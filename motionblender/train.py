import os
from pudb import set_trace
from collections import Counter
import yaml
import random
import gc
from skimage.color import label2rgb
import imageio.v3 as iio
import numpy as np
from PIL import Image
import os.path as osp
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from torchvision.transforms.functional import to_pil_image
import torch
from torch.utils.data import DataLoader
import motionblender.lib.kps_vis as kp_vis
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import tyro
from loguru import logger as guru
from tqdm.auto import trange, tqdm
from pytorch_msssim import SSIM, ms_ssim
import motionblender.init_utils as init_util
from flow3d.data.iphone_dataset import (
    BaseDataset,
    Dataset,
    iPhoneDataset,
    iPhoneDatasetVideoView,
)
from flow3d.data.utils import to_device
from flow3d.metrics import mPSNR, mSSIM, mLPIPS
from flow3d.loss_utils import (
    compute_accel_loss,
    compute_se3_smoothness_loss,
    compute_z_acc_loss,
    masked_l1_loss,
    compute_gradient_loss
)
from flow3d.vis.utils import project_2d_tracks, make_video_divisble
from jaxtyping import Float32
from typing import Annotated
from torch import Tensor
import motionblender.lib.animate as anim
from motionblender.lib.ctrl import AdaptiveController, ControlConfig, create_optimizers_and_schedulers, step_optimizers, LRConfig
from motionblender.lib.dataset import MotionBlenderGeneralDataConfig, MotionBlenderDataset, MotionBlenderIPhoneDataConfig, MotionBlenderIPhoneDataset, MotionBlenderGeneralDataset
from motionblender.lib.misc import (
    add_prefix_to_dict,
    backup_code,
    dump_cpkl,
    dict_items,
    linear_smooth_func,
    load_cpkl,
    loss_dict_to_str,
    loopy,
    render,
    viridis,
    to_fg_names,
    make_guru_once_filter,
)
from motionblender.lib.params import GaussianParams, SplatsDict, merge_splat_dicts
from motionblender.lib.motion import MotionBlender, links_to_pointset, MotionBlenderType
from motionblender.lib.init_graph.human_kinematic_tree_lifting import wholebody_keypoints
import motionblender.lib.pv as pv
import motionblender.lib.convert_utils as cvt


@torch.no_grad()
def save_pv_vis(gs_modules: dict[str, GaussianParams], motion_modules: dict[str, MotionBlender], tag, work_dir, progress=False, chunk=False, names: list[str]=[]):
    vis_path = Path(work_dir) / 'vis' / tag
    vis_path.mkdir(parents=True, exist_ok=True)
    for inst_name in motion_modules:
        if len(names) > 0 and inst_name not in names:
            continue
        gs = gs_modules[inst_name]
        motion = motion_modules[inst_name]
        motion.clear_motion_cache()
        num_frames = motion.num_frames
        plotter = pv.Plotter(backend=f"remote:{str(vis_path / (inst_name + '.pvstate'))}")
        plotter.update_param("colors", gs.get_colors(use_default_sh=True).detach().cpu())
        positions = motion.transform_splats_to_ts(gs.means, range(num_frames), progress=progress, chunk=chunk).transpose(0, 1).detach().cpu().numpy()
        plotter.update_param("means", positions)
        if not motion.is_rigid:
            joints = [motion._joints_tensor_cache[t].detach().cpu().numpy() for t in range(num_frames)]
            plotter.update_param("graph.joints", joints)
            plotter.update_param("graph.links", motion.links.detach().cpu())
        motion.clear_motion_cache()
        plotter.render()



def get_train_val_datasets(
    data_cfg: MotionBlenderGeneralDataConfig | MotionBlenderIPhoneDataConfig, load_val: bool, cameras: list[str]=[], camera_name_override: dict[str, str]={}, **kwargs
) -> tuple[list[MotionBlenderDataset], Dataset | None, Dataset | None]:
    if isinstance(data_cfg, MotionBlenderGeneralDataConfig):
        MBDatasetClass = MotionBlenderGeneralDataset
        MBDatasetValClass = MotionBlenderGeneralDataset
    else:
        assert isinstance(data_cfg, MotionBlenderIPhoneDataConfig)
        MBDatasetClass = MotionBlenderIPhoneDataset
        MBDatasetValClass = iPhoneDataset

    train_video_view = None
    val_img_dataset = None
    if len(cameras) > 0:
        assert MBDatasetClass == MotionBlenderGeneralDataset
        train_datasets = [MBDatasetClass(**asdict(data_cfg), img_prefix=c, **kwargs) for c in cameras]
        train_dataset = train_datasets[0]
    else:
        train_dataset = MBDatasetClass(**asdict(data_cfg), **kwargs)
        train_datasets = [train_dataset]
    train_video_view = iPhoneDatasetVideoView(train_dataset)

    for D in train_datasets + [train_video_view]:
        if D.img_prefix in camera_name_override:
            D.img_prefix = camera_name_override[D.img_prefix]

    val_img_dataset = (
        MBDatasetValClass(
            **asdict(replace(data_cfg, split="val", load_from_cache=True))
        )
        if train_dataset.has_validation and load_val
        else None
    )
    # assert all([len(D) == len(train_datasets[0]) for D in train_datasets])
    return train_datasets, train_video_view, val_img_dataset




#ANCHOR losscfg
@dataclass
class LossesConfig:
    w_mask: float = 1.0
    w_rgb: float = 1.0
    w_track: float = 2.0 # w_track, w_depth_track

    w_depth: float = 0.5
    w_depth_track: float = 0.1
    w_depth_grad: float = 1
    w_scale_var: float = 0.01

    w_smooth_motion: float = 0.1
    w_sparse_link_assignment: float = 0.01
    w_minimal_movement_in_cano: float = 0.3
    w_length_reg: float = 0.0
    w_arap: float = 0.0
    length_reg_names: list[str] = field(default_factory=lambda: [])


    w_kp2d: float = 1.0

    qt_depth: float = 0.98
    qt_depth_grad: float = 0.95
    qt_track2d: float = 0.98
    qt_kp2d: float = 0.9
    qt_mask: float = 0.98
    valid_kp_thresh: float = 0.6


@dataclass
class TrainConfig:
    work_dir: str
    data: (
        Annotated[MotionBlenderIPhoneDataConfig, tyro.conf.subcommand(name="iphone")]
        | Annotated[MotionBlenderGeneralDataConfig, tyro.conf.subcommand(name="general")]
    )
    ctrl: ControlConfig
    lr: LRConfig
    loss: LossesConfig

    rigging_init_steps: int = 1000
    motion_init_steps: int = 2000
    motion_init_batch_size: int = 32
    motion_pretrain_with_means: bool = True
    motion_pretrain_with_kp3d: bool = False

    skip_global_procruste: bool = False
    train_steps: int = 30000

    batch_size: int = 8
    num_workers: int = 4
    val_num_workers: int = 1
    log_every: int = 50
    val_every: int = 2500
    resume_if_possible: str = ""
    output_val_images: bool = False

    num_fg: int = 50_000
    fg_only: bool = False
    min_pts_per_inst: int = 10_000
    num_bg: int = 100_000
    num_tracks_per_frame: int = 800
    num_vertices_for_deformable: list[int] = field(default_factory=lambda: [50])
    blend_method: str = 'dq'
    init_gamma: float = 4.0
    init_temperature: float = 0.01 # 0.01
    length_per_link_learnable: bool = True 
    nearest_k_links: int = 5
    voxel_downsample_size: float = 0.05
    use_radiance_kernel: str = "none"

    test_run: bool = False
    eval_only: bool = False

    step_offset: int = 0
    deformable_link_quantize: int = -1
    # flexible_cano: bool = False

    cameras: list[str] = field(default_factory=lambda: [])
    camera_name_override: str = ""
    init_with_only_one_camera: bool = True
    init_with_rgbd: bool = False
    use_sh_color: bool = False

    render_engine: str = 'gsplat'

    camera_adjustment: bool = False
    camera_learnable: bool = True
    camera_adjust_limits: list[float] = field(default_factory=lambda: [])
    stop_losses_after_iter: int = 2500
    stop_losses: list[str] = field(default_factory=lambda: [])  # scale_var, depth_grad, track_depth, depth, kp2d, track2d, mask
    skip_preprocess_pcd_clustering: list[str] = field(default_factory=lambda: [])

    skip_save_pv: bool = False
    


def run_motion_pretrain(train_datasets: list[MotionBlenderDataset], gs_modules: nn.ModuleDict, motion_modules: dict[str, MotionBlender],
                        dict_of_track3ds: dict[str, init_util.TrackObservations] | None,
                        device="cuda", motion_init_steps: int=1000,  motion_init_batch_size: int=32, loss: LossesConfig=None,
                        init_with_only_one_camera=False, motion_pretrain_with_means=True, motion_pretrain_with_kp3d=False, **_):
    pretrain_data = {}
    loss_cfg = loss
    use_track3d = dict_of_track3ds is not None
    for dataset_i, train_dataset in enumerate(train_datasets):
        if init_with_only_one_camera and dataset_i > 0:
            continue
        _Ks = train_dataset.get_Ks().to(device)
        _w2cs = train_dataset.get_w2cs().to(device)
        num_frames = train_dataset.num_frames
        num_total_frames = train_dataset.num_all_frames_in_scene
        _time_ids = train_dataset.time_ids
        assert _time_ids.shape[0] == num_frames == len(_Ks) == len(_w2cs)
        log_freq = 20

        _inst_track_info = {}
        if use_track3d: 
            guru.info('motion pretraining with 3D tracks')
            for inst_name, tracks_3d in dict_of_track3ds.items():
                tracks_3d = tracks_3d.map(lambda x: x[:, _time_ids].to(device) if len(x.shape) > 1 and x.shape[1] == num_total_frames else x.to(device))
                gt_2d, gt_depth = project_2d_tracks(
                    tracks_3d.xyz.swapaxes(0, 1), _Ks, _w2cs, return_depth=True # (T, G, 2)
                )
                gt_2d = gt_2d.swapaxes(0, 1) # (G, T, 2)
                gt_depth = gt_depth.swapaxes(0, 1)  # (G, T)
                _inst_track_info[inst_name] = {
                    "gt_2d": gt_2d,
                    "gt_depth": gt_depth,
                    "tracks_3d": tracks_3d,
                }
        else:
            guru.info('motion pretraining with only keypoints')

        _all_gt_kps = {}
        for inst_id, all_kps in train_dataset.instance_id_2_keypoints.items():
            if train_dataset.instance_id_2_motion_type[inst_id] == MotionBlenderType.kinematic:
                inst_name = train_dataset.instance_names[inst_id]
                joint_names = motion_modules[inst_name].joint_names
                keypoint_names = motion_modules[inst_name].keypoint_names
                assert joint_names
                joint_indexes = [keypoint_names.index(jn) for jn in joint_names]
                joint_kps = torch.as_tensor(all_kps).float()[:, joint_indexes].to(device)
                _all_gt_kps[inst_name]  = {'2d': joint_kps } # (num_frames, num_joints, 3), x,y,score
                if motion_pretrain_with_kp3d:
                    kp_3d_list = []
                    for fid in trange(len(joint_kps), desc="preparing 3d keypoints"):
                        rgb = train_dataset.imgs[fid]
                        depth = train_dataset.depths[fid]
                        img_h, img_w = train_dataset.imgs[fid].shape[:2]
                        pts, rgbs = cvt.get_pointcloud_from_rgb_depth_cam(rgb, depth, torch.inverse(train_dataset.w2cs[fid]), train_dataset.Ks[fid], img_w, img_h)
                        pts_hw = pts.reshape(img_h, img_w, 3)
                        scores = joint_kps[fid, :, -1].clone()
                        Y, X = joint_kps[fid, :, 1].long(), joint_kps[fid, :, 0].long()
                        joints = pts_hw[Y.cpu(), X.cpu()].to(Y.device)
                        # kp_vis.draw_keypoints((rgb * 255).permute(2, 0, 1).to(torch.uint8), joint_kps[fid]).save('outputs/test.png')
                        scores[train_dataset.instance_masks[fid][Y.cpu(), X.cpu()].flatten() != inst_id] = 0
                        kp_3d_list.append(torch.cat([joints, scores.reshape(-1, 1)], dim=-1))

                    _all_gt_kps[inst_name]['3d'] = torch.stack(kp_3d_list, dim=0) # (num_frames, num_joints, 4), x,y,z,score


        pretrain_data[train_dataset.img_prefix] = {
            'inst_track_info': _inst_track_info,
            'all_gt_kps': _all_gt_kps,  # (num_frames, num_joints, 3), x,y,score
            'time_ids': _time_ids.to(device),
            'Ks': _Ks,
            'w2cs': _w2cs,
        }

    motion_keys = list(motion_modules.keys())
    all_means = [v for mod_name in motion_keys for k, v in gs_modules[mod_name].named_parameters() if k.endswith('means')]
    all_global_ts = [v for k, v in motion_modules.named_parameters() if k.endswith('global_ts')]
    all_global_rot6d = [v for k, v in motion_modules.named_parameters() if k.endswith('global_rot6d')]
    all_local_motions = [v for k, v in motion_modules.named_parameters() if 'global' not in k]

    params =  [
                {"params": all_global_rot6d, "lr": 1e-2},
                {"params": all_global_ts, "lr": 3e-2},

                {'params': all_local_motions, "lr": 1e-3}
    ]
    if motion_pretrain_with_means:
        params.append( {"params": all_means, "lr": 1e-3})
    else:
        guru.warning("Not including gaussian means for motion pretraining, this is safer")

    optimizer = optim.Adam(params)

    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.1 ** (1 / motion_init_steps))
    for i in trange(motion_init_steps, desc="Motion Pretraining"):
        for pretrain_data_key, pretrain_data_val in pretrain_data.items():
            loss_dict = {}
            inst_track_info = pretrain_data_val['inst_track_info']
            all_gt_kps = pretrain_data_val['all_gt_kps']
            time_ids = pretrain_data_val['time_ids']
            Ks = pretrain_data_val['Ks']
            w2cs = pretrain_data_val['w2cs']
            if motion_init_batch_size > 0:
                indices = torch.randperm(len(time_ids), device=device)[:motion_init_batch_size]
            else:
                indices = torch.arange(len(time_ids), device=device)

            for inst_name, motion in motion_modules.items():
                motion.clear_motion_cache()
                inst_loss_dict = {}
                if use_track3d:
                    _ = inst_track_info[inst_name]
                    gt_2d, gt_depth, tracks_3d = _['gt_2d'], _['gt_depth'], _['tracks_3d']
                    positions = motion.transform_splats_to_ts(gs_modules[inst_name].means, time_ids[indices], chunk=True) # [n, t, 3]
                    inst_loss_dict['track_3d'] = masked_l1_loss(positions, tracks_3d.xyz[:, indices], (tracks_3d.visibles[:, indices].float() * tracks_3d.confidences[:, indices])[..., None])

                    pred_2d, pred_depth = project_2d_tracks(positions.swapaxes(0, 1), Ks[indices], w2cs[indices], return_depth=True)
                    pred_2d = pred_2d.swapaxes(0, 1)
                    pred_depth = pred_depth.swapaxes(0, 1)
                    loss_2d = (
                        masked_l1_loss(
                            pred_2d,
                            gt_2d[:, indices],
                            (tracks_3d.invisibles[:, indices].float() * tracks_3d.confidences[:, indices])[..., None],
                            quantile=0.95,
                        )
                        / Ks[0, 0, 0]
                    )
                    inst_loss_dict['track_2d'] = 0.5 * loss_2d
                else:
                    for t in time_ids[indices]:
                        motion.get_transformation_at_t(gs_modules[inst_name].means, t) # [n, t, 3]

                if motion.type == MotionBlenderType.kinematic and inst_name in all_gt_kps:
                    pred_joints = torch.stack([motion._joints_tensor_cache[int(t)] for t in time_ids[indices]]) # t, n, 3
                    pred_joints_2d = project_2d_tracks(pred_joints, Ks[indices], w2cs[indices])
                    loss_joints_2d = masked_l1_loss(
                            pred_joints_2d,
                            all_gt_kps[inst_name]['2d'][indices, :, :2],
                            mask=all_gt_kps[inst_name]['2d'][indices, :, 2:3] > loss_cfg['valid_kp_thresh'],
                            quantile=0.9,
                        ) / Ks[0, 0, 0]
                    inst_loss_dict['kp_2d'] = 0.5 * loss_joints_2d

                    if motion_pretrain_with_kp3d:
                        inst_loss_dict['kp_3d'] = masked_l1_loss(pred_joints, 
                                all_gt_kps[inst_name]['3d'][indices, :, :3], 
                                mask=all_gt_kps[inst_name]['3d'][indices, :, 3:] > loss_cfg['valid_kp_thresh'], quantile=0.9)

                inst_loss_dict.update({k: v * 0.01 for k, v in motion.regularization('sparse_link_assignment').items()})

                if loss_cfg['w_smooth_motion'] > 0:
                    w_smooth = linear_smooth_func(i, 0.01, 0.1, 400, motion_init_steps)
                    inst_loss_dict.update({k: v * w_smooth for k, v in motion.regularization('smooth_motion').items()})
                else:
                    guru.warning('skip smooth_motion', once=True)

                if loss_cfg['w_minimal_movement_in_cano'] > 0:
                    cano_w_smooth = linear_smooth_func(i, 0.05, 0.5, 400, motion_init_steps)
                    inst_loss_dict.update({k: v * cano_w_smooth for k, v in motion.regularization('minimal_movement_in_cano').items()})
                elif loss_cfg['w_minimal_movement_in_cano'] < 0:
                    guru.warning('skip minimal_movement_in_cano (but will keep intiial params the same if possible)', once=True)
       
                if loss_cfg['w_arap'] > 0 and motion.type == MotionBlenderType.deformable:
                    asap_w_smooth = linear_smooth_func(i, 0.01, 0.1, 400, motion_init_steps)
                    inst_loss_dict.update({k: v * asap_w_smooth for k, v in motion.regularization('arap').items()})

                if loss_cfg['w_length_reg'] > 0 and motion.type == MotionBlenderType.deformable and \
                    inst_name.rpartition('-')[0] in loss_cfg['length_reg_names']:
                    length_reg_w_smooth = linear_smooth_func(i, 0.01, 0.1, 400, motion_init_steps)
                    inst_loss_dict.update({k: v * length_reg_w_smooth for k, v in motion.regularization('length_reg').items()})

                inst_loss_dict = add_prefix_to_dict(inst_loss_dict, f"({pretrain_data_key}-{inst_name})")
                loss_dict.update(inst_loss_dict)

        optimizer.zero_grad()
        sum(loss_dict.values()).backward()
        optimizer.step()
        scheduler.step()

        if loss_cfg['w_minimal_movement_in_cano'] < 0:
            for motion in motion_modules.values():
                motion.keep_initial_params_the_same()

        if i % log_freq == 0:
            guru.info(loss_dict_to_str(loss_dict))

    return gs_modules, motion_modules


def apply_global_motion(splats: SplatsDict, t: int, cam_mod: nn.ParameterDict) -> SplatsDict:
    t = int(t)
    device = splats['colors'].device
    rot6d_lim, ts_lim = cam_mod['rot6d_lim'], cam_mod['ts_lim']
    camera_rot6ds, camera_ts = cam_mod['rot6ds'], cam_mod['ts']

    target_t = torch.tanh(camera_ts[t]) * ts_lim * 0.5
    target_rot6d = anim.rmat_to_cont_6d(torch.eye(3, device=device))
    target_rot6d += torch.tanh(camera_rot6ds[t]) * rot6d_lim * 0.5
    target_T = anim.rt_to_mat4(anim.cont_6d_to_rmat(target_rot6d), target_t)

    splats['means'] = anim.apply_mat4(target_T, splats['means'])
    if 'quats' in splats:
        splats['quats'] = anim.apply_mat4_quat(target_T, splats['quats'], format='wxyz')
    return splats

def run_full_training(gs_modules: dict[str, GaussianParams], motion_modules: dict[str, MotionBlender], gaussian_names: list[str], train_datasets: list[MotionBlenderDataset],
                      train_video_view: BaseDataset, val_img_dataset: BaseDataset, cfg: TrainConfig, device: str="cuda",
                      save_callback=lambda: None, camera_state_dict = None):
    if not cfg.fg_only: assert gaussian_names[-1] == 'bg'
    num_instances = len(to_fg_names(gaussian_names))
    num_frames = train_datasets[0].num_all_frames_in_scene # NOTE: num_frames is the total time
    work_dir = Path(cfg.work_dir)

    if cfg.camera_adjustment:
        cam_names = [train_dataset.img_prefix for train_dataset in train_datasets]
        if cam_names != cfg.cameras:
            guru.error(f"Camera names from data ({cam_names}) do not match the camera names in config ({cfg.cameras})")
        if max(list(Counter(cam_names).values())) != 1:
            guru.error(f"Camera names should be unique, cameras from data is not unique: {cam_names}")

        if not cfg.camera_learnable: assert camera_state_dict is not None, "camera_state_dict should be provided when camera_learnable is False"
        if cfg.camera_adjust_limits:
            rot6d_lim_lst, ts_lim_lst = [torch.as_tensor(a).float() for a in cfg.camera_adjust_limits[:len(cam_names)]], [torch.as_tensor(a).float() for a in cfg.camera_adjust_limits[len(cam_names):]]
            guru.warning(f"Using camera adjust limits from config (not from data), rot6d: {rot6d_lim_lst}, ts: {ts_lim_lst}")
        else:
            rot6d_lim_lst, ts_lim_lst = [], [] # len(*_lim_lst) == len(cam_names)
            for cams in [train_dataset.w2cs for train_dataset in train_datasets]:
                ts = cams[:, :3, 3]
                delta_ts = ts[1:] - ts[:-1]
                rot6d = anim.rmat_to_cont_6d(cams[:, :3, :3])
                delta_rot6d = rot6d[1:] - rot6d[:-1]
                rot6d_lim, ts_lim = delta_rot6d.abs().max(0).values.to(device), delta_ts.abs().max(0).values.to(device)
                rot6d_lim_lst.append(rot6d_lim)
                ts_lim_lst.append(ts_lim)

        camera_module = nn.ParameterDict()
        for cam_i, cam_name in enumerate(cam_names):
            camera_module[cam_name] = nn.ParameterDict({
                'ts_lim':  nn.Parameter(ts_lim_lst[cam_i].to(device), requires_grad=False),
                'rot6d_lim':  nn.Parameter(rot6d_lim_lst[cam_i].to(device), requires_grad=False),
                'rot6ds': nn.Parameter(torch.zeros(num_frames, 6, device=device)),
                'ts':  nn.Parameter(torch.zeros(num_frames, 3, device=device)) })

        if camera_state_dict is not None:
            guru.warning("using camera deltas from previous training")
            camera_module.load_state_dict(camera_state_dict)
        else:
            guru.warning('initialize camera deltas')
        
        guru.info(f"camera_module: {camera_module}")
        if not cfg.camera_learnable:
            for p in camera_module.parameters():
                p.requires_grad = False

    optimizers, schedulers = create_optimizers_and_schedulers(gs_modules, motion_modules, max_steps=1 if cfg.eval_only else cfg.train_steps, use_sh_color=cfg.use_sh_color,
                                                              extra_params={'cameras': list(camera_module.parameters())} if cfg.camera_learnable and cfg.camera_adjustment else None,
                                                              **asdict(cfg.lr))
    ctrl = AdaptiveController(cfg.ctrl, gs_modules, optimizers, gaussian_names, device=device, num_frames=num_frames)
    val_dataloader = DataLoader(val_img_dataset, batch_size=1, num_workers=cfg.val_num_workers, shuffle=False, collate_fn=iPhoneDataset.train_collate_fn) if val_img_dataset is not None else None
    video_dataloader = DataLoader(train_video_view, batch_size=1, num_workers=cfg.val_num_workers, shuffle=False, collate_fn=iPhoneDataset.train_collate_fn)
    trainloader = DataLoader(torch.utils.data.ConcatDataset(train_datasets),
                             batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                             shuffle=True, collate_fn=iPhoneDataset.train_collate_fn)
    train_iter = loopy(trainloader)

    metrics = dict(psnr=mPSNR().to(device), fg_psnr=mPSNR().to(device), bg_psnr=mPSNR().to(device))
    val_metrics = dict(psnr=mPSNR().to(device), ssim=mSSIM().to(device), lpips=mLPIPS().to(device))
    ssim = SSIM(data_range=1.0, size_average=True, channel=3)

    def clear_motion_cache_at_t(motions, t=None):
        for v in motions.values():
            if t is None:
                v.clear_motion_cache()
            else:
                if int(t) != v.cano_t:
                    v.clear_motion_cache(int(t))

    def loss_is_stopped(name, step): # scale_var, depth_grad, track_depth, depth, kp2d, track2d, mask
        shall_be_stop = name in cfg.stop_losses and step >= cfg.stop_losses_after_iter
        if shall_be_stop:
            guru.warning(f"Loss {name} is stopped at step {cfg.stop_losses_after_iter}", once=True)
        return shall_be_stop

    def get_full_splat_at_t(gses, motions, t:int, w2c=None):
        full_splats = {}
        for g in gaussian_names:
            if g not in motions:
                means, quats = gses[g].means, gses[g].get_quats()
            else:
                means, quats = motions[g].transform_splats_to_t(gses[g].means, t, cano_quats_wxyz=gses[g].get_quats())
            full_splats.setdefault('means', []).append(means)
            full_splats.setdefault('quats', []).append(quats)
            full_splats.setdefault("scales", []).append(gses[g].get_scales())
            full_splats.setdefault("opacities", []).append(gses[g].get_opacities())

            if w2c is not None:
                assert w2c.dim() == 3
                cam_center = torch.inverse(w2c)[:, :3, 3]
            else:
                cam_center = None
            full_splats.setdefault("colors", []).append(gses[g].get_colors(means, cam_center.to(means.device)))
        full_splats = {k:torch.cat(v) for k, v in full_splats.items()}
        return full_splats

    @torch.no_grad()
    def run_eval(step):
        gc.collect()
        torch.cuda.empty_cache()
        video_dir = work_dir / 'video'
        video_dir.mkdir(exist_ok=True, parents=True)
        val_rgb_dir = video_dir / 'val_rgb' / str(step)
        motion_modules.eval()
        motions = motion_modules
        clear_motion_cache_at_t(motions)
        if val_dataloader is not None:
            for v in val_metrics.values(): v.reset()
            ms_ssim_lst = []
            for batch in tqdm(val_dataloader, desc="evaluation", leave=False):
                fname = batch['frame_names'][0]
                t = batch["ts"][0]
                w2c = batch["w2cs"]  # (1, 4, 4).
                K = batch["Ks"] # (1, 3, 3).
                img = batch["imgs"].to(device) # (1, H, W, 3).
                valid_mask = batch.get("valid_masks", torch.ones_like(batch["imgs"][..., 0])) # (1, H, W).
                fg_mask = batch["masks"]  # (1, H, W).
                covisible_mask = batch.get("covisible_masks", torch.ones_like(fg_mask))  # (H, W).
                W, H = img_wh = img[0].shape[-2::-1]
                valid_mask *= covisible_mask
                valid_mask = valid_mask.to(device)
                val_splats = get_full_splat_at_t(gs_modules, motions, t, w2c)
                if cfg.camera_adjustment:
                    val_splats = apply_global_motion(val_splats, t, camera_module[batch['cam_name'][0]])
                pkg = render(val_splats, w2c.to(device), K.to(device), img_wh, bg_color=torch.ones(3).reshape(1, 3).to(device), engine=cfg.render_engine)
                rendered_img = pkg[0][..., :3]
                if cfg.output_val_images:
                    val_rgb_dir.mkdir(exist_ok=True, parents=True)
                    Image.fromarray((rendered_img[0] * 255).cpu().numpy().astype(np.uint8)).save(val_rgb_dir / f"{fname}.png")
                val_metrics['psnr'].update(rendered_img, img, valid_mask)
                val_metrics['ssim'].update(rendered_img, img, valid_mask)
                val_metrics['lpips'].update(rendered_img, img, valid_mask)
                ms_ssim_lst.append(ms_ssim(rendered_img.permute(0, 3, 1, 2), img.permute(0, 3, 1, 2), data_range=1.0, size_average=True).item())
                clear_motion_cache_at_t(motions, t)
            guru.info(loss_dict_to_str(add_prefix_to_dict({**{k: v.compute() for k, v in val_metrics.items()}, 'ms_ssim': sum(ms_ssim_lst)/len(ms_ssim_lst)}, "val")))

        video = []
        clear_motion_cache_at_t(motions)

        for batch in tqdm(video_dataloader, desc="video rendering", leave=False):
            fname = batch['frame_names'][0]
            t = batch["ts"][0]
            w2c = batch["w2cs"].to(device)  # (1, 4, 4).
            K = batch["Ks"] # (1, 3, 3).
            img = batch["imgs"]
            img_wh = img[0].shape[-2::-1]
            video_splats = get_full_splat_at_t(gs_modules, motions, t, w2c)
            if cfg.camera_adjustment:
                video_splats = apply_global_motion(video_splats, t,
                                                   camera_module[batch['cam_name'][0]])
            pkg = render(video_splats, w2c, K.to(device), img_wh, bg_color=torch.ones(3).reshape(1, 3).to(device), engine=cfg.render_engine)
            rendered_img = pkg[0][..., :3] # h, w, 3
            video.append((rendered_img[0] * 255).cpu())
            clear_motion_cache_at_t(motions, t)
        video = torch.stack(video).numpy().astype(np.uint8)
        video_step = step
        if video_step != 'eval': video_step += cfg.step_offset
        iio.imwrite(video_dir / f"{video_step}.mp4", make_video_divisble(video), fps=30, plugin='FFMPEG')
        guru.info(f"Saved video to {video_dir / f'{video_step}.mp4'}")
        motion_modules.train()

        if step != 'eval': save_callback(f"{step + cfg.step_offset}", camera_params=camera_module.state_dict() if cfg.camera_adjustment else None)
        gc.collect()
        torch.cuda.empty_cache()

    ctrl.control_cache = { 'img_wh': [], 'radii': [], 'xys': [] }

    for step in trange(cfg.train_steps):
        loss_dict = {}
        clear_motion_cache_at_t(motion_modules)
        batch = next(train_iter)
        batch = to_device(batch, device)
        B = batch["imgs"].shape[0]
        W, H = img_wh = batch["imgs"].shape[2:0:-1]

        ts = batch["ts"] # (B,).
        w2cs = batch["w2cs"] # (B, 4, 4).
        cam_centers = torch.inverse(w2cs)[:, :3, 3] # (B, 3)

        Ks = batch["Ks"] # (B, 3, 3).
        imgs = batch["imgs"] # (B, H, W, 3).
        valid_masks = batch.get("valid_masks", torch.ones_like(batch["imgs"][..., 0])) # (B, H, W).
        masks = batch["masks"] # (B, H, W).
        masks *= valid_masks
        imasks = batch["instance_masks"].long() # (B, H, W)
        imasks *= (masks > 0.5) # only keep fg
        imasks_oh = F.one_hot(imasks, num_classes=num_instances+1)[..., 1:] # (B, H, W, num_instances)

        depths = batch["depths"] # (B, H, W).
        gt_kps2d = batch['kps2d'] if 'kps2d' in batch else None

        if cfg.data.use_tracks:
            N = batch["target_ts"][0].shape[0]
            query_tracks_2d = batch["query_tracks_2d"] # track at time t,  [(P, 2), ...].
            target_ts = batch["target_ts"] #  [(N,), ...].
            target_w2cs = batch["target_w2cs"] # [(N, 4, 4), ...].
            # target_cam_centers = torch.inverse(target_w2cs)[:, :3, 3] # (N, 3)
            target_Ks = batch["target_Ks"] # [(N, 3, 3), ...].

            target_tracks_2d = batch["target_tracks_2d"]  # [(N, P, 2), ...]. # N is the neighbor size
            target_visibles = batch["target_visibles"]  # [(N, P), ...].
            target_confidences = batch["target_confidences"]  # [(N, P), ...].
            target_track_depths = batch["target_track_depths"]  # [(N, P), ...].

            tracks_2d = torch.cat([x.reshape(-1, 2) for x in target_tracks_2d], dim=0) # (P_all, 2).
            visibles = torch.cat([x.reshape(-1) for x in target_visibles], dim=0) # (P_all,)
            confidences = torch.cat([x.reshape(-1) for x in target_confidences], dim=0) # (P_all,)
            target_ts_vec = torch.cat(target_ts) # (BN)
            frame_intervals = (ts.repeat_interleave(N) - target_ts_vec).abs()

            all_ts = set(torch.unique(ts).long().tolist() + torch.unique(target_ts_vec).long().tolist())
        else:
            N = 0
            all_ts = set(torch.unique(ts).long().tolist())

        points_cache, gs_cache = {}, {}
        for g in gaussian_names:
            gs_cache[g] = {'scales': gs_modules[g].get_scales(), 'opacities': gs_modules[g].get_opacities() }
            if g in motion_modules:
                for t in all_ts:
                    means, quats = motion_modules[g].transform_splats_to_t(gs_modules[g].means, t, cano_quats_wxyz=gs_modules[g].get_quats())
                    points_cache[(g, t)] = means, quats
            else:
                points_cache[g] = gs_modules[g].means, gs_modules[g].get_quats()

        bg_color = torch.as_tensor([1, 1, 1] + [0] * num_instances + [0] * N * 3).to(device).float().reshape(1, -1)
        rendered_all = {}
        for bi in range(B):
            splats: SplatsDict = {}
            for gi, g in enumerate(gaussian_names):
                t = int(ts[bi])
                means, quats = points_cache[(g, t)] if g in motion_modules else points_cache[g]
                splats.setdefault("means", []).append(means)
                splats.setdefault("quats", []).append(quats)
                splats.setdefault("scales", []).append(gs_cache[g]['scales'])
                splats.setdefault("opacities", []).append(gs_cache[g]['opacities'])
                colors = gs_modules[g].get_colors(means, cam_centers[bi])
                if g == 'bg':
                    colors = torch.cat([colors, bg_color[:, 3:].repeat(len(colors), 1)], dim=-1)
                else:
                    # instance mask
                    inst_flag = [0.]*num_instances
                    inst_flag[gi] = 1.
                    colors = torch.cat([colors, torch.as_tensor(inst_flag).to(device).float().reshape(1, -1).repeat(len(colors), 1)], dim=-1)

                    # track
                    if cfg.data.use_tracks:
                        target_means = []
                        for t in target_ts[bi]: target_means.append(points_cache.get((g, int(t)), points_cache.get(g, None))[0])
                        target_means = torch.stack(target_means).transpose(0, 1) # (P, N, 3)
                        target_means = torch.einsum("nij,pnj->pni", target_w2cs[bi][:, :3], F.pad(target_means, (0, 1), value=1.0))
                        track_3d_vals = target_means.flatten(-2)
                        colors = torch.cat([colors, track_3d_vals], dim=-1)

                assert colors.shape[-1] == bg_color.shape[-1]
                splats.setdefault("colors", []).append(colors)

            splats = {k:torch.cat(v) for k, v in splats.items()}
            if cfg.camera_adjustment:
                _bi_cam_name = batch['cam_name'][bi]
                splats = apply_global_motion(splats, int(ts[bi]), camera_module[_bi_cam_name])
            render_colors, alphas, info = render(splats, w2cs[bi], Ks[bi], img_wh, bg_color, engine=cfg.render_engine)
            pred_colors, pred_masks, pred_tracks_3d, pred_depths = torch.split(render_colors, [3, num_instances, N*3, 1], dim=-1)

            rendered_all.setdefault("color", []).append(pred_colors[0])
            rendered_all.setdefault("mask", []).append(pred_masks[0])
            rendered_all.setdefault("depth", []).append(pred_depths[0])
            rendered_all.setdefault("track_3d", []).append(pred_tracks_3d[0])
            if alphas is not None:
                rendered_all.setdefault("alpha", []).append(alphas[0])

            info['means2d'].retain_grad()
            ctrl.control_cache['xys'].append(info["means2d"])
            ctrl.control_cache['radii'].append(info["radii"])
            ctrl.control_cache['img_wh'].append(img_wh)

        rendered_all = {k: torch.stack(v) for k, v in rendered_all.items() if len(v)> 0}

        # MARK: rgb loss
        if cfg.fg_only:
            fg_masks = imasks_oh.sum(dim=-1, keepdim=True).float()
            imgs = imgs * fg_masks + (1.0 - fg_masks)
            valid_ratio = fg_masks.sum() / fg_masks.numel()
        else:
            imgs = imgs * valid_masks[..., None] + (1.0 - valid_masks[..., None])
            valid_ratio = valid_masks.sum() / valid_masks.numel()
        if step == 0: guru.error(f"note: valid image ratio roughly is {valid_ratio}")

        rendered_imgs = rendered_all['color'] * valid_masks[..., None] + (1.0 - valid_masks[..., None])
        rgb_loss = 0.8 * F.l1_loss(rendered_imgs, imgs) + 0.2 * (1 - ssim(rendered_imgs.permute(0, 3, 1, 2), imgs.permute(0, 3, 1, 2)))
        loss_dict['rgb'] = rgb_loss * cfg.loss.w_rgb #/ valid_ratio

        # MARK: mask loss
        if 'alpha' in rendered_all:
            if cfg.fg_only:
                alpha_loss = F.mse_loss(rendered_all["alpha"],  fg_masks, reduction="none")
                alpha_loss *= valid_masks[..., None]
                loss_dict['alpha'] = alpha_loss.mean() * cfg.loss.w_mask
            else:
                loss_dict['alpha'] = F.mse_loss(rendered_all["alpha"], torch.ones_like(rendered_all["alpha"])) * cfg.loss.w_mask

        if cfg.loss.w_mask > 0 and not loss_is_stopped('mask', step):
            loss_dict['mask'] = masked_l1_loss(rendered_all["mask"], imasks_oh, mask=valid_masks[..., None], quantile=cfg.loss.qt_mask) * cfg.loss.w_mask

        # MARK: track loss
        if cfg.data.use_tracks and cfg.loss.w_track > 0 and not loss_is_stopped('track2d', step):
            pred_tracks_3d = rendered_all["track_3d"].reshape(B, H, W, N, 3).permute(0, 3, 1, 2, 4).reshape(-1, H * W, 3) # B, H, W, N, 3 -> B N H W 3 -> (BN) (HW) 3
            pred_tracks_2d = torch.einsum("bij,bpj->bpi", torch.cat(target_Ks), pred_tracks_3d)
            mapped_depth = torch.clamp(pred_tracks_2d[..., 2:], min=1e-6)  # (B * N, H * W, 1).
            pred_tracks_2d = pred_tracks_2d[..., :2] / mapped_depth  # (B * N, H * W, 2).

            w_interval = torch.exp(-2 * frame_intervals / num_frames)  # (B * N).
            track_weights = confidences[..., None] * w_interval
            masks_flatten = torch.zeros_like(masks)
            for bi in range(B):
                # This takes advantage of the fact that the query 2D tracks are always on the grid.
                query_pixels = query_tracks_2d[bi].to(torch.int64)
                masks_flatten[bi, query_pixels[:, 1], query_pixels[:, 0]] = 1.0

            masks_flatten = masks_flatten.reshape(-1, H * W).tile(1, N).reshape(-1, H * W) > 0.5
            track_2d_loss = masked_l1_loss(
                pred_tracks_2d[masks_flatten][visibles],
                tracks_2d[visibles],
                mask=track_weights[visibles],
                quantile=cfg.loss.qt_track2d,
            ) / max(H, W)
            loss_dict['track_2d'] = track_2d_loss * cfg.loss.w_track

        # MARK: kp loss
        if gt_kps2d is not None and cfg.loss.w_kp2d > 0 and not loss_is_stopped('kp2d', step):
            for g in motion_modules.keys():
                if motion_modules[g].type == MotionBlenderType.kinematic:
                    assert g in gt_kps2d
                    m = motion_modules[g]
                    joint_indexes = [m.keypoint_names.index(jn) for jn in m.joint_names]
                    all_pred_joints3d = torch.stack([m._joints_tensor_cache[int(t)] for t in ts])
                    all_gt_joints2d = gt_kps2d[g][:, joint_indexes]

                    all_pred_joints2d = project_2d_tracks(all_pred_joints3d, Ks, w2cs)

                    loss_joints_2d = masked_l1_loss(
                            all_pred_joints2d,
                            all_gt_joints2d[..., :2],
                            mask=all_gt_joints2d[..., 2:3] > cfg.loss.valid_kp_thresh,
                            quantile=cfg.loss.qt_kp2d) / max(H, W)
                    loss_dict['kp_2d'] = cfg.loss.w_kp2d * loss_joints_2d + loss_dict.get('kp_2d', 0)

        # MARK: depth loss
        depth_masks = valid_masks[..., None]
        if cfg.data.depth_type == "metric_depth":
            depth_masks = depth_masks.bool() & (depths > 1e-3)[..., None]
        if cfg.fg_only:
            depth_masks = depth_masks.bool() & (fg_masks > 0.5)
        depth_masks = depth_masks.float()
        b_depth_masks = depth_masks > 0.5
        has_valid_depth = b_depth_masks.sum() > 0
        pred_depth = rendered_all["depth"]
        pred_disp = 1.0 / (pred_depth + 1e-5)
        tgt_disp = 1.0 / (depths[..., None] + 1e-5)
        if cfg.loss.w_depth > 0 and has_valid_depth and not loss_is_stopped('depth', step):
            depth_loss = masked_l1_loss(
                pred_disp,
                tgt_disp,
                mask=depth_masks,
                quantile=cfg.loss.qt_depth,
            )
            loss_dict['depth'] = depth_loss * cfg.loss.w_depth

        if cfg.data.use_tracks and cfg.loss.w_depth_track > 0 and not loss_is_stopped('track_depth', step):
            mapped_depth_gt = torch.cat([x.reshape(-1) for x in target_track_depths], dim=0)
            mapped_depth_loss = masked_l1_loss(
                1 / (mapped_depth[masks_flatten][visibles] + 1e-5),
                1 / (mapped_depth_gt[visibles, None] + 1e-5),
                track_weights[visibles],
            )
            loss_dict['track_depth'] = mapped_depth_loss * cfg.loss.w_depth_track

        if cfg.loss.w_depth_grad > 0 and has_valid_depth and not loss_is_stopped('depth_grad', step):
            depth_gradient_loss = compute_gradient_loss(
                pred_disp,
                tgt_disp,
                mask=b_depth_masks,
                quantile=cfg.loss.qt_depth_grad,
            )
            loss_dict['depth_grad'] = depth_gradient_loss * cfg.loss.w_depth_grad

        # MARK: scale variance min
        if cfg.loss.w_scale_var > 0 and not loss_is_stopped('scale_var', step):
            loss_scale_var = 0
            for g in gaussian_names:
                loss_scale_var += torch.var(gs_modules[g].params['scales'], dim=-1).mean()
            loss_dict['scale_var'] = loss_scale_var * cfg.loss.w_scale_var

        # MARK: regularization
        for k in ['smooth_motion', 'sparse_link_assignment', 'minimal_movement_in_cano', 'length_reg', 'arap']:
            if getattr(cfg.loss, f'w_{k}') <= 0: continue
            for g in motion_modules.keys():
                if k in ['length_reg', 'arap'] and g.rpartition('-')[0] not in cfg.loss.length_reg_names:
                    continue
                for _name, l in motion_modules[g].regularization(k).items():
                    if _name not in loss_dict: loss_dict[_name] = 0
                    loss_dict[_name] += l * getattr(cfg.loss, f'w_{k}')

        step_optimizers(sum(loss_dict.values()), optimizers, schedulers)
        for motion in motion_modules.values():
            motion.keep_initial_params_the_same()
        ctrl.run_control_steps(step)
        ctrl.control_cache = { 'img_wh': [], 'radii': [], 'xys': [] }

        if step % cfg.log_every == 0:
            with torch.no_grad():
                psnr = metrics['psnr'](rendered_imgs, imgs,  valid_masks)
                if not cfg.fg_only: bg_psnr = metrics['bg_psnr'](rendered_imgs, imgs, 1.0 - masks)
                fg_psnr = metrics['fg_psnr'](rendered_imgs, imgs, masks)
                for k in metrics.keys(): metrics[k].reset()
            msg = f"[step {step}/{cfg.train_steps}] " + \
                      loss_dict_to_str({'psnr': psnr, **({'bg_psnr': bg_psnr} if not cfg.fg_only else {}), 'fg_psnr': fg_psnr}) + " | " + loss_dict_to_str(loss_dict)
            if cfg.camera_adjustment:
                cam_rot6d_norms = []
                cam_ts_norms = []
                for cam_name in cam_names:
                    cam_rot6d_norms.append(camera_module[cam_name]['rot6ds'].norm(dim=1).mean().item())
                    cam_ts_norms.append(camera_module[cam_name]['ts'].norm(dim=1).mean().item())
                msg += f" cam_delta: {np.mean(cam_rot6d_norms):.3f} / {np.mean(cam_ts_norms):.3f}"
            guru.info(msg)

        if (step != 0 and step % cfg.val_every == 0):
           run_eval(step)

        if step % 1000 == 0:
            for gs_model in gs_modules.values():
                gs_model.increase_sh_degree()
    run_eval('eval' if cfg.eval_only else cfg.train_steps)



def main(cfg: TrainConfig):
    ckpt_path = Path(cfg.work_dir) / 'ckpt.cpkl'
    if cfg.test_run:
        cfg.train_steps = 10
        cfg.motion_init_steps = 10
        new_work_dir = '/tmp/motionblender-debug'
        os.makedirs(new_work_dir, exist_ok=True)
        if ckpt_path.exists():
            with open(ckpt_path, 'rb') as fin:
                with open(osp.join(new_work_dir, 'ckpt.cpkl'), 'wb') as fout:
                    fout.write(fin.read())
        cfg.work_dir = new_work_dir
        ckpt_path = Path(new_work_dir) / 'ckpt.cpkl'
        cfg.rigging_init_steps  = 10
        cfg.val_every = 5
        cfg.log_every = 1
        cfg.stop_losses_after_iter = 3
        cfg.skip_save_pv = False
        guru.remove()
        guru.add(lambda msg: tqdm.write(msg, end=""), colorize=True, filter=make_guru_once_filter())
    elif not cfg.eval_only:
        backup_code(cfg.work_dir)
        guru.remove()
        guru.add(lambda msg: tqdm.write(msg, end=""), colorize=True, filter=make_guru_once_filter())
        train_log_txt = osp.join(cfg.work_dir, "train_log.txt")
        if osp.exists(train_log_txt):
            with open(train_log_txt, "r") as f_in:
                from time import time
                backup_log_txt = train_log_txt + f".{int(time())}"
                print('Backup log to', backup_log_txt)
                with open(backup_log_txt, "w") as f_out:
                    f_out.write(f_in.read())
            with open(train_log_txt, "w") as f_out:f_out.write("")
        guru.add(train_log_txt, filter=make_guru_once_filter())
    else:
        cfg.train_steps = 0
        assert 'motion_pretrained' in cfg.resume_if_possible and ckpt_path.exists()

    if "SLURM_JOB_ID" in os.environ:
        guru.info(f"SLURM_JOB_ID = {os.environ['SLURM_JOB_ID']}")

    if cfg.resume_if_possible and not ckpt_path.exists():
        guru.info("No checkpoint found, training from scratch even though resume_if_possible is set")
        cfg.resume_if_possible = ""
    guru.info(yaml.dump(asdict(cfg), sort_keys=False, default_flow_style=False))

    def save_checkpoint(tag="", camera_params=None):
        guru.info(f"Saving checkpoint to {ckpt_path}")
        _track_ = None
        if tag == 'init': _track_ = dict_of_track3ds
        for v in motion_modules.values():
            v.clear_motion_cache()
        dump_cpkl(ckpt_path, [gs_modules, motion_modules, _track_, gaussian_names])
        if tag:
            guru.info(f"Saving checkpoint to {ckpt_path}.{tag}")
            dump_cpkl(str(ckpt_path) + f".{tag}", [gs_modules, motion_modules, _track_, gaussian_names])
            if not cfg.skip_save_pv or not str(tag).isdigit():
                save_pv_vis(gs_modules, motion_modules, str(tag), cfg.work_dir)
        if camera_params is not None:
            dump_cpkl(str(ckpt_path) + ".cam", camera_params)
            guru.info(f"Saving cam params to {ckpt_path}.cam")
            dump_cpkl(str(ckpt_path) + f".{tag}.cam", camera_params)
            guru.info(f"Saving camera params to {ckpt_path}.{tag}.cam")
    
    camera_name_override = {}
    if cfg.camera_name_override:
        for cam_name in cfg.camera_name_override.split(","):
            a, b = cam_name.split(":")
            camera_name_override[a] = b

    train_datasets, train_video_view, val_img_dataset = get_train_val_datasets(cfg.data, load_val=True, cameras=cfg.cameras, 
                                                                               camera_name_override=camera_name_override)

    device = "cuda"
    os.makedirs(cfg.work_dir, exist_ok=True)
    with open(f"{cfg.work_dir}/cfg.yaml", "w") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False)

    if cfg.resume_if_possible:
        gs_modules, motion_modules, dict_of_track3ds, gaussian_names = load_cpkl(ckpt_path)
    else:
        gs_modules, motion_modules, dict_of_track3ds, gaussian_names = init_util.initialize_model(train_datasets, **asdict(cfg))
        guru.info("saving initialized model to {}".format(ckpt_path))
        save_checkpoint("init")

    # track-based motion pretraining
    has_kp2d = any(['kps2d' in D[0] for D in train_datasets])
    if "motion_pretrained" not in cfg.resume_if_possible:
        assert dict_of_track3ds is not None
        motion_modules = motion_modules.to(device)
        gs_modules = gs_modules.to(device)
        if cfg.data.use_tracks and not cfg.init_with_rgbd:
            gs_modules, motion_modules = run_motion_pretrain(train_datasets, gs_modules, motion_modules, dict_of_track3ds, device=device, **asdict(cfg))
            save_checkpoint("motion_pretrained")
        elif has_kp2d:
            gs_modules, motion_modules = run_motion_pretrain(train_datasets, gs_modules, motion_modules, None, device=device, **asdict(cfg))
            save_checkpoint("motion_pretrained")
        else:
            guru.info("skipping motion pretraining")
    else:
        gs_modules, motion_modules, dict_of_track3ds, gaussian_names = load_cpkl(ckpt_path)
        motion_modules = motion_modules.to(device)
        gs_modules = gs_modules.to(device)

    if osp.exists(str(ckpt_path) + ".cam"): camera_state_dict = load_cpkl(str(ckpt_path) + ".cam")
    else: camera_state_dict = None

    gc.collect()
    torch.cuda.empty_cache()

    if cfg.loss.w_minimal_movement_in_cano < 0:
        for m in motion_modules.values():
            initial_frames = [m.cano_t]
            if hasattr(m, 'preinit_joints'):
                initial_frames += list(m.preinit_joints.keys())
            initial_frames = sorted(initial_frames)
            m.save_initial_params(initial_frames)

    run_full_training(gs_modules, motion_modules, gaussian_names, train_datasets,
                      train_video_view,
                      val_img_dataset, cfg, device=device, save_callback=save_checkpoint, camera_state_dict=camera_state_dict)
    if not cfg.eval_only:
        save_checkpoint("end")



if __name__ == "__main__":
    main(tyro.cli(TrainConfig))

