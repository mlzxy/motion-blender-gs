from motionblender.lib.params import GaussianParams
import motionblender.lib.misc as misc
from flow3d.metrics import mPSNR, mSSIM, mLPIPS
from flow3d.data.utils import to_device
from PIL import Image
from torch.utils.data import DataLoader
import random
import numpy as np
import motionblender.lib.pv as pv
import motionblender.lib.kps_vis as kp_vis
from einops import rearrange, repeat, einsum
from tqdm.auto import tqdm, trange
import torch.optim as optim
import torch.nn as nn
from pathlib import Path
from loguru import logger as guru
from motionblender.lib.dataset import MotionBlenderDataset, normalize_coords, iPhoneDataset
from motionblender.lib.motion import MotionBlender, MotionBlenderType
from flow3d.tensor_dataclass import StaticObservations, TensorDataclass, Self
from dataclasses import dataclass, asdict
import torch
import roma 
import torch.nn.functional as F
try:
    from flow3d.init_utils import knn
except Exception as e:
    guru.warning(str(e))
    guru.warning("skipping this exception to continue the code without `cuml`")
import motionblender.lib.init_graph.deformable_graph as defg_init
from motionblender.lib.misc import remap_values, render, loopy
import motionblender.lib.ctrl as ctrl_lib
from motionblender.lib.params import merge_splat_dicts
import motionblender.lib.init_graph.human_kinematic_tree_lifting as khuman_init
import motionblender.lib.convert_utils as cvt
from flow3d.vis.utils import project_2d_tracks
import motionblender.lib.animate as anim
import motionblender.lib.pv as pv


def init_gs(
    points: StaticObservations, 
    filter_outliers: bool = False, 
    init_opacity: float = 0.7,
    use_sh_color: bool = False
) -> GaussianParams:
    """
    using dataclasses instead of individual tensors so we know they're consistent
    and are always masked/filtered together
    """
    num_init_bg_gaussians = points.xyz.shape[0]
    bg_scene_center = points.xyz.mean(0)
    bg_points_centered = points.xyz - bg_scene_center
    bg_min_scale = bg_points_centered.quantile(0.05, dim=0)
    bg_max_scale = bg_points_centered.quantile(0.95, dim=0)
    bg_scene_scale = torch.max(bg_max_scale - bg_min_scale).item() / 2.0
    assert points.colors.max() <= 1.0 and points.colors.min() >= 0.0, "colors should be normalized to [0, 1]"
    bkdg_colors = points.colors

    # Initialize gaussian scales: find the average of the three nearest
    # neighbors in the first frame for each point and use that as the
    # scale.
    dists, _ = knn(points.xyz, 3)
    dists = torch.from_numpy(dists)
    bg_scales = dists.mean(dim=-1, keepdim=True)
    bkdg_scales = torch.log(bg_scales.repeat(1, 3))

    bg_means = points.xyz

    # Initialize gaussian orientations by normals.
    if points.normals is not None:
        local_normals = points.normals.new_tensor([[0.0, 0.0, 1.0]]).expand_as(
            points.normals
        )
        bg_quats = roma.rotvec_to_unitquat(
            F.normalize(local_normals.cross(points.normals), dim=-1)
            * (local_normals * points.normals).sum(-1, keepdim=True).acos_()
        ).roll(1, dims=-1)
    else:
        bg_quats = torch.zeros(num_init_bg_gaussians, 4)
        bg_quats[:, 0] = 1
    
    bg_opacities = torch.logit(torch.full((num_init_bg_gaussians,), init_opacity))

    if filter_outliers:
        outlier_mask = torch.isinf(bkdg_scales.mean(dim=1))
        valid_min_scale = bkdg_scales[~outlier_mask].min() 
        bkdg_scales[outlier_mask] = valid_min_scale

        # valid_mask = ~torch.isinf(bkdg_scales.mean(dim=1))
        # bg_means = bg_means[valid_mask]
        # bg_quats = bg_quats[valid_mask]
        # bkdg_scales = bkdg_scales[valid_mask]
        # bkdg_colors = bkdg_colors[valid_mask]
        # bg_opacities = bg_opacities[valid_mask]
        assert not bkdg_scales.min().isinf()
    
    gaussians = GaussianParams(
        bg_means,
        bg_quats,
        bkdg_scales,
        bkdg_colors if use_sh_color else torch.logit(bkdg_colors),
        bg_opacities,
        scene_center=bg_scene_center,
        scene_scale=bg_scene_scale,
        use_sh_color=use_sh_color,
    )
    return gaussians


@dataclass
class TrackObservations(TensorDataclass):
    xyz: torch.Tensor # all_tracks, frames, 3
    visibles: torch.Tensor # all_tracks, frames
    invisibles: torch.Tensor # all_tracks, frames
    confidences: torch.Tensor # all_tracks, frames
    colors: torch.Tensor # all_tracks, 3
    iids: torch.Tensor # all_tracks

    def check_sizes(self) -> bool:
        dims = self.xyz.shape[:-1]
        return (
            self.visibles.shape == dims 
            and self.iids.shape == dims 
            and self.invisibles.shape == dims
            and self.confidences.shape == dims
            and self.colors.shape[:-1] == dims[:-1]
            and self.xyz.shape[-1] == 3
            and self.colors.shape[-1] == 3
        )

    def filter_valid(self, valid_mask: torch.Tensor) -> Self:
        return self.map(lambda x: x[valid_mask])


def concate_tracks(*track3ds: TrackObservations) -> TrackObservations:
    for k in ['xyz', 'visibles', 'invisibles', 'confidences', 'colors', 'iids']:
        assert all(getattr(track3d, k).shape[1:] == getattr(track3ds[0], k).shape[1:] for track3d in track3ds)
    return TrackObservations(
        torch.cat([track3d.xyz for track3d in track3ds], dim=0),
        torch.cat([track3d.visibles for track3d in track3ds], dim=0),
        torch.cat([track3d.invisibles for track3d in track3ds], dim=0),
        torch.cat([track3d.confidences for track3d in track3ds], dim=0),
        torch.cat([track3d.colors for track3d in track3ds], dim=0),
        torch.cat([track3d.iids for track3d in track3ds], dim=0))


def interpolate_tracks(track3d: TrackObservations, time_ids: torch.Tensor, num_frames: int) -> TrackObservations:
    num_all_tracks = len(track3d.xyz)
    new_xyz = torch.zeros(num_all_tracks, num_frames, 3)
    new_visibles = torch.zeros(num_all_tracks, num_frames)
    new_invisibles = torch.zeros(num_all_tracks, num_frames)
    new_confidences = torch.zeros(num_all_tracks, num_frames)
    available_times = {ti: i for i, ti in enumerate(time_ids.long().tolist())}

    def find_prev_time(ti):
        while ti >= 0:
            if ti in available_times:
                return ti
            ti -= 1
        return -1
    
    def find_next_time(ti):
        while ti < num_frames:
            if ti in available_times:
                return ti
            ti += 1
        return -1

    for t in range(num_frames):
        if t in available_times:
            _xyz = track3d.xyz[:, available_times[t]]
            _visibles = track3d.visibles[:, available_times[t]]
            _invisibles = track3d.invisibles[:, available_times[t]]
            _confidences = track3d.confidences[:, available_times[t]]
        else:
            prev_t = find_prev_time(t)
            next_t = find_next_time(t)
            if prev_t == -1:
                assert next_t != -1
                _xyz = track3d.xyz[:, available_times[next_t]]
                _visibles = track3d.visibles[:, available_times[next_t]]
                _invisibles = track3d.invisibles[:, available_times[next_t]]
                _confidences = track3d.confidences[:, available_times[next_t]]
            elif next_t == -1:
                assert prev_t != -1
                _xyz = track3d.xyz[:, available_times[prev_t]]
                _visibles = track3d.visibles[:, available_times[prev_t]]
                _invisibles = track3d.invisibles[:, available_times[prev_t]]
                _confidences = track3d.confidences[:, available_times[prev_t]]
            else:
                # interpolate between prev and next
                prev_xyz = track3d.xyz[:, available_times[prev_t]]
                prev_visibles = track3d.visibles[:, available_times[prev_t]]
                prev_invisibles = track3d.invisibles[:, available_times[prev_t]]
                prev_confidences = track3d.confidences[:, available_times[prev_t]]

                next_xyz = track3d.xyz[:, available_times[next_t]]
                next_visibles = track3d.visibles[:, available_times[next_t]]
                next_invisibles = track3d.invisibles[:, available_times[next_t]]
                next_confidences = track3d.confidences[:, available_times[next_t]]

                w_prev = (next_t - t) / (next_t - prev_t)
                w_next = (t - prev_t) / (next_t - prev_t)

                # _xyz = prev_xyz * w_prev + next_xyz * w_next
                _visibles = (prev_invisibles.bool() & next_visibles.bool()).to(next_visibles.dtype)
                _invisibles = (prev_visibles.bool() | next_invisibles.bool()).to(next_invisibles.dtype)
                _confidences = prev_confidences * w_prev + next_confidences * w_next
                _xyz = prev_xyz * w_prev + next_xyz * w_next

        new_xyz[:, t] = _xyz
        new_visibles[:, t] = _visibles
        new_invisibles[:, t] = _invisibles
        new_confidences[:, t] = _confidences

    return TrackObservations(new_xyz, new_visibles, new_invisibles, new_confidences, track3d.colors, track3d.iids)


def compute_most_visibles(track3d: TrackObservations):
    return track3d.visibles.sum(dim=0).max().item()



@torch.no_grad()
def initialize_model(train_datasets: list[MotionBlenderDataset], num_bg=-1, num_fg=-1, rigging_init_steps=-1, 
                     num_tracks_per_frame=800, min_pts_per_inst=10000, num_vertices_for_deformable=100, 
                     init_gamma: float=4.0, init_temperature:float=0.01, blend_method: str='dq',
                     length_per_link_learnable: bool=False, nearest_k_links: int=5, use_radiance_kernel:str="none",
                     voxel_downsample_size=0.05, deformable_link_quantize: int=-1, 
                     simple_finger: str = 'no', fg_only: bool=False,  init_with_only_one_camera:bool=True, init_with_rgbd:bool = False, 
                     use_sh_color: bool = False, lr: ctrl_lib.LRConfig = None, ctrl: ctrl_lib.ControlConfig = None,
                     num_workers: int = 4, batch_size: int = 8, work_dir: str = None, render_engine: str='gsplat', 
                     skip_preprocess_pcd_clustering: list[str] = [], skip_global_procruste: bool = False, **_)\
    -> tuple[nn.ModuleDict, dict[str, MotionBlender], list[str]]:

    def get_num_vertices_for_deformable(inst_id: int) -> int:
        num_vertices = -1
        if isinstance(num_vertices_for_deformable, int):
            num_vertices = num_vertices_for_deformable        
        elif isinstance(num_vertices_for_deformable, list):
            if len(num_vertices_for_deformable) == 1:
                num_vertices = num_vertices_for_deformable[0]
            else:
                num_vertices = num_vertices_for_deformable[inst_id - 1]
        return num_vertices

    def get_bkgd_points(num_bg, instance='bg'):
        bkgd_points, bkgd_point_normals, bkgd_point_colors = [], [], []
        for train_dataset in train_datasets:
            _1, _2, _3 = train_dataset.get_bkgd_points(1 + num_bg // len(train_datasets), instance=instance)
            bkgd_points.append(_1)
            bkgd_point_normals.append(_2)
            bkgd_point_colors.append(_3)
        return torch.cat(bkgd_points, dim=0), torch.cat(bkgd_point_normals, dim=0), torch.cat(bkgd_point_colors, dim=0)
    
    def get_rad_kernel(inst_name, inst_type):
        u_rad_kernel = False
        if isinstance(use_radiance_kernel, str) and inst_type in use_radiance_kernel:
            u_rad_kernel = True
            guru.info(f"use radience kernel for kinematic instance {inst_name}")
        return u_rad_kernel
    
    def register_remaining_preinit_motions(inst_preinit_motion_graphs, motion):
        for _frame_t in inst_preinit_motion_graphs:
            if _frame_t != cano_t:
                if not hasattr(motion, 'preinit_joints'): motion.preinit_joints = {}
                guru.warning(f"use pre-init motion graph for {inst_name} at frame {_frame_t}")
                preinit_graph_t = inst_preinit_motion_graphs[_frame_t]
                t_joints = torch.as_tensor(preinit_graph_t['joints'])
                if motion.type == MotionBlenderType.kinematic:
                    t_root_joint = t_joints[root_joint_id].clone()
                    # t_joints -= t_root_joint
                    chain = anim.inverse_kinematic(t_joints - t_root_joint, links, length_inv_activation=torch.log)
                    t_rot6d_tensor = anim.retrieve_tensor_from_chain(chain, 'rot6d')
                    motion.global_ts.data[_frame_t] = t_root_joint
                    motion.rot6d.data[_frame_t] = t_rot6d_tensor
                    motion.preinit_joints[_frame_t] = {'joints': t_joints.cuda()}
                elif motion.type == MotionBlenderType.deformable:
                    # it is possible that the preinit graphs is not complete
                    # in this case, just save the joints and use it for regularization
                    if 'indices' not in preinit_graph_t:
                        t_offset = t_joints.mean(dim=0)
                        # t_joints -= t_offset
                        motion.joints.data[_frame_t] = t_joints - t_offset
                        motion.global_ts.data[_frame_t] = t_offset
                        motion.preinit_joints[_frame_t] = {'joints': t_joints.cuda()}

                if 'indices' in preinit_graph_t:
                    motion.preinit_joints[_frame_t] = {
                        'indices': torch.as_tensor(preinit_graph_t['indices']).cuda(),
                        'joints': t_joints.cuda()
                    }
    
    def init_kinematic_tree_just_from_keypoints(t, D:MotionBlenderDataset, iid):
        rgb = D.imgs[t]
        depth = D.depths[t]
        img_h, img_w = D.imgs[t].shape[:2]
        pts, rgbs = cvt.get_pointcloud_from_rgb_depth_cam(rgb, depth, torch.inverse(D.w2cs[t]), D.Ks[t], img_w, img_h)
        joints_2d = torch.as_tensor(D.instance_id_2_keypoints[iid][t])
        # kp_vis.draw_keypoints((rgb * 255).permute(2, 0, 1).to(torch.uint8), joints_2d).save('outputs/test.png')
        links = D.keypoint_links
        pts_hw = pts.reshape(img_h, img_w, 3)
        joints = pts_hw[joints_2d[:, 1].long(), joints_2d[:, 0].long()]

        root_joint_id = anim.find_root_joint_id(links, n_joints=len(joints))
        global_t = joints[root_joint_id].clone()
        joints = joints - global_t.reshape(1, 3)

        chain = anim.inverse_kinematic(joints, links, length_inv_activation=torch.log)
        links_tensor = torch.as_tensor(links).long()
        hollow_chain = anim.create_hollow_chain_wo_tensor(chain)

        length_tensor, length_linkid2indice = anim.retrieve_tensor_from_chain(chain, 'length', return_linkid2indice=True)
        rot6d_tensor, rot6d_linkid2indice = anim.retrieve_tensor_from_chain(chain, 'rot6d', return_linkid2indice=True)

        return {
            'length': length_tensor,
            'scale': torch.log(torch.ones(1)),
            'links': links_tensor,
            'rot6d': rot6d_tensor,
            'global_t': global_t
        }, {'length_linkid2indice': length_linkid2indice, 'rot6d_linkid2indice': rot6d_linkid2indice, 'hollow_chain': hollow_chain}

    use_tracks = train_datasets[0].use_tracks
    gaussian_names = []
    dict_of_gaussians, dict_of_motions, dict_of_track3ds = {}, {}, {}
    if init_with_rgbd:
        if init_with_only_one_camera:
            train_datasets = [train_datasets[0]]
        use_tracks = False

    if use_tracks:
        guru.info("loading tracks")
        # merging tracks from all datasets
        track3ds = [TrackObservations(*train_dataset.get_instance_tracks_3d(num_tracks_per_frame)) for train_dataset in train_datasets]
        if len(track3ds) > 1:
            guru.warning("merging 3d tracks from multiple cameras, this is experimental")
            track3ds = [interpolate_tracks(track3d, dataset.time_ids, dataset.num_all_frames_in_scene) for track3d, dataset in 
                        tqdm(zip(track3ds, train_datasets), total=len(track3ds), desc="intepolating 3d tracks from each dataset")]
            if init_with_only_one_camera:
                track3d = track3ds[0]
            else:
                all_track3d = concate_tracks(*track3ds)
                track3d = all_track3d
        else:
            track3d = track3ds[0]
        cano_t = int(track3d.visibles.sum(dim=0).argmax().item())
        given_cano_t = train_datasets[0].given_cano_t
        if given_cano_t >= 0:
            guru.warning(f"track cano t is {cano_t}, but given cano t is {given_cano_t}, use the given one")
            cano_t = given_cano_t
        means_cano = track3d.xyz[:, cano_t].clone() 
        num_frames = track3d.xyz.shape[1]
        assert num_frames == train_datasets[0].num_all_frames_in_scene
    else:
        guru.info("not using tracks, initializing model with the first dataset")
        track3d = None
        cano_t = train_datasets[0].given_cano_t
        num_frames = train_datasets[0].num_all_frames_in_scene

    if use_tracks:
        guru.info("initializing instances with 3d tracks")
        preinit_motion_graphs =  getattr(train_datasets[0], 'preinited_motion_graphs', None)

        for inst_id in train_datasets[0].instance_ids:
            inst_name = f"{train_datasets[0].instance_id_2_classname[inst_id]}-{inst_id}"
            motion_type = train_datasets[0].instance_id_2_motion_type[inst_id]
            assert motion_type in [MotionBlenderType.deformable, MotionBlenderType.kinematic, MotionBlenderType.rigid, MotionBlenderType.static]

            if motion_type == MotionBlenderType.static: 
                static_points = StaticObservations(*get_bkgd_points(10000, instance=inst_id))
                assert static_points.check_sizes()
                dict_of_gaussians[inst_name] = init_gs(static_points, use_sh_color=use_sh_color, init_opacity=0.1)
                gaussian_names.append(inst_name)
                guru.info(f"Initialized GS {inst_name} with {len(static_points.xyz)} points")
                continue
            else:
                track_mask = (track3d.iids.long() // 1000) == inst_id
                budget = max(min_pts_per_inst, int(num_fg * track_mask.sum() / len(track_mask)))

                rgbs = track3d.colors[track_mask]
                pts = means_cano[track_mask]
                budget = min(budget, len(rgbs))
                indices = torch.randperm(len(rgbs))[:budget]
                sampled_rgbs = rgbs[indices]
                sampled_pts = pts[indices]

                dict_of_gaussians[inst_name] = init_gs(StaticObservations(sampled_pts, None, sampled_rgbs), filter_outliers=fg_only, use_sh_color=use_sh_color)
                gaussian_names.append(inst_name)
                guru.info(f"Initialized GS {inst_name} with {len(sampled_rgbs)} points")

            dict_of_track3ds[inst_name] = track3d.map(lambda x: x[track_mask][indices])
            num_vertices = get_num_vertices_for_deformable(inst_id)

            if motion_type == MotionBlenderType.rigid:
                guru.info(f"initializing rigid motion for {inst_name}")
                cano_offset = sampled_pts.mean(dim=0)
            elif motion_type == MotionBlenderType.deformable:
                guru.info(f"initializing deformable motion for {inst_name}")
                if preinit_motion_graphs is not None and inst_id in preinit_motion_graphs:
                    joints = torch.as_tensor(preinit_motion_graphs[inst_id][cano_t]['joints']).float()
                    tri_links = torch.as_tensor(preinit_motion_graphs[inst_id][cano_t]['links']).long()
                else:
                    skip_clusters_for_inst = inst_name in skip_preprocess_pcd_clustering
                    if skip_clusters_for_inst:
                        guru.warning(f"skipping preprocess point cloud clustering for {inst_name}, the result pcd maybe more noisy")
                    debug_pcds, traced_indices = defg_init.remove_outliers_from_pointcloud(pts, rgbs, voxel_downsample_size=voxel_downsample_size, skip_clusters=skip_clusters_for_inst)
                    joints, tri_links = defg_init.build_deformable_graph_from_dense_points(debug_pcds[-1], num_vertices=num_vertices)
                    joints, tri_links = torch.from_numpy(joints).float(), torch.from_numpy(tri_links).long()
                cano_offset = joints.mean(dim=0)
                joints -= cano_offset
            elif motion_type == MotionBlenderType.kinematic:
                if preinit_motion_graphs is not None and inst_id in preinit_motion_graphs:
                    joints_lst = preinit_motion_graphs[inst_id][cano_t]['joints']
                    links = preinit_motion_graphs[inst_id][cano_t]['links']
                    root_joint_id = preinit_motion_graphs[inst_id][cano_t].get('root_id', 0)

                    joints = torch.as_tensor(joints_lst)
                    root_joint = joints[root_joint_id].clone()
                    joints -= root_joint

                    assert [l[1] for l in links] == list(range(1, len(joints))), "link ids must equal to joint ids plus 1"

                    chain = anim.inverse_kinematic(joints, links, length_inv_activation=torch.log)
                    links_tensor = torch.as_tensor(links).long()
                    hollow_chain = anim.create_hollow_chain_wo_tensor(chain)

                    length_tensor, length_linkid2indice = anim.retrieve_tensor_from_chain(chain, 'length', return_linkid2indice=True)
                    rot6d_tensor, rot6d_linkid2indice = anim.retrieve_tensor_from_chain(chain, 'rot6d', return_linkid2indice=True)
                    cano_offset = root_joint
                    local_joint_names = preinit_motion_graphs[inst_id][cano_t].get('joint_names', [f'joint-{_v}' for _v in range(len(joints))])
                else:
                    if 'person' in inst_name: # the IPHONE branch
                        guru.info(f"initializing kinematic motion for {inst_name}, lifting 2d human skeleton to 3d")
                        LIFTING_THRESHOLD = 0.002
                        track_link_id = track3d.iids[track_mask].long() % 1000
                        track_pixel_to_2dlink_dist = torch.frac(track3d.iids[track_mask]) * 4

                        track_link_id = khuman_init.cleanup_unreliable_links(track_link_id, track_pixel_to_2dlink_dist, thr=LIFTING_THRESHOLD, simple_finger=simple_finger == 'yes')

                        track_link_id_valid_mask = track_link_id != 500
                        track_link_id = track_link_id[track_link_id_valid_mask]
                        track_pixel_to_2dlink_dist = track_pixel_to_2dlink_dist[track_link_id_valid_mask]
                        valid_pts = pts[track_link_id_valid_mask]
                        valid_rgbs = rgbs[track_link_id_valid_mask]
                        
                        found_link_ids = torch.unique(track_link_id).tolist()
                        if 500 in found_link_ids: found_link_ids.remove(500) 
                        guru.info(f"found {len(found_link_ids)} links in {inst_name}")

                        hollow_chain, length_tensor, rot6d_tensor, rot6d_linkid2indice, length_linkid2indice, links_tensor, \
                            links_globalid2local_ind, joint_local_ind2name = khuman_init.initialize_kinematic_tree_from_links(only_keep_link_ids=found_link_ids)
                        
                        local_joint_ids = set()
                        for _i in range(len(links_globalid2local_ind)):
                            # NOTE: print names to see what links you have
                            linklet = links_tensor[int(links_globalid2local_ind[_i][1])]
                            assert khuman_init.wholebody_connections[links_globalid2local_ind[_i][0]][-1] == joint_local_ind2name[int(linklet[1])]
                            for a in linklet: local_joint_ids.add(int(a))
                        
                        if len(track_link_id.unique()) != len(links_globalid2local_ind):
                            # certain tracks need to be filtered out
                            keep_mask = torch.isin(track_link_id, links_globalid2local_ind[:, 0])
                            track_link_id = track_link_id[keep_mask]
                            track_pixel_to_2dlink_dist = track_pixel_to_2dlink_dist[keep_mask]
                            valid_pts = valid_pts[keep_mask]
                            valid_rgbs = valid_rgbs[keep_mask]
                        
                        # get local joint names
                        local_joint_ids = sorted(local_joint_ids)
                        local_joint_names = [joint_local_ind2name[_j] for _j in local_joint_ids]
                        
                        local_track_link_id = remap_values(track_link_id, links_globalid2local_ind[:, 0], links_globalid2local_ind[:, 1])
                        
                        with torch.enable_grad():
                            param = nn.ParameterDict({
                                'global_t': valid_pts.mean(0),
                                'global_rot6d': nn.Parameter(anim.rmat_to_cont_6d(torch.eye(3)).float(), requires_grad=False),
                                'length': nn.Parameter(length_tensor, requires_grad=False),
                                'scale': torch.log(torch.ones(1)),
                                'rot6d': rot6d_tensor,
                                'links': nn.Parameter(links_tensor, requires_grad=False),
                            }).cuda()
                            meta = {
                                'length_linkid2indice': length_linkid2indice,
                                'rot6d_linkid2indice': rot6d_linkid2indice,
                                'hollow_chain': hollow_chain,
                                'links_globalid2local_ind': links_globalid2local_ind,
                            }
                            buff = { 'pts': valid_pts[track_pixel_to_2dlink_dist <= LIFTING_THRESHOLD].cuda(),  # NOTE: visualize this to see how your kp gonna look
                                    'link_assignments': local_track_link_id[track_pixel_to_2dlink_dist < LIFTING_THRESHOLD].cuda(),}
                            opt = optim.Adam(param.parameters(), lr=5e-3)
                            for i in tqdm(range(rigging_init_steps), desc=f"initializing kinematic tree [{inst_name}]"):
                                opt.zero_grad()
                                loss_dict = {}

                                diff_chain = anim.fill_hollow_chain_with_tensor(meta['hollow_chain'], 
                                                            param['length'] * torch.exp(param['scale']), param['rot6d'], 
                                                            meta['rot6d_linkid2indice'], meta['length_linkid2indice'])
                                global_T = anim.rt_to_mat4(anim.cont_6d_to_rmat(param['global_rot6d']), param['global_t'])
                                t_rel_link_poses = anim.forward_kinematic(diff_chain)
                                t_link_poses = anim.apply_mat4_pose(global_T, t_rel_link_poses)
                                    
                                pred_joint_poses = anim.link_poses_to_joint_positions(t_link_poses, meta['hollow_chain']['id'], global_T[:3, 3])  
                                pred_joint_poses[param['links'][:, 0]], pred_joint_poses[param['links'][:, 1]]

                                distances, _ , _ = anim.compute_distance_from_link(buff['pts'], pred_joint_poses[param['links'][:, 0]], pred_joint_poses[param['links'][:, 1]])
                                pts_to_target_link_dist = torch.gather(distances, 1, buff['link_assignments'].reshape(-1, 1))

                                loss_dict['lifting to 3d points l1'] = pts_to_target_link_dist.mean()
                                sum(loss_dict.values()).backward()
                                opt.step()
                                if i % 50 == 0: guru.info(f"(rigging for {inst_name}) "  + (", ".join([f"{k}: {v.item():.4f}" for k, v in loss_dict.items()])))
                            
                            # plotter = pv.Plotter(backend="remote:/dev/shm/xz653/pvstate.rigging")
                            # plotter.update_param("colors", valid_rgbs[track_pixel_to_2dlink_dist <= LIFTING_THRESHOLD])
                            # plotter.update_param("means", [buff['pts']])
                            # plotter.update_param("graph.links", param['links'].detach().cpu())
                            # plotter.update_param('graph.joints', [pred_joint_poses])
                            # plotter.render()
                    else:
                        param, meta = init_kinematic_tree_just_from_keypoints(cano_t, train_datasets[0], inst_id)
                        hollow_chain = meta['hollow_chain']
                    cano_offset = param['global_t'].detach().cpu()
            else:
                raise ValueError(f"Unknown motion type: {motion_type}")
            
            # initialize global motion
            if skip_global_procruste:
                global_rot6d = anim.rmat_to_cont_6d(torch.eye(3))
                global_ts = torch.zeros(3)
                global_ts += cano_offset
                global_rot6d = repeat(global_rot6d, 'a -> t a', t=num_frames).clone()
                global_ts = repeat(global_ts, 'a -> t a', t=num_frames).clone()
            else:
                guru.info(f"initialize global motion for {inst_name}")
                global_ts, global_rot6d = [], []
                for t in trange(num_frames, leave=False, desc='solving procrustes'):
                    if t != cano_t:
                        source = sampled_pts
                        target =  track3d.xyz[:, t][track_mask][indices]
                        T = anim.solve_procrustes(source[None], target[None])
                        R, trans = anim.mat4_to_rt(T[0])
                        rot6d = anim.rmat_to_cont_6d(R)
                    else:
                        rot6d = anim.rmat_to_cont_6d(torch.eye(3))
                        trans = torch.zeros(3)
                    trans += cano_offset
                    global_ts.append(trans)
                    global_rot6d.append(rot6d)
                global_ts, global_rot6d = torch.stack(global_ts), torch.stack(global_rot6d)
            
            global_T = anim.rt_to_mat4(anim.cont_6d_to_rmat(global_rot6d), global_ts)
            if motion_type == MotionBlenderType.rigid:
                motion = MotionBlender(num_frames, cano_t, global_rot6d, global_ts, type=MotionBlenderType.rigid)
            elif motion_type == MotionBlenderType.deformable:
                motion = MotionBlender(num_frames, cano_t, global_rot6d, global_ts, type=MotionBlenderType.deformable,
                                    links=tri_links, init_gamma=init_gamma, init_temperature=init_temperature, nearest_k=nearest_k_links,
                                    blend_method=blend_method, joints=repeat(joints, "j c -> t j c", t=num_frames).clone(), 
                                    use_radiance_kernel=get_rad_kernel(inst_name, 'deformable'), deformable_link_quantize=deformable_link_quantize)
            else:
                if preinit_motion_graphs is not None and inst_id in preinit_motion_graphs:
                    motion = MotionBlender(num_frames, cano_t, global_rot6d, global_ts, type=MotionBlenderType.kinematic,
                            links=links_tensor, init_gamma=init_gamma, init_temperature=init_temperature,
                            blend_method=blend_method, hollow_chain=hollow_chain, length_tensor=length_tensor, 
                            length_scale_tensor=None, 
                            rot6d_tensor=repeat(rot6d_tensor, 'a b -> t a b', t=num_frames).clone(), 
                            rot6d_linkid2indice=rot6d_linkid2indice, nearest_k=nearest_k_links,
                            length_linkid2indice=length_linkid2indice, length_per_link_learnable=True, use_radiance_kernel=get_rad_kernel(inst_name, 'kinematic'))
                    motion.length.requires_grad = False
                else:
                    length_tensor = param['length'].detach().cpu()
                    if length_per_link_learnable:
                        log_scale = param['scale'].detach().cpu()
                        length_tensor *= torch.exp(log_scale)
                        length_tensor = torch.log(length_tensor)
                    motion = MotionBlender(num_frames, cano_t, global_rot6d, global_ts, type=MotionBlenderType.kinematic,
                        links=param['links'].detach().cpu(), init_gamma=init_gamma, init_temperature=init_temperature,
                        blend_method=blend_method, hollow_chain=hollow_chain, length_tensor=length_tensor, 
                        length_scale_tensor=None if length_per_link_learnable else param['scale'].detach().cpu(), 
                        rot6d_tensor=repeat(param['rot6d'].detach().cpu(), 'a b -> t a b', t=num_frames).clone(), 
                        rot6d_linkid2indice=meta['rot6d_linkid2indice'], nearest_k=nearest_k_links,
                        length_linkid2indice=meta['length_linkid2indice'], length_per_link_learnable=length_per_link_learnable, use_radiance_kernel=get_rad_kernel(inst_name, 'kinematic'))
                    
                if 'person' in inst_name:
                    motion.keypoint_names = khuman_init.wholebody_keypoints # the IPHONE branch
                else:
                    motion.keypoint_names = train_datasets[0].keypoint_names # must come from keypoints folder
                motion.joint_names = local_joint_names # keypoint_names -> all keypoints, joint_names -> available ones
            dict_of_motions[inst_name] = motion
            if preinit_motion_graphs is not None and inst_id in preinit_motion_graphs:
                register_remaining_preinit_motions(preinit_motion_graphs[inst_id], motion)
                
            guru.info(f"finalize the motion initialization of {inst_name}")
    else:
        guru.info("initializing instances with RGB-D")
        train_dataset = train_datasets[0]
        if cano_t < 0:
            cano_img_id = (train_dataset.instance_masks > 0).flatten(1).sum(1).argmax()
            cano_t = int(train_dataset.time_ids[cano_img_id])
            guru.info(f"the canonical frame is {cano_t} (with the maximal foreground mask)")
        else:
            cano_img_id = (train_dataset.time_ids == cano_t).nonzero().flatten().item()

        img_h, img_w = train_dataset.imgs[cano_img_id].shape[:2]

        pts_list, rgbs_list, imasks_list = [], [], []
        for D in train_datasets: # iterate all datasets, get the whole point cloud at time cano_t
            cano_img_id = (D.time_ids == cano_t).nonzero().flatten().item()
            rgb = D.imgs[cano_img_id]
            depth = D.depths[cano_img_id]
            pts, rgbs = cvt.get_pointcloud_from_rgb_depth_cam(rgb, depth, torch.inverse(D.w2cs[cano_img_id]), 
                                                D.Ks[cano_img_id], img_w, img_h)
            imask = D.instance_masks[cano_img_id]
            rgbs_list.append(rgbs)
            pts_list.append(pts)
            imasks_list.append(imask.flatten())

        pts, rgbs, imasks = torch.cat(pts_list), torch.cat(rgbs_list), torch.cat(imasks_list)

        predefined_motion_graphs = train_dataset.predefined_motion_graphs
        preinit_motion_graphs =  train_dataset.preinited_motion_graphs

        all_inst_pts = 0
        for inst_id in train_dataset.instance_ids: all_inst_pts += (imasks == inst_id).sum()

        for inst_id in train_dataset.instance_ids:
            inst_name = f"{train_dataset.instance_id_2_classname[inst_id]}-{inst_id}"
            motion_type = train_dataset.instance_id_2_motion_type[inst_id]

            masks = (imasks == inst_id).flatten()
            inst_pts, inst_rgbs = pts[masks], rgbs[masks]
            budget = max(min_pts_per_inst, int(num_fg * len(inst_pts) / all_inst_pts))
            budget = min(budget, len(inst_rgbs))
            indices = torch.randperm(len(inst_rgbs))[:budget]
            sampled_rgbs = inst_rgbs[indices]
            sampled_pts = inst_pts[indices]
            dict_of_gaussians[inst_name] = init_gs(StaticObservations(sampled_pts, None, sampled_rgbs), 
                                                   filter_outliers=True, 
                                                   use_sh_color=use_sh_color, init_opacity=0.1)
            gaussian_names.append(inst_name)

            if predefined_motion_graphs is not None and inst_id in predefined_motion_graphs: 
                assert not train_dataset.normalize_scene, "predefined motion graph is not supported for normalized scene at this moment"
                assert motion_type == MotionBlenderType.kinematic
                inst_predefined_motion_graphs = predefined_motion_graphs[inst_id]
                links_list = inst_predefined_motion_graphs[0]['links']
                links_tensor = torch.as_tensor(links_list)
                global_rot6d = anim.rmat_to_cont_6d(torch.eye(3)).reshape(1, 6).repeat(num_frames, 1)
                global_ts = torch.zeros(num_frames, 3)
                rot6ds = []

                for a in tqdm(inst_predefined_motion_graphs, desc='reading predefined motion graphs'):
                    tensor_chain = anim.inverse_kinematic(a['joints'], links_list)
                    length_tensor, length_linkid2indice = anim.retrieve_tensor_from_chain(tensor_chain, 'length', return_linkid2indice=True)
                    _rot6d, rot6d_linkid2indice = anim.retrieve_tensor_from_chain(tensor_chain, 'rot6d', return_linkid2indice=True)
                    rot6ds.append(_rot6d)

                hollow_chain = anim.create_hollow_chain_wo_tensor(tensor_chain)
                rot6ds = torch.stack(rot6ds)

                motion = MotionBlender(num_frames, cano_t, global_rot6d, global_ts, type=MotionBlenderType.kinematic,
                    links=links_tensor, init_gamma=init_gamma, init_temperature=init_temperature,
                    blend_method=blend_method, 

                    hollow_chain=hollow_chain, length_tensor=torch.log(length_tensor), length_scale_tensor=None, rot6d_tensor=rot6ds, 

                    rot6d_linkid2indice=rot6d_linkid2indice, nearest_k=nearest_k_links,
                    length_linkid2indice=length_linkid2indice, length_per_link_learnable=True, 
                    use_radiance_kernel=get_rad_kernel(inst_name, 'kinematic'))
                motion.joint_names = a['joint_names']
                motion.global_rot6d.requires_grad = False
                motion.global_ts.requires_grad = False
                motion.length.requires_grad = False
                motion.rot6d.requires_grad = False
            elif preinit_motion_graphs is not None and inst_id in preinit_motion_graphs:
                assert not train_dataset.normalize_scene, "preinit motion graph is not supported for normalized scene at this moment"
                assert motion_type in [MotionBlenderType.deformable, MotionBlenderType.kinematic]
                if motion_type == MotionBlenderType.kinematic:
                    joints_lst = preinit_motion_graphs[inst_id][cano_t]['joints']
                    links = preinit_motion_graphs[inst_id][cano_t]['links']
                    root_joint_id = preinit_motion_graphs[inst_id][cano_t].get('root_id', 0)

                    joints = torch.as_tensor(joints_lst)
                    root_joint = joints[root_joint_id].clone()
                    joints -= root_joint

                    assert [l[1] for l in links] == list(range(1, len(joints))), "link ids must equal to joint ids plus 1"

                    chain = anim.inverse_kinematic(joints, links, length_inv_activation=torch.log)
                    hollow_chain = anim.create_hollow_chain_wo_tensor(chain)

                    length_tensor, length_linkid2indice = anim.retrieve_tensor_from_chain(chain, 'length', return_linkid2indice=True)
                    rot6d_tensor, rot6d_linkid2indice = anim.retrieve_tensor_from_chain(chain, 'rot6d', return_linkid2indice=True)

                    cano_offset = root_joint
                    global_rot6d = anim.rmat_to_cont_6d(torch.eye(3))
                    global_ts = torch.zeros(3)
                    global_ts += cano_offset
                    global_rot6d = repeat(global_rot6d, 'a -> t a', t=num_frames).clone()
                    global_ts = repeat(global_ts, 'a -> t a', t=num_frames).clone()
                    rot6d_tensor_by_t = repeat(rot6d_tensor, 'a b -> t a b', t=num_frames).clone() 

                    motion = MotionBlender(num_frames, cano_t, global_rot6d, global_ts, type=MotionBlenderType.kinematic,
                            links=torch.as_tensor(links).long(), init_gamma=init_gamma, init_temperature=init_temperature,
                            blend_method=blend_method, hollow_chain=hollow_chain, length_tensor=length_tensor, 
                            length_scale_tensor=None, 
                            rot6d_tensor=rot6d_tensor_by_t, 
                            rot6d_linkid2indice=rot6d_linkid2indice, nearest_k=nearest_k_links,
                            length_linkid2indice=length_linkid2indice, length_per_link_learnable=True, 
                            use_radiance_kernel=get_rad_kernel(inst_name, 'kinematic'))
                    motion.joint_names = preinit_motion_graphs[inst_id][cano_t].get('joint_names', [f'joint-{_v}' for _v in range(len(joints))])
                    motion.length.requires_grad = False
                else:
                    joints = torch.as_tensor(preinit_motion_graphs[inst_id][cano_t]['joints']).float()
                    links_tensor = torch.as_tensor(preinit_motion_graphs[inst_id][cano_t]['links']).long()
                    cano_offset = joints.mean(dim=0)
                    joints -= cano_offset

                    global_rot6d = anim.rmat_to_cont_6d(torch.eye(3))
                    global_ts = torch.zeros(3)
                    global_ts += cano_offset
                    global_rot6d = repeat(global_rot6d, 'a -> t a', t=num_frames).clone()
                    global_ts = repeat(global_ts, 'a -> t a', t=num_frames).clone()
                    joints_by_t = repeat(joints, "j c -> t j c", t=num_frames).clone()

                    motion = MotionBlender(num_frames, cano_t, global_rot6d, global_ts, type=MotionBlenderType.deformable,
                                    links=links_tensor, init_gamma=init_gamma, init_temperature=init_temperature, nearest_k=nearest_k_links,
                                    blend_method=blend_method, joints=joints_by_t, 
                                    use_radiance_kernel=False, deformable_link_quantize=deformable_link_quantize)
                
                register_remaining_preinit_motions(preinit_motion_graphs[inst_id], motion)

            elif motion_type == MotionBlenderType.deformable:
                guru.info(f"initializing deformable motion for {inst_name}")
                num_vertices = get_num_vertices_for_deformable(inst_id)
                skip_clusters_for_inst = inst_name in skip_preprocess_pcd_clustering
                if skip_clusters_for_inst:
                    guru.warning(f"skipping preprocess point cloud clustering for {inst_name}, the result pcd maybe more noisy")
                debug_pcds, traced_indices = defg_init.remove_outliers_from_pointcloud(inst_pts, inst_rgbs, skip_clusters=skip_clusters_for_inst, voxel_downsample_size=voxel_downsample_size)
                joints, tri_links = defg_init.build_deformable_graph_from_dense_points(debug_pcds[-1], num_vertices=num_vertices)
                joints, tri_links = torch.from_numpy(joints).float(), torch.from_numpy(tri_links).long()
                cano_offset = joints.mean(dim=0)
                joints -= cano_offset

                global_rot6d = anim.rmat_to_cont_6d(torch.eye(3))
                global_ts = torch.zeros(3)
                global_ts += cano_offset

                global_rot6d = repeat(global_rot6d, 'a -> t a', t=num_frames).clone()
                global_ts = repeat(global_ts, 'a -> t a', t=num_frames).clone()

                motion = MotionBlender(num_frames, cano_t, global_rot6d, global_ts, type=MotionBlenderType.deformable,
                                    links=tri_links, init_gamma=init_gamma, init_temperature=init_temperature, nearest_k=nearest_k_links,
                                    blend_method=blend_method, joints=repeat(joints, "j c -> t j c", t=num_frames).clone(), 
                                    use_radiance_kernel=get_rad_kernel(inst_name, 'deformable'), deformable_link_quantize=deformable_link_quantize)

            elif motion_type == MotionBlenderType.rigid:
                guru.info(f"initializing rigid motion for {inst_name}")
                global_rot6d = anim.rmat_to_cont_6d(torch.eye(3))
                global_rot6d = repeat(global_rot6d, 'a -> t a', t=num_frames).clone()

                cano_offset = sampled_pts.mean(dim=0)
                global_ts = torch.zeros(3)
                global_ts += cano_offset
                global_ts = repeat(global_ts, 'a -> t a', t=num_frames).clone()

                motion = MotionBlender(num_frames, cano_t, global_rot6d, global_ts, type=MotionBlenderType.rigid)
            elif motion_type == MotionBlenderType.kinematic:
                param, meta = init_kinematic_tree_just_from_keypoints(cano_t, train_datasets[0], inst_id)
                hollow_chain = meta['hollow_chain']
                cano_offset = param['global_t'].detach().cpu()

                global_rot6d = anim.rmat_to_cont_6d(torch.eye(3))
                global_ts = torch.zeros(3)
                global_ts += cano_offset
                global_rot6d = repeat(global_rot6d, 'a -> t a', t=num_frames).clone()
                global_ts = repeat(global_ts, 'a -> t a', t=num_frames).clone()

                length_tensor = param['length'].detach().cpu()
                motion = MotionBlender(num_frames, cano_t, global_rot6d, global_ts, type=MotionBlenderType.kinematic,
                    links=param['links'].detach().cpu(), init_gamma=init_gamma, init_temperature=init_temperature,
                    blend_method=blend_method, hollow_chain=hollow_chain, length_tensor=length_tensor, 
                    length_scale_tensor=None if length_per_link_learnable else param['scale'].detach().cpu(), 
                    rot6d_tensor=repeat(param['rot6d'].detach().cpu(), 'a b -> t a b', t=num_frames).clone(), 
                    rot6d_linkid2indice=meta['rot6d_linkid2indice'], nearest_k=nearest_k_links,
                    length_linkid2indice=meta['length_linkid2indice'], length_per_link_learnable=length_per_link_learnable, use_radiance_kernel=get_rad_kernel(inst_name, 'kinematic'))
                motion.joint_names = motion.keypoint_names = train_dataset.keypoint_names 
            elif motion_type == MotionBlenderType.static:
                continue
            else:
                raise ValueError(f"Unknown motion type: {motion_type}")

            dict_of_motions[inst_name] = motion
            guru.info(f"finalize the motion initialization of {inst_name}")

    if not fg_only and 'bg' not in dict_of_gaussians:
        guru.info("initializing background gaussian with RGB-D")
        bg_points = StaticObservations(*get_bkgd_points(num_bg)) # points, normals, colors
        assert bg_points.check_sizes()
        dict_of_gaussians["bg"] = init_gs(bg_points, use_sh_color=use_sh_color, init_opacity=0.1, filter_outliers=True)
        gaussian_names.append("bg")
    
    gs_modules, motion_modules = nn.ModuleDict(dict_of_gaussians), nn.ModuleDict(dict_of_motions)  
    return gs_modules, motion_modules, dict_of_track3ds, gaussian_names