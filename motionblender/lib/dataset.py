from flow3d.data.iphone_dataset import iPhoneDataset,SceneNormDict, BaseDataset, normalize_coords, parse_tapir_track_info, iPhoneDataConfig, masked_median_blur, rt_to_mat4
from PIL import Image
import roma
from dataclasses import dataclass, field
from tqdm.auto import tqdm
import os.path as osp
import torch.nn.functional as F
from loguru import logger as guru
import numpy as np
import torch
from pathlib import Path
from motionblender.lib.misc import compute_link_length, load_cpkl, load_json, viridis, remap_values
import motionblender.lib.animate as anim
import imageio.v3 as iio
import os
from typing import Literal

from motionblender.lib.init_graph.human_kinematic_tree_lifting import from_coco_wholebody_keypoints, kps_to_joint_links, wholebody_keypoints
import motionblender.lib.convert_utils as cvt


@dataclass
class MotionBlenderIPhoneDataConfig(iPhoneDataConfig):
    cache_version: str = ""
    use_tracks: bool = True # placeholder


class MotionBlenderIPhoneDataset(iPhoneDataset):

    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.training: return
        if kwargs.get('skip_load_imgs', False): return
        self.depth_type = kwargs['depth_type']
        self.img_prefix = "1x"
        self.use_tracks = True
        self.given_cano_t = -1

        scene_root = Path(self.data_dir)
        sapiens_root = scene_root / "flow3d_preprocessed/sapiens/sapiens_2b"
        grounding_sam_root = scene_root / "flow3d_preprocessed/grounding_sam2"
        num_frames = self.num_frames
        self.num_all_frames_in_scene = num_frames
        self.time_ids_list = self.time_ids.tolist()

        instance_meta = load_json(grounding_sam_root / "instances.json")
        instance_ids = [v['id'] for v in instance_meta.values()]
        instance_id_2_classname = {v['id']: k for k, v in instance_meta.items()}
        instance_id_2_motion_type = {v['id']: v['type'].lower() for k, v in instance_meta.items()}

        instance_masks = []
        first_mask_path = grounding_sam_root / 'mask_data' / f'mask_0_{str(0).zfill(5)}.npy'
        if first_mask_path.exists():
            for i in range(num_frames):
                arr = torch.from_numpy(np.load(grounding_sam_root / 'mask_data' / f'mask_0_{str(i).zfill(5)}.npy').astype(np.int32))
                instance_masks.append(arr)
            instance_masks = torch.stack(instance_masks)
        else:
            instance_masks = (self.masks > 0.5).long()

        instance_id_2_keypoints = {}
        for inst_id, cls_name in instance_id_2_classname.items():
            if cls_name == 'person':
                all_kps = []
                for i in range(num_frames):
                    kps = None
                    inst_bmask = instance_masks[i] == inst_id
                    if inst_bmask.sum() > 50: # if something smaller than 50 px, nah
                        meta = load_json(sapiens_root / f'0_{str(i).zfill(5)}.json')
                        instance_info = meta['instance_info']
                        if len(instance_info) == 1:
                            kps = instance_info[0]
                        else:
                            max_score = -1
                            for info in instance_info:
                                score = 0
                                for kp, kp_score in zip(info['keypoints'], info['keypoint_scores']):
                                    if kp_score > 0.8:
                                        x, y = int(kp[0]), int(kp[1])
                                        if inst_bmask[max(min(y, inst_bmask.shape[0]-1), 0), max(min(x, inst_bmask.shape[1]-1), 0)]: score += 1
                                if score > max_score:
                                    kps = info
                                    max_score = score
                    all_kps.append(kps)
                instance_id_2_keypoints[inst_id] = [[list(_1) + [_2] for _1, _2 in 
                                                        zip(*from_coco_wholebody_keypoints(a['keypoints'], a['keypoint_scores']))] 
                                                    if a is not None else 
                                                        ([(-1, -1, -1)] * len(wholebody_keypoints))
                                                    for a in all_kps] # list of (x, y, score)
        
        self.instance_ids = instance_ids
        self.instance_id_2_keypoints = instance_id_2_keypoints
        self.instance_id_2_classname = instance_id_2_classname
        self.instance_id_2_motion_type = instance_id_2_motion_type
        self.instance_masks = instance_masks
        self.instance_names = {}
        for inst_id in self.instance_ids:
            self.instance_names[inst_id] = f"{self.instance_id_2_classname[inst_id]}-{inst_id}"
         
        self.masks = (self.masks > 0.5).float()
        self.cache_version = kwargs.get('cache_version', "")
        guru.warning(f"MotionBlenderIPhoneDataset cache tag -> {self.cache_version}")
    
    
    

    def get_instance_tracks_3d(
        self: BaseDataset, num_tracks_per_frame: int, step: int = 1, 
        valid_kp_thresh: float = 0.4, show_pbar: bool = True, 
        candidate_frames: list[int]=None,
        path_proxy: tuple[str, str] = None,
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get 3D tracks from the dataset.
        Args:
            num_tracks_per_frame (int | None): The number of tracks to fetch for each frame
            step (int): The step to temporally subsample the track.
            valid_kp_thresh (float): The threshold for a keypoint to be 
                considered valid
            candidate_frames (List[int]): The frames to consider for track fetching
        """
        assert (
            self.split == "train"
        ), "fetch_tracks_3d is only available for the training split."
        cached_track_3d_path = osp.join(self.cache_dir, f"tracks_3d_version={self.cache_version}.pth")

        if osp.exists(cached_track_3d_path) and self.cache_version and step == 1 and self.load_from_cache:
            guru.info("loading cached 3d tracks data...")
            cached_track_3d_data = torch.load(cached_track_3d_path)
            return cached_track_3d_data
        else:
            if not self.load_from_cache:
                guru.info('load from cache is disabled, recomputing 3d tracks data...')
            else:
                if self.cache_version:
                    guru.info(f"no cached 3d tracks data found at {cached_track_3d_path}, recomputing...")
        
        # Load 2D tracks.
        # each element in raw_tracks_2d is a tensor of (num_frames, num_selected_tracks, 2)
        raw_tracks_2d  = [] # origin_kpid + inst_id * 1000
        candidate_frames = candidate_frames if candidate_frames is not None else list(range(0, self.num_frames, step))
        fg_masks = self.masks * self.valid_masks * (self.depths > 0)
        fg_masks = (fg_masks > 0.5).float()
        H, W = self.imgs.shape[1:3]
        for i in (
            tqdm(candidate_frames, desc="Loading 2D tracks", leave=False)
            if show_pbar
            else candidate_frames
        ):
            curr_num_samples = self.query_tracks_2d[i].shape[0] # persistent tracks from [i]
            init_tracks_2d = self.query_tracks_2d[i][:, :2] # N, 2
            num_samples = min(num_tracks_per_frame, curr_num_samples)

            # note: this is unnecessary, because the saved tracks are already in foreground!
            is_fg = F.grid_sample(
                    fg_masks[i * step: i * step + 1, None],
                    normalize_coords(init_tracks_2d[None, :, None], H, W),
                    align_corners=True,
                ).squeeze()  == 1
            is_fg = is_fg.nonzero()
            track_sels = is_fg[torch.randperm(len(is_fg))[:num_samples]].numpy().flatten()
            
            curr_tracks_2d = []
            for j in range(0, self.num_frames, step):
                if i == j:
                    target_tracks_2d = self.query_tracks_2d[i] # (N, 4)
                else:
                    try:
                        folder_name = f"flow3d_preprocessed/2d_tracks/{self.factor}x/"
                        if not osp.exists(osp.join(self.data_dir, folder_name)):
                            folder_name = f"bootstapir/{self.img_path_suffix}/"
                    except: 
                        folder_name = f"bootstapir/{self.img_path_suffix}/"
                    np_path = osp.join(
                                self.data_dir, folder_name,
                                f"{self.frame_names[i]}_"
                                f"{self.frame_names[j]}.npy",  # the positions of tracks (start at frame i) at frame j
                            )
                    if path_proxy is not None:
                        np_path = np_path.replace(*path_proxy)
                    target_tracks_2d = torch.from_numpy(np.load(np_path).astype(np.float32))
                curr_tracks_2d.append(target_tracks_2d[track_sels]) # all the tracks are the same at the start frame, regardless end frame
            raw_tracks_2d.append(torch.stack(curr_tracks_2d, dim=1)) # tracks(only start from i), every frames , 2 
        guru.info(f"{step=} {len(raw_tracks_2d)=} {raw_tracks_2d[0].shape=}") 


        # Process 3D tracks.
        inv_Ks = torch.linalg.inv(self.Ks)[::step]
        c2ws = torch.linalg.inv(self.w2cs)[::step]
        filtered_tracks_3d, filtered_visibles, filtered_track_colors, filtered_track_iids = [], [], [], []
        filtered_invisibles, filtered_confidences = [], []
        for i, tracks_2d in zip(candidate_frames, raw_tracks_2d): 
            tracks_2d = tracks_2d.swapdims(0, 1) # every frame, tracks (only start from i), 2
            tracks_2d, occs, dists = (
                tracks_2d[..., :2],
                tracks_2d[..., 2],
                tracks_2d[..., 3],
            )
            # visibles = postprocess_occlusions(occs, dists)
            visibles, invisibles, confidences = parse_tapir_track_info(occs, dists)
            # Unproject 2D tracks to 3D.
            track_depths = F.grid_sample(
                self.depths[::step, None],
                normalize_coords(tracks_2d[..., None, :], H, W),
                align_corners=True,
                padding_mode="border",
            )[:, 0]
            tracks_3d = (
                torch.einsum(
                    "nij,npj->npi",
                    inv_Ks,
                    F.pad(tracks_2d, (0, 1), value=1.0),
                )
                * track_depths
            )
            tracks_3d = torch.einsum(
                "nij,npj->npi", c2ws, F.pad(tracks_3d, (0, 1), value=1.0)
            )[..., :3] 

            is_in_masks = (
                F.grid_sample(
                    fg_masks[::step, None],
                    normalize_coords(tracks_2d[..., None, :], H, W),
                    align_corners=True,
                ).squeeze()
                == 1
            )
            visibles *= is_in_masks
            invisibles *= is_in_masks
            confidences *= is_in_masks.float()
            # Get track's color from the query frame.
            track_colors = (
                F.grid_sample(
                    self.imgs[i * step : i * step + 1].permute(0, 3, 1, 2),
                    normalize_coords(tracks_2d[i : i + 1, None, :], H, W),
                    align_corners=True,
                    padding_mode="border",
                )
                .squeeze()
                .T
            ) # [222, 3]
            # at least visible 5% of the time, otherwise discard
            visible_counts = visibles.sum(0)
            valid = visible_counts >= min(
                int(0.05 * self.num_frames),
                visible_counts.float().quantile(0.1).item(),
            )

            inst_masks = self.instance_masks
            inst_masks = inst_masks * fg_masks 
            track_iids =  F.grid_sample(
                    inst_masks[i * step : i * step + 1, None],
                    normalize_coords(tracks_2d[i : i + 1, None, :], H, W),
                    align_corners=True, mode='nearest'
                ).squeeze().float() * 1000
            
            # adding link id
            for inst_id in self.instance_ids:
                if inst_id in self.instance_id_2_keypoints:
                    guru.info(f"postprocess keypoints for {inst_id}", once=True)
                    inst_track_mask = track_iids == inst_id * 1000
                    kps_info = self.instance_id_2_keypoints[inst_id][i * step]
                    kps_info_tensor = torch.as_tensor(kps_info)
                    kps = kps_info_tensor[..., :2].tolist()
                    kp_scores = kps_info_tensor[..., 2].tolist()
                    joints, links, link_ids = kps_to_joint_links(kps, kp_scores, thr=valid_kp_thresh)
                    if len(joints) == 0:
                        track_iids[inst_track_mask] += 500 # invalid mark
                    else:
                        try:
                            # NOTE: we assign each track to a kp, and record the kp id and closest distance 
                            joints, links, link_ids = torch.from_numpy(joints), torch.from_numpy(links), torch.from_numpy(link_ids)
                            joints_int = joints.long() # remove links that are not visible
                            joints_int[:, 0].clamp_(0, W - 1)
                            joints_int[:, 1].clamp_(0, H - 1)
                            joints_visible = fg_masks[i, joints_int[:, 1], joints_int[:, 0]]
                            keep_link_ids = [link_i for link_i, link in enumerate(links) if joints_visible[link].all()]
                            links, link_ids = links[keep_link_ids], link_ids[keep_link_ids]
                            joints = normalize_coords(joints, H, W)
                            pts2d = normalize_coords(tracks_2d[i, :], H, W)
                            pts2d = pts2d[inst_track_mask]
                            gamma = 2 / (compute_link_length(joints, links) + 1e-4)
                            link_weights2d, link_distances = anim.weight_inpaint(pts2d, joints.float(), links, gamma.float(), temperature=0.25, return_distance=True)
                            link_assignments = link_weights2d.argmax(dim=1)
                            link_ids = link_ids[link_assignments]
                            closest_distance = torch.gather(link_distances, 1, link_assignments.reshape(-1, 1))  
                            closest_distance /= 4
                            assert torch.max(closest_distance) < 1
                            track_iids[inst_track_mask] += (link_ids + closest_distance.flatten())
                        except Exception as e:
                            guru.warning("Exception occurred during keypoints postprocessing, skipped" + str(e))
                            track_iids[inst_track_mask] += 500 
            
            filtered_tracks_3d.append(tracks_3d[:, valid])
            filtered_visibles.append(visibles[:, valid])
            filtered_invisibles.append(invisibles[:, valid])
            filtered_confidences.append(confidences[:, valid])
            filtered_track_colors.append(track_colors[valid])
            filtered_track_iids.append(track_iids[valid])


        filtered_tracks_3d = torch.cat(filtered_tracks_3d, dim=1).swapdims(0, 1) # all_tracks, frames, 3
        filtered_visibles = torch.cat(filtered_visibles, dim=1).swapdims(0, 1)
        filtered_invisibles = torch.cat(filtered_invisibles, dim=1).swapdims(0, 1)
        filtered_confidences = torch.cat(filtered_confidences, dim=1).swapdims(0, 1)
        filtered_track_colors = torch.cat(filtered_track_colors, dim=0)
        filtered_track_iids = torch.cat(filtered_track_iids, dim=0)
        result = (filtered_tracks_3d, filtered_visibles, filtered_invisibles, filtered_confidences, filtered_track_colors, filtered_track_iids)
        if step == 1 and self.cache_version: 
            guru.info(f"saving 3d tracks data to {cached_track_3d_path}...")
            torch.save(result, cached_track_3d_path)
        return result
    
    def __getitem__(self, index):
        data = super().__getitem__(index)
        data['cam_name'] = self.img_prefix
        if self.training:
            findex = self.frame_names.index(data['frame_names'])
            data['instance_masks'] = self.instance_masks[findex]
            data['kps2d'] = {}
            for inst_id, all_kps in self.instance_id_2_keypoints.items():
                # kps, kps_scores = all_kps[findex]
                data['kps2d'][self.instance_names[inst_id]] = torch.as_tensor(all_kps[findex]).float()
                # assert len(kps) == len(wholebody_keypoints)

        return data


@dataclass
class MotionBlenderGeneralDataConfig(MotionBlenderIPhoneDataConfig):
    depth_type: Literal["metric_depth", "aligned_depth_anything"]  = "metric_depth"
    normalize_scene: bool = True
    use_tracks: bool = True
    given_cano_t: int = -1 # if use_tracks is False, then cano_t must be given
    cache_version: str = ""
    img_path_suffix: str = ""
    K_scale: int = 1

    mask_insts: list[int] = field(default_factory=lambda: [])



class MotionBlenderGeneralDataset(MotionBlenderIPhoneDataset):

    @property
    def num_frames(self):
        if hasattr(self, 'imgs'): return len(self.imgs)
        else: return 0
    

    def __init__(
        self,
        data_dir: str,
        split: Literal["train", "val", "all"] = "train",
        depth_type: Literal["metric_depth", "aligned_depth_anything"] = "depth_anything_colmap",
        use_median_filter: bool = False,
        num_targets_per_frame: int = 1,
        load_from_cache: bool = False,
        use_tracks: bool = True,
        given_cano_t: int = -1,
        normalize_scene: bool = True,
        scene_norm_dict: SceneNormDict | None = None,
        cache_version: str = "",
        img_path_suffix: str = "",
        img_prefix: str = "",
        K_scale: int = 1,
        mask_insts: list[int] = [],
        **_,
    ):
        BaseDataset.__init__(self)
        self.data_dir = data_dir
        self.training = split == "train"
        self.split = split
        self.depth_type = depth_type
        self.use_median_filter = use_median_filter
        if depth_type == "metric_depth" and not use_median_filter:
            guru.warning("Using metric depth without median filter")

        self.num_targets_per_frame = num_targets_per_frame
        self.scene_norm_dict = None
        self.load_from_cache = load_from_cache
        self.cache_version = cache_version
        self.img_prefix = img_prefix
        self.img_path_suffix = img_path_suffix
        
        if img_prefix: self.cache_version += f"_{img_prefix}"
        if img_path_suffix: self.cache_version += f"_{img_path_suffix}"
            
        self.cache_dir = osp.join(data_dir, "cache")
        self.use_tracks = use_tracks
        self.given_cano_t = given_cano_t
        if not use_tracks and given_cano_t < 0:
            guru.warning("use_tracks is False, but given_cano_t is not set")
        os.makedirs(self.cache_dir, exist_ok=True)

        split_dict = load_json(osp.join(data_dir, "dataset.json"))
        self.num_all_frames_in_scene = len(set([int(_id.split('_')[-1]) for _id in split_dict['ids']]))
        self.has_validation = "val_ids" in split_dict
        if split == "train" and "train_ids" not in split_dict:
            ids = split_dict['ids']
        else:
            ids = split_dict['ids'] if split == "all" else split_dict[f'{split}_ids']
        indices_match_prefix = list(range(len(ids)))
        if img_prefix:
            indices_match_prefix = [i for i, _id in enumerate(ids) if _id.startswith(img_prefix)]

        try:
            if split == "train" and "train_time_ids" not in split_dict:
                time_ids = split_dict['time_ids']
            else:
                time_ids = split_dict['time_ids'] if split == "all" else split_dict[f'{split}_time_ids']
        except KeyError:
            guru.warning(f"time_ids not found in {data_dir}/dataset.json, using range index")
            time_ids = list(range(len(ids)))
        ids = [ids[i] for i in indices_match_prefix]
        time_ids = [time_ids[i] for i in indices_match_prefix]
        
        self.frame_names = ids
        self.time_ids = torch.as_tensor(time_ids).long()
        self.time_ids_list = self.time_ids.tolist()
        self.Ks, self.w2cs = [], []
        for fid in ids:
            intrinsics, extrinsics, img_wh = cvt.from_camera_json(osp.join(data_dir, "camera", f"{fid}.json"))
            self.Ks.append(intrinsics)
            self.w2cs.append(torch.inverse(extrinsics))

        self.Ks = torch.stack(self.Ks)
        if K_scale != 1:
            guru.info(f"scaling Ks by {K_scale}")
            self.Ks[:, :2, 2] /= K_scale
            self.Ks[:, 0, 0] /= K_scale
            self.Ks[:, 1, 1] /= K_scale

        self.w2cs = torch.stack(self.w2cs)
        self.fps = split_dict.get('fps', 30)
        self.scene_norm_dict = scene_norm_dict
        self.normalize_scene = normalize_scene

        imgs = torch.from_numpy(
                np.array(
                    [
                        iio.imread(
                            osp.join(self.data_dir, f"rgb/{img_path_suffix}/{frame_name}.png")
                        )
                        for frame_name in tqdm(
                            self.frame_names,
                            desc=f"Loading {self.split} images",
                            leave=False,
                        )
                    ],
                )
            )
        self.img_wh = list(reversed(imgs.shape[1:3]))
        self.imgs = imgs[..., :3] / 255.0

        data_dir = Path(data_dir)
        if mask_insts:
            for _i in mask_insts:
                assert _i > 0, "you can only mask instance ids (background id is 0)"

        def get_adjust_inst_id(inst_id):
            for _i in mask_insts:
                if _i < inst_id:
                    inst_id -= 1
                elif _i == inst_id:
                    return -1
            return inst_id

        ori_instance_ids = []
        instance_id_2_classname = {}
        instance_id_2_motion_type = {} # important: instance id
        instance_adjusted_id_to_old_id = {}
        for i, name in enumerate(load_json(data_dir / "instance" / img_path_suffix / "names.json")):
            _id = i + 1
            ori_instance_ids.append(_id)
            _name, _type = name.split(':')
            adjust_id = get_adjust_inst_id(_id)
            instance_adjusted_id_to_old_id[adjust_id] = _id
            if adjust_id == -1: continue
            instance_id_2_classname[adjust_id] = _name
            instance_id_2_motion_type[adjust_id] = _type

        ori_instance_masks = []
        for i in range(self.num_frames):
            arr = torch.from_numpy(np.array(Image.open(data_dir / 'instance' / img_path_suffix / 'imask' / f'{self.frame_names[i]}.png')).astype(np.int32))
            ori_instance_masks.append(arr)
        ori_instance_masks = torch.stack(ori_instance_masks)

        if mask_insts:
            remaps = [(0, 0)]
            instance_ids = []
            for inst_id in ori_instance_ids:
                adjust_inst_id = get_adjust_inst_id(inst_id)
                if adjust_inst_id == -1: 
                    remaps.append((inst_id, 0)) # in instance mask, treat as background, in valid_mask, treat as invalid
                    continue
                else:
                    remaps.append((inst_id, adjust_inst_id))
                instance_ids.append(adjust_inst_id)
            instance_masks = remap_values(ori_instance_masks, 
                                        torch.as_tensor([a[0] for a in remaps]).to(ori_instance_masks.dtype), 
                                        torch.as_tensor([a[1] for a in remaps]).to(ori_instance_masks.dtype)) 
        else:
            instance_masks = ori_instance_masks
            instance_ids = ori_instance_ids

        # NOTE: motion graph is predefined, like the robot itself
        self.predefined_motion_graphs = {}
        predefined_motion_graph_path = osp.join(data_dir, "predefined_motion_graphs")
        if osp.exists(osp.join(predefined_motion_graph_path, self.frame_names[0] + '.pkl')):
            for frame_name in self.frame_names:
                mg = load_cpkl(osp.join(predefined_motion_graph_path, frame_name + '.pkl'))
                if 'instances' in mg:
                    mg = {get_adjust_inst_id(int(k)): v for k, v in mg['instances'].items()} # mg: instance_id: int -> dict(links, joints)
                else:
                    assert len(instance_ids) == 1
                    mg = {instance_ids[0]: mg}
                for inst_id, g in mg.items():
                    self.predefined_motion_graphs.setdefault(inst_id, []).append(g)
        
        # NOTE: motion graph is pre-inited externally
        self.preinited_motion_graphs = None
        preinited_motion_graphs_path = osp.join(data_dir, "preinited_motion_graphs.json")
        if osp.exists(preinited_motion_graphs_path):
            _ = load_json(preinited_motion_graphs_path)
            self.given_cano_t = _['cano_t']
            self.preinited_motion_graphs = { get_adjust_inst_id(int(k)): {int(_k): _v for _k, _v in v.items()} for k, v in _['instances'].items() }
            # instance_id: int -> t: int -> dict(joints, links, fixed_joint_ids)
            guru.warning(f"preinited motion graph found, cano_t = {self.given_cano_t}")

        instance_id_2_keypoints = {}
        for inst_id, cls_name in instance_id_2_classname.items():
            if cls_name == 'person' and instance_id_2_motion_type[inst_id] == 'kinematic':
                all_kps = []
                for i, fname in enumerate(self.frame_names):
                    kps = None
                    inst_bmask = instance_masks[i] == inst_id
                    if inst_bmask.sum() > 50: # if something smaller than 50 px, nah
                        meta = load_json(osp.join(self.data_dir, f"sapiens/{img_path_suffix}/sapiens_2b/{fname}.json"))
                        instance_info = meta['instance_info']
                        if len(instance_info) == 1:
                            kps = instance_info[0]
                        else:
                            max_score = -1
                            for info in instance_info:
                                score = 0
                                for kp, kp_score in zip(info['keypoints'], info['keypoint_scores']):
                                    if kp_score > 0.8:
                                        x, y = int(kp[0]), int(kp[1])
                                        if inst_bmask[max(min(y, inst_bmask.shape[0]-1), 0), max(min(x, inst_bmask.shape[1]-1), 0)]: score += 1
                                if score > max_score:
                                    kps = info
                                    max_score = score
                    all_kps.append(kps)
                # list of (N, 3), kp + score
                instance_id_2_keypoints[inst_id] = [[list(_1) + [_2] for _1, _2 in 
                                                        zip(*from_coco_wholebody_keypoints(a['keypoints'], a['keypoint_scores']))] 
                                                    if a is not None else 
                                                        ([(-1, -1, -1)] * len(wholebody_keypoints))
                                                    for a in all_kps]
            else:
                keypoints_folder = osp.join(self.data_dir, f"keypoints/{instance_adjusted_id_to_old_id[inst_id] - 1}/")
                keypoints_connectivity_path = osp.join(keypoints_folder, "keypoints_connectivity.json")
                if osp.exists(keypoints_connectivity_path) and osp.exists(keypoints_folder):
                    keypoints_connectivity = load_json(keypoints_connectivity_path)
                    keypoint_names = []
                    for k, v in keypoints_connectivity.items():
                        keypoint_names += ([k] + v)
                    keypoint_names = sorted(list(set(keypoint_names)))
                    self.keypoint_names = keypoint_names
                    self.keypoint_links = []
                    for k, v in keypoints_connectivity.items():
                        for a in v:
                            self.keypoint_links.append([keypoint_names.index(k), keypoint_names.index(a)])
                    all_kp_data = []
                    for fn in self.frame_names:
                        kp_json = osp.join(keypoints_folder, f"{fn}.json")
                        kp_data = load_json(kp_json)
                        all_kp_data.append([kp_data[kname] for kname in self.keypoint_names])
                    instance_id_2_keypoints[inst_id] = all_kp_data

        self.instance_ids = instance_ids
        self.instance_id_2_keypoints = instance_id_2_keypoints
        self.instance_id_2_classname = instance_id_2_classname
        self.instance_id_2_motion_type = instance_id_2_motion_type
        self.instance_masks = instance_masks
        self.instance_names = {}
        for inst_id in self.instance_ids:
            self.instance_names[inst_id] = f"{self.instance_id_2_classname[inst_id]}-{inst_id}"

        self.masks = (self.instance_masks > 0.5).float()       
        if mask_insts:
            self.valid_masks = ~torch.isin(ori_instance_masks, torch.as_tensor(mask_insts, dtype=self.instance_masks.dtype))
        else:
            self.valid_masks = torch.ones_like(self.masks, dtype=torch.bool)

        if self.training: 
            def load_depth(frame_name):
                if self.depth_type == "metric_depth":
                    depth = np.load(
                        osp.join(
                            self.data_dir,
                            f"metric_depth/{img_path_suffix}/{frame_name}.npy",
                        )
                    )
                else:
                    depth = np.load(
                        osp.join(
                            self.data_dir,
                            f"aligned_depth_anything/{img_path_suffix}/{frame_name}.npy",
                        )
                    )
                depth[depth < 1e-6] = 1e-6
                depth = 1.0 / depth
                return depth

            self.depths = torch.from_numpy(
                np.array(
                    [
                        load_depth(frame_name)
                        for frame_name in tqdm(
                            self.frame_names,
                            desc=f"Loading {self.split} depths",
                            leave=False,
                        )
                    ],
                    np.float32,
                )
            )
            max_depth_values_per_frame = self.depths.reshape(
                self.num_frames, -1
            ).max(1)[0]
            max_depth_value = max_depth_values_per_frame.median() * 2.5
            print("max_depth_value", max_depth_value)
            self.depths = torch.clamp(self.depths, 0, max_depth_value)
            
            if self.use_median_filter:
                for i in tqdm(
                    range(self.num_frames), desc="Processing depths", leave=False
                ):
                    depth = masked_median_blur(
                        self.depths[[i]].unsqueeze(1).to("cuda"),
                        (
                            self.masks[[i]]
                            * self.valid_masks[[i]]
                            * (self.depths[[i]] > 0)
                        )
                        .unsqueeze(1)
                        .to("cuda"),
                    )[0, 0].cpu()
                    self.depths[i] = depth * self.masks[i] + self.depths[i] * (
                        1 - self.masks[i]
                    )

            if self.use_tracks:
                self.query_tracks_2d = [
                    torch.from_numpy(
                        np.load(
                            osp.join(
                                self.data_dir,
                                f"bootstapir/{img_path_suffix}/",
                                f"{frame_name}_{frame_name}.npy",
                            )
                        ).astype(np.float32)
                    )
                    for frame_name in self.frame_names
                ]
                guru.info(
                    f"{len(self.query_tracks_2d)=} {self.query_tracks_2d[0].shape=}"
                )
            else:
                self.query_tracks_2d = None

        self.valid_masks = self.valid_masks.float()

        if self.scene_norm_dict is None and self.normalize_scene:
            cached_scene_norm_dict_path = osp.join(
                self.cache_dir, "scene_norm_dict.pth"
            )
            if osp.exists(cached_scene_norm_dict_path) and self.load_from_cache:
                print("loading cached scene norm dict...")
                self.scene_norm_dict = torch.load(
                    osp.join(self.cache_dir, "scene_norm_dict.pth")
                )
            elif self.training: # ANCHOR
                # Compute the scene scale and transform for normalization.
                # Normalize the scene based on the foreground 3D tracks.
                subsampled_tracks_3d = self.get_tracks_3d(
                    num_samples=10000, step=self.num_frames // 20, show_pbar=False
                )[0]
                scene_center = subsampled_tracks_3d.mean((0, 1))
                tracks_3d_centered = subsampled_tracks_3d - scene_center
                min_scale = tracks_3d_centered.quantile(0.05, dim=0)
                max_scale = tracks_3d_centered.quantile(0.95, dim=0)
                scale = torch.max(max_scale - min_scale).item() / 2.0
                original_up = -F.normalize(self.w2cs[:, 1, :3].mean(0), dim=-1)
                target_up = original_up.new_tensor([0.0, 0.0, 1.0])
                R = roma.rotvec_to_rotmat(
                    F.normalize(original_up.cross(target_up, dim=-1), dim=-1) # axis of the rotation
                    * original_up.dot(target_up).acos_() # angle of the rotation
                )
                transfm = rt_to_mat4(R, torch.einsum("ij,j->i", -R, scene_center))
                self.scene_norm_dict = SceneNormDict(scale=scale, transfm=transfm)
                torch.save(self.scene_norm_dict, cached_scene_norm_dict_path)
            else:
                raise ValueError("scene_norm_dict must be provided for validation.")
        
            # Normalize the scene.
            scale = self.scene_norm_dict["scale"]
            transfm = self.scene_norm_dict["transfm"]
            self.origin_w2cs = self.w2cs.clone()
            self.w2cs = self.w2cs @ torch.linalg.inv(transfm)
            self.w2cs[:, :3, 3] /= scale # translation
            if self.training:
                self.depths /= scale
        else:
            guru.warning(f"no scene normalization is applied for {data_dir}!")
    
    def __getitem__(self, index):
        if self.training:
            index = np.random.randint(0, self.num_frames)
        
        data = {
            # ().
            "frame_names": self.frame_names[index],
            # ().
            "ts": self.time_ids[index],
            # (4, 4).
            "w2cs": self.w2cs[index],
            # (3, 3).
            "Ks": self.Ks[index],
            # (H, W, 3).
            "imgs": self.imgs[index],
            # (H, W).
            "valid_masks": self.valid_masks[index],
            # (H, W).
            "masks": self.masks[index],
        }
        if hasattr(self, 'depths'):
            data["depths"] = self.depths[index] 
        
        data['cam_name'] = self.img_prefix
        if self.training:
            if self.use_tracks:
                data["query_tracks_2d"] = self.query_tracks_2d[index][:, :2] # (P, 2). 
                if self.num_targets_per_frame > 0:
                    target_inds = torch.from_numpy(
                        np.random.choice(
                            self.num_frames, (self.num_targets_per_frame,), replace=False
                        )
                    )
                    # (N, P, 4).
                    target_tracks_2d = torch.stack(
                        [
                            torch.from_numpy(
                                np.load(
                                    osp.join(
                                        self.data_dir,
                                        f"bootstapir/{self.img_path_suffix}/",
                                        f"{self.frame_names[index]}_"
                                        f"{self.frame_names[target_index.item()]}.npy",
                                    )
                                ).astype(np.float32)
                            )
                            for target_index in target_inds
                        ],
                        dim=0,
                    )
                    # (N,).
                    target_ts = self.time_ids[target_inds]
                    data["target_ts"] = target_ts
                    # (N, 4, 4).
                    data["target_w2cs"] = self.w2cs[target_inds]
                    # (N, 3, 3).
                    data["target_Ks"] = self.Ks[target_inds]
                    # (N, P, 2).
                    data["target_tracks_2d"] = target_tracks_2d[..., :2]
                    # (N, P).
                    (
                        data["target_visibles"],
                        data["target_invisibles"],
                        data["target_confidences"],
                    ) = parse_tapir_track_info(
                        target_tracks_2d[..., 2], target_tracks_2d[..., 3]
                    )
                    # (N, P).
                    data["target_track_depths"] = F.grid_sample(
                        self.depths[target_inds, None],
                        normalize_coords(
                            target_tracks_2d[..., None, :2],
                            self.imgs.shape[1],
                            self.imgs.shape[2],
                        ),
                        align_corners=True,
                        padding_mode="border",
                    )[:, 0, :, 0]
            
            findex = self.frame_names.index(data['frame_names'])
            data['instance_masks'] = self.instance_masks[findex]

            if len(self.instance_id_2_keypoints) > 0:
                data['kps2d'] = {}
                for inst_id, all_kps in self.instance_id_2_keypoints.items():
                    data['kps2d'][self.instance_names[inst_id]] = torch.as_tensor(all_kps[findex]).float()
                    assert len(data['kps2d'][self.instance_names[inst_id]]) == len(self.keypoint_names)
        return data


class MotionBlenderDataset(MotionBlenderGeneralDataset, MotionBlenderIPhoneDataset): pass