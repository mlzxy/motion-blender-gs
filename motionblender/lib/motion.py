import torch 
from tqdm.auto import tqdm
from copy import deepcopy
from loguru import logger
from einops import repeat, einsum
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float32, Int64
from flow3d.loss_utils import (
    compute_gradient_loss,
    compute_accel_loss,
    masked_l1_loss
)
import motionblender.lib.animate as anim
from motionblender.lib.misc import reduce_loss_dict
from dataclasses import dataclass
import os
from motionblender.lib.pytorch_arap.arap import ARAPMeshes, compute_energy as arap_loss
import motionblender.lib.mesh2d as mesh2d_lib

class MotionBlenderType:
    deformable = "deformable"
    kinematic = "kinematic"
    rigid = "rigid"
    static = "static"


@dataclass
class PoseStore: 
    global_T: Tensor | None = None # 4x4

    joints: Tensor | None = None # deformable, J x 3
    rot6ds: Tensor | None = None # kinematic, L x 6

    
def smooth_params(p: Float32[Tensor, "t *"]) -> Float32[Tensor, "t *"]:
    p = p.clone()
    avg_middle = torch.stack([p[2:], p[1:-1], p[:-2]]).mean(0)
    p[1:-1] = avg_middle
    return p


class MotionBlender(nn.Module):
    def __init__(self, num_frames:int, cano_t: int, 

                # global parameters
                 global_rot6d: Float32[Tensor, "t 6"],
                 global_ts: Float32[Tensor, "t 3"],
                 links: Int64[Tensor, "l 2"]=None,  type: str="kinematic",
                 init_gamma: float | Float32[Tensor, " l "]=4.0, init_temperature: float=0.01, blend_method:str ='dq', 
                 use_radiance_kernel: bool=False, radiance_kwargs={},
                #  flexible_cano: bool=False,

                # deformable graph parameters
                 joints: Float32[Tensor, "t j 3"]=None, nearest_k: int=5,
                 deformable_link_quantize:int = -1,
                
                # kinematic tree parameters
                 hollow_chain: dict=None,
                 length_tensor: Float32[Tensor, "j 1"] = None, 
                 length_scale_tensor: Float32[Tensor, "1"] = None,
                 rot6d_tensor: Float32[Tensor, "t l 6"] = None, 
                 rot6d_linkid2indice: dict[int, int] = None, length_linkid2indice: dict[int, int] = None,
                 length_per_link_learnable: bool=False):
        super().__init__()

        assert type in [MotionBlenderType.deformable, MotionBlenderType.kinematic, MotionBlenderType.rigid]
        assert blend_method in ['dq', 'linear']
        assert global_rot6d.shape[0] == num_frames
        assert global_ts.shape[0] == num_frames
        self.joint_names = None

        if type == MotionBlenderType.deformable:
            assert joints is not None
            assert len(joints) == num_frames
            assert joints.dim() == 3
            self.register_parameter("joints", nn.Parameter(joints))
            self.deformable_link_quantize = deformable_link_quantize
            if deformable_link_quantize > 0:
                self.register_buffer("link_percentages", torch.linspace(0, 1, deformable_link_quantize))

        elif type == MotionBlenderType.kinematic:
            assert hollow_chain is not None
            assert length_tensor is not None
            assert rot6d_tensor is not None
            assert rot6d_linkid2indice is not None
            assert length_linkid2indice is not None
            assert rot6d_tensor.dim() == 3

            self.length_per_link_learnable = length_per_link_learnable
            self.register_parameter("length", nn.Parameter(length_tensor, requires_grad=length_per_link_learnable))
            if not length_per_link_learnable:
                assert length_scale_tensor is not None
                self.register_parameter("log_length_scale", nn.Parameter(length_scale_tensor))
            else:
                self.log_length_scale = None
            self.register_parameter("rot6d", nn.Parameter(rot6d_tensor))

            self.hollow_chain = hollow_chain
            self.rot6d_linkid2indice = rot6d_linkid2indice
            self.length_linkid2indice = length_linkid2indice
        
        if type in [MotionBlenderType.deformable, MotionBlenderType.kinematic]:
            assert links is not None
            links = links.reshape(-1, 2)
            self.nearest_k = min(nearest_k, len(links))
            self.register_buffer("links", links)
            self.register_parameter("log_temperature", nn.Parameter(torch.log(torch.full((links.shape[0],), fill_value=init_temperature))))
            self.use_radiance_kernel = use_radiance_kernel

            if use_radiance_kernel:
                radiance_kernel_kwargs = {'init_gamma': init_gamma, 'use_embed': True, 'hidden_size': 32, **radiance_kwargs}
                num_links = len(links)
                if isinstance(init_gamma, float):
                    init_gamma = [torch.log(torch.tensor(init_gamma))] * num_links
                funcs = []
                for i in range(num_links):
                    _ = deepcopy(radiance_kernel_kwargs)
                    _['init_gamma'] = float(init_gamma[i])
                    funcs.append(anim.RadianceFunction(**_))
                self.radiance_kernels = nn.ModuleList(funcs)
            else:
                if isinstance(init_gamma, float):
                    self.register_parameter("log_gamma", nn.Parameter(torch.log(torch.full((links.shape[0],), fill_value=init_gamma))))
                else:
                    logger.info('input gamma is a list / tensor, assuming to be log scale')
                    self.register_parameter("log_gamma", nn.Parameter(init_gamma))

        self.register_parameter("global_rot6d", nn.Parameter(global_rot6d))
        self.register_parameter("global_ts", nn.Parameter(global_ts))
        self.blend_method = blend_method
        self.num_frames = num_frames        
        self.cano_t = cano_t
        self.type = type
        self.is_rigid = type == MotionBlenderType.rigid

        # caching cano information for regularization
        self.clear_motion_cache() 
        self.register_buffer('cano_global_ts', global_ts[cano_t].detach())
        self.register_buffer('cano_global_rot6d', global_rot6d[cano_t].detach())
        if not self.is_rigid:
            with torch.no_grad():
                self.compute_link_pose_at_t(cano_t)
            self.register_buffer('cano_joints', self._joints_tensor_cache[cano_t].detach())
        self.clear_motion_cache() 
    
    
    @torch.no_grad()
    def update_cano_info(self):
        self.clear_motion_cache()
        self.cano_global_ts[:] = self.global_ts[self.cano_t].detach()
        self.cano_global_rot6d[:] = self.global_rot6d[self.cano_t].detach()
        if not self.is_rigid:
            self.compute_link_pose_at_t(self.cano_t)
            self.cano_joints[:] = self._joints_tensor_cache[self.cano_t].detach()
    
    @torch.no_grad()
    def save_initial_params(self, frames: list[int]):
        self._initial_anchors = {}
        self._initial_anchors['frames'] = frames
        for k in ['global_ts', 'global_rot6d', 'rot6d', 'joints']:
            if hasattr(self, k):
                self._initial_anchors[k] = getattr(self, k).data[frames].detach()

    @torch.no_grad()
    def keep_initial_params_the_same(self):
        if hasattr(self, '_initial_anchors'):
            initial_frames = self._initial_anchors['frames']
            logger.warning(f"keeping the motion parameters of the {initial_frames} frames the same", once=True)
            for k, v in self._initial_anchors.items():
                if k == 'frames': continue
                getattr(self, k).data[initial_frames] = v
    
    def _shrink_skin_weights(self): # for efficiency
        _ = self._skin_weights.topk(self.nearest_k, dim=1, largest=True, sorted=True)
        self._skin_weights = _.values
        self._skin_weights /= self._skin_weights.sum(dim=1, keepdim=True)
        self._skin_weights_index = _.indices
    
    
    def compute_link_pose_at_t(self, t, skin_pts: Float32[Tensor, "n 3"]=None, pose_store: PoseStore | None=None):
        t = int(t)
        if t in self._global_T_cache: return # already computed
        dev = self.global_rot6d.device
        if pose_store is not None:
            global_T = pose_store.global_T.to(dev).clone()
        else:
            global_ts = self.global_ts[t]
            global_rot6d = self.global_rot6d[t]
            global_T = anim.rt_to_mat4(anim.cont_6d_to_rmat(global_rot6d), global_ts)
        self._global_T_cache[t] = global_T
        if self.is_rigid: return
        
        if self.type == MotionBlenderType.deformable:
            if pose_store is not None:
                joint_poses = pose_store.joints.to(dev).clone()
                assert joint_poses.shape == self.joints[0].shape
            else:
                joint_poses = self.joints[t]
            joint_poses = anim.apply_mat4(global_T, joint_poses)

            if skin_pts is not None and t == self.cano_t:
                self._skin_weights, self._falloffs = anim.weight_inpaint(skin_pts, joint_poses, self.links, 
                    gamma=self.radiance_kernels if getattr(self, 'use_radiance_kernel', False) else torch.exp(self.log_gamma),
                    temperature=torch.exp(self.log_temperature), return_falloff=True) # +100 MB
                self._shrink_skin_weights()
                self._falloffs = torch.gather(self._falloffs, 1, self._skin_weights_index)
                if getattr(self, 'deformable_link_quantize', -1) > 0:
                    self._falloffs_q = (self._falloffs * (self.deformable_link_quantize - 1)).round_().long()

                # (p, nearest_k), (p, nearest_k)
            if self._falloffs is None:
                logger.info("skipping link pose computation since no skinning points are provided. this is ok during joints initialization for canonical frame")
                link_poses = None
            else: # following +(<10 MB)
                link_normals = anim.compute_normals(joint_poses, self.links.reshape(-1, 2, 2))
                if getattr(self, 'deformable_link_quantize', -1) > 0: # improve efficiency here by link quantization
                    link_normals = repeat(link_normals, 'l c -> (q l) c', q = self.deformable_link_quantize)
                    link_percentages = repeat(self.link_percentages, 'q -> (q l)', l=len(self.links))
                    joints_a = repeat(joint_poses[self.links[:, 0]], 'l c ->  (q l) c', q=self.deformable_link_quantize)
                    joints_b = repeat(joint_poses[self.links[:, 1]], 'l c ->  (q l) c', q=self.deformable_link_quantize)
                    if len(link_percentages) >= self._skin_weights_index.numel():
                        logger.warning(f"deformable link quantization requires the computation of {len(link_percentages)} items (> {self._skin_weights_index.numel()} which is without quantization)", once=True)
                    link_poses = anim.find_link_ctrl_pt_pose(joints_a, joints_b, link_normals, link_percentages) 
                    link_poses = link_poses.reshape(self.deformable_link_quantize, len(self.links), 4, 4)
                else:
                    _ = [joint_poses[self.links[:, 0]], joint_poses[self.links[:, 1]], link_normals] # 3x (p, 3)
                    _skin_weights_index = self._skin_weights_index.flatten()
                    link_poses = anim.find_link_ctrl_pt_pose( *[a[_skin_weights_index] for a in _], self._falloffs.flatten())
                    link_poses = link_poses.reshape(*self._falloffs.shape, 4, 4) # p, nearest_k, 4, 4
        else: # kinematic
            length = self.length
            if getattr(self, 'length_per_link_learnable', False): 
                length = torch.exp(length)
            else:
                length = length * torch.exp(self.log_length_scale)
            
            if pose_store is not None:
                rot6d = pose_store.rot6ds.to(dev).clone()
                assert rot6d.shape == self.rot6d[0].shape
            else:
                rot6d = self.rot6d[t]
            
            chain = anim.fill_hollow_chain_with_tensor(self.hollow_chain, length, rot6d, 
                                    self.rot6d_linkid2indice, self.length_linkid2indice)
            t_rel_link_poses = anim.forward_kinematic(chain)
            link_poses = anim.apply_mat4_pose(global_T, t_rel_link_poses)
            joint_poses = anim.link_poses_to_joint_positions(link_poses, self.hollow_chain['id'], global_T[:3, 3])  
            if skin_pts is not None and t == self.cano_t:
                self._skin_weights = anim.weight_inpaint(skin_pts, joint_poses, self.links, 
                    gamma=self.radiance_kernels if getattr(self, 'use_radiance_kernel', False) else torch.exp(self.log_gamma), 
                    temperature=torch.exp(self.log_temperature), return_falloff=False)
        
        self._links_tensor_cache[t] = link_poses
        self._joints_tensor_cache[t] = joint_poses



    def clear_motion_cache(self, t: int | None = None):
        if t is None:
            self._global_T_cache = {}
            self._links_tensor_cache = {}
            self._joints_tensor_cache = {}
            self._skin_weights = None
            self._skin_weights_index = None
            self._falloffs = None
            self._falloffs_q = None
        else:
            self._global_T_cache.pop(t, None)
            self._joints_tensor_cache.pop(t, None)
            self._links_tensor_cache.pop(t, None)

    def transform_splats_to_t(self, cano_means: Float32[Tensor, "n 3"],  t: int, cano_quats_wxyz: Float32[Tensor, "n 4"] = None, pose_store: PoseStore | None = None) -> Float32[Tensor, "n 3"] | tuple[Float32[Tensor, "n 3"], Float32[Tensor, "n 4"]]:
        skinned_mat4 = self.get_transformation_at_t(cano_means, t, pose_store=pose_store)
        pred_means_t = anim.apply_mat4(skinned_mat4, cano_means)
        if cano_quats_wxyz is None: return pred_means_t
        else:
            pred_quats_t = anim.apply_mat4_quat(skinned_mat4, cano_quats_wxyz, format='wxyz')
            return pred_means_t, pred_quats_t
    
    def transform_splats_to_ts(self, cano_means: Float32[Tensor, "n 3"],  ts: list[int], cache_all:bool=False, progress=False, 
                               chunk=True) -> Float32[Tensor, "n t 3"]:
        skinned_mat4s = []
        if progress:
            ts = tqdm(ts)
        if chunk:
            for t in ts:  
                t = int(t)
                skinned_mat4s.append(self.get_transformation_at_t(cano_means, t)) # 100 MB per iter (mostly spend on the distance computation)
                if t != self.cano_t and not cache_all:
                    self._links_tensor_cache.pop(t, None) # for less memory 

            skinned_mat4s = torch.stack(skinned_mat4s).transpose(0, 1) # n,t,4,4
            positions = einsum(skinned_mat4s, F.pad(cano_means, (0, 1), value=1.0), "n t i j , n j -> n t i")[:, :, :3]
            return positions
        else:
            positions = []
            for t in ts:  
                t = int(t)
                mat4  = self.get_transformation_at_t(cano_means, t) # n,4,4
                if t != self.cano_t and not cache_all:
                    self._links_tensor_cache.pop(t, None) # for less memory 
                positions.append(einsum(mat4, F.pad(cano_means, (0, 1), value=1.0), "n i j , n j -> n i")[:, :3])
            return torch.stack(positions).transpose(0, 1) # n,t,3
        
    def get_transformation_at_t(self, cano_means: Float32[Tensor, "n 3"], t: int, pose_store: PoseStore | None = None) -> Float32[Tensor, "n 4 4"]:
        t = int(t)
        if self.cano_t not in self._global_T_cache:
            self.compute_link_pose_at_t(self.cano_t, skin_pts=cano_means)

        self.compute_link_pose_at_t(t, pose_store=pose_store) # +20
        if self.type == MotionBlenderType.rigid:
            cano_link_poses, t_link_poses = self._global_T_cache[self.cano_t], self._global_T_cache[t]
        else:
            cano_link_poses, t_link_poses = self._links_tensor_cache[self.cano_t], self._links_tensor_cache[t]

        # if t == self.cano_t and self.flexible_cano:
        #     # t_link_poses = cano_link_poses
        #     # cano_link_poses = torch.eye(4, device=t_link_poses.device).unsqueeze(0).repeat(len(t_link_poses.reshape(-1, 4, 4)), 1, 1).reshape(*t_link_poses.shape[:-2], 4, 4)
        #     Ts = cano_link_poses.clone()
        #     Ts[..., :3, 3] = 0.0
        # else:
        Ts = anim.find_T_between_poses(cano_link_poses.reshape(-1, 4, 4), 
                        t_link_poses.reshape(-1, 4, 4)).reshape(*t_link_poses.shape[:-2], 4, 4) # + 16MB

        if self.type == MotionBlenderType.rigid:
            skinned_mat4 = repeat(Ts, "a b -> x a b", x=len(cano_means))
        else:
            if self.type == MotionBlenderType.deformable and getattr(self, 'deformable_link_quantize', -1) > 0:
                oshape = self._falloffs.shape
                Ts = Ts[self._falloffs_q.flatten(), self._skin_weights_index.flatten()].reshape(*oshape, 4, 4)
            skinned_mat4 = anim.skinning(self._skin_weights, Ts, blend_mode=self.blend_method) # +20MB
        return skinned_mat4
    

    def regularization(self, reg_type: str, **kwargs):
        loss_dict = {}
        if reg_type == 'smooth_motion':
            rot6d = self.global_rot6d[None]
            ts = self.global_ts[None]
            loss_dict['glb_smooth_loss'] = compute_accel_loss(rot6d) + 2 * compute_accel_loss(ts)
            if self.type == MotionBlenderType.deformable:
                loss_dict['joints_smooth_reg'] = compute_accel_loss(self.joints.transpose(0, 1))
            elif self.type == MotionBlenderType.kinematic:
                loss_dict['rot6d_smooth_reg'] = compute_accel_loss(self.rot6d.transpose(0, 1))

        elif reg_type == 'sparse_link_assignment':
            if self.type != MotionBlenderType.rigid:
                loss_dict['sparse_link_reg'] = (1 - (self._skin_weights**2).sum(dim=-1).mean()) 

        elif reg_type == 'minimal_movement_in_cano':
            loss_dict['glb_cano_ts_reg'] = F.l1_loss(self._global_T_cache[self.cano_t][:3, 3], self.cano_global_ts)
            loss_dict['glb_cano_rot_reg'] = F.l1_loss(anim.rmat_to_cont_6d(self._global_T_cache[self.cano_t][:3, :3]), self.cano_global_rot6d)
            if self.type != MotionBlenderType.rigid:
                loss_dict['cano_joints_reg'] = F.l1_loss(self._joints_tensor_cache[self.cano_t], self.cano_joints)
            
            if hasattr(self, 'preinit_joints'):
                for frame_ind, joints_dict in self.preinit_joints.items():
                    self.compute_link_pose_at_t(frame_ind)
                    if frame_ind in self._joints_tensor_cache:
                        if 'indices' in joints_dict:
                            logger.warning("regulating minimal movements for selected joints in preinited frames", once=True)
                            pred_joints = self._joints_tensor_cache[frame_ind]
                            loss_dict['preinit_joints_reg'] = F.l1_loss(pred_joints[joints_dict['indices']], joints_dict['joints'])
                        else:
                            logger.warning("regulating minimal movements not only in the canonical frame, but also in the preinited frames", once=True)
                            pred_joints = self._joints_tensor_cache[frame_ind]
                            loss_dict['preinit_joints_reg'] = F.l1_loss(pred_joints, joints_dict['joints'])
        
        elif reg_type == 'length_reg' and self.type == MotionBlenderType.deformable:
            assert self.type == MotionBlenderType.deformable
            loss_dict['length_reg'] = []
            start_indices, end_indices = self.links[:, 0], self.links[:, 1]
            start, end = self.joints[:, start_indices], self.joints[:, end_indices]
            edge_lens = (end - start).norm(dim=-1)
            cano_start, cano_end = self.cano_joints[start_indices], self.cano_joints[end_indices]
            cano_edge_len = (cano_end - cano_start).norm(dim=-1).detach()
            loss = (edge_lens - cano_edge_len.unsqueeze(0)).abs().mean()
            loss_dict['length_reg'].append(loss)
        elif reg_type == 'arap' and self.type == MotionBlenderType.deformable:
            verts_template = self.cano_joints.unsqueeze(0)
            if not hasattr(self, 'arap_mesh'):
                faces = mesh2d_lib.edges_to_faces(self.links)
                self.arap_mesh = ARAPMeshes(
                    verts=[verts_template[0]],
                    faces=[faces,] 
                    # NOTE: that not all deformable graphs can be convert to faces
                    # FIXME: update the deformable graph initialization (directly initialize a mesh structure instead of homebrew solution)
                )
            loss_dict['arap_loss'] = []
            for frame_ind, verts in self._joints_tensor_cache.items():
                loss_dict['arap_loss'].append(arap_loss(self.arap_mesh, verts.unsqueeze(0), verts_template))
        else:
            logger.error(f'Unknown regularization type: {reg_type}')
            return {}
        
        return reduce_loss_dict(loss_dict)


def links_to_pointset(joints: Float32[Tensor, "j 3"], links: Int64[Tensor, "l 2"], link_scale: float) -> Float32[Tensor, "p 3"]:
    """ convert a joints and links to a pointset to be rendered with gsplats """
    results = [joints]
    start, end = joints[links[:, 0]], joints[links[:, 1]]
    num_steps = torch.round((torch.linalg.norm(end - start, dim=-1) / link_scale)).to(torch.int32)
    for i in range(len(start)):
        steps = torch.linspace(0, 1, num_steps[i], device=joints.device) + link_scale/2
        results.append(start[i][None] + steps[:, None] * (end[i] - start[i])[None])
    return torch.cat(results)