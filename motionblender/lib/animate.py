"""
notes: 
- all quat fits roma xyzw convention
"""
import torch, roma
import torch.optim as optim
from torch import Tensor
from jaxtyping import Float32, Int64
from beartype import beartype as typechecker
import torch.nn.functional as F
import torch.nn as nn
from copy import deepcopy
import numpy as np
from einops import pack, unpack, reduce, rearrange, repeat, einsum
from typing import TypedDict, Callable
import torch.linalg as tl
import logging
from loguru import logger


#region 3dmath


def rmat_to_cont_6d(matrix):
    """
    :param matrix (*, 3, 3)
    :returns 6d vector (*, 6)
    """
    return torch.cat([matrix[..., 0], matrix[..., 1]], dim=-1)


def cont_6d_to_rmat(cont_6d):
    """
    :param 6d vector (*, 6)
    :returns matrix (*, 3, 3)
    """
    x1 = cont_6d[..., 0:3]
    y1 = cont_6d[..., 3:6]

    x = F.normalize(x1, dim=-1)
    y = F.normalize(y1 - (y1 * x).sum(dim=-1, keepdim=True) * x, dim=-1)
    z = tl.cross(x, y, dim=-1)

    return torch.stack([x, y, z], dim=-1)

def pad1(x):
    return F.pad(x, (0, 1), value=1)

@typechecker
def rt_to_mat4(
    R: Float32[Tensor, "* 3 3"], t: Float32[Tensor, "* 3"], s:  Float32[Tensor, "*"] | None = None
) ->  Float32[Tensor, "* 4 4"]:
    mat34 = torch.cat([R, t[..., None]], dim=-1)
    if s is None:
        bottom = (
            mat34.new_tensor([[0.0, 0.0, 0.0, 1.0]])
            .reshape((1,) * (mat34.dim() - 2) + (1, 4))
            .expand(mat34.shape[:-2] + (1, 4))
        )
    else:
        bottom = F.pad(1.0 / s[..., None, None], (3, 0), value=0.0)
    mat4 = torch.cat([mat34, bottom], dim=-2)
    return mat4

@typechecker
def mat4_to_rt(mat4: Float32[Tensor, "* 4 4"]) -> tuple[Float32[Tensor, "* 3 3"], Float32[Tensor, "* 3"]]:
    """
    Args:
        mat4 (torch.Tensor): (..., 4, 4).

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]: 
            - R (torch.Tensor): (..., 3, 3).
            - t (torch.Tensor): (..., 3).
    """
    R = mat4[..., :3, :3]
    t = mat4[..., :3, 3]
    return R, t


@typechecker
def apply_mat4(mat: Float32[Tensor, "b 4 4"] | Float32[Tensor, "4 4"], pts: Float32[Tensor, "b 3"]) -> Float32[Tensor, "b 3"]:
    if mat.dim() == 2: 
        # mat = repeat(mat, 'a b -> l a b', l=len(pts)).clone()
        return einsum(mat, pad1(pts), 'a b, n b -> n a')[:, :3]
    else:
        return einsum(mat, pad1(pts), 'n a b, n b -> n a')[:, :3]

@typechecker
def apply_mat4_pose(mat: Float32[Tensor, "b 4 4"] | Float32[Tensor, "4 4"], poses: Float32[Tensor, "b 4 4"]) -> Float32[Tensor, "b 4 4"]:
    if mat.dim() == 2:
        # mat = repeat(mat, 'a b -> l a b', l=len(poses)).clone()
        return einsum(mat, poses, 'i j, l j k -> l i k')
    else:
        return einsum(mat, poses, 'l i j, l j k -> l i k')


@typechecker
def apply_mat4_quat(mat: Float32[Tensor, "b 4 4"] | Float32[Tensor, "4 4"], q: Float32[Tensor, "b 4"], format='xyzw') -> Float32[Tensor, "b 4"]:
    if format != 'xyzw':
        q = roma.quat_wxyz_to_xyzw(q)
    
    if mat.dim() == 2: mat = mat.unsqueeze(0)
    q = roma.quat_product(roma.rotmat_to_unitquat(mat[..., :3, :3]), q)
    if format != 'xyzw':
        q = roma.quat_xyzw_to_wxyz(q)
    return q
    
@typechecker
def find_T_between_poses(mat4a: Float32[Tensor, "b 4 4"], mat4b: Float32[Tensor, "b 4 4"]) -> Float32[Tensor, "b 4 4"]:
    """ return the relative transformation from pose-a to pose-b """
    R = einsum(mat4b[:, :3, :3], mat4a[:, :3, :3].transpose(1, 2), 'a i j, a j k -> a i k')
    t = mat4b[:, :3, 3] - einsum(R, mat4a[:, :3, 3], 'a i j, a j -> a i')
    return rt_to_mat4(R,  t)


@typechecker
def solve_procrustes(X: Float32[Tensor, "b p 3"], Y: Float32[Tensor, "b p 3"], w: None | Float32[Tensor, "b p"]=None, 
         eps: float=0.0001) -> Float32[Tensor, "b 4 4"]:
    if w is None: w = torch.ones((X.shape[0], X.shape[1], 1), dtype=torch.float32, device=X.device)
    W1 = torch.abs(w).sum(dim=1, keepdim=True)
    w_norm = w / (W1 + eps)
    mean_X = (w_norm * X).sum(dim=1, keepdim=True)
    mean_Y = (w_norm * Y).sum(dim=1, keepdim=True)
    Sxy = torch.matmul( (Y - mean_Y).transpose(1,2), w_norm * (X - mean_X) ).double()
    U, D, V = Sxy.svd()
    # condition = (D.max(dim=1)[0] / D.min(dim=1)[0]).float()
    S = torch.eye(3)[None].repeat(len(X),1,1).double().to(X.device)
    UV_det = U.det() * V.det()
    S[:, 2:3, 2:3] = UV_det.view(-1, 1, 1)
    svT = torch.matmul(S, V.transpose(1, 2))
    R = torch.matmul(U, svT).float()
    t = mean_Y.transpose(1, 2) - torch.matmul(R, mean_X.transpose(1,2))
    return rt_to_mat4(R.float(), t.float()[..., 0])


#endregion


#region weight-inpaint

@typechecker
def compute_distance_from_link(points: Float32[Tensor, "p c"], 
        link_starts: Float32[Tensor, "b c"], 
        link_ends: Float32[Tensor, "b c"], 
        reference_direction: None | Float32[Tensor, " c "]=None) \
            -> tuple[Float32[Tensor, "p b"], Float32[Tensor, "p b c"], Float32[Tensor, "p b"]]:
    """ 
    A---D-------------B

        C
    AD = AB * (AB dot AC) / (AB dot AB)

    points: (p, 3)
    link_start: (b, 3)
    link_ends: (b, 3)

    return: (p, b), (p, b, 3), (p, b)
    """
    link_directions = link_ends - link_starts # AB, shape: (b, 3)
    points_to_starts = points.unsqueeze(1) - link_starts.unsqueeze(0)  # AC shape: (p, b, 3)
    t = einsum(points_to_starts, link_directions, 'p b c, b c -> p b') \
        / einsum(link_directions, link_directions, 'b c, b c -> b').unsqueeze(0)
    t = torch.clamp(t, 0.0, 1.0)
    
    closest_points = link_starts.unsqueeze(0) + einsum(t, link_directions, 'p b, b c -> p b c') # AD shape: (p, b, 3)
    unnormed_radiance = points.unsqueeze(1) - closest_points
    distances = torch.norm(unnormed_radiance, dim=2)  # shape: (p, b)
    radiance_direction = F.normalize(unnormed_radiance, dim=2) # shape: (p, b, 3)
    return distances, radiance_direction, t

# for raidiance gaussian kernel
class RadianceFunction(nn.Module):
    def __init__(self, init_gamma: float=5.0, bias: bool=False, hidden_size:int=16, num_hidden_layers:int=1,
                 out_activation: Callable=torch.exp, use_embed: bool=False, embed_kwargs: dict={}):
        super().__init__()
        self.register_parameter('base_gamma', nn.Parameter(torch.tensor(init_gamma, dtype=torch.float32)))
        self.register_parameter('beta', nn.Parameter(torch.tensor(0.0, dtype=torch.float32)))
        self.out_activation = out_activation
        self.embedder = lambda x: x 
        input_size = 4
        if use_embed: # from NeRF
            input_size, embed_fns = 0, []
            if embed_kwargs.get('include_input', True): 
                embed_fns.append(lambda x : x)
                input_size += 4
            N_freqs = embed_kwargs.get('num_freqs', 10)
            max_freq = embed_kwargs.get('max_freq_log2', N_freqs-1)
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
            for freq in freq_bands:
                for p_fn in embed_kwargs.get('periodic_fns', [torch.sin, torch.cos]):
                    embed_fns.append(lambda x, p_fn=p_fn, freq=float(freq) : p_fn(x * freq))
                    input_size += 4
            def _(x):
                return torch.cat([fn(x) for fn in embed_fns], -1)
            self.embedder = _
            
        # if input_size > hidden_size:
        #     logger.warning(f"input size ({input_size}) is larger than hidden size ({hidden_size}), possibly because of embedding")

        self.gamma = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=bias), 
            nn.Mish(),
            *sum([[nn.Linear(hidden_size, hidden_size, bias=bias), nn.Mish()] 
                  for _ in range(num_hidden_layers)], []), 
            nn.Linear(hidden_size, 1, bias=False) 
        ) 

    def forward(self, d: Float32[Tensor, "p 3"], t: Float32[Tensor, " p "]):
        x = self.embedder(torch.cat([d, t.unsqueeze(-1)], dim=-1))
        return self.out_activation(self.base_gamma + self.beta * self.gamma(x))


@typechecker
def weight_inpaint(pts: Float32[Tensor, "p c"], joints: Float32[Tensor, "j c"], 
        connections: Int64[Tensor, "l 2"], 
        gamma: float | Float32[Tensor, " l "] | nn.Module | list = 5.0, 
        temperature: float | Float32[Tensor, " l "] = 0.25, 
        reference_direction: None | Float32[Tensor, "3"]=None,
        return_falloff: bool=False, 
        return_distance: bool = False) -> Float32[Tensor, "p l"] | tuple[Float32[Tensor, "p l"], Float32[Tensor, "p l"]]:
    """
    > weight_inpaint(pts, joints, connections, gamma=5., temperature=0.2) 
    tensor([[0.9931, 0.0069],
        [0.5000, 0.5000],
        [0.0069, 0.9931]])
    """
    distances, radiance_dirs, falloff = compute_distance_from_link(pts, joints[connections[:, 0]], joints[connections[:, 1]]) 

    if isinstance(gamma, (nn.Module, list)): 
        lst_of_radiance_functions = gamma
        assert len(lst_of_radiance_functions) == len(connections) == radiance_dirs.shape[1] == falloff.shape[1]
        gamma = []
        for link_id in range(len(connections)):
            assert isinstance(lst_of_radiance_functions[link_id], RadianceFunction), "radiance function must be a subclass of RadianceFunction"
            gamma.append(lst_of_radiance_functions[link_id](radiance_dirs[:, link_id], falloff[:, link_id]))
        gamma = torch.cat(gamma, dim=1)

    if isinstance(gamma, Tensor) and gamma.dim() == 1: gamma = gamma.unsqueeze(0)
    if isinstance(temperature, Tensor) and temperature.dim() == 1: temperature = temperature.unsqueeze(0)

    weights = torch.exp(-gamma * distances) / temperature
    weights = F.softmax(weights, dim=1)
    if return_distance: return weights, distances
    if return_falloff: return weights, falloff
    else: return weights

#endregion


#region deform-grid

DEFAULT_UP_TENSOR = torch.as_tensor([0, 0, 1])

@typechecker
def compute_normals(pts: Float32[Tensor, "p 3"], triangles: Int64[Tensor, "t 2 2"], default_uptensor=DEFAULT_UP_TENSOR) -> Float32[Tensor, "p 3"]:
    e0 = pts[triangles[:, 0, 1]] - pts[triangles[:, 0, 0]]
    e1 = pts[triangles[:, 1, 1]] - pts[triangles[:, 1, 0]]
    normals = tl.cross(e0, e1) 
    is_colinear  = (normals == 0).all(1)
    normals[is_colinear] = default_uptensor.to(normals.dtype).to(normals.device)
    normals = F.normalize(normals, dim=1)
    normals = repeat(normals, 't c -> (t x) c', x=2)
    return normals

@typechecker
def find_link_ctrl_pt_pose(
                link_start: Float32[Tensor, "b 3"], link_end: Float32[Tensor, "b 3"], 
                normal: Float32[Tensor, "b 3"], falloff: Float32[Tensor, " b "] | float = 0.5) -> Float32[Tensor, "b 4 4"]:
    dir = link_end - link_start
    if isinstance(falloff, Tensor): falloff = falloff.unsqueeze(-1)
    o = link_start + falloff * dir
    z = F.normalize(dir, dim=1)
    x= F.normalize(tl.cross(normal, z), dim=1)
    y = F.normalize(tl.cross(z, x), dim=1)
    return rt_to_mat4(pack([x, y, z], "b c *")[0], o)


#endregion



#region kinematic-chain


class KinematicLink(TypedDict):
    id: int
    name: str | None
    length: Float32[Tensor, "1"] | None
    rot6d: Float32[Tensor, "6"] | None
    chain: list['KinematicLink'] | None


def find_root_joint_id(links: list[tuple[int, int]], n_joints: int | list | None = None, return_all_roots: bool=False, return_largest: bool=False, return_smallest: bool=False) -> int | list[int] | tuple[int, list[int]]:
    indegree = {}
    if n_joints is None: n_joints = sorted(set(sum([list(l) for l in links], [])))
    for n in (range(n_joints) if isinstance(n_joints, int) else n_joints): indegree[n] = 0
    for link in links: 
        indegree[link[1]] += 1
    roots = [i for i in indegree if indegree[i] == 0]
    assert len(roots) != 0, "no root joint found, cycle detected"
    if return_all_roots: 
        adj_list = {}
        for a, b in links: adj_list.setdefault(a, []).append(b)
        def collect_children(node):
            if node not in adj_list: return [node]
            return [node] + sum([collect_children(child) for child in adj_list[node]], [])
        roots = sorted([(collect_children(r), r) for r in roots], key=lambda x: len(x[0]), reverse=True) # put largest tree first
        if return_largest:
            return roots[0][1], roots[0][0]
        elif return_smallest:
            return roots[-1][1], roots[-1][0]
        else:
            return [r[1] for r in roots]
    else:
        assert len(roots) == 1, "only one root is allowed in a kinematic tree"
        return roots[0]


def link_poses_to_joint_positions(link_poses: Float32[Tensor, "l 4 4"], 
            root_joint_id: int, root_joint_position: None | Float32[Tensor, "3"] = None) -> Float32[Tensor, "l+1 3"]:
    """ assuming the link poses are ordered by the id of the end joint 
        convert the link poses to normalized joint positions 
    """
    joints = torch.zeros(len(link_poses) + 1, 3, device=link_poses.device)
    indices = sorted(set(range(len(joints))) - {root_joint_id})  
    joints[indices] = link_poses[:, :3, 3]
    if root_joint_position is not None: 
        joints[root_joint_id] = root_joint_position
    return joints


@typechecker
def forward_kinematic(chain: KinematicLink | nn.ParameterDict, length_activation: Callable = lambda x: x,
                      rot_activation: Callable = lambda x: x) -> Float32[Tensor, "l 4 4"]:
    """
    the return link poses are ordered by the id of the end joint
    """
    root_mat = torch.eye(4)
    poses = {}

    def walk(link, T):
        for child in link['chain']:
            if T.device != child['length'].device: T = T.to(child['length'].device)
            T_trans, T_rot = torch.eye(4, device=T.device), torch.eye(4, device=T.device)
            T_trans[2, 3] = length_activation(child['length'])
            T_rot[:3, :3] = cont_6d_to_rmat(rot_activation(child['rot6d']))
            new_T = T @ (T_rot @ T_trans) # first translate, then rotate in local frame
            poses[child['id']] = new_T
            if child['chain']:
                walk(child, new_T) 

    walk(chain, root_mat)
    return pack([v for k, v in sorted(poses.items())], '* a b')[0]


@typechecker
def inverse_kinematic(joints: Float32[Tensor, "j 3"], connections: list[tuple[int, int] | list[int]], 
                      length_inv_activation: Callable = lambda x: x, log: bool=False) -> KinematicLink:
    """ this is a simplified version of inverse kinematic (just tracing back joints without any gradient-descent going on), 
        because we already know the joint positions 
    """
    adj, indegree = {}, {i: 0 for i in range(len(joints))}
    last_b = -1
    for a, b in connections: 
        adj.setdefault(a, []).append(b) 
        indegree[b] += 1
        assert b > last_b, "the connections must be sorted by the end joint id"
        last_b = b
    roots = [i for i in range(len(joints)) if indegree[i] == 0]
    assert len(roots) == 1, "only one root is allowed"
    root = roots[0]
    assert torch.all(joints[root] == 0.), "root joint must be at origin"

    root_mat = torch.eye(4).to(joints.device)
    up = torch.tensor([0, 0, 1], dtype=joints.dtype, device=joints.device)

    def walk(node_id, T, lvl=0, parent=None):
        chain = []
        if log: logger.info("  " * lvl, node_id) 
        new_node = KinematicLink(id=node_id, length=None, rot6d=None)
        if parent is not None:
            end, start = joints[node_id], joints[parent['id']]
            raw_length = torch.norm(end - start)
            length = length_inv_activation(raw_length)
            new_node['length'] = length
            
            end = einsum(T, pad1(end), 'a b, b -> a')[:3]
            # start = einsum(T, pad1(start), 'a b, b -> a')[:3] # start always 0
            ndir = F.normalize(end, dim=0)
            if torch.allclose(ndir.abs(), up):
                T_ = torch.eye(4).to(joints.device)
                if ndir[-1] < 0:
                    T_[0, 0] = -1
                    T_[2, 2] = -1
                T_[:3, 3] = -end
            else:
                vec = F.normalize(tl.cross(ndir, up, dim=0), dim=0)
                angle = torch.acos(torch.dot(ndir, up))
                R = roma.rotvec_to_rotmat(angle * vec)
                T_ = rt_to_mat4(R, torch.einsum("ij,j->i", -R, end))

            new_node['rot6d'] = rmat_to_cont_6d(T_[:3, :3].t())
            new_T = T_ @ T 
        else:
            new_T = T

        for child in adj.get(node_id, []):
            chain.append(walk(child, new_T, lvl=lvl+1, parent=new_node))
        new_node['chain'] = chain
        return new_node
    
    if log: logger.info("chain: ")
    chain = walk(root, root_mat) 
    return chain 


def retrieve_node_from_chain(chain: KinematicLink, node_id: int):
    """ return the node with the given id """
    def walk(node):
        if node['id'] == node_id:
            return node
        for child in node['chain']:
            res = walk(child)
            if res is not None:
                return res
        return None
    return walk(chain) 


def retrieve_tensor_from_chain(chain: KinematicLink, key: str, return_linkid2indice: bool=False) -> Tensor:
    """ return the tensor of the key from the chain (all nodes concated) """
    params = {}
    def walk(link):
        for child in link['chain']:
            params[child['id']] = child[key]
            if child['chain']: walk(child)
    walk(chain)
    if return_linkid2indice:
        linkid2indice = {}
        for i, k in enumerate(sorted(params.keys())):
            linkid2indice[k] = i
        
    params = torch.stack([v for k, v in sorted(params.items())])
    if return_linkid2indice:
        return params, linkid2indice
    else:
        return params

def create_hollow_chain_wo_tensor(chain: KinematicLink) -> KinematicLink:
    """ remove the all tensor from the chain, so the chain only works as a structure """
    def walk(node):
        new_node = {
            'id': node['id'],
            'name': node.get('name', None),
            'chain': []
        }
        for child in node['chain']:
            new_node['chain'].append(walk(child))
        return new_node
    return walk(chain)


def fill_hollow_chain_with_tensor(hollow_chain: KinematicLink, length_tensor: Float32[Tensor, " l "], 
                                  rot6d_tensor: Float32[Tensor, "b 6"], 
                                  linkid2indice: dict[int, int], 
                                  length_linkid2indice: dict[int, int] | None=None) -> KinematicLink:
    """ fill the hollow chain with the given tensors """
    if length_linkid2indice is None: length_linkid2indice = linkid2indice

    hollow_chain = deepcopy(hollow_chain)
    def walk(node):
        if node['id'] in linkid2indice:
            node['length'] = length_tensor[length_linkid2indice[node['id']]]
            node['rot6d'] = rot6d_tensor[linkid2indice[node['id']]]
        else:
            node['length'] = None
            node['rot6d'] = None
        for child in node['chain']:
            walk(child)
        return node
    return walk(hollow_chain)
    
#endregion


#region skinning

@typechecker
def dual_quat_normalize(q: Float32[Tensor, "b 8"]) -> Float32[Tensor, "b 8"]:
    norm_qr = q[:, :4].norm(dim=1, keepdim=True)
    norm_qr_sq = norm_qr ** 2
    q = q.clone() / norm_qr
    real, dual = q[:, :4], q[:, 4:]
    dual = dual - (einsum(real, dual, 'b a, b a -> b')[:, None] * norm_qr_sq * real)
    return pack([real, dual], 'b *')[0]


@typechecker
def dual_quat_mult(q1: Float32[Tensor, "b 8"], q2: Float32[Tensor, "b 8"]) -> Float32[Tensor, "b 8"]:
    q1r, q1d, q2r, q2d = q1[:, :4], q1[:, 4:], q2[:, :4], q2[:, 4:]
    qr_prod = roma.quat_product(q1r, q2r)
    qd_prod = roma.quat_product(q1r, q2d) + roma.quat_product(q1d, q2r)
    return pack([qr_prod, qd_prod], 'b *')[0]


@typechecker
def dual_quat_to_mat(q: Float32[Tensor, "b 8"]) -> Float32[Tensor, "b 4 4"]:
    """ assume q is normalized """
    mat = repeat(torch.eye(4, device=q.device), "a b -> c a b", c=len(q)).clone()
    mat[:, :3, :3] = roma.unitquat_to_rotmat(q[:, :4])
    qr = q[:, :4] # qr = qr * torch.sign(qr[:, -1:])
    qd = q[:, 4:]
    qr_conj = roma.quat_conjugation(qr) # conjugate
    trans = roma.quat_product(2.0 * qd, qr_conj)
    mat[:, :3, 3] = trans[:, 0:3]
    return mat


@typechecker
def dual_quat_from_mat(mat: Float32[Tensor, "b 4 4"]) -> Float32[Tensor, "b 8"]:
    qr = roma.rotmat_to_unitquat(mat[:, :3, :3])
    qd_ = pack([mat[:, :3, 3], torch.zeros(len(mat), 1, device=mat.device)], 'b *')[0]
    qd = roma.quat_product(qd_, qr) * 0.5
    return pack([qr, qd], 'b *')[0]


@typechecker
def skinning(weights: Float32[Tensor, "p m"],  mat4s: Float32[Tensor, "m 4 4"] | Float32[Tensor, "p m 4 4"],
             blend_mode: str='linear') -> Float32[Tensor, "p 4 4"]:
    """ weights must be normalized to sum 1 at last dim"""
    assert blend_mode in ['dq', 'linear']
    if blend_mode == 'linear':
        if len(mat4s.shape) == 3:
            mat4s = repeat(mat4s, 'm a b -> p m a b', p=len(weights))
        else: assert len(mat4s.shape) == 4
        return einsum(weights, mat4s, 'p m, p m a b -> p a b')
    else:
        if len(mat4s.shape) == 3:
            dqs = dual_quat_from_mat(mat4s) # m 8
            dqs = repeat(dqs, "m a -> p m a", p=len(weights))
        else:
            dqs = rearrange(dual_quat_from_mat(rearrange(mat4s, 'p m a b -> (p m) a b')), 
                            '(p m) a -> p m a', p=len(weights))
        q = einsum(weights, dqs, 'p m, p m a -> p a')
        q = dual_quat_normalize(q)
        return dual_quat_to_mat(q)

#endregion