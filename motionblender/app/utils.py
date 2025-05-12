import numpy as np
from copy import deepcopy
import torch.nn.functional as F
from einops import einsum
from PIL import Image, ImageDraw, ImageFont
import torch
from viser import ScenePointerEvent
import matplotlib.pyplot as plt
import roma
from jaxtyping import Float32, UInt8

from motionblender.lib.motion import PoseStore, MotionBlender
from scipy.spatial.transform import Rotation

Tensor = torch.Tensor

def X_to_pose7(X):
    t = X[:3, -1]
    q = Rotation.from_matrix(X[:3, :3]).as_quat()
    return np.concatenate([t, q])



def pose7_to_frame(pose, scale=0.2):
    pose = pose.copy()
    R = Rotation.from_quat(pose[3:]).as_matrix() * scale
    t = pose[:3]
    return np.array([t, R[0] + t, R[1] + t, R[2] + t])


def X_to_frame(X, scale=0.5):
    return pose7_to_frame(X_to_pose7(X), scale=scale)

viridis = plt.get_cmap('viridis')
torch_viridis = torch.from_numpy(viridis(np.linspace(0, 1, 1000))[:, :3]).float().cuda()

jet = plt.get_cmap('jet')
torch_jet = torch.from_numpy(jet(np.linspace(0, 1, 1000))[:, :3]).float().cuda()

def torch_cmap(val, cmap='jet'):
    ival = val * 1000.
    ival = torch.clamp(ival, 0, 999).long()
    if cmap == 'viridis':
        return torch_viridis[ival]
    elif cmap == 'jet':
        return torch_jet[ival]
    else:
        raise ValueError(f"Unknown colormap {cmap}")


def interpolate_gripper_degree(start_degree: float, end_degree: float, T: int, window_size: int) -> list[float]:
    result = [start_degree] * T
    window_size = min(window_size, T)
    result[-window_size:] = np.linspace(start_degree, end_degree, window_size).tolist()
    return result


def interpolate_se3(pose1: Float32[Tensor, "4 4"], pose2: Float32[Tensor, "4 4"], t: int) -> Float32[Tensor, "t 4 4"]:
    from motionblender.lib.animate import rt_to_mat4
    ts = torch.linspace(0, 1, int(t)).to(pose1.device)
    trans = pose1[:3, 3].reshape(1, 3) + ts.reshape(-1, 1) * (pose2[:3, 3].reshape(1, 3) - pose1[:3, 3].reshape(1, 3))  # t x 3 

    rotvec1, rotvec2 = roma.rotmat_to_rotvec(pose1[:3, :3]), roma.rotmat_to_rotvec(pose2[:3, :3])
    rotvec_interpolated = roma.rotvec_slerp(rotvec1, rotvec2, ts)
    rots = roma.rotvec_to_rotmat(rotvec_interpolated)
    return rt_to_mat4(rots, trans)


def compute_axis_from_pose(ee_pose: Float32[Tensor, "4 4"]):
    frame = X_to_frame(ee_pose.cpu().numpy(), scale=0.1)
    return {
        'joints': torch.from_numpy(frame),
        'links': torch.as_tensor([[0, 1], [0, 2], [0, 3]]),
        'joints_color': ['yellow', 'red', 'green', 'blue'],
        'links_color': ['red', 'green', 'blue']
    }


class RobotInterface:
    def __init__(self):
        self.buf = {
            'degree': 50,
            'pose': None,
            'stale': True,
            'rot6d': None
        }
        self.inited = False

    def initialize(self, motion_module: MotionBlender) -> None:
        raise NotImplementedError
    
    def get_joint_rotations(self) -> Float32[Tensor, "j 6"]:
        raise NotImplementedError

    def set_gripper_degree(self, degree: int) -> None:
        self.buf['degree'] = degree
        self.buf['stale'] = True

    def set_ee_pose(self, ee_pose: Float32[Tensor, "4 4"]) -> None:
        self.buf['pose'] = ee_pose
        self.buf['stale'] = True

    def get_ee_pose(self) -> Float32[Tensor, "4 4"]:
        return self.buf['pose']



def textsize(text, font):
    im = Image.new(mode="P", size=(0, 0))
    draw = ImageDraw.Draw(im)
    _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
    return width, height

def text_to_image(text, font_size=14, image_size=(400, 300), text_color=(0, 0, 0), bg_color=(255, 255, 255)):
    image = Image.new('RGB', image_size, bg_color)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    y_text = 10
    for line in text.split('\n'):
        width, height = textsize(line, font=font)
        draw.text(((image_size[0] - width) / 2, y_text), line, fill=text_color, font=font)
        y_text += height + 5

    return np.array(image)


def w2c_to_opengl_camera_pose(w2c):
    w2c = deepcopy(w2c)
    R_flip = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    w2c[:3, :3] = w2c[:3, :3] @ torch.from_numpy(R_flip).float().to(w2c.device)

    T_conversion = torch.from_numpy(np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ])).float().to(w2c.device)
    camera_pose = torch.linalg.inv(w2c)
    camera_pose =  T_conversion @ camera_pose @ T_conversion

    return camera_pose

    
    
def client_message(client, message, title="Notice!", warning=False, danger=False, success=False, auto_close=True, loading=False):
    color = None
    if warning:
        color = 'orange'
    elif danger:
        color = 'red'
    elif success:
        color = 'green'
    client.add_notification(title, message, auto_close=auto_close, color=color, loading=loading)
    


def find_closest_click_id(points: Float32[Tensor, "n 3"], click: ScenePointerEvent, dev: torch.device, cosine_thr=0.95):
    ray_origin, ray_dir = torch.as_tensor(click.ray_origin).to(dev), torch.as_tensor(click.ray_direction).to(dev)
    many_rays = points - ray_origin[None]
    dotprod = einsum(F.normalize(many_rays, dim=1), F.normalize(ray_dir, dim=0), 'n d, d -> n')
    contact_ind = dotprod.argmax()
    if dotprod[contact_ind] < cosine_thr:
        return -1
    else:
        return contact_ind.item()
    # colinear_ind = (dotprod > cosine_thr).nonzero().flatten()
    # # if len(colinear_ind) == 0:
    # #     return -1
    # contact_ind = colinear_ind[many_rays[colinear_ind].norm(dim=1).argmin()]
    # return contact_ind.item()


def evt_to_dir(evt): 
    value = evt.target.value
    sign = 1 if '+' in value else -1
    value = value[:1]
    if value == 'x':
        axis = 0
    elif value == 'y':
        axis = 1
    elif value == 'z':
        axis = 2
    else:
        raise ValueError(f"invalid value {value}")
    return sign, axis


def set_modules_grad_enable(modules, val):
    if 'dict' in modules.__class__.__name__.lower(): modules = modules.values()
    for m in modules:
        for p in m.parameters():
            if p.dtype in [torch.float32, torch.double]:
                p.requires_grad = val