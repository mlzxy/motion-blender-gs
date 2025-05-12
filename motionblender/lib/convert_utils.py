from pathlib import Path
import ujson as json
import numpy as np
import torch
import torch.nn.functional as F
import math
from plyfile import PlyData, PlyElement
from sklearn.neighbors import NearestNeighbors

def T_transform(transformation_matrix, points):
    transformed_points_homogeneous = torch.einsum('ij,kj->ki', transformation_matrix, F.pad(points, (0, 1), value=1.0))
    transformed_points = transformed_points_homogeneous[:, :3] / transformed_points_homogeneous[:, 3:]
    return transformed_points

def get_pointcloud_from_rgb_depth_cam(rgb, depths, c2w, X_2d3d, image_w, image_h):
    """ rgb: [H, W, 3], depths: [H, W], X_WC: [4, 4], X_2d3d: [3, 3] """
    u, v = np.meshgrid(np.arange(image_w), np.arange(image_h))
    u = u.reshape(-1).astype(dtype=np.float32) 
    v = v.reshape(-1).astype(dtype=np.float32)
    rays = np.stack((u, v, np.ones_like(u)), axis=0)  # (3, H*W)
    rays = torch.from_numpy(rays)
    rgbs = rgb.flatten(0, 1)
    
    p_Wraysd = torch.inverse(X_2d3d) @ rays
    p_Wraysd = p_Wraysd.permute(1, 0)
    pts = p_Wraysd * depths.reshape(-1, 1)

    pts = T_transform(c2w, pts)
    return pts, rgbs


def np_RT_from_extrinsics(extrinsics):
    X_WC = extrinsics.cpu().numpy() if isinstance(extrinsics, torch.Tensor) else extrinsics
    X_CW = np.linalg.inv(X_WC)
    R = np.transpose(X_CW[:3, :3])  
    T = X_CW[:3, 3]
    return R, T

def extrinsics_from_np_RT(R, T):
    X_CW = np.eye(4)
    X_CW[:3, 3] = T
    X_CW[:3, :3] = np.transpose(R)
    extrinsics = torch.from_numpy(np.linalg.inv(X_CW)).float()
    return extrinsics


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def intrinsic_matrix_to_fov(K):
    """
    Convert intrinsic matrix to FOV angles in radians.
    Args:
        K (torch.Tensor): 3x3 intrinsic matrix.
    Returns:
        fov_x (float): FOV in radians along the x-axis.
        fov_y (float): FOV in radians along the y-axis.
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    image_width, image_height = cx * 2, cy * 2
    fov_x = 2 * torch.atan(image_width / (2 * fx))
    fov_y = 2 * torch.atan(image_height / (2 * fy))
    return fov_x.item(), fov_y.item()


def th_projection_matrix_from_fov(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right
    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P




def th_intrinsics_from_fov(fovx_rad, fovy_rad, img_w, img_h): # X_2d3d
    fx = img_w / (2 * np.tan(fovx_rad / 2))
    fy = img_h / (2 * np.tan(fovy_rad / 2))
    cx = img_w / 2
    cy = img_h / 2
    K = torch.from_numpy(np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])).float()
    return K


def cam_params_from_intrinsics(K):
    if tuple(K.shape) != (3, 3):
        raise ValueError("The intrinsics matrix K must be a 3x3 matrix.")
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    return {
        'focal_length': float(fy),
        'cx': float(cx),
        'cy': float(cy),
    }


def to_camera_json(intrinsics, extrinsics, width, height):
    R, T = np_RT_from_extrinsics(extrinsics)
    cam = cam_params_from_intrinsics(intrinsics)
    return {
        "focal_length": cam['focal_length'],
        "image_size": [
            width,
            height
        ],
        "orientation": R.T.tolist(),
        "pixel_aspect_ratio": 1.0,
        "position": (- T @ np.linalg.inv(R)).tolist(),
        "principal_point": [
            cam['cx'],
            cam['cy']
        ],
        "radial_distortion": [
            0.0,
            0.0,
            0.0
        ],
        "skew": 0.0,
        "tangential_distortion": [
            0.0,
            0.0
        ]
    }


def from_camera_json(json_path):
    with open(json_path, 'r') as f:
        cam_json = json.load(f)

    w, h = cam_json['image_size']
    FovY = focal2fov(cam_json['focal_length'], h)
    FovX = focal2fov(cam_json['focal_length'], w)
    orientation, position = np.asarray(cam_json['orientation']), np.asarray(cam_json['position'])
    R = orientation.T
    T = - position @ R
    intrinsics = th_intrinsics_from_fov(FovX, FovY, w, h).float()
    extrinsics = extrinsics_from_np_RT(R, T).float()

    return intrinsics, extrinsics, [w, h]

    
    
def knn(x, K: int = 4):
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def init_splat_dict_from_pcd(pts, rgbs):
    if not isinstance(pts, torch.Tensor):
        pts, rgbs = torch.from_numpy(pts).float(), torch.from_numpy(rgbs).float()
    device = pts.device
    assert rgbs.max() <= 1.0 and rgbs.min() >= 0.0

    rotations = torch.zeros((len(pts), 4)) # quaternion
    rotations[:, 0] = 1

    dist2_avg = (knn(pts, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg).unsqueeze(-1).repeat(1, 3)
    
    return dict(
        means = pts, colors = rgbs, scales = dist_avg.to(device), 
        quats = rotations.to(device), opacities = 0.1*torch.ones((len(pts),), dtype=torch.float).to(device)
    )

def fetch_ply(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    return positions, colors