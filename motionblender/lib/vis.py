import torch
import torch.nn.functional as F 
import pyvista as pv
from torch import Tensor
import numpy as np
from PIL import Image

    
def pose_to_mocap(X, name=0, radius=0.02, return_joints_links=False, **kwargs):
    colors = ['red', 'green', 'blue', 'yellow'] 
    if len(X) == 7:
        frame = pose7_to_frame(X)
    else:
        frame = X_to_frame(X)
    if return_joints_links:
        links = [(0, 1), (0, 2), (0, 3)]
        joints = []
        joints_color = ['yellow', 'red', 'green', 'blue']
        links_color = ['red', 'green', 'blue']
        for i in range(4):
            joints.append(frame[i])
        return {'joints': joints, 'links': links, 'joints_color': joints_color, 'links_color': links_color}
    else:
        meshes = []
        for i in range(4):
            sphere = pv.Sphere(radius=radius, center=frame[i])
            meshes.append({'mesh': sphere, 'kwargs': {'color': 'yellow', 'opacity': 0.8, **kwargs}, 'name': f'mocap-{name}.dot.{i}'})
        
        ox = pv.Tube(frame[0], frame[1], radius=radius/2) 
        oy = pv.Tube(frame[0], frame[2], radius=radius/2) 
        oz = pv.Tube(frame[0], frame[3], radius=radius/2)
        for i, l in enumerate([ox, oy, oz]):   
            meshes.append({'mesh': l, 'kwargs': {'color': colors[i], 'line_width': 1, }, 'name': f'mocap-{name}.line.{i}'})
        return meshes


def to_pil(img, mask=None):
    if isinstance(img, Image.Image):
        return img
    scale = 1
    if len(img.shape) == 4:
        img = img[0]
    if img.shape[0] <= 3:
        img = img.permute(1, 2, 0)    
    img = img[..., :3]
    
    if isinstance(img, np.ndarray): img = torch.from_numpy(img.copy())
    img = img.cpu()    
    if img.max() < 3.0:
        scale = 255
    if mask is not None:
        if len(mask.shape) == 2:
            mask = mask[..., None]
    if mask is None:
        return Image.fromarray((scale*img).to(torch.uint8).numpy())
    else:
        mask = (mask > 0).float()
        return Image.fromarray(((img * mask + (1 - mask.float())) * scale).to(torch.uint8).numpy())


# https://qroph.github.io/2018/07/30/smooth-paths-using-catmull-rom-splines.html
def catmoll_rom_spline(p0: Tensor, p1: Tensor, p2: Tensor, p3: Tensor, 
                       alpha: Tensor, tensions: Tensor, t: Tensor) -> Tensor:
    """
    p0, p1, p2, p3: (N, D)
    tensions, alpha, t: (N, ), all range in [0, 1]
    return: (N, D)
    """
    dt0 = torch.norm(p1 - p0, dim=-1, keepdim=True).pow(alpha).clamp_min_(1e-4) 
    dt1 = torch.norm(p2 - p1, dim=-1, keepdim=True).pow(alpha).clamp_min_(1e-4)
    dt2 = torch.norm(p3 - p2, dim=-1, keepdim=True).pow(alpha).clamp_min_(1e-4)
    
    m1 = (p1 - p0) / dt0 - (p2 - p0) / (dt0 + dt1) + (p2 - p1) / dt1
    m2 = (p2 - p1) / dt1 - (p3 - p1) / (dt1 + dt2) + (p3 - p2) / dt2
    m1 *= (dt1 * tensions)
    m2 *= (dt1 * tensions)

    d = p1
    c = m1
    b = -3 * p1 + 3 * p2 - 2 * m1 - m2
    a = 2 * p1 - 2 * p2 + m1 + m2

    t2 = t * t
    t3 = t2 * t
    return a * t3 + b * t2 + c * t + d


# https://theorangeduck.com/page/cubic-interpolation-quaternions
def catmoll_rom_spline_quat(q0: Tensor, q1: Tensor, q2: Tensor, q3: Tensor, t: Tensor) -> Tensor:
    """
    q0, q1, q2, q3: (N, 4) 
    t: (N, )
    """
    t2 = t * t
    t3 = t2 * t

    w1 = 3*t2 - 2*t3
    w2 = t3 - 2*t2 + t
    w3 = t3 - t2

    r1_sub_r0 = quaternion_to_axis_angle(quaternion_multiply(q1, quaternion_invert(q0)))
    r2_sub_r1 = quaternion_to_axis_angle(quaternion_multiply(q2, quaternion_invert(q1)))
    r3_sub_r2 = quaternion_to_axis_angle(quaternion_multiply(q3, quaternion_invert(q2)))
    
    v1 = (r1_sub_r0 + r2_sub_r1) / 2
    v2 = (r2_sub_r1 + r3_sub_r2) / 2

    result = quaternion_multiply(axis_angle_to_quaternion(w1*r2_sub_r1 + w2*v1 + w3*v2), q1);
    return result

####

def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)


def _copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)

def random_quaternions(
    n: int, dtype: torch.dtype = None, device = None
) -> torch.Tensor:
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.

    Args:
        n: Number of quaternions in a batch to return.
        dtype: Type to return.
        device: Desired device of returned tensor. Default:
            uses the current device for the default tensor type.

    Returns:
        Quaternions as tensor of shape (N, 4).
    """
    if isinstance(device, str):
        device = torch.device(device)
    o = torch.randn((n, 4), dtype=dtype, device=device)
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def random_rotations(
    n: int, dtype: torch.dtype = None, device = None
) -> torch.Tensor:
    """
    Generate random rotations as 3x3 rotation matrices.

    Args:
        n: Number of rotation matrices in a batch to return.
        dtype: Type to return.
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type.

    Returns:
        Rotation matrices as tensor of shape (n, 3, 3).
    """
    quaternions = random_quaternions(n, dtype=dtype, device=device)
    return quaternion_to_matrix(quaternions)

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def quaternion_invert(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).
    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """
    scaling = torch.tensor([1, -1, -1, -1], device=quaternion.device)
    return quaternion * scaling


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)



def X_to_pose7(X):
    t = X[:3, -1]
    q = matrix_to_quaternion(torch.from_numpy(X[:3, :3][None]))[0].numpy()
    return np.concatenate([t, q])


def pose7_to_frame(pose, scale=0.1, **kwargs):
    pose = pose.copy()
    R = quaternion_to_matrix(torch.from_numpy(pose[3:])[None]).numpy()[0] * scale
    t = pose[:3]
    return np.array([t, R[:, 0] + t, R[:, 1] + t, R[:, 2] + t])


def X_to_frame(X):
    return pose7_to_frame(X_to_pose7(X))

def frame_to_X(frame):
    frame = np.copy(frame)
    t, x, y, z = frame
    X = np.eye(4)
    X[:3, -1] = t
    x -= t
    y -= t
    z -= t
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    z = z / np.linalg.norm(z)
    # X[0, :3] = x
    # X[1, :3] = y
    # X[2, :3] = z
    X[:3, 0] = x
    X[:3, 1] = y
    X[:3, 2] = z
    return X


def batch_T_to_frame(T, scale=0.1):
    t = T[:, :3, -1]
    R0 = T[:, :3, 0] * scale
    R1 = T[:, :3, 1] * scale
    R2 = T[:, :3, 2] * scale
    return torch.cat([a[:, None, :] for a in [t, R0+t, R1+t, R2+t]], dim=1)


def batch_frame_to_T(frame):
    t, x, y, z = frame[:, 0], frame[:, 1], frame[:, 2], frame[:, 3]
    X = torch.zeros(len(frame), 4, 4)
    X[:, :3, -1] = t
    x -= t
    y -= t
    z -= t
    x = x / torch.norm(x, dim=-1, keepdim=True)
    y = y / torch.norm(y, dim=-1, keepdim=True)
    z = z / torch.norm(z, dim=-1, keepdim=True)
    X[:, :3, 0] = x
    X[:, :3, 1] = y
    X[:, :3, 2] = z
    X[:, 3, 3] = 1
    return X