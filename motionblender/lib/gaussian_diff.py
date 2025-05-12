import math
import torch
from diff_gaussian_rasterization_ext import GaussianRasterizationSettings, GaussianRasterizer
from motionblender.lib.params import SplatsDict
import motionblender.lib.convert_utils as cvt


def get_rasterization_settings(w2c, K, active_sh_degree:int, bg_tensor=None, img_wh=[100, 100], zfar=100.0, znear=0.01):
    """ G4DSample, int, tensor"""
    device = w2c.device
    if 'cuda' in str(device):
        device = 'cuda'
    
    assert w2c.shape == (4, 4)
    assert K.shape == (3, 3)

    if bg_tensor is None: bg_tensor = torch.ones(3).to(device)
    camera_center = torch.inverse(w2c)[:3, 3]
    world_view_matrix = w2c.T

    fov_x, fov_y = cvt.intrinsic_matrix_to_fov(K)
    projection_matrix = cvt.th_projection_matrix_from_fov(znear, zfar, fov_x, fov_y).T.cuda()
    full_proj_transform = (world_view_matrix.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

    tanfovx = math.tan(fov_x * 0.5)
    tanfovy = math.tan(fov_y * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(img_wh[1]),
        image_width=int(img_wh[0]),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_tensor,
        scale_modifier=1.0,
        viewmatrix=world_view_matrix.to(device),
        projmatrix=full_proj_transform.to(device),
        sh_degree=active_sh_degree,
        campos=camera_center.to(device),
        prefiltered=False,
        debug=False
    )
    return raster_settings


def render_with_diffg(rasterizer, splat_dict: SplatsDict, reverse_render_features: torch.Tensor|None=None):
    means2D = torch.zeros_like(splat_dict['means'], dtype=splat_dict['means'].dtype, requires_grad=True, device="cuda") + 0
    try: means2D.retain_grad()
    except: pass

    raster_kwargs = dict( means3D = splat_dict['means'],
            means2D = means2D,
            shs = None,
            colors_precomp = splat_dict['colors'],
            opacities = splat_dict['opacities'].reshape(-1, 1),
            scales = splat_dict['scales'],
            rotations = splat_dict['quats'],
            cov3D_precomp = None)
    
    if reverse_render_features is not None:
        raster_kwargs["colors_to_reverse"] = reverse_render_features

    return_info = rasterizer(**raster_kwargs)

    rendered_feat, radii, depth, accum_T  = return_info[:4]

    pkg = {
            "features": rendered_feat,
            "depth": depth.squeeze(),
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
    }
    if reverse_render_features is not None:
        pkg["reversed_features"] = return_info[4]
    return pkg
