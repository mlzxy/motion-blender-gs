from functools import partial
from loguru import logger
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim
import math
import torch 
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import TypedDict
from motionblender.lib.misc import exponential_decay_func, make_constant_func, quat_activation
from motionblender.lib.eval_sh import eval_sh, SH2RGB, RGB2SH
from jaxtyping import Float32


class SplatsDict(TypedDict):
    means: Float32[Tensor, "g 3"] | Float32[Tensor, "t g 3"]
    quats: Float32[Tensor, "g 4"] | Float32[Tensor, "t g 4"]

    colors:  Float32[Tensor, "g 3"]
    scales:  Float32[Tensor, "g 3"]
    opacities:  Float32[Tensor, " g "]





class GaussianParams(nn.Module):
    def __init__(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        scene_center: torch.Tensor | None = None,
        scene_scale: torch.Tensor | float = 1.0,
        use_sh_color: bool = False,
        sh0: torch.Tensor | None = None,
        sh_rest: torch.Tensor | None = None,
        motion_coefs: torch.Tensor | None = None,
    ):
        super().__init__()
        self.use_sh_color = use_sh_color
        self.active_sh_degree = 0
        params_dict = {
            "means": nn.Parameter(means),
            "quats": nn.Parameter(quats),
            "scales": nn.Parameter(scales),
            "opacities": nn.Parameter(opacities),
        }
        if use_sh_color:
            logger.info('initialize gaussian with spherical harmonics color!')
            # N, K, 1 / N, K, 2
            if sh0 is not None:
                assert sh_rest is not None
                params_dict["sh0"] = nn.Parameter(sh0)
                params_dict["sh_rest"] = nn.Parameter(sh_rest)
            else:
                features = torch.zeros([len(colors), 16, 3])
                assert colors.max() <= 1.0 and colors.min() >= 0.0, "colors should be normalized to [0, 1]"
                colors = RGB2SH(colors)
                features[:, 0, :3] = colors
                params_dict["sh0"] =  nn.Parameter(features[:,:1,:].contiguous().requires_grad_(True)) 
                params_dict["sh_rest"] = nn.Parameter(features[:,1:,:].contiguous().requires_grad_(True))
        else:
            params_dict["colors"] = nn.Parameter(colors)
        
        if motion_coefs is not None:
            params_dict["motion_coefs"] = nn.Parameter(motion_coefs)
        self.params = nn.ParameterDict(params_dict)
        self.quat_activation = quat_activation
        self.color_activation = torch.sigmoid
        self.scale_activation = torch.exp
        self.opacity_activation = torch.sigmoid

        if scene_center is None:
            scene_center = torch.zeros(3, device=means.device)
        self.register_buffer("scene_center", scene_center)
        self.register_buffer("scene_scale", torch.as_tensor(scene_scale))
    
    
    def subset(self, mask: torch.Tensor) -> "GaussianParams":
        if self.use_sh_color:
            gs = GaussianParams(
                means=self.params["means"][mask],
                quats=self.params["quats"][mask],
                scales=self.params["scales"][mask],
                colors=None,
                opacities=self.params["opacities"][mask],
                sh0=self.params["sh0"][mask],
                sh_rest=self.params["sh_rest"][mask],
                use_sh_color=self.use_sh_color,
                scene_center=self.scene_center,
                scene_scale=self.scene_scale,
            )
        else:
            gs = GaussianParams(
                means=self.params["means"][mask],
                quats=self.params["quats"][mask],
                scales=self.params["scales"][mask],
                colors=self.params["colors"][mask],
                opacities=self.params["opacities"][mask],
                use_sh_color=self.use_sh_color,
                scene_center=self.scene_center,
                scene_scale=self.scene_scale,
            )
        gs.active_sh_degree = self.active_sh_degree
        return gs


    @staticmethod
    def init_from_state_dict(state_dict, prefix="params."):
        req_keys = ["means", "quats", "scales", "colors", "opacities"]
        assert all(f"{prefix}{k}" in state_dict for k in req_keys)
        args = {
            "scene_center": torch.zeros(3),
            "scene_scale": torch.tensor(1.0),
        }
        for k in req_keys + list(args.keys()):
            if f"{prefix}{k}" in state_dict:
                args[k] = state_dict[f"{prefix}{k}"]
        return GaussianParams(**args)

    @property
    def num_gaussians(self) -> int:
        return self.params["means"].shape[0]
    
    @property
    def means(self) -> torch.Tensor:
        return  self.params["means"]
    
    def increase_sh_degree(self):
        if self.active_sh_degree < 3 and self.use_sh_color:
            logger.warning(f"increasing sh degree from {self.active_sh_degree} to {self.active_sh_degree + 1}")
            self.active_sh_degree += 1

    def get_colors(self, means3D_final: Float32[Tensor, "N 3"] | None = None, 
                   camera_center: Float32[Tensor, "3"] | None = None, use_default_sh: bool=False) -> torch.Tensor:
        if getattr(self, 'use_sh_color', False):
            if use_default_sh:
                shs_final = torch.cat([self.params["sh0"], self.params["sh_rest"]], dim=1) 
                return SH2RGB(shs_final[:, 0, :3])
                
            assert means3D_final is not None and camera_center is not None
            assert 0 <= self.active_sh_degree <= 3
            shs_final = torch.cat([self.params["sh0"], self.params["sh_rest"]], dim=1) # N, K, 3
            assert tuple(shs_final.shape[1:]) == (16, 3)

            shs_view = shs_final.transpose(1, 2).view(-1, 3, 16)
            dir_pp = means3D_final - camera_center.reshape(1, -1).repeat(shs_final.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
            # assert sh2rgb.max() <= 0.5 and sh2rgb.min() >= -0.5, "sh2rgb should be normalized to [-0.5, 0.5]"
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            return colors_precomp
        else:
            return self.color_activation(self.params["colors"])

    def get_scales(self) -> torch.Tensor:
        return self.scale_activation(self.params["scales"])

    def get_opacities(self) -> torch.Tensor:
        return self.opacity_activation(self.params["opacities"])

    def get_quats(self) -> torch.Tensor:
        return self.quat_activation(self.params["quats"])

    def densify_params(self, should_split, should_dup):
        """
        densify gaussians
        """
        updated_params = {}
        for name, x in self.params.items():
            x_dup = x[should_dup]
            x_split = x[should_split].repeat([2] + [1] * (x.ndim - 1))
            if name == "scales":
                x_split -= math.log(1.6)
            x_new = nn.Parameter(torch.cat([x[~should_split], x_dup, x_split], dim=0))
            updated_params[name] = x_new
            self.params[name] = x_new
        return updated_params

    def cull_params(self, should_cull):
        """
        cull gaussians
        """
        updated_params = {}
        for name, x in self.params.items():
            x_new = nn.Parameter(x[~should_cull])
            updated_params[name] = x_new
            self.params[name] = x_new
        return updated_params

    def reset_opacities(self, new_val):
        """
        reset all opacities to new_val
        """
        self.params["opacities"].data.fill_(new_val)
        updated_params = {"opacities": self.params["opacities"]}
        return updated_params
    
    def to_dict(self, cam_center: Float32[Tensor, "3"] | None = None, use_default_sh: bool=False) -> SplatsDict:
        return {
            "means": self.params['means'],
            "quats": self.get_quats(),
            "scales": self.get_scales(),
            "colors": self.get_colors(means3D_final=self.params['means'], camera_center=cam_center, use_default_sh=use_default_sh),
            "opacities": self.get_opacities()
        }
    


def merge_splat_dicts(*splats: list[SplatsDict]) -> SplatsDict:
    r = {k:v for k , v in splats[0].items()}
    for sp in splats[1:]: r = {k: torch.cat([r[k], sp[k]]) for k in sp.keys()}
    return r

def create_optimizers_for_gses(max_steps: int, dict_of_gsparams: dict[str, GaussianParams], lr_attributes, annealing_for: list[str] = []) -> tuple[dict, dict]:
    optimizers, schedulers = {}, {}
    for gs_name, gs in dict_of_gsparams.items():
        for pname, params in gs.named_parameters.values():
            full_name = gs_name + '.' + pname
            attr_name = pname.split('.')[-1]
            lr = getattr(lr_attributes, attr_name)
            optimizers[full_name] = optim.Adam([{"params": params, "lr": lr, "name": full_name}])
            fnc = partial(exponential_decay_func, max_steps=max_steps, final=0.1 * lr) if attr_name in annealing_for else make_constant_func(1.0)
            schedulers[full_name] = LambdaLR(optimizers[full_name], partial(fnc, init=lr))
    return optimizers, schedulers