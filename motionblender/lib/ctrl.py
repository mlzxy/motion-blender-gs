import torch
from functools import partial
from motionblender.lib.params import GaussianParams
from motionblender.lib.misc import make_constant_func, exponential_decay_func, get_learnable_parameters, to_fg_names
import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from dataclasses import dataclass, field
from argparse import Namespace
from loguru import logger


@dataclass
class LRConfig:
    means: float = 1.6e-4
    opacities: float = 5e-2
    scales: float = 5e-3
    quats: float = 1e-3 # wxyz
    colors: float = 1e-2

    sh0: float = 0.0025
    sh_rest: float = 0.000125

    motions: float = 1e-3
    motion_coefs: float = 1.6e-4
    cameras: float = 1e-3
    annealing_for: list[str] = field(default_factory=lambda: ['scales', 'motions', 'cameras'])
    annealing_factor: float = 0.1




@dataclass
class ControlConfig:
    warmup_steps: int = 200
    control_every: int = 100
    reset_opacity_every_n_controls: int = 30 # 30 * 100 -> 3000
    stop_control_by_screen_steps: int = 4000
    stop_control_steps: int = 4000
    ### Densify.
    densify_xys_grad_threshold: float = 0.0002
    densify_scale_threshold: float = 0.01
    densify_screen_threshold: float = 0.05
    stop_densify_steps: int = 15000
    ### Cull.
    cull_opacity_threshold: float = 0.1
    cull_scale_threshold: float = 0.5
    cull_screen_threshold: float = 0.15


def dup_in_optim(optimizer, new_params: list, should_dup: torch.Tensor, num_dups: int):
    assert len(optimizer.param_groups) == len(new_params)
    for i, p_new in enumerate(new_params):
        old_params = optimizer.param_groups[i]["params"][0]
        param_state = optimizer.state[old_params]
        if len(param_state) == 0:
            return
        for key in param_state:
            if key == "step":
                continue
            p = param_state[key]
            param_state[key] = torch.cat(
                [p[~should_dup], p.new_zeros(num_dups, *p.shape[1:])],
                dim=0,
            )
        del optimizer.state[old_params]
        optimizer.state[p_new] = param_state
        optimizer.param_groups[i]["params"] = [p_new]
        del old_params
        torch.cuda.empty_cache()


def remove_from_optim(optimizer, new_params: list, _should_cull: torch.Tensor):
    assert len(optimizer.param_groups) == len(new_params)
    for i, p_new in enumerate(new_params):
        old_params = optimizer.param_groups[i]["params"][0]
        param_state = optimizer.state[old_params]
        if len(param_state) == 0:
            return
        for key in param_state:
            if key == "step":
                continue
            param_state[key] = param_state[key][~_should_cull]
        del optimizer.state[old_params]
        optimizer.state[p_new] = param_state
        optimizer.param_groups[i]["params"] = [p_new]
        del old_params
        torch.cuda.empty_cache()


def reset_in_optim(optimizer, new_params: list):
    assert len(optimizer.param_groups) == len(new_params)
    for i, p_new in enumerate(new_params):
        old_params = optimizer.param_groups[i]["params"][0]
        param_state = optimizer.state[old_params]
        if len(param_state) == 0:
            return
        for key in param_state:
            param_state[key] = torch.zeros_like(param_state[key])
        del optimizer.state[old_params]
        optimizer.state[p_new] = param_state
        optimizer.param_groups[i]["params"] = [p_new]
        del old_params
        torch.cuda.empty_cache()



class AdaptiveController:
    """ refactor from `flow3d.trainer` """

    def __init__(self, ctrl_cfg: ControlConfig, 
                 params: dict[str, GaussianParams],
                 optimizers: dict[str, optim.Optimizer], 
                 gaussian_names: list[str],
                 device: str="cuda", num_frames: int=500):
        self.ctrl_cfg = ctrl_cfg
        self.optimizers = optimizers
        self.params = params
        num_gaussians = self.num_gaussians
        self.gaussian_names = gaussian_names
        self.num_frames = num_frames

        self.reset_opacity_every = self.ctrl_cfg.reset_opacity_every_n_controls * self.ctrl_cfg.control_every
        self.running_stats = {
            "xys_grad_norm_acc": torch.zeros(num_gaussians, device=device),
            "vis_count": torch.zeros(num_gaussians, device=device, dtype=torch.int64),
            "max_radii": torch.zeros(num_gaussians, device=device),
        }
        self.control_cache = { 'img_wh': [], 'radii': [], 'xys': [] }
    
    @property
    def num_gaussians(self):
        return sum([p.num_gaussians for p in self.params.values()])
    
    @property
    def num_fg_gaussians(self):
        return sum([self.params[p].num_gaussians for p in to_fg_names(self.gaussian_names)])

    def get_scales_all(self):
        return torch.cat([self.params[p].get_scales() for p in self.gaussian_names])
    
    def get_opacities_all(self):
        return torch.cat([self.params[p].get_opacities() for p in self.gaussian_names])

    def run_control_steps(self, global_step):
        cfg = self.ctrl_cfg
        ready = self._prepare_control_step()
        if (
            ready
            and global_step > cfg.warmup_steps
            and global_step % cfg.control_every == 0
            and global_step < cfg.stop_control_steps
        ):
            if (
                global_step < cfg.stop_densify_steps
                and global_step % self.reset_opacity_every > self.num_frames
            ):
                self._densify_control_step(global_step)
            if global_step % self.reset_opacity_every > min(3 * self.num_frames, 1000):
                self._cull_control_step(global_step)
            if global_step % self.reset_opacity_every == 0:
                self._reset_opacity_control_step()

            # Reset stats after every control.
            for k in self.running_stats:
                self.running_stats[k].zero_()

    @torch.no_grad()
    def _prepare_control_step(self) -> bool:
        num_gaussians = self.num_gaussians
        # Prepare for adaptive gaussian control based on the current stats.
        if len(self.control_cache['xys']) == 0:
            logger.warning("Model not training, skipping control step preparation")
            return False

        batch_size = len(self.control_cache['xys'])
        # these quantities are for each rendered view and have shapes (C, G, *)
        # must be aggregated over all views
        for _current_xys, _current_radii, _current_img_wh in zip(
            self.control_cache['xys'], self.control_cache['radii'], self.control_cache['img_wh']
        ):
            _current_radii = _current_radii[:, :num_gaussians]
            sel = _current_radii > 0
            gidcs = torch.where(sel)[1]
            # normalize grads to [-1, 1] screen space
            xys_grad = _current_xys.grad.clone()
            if xys_grad.dim() == 2:
                xys_grad = xys_grad[None, :num_gaussians, :2]
            else:
                xys_grad = xys_grad[:, :num_gaussians]
            xys_grad[..., 0] *= _current_img_wh[0] / 2.0 * batch_size
            xys_grad[..., 1] *= _current_img_wh[1] / 2.0 * batch_size
            self.running_stats["xys_grad_norm_acc"].index_add_(
                0, gidcs, xys_grad[sel].norm(dim=-1)
            )
            self.running_stats["vis_count"].index_add_(
                0, gidcs, torch.ones_like(gidcs, dtype=torch.int64)
            )
            max_radii = torch.maximum(
                self.running_stats["max_radii"].index_select(0, gidcs),
                _current_radii[sel] / max(_current_img_wh),
            )
            self.running_stats["max_radii"].index_put((gidcs,), max_radii)
        return True
    

    @torch.no_grad()
    def _densify_control_step(self, global_step):
        assert (self.running_stats["vis_count"] > 0).any()

        cfg = self.ctrl_cfg
        xys_grad_avg = self.running_stats["xys_grad_norm_acc"] / self.running_stats[
            "vis_count"
        ].clamp_min(1)
        is_grad_too_high = xys_grad_avg > cfg.densify_xys_grad_threshold
        # Split gaussians.
        scales = self.get_scales_all()
        is_scale_too_big = scales.amax(dim=-1) > cfg.densify_scale_threshold
        if global_step < cfg.stop_control_by_screen_steps:
            is_radius_too_big = (
                self.running_stats["max_radii"] > cfg.densify_screen_threshold
            )
        else:
            is_radius_too_big = torch.zeros_like(is_grad_too_high, dtype=torch.bool)

        all_should_split = is_grad_too_high & (is_scale_too_big | is_radius_too_big)
        all_should_dup = is_grad_too_high & ~is_scale_too_big

        count = 0
        new_running_stats = {k: list() for k in self.running_stats}
        for gsname in self.gaussian_names:
            num_gs = self.params[gsname].num_gaussians

            should_split = all_should_split[count:count+num_gs]
            num_splits = int(should_split.sum().item())
            should_dup = all_should_dup[count:count+num_gs]
            num_dups = int(should_dup.sum().item())
            param_map = self.params[gsname].densify_params(should_split, should_dup)

            for param_name, new_params in param_map.items():
                dup_in_optim(self.optimizers[f"{gsname}.params.{param_name}"], [new_params], should_split, num_splits * 2 + num_dups)
            
            for k, v in self.running_stats.items():
                v_gs = v[count:count+num_gs]
                new_running_stats[k].extend([
                    v_gs[~should_split],
                    v_gs[should_dup],
                    v_gs[should_split].repeat(2)])

            count += num_gs
        new_running_stats = {k: torch.cat(v) for k, v in new_running_stats.items()}
        self.running_stats = new_running_stats
        logger.info(
            f"Split {should_split.sum().item()} gaussians, "
            f"Duplicated {should_dup.sum().item()} gaussians, "
            f"{self.num_gaussians} gaussians left"
        )

    @torch.no_grad()
    def _cull_control_step(self, global_step):
        # Cull gaussians.
        cfg = self.ctrl_cfg
        opacities = self.get_opacities_all()
        device = opacities.device
        is_opacity_too_small = opacities < cfg.cull_opacity_threshold
        is_radius_too_big = torch.zeros_like(is_opacity_too_small, dtype=torch.bool)
        is_scale_too_big = torch.zeros_like(is_opacity_too_small, dtype=torch.bool)
        cull_scale_threshold = (
            torch.ones(len(is_scale_too_big), device=device) * cfg.cull_scale_threshold
        )

        num_fg = self.num_fg_gaussians
        if 'bg' in self.params:
            cull_scale_threshold[num_fg:] *= self.params['bg'].scene_scale

        if global_step > self.reset_opacity_every:
            scales = self.get_scales_all()
            is_scale_too_big = scales.amax(dim=-1) > cull_scale_threshold
            if global_step < cfg.stop_control_by_screen_steps:
                is_radius_too_big = (
                    self.running_stats["max_radii"] > cfg.cull_screen_threshold
                )
        should_cull = is_opacity_too_small | is_radius_too_big | is_scale_too_big

        count = 0
        for gsname in self.gaussian_names:
            num_gs = self.params[gsname].num_gaussians
            should_gs_cull = should_cull[count:count+num_gs]
            param_map = self.params[gsname].cull_params(should_gs_cull)
            for param_name, new_params in param_map.items():
                remove_from_optim(self.optimizers[f"{gsname}.params.{param_name}"], [new_params], should_gs_cull)
            count += num_gs

        # update running stats
        for k, v in self.running_stats.items():
            self.running_stats[k] = v[~should_cull]

        logger.info(
            f"Culled {should_cull.sum().item()} gaussians, "
            f"{self.num_gaussians} gaussians left"
        )

    @torch.no_grad()
    def _reset_opacity_control_step(self):
        new_val = torch.logit(torch.tensor(0.8 * self.ctrl_cfg.cull_opacity_threshold))
        for gsname in self.gaussian_names:
            params = self.params[gsname].params
            params["opacities"].data.fill_(new_val)
            reset_in_optim(self.optimizers[f"{gsname}.params.opacities"], [params["opacities"]])
        logger.info("Reset opacities")
        
        

def step_optimizers(loss: torch.Tensor, optimizers: dict[str, optim.Optimizer], schedulers: dict[str, lr_scheduler.LRScheduler]):
    for opt in optimizers.values():
        opt.zero_grad(set_to_none=True)
    loss.backward()
    for opt in optimizers.values():
        opt.step()
    for sched in schedulers.values():
        sched.step()


def create_optimizers_and_schedulers(gs_modules: dict[str, GaussianParams], 
        motion_modules: nn.ModuleDict | None=None, max_steps: int=1000, annealing_for: list[str]=[], annealing_factor:float = 0.1, extra_params=None, 
        use_sh_color: bool = False, **lr_kwargs) \
            -> tuple[dict[str, optim.Optimizer], dict[str,lr_scheduler.LRScheduler]]:
    optimizers, schedulers = {}, {}
    extra_params = extra_params or {}
    if use_sh_color:
        gs_keys = ['means', 'opacities', 'scales', 'quats', 'sh0', 'sh_rest', 'motion_coefs']
    else:
        gs_keys = ['means', 'opacities', 'scales', 'quats', 'colors', 'motion_coefs']
    
    if motion_modules is None:
        motion_keys = []
    else:
        motion_keys = ['motions']

    for k in gs_keys + motion_keys + list(extra_params.keys()):
        lr = lr_kwargs[k]
        
        fnc = partial(exponential_decay_func, max_steps=max_steps, final=annealing_factor * lr) if k in annealing_for else make_constant_func(1.0)
        if k == 'motions':
            # motions parameters include all instances
            optimizers[k] = optim.Adam(get_learnable_parameters(motion_modules), lr=lr)
            schedulers[k] = lr_scheduler.LambdaLR(optimizers[k], partial(fnc, init=lr))
            logger.info("creating optimizer/scheduler for motions (all instances)")
        elif k in gs_keys:
            for inst_name, mod in gs_modules.items():
                if k == 'motion_coefs' and k not in mod.params:
                    logger.warning(f"motion_coefs not found in {inst_name}, skipping")
                    continue

                full_name = f"{inst_name}.params.{k}"
                optimizers[full_name] = optim.Adam([mod.params[k]], lr=lr)
                schedulers[full_name] = lr_scheduler.LambdaLR(optimizers[full_name], partial(fnc, init=lr))
                logger.info(f"creating optimizer/scheduler for {full_name}")   
        else:
            optimizers[k] = optim.Adam(extra_params[k], lr=lr)
            schedulers[k] = lr_scheduler.LambdaLR(optimizers[k], partial(fnc, init=lr))
            logger.info(f"creating optimizer/scheduler for {k} (extra parameters)")
            
    return optimizers, schedulers
    
        
