import numpy as np
import os
import yaml
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import shutil
import os.path as osp
from datetime import datetime   
import json
from loguru import logger as guru
import torch
from gsplat.rendering import rasterization
from collections import defaultdict
import cloudpickle as cpickle
from dataclasses import dataclass, is_dataclass, fields, field
import sys
import importlib.util


def import_py_file(file_path, module_name=None):
    if module_name is None:
        module_name = osp.basename(file_path).replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def make_guru_once_filter():
    _msg_history = defaultdict(set)
    def filter(record):
        if "once" in record["extra"]:
            level = record["level"].no
            message = record["message"]
            if message in _msg_history[level]:
                return False
            _msg_history[level].add(message)
        return True
    return filter


def add_log_txt_guru(work_dir, log_name="train_log.txt", clear=False):
    train_log_txt = osp.join(work_dir, log_name)
    if osp.exists(train_log_txt):
        if not clear:
            with open(train_log_txt, "r") as f_in:
                from time import time
                backup_log_txt = train_log_txt + f".{int(time())}"
                print('Backup log to', backup_log_txt)
                with open(backup_log_txt, "w") as f_out:
                    f_out.write(f_in.read())
        with open(train_log_txt, "w") as f_out:f_out.write("")
    guru.add(train_log_txt, filter=make_guru_once_filter())


def dict_to_config(cls, data, classes=None, parent=""):
    if not is_dataclass(cls):
        raise ValueError(f"{cls} is not a dataclass")
    kwargs = {}
    for field in fields(cls):
        scope_name = ("" if not parent else (parent + '.')) + field.name
        if field.name in data:
            value = data[field.name]
            if is_dataclass(field.type):
                # Recursively convert nested dictionaries
                kwargs[field.name] = dict_to_config(field.type, value, classes, parent=scope_name)
            elif scope_name in classes:
                kwargs[field.name] = dict_to_config(classes[scope_name], value, classes, parent=scope_name)
            else:
                kwargs[field.name] = value
    return cls(**kwargs)


def dcdefault(v):
    return field(default_factory=lambda: v)


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def dump_json(path, obj, **kwargs):
    if isinstance(obj, str):
        path, obj = obj, path
    with open(path, 'w') as f:
        json.dump(obj, f, **kwargs)
    
def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)
    
def dump_cpkl(path, obj):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        cpickle.dump(obj, f)

def load_cpkl(path):
    with open(path, 'rb') as f:
        return cpickle.load(f)
    
def remap_values(x, from_values, to_values=None):
    """
    x: tensor of any shape
    from_values: (N,) dict keys
    to_values: (N,) dict values
    """
    if to_values is None: to_values = torch.arange(len(from_values), device=x.device)
    index = torch.bucketize(x.ravel(), from_values)
    return to_values[index].reshape(x.shape)


def reduce_loss_dict(loss_dict, itemize=False):
    def mean_lst(x):
        return (sum(x) / len(x)) if len(x) > 0 else 0.0
    def item(x):
        if itemize: return x.item() if hasattr(x, 'item') else x 
        else: return x
    return {k: item(mean_lst(v) if isinstance(v, list) else v)  for k, v in loss_dict.items()}


def itemize(x):
    return x.item() if hasattr(x, 'item') else x


def loss_dict_to_str(loss_dict):
    return ", ".join([f"{k}: {itemize(v):.4f}" for k, v in loss_dict.items()])


def add_prefix_to_dict(dct, prefix):
    prefix += '.' if prefix else ''
    return {f"{prefix}{k}": v for k, v in dct.items()}

def linear_smooth_func(i, min_v, max_v, th, num_iters): 
    """ 
    i: step
    min_v: min value 
    max_v: max value
    th: warmup steps
    num_iters: total number of iterations
    """
    if i <= th:
        return min_v  
    if i > num_iters:
        return max_v
    else:
        return (max_v - min_v) * (i - th) / (num_iters - th) + min_v

def exponential_decay_func(step, max_steps, init, final):
    t = np.clip(step / max_steps, 0.0, 1.0)
    v = np.exp(np.log(init) * (1 - t) + np.log(final) * t)
    return v / init # turn it into a multiplier

def make_constant_func(v):
    return lambda *_, **__: v


def render(splats: dict, w2c, K, img_wh, bg_color=None, depth=True, engine='gsplat'):
    if bg_color is None: bg_color = torch.ones(3).to(w2c.device)   
    if w2c.dim() == 3: assert w2c.shape[0] == 1

    if engine == 'gsplat':    
        if bg_color.dim() == 1: bg_color = bg_color.unsqueeze(0)
        if w2c.dim() == 2: w2c = w2c.unsqueeze(0)
        if K.dim() == 2: K = K.unsqueeze(0)
        W, H = img_wh
        render_colors, alphas, info = rasterization(
            **splats,
            backgrounds=bg_color,
            viewmats=w2c, 
            Ks=K, 
            width=W,
            height=H,
            packed=False,
            render_mode="RGB+ED" if depth else "RGB"
        )
        return render_colors, alphas, info
    else:
        import motionblender.lib.gaussian_diff as gdiff
        assert splats['colors'].shape[-1] <= 16
        rasterize_settings = gdiff.get_rasterization_settings(w2c, K, 4, bg_tensor=bg_color, img_wh=img_wh)
        rasterizer = gdiff.GaussianRasterizer(rasterize_settings)
        pkg = gdiff.render_with_diffg(rasterizer, splats)
        info = {
            'means2d': pkg['viewspace_points'],
            'radii': pkg['radii'][None],
        }
        if depth:
            pkg['features'] = torch.cat([pkg['features'], pkg['depth'][None]])
        return pkg['features'].permute(1, 2, 0)[None], None, info


def set_modules_grad_enable(modules, val):
    if 'dict' in modules.__class__.__name__.lower(): modules = modules.values()
    for m in modules:
        for p in m.parameters():
            if p.dtype in [torch.float32, torch.double]:
                p.requires_grad = val


def get_learnable_parameters(module:nn.Module):
    return [p for p in module.parameters() if p.requires_grad]
                
                
def compute_link_length(kps, connections):
    """ (j, c), (l, 2) -> (l) """ 
    return torch.norm(kps[connections[:, 0]] - kps[connections[:, 1]], dim=1).flatten()



def backup_code(work_dir):
    root_dir = osp.abspath(osp.join(osp.dirname(__file__)))
    tracked_dirs = [osp.join(root_dir, dirname) for dirname in ["motionblender", ]]
    dst_dir = osp.join(work_dir, "code", datetime.now().strftime("%Y-%m-%d-%H%M%S"))
    for tracked_dir in tracked_dirs:
        if osp.exists(tracked_dir):
            shutil.copytree(tracked_dir, osp.join(dst_dir, osp.basename(tracked_dir)))


def quat_activation(x): return F.normalize(x, dim=-1, p=2)



def dict_items(dct, exclude=[]):
    for k, v in dct.items():
        if k in exclude: continue
        else: yield k, v


def loopy(dl):
    while True:
        for x in dl: yield x


def viridis(tensor, cmap='viridis'):
    tensor = tensor.detach().cpu().squeeze()
    assert tensor.dim() == 2
    return to_pil_image(plt.get_cmap(cmap)(tensor)[:, :, :3])



def to_fg_names(gaussian_names):
    return [name for name in gaussian_names if name != 'bg']


def save_debug_img(tensor, path="outputs/test.png"):
    if tensor.dim() == 4: tensor = tensor[0]
    if tensor.shape[-1] == 3:
        tensor = tensor.permute(2, 0, 1)
    if tensor.max() > 10:
        tensor = tensor / 255.0
    to_pil_image(tensor).save(path)
    


def remap_values(x, from_values, to_values=None):
    """
    x: tensor of any shape
    from_values: (N,) dict keys
    to_values: (N,) dict values
    """
    if to_values is None: to_values = torch.arange(len(from_values), device=x.device)
    index = torch.bucketize(x.ravel(), from_values)
    return to_values[index].reshape(x.shape)
