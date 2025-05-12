import os
import shutil

if "CUDA_VISIBLE_DEVICES" in os.environ:
    dev = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]
else:
    dev = "0"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["EGL_DEVICE_ID"] = dev

import pyrender
import trimesh

from collections import defaultdict
from einops import rearrange, einsum
import yaml
import os.path as osp
import json
from datetime import datetime
import os
import fire
from tqdm.auto import trange, tqdm
import time
import viser
import imageio.v3 as iio
import numpy as np
import roma
import traceback
import torch
import torch.nn.functional as F
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from loguru import logger as guru

from flow3d.vis.utils import get_server
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple, Union
from jaxtyping import Float32, UInt8

from motionblender.app._patch.nerfview_viewer import CameraState, Viewer, RenderTask
import motionblender.lib.animate as anim

from viser import Icon, ViserServer, GuiEvent, ScenePointerEvent

from flow3d.vis.playback_panel import add_gui_playback_group
from flow3d.vis.render_panel import populate_render_tab
from pudb.remote import set_trace

from gsplat.rendering import rasterization
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import motionblender.app.utils as app_utils
import motionblender.lib.misc as misc
import motionblender.train as train
import motionblender.lib.pv as pv
from motionblender.lib.motion import PoseStore
from dataclasses import dataclass, field
import threading
import sys

Tensor = torch.Tensor
torch.set_grad_enabled(False)




@dataclass
class PoseStoreWithIndex(PoseStore):
    pivot_nodes: list[int] = field(default_factory=list)
    selected_nodes: list[int] = field(default_factory=list)

    abs_joints: Tensor | None = None

@dataclass
class AppBuffer:
    images: list[np.ndarray] = field(default_factory=list)
    splats: train.SplatsDict = field(default_factory=dict)
    tmp_pose: dict[str, PoseStoreWithIndex] = field(default_factory=dict)
    last_frame: np.ndarray | None = None


@dataclass
class AppState:
    animate_target_pose: str = ""
    vis_cam_frame_id: int = 0
    curr_pose_name: str | int = 0
    curr_img_wh: tuple[int, int] = (0, 0)
    curr_w2c: Tensor | None  = None
    curr_K: Tensor | None = None
    curr_cam_position: Tensor | None = None
    curr_cam_wxyz: Tensor | None = None

    cam_poses: dict[str, Tensor] = field(default_factory=dict)
    poses: dict[str, dict[str, PoseStoreWithIndex]] = field(default_factory=dict)
    # pose_name -> inst_name -> PoseStore

    xyz_step_size: float = 0.05
    rot_step_size: float = 0.05

    selected_instance: str = ""

    node_selecting_mode: str = "none"
    show_impainted_weights: bool = False

    link_radius: float = 0.02
    link_color: tuple[int, int, int] = '#0000FF'
    node_color: tuple[int, int, int] = '#FF0000'
    active_link_color: tuple[int, int, int] = '#00FFFF'
    active_node_color: tuple[int, int, int] = '#FFFF00'
    pivot_node_color: tuple[int, int, int] = '#00FF00'
    link_intensity: float = 10.
    link_opacity: float = 0.5
    scene_opacity: float = 1.0

    render_bg: bool = True
    render_links: bool = True


class MeshRenderBackend:
    def __init__(self):
        self.pv = pv.Plotter()
        self.pv.url
        self.pyrender = None # pyrender.OffscreenRenderer(viewport_width=img_wh[0], viewport_height=img_wh[1])
        self.scene = None
        self.cam = None
        self.light = None
        self.enabled = True

    def update_scene(self, joints, links, joints_color, links_color, K, intensity=30., link_radius=0.03, scene_obj_path='/dev/shm/scene.obj',
                     robot_gripper_pose: Float32[Tensor, "4 4"] | None = None):
        if robot_gripper_pose is not None:
            axis = app_utils.compute_axis_from_pose(robot_gripper_pose)
            len_joints = len(joints)
            joints = torch.cat([joints, axis['joints'].to(joints.device)], dim=0)
            links = torch.cat([links, axis['links'].to(joints.device) + len_joints], dim=0)
            joints_color = joints_color + axis['joints_color']
            links_color = links_color + axis['links_color']

        self.pv.update_params(graph={
            'joints': [joints],
            'links': links,
            'joints_color': joints_color,
            'links_color': links_color,
        })
        self.pv.gui_state['ui']['frame_id'] = 0
        self.pv.gui_state['ui']['node_radius'] = link_radius

        self.pv.render(render=False, pyrender_fix=True)
        scene_obj = scene_obj_path
        self.pv.plotter.export_obj(scene_obj)

        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        scene = pyrender.Scene.from_trimesh_scene(trimesh.load(scene_obj))
        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)

        eye = np.eye(4)
        self.cam = scene.add(camera, pose=eye)
        light = pyrender.SpotLight(color=np.ones(3), intensity=intensity,
                                innerConeAngle=np.pi/2.0, outerConeAngle=np.pi/2.0)
        self.light = scene.add(light, pose=eye)

        self.scene = scene

    def update_camera(self, w2c: Float32[Tensor, "4 4"]):
        assert self.scene is not None
        assert self.cam is not None
        assert self.light is not None
        cam_pose = app_utils.w2c_to_opengl_camera_pose(w2c).cpu().numpy()
        self.scene.set_pose(self.cam, cam_pose)
        self.scene.set_pose(self.light, cam_pose)

    def render(self):
        if not self.enabled:
            return None, None, None
        assert self.scene is not None
        color, depth = self.pyrender.render(self.scene)
        mask = depth >= 1e-6
        return color, depth, mask



def get_full_splat_at_t(gses, motions, gaussian_names, pose_stores: dict[str, PoseStore], t: int, include_background=True,
                        weight_impainting=False, weight_for_links: list[int] = []):
    full_splats = {}
    for g in gaussian_names:
        if g == 'bg':
            if not include_background or weight_impainting: continue
            means, quats = gses[g].means, gses[g].get_quats()
        else:
            motions[g].clear_motion_cache(t)
            means, quats = motions[g].transform_splats_to_t(gses[g].means, t=t, cano_quats_wxyz=gses[g].get_quats(), pose_store=pose_stores[g])
        full_splats.setdefault('means', []).append(means)
        full_splats.setdefault('quats', []).append(quats)
        full_splats.setdefault("scales", []).append(gses[g].get_scales())
        full_splats.setdefault("opacities", []).append(gses[g].get_opacities())
        if weight_impainting:
            m = motions[g]
            weight_for_links = torch.as_tensor(weight_for_links).to(m._skin_weights.device)
            if m._skin_weights_index is None:
                weights = m._skin_weights[:, weight_for_links].sum(dim=1)
            else:
                _skin_weights_mask = torch.isin(m._skin_weights_index, weight_for_links)
                weights = (m._skin_weights * _skin_weights_mask).sum(dim=1)

            color_indices = (weights / 0.001).round().clamp(0, 999)
            colors = app_utils.torch_viridis[color_indices.long()]
        else:
            colors = gses[g].get_colors()
        full_splats.setdefault("colors", []).append(colors)
    full_splats = {k:torch.cat(v) for k, v in full_splats.items()}
    return full_splats



class MotionBlenderApp(Viewer):
    def __init__(
        self, vise_server, train_view: train.MotionBlenderDataset, val_view: train.MotionBlenderDataset,
        app_state:AppState, gs_modules: dict[str, train.GaussianParams], motion_modules: dict[str, train.MotionBlender],
        gaussian_names: list[str], cfg: train.TrainConfig, device: str = 'cuda', lazy_render_interval: float=0.5,
        robot_module: app_utils.RobotInterface | None = None, vis_robot_pose: bool = False, restore_camera_pose: bool = False
    ):
        self.device = device
        self.server = vise_server
        self.train_view = train_view
        self.val_view = val_view
        self.gs_modules = gs_modules
        self.motion_modules = motion_modules
        self.gaussian_names = gaussian_names
        self.robot_module = robot_module
        self.cfg = cfg
        self.work_dir = Path(cfg.work_dir)
        self.updated = False
        self.vis_robot_pose = vis_robot_pose
        self.restore_camera_pose = restore_camera_pose
        self.gui_state = app_state
        self.buf = AppBuffer()

        self.gui_state.selected_instance = self.gaussian_names[0]
        self.gui_state.curr_pose_name = 0
        self.update_tmp_pose()

        if self.robot_module is not None:
            assert "robot" in self.gaussian_names, "robot instance is required for robot interface"

            self.robot_module.initialize(self.motion_modules['robot'])
            rot6d = self.robot_module.get_joint_rotations()
            self.buf.tmp_pose['robot'].rot6ds = rot6d

            if not hasattr(self.motion_modules['robot'], 'gripper_poses'):
                ee_pose = self.robot_module.get_ee_pose()
                self.motion_modules['robot'].gripper_poses = [ee_pose.clone() for _ in range(len(self.train_view))]

            if not hasattr(self.motion_modules['robot'], 'gripper_degrees'):
                self.motion_modules['robot'].gripper_degrees = [50.] * len(self.train_view)

            self.gui_state.selected_instance = "robot"

        self.mesh_renderer = MeshRenderBackend()
        self.bg_color = torch.as_tensor([1., 1., 1.]).reshape(1, -1).float().to(device)
        self.device = device
        self.lazy_render_interval = lazy_render_interval
        self.num_frames = len(self.train_view)
        self.gui_errors = []
        self.gui_messages = []
        super().__init__(self.server, self.render_fn, "rendering")
        self._inited = True

        self.lock = threading.Lock()
        self.last_render_time = 0
        self.animation_frames = 20

    def rerender(self, msg, allow_cache=False, **kwargs):
        curr_time = time.perf_counter()
        if curr_time - self.last_render_time < self.lazy_render_interval:
            self.gui_messages.append("Too fast to full render, wait a bit")
            return
        else:
            self.last_render_time = curr_time
        clients = self.server.get_clients()
        for client_id in clients:
            camera_state = self.get_camera_state(clients[client_id])
            assert camera_state is not None
            self._renderers[client_id].submit(RenderTask("rerender", camera_state, message=msg, allow_cache=allow_cache, **kwargs))

    def _connect_client(self, client: viser.ClientHandle):
        super()._connect_client(client)
        if getattr(self.gui_state, 'curr_cam_position', None) is not None and self.restore_camera_pose:
            guru.info(f"restore camera pose to {self.gui_state.curr_cam_position}")
            client.camera.position = self.gui_state.curr_cam_position
            client.camera.wxyz = self.gui_state.curr_cam_wxyz

    def _define_guis(self):
        server = self.server
        gui = server.gui
        state = self.gui_state
        dev = self.device
        toast = app_utils.client_message
        self.body = body = {}
        gui.configure_theme(control_width="large")

        def callback(path, name='on_update', render=True, cache=False, trace=False):
            def decorator(func):
                def new_func(evt):
                    with self.lock:
                        o = True
                        try:
                            if trace: set_trace()
                            o = func(evt)
                        except (Exception, AssertionError, RuntimeError) as e:
                            tb = traceback.format_exc()
                            msg = f"Error in {path}.{name}: \n {tb}"
                            self.gui_errors.append(msg)
                            toast(evt.client, str(e), title="Error", danger=True)

                        if render and o is not False: self.rerender(f"{path}.{name}", allow_cache=cache)
                        self.updated = True
                getattr(body[path], name)(new_func)
                return new_func
            return decorator

        body['general'] = server.gui.add_folder("General")
        with body['general']:
            body['general.res_slider'] = self._max_img_res_slider = server.gui.add_slider("Img Res", min=64, max=2048, step=1, initial_value=1536)
            body['general.res_slider'].on_update(self.rerender)

            body['general.t_slider'] = gui.add_slider("Time", min=0, max=self.num_frames-1, step=1, initial_value=0)
            def make_robot_time_hint(): return f"cur:{self.gui_state.curr_pose_name}, max:{self.num_frames-1}"
            @callback('general.t_slider')
            def _(evt: GuiEvent):
                state.curr_pose_name = evt.target.value
                self.update_tmp_pose()
                if self.robot_module is not None:
                    body['robot.edit_pose.target_t'].hint = make_robot_time_hint()
                update_preview_img(evt)
                toast(evt.client, f"Switched to pose {evt.target.value}", success=True)

            body['general.preview'] = server.gui.add_button("Preview & Sync (RGB)", color="green")
            def make_cam_pose_hint(): return f"0-{self.num_frames-1}," + (",".join(sorted(state.cam_poses.keys()) + ['$']))
            # body['general.preview_cam_id'] = gui.add_text("Preview Cam ID", initial_value="0", hint=make_cam_pose_hint())
            body['general.preview.img'] = gui.add_image(np.full([100, 100, 3], fill_value=0, dtype=np.uint8), "Preview Image")
            def update_preview_img(evt:GuiEvent):
                w2c, K, img_wh = get_render_cam(evt)
                if w2c is None: return
                img, _, _ = train.render(self.buf.splats, w2c, K, img_wh, self.bg_color, depth=False)
                img = (img.cpu().numpy() * 255.0).astype(np.uint8)
                body['general.preview.img'].image = img
            @callback('general.preview', name='on_click')
            def _(evt: GuiEvent):
                update_preview_img(evt)
                toast(evt.client, "Image rendered!", success=True)

        body['edit'] = server.gui.add_folder("Edit (basic)")
        with body['edit']:
            body['edit.mode'] = gui.add_dropdown("Node Selection Mode", ["none", "edit", "pivot"], initial_value=state.node_selecting_mode)
            @callback('edit.mode', render=False)
            def _(evt: GuiEvent):
                flag = evt.target.value
                state.node_selecting_mode = flag
                if flag != 'none':
                    @server.scene.on_pointer_event(event_type="click")
                    def __(click: ScenePointerEvent):
                        g = state.selected_instance
                        if not g:
                            toast(click.client, "Please select an instance first!", warning=True)
                        else:
                            joint_id = app_utils.find_closest_click_id(self.buf.tmp_pose[state.selected_instance].abs_joints, click, dev, cosine_thr=0.98)
                            nodes = self.buf.tmp_pose[g].selected_nodes if flag == 'edit' else self.buf.tmp_pose[g].pivot_nodes
                            if joint_id >= 0 and joint_id not in nodes:
                                nodes.append(joint_id)
                                joint_name = joint_id
                                m = self.motion_modules[state.selected_instance]
                                if getattr(m, 'joint_names', []):
                                    joint_name = m.joint_names[joint_id]
                                toast(click.client, f"Node {joint_name} selected!", success=True)
                                self.rerender("edit.mode.on_pointer_event")
                            else:
                                if joint_id >= 0:
                                    toast(click.client, "Node already selected!")
                                else:
                                    toast(click.client, "Not accurate enough to pin down a node!", warning=True)
                else:
                    server.scene.remove_pointer_callback()

            body['edit.reset_nodes'] = gui.add_button("Reset Selected Nodes", color="purple")
            @callback('edit.reset_nodes', name='on_click')
            def _(evt: GuiEvent):
                self.buf.tmp_pose[state.selected_instance].selected_nodes = []
                self.buf.tmp_pose[state.selected_instance].pivot_nodes = []


            def make_pose_hint(): return f"0-{self.num_frames-1}," + (",".join(sorted(state.poses.keys())))
            body['edit.pose_switch'] = gui.add_text("Switch to Pose", initial_value=str(state.curr_pose_name), hint=make_pose_hint())
            @callback('edit.pose_switch')
            def _(evt: GuiEvent):
                v = evt.target.value
                if v.isdigit():
                    v = int(v)
                    if 0 <= v < self.num_frames and v != state.curr_pose_name:
                        state.curr_pose_name = v
                        self.update_tmp_pose()
                        toast(evt.client, f"Switched to pose {v} (frame index)", success=True)
                        update_preview_img(evt)
                    else:
                        return False
                else:
                    if v in state.poses and v != state.curr_pose_name:
                        state.curr_pose_name = v
                        self.update_tmp_pose()
                        update_preview_img(evt)
                        toast(evt.client, f"Switched to pose {v}", success=True)
                    else:
                        return False

            body['edit.inst_select'] = gui.add_dropdown("Instance", list(self.buf.tmp_pose.keys()), initial_value=state.selected_instance)
            @callback('edit.inst_select')
            def _(evt: GuiEvent): state.selected_instance = evt.target.value


            body['edit.pose_name'] = gui.add_text("Pose Name", "", hint=make_pose_hint())
            body['edit.save_pose'] = server.gui.add_button("Save Curr Pose", color="yellow")
            @callback('edit.save_pose', name='on_click', render=False)
            def _(evt: GuiEvent):
                pose_name = body['edit.pose_name'].value
                if pose_name.isdigit():
                    toast(evt.client, "Pose name cannot be a number!", warning=True)
                if not pose_name:
                    toast(evt.client, "Pose name cannot be empty!", warning=True)
                else:
                    state.poses[pose_name] = deepcopy(self.buf.tmp_pose)
                    body['edit.pose_switch'].hint = make_pose_hint()
                    body['edit.save_pose'].hint = make_pose_hint()
                    toast(evt.client, f"Saved pose {pose_name}!", success=True)


        body['adjust'] = server.gui.add_folder("Edit (adjust)")
        with body['adjust']:
            body["adjust.trans_step_size"] = gui.add_text("Translation Step", initial_value=str(state.xyz_step_size))
            @callback('adjust.trans_step_size', render=False)
            def _(evt: GuiEvent):
                try: state.xyz_step_size = float(evt.target.value)
                except ValueError: pass

            body["adjust.rot_step_size"] = gui.add_text("Rotation Step", initial_value=str(state.rot_step_size))
            @callback('adjust.rot_step_size', render=False)
            def _(evt: GuiEvent):
                try: state.rot_step_size = float(evt.target.value)
                except ValueError: pass

            xxyyzz ="x+,x-,y+,y-,z+,z-".split(',')
            body['adjust.global.rotation'] = gui.add_button_group("🌎 Rotation", xxyyzz)
            body['adjust.global.xyz'] = gui.add_button_group("🌎 Translation", xxyyzz)
            body['adjust.joints.rotation'] = gui.add_button_group("🦴 Rotation", xxyyzz)
            body['adjust.joints.xyz'] = gui.add_button_group("🦴 Translation", xxyyzz)

            @callback('adjust.global.xyz', name='on_click')
            def _(evt: GuiEvent):
                sign, axis_ind = app_utils.evt_to_dir(evt)
                axis = state.curr_w2c[axis_ind, :3]
                step = sign * state.xyz_step_size * axis
                self.buf.tmp_pose[state.selected_instance].global_T[:3, 3] += step
                update_preview_img(evt)

            @callback('adjust.global.rotation', name='on_click')
            def _(evt: GuiEvent):
                sign, axis_ind = app_utils.evt_to_dir(evt)
                axis = state.curr_w2c[axis_ind, :3]
                rotvec = (axis / torch.linalg.norm(axis)) * sign * state.rot_step_size
                mat33 = roma.rotvec_to_rotmat(rotvec)
                global_R = self.buf.tmp_pose[state.selected_instance].global_T[:3, :3]
                self.buf.tmp_pose[state.selected_instance].global_T[:3, :3] = mat33 @ global_R
                update_preview_img(evt)

            @callback('adjust.joints.xyz', name='on_click')
            def _(evt: GuiEvent):
                g = state.selected_instance
                motion_type = self.motion_modules[g].type
                if motion_type in [train.MotionBlenderType.kinematic, train.MotionBlenderType.rigid]:
                    toast(evt.client, "kinematic/rigid instance does not support joint position adjustment")
                    return False

                tmp = self.buf.tmp_pose[g]

                selected_nodes = tmp.selected_nodes
                if len(selected_nodes) == 0:
                    toast(evt.client, "no joint selected", warning=True)
                    return False

                sign, axis_ind = app_utils.evt_to_dir(evt)
                axis = state.curr_w2c[axis_ind, :3]
                step = sign * state.xyz_step_size * axis # (3,)

                step = torch.linalg.inv(tmp.global_T[:3, :3]) @ step.reshape(3, 1)
                step = step.reshape(1, 3)
                tmp.joints[selected_nodes] += step
                update_preview_img(evt)

            @callback('adjust.joints.rotation', name='on_click')
            def _(evt: GuiEvent):
                # use the mean of pivot nodes as the rotation center
                # if pivot nodes are not selected, use the mean of all selected nodes
                g = state.selected_instance
                m = self.motion_modules[g]
                selected_nodes = self.buf.tmp_pose[g].selected_nodes
                pivot_nodes = self.buf.tmp_pose[g].pivot_nodes
                if len(selected_nodes) == 0:
                    toast(evt.client, "no joint selected", warning=True)
                    return False
                if m.is_rigid:
                    toast(evt.client, "rigid instance does not support joint rotation adjustment", warning=True)
                    return False

                if m.type == train.MotionBlenderType.kinematic:
                    for node_i in selected_nodes:
                        if node_i not in m.joint_i_2_link_i:
                            toast(evt.client, "the selected joint does not have a following link", warning=True)
                            return False

                sign, axis_ind = app_utils.evt_to_dir(evt)

                if m.type == train.MotionBlenderType.deformable:
                    axis = state.curr_w2c[axis_ind, :3]
                    rotvec = (axis / torch.linalg.norm(axis)) * sign * state.rot_step_size
                    mat33 = roma.rotvec_to_rotmat(rotvec)

                    global33 = self.buf.tmp_pose[g].global_T[:3, :3]
                    inv_global33 = torch.linalg.inv(global33)
                    mat33 = inv_global33 @ mat33 @ global33

                    if len(pivot_nodes) > 0:
                        center = self.buf.tmp_pose[g].joints[pivot_nodes].mean(0)
                    else:
                        center = self.buf.tmp_pose[g].joints[selected_nodes].mean(0)
                    self.buf.tmp_pose[g].joints[selected_nodes] -= center
                    self.buf.tmp_pose[g].joints[selected_nodes] = (mat33 @ self.buf.tmp_pose[g].joints[selected_nodes].T).T
                    self.buf.tmp_pose[g].joints[selected_nodes] += center
                else:
                    for node_i in selected_nodes:
                        link_i = m.joint_i_2_link_i[node_i]
                        axis = torch.eye(3, device=self.device)[axis_ind, :3]
                        rotvec = axis * sign * state.rot_step_size
                        mat33 = roma.rotvec_to_rotmat(rotvec)

                        curr_rmat = anim.cont_6d_to_rmat(self.buf.tmp_pose[g].rot6ds[link_i])
                        new_rmat = mat33 @ curr_rmat
                        self.buf.tmp_pose[g].rot6ds[link_i] = anim.rmat_to_cont_6d(new_rmat)
                update_preview_img(evt)


            body['adjust.instance.t_input'] = gui.add_text("T", initial_value="0", hint=make_robot_time_hint())
            body['adjust.instance.t_input.apply'] = gui.add_button("Set Instance Pose to [T]", color="red")
            @callback('adjust.instance.t_input.apply', name='on_click')
            def _(evt: GuiEvent):
                t = int(body['adjust.instance.t_input'].value)
                assert 0 <= t < self.num_frames
                g = state.selected_instance
                ps = PoseStoreWithIndex()
                m = self.motion_modules[g]
                m.compute_link_pose_at_t(t)
                ps.global_T = m._global_T_cache[t].clone()
                ps.abs_joints = m._joints_tensor_cache[t].clone()
                if m.type == train.MotionBlenderType.kinematic:
                    ps.rot6ds = m.rot6d[t].clone()
                elif m.type == train.MotionBlenderType.deformable:
                    ps.joints = m.joints[t].clone()
                self.buf.tmp_pose[g] = ps
                toast(evt.client, f"Set instance {g} pose to its pose at frame {t}!", success=True)


        body['tweaks'] = server.gui.add_folder("Tweaks")
        with body['tweaks']:
            body['tweaks.link_radius'] = gui.add_text("Link Radius", initial_value=str(state.link_radius))
            @callback('tweaks.link_radius')
            def _(evt: GuiEvent):
                try: state.link_radius = float(evt.target.value)
                except ValueError: return False

            color_varnames = "link_color,node_color,active_link_color,active_node_color,pivot_node_color"
            color_varnames_lst = color_varnames.split(',')
            body['tweaks.color_profile'] = gui.add_text("Color Profile #hex", initial_value=",".join([getattr(state, c) for c in color_varnames_lst]), hint=color_varnames)
            @callback('tweaks.color_profile')
            def _(evt: GuiEvent):
                try:
                    colors = [c.strip() for c in evt.target.value.split(',')]
                    assert len(colors) == len(color_varnames_lst)
                    for c in colors: assert len(c) > 1
                    for cname, cv in zip(color_varnames_lst, colors): setattr(state, cname, cv)
                except Exception:
                    self.gui_errors.append(traceback.format_exc())
                    return False

            body['tweaks.link_intensity'] = gui.add_text("Link Intensity", initial_value=str(state.link_intensity))
            @callback('tweaks.link_intensity')
            def _(evt: GuiEvent):
                try:
                    v = float(evt.target.value)
                    if v > 0: state.link_intensity = v
                    else: return False
                except ValueError: return False

            body['tweaks.link_opacity'] = gui.add_text("Link Opacity", initial_value=str(state.link_opacity))
            @callback('tweaks.link_opacity', cache=True)
            def _(evt: GuiEvent):
                try:
                    v = float(evt.target.value)
                    if 0 <= v <= 1: state.link_opacity = v
                    else: return False
                except ValueError: return False

            body['tweaks.scene_opacity'] = gui.add_text("Scene Opacity", initial_value=str(state.scene_opacity))
            @callback('tweaks.scene_opacity', cache=True)
            def _(evt: GuiEvent):
                try:
                    v = float(evt.target.value)
                    if 0 <= v <= 1: state.scene_opacity = v
                    else: return False
                except ValueError: return False

            body['tweaks.render_bg'] = gui.add_checkbox("Render Background", initial_value=state.render_bg)
            @callback('tweaks.render_bg')
            def _(evt: GuiEvent): state.render_bg = bool(evt.target.value)

            body['tweaks.render_links'] = gui.add_checkbox("Render Links", initial_value=state.render_links)
            @callback('tweaks.render_links')
            def _(evt: GuiEvent): state.render_links = bool(evt.target.value)

            body['tweak.show_impainted_weights'] = gui.add_checkbox("Show Impainted Weights", initial_value=state.show_impainted_weights)
            @callback('tweak.show_impainted_weights')
            def _(evt: GuiEvent):
                if evt.target.value and len(self.buf.tmp_pose[state.selected_instance].selected_nodes) == 0:
                    toast(evt.client, "No node selected, please select a node first!", warning=True)
                    return False
                else:
                    state.show_impainted_weights = bool(evt.target.value)

        body['export'] = server.gui.add_folder("Export")
        def get_render_cam(evt):
            cam_id = str(self.gui_state.curr_pose_name)
            if not cam_id.isdigit():
                cam_id = cam_id[1:]
                assert cam_id.isdigit()

            cam_id = int(cam_id)
            if cam_id < 0 or cam_id >= self.num_frames:
                toast(evt.client, f"Invalid camera frame id, shall be [0, {self.num_frames - 1}]", warning=True)
                return None, None, None
            else:
                batch = self.train_view[cam_id]
                w2c = batch["w2cs"].to(self.device)
                K = batch["Ks"].to(self.device)
                img_wh = batch['imgs'].shape[-2::-1]
                return w2c, K, img_wh

        with body['export']:
            body['export.animation_target'] = gui.add_text("Animation Target", initial_value="0", hint=make_pose_hint())
            body['export.save_path'] = gui.add_text("Save Path", initial_value=str(self.work_dir / "export"))
            body['export.save_animation'] = gui.add_button("Save Animation", color="red")
            @callback('export.save_animation', name='on_click', render=False)
            def _(evt: GuiEvent):
                w2c, K, img_wh = get_render_cam(evt)
                if w2c is None: return
                animate_target_pose = body['export.animation_target'].value
                roast_target_pose = lambda: toast(evt.client, f"target pose {animate_target_pose} not found! Available: {make_pose_hint()}", warning=True)
                if animate_target_pose not in state.poses:
                    if animate_target_pose.isdigit():
                        target_pose = int(animate_target_pose)
                        if 0 <= target_pose < self.num_frames:
                            pass
                        else:
                            roast_target_pose()
                            return
                    else:
                        roast_target_pose()
                        return
                else:
                    target_pose = animate_target_pose

                self.rerender(msg=f"Exporting animation to {body['export.save_path'].value} ...",
                              buffer=misc.cpickle.dumps({'command': "animation", "kwargs": dict(target_pose=target_pose, cam_info=[w2c, K, img_wh])}))
                toast(evt.client, "Rendering in progress, don't touch screen before finished, check terminal for progress.", auto_close=False, loading=True)

        if self.robot_module is not None:
            body['robot'] = server.gui.add_folder("Robot")
            with body['robot']:
                body['robot.gripper.rotation'] = gui.add_button_group("🤖 Rotation", xxyyzz)
                body['robot.gripper.xyz'] = gui.add_button_group("🤖 Translation", xxyyzz)
                @callback('robot.gripper.xyz', name='on_click')
                def _(evt: GuiEvent):
                    sign, axis_ind = app_utils.evt_to_dir(evt)
                    axis = torch.eye(3, device=self.device)[axis_ind, :3]
                    step = sign * state.xyz_step_size * axis
                    ee_pose = self.robot_module.get_ee_pose()
                    ee_pose[:3, 3] += step
                    self.robot_module.set_ee_pose(ee_pose)
                    self.buf.tmp_pose['robot'].rot6ds = self.robot_module.get_joint_rotations()

                @callback('robot.gripper.rotation', name='on_click')
                def _(evt: GuiEvent):
                    sign, axis_ind = app_utils.evt_to_dir(evt)
                    axis = torch.eye(3, device=self.device)[axis_ind, :3]
                    step = sign * state.rot_step_size * axis
                    ee_pose = self.robot_module.get_ee_pose()
                    ee_pose[:3, :3] = roma.rotvec_to_rotmat(step) @ ee_pose[:3, :3]
                    self.robot_module.set_ee_pose(ee_pose)
                    self.buf.tmp_pose['robot'].rot6ds = self.robot_module.get_joint_rotations()

                body['robot.gripper.degree'] = gui.add_text("🤖 Degree", initial_value="50")
                @callback('robot.gripper.degree')
                def _(evt: GuiEvent):
                    try:
                        degree = int(evt.target.value)
                        assert 1 <= degree <= 100
                        self.robot_module.set_gripper_degree(degree)
                        self.buf.tmp_pose['robot'].rot6ds = self.robot_module.get_joint_rotations()
                    except (Exception, AssertionError):
                        toast(evt.client, "Invalid degree, shall be an integer [1, 100]", warning=True)
                        return False

                body['robot.edit_pose'] = gui.add_folder("Edit Pose")
                with body['robot.edit_pose']:
                    body['robot.edit_pose.target_t'] = gui.add_text("Target Time", initial_value="0", hint=make_robot_time_hint())
                    MAX_GRIPPER_SPEED = 20
                    body['robot.edit_pose.gripper_speed'] = gui.add_slider("Gripper Speed", min=1, max=MAX_GRIPPER_SPEED-1, step=1, initial_value=10)
                    body['robot.edit_pose.method'] = gui.add_dropdown("Method", ["constant", "linear", "procrustes"], initial_value="constant")
                    body['robot.edit_pose.procruste_object'] = gui.add_dropdown("Procruste Object", list(set(self.gaussian_names) - {"robot", "bg"}))
                    body['robot.edit_pose.apply'] = gui.add_button("Apply Edits", color="red")


                    @callback('robot.edit_pose.apply', name='on_click', render=False, trace=False)
                    def _(evt: GuiEvent):
                        toast(evt.client, "Applying robot edits, please wait...", auto_close=False, loading=True, danger=True)
                        robot_module = deepcopy(self.robot_module)
                        start_t = int(body['robot.edit_pose.target_t'].value)
                        assert 0 <= start_t < self.num_frames
                        if not str(state.curr_pose_name).isdigit():
                            raise ValueError(f"please select a frame pose [0-{self.num_frames-1}] first")
                        end_t = int(state.curr_pose_name)
                        method = body['robot.edit_pose.method'].value
                        procrustes_object = body['robot.edit_pose.procruste_object'].value

                        start_pose = self.motion_modules['robot'].gripper_poses[start_t].to(self.device)
                        end_pose = self.robot_module.get_ee_pose().to(self.device)
                        T = abs(end_t - start_t) + 1

                        if method == "linear":
                            ee_poses = app_utils.interpolate_se3(start_pose, end_pose, T)
                        elif method == "procrustes":
                            step = 1 if end_t > start_t else -1
                            frame_ids = range(start_t, end_t+step, step)
                            N = 10000
                            self.motion_modules[procrustes_object].clear_motion_cache()
                            N = min(N, len(self.gs_modules[procrustes_object].means))
                            indices = torch.randperm(len(self.gs_modules[procrustes_object].means))[:N].to(self.device)
                            ee_poses = []
                            from_means = self.motion_modules[procrustes_object].transform_splats_to_t(self.gs_modules[procrustes_object].means, end_t)[indices]

                            for fi in tqdm(frame_ids, total=len(frame_ids), desc="procrustes"):
                                to_means = self.motion_modules[procrustes_object].transform_splats_to_t(self.gs_modules[procrustes_object].means, fi)[indices]
                                registration_mat = anim.solve_procrustes(from_means[None], to_means[None])
                                new_pose = anim.apply_mat4_pose(registration_mat, end_pose[None])[0]
                                ee_poses.append(new_pose)

                            self.motion_modules[procrustes_object].clear_motion_cache()
                        elif method == "constant":
                            ee_poses = [end_pose.clone() for _ in range(T)]
                        else:
                            raise ValueError(f"invalid method: {method}")

                        # interpolate gripper degree
                        start_degree = self.motion_modules['robot'].gripper_degrees[start_t]
                        end_degree = int(body['robot.gripper.degree'].value)
                        if method == "constant":
                            gripper_degrees = [end_degree] * T
                        else:
                            gripper_degrees = app_utils.interpolate_gripper_degree(start_degree, end_degree, T,
                                                                                window_size=MAX_GRIPPER_SPEED - body['robot.edit_pose.gripper_speed'].value)

                        # synchronize global pose
                        global_T = self.buf.tmp_pose['robot'].global_T
                        global_t = global_T[:3, 3]
                        global_rot6d = anim.rmat_to_cont_6d(global_T[:3, :3])
                        self.motion_modules['robot'].global_rot6d[:] = global_rot6d # no mobile robot
                        self.motion_modules['robot'].global_ts[:] = global_t

                        if start_t > end_t:
                            times = list(range(end_t, start_t+1))
                            start_t, end_t = end_t, start_t
                            gripper_degrees.reverse()
                            if isinstance(ee_poses, list):
                                ee_poses.reverse()
                            elif isinstance(ee_poses, Tensor):
                                ee_poses = ee_poses.flip(0)
                            else:
                                raise ValueError(f"invalid ee_poses type: {type(ee_poses)}")
                        else:
                            times = list(range(start_t, end_t+1))
                        assert start_t in times and end_t in times

                        # flushing out joint rotaitons and gripper values
                        assert len(times) == len(ee_poses) == len(gripper_degrees)
                        links_i = self.motion_modules['robot'].links.tolist()
                        links_i = [tuple(l) for l in links_i]
                        for t, ee_pose, ee_degree in tqdm(zip(times, ee_poses, gripper_degrees), total=len(times), desc="flushing out joint rotaitons and gripper values"):
                            robot_module.set_ee_pose(ee_pose)
                            robot_module.set_gripper_degree(ee_degree)
                            rot6d = robot_module.get_joint_rotations()

                            self.motion_modules['robot'].gripper_poses[t] = ee_pose
                            self.motion_modules['robot'].gripper_degrees[t] = ee_degree
                            self.motion_modules['robot'].rot6d[t] = rot6d

                        misc.dump_cpkl(self.work_dir / "robot_gs.cpkl", [self.gs_modules, self.motion_modules, None, self.gaussian_names])
                        toast(evt.client, "Robot edits done!", success=True)


    def animation_task(self, target_pose, cam_info):
        w2c, K, img_wh = cam_info
        num_frames = self.animation_frames
        self.mesh_renderer.pyrender.delete()
        self.mesh_renderer.pyrender = pyrender.OffscreenRenderer(viewport_width=img_wh[0], viewport_height=img_wh[1])

        start_pose, end_pose = deepcopy(self.buf.tmp_pose), self.get_pose_from_name(target_pose)
        all_poses: list[dict[str, PoseStoreWithIndex]] = [{g: PoseStoreWithIndex() for g in self.gaussian_names[:-1]} for _ in range(num_frames)]
        linspace = torch.linspace(0, 1.0, num_frames).to(self.device)
        selected_links = defaultdict(list)
        for g in self.gaussian_names[:-1]:
            m = self.motion_modules[g]
            global_ts = start_pose[g].global_T[:3, 3].reshape(1, 3) + (end_pose[g].global_T[:3, 3].reshape(1, 3) - start_pose[g].global_T[:3, 3].reshape(1, 3)) * linspace.reshape(-1, 1) # (num_frames, 3)
            start_global_uquat, end_global_uquat = roma.rotmat_to_unitquat(start_pose[g].global_T[:3, :3]), roma.rotmat_to_unitquat(end_pose[g].global_T[:3, :3])
            global_uquats = roma.unitquat_slerp(start_global_uquat, end_global_uquat, linspace)
            global_Rs = roma.unitquat_to_rotmat(global_uquats) # (num_frames, 3, 3)
            global_Ts = anim.rt_to_mat4(global_Rs, global_ts)
            if m.type == train.MotionBlenderType.deformable: # jonits
                joints = start_pose[g].joints.reshape(1, -1, 3) + (end_pose[g].joints.reshape(1, -1, 3) - start_pose[g].joints.reshape(1, -1, 3)) * linspace.reshape(-1, 1, 1)
            elif m.type == train.MotionBlenderType.kinematic: # rot6ds
                start_rot6d_uquat, end_rot6d_uquat = roma.rotmat_to_unitquat(anim.cont_6d_to_rmat(start_pose[g].rot6ds)), roma.rotmat_to_unitquat(anim.cont_6d_to_rmat(end_pose[g].rot6ds))
                rot6ds = anim.rmat_to_cont_6d(roma.unitquat_to_rotmat(roma.unitquat_slerp(start_rot6d_uquat, end_rot6d_uquat, linspace)))
            else:
                raise NotImplementedError

            for fi in range(num_frames):
                all_poses[fi][g].global_T = global_Ts[fi]
                if m.type == train.MotionBlenderType.deformable:
                    all_poses[fi][g].joints = joints[fi]
                elif m.type == train.MotionBlenderType.kinematic:
                    all_poses[fi][g].rot6ds = rot6ds[fi]

            selected_nodes = self.buf.tmp_pose[g].selected_nodes
            for li, (la, lb) in enumerate(m.links.tolist()):
                if la in selected_nodes or lb in selected_nodes:
                    selected_links[g].append(li)

        assert len(all_poses) == num_frames
        rendered = defaultdict(list)
        rendered_states = []
        write_to = self.body['export.save_path'].value
        write_to = Path(write_to) / (f"{self.gui_state.curr_pose_name}-{target_pose}.{int(time.time())}")
        write_mesh_to = write_to / 'meshes'
        write_mesh_to.mkdir(exist_ok=True, parents=True)
        for fi in trange(num_frames):
            pose = all_poses[fi]
            splats = get_full_splat_at_t(self.gs_modules, self.motion_modules, self.gaussian_names, pose, t=-1,
                                            weight_impainting=self.gui_state.show_impainted_weights,
                                            weight_for_links=selected_links[g])
            img, _, _ = train.render(splats, w2c, K, img_wh, self.bg_color, depth=False)
            img = (img[0] * 255).to(torch.uint8).cpu().numpy()
            rendered['img'].append(img)
            graphs = {}
            for g in self.gaussian_names[:-1]:
                pose[g].abs_joints = self.motion_modules[g]._joints_tensor_cache[-1]
                joints = pose[g].abs_joints
                links = self.motion_modules[g].links
                scene_obj = '/dev/shm/scene.obj'
                self.mesh_renderer.update_scene(
                    joints=joints.cpu(),
                    links=links.cpu(),
                    joints_color=[self.gui_state.node_color] * len(joints),
                    links_color=[self.gui_state.link_color] * len(links),
                    K=K.cpu(),
                    intensity=self.gui_state.link_intensity,
                    link_radius=self.gui_state.link_radius,
                    scene_obj_path=scene_obj
                )
                graphs[g] = dict(
                    joints=joints.cpu(),
                    links=links.cpu(),
                    joints_color=[self.gui_state.node_color] * len(joints),
                    links_color=[self.gui_state.link_color] * len(links),
                )
                self.mesh_renderer.update_camera(w2c.cpu())
                link_rgb, _, link_mask = self.mesh_renderer.render()
                shutil.copy(scene_obj, write_mesh_to / f"{g}_{fi}.obj")
                shutil.copy(scene_obj.replace('.obj', '.mtl'), write_mesh_to / f"{g}_{fi}.mtl")
                rendered[f"{g}_link_rgb"].append(link_rgb)
                rendered[f"{g}_link_mask"].append(link_mask.astype(np.uint8) * 255)

            rendered_states.append({
                'w2c': w2c,
                'K': K,
                'pose': pose,
                'weight_for_links': selected_links,
                'graphs': graphs
            })

        guru.info(f"Rendering done! clean up and writing to {write_to}...")
        self.mesh_renderer.pyrender.delete()
        self.mesh_renderer.pyrender = pyrender.OffscreenRenderer(viewport_width=self.gui_state.curr_img_wh[0],
                                                                    viewport_height=self.gui_state.curr_img_wh[1])
        rendered = {k:np.stack(v) for k, v in rendered.items()}
        for k, v in rendered.items():
            iio.imwrite(write_to / f"{k}.mp4", v, plugin='FFMPEG', fps=30)
            guru.info(f"writing to {write_to / f'{k}.mp4'}")
        misc.dump_cpkl(write_to / "states.cpkl", rendered_states)
        misc.dump_cpkl(write_to / "app_state.cpkl", self.gui_state)
        misc.dump_json(write_to / "info.json", {
            "target_pose_name": self.body['export.animation_target'].value,
            "curr_pose_name": self.gui_state.curr_pose_name,
        })
        guru.info("animation done!")

    def get_pose_from_name(self, pose_name: str | int, instance_name: str | None = None) -> PoseStoreWithIndex:
        if isinstance(pose_name, str) and pose_name.isdigit():
            pose_name = int(pose_name)
        if instance_name is None:
            gs = self.gaussian_names[:-1]
        else:
            gs = [instance_name]
        if isinstance(pose_name, int):
            t = int(pose_name)
            result = {}
            for g in gs:
                ps = PoseStoreWithIndex()
                m = self.motion_modules[g]
                m.compute_link_pose_at_t(m.cano_t, skin_pts=self.gs_modules[g].means)
                m.compute_link_pose_at_t(t)
                ps.global_T = m._global_T_cache[t].clone()
                ps.abs_joints = m._joints_tensor_cache[t].clone()
                if m.type == train.MotionBlenderType.kinematic:
                    ps.rot6ds = m.rot6d[t].clone()
                elif m.type == train.MotionBlenderType.deformable:
                    ps.joints = m.joints[t].clone()
                result[g] = ps
            return result
        else:
            assert pose_name in self.gui_state.poses
            return deepcopy(self.gui_state.poses[pose_name])

    def update_tmp_pose(self):
        self.buf.tmp_pose = self.get_pose_from_name(self.gui_state.curr_pose_name)
        if self.robot_module is not None and self.robot_module.inited and str(self.gui_state.curr_pose_name).isdigit():
            self.robot_module.set_ee_pose(self.motion_modules['robot'].gripper_poses[int(self.gui_state.curr_pose_name)])

    def handle_gui_logs(self):
        if len(self.gui_errors) > 0:
            es = list(self.gui_errors)
            self.gui_errors = []
            for e in es: guru.error(e)
        if len(self.gui_messages) > 0:
            ms = list(self.gui_messages)
            self.gui_messages = []
            for m in ms: guru.info(m)

    @torch.inference_mode()
    def render_fn(self, camera_state: CameraState, img_wh: tuple[int, int], cache_refresh=False, buffer=b""):
        try:
            if not getattr(self, '_inited', False):
                return np.full((img_wh[1], img_wh[0], 3), 255, dtype=np.uint8)

            if len(buffer) > 0:
                _ = misc.cpickle.loads(buffer)
                if _['command'] == 'animation':
                    guru.info("Animation command received, starting... (pause rendering)")
                    self.animation_task(**_['kwargs'])
                else:
                    guru.info("Unknown command received, ignoring...")

            img_size_changed =  self.gui_state.curr_img_wh != tuple(img_wh)
            heavy_load = cache_refresh or len(self.buf.splats) == 0 or img_size_changed
            W, H = img_wh

            focal = 0.5 * H / np.tan(0.5 * camera_state.fov).item()
            K = torch.tensor(
                [[focal, 0.0, W / 2.0], [0.0, focal, H / 2.0], [0.0, 0.0, 1.0]],
                device=self.device,
            )
            w2c = torch.linalg.inv(
                torch.from_numpy(camera_state.c2w.astype(np.float32)).to(self.device)
            )

            old_img_wh = self.gui_state.curr_img_wh
            self.gui_state.curr_img_wh = tuple(img_wh)
            self.gui_state.curr_w2c = w2c
            self.gui_state.curr_K = K
            self.gui_state.curr_cam_position = camera_state.position
            self.gui_state.curr_cam_wxyz = camera_state.wxyz
            links_need_to_show = self.gui_state.render_links and not self.gui_state.show_impainted_weights
            if heavy_load:
                guru.info(f"cache_refresh {cache_refresh}, img_size_changed {img_size_changed} {old_img_wh} -> {img_wh}")
                joints_color, links_color = [], []
                g = self.gui_state.selected_instance
                links  = self.motion_modules[g].links
                pivot_nodes = self.buf.tmp_pose[g].pivot_nodes
                selected_nodes = self.buf.tmp_pose[g].selected_nodes
                selected_links = []

                for li, (la, lb) in enumerate(links.tolist()):
                    if la in selected_nodes or lb in selected_nodes:
                        links_color.append(self.gui_state.active_link_color)
                        selected_links.append(li)
                    else:
                        links_color.append(self.gui_state.link_color)

                splats = get_full_splat_at_t(self.gs_modules, self.motion_modules, self.gaussian_names, self.buf.tmp_pose, t=-1,
                                            include_background=self.gui_state.render_bg, weight_impainting=self.gui_state.show_impainted_weights,
                                            weight_for_links=selected_links)
                self.buf.splats = splats
                joints = self.motion_modules[g]._joints_tensor_cache[-1]
                self.buf.tmp_pose[g].abs_joints = joints.clone()
                for j in range(len(joints)):
                    if j in pivot_nodes:
                        joints_color.append(self.gui_state.pivot_node_color)
                    elif j in selected_nodes:
                        joints_color.append(self.gui_state.active_node_color)
                    else:
                        joints_color.append(self.gui_state.node_color)

                if links_need_to_show:
                    if self.mesh_renderer.pyrender is None or img_size_changed:
                        if self.mesh_renderer.pyrender is not None:
                            self.mesh_renderer.pyrender.delete()
                        self.mesh_renderer.pyrender = pyrender.OffscreenRenderer(viewport_width=img_wh[0], viewport_height=img_wh[1])
                    self.mesh_renderer.update_scene(
                        joints=joints.cpu(),
                        links=links.cpu(),
                        joints_color=joints_color,
                        links_color=links_color,
                        K=K.cpu(),
                        intensity=self.gui_state.link_intensity,
                        link_radius=self.gui_state.link_radius,
                        robot_gripper_pose=None if (self.gui_state.selected_instance != "robot" or not self.vis_robot_pose) else (self.buf.tmp_pose['robot'].global_T @ self.robot_module.get_ee_pose())
                    )


            if links_need_to_show:
                self.mesh_renderer.update_camera(w2c.cpu())
                link_rgb, link_depth, link_mask = self.mesh_renderer.render()
                link_rgb = torch.from_numpy(link_rgb.copy()).float().to(self.device) / 255.0
                link_mask = torch.from_numpy(link_mask.copy()).float().to(self.device).reshape(*link_rgb.shape[:2], 1)

            splats = {k:v for k, v in self.buf.splats.items()}
            splats['opacities'] = self.gui_state.scene_opacity * splats['opacities']
            img, _, _ = train.render(splats, w2c, K, img_wh, self.bg_color, depth=False)
            img = img[0]

            if links_need_to_show:
                img = img * (1 - link_mask) + link_mask * (link_rgb * self.gui_state.link_opacity + img * (1 - self.gui_state.link_opacity))
            img = (img.cpu().numpy() * 255.0).astype(np.uint8)
            self.updated = True
            self.handle_gui_logs()
            self.buf.last_frame = img
            return img
        except (Exception, AssertionError, RuntimeError):
            tb = str(traceback.format_exc())
            guru.error(tb)
            return app_utils.text_to_image(tb, image_size=img_wh)


def main(ckpt_path='outputs/mb/robot/okish/toy/ckpt.robot.cpkl', port=6060, refresh=False, save_state=True, lazy_render_interval=0.5,
         robot_module: str | None = None, vis_robot_pose: bool = False, work_dir_from_ckpt: bool = False, restore_camera_pose: bool = False, data_dir=""):
    if robot_module: robot_module: app_utils.RobotInterface = misc.import_py_file(robot_module).robot
    # for initialization einops
    einsum( torch.randn(10, 3), torch.randn(3), 'n d, d -> n')
    vise_server = get_server(port=port)
    with torch.no_grad():
        gs_modules, motion_modules, _, gaussian_names = misc.load_cpkl(ckpt_path)
        misc.set_modules_grad_enable(gs_modules, False)
        misc.set_modules_grad_enable(motion_modules, False)

    for v in motion_modules.values():
        if v.type == train.MotionBlenderType.kinematic:
            v.joint_i_2_link_i = {} # this is used to check whether a joint has a link following
            for link_i, (joint_a, joint_b) in enumerate(v.links):
                v.joint_i_2_link_i[int(joint_b)] = link_i
    guru.info(f"instances: {gaussian_names[:-1]}")

    cfg_path = osp.dirname(ckpt_path) + '/cfg.yaml'
    if not osp.exists(cfg_path):
        cfg_path = osp.dirname(ckpt_path) + '/../cfg.yaml'
        if not osp.exists(cfg_path):
            raise ValueError(f"cfg.yaml cannot be resolved from {ckpt_path}")

    cfg_dict = yaml.load(open(cfg_path), Loader=yaml.FullLoader)
    if 'use_tracks' in cfg_dict['data']:
        data_class = train.MotionBlenderGeneralDataConfig
    else:
        data_class = train.MotionBlenderIPhoneDataConfig
    cfg = misc.dict_to_config(train.TrainConfig, cfg_dict, classes=dict(data=data_class))
    if work_dir_from_ckpt:
        cfg.work_dir = osp.dirname(ckpt_path)
    cfg.work_dir = osp.join(cfg.work_dir, 'app')
    os.makedirs(cfg.work_dir, exist_ok=True)
    misc.add_log_txt_guru(cfg.work_dir, "log.txt", clear=True)

    if data_dir:
        cfg.data.data_dir = data_dir
    else:
        cfg.data.data_dir = f'./datasets/iphone/{osp.basename(osp.dirname(ckpt_path))}'
    train_dataset, train_video_view, val_img_dataset = train.get_train_val_datasets(cfg.data, load_val=False)
    device = "cuda"
    app_state_path = osp.join(cfg.work_dir, 'app_state.pkl')

    if osp.exists(app_state_path) and not refresh:
        app_state = misc.load_cpkl(app_state_path)
        guru.info(f'loading app state from {app_state_path}')
    else:
        app_state = AppState()

    app = MotionBlenderApp(vise_server, train_view=train_video_view, val_view=val_img_dataset, app_state=app_state,
                            gs_modules=gs_modules, motion_modules=motion_modules, gaussian_names=gaussian_names, cfg=cfg, device=device,
                            lazy_render_interval=lazy_render_interval, robot_module=robot_module, vis_robot_pose=vis_robot_pose, restore_camera_pose=restore_camera_pose)

    guru.info(f"App is running (saved poses: {list(app.gui_state.poses.keys())})")
    while True:
        time.sleep(2)
        if app.updated and save_state:
            misc.dump_cpkl(osp.join(cfg.work_dir, 'app_state.pkl'), app.gui_state)
            guru.info('app state saved, with all saved poses: {}'.format(list(app.gui_state.poses.keys())))
            app.updated = False



if __name__ == "__main__":
    fire.Fire(main)
