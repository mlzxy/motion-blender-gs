import asyncio
from copy import deepcopy
from loguru import logger
import pickle 
import pyvista as pv
import torch
from jaxtyping import Float32, Int64
import numpy as np
import traceback
from torch import Tensor
import os.path as osp
import motionblender.lib.mesh2d as mesh_lib
import os

from pyvista.trame.ui.vuetify3 import button, divider, select, slider, text_field, checkbox
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

pv.set_jupyter_backend('trame')
pv.set_plot_theme('paraview')

COLORS = '#4169E1,#40E0D0,#6B8E23,#E6E6FA,#FFD700,#FFDAB9,#8A2BE2,#00CED1,#00FA9A,#6495ED,#ADD8E6,#BDB76B,#00BFFF,#EEE8AA,#6B8E23,#8B4513,#48D1CC,#FFD700,#D8BFD8,#7FFFD4,#F0E68C,#6A5ACD,#D2B48C,#FFEBCD,#00FF00,#FF69B4,#F5DEB3,#32CD32,#9932CC,#483D8B,#228B22,#FFDAB9,#40E0D0,#20B2AA,#FFEFD5,#FFDEAD,#F5FFFA,#3CB371,#FFFACD,#556B2F,#F4A460,#B0C4DE,#B8860B,#87CEFA,#BA55D3,#FF4500,#006400,#BA55D3,#FAEBD7,#CD853F,#87CEEB,#FFF0F5,#FAFAD2,#00CED1,#DDA0DD,#8A2BE2,#FFE4B5,#8FBC8F,#66CDAA,#FDF5E6,#FFB6C1,#4682B4,#AFEEEE,#E0FFFF,#8B008B,#FFEFD5,#FF1493,#F5F5DC,#FF7F50,#9400D3,#008000,#FFE4B5,#008B8B,#5F9EA0,#FFE4C4,#7CFC00,#DAA520,#DA70D6,#6A5ACD,#7B68EE,#7B68EE,#FFFFE0,#FF6347,#00FFFF,#EEE8AA,#483D8B,#FFF5EE,#FFC0CB,#DB7093,#DEB887,#FFFFFF,#FFFF00,#F5F5F5,#4B0082,#DA70D6,#FFE4E1,#800080,#FAFAD2,#BC8F8F,#48D1CC,#F0E68C,#BDB76B,#EE82EE,#9ACD32,#90EE90,#808000,#A0522D,#556B2F,#008080,#C71585,#008080,#98FB98,#9370DB,#D2691E,#2E8B57,#1E90FF,#20B2AA,#FFA500,#00FFFF,#FF8C00,#9370DB'.split(',')


DEFAULT_COLORS = (
    'red',
    'blue',
    'yellow',
    'magenta',
    'green',
    'indigo',
    'darkorange',
    'cyan',
    'pink',
    'yellowgreen',
)


def vis_link(plotter, joints, connections, prefix="", radius=0.1, render=True, 
             joints_color=None, links_color=None):
    actors = []
    for i, joint in enumerate(joints):
        if joints_color is not None:
            color = joints_color[i]
        else:
            color = COLORS[i % len(COLORS)]
        a = plotter.add_mesh(pv.Sphere(center=joint, radius=radius), color=color, name=f'{prefix}joint-{i}', render=render)
        actors.append(a)

    for i, connection in enumerate(connections):
        joint1 = joints[connection[0]]
        joint2 = joints[connection[1]]
        if links_color is not None:
            if isinstance(links_color, list):
                color = links_color[i]
            else:
                color = links_color
        else:
            color = 'red'
        try:
            a = plotter.add_mesh(pv.Tube(joint1, joint2, radius=radius/2), color=color, name=f'{prefix}tube-{i}', render=render)
            actors.append(a)
        except Exception as e:
            logger.warning(f'Failed to add tube {i} from {joint1} to {joint2}: {e}')
            continue
    return actors

    
def to_np(x):
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return x


def to255(rgbs):
    if rgbs.max() <= 1:
        return (rgbs * 255).astype(np.uint8)
    else:
        return rgbs


def mask2color(mask: np.ndarray):
    mask = to_np(mask)
    mask = mask.astype(int) 
    mask = mask - mask.min()
    num_classes = mask.max() + 1
    if num_classes > len(DEFAULT_COLORS):
        colors = COLORS
    else:
        colors = DEFAULT_COLORS
    colors = np.array([mcolors.to_rgb(c) for c in colors])
    oshape = mask.shape
    colored_mask = to255(colors[mask.flatten() % len(colors)])
    colored_mask = colored_mask.reshape(oshape + (3,))
    return colored_mask


def camera_to_world_matrix(position, azimuth, roll, elevation):
    """
    Construct a 4x4 camera-to-world matrix from camera parameters.

    Args:
        position (tuple): (x, y, z) position of the camera in world coordinates.
        azimuth (float): Rotation around the Y-axis (yaw) in degrees.
        roll (float): Rotation around the Z-axis (roll) in degrees.
        elevation (float): Rotation around the X-axis (pitch) in degrees.

    Returns:
        np.ndarray: 4x4 camera-to-world transformation matrix.
    """
    from math import radians, sin, cos
    # Convert angles from degrees to radians
    yaw = radians(azimuth)
    pitch = radians(elevation)
    roll_rad = radians(roll)

    # Compute rotation matrices for each axis
    # Rotation around Y-axis (azimuth/yaw)
    R_y = np.array([
        [cos(yaw), 0, sin(yaw)],
        [0, 1, 0],
        [-sin(yaw), 0, cos(yaw)]
    ])

    # Rotation around X-axis (elevation/pitch)
    R_x = np.array([
        [1, 0, 0],
        [0, cos(pitch), -sin(pitch)],
        [0, sin(pitch), cos(pitch)]
    ])

    # Rotation around Z-axis (roll)
    R_z = np.array([
        [cos(roll_rad), -sin(roll_rad), 0],
        [sin(roll_rad), cos(roll_rad), 0],
        [0, 0, 1]
    ])

    # Combined rotation: R = R_z * R_x * R_y (applied in reverse order)
    R = R_z @ R_x @ R_y

    # Construct 4x4 transformation matrix
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = position

    return M
        

class Plotter:
    def render(self, graph_only=False):
        def render_points():
            try:
                fid = self.frame
                self.actors['points'] = self.plotter.add_points(to_np(self.params['pts'][fid]), 
                    opacity=1.0,
                    render_points_as_spheres=True, emissive=True, 
                    rgb=True, name='splats', scalars=to255(to_np(self.params['rgbs'][fid])), render=False)
            except Exception as e:
                points = self.actors.pop('points', None)
                if points is not None: self.plotter.remove_actor(points)
                msg = str(traceback.format_exc())
                self.messages.append(msg)
                logger.info(msg)

        def render_graph():
            # logger.info('Rendering graph')
            try:
                if self.is_2dmesh:
                    joints = deepcopy(self.params['joints'][self.frame])
                    joints, faces, keypoint_inds = mesh_lib.generate_parallelogram_mesh_3d(joints[0], joints[1], joints[2], nx=self.nxy[0], ny=self.nxy[1])
                    joints = torch.as_tensor(joints)
                    links = mesh_lib.faces_to_edges(torch.as_tensor(faces)).tolist() 
                    node_id = keypoint_inds[['top_left', 'top_right', 'bottom_right'][int(self.node_id)]]
                else:
                    joints = self.params['joints'][self.frame]
                    links = self.params['links']
                    node_id = int(self.node_id)

                self.actors['graph'] = vis_link(self.plotter, 
                            to_np(joints), links,
                            prefix='splats-', radius=self.node_size, render=False, 
                            joints_color=['yellow' if ji == node_id else 'red'  for ji in range(len(joints))], 
                            links_color='blue')
                # logger.info('Rendering graph finish')
            except Exception as e:
                for a in self.actors.pop('graph', []): self.plotter.remove_actor(a)
                msg = str(traceback.format_exc())
                self.messages.append(msg)
                logger.info(msg)
        
        if not graph_only:
            render_points()
        render_graph()
        self.plotter.render()
        
        
    def __init__(self, params, node_radius=0.01):
        num_nodes =len(params['joints'][0])
        num_frames=len(params['pts'])
        self.nxy = (7, 7)

        self.plotter = None
        self._viewer = None

        for k in ['pts', 'rgbs', 'joints', 'links']:
            assert k in params

        self.params = {}
        self.set_params(params)      
        self.actors = {}

        self.num_nodes = num_nodes
        self.num_frames = num_frames
        self.is_2dmesh = False

        self.function = "init-graph"
        self.frame = 0
        self.node_size = node_radius
        self.node_id = 0
        self.cursor_step_size = 0.01

        self.messages = []
    
    def set_params(self, params):
        for k in ['joints', 'links']:
            if k in params:
                params[k] = torch.as_tensor(params[k])
        self.params.update(params)
    @property
    def url(self):
        if self._viewer is None:
            def _menu():
                select(
                    model=("func", "init-graph"),
                    tooltip="func",
                    items=['Func', ['init-graph', 'global-move', 'node-adjustment']],
                    hide_details=True,
                    dense=True,
                )

                checkbox(model=("is_2dmesh", False), 
                         tooltip="visualize as 2d mesh", 
                         icons=["mdi-checkbox-multiple-blank-circle",  "mdi-checkbox-multiple-blank-circle-outline"])
                
                text_field(
                    model=("pre_inited_frames", self.params.get('pre_inited_frames', "")),
                    tooltip="Pre-inited Frames",
                    readonly=False,
                    type="text",
                    hide_details=True,
                    style="min-width: 40px; width: 100px",
                    classes='my-0 py-0 ml-1 mr-1',
                )

                text_field(
                    model=("frame", 0),
                    tooltip="Frames",
                    readonly=False,
                    type="number",
                    dense=True,
                    hide_details=True,
                    style="min-width: 40px; width: 100px",
                    classes='my-0 py-0 ml-1 mr-1',
                )
                text_field(
                    model=("node_size", self.node_size),
                    tooltip="node size",
                    readonly=False,
                    type="number",
                    dense=True,
                    hide_details=True,
                    style="min-width: 40px; width: 100px",
                    classes='my-0 py-0 ml-1 mr-1',
                )
                select(
                    model=("node_id", "0"),
                    tooltip="select node",
                    items=['Node', [str(v) for v in range(self.num_nodes)]],
                    hide_details=True,
                    dense=True,
                )

                def make_cursor_func(key):
                    def func():
                        try:
                            step_size = self.cursor_step_size
                            dir = {'w': 1, 's': -1, 'a': 1, 'd': -1, 'z': 1, 'x': -1}[key]
                            axis_ind = {'w': 0, 's': 0, 'a': 1, 'd': 1, 'z': 2, 'x': 2}[key]
                            _ = [0, 0, 0]
                            _[axis_ind] = 1
                            dir = dir * np.array(_) * step_size 
                            if self.function == "global-move":
                                self.params['joints'][self.frame] += torch.as_tensor(dir).reshape(1, 3)
                                # logger.info(f'global move {dir}')
                            elif self.function == "node-adjustment":
                                self.params['joints'][self.frame][self.node_id] += torch.as_tensor(dir)
                                # logger.info(f'node adjustment {dir}')
                            else:
                                return 
                            self.render(graph_only=True)
                        except Exception as e:
                            msg = str(traceback.format_exc())
                            self.messages.append(msg)
                            logger.info(msg)
                    return func
                                                                                                   
                button(
                    click=make_cursor_func('w'),
                    icon='mdi-arrow-left',
                    tooltip='Move Cursor X',
                )
                button(
                    click=make_cursor_func('s'),
                    icon='mdi-arrow-right',
                    tooltip='Move Cursor -X',
                )
                button(
                    click=make_cursor_func('a'),
                    icon='mdi-arrow-up',
                    tooltip='Move Cursor Y',
                )
                button(
                    click=make_cursor_func('d'),
                    icon='mdi-arrow-down',
                    tooltip='Move Cursor -Y',
                )
                button(
                    click=make_cursor_func('z'),
                    icon='mdi-arrow-top-right',
                    tooltip='Move Cursor Z',
                )
                button(
                    click=make_cursor_func('x'),
                    icon='mdi-arrow-bottom-left',
                    tooltip='Move Cursor -Z',
                )

                text_field(
                    model=("cursor_step_size", self.cursor_step_size),
                    tooltip="Cursor Step size",
                    readonly=False,
                    type="number",
                    dense=True,
                    hide_details=True,
                    style="min-width: 40px; width: 100px",
                    classes='my-0 py-0 ml-1 mr-1',
                )

            self.plotter = pv.Plotter(notebook=True, off_screen=False)
            self._viewer = self.plotter.show(return_viewer=True, jupyter_kwargs=dict(add_menu_items=_menu))
            state, ctrl = self._viewer.viewer.server.state, self._viewer.viewer.server.controller
            ctrl.view_update = self._viewer.viewer.update

            @state.change("frame")
            def update_frame_id(frame, **kwargs):
                try:
                    frame = int(float(frame))
                except:
                    return
                if 0 <= frame < self.num_frames:
                    self.frame = frame
                    self.render()
                    ctrl.view_update()
            
            @state.change("is_2dmesh")
            def update_is_2dmesh(is_2dmesh, **kwargs):
                self.is_2dmesh = bool(is_2dmesh)
                self.render()
                ctrl.view_update()
            
            @state.change("pre_inited_frames")
            def update_pre_inited_frames(pre_inited_frames, **kwargs):
                self.params['pre_inited_frames'] = pre_inited_frames

            @state.change("func")
            def set_function(func, **kwargs):
                self.function = func
                if func == 'init-graph':
                    self.params['joints'][:] = self.params['joints'][[self.frame]].clone()
                    logger.warning(f'init all joints at all frames to the frame {self.frame}')
                self.render()
                ctrl.view_update()

            @state.change("node_id")
            def set_node_id(node_id, **kwargs):
                try:
                    self.node_id = int(node_id)
                except:
                    return
                self.render(graph_only=True)
                ctrl.view_update()
            
            @state.change("node_size")
            def set_node_size(node_size, **kwargs):
                try:
                    node_size = float(node_size)
                except:
                    return
                self.node_size = node_size
                self.render(graph_only=True)
                ctrl.view_update()
                
            @state.change("cursor_step_size")
            def set_cursor_step_size(cursor_step_size, **kwargs):
                try:
                    cursor_step_size = float(cursor_step_size)
                except:
                    return 
                self.cursor_step_size = cursor_step_size 
                ctrl.view_update()

        url = self._viewer.value.split("src=\"")[1].split("\"")[0]
        logger.info(f'pyvista viewer url: {url}')
        return url
    
    def __del__(self):
        try:
            self.plotter.close()
        except: pass
    