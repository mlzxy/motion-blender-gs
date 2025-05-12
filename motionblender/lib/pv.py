import asyncio
from loguru import logger
import pickle 
import pyvista as pv
import torch
from jaxtyping import Float32, Int64
import numpy as np
import traceback
from torch import Tensor
import os.path as osp
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
            color = links_color[i]
        else:
            color = 'red'
        a = plotter.add_mesh(pv.Tube(joint1, joint2, radius=radius/2), color=color, name=f'{prefix}tube-{i}', render=render)
        actors.append(a)
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
    def render(self, render=True, offline=False, reset_before_render=False, pyrender_fix=False):
        if self.backend.startswith('remote:'):
            backend = self.backend.split('remote:')[-1]
            self.dump(backend)
            logger.info(f'pyvista rendering to {backend}')
            return 
        elif self.backend.startswith('from:'):
            backend = self.backend.split('from:')[-1]
            if not osp.exists(backend):
                logger.info(f'file {backend} does not exist yet')
                return
            remote_mtime = osp.getmtime(backend)
            if self.gui_state['pv'].get('remote_mtime', -1) != remote_mtime:
                logger.info(f'rendering from {backend}')
                self.load(backend)
            self.gui_state['pv']['remote_mtime'] = remote_mtime

        if self._viewer is None:
            logger.info('Viewer not initialized, skip rendering') 
            return

        fid = self.gui_state['ui']['frame_id']
        view_mode = self.gui_state['ui']['view']
        node_radius = self.gui_state['ui']['node_radius']
        
        def remove_points():
            points = self.actors.pop('points', None)
            if points is not None: self.plotter.remove_actor(points)

        def remove_graph():
            for a in self.actors.pop('graph', []): self.plotter.remove_actor(a)

        def render_points():
            if 'means' in self.params and self.params['means'] is not None:
                try:
                    opacity = self.params.get('opacities', 1.0)
                    # if not isinstance(opacity, (float, int)):
                    #     opacity = opacity.reshape(-1, 1)
                    
                    self.actors['points'] = self.plotter.add_points(to_np(self.params['means'][fid]), 
                        opacity=opacity,
                        render_points_as_spheres=True, emissive=True, 
                        rgb=True, name='splats', scalars=to255(to_np(self.params.get('colors', None))), render=False)
                except Exception as e:
                    remove_points()
                    logger.info(str(traceback.format_exc()))

        def render_graph():
            if 'graph' in self.params and 'joints' in self.params['graph'] and self.params['graph']['joints'] is not None and \
                'links' in self.params['graph'] and self.params['graph']['links'] is not None:
                try:
                    if pyrender_fix:
                        for a in self.actors.get('graph', []):
                            a.SetVisibility(False)
                    
                    self.actors['graph'] = vis_link(self.plotter, 
                                to_np(self.params['graph']['joints'][fid]), self.params['graph']['links'], 
                                prefix='splats-', radius=node_radius, render=False, 
                                joints_color=self.params['graph'].get('joints_color', None), 
                                links_color=self.params['graph'].get('links_color', None))

                    if pyrender_fix:
                        for a in self.actors.get('graph', []):
                            a.SetVisibility(True)
                except Exception as e:
                    remove_graph()
                    logger.info(str(traceback.format_exc()))
        

        def render_meshes():
            if len(self.params.get('meshes', [])) > 0:
                try:
                    actors = []
                    for i, m_dict in enumerate(self.params['meshes']):
                        if isinstance(m_dict, list):
                            m_dict = m_dict[int(fid)]
                        actors.append(self.plotter.add_mesh(m_dict['mesh'], **m_dict['kwargs'], render=False, name=m_dict.get('name', str(i))))
                    self.actors['meshes'] = actors
                except Exception as e:
                    remove_meshes()
                    logger.info(str(traceback.format_exc()))

        def remove_meshes():
            for a in self.actors.pop('meshes', []): self.plotter.remove_actor(a)

        if reset_before_render:
            remove_points()
            remove_graph()
            remove_meshes()

        if view_mode == 'points':
            remove_graph()
            remove_meshes()
            render_points()
        elif view_mode == 'graph':
            remove_points() 
            render_graph()
            render_meshes()
        else:
            render_points()
            render_graph()
            render_meshes()
        
        if render:
            if offline:
                self.plotter.write_frame()
            else:
                self.plotter.render()
        
        
    def __init__(self, node_radius=0.1, num_frames=500, backend='/dev/shm/pvlib.state', notebook=True, off_screen=False, 
                 enable_point_cursor=False, frame_selection=True):
        self.plotter = None
        self._viewer = None
        self.params = {}
        self.frame_selection = frame_selection
        self.actors = {}
        self.backend = backend
        self.gui_state = {
            'pv': {},
            'ui': {
                'node_radius': node_radius,
                'cursor_step_size': 0.1,
                'frame_id': 0,
                'view': 'all',
                'play_step': 10,
                'play_sleep': 0.5
            },
            'app': {
                'num_frames': num_frames,
            }
        }
        self.notebook = notebook
        self.off_screen = off_screen
        
        self.enable_point_cursor_function = enable_point_cursor
        self.enable_pc = False
        self.pc_position = [0., 0., 0.]
        self.messeges = []
    
    def dump(self, path='/dev/shm/pvlib.state'):
        state = {'params': self.params, 'gui_state': {'app': self.gui_state['app']}}
        with open(path, 'wb') as f:
            f.write(pickle.dumps(state))
    
    def load(self, path='/dev/shm/pvlib.state'):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.params = state['params']
        self.gui_state['app'] = state['gui_state']['app']
    
    def reset(self):
        if self.plotter is not None:
            self.plotter.close()
        self._viewer = None
        self.plotter = pv.Plotter(notebook=self.notebook, off_screen=self.off_screen)
    
    def clear(self):
        self.plotter.clear_actors()
    
    def update_params(self, **kwargs):
        for k, v in kwargs.items():
            self.params[k] = v  
    
    def update_param(self, key, value):
        params = self.params
        keys = key.split('.')
        for i, k in enumerate(keys):
            if i == len(keys) - 1:
                params[k] = value
            else: 
                if k not in params:
                    params[k] = {}
                params = params[k] 

    @property
    def url(self):
        if (not self.backend.startswith('remote:')) and self._viewer is None:
            self.reset()
            if self.off_screen: return None
            def _menu():
                if self.frame_selection:
                    button(
                        click=button_play,
                        icon='mdi-play',
                        tooltip='Play',
                    )
                    text_field(
                        model=("frames", 0),
                        tooltip="Frames",
                        readonly=False,
                        type="number",
                        dense=True,
                        hide_details=True,
                        style="min-width: 40px; width: 100px",
                        classes='my-0 py-0 ml-1 mr-1',
                    )
                    slider(
                        model=("frames", 0),
                        tooltip="Frames",
                        min=0,
                        max=self.gui_state['app']['num_frames'] - 1,
                        step=1,
                        dense=True,
                        hide_details=True,
                        style="width: 300px",
                        classes='my-0 py-0 ml-1 mr-1',
                    )
                slider(
                    model=("node_size", 0.1),
                    tooltip="Node size",
                    min=0,
                    max=self.gui_state['ui']['node_radius'],
                    step=0.002,
                    dense=True,
                    hide_details=True,
                    style="width: 100px",
                    classes='my-0 py-0 ml-1 mr-1',
                )
                select(
                    model=("view", "all"),
                    tooltip="Toggle data view",
                    items=['View', ["points", "graph", "all"]],
                    hide_details=True,
                    dense=True,
                )

                if self.enable_point_cursor_function:
                    # icons: https://pictogrammers.github.io/@mdi/font/4.5.95/
                    checkbox(model=("enable_pc", False), tooltip="Enable Point Cursor", icons=["mdi-checkbox-multiple-blank-circle",  "mdi-checkbox-multiple-blank-circle-outline"])

                    def make_cursor_func(key):
                        def func():
                            step_size = self.gui_state['ui']['cursor_step_size']
                            if 'pc' in self.actors and self.enable_pc:
                                dir = {'w': 1, 's': -1, 'a': 1, 'd': -1, 'z': 1, 'x': -1}[key]
                                axis_ind = {'w': 0, 's': 0, 'a': 1, 'd': 1, 'z': 2, 'x': 2}[key]

                                # cam = self.plotter.camera
                                # cam_mat = camera_to_world_matrix(cam.position, cam.azimuth, cam.roll, cam.elevation)
                                # cam_mat[axis_ind, :3]
                                _ = [0, 0, 0]
                                _[axis_ind] = 1
                                dir = dir * np.array(_) * step_size 
                                
                                self.pc_mesh.translate(dir, inplace=True)
                                self.pc_position = self.pc_mesh.center.copy()
                                self.actors['pc'] = self.plotter.add_mesh(self.pc_mesh, color='yellow', name='pc', render=True)
                                logger.info(f'point cursor position: {self.pc_position}')
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

                    slider(
                        model=("cursor_step_size", 0.1),
                        tooltip="Cursor Step size",
                        min=0,
                        max=self.gui_state['ui']['cursor_step_size'],
                        step=0.002,
                        dense=True,
                        hide_details=True,
                        style="width: 100px",
                        classes='my-0 py-0 ml-1 mr-1',
                    )

            def button_play():
                state.play = not state.play
                state.flush()
        
            self._viewer = self.plotter.show(return_viewer=True, jupyter_kwargs=dict(add_menu_items=_menu))
            state, ctrl = self._viewer.viewer.server.state, self._viewer.viewer.server.controller
            ctrl.view_update = self._viewer.viewer.update
            self.gui_state['pv'] = {'state': state, 'ctrl': ctrl}


            if self.frame_selection:
                @state.change("play")
                async def _play(play, **kwargs):
                    while state.play:
                        if state.frames + self.gui_state['ui']['play_step'] >= len(self.params['means']):
                            state.frames = len(self.params['means']) - 1
                            state.play = False
                        else:
                            state.frames += self.gui_state['ui']['play_step']
                            state.flush()
                        await asyncio.sleep(self.gui_state['ui']['play_sleep'])

                @state.change("frames")
                def update_frame_id(frames, **kwargs):
                    frame_id = frames
                    self.gui_state['ui']['frame_id'] = int(frame_id)
                    self.render()
                    ctrl.view_update()
            
            @state.change("view")
            def set_view(view, **kwargs):
                self.gui_state['ui']['view'] = view 
                self.render()
                ctrl.view_update()
            
            @state.change("node_size")
            def set_node_size(node_size, **kwargs):
                self.gui_state['ui']['node_radius'] = node_size 
                self.render()
                ctrl.view_update()
         
            if self.enable_point_cursor_function:

                @state.change("enable_pc")
                def set_enable_pc(enable_pc, **kwargs):
                    self.enable_pc = bool(enable_pc)
                    if self.enable_pc and 'pc' not in self.actors:
                        self.pc_mesh = pv.Sphere(radius=self.gui_state['ui']['node_radius'] * 2, center=self.pc_position)
                        self.actors['pc'] = self.plotter.add_mesh(self.pc_mesh, color='yellow', name='pc', render=True)
                    elif not self.enable_pc and 'pc' in self.actors:
                        pc = self.actors.pop('pc', None)
                        if pc: self.plotter.remove_actor(pc)
                    ctrl.view_update()
                
                @state.change("cursor_step_size")
                def set_cursor_step_size(cursor_step_size, **kwargs):
                    self.gui_state['ui']['cursor_step_size'] = cursor_step_size 
                    ctrl.view_update()

        if not self.backend.startswith('remote:'):
            url = self._viewer.value.split("src=\"")[1].split("\"")[0]
            logger.info(f'pyvista viewer url: {url}')
            return url
        else:
            return self.backend
    
    def __del__(self):
        try:
            self.plotter.close()
        except: pass
    

if __name__ == "__main__":
    import sys, time
    p = Plotter(backend="")
    logger.info(p.url)
    while True:
        txt = input("provide a pvstate full path>").strip()
        if txt == '':
            continue
        if txt == 'q': 
            break
        else:
            if osp.exists(txt):
                p.backend = f"from:{txt}"
                p.render()
            else:
                logger.info(f"file {txt} does not exist")
                continue