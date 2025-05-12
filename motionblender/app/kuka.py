from motionblender.app.utils import RobotInterface, MotionBlender
import os
import os.path as osp
import roma
import motionblender.lib.misc as misc
import torch.nn.functional as F
from loguru import logger
import torch.nn as nn
import torch.optim as optim
import numpy as np
import motionblender.lib.animate as anim
from jaxtyping import Float32
from tqdm.auto import tqdm, trange
import kinpy as kp
import torch
import transformations as tf
from torch import Tensor

def map_gripper_ctrl_to_joint_v(gv):
    v = max(min(gv, 100), 30)
    v = 0.8 * (v - 30) / 70
    return {
        'finger_1_joint_1': v,
        'finger_2_joint_1': v,
        'finger_middle_joint_1': v
    }

robot_joints = [
    'joint_0',
    'joint_1',
    'joint_2',
    'joint_3',
    'joint_4',
    'joint_5',
    'joint_6',
    'joint_7', 
    'ee',
    'palm',
    'left_finger_1',
    'right_finger_1',
    'left_finger_2',
    'right_finger_2',
]

robot_joints_from_origin = {
    'joint_0': 'iiwa_link_0',
    'joint_1': 'iiwa_link_1',
    'joint_2': 'iiwa_link_2',
    'joint_3': 'iiwa_link_3',
    'joint_4': 'iiwa_link_4',
    'joint_5': 'iiwa_link_5',
    'joint_6': 'iiwa_link_6',
    'joint_7': 'iiwa_link_7',
    'ee': 'iiwa_link_ee',
    'palm': 'palm',
    'left_finger_1': ['finger_middle_link_0'],
    'right_finger_1': ['finger_1_link_0', 'finger_2_link_0'],
    'left_finger_2': ['finger_middle_link_3'],
    'right_finger_2': ['finger_1_link_3', 'finger_2_link_3'],
}

robot_connections = [
    ['joint_0', 'joint_1'],
    ['joint_1', 'joint_2'],
    ['joint_2', 'joint_3'],
    ['joint_3', 'joint_4'],
    ['joint_4', 'joint_5'],
    ['joint_5', 'joint_6'],
    ['joint_6', 'joint_7'],
    ['joint_7', 'ee'],
    ['ee', 'palm'],
    ['palm', 'left_finger_1'],
    ['palm', 'right_finger_1'],
    ['left_finger_1', 'left_finger_2'],
    ['right_finger_1', 'right_finger_2'],
]

robot_connections_int = [(robot_joints.index(a), robot_joints.index(b)) for a, b in robot_connections]


default_joint_values = {'iiwa_joint_1': -0.15783709287643433,
 'iiwa_joint_2': 0.48583802580833435,
 'iiwa_joint_3': 1.0546152225288097e-05,
 'iiwa_joint_4': -1.6795016527175903,
 'iiwa_joint_5': 0.9391155242919922,
 'iiwa_joint_6': 1.027316689491272,
 'iiwa_joint_7': -1.273979663848877,
 'palm_finger_1_joint': -0.16,
 'palm_finger_2_joint': 0.16,
 'finger_middle_joint_3': 0,
 'finger_middle_joint_2': 0,
 'finger_2_joint_3': 0,
 'finger_2_joint_2': 0,
 'finger_1_joint_3': 0,
 'finger_1_joint_2': 0,
 'finger_1_joint_1': 0.0,
 'finger_2_joint_1': 0.0,
 'finger_middle_joint_1': 0.0}


class Kuka(RobotInterface):
    def __init__(self, iiwa_path=os.path.dirname(__file__) + "/iiwa/kuka.urdf"):
        super().__init__()
        txt = open(iiwa_path).read()
        self.iiwa_serial_chain = kp.build_serial_chain_from_urdf(txt, 'palm')
        self.iiwa_chain = kp.build_chain_from_urdf(txt)
        self.joint_names = [f'iiwa_joint_{i}' for i in range(1, 8)]
        self.curr_jvs = np.array([default_joint_values[jname] for jname in self.joint_names])
        self.buf['rot6d'] = None
    
    def _joint_values_to_rot6d(self, joint_values):
        link_poses = self.iiwa_chain.forward_kinematics(joint_values)
        robot_joints_values = {}
        for rj_name, parents in robot_joints_from_origin.items(): 
            if isinstance(parents, list):
                values = []
                for p in parents:
                    values.append(link_poses[p].pos)
                robot_joints_values[rj_name] = sum(values) / len(values)
            else:
                robot_joints_values[rj_name] = link_poses[parents].pos

        joint_positions = []
        for rj_name in robot_joints:
            joint_positions.append(robot_joints_values[rj_name])
        joint_positions = torch.from_numpy(np.array(joint_positions)).float()
        anim_chain = anim.inverse_kinematic(joint_positions, robot_connections_int)
        rot6d = anim.retrieve_tensor_from_chain(anim_chain, 'rot6d')
        return rot6d

    def initialize(self, motion_module: MotionBlender) -> None:
        gripper_poses = getattr(motion_module, 'gripper_poses', [])
        if len(gripper_poses) == 0:
            if hasattr(motion_module, 'original_cano_t'):
                original_cano_t = motion_module.original_cano_t
            else:
                original_cano_t = 323

            ROBOT_ROOT_DATASET_DIR = './datasets/robot/'
            img_id = misc.load_json(osp.join(ROBOT_ROOT_DATASET_DIR, 'dataset.json'))['ids'][original_cano_t]
            jv = misc.load_cpkl(osp.join(ROBOT_ROOT_DATASET_DIR, 'robot_rawdata.pkl'))['joint_pos_list'][int(img_id)]
            self.curr_jvs = np.array(jv)
            end_pose_tf = self.iiwa_serial_chain.forward_kinematics(self.curr_jvs)
            ee_pose_robot_base = torch.from_numpy(end_pose_tf.matrix()).float().to(next(motion_module.parameters()).device)
            self.buf['pose'] = ee_pose_robot_base
            self.buf['degree'] = 50.
        else:
            self.buf['pose'] = gripper_poses[0]
            self.buf['degree'] = motion_module.gripper_degrees[0]

        self.get_joint_rotations()
        self.inited = True
    
    def get_joint_rotations(self, refresh=False) -> Float32[Tensor, "j 6"]:
        if not self.buf['stale'] and self.buf['rot6d'] is not None and not refresh:
            return self.buf['rot6d']

        matrix = self.buf['pose'].cpu().numpy()
        trans = kp.Transform(rot=tf.quaternion_from_matrix(matrix), pos=matrix[:3, 3])
        self.curr_jvs = self.iiwa_serial_chain.inverse_kinematics(trans, self.curr_jvs)
        joint_values = {**default_joint_values, **map_gripper_ctrl_to_joint_v(self.buf['degree']), 
                        **dict(zip(self.joint_names, self.curr_jvs))}
        rot6d = self._joint_values_to_rot6d(joint_values)
        self.buf['rot6d'] = rot6d.to(self.buf['pose'].device)
        self.buf['stale'] = False
        return self.buf['rot6d']


robot = Kuka()

if __name__ == "__main__":
    gs_modules, motion_modules, _, gaussian_names = misc.load_cpkl("outputs/mb/robot/okish/toy/ckpt.robot.cpkl")
    robot.initialize(motion_modules['robot'])
    robot.buf['pose'][:3, 3] += (torch.rand(3).cuda() * 0.3)
    robot.buf['pose'][:3, :3] = roma.random_rotmat().cuda() @ robot.buf['pose'][:3, :3]
    print(robot.get_joint_rotations(refresh=True))