import meshcat.geometry as mg
import numpy as np
from pinocchio.visualize import MeshcatVisualizer
from pinocchio import BODY
import pinocchio as pin

from loop_rate_limiters import RateLimiter

import pink
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask

import os
import meshcat.geometry as mg
import qpsolvers
import sys
# from scipy.spatial.transform import Rotation as R
# import ipdb
# import torch
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# from scipy.spatial.transform import Rotation as sRot

# from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
# from teleop.robot_control.hand_retargeting import HandRetargeting, HandType
# from multiprocessing import shared_memory, Array, Lock

import logging
import os
from typing import Tuple
from dex_retargeting.retargeting_config import RetargetingConfig
from pathlib import Path
import yaml
from teleop.utils.other_utils import KalmanFilterSmoother

current_dir = os.path.dirname(os.path.abspath(__file__))

class GR1T2_ArmIK:

    def __init__(self, urdf_path: str, visualize=False) -> None:
        self.visualize = visualize

        self.mixed_jointsToLockIDs = [
            'left_hip_roll_joint',
            'left_hip_yaw_joint',
            'left_hip_pitch_joint',
            'left_knee_pitch_joint',
            'left_ankle_pitch_joint',
            'left_ankle_roll_joint',
            'right_hip_roll_joint',
            'right_hip_yaw_joint',
            'right_hip_pitch_joint',
            'right_knee_pitch_joint',
            'right_ankle_pitch_joint',
            'right_ankle_roll_joint',
            'waist_yaw_joint',
            'waist_pitch_joint',
            'waist_roll_joint',
            # "head_roll_joint",
            # "head_pitch_joint",
            # "head_yaw_joint", # this part apply directly ?

            # "left_shoulder_pitch_joint",
            # "left_shoulder_roll_joint",
            # "left_shoulder_yaw_joint",
            # "left_elbow_pitch_joint",
            # "left_wrist_yaw_joint",
            # "left_wrist_pitch_joint",
            # "left_wrist_roll_joint",
            # "right_shoulder_pitch_joint",
            # "right_shoulder_roll_joint",
            # "right_shoulder_yaw_joint",
            # "right_elbow_pitch_joint",
            # "right_wrist_yaw_joint",
            # "right_wrist_pitch_joint",
            # "right_wrist_roll_joint",   # arm related, used for ik
            'L_index_proximal_joint',
            'L_index_intermediate_joint',
            'L_middle_proximal_joint',
            'L_middle_intermediate_joint',
            'L_pinky_proximal_joint',
            'L_pinky_intermediate_joint',
            'L_ring_proximal_joint',
            'L_ring_intermediate_joint',
            'L_thumb_proximal_yaw_joint',
            'L_thumb_proximal_pitch_joint',
            "L_thumb_intermediate_joint",  # fu
            'L_thumb_distal_joint',
            'R_index_proximal_joint',
            'R_index_intermediate_joint',
            'R_middle_proximal_joint',
            'R_middle_intermediate_joint',
            'R_pinky_proximal_joint',
            'R_pinky_intermediate_joint',
            'R_ring_proximal_joint',
            'R_ring_intermediate_joint',
            'R_thumb_proximal_yaw_joint',
            'R_thumb_proximal_pitch_joint',
            "R_thumb_intermediate_joint", # fu
            'R_thumb_distal_joint',  # all hand-related joints
        ]

        # Handle relative path in urdf which is not supported by pinocchio :<
        urdf_path = os.path.abspath(urdf_path)
        with open(urdf_path, 'r') as f:
            urdf_text = f.read()
        urdf_text = urdf_text.replace('filename="', 'filename="' + os.path.dirname(urdf_path) + '/')
        tmp_urdf_path = urdf_path + '.tmp.urdf'
        with open(tmp_urdf_path, 'w') as f:
            f.write(urdf_text)
        self.robot_wrapper = pin.RobotWrapper.BuildFromURDF(tmp_urdf_path)
        os.remove(tmp_urdf_path)


        self.reduced_robot = self.robot_wrapper.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot_wrapper.model.nq),
        )

        # # Initialize the Meshcat visualizer
        if self.visualize:
            self.vis = MeshcatVisualizer(
                self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model
            )
            self.reduced_robot.setVisualizer(self.vis, init=False)
            self.vis.initViewer(open=True)
            self.vis.loadViewerModel('pinocchio')

        self.config = pink.Configuration(self.reduced_robot.model, self.reduced_robot.data, self.reduced_robot.q0)
        self.link_names = [frame.name for frame in self.reduced_robot.model.frames if frame.type == BODY]
        self.q_max, self.q_min = self.make_joint_config(self.config, 0.9)

        # self.left_hand_index = self.reduced_robot.model.getFrameId('l_hand_base_link')
        # self.right_hand_index = self.reduced_robot.model.getFrameId('r_hand_base_link')
        self.left_hand_index = self.reduced_robot.model.getFrameId('L_hand_base_link')
        self.right_hand_index = self.reduced_robot.model.getFrameId('R_hand_base_link')
        self.head_index = self.reduced_robot.model.getFrameId('head_yaw_link')

        if self.visualize:
            self.vis.displayFrames(True, frame_ids=[self.left_hand_index, self.right_hand_index, self.head_index])
            self.vis.display(pin.neutral(self.reduced_robot.model))
            # Enable the display of end effector target frames with short axis lengths and greater width.
            frame_viz_names = ['L_ee_target', 'R_ee_target', 'head_target']
            FRAME_AXIS_POSITIONS = (
                np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]]).astype(np.float32).T
            )
            FRAME_AXIS_COLORS = (
                np.array([[1, 0, 0], [1, 0.6, 0], [0, 1, 0], [0.6, 1, 0], [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T
            )
            axis_length = 0.1
            axis_width = 100
            for frame_viz_name in frame_viz_names:
                self.vis.viewer[frame_viz_name].set_object(
                    mg.LineSegments(
                        mg.PointsGeometry(
                            position=axis_length * FRAME_AXIS_POSITIONS,
                            color=FRAME_AXIS_COLORS,
                        ),
                        mg.LineBasicMaterial(
                            linewidth=axis_width,
                            vertexColors=True,
                        ),
                    )
                )

            self.vis.viewer['Origin'].set_object(
                mg.LineSegments(
                    mg.PointsGeometry(
                        position=0.3 * FRAME_AXIS_POSITIONS,
                        color=FRAME_AXIS_COLORS,
                    ),
                    mg.LineBasicMaterial(
                        linewidth=30,
                        vertexColors=True,
                    ),
                )
            )

        self.default_pose = self.reduced_robot.q0.copy()
        self.default_pose_list = {'left_elbow_pitch_joint': -90.0, 'right_elbow_pitch_joint': -90.0}

        for item in self.default_pose_list.keys():
            self.default_pose[self.reduced_robot.model.getJointId(item) - 1] = np.deg2rad(self.default_pose_list[item])

        self.tasks = {}
        
        # self.tasks['right_hand'] = FrameTask(
        #     frame='r_hand_base_link', position_cost=1.0, orientation_cost=1.0, lm_damping=0.5
        # )
        # self.tasks['left_hand'] = FrameTask(
        #     frame='l_hand_base_link', position_cost=1.0, orientation_cost=1.0, lm_damping=0.5
        # )

        self.tasks['right_hand'] = FrameTask(
            frame='R_hand_base_link', position_cost=1.0, orientation_cost=1.0, lm_damping=0.5
        )
        self.tasks['left_hand'] = FrameTask(
            frame='L_hand_base_link', position_cost=1.0, orientation_cost=1.0, lm_damping=0.5
        )

        self.tasks['head'] = FrameTask(frame='head_yaw_link', position_cost=0.0, orientation_cost=1.0, lm_damping=1.0)
        self.posture_task = PostureTask(cost=0.001, lm_damping=1.0, gain=0.1)
        self.posture_task.set_target(self.default_pose)

        # define QP-solver
        self.solver = qpsolvers.available_solvers[0]
        if 'quadprog' in qpsolvers.available_solvers:
            solver = 'quadprog'
        print(f'Using {solver} QP Solver')
        self.rate = RateLimiter(frequency=200.0)
        self.dt = self.rate.period

    def get_root_joint_dim(self, model):
        if model.existJointName('root_joint'):
            root_joint_id = model.getJointId('root_joint')
            root_joint = model.joints[root_joint_id]
            return root_joint.nq, root_joint.nv
        return 0, 0

    def qpos_clamp(self):
        start, _ = self.get_root_joint_dim(self.config.model)
        end = self.config.model.nq
        qpos = self.config.q.copy()
        qpos[start:end] = np.clip(qpos[start:end], a_min=self.q_min[start:end], a_max=self.q_max[start:end])
        qpos.setflags(write=False)
        self.config.q = qpos

    def make_joint_config(self, config, limit=0.95):
        q_max = config.model.upperPositionLimit
        q_min = config.model.lowerPositionLimit
        q_mean = (q_max + q_min) / 2
        q_scale = (q_max - q_min) * limit
        q_max = q_mean + q_scale / 2
        q_min = q_mean - q_scale / 2
        return q_max, q_min

    def array2SE3(self, pose: np.ndarray):
        transform = pose[:3, :3]
        translation = pose[:3, 3]
        return pin.SE3(pin.SE3(transform, translation))

    def solve_ik(self, left_pose: np.ndarray, right_pose: np.ndarray, head_pose: np.ndarray) -> Tuple[np.ndarray, bool]:

        if self.visualize:
            self.vis.viewer['L_ee_target'].set_transform(left_pose)
            self.vis.viewer['R_ee_target'].set_transform(right_pose)
            self.vis.viewer['head_target'].set_transform(head_pose)
            origin_pose = np.eye(4)
            self.vis.viewer['Origin'].set_transform(origin_pose)

        try:
            self.tasks['right_hand'].set_target(self.array2SE3(right_pose))
            self.tasks['left_hand'].set_target(self.array2SE3(left_pose))
            self.tasks['head'].set_target(self.array2SE3(head_pose))
            velocity = solve_ik(
                self.config, list(self.tasks.values()) + [self.posture_task], self.dt, solver=self.solver
            )

            self.config.integrate_inplace(velocity, self.dt)
            self.qpos_clamp()
            sol_q = self.config.q
            if self.visualize:
                self.vis.display(sol_q)
            return sol_q, True  # target position; needed torques

        except Exception as e:
            logging.error(f'exception in ik solver: {e}')
            return np.zeros(self.reduced_robot.model.nq), False


class Camera_IK:
    def __init__(self, visualize):
        urdf_path = "../assets/GRX/GR1/GR1T2/urdf/GR1T2_inspire_hand.urdf"
        self.arm_ik = GR1T2_ArmIK(urdf_path, visualize)

        # retargeting_config_path = "../assets/inspire_hand_gr1t2/inspire_hand.yml"
        retargeting_config_path = "../assets/dex_inspire_hand/inspire_hand.yml"
        RetargetingConfig.set_default_urdf_dir(os.path.dirname(retargeting_config_path))
        with Path(retargeting_config_path).open('r') as f:
            cfg = yaml.safe_load(f)

        self.left_retargeting = RetargetingConfig.from_dict(cfg['left']).build()
        self.right_retargeting = RetargetingConfig.from_dict(cfg['right']).build()
        self.tip_indices = [4,8,12,16,20]

        # 添加卡尔曼滤波
        initial_q = np.zeros(17)
        dt = 1
        self.kalman_filter = KalmanFilterSmoother(initial_state=initial_q, dt=dt)

        self.previous_left_qpos = np.zeros(12)
        self.previous_right_qpos = np.zeros(12)
 


    def ik_fun(self,data):

        target_joint_positions = np.zeros(41)
        head_mat = np.eye(4)
        sol_q, success = self.arm_ik.solve_ik(data[0], data[1], head_mat)
        # 卡尔曼滤波平滑一下
        sol_q = self.kalman_filter.update(sol_q)

        # 将小拇指进行缩放
        data[2][20] = data[2][20] * 1.2
        data[3][20] = data[3][20] * 1.2
        # 判断data[2]是否是np.zeors(21,3)
        if np.sum(data[2])==0:
            left_qpos = self.previous_left_qpos
        else:    
            left_qpos = self.left_retargeting.retarget(data[2][self.tip_indices])
            self.previous_left_qpos = left_qpos

        if np.sum(data[3])==0:
            right_qpos = self.previous_right_qpos
        else:
            right_qpos = self.right_retargeting.retarget(data[3][self.tip_indices])
            self.previous_right_qpos = right_qpos
        

        # 如果感觉大拇指抓取有一些吃力，就采用大拇指弯曲来自其他四个手指的平均值
        mean_thumb_flag = False
        if mean_thumb_flag:
            other_left_qpos = np.sum(left_qpos[:8])/8
            # print("other_left_qpos", other_left_qpos)
            left_thumb1 = other_left_qpos/1.7*1.3
            left_thumb2 = other_left_qpos/1.7*0.6
            left_thumb3 = other_left_qpos/1.7*0.8
            left_thumb4 = other_left_qpos/1.7*1.2
            left_qpos[8:] = np.array([left_thumb1, left_thumb2, left_thumb3, left_thumb4])
            # print("left_qpos[8:]", left_qpos[8:])

            other_right_qpos = np.sum(right_qpos[:8])/8
            # print("other_right_qpos", other_right_qpos)
            right_thumb1 = other_right_qpos/1.7*1.3
            right_thumb2 = other_right_qpos/1.7*0.6
            right_thumb3 = other_right_qpos/1.7*0.8
            right_thumb4 = other_right_qpos/1.7*1.2
            right_qpos[8:] = np.array([right_thumb1, right_thumb2, right_thumb3, right_thumb4])

        target_joint_positions[:3] = sol_q[:3] 
        target_joint_positions[3:10] = sol_q[3:10] 
        target_joint_positions[10:22] = left_qpos
        target_joint_positions[22:29] = sol_q[10:17] 
        target_joint_positions[29:41] = right_qpos

        return target_joint_positions

