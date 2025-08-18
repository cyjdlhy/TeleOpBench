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
from scipy.spatial.transform import Rotation as R
current_dir = os.path.dirname(os.path.abspath(__file__))
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial.transform import Rotation as sRot
from pinocchio import casadi as cpin  
import casadi 
from teleop.utils.weighted_moving_filter import WeightedMovingFilter
from dex_retargeting.retargeting_config import RetargetingConfig
import yaml
from pathlib import Path

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
from teleop.utils.other_utils import KalmanFilterSmoother

class H1_2_ArmIK:
    def __init__(self, Unit_Test = False, Visualization = False):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.Unit_Test = Unit_Test
        self.Visualization = Visualization

        if not self.Unit_Test:
            self.robot = pin.RobotWrapper.BuildFromURDF('../assets/h1_2/h1_2.urdf', '../assets/h1_2/')
        else:
            self.robot = pin.RobotWrapper.BuildFromURDF('../../assets/h1_2/h1_2.urdf', '../../assets/h1_2/') # for test

        for i in range(self.robot.model.njoints):
            print(f"{self.robot.model.names[i]}")

        self.mixed_jointsToLockIDs = [
                                      "left_hip_yaw_joint",
                                      "left_hip_pitch_joint",
                                      "left_hip_roll_joint",
                                      "left_knee_joint",
                                      "left_ankle_pitch_joint",
                                      "left_ankle_roll_joint",
                                      "right_hip_yaw_joint",
                                      "right_hip_pitch_joint",
                                      "right_hip_roll_joint",
                                      "right_knee_joint",
                                      "right_ankle_pitch_joint",
                                      "right_ankle_roll_joint",
                                      "torso_joint",
                                    # "left_shoulder_pitch_joint",
                                    # "left_shoulder_roll_joint",
                                    # "left_shoulder_yaw_joint",
                                    # "left_elbow_joint",
                                    # "left_wrist_roll_joint",
                                    # "left_wrist_pitch_joint",
                                    # "left_wrist_yaw_joint",
                                      "L_index_proximal_joint",
                                      "L_index_intermediate_joint",
                                      "L_middle_proximal_joint",
                                      "L_middle_intermediate_joint",
                                      "L_pinky_proximal_joint",
                                      "L_pinky_intermediate_joint",
                                      "L_ring_proximal_joint",
                                      "L_ring_intermediate_joint",
                                      "L_thumb_proximal_yaw_joint",
                                      "L_thumb_proximal_pitch_joint",
                                      "L_thumb_intermediate_joint",
                                      "L_thumb_distal_joint",
                                    # "right_shoulder_pitch_joint",
                                    # "right_shoulder_roll_joint",
                                    # "right_shoulder_yaw_joint",
                                    # "right_elbow_joint",
                                    # "right_wrist_roll_joint",
                                    # "right_wrist_pitch_joint",
                                    # "right_wrist_yaw_joint",
                                      "R_index_proximal_joint",
                                      "R_index_intermediate_joint",
                                      "R_middle_proximal_joint",
                                      "R_middle_intermediate_joint",
                                      "R_pinky_proximal_joint",
                                      "R_pinky_intermediate_joint",
                                      "R_ring_proximal_joint",
                                      "R_ring_intermediate_joint",
                                      "R_thumb_proximal_yaw_joint",
                                      "R_thumb_proximal_pitch_joint",
                                      "R_thumb_intermediate_joint",
                                      "R_thumb_distal_joint"
                                    ]

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )

        self.reduced_robot.model.addFrame(
            pin.Frame('L_ee',
                      self.reduced_robot.model.getJointId('left_wrist_yaw_joint'),
                      pin.SE3(np.eye(3),
                              np.array([0.05,0,0]).T),
                      pin.FrameType.OP_FRAME)
        )
        
        self.reduced_robot.model.addFrame(
            pin.Frame('R_ee',
                      self.reduced_robot.model.getJointId('right_wrist_yaw_joint'),
                      pin.SE3(np.eye(3),
                              np.array([0.05,0,0]).T),
                      pin.FrameType.OP_FRAME)
        )

        # for i in range(self.reduced_robot.model.nframes):
        #     frame = self.reduced_robot.model.frames[i]
        #     frame_id = self.reduced_robot.model.getFrameId(frame.name)
        #     print(f"Frame ID: {frame_id}, Name: {frame.name}")
        
        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1) 
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Get the hand joint ID and define the error function
        self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")
        self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")

        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.L_hand_id].translation - self.cTf_l[:3,3],
                    self.cdata.oMf[self.R_hand_id].translation - self.cTf_r[:3,3]
                )
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.L_hand_id].rotation @ self.cTf_l[:3,:3].T),
                    cpin.log3(self.cdata.oMf[self.R_hand_id].rotation @ self.cTf_r[:3,:3].T)
                )
            ],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)   # for smooth
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        # Setting optimization constraints and goals
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit)
        )
        self.opti.minimize(50 * self.translational_cost + self.rotation_cost + 0.02 * self.regularization_cost + 0.1 * self.smooth_cost)

        opts = {
            'ipopt':{
                'print_level':0,
                'max_iter':50,
                'tol':1e-6
            },
            'print_time':False,# print or not
            'calc_lam_p':False # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 14)
        self.vis = None

        if self.Visualization:
            # Initialize the Meshcat visualizer for visualization
            self.vis = MeshcatVisualizer(self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model)
            self.vis.initViewer(open=True) 
            self.vis.loadViewerModel("pinocchio") 
            self.vis.displayFrames(True, frame_ids=[101, 102], axis_length = 0.15, axis_width = 5)
            self.vis.display(pin.neutral(self.reduced_robot.model))

            # Enable the display of end effector target frames with short axis lengths and greater width.
            frame_viz_names = ['L_ee_target', 'R_ee_target']
            FRAME_AXIS_POSITIONS = (
                np.array([[0, 0, 0], [1, 0, 0],
                          [0, 0, 0], [0, 1, 0],
                          [0, 0, 0], [0, 0, 1]]).astype(np.float32).T
            )
            FRAME_AXIS_COLORS = (
                np.array([[1, 0, 0], [1, 0.6, 0],
                          [0, 1, 0], [0.6, 1, 0],
                          [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T
            )
            axis_length = 0.1
            axis_width = 10
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
    


    def solve_ik(self, left_wrist, right_wrist, current_lr_arm_motor_q = None, current_lr_arm_motor_dq = None):
        if current_lr_arm_motor_q is not None:
            self.init_data = current_lr_arm_motor_q
        self.opti.set_initial(self.var_q, self.init_data)

        if self.Visualization:
            self.vis.viewer['L_ee_target'].set_transform(left_wrist)   # for visualization
            self.vis.viewer['R_ee_target'].set_transform(right_wrist)  # for visualization

        self.opti.set_value(self.param_tf_l, left_wrist)
        self.opti.set_value(self.param_tf_r, right_wrist)
        self.opti.set_value(self.var_q_last, self.init_data) # for smooth

        try:
            sol = self.opti.solve()
            # sol = self.opti.solve_limited()

            sol_q = self.opti.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            return sol_q, sol_tauff
        
        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")

            sol_q = self.opti.debug.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            # print(f"sol_q:{sol_q} \nmotorstate: \n{current_lr_arm_motor_q} \nleft_pose: \n{left_wrist} \nright_pose: \n{right_wrist}")
            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            # return sol_q, sol_tauff
            return current_lr_arm_motor_q, np.zeros(self.reduced_robot.model.nv)

class Camera_IK:
    def __init__(self, visualize):
        # urdf_path = "../assets/robots/gr1/urdf/robot.urdf"
        urdf_path = "../assets/GRX/GR1/GR1T2/urdf/GR1T2_inspire_hand.urdf"
        self.arm_ik = H1_2_ArmIK(Visualization=visualize)

        # retargeting_config_path = "../assets/inspire_hand_gr1t2/inspire_hand.yml"
        retargeting_config_path = "../assets/dex_inspire_hand/inspire_hand.yml"
        RetargetingConfig.set_default_urdf_dir(os.path.dirname(retargeting_config_path))
        with Path(retargeting_config_path).open('r') as f:
            cfg = yaml.safe_load(f)

        self.left_retargeting = RetargetingConfig.from_dict(cfg['left']).build()
        self.right_retargeting = RetargetingConfig.from_dict(cfg['right']).build()
        self.tip_indices = [4,8,12,16,20]

        # 添加卡尔曼滤波
        initial_q = np.zeros(14)
        dt = 1
        self.kalman_filter = KalmanFilterSmoother(initial_state=initial_q, dt=dt)

        self.previous_left_qpos = np.zeros(12)
        self.previous_right_qpos = np.zeros(12)

    def ik_fun(self,data):

        target_joint_positions = np.zeros(38)
        head_mat = np.eye(4)
        sol_q, success = self.arm_ik.solve_ik(data[0], data[1])

        # 卡尔曼滤波平滑一下
        sol_q = self.kalman_filter.update(sol_q)

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

        target_joint_positions[0:7] = sol_q[0:7] # 7
        target_joint_positions[7:19] = left_qpos
        target_joint_positions[19:26] = sol_q[7:14] # 7
        target_joint_positions[26:38] = right_qpos

        return target_joint_positions