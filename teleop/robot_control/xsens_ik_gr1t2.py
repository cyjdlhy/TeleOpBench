import meshcat.geometry as mg
import numpy as np
import pinocchio as pin
from pinocchio import BODY
from pinocchio.visualize import MeshcatVisualizer
from loop_rate_limiters import RateLimiter

import pink
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask

import os
import meshcat.geometry as mg
import qpsolvers


class Xsense_IK:
    def __init__(self, visualize):
        np.set_printoptions(precision=6, suppress=True, linewidth=300)
        urdf_abs_path = "../assets/GRX/GR1/GR1T2/urdf/GR1T2_inspire_hand.urdf"
        self.robot = pin.RobotWrapper.BuildFromURDF(
            urdf_abs_path,
            package_dirs=[
                "../assets/GRX/GR1/GR1T2",
                "../assets/GRX/GR1/GR1T2/meshes",
            ],
        )
        self.visualize = visualize
        # set fixed joint; in current setting, lower part / waist / head are fixed
        self.mixed_jointsToLockIDs = [
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_hip_pitch_joint",
            "left_knee_pitch_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_hip_pitch_joint",
            "right_knee_pitch_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_pitch_joint",
            "waist_roll_joint",
        ]

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )
        self.curr_position = np.zeros((42, 3))
        # # Initialize the Meshcat visualizer
        if self.visualize:
            self.vis = MeshcatVisualizer(
                self.reduced_robot.model,
                self.reduced_robot.collision_model,
                self.reduced_robot.visual_model,
            )
            self.reduced_robot.setVisualizer(self.vis, init=False)
            self.vis.initViewer(open=True)
            self.vis.loadViewerModel("pinocchio")

        self.config = pink.Configuration(
            self.reduced_robot.model, self.reduced_robot.data, self.reduced_robot.q0
        )
        self.link_names = [
            frame.name
            for frame in self.reduced_robot.model.frames
            if frame.type == BODY
        ]
        self.q_max, self.q_min = self.make_joint_config(
            self.config, 0.95
        )  # train policy with max=0.95

        # # 上肢肢体
        self.left_hand_index = self.reduced_robot.model.getFrameId(
            "left_wrist_roll_joint"
        )
        self.right_hand_index = self.reduced_robot.model.getFrameId(
            "right_wrist_roll_joint"
        )
        self.left_forearm_index = self.reduced_robot.model.getFrameId(
            "left_elbow_pitch_joint"
        )
        self.right_forearm_index = self.reduced_robot.model.getFrameId(
            "right_elbow_pitch_joint"
        )
        self.left_upper_arm_index = self.reduced_robot.model.getFrameId(
            "left_shoulder_roll_joint"
        )
        self.right_upper_arm_index = self.reduced_robot.model.getFrameId(
            "right_shoulder_roll_joint"
        )

        self.body_index = self.reduced_robot.model.getFrameId("torso_link")

        if self.visualize:
            self.vis.displayFrames(
                True,
                frame_ids=[
                    self.left_hand_index,
                    self.right_hand_index,
                    self.left_forearm_index,
                    self.right_forearm_index,
                    self.left_upper_arm_index,
                    self.right_upper_arm_index,
                    self.body_index,
                ],
            )
            self.vis.display(pin.neutral(self.reduced_robot.model))
            # Enable the display of end effector target frames with short axis lengths and greater width.
            frame_viz_names = [
                "L_hand_target",
                "R_hand_target",
                "L_forearm_target",
                "R_forearm_target",
                "L_arm_target",
                "R_arm_target",
            ]  # , 'body_target']
            FRAME_AXIS_POSITIONS = (
                np.array(
                    [[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]]
                )
                .astype(np.float32)
                .T
            )
            FRAME_AXIS_COLORS = (
                np.array(
                    [
                        [1, 0, 0],
                        [1, 0.6, 0],
                        [0, 1, 0],
                        [0.6, 1, 0],
                        [0, 0, 1],
                        [0, 0.6, 1],
                    ]
                )
                .astype(np.float32)
                .T
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

            self.vis.viewer["Origin"].set_object(
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
        self.default_pose_list = {
            # for G1 all upper default as 0
        }

        for item in self.default_pose_list.keys():
            self.default_pose[self.reduced_robot.model.getJointId(item) - 1] = (
                np.deg2rad(self.default_pose_list[item])
            )

        self.tasks = {}

        self.tasks["right_arm"] = FrameTask(
            frame="right_shoulder_roll_joint",
            position_cost=3.0,
            orientation_cost=0.6,
            lm_damping=1.0,
        )

        self.tasks["left_arm"] = FrameTask(
            frame="left_shoulder_roll_joint",
            position_cost=3.0,
            orientation_cost=0.6,
            lm_damping=1.0,
        )

        self.tasks["right_forearm"] = FrameTask(
            frame=("right_elbow_pitch_joint"),
            position_cost=3.0,
            orientation_cost=0.6,
            lm_damping=1.0,
        )

        self.tasks["left_forearm"] = FrameTask(
            frame=("left_elbow_pitch_joint"),
            position_cost=3.0,
            orientation_cost=0.6,
            lm_damping=1.0,
        )

        self.tasks["right_hand"] = FrameTask(
            frame="right_wrist_roll_joint",
            position_cost=4.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        self.tasks["left_hand"] = FrameTask(
            frame="left_wrist_roll_joint",
            position_cost=4.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )

        self.posture_task = PostureTask(cost=0.01, lm_damping=1.0, gain=0.1)
        self.posture_task.set_target(self.default_pose)

        # define QP-solver
        self.solver = qpsolvers.available_solvers[0]
        if "quadprog" in qpsolvers.available_solvers:
            solver = "quadprog"
        print(f"Using {solver} QP Solver")
        self.rate = RateLimiter(frequency=120.0)
        self.dt = self.rate.period

    def get_root_joint_dim(self, model):
        if model.existJointName("root_joint"):
            root_joint_id = model.getJointId("root_joint")
            root_joint = model.joints[root_joint_id]
            return root_joint.nq, root_joint.nv
        return 0, 0

    # def qpos_clamp(self):
    #     start, _ = self.get_root_joint_dim(self.config.model)
    #     end = self.config.model.nq
    #     qpos = self.config.q.copy()
    #     qpos[start:end] = np.clip(qpos[start:end], a_min=self.q_min[start:end], a_max=self.q_max[start:end])
    #     qpos.setflags(write=False)
    #     self.config.q = qpos

    # 限制关节角度范围
    def qpos_clamp(self):
        # nq 是机器人的关节数量
        # get_root_joint_dim 是一个方法，用于获取机器人根关节(root joint)的自由度(nq)和速度维度(nv)
        start, _ = self.get_root_joint_dim(self.config.model)  # 第一个关节的索引
        end = self.config.model.nq  # 最后一个关节的索引
        qpos = self.config.q.copy()  # 获取当前所有关节角度
        # print(len(qpos[0])
        # print(qpos)
        # 缩放手指的旋转

        qpos[start:end] = np.clip(
            qpos[start:end], a_min=self.q_min[start:end], a_max=self.q_max[start:end]
        )  # 限制关节角度范围
        # print(len(qpos[start:end]))

        qpos.setflags(write=False)  # 设置为只读
        self.config.q = qpos

    def make_joint_config(self, config, limit=0.95):
        q_max = config.model.upperPositionLimit
        q_min = config.model.lowerPositionLimit
        q_mean = (q_max + q_min) / 2
        q_scale = (q_max - q_min) * limit
        q_max = q_mean + q_scale / 2
        q_min = q_mean - q_scale / 2
        return q_max, q_min

    def array2SE3(self, pose):
        transform = pose[:3, :3]
        translation = pose[:3, 3]
        return pin.SE3(pin.SE3(transform, translation))

    def get_joint_positions(self, q):
        """
        计算给定关节角度下各关节的xyz坐标

        Args:
            q: 关节角度向量

        Returns:
            List[np.ndarray]: 各关节的xyz坐标列表
        """
        # 假设模型中有获取关节位置的功能
        # 如果您使用的是特定的机器人库(如RBDL, pinocchio等)，方法会有所不同
        joint_positions = []

        # 示例实现 - 需要根据您的具体机器人模型调整
        model = self.reduced_robot  # 假设有一个机器人模型对象
        for joint_id in range(model.get_n_joints()):
            # 计算关节位置 - 需要替换为您机器人模型中的实际方法
            position = model.compute_joint_position(q, joint_id)
            joint_positions.append(position)

        return joint_positions

    def ik_fun(self, data):
        (
            left_hand,
            right_hand,
            left_forearm,
            right_forearm,
            left_arm,
            right_arm,
        ) = data

        if self.visualize:
            self.vis.viewer["L_hand_target"].set_transform(left_hand)
            self.vis.viewer["R_hand_target"].set_transform(right_hand)
            self.vis.viewer["L_arm_target"].set_transform(left_arm)
            self.vis.viewer["R_arm_target"].set_transform(right_arm)
            self.vis.viewer["L_forearm_target"].set_transform(left_forearm)
            self.vis.viewer["R_forearm_target"].set_transform(right_forearm)
        origin_pose = np.eye(4)
        self.vis.viewer["Origin"].set_transform(origin_pose)

        self.tasks["left_hand"].set_target(self.array2SE3(left_hand))
        self.tasks["right_hand"].set_target(self.array2SE3(right_hand))
        self.tasks["left_arm"].set_target(self.array2SE3(left_arm))
        self.tasks["right_arm"].set_target(self.array2SE3(right_arm))
        self.tasks["left_forearm"].set_target(self.array2SE3(left_forearm))
        self.tasks["right_forearm"].set_target(self.array2SE3(right_forearm))

        velocity = solve_ik(
            self.config,
            list(self.tasks.values()) + [self.posture_task],
            self.dt,
            solver=self.solver,
        )

        self.config.integrate_inplace(velocity, self.dt)
        self.qpos_clamp()

        sol_q = self.config.q.copy()
        # print(sol_q)
        if self.visualize:
            self.vis.display(sol_q)

        return sol_q # target position; needed torques

        # except Exception as e:
        #     return np.zeros(self.reduced_robot.model.nq), False
