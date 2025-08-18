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

# import ipdb
current_dir = os.path.dirname(os.path.abspath(__file__))
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial.transform import Rotation as sRot

from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES, SMPLH_BONE_ORDER_NAMES, SMPLH_MUJOCO_NAMES
from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
)

from torch.autograd import Variable
from tqdm import tqdm

SMPLX_BONE_ORDER_NAMES = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Torso",
    "L_Knee",
    "R_Knee",
    "Spine",
    "L_Ankle",
    "R_Ankle",
    "Chest",
    "L_Toe",
    "R_Toe",
    "Neck",
    "L_Thorax",
    "R_Thorax",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "1",
    "2",
    "3",
    "L_Index1",
    "L_Index2",
    "L_Index3",
    "L_Middle1",
    "L_Middle2",
    "L_Middle3",
    "L_Pinky1",
    "L_Pinky2",
    "L_Pinky3",
    "L_Ring1",
    "L_Ring2",
    "L_Ring3",
    "L_Thumb1",
    "L_Thumb2",
    "L_Thumb3",
    "R_Index1",
    "R_Index2",
    "R_Index3",
    "R_Middle1",
    "R_Middle2",
    "R_Middle3",
    "R_Pinky1",
    "R_Pinky2",
    "R_Pinky3",
    "R_Ring1",
    "R_Ring2",
    "R_Ring3",
    "R_Thumb1",
    "R_Thumb2",
    "R_Thumb3",
]


def visualize_joint_positions(joint_positions):
    """
    可视化机器人关节位置的3D散点图

    Args:
        joint_positions (np.ndarray): 形状为(n,3)的关节位置数组
    """
    # 创建3D图形
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # 提取x、y、z坐标
    x = joint_positions[:, 0]
    y = joint_positions[:, 1]
    z = joint_positions[:, 2]

    # 绘制散点图
    ax.scatter(x, y, z, c="b", marker="o")

    # 找到所有坐标的最大范围
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5

    # 设置相同的坐标轴范围
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # 设置坐标轴标签
    ax.set_xlabel("X轴")
    ax.set_ylabel("Y轴")
    ax.set_zlabel("Z轴")

    # 设置标题
    ax.set_title("机器人关节位置可视化")

    # 添加网格
    ax.grid(True)

    # 设置坐标轴比例相等
    ax.set_box_aspect([1, 1, 1])

    plt.show()


# 已经将机器人转换到t_pose,for gr1t2
def Tpose_robot_joint_positions(robot_type="g1", robot_key_joints=None,print_all=False):
    """
    加载机器人模型并打印所有关节的位置信息

    Args:
        urdf_path (str): URDF文件的路径
        mesh_path (str): 网格文件目录的路径
    """
    if robot_type == "g1":
        urdf_path = "../../assets/g1/g1_body29_hand14.urdf"
        mesh_path=["../../assets/g1"]
    elif robot_type == "gr1t2":
        urdf_path = "../../assets/GRX/GR1/GR1T2/urdf/GR1T2_inspire_hand.urdf"
        mesh_path=["../../assets/GRX/GR1/GR1T2/urdf"]
    elif robot_type == "h1_2":
        urdf_path = '../../assets/h1_2/h1_2.urdf'
        mesh_path=['../../assets/h1_2/']

    # 加载机器人模型
    robot = pin.RobotWrapper.BuildFromURDF(urdf_path, mesh_path)

    # 打印所有关节的名称
    if print_all:
        print("机器人所有关节名称：")
        for i in range(robot.model.njoints):
            print(f"{robot.model.names[i]}")


    joint_angles = np.zeros(robot.model.nq)

    # 不同的机器人需要旋转的角度不同
    if robot_type == "g1":
        left_shoulder_roll_idx = robot.model.getJointId("left_shoulder_roll_joint")
        joint_angles[left_shoulder_roll_idx-1] = np.pi/2  

        left_elbow_joint_idx = robot.model.getJointId("left_elbow_joint")
        joint_angles[left_elbow_joint_idx-1] = np.pi/2 

        right_shoulder_roll_idx = robot.model.getJointId("right_shoulder_roll_joint")
        joint_angles[right_shoulder_roll_idx-1] = -np.pi/2  

        right_elbow_joint_idx = robot.model.getJointId("right_elbow_joint")
        joint_angles[right_elbow_joint_idx-1] = np.pi/2 

    elif robot_type == "gr1t2":

        left_shoulder_roll_joint_idx = robot.model.getJointId("left_shoulder_roll_joint")
        joint_angles[left_shoulder_roll_joint_idx-1] = np.pi/2  

        right_shoulder_roll_joint_idx = robot.model.getJointId("right_shoulder_roll_joint")
        joint_angles[right_shoulder_roll_joint_idx-1] = -np.pi/2 

    elif robot_type == "h1_2":
        left_shoulder_roll_idx = robot.model.getJointId("left_shoulder_roll_joint")
        joint_angles[left_shoulder_roll_idx-1] = np.pi/2 

        left_elbow_joint_idx = robot.model.getJointId("left_elbow_pitch_joint")
        joint_angles[left_elbow_joint_idx-1] = np.pi/2 

        right_shoulder_roll_idx = robot.model.getJointId("right_shoulder_roll_joint")
        joint_angles[right_shoulder_roll_idx-1] = -np.pi/2 

        right_elbow_joint_idx = robot.model.getJointId("right_elbow_pitch_joint")
        joint_angles[right_elbow_joint_idx-1] = np.pi/2 


    # 更新机器人状态
    pin.forwardKinematics(robot.model, robot.data, joint_angles)
    pin.updateFramePlacements(robot.model, robot.data)

    # 输出每个关节的位置信息
    if print_all:
        print("所有关节位置：")
        for i in range(robot.model.njoints):
            joint_name = robot.model.names[i]
            joint_id = robot.model.getJointId(joint_name)
            joint_position = robot.data.oMi[joint_id].translation
            print(f"{joint_name}: {joint_position}")

    if robot_key_joints is None:
        # 直接获取所有关节位置
        joint_positions = np.array([oMi.translation for oMi in robot.data.oMi])
    else:
        # 只获取指定关键关节的位置
        joint_positions = []
        for joint_name in robot_key_joints:
            # print(joint_name)
            joint_id = robot.model.getJointId(joint_name)
            joint_position = robot.data.oMi[joint_id].translation
            joint_positions.append(joint_position)


        joint_positions = np.array(joint_positions)

    return joint_positions

def Tpose_smpl_joint_positions(beta=None, scale=None, smpl_key_joints=None, smpl_init=None, device="cpu"):
    if smpl_init is None:
        smpl_init = {"Pelvis": "[np.pi/2, 0, np.pi/2]"}
    pose_aa_stand = np.zeros((1, 156))
    pose_aa_stand = pose_aa_stand.reshape(-1, 52, 3)  # 所以这里其实是指旋转

    for init_key, init_value in smpl_init.items():
        pose_aa_stand[:, SMPLH_BONE_ORDER_NAMES.index(init_key)] = sRot.from_euler("xyz", eval(init_value), degrees=False).as_rotvec()

    # 下面是smpl的可视化
    pose_aa_stand = torch.from_numpy(pose_aa_stand.reshape(-1, 156)).to(device)
    # print(pose_aa_stand.shape)
    # smpl_parser_x  = SMPLX_Parser(model_path="data/smpl", gender="neutral",ext="pkl")
    smpl_parser_x = SMPLX_Parser(
        model_path="../../data/smpl",
        gender="neutral",
        use_pca=False,
        create_transl=False,
        flat_hand_mean=True,
        num_betas=20,
        device=device,
    ).to(device)

    trans = torch.zeros([1, 3], device=device)
    if scale == None and beta == None:
        beta = torch.zeros([1, 20], device=device)  # 添加device
        scale = torch.ones([1], device=device)  # 添加device

    verts, joints = smpl_parser_x.get_joints_verts(pose_aa_stand, beta, trans)

    if smpl_key_joints is not None:
        # 创建索引张量
        indices = [SMPLX_BONE_ORDER_NAMES.index(name) for name in smpl_key_joints]
        # 直接使用索引获取选定关节的位置
        joints = joints[:, indices]

    # 将其放入到原点
    root_pos = joints[:, 0]
    joints = (joints - joints[:, 0]) * scale + root_pos

    return joints

# 简单
def visualize_smpl_robot(beta,scale,robot_type,smpl_key_joints=None, robot_key_joints=None):

    smpl_joint_positions = Tpose_smpl_joint_positions(beta=beta, scale=scale, smpl_key_joints=smpl_key_joints)
    smpl_joint_positions = smpl_joint_positions[0].detach().cpu().clone()
    robot_joint_positions = Tpose_robot_joint_positions(robot_type=robot_type, robot_key_joints=robot_key_joints)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(90, 0)
    ax.scatter(smpl_joint_positions[:, 0], smpl_joint_positions[:, 1], smpl_joint_positions[:, 2], label="Humanoid Shape", c="blue")
    ax.scatter(robot_joint_positions[:, 0], robot_joint_positions[:, 1], robot_joint_positions[:, 2], label="Fitted Shape", c="red")

    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    drange = 1
    ax.set_xlim(-drange, drange)
    ax.set_ylim(-drange, drange)
    ax.set_zlim(-drange, drange)
    ax.legend()
    plt.show()

# 写一个meshcat 跟着不同的solq可视化的code
class Visulize_solq:

    def __init__(self, urdf_path=None, mesh_dir=None,robot_type="g1"):
        print(urdf_path)
        self.robot = pin.RobotWrapper.BuildFromURDF(
            urdf_path,
            package_dirs=mesh_dir,
        )

        if robot_type == "g1":
            # for g1
            self.mixed_jointsToLockIDs = [
                "left_hip_pitch_joint",
                "left_hip_roll_joint",
                "left_hip_yaw_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
                "right_hip_pitch_joint",
                "right_hip_roll_joint",
                "right_hip_yaw_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
                "waist_yaw_joint",
                'waist_pitch_joint',
                'waist_roll_joint',
                #  "left_shoulder_pitch_joint",
                #  "left_shoulder_roll_joint",
                #  "left_shoulder_yaw_joint",
                #  "left_elbow_joint",
                #  "left_wrist_roll_joint",
                #  "left_wrist_pitch_joint",
                #  "left_wrist_yaw_joint",
                #  "right_shoulder_pitch_joint",
                #  "right_shoulder_roll_joint",
                #  "right_shoulder_yaw_joint",
                #  "right_elbow_joint",
                #  "right_wrist_roll_joint",
                #  "right_wrist_pitch_joint",
                #  "right_wrist_yaw_joint",
            ]
        elif robot_type == "gr1t2":
        # for gr1t2
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
                # upper was locked
                # "head_roll_joint",
                # "head_pitch_joint",
                # "head_yaw_joint",
                # "left_shoulder_pitch_joint",
                # "left_shoulder_roll_joint",
                # "left_shoulder_yaw_joint",
                # "left_elbow_pitch_joint",
                # "left_wrist_yaw_joint",
                # "left_wrist_roll_joint",
                # "left_wrist_pitch_joint",
                # "L_index_proximal_joint",  #10
                # "L_index_intermediate_joint",
                # "L_middle_proximal_joint",
                # "L_middle_intermediate_joint",
                # "L_pinky_proximal_joint",
                # "L_pinky_intermediate_joint",
                # "L_ring_proximal_joint",
                # "L_ring_intermediate_joint",
                # "L_thumb_proximal_yaw_joint",
                # "L_thumb_proximal_pitch_joint",
                # "L_thumb_intermediate_joint",
                # "L_thumb_distal_joint",   # 21
                # "right_shoulder_pitch_joint",
                # "right_shoulder_roll_joint",
                # "right_shoulder_yaw_joint",
                # "right_elbow_pitch_joint",
                # "right_wrist_yaw_joint",
                # "right_wrist_roll_joint",
                # "right_wrist_pitch_joint",
                # "R_index_proximal_joint", # 29
                # "R_index_intermediate_joint",
                # "R_middle_proximal_joint",
                # "R_middle_intermediate_joint",
                # "R_pinky_proximal_joint",
                # "R_pinky_intermediate_joint",
                # "R_ring_proximal_joint",
                # "R_ring_intermediate_joint",
                # "R_thumb_proximal_yaw_joint",
                # "R_thumb_proximal_pitch_joint",
                # "R_thumb_intermediate_joint",
                # "R_thumb_distal_joint",
            ]
        elif robot_type == "h1_2":
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
                                        # "L_index_proximal_joint",
                                        # "L_index_intermediate_joint",
                                        # "L_middle_proximal_joint",
                                        # "L_middle_intermediate_joint",
                                        # "L_pinky_proximal_joint",
                                        # "L_pinky_intermediate_joint",
                                        # "L_ring_proximal_joint",
                                        # "L_ring_intermediate_joint",
                                        # "L_thumb_proximal_yaw_joint",
                                        # "L_thumb_proximal_pitch_joint",
                                        # "L_thumb_intermediate_joint",
                                        # "L_thumb_distal_joint",
                                        # "right_shoulder_pitch_joint",
                                        # "right_shoulder_roll_joint",
                                        # "right_shoulder_yaw_joint",
                                        # "right_elbow_joint",
                                        # "right_wrist_roll_joint",
                                        # "right_wrist_pitch_joint",
                                        # "right_wrist_yaw_joint",
                                        # "R_index_proximal_joint",
                                        # "R_index_intermediate_joint",
                                        # "R_middle_proximal_joint",
                                        # "R_middle_intermediate_joint",
                                        # "R_pinky_proximal_joint",
                                        # "R_pinky_intermediate_joint",
                                        # "R_ring_proximal_joint",
                                        # "R_ring_intermediate_joint",
                                        # "R_thumb_proximal_yaw_joint",
                                        # "R_thumb_proximal_pitch_joint",
                                        # "R_thumb_intermediate_joint",
                                        # "R_thumb_distal_joint"
                                        ]

        # 简化机器人模型,锁定了下半身关节
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )  # refrerence_configuration 为锁定关节的默认位置

        self.vis = MeshcatVisualizer(
            self.reduced_robot.model,
            self.reduced_robot.collision_model,
            self.reduced_robot.visual_model,
        )

        self.vis.initViewer(open=True)
        self.vis.loadViewerModel("pinocchio")

    def __call__(self, q):
        # print(len(q))
        self.vis.display(q)

# 得到优化以后的beta
def optimize_beta(robot_type="g1", robot_key_joints=None, smpl_key_joints=None):


    if robot_type == "g1":
        urdf_abs_path = "../../assets/g1/g1_body29_hand14.urdf"
        package_dirs = ["../../assets/g1"]
    elif robot_type == "gr1t2":
        urdf_abs_path = "../../assets/GRX/GR1/GR1T2/urdf/GR1T2_inspire_hand.urdf"
        package_dirs = ["../../assets/GRX/GR1/GR1T2/urdf"]
    elif robot_type == "h1_2":
        urdf_abs_path = '../../assets/h1_2/h1_2.urdf'
        package_dirs = ['../../assets/h1_2/']


    device = "cuda"
    robot_joint_positions = Tpose_robot_joint_positions(robot_type=robot_type,robot_key_joints=robot_key_joints)
    robot_joint_positions = torch.from_numpy(robot_joint_positions).to(device).unsqueeze(0)

    # 现在不需要优化scale,scale用其他方式优化
    shape_new = Variable(torch.zeros([1, 20]).to(device), requires_grad=True)
    scale = torch.ones([1]).to(device)  # 这里初始化scale为1

    # scale = Variable(torch.ones([1]).to(device), requires_grad=True)  # 这里初始化scale为1
    # optimizer_shape = torch.optim.Adam([shape_new, scale], lr=0.1)
    optimizer_shape = torch.optim.Adam([shape_new], lr=0.1)

    train_iterations = 5000
    print("start fitting shapes")
    pbar = tqdm(range(train_iterations))
    for iteration in pbar:

        # 待优化
        smpl_joint_positions = Tpose_smpl_joint_positions(beta=shape_new, scale=scale, smpl_key_joints=smpl_key_joints, device=device)

        diff = smpl_joint_positions - robot_joint_positions

        # loss_g = diff.norm(dim = -1).mean()
        loss_g = diff.norm(dim=-1).square().sum()

        loss = loss_g
        pbar.set_description_str(f"{iteration} - Loss: {loss.item() * 1000}")

        optimizer_shape.zero_grad()
        loss.backward()
        optimizer_shape.step()

    # print the fitted shape and scale parameters
    print("shape:", shape_new.detach())
    return shape_new.detach()

def calculate_joint_distances(joint_positions):
    """
    计算肩部、肘部、手腕之间的距离
    :param joint_positions: 关节位置数组，形状为 (N, 3)
    :return: 各个距离的字典
    """
    distances = {}
    distances['z_distance'] = joint_positions[1, 2] - joint_positions[0, 2]
    # 计算 y 和 z 轴的距离
    distances['y_distance'] = joint_positions[1, 1] - joint_positions[0, 1]
    # 计算肩部到肘部的距离
    distances['shoulder_elbow_distance'] = np.linalg.norm(joint_positions[1] - joint_positions[3])
    # 计算肘部到手腕的距离
    distances['elbow_wrist_distance'] = np.linalg.norm(joint_positions[3] - joint_positions[5])

    return distances

def calculate_scale_list(robot_type, robot_key_joints, smpl_key_joints, beta, scale):
    # 获取机器人关节位置
    joint_positions = Tpose_robot_joint_positions(robot_type=robot_type, robot_key_joints=robot_key_joints, print_all=False)
    # 获取对应beta下的SMPL关节位置
    smpl_positions = Tpose_smpl_joint_positions(beta=beta, scale=scale, smpl_key_joints=smpl_key_joints)[0].detach().cpu().clone().numpy()
    
    # 计算各自的关节距离
    smpl_distance = calculate_joint_distances(smpl_positions)
    distance = calculate_joint_distances(joint_positions)
    
    # 计算scale_list
    scale_list = [distance["z_distance"]]
    keys = ["y_distance", "shoulder_elbow_distance", "elbow_wrist_distance"]
    
    for key in keys:
        scale_list.append(distance[key] / smpl_distance[key])
    
    return scale_list

def scale_joints_ori(joints,scale_list=None,robot_type="g1"):
    # 左手
    # g1
    if robot_type == "g1" and scale_list is None:
        scale_list = [0.10022/0.2472068965435028, 0.224639496822895/0.29666388034820557, 0.18428120957972927/0.31173112988471985]
        scale_left_joints_list = [[0,0,0.29178]]
        scale_right_joints_list = [[0,0,0.29178]]
        # joints[16][2] = 0.29178
        # joints[17][2] = 0.29178
    # gr1t2
    elif robot_type == "gr1t2" and scale_list is None:
        scale_list = [0.1877331147298219/0.2472068965435028, 0.2528544583947256/0.29666388034820557, 0.25060000000000004/0.31173112988471985]
        scale_left_joints_list = [[0,0,0.4148039488176755]]
        scale_right_joints_list = [[0,0,0.4148039488176755]]
        # joints[16][2] = 0.4148039488176755
        # joints[17][2] = 0.4148039488176755
    elif robot_type == "h1_2" and scale_list is None:
        scale_list = [0.14806/0.2472068965435028, 0.39147126287809864/0.29666388034820557, 0.22864691119715566/0.31173112988471985]
        scale_left_joints_list = [[0,0,0.42333]]
        scale_right_joints_list = [[0,0,0.42333]]
    # y_distance: 0.14806, z_distance: 0.42333, shouler_ebow_distance: 0.39147126287809864, elbow_wrist_distance: 0.22864691119715566
        # joints[16][2] = 0.42333
        # joints[17][2] = 0.42333
    direction1 = joints[16] - scale_left_joints_list[0] # left_shoulder
    direction1[2] = 0
    direction1[0] = 0
    left_shoulder_joint = scale_left_joints_list[-1] + direction1 * scale_list[0]
    scale_left_joints_list.append(left_shoulder_joint)
    direction2 = joints[18] - joints[16] # left_shoulder -> left_elbow
    left_elbow_joint = scale_left_joints_list[-1] + direction2 * scale_list[1]
    scale_left_joints_list.append(left_elbow_joint)
    direction3 = joints[20] - joints[18] # left_elbow -> left_wrist
    left_wrist_joint = scale_left_joints_list[-1] + direction3 * scale_list[2]
    scale_left_joints_list.append(left_wrist_joint)

    # 先处理手指
    joints[25:40] = joints[25:40]-joints[20]
    joints[16] = scale_left_joints_list[1]
    joints[18] = scale_left_joints_list[2]
    joints[20] = scale_left_joints_list[3]
    joints[25:40] = joints[25:40] + joints[20]


    # 右手
    # scale_list = [0.10022/0.2472068965435028, 0.224639496822895/0.29666388034820557, 0.18428120957972927/0.31173112988471985]
    direction1 = joints[17] - scale_right_joints_list[0] # right_shoulder
    direction1[2] = 0
    direction1[0] = 0
    right_shoulder_joint = scale_right_joints_list[-1] + direction1 * scale_list[0]
    scale_right_joints_list.append(right_shoulder_joint)
    direction2 = joints[19] - joints[17] # right_shoulder -> right_elbow
    right_elbow_joint = scale_right_joints_list[-1] + direction2 * scale_list[1]
    scale_right_joints_list.append(right_elbow_joint)
    direction3 = joints[21] - joints[19] # right_elbow -> right_wrist
    right_wrist_joint = scale_right_joints_list[-1] + direction3 * scale_list[2]
    scale_right_joints_list.append(right_wrist_joint)

    # 先处理手指
    joints[40:55] = joints[40:55]-joints[21]
    joints[17] = scale_right_joints_list[1]
    joints[19] = scale_right_joints_list[2]
    joints[21] = scale_right_joints_list[3]
    joints[40:55] = joints[40:55] + joints[21]

    return joints


def scale_joints(joints,robot_type="g1"):
    # 直接从shoulder 与 Pelvis 的交点出发 然后是各个link的ratio

    if robot_type == "g1":
        scale_list = [0.29178, 0.9530646994781607, 1.0209423038821641, 0.9934842306486995]
        scale_left_joints_list = [[0,0,scale_list[0]]]
        scale_right_joints_list = [[0,0,scale_list[0]]]
        scale_list = scale_list[1:]
    elif robot_type == "gr1t2":
        scale_list = [0.4148039488176755, 0.9426624260477016, 1.0560934022512718, 0.9853709280947656]
        scale_left_joints_list = [[0,0,scale_list[0]]]
        scale_right_joints_list = [[0,0,scale_list[0]]]
        scale_list = scale_list[1:]
    elif robot_type == "h1_2":
        scale_list = [0.42333, 0.9993858935507919, 1.0076461701629893, 0.9891067432243611]
        scale_left_joints_list = [[0,0,scale_list[0]]]
        scale_right_joints_list = [[0,0,scale_list[0]]]
        scale_list = scale_list[1:]

    direction1 = joints[16] - scale_left_joints_list[0] # left_shoulder
    direction1[2] = 0
    direction1[0] = 0
    left_shoulder_joint = scale_left_joints_list[-1] + direction1 * scale_list[0]
    scale_left_joints_list.append(left_shoulder_joint)
    direction2 = joints[18] - joints[16] # left_shoulder -> left_elbow
    left_elbow_joint = scale_left_joints_list[-1] + direction2 * scale_list[1]
    scale_left_joints_list.append(left_elbow_joint)
    direction3 = joints[20] - joints[18] # left_elbow -> left_wrist
    left_wrist_joint = scale_left_joints_list[-1] + direction3 * scale_list[2]
    scale_left_joints_list.append(left_wrist_joint)

    # 左手
    joints[16] = scale_left_joints_list[1]
    joints[18] = scale_left_joints_list[2]
    joints[20] = scale_left_joints_list[3]


    # 右手
    direction1 = joints[17] - scale_right_joints_list[0] # right_shoulder
    direction1[2] = 0
    direction1[0] = 0
    right_shoulder_joint = scale_right_joints_list[-1] + direction1 * scale_list[0]
    scale_right_joints_list.append(right_shoulder_joint)
    direction2 = joints[19] - joints[17] # right_shoulder -> right_elbow
    right_elbow_joint = scale_right_joints_list[-1] + direction2 * scale_list[1]
    scale_right_joints_list.append(right_elbow_joint)
    direction3 = joints[21] - joints[19] # right_elbow -> right_wrist
    right_wrist_joint = scale_right_joints_list[-1] + direction3 * scale_list[2]
    scale_right_joints_list.append(right_wrist_joint)


    joints[17] = scale_right_joints_list[1]
    joints[19] = scale_right_joints_list[2]
    joints[21] = scale_right_joints_list[3]

    return joints




if __name__ == "__main__":

    # 测试是否为T_pose
    # g1_joint_positions = Tpose_robot_joint_positions(robot_type="g1", robot_key_joints=None, print_all=False)
    # gr1t2_joint_positions = Tpose_robot_joint_positions(robot_type="gr1t2", robot_key_joints=None, print_all=False)
    # h1_2_joint_positions = Tpose_robot_joint_positions(robot_type="h1_2", robot_key_joints=None, print_all=False)
    # visualize_joint_positions(g1_joint_positions)
    # visualize_joint_positions(gr1t2_joint_positions)  
    # visualize_joint_positions(h1_2_joint_positions)

    # 优化beta
    smpl_key_joints = [
        "Pelvis", # 0 
        # "Neck",
        "L_Shoulder", # 1
        "R_Shoulder", # 2
        "L_Elbow", # 3
        "R_Elbow", # 4
        "L_Wrist", # 5
        "R_Wrist", # 6
    ]
    robot_key_joints_gr1t2 = [
        "universe",
        # "head_pitch_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "left_elbow_pitch_joint",
        "right_elbow_pitch_joint",
        "left_wrist_pitch_joint",
        "right_wrist_pitch_joint",
    ]

    robot_key_joints_g1 = [
        "universe",
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_elbow_joint",
        "right_elbow_joint",
        "left_wrist_yaw_joint",
        "right_wrist_yaw_joint",
    ]

    robot_key_joints_h1_2 = [
        "universe",
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_elbow_pitch_joint",
        "right_elbow_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_yaw_joint",
    ]

    # g1_beta = optimize_beta(robot_type="g1", robot_key_joints=robot_key_joints_g1, smpl_key_joints=smpl_key_joints)
    # 获得g1 beta
    g1_beta = torch.tensor([[ -4.9129,  -4.5723, -19.7881,  28.6458, -36.9779,  -7.0731,   5.2939,
            -3.5626, -12.6291, -61.8798,  17.2138, -19.0186, -15.0479,  13.8146,
            -4.4299, -15.1081,  14.6959, -19.5210,  -2.8372, -22.3335]])
    
    # gr1t2_beta = optimize_beta(robot_type="gr1t2", robot_key_joints=robot_key_joints_gr1t2, smpl_key_joints=smpl_key_joints)
    # 获得gr1t2 beta
    gr1t2_beta = torch.tensor([[-2.2209, 0.1585, -29.5929, 37.2222, -55.1805, -18.0210, 5.9216, 
                                5.6700, -13.7881, -64.6534, 25.1731, -36.0609, -25.4610, 14.0997, 
                                -5.0708, -19.2019, 42.7658, -14.8146, -1.6842, -41.553]])

    # 获得h1_2 beta
    # h1_2_beta = optimize_beta(robot_type="h1_2", robot_key_joints=robot_key_joints_h1_2, smpl_key_joints=smpl_key_joints)
    h1_2_beta = torch.tensor([[ 0.2073,  -21.8104,  -34.4264,   58.8596,  -37.0272,  -36.6192,
                                4.2660,   -6.1962,   -7.6354, -189.1553,   15.9884,   51.2717,
                                35.2791,   94.5679,  -45.7144,  -40.6734, -139.4461,   47.5057,
                                -85.6297,   19.0058]])


    # 可视化优化以后的beta与机器人的对比
    scale = torch.ones([1])
    # visualize_smpl_robot(g1_beta,scale,robot_type = "g1",smpl_key_joints=smpl_key_joints, robot_key_joints=robot_key_joints_g1)
    # visualize_smpl_robot(gr1t2_beta,scale,robot_type = "gr1t2",smpl_key_joints=smpl_key_joints, robot_key_joints=robot_key_joints_gr1t2)
    # visualize_smpl_robot(h1_2_beta,scale,robot_type = "h1_2",smpl_key_joints=smpl_key_joints, robot_key_joints=robot_key_joints_h1_2)

    # 需要返回每个机器人,因为左右是对称的,所以就只是返回左右就可以了
    # universe->left_shoulder 的y和z距离
    # left_shoulder->left_elbow的距离
    # left_elbow->left_wrist的距离
    g1_scale_list = calculate_scale_list("g1", robot_key_joints_g1, smpl_key_joints, g1_beta, scale)
    print(f"g1_scale_list: {g1_scale_list}")

    gr1t2_scale_list = calculate_scale_list("gr1t2", robot_key_joints_gr1t2, smpl_key_joints, gr1t2_beta, scale)
    print(f"gr1t2_scale_list: {gr1t2_scale_list}")

    h1_2_scale_list = calculate_scale_list("h1_2", robot_key_joints_h1_2, smpl_key_joints, h1_2_beta, scale)
    print(f"h1_2_scale_list: {h1_2_scale_list}")


    # g1_joint_positions = Tpose_robot_joint_positions(robot_type="g1", robot_key_joints=robot_key_joints_g1, print_all=False)
    # # g1_beta下的smpl关节
    # g1_smpl_positions = Tpose_smpl_joint_positions(beta=g1_beta, scale=scale, smpl_key_joints=smpl_key_joints)[0].detach().cpu().clone()
    # # 计算各自的距离
    # g1_smpl_distance = calculate_joint_distances(g1_smpl_positions)
    # g1_distance = calculate_joint_distances(g1_joint_positions)
    # # 然后获得scale_list
    # g1_scale_list = [g1_distance["z_distance"]]
    # keys = ["y_distance", "shoulder_elbow_distance", "elbow_wrist_distance"]
    # for key in keys:
    #     g1_scale_list.append(g1_distance[key] / g1_smpl_distance[key])
    # print(f"g1_scale_list: {g1_scale_list}")

    # gr1t2_joint_positions = Tpose_robot_joint_positions(robot_type="gr1t2", robot_key_joints=robot_key_joints_gr1t2, print_all=False)
    # # gr1t2_beta下的smpl关节
    # gr1t2_smpl_positions = Tpose_smpl_joint_positions(beta=gr1t2_beta, scale=scale, smpl_key_joints=smpl_key_joints)[0].detach().cpu().clone()
    # # 计算各自的距离
    # gr1t2_smpl_distance = calculate_joint_distances(gr1t2_smpl_positions)
    # gr1t2_distance = calculate_joint_distances(gr1t2_joint_positions)
    # # 然后获得scale_list
    # gr1t2_scale_list = [gr1t2_distance["z_distance"]]
    # keys = ["y_distance", "shoulder_elbow_distance", "elbow_wrist_distance"]
    # for key in keys:
    #     gr1t2_scale_list.append(gr1t2_distance[key] / gr1t2_smpl_distance[key])
    # print(f"gr1t2_scale_list: {g1_scale_list}")

    # h1_2_joint_positions = Tpose_robot_joint_positions(robot_type="h1_2", robot_key_joints=robot_key_joints_h1_2, print_all=False)
    # # h1_2_beta下的smpl关节
    # h1_2_smpl_positions = Tpose_smpl_joint_positions(beta=h1_2_beta, scale=scale, smpl_key_joints=smpl_key_joints)[0].detach().cpu().clone()
    # # 计算各自的距离
    # h1_2_smpl_distance = calculate_joint_distances(h1_2_smpl_positions)
    # h1_2_distance = calculate_joint_distances(h1_2_joint_positions)
    # # 然后获得scale_list
    # h1_2_scale_list = [h1_2_distance["z_distance"]]
    # keys = ["y_distance", "shoulder_elbow_distance", "elbow_wrist_distance"]
    # for key in keys:
    #     h1_2_scale_list.append(h1_2_distance[key] / h1_2_smpl_distance[key])
    # print(f"h1_2_scale_list: {g1_scale_list}")


 