from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser, 
)
import torch
import numpy as np

from teleop.utils.smpl_utils import batch_rodrigues, get_global_rotation
from scipy.spatial.transform import Rotation as sRot
import socket
from threading import Thread
import struct
import select

import sys

sys.path.append("./")
sys.path.append("../")
from lcm_with_hand.xsens_gr1_lcmt import xsens_gr1_lcmt
import lcm
from teleop.utils.scale_utils import scale_joints

class CameraWrapper:
    # scale = 0.6546
    # trans = torch.zeros([1, 3])
    # beta = torch.tensor([[  2.8964,   3.2512,  -6.3317,   0.7466,   8.5855,   4.3732, -16.6232, -6.0283, -29.3488,  16.7490]])
    def __init__(self, scale, beta, trans = None,host="0.0.0.0", port=1234, timeout=3, segment_id=1):
        
        self.smpl_parser_x = SMPLX_Parser(model_path="../data/smpl", gender="neutral", use_pca=False, create_transl=False, flat_hand_mean = True, num_betas=20)
        self.scale = scale
        if trans == None:
            self.trans = torch.zeros([1, 3])
        self.beta = beta
        self.host = host
        self.port = port
        self.timeout = timeout
        self.segment_id = segment_id
        self.lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
        self.action = np.zeros(41)
        # Create TCP socket
        # self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.server_socket.bind((self.host, self.port))  # Bind the socket to host and port
        # # self.server_socket.listen(1)  # Listen for incoming connections
        # print(f"Server listening on {self.host}:{self.port}...")
        # self.server_socket.setblocking(False)
        self.running = False    # 线程运行标志
        self.receiver_thread = None  # Will store the receiver thread
        self.sock = None        # UDP套接字
        self.data = None
        self.pose_aa = np.zeros((64,3))
        # 接收pose_aa的数据

    def socket_receiver(self):
        """线程内部的接收循环"""
        while self.running:
            try:
                # 使用select检测可读事件，超时0.1秒
                ready, _, _ = select.select([self.sock], [], [], 0.1)
                if ready:
                    data, addr = self.sock.recvfrom(1024)
                    # 解析并存储数据
                    self.parse_position_packet(data)
                    self.process()
                    # self.data_array = np.frombuffer(data, dtype=np.float32).reshape(1, 152)
                    # print(f"Received data from {addr}")
            except Exception as e:
                print(f"Error receiving data: {e}")
                self.stop_receiving()  # 发生错误时自动停止

    def parse_position_packet(self, data):

        # 解析数据
        data_array = np.frombuffer(data, dtype=np.float32).reshape(64, 3)
        self.pose_aa = data_array
        # self.pose_aa = torch.from_numpy(data_array)

    # 开启接收数据
    def start_receiving(self):
        """启动接收线程"""
        if not self.running:
            self.running = True
            # 创建并配置UDP套接字
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind((self.host, self.port))
            self.sock.setblocking(False)  # 非阻塞模式
            # 启动线程
            self.receiver_thread = Thread(target=self.socket_receiver)
            self.receiver_thread.daemon = True  # 设为守护线程
            self.receiver_thread.start()
            print(f"UDP receiver started on {self.host}:{self.port}")

    def stop_receiving(self):
        # Set running flag to False to stop the thread
        self.running = False

        # Wait for the thread to finish
        if self.receiver_thread is not None:
            self.receiver_thread.join()

        # Close sockets
        self.client_socket.close()
        self.server_socket.close()

    def process(self):
        '''
        pose_aa, torch.tensor, shape: [1,52,3] 不包含 [22: 'jaw',23: 'left_eye',24: 'right_eye']
        这里我暂时将pose_aa和betas放到这里来处理，
        这里scales,trans和beta是需要逐优化来获得结果的
        '''
        
        pose_aa = self.pose_aa[:22].copy()
        left_hand = self.pose_aa[22:43].copy()
        right_hand = self.pose_aa[43:64].copy()

        # 将机器人的坐标系转换到gr1t2的坐标系
        smpl_axis = sRot.from_euler("xyz", [np.pi/2, 0, np.pi/2],  degrees = False).as_rotvec()
        pose_aa[0] = torch.tensor(smpl_axis)
        whole_pose_aa = torch.cat([torch.tensor(pose_aa), torch.zeros(52-22, 3)], dim=0)
        verts, joints = self.smpl_parser_x.get_joints_verts(whole_pose_aa, self.beta , self.trans)

        # 将机器人落到原点,并且进行缩放
        joints = joints.detach().clone().numpy()[0]
        joints = joints - joints[0][:3]
        joints = scale_joints(joints, robot_type="gr1t2")

        # root到wrist的chain
        l_parents = [0,3,6,9,13,16,18,20]
        r_parents = [0,3,6,9,14,17,19,21]

        # 获得全局旋转矩阵
        rot_mats = batch_rodrigues(whole_pose_aa).reshape(52, 3, 3)
        l_globle_rot = get_global_rotation(rot_mats, l_parents) # l_wrist
        r_globle_rot = get_global_rotation(rot_mats, r_parents) # r_wrist

        # 获得左右手腕
        left_wrist_mat = np.eye(4)  # 初始化单位矩阵
        left_wrist_mat[:3, :3] = l_globle_rot.clone().numpy() @ np.array([[0,0,-1],[0,1,0],[1,0,0]]) # 填入旋转部分
        left_wrist_mat[:3, 3] =  joints[20][:3]  # 填入平移部分

        # 右手腕
        right_wrist_mat = np.eye(4)  # 初始化单位矩阵
        right_wrist_mat[:3, :3] = r_globle_rot.clone().numpy() @ np.array([[0,0,1],[0,-1,0],[1,0,0]])  # 填入旋转部分
        right_wrist_mat[:3, 3] = joints[21][:3]  # 填入平移部分


        left_hand = left_hand - left_hand[0]
        right_hand = right_hand - right_hand[0]

        self.data = (left_wrist_mat,right_wrist_mat,left_hand,right_hand)
        
        # 将数据给发出去，这里可能需要放到解完ik哪里
    def publish(self):
        upper_action = xsens_gr1_lcmt()
        upper_action.action = self.action
        self.lc.publish("upper_action", upper_action.encode())

    def get_data(self):
        return self.data


if __name__ == "__main__":
    # 设置基本参数
    scale = 0.6546
    beta = torch.tensor([[2.8964, 3.2512, -6.3317, 0.7466, 8.5855, 4.3732, 
                         -16.6232, -6.0283, -29.3488, 16.7490]])
    
    # 创建相机包装器实例
    camera = CameraWrapper(scale=scale, beta=beta)
    
    try:
        # 开始接收数据
        camera.start_receiving()
        # 主循环
        while True:
            data = camera.get_data()
            if data is not None:
                # 打印数据示例（这里只打印左手腕矩阵作为示例）
                print("左手腕矩阵：")
                print(data[0])
            
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        # 停止接收并清理
        camera.stop_receiving()
