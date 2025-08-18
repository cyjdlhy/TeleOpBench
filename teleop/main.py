import numpy as np
import time
import argparse
import cv2
from multiprocessing import shared_memory, Array, Lock
import threading

# import pyrealsense2 as rs
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from teleop.image_server.image_client import ImageClient
import socket
import torch


# utils
from teleop.utils.scale_utils import Visulize_solq
import traceback  # 导入 traceback 以打印完整错误信息


class FusedController:
    def __init__(self, input_type, robot_type="g1", visualize=False):
        """
        判断输入的类型，然后进行处理,
        初步架构就是: 有一个发送数据的进程（应该不会发到这个类里面）
        （1）来一个接收数据的进程（接收不同的数据，visionpro，camera，xsense）
        （2）将接收到的数据进行处理，然后获得对应的旋转矩阵
        （3）将旋转矩阵放入ik中，解开获得q，返回q
        """
        self.visualize = visualize
        self.input_type = input_type
        self.robot_type = robot_type
        if input_type == "visionpro":
            self.init_visionpro()
        elif input_type == "camera":
            self.init_camera()
        elif input_type == "xsense":
            self.init_xsense()
        elif input_type == "exoskeleton":
            self.input_exoskeleton()
        else:
            raise ValueError(f"不支持的输入类型: {input_type}。支持的类型包括: visionpro, camera, xsense, exoskeleton")

    def init_visionpro(self):
        if self.robot_type == "g1":
            from teleop.robot_control.tv_ik_g1 import TV_IK
            from teleop.receiver_wrapper.tv_wrapper_g1 import TeleVisionWrapper
        elif self.robot_type == "gr1t2":
            from teleop.robot_control.tv_ik_gr1t2 import TV_IK
            from teleop.receiver_wrapper.tv_wrapper_gr1t2 import TeleVisionWrapper
        elif self.robot_type == "h1_2":
            from teleop.robot_control.tv_ik_h1_2 import TV_IK
            from teleop.receiver_wrapper.tv_wrapper_h1_2 import TeleVisionWrapper

        # 初始化共享内存和图像处理
        self.tv_img_shape = [480, 640, 3]
        self.tv_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(self.tv_img_shape) * np.uint8().itemsize)
        self.tv_img_array = np.ndarray(self.tv_img_shape, dtype=np.uint8, buffer=self.tv_img_shm.buf)
        # 初始化各个组件
        self.img_client = ImageClient(
            tv_img_shape=self.tv_img_shape, tv_img_shm_name=self.tv_img_shm.name, server_address="0.0.0.0", port=1213
        )

        # 启动图像接收线程
        self.image_receive_thread = threading.Thread(target=self.img_client.receive_process, daemon=True)
        self.image_receive_thread.daemon = True
        self.image_receive_thread.start()

        self.wrapper = TeleVisionWrapper(False, self.tv_img_shape, self.tv_img_shm.name)
        self.ik = TV_IK(visualize=self.visualize)

    def init_camera(self):

        if self.robot_type == "g1":
            from teleop.robot_control.camera_ik_g1_media import Camera_IK
            from teleop.receiver_wrapper.camera_wrapper_g1_media import CameraWrapper
        elif self.robot_type == "gr1t2":
            from teleop.robot_control.camera_ik_gr1t2_media import Camera_IK
            from teleop.receiver_wrapper.camera_wrapper_gr1t2_media import CameraWrapper
        elif self.robot_type == "h1_2":
            from teleop.robot_control.camera_ik_h1_2_media import Camera_IK
            from teleop.receiver_wrapper.camera_wrapper_h1_2_media import CameraWrapper

        scale = 1.0
        beta = torch.tensor(
            [[2.8964, 3.2512, -6.3317, 0.7466, 8.5855, 4.3732, -16.6232, -6.0283, -29.3488, 16.7490]]
        )
        self.wrapper = CameraWrapper(scale=scale, beta=beta)
        self.wrapper.start_receiving()
        self.ik = Camera_IK(visualize=self.visualize)

    def init_xsense(self):
        if self.robot_type == "g1":
            from teleop.robot_control.xsens_ik_g1 import Xsense_IK
            from teleop.receiver_wrapper.xsens_wrapper_g1 import XsenseWrapper

            self.wrapper = XsenseWrapper()
            # 初始化就打开数据的接收
            self.wrapper.start_receiving()
            time.sleep(1)
            self.xsense_ik = Xsense_IK(visualize=self.visualize)

        elif self.robot_type == "gr1t2":
            from teleop.robot_control.xsens_ik_gr1t2 import Xsense_IK
            from teleop.receiver_wrapper.xsens_wrapper_gr1t2 import XsenseWrapper

            self.wrapper = XsenseWrapper()
            # 初始化就打开数据的接收
            
            self.wrapper.start_receiving()
            time.sleep(1)
            self.wrapper.autofit()
            self.xsense_ik = Xsense_IK(visualize=self.visualize)
        elif self.robot_type == "h1_2":
            from teleop.robot_control.xsens_ik_h1_2 import Xsense_IK
            from teleop.receiver_wrapper.xsens_wrapper_h1_2 import XsenseWrapper
            self.wrapper = XsenseWrapper()
            self.wrapper.start_receiving()
            time.sleep(1)
            self.wrapper.autofit()
            self.xsense_ik = Xsense_IK(visualize=self.visualize)

    def init_exoskeleton(self):
        if self.robot_type == "g1":
            from teleop.receiver_wrapper.exoskeleton_wrapper_g1 import ExoskeletonWrapper

            self.wrapper = ExoskeletonWrapper()
            self.wrapper.start_receiving()
            time.sleep(1)
        elif self.robot_type == "gr1t2":
            from teleop.receiver_wrapper.exoskeleton_wrapper_gr1t2 import ExoskeletonWrapper

            self.wrapper = ExoskeletonWrapper()
            self.wrapper.start_receiving()
            time.sleep(1)
        elif self.robot_type == "h1_2":
            from teleop.receiver_wrapper.exoskeleton_wrapper_h1_2 import ExoskeletonWrapper

            self.wrapper = ExoskeletonWrapper()
            self.wrapper.start_receiving()
            time.sleep(1)
        else:
            raise ValueError(f"不支持的机器人类型: {self.robot_type}。支持的类型包括: g1, gr1t2, h1_2")
        
    def get_q(self, *args, **kwargs):
        data = self.wrapper.get_data(*args, **kwargs)
        if data == None:
            return False
        
        # 如果是 exoskeleton 类型，直接返回数据
        if self.input_type == "exoskeleton":
            sol_q = data
        else:
            sol_q = self.ik.ik_fun(data)
        
        return sol_q



if __name__ == "__main__":
    # 根据命令行参数选择输入类型
    parser = argparse.ArgumentParser(description="选择输入类型")
    parser.add_argument("--input_type", type=str, default="visionpro", choices=["camera", "xsense", "visionpro", "exoskeleton"], help="teleop type")
    parser.add_argument("--robot_type", type=str, default="gr1t2", choices=["gr1t2", "g1", "h1_2"], help="robot_type")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    args = parser.parse_args()

    # 创建 FusedController 实例
    controller = FusedController(args.input_type, args.robot_type, args.visualize)


    if args.visualize == False and args.robot_type == "g1":
        # g1
        urdf_abs_path = "../assets/g1/g1_body29_hand14.urdf"
        package_dirs = ["../assets/g1"]
        visulize_mesh = Visulize_solq(urdf_path=urdf_abs_path, mesh_dir=package_dirs, robot_type=args.robot_type)

    elif args.visualize == False and args.robot_type == "gr1t2":
        # gr1t2
        visulize_mesh = Visulize_solq(
            urdf_path="../assets/GRX/GR1/GR1T2/urdf/GR1T2_inspire_hand.urdf",
            mesh_dir=["../assets/GRX/GR1/GR1T2/urdf"],
            robot_type=args.robot_type,
        )

    elif args.visualize == False and args.robot_type == "h1_2":
        # h1_2
        visulize_mesh = Visulize_solq(
            urdf_path="../assets/h1_2/h1_2.urdf", 
            mesh_dir=["../assets/h1_2/"], 
            robot_type=args.robot_type
        )


    try:
        print(f"开始{args.input_type}控制演示...")
        while True:
            loop_start = time.time()  # 记录当前循环的开始时间
            # 获取关节角度
            q = controller.get_q()
            if q is not False and args.visualize == False:
                visulize_mesh(q)
            if q is False:
                continue

            # if args.input_type == "xsense":
            #     controller.wrapper.action = q
            #     controller.wrapper.publish()
            # elif args.input_type == "camera":
            #     controller.camera_wrapper.action = q
            #     controller.camera_wrapper.publish()
            # elif args.input_type == "visionpro":
            #     controller.tv_wrapper.action = q
            #     controller.tv_wrapper.publish()
            loop_end = time.time()  # 记录当前循环结束时间
            loop_time = loop_end - loop_start  # 计算循环时长
            # print("frequency:{}".format(1/loop_time))

    except KeyboardInterrupt:
        print("\n程序已停止")
    except Exception as e:
        print(f"发生错误: {e}")
        traceback.print_exc()  # 这里会打印完整的错误堆栈
    finally:
        # 清理资源
        cv2.destroyAllWindows()
