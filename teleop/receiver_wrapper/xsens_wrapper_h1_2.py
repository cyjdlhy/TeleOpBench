import socket
import numpy as np
import struct
from threading import Thread
from scipy.spatial.transform import Rotation as R
import lcm
from lcm_glove.glove import glove
from lcm_types.h1_lcmt import h1_lcmt
import logging
import time
import sys
import threading

sys.path.append("./")
sys.path.append("../")

logging.basicConfig(level=logging.DEBUG)

class XsenseWrapper:
    def __init__(self, host="0.0.0.0", port=9764, timeout=3, segment_id=1):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.segment_id = segment_id
        self.lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
        self.control_msg_subscription = self.lc.subscribe("glove", self.get_glove)
        
        # self.lcm_thread = threading.Thread(target=self._lcm_loop)
        # self.lcm_thread.daemon = True
        # self.lcm_thread.start()
        self.glove_data = np.zeros(40)
        self.action = np.zeros(38)
        self.ori = np.zeros((63, 4))  # wxyz
        self.pos = np.zeros((63, 3))  # Initialize orientation array

        self.rev_pos = np.zeros((63, 3))
        self.robot_pos = np.zeros((42, 3))
        self.robot_limbs = np.array([0.2122727823636612, 0.3304533401253496, 0.23062395799222593])
        self.scale = {}
        self.scale["default"] = np.ones(3)
        self.scale["character1"] = np.array(
            # [1.0785723728586174, 0.8293056336515555, 1.0054510227890003, 1.6997153499201505]
            [1.0785723728586174, 0.8293056336515555, 1.0054510227890003]
        )
        self.scale["character2"] = np.array(
            # [1.0785723728586174, 0.8293056336515555, 1.0054510227890003, 1.30]
            [1.0785723728586174, 0.8293056336515555, 1.0054510227890003]
        )

        # tcp  with Xsens
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))  # Bind the socket to host and port
        self.server_socket.listen(1)  # Listen for incoming connections
        print(f"Server listening on {self.host}:{self.port}...")
        # Accept incoming connections
        self.client_socket, self.client_address = self.server_socket.accept()
        print(f"Connected to client at {self.client_address}")

        self.last_received = np.nan
        self.hand_pose = np.zeros((24))
        self.angle = False
        self.running = True  # This flag controls the thread loop
        # self.ik_Wo_hand = Xsense_IK_Wo_hand(True)
        # self.hand_pos = gr1_hand()
        self.receiver_thread = None  # Will store the receiver thread
        self.hand_thread = None
        self.arm_thread = None

        # 来一个data变量
        self.data = None  # 记录关节的齐次坐标

        # 处理接收的数据

    def _lcm_loop(self):
        """LCM消息循环"""
        while True:
            self.lc.handle()

    def get_glove(self, channel, data):
        msg = glove.decode(data)
        self.glove_data = np.array(msg.action)

    def _parse_position_packet(self, message):
        # Parse the packet and update pos and ori arrays
        try:
            message_id = message[:6].decode("utf-8")
            message_type = int(message_id[-2:])
            sample_counter = struct.unpack(">I", message[6:10])[0] + 1
            all_segments = struct.unpack(">B", message[11:12])[0]
            # additional info
            datagram_counter = bin(struct.unpack(">B", message[10:11])[0])[2:].zfill(8)
            time_code = struct.unpack(">I", message[12:16])[0]
            figure_nums = struct.unpack(">B", message[19:20])[0]
            new_packet_flag = 0

            packet_size = 32
            if sample_counter == self.last_received:
                new_packet_flag = 0
            else:
                new_packet_flag = 1

            if message_type == 2:
                for s in range(all_segments):
                    offset = s * packet_size
                    segment_id = struct.unpack_from(">I", message, offset + 24)[0]
                    self.pos[s, 0] = struct.unpack_from(">f", message, offset + 4 + 24)[0]
                    self.pos[s, 1] = struct.unpack_from(">f", message, offset + 8 + 24)[0]
                    self.pos[s, 2] = struct.unpack_from(">f", message, offset + 12 + 24)[0]
                    self.ori[s, 3] = struct.unpack_from(">f", message, offset + 16 + 24)[0]
                    self.ori[s, 0] = struct.unpack_from(">f", message, offset + 20 + 24)[0]
                    self.ori[s, 1] = struct.unpack_from(">f", message, offset + 24 + 24)[0]
                    self.ori[s, 2] = struct.unpack_from(">f", message, offset + 28 + 24)[0]

                self.angle = False
            elif message_type == 20:
                self.angle = True
            self.last_received = sample_counter
            return True
        except Exception as e:
            logging.error(f"Error parsing packet, and the exception is {e}")
            return False

    def parse_position_packet(self, message):
        # Parse the packet and update pos and ori arrays
        try:
            message_id = message[:6].decode("utf-8")
            message_type = int(message_id[-2:])
            sample_counter = struct.unpack(">I", message[6:10])[0] + 1
            datagram_counter = bin(struct.unpack(">B", message[10:11])[0])[2:].zfill(8)
            all_segments = struct.unpack(">B", message[11:12])[0]
            time_code = struct.unpack(">I", message[12:16])[0]
            figure_nums = struct.unpack(">B", message[19:20])[0]
            packet_size = 32
            new_packet_flag = 0

            if sample_counter == self.last_received:
                new_packet_flag = 0
            else:
                new_packet_flag = 1

            if message_type == 2:
                for s in range(all_segments):
                    offset = s * packet_size
                    segment_id = struct.unpack_from(">I", message, offset + 24)[0]

                    self.pos[s, 0] = struct.unpack_from(">f", message, offset + 4 + 24)[0]
                    self.pos[s, 1] = struct.unpack_from(">f", message, offset + 8 + 24)[0]
                    self.pos[s, 2] = struct.unpack_from(">f", message, offset + 12 + 24)[0]
                    self.ori[s, 3] = struct.unpack_from(">f", message, offset + 16 + 24)[0]
                    self.ori[s, 0] = struct.unpack_from(">f", message, offset + 20 + 24)[0]
                    self.ori[s, 1] = struct.unpack_from(">f", message, offset + 24 + 24)[0]
                    self.ori[s, 2] = struct.unpack_from(">f", message, offset + 28 + 24)[0]

                self.angle = False
                # print(f"ori: {self.ori}")
            elif message_type == 20:
                self.angle = True
            self.last_received = sample_counter
            return True
        except Exception as e:
            # logging.error("Error parsing packet")
            # print("Error parsing packet")
            return False

    def autofit(self):
        print(f"debug: {self.pos}")
        human_limb1 = np.linalg.norm(self.pos[8][:2]-self.pos[0][:2])
        human_limb2 = np.linalg.norm(self.pos[9]-self.pos[8])
        human_limb3 = np.linalg.norm(self.pos[10]-self.pos[9])
        scale = np.array([
            # self.robot_limbs[0] / human_limb1,
            1.15,
            self.robot_limbs[1] / human_limb2,
            self.robot_limbs[2] / human_limb3
        ])
        self.scale["default"] = scale
        print(f"debug: {scale}")
    def socket_receiver(self):
        while self.running:
            try:
                message = self.client_socket.recv(8 * 2000)  # Max packet length
                self.parse_position_packet(message)
                # self.process_arm()
            except socket.error or KeyboardInterrupt as e:
                print(f"Socket error or keyboard: {e}")
                self.stop_receiving()
                break

    # 开启接收数据
    def start_receiving(self):
        # Start a thread to handle the socket reception
        if self.receiver_thread is None or not self.receiver_thread.is_alive():
            self.running = True
            self.receiver_thread = Thread(target=self.socket_receiver)
            self.receiver_thread.daemon = True
            self.receiver_thread.start()

    # 继续处理收到的数据
    def relative_position_orientation(self, index, direc):
        if self.ori[0].all() == 0:
            pass
        # print("DEBUG: self.ori[0] =", self.ori[0])
        q_ref = R.from_quat(self.ori[0])  # pelvis orientation
        pelvis_rotation = q_ref.as_matrix()

        if index == 12 or index == 8:  # 肩膀
            relative_position = self.pos[index] - self.pos[0]
            # relative_position[2] = self.robot_pos[4][2] # lock shoulder to robot
            relative_position[2] = 0.41
            relative_position[:2] = relative_position[:2] * self.scale["default"][0]
            self.rev_pos[index] = relative_position
            relative_position = np.linalg.inv(pelvis_rotation).dot(relative_position)

        elif index == 13 or index == 9:  # 上臂（手肘）
            relative_position = self.pos[index] - self.pos[index - 1]
            relative_position = relative_position * self.scale["default"][1] + self.rev_pos[index - 1]
            self.rev_pos[index] = relative_position
            relative_position = np.linalg.inv(pelvis_rotation).dot(relative_position)

        elif index == 14 or index == 10:  # 下臂（手腕）
            relative_position = self.pos[index] - self.pos[index - 1]
            relative_position = relative_position * self.scale["default"][2] + self.rev_pos[index - 1]
            self.rev_pos[index] = relative_position
            relative_position = np.linalg.inv(pelvis_rotation).dot(relative_position)

        q_target = R.from_quat(self.ori[index])
        # q_ref_inv = q_ref.inv() #逆矩阵保留
        relative_mat_ori = np.linalg.inv(pelvis_rotation).dot(q_target.as_matrix())
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = relative_mat_ori
        transform_matrix[:3, 3] = relative_position
        return transform_matrix

    def process(self):
        # head = self.relative_position_orientation(6, "l")
        left_arm = self.relative_position_orientation(12, "l")
        right_arm = self.relative_position_orientation(8, "r")
        left_forearm = self.relative_position_orientation(13, "l")
        right_forearm = self.relative_position_orientation(9, "r")
        left_hand = self.relative_position_orientation(14, "l")
        right_hand = self.relative_position_orientation(10, "r")

        # left_hand_target_axis = np.eye(3)
        left_hand_target_axis = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        left_hand[:3, :3] = (left_hand[:3, :3]).dot(left_hand_target_axis)

        # right_hand_target_axis = np.eye(3)
        right_hand_target_axis = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        right_hand[:3, :3] = (right_hand[:3, :3]).dot(right_hand_target_axis)

        # left_forearm_target_axis = np.eye(3)
        left_forearm_target_axis = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        left_forearm[:3, :3] = (left_forearm[:3, :3]).dot(left_forearm_target_axis)

        # right_forearm_target_axis = np.eye(3)
        right_forearm_target_axis = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        right_forearm[:3, :3] = (right_forearm[:3, :3]).dot(right_forearm_target_axis)

        # left_arm_target_axis = np.eye(3)
        left_arm_target_axis = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        left_arm[:3, :3] = (left_arm[:3, :3]).dot(left_arm_target_axis)

        # right_arm_target_axis = np.eye(3)
        right_arm_target_axis = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        right_arm[:3, :3] = (right_arm[:3, :3]).dot(right_arm_target_axis)

        # fmt: off
        self.data = (left_hand, right_hand, left_forearm, right_forearm, left_arm, right_arm)
        # fmt: on
        # if rev:
        #     print(f"IK solved \n action: {action} \n action_shape: {action_shape}")
        # end = time.time()
        # print(f"Arm ik full: Time: {end - start},freq: {1/(end - start)}")

    # 将数据给发出去，这里可能需要放到解完ik哪里
    def publish(self):
        h1_action = h1_lcmt()
        h1_action.action = list(self.action)
        self.lc.publish("h1", h1_action.encode())

    def stop_receiving(self):
        # Set running flag to False to stop the thread
        self.running = False

        # Wait for the thread to finish
        if self.receiver_thread is not None:
            self.receiver_thread.join()

    # 这里返回已经在pink坐标系下的数据
    def get_data(self):
        # self.start_receiving()
        return self.data


if __name__ == "__main__":
    my_Wrapper = Wrapper()

    my_Wrapper.start_receiving()
    time.sleep(1)
    freq = 240
    start = time.time()
    my_Wrapper.autofit()
    with_hand = False # If need hands pos, set to True
    try:
        while True:
            start = time.time()
            my_Wrapper.lc.handle()
            # Usage example
            # my_Wrapper.autofit()
            glove_data = my_Wrapper.glove_data
            glove_data[1] = 50 - glove_data[1]
            glove_data[21] = 50 - glove_data[21]
            glove_data = glove_data / 180 * np.pi
            glove_data[0] = glove_data[0] * 1.5
            glove_data[20] = glove_data[20] * 1.5
            glove_data[1] = glove_data[1] 
            glove_data[21] = glove_data[21] 

            glove_data[3] = glove_data[3] * 2.5
            glove_data[23] = glove_data[23] * 2.5

            left_hand = [5, 6, 9, 10, 17, 18, 13, 14, 0, 1, 2, 3]
            right_hand = [25, 26, 29, 30, 37, 38, 33, 34, 20, 21, 22, 23]
            left_hand_data = glove_data[left_hand]
            right_hand_data = glove_data[right_hand]
            my_Wrapper.process()
            my_Wrapper.action[7:19] = left_hand_data
            my_Wrapper.action[26:] = right_hand_data
            my_Wrapper.ik.vis.display(my_Wrapper.action)
            my_Wrapper.publish()
            # logging.debug(f"Time: {time.time() - start}\n Total freq: {1 / (time.time() - start)}")

    except KeyboardInterrupt:
        # Stop the receiver thread when needed
        my_Wrapper.stop_receiving()
        print("closed")
