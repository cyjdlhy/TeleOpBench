import sys

sys.path.append("./")
sys.path.append("../")
import socket
import numpy as np
import struct
from threading import Thread
# from ik.g1_ik import G1_IK
# from ik.g1_ik_with_hand import G1_IK
from scipy.spatial.transform import Rotation as R
import lcm
from lcm_with_hand.xsense_lcmt import xsense_lcmt

class XsenseWrapper:

    def __init__(self, host="0.0.0.0", port=9764, timeout=3, segment_id=1):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.segment_id = segment_id
        # 暂时不用lcm
        self.lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
        # self.ori = np.zeros((23, 4))  # wxyz?
        # self.pos = np.zeros((23, 3))  # Initialize orientation array
        self.ori = np.zeros((63, 4))  # wxyz
        self.pos = np.zeros((63, 3))  # Initialize orientation array
        self.last_received = np.nan
        self.angle = False
        self.running = True  # This flag controls the thread loop
        # self.ik = G1_IK(True)
        self.action = np.zeros(28)
        # Create TCP socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))  # Bind the socket to host and port
        self.server_socket.listen(1)  # Listen for incoming connections
        print(f"Server listening on {self.host}:{self.port}...")
        # Accept incoming connections
        self.client_socket, self.client_address = self.server_socket.accept()
        print(f"Connected to client at {self.client_address}")
        self.receiver_thread = None  # Will store the receiver thread

        # 来一个data变量
        self.data = None

    # 处理接收的数据
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

            # pos = np.zeros((all_segments, 3), dtype=float)
            # ori = np.zeros((all_segments, 4), dtype=float)

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
            # import ipdb; ipdb.set_trace()
            # self.pos = pos
            # self.ori = ori
            # import ipdb; ipdb.set_trace()
            # print(f"Quaternion at index 2: {self.ori[2]}")
            self.last_received = sample_counter
            return True
        except:
            return False

    def socket_receiver(self):
        while self.running:
            try:
                message = self.client_socket.recv(8 * 2000)  # Max packet length
                # if not message:
                # break
                self.parse_position_packet(message)
                # print(self.ori[2])
                # import ipdb; ipdb.set_trace()
                self.process()
            except socket.error as e:
                print(f"Socket error: {e}")
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
    def relative_position_orientation(self, index, direc, scale=0.6):
        if direc == "l":
            shoulder_pos = self.pos[11]
            modi_pos = np.array([0.02, 0.10, 0.3])
            # modi_pos = np.array([0.02, 0.13, 0.3])
        elif direc == "r":
            shoulder_pos = self.pos[7]
            modi_pos = np.array([0.02, -0.10, 0.3])
            # modi_pos = np.array([0.02, -0.13, 0.3])
        relative_position = self.pos[index] - shoulder_pos
        # if np.linalg.norm(self.ori[2]) == 0:
        #     raise ValueError("Invalid quaternion at self.ori[2]: zero norm quaternion")
        q_ref = R.from_quat(self.ori[2])
        pelvis_rotation = q_ref.as_matrix()
        relative_position = np.linalg.inv(pelvis_rotation).dot(relative_position) * scale + modi_pos
        # import ipdb; ipdb.set_trace()
        # import ipdb; ipdb.set_trace()
        q_target = R.from_quat(self.ori[index])
        q_ref_inv = q_ref.inv()
        relative_mat_ori = np.linalg.inv(pelvis_rotation).dot(q_target.as_matrix())
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = relative_mat_ori
        transform_matrix[:3, 3] = relative_position
        return transform_matrix

    # 处理数据获得最后的矩阵
    def process(self):
        scale = 0.45 / 0.7
        left_hand = self.relative_position_orientation(14, "l", scale)
        right_hand = self.relative_position_orientation(10, "r", scale)
        left_arm = self.relative_position_orientation(12, "l", scale)
        right_arm = self.relative_position_orientation(8, "r", scale)
        left_forearm = self.relative_position_orientation(13, "l", scale)
        right_forearm = self.relative_position_orientation(9, "r", scale)
        left_thumb = self.relative_position_orientation(26, "l", scale)
        right_thumb = self.relative_position_orientation(46, "r", scale)
        left_index = self.relative_position_orientation(30, "l", scale)
        right_index = self.relative_position_orientation(50, "r", scale)
        left_middle = self.relative_position_orientation(34, "l", scale)
        right_middle = self.relative_position_orientation(54, "r", scale) 

    
        # left_thumb_target_axis = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        # left_thumb_target_axis = np.eye(3)
        left_thumb_target_axis = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])
        left_thumb[:3, :3] = (left_thumb[:3, :3]).dot(left_thumb_target_axis)

        # right_thumb_target_axis = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        # right_thumb_target_axis = np.eye(3)
        right_thumb_target_axis = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])
        right_thumb[:3, :3] = (right_thumb[:3, :3]).dot(right_thumb_target_axis)

        left_index_target_axis = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        left_index[:3, :3] = (left_index[:3, :3]).dot(left_index_target_axis)

        right_index_target_axis = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        right_index[:3, :3] = (right_index[:3, :3]).dot(right_index_target_axis)

        left_middle_target_axis = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        left_middle[:3, :3] = (left_middle[:3, :3]).dot(left_middle_target_axis)

        right_middle_target_axis = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        right_middle[:3, :3] = (right_middle[:3, :3]).dot(right_middle_target_axis)

        right_hand_target_axis = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        right_hand[:3, :3] = (right_hand[:3, :3]).dot(right_hand_target_axis)

        left_hand_target_axis = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        left_hand[:3, :3] = (left_hand[:3, :3]).dot(left_hand_target_axis)

        right_arm_target_axis = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        right_arm[:3, :3] = (right_arm[:3, :3]).dot(right_arm_target_axis)

        left_arm_target_axis = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        left_arm[:3, :3] = (left_arm[:3, :3]).dot(left_arm_target_axis)

        left_forearm_target_axis = np.eye(3)
        # left_forearm_target_axis = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        left_forearm_target_axis = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        left_forearm[:3, :3] = (left_forearm[:3, :3]).dot(left_forearm_target_axis)

        right_arm_target_axis = np.eye(3)
        # right_arm_target_axis = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        right_arm_target_axis = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        right_forearm[:3, :3] = (right_forearm[:3, :3]).dot(right_arm_target_axis)


        self.data = (
            left_thumb,
            right_thumb,
            left_index,
            right_index,
            left_middle,
            right_middle,
            left_hand,
            right_hand,
            left_forearm,
            right_forearm,
            left_arm,
            right_arm,
        )

    # 将数据给发出去，这里可能需要放到解完ik哪里
    def publish(self):
        upper_action = xsense_lcmt()
        upper_action.action = self.action
        self.lc.publish("upper_action", upper_action.encode())

    def stop_receiving(self):
        # Set running flag to False to stop the thread
        self.running = False

        # Wait for the thread to finish
        if self.receiver_thread is not None:
            self.receiver_thread.join()

        # Close sockets
        self.client_socket.close()
        self.server_socket.close()

    # 这里返回已经在pink坐标系下的数据
    def get_data(self):
        # self.start_receiving()
        return self.data

        