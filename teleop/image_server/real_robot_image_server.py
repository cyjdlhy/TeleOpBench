import cv2
import zmq
import time
import struct
from collections import deque
import numpy as np
import os
import datetime


class ImageServer:
    def __init__(self, config, port = 1213, Unit_Test = False):
        """
        config example1:
        {
            'fps':30                                                          # frame per second
            'head_camera_type': 'opencv',                                     # opencv or realsense
            'head_camera_image_shape': [480, 1280],                           # Head camera resolution  [height, width]
            'head_camera_id_numbers': [0],                                    # '/dev/video0' (opencv)
            'wrist_camera_type': 'realsense', 
            'wrist_camera_image_shape': [480, 640],                           # Wrist camera resolution  [height, width]
            'wrist_camera_id_numbers': ["218622271789", "241222076627"],      # realsense camera's serial number
        }

        config example2:
        {
            'fps':30                                                          # frame per second
            'head_camera_type': 'realsense',                                  # opencv or realsense
            'head_camera_image_shape': [480, 640],                            # Head camera resolution  [height, width]
            'head_camera_id_numbers': ["218622271739"],                       # realsense camera's serial number
            'wrist_camera_type': 'opencv', 
            'wrist_camera_image_shape': [480, 640],                           # Wrist camera resolution  [height, width]
            'wrist_camera_id_numbers': [0,1],                                 # '/dev/video0' and '/dev/video1' (opencv)
        }

        If you are not using the wrist camera, you can comment out its configuration, like this below:
        config:
        {
            'fps':30                                                          # frame per second
            'head_camera_type': 'opencv',                                     # opencv or realsense
            'head_camera_image_shape': [480, 1280],                           # Head camera resolution  [height, width]
            'head_camera_id_numbers': [0],                                    # '/dev/video0' (opencv)
            #'wrist_camera_type': 'realsense', 
            #'wrist_camera_image_shape': [480, 640],                           # Wrist camera resolution  [height, width]
            #'wrist_camera_id_numbers': ["218622271789", "241222076627"],      # serial number (realsense)
        }
        """
        print(config)
        self.fps = config.get('fps', 30)
        self.head_camera_type = config.get('head_camera_type', 'opencv')
        self.head_image_shape = config.get('head_camera_image_shape', [480, 640])      # (height, width)
        self.head_camera_id_numbers = config.get('head_camera_id_numbers', [0])

        self.port = port
        self.save_dir = config.get('save_dir', 'real_record')


        # Set ZeroMQ context and socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{self.port}")


    def _close(self):
        self.socket.close()
        self.context.term()
        print("[Image Server] The server has been closed.")

    # def send_process(self):
    #     try:

    #         # Initialize camera capture
    #         cap = cv2.VideoCapture(0)  # 0 is typically the default camera
            
    #         if not cap.isOpened():
    #             print("[Image Server] Failed to open camera")
    #             return

    #         # 创建保存图片的文件夹
    #         save_dir = self.save_dir
    #         os.makedirs(save_dir, exist_ok=True)  # 自动创建文件夹（如果不存在）
    #         num_frame = 0  # 初始化帧计数器


    #         while True:
    #             # Capture frame-by-frame
    #             ret, frame = cap.read()
    #             if not ret:
    #                 print("[Image Server] Failed to capture frame")
    #                 break
                
    #             # Process the frame (you can keep your original processing if needed)
    #             head_color = frame
    #             full_color = head_color


    #             # 保存图片到本地
    #             timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 时间戳（精确到毫秒）
    #             filename = f"frame_{num_frame}_{timestamp}.jpg"
    #             filepath = os.path.join(save_dir, filename)
    #             cv2.imwrite(filepath, full_color)
    #             num_frame += 1  # 更新帧计数器

    #             print(f"nums_frame have saved", num_frame)

    #             ret, buffer = cv2.imencode('.jpg', full_color)
    #             if not ret:
    #                 print("[Image Server] Frame imencode is failed.")
    #                 continue

    #             jpg_bytes = buffer.tobytes()

    #             message = jpg_bytes

    #             self.socket.send(message)


    #     except KeyboardInterrupt:
    #         print("[Image Server] Interrupted by user.")
    #     finally:
    #         self._close()

    def send_process(self):
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("[Image Server] Failed to open camera")
                return

            is_recording = False
            current_save_dir = None
            num_frame = 0

            cv2.namedWindow('Camera')  # 创建窗口用于接收按键事件

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[Image Server] Failed to capture frame")
                    break

                # 图像处理（保持原逻辑）
                full_color = frame.copy()

                # 显示实时画面（可隐藏）
                cv2.imshow('Camera', full_color)

                # 检测按键输入
                key = cv2.waitKey(1) & 0xFF
                if key == ord('n') or key == ord('N'):
                    if not is_recording:
                        # 生成时间戳并创建文件夹
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        current_save_dir = os.path.join(self.save_dir, timestamp)
                        os.makedirs(current_save_dir, exist_ok=True)
                        num_frame = 0  # 重置帧计数器
                        is_recording = True
                        print(f"[Image Server] 开始记录到: {current_save_dir}")
                elif key == ord('m') or key == ord('M'):
                    if is_recording:
                        is_recording = False
                        print(f"[Image Server] 停止记录，数据保存在: {current_save_dir}")

                # 记录帧到指定目录
                if is_recording:
                    frame_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    filename = f"frame_{num_frame}_{frame_timestamp}.jpg"
                    cv2.imwrite(os.path.join(current_save_dir, filename), full_color)
                    num_frame += 1
                    print(f"[Image Server] 已保存帧: {num_frame}", end='\r')

                # 保持原有发送逻辑
                ret, buffer = cv2.imencode('.jpg', full_color)
                if not ret:
                    continue
                self.socket.send(buffer.tobytes())

                # 退出检测
                if key == 27:  # ESC键退出
                    break

        except KeyboardInterrupt:
            print("[Image Server] 用户中断")
        finally:
            cv2.destroyAllWindows()
            cap.release()
            self._close()
            print("[Image Server] 资源已释放")


if __name__ == "__main__":
    config = {
        'fps': 30,
        'head_camera_type': 'opencv',
        'head_camera_image_shape': [480, 1280],  # Head camera resolution
        'head_camera_id_numbers': [0],
        'wrist_camera_type': 'opencv',
        'wrist_camera_image_shape': [480, 640],  # Wrist camera resolution
        'wrist_camera_id_numbers': [2, 4],
        'save_dir':'../real_record',
    }

    server = ImageServer(config, Unit_Test=False)
    server.send_process()