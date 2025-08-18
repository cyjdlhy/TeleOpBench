import numpy as np
from teleop.receiver_wrapper.television import TeleVision
from teleop.receiver_wrapper.constants_gr1t2 import *
import sys

sys.path.append("./")
sys.path.append("../")
from lcm_with_hand.xsens_gr1_lcmt import xsens_gr1_lcmt
import lcm
# from teleop.utils.mat_tool import mat_update, fast_mat_inv

"""
(basis) OpenXR Convention : y up, z back, x right. 
(basis) Robot  Convention : z up, y left, x front.  
p.s. Vuer's all raw data follows OpenXR Convention, WORLD coordinate.

under (basis) Robot Convention, wrist's initial pose convention:

    # (Left Wrist) XR/AppleVisionPro Convention:
        - the x-axis pointing from wrist toward middle.
        - the y-axis pointing from index toward pinky.
        - the z-axis pointing from palm toward back of the hand.

    # (Right Wrist) XR/AppleVisionPro Convention:
        - the x-axis pointing from wrist toward middle.
        - the y-axis pointing from pinky toward index.
        - the z-axis pointing from palm toward back of the hand.
  
    # (Left Wrist URDF) Unitree Convention:
        - the x-axis pointing from wrist toward middle.
        - the y-axis pointing from palm toward back of the hand.
        - the z-axis pointing from pinky toward index.

    # (Right Wrist URDF) Unitree Convention:
        - the x-axis pointing from wrist toward middle.
        - the y-axis pointing from back of the hand toward palm. 
        - the z-axis pointing from pinky toward index.

under (basis) Robot Convention, hand's initial pose convention:

    # (Left Hand) XR/AppleVisionPro Convention:
        - the x-axis pointing from wrist toward middle.
        - the y-axis pointing from index toward pinky.
        - the z-axis pointing from palm toward back of the hand.

    # (Right Hand) XR/AppleVisionPro Convention:
        - the x-axis pointing from wrist toward middle.
        - the y-axis pointing from pinky toward index.
        - the z-axis pointing from palm toward back of the hand.

    # (Left Hand URDF) Unitree Convention:   
        - The x-axis pointing from palm toward back of the hand. 
        - The y-axis pointing from middle toward wrist.
        - The z-axis pointing from pinky toward index.

    # (Right Hand URDF) Unitree Convention: 
        - The x-axis pointing from palm toward back of the hand. 
        - The y-axis pointing from middle toward wrist.
        - The z-axis pointing from index toward pinky. 

    p.s. From website: https://registry.khronos.org/OpenXR/specs/1.1/man/html/openxr.html.
         You can find **(Left/Right Wrist) XR/AppleVisionPro Convention** related information like this below:
           "The wrist joint is located at the pivot point of the wrist, which is location invariant when twisting the hand without moving the forearm. 
            The backward (+Z) direction is parallel to the line from wrist joint to middle finger metacarpal joint, and points away from the finger tips. 
            The up (+Y) direction points out towards back of the hand and perpendicular to the skin at wrist. 
            The X direction is perpendicular to the Y and Z directions and follows the right hand rule."
         Note: The above context is of course under **(basis) OpenXR Convention**.

    p.s. **(Wrist/Hand URDF) Unitree Convention** information come from URDF files.
"""

class TeleVisionWrapper:
    def __init__(self, binocular, img_shape, img_shm_name):
        self.tv = TeleVision(binocular, img_shape, img_shm_name)

        self.vuer_head_mat = np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 1.5],
                                       [0, 0, 1, -0.2],
                                       [0, 0, 0, 1]])
        self.vuer_right_wrist_mat = np.array([[1, 0, 0, 0.2],
                                              [0, 1, 0, 1],
                                              [0, 0, 1, -0.5],
                                              [0, 0, 0, 1]])
        self.vuer_left_wrist_mat = np.array([[1, 0, 0, -0.2],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, -0.5],
                                             [0, 0, 0, 1]])

        self.lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
        self.action = np.zeros(41)

    def get_data(self):

        # first, y-up z-forward x-left axis
        # head at 1.5 height
        self.vuer_head_mat = mat_update(self.vuer_head_mat, self.tv.head_matrix.copy())
        self.vuer_right_wrist_mat = mat_update(self.vuer_right_wrist_mat, self.tv.right_hand.copy())
        self.vuer_left_wrist_mat = mat_update(self.vuer_left_wrist_mat, self.tv.left_hand.copy())
        # change of basis
        head_mat = grd_yup2grd_zup @ self.vuer_head_mat @ fast_mat_inv(grd_yup2grd_zup)
        right_wrist_mat = grd_yup2grd_zup @ self.vuer_right_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)
        left_wrist_mat = grd_yup2grd_zup @ self.vuer_left_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)

        rel_left_wrist_mat = left_wrist_mat @ hand2inspire_l_arm
        rel_left_wrist_mat[0:3, 3] = (
            rel_left_wrist_mat[0:3, 3] - head_mat[0:3, 3]
        )  # relative position in world frame; orientation is inspire-related

        rel_right_wrist_mat = right_wrist_mat @ hand2inspire_r_arm  # wTr = wTh @ hTr
        rel_right_wrist_mat[0:3, 3] = rel_right_wrist_mat[0:3, 3] - head_mat[0:3, 3]

        # homogeneous
        left_fingers = np.concatenate([self.tv.left_landmarks.copy().T, np.ones((1, self.tv.left_landmarks.shape[0]))])
        right_fingers = np.concatenate([self.tv.right_landmarks.copy().T, np.ones((1, self.tv.right_landmarks.shape[0]))])

        # change of basis
        left_fingers = grd_yup2grd_zup @ left_fingers
        right_fingers = grd_yup2grd_zup @ right_fingers

        rel_left_fingers = fast_mat_inv(left_wrist_mat) @ left_fingers
        rel_right_fingers = fast_mat_inv(right_wrist_mat) @ right_fingers
        rel_left_fingers = (hand2inspire_l_finger.T @ rel_left_fingers)[0:3, :].T
        rel_right_fingers = (hand2inspire_r_finger.T @ rel_right_fingers)[0:3, :].T

        # it should be mat of inspire dexterous hand

        rel_left_wrist_mat[:3, :3] = rel_left_wrist_mat[:3, :3] @ align_transform_l.T
        rel_right_wrist_mat[:3, :3] = rel_right_wrist_mat[:3, :3] @ align_transform_r.T
        # head_mat[:3, :3] = align_transform_head @ head_mat[:3, :3]
        return head_mat, rel_left_wrist_mat, rel_right_wrist_mat, rel_left_fingers, rel_right_fingers
    

        # 将数据给发出去，这里可能需要放到解完ik哪里
    def publish(self):
        upper_action = xsens_gr1_lcmt()
        upper_action.action = self.action
        self.lc.publish("upper_action", upper_action.encode())
        print(upper_action.action)


