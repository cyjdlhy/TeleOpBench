import numpy as np
JointsToLock = [
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
    
]

FRAME_AXIS_POSITIONS = (
    np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]]).astype(np.float32).T
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
