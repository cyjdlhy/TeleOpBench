import numpy as np
from filterpy.kalman import KalmanFilter

class KalmanFilterSmoother:
    def __init__(self, initial_state, dt=1, process_variance=1e-5, measurement_variance=1e-4):
        self.kf = KalmanFilter(dim_x=len(initial_state), dim_z=len(initial_state))
        
        # State transition matrix
        self.kf.F = np.eye(len(initial_state))
        
        # Measurement function
        self.kf.H = np.eye(len(initial_state))
        
        # Initial state
        self.kf.x = np.array(initial_state)
        
        # Covariance matrix
        self.kf.P *= 1000
        
        # Process noise covariance
        self.kf.Q = np.eye(len(initial_state)) * process_variance
        
        # Measurement noise covariance
        self.kf.R = np.eye(len(initial_state)) * measurement_variance
        
    def update(self, measurement):
        self.kf.predict()
        self.kf.update(measurement)
        return self.kf.x