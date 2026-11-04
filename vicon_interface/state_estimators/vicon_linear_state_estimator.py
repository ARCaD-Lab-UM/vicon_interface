import numpy as np

from .floating_base_linear_state_estimator import linear_motion_model
from .kalman_filter import KalmanFilter


class ViconLinearStateEstimator(KalmanFilter):
    """Constant-velocity KF for a Vicon object: world (px, py, pz, vx, vy, vz), all observed."""

    def __init__(self, dt, Q, R, P_init):
        A, B = linear_motion_model(dt)
        C = np.eye(6)
        x_init = np.zeros(6)
        super().__init__(A, B, C, Q, R, x_init, P_init)

    def set_dt(self, dt):
        """Rebuild A and B for a new predict interval dt [s]."""
        self.A, self.B = linear_motion_model(dt)
