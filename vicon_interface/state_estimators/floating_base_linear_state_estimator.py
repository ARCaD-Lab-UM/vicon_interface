import numpy as np

from .kalman_filter import KalmanFilter


def linear_motion_model(dt):
    """Return (A, B) of p' = v, v' = a over dt; Ac is nilpotent, so I + dt*Ac is exact."""
    A = np.eye(6)
    A[:3, 3:] = dt * np.eye(3)
    B = np.zeros((6, 3))
    B[3:, :] = dt * np.eye(3)
    return A, B


class FloatingBaseLinearStateEstimator(KalmanFilter):

    def __init__(self, dt, Q, R, height_init):
        # state (px, py, pz, vx, vy, vz), control (ax, ay, az), observation (pz, vx, vy, vz)
        dim_state = 6
        dim_obs = 4
        A, B = linear_motion_model(dt)
        C = np.zeros((dim_obs, dim_state))
        C[:, 2:] = np.eye(4)

        self.x_init = np.array([0, 0, height_init, 0, 0, 0])   # initial pos and vel in 3D
        self.P_init = np.eye(dim_state) * 1e-5         # initial state covariance
        super().__init__(A, B, C, Q, R, self.x_init, self.P_init)
