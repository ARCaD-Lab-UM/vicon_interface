import numpy as np

from .kalman_filter import KalmanFilter


def constant_acceleration_model(dt, jerk_psd):
    """Return (A, Q) of p' = v, v' = a, a' = white jerk over dt [s], for three axes.

    State order (px, py, pz, vx, vy, vz, ax, ay, az); jerk_psd is the jerk power spectral density
    per axis [m^2/s^5]. Both are exact: A = expm(Ac dt) ends after the dt^2 term (Ac is nilpotent),
    and Q = integral over [0, dt] of expm(Ac s) G q G^T expm(Ac s)^T ds in closed form. Two steps
    of dt/2 therefore equal one step of dt, so a dropped Vicon frame is just a longer step.
    """
    A1 = np.array([[1.0, dt, 0.5 * dt * dt],
                   [0.0, 1.0, dt],
                   [0.0, 0.0, 1.0]])
    Q1 = np.array([[dt ** 5 / 20.0, dt ** 4 / 8.0, dt ** 3 / 6.0],
                   [dt ** 4 / 8.0, dt ** 3 / 3.0, dt ** 2 / 2.0],
                   [dt ** 3 / 6.0, dt ** 2 / 2.0, dt]])
    A = np.kron(A1, np.eye(3))
    Q = np.kron(Q1, np.diag(np.asarray(jerk_psd, dtype=float)))
    return A, Q


class ViconLinearStateEstimator(KalmanFilter):
    """Constant-acceleration KF for a Vicon object: world (p, v, a) per axis, position observed."""

    def __init__(self, jerk_psd, R, P_init):
        self.jerk_psd_ = np.asarray(jerk_psd, dtype=float)  # m^2/s^5 per axis
        assert self.jerk_psd_.shape == (3,)
        A, Q = constant_acceleration_model(0.0, self.jerk_psd_)
        B = np.zeros((9, 1))  # no input: the model is driven by the white jerk alone
        C = np.zeros((3, 9))
        C[:, :3] = np.eye(3)  # z = p
        super().__init__(A, B, C, Q, R, np.zeros(9), P_init)

    def predict_dt(self, dt):
        """Move the state dt [s] along the model and add the process noise of that interval."""
        self.A, self.Q = constant_acceleration_model(dt, self.jerk_psd_)
        return self.predict(np.zeros(1))

    def correct(self, y):
        x = super().correct(y)
        self.P = 0.5 * (self.P + self.P.T)  # keep P symmetric against round-off
        return x

    def predicted_state(self, dt):
        """Return the state dt [s] ahead along the model; the filter itself is not changed."""
        A, _ = constant_acceleration_model(dt, self.jerk_psd_)
        return A @ self.x
