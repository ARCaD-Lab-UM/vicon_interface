"""Kalman filter of a Vicon rigid body's pose, without an IMU.

Position lives in the world frame, orientation on SO(3) with its error in the body frame,
R = R_hat exp([d]x). Each axis carries its value and one derivative ('cv') or two ('ca'); the
next derivative is white noise of a given spectral density, so the process noise follows the
true frame interval. Only the pose is measured.
"""
from dataclasses import dataclass
import math

import numpy as np

ORDERS = {'cv': 2, 'ca': 3}  # states per axis: value, velocity[, acceleration]


def skew(w):
    """Return [w]x, the matrix with [w]x u = w x u."""
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def so3_exp(phi):
    """Return the rotation matrix exp([phi]x) of a rotation vector phi (rad)."""
    theta2 = float(phi @ phi)
    K = skew(phi)
    if theta2 < 1e-12:
        return np.eye(3) + K + 0.5 * (K @ K)
    theta = math.sqrt(theta2)
    return (np.eye(3) + (math.sin(theta) / theta) * K
            + ((1.0 - math.cos(theta)) / theta2) * (K @ K))


def so3_log(R):
    """Return the rotation vector phi (rad), |phi| <= pi, with exp([phi]x) = R."""
    w = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])  # sin * axis
    c = min(max(0.5 * (np.trace(R) - 1.0), -1.0), 1.0)  # cos(theta)
    s = float(np.linalg.norm(w))
    theta = math.atan2(s, c)
    if theta < 1e-6:
        return (1.0 + theta * theta / 6.0) * w
    if theta < math.pi - 1e-6:
        return (theta / s) * w
    B = 0.5 * (R + R.T) - c * np.eye(3)  # (1 - cos(theta)) axis axis^T; sin(theta) is too small
    column = B[:, int(np.argmax(np.diag(B)))]
    axis = column / np.linalg.norm(column)
    return theta * (axis if axis @ w >= 0.0 else -axis)


def so3_right_jacobian(phi):
    """Return J_r(phi): exp([phi + d]x) = exp([phi]x) exp([J_r(phi) d]x) to first order in d."""
    theta2 = float(phi @ phi)
    K = skew(phi)
    if theta2 < 1e-12:
        return np.eye(3) - 0.5 * K + (K @ K) / 6.0
    theta = math.sqrt(theta2)
    return (np.eye(3) - ((1.0 - math.cos(theta)) / theta2) * K
            + ((theta - math.sin(theta)) / (theta2 * theta)) * (K @ K))


def kinematic_blocks(order, dt):
    """Return (F, Q) of one axis over dt (s): its value and order - 1 derivatives.

    The next derivative is white noise of unit spectral density, so
    Q[i, j] = dt^n / (n (order - 1 - i)! (order - 1 - j)!) with n = 2 order - 1 - i - j.
    """
    F = np.zeros((order, order))
    Q = np.zeros((order, order))
    for i in range(order):
        for j in range(order):
            if j >= i:
                F[i, j] = dt ** (j - i) / math.factorial(j - i)
            n = 2 * order - 1 - i - j
            Q[i, j] = dt ** n / (n * math.factorial(order - 1 - i) * math.factorial(order - 1 - j))
    return F, Q


def _correct(P, r, R_meas):
    """Return (dx, P) of a Kalman update by a measurement of the first three states, residual r."""
    S = P[:3, :3] + R_meas
    K = np.linalg.solve(S, P[:3, :]).T  # P H^T S^-1, H = [I 0]
    I_KH = np.eye(P.shape[0])
    I_KH[:, :3] -= K
    P = I_KH @ P @ I_KH.T + K @ R_meas @ K.T  # Joseph form
    return K @ r, 0.5 * (P + P.T)


def _mahalanobis(P, r, R_meas):
    """Return sqrt(r^T S^-1 r), S the innovation covariance of a measurement of the first three."""
    return math.sqrt(max(float(r @ np.linalg.solve(P[:3, :3] + R_meas, r)), 0.0))


@dataclass
class PoseEstimate:
    t: float  # s, Vicon time
    p: np.ndarray  # m, world
    v: np.ndarray  # m/s, world
    a: np.ndarray  # m/s^2, world; zero for 'cv'
    R: np.ndarray  # world from body
    w: np.ndarray  # rad/s, body
    alpha: np.ndarray  # rad/s^2, body; zero for 'cv'
    P_trans: np.ndarray  # covariance of (p, v[, a])
    P_rot: np.ndarray  # covariance of (d, w[, alpha]), d the body-frame error of R

    def odometry_covariances(self):
        """Return the (pose, twist) 6x6 covariances in nav_msgs/Odometry's conventions.

        pose: position, then rotation about the world axes, in the world frame. twist: linear,
        then angular velocity, in the body frame. Translation and rotation are independent here.
        """
        R = self.R
        pose = np.zeros((6, 6))
        pose[:3, :3] = self.P_trans[:3, :3]
        pose[3:, 3:] = R @ self.P_rot[:3, :3] @ R.T  # body-frame error moved to the world
        twist = np.zeros((6, 6))
        twist[:3, :3] = R.T @ self.P_trans[3:6, 3:6] @ R  # world velocity moved to the body
        twist[3:, 3:] = self.P_rot[3:6, 3:6]
        return pose, twist


class PoseFilter:
    """Pose-only Kalman filter of one rigid body, with the 'cv' or 'ca' model described above."""

    def __init__(self, model, noise_psd, measurement_std, initial_std, gate_sigma,
                 max_rejections, reset_gap):
        """Build the filter.

        noise_psd: spectral densities of the white derivative, translation then rotation:
            m^2/s^3 and rad^2/s^3 for 'cv', m^2/s^5 and rad^2/s^5 for 'ca'.
        measurement_std: position (m) and orientation (rad).
        initial_std: velocity (m/s), angular velocity (rad/s), acceleration (m/s^2) and angular
            acceleration (rad/s^2) after a reset; 'ca' alone reads the last two.
        gate_sigma: Mahalanobis distance above which a frame is rejected.
        max_rejections: rejected frames in a row that reset the filter.
        reset_gap: time (s) between frames above which the filter restarts from the measurement.
        """
        if model not in ORDERS:
            raise ValueError(f'PoseFilter: model must be one of {sorted(ORDERS)}, got {model!r}')
        if len(noise_psd) != 2 or len(measurement_std) != 2 or len(initial_std) != 4:
            raise ValueError(
                'PoseFilter: noise_psd and measurement_std take 2 entries, initial_std 4')
        if min(noise_psd) < 0.0 or min(measurement_std) <= 0.0 or min(initial_std) <= 0.0:
            raise ValueError(
                'PoseFilter: noise_psd must be >= 0, measurement_std and initial_std > 0')
        self.order_ = ORDERS[model]
        self.q_trans_, self.q_rot_ = (float(q) for q in noise_psd)
        std_p, std_theta = (float(s) for s in measurement_std)
        std_v, std_w, std_a, std_alpha = (float(s) for s in initial_std)
        self.R_trans_ = std_p ** 2 * np.eye(3)  # m^2
        self.R_rot_ = std_theta ** 2 * np.eye(3)  # rad^2
        trans = np.square([std_p, std_v, std_a][:self.order_])
        rot = np.square([std_theta, std_w, std_alpha][:self.order_])
        self.P0_trans_ = np.diag(np.repeat(trans, 3))
        self.P0_rot_ = np.diag(np.repeat(rot, 3))
        self.gate_sigma_ = float(gate_sigma)
        self.max_rejections_ = int(max_rejections)
        self.reset_gap_ = float(reset_gap)
        self.initialized_ = False
        self.rejections_ = 0  # rejected frames in a row
        self.t_ = 0.0
        self.x_ = np.zeros(3 * self.order_)  # p, v[, a]: world
        self.P_trans_ = self.P0_trans_.copy()
        self.R_ = np.eye(3)  # world from body
        self.rates_ = np.zeros(3 * self.order_ - 3)  # w[, alpha]: body
        self.P_rot_ = self.P0_rot_.copy()

    @property
    def initialized(self):
        return self.initialized_

    def reset(self, t, p, R):
        """Restart at rest at time t (s) from the pose p (m, world) and R (world from body)."""
        self.t_ = float(t)
        self.x_ = np.zeros(3 * self.order_)
        self.x_[:3] = p
        self.P_trans_ = self.P0_trans_.copy()
        self.R_ = np.array(R, dtype=float)
        self.rates_ = np.zeros(3 * self.order_ - 3)
        self.P_rot_ = self.P0_rot_.copy()
        self.rejections_ = 0
        self.initialized_ = True

    def update(self, t, p, R):
        """Move to time t (s) and correct with the measured pose.

        Return 'reset', 'accepted', 'rejected' or 'stale'. A frame further than reset_gap from
        the last, either way (a gap, or a restarted Vicon clock), resets the filter.
        """
        if not self.initialized_ or abs(t - self.t_) > self.reset_gap_:
            self.reset(t, p, R)
            return 'reset'
        if t <= self.t_:
            return 'stale'
        x, P_trans, R_pred, rates, P_rot = self._propagate(t - self.t_)
        r_trans = np.asarray(p, dtype=float) - x[:3]  # m, world
        r_rot = so3_log(R_pred.T @ np.asarray(R, dtype=float))  # rad, body
        self.t_ = float(t)
        if max(_mahalanobis(P_trans, r_trans, self.R_trans_),
               _mahalanobis(P_rot, r_rot, self.R_rot_)) > self.gate_sigma_:
            self.rejections_ += 1
            if self.rejections_ >= self.max_rejections_:
                self.reset(t, p, R)
                return 'reset'
            self.x_, self.P_trans_, self.R_ = x, P_trans, R_pred
            self.rates_, self.P_rot_ = rates, P_rot
            return 'rejected'
        self.rejections_ = 0
        dx, self.P_trans_ = _correct(P_trans, r_trans, self.R_trans_)
        self.x_ = x + dx
        dx, self.P_rot_ = _correct(P_rot, r_rot, self.R_rot_)
        self.R_ = R_pred @ so3_exp(dx[:3])
        self.rates_ = rates + dx[3:]
        return 'accepted'

    def state(self):
        """Return the current estimate as a PoseEstimate of copies."""
        return self._estimate(self.t_, self.x_, self.P_trans_, self.R_, self.rates_, self.P_rot_)

    def predicted(self, t):
        """Return the estimate moved by the model to time t (s), never back; the filter is kept."""
        dt = max(t - self.t_, 0.0)
        return self._estimate(self.t_ + dt, *self._propagate(dt))

    def _propagate(self, dt):
        """Return (x, P_trans, R, rates, P_rot): the current estimate moved forward by dt (s)."""
        F1, Q1 = kinematic_blocks(self.order_, dt)
        F = np.kron(F1, np.eye(3))
        Q = np.kron(Q1, np.eye(3))
        x = F @ self.x_
        P_trans = F @ self.P_trans_ @ F.T + self.q_trans_ * Q
        rates = F[3:, 3:] @ self.rates_  # w (+ alpha dt)[, alpha]
        phi = F[:3, 3:] @ self.rates_  # rad, body: w dt (+ alpha dt^2 / 2)
        F[:3, :3] = so3_exp(phi).T
        F[:3, 3:] = so3_right_jacobian(phi) @ F[:3, 3:]
        P_rot = F @ self.P_rot_ @ F.T + self.q_rot_ * Q
        return x, P_trans, self.R_ @ so3_exp(phi), rates, P_rot

    def _estimate(self, t, x, P_trans, R, rates, P_rot):
        zero = np.zeros(3)
        ca = self.order_ == 3
        return PoseEstimate(
            t=float(t), p=x[:3].copy(), v=x[3:6].copy(), a=x[6:9].copy() if ca else zero.copy(),
            R=R.copy(), w=rates[:3].copy(), alpha=rates[3:6].copy() if ca else zero.copy(),
            P_trans=P_trans.copy(), P_rot=P_rot.copy())
