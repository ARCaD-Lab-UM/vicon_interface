"""Kalman filter of a Vicon rigid body's pose, without an IMU.

Position lives in the world frame, orientation on SO(3) with its error in the body frame,
R = R_hat exp([d]x). Each axis carries its value and its rate, constant up to a white
acceleration of a given spectral density, so the process noise follows the true frame interval.
Only the pose is measured.
"""
from dataclasses import dataclass
import math

import numpy as np


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


def kinematic_blocks(dt):
    """Return (F, Q) of one axis over dt (s): its value and its rate.

    The rate's derivative is white noise of unit spectral density.
    """
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = np.array([[dt ** 3 / 3.0, dt ** 2 / 2.0], [dt ** 2 / 2.0, dt]])
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
    R: np.ndarray  # world from body
    w: np.ndarray  # rad/s, body
    P_trans: np.ndarray  # covariance of (p, v)
    P_rot: np.ndarray  # covariance of (d, w), d the body-frame error of R
    P_cross: np.ndarray = None  # covariance of (p, v) with (d, w); None is zero

    def transported(self, offset_xyz, offset_R):
        """Return the estimate of a frame fixed on the body.

        The frame's origin is offset_xyz (m, body frame), its orientation offset_R (body from
        frame). The rigid-body formulas move the state: p + R d, v + R (w x d), R offset_R and
        offset_R^T w. Their first-order map moves the covariance, so the new frame's translation
        and rotation errors are correlated.
        """
        d = np.asarray(offset_xyz, dtype=float)
        R, w = self.R, self.w
        wd = np.cross(w, d)
        Sigma = np.zeros((12, 12))
        Sigma[:6, :6], Sigma[6:, 6:] = self.P_trans, self.P_rot
        if self.P_cross is not None:
            Sigma[:6, 6:], Sigma[6:, :6] = self.P_cross, self.P_cross.T
        J = np.eye(12)  # errors of the new frame on the body's, rows on columns
        J[0:3, 6:9] = -R @ skew(d)  # R exp(d theta) d = R d - R [d]x d theta
        J[3:6, 6:9] = -R @ skew(wd)
        J[3:6, 9:12] = -R @ skew(d)
        J[6:9, 6:9] = J[9:12, 9:12] = offset_R.T
        P = J @ Sigma @ J.T
        return PoseEstimate(
            t=self.t, p=self.p + R @ d, v=self.v + R @ wd, R=R @ offset_R, w=offset_R.T @ w,
            P_trans=P[:6, :6], P_rot=P[6:, 6:], P_cross=P[:6, 6:])

    def odometry_covariances(self):
        """Return the (pose, twist) 6x6 covariances in nav_msgs/Odometry's conventions.

        pose: position, then rotation about the world axes, in the world frame. twist: linear,
        then angular velocity, in the body frame.
        """
        R = self.R
        pose = np.zeros((6, 6))
        pose[:3, :3] = self.P_trans[:3, :3]
        pose[3:, 3:] = R @ self.P_rot[:3, :3] @ R.T  # body-frame error moved to the world
        twist = np.zeros((6, 6))
        twist[:3, :3] = R.T @ self.P_trans[3:6, 3:6] @ R  # world velocity moved to the body
        twist[3:, 3:] = self.P_rot[3:6, 3:6]
        if self.P_cross is not None:
            pose[:3, 3:] = self.P_cross[:3, :3] @ R.T
            pose[3:, :3] = pose[:3, 3:].T
            twist[:3, 3:] = R.T @ self.P_cross[3:6, 3:6]
            twist[3:, :3] = twist[:3, 3:].T
        return pose, twist


class PoseFilter:
    """Pose-only Kalman filter of one rigid body, with the constant-velocity model above."""

    def __init__(self, noise_psd, measurement_std, initial_std, gate, max_rejections, reset_gap,
                 rot_huber_k=0.0):
        """Build the filter.

        noise_psd: spectral densities of the white acceleration, m^2/s^3 for the translation and
            rad^2/s^3 for the rotation.
        measurement_std: position (m) and orientation (rad).
        initial_std: velocity (m/s) and angular velocity (rad/s) after a reset.
        gate: position (m) and rotation (rad) innovations above which a frame is rejected, as a
            marker swap or a flipped fit; motion, however sharp, stays well below them.
        max_rejections: rejected frames in a row that reset the filter.
        reset_gap: time (s) between frames above which the filter restarts from the measurement.
        rot_huber_k: Mahalanobis distance above which a rotation residual's measurement variance
            grows by distance / rot_huber_k (a Huber weight), so a glitch pulls less; 0 is off.
        """
        if len(noise_psd) != 2 or len(measurement_std) != 2 or len(initial_std) != 2:
            raise ValueError(
                'PoseFilter: noise_psd, measurement_std and initial_std take 2 entries each')
        if min(noise_psd) < 0.0 or min(measurement_std) <= 0.0 or min(initial_std) <= 0.0:
            raise ValueError(
                'PoseFilter: noise_psd must be >= 0, measurement_std and initial_std > 0')
        if len(gate) != 2 or min(gate) <= 0.0 or rot_huber_k < 0.0:
            raise ValueError('PoseFilter: gate takes 2 entries > 0, rot_huber_k must be >= 0')
        self.q_trans_, self.q_rot_ = (float(q) for q in noise_psd)
        std_p, std_theta = (float(s) for s in measurement_std)
        std_v, std_w = (float(s) for s in initial_std)
        self.R_trans_ = std_p ** 2 * np.eye(3)  # m^2
        self.R_rot_ = std_theta ** 2 * np.eye(3)  # rad^2
        self.P0_trans_ = np.diag(np.repeat(np.square([std_p, std_v]), 3))
        self.P0_rot_ = np.diag(np.repeat(np.square([std_theta, std_w]), 3))
        self.gate_position_, self.gate_rotation_ = (float(g) for g in gate)  # m, rad
        self.rot_huber_k_ = float(rot_huber_k)
        self.max_rejections_ = int(max_rejections)
        self.reset_gap_ = float(reset_gap)
        self.initialized_ = False
        self.rejections_ = 0  # rejected frames in a row
        self.t_ = 0.0
        self.x_ = np.zeros(6)  # p, v: world
        self.P_trans_ = self.P0_trans_.copy()
        self.R_ = np.eye(3)  # world from body
        self.w_ = np.zeros(3)  # rad/s, body
        self.P_rot_ = self.P0_rot_.copy()

    @property
    def initialized(self):
        return self.initialized_

    def reset(self, t, p, R):
        """Restart at rest at time t (s) from the pose p (m, world) and R (world from body)."""
        self.t_ = float(t)
        self.x_ = np.zeros(6)
        self.x_[:3] = p
        self.P_trans_ = self.P0_trans_.copy()
        self.R_ = np.array(R, dtype=float)
        self.w_ = np.zeros(3)
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
        x, P_trans, R_pred, w, P_rot = self._propagate(t - self.t_)
        r_trans = np.asarray(p, dtype=float) - x[:3]  # m, world
        r_rot = so3_log(R_pred.T @ np.asarray(R, dtype=float))  # rad, body
        self.t_ = float(t)
        if (np.linalg.norm(r_trans) > self.gate_position_
                or np.linalg.norm(r_rot) > self.gate_rotation_):
            self.rejections_ += 1
            if self.rejections_ >= self.max_rejections_:
                self.reset(t, p, R)
                return 'reset'
            self.x_, self.P_trans_, self.R_ = x, P_trans, R_pred
            self.w_, self.P_rot_ = w, P_rot
            return 'rejected'
        self.rejections_ = 0
        dx, self.P_trans_ = _correct(P_trans, r_trans, self.R_trans_)
        self.x_ = x + dx
        R_rot = self.R_rot_
        if self.rot_huber_k_ > 0.0:
            distance = _mahalanobis(P_rot, r_rot, R_rot)
            if distance > self.rot_huber_k_:
                R_rot = R_rot * (distance / self.rot_huber_k_)  # weight k / distance
        dx, self.P_rot_ = _correct(P_rot, r_rot, R_rot)
        self.R_ = R_pred @ so3_exp(dx[:3])
        self.w_ = w + dx[3:]
        return 'accepted'

    def state(self):
        """Return the current estimate as a PoseEstimate of copies."""
        return self._estimate(self.t_, self.x_, self.P_trans_, self.R_, self.w_, self.P_rot_)

    def predicted(self, t):
        """Return the estimate moved by the model to time t (s), never back; the filter is kept."""
        dt = max(t - self.t_, 0.0)
        return self._estimate(self.t_ + dt, *self._propagate(dt))

    def _propagate(self, dt):
        """Return (x, P_trans, R, w, P_rot): the current estimate moved forward by dt (s)."""
        F1, Q1 = kinematic_blocks(dt)
        F = np.kron(F1, np.eye(3))
        Q = np.kron(Q1, np.eye(3))
        x = F @ self.x_
        P_trans = F @ self.P_trans_ @ F.T + self.q_trans_ * Q
        phi = dt * self.w_  # rad, body
        F[:3, :3] = so3_exp(phi).T
        F[:3, 3:] = dt * so3_right_jacobian(phi)
        P_rot = F @ self.P_rot_ @ F.T + self.q_rot_ * Q
        return x, P_trans, self.R_ @ so3_exp(phi), self.w_.copy(), P_rot

    def _estimate(self, t, x, P_trans, R, w, P_rot):
        return PoseEstimate(
            t=float(t), p=x[:3].copy(), v=x[3:6].copy(), R=R.copy(), w=w.copy(),
            P_trans=P_trans.copy(), P_rot=P_rot.copy())
