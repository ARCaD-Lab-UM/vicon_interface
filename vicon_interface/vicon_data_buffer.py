import time
from threading import Lock
import numpy as np
import tf_transformations

from vicon_interface.state_estimators.pose_filter import so3_log


class ViconDataBuffer:
    def __init__(self, state_dim, rolling_window_size, offset_xyz=None, offset_R=None) -> None:
        """Rolling pose buffer with finite-difference velocity.

        offset_xyz / offset_R describe the pose of the frame to report relative to the tracked rigid
        body's own frame (T_marker->child): the child origin expressed in marker coordinates,
        and the child orientation as a rotation matrix. They are applied per sample BEFORE the
        finite difference, so the reported velocity is the child frame's velocity.
        """
        self.past_poses_ = np.zeros((0, state_dim))
        self.past_times_ = np.zeros((0, 1))
        self.past_frame_times_ = np.zeros(0)  # s, Vicon clock; NaN where a frame had none
        self.rolling_window_size_ = rolling_window_size
        self.lock_ = Lock()

        self.offset_xyz_ = (
            np.zeros(3) if offset_xyz is None else np.asarray(offset_xyz, dtype=float)
        )
        self.offset_R_ = np.eye(3) if offset_R is None else np.asarray(offset_R, dtype=float)
        self.has_offset_ = bool(
            np.any(self.offset_xyz_) or not np.allclose(self.offset_R_, np.eye(3))
        )

        self.v_ = None
        self.a_ = None
        self.R_child_ = None

    def add_pose(self, pose, latency, frame_time=None):
        """Buffer a pose captured latency (s) ago, at frame_time (s) on the Vicon clock if given.

        The finite difference is timed by the Vicon clock when every pose in the window has a
        frame time, by the host capture times otherwise. A frame time that does not increase
        (a restarted Vicon) starts the window over.
        """
        with self.lock_:
            time_stamp = time.monotonic() - latency
            frame_time = np.nan if frame_time is None else float(frame_time)
            if len(self.past_frame_times_) and frame_time <= self.past_frame_times_[-1]:
                self.past_poses_ = self.past_poses_[:0]  # NaN on either side compares False
                self.past_times_ = self.past_times_[:0]
                self.past_frame_times_ = self.past_frame_times_[:0]
            x, y, z, roll, pitch, yaw = pose
            x /= 1000.0
            y /= 1000.0
            z /= 1000.0

            # marker -> child frame, applied here so the finite difference below sees the
            # child's trajectory.
            # R is used to express the velocity in the child's own frame; in CHILD orientation, not the marker's.
            R = tf_transformations.euler_matrix(roll, pitch, yaw, "sxyz")[:3, :3]
            if self.has_offset_:
                x, y, z = np.array([x, y, z]) + R @ self.offset_xyz_
                R = R @ self.offset_R_
                m = np.eye(4)
                m[:3, :3] = R
                roll, pitch, yaw = tf_transformations.euler_from_matrix(m, "sxyz")
            self.R_child_ = R  # child orientation in the world frame, for ViconDataFilter
            if len(self.past_poses_) == self.rolling_window_size_:
                self.past_poses_ = self.past_poses_[1:]
                self.past_times_ = self.past_times_[1:]
                self.past_frame_times_ = self.past_frame_times_[1:]
            self.past_poses_ = np.vstack(
                (self.past_poses_, np.array([x, y, z, roll, pitch, yaw]))
            )
            self.past_times_ = np.append(self.past_times_, time_stamp)
            self.past_frame_times_ = np.append(self.past_frame_times_, frame_time)

            if len(self.past_poses_) == self.rolling_window_size_:
                times = self.past_frame_times_
                if np.any(np.isnan(times)):
                    times = self.past_times_  # a frame without Vicon time: the host clock
                dt = times[-1] - times[0]  # s, across the window
                self.v_ = (self.past_poses_[-1, :3] - self.past_poses_[0, :3]) / dt
                R_first = tf_transformations.euler_matrix(*self.past_poses_[0, 3:], "sxyz")
                self.a_ = so3_log(R_first[:3, :3].T @ R) / dt  # rad/s, body, across the window
            else:
                self.v_ = np.zeros(3)
                self.a_ = np.zeros(3)

            self.v_ = np.dot(R.T, self.v_)

    def get_latest_pose(self):
        with self.lock_:
            if len(self.past_poses_) == 0:
                return np.zeros(6), 0
            return self.past_poses_[-1].copy(), self.past_times_[-1]

    def get_latest_velocity(self):
        with self.lock_:
            if self.v_ is None:
                return np.zeros(3), np.zeros(3), 0
            return self.v_.copy(), self.a_.copy(), self.past_times_[-1]

    def is_ready(self):
        with self.lock_:
            return len(self.past_poses_) == self.rolling_window_size_
