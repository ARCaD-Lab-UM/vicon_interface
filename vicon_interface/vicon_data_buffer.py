import time
from threading import Lock
import numpy as np
import tf_transformations


class ViconDataBuffer:
    def __init__(self, state_dim, rolling_window_size, offset_xyz=None, offset_R=None) -> None:
        """Rolling window of the reported frame's poses, and the tracked rigid body's latest pose.

        offset_xyz / offset_R describe the pose of the frame to report relative to the tracked rigid
        body's own frame (T_marker->child): the child origin expressed in marker coordinates,
        and the child orientation as a rotation matrix. They are applied per sample.
        """
        self.past_poses_ = np.zeros((0, state_dim))
        self.past_times_ = np.zeros((0, 1))
        self.rolling_window_size_ = rolling_window_size
        self.lock_ = Lock()

        self.offset_xyz_ = (
            np.zeros(3) if offset_xyz is None else np.asarray(offset_xyz, dtype=float)
        )
        self.offset_R_ = np.eye(3) if offset_R is None else np.asarray(offset_R, dtype=float)
        self.has_offset_ = bool(
            np.any(self.offset_xyz_) or not np.allclose(self.offset_R_, np.eye(3))
        )

        self.marker_p_ = None  # m, the rigid body's origin in the world, for ViconDataFilter
        self.marker_R_ = None  # the rigid body's orientation in the world

    def add_pose(self, pose, latency):
        """Buffer a pose (mm, sxyz rad) of the rigid body, captured latency (s) ago."""
        with self.lock_:
            time_stamp = time.monotonic() - latency
            x, y, z, roll, pitch, yaw = pose
            x /= 1000.0
            y /= 1000.0
            z /= 1000.0

            R = tf_transformations.euler_matrix(roll, pitch, yaw, "sxyz")[:3, :3]
            self.marker_p_ = np.array([x, y, z])
            self.marker_R_ = R
            if self.has_offset_:  # marker -> child frame
                x, y, z = np.array([x, y, z]) + R @ self.offset_xyz_
                R = R @ self.offset_R_
                m = np.eye(4)
                m[:3, :3] = R
                roll, pitch, yaw = tf_transformations.euler_from_matrix(m, "sxyz")
            if len(self.past_poses_) == self.rolling_window_size_:
                self.past_poses_ = self.past_poses_[1:]
                self.past_times_ = self.past_times_[1:]
            self.past_poses_ = np.vstack(
                (self.past_poses_, np.array([x, y, z, roll, pitch, yaw]))
            )
            self.past_times_ = np.append(self.past_times_, time_stamp)

    def get_latest_pose(self):
        with self.lock_:
            if len(self.past_poses_) == 0:
                return np.zeros(6), 0
            return self.past_poses_[-1].copy(), self.past_times_[-1]

    def is_ready(self):
        with self.lock_:
            return len(self.past_poses_) == self.rolling_window_size_
