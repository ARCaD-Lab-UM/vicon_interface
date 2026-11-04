import time

import numpy as np

from vicon_interface.state_estimators import ViconLinearStateEstimator
from vicon_interface.utils import Daemon, DaemonConfig
from vicon_interface.vicon_data_buffer import ViconDataBuffer


class ViconDataFilter(ViconDataBuffer):
    """Rolling buffer plus a constant-velocity Kalman filter and a watchdog (see AGENTS.md)."""

    def __init__(self, state_dim, rolling_window_size, offset_xyz=None, offset_R=None, *,
                 process_noise, measurement_noise, initial_covariance, max_delay, predict_dt,
                 offline_timeout, online_jitter_time, owner_id=None):
        super().__init__(state_dim, rolling_window_size, offset_xyz, offset_R)
        self.kf_ = ViconLinearStateEstimator(
            predict_dt,
            np.diag(np.asarray(process_noise, dtype=float)),
            np.diag(np.asarray(measurement_noise, dtype=float)),
            np.diag(np.asarray(initial_covariance, dtype=float)),
        )
        self.daemon_ = Daemon(DaemonConfig(
            set_online_jitter_time_ms=1000.0 * online_jitter_time,
            set_offline_time_ms=1000.0 * offline_timeout,
            owner_id=owner_id,
        ))
        self.max_delay_ = float(max_delay)  # s, cap on the move of a measurement to now
        self.kf_initialized_ = False
        self.kf_needs_reset_ = True
        self.frames_since_online_ = 0
        self.kf_resets_ = 0

    def add_pose(self, pose, latency):
        super().add_pose(pose, latency)
        with self.lock_:
            now = time.monotonic()
            now_ms = 1000.0 * now
            stamp = self.past_times_[-1]
            # s, frame age (Vicon latency plus processing delay since the grab), capped
            d = min(max(now - stamp, 0.0), self.max_delay_)
            self.daemon_.update(now_ms)  # catches a gap even if the timer did not run
            if self.daemon_.is_offline():
                self.frames_since_online_ = 0
                self.kf_needs_reset_ = True
            self.daemon_.reload(now_ms)
            self.frames_since_online_ += 1
            if self.frames_since_online_ < self.rolling_window_size_:
                # the window still holds pre-gap samples, or the zero velocity of a short window
                return
            v_world = self.R_child_ @ self.v_  # m/s, world; undoes the base's R.T
            p_now = self.past_poses_[-1, :3] + v_world * d  # m, moved to now, a = 0
            y = np.concatenate((p_now, v_world))
            if not np.all(np.isfinite(y)):
                return
            if self.kf_needs_reset_ or not np.all(np.isfinite(self.kf_.get_state())):
                self.kf_.reset(y)
                self.kf_resets_ += 1
                self.kf_needs_reset_ = False
                self.kf_initialized_ = True
            else:
                self.kf_.correct(y)

    def update(self, now=None):
        """Run one timer step: watchdog update and predict; True when the filter may publish."""
        with self.lock_:
            if now is None:
                now = time.monotonic()
            self.daemon_.update(1000.0 * now)
            if self.daemon_.is_offline():
                self.frames_since_online_ = 0
                self.kf_needs_reset_ = True
                return False
            if not self.kf_initialized_ or self.kf_needs_reset_:
                return False
            self.kf_.predict(np.zeros(3))
            return not self.daemon_.is_error()

    def get_filtered_state(self):
        """Return copies: world p and v, latest raw rpy, body Euler rate, frame stamp."""
        with self.lock_:
            x = self.kf_.get_state()
            if len(self.past_poses_) == 0:
                return x[:3], x[3:], np.zeros(3), np.zeros(3), 0
            rpy = self.past_poses_[-1, 3:].copy()
            return x[:3], x[3:], rpy, self.a_.copy(), self.past_times_[-1]

    def is_online(self):
        with self.lock_:
            return self.daemon_.is_online()

    def is_offline(self):
        with self.lock_:
            return self.daemon_.is_offline()
