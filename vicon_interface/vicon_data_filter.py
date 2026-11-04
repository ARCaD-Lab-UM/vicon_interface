from collections import deque
import time

import numpy as np

from vicon_interface.state_estimators import PoseFilter
from vicon_interface.utils import Daemon, DaemonConfig
from vicon_interface.vicon_data_buffer import ViconDataBuffer


class ViconDataFilter(ViconDataBuffer):
    """Rolling buffer, a pose Kalman filter on the Vicon clock and a watchdog (see AGENTS.md)."""

    def __init__(self, state_dim, rolling_window_size, offset_xyz=None, offset_R=None, *,
                 motion_model, noise_psd, measurement_std, initial_std, gate, max_rejections,
                 max_delay, clock_window, offline_timeout, online_jitter_time, rot_huber_k=0.0,
                 owner_id=None):
        super().__init__(state_dim, rolling_window_size, offset_xyz, offset_R)
        self.kf_ = PoseFilter(motion_model, noise_psd, measurement_std, initial_std, gate,
                              max_rejections, offline_timeout, rot_huber_k)
        self.daemon_ = Daemon(DaemonConfig(
            set_online_jitter_time_ms=1000.0 * online_jitter_time,
            set_offline_time_ms=1000.0 * offline_timeout,
            owner_id=owner_id,
        ))
        self.max_delay_ = float(max_delay)  # s, cap on the prediction from the last frame to now
        self.max_frame_step_ = float(offline_timeout)  # s of Vicon time; a longer step resets
        self.clock_window_ = float(clock_window)  # s of host time behind the clock offset minimum
        self.clock_offsets_ = deque()  # (host s, host minus Vicon s), offsets increasing
        self.frame_number_ = None  # Vicon frame number of the last fused frame
        self.frame_time_ = 0.0  # s, Vicon time of the last fused frame, 0 at the last reset
        self.frame_clock_ok_ = True  # False while frames come without a usable frame rate
        self.kf_initialized_ = False
        self.kf_needs_reset_ = True
        self.frames_since_online_ = 0
        self.kf_resets_ = 0
        self.kf_rejections_ = 0  # frames the filter's gate rejected
        self.fused_ = False  # the latest add_pose stepped the filter

    def add_pose(self, pose, latency, frame_number=None, frame_rate=None):
        """Buffer a pose, step the filter by the frame-number step over frame_rate, then fuse."""
        clock_ok = frame_number is not None and frame_rate is not None and frame_rate > 0.0
        super().add_pose(pose, latency, frame_number / frame_rate if clock_ok else None)
        with self.lock_:
            self.fused_ = False
            now = time.monotonic()
            now_ms = 1000.0 * now
            self.daemon_.update(now_ms)  # catches a gap even if the timer did not run
            if self.daemon_.is_offline():
                self.frames_since_online_ = 0
                self.kf_needs_reset_ = True
            self.daemon_.reload(now_ms)
            self.frames_since_online_ += 1
            self.frame_clock_ok_ = clock_ok
            if not self.frame_clock_ok_:
                return  # no Vicon time for this frame: the filter is not stepped
            # The rigid body itself: its estimate moves to the child at the output. Fed the child,
            # the filter would take orientation noise times the offset arm as position noise.
            p = self.marker_p_  # m, the rigid body's origin in the world
            R = self.marker_R_  # the rigid body's orientation in the world
            if not (np.all(np.isfinite(p)) and np.all(np.isfinite(R))):
                return
            step = None
            if self.frame_number_ is not None:
                step = (frame_number - self.frame_number_) / frame_rate  # s, Vicon clock
                if step == 0.0:
                    return  # the same frame twice
            if self.kf_needs_reset_ or step is None or not 0.0 < step <= self.max_frame_step_:
                self.frame_time_ = 0.0
                self.kf_.reset(self.frame_time_, p, R)  # at rest, with the initial covariance
                self.clock_offsets_.clear()  # the Vicon frame count may have restarted
                self.kf_resets_ += 1
                self.kf_needs_reset_ = False
                self.kf_initialized_ = True
            else:
                self.frame_time_ += step  # a dropped frame is one longer step
                outcome = self.kf_.update(self.frame_time_, p, R)
                if outcome == 'rejected':
                    self.kf_rejections_ += 1
                elif outcome == 'reset':  # too many rejected in a row: restarted at the frame
                    self.kf_resets_ += 1
                    self.frames_since_online_ = 1
            self.frame_number_ = frame_number
            # capture time on the host clock (frame stamp) minus the Vicon time of the same frame
            self._add_clock_offset(now, self.past_times_[-1] - self.frame_time_)
            self.fused_ = True

    def _add_clock_offset(self, now, offset):
        """Keep the minimum host-minus-Vicon offset over clock_window: delays only add to it."""
        while self.clock_offsets_ and self.clock_offsets_[-1][1] >= offset:
            self.clock_offsets_.pop()
        self.clock_offsets_.append((now, offset))
        while now - self.clock_offsets_[0][0] > self.clock_window_:
            self.clock_offsets_.popleft()

    def update(self, now=None):
        """Run one timer step of the watchdog; True when the filter may publish."""
        with self.lock_:
            if now is None:
                now = time.monotonic()
            self.daemon_.update(1000.0 * now)
            if self.daemon_.is_offline():
                self.frames_since_online_ = 0
                self.kf_needs_reset_ = True
                return False
            if not self.kf_initialized_ or self.kf_needs_reset_ or not self.frame_clock_ok_:
                return False  # without a frame rate, frames still come but the filter stands still
            if self.frames_since_online_ < self.rolling_window_size_:
                return False  # the filter has fused only a few post-gap frames
            return not self.daemon_.is_error()

    def get_filtered_state(self, now=None):
        """Return (estimate, stamp): the child frame's PoseEstimate at now and the frame's capture.

        The filter stays at its last fused frame; the estimate is predicted from there to now
        (host monotonic time) along the model, at most max_delay ahead, then moved to the child.
        stamp is that frame's capture time on the host monotonic clock. (None, 0) before the
        first fused frame.
        """
        with self.lock_:
            if not self.kf_initialized_ or not self.clock_offsets_:
                return None, 0
            if now is None:
                now = time.monotonic()
            stamp = self.frame_time_ + self.clock_offsets_[0][1]  # s, host monotonic
            ahead = min(max(now - stamp, 0.0), self.max_delay_)  # s
            estimate = self.kf_.predicted(self.frame_time_ + ahead)
            return estimate.transported(self.offset_xyz_, self.offset_R_), stamp

    def get_frame_state(self):
        """Return (estimate, stamp) for the latest frame, as odometry/<obj> reads it.

        stamp is the frame's capture time on the host monotonic clock, from the Vicon clock and
        the minimum offset; None when the frame was not fused. estimate is the filter's
        PoseEstimate at that frame, moved to the child; None as well while the filter has fused
        fewer than rolling_window_size frames since it started.
        """
        with self.lock_:
            if not self.fused_:
                return None, None
            stamp = self.frame_time_ + self.clock_offsets_[0][1]  # s, host monotonic
            if self.frames_since_online_ < self.rolling_window_size_:
                return None, stamp
            return self.kf_.state().transported(self.offset_xyz_, self.offset_R_), stamp

    def has_frame_clock(self):
        with self.lock_:
            return self.frame_clock_ok_

    def is_online(self):
        with self.lock_:
            return self.daemon_.is_online()

    def is_offline(self):
        with self.lock_:
            return self.daemon_.is_offline()
