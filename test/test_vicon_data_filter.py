"""Behavioural tests for ViconDataFilter with a fake clock; no ROS needed."""
import time

import numpy as np
import pytest
import tf_transformations

from vicon_interface.vicon_data_buffer import ViconDataBuffer
from vicon_interface.vicon_data_filter import ViconDataFilter

DT = 0.004  # Vicon frame interval and timer period, s
WINDOW = 5
PARAMS = {
    'process_noise': [1e-8] * 3 + [1.44e-4] * 3,
    'measurement_noise': [4e-8] * 3 + [2.5e-4] * 3,
    'initial_covariance': [1e-4] * 3 + [1.0] * 3,
    'max_delay': 0.02,
    'predict_dt': DT,
    'offline_timeout': 0.1,
    'online_jitter_time': 0.0,
    'owner_id': 'test',
}


class FakeClock:

    def __init__(self):
        self.t = 100.0

    def monotonic(self):
        return self.t


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(time, 'monotonic', fake.monotonic)
    return fake


def make_filter(**overrides):
    return ViconDataFilter(6, WINDOW, **{**PARAMS, **overrides})


def node_angles(roll, pitch, yaw):
    """Do what vicon_stream_node does with the SDK's XYZ Euler triple."""
    q = tf_transformations.quaternion_from_euler(roll, pitch, yaw, 'rxyz')
    return tf_transformations.euler_from_quaternion(q, 'sxyz')


def feed(f, clock, t_frame, pos_m, rpy, latency=0.0):
    """Run one timer tick, then deliver the frame captured at t_frame, latency seconds later."""
    clock.t = 100.0 + t_frame + latency - DT / 2
    f.update()
    clock.t = 100.0 + t_frame + latency
    r, p, y = node_angles(*rpy)
    f.add_pose((1000.0 * pos_m[0], 1000.0 * pos_m[1], 1000.0 * pos_m[2], r, p, y), latency)


def test_startup_waits_for_a_full_window_then_resets_the_filter(clock):
    f = make_filter()
    assert f.update() is False
    v = np.array([0.3, -0.1, 0.05])
    for k in range(WINDOW - 1):
        feed(f, clock, k * DT, v * k * DT, (0.0, 0.0, 0.0))
        assert f.update() is False and f.kf_resets_ == 0
    feed(f, clock, (WINDOW - 1) * DT, v * (WINDOW - 1) * DT, (0.0, 0.0, 0.0))
    assert f.kf_resets_ == 1
    p, vw, rpy, a, stamp = f.get_filtered_state()
    assert np.allclose(vw, v, atol=1e-9)  # the reset takes the window velocity verbatim
    assert np.allclose(p, v * (WINDOW - 1) * DT, atol=1e-9)  # latency 0: nothing to move
    assert stamp == clock.t
    assert f.update() is True
    p[:] = 99.0  # the getter hands out copies
    assert not np.any(f.kf_.get_state() == 99.0)


def test_constant_velocity_is_tracked_exactly(clock):
    f = make_filter()
    v = np.array([0.3, -0.1, 0.05])
    for k in range(200):
        feed(f, clock, k * DT, v * k * DT, (0.1, -0.2, 0.5))
    p, vw, _, _, _ = f.get_filtered_state()
    assert np.allclose(vw, v, atol=1e-4)
    assert np.allclose(p, v * 199 * DT, atol=1e-4)


def test_latency_is_compensated_in_the_measurement(clock):
    f = make_filter()
    v = np.array([0.3, -0.1, 0.05])
    latency = 0.010
    for k in range(200):
        feed(f, clock, k * DT, v * k * DT, (0.0, 0.0, 0.0), latency=latency)
    now = clock.t - 100.0  # the last frame arrived latency after its capture
    p, vw, _, _, stamp = f.get_filtered_state()
    assert np.allclose(p, v * now, atol=2e-4)  # position at now, not at the frame time
    assert np.isclose(stamp, 100.0 + 199 * DT)


def test_the_measurement_move_is_capped_at_max_delay(clock):
    f = make_filter()
    v = np.array([0.3, -0.1, 0.05])
    latency = 0.05  # well above the 20 ms cap
    for k in range(200):
        feed(f, clock, k * DT, v * k * DT, (0.0, 0.0, 0.0), latency=latency)
    p, vw, _, _, stamp = f.get_filtered_state()
    assert np.allclose(p, v * (199 * DT + 0.02), atol=2e-4)  # moved by the cap, not the latency
    assert np.array_equal(f.kf_.R, np.diag(PARAMS['measurement_noise']))  # R is never rescaled


def test_sinusoid_velocity_error_is_small_and_close_to_the_finite_difference(clock):
    f = make_filter()
    err_kf, err_fd = [], []
    for k in range(500):
        t = k * DT
        pos = np.array([0.1 * np.sin(2 * np.pi * t), 0.0, 0.02 * np.sin(4 * np.pi * t)])
        feed(f, clock, t, pos, (0.0, 0.0, 0.0))
        if t >= 1.0:
            v_true = np.array([0.2 * np.pi * np.cos(2 * np.pi * t), 0.0,
                               0.08 * np.pi * np.cos(4 * np.pi * t)])
            _, vw, _, _, _ = f.get_filtered_state()
            v_fd, _, _ = f.get_latest_velocity()
            err_kf.append(np.linalg.norm(vw - v_true))
            err_fd.append(np.linalg.norm(v_fd - v_true))
    rms_kf = np.sqrt(np.mean(np.square(err_kf)))
    rms_fd = np.sqrt(np.mean(np.square(err_fd)))
    print(f'sinusoid velocity RMS error: kf {rms_kf:.4f} m/s, finite difference {rms_fd:.4f} m/s')
    assert rms_kf < 0.05
    assert rms_kf <= 1.5 * rms_fd


def test_static_object_filtered_velocity_is_quieter_than_the_finite_difference(clock):
    f = make_filter()
    rng = np.random.default_rng(0)
    v_kf, v_fd = [], []
    for k in range(2000):
        pos = np.array([1.0, 2.0, 0.5]) + 1e-4 * rng.standard_normal(3)  # 0.1 mm white noise
        feed(f, clock, k * DT, pos, (0.0, 0.0, 0.0))
        if k >= 100:
            v_kf.append(f.get_filtered_state()[1])
            v_fd.append(f.get_latest_velocity()[0])
    std_kf = np.std(np.array(v_kf), axis=0)
    std_fd = np.std(np.array(v_fd), axis=0)
    print(f'static velocity std: kf {std_kf}, finite difference {std_fd}')
    assert np.all(std_kf < std_fd)


def test_yaw_wrap_does_not_disturb_the_filtered_velocity(clock):
    f = make_filter()
    v = np.array([0.3, 0.0, 0.0])
    rate = 2.0  # rad/s; yaw passes +pi at t = 0.05 s
    for k in range(120):
        t = k * DT
        feed(f, clock, t, v * t, (0.0, 0.0, np.pi - 0.1 + rate * t))
        if k >= WINDOW + 2:
            _, vw, rpy, a, _ = f.get_filtered_state()
            assert np.allclose(vw, v, atol=1e-3)
            R = tf_transformations.euler_matrix(*rpy, 'sxyz')[:3, :3]
            assert np.allclose(R.T @ vw, R.T @ v, atol=1e-3)
            assert abs(a[2] - rate) < 1e-6


def test_offline_gap_pauses_the_filter_and_resets_it_on_return(clock):
    f = make_filter()
    v = np.array([0.3, -0.1, 0.05])
    for k in range(50):
        feed(f, clock, k * DT, v * k * DT, (0.0, 0.0, 0.0))
    assert f.kf_resets_ == 1 and f.update() is True
    t_gap = clock.t
    results = []
    for i in range(1, 126):  # 0.5 s of timer ticks without a frame
        clock.t = t_gap + i * DT
        results.append(f.update())
    assert all(results[:25])  # up to 100 ms: still online, free-running
    assert not any(results[25:])  # after the timeout
    frozen = f.kf_.get_state()
    clock.t += DT
    assert f.update() is False
    assert np.array_equal(f.kf_.get_state(), frozen)
    assert f.is_offline()
    t0 = clock.t - 100.0 + DT  # frames resume, continuing the same motion
    for j in range(40):
        t = t0 + j * DT
        feed(f, clock, t, v * t, (0.0, 0.0, 0.0))
        if j < WINDOW - 1:
            assert f.update() is False and f.kf_resets_ == 1
    assert f.kf_resets_ == 2 and f.update() is True
    p, vw, _, _, _ = f.get_filtered_state()
    assert np.allclose(vw, v, atol=1e-3)
    assert np.allclose(p, v * (t0 + 40 * DT), atol=1e-3)  # one predict step after the last frame


def test_a_nan_frame_is_skipped_and_the_filter_recovers(clock):
    f = make_filter()
    v = np.array([0.3, -0.1, 0.05])
    for k in range(50):
        pos = v * k * DT
        if k == 30:
            pos = np.array([np.nan, pos[1], pos[2]])
        feed(f, clock, k * DT, pos, (0.0, 0.0, 0.0))
        assert np.all(np.isfinite(f.kf_.get_state()))
    assert f.kf_resets_ == 1
    p, vw, _, _, _ = f.get_filtered_state()
    assert np.allclose(vw, v, atol=1e-3)
    assert np.allclose(p, v * 49 * DT, atol=1e-3)


def test_raw_buffer_outputs_are_unchanged_by_the_filter(clock):
    offset_xyz = [0.1, -0.05, -0.266]
    offset_R = tf_transformations.euler_matrix(*np.deg2rad([10.0, -20.0, 180.0]), 'sxyz')[:3, :3]
    raw = ViconDataBuffer(6, WINDOW, offset_xyz=offset_xyz, offset_R=offset_R)
    f = make_filter(offset_xyz=offset_xyz, offset_R=offset_R)
    rng = np.random.default_rng(1)
    for k in range(300):
        t = k * DT
        pos = np.array([0.3 * t + 0.02 * np.sin(3 * t), 0.05 * np.cos(2 * t),
                        0.5 + 0.03 * np.sin(5 * t)])
        pos = pos + 1e-4 * rng.standard_normal(3)
        rpy = (0.3 * np.sin(2 * t), 0.4 * np.sin(1.5 * t), 2.0 * np.sin(0.7 * t) - 1.0)
        clock.t = 100.0 + t - DT / 2
        f.update()
        clock.t = 100.0 + t
        r, p, y = node_angles(*rpy)
        sample = (1000.0 * pos[0], 1000.0 * pos[1], 1000.0 * pos[2], r, p, y)
        raw.add_pose(sample, 0.005)
        f.add_pose(sample, 0.005)
        for a, b in zip(raw.get_latest_pose(), f.get_latest_pose()):
            assert np.array_equal(a, b)
        for a, b in zip(raw.get_latest_velocity(), f.get_latest_velocity()):
            assert np.array_equal(a, b)
        assert raw.is_ready() == f.is_ready()
