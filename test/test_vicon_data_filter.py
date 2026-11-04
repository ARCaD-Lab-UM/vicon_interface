"""Behavioural tests for ViconDataFilter with a fake clock; no ROS needed."""
import time

import numpy as np
import pytest
import tf_transformations

from vicon_interface.vicon_data_buffer import ViconDataBuffer
from vicon_interface.vicon_data_filter import ViconDataFilter

DT = 0.004  # Vicon frame interval and timer period, s
RATE = 1.0 / DT  # Hz, the frame rate the SDK reports
WINDOW = 5
PARAMS = {
    'jerk_psd': [10.0] * 3,
    'measurement_noise': [4e-8] * 3,
    'initial_covariance': [4e-8] * 3 + [1.0] * 3 + [25.0] * 3,
    'max_delay': 0.02,
    'clock_window': 2.0,
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


def feed(f, clock, k, pos_m, rpy, latency=0.0, delay=0.0, frame_number=None, frame_rate=RATE):
    """Run one timer tick, then deliver frame k, captured at k * DT, latency + delay seconds later.

    latency is what the SDK reports; delay is host-side waiting that the SDK does not report.
    """
    tick = 100.0 + k * DT + latency - DT / 2
    if tick > clock.t:
        clock.t = tick
        f.update()
    clock.t = 100.0 + k * DT + latency + delay
    r, p, y = node_angles(*rpy)
    sample = (1000.0 * pos_m[0], 1000.0 * pos_m[1], 1000.0 * pos_m[2], r, p, y)
    f.add_pose(sample, latency, k if frame_number is None else frame_number, frame_rate)


def test_startup_waits_for_a_full_window_then_publishes(clock):
    f = make_filter()
    assert f.update() is False
    v = np.array([0.3, -0.1, 0.05])
    for k in range(WINDOW - 1):
        feed(f, clock, k, v * k * DT, (0.0, 0.0, 0.0))
        assert f.kf_resets_ == 1  # the first frame starts the filter
        assert f.update() is False
    feed(f, clock, WINDOW - 1, v * (WINDOW - 1) * DT, (0.0, 0.0, 0.0))
    assert f.kf_resets_ == 1
    assert f.update() is True
    p, vw, rpy, a, stamp = f.get_filtered_state()
    assert stamp == pytest.approx(clock.t)  # latency 0: captured now
    p[:] = 99.0  # the getter hands out copies
    vw[:] = 99.0
    assert not np.any(f.kf_.get_state() == 99.0)


def test_constant_velocity_is_tracked_exactly(clock):
    f = make_filter()
    v = np.array([0.3, -0.1, 0.05])
    for k in range(200):
        feed(f, clock, k, v * k * DT, (0.1, -0.2, 0.5))
    p, vw, _, _, _ = f.get_filtered_state()
    assert np.allclose(vw, v, atol=1e-4)
    assert np.allclose(p, v * 199 * DT, atol=1e-5)


def test_constant_acceleration_is_tracked_without_lag(clock):
    f = make_filter()
    v0 = np.array([0.2, 0.0, -0.1])
    acc = np.array([0.5, -1.0, 2.0])
    for k in range(300):
        t = k * DT
        feed(f, clock, k, v0 * t + 0.5 * acc * t * t, (0.0, 0.0, 0.0))
    t = 299 * DT
    p, vw, _, _, _ = f.get_filtered_state()
    assert np.allclose(vw, v0 + acc * t, atol=1e-4)  # a constant-velocity model would lag here
    assert np.allclose(f.kf_.get_state()[6:], acc, atol=1e-2)


def test_latency_is_compensated_by_predicting_to_now(clock):
    f = make_filter()
    v = np.array([0.3, -0.1, 0.05])
    latency = 0.010
    for k in range(200):
        feed(f, clock, k, v * k * DT, (0.0, 0.0, 0.0), latency=latency)
    now = clock.t - 100.0  # the last frame arrived latency after its capture
    p, vw, _, _, stamp = f.get_filtered_state()
    assert np.allclose(p, v * now, atol=1e-5)  # position at now, not at the frame time
    assert np.isclose(stamp, 100.0 + 199 * DT)  # the capture time on the host clock
    p_later = f.get_filtered_state(now=clock.t + 0.005)[0]
    assert np.allclose(p_later, v * (now + 0.005), atol=1e-5)
    assert np.allclose(f.get_filtered_state()[0], p)  # looking ahead does not move the filter


def test_the_prediction_to_now_is_capped_at_max_delay(clock):
    f = make_filter()
    v = np.array([0.3, -0.1, 0.05])
    latency = 0.05  # well above the 20 ms cap
    for k in range(200):
        feed(f, clock, k, v * k * DT, (0.0, 0.0, 0.0), latency=latency)
    p, vw, _, _, stamp = f.get_filtered_state()
    assert np.allclose(p, v * (199 * DT + 0.02), atol=1e-5)  # moved by the cap, not the latency
    assert np.array_equal(f.kf_.R, np.diag(PARAMS['measurement_noise']))  # R is never rescaled


def test_dropped_frames_and_host_delays_do_not_reach_the_velocity(clock):
    """The SDK drops frames and the host picks them up late; the Vicon clock keeps dt exact."""
    rng = np.random.default_rng(2)
    f = make_filter()
    raw = ViconDataBuffer(6, WINDOW)
    v = np.array([0.5, 0.2, 0.0])
    err_kf, err_fd = [], []
    k = 0
    for i in range(400):
        k += 2 if rng.random() < 0.05 else 1  # 5 % of the frames never reach the node
        delay = rng.uniform(0.0, 0.9 * DT)  # s, waiting the SDK latency does not include
        feed(f, clock, k, v * k * DT, (0.0, 0.0, 0.0), delay=delay)
        raw.add_pose((1000.0 * v[0] * k * DT, 1000.0 * v[1] * k * DT, 0.0, 0.0, 0.0, 0.0), 0.0)
        if i >= 100:
            err_kf.append(np.max(np.abs(f.get_filtered_state()[1] - v)))
            err_fd.append(np.max(np.abs(raw.get_latest_velocity()[0] - v)))
    print(f'worst velocity error: kf {max(err_kf):.2e} m/s, '
          f'host-clocked finite difference {max(err_fd):.3f} m/s')
    assert max(err_kf) < 1e-4


def test_sinusoid_velocity_error_is_small_and_below_the_finite_difference(clock):
    f = make_filter()
    err_kf, err_fd = [], []
    for k in range(500):
        t = k * DT
        pos = np.array([0.1 * np.sin(2 * np.pi * t), 0.0, 0.02 * np.sin(4 * np.pi * t)])
        feed(f, clock, k, pos, (0.0, 0.0, 0.0))
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
    assert rms_kf < 0.02  # about 2 (f / f_n)^2 of the amplitude, f_n ~ 10 Hz
    assert rms_kf < rms_fd


def test_static_object_filtered_velocity_is_quieter_than_the_finite_difference(clock):
    f = make_filter()
    rng = np.random.default_rng(0)
    v_kf, v_fd = [], []
    for k in range(2000):
        pos = np.array([1.0, 2.0, 0.5]) + 1e-4 * rng.standard_normal(3)  # 0.1 mm white noise
        feed(f, clock, k, pos, (0.0, 0.0, 0.0))
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
    rate = 2.0  # rad/s; yaw passes +pi at t = 0.3 s, frame 75
    for k in range(150):
        t = k * DT
        feed(f, clock, k, v * t, (0.0, 0.0, np.pi - 0.6 + rate * t))
        if k >= 50:
            _, vw, rpy, a, _ = f.get_filtered_state()
            assert np.allclose(vw, v, atol=1e-4)
            R = tf_transformations.euler_matrix(*rpy, 'sxyz')[:3, :3]
            assert np.allclose(R.T @ vw, R.T @ v, atol=1e-4)
            assert abs(a[2] - rate) < 1e-6


def test_offline_gap_pauses_the_filter_and_resets_it_on_return(clock):
    f = make_filter()
    v = np.array([0.3, -0.1, 0.05])
    for k in range(50):
        feed(f, clock, k, v * k * DT, (0.0, 0.0, 0.0))
    assert f.kf_resets_ == 1 and f.update() is True
    t_gap = clock.t
    results = []
    for i in range(1, 126):  # 0.5 s of timer ticks without a frame
        clock.t = t_gap + i * DT
        results.append(f.update())
    assert all(results[:25])  # up to 100 ms: still online, the output is a prediction
    assert not any(results[25:])  # after the timeout
    p_held = f.get_filtered_state()[0]
    assert np.allclose(p_held, v * (49 * DT + PARAMS['max_delay']), atol=1e-5)  # held at the cap
    frozen = f.kf_.get_state()
    clock.t += DT
    assert f.update() is False
    assert np.array_equal(f.kf_.get_state(), frozen)
    assert f.is_offline()
    k0 = int(round((clock.t - 100.0) / DT)) + 1  # frames resume, continuing the same motion
    for j in range(40):
        feed(f, clock, k0 + j, v * (k0 + j) * DT, (0.0, 0.0, 0.0))
        assert f.kf_resets_ == 2  # the first frame after the gap restarts the filter
        if j < WINDOW - 1:
            assert f.update() is False
    assert f.update() is True
    p, vw, _, _, _ = f.get_filtered_state()
    assert np.allclose(vw, v, atol=1e-3)
    assert np.allclose(p, v * (k0 + 39) * DT, atol=1e-4)


def test_a_nan_frame_is_skipped_and_the_filter_recovers(clock):
    f = make_filter()
    v = np.array([0.3, -0.1, 0.05])
    for k in range(50):
        pos = v * k * DT
        if k == 30:
            pos = np.array([np.nan, pos[1], pos[2]])
        feed(f, clock, k, pos, (0.0, 0.0, 0.0))
        assert np.all(np.isfinite(f.kf_.get_state()))
    assert f.kf_resets_ == 1  # the next frame is just a two-frame step
    p, vw, _, _, _ = f.get_filtered_state()
    assert np.allclose(vw, v, atol=1e-3)
    assert np.allclose(p, v * 49 * DT, atol=1e-4)


def test_a_restarted_frame_count_restarts_the_filter(clock):
    f = make_filter()
    v = np.array([0.3, -0.1, 0.05])
    for k in range(50):
        feed(f, clock, k, v * k * DT, (0.0, 0.0, 0.0))
    assert f.kf_resets_ == 1
    for j in range(50):  # Vicon restarts its frame count; frames keep coming on time
        feed(f, clock, 50 + j, v * (50 + j) * DT, (0.0, 0.0, 0.0), frame_number=7 + j)
    assert f.kf_resets_ == 2
    p, vw, _, _, _ = f.get_filtered_state()
    assert np.allclose(vw, v, atol=1e-3)
    assert np.allclose(p, v * 99 * DT, atol=1e-4)


def test_without_a_frame_rate_the_filter_waits(clock):
    f = make_filter()
    for k in range(20):
        feed(f, clock, k, np.zeros(3), (0.0, 0.0, 0.0), frame_rate=0.0)
    assert f.kf_resets_ == 0 and not f.has_frame_clock()
    assert f.is_online() and f.update() is False
    feed(f, clock, 20, np.zeros(3), (0.0, 0.0, 0.0))
    assert f.has_frame_clock() and f.kf_resets_ == 1


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
        f.add_pose(sample, 0.005, k, RATE)
        for a, b in zip(raw.get_latest_pose(), f.get_latest_pose()):
            assert np.array_equal(a, b)
        for a, b in zip(raw.get_latest_velocity(), f.get_latest_velocity()):
            assert np.array_equal(a, b)
        assert raw.is_ready() == f.is_ready()
