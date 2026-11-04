"""Behavioural tests for ViconDataFilter with a fake clock; no ROS needed."""
import time

import numpy as np
import pytest
import tf_transformations

from vicon_interface.state_estimators.pose_filter import so3_exp, so3_log
from vicon_interface.vicon_data_buffer import ViconDataBuffer
from vicon_interface.vicon_data_filter import ViconDataFilter

DT = 0.004  # Vicon frame interval and timer period, s
RATE = 1.0 / DT  # Hz, the frame rate the SDK reports
WINDOW = 5
PARAMS = {
    'motion_model': 'cv',
    'noise_psd': (0.036, 0.9),
    'measurement_std': (2e-4, 1e-3),
    'initial_std': (1.0, 2.0, 5.0, 20.0),
    'gate_sigma': 10.0,
    'max_rejections': 5,
    'max_delay': 0.02,
    'clock_window': 2.0,
    'offline_timeout': 0.1,
    'online_jitter_time': 0.0,
    'owner_id': 'test',
}
CA = {'motion_model': 'ca', 'noise_psd': (10.0, 250.0)}  # the param file's "ca" densities
MODELS = pytest.mark.parametrize('model', [{}, CA], ids=['cv', 'ca'])


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


def filtered(f, now=None):
    """Return the filter's estimate at now (host monotonic), as odometry/filtered reads it."""
    return f.get_filtered_state(now)[0]


def node_angles(roll, pitch, yaw):
    """Do what vicon_stream_node does with the SDK's XYZ Euler triple."""
    q = tf_transformations.quaternion_from_euler(roll, pitch, yaw, 'rxyz')
    return tf_transformations.euler_from_quaternion(q, 'sxyz')


def sdk_angles(R):
    """Return the SDK's XYZ Euler triple of the rotation matrix R."""
    m = np.eye(4)
    m[:3, :3] = R
    return tf_transformations.euler_from_matrix(m, 'rxyz')


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
    est, stamp = f.get_filtered_state()
    assert stamp == pytest.approx(clock.t)  # latency 0: captured now
    est.p[:] = 99.0  # the getter hands out copies
    est.v[:] = 99.0
    assert not np.any(f.kf_.state().p == 99.0) and not np.any(f.kf_.state().v == 99.0)


@MODELS
def test_constant_velocity_is_tracked_exactly(clock, model):
    f = make_filter(**model)
    v = np.array([0.3, -0.1, 0.05])
    for k in range(200):
        feed(f, clock, k, v * k * DT, (0.1, -0.2, 0.5))
    est = filtered(f)
    assert np.allclose(est.v, v, atol=1e-4)
    assert np.allclose(est.p, v * 199 * DT, atol=1e-5)


def test_constant_acceleration_is_tracked_without_lag_by_ca(clock):
    f = make_filter(**CA)
    v0 = np.array([0.2, 0.0, -0.1])
    acc = np.array([0.5, -1.0, 2.0])
    for k in range(300):
        t = k * DT
        feed(f, clock, k, v0 * t + 0.5 * acc * t * t, (0.0, 0.0, 0.0))
    t = 299 * DT
    assert np.allclose(filtered(f).v, v0 + acc * t, atol=1e-4)  # "cv" lags here by ~11 ms
    assert np.allclose(f.kf_.state().a, acc, atol=1e-2)


def test_latency_is_compensated_by_predicting_to_now(clock):
    f = make_filter()
    v = np.array([0.3, -0.1, 0.05])
    latency = 0.010
    for k in range(200):
        feed(f, clock, k, v * k * DT, (0.0, 0.0, 0.0), latency=latency)
    now = clock.t - 100.0  # the last frame arrived latency after its capture
    est, stamp = f.get_filtered_state()
    assert np.allclose(est.p, v * now, atol=1e-5)  # position at now, not at the frame time
    assert np.isclose(stamp, 100.0 + 199 * DT)  # the capture time on the host clock
    later = filtered(f, now=clock.t + 0.005)
    assert np.allclose(later.p, v * (now + 0.005), atol=1e-5)
    assert np.allclose(filtered(f).p, est.p)  # looking ahead does not move the filter


def test_the_prediction_to_now_is_capped_at_max_delay(clock):
    f = make_filter()
    v = np.array([0.3, -0.1, 0.05])
    latency = 0.05  # well above the 20 ms cap
    for k in range(200):
        feed(f, clock, k, v * k * DT, (0.0, 0.0, 0.0), latency=latency)
    est = filtered(f)
    assert np.allclose(est.p, v * (199 * DT + 0.02), atol=1e-5)  # by the cap, not the latency
    std_p = PARAMS['measurement_std'][0]
    assert np.array_equal(f.kf_.R_trans_, std_p ** 2 * np.eye(3))  # R is never rescaled


@MODELS
def test_dropped_frames_and_host_delays_do_not_reach_the_velocity(clock, model):
    """The SDK drops frames and the host picks them up late; the Vicon clock keeps dt exact."""
    rng = np.random.default_rng(2)
    f = make_filter(**model)
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
            err_kf.append(np.max(np.abs(filtered(f).v - v)))
            err_fd.append(np.max(np.abs(raw.get_latest_velocity()[0] - v)))
    print(f'worst velocity error: kf {max(err_kf):.2e} m/s, '
          f'host-clocked finite difference {max(err_fd):.3f} m/s')
    assert max(err_kf) < 1e-4


def test_the_finite_difference_is_timed_by_the_frame_clock(clock):
    """The fallback twist: host delays and dropped frames reach it only through the host clock."""
    rng = np.random.default_rng(5)
    f = make_filter()
    host = ViconDataBuffer(6, WINDOW)  # given no frame times, as before
    v = np.array([0.5, 0.2, 0.0])
    err_frame, err_host = [], []
    k = 0
    for i in range(300):
        k += 2 if rng.random() < 0.05 else 1  # 5 % of the frames never reach the node
        delay = rng.uniform(0.0, 0.9 * DT)  # s, waiting the SDK latency does not include
        feed(f, clock, k, v * k * DT, (0.0, 0.0, 0.0), delay=delay)
        host.add_pose((1000.0 * v[0] * k * DT, 1000.0 * v[1] * k * DT, 0.0, 0.0, 0.0, 0.0), 0.0)
        if i >= WINDOW:
            err_frame.append(np.max(np.abs(f.get_latest_velocity()[0] - v)))
            err_host.append(np.max(np.abs(host.get_latest_velocity()[0] - v)))
    assert max(err_frame) < 1e-9
    assert max(err_host) > 0.05  # what the host clock did to the same frames
    for j in range(WINDOW):  # Vicon restarts its frame count: the window starts over
        feed(f, clock, k + 1 + j, v * (k + 1 + j) * DT, (0.0, 0.0, 0.0), frame_number=3 + j)
        assert f.is_ready() == (j == WINDOW - 1)
    assert np.allclose(f.get_latest_velocity()[0], v, atol=1e-9)


@pytest.mark.parametrize('model, rms_max, fd_ratio', [({}, 0.05, 1.5), (CA, 0.02, 1.0)],
                         ids=['cv', 'ca'])
def test_sinusoid_velocity_error_is_small(clock, model, rms_max, fd_ratio):
    """1 and 2 Hz sinusoids: "cv" lags about as much as the 5-frame difference, "ca" less."""
    f = make_filter(**model)
    err_kf, err_fd = [], []
    for k in range(500):
        t = k * DT
        pos = np.array([0.1 * np.sin(2 * np.pi * t), 0.0, 0.02 * np.sin(4 * np.pi * t)])
        feed(f, clock, k, pos, (0.0, 0.0, 0.0))
        if t >= 1.0:
            v_true = np.array([0.2 * np.pi * np.cos(2 * np.pi * t), 0.0,
                               0.08 * np.pi * np.cos(4 * np.pi * t)])
            v_fd, _, _ = f.get_latest_velocity()
            err_kf.append(np.linalg.norm(filtered(f).v - v_true))
            err_fd.append(np.linalg.norm(v_fd - v_true))
    rms_kf = np.sqrt(np.mean(np.square(err_kf)))
    rms_fd = np.sqrt(np.mean(np.square(err_fd)))
    print(f'sinusoid velocity RMS error: kf {rms_kf:.4f} m/s, finite difference {rms_fd:.4f} m/s')
    assert rms_kf < rms_max
    assert rms_kf <= fd_ratio * rms_fd


@MODELS
def test_static_object_filtered_velocity_is_quieter_than_the_finite_difference(clock, model):
    f = make_filter(**model)
    rng = np.random.default_rng(0)
    v_kf, v_fd = [], []
    for k in range(2000):
        pos = np.array([1.0, 2.0, 0.5]) + 1e-4 * rng.standard_normal(3)  # 0.1 mm white noise
        feed(f, clock, k, pos, (0.0, 0.0, 0.0))
        if k >= 100:
            v_kf.append(filtered(f).v)
            v_fd.append(f.get_latest_velocity()[0])
    std_kf = np.std(np.array(v_kf), axis=0)
    std_fd = np.std(np.array(v_fd), axis=0)
    print(f'static velocity std: kf {std_kf}, finite difference {std_fd}')
    assert np.all(std_kf < std_fd)


@MODELS
def test_yaw_wrap_does_not_disturb_the_filtered_estimate(clock, model):
    f = make_filter(**model)
    v = np.array([0.3, 0.0, 0.0])
    rate = 2.0  # rad/s; yaw passes +pi at t = 0.6 s, frame 150, once the start has settled
    for k in range(250):
        t = k * DT
        feed(f, clock, k, v * t, (0.0, 0.0, np.pi - 1.2 + rate * t))
        if k >= 100:
            est = filtered(f)
            assert np.allclose(est.v, v, atol=1e-4)
            assert np.allclose(est.w, [0.0, 0.0, rate], atol=1e-6)


@MODELS
def test_orientation_and_body_rate_follow_the_frames_at_a_ninety_degree_heading(clock, model):
    f = make_filter(**model)
    w_body = np.array([1.0, 0.0, 0.0])  # rad/s, rolling about the body's x axis
    heading = so3_exp(np.array([0.0, 0.0, np.pi / 2]))
    for k in range(250):
        feed(f, clock, k, np.zeros(3), sdk_angles(heading @ so3_exp(w_body * k * DT)))
    est = filtered(f)
    assert np.allclose(so3_log(est.R.T @ heading @ so3_exp(w_body * 249 * DT)), 0.0, atol=1e-6)
    assert np.allclose(est.w, w_body, atol=1e-5)  # the Euler-rate twist reads [0, -1, 0] here


@MODELS
def test_a_glitch_frame_is_rejected_and_does_not_reach_the_velocity(clock, model):
    f = make_filter(**model)
    v = np.array([0.3, -0.1, 0.05])
    for k in range(200):
        glitch = np.array([0.05, 0.0, 0.0]) if k == 150 else 0.0  # m, one swapped marker
        feed(f, clock, k, v * k * DT + glitch, (0.0, 0.0, 0.0))
        if k >= 150:
            assert np.allclose(filtered(f).v, v, atol=1e-4)
    assert f.kf_rejections_ == 1 and f.kf_resets_ == 1


def test_the_frame_state_is_the_filter_at_the_capture_time(clock):
    """odometry/<obj> reads it: no host delay reaches the stamp, nothing during the warm-up."""
    rng = np.random.default_rng(4)
    f = make_filter()
    v = np.array([0.3, -0.1, 0.05])
    latency = 0.008  # s, as the SDK reports it
    for k in range(200):
        delay = 0.0 if k % 10 == 0 else rng.uniform(0.0, 0.9 * DT)  # s, the host's own waiting
        feed(f, clock, k, v * k * DT, (0.0, 0.0, 0.0), latency=latency, delay=delay)
        estimate, stamp = f.get_frame_state()
        if k >= 10:  # one frame on time in each ten sets the minimum offset
            assert stamp == pytest.approx(100.0 + k * DT, abs=1e-9)
        if k < WINDOW - 1:
            assert estimate is None and stamp is not None
        elif k >= 100:
            assert np.allclose(estimate.v, v, atol=1e-4)
            assert np.allclose(estimate.p, v * k * DT, atol=1e-5)  # at the frame, not moved
    feed(f, clock, 200, np.full(3, np.nan), (0.0, 0.0, 0.0))
    assert f.get_frame_state() == (None, None)  # a frame not fused has no stamp either


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
    held = filtered(f)
    assert np.allclose(held.p, v * (49 * DT + PARAMS['max_delay']), atol=1e-5)  # held at the cap
    frozen = f.kf_.state()
    clock.t += DT
    assert f.update() is False
    assert np.array_equal(f.kf_.state().p, frozen.p)
    assert np.array_equal(f.kf_.state().v, frozen.v)
    assert f.is_offline()
    k0 = int(round((clock.t - 100.0) / DT)) + 1  # frames resume, continuing the same motion
    for j in range(40):
        feed(f, clock, k0 + j, v * (k0 + j) * DT, (0.0, 0.0, 0.0))
        assert f.kf_resets_ == 2  # the first frame after the gap restarts the filter
        if j < WINDOW - 1:
            assert f.update() is False
    assert f.update() is True
    est = filtered(f)
    assert np.allclose(est.v, v, atol=1e-3)
    assert np.allclose(est.p, v * (k0 + 39) * DT, atol=1e-4)


def test_a_nan_frame_is_skipped_and_the_filter_recovers(clock):
    f = make_filter()
    v = np.array([0.3, -0.1, 0.05])
    for k in range(50):
        pos = v * k * DT
        if k == 30:
            pos = np.array([np.nan, pos[1], pos[2]])
        feed(f, clock, k, pos, (np.nan, 0.0, 0.0) if k == 35 else (0.0, 0.0, 0.0))
        state = f.kf_.state()
        assert np.all(np.isfinite(state.p)) and np.all(np.isfinite(state.R))
    assert f.kf_resets_ == 1  # the next frame is just a two-frame step
    est = filtered(f)
    assert np.allclose(est.v, v, atol=1e-3)
    assert np.allclose(est.p, v * 49 * DT, atol=1e-4)


def test_a_restarted_frame_count_restarts_the_filter(clock):
    f = make_filter()
    v = np.array([0.3, -0.1, 0.05])
    for k in range(50):
        feed(f, clock, k, v * k * DT, (0.0, 0.0, 0.0))
    assert f.kf_resets_ == 1
    for j in range(50):  # Vicon restarts its frame count; frames keep coming on time
        feed(f, clock, 50 + j, v * (50 + j) * DT, (0.0, 0.0, 0.0), frame_number=7 + j)
    assert f.kf_resets_ == 2
    est = filtered(f)
    assert np.allclose(est.v, v, atol=1e-3)
    assert np.allclose(est.p, v * 99 * DT, atol=1e-4)


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
        raw.add_pose(sample, 0.005, k / RATE)
        f.add_pose(sample, 0.005, k, RATE)
        for a, b in zip(raw.get_latest_pose(), f.get_latest_pose()):
            assert np.array_equal(a, b)
        for a, b in zip(raw.get_latest_velocity(), f.get_latest_velocity()):
            assert np.array_equal(a, b)
        assert raw.is_ready() == f.is_ready()
