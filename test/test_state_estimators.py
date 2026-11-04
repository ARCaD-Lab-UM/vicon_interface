"""Unit tests for the ported state estimators and watchdog; no ROS needed."""
import numpy as np

from vicon_interface.state_estimators import (
    FloatingBaseLinearStateEstimator,
    KalmanFilter,
    linear_motion_model,
)
from vicon_interface.utils import Daemon, DaemonConfig


def test_kalman_filter_predict_and_correct_match_the_closed_form():
    dt, q, r = 0.1, 0.01, 0.04
    A = np.array([[1.0, dt], [0.0, 1.0]])
    B = np.array([[0.0], [dt]])
    C = np.array([[1.0, 0.0]])
    Q = np.diag([0.0, q])
    R = np.array([[r]])
    x0 = np.array([1.0, 2.0])
    P0 = np.diag([0.5, 0.3])
    kf = KalmanFilter(A, B, C, Q, R, x0, P0)

    x_pred = kf.predict(np.array([3.0]))
    assert np.allclose(x_pred, [1.0 + 2.0 * dt, 2.0 + 3.0 * dt])
    P_pred = A @ P0 @ A.T + Q
    assert np.allclose(kf.P, P_pred)

    x_corr = kf.correct(np.array([1.5]))
    K = P_pred[:, 0] / (P_pred[0, 0] + r)  # P C^T (C P C^T + R)^-1 for a scalar measurement
    assert np.allclose(x_corr, x_pred + K * (1.5 - x_pred[0]))
    assert np.allclose(kf.P, (np.eye(2) - np.outer(K, C[0])) @ P_pred)
    assert np.allclose(kf.get_state(), x_corr)

    kf.reset()
    assert np.array_equal(kf.x, x0) and np.array_equal(kf.P, P0)
    kf.reset(np.array([5.0, 6.0]))
    assert np.array_equal(kf.x, [5.0, 6.0])


def test_floating_base_matrices_match_the_original_expm_form():
    dt = 0.004
    Ac = np.zeros((6, 6))
    Ac[:3, 3:] = np.eye(3)
    Bc = np.zeros((6, 3))
    Bc[3:] = np.eye(3)
    series = np.zeros((6, 6))  # expm(dt*Ac) = sum_k (dt*Ac)^k / k!, summed by hand
    term = np.eye(6)
    for k in range(8):
        series += term
        term = term @ (dt * Ac) / (k + 1)

    est = FloatingBaseLinearStateEstimator(dt, np.eye(6) * 1e-3, np.eye(4) * 1e-2, 0.7)
    assert np.allclose(est.A, series, atol=1e-15)
    assert np.allclose(est.B, dt * Bc)
    C = np.zeros((4, 6))
    C[:, 2:] = np.eye(4)
    assert np.array_equal(est.C, C)
    assert np.array_equal(est.x, [0.0, 0.0, 0.7, 0.0, 0.0, 0.0])
    assert np.array_equal(est.P, np.eye(6) * 1e-5)
    A, B = linear_motion_model(dt)
    assert np.array_equal(A, est.A) and np.array_equal(B, est.B)


def test_daemon_offline_online_and_jitter_semantics():
    config = DaemonConfig(
        set_online_jitter_time_ms=20.0, set_offline_time_ms=100.0, owner_id='box')
    d = Daemon(config)
    d.update(now_ms=150.0)  # never fed: offline
    assert d.is_offline() and d.is_error()
    d.reload(now_ms=160.0)  # first frame: online, inside the jitter window
    d.update(now_ms=165.0)
    assert d.is_online() and d.is_error()
    d.update(now_ms=181.0)  # jitter window over
    assert d.is_online() and not d.is_error()
    d.update(now_ms=281.0)  # 100 ms without a frame
    assert d.is_offline()
    d.reload(now_ms=282.0)
    d.update(now_ms=282.5)
    assert d.is_online() and d.is_error()  # back, settling
    d.update(now_ms=310.0)
    assert not d.is_error()
    assert d.frequency_hz == 0.0 or d.frequency_hz > 0.0  # stats exist even without a spy
