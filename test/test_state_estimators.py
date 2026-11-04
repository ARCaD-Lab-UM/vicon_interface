"""Unit tests for the ported state estimators and watchdog; no ROS needed."""
import numpy as np

from vicon_interface.state_estimators import (
    FloatingBaseLinearStateEstimator,
    KalmanFilter,
    linear_motion_model,
    ViconLinearStateEstimator,
)
from vicon_interface.state_estimators.vicon_linear_state_estimator import (
    constant_acceleration_model,
)
from vicon_interface.utils import Daemon, DaemonConfig


def expm_series(M, terms=10):
    """expm(M) = sum_k M^k / k!, summed by hand (exact here: every M used below is nilpotent)."""
    out = np.zeros_like(M)
    term = np.eye(M.shape[0])
    for k in range(terms):
        out += term
        term = term @ M / (k + 1)
    return out


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


def test_constant_acceleration_model_is_the_exact_discretization():
    q = np.array([10.0, 2.0, 0.5])  # m^2/s^5 per axis
    Ac = np.zeros((9, 9))
    Ac[:3, 3:6] = np.eye(3)
    Ac[3:6, 6:] = np.eye(3)
    G = np.zeros((9, 3))
    G[6:] = np.eye(3)
    for dt in (0.004, 0.008, 0.05):
        # Van Loan: expm([[-Ac, G q G^T], [0, Ac^T]] dt) = [[., A^-1 Q], [0, A^T]]
        M = np.zeros((18, 18))
        M[:9, :9] = -Ac
        M[:9, 9:] = G @ np.diag(q) @ G.T
        M[9:, 9:] = Ac.T
        E = expm_series(M * dt)
        A_ref = E[9:, 9:].T
        Q_ref = A_ref @ E[:9, 9:]
        A, Q = constant_acceleration_model(dt, q)
        assert np.allclose(A, A_ref, rtol=0.0, atol=1e-15)
        assert np.allclose(Q, Q_ref, rtol=1e-12, atol=1e-30)
        A_half, Q_half = constant_acceleration_model(dt / 2, q)
        assert np.allclose(A_half @ A_half, A, rtol=0.0, atol=1e-15)
        # two half steps without a measurement equal one full step: a dropped frame costs nothing
        assert np.allclose(A_half @ Q_half @ A_half.T + Q_half, Q, rtol=1e-12, atol=1e-30)


def test_vicon_estimator_observes_position_and_predicts_along_the_model():
    R = np.diag([4e-8] * 3)
    P0 = np.diag([4e-8] * 3 + [1.0] * 3 + [25.0] * 3)
    est = ViconLinearStateEstimator([10.0] * 3, R, P0)
    C = np.zeros((3, 9))
    C[:, :3] = np.eye(3)
    assert np.array_equal(est.C, C)
    assert np.array_equal(est.x, np.zeros(9))

    x0 = np.array([1.0, 2.0, 3.0, 0.5, -0.5, 0.25, 1.0, 0.0, -2.0])
    est.reset(x0)
    dt = 0.01
    ahead = est.predicted_state(dt)
    assert np.array_equal(est.x, x0)  # looking ahead does not move the filter
    x = est.predict_dt(dt)
    assert np.allclose(x, ahead)
    assert np.allclose(x[:3], x0[:3] + x0[3:6] * dt + 0.5 * x0[6:] * dt * dt)
    assert np.allclose(x[3:6], x0[3:6] + x0[6:] * dt)
    assert np.allclose(x[6:], x0[6:])

    est.correct(x[:3])
    assert np.array_equal(est.R, R)
    assert np.allclose(est.x, x)  # a measurement equal to the prediction leaves the state alone
    assert np.array_equal(est.P, est.P.T)


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
