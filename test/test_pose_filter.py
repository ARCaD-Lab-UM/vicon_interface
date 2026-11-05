"""Tests for PoseFilter, the pose-only Kalman filter; numpy only, no ROS."""
import copy
import math

import numpy as np
import pytest

from vicon_interface.state_estimators.pose_filter import (
    _mahalanobis,
    kinematic_blocks,
    PoseEstimate,
    PoseFilter,
    so3_exp,
    so3_log,
    so3_right_jacobian,
)

DT = 0.004  # s, Vicon frame interval at 250 Hz
NOISE_PSD = (0.036, 0.9)  # m^2/s^3, rad^2/s^3
STD = (2e-4, 1e-3)  # m, rad
FILTER_ARGS = {
    'measurement_std': STD,
    'initial_std': (1.0, 2.0),
    'gate': (0.02, 0.2),
    'max_rejections': 5,
    'reset_gap': 0.1,
}
AXIS = np.array([0.3, -0.4, 0.866]) / np.linalg.norm([0.3, -0.4, 0.866])


def make_filter(**overrides):
    return PoseFilter(NOISE_PSD, **{**FILTER_ARGS, **overrides})


def rot_z(angle):
    return so3_exp(np.array([0.0, 0.0, angle]))


def motion(t):
    """Return (p, R, v, w) of a constant-velocity motion; the rotation keeps one axis."""
    v = np.array([0.4, -0.2, 0.05])  # m/s
    p = np.array([1.0, -2.0, 0.8]) + v * t
    R = so3_exp(np.array([0.2, -0.1, 2.5])) @ so3_exp(AXIS * 0.8 * t)
    return p, R, v, AXIS * 0.8


def run(f, frames, noise=None):
    """Feed f the frames of motion, noise(k) added to frame k if given; return the outcomes."""
    outcomes = []
    for k in frames:
        p, R, _, _ = motion(k * DT)
        if noise is not None:
            dp, dtheta = noise(k)
            p, R = p + dp, R @ so3_exp(dtheta)
        outcomes.append(f.update(k * DT, p, R))
    return outcomes


def expm_series(M, terms=10):
    """expm(M) = sum_k M^k / k!, summed by hand (exact here: every M used below is nilpotent)."""
    out = np.zeros_like(M)
    term = np.eye(M.shape[0])
    for k in range(terms):
        out += term
        term = term @ M / (k + 1)
    return out


def steady_rate_std(q, std):
    """Return the std of the steady filter's rate estimate under white measurement noise alone."""
    F, Q = kinematic_blocks(DT)
    H = np.array([[1.0, 0.0]])
    P = np.eye(2)
    for _ in range(3000):
        P = F @ (P - P @ H.T @ H @ P / (P[0, 0] + std ** 2)) @ F.T + q * Q
    K = P[:, :1] / (P[0, 0] + std ** 2)
    A = (np.eye(2) - K @ H) @ F  # the posterior's closed loop
    S = np.zeros((2, 2))
    for _ in range(3000):
        S = A @ S @ A.T + std ** 2 * (K @ K.T)
    return math.sqrt(S[1, 1])


def test_so3_log_inverts_exp_from_tiny_angles_up_to_pi():
    rng = np.random.default_rng(0)
    for norm in (1e-12, 1e-7, 1e-3, 0.5, 2.0, 3.0, math.pi - 1e-4, math.pi - 1e-8):
        for _ in range(20):
            axis = rng.standard_normal(3)
            phi = norm * axis / np.linalg.norm(axis)
            R = so3_exp(phi)
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-14)
            assert np.allclose(so3_log(R), phi, atol=1e-9)


def test_the_right_jacobian_composes_exponentials_to_first_order():
    rng = np.random.default_rng(1)
    for norm in (1e-9, 1e-3, 0.7, 2.5):
        axis = rng.standard_normal(3)
        phi = norm * axis / np.linalg.norm(axis)
        d = 1e-6 * rng.standard_normal(3)
        error = so3_exp(phi + d) - so3_exp(phi) @ so3_exp(so3_right_jacobian(phi) @ d)
        assert np.max(np.abs(error)) < 1e-10  # second order in |d|


def test_kinematic_blocks_compose():
    dt = 0.01
    F, Q = kinematic_blocks(dt)
    F2, Q2 = kinematic_blocks(2 * dt)  # the noise is integrated exactly: two steps are one
    assert np.allclose(F2, F @ F, rtol=1e-14, atol=0.0)
    assert np.allclose(Q2, F @ Q @ F.T + Q, rtol=1e-12, atol=0.0)


def test_kinematic_blocks_are_the_exact_discretization():
    Ac = np.array([[0.0, 1.0], [0.0, 0.0]])  # the value's derivative is the rate
    G = np.array([[0.0], [1.0]])  # unit white noise drives the rate
    for dt in (0.004, 0.008, 0.05):
        # Van Loan: expm([[-Ac, G G^T], [0, Ac^T]] dt) = [[., F^-1 Q], [0, F^T]]
        M = np.zeros((4, 4))
        M[:2, :2] = -Ac
        M[:2, 2:] = G @ G.T
        M[2:, 2:] = Ac.T
        E = expm_series(M * dt)
        F_ref = E[2:, 2:].T
        F, Q = kinematic_blocks(dt)
        assert np.allclose(F, F_ref, rtol=0.0, atol=1e-15)
        assert np.allclose(Q, F_ref @ E[:2, 2:], rtol=1e-12, atol=1e-30)


def test_constant_velocity_is_tracked_exactly():
    f = make_filter()
    assert run(f, range(400)) == ['reset'] + ['accepted'] * 399
    est = f.state()
    p, R, v, w = motion(399 * DT)
    assert np.allclose(est.p, p, atol=1e-9)
    assert np.allclose(est.v, v, atol=1e-7)
    assert np.allclose(so3_log(est.R.T @ R), 0.0, atol=1e-9)
    assert np.allclose(est.w, w, atol=1e-7)


def test_static_noise_reaches_the_rates_as_the_steady_riccati_solution_predicts():
    f = make_filter()
    rng = np.random.default_rng(2)
    p0, R0 = np.array([0.5, 1.0, 0.9]), so3_exp(np.array([0.1, 0.2, -1.0]))
    v, w = [], []
    for k in range(10000):
        p = p0 + STD[0] * rng.standard_normal(3)
        R = R0 @ so3_exp(STD[1] * rng.standard_normal(3))
        f.update(k * DT, p, R)
        if k >= 500:
            est = f.state()
            v.append(est.v)
            w.append(est.w)
    assert np.allclose(np.std(v, axis=0), steady_rate_std(NOISE_PSD[0], STD[0]), rtol=0.2)
    assert np.allclose(np.std(w, axis=0), steady_rate_std(NOISE_PSD[1], STD[1]), rtol=0.2)


def test_dropped_frames_keep_the_exact_interval():
    f = make_filter()
    frames = [k for k in range(400) if k % 7 != 3 and not 200 <= k < 203]
    assert set(run(f, frames)[1:]) == {'accepted'}
    est = f.state()
    _, _, v, w = motion(frames[-1] * DT)
    assert np.allclose(est.v, v, atol=1e-7)
    assert np.allclose(est.w, w, atol=1e-7)


def test_a_frame_at_or_before_the_last_one_is_stale_and_changes_nothing():
    f = make_filter()
    run(f, range(50))
    before = f.state()
    p, R, _, _ = motion(49 * DT)
    assert f.update(49 * DT, p + 0.01, R) == 'stale'
    assert f.update(45 * DT, p, R) == 'stale'
    after = f.state()
    for name in ('t', 'p', 'v', 'R', 'w', 'P_trans', 'P_rot'):
        assert np.array_equal(getattr(before, name), getattr(after, name))


@pytest.mark.parametrize('kind', ('position', 'orientation'))
def test_a_single_outlier_is_rejected_as_if_the_frame_never_came(kind):
    rng = np.random.default_rng(3)
    noise = {k: (STD[0] * rng.standard_normal(3), STD[1] * rng.standard_normal(3))
             for k in range(300)}
    jump = {
        'position': (np.array([0.05, 0.0, 0.0]), np.zeros(3)),  # m
        'orientation': (np.zeros(3), np.array([0.0, 0.3, 0.0])),  # rad
    }
    seen, unseen = make_filter(), make_filter()
    outcomes = run(seen, range(300), lambda k: jump[kind] if k == 200 else noise[k])
    assert outcomes[200] == 'rejected' and set(outcomes[201:]) == {'accepted'}
    run(unseen, [k for k in range(300) if k != 200], lambda k: noise[k])
    a, b = seen.state(), unseen.state()
    assert np.allclose(a.p, b.p, atol=1e-8)
    assert np.allclose(a.v, b.v, atol=1e-6)
    assert np.allclose(a.w, b.w, atol=1e-5)


@pytest.mark.parametrize('kind, size, outcome', [
    ('position', 0.03, 'rejected'), ('position', 0.01, 'accepted'),  # m
    ('orientation', 0.3, 'rejected'), ('orientation', 0.1, 'accepted'),  # rad
])
def test_the_gate_rejects_only_beyond_its_physical_thresholds(kind, size, outcome):
    f = make_filter()
    run(f, range(200))
    p, R, _, _ = motion(200 * DT)
    if kind == 'position':
        p = p + np.array([size, 0.0, 0.0])
    else:
        R = R @ so3_exp(np.array([0.0, size, 0.0]))
    assert f.update(200 * DT, p, R) == outcome


def test_the_huber_weight_leaves_small_residuals_alone_and_scales_a_large_one():
    rng = np.random.default_rng(5)
    small = {k: (np.zeros(3), 1e-5 * rng.standard_normal(3)) for k in range(200)}  # rad, << k
    plain = make_filter()
    huber = make_filter(rot_huber_k=2.0)
    run(plain, range(200), lambda k: small[k])
    run(huber, range(200), lambda k: small[k])
    for name in ('p', 'v', 'R', 'w', 'P_trans', 'P_rot'):
        assert np.array_equal(getattr(plain.state(), name), getattr(huber.state(), name))
    p, R, _, _ = motion(200 * DT)
    R = R @ so3_exp(np.array([0.0, 0.03, 0.0]))  # rad, a glitch inside the gate
    scaled = copy.deepcopy(plain)  # the Huber update is a plain one with the variance scaled
    _, _, R_pred, _, P_rot = scaled._propagate(DT)
    distance = _mahalanobis(P_rot, so3_log(R_pred.T @ R), scaled.R_rot_)
    assert distance > 2.0
    scaled.R_rot_ = scaled.R_rot_ * (distance / 2.0)
    for f in (plain, huber, scaled):
        assert f.update(200 * DT, p, R) == 'accepted'
    assert np.allclose(huber.state().R, scaled.state().R, rtol=0.0, atol=1e-14)
    assert np.allclose(huber.state().w, scaled.state().w, rtol=1e-12, atol=1e-14)
    pulled = [np.linalg.norm(so3_log(R_pred.T @ f.state().R)) for f in (plain, huber)]
    assert pulled[1] < 0.5 * pulled[0]  # the glitch pulls the weighted filter less


def test_a_lasting_jump_resets_after_max_rejections():
    f = make_filter()
    shift = np.array([0.0, 0.05, 0.0])  # m: the body moved for good, or was redefined
    outcomes = run(f, range(300), lambda k: (shift if k >= 200 else np.zeros(3), np.zeros(3)))
    assert outcomes[200:205] == ['rejected'] * 4 + ['reset']
    assert set(outcomes[205:]) == {'accepted'}
    est = f.state()
    p, _, v, w = motion(299 * DT)
    assert np.allclose(est.p, p + shift, atol=1e-9)
    assert np.allclose(est.v, v, atol=1e-6)
    assert np.allclose(est.w, w, atol=1e-6)


def test_a_gap_or_a_restarted_clock_restarts_from_the_measurement():
    f = make_filter()
    run(f, range(50))
    p, R, _, _ = motion(50 * DT + 0.2)
    assert f.update(50 * DT + 0.2, p, R) == 'reset'  # 200 ms without a frame
    est = f.state()
    assert np.array_equal(est.p, p) and np.array_equal(est.R, R)
    assert not np.any(est.v) and not np.any(est.w)
    assert f.update(0.0, p, R) == 'reset'  # a Vicon restart sends the frame numbers back


def test_the_angular_velocity_is_the_body_one_at_a_heading_of_ninety_degrees():
    f = make_filter()
    w_body = np.array([1.0, 0.0, 0.0])  # rad/s, rolling about the body's x axis
    for k in range(200):
        f.update(k * DT, np.zeros(3), rot_z(math.pi / 2) @ so3_exp(w_body * k * DT))
    assert np.allclose(f.state().w, w_body, atol=1e-7)  # the Euler-rate twist read [0, -1, 0]


def test_yaw_through_pi_does_not_disturb_the_estimate():
    f = make_filter()
    rate = 2.0  # rad/s; yaw passes pi at t = 0.6 s, once the start has settled
    worst = 0.0
    for k in range(250):
        f.update(k * DT, np.zeros(3), rot_z(math.pi - 1.2 + rate * k * DT))
        if k >= 100:
            worst = max(worst, float(np.max(np.abs(f.state().w - [0.0, 0.0, rate]))))
    assert worst < 1e-6


def test_predicted_moves_the_estimate_and_leaves_the_filter_unchanged():
    f = make_filter()
    run(f, range(300))
    before = f.state()
    t = 299 * DT + 0.02  # s, 20 ms ahead
    ahead = f.predicted(t)
    p, R, v, w = motion(t)
    assert ahead.t == pytest.approx(t)
    assert np.allclose(ahead.p, p, atol=1e-9)
    assert np.allclose(ahead.v, v, atol=1e-7)
    assert np.allclose(so3_log(ahead.R.T @ R), 0.0, atol=1e-9)
    assert np.allclose(ahead.w, w, atol=1e-7)
    assert np.all(np.diag(ahead.P_trans) > np.diag(before.P_trans))
    assert np.all(np.diag(ahead.P_rot) > np.diag(before.P_rot))
    after = f.state()
    for name in ('t', 'p', 'v', 'R', 'w', 'P_trans', 'P_rot'):
        assert np.array_equal(getattr(before, name), getattr(after, name))
    assert np.array_equal(f.predicted(0.0).p, before.p)  # never before the current estimate


OFFSET = (np.array([0.05, -0.1, -0.266]), so3_exp(np.array([0.1, -0.2, 3.0])))  # m, rotation


def test_transported_is_exact_for_a_rigid_motion():
    f = make_filter()
    run(f, range(400))
    d, R_off = OFFSET
    child = f.state().transported(d, R_off)
    p, R, v, w = motion(399 * DT)
    assert np.allclose(child.p, p + R @ d, atol=1e-9)
    assert np.allclose(so3_log((R @ R_off).T @ child.R), 0.0, atol=1e-9)
    assert np.allclose(child.v, v + R @ np.cross(w, d), atol=1e-7)  # the lever arm's w x d
    assert np.allclose(child.w, R_off.T @ w, atol=1e-7)


def test_transported_covariance_matches_the_spread_of_sampled_states():
    rng = np.random.default_rng(6)
    f = make_filter()
    run(f, range(100))
    est = f.state()
    A = rng.standard_normal((12, 12))
    Sigma = 1e-8 * (A @ A.T + np.eye(12))  # small enough for first order, and correlated
    est.P_trans, est.P_rot, est.P_cross = Sigma[:6, :6], Sigma[6:, 6:], Sigma[:6, 6:]
    d, R_off = OFFSET
    child = est.transported(d, R_off)
    rows = []
    for e in rng.multivariate_normal(np.zeros(12), Sigma, size=10000):
        moved = PoseEstimate(est.t, est.p + e[0:3], est.v + e[3:6], est.R @ so3_exp(e[6:9]),
                             est.w + e[9:12], est.P_trans, est.P_rot).transported(d, R_off)
        rows.append(np.concatenate([moved.p - child.p, moved.v - child.v,
                                    so3_log(child.R.T @ moved.R), moved.w - child.w]))
    P = np.block([[child.P_trans, child.P_cross], [child.P_cross.T, child.P_rot]])
    assert np.allclose(np.cov(np.array(rows).T), P, rtol=0.1, atol=0.05 * np.abs(P).max())


def test_odometry_covariances_move_each_block_to_its_message_frame():
    f = make_filter()
    f.reset(0.0, np.zeros(3), rot_z(math.pi / 2))
    f.P_trans_ = np.diag([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])  # p, v in the world
    f.P_rot_ = np.diag([0.1, 0.2, 0.3, 1.0, 2.0, 3.0])  # d, w in the body
    pose, twist = f.state().odometry_covariances()
    # the body's x is the world's y at a heading of ninety degrees, and the other way round
    assert np.allclose(pose, np.diag([1.0, 2.0, 3.0, 0.2, 0.1, 0.3]), rtol=0.0, atol=1e-15)
    assert np.allclose(twist, np.diag([20.0, 10.0, 30.0, 1.0, 2.0, 3.0]), rtol=0.0, atol=1e-14)


def test_a_bad_parameter_is_refused_by_name():
    sizes = 'PoseFilter: noise_psd, measurement_std and initial_std take 2 entries each'
    with pytest.raises(ValueError, match=sizes):
        PoseFilter((1.0,), **FILTER_ARGS)
    with pytest.raises(ValueError, match=sizes):
        PoseFilter((1.0, 1.0), **{**FILTER_ARGS, 'initial_std': (1.0, 2.0, 5.0, 20.0)})
    with pytest.raises(ValueError, match='PoseFilter: noise_psd must be >= 0'):
        PoseFilter((-1.0, 1.0), **FILTER_ARGS)
    with pytest.raises(ValueError, match='PoseFilter: gate takes 2 entries > 0'):
        PoseFilter((1.0, 1.0), **{**FILTER_ARGS, 'gate': (0.02,)})
    with pytest.raises(ValueError, match='rot_huber_k must be >= 0'):
        PoseFilter((1.0, 1.0), **{**FILTER_ARGS, 'rot_huber_k': -1.0})
