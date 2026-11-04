"""Tests for PoseFilter, the pose-only Kalman filter; numpy only, no ROS."""
import copy
import math

import numpy as np
import pytest

from vicon_interface.state_estimators.pose_filter import (
    _mahalanobis,
    kinematic_blocks,
    PoseFilter,
    so3_exp,
    so3_log,
    so3_right_jacobian,
)

DT = 0.004  # s, Vicon frame interval at 250 Hz
MODELS = ('cv', 'ca')
NOISE_PSD = {'cv': (0.036, 0.9), 'ca': (36.0, 900.0)}  # "ca" at "cv"'s velocity noise
STD = (2e-4, 1e-3)  # m, rad
FILTER_ARGS = {
    'measurement_std': STD,
    'initial_std': (1.0, 2.0, 10.0, 20.0),
    'gate': (0.02, 0.2),
    'max_rejections': 5,
    'reset_gap': 0.1,
}
AXIS = np.array([0.3, -0.4, 0.866]) / np.linalg.norm([0.3, -0.4, 0.866])


def make_filter(model):
    return PoseFilter(model, NOISE_PSD[model], **FILTER_ARGS)


def rot_z(angle):
    return so3_exp(np.array([0.0, 0.0, angle]))


def motion(model, t):
    """Return (p, R, v, w) of a motion the model holds exactly.

    Constant velocity for 'cv', constant acceleration for 'ca'; the rotation keeps one axis,
    so that its exponential stays exact.
    """
    a = np.array([0.5, 0.3, -0.2]) if model == 'ca' else np.zeros(3)  # m/s^2
    v0 = np.array([0.4, -0.2, 0.05])  # m/s
    p = np.array([1.0, -2.0, 0.8]) + v0 * t + 0.5 * a * t * t
    alpha = 1.5 if model == 'ca' else 0.0  # rad/s^2
    R = so3_exp(np.array([0.2, -0.1, 2.5])) @ so3_exp(AXIS * (0.8 * t + 0.5 * alpha * t * t))
    return p, R, v0 + a * t, AXIS * (0.8 + alpha * t)


def run(f, model, frames, noise=None):
    """Feed f the frames of motion(model), noise(k) added to frame k if given; return outcomes."""
    outcomes = []
    for k in frames:
        p, R, _, _ = motion(model, k * DT)
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


def steady_rate_std(order, q, std):
    """Return the std of the steady filter's rate estimate under white measurement noise alone."""
    F, Q = kinematic_blocks(order, DT)
    H = np.zeros((1, order))
    H[0, 0] = 1.0
    P = np.eye(order)
    for _ in range(3000):
        P = F @ (P - P @ H.T @ H @ P / (P[0, 0] + std ** 2)) @ F.T + q * Q
    K = P[:, :1] / (P[0, 0] + std ** 2)
    A = (np.eye(order) - K @ H) @ F  # the posterior's closed loop
    S = np.zeros((order, order))
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


def test_kinematic_blocks_are_the_closed_forms_and_compose():
    dt = 0.01
    F, Q = kinematic_blocks(2, dt)
    assert np.allclose(F, [[1.0, dt], [0.0, 1.0]], rtol=0.0, atol=1e-15)
    assert np.allclose(Q, [[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]], rtol=1e-14, atol=0.0)
    F, Q = kinematic_blocks(3, dt)
    assert np.allclose(F, [[1.0, dt, dt ** 2 / 2], [0.0, 1.0, dt], [0.0, 0.0, 1.0]],
                       rtol=0.0, atol=1e-15)
    assert np.allclose(Q, [[dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
                           [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
                           [dt ** 3 / 6, dt ** 2 / 2, dt]], rtol=1e-14, atol=0.0)
    F2, Q2 = kinematic_blocks(3, 2 * dt)  # the noise is integrated exactly: two steps are one
    assert np.allclose(F2, F @ F, rtol=1e-14, atol=0.0)
    assert np.allclose(Q2, F @ Q @ F.T + Q, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize('order', (2, 3))
def test_kinematic_blocks_are_the_exact_discretization(order):
    Ac = np.diag(np.ones(order - 1), 1)  # each state is the rate of the one before
    G = np.zeros((order, 1))
    G[-1, 0] = 1.0  # unit white noise drives the last state
    for dt in (0.004, 0.008, 0.05):
        # Van Loan: expm([[-Ac, G G^T], [0, Ac^T]] dt) = [[., F^-1 Q], [0, F^T]]
        M = np.zeros((2 * order, 2 * order))
        M[:order, :order] = -Ac
        M[:order, order:] = G @ G.T
        M[order:, order:] = Ac.T
        E = expm_series(M * dt)
        F_ref = E[order:, order:].T
        F, Q = kinematic_blocks(order, dt)
        assert np.allclose(F, F_ref, rtol=0.0, atol=1e-15)
        assert np.allclose(Q, F_ref @ E[:order, order:], rtol=1e-12, atol=1e-30)


@pytest.mark.parametrize('model', MODELS)
def test_motion_the_model_holds_is_tracked_exactly(model):
    f = make_filter(model)
    assert run(f, model, range(400)) == ['reset'] + ['accepted'] * 399
    est = f.state()
    p, R, v, w = motion(model, 399 * DT)
    assert np.allclose(est.p, p, atol=1e-9)
    assert np.allclose(est.v, v, atol=1e-7)
    assert np.allclose(so3_log(est.R.T @ R), 0.0, atol=1e-9)
    assert np.allclose(est.w, w, atol=1e-7)


@pytest.mark.parametrize('model', MODELS)
def test_static_noise_reaches_the_rates_as_the_steady_riccati_solution_predicts(model):
    f = make_filter(model)
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
    order = 2 if model == 'cv' else 3
    v_std = steady_rate_std(order, NOISE_PSD[model][0], STD[0])
    w_std = steady_rate_std(order, NOISE_PSD[model][1], STD[1])
    assert np.allclose(np.std(v, axis=0), v_std, rtol=0.2)
    assert np.allclose(np.std(w, axis=0), w_std, rtol=0.2)


@pytest.mark.parametrize('model', MODELS)
def test_dropped_frames_keep_the_exact_interval(model):
    f = make_filter(model)
    frames = [k for k in range(400) if k % 7 != 3 and not 200 <= k < 203]
    assert set(run(f, model, frames)[1:]) == {'accepted'}
    est = f.state()
    _, _, v, w = motion(model, frames[-1] * DT)
    assert np.allclose(est.v, v, atol=1e-7)
    assert np.allclose(est.w, w, atol=1e-7)


@pytest.mark.parametrize('model', MODELS)
def test_a_frame_at_or_before_the_last_one_is_stale_and_changes_nothing(model):
    f = make_filter(model)
    run(f, model, range(50))
    before = f.state()
    p, R, _, _ = motion(model, 49 * DT)
    assert f.update(49 * DT, p + 0.01, R) == 'stale'
    assert f.update(45 * DT, p, R) == 'stale'
    after = f.state()
    for name in ('t', 'p', 'v', 'R', 'w', 'P_trans', 'P_rot'):
        assert np.array_equal(getattr(before, name), getattr(after, name))


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('kind', ('position', 'orientation'))
def test_a_single_outlier_is_rejected_as_if_the_frame_never_came(model, kind):
    rng = np.random.default_rng(3)
    noise = {k: (STD[0] * rng.standard_normal(3), STD[1] * rng.standard_normal(3))
             for k in range(300)}
    jump = {
        'position': (np.array([0.05, 0.0, 0.0]), np.zeros(3)),  # m
        'orientation': (np.zeros(3), np.array([0.0, 0.3, 0.0])),  # rad
    }
    seen, unseen = make_filter(model), make_filter(model)
    outcomes = run(seen, model, range(300), lambda k: jump[kind] if k == 200 else noise[k])
    assert outcomes[200] == 'rejected' and set(outcomes[201:]) == {'accepted'}
    run(unseen, model, [k for k in range(300) if k != 200], lambda k: noise[k])
    a, b = seen.state(), unseen.state()
    assert np.allclose(a.p, b.p, atol=1e-8)
    assert np.allclose(a.v, b.v, atol=1e-6)
    assert np.allclose(a.w, b.w, atol=1e-5)


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('kind, size, outcome', [
    ('position', 0.03, 'rejected'), ('position', 0.01, 'accepted'),  # m
    ('orientation', 0.3, 'rejected'), ('orientation', 0.1, 'accepted'),  # rad
])
def test_the_gate_rejects_only_beyond_its_physical_thresholds(model, kind, size, outcome):
    f = make_filter(model)
    run(f, model, range(200))
    p, R, _, _ = motion(model, 200 * DT)
    if kind == 'position':
        p = p + np.array([size, 0.0, 0.0])
    else:
        R = R @ so3_exp(np.array([0.0, size, 0.0]))
    assert f.update(200 * DT, p, R) == outcome


@pytest.mark.parametrize('model', MODELS)
def test_the_huber_weight_leaves_small_residuals_alone_and_scales_a_large_one(model):
    rng = np.random.default_rng(5)
    small = {k: (np.zeros(3), 1e-5 * rng.standard_normal(3)) for k in range(200)}  # rad, << k
    plain = make_filter(model)
    huber = PoseFilter(model, NOISE_PSD[model], **{**FILTER_ARGS, 'rot_huber_k': 2.0})
    run(plain, model, range(200), lambda k: small[k])
    run(huber, model, range(200), lambda k: small[k])
    for name in ('p', 'v', 'R', 'w', 'P_trans', 'P_rot'):
        assert np.array_equal(getattr(plain.state(), name), getattr(huber.state(), name))
    p, R, _, _ = motion(model, 200 * DT)
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


@pytest.mark.parametrize('model', MODELS)
def test_a_lasting_jump_resets_after_max_rejections(model):
    f = make_filter(model)
    shift = np.array([0.0, 0.05, 0.0])  # m: the body moved for good, or was redefined
    outcomes = run(f, model, range(300),
                   lambda k: (shift if k >= 200 else np.zeros(3), np.zeros(3)))
    assert outcomes[200:205] == ['rejected'] * 4 + ['reset']
    assert set(outcomes[205:]) == {'accepted'}
    est = f.state()
    p, _, v, w = motion(model, 299 * DT)
    assert np.allclose(est.p, p + shift, atol=1e-9)
    assert np.allclose(est.v, v, atol=1e-6)
    assert np.allclose(est.w, w, atol=1e-6)


@pytest.mark.parametrize('model', MODELS)
def test_a_gap_or_a_restarted_clock_restarts_from_the_measurement(model):
    f = make_filter(model)
    run(f, model, range(50))
    p, R, _, _ = motion(model, 50 * DT + 0.2)
    assert f.update(50 * DT + 0.2, p, R) == 'reset'  # 200 ms without a frame
    est = f.state()
    assert np.array_equal(est.p, p) and np.array_equal(est.R, R)
    assert not np.any(est.v) and not np.any(est.w)
    assert f.update(0.0, p, R) == 'reset'  # a Vicon restart sends the frame numbers back


@pytest.mark.parametrize('model', MODELS)
def test_the_angular_velocity_is_the_body_one_at_a_heading_of_ninety_degrees(model):
    f = make_filter(model)
    w_body = np.array([1.0, 0.0, 0.0])  # rad/s, rolling about the body's x axis
    for k in range(200):
        f.update(k * DT, np.zeros(3), rot_z(math.pi / 2) @ so3_exp(w_body * k * DT))
    assert np.allclose(f.state().w, w_body, atol=1e-7)  # the Euler-rate twist read [0, -1, 0]


@pytest.mark.parametrize('model', MODELS)
def test_yaw_through_pi_does_not_disturb_the_estimate(model):
    f = make_filter(model)
    rate = 2.0  # rad/s; yaw passes pi at t = 0.6 s, once the start has settled
    worst = 0.0
    for k in range(250):
        f.update(k * DT, np.zeros(3), rot_z(math.pi - 1.2 + rate * k * DT))
        if k >= 100:
            worst = max(worst, float(np.max(np.abs(f.state().w - [0.0, 0.0, rate]))))
    assert worst < 1e-6


@pytest.mark.parametrize('model', MODELS)
def test_predicted_moves_the_estimate_and_leaves_the_filter_unchanged(model):
    f = make_filter(model)
    run(f, model, range(300))
    before = f.state()
    t = 299 * DT + 0.02  # s, 20 ms ahead
    ahead = f.predicted(t)
    p, R, v, w = motion(model, t)
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


def test_odometry_covariances_move_each_block_to_its_message_frame():
    f = make_filter('cv')
    f.reset(0.0, np.zeros(3), rot_z(math.pi / 2))
    f.P_trans_ = np.diag([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])  # p, v in the world
    f.P_rot_ = np.diag([0.1, 0.2, 0.3, 1.0, 2.0, 3.0])  # d, w in the body
    pose, twist = f.state().odometry_covariances()
    # the body's x is the world's y at a heading of ninety degrees, and the other way round
    assert np.allclose(pose, np.diag([1.0, 2.0, 3.0, 0.2, 0.1, 0.3]), rtol=0.0, atol=1e-15)
    assert np.allclose(twist, np.diag([20.0, 10.0, 30.0, 1.0, 2.0, 3.0]), rtol=0.0, atol=1e-14)


def test_a_bad_model_or_parameter_is_refused_by_name():
    with pytest.raises(ValueError, match="model must be one of \\['ca', 'cv'\\], got 'cj'"):
        PoseFilter('cj', (1.0, 1.0), **FILTER_ARGS)
    with pytest.raises(ValueError, match='PoseFilter: noise_psd and measurement_std take 2'):
        PoseFilter('cv', (1.0,), **FILTER_ARGS)
    with pytest.raises(ValueError, match='PoseFilter: noise_psd must be >= 0'):
        PoseFilter('cv', (-1.0, 1.0), **FILTER_ARGS)
    with pytest.raises(ValueError, match='PoseFilter: gate takes 2 entries > 0'):
        PoseFilter('cv', (1.0, 1.0), **{**FILTER_ARGS, 'gate': (0.02,)})
    with pytest.raises(ValueError, match='rot_huber_k must be >= 0'):
        PoseFilter('cv', (1.0, 1.0), **{**FILTER_ARGS, 'rot_huber_k': -1.0})
