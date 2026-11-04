#!/usr/bin/env python3
"""Replay a Vicon recording through the node's estimators and score their four outputs.

The input is the node's raw log (raw_log_path, CSV) or a recording's markers file
(data/<recording>/*_markers.npz). From the repository root, with ROS sourced (for
tf_transformations):

    python3 replay_vicon_log.py data/2026-11-04_g1_static_dynamic/dynamic_markers.npz --object g1
    python3 replay_vicon_log.py /tmp/vicon_raw.csv --object g1 --static 0 20

For one object it scores, frame by frame, what the odometry topics can carry against zero-phase
references: quadratic fits over +-12 exact Vicon frame times of the rigid body's position and
orientation, moved to the child frame like the node moves them:

  vicon              the pose as Vicon gives it
  host difference    the rolling-window twist timed by host receive times (the old twist)
  frame difference   the same twist timed by the Vicon frame clock
  filter cv, ca      the pose filter at each frame, with either motion model

For position (mm), rotation (mrad), velocity (mm/s) and angular velocity (mrad/s) it prints the
error's RMS, 99.9th percentile and maximum; the 99.9th percentile and maximum of the error's change
from one frame to the next, which is a jump once the motion itself is taken out; and, for the
twists, the lag that minimises the RMS. It also prints how late the host picked frames up, and
with --static the rigid body's pose noise at rest, the values for kf_measurement_std.
Parameters can be overridden for a sweep: --set kf_cv_noise_psd=[0.036,0.05].
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import tf_transformations
import yaml

from vicon_interface.state_estimators.pose_filter import so3_exp, so3_log, so3_right_jacobian
from vicon_interface.vicon_data_buffer import ViconDataBuffer
from vicon_interface.vicon_data_filter import ViconDataFilter

QUANTITIES = (('position', 'mm'), ('rotation', 'mrad'), ('velocity', 'mm/s'),
              ('angular velocity', 'mrad/s'))
ESTIMATORS = ('vicon', 'host difference', 'frame difference', 'filter cv', 'filter ca')
WARMUP = 50  # frames left out of every score while the estimators start


class ReplayClock:
    """Stands in for time.monotonic(): the host time each frame was logged at."""

    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t


def read_log(path, name):
    """Return the object's rows of the raw log as a dict of numpy arrays, one entry per column."""
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r['object'] == name]
    if not rows:
        raise SystemExit(f'replay_vicon_log: no frame of `{name}` in {path}')
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k != 'object'}


def read_markers(path, rate):
    """Return a recording's markers file in read_log's columns; rate (Hz) is Vicon's frame rate."""
    rows = np.load(path, allow_pickle=False)['rows']
    keys = ('x_mm', 'y_mm', 'z_mm', 'euler_x', 'euler_y', 'euler_z')
    log = {k: rows[:, 5 + i] for i, k in enumerate(keys)}
    log.update(frame_number=rows[:, 2], frame_rate=np.full(len(rows), float(rate)),
               host_time=rows[:, 0], latency=rows[:, 3])
    return log


def read_params(path, overrides=()):
    """Return the node's parameters, with key=value overrides (values in YAML) applied."""
    with open(path) as f:
        params = yaml.safe_load(f)['/**']['ros__parameters']
    for item in overrides:
        key, _, value = item.partition('=')
        if key not in params:
            raise SystemExit(f'replay_vicon_log: --set names no parameter `{key}`')
        params[key] = yaml.safe_load(value)
    return params


def offsets(params, index):
    """Return the child frame's offset (m) and rotation in the rigid body's frame."""
    xyz = np.array(params['child_frame_offset_xyz'][3 * index:3 * index + 3], dtype=float)
    rpy = np.deg2rad(params['child_frame_offset_rpy_deg'][3 * index:3 * index + 3])
    return xyz, tf_transformations.euler_matrix(*rpy, 'sxyz')[:3, :3]


def make_filter(params, index, model):
    """Return the node's ViconDataFilter for the object at index, with the given motion model."""
    xyz, R_off = offsets(params, index)
    return ViconDataFilter(
        6, params['rolling_window_size'], offset_xyz=xyz, offset_R=R_off,
        motion_model=model, noise_psd=params[f'kf_{model}_noise_psd'],
        measurement_std=params['kf_measurement_std'], initial_std=params['kf_initial_std'],
        gate_sigma=params['kf_gate_sigma'], max_rejections=params['kf_max_rejections'],
        max_delay=params['kf_max_delay'], clock_window=params['kf_clock_window'],
        offline_timeout=params['vicon_offline_timeout'],
        online_jitter_time=params['vicon_online_jitter_time'])


def sdk_rotation(ex, ey, ez):
    """Return the rotation matrix of the SDK's XYZ Euler triple, as the node converts it."""
    q = tf_transformations.quaternion_from_euler(ex, ey, ez, 'rxyz')
    return tf_transformations.quaternion_matrix(q)[:3, :3]


def replay(log, params, index, clock, models=('cv', 'ca')):
    """Feed every frame to the estimators as the node does.

    Return t (s, Vicon clock), the rigid body's pose per frame and, per estimator, its child-frame
    position, rotation, body velocity and body angular velocity (None where it has none).
    """
    filters = {m: make_filter(params, index, m) for m in models}
    first = filters[models[0]]
    host = ViconDataBuffer(6, params['rolling_window_size'], first.offset_xyz_, first.offset_R_)
    n = len(log['frame_number'])
    nan3, nan33 = np.full(3, np.nan), np.full((3, 3), np.nan)
    names = ('vicon', 'host difference', 'frame difference') + tuple(f'filter {m}' for m in models)
    est = {name: {k: [] for k in 'pRvw'} for name in names}
    body_p, body_R = [], []
    for i in range(n):
        clock.t = log['host_time'][i]
        R_sdk = sdk_rotation(log['euler_x'][i], log['euler_y'][i], log['euler_z'][i])
        m = np.eye(4)
        m[:3, :3] = R_sdk
        roll, pitch, yaw = tf_transformations.euler_from_matrix(m, 'sxyz')
        sample = (log['x_mm'][i], log['y_mm'][i], log['z_mm'][i], roll, pitch, yaw)
        rate = log['frame_rate'][i] if np.isfinite(log['frame_rate'][i]) else None
        for g in filters.values():
            g.add_pose(sample, log['latency'][i], int(log['frame_number'][i]), rate)
        host.add_pose(sample, log['latency'][i])
        body_p.append(np.array(sample[:3]) / 1000.0)  # m, the rigid body's origin
        body_R.append(R_sdk)
        est['vicon']['p'].append(first.get_latest_pose()[0][:3])
        est['vicon']['R'].append(first.R_child_.copy())
        for name, buffer in (('host difference', host), ('frame difference', first)):
            ready = buffer.is_ready()  # zero velocity until the window is full
            v, w, _ = buffer.get_latest_velocity()
            est[name]['v'].append(v if ready else nan3)
            est[name]['w'].append(w if ready else nan3)
        for model, g in filters.items():
            e, _ = g.get_frame_state()
            out = est[f'filter {model}']
            out['p'].append(nan3 if e is None else e.p)
            out['R'].append(nan33 if e is None else e.R)
            out['v'].append(nan3 if e is None else e.R.T @ e.v)
            out['w'].append(nan3 if e is None else e.w)
    est = {name: {k: (np.array(x) if x else None) for k, x in out.items()}
           for name, out in est.items()}
    t = log['frame_number'] / log['frame_rate']
    return t, np.array(body_p), np.array(body_R), est


def references(t, p, R, offset_xyz, offset_R, half=12):
    """Return the child's zero-phase position, rotation, body velocity and body angular velocity.

    Quadratic fits over +-half frames of the rigid body's position and of its orientation's body
    increments, moved to the child frame; NaN at the ends and across a restarted frame count.
    """
    n = len(t)
    P, V, W = (np.full((n, 3), np.nan) for _ in range(3))
    RR = np.full((n, 3, 3), np.nan)
    for k in range(half, n - half):
        s = slice(k - half, k + half + 1)
        tw = t[s] - t[k]  # s, exact frame times
        if not np.all(np.diff(tw) > 0.0):
            continue
        A = np.c_[np.ones_like(tw), tw, tw * tw]
        c_th = np.linalg.lstsq(A, [so3_log(R[k].T @ Rj) for Rj in R[s]], rcond=None)[0]
        c_p = np.linalg.lstsq(A, p[s], rcond=None)[0]
        Rk = R[k] @ so3_exp(c_th[0])
        w = so3_right_jacobian(c_th[0]) @ c_th[1]  # rad/s, body
        RR[k] = Rk @ offset_R
        P[k] = c_p[0] + Rk @ offset_xyz
        V[k] = RR[k].T @ (c_p[1] + Rk @ np.cross(w, offset_xyz))
        W[k] = offset_R.T @ w
    return P, RR, V, W


def errors(est, ref):
    """Return the estimator's per-frame error vectors, keyed by quantity, where it has them."""
    P, RR, V, W = ref
    out = {}
    if est['p'] is not None:
        out['position'] = est['p'] - P
    if est['R'] is not None:
        out['rotation'] = np.array([
            so3_log(Rr.T @ Re) if np.all(np.isfinite(Rr)) and np.all(np.isfinite(Re))
            else np.full(3, np.nan) for Rr, Re in zip(RR, est['R'])])
    if est['v'] is not None:
        out['velocity'] = est['v'] - V
    if est['w'] is not None:
        out['angular velocity'] = est['w'] - W
    return out


def score(err, mask):
    """Return RMS, 99.9th percentile and max of |err|, and of its frame-to-frame change."""
    ok = mask & np.all(np.isfinite(err), axis=1)
    e = np.linalg.norm(err[ok], axis=1)
    both = ok[1:] & ok[:-1]
    j = np.linalg.norm(np.diff(err, axis=0)[both], axis=1)
    if not len(e) or not len(j):
        return (np.nan,) * 5
    return (np.sqrt(np.mean(e ** 2)), np.percentile(e, 99.9), e.max(),
            np.percentile(j, 99.9), j.max())


def lag(x, ref, mask, max_shift=25):
    """Return the frames x trails ref by: the shift that minimises their RMS difference."""
    best, rms_best = 0, np.inf
    for s in range(max_shift + 1):
        a, b = x[s:], ref[:len(ref) - s]
        m = mask[s:] & np.all(np.isfinite(a), axis=1) & np.all(np.isfinite(b), axis=1)
        if m.any():
            rms = np.sqrt(np.mean(np.sum((a[m] - b[m]) ** 2, axis=1)))
            if rms < rms_best:
                best, rms_best = s, rms
    return best


def evaluate(log, params, index, models=('cv', 'ca'), skip=()):
    """Replay log and return t, the scores per quantity and estimator, and the lags in frames.

    skip holds (t0, t1) stretches, in seconds from the first frame, left out of every score.
    """
    clock = ReplayClock()
    monotonic, time.monotonic = time.monotonic, clock.monotonic  # the logged host times
    try:
        t, p, R, est = replay(log, params, index, clock, models)
    finally:
        time.monotonic = monotonic
    ref = references(t, p, R, *offsets(params, index))
    mask = np.ones(len(t), bool)
    mask[:WARMUP] = False
    for t0, t1 in skip:
        mask &= ~((t - t[0] >= t0) & (t - t[0] <= t1))
    scores, lags = {}, {}
    for name, e in est.items():
        for quantity, err in errors(e, ref).items():
            scores.setdefault(quantity, {})[name] = score(err, mask)
        for quantity, key, r in (('velocity', 'v', ref[2]), ('angular velocity', 'w', ref[3])):
            if e[key] is not None:
                lags.setdefault(quantity, {})[name] = lag(e[key], r, mask)
    return t, p, R, scores, lags


def static_noise(t, p, R, t0, t1):
    """Print the rigid body's pose noise over [t0, t1] s from the first frame."""
    m = (t - t[0] >= t0) & (t - t[0] <= t1)
    if m.sum() < 10:
        raise SystemExit('replay_vicon_log: fewer than 10 frames in --static')
    std_p = np.std(p[m], axis=0)  # m
    R_mid = R[m][m.sum() // 2]
    std_theta = np.std([so3_log(R_mid.T @ Rk) for Rk in R[m]], axis=0)  # rad, body
    print(f'\nrigid body at rest from {t0:g} to {t1:g} s, {m.sum()} frames: position std '
          f'{np.round(1e3 * std_p, 3)} mm, orientation std {np.round(1e3 * std_theta, 3)} mrad')
    print(f'kf_measurement_std: [{np.sqrt(np.mean(std_p ** 2)):.2e}, '
          f'{np.sqrt(np.mean(std_theta ** 2)):.2e}]')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('log', help='raw log the node wrote (raw_log_path), or a *_markers.npz')
    ap.add_argument('--object', help='object name; the raw log\'s first by default')
    ap.add_argument('--params', default='param/vicon_interface.param.yaml')
    ap.add_argument('--rate', type=float, default=250.0, help='Vicon frame rate of a .npz, Hz')
    ap.add_argument('--models', default='cv,ca', help='filter models to replay, comma-separated')
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                    help='override a parameter; the value is read as YAML')
    ap.add_argument('--skip', nargs=2, type=float, action='append', default=[],
                    metavar=('T0', 'T1'), help='seconds from the first frame to leave out')
    ap.add_argument('--static', nargs=2, type=float, metavar=('T0', 'T1'),
                    help='seconds from the first frame of a stretch at rest')
    args = ap.parse_args()
    params = read_params(args.params, args.set)
    npz = Path(args.log).suffix == '.npz'
    if args.object is None:
        if npz:
            raise SystemExit('replay_vicon_log: a .npz needs --object')
        with open(args.log) as f:
            args.object = next(csv.DictReader(f))['object']
    index = params['vicon_object_names'].index(args.object)
    log = read_markers(args.log, args.rate) if npz else read_log(args.log, args.object)
    t, p, R, scores, lags = evaluate(log, params, index, tuple(args.models.split(',')), args.skip)

    period = float(np.median(np.diff(t)))  # s
    steps = np.diff(log['frame_number'])
    late = log['host_time'] - log['latency'] - t
    late = 1e3 * (late - np.nanmin(late))  # ms the host picked each frame up after the earliest
    print(f'{args.object}: {len(t)} frames over {t[-1] - t[0]:.1f} s, '
          f'period {1e3 * period:.2f} ms, {int(np.sum(steps[steps > 1] - 1))} frames dropped')
    print(f'host pick-up delay: median {np.nanmedian(late):.2f} ms, '
          f'99.9 % {np.nanpercentile(late, 99.9):.2f} ms, max {np.nanmax(late):.2f} ms')
    for quantity, unit in QUANTITIES:
        print(f'\n{quantity + ", " + unit:22s} {"RMS":>8s} {"99.9 %":>8s} {"max":>8s} '
              f'{"jump 99.9 %":>12s} {"jump max":>9s}' + (f' {"lag ms":>7s}' if quantity in lags
                                                         else ''))
        for name, s in scores.get(quantity, {}).items():
            line = f'  {name:20s} ' + ' '.join(f'{1e3 * x:8.2f}' for x in s[:3])
            line += f' {1e3 * s[3]:12.2f} {1e3 * s[4]:9.2f}'
            if quantity in lags:
                line += f' {1e3 * lags[quantity][name] * period:7.1f}'
            print(line)
    if args.static is not None:
        static_noise(t, p, R, *args.static)


if __name__ == '__main__':
    main()
