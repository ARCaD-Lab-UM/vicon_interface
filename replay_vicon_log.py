#!/usr/bin/env python3
"""Replay a raw Vicon log through the node's estimators, offline and without ROS.

Record a log by setting raw_log_path in param/vicon_interface.param.yaml, then, from the
repository root with ROS sourced (for tf_transformations):

    python3 replay_vicon_log.py /tmp/vicon_raw.csv --object g1 --static 0 20

For one object it compares, frame by frame, the twists odometry/<obj> can carry with a
zero-phase reference, a line fit over +-12 exact Vicon frame times:

  host difference    the rolling-window difference timed by host receive times (the old twist)
  frame difference   the same difference timed by the Vicon frame clock
  filter cv, ca      the pose filter at each frame, with either motion model

For each it prints the RMS deviation from the reference at the lag that minimises it, that lag,
and the frame-to-frame jumps. It also prints how late the host picked frames up, which is what
reached the old twist, and with --static the pose noise of a stretch at rest, which is what
kf_measurement_std wants.
"""
import argparse
import csv
import time

import numpy as np
import tf_transformations
import yaml

from vicon_interface.state_estimators.pose_filter import so3_log
from vicon_interface.vicon_data_buffer import ViconDataBuffer
from vicon_interface.vicon_data_filter import ViconDataFilter

TWISTS = ('host difference', 'frame difference', 'filter cv', 'filter ca')


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


def read_params(path):
    with open(path) as f:
        return yaml.safe_load(f)['/**']['ros__parameters']


def make_filter(params, index, model):
    """Return the node's ViconDataFilter for the object at index, with the given motion model."""
    xyz = params['child_frame_offset_xyz'][3 * index:3 * index + 3]
    rpy = np.deg2rad(params['child_frame_offset_rpy_deg'][3 * index:3 * index + 3])
    return ViconDataFilter(
        6, params['rolling_window_size'],
        offset_xyz=xyz, offset_R=tf_transformations.euler_matrix(*rpy, 'sxyz')[:3, :3],
        motion_model=model, noise_psd=params[f'kf_{model}_noise_psd'],
        measurement_std=params['kf_measurement_std'], initial_std=params['kf_initial_std'],
        gate_sigma=params['kf_gate_sigma'], max_rejections=params['kf_max_rejections'],
        max_delay=params['kf_max_delay'], clock_window=params['kf_clock_window'],
        offline_timeout=params['vicon_offline_timeout'],
        online_jitter_time=params['vicon_online_jitter_time'])


def replay(log, params, index, clock):
    """Feed every frame to the estimators as the node does; return per-frame arrays."""
    filters = {m: make_filter(params, index, m) for m in ('cv', 'ca')}
    f = filters['cv']
    host = ViconDataBuffer(6, params['rolling_window_size'], f.offset_xyz_, f.offset_R_)
    out = {k: [] for k in ('t', 'p', 'R') + TWISTS}
    for i in range(len(log['frame_number'])):
        clock.t = log['host_time'][i]
        q = tf_transformations.quaternion_from_euler(
            log['euler_x'][i], log['euler_y'][i], log['euler_z'][i], 'rxyz')
        roll, pitch, yaw = tf_transformations.euler_from_quaternion(q, 'sxyz')
        sample = (log['x_mm'][i], log['y_mm'][i], log['z_mm'][i], roll, pitch, yaw)
        rate = log['frame_rate'][i] if np.isfinite(log['frame_rate'][i]) else None
        for g in filters.values():
            g.add_pose(sample, log['latency'][i], int(log['frame_number'][i]), rate)
        host.add_pose(sample, log['latency'][i])
        out['t'].append(log['frame_number'][i] / log['frame_rate'][i])  # s, Vicon clock
        out['p'].append(f.get_latest_pose()[0][:3])  # m, child frame in the world
        out['R'].append(f.R_child_.copy())
        for name, buffer in (('host difference', host), ('frame difference', f)):
            ready = buffer.is_ready()  # zero velocity until the window is full
            out[name].append(buffer.get_latest_velocity()[0] if ready else np.full(3, np.nan))
        for m, g in filters.items():
            est, _ = g.get_frame_state()
            out[f'filter {m}'].append(np.full(3, np.nan) if est is None else est.R.T @ est.v)
    return {k: np.array(v) for k, v in out.items()}


def reference(t, p, R, half=12):
    """Return the body-frame velocity of a line fit over +-half frames, NaN at the ends.

    A centred fit has no lag, and spreads its weight so thinly that the estimates' noise,
    which shares its samples, does not pull the lag search to one alignment.
    """
    v = np.full_like(p, np.nan)
    for k in range(half, len(t) - half):
        tw = t[k - half:k + half + 1] - t[k]  # s, exact frame times
        if not np.all(np.diff(tw) > 0.0):
            continue  # a restarted frame count
        tc = tw - tw.mean()
        pw = p[k - half:k + half + 1]
        v[k] = tc @ (pw - pw.mean(axis=0)) / (tc @ tc)  # m/s, world
    return np.einsum('kji,kj->ki', R, v)  # R^T v, frame by frame


def compare(name, v, ref, period):
    """Print the RMS deviation from ref at the lag that minimises it, the lag and the jumps."""
    ok = np.all(np.isfinite(v), axis=1)
    best = None
    for shift in range(26):  # frames the estimate may trail the reference by
        a, b = v[shift:], ref[:len(ref) - shift]
        m = ok[shift:] & np.all(np.isfinite(b), axis=1)
        rms = np.sqrt(np.mean(np.sum((a[m] - b[m]) ** 2, axis=1)))
        if best is None or rms < best[1]:
            best = (shift, rms)
    jumps = np.linalg.norm(np.diff(v[ok], axis=0), axis=1)  # m/s, frame to frame
    print(f'{name:18s} {1e3 * best[1]:10.1f} {1e3 * best[0] * period:8.1f} '
          f'{1e3 * np.percentile(jumps, 99.9):12.1f} {1e3 * jumps.max():10.1f}')


def static_noise(t, p, R, t0, t1):
    """Print the pose noise over [t0, t1] s from the first frame, and the kf_measurement_std."""
    m = (t - t[0] >= t0) & (t - t[0] <= t1)
    if m.sum() < 10:
        raise SystemExit('replay_vicon_log: fewer than 10 frames in --static')
    std_p = np.std(p[m], axis=0)  # m
    R_mid = R[m][m.sum() // 2]
    std_theta = np.std([so3_log(R_mid.T @ Rk) for Rk in R[m]], axis=0)  # rad, body
    print(f'\nat rest from {t0:g} to {t1:g} s, {m.sum()} frames: position std '
          f'{np.round(1e3 * std_p, 3)} mm, orientation std {np.round(1e3 * std_theta, 3)} mrad')
    print(f'kf_measurement_std: [{np.sqrt(np.mean(std_p ** 2)):.2e}, '
          f'{np.sqrt(np.mean(std_theta ** 2)):.2e}]')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('log', help='raw log the node wrote (raw_log_path)')
    ap.add_argument('--object', help='object name; the log\'s first by default')
    ap.add_argument('--params', default='param/vicon_interface.param.yaml')
    ap.add_argument('--static', nargs=2, type=float, metavar=('T0', 'T1'),
                    help='seconds from the first frame of a stretch at rest')
    args = ap.parse_args()
    params = read_params(args.params)
    if args.object is None:
        with open(args.log) as f:
            args.object = next(csv.DictReader(f))['object']
    index = params['vicon_object_names'].index(args.object)
    clock = ReplayClock()
    time.monotonic = clock.monotonic  # the estimators read the logged host times
    log = read_log(args.log, args.object)
    out = replay(log, params, index, clock)

    t = out['t']
    period = float(np.median(np.diff(t)))  # s
    steps = np.diff(log['frame_number'])
    late = log['host_time'] - log['latency'] - t
    late = 1e3 * (late - np.nanmin(late))  # ms the host picked each frame up after the earliest
    print(f'{args.object}: {len(t)} frames over {t[-1] - t[0]:.1f} s, '
          f'period {1e3 * period:.2f} ms, {int(np.sum(steps[steps > 1] - 1))} frames dropped')
    print(f'host pick-up delay: median {np.nanmedian(late):.2f} ms, '
          f'99.9 % {np.nanpercentile(late, 99.9):.2f} ms, max {np.nanmax(late):.2f} ms\n')
    ref = reference(t, out['p'], out['R'])
    print(f'{"twist, body frame":18s} {"RMS mm/s":>10s} {"lag ms":>8s} '
          f'{"jump 99.9 %":>12s} {"jump max":>10s}   (jumps in mm/s per frame)')
    for name in TWISTS:
        compare(name, out[name], ref, period)
    if args.static is not None:
        static_noise(t, out['p'], out['R'], *args.static)


if __name__ == '__main__':
    main()
