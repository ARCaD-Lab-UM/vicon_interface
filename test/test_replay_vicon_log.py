"""The recording replay and its scores, on small synthetic recordings; needs tf_transformations."""
import csv

import numpy as np
import tf_transformations

import replay_vicon_log as rv
from vicon_interface.state_estimators.pose_filter import so3_exp, so3_log

RATE = 250.0  # Hz
COLUMNS = ('object', 'frame_number', 'frame_rate', 'host_time', 'latency',
           'x_mm', 'y_mm', 'z_mm', 'euler_x', 'euler_y', 'euler_z')  # the node's RAW_LOG_COLUMNS
V = np.array([0.4, -0.2, 0.05])  # m/s, the rigid body's origin, world
W = np.array([0.3, -0.2, 0.9])  # rad/s, the rigid body's, body frame
R0 = so3_exp(np.array([0.1, -0.2, 2.0]))


def body(t):
    """Return the rigid body's position (m) and rotation at t (s): constant v and body rate."""
    return np.array([1.0, 2.0, 0.9]) + V * t, R0 @ so3_exp(W * t)


def frames(n, stall_every=50, drop_every=97):
    """Yield (k, host time, latency, x, y, z mm, SDK Euler triple) of n frames, a few dropped."""
    for k in range(n):
        if k % drop_every == 5:
            continue
        p, R = body(k / RATE)
        m = np.eye(4)
        m[:3, :3] = R
        delay = 0.006 if k % stall_every == 0 else 0.0  # s, the host's own waiting
        yield (k, 100.0 + k / RATE + 0.005 + delay, 0.005, *(1000.0 * p),
               *tf_transformations.euler_from_matrix(m, 'rxyz'))


def write_csv(path, n):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for k, host, latency, *rest in frames(n):
            writer.writerow(['g1', k, RATE, host, latency, *rest])


def write_npz(path, n):
    rows = []
    for k, host, latency, x, y, z, ex, ey, ez in frames(n):
        rows.append([host, 1.7e9 + host, k, latency, np.nan, x, y, z, ex, ey, ez, 0.0]
                    + [0.0] * 18)  # six markers' xyz, unused
    np.savez_compressed(path, rows=np.array(rows), names=np.array(['m1'] * 6))


def params():
    p = rv.read_params('param/vicon_interface.param.yaml')
    assert p['vicon_object_names'][0] == 'g1'
    return p


def test_the_references_are_exact_for_a_rigid_motion_with_an_offset():
    p = params()
    d, R_off = rv.offsets(p, 0)
    assert np.any(d)  # the param file's g1 sits below its rigid body
    t = np.arange(200) / RATE
    pose = [body(tk) for tk in t]
    p_body, R_body = np.array([x for x, _ in pose]), np.array([R for _, R in pose])
    P, RR, Vc, Wc = rv.references(t, p_body, R_body, d, R_off)
    for k in range(12, 188):
        p_k, R_k = pose[k]
        assert np.allclose(P[k], p_k + R_k @ d, atol=1e-9)
        assert np.allclose(so3_log((R_k @ R_off).T @ RR[k]), 0.0, atol=1e-9)
        v_child = (R_k @ R_off).T @ (V + R_k @ np.cross(W, d))  # body frame of the child
        assert np.allclose(Vc[k], v_child, atol=1e-8)
        assert np.allclose(Wc[k], R_off.T @ W, atol=1e-8)
    assert np.all(np.isnan(P[:12])) and np.all(np.isnan(P[-12:]))


def test_a_markers_file_reads_as_the_raw_log_does(tmp_path):
    write_csv(tmp_path / 'raw.csv', 300)
    write_npz(tmp_path / 'rec_markers.npz', 300)
    a = rv.read_log(tmp_path / 'raw.csv', 'g1')
    b = rv.read_markers(tmp_path / 'rec_markers.npz', RATE)
    assert set(a) == set(b)
    for key in a:
        assert np.allclose(a[key], b[key], rtol=1e-12, atol=0.0), key


def test_clean_frames_score_near_zero_and_host_stalls_reach_only_the_host_difference(tmp_path):
    write_npz(tmp_path / 'rec_markers.npz', 600)
    log = rv.read_markers(tmp_path / 'rec_markers.npz', RATE)
    # g1 sits below its rigid body: the filter tracks the body and moves the estimate down,
    # which is exact; tracking the child, swung on an arc by the rotation, would lag
    t, p, R, scores, lags = rv.evaluate(log, params(), 0)
    for name in ('vicon', 'filter cv', 'filter ca'):  # "ca" settles last, by 1e-8 at frame 50
        assert scores['position'][name][2] < 1e-7, name  # max error, m
        assert scores['rotation'][name][2] < 1e-7, name  # rad
    for name in ('filter cv', 'filter ca'):
        assert scores['velocity'][name][2] < 1e-5, name  # m/s
        assert scores['angular velocity'][name][2] < 1e-5, name  # rad/s
        assert lags['velocity'][name] == 0, name
    # the difference trails by half its window, which on the child's arc is about a mm/s
    assert scores['velocity']['frame difference'][2] < 2e-3
    assert scores['angular velocity']['frame difference'][2] < 1e-5
    assert scores['velocity']['host difference'][2] > 0.05  # the 6 ms stalls, through host time
    assert scores['angular velocity']['host difference'][2] > 0.05


def test_the_replay_prints_every_quantity_and_estimator(tmp_path, monkeypatch, capsys):
    write_csv(tmp_path / 'raw.csv', 400)
    monkeypatch.setattr('sys.argv', ['replay_vicon_log.py', str(tmp_path / 'raw.csv'),
                                     '--static', '0', '1', '--set', 'kf_rot_huber_k=2.0'])
    rv.main()
    printed = capsys.readouterr().out
    for word in [q for q, _ in rv.QUANTITIES] + list(rv.ESTIMATORS):
        assert word in printed, word
    assert 'host pick-up delay' in printed and 'kf_measurement_std' in printed
