"""The raw-log replay on a small synthetic log; no ROS needed but tf_transformations."""
import csv
import time

import numpy as np
import tf_transformations

import replay_vicon_log as rv

RATE = 250.0  # Hz
COLUMNS = ('object', 'frame_number', 'frame_rate', 'host_time', 'latency',
           'x_mm', 'y_mm', 'z_mm', 'euler_x', 'euler_y', 'euler_z')  # the node's RAW_LOG_COLUMNS


def write_log(path, v, frames, stall_every=50):
    """Write g1 moving at v (m/s, world) and turning, picked up 6 ms late every stall_every."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for k in frames:
            t = k / RATE
            m = tf_transformations.euler_matrix(0.0, 0.0, 0.5 * t, 'sxyz')
            ex, ey, ez = tf_transformations.euler_from_matrix(m, 'rxyz')
            delay = 0.006 if k % stall_every == 0 else 0.0  # s, the host's own waiting
            writer.writerow(['g1', k, RATE, 100.0 + t + 0.005 + delay, 0.005,
                             *(1000.0 * v * t), ex, ey, ez])


def test_the_replay_separates_host_delays_from_the_frame_clock(tmp_path, monkeypatch):
    params = rv.read_params('param/vicon_interface.param.yaml')
    params['child_frame_offset_xyz'] = [0.0] * 6  # markers on the child, so v is its velocity
    params['child_frame_offset_rpy_deg'] = [0.0] * 6
    v = np.array([0.4, -0.2, 0.05])
    write_log(tmp_path / 'raw.csv', v, [k for k in range(500) if k % 97 != 5])  # some dropped
    clock = rv.ReplayClock()
    monkeypatch.setattr(time, 'monotonic', clock.monotonic)
    log = rv.read_log(tmp_path / 'raw.csv', 'g1')
    out = rv.replay(log, params, params['vicon_object_names'].index('g1'), clock)
    v_body = np.einsum('kji,j->ki', out['R'], v)  # m/s, the child frame's
    late = slice(100, None)  # once every estimate has settled
    for name in ('frame difference', 'filter cv', 'filter ca'):
        assert np.nanmax(np.abs(out[name][late] - v_body[late])) < 1e-3, name
    assert np.nanmax(np.abs(out['host difference'][late] - v_body[late])) > 0.05


def test_the_replay_prints_its_table(tmp_path, monkeypatch, capsys):
    write_log(tmp_path / 'raw.csv', np.array([0.3, 0.0, 0.0]), range(400))
    monkeypatch.setattr('sys.argv', ['replay_vicon_log.py', str(tmp_path / 'raw.csv'),
                                     '--static', '0', '1'])
    rv.main()
    printed = capsys.readouterr().out
    for name in rv.TWISTS + ('host pick-up delay', 'kf_measurement_std'):
        assert name in printed
