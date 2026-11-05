"""The node's estimators on the 2026-11-04 g1 recordings (data/); needs tf_transformations.

The param file's filter must keep each of the four outputs within 20 % of what its defaults scored
when they were set, and must stay better than what the node published before: the raw Vicon pose
on jumps, the frame-clocked finite difference on the twists.
"""
from pathlib import Path

import pytest

import replay_vicon_log as rv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / '2026-11-04_g1_static_dynamic'
SKIP = {'static': ((18.0, 25.0),), 'dynamic': ()}  # static: g1 disturbed from 18 to 25 s
SCORED = {  # the defaults' error RMS and 99.9th percentile jump: m, rad, m/s and rad/s
    'static': {'position': (2.916e-5, 1.227e-4), 'rotation': (9.803e-5, 3.987e-4),
               'velocity': (1.805e-3, 7.036e-3), 'angular velocity': (5.831e-3, 2.136e-2)},
    'dynamic': {'position': (4.063e-4, 6.943e-4), 'rotation': (1.569e-3, 2.500e-3),
                'velocity': (2.456e-2, 3.112e-2), 'angular velocity': (8.899e-2, 9.978e-2)},
}


@pytest.fixture(scope='module', params=sorted(SKIP))
def scored(request):
    params = rv.read_params(ROOT / 'param' / 'vicon_interface.param.yaml')
    log = rv.read_markers(DATA / f'{request.param}_markers.npz', 250.0)
    model = params['kf_motion_model']
    index = params['vicon_object_names'].index('g1')
    *_, scores, _ = rv.evaluate(log, params, index, (model,), SKIP[request.param])
    return request.param, f'filter {model}', scores


@pytest.mark.parametrize('quantity', [q for q, _ in rv.QUANTITIES])
def test_the_filter_keeps_its_scores(scored, quantity):
    recording, name, scores = scored
    rms, _, _, jump, _ = scores[quantity][name]
    rms_set, jump_set = SCORED[recording][quantity]
    assert rms <= 1.2 * rms_set, (recording, rms)
    assert jump <= 1.2 * jump_set, (recording, jump)


def test_the_filter_stays_better_than_what_was_published_before(scored):
    recording, name, scores = scored
    for quantity in ('position', 'rotation'):  # the raw Vicon pose flickers
        assert scores[quantity][name][3] < 0.5 * scores[quantity]['vicon'][3], quantity
    for quantity in ('velocity', 'angular velocity'):
        assert scores[quantity][name][0] < 0.7 * scores[quantity]['frame difference'][0], quantity
        assert scores[quantity][name][3] < 0.3 * scores[quantity]['frame difference'][3], quantity
