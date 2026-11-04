"""Regression checks for distance gradients, neighbour rules and frame tunnelling."""

# Support both direct execution and python -m from the repository layout.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_root = str(_ToolPath(__file__).resolve().parents[2])
if _tool_root not in _tool_sys.path:
    _tool_sys.path.insert(0, _tool_root)

import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from supermimic.s54 import model_contract
from supermimic.motion import MotionLib
from supermimic.kinematics import S54Kinematics
from tools.data.s54_self_collision import CollisionGeometry, dense_contacts


def main():
    source = REPO / 'data/retarget_data/reference_grounded/dance1_subject1.csv'
    q = np.loadtxt(source, delimiter=',', ndmin=2)[:, 7:34]
    model, _ = model_contract()
    g = CollisionGeometry(model)
    x = q[943]
    distances, ids, jac = g.distances(x, .035, True)
    assert distances.min() < -.1, 'Regression input must contain the reported forearm/torso collision'
    errors = []
    for row, pair in enumerate(ids):
        if abs(distances[pair]) < 1e-4:
            continue
        for j in np.flatnonzero(abs(jac[row]) > 1e-4):
            left, right = x.copy(), x.copy()
            left[j] -= 1e-5; right[j] += 1e-5
            a, _, _ = g.distances(left, .035)
            b, _, _ = g.distances(right, .035)
            errors.append(abs((b[pair] - a[pair]) / 2e-5 - jac[row, j]))
    assert max(errors) < 1e-4, max(errors)
    assert g.certificate(q[942:945])['status'] == 'FAIL'
    names = set(tuple(sorted((model.body(int(model.geom_bodyid[a])).name,
                              model.body(int(model.geom_bodyid[b])).name))) for a, b in g.pairs)
    for pair in (('waist_yaw', 'zarm_l4_link'), ('leg_l3_link', 'zarm_l7_link'),
                 ('zarm_l7_link', 'zarm_r7_link'), ('leg_l6_link', 'leg_r6_link')):
        assert tuple(sorted(pair)) in names
    assert tuple(sorted(('zarm_l3_link', 'zarm_l4_link'))) not in names
    rng = np.random.default_rng(54)
    safe = [i for i in np.linspace(0, len(q) - 1, 160).astype(int) if g.distances(q[i])[0].min() > .008]
    crossing = None
    for _ in range(500):
        a, b = rng.choice(safe, 2, replace=False)
        if g.distances((q[a] + q[b]) * .5)[0].min() < -.005:
            crossing = (int(a), int(b))
            break
    assert crossing is not None, 'No frame-tunnelling regression fixture found'
    assert g.certificate(q[list(crossing)])['status'] == 'FAIL', 'Safe endpoints must not hide interpolation collisions'
    fixture = json.loads((REPO / 'tools/tests/fixtures/s54_subframe_collision.json').read_text())
    segment = np.array(fixture['joint_positions'])
    cfg = json.loads((REPO / 'configs/s54/retarget/s54_self_collision.json').read_text())
    _, coarse, _, _ = dense_contacts(g, segment, cfg)
    active, exact, constraints, _ = dense_contacts(g, segment, cfg, True, [fixture['fraction']])
    assert coarse > .003 and exact < .001 and len(active) > 0 and constraints.shape[0] > 0
    assert g.certificate(segment)['status'] == 'FAIL'
    for f in rng.choice(safe, 5, replace=False):
        delta = rng.uniform(-.02, .02, 27)
        a, _, _ = g.distances(q[f], detect=10., exhaustive=True)
        b, _, _ = g.distances(q[f] + delta, detect=10., exhaustive=True)
        assert np.all(abs(a - b) <= g.levers @ abs(delta) + 1e-5)
    try:
        MotionLib([str(source)], S54Kinematics(), 'cpu')
    except ValueError as exc:
        assert 'self-collision certificate' in str(exc), exc
    else:
        raise AssertionError('Uncertified legacy data must not enter training')
    report = dict(status='PASS', geom_pairs=len(g.pairs), gradient_max_error=max(errors),
                  interpolation_tunnelling_fixture=list(crossing),
                  known_collision_rejected=True, adaptive_subframe_regression=True,
                  lever_bounds=True, legacy_training_rejected=True)
    destination = REPO / 'results/s54/self_collision_regression.json'
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
