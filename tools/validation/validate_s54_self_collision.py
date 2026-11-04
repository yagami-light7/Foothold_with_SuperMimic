"""Fail-closed self collision acceptance for serialized S54 references."""

# Support both direct execution and python -m from the repository layout.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_root = str(_ToolPath(__file__).resolve().parents[2])
if _tool_root not in _tool_sys.path:
    _tool_sys.path.insert(0, _tool_root)

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from supermimic.s54 import model_contract, JOINT_NAMES, actuator_parameters
from supermimic.motion import MotionLib
from supermimic.kinematics import S54Kinematics
from tools.data.prepare_s54_motion import sha
from tools.data.s54_self_collision import CollisionGeometry, human_targets, quality, VERSION


def validate(csv_path):
    csv_path = Path(csv_path)
    out, stem = csv_path.parent, csv_path.stem
    meta = json.loads(csv_path.with_suffix('.json').read_text())
    cfg = meta['self_collision_repair']['config']
    source = np.loadtxt(csv_path, delimiter=',', ndmin=2)
    model, _ = model_contract()
    geometry = CollisionGeometry(model)
    torch.set_num_threads(4)
    motion = MotionLib([str(csv_path)], S54Kinematics(), 'cpu', require_collision_safe=False)
    certificate = geometry.certificate(source[:, 7:34], cfg['minimum_clearance_m'], cfg['certificate_max_depth'])
    control_minimum = float('inf')
    for values in motion.q.numpy():
        distances, _, _ = geometry.distances(values[:27])
        control_minimum = min(control_minimum, float(distances.min()))
    # Audit the broadphase with exhaustive native narrow-phase queries.
    close_error = 0.
    for f in np.linspace(0, len(source) - 1, min(32, len(source))).astype(int):
        broad, _, _ = geometry.distances(source[f, 7:34])
        native, _, _ = geometry.distances(source[f, 7:34], exhaustive=True)
        near = native < .04 - 1e-7
        if near.any():
            close_error = max(close_error, float(abs(broad[near] - native[near]).max()))
    actual_quality = quality(geometry, source[:, 7:34], human_targets(out, stem), cfg)
    parameters = actuator_parameters()
    speeds = np.array([parameters[n]['velocity'] for n in JOINT_NAMES[:27]])
    adjustment = np.load(out / 'diagnostics' / stem / 'self_collision_adjustment.npz')
    checks = dict(continuous_interpolation=certificate['status'] == 'PASS',
                  control_50hz=control_minimum >= cfg['minimum_clearance_m'] - 1e-5,
                  broadphase_no_missed_contacts=close_error < 1e-7,
                  recorded_adjustments_match=bool(np.array_equal(source[:, 7:34], adjustment['joint_positions'])),
                  geometry_matches_repair=certificate['geometry_sha256'] == meta['self_collision_repair']['certificate']['geometry_sha256'],
                  all_joint_velocity_limits=bool(np.all(abs(np.diff(source[:, 7:34], axis=0)) * 30 <= speeds + 1e-5)),
                  motion_quality=all(actual_quality['checks'].values()),
                  heads_remain_neutral=bool(np.all(source[:, 34:] == 0)),
                  finite_motionlib=bool(torch.isfinite(motion.q).all() and torch.isfinite(motion.qd).all()),
                  foot_grounding=all(meta.get('grounding', {}).get('checks', {'missing': False}).values()))
    report = dict(version=VERSION, status='PASS' if all(checks.values()) else 'FAIL', clip=stem,
                  csv_sha256=sha(csv_path), source_frames=len(source), control_frames=motion.num_frames,
                  continuous_certificate=certificate, control_minimum_distance_m=control_minimum,
                  exhaustive_native_close_contact_error_m=close_error, quality=actual_quality, checks=checks)
    report_path = out / 'diagnostics' / stem / 'self_collision_validation.json'
    report_path.write_text(json.dumps(report, indent=2))
    if report['status'] != 'PASS':
        raise ValueError(f'Self collision acceptance failed: {report_path}')
    meta['self_collision_validation'] = report
    csv_path.with_suffix('.json').write_text(json.dumps(meta, indent=2))
    print(f'Self collision validation PASS {stem}: continuous intervals + {motion.num_frames} control frames; '
          f'minimum control gap={control_minimum * 1000:.3f} mm', flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv', type=Path)
    validate(p.parse_args().csv)
