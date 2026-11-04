"""Independent native-MuJoCo and actual 50 Hz MotionLib foot validation."""

# Support both direct execution and python -m from the repository layout.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_root = str(_ToolPath(__file__).resolve().parents[2])
if _tool_root not in _tool_sys.path:
    _tool_sys.path.insert(0, _tool_root)

import argparse
import json
from pathlib import Path
import pickle
import sys

import mujoco as mj
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from supermimic.s54 import JOINT_NAMES, model_contract
from supermimic.motion import MotionLib
from supermimic.kinematics import S54Kinematics
from supermimic.rotations import mat_to_quat
from tools.tests.test_s54_cpu import fk_test
from tools.data.s54_grounding import FootGeometry


def native_gaps(model, values):
    """Use native signed collision distances, not the export support formula."""
    geometry = FootGeometry(model)
    data = mj.MjData(model)
    floor = model.geom('floor').id
    gaps = np.empty((len(values), 2))
    support_error = 0.
    samples = set(np.linspace(0, len(values) - 1, min(128, len(values))).astype(int))
    analytic = geometry.gaps(values)
    for i, row in enumerate(values):
        data.qpos[:] = model.qpos0
        data.qpos[:3] = row[:3]
        data.qpos[3:7] = row[[6, 3, 4, 5]]
        data.qpos[geometry.addresses] = row[7:]
        mj.mj_forward(model, data)
        for side in range(2):
            gap = min(mj.mj_geomDistance(model, data, floor, g, 10., None)
                      for g in geometry.collision_ids[side])
            gaps[i, side] = gap
            if i in samples:
                # Independently transform ALL mesh vertices using MuJoCo's
                # world geom frame, rather than the exported hull/body frame.
                for g in np.flatnonzero((model.geom_bodyid == geometry.bodies[side])
                                        & (model.geom_type == mj.mjtGeom.mjGEOM_MESH)):
                    mesh = model.geom_dataid[g]
                    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                    z = model.mesh_vert[start:start + count] @ data.geom_xmat[g].reshape(3, 3)[2] + data.geom_xpos[g, 2]
                    gap = min(gap, float(z.min()))
                support_error = max(support_error, abs(gap - analytic[i, side]))
    return gaps, support_error, analytic


def validate(csv_path):
    csv_path = Path(csv_path)
    stem, out = csv_path.stem, csv_path.parent
    torch.set_num_threads(4)
    kin = S54Kinematics()
    motion = MotionLib([str(csv_path)], kin, 'cpu', require_collision_safe=False)
    source = np.loadtxt(csv_path, delimiter=',', ndmin=2)
    root_quat = mat_to_quat(motion.root_R).numpy()[:, [1, 2, 3, 0]]
    control = np.c_[motion.root_p.numpy(), root_quat, motion.q.numpy()]
    model, _ = model_contract()
    src_collision, src_error, src_all = native_gaps(model, source)
    ctl_collision, ctl_error, ctl_all = native_gaps(model, control)
    with (out / 'reference' / f'{stem}_30fps.pkl').open('rb') as f:
        pk = pickle.load(f)
    with (out / 'baseline' / f'{stem}_30fps.pkl').open('rb') as f:
        baseline = pickle.load(f)
    raw = np.loadtxt(out / 'raw' / csv_path.name, delimiter=',', ndmin=2)
    original = np.c_[baseline['root_pos'], baseline['root_rot'], baseline['dof_pos']]
    metadata = json.loads(csv_path.with_suffix('.json').read_text())
    upper = json.loads((out / 'diagnostics' / stem / 'upper_body_metrics.json').read_text())
    tolerance = metadata['grounding']['config']['validation_tolerance_m']
    minimum = metadata['grounding']['config']['clearance_m'] - tolerance
    recorded_joints = original[:, 7:20]
    if (out / 'diagnostics' / stem / 'self_collision_adjustment.npz').exists():
        adjustment = np.load(out / 'diagnostics' / stem / 'self_collision_adjustment.npz')
        recorded_joints = adjustment['joint_positions'][:, :13]
    checks = dict(
        source_30hz_nonpenetrating=bool(min(src_collision.min(), src_all.min()) >= minimum),
        motionlib_50hz_nonpenetrating=bool(min(ctl_collision.min(), ctl_all.min()) >= minimum),
        full_mesh_and_native_support_agree=bool(max(src_error, ctl_error) < 1e-7),
        csv_raw_pkl_equal=bool(np.array_equal(source[:, :34], raw)
                              and np.array_equal(np.c_[pk['root_pos'], pk['root_rot'], pk['dof_pos']], raw)),
        unchanged_root_xy_rotation=bool(np.array_equal(original[:, [0, 1, 3, 4, 5, 6]], source[:, [0, 1, 3, 4, 5, 6]])),
        legs_waist_match_recorded_trajectory=bool(np.array_equal(recorded_joints, source[:, 7:20])),
        upper_body_acceptance=bool(all(upper['checks'].values())),
        finite_motionlib=all(bool(torch.isfinite(x).all()) for x in (motion.q, motion.qd, motion.root_p, motion.root_v, motion.body_p)))
    report = dict(clip=stem, status='PASS' if all(checks.values()) else 'FAIL', checks=checks,
                  source_frames=len(source), control_frames=motion.num_frames,
                  source_min_collision_gap_m=float(src_collision.min()),
                  control_min_collision_gap_m=float(ctl_collision.min()),
                  source_min_visual_collision_gap_m=float(src_all.min()),
                  control_min_visual_collision_gap_m=float(ctl_all.min()),
                  max_independent_support_error_m=max(src_error, ctl_error),
                  fk=fk_test(kin, motion, min(128, motion.num_frames)))
    path = out / 'diagnostics' / stem / 'grounding_validation.json'
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if not all(checks.values()):
        raise ValueError(f'Independent grounding validation failed: {path}')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    validate(parser.parse_args().csv)
