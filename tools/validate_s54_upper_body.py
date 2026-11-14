"""Independent CPU validation of the exported upper-body reference and native FK."""
import argparse
import json
from pathlib import Path
import pickle
import sys

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from supermimic.s54 import JOINT_NAMES, SOURCE_MJCF, actuator_parameters, default_joint_pos_list
from supermimic.motion import MotionLib
from supermimic.kinematics import S54Kinematics
from tools.test_s54_cpu import fk_test


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion_dir', type=Path, default=Path('library/motion_reference/kuavo_s54/retarget_data/reference'))
    p.add_argument('--baseline', type=Path, help='Baseline PKL/CSV; defaults to the newly generated dance1 baseline PKL')
    p.add_argument('--out', type=Path, default=Path('results/s54/upper_body_validation.json'))
    a = p.parse_args()
    a.baseline = a.baseline or a.motion_dir / 'baseline/dance1_subject1_30fps.pkl'
    stem = a.baseline.stem.removesuffix('_30fps')
    if (a.motion_dir / 'diagnostics' / stem / 'self_collision_metrics.json').exists():
        # Collision repair can intentionally change legs/waist and arm targets.
        # Recheck actual final geometry and human tracking rather than asserting
        # that the superseded arm-only intermediate is numerically unchanged.
        from validate_s54_self_collision import validate
        result = validate(a.motion_dir / f'{stem}.csv')
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=2))
        return
    if a.baseline.suffix == '.pkl':
        with a.baseline.open('rb') as f:
            source = pickle.load(f)
        heads = np.tile(default_joint_pos_list()[-2:], (len(source['root_pos']), 1))
        old = np.c_[source['root_pos'], source['root_rot'], source['dof_pos'], heads]
    else:
        old = np.loadtxt(a.baseline, delimiter=',')
    new = np.loadtxt(a.motion_dir / f'{stem}.csv', delimiter=',')
    assert new.shape == old.shape and new.shape[1] == 36
    ground_path = a.motion_dir / 'diagnostics' / stem / 'grounding_metrics.json'
    root_lift = np.zeros(len(new))
    columns = list(range(20))
    if ground_path.exists():
        guard = np.load(a.motion_dir / 'diagnostics' / stem / 'before_ground_guard.npz')
        root_lift = guard['lift_m']
        assert np.allclose(new[:, 2] - old[:, 2], root_lift, rtol=0, atol=1e-12)
        columns.remove(2)
    assert np.array_equal(old[:, columns], new[:, columns]), 'Root XY/rotation, legs or waist changed'
    assert np.array_equal(old[:, -2:], new[:, -2:]), 'Head defaults changed'
    assert np.isfinite(new).all()
    with (a.motion_dir / 'reference' / f'{stem}_30fps.pkl').open('rb') as f:
        pk = pickle.load(f)
    assert pk['fps'] == 30 and np.array_equal(np.c_[pk['root_pos'], pk['root_rot'], pk['dof_pos']], new[:, :34])
    velocity = actuator_parameters()
    above = {}
    for i, name in enumerate(JOINT_NAMES):
        count = int(np.sum(abs(np.diff(new[:, 7 + i])) * 30 > velocity[name]['velocity'] + 1e-6))
        if count:
            above[name] = count
    assert not any(n.startswith('zarm') for n in above), above
    torch.set_num_threads(4)
    kin = S54Kinematics()
    motion = MotionLib([str(a.motion_dir / f'{stem}.csv')], kin, 'cpu', require_collision_safe=False)
    assert motion.num_frames == int(np.floor((len(new) - 1) / 30 / .02)) + 1
    assert all(torch.isfinite(x).all() for x in (motion.q, motion.qd, motion.body_p, motion.body_R))
    model = mj.MjModel.from_xml_path(str(SOURCE_MJCF))
    data = mj.MjData(model)
    feet = [model.body(f'leg_{s}6_link').id for s in 'lr']
    floor = model.geom('floor').id
    geoms = [np.flatnonzero((model.geom_bodyid == b) & ((model.geom_contype | model.geom_conaffinity) != 0)) for b in feet]
    h = np.load(a.motion_dir / 'reference' / f'{stem}_human.npz')
    names = list(h['names'])
    metrics = json.loads((a.motion_dir / 'diagnostics' / stem / 'upper_body_metrics.json').read_text())
    cfg = metrics['config']
    assert all(metrics['checks'].values()), metrics['checks']
    offset = Rotation.from_quat(np.array(cfg['human_root_to_robot_wxyz'])[[1, 2, 3, 0]]).as_matrix()
    frames = np.unique(np.r_[np.linspace(0, len(new) - 1, min(128, len(new))).astype(int),
                             [v for v in (150, 300, 429, 3069, 3491) if v < len(new)]])
    gap_change, errors = 0., []
    for f in frames:
        gaps = []
        for values in (old, new):
            data.qpos[:] = model.qpos0
            data.qpos[:3] = values[f, :3]
            data.qpos[3:7] = values[f, [6, 3, 4, 5]]
            for i, name in enumerate(JOINT_NAMES):
                data.qpos[model.jnt_qposadr[model.joint(name).id]] = values[f, 7 + i]
            mj.mj_forward(model, data)
            gaps.append([min(mj.mj_geomDistance(model, data, floor, int(g), 10., None) for g in side) for side in geoms])
        gap_change = max(gap_change, float(abs(np.array(gaps[1]) - gaps[0] - root_lift[f]).max()))
        robot_root = Rotation.from_quat(new[f, 3:7]).as_matrix()
        human_root = Rotation.from_quat(h['quats_wxyz'][f, names.index('Hips')][[1, 2, 3, 0]]).as_matrix() @ offset
        row = []
        for side, label in [('l', 'Left'), ('r', 'Right')]:
            ids = [model.body(f'zarm_{side}{i}_link').id for i in (1, 4, 6)]
            robot = np.diff(data.xpos[ids], axis=0) @ robot_root
            human = np.diff(h['positions'][f, [names.index(label + s) for s in ('Arm', 'ForeArm', 'Hand')]], axis=0) @ human_root
            robot /= np.linalg.norm(robot, axis=-1, keepdims=True)
            human /= np.linalg.norm(human, axis=-1, keepdims=True)
            row.extend(np.rad2deg(np.arccos(np.clip(np.sum(robot * human, axis=-1), -1, 1))))
        errors.append(row)
    assert gap_change < 1e-12
    generated = np.loadtxt(a.motion_dir / 'diagnostics' / stem / 'upper_body_frames.csv', delimiter=',', skiprows=1)
    measured = np.array(errors).reshape(len(frames), 2, 2)
    stored = generated.reshape(len(new), 2, 7)[frames, :, 5:7]
    metric_error = float(abs(measured - stored).max())
    assert metric_error < 1e-5, metric_error
    report = dict(status='PASS', source_shape=list(new.shape), source_fps=30, control_frames=motion.num_frames,
                  unchanged_root_xy_rotation_legs_waist_heads=True, pkl_csv_equal=True, finite=True,
                  root_z_lift_max_m=float(root_lift.max()),
                  inherited_lower_body_speed_violations=above, new_arm_speed_violations=0,
                  independent_native_samples=len(frames), max_unexplained_foot_gap_change_m=gap_change,
                  independent_direction_metric_max_difference_deg=metric_error, fk=fk_test(kin, motion, 128))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
