"""Reconstruct S54 arms from BVH bone directions with bounded anatomical IK.

The baseline PKL supplies root/legs/waist, which remain numerically unchanged.
Run in the existing Leju-GMR Python. No Leju-GMR source/config is modified.
Outputs are isolated from existing training data. This is kinematic retargeting,
not a claim that the baseline lower-body reference is ground-safe.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import pickle
import re
import sys
import time

import mujoco as mj
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from supermimic.s54 import JOINT_NAMES, SOURCE_MJCF, actuator_parameters, model_contract
from prepare_s54_motion import prepare


def rotation(wxyz):
    return Rotation.from_quat(np.asarray(wxyz)[..., [1, 2, 3, 0]]).as_matrix()


def unit(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError('Degenerate human/robot limb segment')
    return v / norm


def load_bvh_frames(gmr, bvh):
    # Reuse the exact loader that produced the baseline, without running its IK.
    sys.path.insert(0, str(gmr))
    spec = importlib.util.spec_from_file_location('s54_upper_bvh_loader', gmr / 'scripts/bvh_to_pkl.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.load_motion_data_kuavo_s54('lafan1', str(bvh), 0.01)


def load_human(gmr, bvh, cached=None):
    if cached:
        with np.load(cached) as h:
            return {k: h[k].copy() for k in ('names', 'positions', 'quats_wxyz')}
    frames, _ = load_bvh_frames(gmr, bvh)
    names = list(frames[0])
    return dict(names=np.array(names), positions=np.array([[f[n][0] for n in names] for f in frames]),
                quats_wxyz=np.array([[f[n][1] for n in names] for f in frames]))


class ArmIK:
    def __init__(self, model, side, cfg, velocities):
        self.m, self.d, self.cfg = model, mj.MjData(model), cfg
        self.names = [f'zarm_{side}{i}_joint' for i in range(1, 8)]
        joints = [model.joint(n).id for n in self.names]
        self.adr = model.jnt_qposadr[joints]
        self.lo, self.hi = model.jnt_range[joints].T
        self.speed = np.array([velocities[n]['velocity'] for n in self.names]) * cfg['urdf_velocity_scale']
        # Shoulder, elbow, wrist centre, terminal hand frame. The wrist centre
        # excludes the final wrist-roll/pitch offset from the forearm direction.
        self.bodies = [model.body(f'zarm_{side}{i}_link').id for i in (1, 4, 6, 7)]
        mj.mj_kinematics(model, self.d)
        segments = np.diff(self.d.xpos[self.bodies[:3]], axis=0)
        self.lengths = np.linalg.norm(segments, axis=1)
        # S54 has nonzero X offsets in both segments. Its geometric straight
        # elbow is about -8.916 deg, not q4=0. Restrict IK to the human flexion
        # branch; otherwise a local solver can cross straight and get trapped
        # at q4=0 with shoulder twist saturated. Physical MJCF limits stay intact.
        if np.max(abs(segments[:, 1])) > 1e-8 or not np.allclose(model.jnt_axis[joints[3]], [0, 1, 0]):
            raise ValueError('Unexpected S54 elbow geometry/axis')
        self.straight_elbow = float(np.arctan2(segments[1, 0], -segments[1, 2]) - np.arctan2(segments[0, 0], -segments[0, 2]))
        self.previous = None

    def pose(self, full_qpos, q):
        self.d.qpos[:] = full_qpos
        self.d.qpos[self.adr] = q
        mj.mj_kinematics(self.m, self.d)
        points = self.d.xpos[self.bodies[:3]].copy()
        directions = unit(np.diff(points, axis=0))
        hand_R = self.d.xmat[self.bodies[3]].reshape(3, 3).copy()
        return directions, points, hand_R

    def solve(self, full_qpos, target_dir, target_hand_R, dt):
        cfg = self.cfg
        previous = self.previous
        # Position bounds always apply. Offline mode subsequently projects the
        # complete arm trajectory onto velocity limits, anticipating fast moves;
        # causal mode enforces them here and can lag abrupt human movements.
        lo, hi = self.lo.copy(), self.hi.copy()
        if cfg['anatomical_elbow_branch']:
            hi[3] = min(hi[3], self.straight_elbow)
        if previous is not None and cfg['temporal_mode'] == 'causal':
            lo = np.maximum(lo, previous - dt * self.speed)
            hi = np.minimum(hi, previous + dt * self.speed)
        start = previous.copy() if previous is not None else np.array([0, 0, 0, -.4, 0, 0, 0.])
        # The two bone directions determine flexion independently of shoulder
        # orientation; use this as a seed, within the actual per-frame bounds.
        start[3] = np.clip(self.straight_elbow - np.arccos(np.clip(target_dir[0] @ target_dir[1], -1, 1)), lo[3], hi[3])

        def residual(q):
            dirs, _, hand_R = self.pose(full_qpos, q)
            orientation = Rotation.from_matrix(target_hand_R.T @ hand_R).as_rotvec()
            result = [(dirs[0] - target_dir[0]) * cfg['upper_direction_weight'],
                      (dirs[1] - target_dir[1]) * cfg['forearm_direction_weight'],
                      orientation * cfg['hand_orientation_weight'],
                      q * cfg['neutral_pose_weight']]
            if previous is not None:
                result.append((q - previous) * cfg['previous_pose_weight'])
            return np.concatenate(result)

        result = least_squares(residual, np.clip(start, lo + 1e-10, hi - 1e-10), bounds=(lo, hi),
                               max_nfev=cfg['max_nfev'], ftol=1e-7, xtol=1e-7, gtol=1e-7)
        # First-frame multistart avoids locking a whole sequence onto a poor
        # shoulder branch. Later frames respect the same velocity bounds.
        dirs, _, _ = self.pose(full_qpos, result.x)
        large_error = np.max(np.rad2deg(np.arccos(np.clip(np.sum(dirs * target_dir, axis=1), -1, 1)))) > 3
        if previous is None or (cfg['temporal_mode'] != 'causal' and large_error):
            for seed in (full_qpos[self.adr], np.array([-.5, 0, 0, -.8, 0, 0, 0.])):
                candidate = least_squares(residual, np.clip(seed, lo + 1e-10, hi - 1e-10), bounds=(lo, hi),
                                          max_nfev=cfg['max_nfev'], ftol=1e-7, xtol=1e-7, gtol=1e-7)
                if candidate.cost < result.cost:
                    result = candidate
        if not np.isfinite(result.x).all():
            raise ValueError('Nonfinite arm IK result')
        self.previous = result.x.copy()
        return result.x, result.nfev, result.status


def summarize(x):
    return dict(median=float(np.median(x)), p95=float(np.percentile(x, 95)), max=float(np.max(x)))


def project_velocity(q, arms, fps):
    """Closest joint trajectory with symmetric, hard velocity/position bounds.

    Sparse convex QPs use the already installed GMR proxqp backend. This is an
    explicit offline temporal adjustment; no frames are dropped or retimed.
    """
    from qpsolvers import solve_qp
    from scipy import sparse
    n = len(q)
    difference = sparse.diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n), format='csc')
    constraints = sparse.vstack([difference, -difference], format='csc')
    identity = sparse.eye(n, format='csc')
    result = q.copy()
    for side, arm in enumerate(arms):
        for j in range(7):
            idx = 13 + side * 7 + j
            hi = min(arm.hi[j], arm.straight_elbow) if j == 3 and arm.cfg['anatomical_elbow_branch'] else arm.hi[j]
            # Leave a tiny numerical margin so a converged QP does not appear
            # above an exact URDF limit merely due to floating-point residuals.
            step_limit = arm.speed[j] * (1 - arm.cfg['velocity_numerical_margin']) / fps
            projected = solve_qp(identity, -q[:, idx], constraints, np.full(2 * (n - 1), step_limit),
                                 lb=np.full(n, arm.lo[j]), ub=np.full(n, hi), solver='proxqp',
                                 eps_abs=1e-10, eps_rel=0., max_iter=1000)
            if projected is None or not np.isfinite(projected).all():
                raise RuntimeError(f'Offline velocity QP failed: {arm.names[j]}')
            result[:, idx] = projected
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gmr', type=Path, default=Path('/mnt/sage_data/workspace/Leju-GMR'))
    p.add_argument('--bvh', type=Path)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--human_npz', type=Path, help='Optional cache from diagnose_s54_human.py')
    p.add_argument('--config', type=Path, default=REPO / 'configs/s54_upper_body_retarget.json')
    p.add_argument('--out_dir', type=Path, default=REPO / 'library/motion_reference/kuavo_s54/retarget_data/reference')
    p.add_argument('--max_frames', type=int, help='Prefix for a quick probe; does not change source FPS')
    a = p.parse_args()
    a.bvh = a.bvh or REPO / 'library/motion_reference/kuavo_s54/retarget_data/raw/dance1_subject1.bvh'
    a.baseline = a.baseline or REPO / 'library/motion_reference/kuavo_s54/retarget_data/reference/baseline/dance1_subject1_30fps.pkl'
    cfg = json.loads(a.config.read_text())
    if cfg['temporal_mode'] not in ('causal', 'offline_projection'):
        raise ValueError('temporal_mode must be causal or offline_projection')
    # Only trusted local PKLs created by the user's GMR pipeline are supported.
    with a.baseline.open('rb') as f:
        baseline = pickle.load(f)
    fps = float(baseline['fps'])
    header = a.bvh.read_text().split('MOTION', 1)[1][:150]
    source_count = int(re.search(r'Frames:\s*(\d+)', header)[1])
    bvh_fps = 1 / float(re.search(r'Frame Time:\s*([\d.]+)', header)[1])
    if abs(fps - 30) > 1e-8 or abs(fps - bvh_fps) > .001:
        raise ValueError(f'Expected the original 30 FPS BVH/PKL, got {fps}/{bvh_fps}')
    human = load_human(a.gmr, a.bvh, a.human_npz)
    count = min(a.max_frames or source_count, source_count)
    baseline_count = len(baseline['dof_pos'])
    if len(human['positions']) != source_count or baseline_count not in (source_count, count):
        raise ValueError('BVH/human cache/PKL frame counts differ')
    if baseline_count != source_count and a.max_frames is None:
        raise ValueError('A partial baseline requires explicit --max_frames')
    if np.asarray(baseline['dof_pos']).shape[1] != 27 or baseline['robot'] != 'kuavo_s54':
        raise ValueError('Expected original S54 27-DoF PKL')
    if baseline.get('dof_layout') != dict(leg_indices=list(range(12)), waist_indices=[12], arm_indices=list(range(13, 27))):
        raise ValueError('PKL joint layout differs from the named S54 contract')
    if count < 2:
        raise ValueError('At least two frames are required')
    stem = a.bvh.stem
    out = a.out_dir.resolve()
    diagnostics = out / 'diagnostics' / stem
    if (out / f'{stem}.csv').exists() or (diagnostics / 'upper_body_metrics.json').exists():
        raise FileExistsError(f'Choose a new --out_dir; refusing to replace {out}')
    m, rows = model_contract()
    velocities = actuator_parameters()
    q = np.array(baseline['dof_pos'][:count], copy=True)
    old = q.copy()
    root_p = np.asarray(baseline['root_pos'][:count])
    root_quat = np.asarray(baseline['root_rot'][:count])
    robot_root_R = Rotation.from_quat(root_quat).as_matrix()
    names = list(human['names'])
    human_root_R = rotation(human['quats_wxyz'][:count, names.index('Hips')]) @ rotation(cfg['human_root_to_robot_wxyz'])
    # Remove the baseline's small root orientation residual: human limb
    # directions are compared in the same root frame as the fixed robot base.
    align = robot_root_R @ human_root_R.transpose(0, 2, 1)
    arms = [ArmIK(m, s, cfg, velocities) for s in 'lr']
    target_dirs, target_rotations = [], []
    for label in ('Left', 'Right'):
        idx = [names.index(label + x) for x in ('Arm', 'ForeArm', 'Hand')]
        dirs = unit(np.diff(human['positions'][:count, idx], axis=1))
        target_dirs.append(np.einsum('tij,tsj->tsi', align, dirs))
        hand_R = rotation(human['quats_wxyz'][:count, idx[2]]) @ rotation(cfg['hand_rotation_offset_wxyz'])
        target_rotations.append(align @ hand_R)
    measurements = {v: dict(upper_deg=[], forearm_deg=[], wrist_error_m=[], hand_rotation_deg=[],
                             horizontal=[], elevation_deg=[]) for v in ('original', 'corrected')}
    full_qpos = np.tile(m.qpos0, (count, 1))
    full_qpos[:, :3] = root_p
    full_qpos[:, 3:7] = root_quat[:, [3, 0, 1, 2]]
    # Names, never an assumed simulator array order.
    for i, row in enumerate(rows[:27]):
        full_qpos[:, row['qposadr']] = old[:, i]
    frames_data, nfevs, statuses = [], [], []
    started = time.monotonic()
    for f in range(count):
        for side, arm in enumerate(arms):
            target = target_dirs[side][f]
            solution, nfev, status = arm.solve(full_qpos[f], target, target_rotations[side][f], 1 / fps)
            for j, name in enumerate(arm.names):
                q[f, JOINT_NAMES.index(name)] = solution[j]
            nfevs.append(nfev); statuses.append(status)
        if f == 0 or (f + 1) % 300 == 0 or f + 1 == count:
            print(f'Arm IK {f + 1}/{count}; elapsed={time.monotonic() - started:.1f}s', flush=True)
    unconstrained_q = q.copy()
    if cfg['temporal_mode'] == 'offline_projection':
        print('Projecting complete arm trajectories onto URDF velocity bounds...', flush=True)
        q = project_velocity(q, arms, fps)
    refinement = None
    if cfg.get('global_refinement_iterations', 0):
        from s54_arm_trajectory import refine_trajectory
        q, refinement = refine_trajectory(q, arms, full_qpos, target_dirs, target_rotations, fps,
                                          cfg['global_refinement_iterations'])
        q = project_velocity(q, arms, fps)
    for f in range(count):
        for side, arm in enumerate(arms):
            target = target_dirs[side][f]
            for variant, values in [('original', old[f, 13 + side * 7:20 + side * 7]), ('corrected', q[f, 13 + side * 7:20 + side * 7])]:
                dirs, points, hand_R = arm.pose(full_qpos[f], values)
                error = np.rad2deg(np.arccos(np.clip(np.sum(dirs * target, axis=1), -1, 1)))
                target_wrist = points[0] + arm.lengths @ target
                local_upper = robot_root_R[f].T @ dirs[0]
                elevation = np.rad2deg(np.arctan2(local_upper[2], np.linalg.norm(local_upper[:2])))
                measurement = measurements[variant]
                measurement['upper_deg'].append(error[0]); measurement['forearm_deg'].append(error[1])
                measurement['wrist_error_m'].append(np.linalg.norm(points[2] - target_wrist))
                measurement['hand_rotation_deg'].append(np.rad2deg(Rotation.from_matrix(target_rotations[side][f].T @ hand_R).magnitude()))
                measurement['horizontal'].append(abs(elevation) < 15 and abs(local_upper[1]) > .866)
                measurement['elevation_deg'].append(elevation)
            human_local = human_root_R[f].T @ unit(human['positions'][f, names.index(('Left', 'Right')[side] + 'ForeArm')] - human['positions'][f, names.index(('Left', 'Right')[side] + 'Arm')])
            human_el = np.rad2deg(np.arctan2(human_local[2], np.linalg.norm(human_local[:2])))
            frames_data.append([f / fps, side, human_el, measurements['original']['elevation_deg'][-1],
                                measurements['corrected']['elevation_deg'][-1], *error])
    metrics = dict(frames=count, fps=fps, source_bvh=str(a.bvh), baseline_pkl=str(a.baseline), config=cfg,
                   root_position_max_change_m=0., root_rotation_max_change_rad=0.,
                   lower_body_max_change_rad=float(np.max(abs(q[:, :13] - old[:, :13]))),
                   solver_nfev=summarize(nfevs), solver_max_nfev_frames=int(np.sum(np.array(statuses) == 0)))
    metrics['geometric_straight_elbow_deg'] = {s: float(np.rad2deg(arm.straight_elbow)) for s, arm in zip(('left', 'right'), arms)}
    metrics['temporal_projection_joint_adjustment_rad'] = summarize(abs(q[:, 13:] - unconstrained_q[:, 13:]))
    metrics['global_refinement'] = refinement
    metrics['human'] = {}
    for window, end in [('first10s', min(count, 301)), ('full', count)]:
        metrics['human'][window] = {}
        for side, label in enumerate(('Left', 'Right')):
            v = unit(human['positions'][:end, names.index(label + 'ForeArm')] - human['positions'][:end, names.index(label + 'Arm')])
            v = np.einsum('tji,tj->ti', human_root_R[:end], v)
            el = np.rad2deg(np.arctan2(v[:, 2], np.linalg.norm(v[:, :2], axis=-1)))
            metrics['human'][window][('left', 'right')[side]] = dict(horizontal_fraction=float(np.mean((abs(el) < 15) & (abs(v[:, 1]) > .866))))
    assert np.array_equal(q[:, :13], old[:, :13])
    for variant, values in [('original', old), ('corrected', q)]:
        metrics[variant] = {}
        mm = {k: np.asarray(v).reshape(count, 2) for k, v in measurements[variant].items()}
        for window, end in [('first10s', min(count, 301)), ('full', count)]:
            metrics[variant][window] = {}
            for side, label in enumerate(('left', 'right')):
                arm = arms[side]
                qq = values[:end, 13 + side * 7:20 + side * 7]
                speed = abs(np.diff(qq, axis=0)) * fps
                near = np.minimum(qq - arm.lo, arm.hi - qq) < np.deg2rad(1)
                metrics[variant][window][label] = {
                    **{key: summarize(mm[key][:end, side]) for key in ('upper_deg', 'forearm_deg', 'wrist_error_m', 'hand_rotation_deg')},
                    'horizontal_fraction': float(np.mean(mm['horizontal'][:end, side])),
                    'joint3_near_limit_fraction': float(np.mean(near[:, 2])),
                    'joint_near_limit_fraction': np.mean(near, axis=0).tolist(),
                    'max_joint_speed_rad_s': np.max(speed, axis=0).tolist(),
                    'speed_above_urdf_intervals': int(np.sum(speed > arm.speed + 1e-6)),
                    'largest_joint_step_rad': float(np.max(abs(np.diff(qq, axis=0))))}
    quality = cfg['quality']
    checks = dict(finite=bool(np.isfinite(q).all()), unchanged_lower_body=bool(np.array_equal(q[:, :13], old[:, :13])),
                  joint_ranges=all(np.all((q[:, i] >= r['range'][0] - 1e-8) & (q[:, i] <= r['range'][1] + 1e-8)) for i, r in enumerate(rows[:27])),
                  arm_velocity_limits=all(metrics['corrected']['full'][s]['speed_above_urdf_intervals'] == 0 for s in ('left', 'right')),
                  upper_direction_quality=all(metrics['corrected'][w][s]['upper_deg']['p95'] < quality['upper_p95_deg'] for w in ('first10s', 'full') for s in ('left', 'right')),
                  forearm_direction_quality=all(metrics['corrected']['full'][s]['forearm_deg']['p95'] < quality['forearm_p95_deg'] for s in ('left', 'right')),
                  wrist_position_quality=all(metrics['corrected']['full'][s]['wrist_error_m']['p95'] < quality['wrist_p95_m'] for s in ('left', 'right')),
                  no_artificial_persistent_spread=all(metrics['corrected'][w][s]['horizontal_fraction'] - metrics['human'][w][s]['horizontal_fraction'] < quality['artificial_horizontal_excess_fraction'] for w in ('first10s', 'full') for s in ('left', 'right')),
                  joint3_not_persistently_saturated=all(metrics['corrected']['full'][s]['joint3_near_limit_fraction'] < quality['joint3_near_limit_fraction'] for s in ('left', 'right')))
    metrics['checks'] = checks
    out.mkdir(parents=True, exist_ok=True)
    diagnostics.mkdir(parents=True, exist_ok=True)
    (diagnostics / 'upper_body_metrics.json').write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(diagnostics / 'arm_ik_before_velocity_projection.npz', q=unconstrained_q)
    np.savetxt(diagnostics / 'upper_body_frames.csv', frames_data, delimiter=',',
               header='time_s,side,human_elevation_deg,original_elevation_deg,corrected_elevation_deg,corrected_upper_error_deg,corrected_forearm_error_deg', comments='')
    raw = out / 'raw' / f'{stem}.csv'
    raw.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(raw, np.c_[root_p, root_quat, q], delimiter=',', fmt='%.17g')
    prepare(raw, out / f'{stem}.csv', expected_frames=count, raw_dir=out / 'raw')
    result = dict(baseline, root_pos=root_p.copy(), root_rot=root_quat.copy(), dof_pos=q, local_body_pos=None)
    (out / 'reference').mkdir(exist_ok=True)
    with (out / 'reference' / f'{stem}_30fps.pkl').open('wb') as f:
        pickle.dump(result, f)
    np.savez_compressed(out / 'reference' / f'{stem}_human.npz', **{k: v[:count] if k != 'names' else v for k, v in human.items()})
    print(json.dumps({'checks': checks, 'first10s': metrics['corrected']['first10s'], 'out_dir': str(out)}, indent=2))
    if not all(checks.values()):
        raise SystemExit('Reference candidate saved, but some acceptance checks FAIL; inspect upper_body_metrics.json')


if __name__ == '__main__':
    main()
