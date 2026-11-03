"""Ground-aware S54 IK and a geometry-based, temporally smooth export guard.

Foot origins are ankle frames, NOT soles. Both visual meshes and collision
primitives determine support height. No model, floor or human data is moved.
"""
import json
from pathlib import Path
import pickle

import mujoco as mj
import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp

from supermimic.s54 import JOINT_NAMES, model_contract


def stats(values):
    values = np.asarray(values)
    return dict(min=float(values.min()), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), max=float(values.max()))


from supermimic.foot_geometry import FootGeometry


class GroundedTargets:
    """Adjust GMR tasks in memory; leave the installed Leju-GMR untouched."""
    def __init__(self, solver, frames, cfg):
        import mink
        self.solver, self.cfg, self.index = solver, cfg, 0
        self.geometry = FootGeometry(solver.model)
        # Ankle pitch and roll bodies have coincident origins. The old Foot
        # and Toe position tasks demanded two distinct human points there.
        removed = []
        for label in ('Left', 'Right'):
            task = solver.human_body_to_task1.pop(label + 'Toe', None)
            if task is not None:
                solver.tasks1.remove(task)
                removed.append(label + 'Toe')
        self.removed = removed
        self.contacts, self.flat = [], []
        for label in ('Left', 'Right'):
            foot = np.array([f[label + 'Foot'][0] for f in frames])
            toe = np.array([f[label + 'Toe'][0] for f in frames])
            heights = [v[:, 2] - np.percentile(v[:, 2], 10) for v in (foot, toe)]
            speeds = [np.linalg.norm(np.gradient(v, axis=0) * 30, axis=1) for v in (foot, toe)]
            contacts = [np.clip(1 - np.maximum(h, 0) / cfg['contact_height_m'], 0, 1)
                        * np.clip(2 - s / cfg['contact_speed_m_s'], 0, 1)
                        for h, s in zip(heights, speeds)]
            smooth = lambda x: gaussian_filter1d(x, cfg['contact_smoothing_frames'], mode='nearest')
            # Heel-only / toe-only contact should still allow foot rolling.
            self.contacts.append(smooth(np.maximum(*contacts)))
            self.flat.append(smooth(np.minimum(*contacts)))
        self.targets = []
        original = solver.update_targets

        def update(human_data, offset_to_ground=False):
            original(human_data, offset_to_ground=False)
            row = []
            for side, label in enumerate(('Left', 'Right')):
                name = label + 'Foot'
                pos, quat = solver.scaled_human_data[name]
                pos = pos.copy()
                R = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
                yaw = np.arctan2(R[1, 0], R[0, 0])
                flat_R = Rotation.from_euler('z', yaw).as_matrix()
                # Do not flatten an inverted foot during acrobatics.
                upright = np.clip((R[2, 2] - .3) / .4, 0, 1)
                weight = self.flat[side][self.index] * upright
                R = R @ Rotation.from_rotvec(Rotation.from_matrix(R.T @ flat_R).as_rotvec() * weight).as_matrix()
                sole = self.geometry.local_low(side, R)
                safe_z = cfg['ik_clearance_m'] - sole
                contact = self.contacts[side][self.index] * upright
                pos[2] = max(safe_z, (1 - contact) * pos[2] + contact * safe_z)
                q = Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]]
                solver.human_body_to_task1[name].set_target(
                    mink.SE3.from_rotation_and_translation(mink.SO3(q), pos))
                row.extend([contact, weight, pos[2], sole])
            self.targets.append(row)

        solver.update_targets = update
        floor = solver.model.geom('floor').id
        solver.ik_limits.append(mink.CollisionAvoidanceLimit(
            solver.model, [(ids, [floor]) for ids in self.geometry.collision_ids],
            gain=.85, minimum_distance_from_collisions=cfg['ik_clearance_m'],
            collision_detection_distance=.06, broadphase=False))
        # Collision limits prevent worsening penetration, so start above the
        # floor rather than at the source model's underground default pose.
        q = solver.configuration.q.copy()
        row = np.r_[q[:3], q[[4, 5, 6, 3]], q[self.geometry.addresses]]
        q[2] += max(0., .02 - self.geometry.gaps(row).min())
        solver.configuration.update(q)

    def report(self):
        return dict(removed_conflicting_position_tasks=self.removed,
                    contact_fraction=[float(np.mean(v > .5)) for v in self.contacts],
                    flat_contact_fraction=[float(np.mean(v > .5)) for v in self.flat],
                    target_sole_clearance_m=self.cfg['ik_clearance_m'])


def interpolate(csv, fps):
    """Linear joints/position and shortest-arc SLERP, matching MotionLib."""
    n = len(csv)
    t = np.arange(int(np.floor((n - 1) / 30 * fps)) + 1) / fps * 30
    i = np.minimum(np.floor(t).astype(int), n - 2)
    a = np.clip(t - i, 0, 1)
    values = csv[i] * (1 - a[:, None]) + csv[i + 1] * a[:, None]
    values[:, 3:7] = Slerp(np.arange(n), Rotation.from_quat(csv[:, 3:7]))(t).as_quat()
    return values, i, a


def ground_export(out, stem, cfg):
    """Lift root Z only as needed, smoothly, with 30/50/150 Hz constraints.

    The 150 Hz grid includes both 30 Hz exports and all 50 Hz control samples.
    Pose, root XY/rotation and duration stay unchanged. This is kinematic
    nonpenetration; it does not imply dynamic balance or eliminate foot slip.
    """
    from qpsolvers import solve_qp
    from prepare_s54_motion import sha
    out = Path(out)
    csv_path = out / f'{stem}.csv'
    raw_path = out / 'raw' / csv_path.name
    pkl_path = out / 'reference' / f'{stem}_30fps.pkl'
    diagnostic = out / 'diagnostics' / stem
    report_path = diagnostic / 'grounding_metrics.json'
    if report_path.exists():
        raise FileExistsError(f'Grounding already applied: {report_path}')
    x = np.loadtxt(csv_path, delimiter=',', ndmin=2)
    m, _ = model_contract()
    geometry = FootGeometry(m)
    dense, indices, alpha = interpolate(x, 150)
    gaps = geometry.gaps(dense)
    required = np.maximum(cfg['clearance_m'] - gaps.min(axis=1), 0.)
    n, samples = len(x), len(dense)
    rows = np.repeat(np.arange(samples), 2)
    cols = np.c_[indices, indices + 1].ravel()
    A = sparse.coo_matrix((np.c_[1 - alpha, alpha].ravel(), (rows, cols)), shape=(samples, n)).tocsc()
    # Bound both endpoints by the largest required lift inside their interval.
    # Every interpolation is then safe. This conservative reduction avoids
    # thousands of almost identical, poorly conditioned active inequalities
    # in long clips; it adds at most one source frame of anticipation.
    bounds = np.zeros(n)
    np.maximum.at(bounds, indices, required)
    np.maximum.at(bounds, indices + 1, required)
    D = sparse.diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n), format='csc')
    DD = (sparse.diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2],
                       shape=(n - 2, n), format='csc') if n > 2 else sparse.csc_matrix((0, n)))
    H = sparse.eye(n, format='csc') + cfg['root_lift_smoothing'] * D.T @ D + cfg['root_lift_acceleration_smoothing'] * DD.T @ DD
    print(f'Ground guard {stem}: {samples} geometry samples -> {n} endpoint bounds', flush=True)
    lift = solve_qp(H.tocsc(), np.zeros(n), lb=bounds,
                    solver='proxqp', eps_abs=1e-7, eps_rel=0., max_iter=2000)
    if lift is None or not np.isfinite(lift).all():
        raise RuntimeError('Sole clearance trajectory QP failed')
    lift = np.maximum(lift, bounds) + 1e-5  # exact feasibility + 0.01 mm slack
    y = x.copy()
    y[:, 2] += lift
    after_dense = gaps + (A @ lift)[:, None]
    checks = dict(finite=bool(np.isfinite(y).all()),
                  nonpenetrating_150hz=bool(after_dense.min() >= cfg['clearance_m'] - cfg['validation_tolerance_m']),
                  unchanged_joints=bool(np.array_equal(x[:, 7:], y[:, 7:])),
                  unchanged_root_xy_rotation=bool(np.array_equal(x[:, [0, 1, 3, 4, 5, 6]], y[:, [0, 1, 3, 4, 5, 6]])))
    report = dict(clip=stem, frames=n, fps=30, config=cfg, validation_fps=150,
                  before_gap_m={s: stats(gaps[:, i]) for i, s in enumerate(('left', 'right'))},
                  after_gap_m={s: stats(after_dense[:, i]) for i, s in enumerate(('left', 'right'))},
                  minimum_support_gap_m=stats(after_dense.min(axis=1)), root_lift_m=stats(lift),
                  root_lift_max_speed_m_s=float(np.max(abs(np.diff(lift))) * 30), checks=checks,
                  scope='Geometric foot nonpenetration; no dynamic stability/contact-force guarantee.')
    diagnostic.mkdir(parents=True, exist_ok=True)
    if not all(checks.values()):
        report_path.write_text(json.dumps(report, indent=2))
        raise ValueError(f'Grounding checks failed: {report_path}')
    # Keep the exact pre-guard trajectory for review without polluting the
    # top-level MotionLib CSV glob. Only this newly generated clip is changed.
    np.savez_compressed(diagnostic / 'before_ground_guard.npz', csv=x, lift_m=lift)
    np.savetxt(csv_path, y, delimiter=',', fmt='%.17g')
    np.savetxt(raw_path, y[:, :34], delimiter=',', fmt='%.17g')
    with pkl_path.open('rb') as f:
        motion = pickle.load(f)
    motion['root_pos'] = y[:, :3].copy()
    motion['grounding'] = dict(clearance_m=cfg['clearance_m'], validation_fps=150)
    with pkl_path.open('wb') as f:
        pickle.dump(motion, f)
    metadata_path = csv_path.with_suffix('.json')
    metadata = json.loads(metadata_path.read_text())
    metadata.update(csv_sha256=sha(csv_path), raw_sha256=sha(raw_path),
                    root_max_speed_m_s=float(np.linalg.norm(np.diff(y[:, :3], axis=0) * 30, axis=1).max()),
                    grounding=report,
                    modifications='Ground-aware foot IK; bounded upper-body IK; joint self-collision/velocity repair when recorded; smooth geometry-constrained root Z lift; model-default heads. No retiming.')
    metadata_path.write_text(json.dumps(metadata, indent=2))
    report_path.write_text(json.dumps(report, indent=2))
    print(f'Grounding PASS {stem}: minimum sole gap={after_dense.min() * 1000:.3f} mm; '
          f'root lift median/max={np.median(lift) * 1000:.2f}/{lift.max() * 1000:.2f} mm', flush=True)
    return report
