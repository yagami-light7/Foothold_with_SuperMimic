"""S54 self collision constraints, trajectory repair and continuous validation.

Explicit distances ignore disabled contact masks. Both visual mesh convex hulls
and physical collision primitives are checked. Only mechanical neighbours in
the collision-bearing link chain are excluded, using a pose-independent rule.
"""
import hashlib
import json
from pathlib import Path
import pickle
import time

import mujoco as mj
import numpy as np
from qpsolvers import solve_qp
from scipy import sparse
from scipy.spatial.transform import Rotation

from supermimic.s54 import JOINT_NAMES, model_contract, actuator_parameters

VERSION = 1


class CollisionGeometry:
    def __init__(self, model):
        self.m, self.d = model, mj.MjData(model)
        model.opt.ccd_iterations = max(100, model.opt.ccd_iterations)
        model.opt.ccd_tolerance = min(1e-8, model.opt.ccd_tolerance)
        self.addresses = np.array([model.jnt_qposadr[model.joint(n).id] for n in JOINT_NAMES[:27]])
        self.dofs = np.array([model.jnt_dofadr[model.joint(n).id] for n in JOINT_NAMES[:27]])
        self.lo, self.hi = np.array([model.jnt_range[model.joint(n).id] for n in JOINT_NAMES[:27]]).T
        # The physical elbow limit stays unchanged; retain the previously
        # established anatomical flexion branch in the retarget solver.
        self.hi[[16, 23]] = np.minimum(self.hi[[16, 23]], np.deg2rad(-8.916))
        physical = np.flatnonzero((model.geom_contype | model.geom_conaffinity) != 0)
        bearing = set(int(b) for b in model.geom_bodyid[physical]) - {0}
        parent = {}
        for b in bearing:
            a = int(model.body_parentid[b])
            while a and a not in bearing:
                a = int(model.body_parentid[a])
            parent[b] = a
        owner = {}
        for body in range(1, model.nbody):
            a = body
            while a and a not in bearing:
                a = int(model.body_parentid[a])
            owner[body] = a
        geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] and model.geom_rbound[g] > 0]
        pairs, excluded = [], set()
        for i, g in enumerate(geoms):
            for h in geoms[i + 1:]:
                a, b = owner[int(model.geom_bodyid[g])], owner[int(model.geom_bodyid[h])]
                if a == b or parent.get(a) == b or parent.get(b) == a:
                    excluded.add(tuple(sorted((a, b))))
                else:
                    pairs.append((g, h))
        self.pairs = np.asarray(pairs, dtype=int)
        self.radii = model.geom_rbound[self.pairs].sum(axis=1)
        self.excluded = [[model.body(a).name, model.body(b).name] for a, b in sorted(excluded)]
        self.body_pairs = np.sort(model.geom_bodyid[self.pairs], axis=1)
        self.constraint_groups = np.array([min(owner[int(model.geom_bodyid[a])], owner[int(model.geom_bodyid[b])]) * model.nbody
                                           + max(owner[int(model.geom_bodyid[a])], owner[int(model.geom_bodyid[b])])
                                           for a, b in pairs])
        self._fromto, self._j1, self._j2 = np.empty(6), np.empty((3, model.nv)), np.empty((3, model.nv))
        self.levers = self._lever_bounds()
        digest = hashlib.sha256()
        for field in ('body_parentid', 'body_pos', 'body_quat', 'jnt_type', 'jnt_axis', 'jnt_pos',
                      'geom_bodyid', 'geom_type', 'geom_pos', 'geom_quat', 'geom_size', 'mesh_vert', 'mesh_face'):
            digest.update(np.asarray(getattr(model, field)).tobytes())
        digest.update(self.pairs.tobytes())
        self.geometry_sha256 = digest.hexdigest()
        self.arm_bodies = [[model.body(f'zarm_{s}{j}_link').id for j in (1, 4, 6)] for s in 'lr']

    def _lever_bounds(self):
        """Global joint-angle Lipschitz bounds; common rigid motion cancels.

        The lever for a descendant geom is bounded by the sum of link lengths,
        its local centre offset and its bounding radius. Shared ancestor joint
        rotations move both geoms rigidly and do not change their distance.
        """
        m = self.m
        lengths = np.zeros((m.ngeom, 27))
        affected = np.zeros((m.ngeom, 27), dtype=bool)
        for g in np.unique(self.pairs):
            for i, name in enumerate(JOINT_NAMES[:27]):
                j = m.joint(name).id
                ancestor = int(m.jnt_bodyid[j])
                b, length = int(m.geom_bodyid[g]), float(np.linalg.norm(m.geom_pos[g]) + m.geom_rbound[g])
                while b and b != ancestor:
                    length += np.linalg.norm(m.body_pos[b])
                    b = int(m.body_parentid[b])
                if b == ancestor:
                    affected[g, i] = True
                    lengths[g, i] = length + np.linalg.norm(m.jnt_pos[j])
        a, b = self.pairs.T
        return np.where(affected[a] ^ affected[b], lengths[a] + lengths[b], 0.)

    def pose(self, q):
        # Self distances are invariant under the floating base's rigid motion.
        self.d.qpos[:] = self.m.qpos0
        self.d.qpos[:7] = [0, 0, 1, 1, 0, 0, 0]
        self.d.qpos[self.addresses] = q
        mj.mj_kinematics(self.m, self.d)
        mj.mj_comPos(self.m, self.d)

    def distances(self, q, detect=.04, jacobian=False, exhaustive=False):
        self.pose(q)
        a, b = self.pairs.T
        bound = np.linalg.norm(self.d.geom_xpos[a] - self.d.geom_xpos[b], axis=1) - self.radii
        ids = np.arange(len(a)) if exhaustive else np.flatnonzero(bound < detect)
        distances = np.maximum(bound, detect)
        active, gradients, witnesses = [], [], {}
        for index in ids:
            g, h = self.pairs[index]
            dist = mj.mj_geomDistance(self.m, self.d, int(g), int(h), detect, self._fromto)
            distances[index] = dist
            if jacobian and dist < detect:
                group = self.constraint_groups[index]
                old = witnesses.get(group)
                if old is None or dist < old[1]:
                    witnesses[group] = (index, dist, self._fromto.copy())
        # Meshes and their collision proxies often describe the same contact.
        # Linearize the minimum per mechanical link pair, then re-query every
        # geom after each step. Final validation still checks ALL geom pairs.
        for index, dist, fromto in witnesses.values():
            g, h = self.pairs[index]
            normal = fromto[3:] - fromto[:3]
            length = np.linalg.norm(normal)
            if length < 1e-12:
                normal = self.d.geom_xpos[h] - self.d.geom_xpos[g]
                length = max(np.linalg.norm(normal), 1e-12)
            normal = normal / length * (1 if dist >= 0 else -1)
            mj.mj_jac(self.m, self.d, self._j1, None, fromto[:3], self.m.geom_bodyid[g])
            mj.mj_jac(self.m, self.d, self._j2, None, fromto[3:], self.m.geom_bodyid[h])
            gradients.append((normal @ (self._j2 - self._j1))[self.dofs])
            active.append(index)
        return distances, np.asarray(active, dtype=int), np.asarray(gradients).reshape(-1, 27)

    def penalty(self, distances, clearance):
        groups = np.unique(self.constraint_groups)
        minima = np.full(int(self.constraint_groups.max()) + 1, np.inf)
        np.minimum.at(minima, self.constraint_groups, distances)
        return float(np.sum(np.maximum(clearance - minima[groups], 0)**2))

    def subset_distances(self, q, indices, detect):
        self.pose(q)
        a, b = self.pairs[indices].T
        bound = np.linalg.norm(self.d.geom_xpos[a] - self.d.geom_xpos[b], axis=1) - self.radii[indices]
        values = np.maximum(bound, detect)
        for local in np.flatnonzero(bound < detect):
            values[local] = mj.mj_geomDistance(self.m, self.d, int(a[local]), int(b[local]), detect, None)
        return values

    def directions(self, jacobian=False):
        values, gradients = [], []
        for bodies in self.arm_bodies:
            segments = np.diff(self.d.xpos[bodies], axis=0)
            lengths = np.linalg.norm(segments, axis=1)
            values.append(segments / lengths[:, None])
            if jacobian:
                J = []
                for b in bodies:
                    mj.mj_jacBody(self.m, self.d, self._j1, None, b)
                    J.append(self._j1[:, self.dofs].copy())
                gradients.append(np.diff(J, axis=0) / lengths[:, None, None])
        return np.asarray(values), np.asarray(gradients).reshape(-1, 27) if jacobian else None

    def certificate(self, q, minimum=.001, max_depth=10):
        """Certify every linear-joint interval, not just its sampled endpoints."""
        worst, samples, bad = float('inf'), 0, []

        all_pairs = np.arange(len(self.pairs))

        def evaluate(t, values, indices=all_pairs):
            nonlocal worst, samples
            distances = self.subset_distances(values, indices, minimum + .05)
            samples += 1
            worst = min(worst, float(distances.min()))
            return distances

        def interval(i, left, right, a, b, depth, indices=all_pairs):
            mid = (a + b) / 2
            values = left * .5 + right * .5
            distances = evaluate(i + mid, values, indices)
            gap = float(distances.min())
            if gap < minimum:
                pair = self.pairs[indices[int(np.argmin(distances))]]
                bad.append(dict(frame=float(i + mid), distance_m=gap,
                                bodies=[self.m.body(int(self.m.geom_bodyid[g])).name for g in pair], reason='clearance'))
                return False
            movement = self.levers[indices] @ abs(right - left) * .5
            uncertain = indices[distances - movement < minimum]
            if not len(uncertain):
                return True
            if depth >= max_depth:
                bad.append(dict(frame=float(i + mid), distance_m=gap, reason='uncertified_interval'))
                return False
            # Pairs already proven safe over the parent interval need no more
            # narrow-phase queries in either child. The proof still covers all
            # pairs, while long clips avoid rechecking faraway limbs at depth.
            return (interval(i, left, values, a, mid, depth + 1, uncertain)
                    and interval(i, values, right, mid, b, depth + 1, uncertain))

        for i, values in enumerate(q):
            distances = evaluate(float(i), values)
            if distances.min() < minimum:
                pair = self.pairs[int(np.argmin(distances))]
                bad.append(dict(frame=float(i), distance_m=float(distances.min()),
                                bodies=[self.m.body(int(self.m.geom_bodyid[g])).name for g in pair], reason='clearance'))
            if i and not bad:
                interval(i - 1, q[i - 1], values, 0., 1., 0)
            if bad:
                break
        return dict(status='PASS' if not bad else 'FAIL', continuous_joint_interpolation=not bad,
                    minimum_clearance_m=minimum, minimum_evaluated_distance_m=worst,
                    evaluated_poses=samples, first_failure=bad[0] if bad else None,
                    geometry_sha256=self.geometry_sha256, geom_pair_count=len(self.pairs),
                    excluded_mechanical_neighbours=self.excluded)


def objective(geometry, q, original, target_dirs, cfg, jacobian=True):
    weights = np.r_[np.full(12, cfg['leg_pose_weight']), cfg['waist_pose_weight'], np.full(14, cfg['arm_pose_weight'])]
    dirs, J = geometry.directions(jacobian)
    r = np.r_[(q - original) * weights, ((dirs - target_dirs) * cfg['direction_weight']).ravel()]
    return r, np.vstack([np.diag(weights), J * cfg['direction_weight']]) if jacobian else None


def repair_frame(geometry, q, target, original, cfg):
    for _ in range(cfg['frame_iterations']):
        distances, ids, Jc = geometry.distances(q, cfg['activation_distance_m'], True)
        if distances.min() >= cfg['target_clearance_m'] - 1e-5:
            return q
        r, J = objective(geometry, q, original, target, cfg)
        lo = np.maximum(geometry.lo - q, -cfg['trust_region_rad'])
        hi = np.minimum(geometry.hi - q, cfg['trust_region_rad'])
        # Elastic constraints keep the subproblem solvable through a deep or
        # opposing contact. Final acceptance never allows this elastic slack.
        n = len(ids)
        H = sparse.block_diag([J.T @ J + .01 * np.eye(27), cfg['slack_weight'] * sparse.eye(n)], format='csc')
        G = sparse.hstack([-sparse.csc_matrix(Jc), -sparse.eye(n)], format='csc')
        result = solve_qp(H.toarray(), np.r_[J.T @ r, np.zeros(n)], G.toarray(),
                          distances[ids] - cfg['target_clearance_m'],
                          lb=np.r_[lo, np.zeros(n)], ub=np.r_[hi, np.full(n, np.inf)], solver='daqp')
        if result is None:
            return q
        old_merit = .5 * (r @ r) + cfg['slack_weight'] * geometry.penalty(distances, cfg['target_clearance_m'])
        accepted = False
        for scale in (1., .5, .25, .125):
            candidate = q + scale * result[:27]
            d, _, _ = geometry.distances(candidate, cfg['activation_distance_m'])
            r1, _ = objective(geometry, candidate, original, target, cfg, False)
            merit = .5 * (r1 @ r1) + cfg['slack_weight'] * geometry.penalty(d, cfg['target_clearance_m'])
            if merit < old_merit:
                q, accepted = candidate, True
                break
        if not accepted:
            break
    return q


def project_velocities(q, geometry, speeds):
    n = len(q)
    D = sparse.diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n), format='csc')
    G = sparse.vstack([D, -D], format='csc')
    for j in range(27):
        step = speeds[j] / 30 * (1 - 1e-6)
        x = solve_qp(sparse.eye(n, format='csc'), -q[:, j], G, np.full(2 * (n - 1), step),
                     lb=np.full(n, geometry.lo[j]), ub=np.full(n, geometry.hi[j]),
                     solver='proxqp', eps_abs=1e-9, eps_rel=0., max_iter=1000)
        if x is None:
            raise RuntimeError(f'Self collision velocity projection failed: {JOINT_NAMES[j]}')
        q[:, j] = x
    return q


def sample_times(n, extra=()):
    times = np.arange((n - 1) * 5 + 1, dtype=float) / 5
    if len(extra):
        times = np.unique(np.r_[times, [v for v in extra if 0 <= v <= n - 1]])
    return times


def dense_contacts(geometry, q, cfg, jacobian=False, extra_samples=()):
    """150 Hz collocation constraints; interpolate joint values, not distances."""
    rows, cols, values, bounds = [], [], [], []
    bad = set()
    minimum = float('inf')
    for t in sample_times(len(q), extra_samples):
        i = min(int(t), len(q) - 2)
        a = min(1., t - i)
        x = q[i] * (1 - a) + q[i + 1] * a
        d, ids, J = geometry.distances(x, cfg['activation_distance_m'], jacobian)
        minimum = min(minimum, float(d.min()))
        if d.min() < cfg['minimum_clearance_m'] + .002:
            bad.update((i, i + 1))
        if jacobian:
            for k, idx in enumerate(ids):
                row = len(bounds)
                bounds.append(float(d[idx] - cfg['target_clearance_m']))
                for frame, weight in ((i, 1 - a), (i + 1, a)):
                    if weight:
                        nz = np.flatnonzero(abs(J[k]) > 1e-12)
                        rows.extend([row] * len(nz)); cols.extend((frame * 27 + nz).tolist())
                        values.extend((-weight * J[k, nz]).tolist())
    C = sparse.coo_matrix((values, (rows, cols)), shape=(len(bounds), len(q) * 27)).tocsc()
    return sorted(bad), minimum, C, np.asarray(bounds)


def refine_window(geometry, q, original, targets, speeds, cfg, extra_samples=()):
    n, width = len(q), len(q) * 27
    D = sparse.kron(sparse.diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n)), sparse.eye(27), format='csc')
    V = sparse.vstack([D, -D], format='csc')
    limits = np.tile(speeds / 30 * (1 - 1e-6), n - 1)
    regularizer = cfg['temporal_weight'] * D.T @ D
    for iteration in range(cfg['trajectory_iterations']):
        bad, minimum, C, b = dense_contacts(geometry, q, cfg, True, extra_samples)
        if minimum >= cfg['minimum_clearance_m'] + .003:
            break
        residuals, jacobians = [], []
        for f in range(n):
            geometry.pose(q[f])
            r, J = objective(geometry, q[f], original[f], targets[f], cfg)
            residuals.append(r); jacobians.append(J)
        J = sparse.block_diag(jacobians, format='csc')
        r = np.concatenate(residuals)
        flat = q.ravel()
        H = J.T @ J + regularizer + .02 * sparse.eye(width, format='csc')
        gradient = J.T @ r + regularizer @ flat
        lo = np.maximum((geometry.lo - q).ravel(), -cfg['trust_region_rad'])
        hi = np.minimum((geometry.hi - q).ravel(), cfg['trust_region_rad'])
        # Safe surrounding poses anchor the window, retaining exact velocity
        # continuity across overlapping windows.
        lo[:27] = hi[:27] = 0
        lo[-27:] = hi[-27:] = 0
        count = len(b)
        matrix = sparse.bmat([[V, sparse.csc_matrix((V.shape[0], count))], [C, -sparse.eye(count)]], format='csc')
        h = np.r_[limits - D @ flat, limits + D @ flat, b]
        delta = solve_qp(sparse.block_diag([H, cfg['slack_weight'] * sparse.eye(count)], format='csc'),
                         np.r_[gradient, np.zeros(count)], matrix, h,
                         lb=np.r_[lo, np.zeros(count)], ub=np.r_[hi, np.full(count, np.inf)],
                         solver='proxqp', eps_abs=1e-7, eps_rel=0., max_iter=1000)
        if delta is None:
            raise RuntimeError('Collision-constrained trajectory QP failed')
        old_violation = np.maximum(-b, 0)
        old_merit = .5 * (r @ r + flat @ (regularizer @ flat)) + cfg['slack_weight'] * (old_violation @ old_violation)
        accepted = False
        for scale in (1., .5, .25, .125):
            candidate = q + scale * delta[:width].reshape(n, 27)
            violation = collision_penalty(geometry, candidate, cfg, extra_samples)
            residual = []
            for f in range(n):
                geometry.pose(candidate[f]); rr, _ = objective(geometry, candidate[f], original[f], targets[f], cfg, False)
                residual.extend(rr)
            residual = np.asarray(residual)
            z = candidate.ravel()
            merit = .5 * (residual @ residual + z @ (regularizer @ z)) + cfg['slack_weight'] * violation
            if merit < old_merit:
                q, accepted = candidate, True
                break
        if not accepted:
            break
    return q


def collision_penalty(geometry, q, cfg, extra_samples=()):
    total = 0.
    for t in sample_times(len(q), extra_samples):
        i = min(int(t), len(q) - 2); a = min(1., t - i)
        d, _, _ = geometry.distances(q[i] * (1 - a) + q[i + 1] * a, cfg['activation_distance_m'])
        total += geometry.penalty(d, cfg['target_clearance_m'])
    return total


def repair(q, geometry, targets, cfg, checkpoint=None):
    original = q.copy()
    velocities = actuator_parameters()
    speeds = np.array([velocities[n]['velocity'] for n in JOINT_NAMES[:27]])
    started = time.monotonic()
    print(f'Checking {len(q)} frames against {len(geometry.pairs)} self collision pairs...', flush=True)
    source_gaps = [float(geometry.distances(values)[0].min()) for values in q]
    before_bad = np.flatnonzero(np.array(source_gaps) < cfg['minimum_clearance_m'])
    before_minimum = min(source_gaps)
    if checkpoint and Path(checkpoint).exists():
        with np.load(checkpoint) as saved:
            candidate = saved['q']
        if candidate.shape != q.shape or not np.isfinite(candidate).all():
            raise ValueError('Invalid collision progress checkpoint')
        q = candidate.copy()
        print(f'Resuming collision trajectory from {checkpoint}', flush=True)
    else:
        for f in range(len(q)):
            q[f] = repair_frame(geometry, q[f].copy(), targets[f], original[f], cfg)
            if f == 0 or (f + 1) % 300 == 0 or f + 1 == len(q):
                print(f'Self collision frame repair {f + 1}/{len(q)}; elapsed={time.monotonic() - started:.1f}s', flush=True)
        q = project_velocities(q, geometry, speeds)
    if checkpoint:
        np.savez_compressed(checkpoint, q=q)
    windows = []
    certificate = None
    adaptive_samples = []
    for outer in range(cfg['trajectory_passes']):
        bad, minimum, _, _ = dense_contacts(geometry, q, cfg, extra_samples=adaptive_samples)
        print(f'Self collision trajectory pass {outer + 1}: {len(bad)} active frames; min gap={minimum * 1000:.2f} mm', flush=True)
        if minimum >= cfg['minimum_clearance_m'] * 2:
            certificate = geometry.certificate(q, cfg['minimum_clearance_m'], cfg['certificate_max_depth'])
            if certificate['status'] == 'PASS':
                break
            exact = certificate['first_failure']['frame']
            adaptive_samples = sorted(set(adaptive_samples) | {exact})
            print(f'Adding adaptive collision constraint at source frame {exact:.8f}', flush=True)
            bad = sorted(set(bad) | {min(len(q) - 1, max(0, int(exact)))})
        mask = np.zeros(len(q), dtype=bool)
        for f in bad:
            mask[max(0, f - cfg['context_frames']):min(len(q), f + cfg['context_frames'] + 1)] = True
        edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
        certificate = None
        for first, stop in zip(edges[::2], edges[1::2]):
            start = max(0, first - 1); stop = min(len(q), stop + 1)
            while start < stop - 2:
                end = min(stop, start + cfg['window_frames'])
                print(f'Collision window {start}:{end}', flush=True)
                local_samples = [v - start for v in adaptive_samples if start <= v <= end - 1]
                q[start:end] = refine_window(geometry, q[start:end].copy(), original[start:end], targets[start:end], speeds, cfg, local_samples)
                windows.append([int(start), int(end)])
                if checkpoint:
                    np.savez_compressed(checkpoint, q=q)
                if end == stop:
                    break
                start = end - 2 * cfg['context_frames']
    if certificate is None:
        certificate = geometry.certificate(q, cfg['minimum_clearance_m'], cfg['certificate_max_depth'])
    report = dict(version=VERSION, config=cfg, before_minimum_distance_m=before_minimum,
                  before_active_frames=len(before_bad), before_sampling_fps=30, certificate=certificate,
                  max_joint_change_rad=float(abs(q - original).max()),
                  max_arm_joint_change_rad=float(abs(q[:, 13:] - original[:, 13:]).max()),
                  max_leg_joint_change_rad=float(abs(q[:, :12] - original[:, :12]).max()),
                  windows=windows, adaptive_constraint_frames=adaptive_samples, elapsed_s=time.monotonic() - started,
                  checks=dict(continuous_self_collision=certificate['status'] == 'PASS',
                              finite=bool(np.isfinite(q).all()),
                              joint_ranges=bool(np.all(q >= geometry.lo - 1e-6) and np.all(q <= geometry.hi + 1e-6)),
                              velocity_limits=bool(np.all(abs(np.diff(q, axis=0)) * 30 <= speeds + 1e-5))))
    return q, report


def human_targets(out, stem):
    with np.load(Path(out) / 'reference' / f'{stem}_human.npz') as h:
        names, positions, quats = list(h['names']), h['positions'], h['quats_wxyz']
    upper = json.loads((Path(out) / 'diagnostics' / stem / 'upper_body_metrics.json').read_text())
    offset = Rotation.from_quat(np.array(upper['config']['human_root_to_robot_wxyz'])[[1, 2, 3, 0]]).as_matrix()
    root = Rotation.from_quat(quats[:, names.index('Hips')][:, [1, 2, 3, 0]]).as_matrix() @ offset
    targets = []
    for label in ('Left', 'Right'):
        ids = [names.index(label + v) for v in ('Arm', 'ForeArm', 'Hand')]
        directions = np.diff(positions[:, ids], axis=1)
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        targets.append(np.einsum('tji,tsj->tsi', root, directions))
    return np.stack(targets, axis=1)


def quality(geometry, q, targets, cfg):
    actual = []
    lengths = []
    for values in q:
        geometry.pose(values)
        dirs, _ = geometry.directions()
        actual.append(dirs)
        if not lengths:
            lengths = [np.linalg.norm(np.diff(geometry.d.xpos[bs], axis=0), axis=1) for bs in geometry.arm_bodies]
    actual = np.asarray(actual)
    angles = np.rad2deg(np.arccos(np.clip(np.sum(actual * targets, axis=-1), -1, 1)))
    wrist = np.linalg.norm(np.sum((actual - targets) * np.asarray(lengths)[None, :, :, None], axis=2), axis=-1)
    horizontal = lambda v: (abs(v[..., 2]) < np.sin(np.deg2rad(15))) & (abs(v[..., 1]) > .866)
    records = {}
    checks = {}
    for window, end in (('full', len(q)), ('first10s', min(301, len(q)))):
        upper = np.percentile(angles[:end, :, 0], 95, axis=0)
        fore = np.percentile(angles[:end, :, 1], 95, axis=0)
        hand = np.percentile(wrist[:end], 95, axis=0)
        excess = np.mean(horizontal(actual[:end, :, 0]), axis=0) - np.mean(horizontal(targets[:end, :, 0]), axis=0)
        records[window] = dict(upper_p95_deg=upper.tolist(), forearm_p95_deg=fore.tolist(),
                               wrist_p95_m=hand.tolist(), artificial_horizontal_excess_fraction=excess.tolist())
        checks[window + '_direction'] = bool(np.max(upper) <= cfg['quality_upper_p95_deg'] and np.max(fore) <= cfg['quality_forearm_p95_deg'])
        checks[window + '_wrist'] = bool(np.max(hand) <= cfg['quality_wrist_p95_m'])
        checks[window + '_no_persistent_spread'] = bool(np.max(excess) <= cfg['quality_horizontal_excess_fraction'])
    return dict(measurements=records, checks=checks)


def repair_export(out, stem, cfg):
    """Repair one private staging export; callers publish only after validation."""
    from prepare_s54_motion import prepare
    if (cfg.get('version') != VERSION or cfg['minimum_clearance_m'] < .001
            or cfg['target_clearance_m'] < cfg['minimum_clearance_m'] + .003
            or cfg['validation_fps'] != 150 or cfg['window_frames'] <= 2 * cfg['context_frames'] + 2):
        raise ValueError('Invalid self collision safety configuration')
    out = Path(out)
    path = out / f'{stem}.csv'
    x = np.loadtxt(path, delimiter=',', ndmin=2)
    model, _ = model_contract()
    geometry = CollisionGeometry(model)
    targets = human_targets(out, stem)
    checkpoint = out / 'diagnostics' / stem / 'collision_progress.npz'
    q, report = repair(x[:, 7:34].copy(), geometry, targets, cfg, checkpoint)
    report['quality'] = quality(geometry, q, targets, cfg)
    report['checks'].update(report['quality']['checks'])
    report['status'] = 'PASS' if all(report['checks'].values()) else 'FAIL'
    diagnostic = out / 'diagnostics' / stem
    (diagnostic / 'self_collision_metrics.json').write_text(json.dumps(report, indent=2))
    np.savez_compressed(diagnostic / 'self_collision_adjustment.npz', before_csv=x, joint_positions=q)
    if report['status'] != 'PASS':
        raise ValueError(f'Self collision repair/quality checks failed: {diagnostic / "self_collision_metrics.json"}')
    y = x.copy()
    y[:, 7:34] = q
    raw = out / 'raw' / f'{stem}.csv'
    np.savetxt(raw, y[:, :34], delimiter=',', fmt='%.17g')
    # These files belong to the private candidate, never the source dataset.
    path.unlink()
    path.with_suffix('.json').unlink()
    prepare(raw, path, expected_frames=len(y), raw_dir=raw.parent)
    metadata = json.loads(path.with_suffix('.json').read_text())
    metadata['self_collision_repair'] = report
    path.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
    pkl_path = out / 'reference' / f'{stem}_30fps.pkl'
    with pkl_path.open('rb') as f:
        motion = pickle.load(f)
    motion['dof_pos'] = q
    with pkl_path.open('wb') as f:
        pickle.dump(motion, f)
    return report
