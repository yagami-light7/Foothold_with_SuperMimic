"""Sparse constrained refinement of S54 arm trajectories using native FK Jacobians."""
import mujoco as mj
import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation
from qpsolvers import solve_qp


def left_log_jacobian_inverse(r):
    theta = np.linalg.norm(r)
    x, y, z = r
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    c = 1 / 12 if theta < 1e-5 else 1 / theta**2 - .5 / (theta * np.tan(theta / 2))
    return np.eye(3) - .5 * K + c * K @ K


def residual_jacobian(arm, full_qpos, q, target, target_R, jacobian=True):
    directions, _, hand_R = arm.pose(full_qpos, q)
    cfg = arm.cfg
    r = Rotation.from_matrix(target_R.T @ hand_R).as_rotvec()
    weights = np.array([cfg['upper_direction_weight'], cfg['forearm_direction_weight']])
    residual = np.r_[((directions - target) * weights[:, None]).ravel(), cfg['hand_orientation_weight'] * r]
    if not jacobian:
        return residual
    m, d = arm.m, arm.d
    mj.mj_comPos(m, d)
    dofs = [m.jnt_dofadr[m.joint(n).id] for n in arm.names]
    jp, jw = [], []
    for b in arm.bodies:
        p, w = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mj.mj_jacBody(m, d, p, w, b)
        jp.append(p[:, dofs]); jw.append(w[:, dofs])
    J = np.vstack([(jp[1] - jp[0]) * weights[0] / arm.lengths[0],
                   (jp[2] - jp[1]) * weights[1] / arm.lengths[1],
                   cfg['hand_orientation_weight'] * left_log_jacobian_inverse(r) @ target_R.T @ jw[3]])
    return residual, J


def _refine_window(q, arms, full_qpos, target_dirs, target_rotations, fps, iterations):
    """Sequential convex subproblems with exact linear velocity constraints.

    FK is nonlinear, so use native analytical Jacobians plus line search. Every
    accepted step is a convex combination of feasible trajectories. The method
    improves bone-direction tracking directly, rather than just joint closeness.
    """
    n = len(q)
    D = sparse.kron(sparse.diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1],
                                 shape=(n - 1, n), format='csc'), sparse.eye(7), format='csc')
    G = sparse.vstack([D, -D], format='csc')
    identity = sparse.eye(n * 7, format='csc')
    report = {}
    for side, arm in enumerate(arms):
        X = q[:, 13 + side * 7:20 + side * 7].copy()
        cfg = arm.cfg
        smooth = cfg['previous_pose_weight']**2
        neutral = cfg['neutral_pose_weight']**2
        regularizer = smooth * D.T @ D + neutral * identity
        limits = np.tile(arm.speed * (1 - cfg['velocity_numerical_margin']) / fps, n - 1)
        lo, hi = np.tile(arm.lo, (n, 1)), np.tile(arm.hi, (n, 1))
        if cfg['anatomical_elbow_branch']:
            hi[:, 3] = np.minimum(hi[:, 3], arm.straight_elbow)
        # Fixed boundary poses preserve velocity continuity with adjacent windows.
        lo[0] = hi[0] = X[0]
        lo[-1] = hi[-1] = X[-1]

        def evaluate(values, jacobian):
            residuals, blocks = [], []
            for f in range(n):
                result = residual_jacobian(arm, full_qpos[f], values[f], target_dirs[side][f], target_rotations[side][f], jacobian)
                if jacobian:
                    r, J = result; residuals.append(r); blocks.append(J)
                else:
                    residuals.append(result)
            r = np.concatenate(residuals)
            flat = values.ravel()
            cost = .5 * (r @ r + flat @ (regularizer @ flat))
            return cost, r, sparse.block_diag(blocks, format='csc') if jacobian else None

        initial, _, _ = evaluate(X, False)
        seed_selected = False
        if cfg.get('refinement_causal_seed', False) and initial > .01:
            # A short exact-IK branch switch may look good before velocity
            # projection and very poor afterwards. Also try the continuous
            # causal branch, then project it while retaining both endpoints.
            saved_cfg, saved_previous = arm.cfg, arm.previous
            arm.cfg = dict(cfg, temporal_mode='causal')
            arm.previous = X[0].copy()
            Y = X.copy()
            try:
                for f in range(1, n):
                    Y[f], _, _ = arm.solve(full_qpos[f], target_dirs[side][f], target_rotations[side][f], 1 / fps)
            finally:
                arm.cfg, arm.previous = saved_cfg, saved_previous
            candidate = solve_qp(identity, -Y.ravel(), G, np.r_[limits, limits],
                                 lb=lo.ravel(), ub=hi.ravel(), solver='proxqp', eps_abs=1e-9, eps_rel=0., max_iter=1000)
            if candidate is not None:
                Y = candidate.reshape(n, 7)
                seed_cost, _, _ = evaluate(Y, False)
                if seed_cost < initial:
                    X, seed_selected = Y, True
        start_cost, _, _ = evaluate(X, False)
        history = [float(start_cost)]
        fallbacks = 0
        for iteration in range(iterations):
            cost, r, J = evaluate(X, True)
            flat = X.ravel()
            H = J.T @ J + regularizer + cfg['refinement_damping'] * identity
            gradient = J.T @ r + regularizer @ flat
            diff = D @ flat
            # Keep a small extra tolerance margin for the nonlinear line search.
            step_lo = np.maximum((lo - X).ravel(), -cfg['refinement_trust_region_rad'])
            step_hi = np.minimum((hi - X).ravel(), cfg['refinement_trust_region_rad'])
            delta = solve_qp(H, gradient, G, np.r_[limits - diff, limits + diff],
                             lb=step_lo, ub=step_hi, solver='proxqp', eps_abs=1e-8, eps_rel=0., max_iter=300)
            if delta is None:
                # Small windows also fit the same dense DAQP solver used by
                # GMR. It handles active/fixed bounds differently from ProxQP.
                delta = solve_qp(H.toarray(), gradient, G.toarray(), np.r_[limits - diff, limits + diff],
                                 lb=step_lo, ub=step_hi, solver='daqp')
                fallbacks += 1
            if delta is None:
                raise RuntimeError('Arm trajectory refinement failed in both proxqp and daqp')
            accepted = False
            for scale in (1., .5, .25, .125, .0625):
                candidate = X + scale * delta.reshape(n, 7)
                new_cost, _, _ = evaluate(candidate, False)
                if new_cost < cost:
                    X, cost, accepted = candidate, new_cost, True
                    break
            history.append(float(cost))
            if not accepted or abs(history[-2] - history[-1]) < 1e-5:
                break
        q[:, 13 + side * 7:20 + side * 7] = X
        report[('left', 'right')[side]] = dict(initial_objective=float(initial), final_objective=float(history[-1]), history=history, daqp_fallbacks=fallbacks, causal_seed_selected=seed_selected)
    return q, report


def refine_trajectory(q, arms, full_qpos, target_dirs, target_rotations, fps, iterations):
    """Refine overlapping windows around poor tracking, fixing their endpoints.

    This keeps sparse QPs small enough for the existing laptop CPU. Window
    boundaries retain exact poses, so the complete trajectory stays continuous.
    """
    n = len(q)
    bad = np.zeros(n, dtype=bool)
    for side, arm in enumerate(arms):
        for f in range(n):
            dirs, _, _ = arm.pose(full_qpos[f], q[f, 13 + 7 * side:20 + 7 * side])
            angles = np.rad2deg(np.arccos(np.clip(np.sum(dirs * target_dirs[side][f], axis=1), -1, 1)))
            bad[f] |= np.max(angles) > arm.cfg['refinement_trigger_deg']
    context = min(arms[0].cfg['refinement_context_frames'], max(0, (n - 1) // 2))
    active = np.convolve(bad.astype(int), np.ones(2 * context + 1, dtype=int), mode='same') > 0
    edges = np.flatnonzero(np.diff(np.r_[False, active, False]))
    size = arms[0].cfg['refinement_window_frames']
    windows = []
    for first, end in zip(edges[::2], edges[1::2]):
        start = max(0, first - 1)
        end = min(n, end + 1)
        while start < end - 2:
            stop = min(start + size, end)
            windows.append((start, stop))
            if stop == end:
                break
            start = stop - 2 * context
    records = []
    for index, (first, end) in enumerate(windows):
        print(f'Refining window {index + 1}/{len(windows)}: frames {first}:{end}', flush=True)
        values, result = _refine_window(q[first:end].copy(), arms, full_qpos[first:end],
                                         [v[first:end] for v in target_dirs],
                                         [v[first:end] for v in target_rotations], fps, iterations)
        q[first:end] = values
        records.append(dict(first_frame=int(first), end_frame_exclusive=int(end), **result))
        print(f'Trajectory refinement window {index + 1}/{len(windows)}: {first / fps:.2f}-{end / fps:.2f}s', flush=True)
    return q, dict(windows=records, window_count=len(windows), trigger_deg=arms[0].cfg['refinement_trigger_deg'])
