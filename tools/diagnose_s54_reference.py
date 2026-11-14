"""Numerical, read-only diagnosis of the existing S54 reference and evaluation.

Prepare optional human targets with diagnose_s54_human.py in the existing GMR env.
All generated tables/figures go under reports; source assets, data and runs are read-only.
"""
import argparse
import csv
import json
from pathlib import Path
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.kinematics import S54Kinematics
from supermimic.motion import MotionLib, CONTROL_DT
from supermimic.mdp import MDP, RewardCfg
from supermimic.rotations import quat_to_mat
from supermimic.s54 import JOINT_NAMES, SOURCE_MJCF, MJCF_PATH, default_joint_pos_list
from render import two_robot_xml, set_pose


def stats(x):
    return dict(min=float(np.min(x)), p5=float(np.percentile(x, 5)), median=float(np.median(x)),
                p95=float(np.percentile(x, 95)), max=float(np.max(x)))


def write_csv(path, rows):
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def rotation_error(a, b):
    return Rotation.from_matrix(np.swapaxes(a, -1, -2) @ b).magnitude()


def set_reference(m, d, p, quat_xyzw, joints):
    d.qpos[:] = m.qpos0
    d.qpos[:3] = p; d.qpos[3:7] = quat_xyzw[[3, 0, 1, 2]]
    for name, value in zip(JOINT_NAMES, joints):
        d.qpos[m.jnt_qposadr[m.joint(name).id]] = value
    mj.mj_forward(m, d)


def geom_gap(m, d, g, normal, plane_point):
    """Independent support-point distance for the actual box/capsule/visual mesh.

    This is geometry only, not a second kinematics implementation.
    """
    local_n = d.geom_xmat[g].reshape(3, 3).T @ normal
    center_gap = float((d.geom_xpos[g] - plane_point) @ normal)
    kind = m.geom_type[g]
    if kind == mj.mjtGeom.mjGEOM_BOX:
        return center_gap - np.abs(local_n) @ m.geom_size[g]
    if kind == mj.mjtGeom.mjGEOM_CAPSULE:
        return center_gap - m.geom_size[g, 0] - m.geom_size[g, 1] * abs(local_n[2])
    if kind == mj.mjtGeom.mjGEOM_MESH:
        mesh = m.geom_dataid[g]; start = m.mesh_vertadr[mesh]; count = m.mesh_vertnum[mesh]
        return center_gap + float(np.min(m.mesh_vert[start:start + count] @ local_n))
    raise ValueError(f'Unexpected foot geometry type: {kind}')


def foot_analysis(m, p, quat, q, fps, out, human_path):
    d = mj.MjData(m); floor = m.geom('floor').id
    mj.mj_forward(m, d)
    normal = d.geom_xmat[floor].reshape(3, 3)[:, 2].copy(); origin = d.geom_xpos[floor].copy()
    assert np.allclose(normal, [0, 0, 1]), 'This diagnostic expects the actual horizontal floor'
    feet = [m.body(f'leg_{s}6_link').id for s in 'lr']
    hands = [m.body(f'zarm_{s}7_link').id for s in 'lr']
    shoulders = [m.body(f'zarm_{s}1_link').id for s in 'lr']
    elbows = [m.body(f'zarm_{s}4_link').id for s in 'lr']
    collision = [np.flatnonzero((m.geom_bodyid == b) & ((m.geom_contype | m.geom_conaffinity) != 0)) for b in feet]
    visual = [np.flatnonzero((m.geom_bodyid == b) & (m.geom_type == mj.mjtGeom.mjGEOM_MESH)) for b in feet]
    sites = [m.site(f'{s}_ft_frame').id for s in 'lr']
    # Explicitly verify positive separation and negative penetration, for each actual primitive type.
    sign_tests = []
    for side in collision:
        for kind in np.unique(m.geom_type[side]):
            g = int(next(g for g in side if m.geom_type[g] == kind))
            for expected in (-0.02, 0.03):
                d.qpos[:] = m.qpos0; mj.mj_forward(m, d)
                d.qpos[2] += expected - geom_gap(m, d, g, normal, origin); mj.mj_forward(m, d)
                native = mj.mj_geomDistance(m, d, floor, g, 10.0, None)
                assert abs(native - expected) < 1e-8, (g, expected, native)
                sign_tests.append(dict(geom=g, type=int(kind), expected_m=expected, measured_m=float(native)))
    gaps, visual_gaps, links, site_z, hand_p, shoulder_p, elbow_p, hand_R = [], [], [], [], [], [], [], []
    foot_p, foot_R = [], []
    support_error = 0.0
    for root_p, root_q, joint_q in zip(p, quat, q):
        set_reference(m, d, root_p, root_q, joint_q)
        row, mesh_row = [], []
        for side, mesh in zip(collision, visual):
            native = np.array([mj.mj_geomDistance(m, d, floor, int(g), 10.0, None) for g in side])
            analytic = np.array([geom_gap(m, d, int(g), normal, origin) for g in side])
            support_error = max(support_error, float(np.max(abs(native - analytic))))
            row.append(native.min()); mesh_row.append(min(geom_gap(m, d, int(g), normal, origin) for g in mesh))
        gaps.append(row); visual_gaps.append(mesh_row); links.append(d.xpos[feet, 2].copy())
        site_z.append(d.site_xpos[sites, 2].copy()); hand_p.append(d.xpos[hands].copy())
        shoulder_p.append(d.xpos[shoulders].copy()); elbow_p.append(d.xpos[elbows].copy())
        hand_R.append(d.xmat[hands].reshape(2, 3, 3).copy())
        foot_p.append(d.xpos[feet].copy()); foot_R.append(d.xmat[feet].reshape(2, 3, 3).copy())
    assert support_error < 1e-8, support_error
    gaps, visual_gaps, links, site_z = map(np.asarray, (gaps, visual_gaps, links, site_z))
    t = np.arange(len(p)) / fps
    report = dict(plane_position_m=origin.tolist(), plane_normal=normal.tolist(), sign_tests=sign_tests,
                  native_support_max_error_m=support_error, root_z_m=stats(p[:, 2]))
    for i, side in enumerate(('left', 'right')):
        worst = np.argsort(gaps[:, i])[:5]
        report[side] = dict(collision_gap_m=stats(gaps[:, i]), visual_gap_m=stats(visual_gaps[:, i]),
                            below_5mm_fraction=float(np.mean(gaps[:, i] < -0.005)),
                            below_20mm_fraction=float(np.mean(gaps[:, i] < -0.02)),
                            visual_below_5mm_fraction=float(np.mean(visual_gaps[:, i] < -0.005)),
                            visual_minus_collision_m=stats(visual_gaps[:, i] - gaps[:, i]),
                            foot_origin_above_collision_m=stats(links[:, i] - gaps[:, i]),
                            site_minus_link_max_m=float(abs(site_z[:, i] - links[:, i]).max()),
                            worst=[dict(frame=int(f), time_s=float(t[f]), gap_m=float(gaps[f, i]), root_z_m=float(p[f, 2])) for f in worst])
    report['both_below_5mm_fraction'] = float(np.mean((gaps < -0.005).all(1)))
    report['both_below_20mm_fraction'] = float(np.mean((gaps < -0.02).all(1)))
    report['lower_foot_gap_m'] = stats(gaps.min(1))
    write_csv(out / 'foot_ground.csv', [dict(time_s=t[f], root_z_m=p[f, 2], left_gap_m=gaps[f, 0],
              right_gap_m=gaps[f, 1], left_visual_gap_m=visual_gaps[f, 0], right_visual_gap_m=visual_gaps[f, 1],
              left_link_z_m=links[f, 0], right_link_z_m=links[f, 1]) for f in range(len(p))])
    geometry = []
    for label, side in zip(('left', 'right'), collision):
        for g in side:
            geometry.append(dict(side=label, geom=int(g), body=m.body(int(m.geom_bodyid[g])).name,
                                 type=int(m.geom_type[g]), local_pos=m.geom_pos[g].tolist(),
                                 local_quat_wxyz=m.geom_quat[g].tolist(), size=m.geom_size[g].tolist(),
                                 contype=int(m.geom_contype[g]), conaffinity=int(m.geom_conaffinity[g])))
    report['collision_geometry'] = geometry
    report['foot_sites'] = [dict(name=m.site(s).name, local_position=m.site_pos[s].tolist()) for s in sites]
    if human_path.exists():
        h = np.load(human_path); names = list(h['names']); foot_p, foot_R = np.asarray(foot_p), np.asarray(foot_R)
        report['ik_foot_tasks'] = {}
        for i, label in enumerate(('Left', 'Right')):
            fi, ti = names.index(label + 'Foot'), names.index(label + 'Toe')
            tp = h['target_positions'][:, fi]
            tR = Rotation.from_quat(h['target_quats_wxyz'][:, fi][:, [1, 2, 3, 0]]).as_matrix()
            target_gap = []
            # Rigidly place only the foot geoms at the exact IK foot-body target.
            # This tests whether the target itself is ground-safe, without solving IK.
            for target_p, target_R in zip(tp, tR):
                for g in collision[i]:
                    d.geom_xpos[g] = target_p + target_R @ m.geom_pos[g]
                    d.geom_xmat[g] = (target_R @ Rotation.from_quat(m.geom_quat[g][[1, 2, 3, 0]]).as_matrix()).reshape(-1)
                target_gap.append(min(geom_gap(m, d, int(g), normal, origin) for g in collision[i]))
            report['ik_foot_tasks'][label] = dict(
                actual_ankle5_to_foot6_origin_distance_m=float(np.linalg.norm(m.body_pos[feet[i]])),
                target_foot_to_toe_distance_m=stats(np.linalg.norm(h['target_positions'][:, ti] - tp, axis=-1)),
                foot_task_position_residual_m=stats(np.linalg.norm(foot_p[:, i] - tp, axis=-1)),
                foot_task_z_residual_m=stats(foot_p[:, i, 2] - tp[:, 2]),
                foot_task_orientation_residual_deg=stats(np.rad2deg(rotation_error(foot_R[:, i], tR))),
                first10_foot_normal_tilt_deg=stats(np.rad2deg(np.arccos(np.clip(foot_R[:301, i, 2, 2], -1, 1)))),
                exact_foot_target_sole_gap_m=stats(target_gap),
                exact_foot_target_below5mm_fraction=float(np.mean(np.array(target_gap) < -.005)))
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for i, label in enumerate(('left', 'right')):
        ax[0].plot(t, gaps[:, i] * 1000, label=f'{label} collision', lw=.7)
        ax[0].plot(t, visual_gaps[:, i] * 1000, label=f'{label} visual', lw=.5, alpha=.5)
    ax[0].axhline(0, color='k', lw=.7); ax[0].axhline(-20, color='red', ls='--', lw=.6)
    ax[0].set(ylabel='Foot-plane gap (mm)', ylim=(-220, 350)); ax[0].legend(ncol=2)
    ax[1].plot(t, p[:, 2], label='root Z'); ax[1].plot(t, gaps.min(1), label='lower foot gap')
    ax[1].set(xlabel='Time (s)', ylabel='Height (m)'); ax[1].legend()
    fig.suptitle('Original 30 FPS PKL, native MuJoCo geometry distances (swing heights clipped in plot)')
    fig.tight_layout(); fig.savefig(out / 'foot_ground_and_root.png', dpi=160); plt.close(fig)
    return report, dict(gaps=gaps, visual_gaps=visual_gaps, hands=np.array(hand_p), shoulders=np.array(shoulder_p),
                        elbows=np.array(elbow_p), hand_R=np.array(hand_R))


def arms_analysis(m, p, R, q, fps, body, out, human_path):
    rows, hand_stats = [], {}
    qd = np.gradient(q, 1 / fps, axis=0)
    for window, count in [('first10s', min(len(q), int(10 * fps) + 1)), ('full', len(q))]:
        for j, name in enumerate(JOINT_NAMES):
            if not name.startswith('zarm_'): continue
            lo, hi = m.jnt_range[m.joint(name).id]; values = q[:count, j]
            rows.append(dict(window=window, joint=name, min_deg=float(np.rad2deg(values.min())),
                             max_deg=float(np.rad2deg(values.max())), range_deg=float(np.rad2deg(np.ptp(values))),
                             std_deg=float(np.rad2deg(values.std())), rms_velocity_rad_s=float(np.sqrt(np.mean(qd[:count, j] ** 2))),
                             near_limit_1deg_fraction=float(np.mean((values - lo < np.deg2rad(1)) | (hi - values < np.deg2rad(1))))))
        relative = np.einsum('tji,tsj->tsi', R[:count], body['hands'][:count] - p[:count, None])
        arm_vec = np.einsum('tji,tsj->tsi', R[:count], body['elbows'][:count] - body['shoulders'][:count])
        elevated = np.rad2deg(np.arctan2(arm_vec[..., 2], np.linalg.norm(arm_vec[..., :2], axis=-1)))
        lateral = abs(arm_vec[..., 1]) / np.linalg.norm(arm_vec, axis=-1)
        hand_stats[window] = {s:dict(world_range_xyz_m=np.ptp(body['hands'][:count, i], axis=0).tolist(),
                                       root_frame_range_xyz_m=np.ptp(relative[:, i], axis=0).tolist(),
                                       upper_arm_elevation_deg=stats(elevated[:, i]),
                                       lateral_horizontal_fraction=float(np.mean((abs(elevated[:, i]) < 15) & (lateral[:, i] > .866))))
                              for i, s in enumerate(('left', 'right'))}
    write_csv(out / 'arm_joint_statistics.csv', rows)
    report = dict(joints=rows, hand_motion=hand_stats,
                  finger_joints=[m.joint(j).name for j in range(m.njnt) if any(x in (m.joint(j).name or '').lower() for x in ('finger', 'thumb'))])
    if human_path.exists():
        h = np.load(human_path); names = list(h['names']); human_stats = {}
        for i, label in enumerate(('Left', 'Right')):
            idx = names.index(label + 'Hand'); target = h['target_positions'][:, idx]
            target_R = Rotation.from_quat(h['target_quats_wxyz'][:, idx][:, [1, 2, 3, 0]]).as_matrix()
            human_stats[label] = dict(
                zero_weight_hand_position_residual_m=stats(np.linalg.norm(body['hands'][:, i] - target, axis=-1)),
                hand_orientation_residual_deg=stats(np.rad2deg(rotation_error(body['hand_R'][:, i], target_R))))
            for window, count in [('first10s', 301), ('full', len(q))]:
                hips = names.index('Hips'); rawp = h['positions'][:count]
                # Use the target root orientation so the human axes match the robot's root convention.
                root_R = Rotation.from_quat(h['target_quats_wxyz'][:count, hips][:, [1, 2, 3, 0]]).as_matrix()
                local = np.einsum('tji,tj->ti', root_R, rawp[:, idx] - rawp[:, hips])
                upper = rawp[:, names.index(label + 'ForeArm')] - rawp[:, names.index(label + 'Arm')]
                upper = np.einsum('tji,tj->ti', root_R, upper)
                elevation = np.rad2deg(np.arctan2(upper[:, 2], np.linalg.norm(upper[:, :2], axis=-1)))
                lateral = abs(upper[:, 1]) / np.linalg.norm(upper, axis=-1)
                human_stats[label][window] = dict(human_hand_root_range_xyz_m=np.ptp(local, axis=0).tolist(),
                    human_lateral_horizontal_fraction=float(np.mean((abs(elevation) < 15) & (lateral > .866))),
                    human_upper_arm_elevation_deg=stats(elevation))
        report['human_comparison'] = human_stats
        report['human_height_assumption_actual_m'] = float(h['human_height'])
        report['ground_offset_m'] = float(h['ground_offset'])
        hips = names.index('Hips')
        target_root_R = Rotation.from_quat(h['target_quats_wxyz'][:, hips][:, [1, 2, 3, 0]]).as_matrix()
        report['root_ik_orientation_residual_deg'] = stats(np.rad2deg(rotation_error(R, target_root_R)))
    names = [r['joint'] for r in rows if r['window'] == 'full']; x = np.arange(len(names))
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for window, offset in [('first10s', -.18), ('full', .18)]:
        values = [r for r in rows if r['window'] == window]
        ax[0].bar(x + offset, [r['range_deg'] for r in values], width=.36, label=window)
        ax[1].bar(x + offset, [100 * r['near_limit_1deg_fraction'] for r in values], width=.36, label=window)
    ax[0].set(ylabel='Joint range (degrees)'); ax[0].legend()
    ax[1].set(ylabel='Within 1 degree of hard limit (%)', xticks=x, xticklabels=names)
    ax[1].tick_params(axis='x', rotation=55); fig.tight_layout(); fig.savefig(out / 'arm_motion.png', dpi=160); plt.close(fig)
    return report


def model_and_fk(gmr_m, p, quat, q, out):
    copied = mj.MjModel.from_xml_path(str(SOURCE_MJCF)); sim = mj.MjModel.from_xml_path(str(MJCF_PATH))
    fields = ['body_parentid', 'body_pos', 'body_quat', 'body_mass', 'body_ipos', 'body_inertia', 'body_iquat',
              'jnt_type', 'jnt_axis', 'jnt_pos', 'jnt_range', 'qpos0']
    report = {}
    for label, other in [('copied_source', copied), ('simulation_copy', sim)]:
        report[label] = {field:float(np.max(abs(getattr(gmr_m, field) - getattr(other, field)))) for field in fields}
        src_g = np.flatnonzero(gmr_m.geom_bodyid != 0); dst_g = np.flatnonzero(other.geom_bodyid != 0)
        report[label]['geometry_fields_max_error'] = {field:float(np.max(abs(getattr(gmr_m, field)[src_g] - getattr(other, field)[dst_g])))
                                                        for field in ('geom_pos', 'geom_quat', 'geom_size', 'geom_type')}
    kin = S54Kinematics(dtype=torch.float64)
    frames = np.unique(np.r_[np.linspace(0, len(p) - 1, 64).astype(int), np.array([0, 53, 150, 300, 900, 3216, 3491])])
    zero = torch.zeros(len(frames), 3, dtype=torch.float64)
    R = Rotation.from_quat(quat[frames]).as_matrix()
    tensor = lambda a:torch.tensor(a, dtype=torch.float64)
    bp, bR, _, _ = kin.forward(tensor(p[frames]), tensor(R), zero, zero, tensor(q[frames]), torch.zeros(len(frames), len(JOINT_NAMES), dtype=torch.float64), all_bodies=True)
    mrender = mj.MjModel.from_xml_string(two_robot_xml()); drender = mj.MjData(mrender)
    dgmr, dsim = mj.MjData(gmr_m), mj.MjData(sim)
    fk_pos, fk_rot, sim_pos, render_pos, render_rot = [], [], [], [], []
    for i, f in enumerate(frames):
        set_reference(gmr_m, dgmr, p[f], quat[f], q[f]); set_reference(sim, dsim, p[f], quat[f], q[f])
        set_pose(mrender, drender, p[f], R[i], q[f], prefix='ref_'); mj.mj_forward(mrender, drender)
        for b, name in enumerate(kin.body_names):
            gb = gmr_m.body(name).id; sb = sim.body(name).id; rb = mrender.body('ref_' + name).id
            fk_pos.append(np.linalg.norm(bp[i, b].numpy() - dgmr.xpos[gb]))
            fk_rot.append(float(rotation_error(bR[i, b].numpy(), dgmr.xmat[gb].reshape(3, 3))))
            sim_pos.append(np.linalg.norm(dsim.xpos[sb] - dgmr.xpos[gb]))
            render_pos.append(np.linalg.norm(drender.xpos[rb] - dgmr.xpos[gb]))
            render_rot.append(float(rotation_error(drender.xmat[rb].reshape(3, 3), dgmr.xmat[gb].reshape(3, 3))))
    report['sampled_frames'] = len(frames)
    report['fk_position_max_m'] = float(max(fk_pos)); report['fk_rotation_max_rad'] = float(max(fk_rot))
    report['simulation_mjcf_position_max_m'] = float(max(sim_pos))
    report['renderer_ghost_position_max_m'] = float(max(render_pos)); report['renderer_ghost_rotation_max_rad'] = float(max(render_rot))
    return report


def pipeline(pkl, raw, full, motion, recording, out):
    data = np.concatenate([pkl['root_pos'], pkl['root_rot'], pkl['dof_pos']], axis=1)
    fps = float(pkl['fps']); n = len(motion.q); times = np.arange(n) * CONTROL_DT
    # Independent SciPy interpolation at physical time; compare to the real MotionLib too.
    src_t = np.arange(len(data)) / fps
    independent_p = np.stack([np.interp(times, src_t, full[:, j]) for j in range(3)], -1)
    independent_q = np.stack([np.interp(times, src_t, full[:, j]) for j in range(7, 36)], -1)
    independent_R = Slerp(src_t, Rotation.from_quat(full[:, 3:7]))(times).as_matrix()
    report = dict(source_frames=len(data), source_fps=fps, source_duration_s=float(src_t[-1]),
                  control_frames=n, control_dt=CONTROL_DT, control_last_time_s=float(times[-1]),
                  pkl_csv_max_abs=float(abs(data - raw).max()), raw_training_first27_max_abs=float(abs(raw - full[:, :34]).max()),
                  head_values_rad=np.unique(full[:, -2:], axis=0).tolist(),
                  source_nonfinite=int(np.sum(~np.isfinite(data))), training_nonfinite=int(np.sum(~np.isfinite(full))),
                  root_quaternion_norm_max_error=float(abs(np.linalg.norm(full[:, 3:7], axis=-1) - 1).max()),
                  pkl_dof_layout=pkl['dof_layout'], npz_joint_names=list(recording['joint_names']),
                  motion_npz_root_max_m=float(abs(motion.root_p.numpy() - recording['ref_root_p']).max()),
                  motion_npz_joint_max_rad=float(abs(motion.q.numpy() - recording['ref_q']).max()),
                  motion_npz_rotation_max_rad=float(rotation_error(motion.root_R.numpy(), recording['ref_root_R']).max()),
                  independent_motion_root_max_m=float(abs(independent_p - motion.root_p.numpy()).max()),
                  independent_motion_joint_max_rad=float(abs(independent_q - motion.q.numpy()).max()),
                  independent_motion_rotation_max_rad=float(rotation_error(independent_R, motion.root_R.numpy()).max()))
    rows = []
    for t in (0, 1.78, 5, 10, 30):
        f = int(round(t / CONTROL_DT))
        rows.append(dict(time_s=t, control_frame=f, source_fractional_frame=t * fps,
                         root_error_m=float(np.linalg.norm(independent_p[f] - recording['ref_root_p'][f])),
                         rotation_error_rad=float(rotation_error(independent_R[f], recording['ref_root_R'][f])),
                         joint_max_error_rad=float(abs(independent_q[f] - recording['ref_q'][f]).max())))
    report['time_alignment'] = rows; write_csv(out / 'pipeline_time_alignment.csv', rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([r['time_s'] for r in rows], [r['joint_max_error_rad'] for r in rows], 'o-', label='joint max error (rad)')
    ax.plot([r['time_s'] for r in rows], [r['root_error_m'] for r in rows], 'o-', label='root error (m)')
    ax.set(xlabel='Time (s)', ylabel='Absolute interpolation difference', title='Independent 30-to-50 Hz reconstruction vs evaluation reference')
    ax.legend(); fig.tight_layout(); fig.savefig(out / 'pipeline_alignment.png', dpi=160); plt.close(fig)
    return report


def evaluation(recording, motion, kin, run, out):
    p = recording['root_p'][:, 0]; q = recording['q'][:, 0]; f = recording['f'][:, 0].astype(int)
    R = Rotation.from_quat(recording['root_quat'][:, 0][:, [1, 2, 3, 0]]).as_matrix()
    rp, rR = recording['ref_root_p'][f], recording['ref_root_R'][f]
    ez = abs(p[:, 2] - rp[:, 2]); et = abs(R[:, 2, 2] - rR[:, 2, 2]); cfg = RewardCfg()
    first = int(np.flatnonzero(~recording['alive'][:, 0])[0])
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list()))
    zero = torch.zeros(len(f), 3); states = (torch.tensor(p), torch.tensor(R, dtype=torch.float32), zero, zero, torch.tensor(q), torch.zeros_like(torch.tensor(q)))
    terminated = mdp.termination(states, torch.tensor(f)).numpy()
    # Pose-only terms do not depend on the unrecorded velocities/actions. Do not
    # report the aggregate reward from these placeholder inputs as an actual return.
    with torch.no_grad():
        _, pose_terms = mdp.reward(states, torch.tensor(f), torch.zeros_like(states[4]), torch.zeros_like(states[4]), terms=True)
    observed = np.flatnonzero(terminated)
    result = json.loads((run / 'ckpt_010000_eval.json').read_text())
    report = dict(result=result, recorded_frames=len(p), first_dead_index=first, first_dead_reference_frame=int(f[first]),
                  first_dead_time_s=float(f[first] * CONTROL_DT), first_threshold_crossing_index=int(observed[0]),
                  height_error_m=float(ez[first]), height_threshold_m=cfg.term_height,
                  tilt_scalar_error=float(et[first]), tilt_scalar_threshold=cfg.term_tilt,
                  root_z_m=float(p[first, 2]), reference_root_z_m=float(rp[first, 2]),
                  root_euler_xyz_deg=Rotation.from_matrix(R[first]).as_euler('xyz', degrees=True).tolist(),
                  reference_root_euler_xyz_deg=Rotation.from_matrix(rR[first]).as_euler('xyz', degrees=True).tolist(),
                  previous_height_error_m=float(ez[first - 1]), previous_tilt_error=float(et[first - 1]),
                  root_quat_wxyz=recording['root_quat'][first, 0].tolist(),
                  joint_positions_rad=dict(zip(JOINT_NAMES, q[first].tolist())),
                  previous_joint_positions_rad=dict(zip(JOINT_NAMES, q[first - 1].tolist())),
                  max_joint_error_rad=float(abs(q[first] - recording['ref_q'][f[first]]).max()),
                  height_trigger=bool(ez[first] > cfg.term_height), tilt_trigger=bool(et[first] > cfg.term_tilt),
                  alive_transition_fraction=float(first / (int(recording['clip_len'][0]) - 1)),
                  normalized_return_over_executed_steps=float(result['norm_return'] * (int(recording['clip_len'][0]) - 1) / first),
                  frame_sequence_contiguous=bool(np.array_equal(f, np.arange(len(f)))))
    report['pose_tracking'] = dict(
        anchored_body_position_rmse_m=stats(pose_terms['e_body_pos'][:first].sqrt().numpy()),
        root_position_error_m=stats(np.linalg.norm(p[:first] - rp[:first], axis=-1)),
        root_xy_error_m=stats(np.linalg.norm(p[:first, :2] - rp[:first, :2], axis=-1)),
        joint_rms_error_rad=float(np.sqrt(np.mean((q[:first] - recording['ref_q'][f[:first]]) ** 2))))
    violation = torch.relu(motion.q - mdp.soft_hi) + torch.relu(mdp.soft_lo - motion.q)
    report['reference_soft_limit_normalized_penalty_mean'] = float(
        violation.sum(-1).mean() * cfg.w_joint_limit * CONTROL_DT / 0.1)
    model = mj.MjModel.from_xml_path(str(SOURCE_MJCF)); data = mj.MjData(model)
    feet = [model.body(f'leg_{s}6_link').id for s in 'lr']
    geoms = [np.flatnonzero((model.geom_bodyid == b) & ((model.geom_contype | model.geom_conaffinity) != 0)) for b in feet]
    report['pose_samples'] = []
    for t in (0, 1.78, 5, 10, 30, 60, 100, 107.2):
        i = int(round(t / CONTROL_DT)); sample = dict(time_s=t, xy_error_m=float(np.linalg.norm(p[i, :2] - rp[i, :2])))
        for label, pos, rot, joints in [('policy', p[i], R[i], q[i]), ('reference', rp[i], rR[i], recording['ref_q'][f[i]])]:
            set_reference(model, data, pos, Rotation.from_matrix(rot).as_quat(), joints)
            sample[label + '_foot_gap_m'] = [float(min(mj.mj_geomDistance(model, data, model.geom('floor').id, int(g), 10.0, None) for g in side)) for side in geoms]
        report['pose_samples'].append(sample)
    for i, label in [(first - 1, 'previous_state'), (first, 'terminal_state')]:
        report[label] = dict(time_s=float(f[i] * CONTROL_DT), root_z_m=float(p[i, 2]), ref_z_m=float(rp[i, 2]),
                            policy_Rzz=float(R[i, 2, 2]), reference_Rzz=float(rR[i, 2, 2]),
                            joint_positions_rad=dict(zip(JOINT_NAMES, q[i].tolist())))
    table = np.genfromtxt(run / 'train_log.csv', delimiter=',', names=True)
    metrics = ['wm_loss', 'wm_err8', 'sim_rew', 'ep_ret_norm', 'imag_survival', 'term_frac', 'pi_loss', 'ep_len']
    report['training_windows'] = {}
    for start, end in [(0, 1000), (1000, 2000), (4000, 6000), (8000, 9000), (9000, 10000)]:
        mask = (table['iteration'] > start) & (table['iteration'] <= end)
        report['training_windows'][f'{start}-{end}'] = {n:float(np.mean(table[n][mask])) for n in metrics}
    report['training_last_row'] = {n:float(table[-1][n]) for n in table.dtype.names}
    time = f * CONTROL_DT
    write_csv(out / 'evaluation_state.csv', [dict(frame=int(f[i]), time_s=float(time[i]), alive=bool(recording['alive'][i, 0]),
              root_z_m=float(p[i, 2]), ref_z_m=float(rp[i, 2]), height_error_m=float(ez[i]),
              tilt_error=float(et[i]), joint_max_error_rad=float(abs(q[i] - recording['ref_q'][f[i]]).max())) for i in range(len(f))])
    fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    ax[0].plot(time, p[:, 2], label='policy root Z'); ax[0].plot(time, rp[:, 2], label='reference Z'); ax[0].set(ylabel='Height (m)'); ax[0].legend()
    ax[1].plot(time, ez, label='height error'); ax[1].axhline(cfg.term_height, color='r', ls='--'); ax[1].set(ylabel='Height error (m)')
    ax[2].plot(time, et, label='Rzz difference'); ax[2].axhline(cfg.term_tilt, color='r', ls='--'); ax[2].set(ylabel='Tilt scalar error')
    ax[3].plot(time, np.linalg.norm(p[:, :2] - rp[:, :2], axis=-1)); ax[3].set(xlabel='Time (s)', ylabel='Global XY error (m)')
    for a in ax: a.axvline(time[first], color='k', ls=':')
    fig.tight_layout(); fig.savefig(out / 'evaluation_failure.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True); it = table['iteration']
    ax[0].semilogy(it, table['wm_loss'], label='wm_loss'); ax[0].semilogy(it, table['wm_err8'], label='wm_err8'); ax[0].legend()
    ax[1].plot(it, table['sim_rew'], label='sim_rew'); ax[1].plot(it, table['imag_rew'], label='imag_rew'); ax[1].plot(it, table['ep_ret_norm'], label='episode return / executed length'); ax[1].legend()
    ax[2].plot(it, table['imag_survival'], label='imag_survival'); ax[2].plot(it, table['term_frac'], label='term_frac'); ax[2].set(xlabel='Iteration'); ax[2].legend()
    fig.tight_layout(); fig.savefig(out / 'training_curves.png', dpi=160); plt.close(fig)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gmr', type=Path, default=Path('/mnt/sage_data/workspace/Leju-GMR'))
    p.add_argument('--run_dir', type=Path, default=Path('logs/s54_dance1'))
    p.add_argument('--out', type=Path, default=Path('reports/s54_reference_figures'))
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True); torch.set_num_threads(4)
    pkl = pickle.load((a.gmr / 'output/kuavo_s54/pkl/dance1_subject1_30fps.pkl').open('rb'))
    raw = np.loadtxt(a.gmr / 'output/kuavo_s54/csv/dance1_subject1.csv', delimiter=',')
    full = np.loadtxt('library/motion_reference/kuavo_s54/retarget_data/reference/dance1_subject1.csv', delimiter=',')
    if raw.shape != (len(pkl['root_pos']), 34) or full.shape != (len(raw), 36): raise ValueError('Unexpected S54 reference shapes')
    source = mj.MjModel.from_xml_path(str(a.gmr / 'assets/kuavo_s54/xml/biped_s54.xml'))
    actual_names = [source.joint(j).name for j in range(source.njnt) if source.jnt_type[j] != mj.mjtJoint.mjJNT_FREE]
    if actual_names != JOINT_NAMES: raise ValueError('Source joint names/order differs from S54 contract')
    recording = dict(np.load(a.run_dir / 'traj_10000.npz', allow_pickle=False))
    if list(recording['joint_names']) != JOINT_NAMES: raise ValueError('NPZ joint names/order differs')
    fps = float(pkl['fps']); kin = S54Kinematics(); motion = MotionLib.from_dir('library/motion_reference/kuavo_s54/retarget_data/reference', kin, 'cpu')
    report = {'pipeline':pipeline(pkl, raw, full, motion, recording, a.out)}
    print('Data/time alignment complete', flush=True)
    report['feet'], body = foot_analysis(source, full[:, :3], full[:, 3:7], full[:, 7:], fps, a.out, a.out / 'human_targets.npz')
    print('Full-clip foot geometry complete', flush=True)
    R = Rotation.from_quat(full[:, 3:7]).as_matrix()
    report['arms'] = arms_analysis(source, full[:, :3], R, full[:, 7:], fps, body, a.out, a.out / 'human_targets.npz')
    report['models'] = model_and_fk(source, full[:, :3], full[:, 3:7], full[:, 7:], a.out)
    report['evaluation'] = evaluation(recording, motion, kin, a.run_dir, a.out)
    (a.out / 'diagnosis_metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(feet={s:{k:v for k,v in report['feet'][s].items() if k in ('collision_gap_m', 'below_5mm_fraction')} for s in ('left','right')},
                          first_failure_s=report['evaluation']['first_dead_time_s'],
                          height_trigger=report['evaluation']['height_trigger'], tilt_trigger=report['evaluation']['tilt_trigger'],
                          output=str(a.out)), indent=2))
    print('Confirmed: persistent foot penetration, co-located Foot/Toe robot frames with separated targets, saturated arm3 joints.')
    print('Not established: unique causal contribution of hand position cost, PD gains, or batch size; no ablation/retraining performed.')
    print('First fixes: correct Toe task frame; calibrate sole/ground targets; review arm frame offsets and limits.')
    print('Report: reports/s54_reference_diagnosis.md')


if __name__ == '__main__':
    main()
