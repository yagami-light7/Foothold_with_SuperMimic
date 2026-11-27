"""复用已归档的 Newton 轨迹，审计根部姿态、目标编码和反事实运动学。"""

from argparse import ArgumentParser, Namespace
from contextlib import chdir
from copy import copy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

if __name__ == '__main__':
    os.environ['SUPERMIMIC_ROBOT'] = 'roban'

import numpy as np
import torch

from ..kinematics import RobotKinematics
from ..motion import CONTROL_DT
from ..roban import JOINT_NAMES, ROOT_BODY, model_contract
from ..rotations import mat_to_6d, quat_to_mat, so3_exp, yaw_mat
from .m4_task import build_reference_tracking
from .m50_reference import MANIFEST, REFERENCE_ROOT, REPO, load_clip, read_manifest
from .m50_task import MultiReferenceGoalSpec, build_m50_task
from .m410_evaluation import mean_metrics
from .m50_evaluation import DEFAULT_SEEDS, HOLDOUT_SEEDS, PERTURB_SEEDS
from .m51_task import OUTPUT as M51_OUTPUT, StanceFootMDP
from .m52_orientation import (WAIST, angle_error, counterfactual, distribution, euler, posture_metrics,
                              python, rms, tilt_angle, trace_states)
from .m52_task import OUTPUT, SOURCE


ARCHIVE = M51_OUTPUT/'cuda_evaluation.npz'
ORIGINAL = REPO.parent/'PMT/TCRS/outputs/roban_motion_dataset/train/flat/forward_012/npz/body_reference.npz'
BASELINES = ('m50', 'a1000', 'b2000', 'c1500', 'c2000')


def update_audit(path, **changes):
    saved = json.loads(path.read_text()) if path.is_file() else {}
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as scratch:
        tmp = Path(scratch)/path.name
        tmp.write_text(json.dumps(saved | changes, ensure_ascii=False, allow_nan=False, indent=2)+'\n')
        tmp.replace(path)


def phases(motion, count, device):
    t = torch.arange(count, device=device) * CONTROL_DT
    return dict(startup=t < float(motion.liftoff_time[0]),
                walking=(t >= float(motion.liftoff_time[0])) & (t <= float(motion.touchdown_time[-1]) + 1e-6))


def trace_posture(trace, motion, kin, *, steps=None):
    state, ref = trace_states(trace, motion, None if steps is None else steps+1)
    return posture_metrics(state, ref, kin, segments=phases(motion, len(state[0]), state[0].device))


def reference_stats(R, omega, joints):
    return dict(rpy=distribution(euler(R)), omega_rms_rad_s=python(rms(omega)),
                quaternion_order='wxyz', rotation_matrix='world_from_body_columns',
                z_axis_tilt_rms_deg=python(rms(torch.atan2(R[..., :2, 2].norm(dim=-1), R[..., 2, 2]).rad2deg())),
                waist_rms_rad=python(rms(joints[..., WAIST])))


def correction(actual, prior):
    angles = angle_error(actual, prior).rad2deg()
    tilt = tilt_angle(actual, prior).rad2deg()
    return dict(rpy_rms_deg=python(rms(angles)), rpy_abs_max_deg=python(angles.abs().amax(0)),
                tilt_rms_deg=python(rms(tilt)), tilt_abs_max_deg=python(tilt.max()))


def frozen_task(kin, motion, saved, mdp_type=StanceFootMDP, args=None):
    params = Namespace(**saved['args']) if args is None else args
    spec = MultiReferenceGoalSpec('roban_foot', params.goal_frames, params.goal_dt)
    return build_reference_tracking(params, kin, motion, spec, mdp_type)[2]


def offline_goals(mdp):
    motion = mdp.motion
    starts = motion.clip_start
    clock = motion.clip_len.new_tensor((0, 31, 73))
    frame = starts+clock
    anchor = motion.anchor(starts)
    state = motion.ref_state(frame)
    # 当前机器人有自身航向和倾斜，参考的朝向仍应完整保留。
    state = (state[0], so3_exp(state[0].new_tensor([.13, -.11, .37])) @ state[1], *state[2:])
    goal = mdp.goals.goal(state, frame, clock, anchor).reshape(3, 10, 32)
    frames = motion.indices(frame, mdp.goals.spec.frames, mdp.goals.spec.dt)
    ref = motion.sample(frames, anchor)
    torch.testing.assert_close(goal[..., 26:], mat_to_6d(yaw_mat(state[1]).mT[:, None] @ ref['root_R']),
                               atol=0., rtol=0.)
    changed = copy(motion)
    changed.root_R = motion.root_R @ so3_exp(state[0].new_tensor([.08, 0., 0.]))
    # 右乘 Roll 只改变参考倾斜，参考前向与参考 Yaw 逐值不变。
    assert torch.equal(changed.root_R[..., :, 0], motion.root_R[..., :, 0])
    perturbed = copy(mdp.goals)
    perturbed.motion = changed
    other = perturbed.goal(state, frame, clock, anchor).reshape_as(goal)
    assert torch.equal(goal[..., :26], other[..., :26])
    # 在同一参考 Yaw 下，分别施加纯 Roll 和纯 Pitch。
    desired = yaw_mat(motion.root_R)
    for axis in ((.08, 0., 0.), (0., -.09, 0.)):
        changed.root_R = desired @ so3_exp(goal.new_tensor(axis))
        base_R = changed.root_R
        changed.root_R = desired
        neutral = perturbed.goal(state, frame, clock, anchor).reshape_as(goal)
        changed.root_R = base_R
        mixed = perturbed.goal(state, frame, clock, anchor).reshape_as(goal)
        assert torch.equal(neutral[..., :26], mixed[..., :26])
        assert ((neutral[..., 26:]-mixed[..., 26:]).norm(dim=-1) > .05).all()
    end = motion.clip_end
    near = end-1
    result = mdp.goals.goal(motion.ref_state(near), near, near-starts, anchor).reshape_as(goal)
    assert torch.equal(result[:, :, 22], result.new_full((3, 10), CONTROL_DT))
    assert (motion.indices(near, 10, .10) == end[:, None]).all()
    return dict(obs_dim=mdp.obs_dim, goals=list(goal.shape[1:]), critic_dim=mdp.critic_dim+mdp.goals.spec.dim,
                rotation_flat_indices=[26, 27, 28, 29, 30, 31], tilt_only_goal_max_delta=python((other-goal).abs().max()),
                foot_and_position_unchanged=True, packed_boundary_clamps=True)


@torch.no_grad()
def online_reference(cpu, mdp, entries):
    from roban_motion.m313_reference import ACTOR_FIELDS, HybridReference, LiveCache, StaticBank
    from roban_motion.m313_reward import make_task
    from roban_motion.m313_support import engine

    motion = mdp.motion
    g = engine(cpu, len(entries), str(motion.q.device), temporal=True)
    ids = g.envs
    initial = motion.ref_state(motion.clip_start)
    from roban_motion.m313_training import pack_initial
    from ..rotations import mat_to_quat

    commands = [(row['step_length'], row['step_frequency'], row['yaw_rate']) for row in entries]
    g.reset(ids, pack_initial(g, initial, ids, mat_to_quat), g.fk.tensor(commands))
    assert g.ensure_future().all()
    task = make_task(mdp)
    bank = StaticBank.from_motion(motion, mdp)
    reference = HybridReference(bank, LiveCache(g, task.kin, motion, task), len(ids), str(motion.q.device))
    zero = torch.zeros_like(ids)
    reference.reset(ids, ids, zero.bool())
    offline_ticket = reference.issue(zero)
    offline = mdp.goals.goal(initial, motion.clip_start, zero, motion.anchor(motion.clip_start))
    cached = task.goals.goal(initial, zero, zero, offline_ticket)
    torch.testing.assert_close(cached, offline, atol=1e-7, rtol=0.)
    reference.reset(ids, ids, ~zero.bool())
    live_ticket = reference.issue(zero)
    goal = task.goals.goal(initial, zero, zero, live_ticket).reshape(len(ids), 10, 32)
    values = live_ticket.sample(live_ticket.indices(zero, 10, .10), ACTOR_FIELDS)
    reconstructed = mat_to_6d(yaw_mat(initial[1]).mT[:, None] @ values['root_R'])
    torch.testing.assert_close(goal[..., 26:], reconstructed, atol=0., rtol=0.)
    assert (euler(values['root_R'])[..., :2].abs().amax(dim=(1, 2)) > .01).all()
    result = dict(goal_static_cache_delta=python((offline-cached).abs().max()),
                  goal_live_matrix_delta=python((reconstructed-goal[..., 26:]).abs().max()), clips={})
    length = [x['num_frames']-1 for x in entries]
    observations, guide = [], []
    for i in range(max(length)+1):
        node = g.provider.get_current(ids)
        assert node.valid.all()
        frame = node.frame
        observations.append((quat_to_mat(frame.root_orientation), frame.root_angular_velocity, frame.joint_position))
        guide.append(quat_to_mat(g.trajectory.read(ids, node.tick).guide_quaternion))
        if i < max(length):
            assert g.advance(ids.new_tensor(length)>i).valid.all()
    rotation, omega, joints = map(torch.stack, zip(*observations))
    guide = torch.stack(guide)
    for i, entry in enumerate(entries):
        select = (slice(0, length[i]+1), i)
        R = rotation[select]
        result['clips'][entry['clip_id']] = reference_stats(R, omega[select], joints[select]) | {
            'fk_minus_guide': correction(R, guide[select])}
    result['solver_level_counts'] = g.solver.level_counts.tolist()
    return result


def run_reference(root, original, prior, output):
    from roban_motion.m3_11 import PROJECT, setup

    kin = RobotKinematics('cpu')
    saved = torch.load(SOURCE, map_location='cpu', weights_only=False, mmap=True)
    manifest = read_manifest()
    c = Namespace(**saved['args'])
    c.m50_reference_root, c.m50_library = str(root), str(MANIFEST)
    _, _, mdp, _ = build_m50_task(c, kin, 'cpu')
    audit = dict(reference={}, policy={}, counterfactual={}, goal=offline_goals(mdp))
    model, _ = model_contract()
    chest, waist = model.body(ROOT_BODY).id, model.body('waist_yaw_link').id
    assert int(model.body_parentid[waist]) == chest
    assert all(int(model.body_parentid[model.body(f'zarm_{side}1_link').id]) == chest for side in 'lr')
    audit['chain'] = dict(root_and_chest=ROOT_BODY, waist_child='waist_yaw_link', waist_axis=[0, 0, 1],
                          shoulder_parent=ROOT_BODY)
    with np.load(original, allow_pickle=False) as raw:
        assert raw['quaternion_order'].item() == 'wxyz' and tuple(raw['joint_names']) == tuple(JOINT_NAMES)
        i = list(raw['body_names']).index(ROOT_BODY)
        rotation = quat_to_mat(torch.from_numpy(raw['root_quat']))
        omega = torch.from_numpy(raw['body_ang_vel_w'][:, i])
        joints = torch.from_numpy(raw['joint_pos'])
        audit['reference']['forward_012'] = reference_stats(rotation, omega, joints)
    with np.load(prior, allow_pickle=False) as f:
        raw, periodic = (quat_to_mat(torch.from_numpy(f[key])) for key in ('raw_root_local_quaternion', 'root_local_quaternion'))
        audit['prior'] = dict(source_time_s=f['source_time'][[0, -1]].tolist(),
                              raw=distribution(euler(raw)), periodic=distribution(euler(periodic)),
                              seam_change=correction(periodic, raw))
    with chdir(PROJECT):
        _, _, online = setup()
        audit['online'] = online_reference(online, mdp, manifest['train'])
    output.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(3, 2, figsize=(12, 8), layout='constrained')
    with np.load(ARCHIVE, allow_pickle=False) as package:
        cases = json.loads(package['metadata'].item())['cases']
        for split in ('train', 'heldout'):
            for col, entry in enumerate(manifest[split]):
                name = entry['clip_id']
                clip = load_clip(entry, kin, 'cpu', root, split)
                source = (root if split == 'train' else REPO) / entry['file_path']
                with np.load(source, allow_pickle=False) as file:
                    roots = ['diag__source_root_quaternion', 'diag__guide_root_quaternion']
                    captured, guide = (quat_to_mat(torch.from_numpy(file[key])).to(clip.root_R) for key in roots)
                R = clip.root_R
                audit['reference'][name] = reference_stats(R, clip.root_w, clip.q) | {
                    'sqp_minus_guide': correction(R, guide), 'guide_minus_source': correction(guide, captured)}
                if split != 'train':
                    continue
                t = np.arange(clip.num_frames)*CONTROL_DT
                ref_euler = euler(R).rad2deg()
                for i, axis in enumerate(('Roll', 'Pitch')):
                    ax[col, i].plot(t, ref_euler[:, i], color='black', lw=1.8, label='Reference')
                    ax[col, i].set(title=f'{name} {axis}', ylabel='deg', xlabel='s')
                    ax[col, i].axvspan(0, float(clip.liftoff_time[0]), color='0.92', zorder=-1)
                    ax[col, i].grid(alpha=.25)
                for key in BASELINES:
                    rows = cases[key]['seen'][name]
                    values, diagnostics = {}, []
                    for seed, row in rows.items():
                        trace = {field: package[row['prefix']+field] for field in ('state', 'frame')}
                        values[int(seed)] = trace_posture(trace, clip, kin)
                        if int(seed) == DEFAULT_SEEDS[0]:
                            state, _ = trace_states(trace, clip)
                            angles = euler(state[1]).rad2deg()
                            for i in range(2):
                                ax[col, i].plot(t[:len(angles)], angles[:, i], lw=1, label=key.upper())
                        if key == 'c1500' and int(seed) in DEFAULT_SEEDS:
                            diagnostics.append(counterfactual(*trace_states(trace, clip), kin))
                    cohorts = dict(nominal=DEFAULT_SEEDS, perturb=PERTURB_SEEDS, holdout=HOLDOUT_SEEDS)
                    audit['policy'].setdefault(key, {})[name] = {
                        group: mean_metrics([values[seed] for seed in seeds if seed in values])
                        for group, seeds in cohorts.items() if any(seed in values for seed in seeds)}
                    if diagnostics:
                        audit['counterfactual'][name] = mean_metrics(diagnostics)
                expected = audit['reference'][name]['rpy']
                actual = audit['policy']['c1500'][name]['nominal']['all']
                hands = audit['counterfactual'][name]
                print(f'{name} 参考 / C1500 平均 Roll-Pitch {expected["mean_deg"][:2]} / '
                      f'{actual["angles"]["mean_deg"][:2]}，Tilt RMS {actual["tilt_rms_deg"]:.2f}°，'
                      f'手部 D0→D1 {[hands[k]["hand_world_m_rms"] for k in ("D0", "D1")]} m', flush=True)
    ax[0, 0].legend(ncol=2, fontsize=8)
    fig.savefig(output/'root_orientation_audit.pdf')
    plt.close(fig)
    return audit


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, default=REFERENCE_ROOT)
    parser.add_argument('--original', type=Path, default=ORIGINAL)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(2)
    prior = args.reference_root/'m3_10/fixed_prior/periodic_prior.npz'
    audit = run_reference(args.reference_root, args.original, prior, args.out)
    update_audit(args.out/'preflight.json', orientation=audit)


if __name__ == '__main__':
    main()
