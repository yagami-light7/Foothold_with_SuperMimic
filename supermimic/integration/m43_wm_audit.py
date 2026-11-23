"""同一真实 F2 轨迹上的 M4.3 多阶段 WM、Newton 差分和 Actor BPTT 审计。"""

import argparse
import json
import math
import os
from contextlib import ExitStack
from pathlib import Path

if __name__ == '__main__':
    os.environ['SUPERMIMIC_ROBOT'] = 'roban'

import numpy as np
import torch

from ..kinematics import RobotKinematics
from ..mdp import normalize_xi, unflatten_state
from ..motion import CONTROL_DT
from ..networks import Critic, Policy, WorldModel
from ..rotations import mat_to_quat
from .m4_task import build_m4_task
from .m42v3_ac import (episode_input, gradient, gradient_audit, load_trace, objective,
                       sample_starts, update_policy)
from .m42v3_wm import CONTACT_FORCE_N, evaluate_windows, normalizer_summary, select_windows

SLOPE_FLOOR = 1e-5


def scores(mdp, state, frame, action, previous):
    anchor = mdp.motion.anchor(torch.zeros_like(frame))
    reward, terms = mdp.reward(state, frame + 1, action, previous, terms=True,
                               command=anchor, ep_step=frame + 1)
    values = dict(foot=terms['foot_objective'], landing=terms['touchdown_contribution'], total=reward)
    if 'swing_clearance_contribution' in terms:
        values['clearance'] = terms['swing_clearance_contribution']
    return {name: value / mdp.max_step_reward for name, value in values.items()}


@torch.enable_grad()
def model_slopes(model, scoring, samples):
    action = samples['action'].detach().requires_grad_(True)
    state = unflatten_state(samples['state'])
    frame = samples['frame']
    successor = model.step(state, scoring[0][1].wm_action(action, frame), normalize_xi(samples['xi']))
    result = {}
    for label, mdp in scoring:
        values = scores(mdp, successor, frame, action, samples['previous'])
        result[label] = {name: torch.autograd.grad(value.sum(), action, retain_graph=True)[0].detach()
                         for name, value in values.items()}
    return result


def collect_samples(record, prefix, motion, mdp, count, device, *, joint_index=None):
    columns = {}
    support_channels = None
    for seed in record['seeds']:
        key = f'{prefix}_s{seed}_'
        f = len(record[key + 'state']) - 1
        if f < 1:
            continue

        def read(name):
            return torch.as_tensor(record[key + name], device=device)

        action = read('action')[:f]
        previous = read('previous_action')[:f]
        force = read('force')[:f, :, 2]
        pd = mdp.default_q + mdp.alpha * action
        margin = torch.minimum(pd - mdp.kin.q_lo, mdp.kin.q_hi - pd)
        action_change = (action - previous).square().mean(-1).sqrt()
        time = read('time')[:f]
        foot_error = read('foot_target_error')[:f]
        planned_swing = ~read('planned_contact')[:f].bool()
        error = torch.where(planned_swing.any(-1), (foot_error * planned_swing).amax(-1), foot_error.amax(-1))
        support = (force >= CONTACT_FORCE_N).sum(-1)
        high_change = action_change >= torch.quantile(action_change, .9)
        close_clamp = ((margin.abs() / mdp.alpha) <= .30).any(-1)
        touch_time = float(motion.touchdown_time[0])
        lift_time = float(motion.liftoff_time[0])
        masks = (support == 2, support == 1, high_change, close_clamp,
                 (time >= lift_time) & (time < touch_time),
                 (time - touch_time).abs() <= .20, error >= .30)
        if joint_index is not None:
            # 另外保留被驱动到限位及力矩饱和的真实控制步。
            j = joint_index
            torque = read('actuator_torque')[1:f + 1, j]
            q = unflatten_state(read('state')[:f])[4][:, j]
            near_stop = torch.minimum(q-mdp.kin.q_lo[j], mdp.kin.q_hi[j]-q) <= .015
            saturated = torque.abs() >= .99*mdp.effort[j]
            masks += (near_stop, saturated, ~saturated, margin[:, j] < 0., margin[:, j] >= 0.)
        uniform = torch.linspace(0, f-1, min(count, f), device=device).round().long()
        # 对少量但关键的姿态额外取样，避免均匀取帧错过转换。
        selected = [uniform]
        for mask in masks:
            ids = mask.nonzero(as_tuple=True)[0]
            if len(ids):
                selected.append(ids[torch.linspace(0, len(ids)-1, min(2, len(ids)),
                                                   device=device).round().long()])
        idx = torch.cat(selected).unique()
        warm = 'solver_warmstart' if key + 'solver_warmstart' in record else None
        quat = 'root_quat_xyzw' if key + 'root_quat_xyzw' in record else None
        channels = (warm is not None, quat is not None)
        if support_channels is not None and support_channels != channels:
            raise ValueError(f'{prefix} 的种子轨迹包含不一致的求解器快照字段')
        support_channels = channels
        past = (idx[:, None] + torch.arange(-9, 1, device=device)).clamp_min(0)
        rows = dict(state=read('state')[idx], next_foot=read('foot')[idx+1],
                    action=action[idx], previous=previous[idx], frame=read('frame')[idx].long(),
                    xi=read('xi')[idx], foot_error=error[idx], force=force[idx],
                    action_change=action_change[idx], near_clamp=close_clamp[idx],
                    history=read('observation')[past], goal=read('goal')[idx],
                    seed=torch.full_like(idx, int(seed)))
        if warm is not None:
            rows['warmstart'] = read(warm)[idx]
        if quat is not None:
            rows['root_quat'] = read(quat)[idx]
        if joint_index is not None:
            rows.update(ankle_torque=torque[idx], target=read('target')[idx])
        for name, value in rows.items():
            columns.setdefault(name, []).append(value)
    if support_channels is None:
        raise ValueError(f'{prefix} 没有可用控制步')
    return {name: torch.cat(value) for name, value in columns.items()}


def directions(action, mdp, gradients, eps_max):
    proposed = mdp.default_q + mdp.alpha * action
    active = ((proposed > mdp.kin.q_lo + mdp.alpha * eps_max)
              & (proposed < mdp.kin.q_hi - mdp.alpha * eps_max))
    rng = torch.Generator(device=action.device).manual_seed(4203)
    random = torch.randn((2, *action.shape), device=action.device, generator=rng)
    return torch.nn.functional.normalize(torch.cat((random, gradients[None]), 0) * active, dim=-1)


def restore(env, probe, samples, repeat):
    ids = torch.arange(env.N, device=env.device)
    env.reset(ids)
    env.f.copy_(samples['frame'].repeat(repeat))
    env.ep_step.copy_(env.f)
    repeated = samples['state'].repeat(repeat, 1)
    p, R, v, w, q, qd = unflatten_state(repeated)
    quat = samples.get('root_quat')
    quat = mat_to_quat(R)[:, [1, 2, 3, 0]] if quat is None else quat.repeat(repeat, 1)
    robot = env.robot
    robot.write_root_link_pose_to_sim_index(root_pose=torch.cat((p + env.scene.env_origins, quat), -1))
    robot.write_root_link_velocity_to_sim_index(root_velocity=torch.cat((v, w), -1))
    robot.write_joint_position_to_sim_index(position=q[:, env.sim_to_canonical])
    robot.write_joint_velocity_to_sim_index(velocity=qd[:, env.sim_to_canonical])
    env.scene.write_data_to_sim()
    env.refresh()
    if 'warmstart' in samples:
        # 求解器在重置时清理暖启动，须等刷新完成后写入轨迹当步保存的值。
        probe.warmstart.copy_(samples['warmstart'].repeat(repeat, 1))
    probe.overflow.zero_()
    return env.state()


def bootstrap_ci(sign, mask, frame):
    # 三个确定性 Reset 的同帧方向高度相关，按控制帧整簇重采样。
    frame = frame.expand_as(sign)
    ids, inverse = frame[mask].unique(return_inverse=True)
    if len(ids) < 2:
        return None
    bins = torch.zeros(len(ids), device=sign.device)
    successes = bins.scatter_add(0, inverse, sign[mask].float())
    counts = torch.bincount(inverse, minlength=len(ids))
    rng = torch.Generator(device=sign.device).manual_seed(4303)
    idx = torch.randint(len(ids), (400, len(ids)), generator=rng, device=sign.device)
    ratio = successes[idx].sum(1) / counts[idx].sum(1)
    return torch.quantile(ratio, ratio.new_tensor([.025, .975])).tolist()


def slope_metrics(pred, true, valid, frame, group):
    valid = valid & group
    true_small, pred_small = true.abs() <= SLOPE_FLOOR, pred.abs() <= SLOPE_FLOOR
    usable = valid & ~true_small & ~pred_small
    agree = (pred * true) > 0
    improve = usable & (pred > 0)
    m, n = pred[usable], true[usable]
    result = dict(valid=int(valid.sum()), informative=len(m),
                  newton_small=int((valid & true_small).sum()), wm_small=int((valid & pred_small).sum()),
                  wm_positive=int(improve.sum()), wm_positive_real_negative=int((improve & (true < 0)).sum()))
    if len(m):
        result.update(sign=float(agree[usable].float().mean()), sign_ci95=bootstrap_ci(agree, usable, frame),
                      cosine=float(torch.nn.functional.cosine_similarity(m, n, dim=0)),
                      correlation=float(torch.corrcoef(torch.stack((m,n)))[0,1])
                      if len(m) >= 2 and bool((torch.stack((m,n)).std(-1) > 0).all()) else None,
                      magnitude_mae=float((m.abs()-n.abs()).abs().mean()),
                      magnitude_rmse=float((m.abs()-n.abs()).square().mean().sqrt()),
                      magnitude_ratio=float(m.abs().mean()/n.abs().mean()))
    return result


def comparison_groups(samples, motion):
    time = samples['frame'] * CONTROL_DT
    lift, land = (float(x[0]) for x in (motion.liftoff_time, motion.touchdown_time))
    support = (samples['force'] >= CONTACT_FORCE_N).sum(-1)
    err = samples['foot_error']
    return dict(all=torch.ones_like(support, dtype=torch.bool),
                double_support=support == 2, single_support=support == 1,
                first_swing=(time >= lift-.000001) & (time < land-.000001),
                landing=(time-land).abs() <= .200001,
                high_action_change=samples['action_change'] >= torch.quantile(samples['action_change'], .9),
                near_clamp=samples['near_clamp'], err_below_08=err < .08,
                err_08_to_20=(err >= .08) & (err < .20),
                err_20_to_35=(err >= .20) & (err < .35), err_over_35=err >= .35)


@torch.no_grad()
def difference_batch(env, probe, samples, direction, scoring, epsilons):
    from .m4_dynamics import fault_masks

    n = len(samples['state'])
    count = len(direction)
    repeat = 2 * count + 1
    frame = samples['frame'].repeat(repeat)
    prior = samples['previous'].repeat(repeat, 1)
    result, replay = {}, None
    for eps in epsilons:
        before = restore(env, probe, samples, repeat)
        action = samples['action']
        command = torch.cat((action + eps * direction, action - eps * direction, action[None])).flatten(0, 1)
        after = env.advance_physics(env.pd_target(command))
        bad = torch.stack(fault_masks(before, after, probe, env.mdp)[:3]).any(0).view(repeat, n)
        valid = ~(bad[:count] | bad[count:2*count]) & (direction.square().sum(-1) > 0)
        result[eps] = {}
        for label, mdp in scoring:
            terms = scores(mdp, after, frame, command, prior)
            for name, value in terms.items():
                r = value.view(repeat, n)
                result[eps][f'{label}/{name}'] = ((r[:count] - r[count:2*count])/(2*eps), valid)
        if replay is None:
            foot = env.motion.feet(tuple(s[-n:] for s in after))[0] - samples['next_foot']
            replay = dict(foot_rmse_m=float(foot.square().sum(-1).mean().sqrt()),
                          uncertain_worlds=int(bad[-1].sum()), exact_quaternion='root_quat' in samples,
                          warmstart='warmstart' in samples)
    return result, replay


@torch.no_grad()
def true_update(env, probe, samples, pi, updated, horizon=8):
    from .m4_dynamics import fault_masks

    n = len(samples['state'])
    repeats = env.N // n
    base = restore(env, probe, samples, repeats)
    previous = samples['previous'].repeat(repeats, 1)
    hist = samples['history'].repeat(repeats, 1, 1)
    frame = samples['frame'].repeat(repeats)
    xi = samples['xi'].repeat(repeats, 1)
    anchor = env.motion.anchor(torch.zeros_like(frame))
    limit = env.motion.num_frames - 1
    start = frame.clone()
    start_height = env.motion.feet(base)[0][..., 2]
    alive = torch.ones(env.N, dtype=torch.bool, device=env.device)
    aggregate = {}
    mdp, goals, motion = env.mdp, env.goals, env.motion
    lift_gain = torch.zeros_like(start_height)
    swing_seen = torch.zeros_like(start_height, dtype=torch.bool)

    for step in range(horizon):
        g = goals.goal(base, frame, frame, anchor)
        command = samples['action'].repeat(repeats, 1)
        command[:n] = pi(hist[:n], g[:n])
        command[n:2*n] = updated(hist[n:2*n], g[n:2*n])
        successor = env.advance_physics(env.pd_target(command))
        bad = torch.stack(fault_masks(base, successor, probe, mdp)[:3]).any(0)
        if bad.any():
            # 仅在离线实验丢弃该方向，主训练的故障保护不受影响。
            return dict(invalid=int(bad.sum()), horizon=step+1)
        evaluate = alive & (start+step < limit)
        next_frame = (frame + 1).clamp_max(limit)
        obs = goals.obs(mdp, successor, command, xi, next_frame, anchor)
        hist = torch.cat((hist[:, 1:], obs[:, None]), 1)
        values = scores(mdp, successor, frame.clamp_max(limit - 1), command, previous)
        for name, value in values.items():
            aggregate.setdefault(name, torch.zeros_like(value)).add_(value * (env.gamma**step) * evaluate)
        swing = ~motion.contact[next_frame] & evaluate[:, None]
        swing_seen |= swing
        foot = motion.feet(successor)[0][..., 2]
        lift_gain = torch.maximum(lift_gain, torch.where(swing, foot-start_height, 0.))
        alive = evaluate & ~mdp.termination(successor)
        frame = next_frame
        previous, base = command, successor
    report = {}
    for name, value in aggregate.items():
        old, new = value[:n], value[n:2*n]
        report[name] = dict(before=float(old.mean()), after=float(new.mean()),
                            paired_improved=int((new > old+SLOPE_FLOOR).sum()))
    lift = [lift_gain[i*n:(i+1)*n][swing_seen[i*n:(i+1)*n]] for i in range(2)]
    report['mean_max_lift_m'] = [float(x.mean()) if len(x) else None for x in lift]
    return dict(horizon=horizon, starts=n, **report)


def compact_prediction(result):
    return {h: {phase: value for phase, value in rows.items() if phase in (
            'all', 'double_support', 'single_support', 'first_swing', 'first_touchdown',
            'clamped', 'unclamped', 'high_action_change', 'high_qd')}
            for h, rows in result['horizons'].items()}


def read_checkpoints(root, iterations):
    result = {}
    preferred = ('early.pt', 'iter_1000.pt', 'iter_2000.pt', 'iter_3000.pt', 'iter_5000.pt', 'last.pt')
    files = sorted(root.glob('*.pt'), key=lambda p: preferred.index(p.name) if p.name in preferred else len(preferred))
    for path in files:
        if path.name == 'initial.pt':
            continue
        ck = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        it = int(ck['iteration'])
        if it in iterations and it not in result:
            result[it] = ck
    return result


def make_policy(ck, mdp, goals, device):
    a = argparse.Namespace(**ck['args'])
    policy = Policy(mdp.obs_dim, goals.spec.dim, hist=a.hist, obs_adapter=a.obs_adapter).to(device).eval()
    policy.load_state_dict(ck['pi'])
    return policy


def make_wm(ck, device):
    model = WorldModel().to(device).eval().requires_grad_(False)
    model.load_state_dict(ck['wm'])
    return model


def write_json(path, report):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    try:
        tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def actor_diagnostics(checkpoint, task, model, record, iteration, device, physical):
    args, motion, goals, mdp = task
    policy = make_policy(checkpoint, mdp, goals, device)
    critic = Critic(mdp.critic_dim + goals.spec.dim).to(device).eval().requires_grad_(False)
    critic.load_state_dict(checkpoint['critic'])
    if not any(record[f'i{iteration}_s{seed}_action'].size for seed in record['seeds']):
        return None, policy, None
    with torch.no_grad():
        traces = [load_trace(record, iteration, int(seed), device) for seed in record['seeds']
                  if record[f'i{iteration}_s{seed}_action'].size]
        trajectories = [(row, episode_input(motion, mdp, goals, row, args.hist)) for row in traces]
        batch = sample_starts(trajectories, motion.num_frames-1, args.horizon, 64, 381, device)
    report = gradient_audit(batch, policy, critic, model, motion, mdp, goals, args, checkpoint, 381,
                            single_updates=True)
    if not physical:
        return report, policy, None
    value_upper = checkpoint['imagination_contract']['bootstrap_upper_bound']
    params = tuple(policy.parameters())
    graph = objective(batch, policy, critic, model, motion, mdp, goals, args, value_upper, 381)
    g = gradient(graph['A'], params)
    decay = (.03 + .97 * .5 * (1 + math.cos(math.pi * iteration / args.lr_decay_iters))
             if iteration < args.lr_decay_iters else .03)
    lr = (args.lr if args.actor_lr is None else args.actor_lr) * decay
    updated = update_policy(policy, checkpoint, params, g, lr, batch, critic, model, motion, mdp, goals,
                            args, value_upper, 381, return_policy=True)
    return report, policy, updated


def task_for(checkpoint, kin, reference, device):
    cfg = argparse.Namespace(**checkpoint['args'])
    cfg.device = str(device)
    motion, goals, mdp, _ = build_m4_task(cfg, kin, [str(reference)], device)
    mdp.assert_checkpoint_task(checkpoint)
    return cfg, motion, goals, mdp


@torch.no_grad()
def local_audit(task, samples, scoring, gradients, epsilons, physical=None):
    from isaaclab.app import launch_simulation

    from .m4_dynamics import SolverReadback
    from .m4_task import M4TrackingSim, make_m4_sim

    cfg, motion, goals, mdp = task
    # 公共数据上各模型使用同样的两个随机方向及一个固定参考 WM 的上升方向。
    anchor_gradient = next(iter(gradients.values()))[scoring[0][0]]['foot']
    direction = directions(samples['action'], mdp, anchor_gradient, max(epsilons))
    n_envs = (2*len(direction)+1) * len(samples['state'])
    sim = make_m4_sim(samples['state'].device, cfg)
    with launch_simulation(sim, cfg):
        env = M4TrackingSim(sim, n_envs, motion, mdp, eval_mode=True, goals=goals, gamma=cfg.gamma)
        probe = SolverReadback(env)
        differences, replay = difference_batch(env, probe, samples, direction, scoring, epsilons)
        update = true_update(env, probe, samples, *physical) if physical is not None else None
    reports = {}
    frame = samples['frame'][None]
    groups = comparison_groups(samples, motion)
    center = min(epsilons, key=lambda eps: abs(eps-.15))
    for model, reward_defs in gradients.items():
        grouped = {}
        for reward_name, terms in reward_defs.items():
            for name, grad in terms.items():
                key = f'{reward_name}/{name}'
                pred = (grad * direction).sum(-1)
                by_eps = {}
                for eps, series in differences.items():
                    truth, valid = series[key]
                    primary_truth, primary_valid = differences[center][key]
                    info = valid & primary_valid & (truth.abs() > SLOPE_FLOOR) & (primary_truth.abs() > SLOPE_FLOOR)
                    by_eps[str(eps)] = slope_metrics(pred, truth, valid, frame, groups['all'])
                    by_eps[str(eps)]['sign_flip_vs_center'] = int((info & (truth*primary_truth < 0)).sum())
                    by_eps[str(eps)]['epsilon_comparable'] = int(info.sum())
                truth, valid = differences[center][key]
                by_group = {label: slope_metrics(pred, truth, valid, frame, mask)
                            for label,mask in groups.items() if mask.any() and label != 'all'}
                ascent_true = truth[-1]
                ascent_valid = valid[-1] & (pred[-1] > SLOPE_FLOOR)
                grouped[key] = dict(epsilons=by_eps, groups_at_center=by_group,
                                    ascent_true_negative=int((ascent_valid & (ascent_true < -SLOPE_FLOOR)).sum()),
                                    ascent_true_positive=int((ascent_valid & (ascent_true > SLOPE_FLOOR)).sum()))
        reports[model] = grouped
    return dict(starts=len(samples['state']), seeds=samples['seed'].unique().tolist(),
                center_epsilon=center, nominal_replay=replay, models=reports, real_adam_update=update)


def summary_line(label, stats):
    selected = {name: item for name,item in stats.items() if name != 'models'}
    selected['scores'] = {}
    center = str(stats['center_epsilon'])
    for model, rows in stats['models'].items():
        selected['scores'][model] = {name: item['epsilons'][center] for name,item in rows.items()}
    print(label, json.dumps(selected, ensure_ascii=False, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m4_reference', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, default=Path('outputs/m4_3/checkpoints'))
    parser.add_argument('--baseline-checkpoint', type=Path,
                        default=Path('outputs/m4_2/m4_2_v2/checkpoints/last.pt'))
    parser.add_argument('--rollouts', type=Path, default=Path('outputs/m4_3/evaluation/rollouts.npz'))
    parser.add_argument('--public', type=Path, default=Path('outputs/m4_2/m4_2_v3/rollouts.npz'))
    parser.add_argument('--iterations', type=int, nargs='+', default=[1000, 2000, 3000, 5000])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--local-newton', action='store_true')
    parser.add_argument('--scope', choices=('all', 'own', 'public'), default='all',
                        help='同策略或公共数据分批审计时使用，默认同时检查两者')
    parser.add_argument('--baseline-only', action='store_true')
    parser.add_argument('--starts-per-seed', type=int, default=14)
    parser.add_argument('--epsilons', nargs='+', type=float, default=[.075,.15,.30])
    parser.add_argument('--output', type=Path)
    cli = parser.parse_args()
    device = torch.device(cli.device)
    if cli.local_newton and device.type != 'cuda':
        parser.error('--local-newton 需要 CUDA Newton')
    if cli.starts_per_seed < 1 or not all(math.isfinite(x) and x > 0 for x in cli.epsilons):
        parser.error('需要正的采样数量和扰动幅值')
    torch.set_num_threads(min(6,torch.get_num_threads()))
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True

    old_check = torch.load(cli.baseline_checkpoint, map_location='cpu', weights_only=False, mmap=True)
    stages = {} if cli.baseline_only else read_checkpoints(cli.checkpoints, cli.iterations)
    if not cli.baseline_only and not stages:
        parser.error('缺少匹配的新训练检查点')
    kin = RobotKinematics(device)
    old_task = task_for(old_check, kin, cli.m4_reference, device)
    new_task = task_for(next(iter(stages.values())), kin, cli.m4_reference, device) if stages else None
    scoring = [('m42_reward', old_task[3])]
    if new_task is not None:
        scoring.append(('m43_reward', new_task[3]))
    for it, ck in stages.items():
        if ck['physical_contract'] != old_check['physical_contract']:
            raise ValueError(f'第 {it} 轮的动力学契约与 M4.2 基线不一致')
        new_task[3].assert_checkpoint_task(ck)
    epsilons = sorted(set(cli.epsilons))
    report = dict(main_epsilons=epsilons, slope_floor=SLOPE_FLOOR,
                  intervals='95%控制帧整簇自举，400次；同一F2确定性初态',
                  prediction={}, local_own={}, bptt={})
    public_check = cli.local_newton and cli.scope != 'own'
    with ExitStack() as stack:
        pub = stack.enter_context(np.load(cli.public, allow_pickle=False))
        new = stack.enter_context(np.load(cli.rollouts, allow_pickle=False)) if stages else None
        pub_window = select_windows(pub, int(old_check['iteration']), pub['seeds'], old_task[0].hist, device)
        old_model = make_wm(old_check, device)
        public_grad = {}
        if public_check:
            archive = new if new is not None and f"baseline_s{new['seeds'][0]}_state" in new else pub
            prefix = 'baseline' if archive is new else f"i{old_check['iteration']}"
            public_samples = collect_samples(archive, prefix, old_task[1], old_task[3], cli.starts_per_seed, device)
            public_grad['m42_5000'] = model_slopes(old_model, scoring, public_samples)

        for model_name, checkpoint in [('m42_5000',old_check)]+[(f'm43_{i}',stages[i]) for i in sorted(stages)]:
            iteration = int(checkpoint['iteration'])
            model = old_model if model_name == 'm42_5000' else make_wm(checkpoint,device)
            result = {'public':{}}
            for name, mdp in scoring:
                with torch.no_grad():
                    pred = evaluate_windows(model, mdp, old_task[1], old_task[2], pub_window, old_task[0].gamma,
                                            rescore=name != 'm42_reward', measure_lift=True)
                result['public'][name] = compact_prediction(pred)
            result['public']['action_distribution'] = normalizer_summary(model, old_task[3], pub_window)
            report['prediction'][model_name] = result
            if public_check and model is not old_model:
                public_grad[model_name] = model_slopes(model, scoring, public_samples)
            if model is old_model or cli.scope == 'public' or iteration not in new['iterations']:
                continue
            args, motion, goals, mdp = new_task
            own = select_windows(new, iteration, new['seeds'], args.hist, device)
            if not own:
                continue
            bptt, policy, updated = actor_diagnostics(checkpoint,new_task,model,new,iteration,device,cli.local_newton)
            report['bptt'][model_name] = bptt
            with torch.no_grad():
                result['own'] = {name: compact_prediction(evaluate_windows(
                    model, task_mdp, motion, goals, own, args.gamma, rescore=name != 'm43_reward',
                    measure_lift=True)) for name, task_mdp in scoring}
                result['own_closed'] = compact_prediction(evaluate_windows(model,mdp,motion,goals,own,args.gamma,
                                                                           policy,measure_lift=True))
            result['own']['action_distribution'] = normalizer_summary(model,mdp,own)
            print('WM 阶段', iteration, json.dumps(dict(
                foot_rmse_m=[result['own']['m43_reward'][h]['all']['foot_m'] for h in (1,4,8)],
                public_foot_rmse_m=[result['public']['m43_reward'][h]['all']['foot_m'] for h in (1,4,8)],
                bptt_terms={k:bptt['terms'][k]['norm'] for k in ('swing_position','swing_clearance','touchdown')
                            if k in bptt['terms']}), ensure_ascii=False, allow_nan=False), flush=True)
            if cli.local_newton:
                own_samples = collect_samples(new,f'i{iteration}',motion,mdp,cli.starts_per_seed,device)
                own_grad = {model_name:model_slopes(model,scoring,own_samples)}
                phys = (policy,updated) if updated is not None else None
                own_local = local_audit(new_task,own_samples,scoring,own_grad,epsilons,phys)
                report['local_own'][model_name] = own_local
                summary_line(f'Newton 阶段 {iteration}', own_local)
            write_json(cli.output, report)
        if public_check:
            report['local_public'] = local_audit(old_task,public_samples,scoring,public_grad,epsilons)
            summary_line('Newton 公共', report['local_public'])
        write_json(cli.output,report)


if __name__ == '__main__':
    main()
