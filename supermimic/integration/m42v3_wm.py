"""用冻结 Newton 轨迹检查 Roban F2 世界模型及 Actor 闭环。"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

if __name__ == '__main__':
    os.environ['SUPERMIMIC_ROBOT'] = 'roban'

import numpy as np
import torch

from ..kinematics import RobotKinematics
from ..mdp import INV_DIM, NUM_J, normalize_xi, unflatten_state
from ..motion import CONTROL_DT
from ..networks import Policy, WorldModel
from ..rotations import so3_log
from .m4_task import build_m4_task


HORIZONS = (1, 4, 8)
CONTACT_FORCE_N = 10.


def scalar_metrics(x):
    magnitude = x.abs().flatten()
    p95, p99 = torch.quantile(magnitude, magnitude.new_tensor([.95, .99])).tolist()
    return dict(mean=float(x.mean()), std=float(x.std(unbiased=False)), abs_p95=p95,
                abs_p99=p99, abs_max=float(magnitude.max()))


def select_windows(record, iteration, seeds, history_size, device):
    columns = {}
    H = HORIZONS[-1]
    for seed in seeds:
        prefix = f'i{iteration}_s{seed}_'

        def read(key, dtype=None):
            return torch.as_tensor(record[prefix + key], dtype=dtype, device=device)

        frame = read('frame', torch.long)
        state, obs, goal = read('state'), read('observation'), read('goal')
        action, xi, force = read('action'), read('xi'), read('force')
        raw, clip = read('raw_action'), read('clamp_mask', torch.bool)
        reward, term, timeout = read('reward'), read('term', torch.bool), read('timeout', torch.bool)
        if len(action) < H:
            continue
        torch.testing.assert_close(frame, torch.arange(len(state), device=device))
        if int(read('command_start_frame')) != 0 or int(read('command_clip_id')) != 0:
            raise ValueError('仅能比较从同一 F2 首帧起步的轨迹')
        torch.testing.assert_close(read('command_rotation'), torch.eye(3, device=device))
        torch.testing.assert_close(read('command_translation'), torch.zeros(3, device=device))
        start = torch.arange(len(action) - H + 1, device=device)
        path = start[:, None] + torch.arange(H, device=device)
        history = (start[:, None] + torch.arange(1 - history_size, 1, device=device)).clamp_min(0)
        # 历史在第 0 帧补齐。电机动作 t 与步后状态和参考帧 t+1 配对。
        batch = dict(start=state[start], history=obs[history], goal=goal[start], frame=frame[path],
                     next_state=state[path + 1], next_foot=read('foot')[path + 1],
                     xi=xi[path], raw=raw[path], action=action[path], prior=read('previous_action')[path],
                     clamped=clip[path], reward=reward[path], term=term[path], timeout=timeout[path],
                     force=force[start, :, 2], reset=(start == 0))
        for key, value in batch.items():
            columns.setdefault(key, []).append(value)
    if not columns:
        raise ValueError(f'第 {iteration} 轮没有达到 {H} 步的有效 Newton 轨迹')
    return {key: torch.cat(value) for key, value in columns.items()}


def ranges(batch, motion):
    qd = unflatten_state(batch['start'])[-1]
    speed = qd.square().mean(-1).sqrt()
    change = (batch['action'][:, 0] - batch['prior'][:, 0]).square().mean(-1).sqrt()
    support = (batch['force'] >= CONTACT_FORCE_N).sum(-1)
    clip = batch['clamped'][:, 0].any(-1)
    speed_p90 = torch.quantile(speed, .9)
    change_p90 = torch.quantile(change, .9)
    frame = batch['frame'][:, 0]
    lift = round(float(motion.liftoff_time[0]) / CONTROL_DT)
    land = round(float(motion.touchdown_time[0]) / CONTROL_DT)
    # 参考事件以整数控制帧划分，避免 0.8 s 边界被浮点舍入移入上一相位。
    landing_window = round(.2 / CONTROL_DT)
    groups = dict(all=torch.ones_like(clip), clamped=clip, unclamped=~clip,
                  double_support=support == 2, single_support=support == 1,
                  no_contact=(support == 0) & ~batch['reset'], high_qd=speed >= speed_p90,
                  high_action_change=change >= change_p90, pre_swing=frame < lift,
                  first_swing=(frame >= lift) & (frame < land), first_touchdown=(frame - land).abs() <= landing_window)
    return groups, dict(high_qd_rms_p90=float(speed_p90), high_action_change_rms_p90=float(change_p90))


def evaluate_windows(model, mdp, motion, goals, batch, gamma, policy=None, *, rescore=False, measure_lift=False):
    groups, thresholds = ranges(batch, motion)
    state = unflatten_state(batch['start'])
    frame, history = batch['frame'], batch['history']
    n, horizon = frame.shape
    command = motion.anchor(torch.zeros(n, dtype=torch.long, device=frame.device))
    previous = batch['prior'][:, 0]
    real_weight = pred_weight = torch.ones_like(previous[:, 0])
    sum_real = torch.zeros_like(real_weight)
    sum_pred = torch.zeros_like(real_weight)
    report = {}
    max_fk_error = first_action_error = real_reward_error = 0.
    discount = gamma ** torch.arange(horizon, dtype=previous.dtype, device=frame.device)
    floor = model.out_norm.mean.new_tensor([.003] * 3 + [.005] * 3 + [.02] * 3 + [.05] * 3 +
                                          [.003] * NUM_J + [.05] * NUM_J)
    normalizer = torch.maximum(model.out_norm.std(), floor)
    diverged = []
    start_foot = motion.feet(state)[0] if measure_lift else None

    for k in range(horizon):
        f = frame[:, k]
        true_action = batch['action'][:, k]
        xi = batch['xi'][:, k]
        if policy is None:
            action = true_action
            previous = batch['prior'][:, k]
        else:
            g = batch['goal'] if k == 0 else goals.goal(state, f, f, command)
            action = policy(history, g)
            if k == 0:
                first_action_error = float((action - true_action).abs().max())
        state = model.step(state, mdp.wm_action(action, f), normalize_xi(xi))
        if policy is not None and k + 1 < horizon:
            o = goals.obs(mdp, state, action, xi, f + 1, command)
            history = torch.cat((history[:, 1:], o[:, None]), 1)
        true = unflatten_state(batch['next_state'][:, k])
        ref_end = f + 1 >= motion.num_frames - 1
        estimated_reward, estimated_terms = mdp.reward(state, f + 1, action, previous, terms=True,
                                                       command=command, ep_step=f + 1)
        true_reward = batch['reward'][:, k]
        reward_true_check, true_terms = mdp.reward(true, f + 1, true_action, batch['prior'][:, k],
                                                 terms=True, command=command, ep_step=f + 1)
        # 公共轨迹可用另一套奖励重算，状态及下发动作保持原样。
        if rescore:
            true_reward = reward_true_check
        real_reward_error = max(real_reward_error, float((reward_true_check - true_reward).abs().max()))
        previous = action
        estimate_soft = torch.where(ref_end, 1., mdp.soft_termination(state))
        true_soft = torch.where(ref_end, 1., mdp.soft_termination(true))
        sum_pred += discount[k] * pred_weight * estimated_reward
        sum_real += discount[k] * real_weight * true_reward
        pred_weight *= 1 - estimate_soft
        real_weight *= 1 - true_soft
        diverged.append((action - true_action).square().mean(-1))
        if k + 1 not in HORIZONS:
            continue
        pred_foot = motion.feet(state)[0]
        true_foot = batch['next_foot'][:, k]
        max_fk_error = max(max_fk_error, float((motion.feet(true)[0] - true_foot).norm(dim=-1).max()))
        error = model.state_error(state, true)
        pred_term = mdp.termination(state) | ref_end
        true_term = batch['term'][:, k] | batch['timeout'][:, k]
        foot_bias = (estimated_terms['foot_objective'] - true_terms['foot_objective']) / mdp.max_step_reward
        values = dict(state_mse=(error / normalizer).square().mean(-1),
                      q_rad=(state[4] - true[4]).square().mean(-1),
                      qd_rad_s=(state[5] - true[5]).square().mean(-1),
                      foot_m=(pred_foot - true_foot).square().sum(-1).mean(-1),
                      root_m=(state[0] - true[0]).square().sum(-1),
                      tilt_deg=so3_log(true[1].transpose(-1, -2) @ state[1]).norm(dim=-1).rad2deg().square(),
                      reward_bias=(estimated_reward - true_reward) / mdp.max_step_reward,
                      foot_reward_bias=foot_bias,
                      return_bias=(sum_pred - sum_real) / (mdp.max_step_reward * discount[:k+1].sum()),
                      raw_action_rmse=torch.stack(diverged, 0).mean(0),
                      soft_termination_brier=(estimate_soft - true_term.float()).square())
        step = {}
        for key, mask in groups.items():
            if not bool(mask.any()):
                continue
            values_mean = {label: float(v[mask].mean()) for label, v in values.items()}
            for label in ('q_rad', 'qd_rad_s', 'foot_m', 'root_m', 'tilt_deg', 'raw_action_rmse'):
                values_mean[label] = values_mean[label] ** .5
            values_mean.update(reward_mae=float(values['reward_bias'][mask].abs().mean()),
                               terms=int((mask & true_term).sum()),
                               false_positive=int((mask & pred_term & ~true_term).sum()),
                               false_negative=int((mask & ~pred_term & true_term).sum()),
                               n=int(mask.sum()))
            if measure_lift:
                swing = (~motion.contact[f + 1]) & motion.contact_valid[f + 1] & mask[:, None]
                if swing.any():
                    values_mean.update(predicted_swing_lift_m=float((pred_foot[..., 2] - start_foot[..., 2])[swing].mean()),
                                       real_swing_lift_m=float((true_foot[..., 2] - start_foot[..., 2])[swing].mean()))
            step[key] = values_mean
        report[k + 1] = step
    return dict(horizons=report, thresholds=thresholds, fk_error_max_m=max_fk_error,
                true_reward_error_max=real_reward_error, first_action_error_max=first_action_error)


def normalizer_summary(model, mdp, batch):
    raw = batch['raw'][:, 0]
    action = batch['action'][:, 0]
    block = slice(INV_DIM, INV_DIM + NUM_J)
    mean, std = model.in_norm.mean[block], model.in_norm.std()[block]
    a_eff = mdp.wm_action(action, batch['frame'][:, 0])
    return dict(raw=scalar_metrics(raw), effective=scalar_metrics(a_eff),
                normalizer_count=int(model.in_norm.count),
                normalizer_joint_mean_range=[float(mean.min()), float(mean.max())],
                normalizer_std_range=[float(std.min()), float(std.max())],
                effective_abs_z=scalar_metrics(((a_eff - mean) / std).abs()),
                clipped=float(batch['clamped'][:, 0].float().mean()))


@torch.enable_grad()
def action_directions(model, mdp, motion, state, frame, xi, actions, prior, epsilon):
    a = actions.detach().requires_grad_(True)
    anchor = motion.anchor(torch.zeros_like(frame))
    future = model.step(state, mdp.wm_action(a, frame), normalize_xi(xi))
    _, terms = mdp.reward(future, frame + 1, a, prior, terms=True, command=anchor, ep_step=frame + 1)
    grad, = torch.autograd.grad(terms['foot_objective'].sum() / mdp.max_step_reward, a)
    # 两侧扰动均留在夹紧边界内，避免将截断的导数误当作碰撞响应。
    proposed = mdp.default_q + mdp.alpha * actions
    interior = (proposed > mdp.kin.q_lo + mdp.alpha * epsilon) & (proposed < mdp.kin.q_hi - mdp.alpha * epsilon)
    rng = torch.Generator(device=actions.device).manual_seed(4203)
    random = torch.randn((2, *actions.shape), device=actions.device, generator=rng)
    direction = torch.nn.functional.normalize(torch.cat((grad[None], random), 0) * interior, dim=-1).detach()
    return direction, (grad * direction).sum(-1).detach()


def local_newton(model, mdp, motion, goals, batch, config):
    from isaaclab.app import launch_simulation

    from ..rotations import mat_to_quat
    from .m4_dynamics import SolverReadback, fault_masks
    from .m4_task import M4TrackingSim, make_m4_sim

    # 三组固定种子回放通常重合。局部响应只从第一组挑选沿 F2 均匀覆盖的状态。
    dev = model.in_norm.mean.device
    starts = batch['reset'].nonzero().flatten()
    count = int(starts[1]) if len(starts) > 1 else len(batch['start'])
    idx = torch.linspace(0, count - 1, min(count, 32), device=dev).round().long()
    n = len(idx)
    epsilon = .15
    frame, xi = batch['frame'][idx, 0], batch['xi'][idx, 0]
    a, prior = batch['action'][idx, 0], batch['prior'][idx, 0]
    state = unflatten_state(batch['start'][idx])
    d, wm = action_directions(model, mdp, motion, state, frame, xi, a, prior, epsilon)
    ndir = len(d)
    actions = torch.cat((a[None] + epsilon * d, a[None] - epsilon * d, a[None]), 0).flatten(0, 1)
    worlds = (2 * ndir + 1) * n
    repeated = tuple(s.repeat((2 * ndir + 1,) + (1,) * (s.ndim - 1)) for s in state)
    p, R, v, w, q, qd = repeated
    sim_cfg = make_m4_sim(dev, config)
    with launch_simulation(sim_cfg, config):
        env = M4TrackingSim(sim_cfg, worlds, motion, mdp, goals=goals, eval_mode=True)
        env_ids = torch.arange(worlds, device=dev)
        env.reset(env_ids)
        # 两侧世界从同一实测状态重新起步，不复制原轨迹的求解器暖启动。
        robot = env.robot
        root_pose = torch.cat((p + env.scene.env_origins, mat_to_quat(R)[:, [1, 2, 3, 0]]), -1)
        robot.write_root_link_pose_to_sim_index(root_pose=root_pose)
        robot.write_root_link_velocity_to_sim_index(root_velocity=torch.cat((v, w), -1))
        robot.write_joint_position_to_sim_index(position=q[:, env.sim_to_canonical])
        robot.write_joint_velocity_to_sim_index(velocity=qd[:, env.sim_to_canonical])
        env.scene.write_data_to_sim()
        env.refresh()
        before = env.state()
        probe = SolverReadback(env)
        after = env.advance_physics(env.pd_target(actions))
        bad = torch.stack(fault_masks(before, after, probe, mdp)[:3]).any(0)
        f = frame.repeat(2 * ndir + 1)
        command = motion.anchor(torch.zeros_like(f))
        _, terms = mdp.reward(after, f + 1, actions, prior.repeat(2 * ndir + 1, 1), terms=True,
                              command=command, ep_step=f + 1)
    rewards = terms['foot_objective'].reshape(2 * ndir + 1, n) / mdp.max_step_reward
    truth = (rewards[:ndir] - rewards[ndir:2*ndir]) / (2 * epsilon)
    bad = bad.reshape(2 * ndir + 1, n)
    valid = ~(bad[:ndir] | bad[ndir:2*ndir]) & (d.square().sum(-1) > 0.)
    meaningful = valid & (truth.abs() > 1e-5) & (wm.abs() > 1e-5)
    nominal = tuple(s[-n:] for s in after)
    real = unflatten_state(batch['next_state'][idx, 0])
    foot = motion.feet(nominal)[0] - motion.feet(real)[0]
    both = torch.stack((wm[valid], truth[valid]))
    return dict(n=n, perturbation=epsilon, included_pairs=int(valid.sum()), informative_pairs=int(meaningful.sum()),
                direction_agreement=float((wm[meaningful] * truth[meaningful] > 0).float().mean()) if meaningful.any() else None,
                correlation=float(torch.corrcoef(both)[0, 1]) if both.shape[1] > 1 and bool((both.std(1) > 0).all()) else None,
                wm_ascent_true_positive=int((valid[0] & (truth[0] > 1e-5)).sum()),
                wm_ascent_true_negative=int((valid[0] & (truth[0] < -1e-5)).sum()),
                wm_grad_abs_mean=float(wm[valid].abs().mean()), newton_grad_abs_mean=float(truth[valid].abs().mean()),
                nominal_foot_rmse_m=float(foot.square().sum(-1).mean().sqrt()))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='训练输出目录，包含 checkpoints')
    parser.add_argument('--rollouts', type=Path, required=True, help='同一任务的冻结 Newton 轨迹 NPZ')
    parser.add_argument('--reference', type=Path, required=True, help='原始 F2 参考文件')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--local-newton', action='store_true', help='在 CUDA Newton 中检查单步足部奖励动作方向')
    args = parser.parse_args()
    if args.device == 'cpu':
        torch.set_num_threads(min(8, torch.get_num_threads()))
    checks = {}
    for name in ('early', 'last', 'best'):
        path = args.run / 'checkpoints' / f'{name}.pt'
        if path.is_file():
            check = torch.load(path, map_location='cpu', weights_only=False)
            checks.setdefault(check['iteration'], check)
    if not checks:
        raise ValueError('缺少可验证检查点')
    config = SimpleNamespace(**next(iter(checks.values()))['args'])
    motion, goals, mdp, terrain = build_m4_task(config, RobotKinematics(args.device),
                                                [str(args.reference)], args.device)
    with np.load(args.rollouts, allow_pickle=False) as record:
        seeds = record['seeds'].tolist()
        windows = {it: select_windows(record, it, seeds, config.hist, args.device)
                   for it in record['iterations'].tolist() if it in checks}
    if not windows:
        raise ValueError('冻结轨迹轮次与检查点不匹配')
    for wm_it, check in checks.items():
        if wm_it not in windows:
            continue
        model = WorldModel(terrain_provider=terrain).to(args.device).eval()
        model.load_state_dict(check['wm'])
        if args.local_newton:
            if not args.device.startswith('cuda'):
                parser.error('--local-newton 需要 CUDA 设备')
            if wm_it == max(windows):
                launches = argparse.Namespace(**check['args'])
                launches.device = args.device
                result = local_newton(model.requires_grad_(False), mdp, motion, goals, windows[wm_it], launches)
                print('Newton 方向', json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
            continue
        for data_it, batch in windows.items():
            result = evaluate_windows(model, mdp, motion, goals, batch, config.gamma)
            norm = normalizer_summary(model, mdp, batch)
            print('WM', json.dumps(dict(model_iteration=wm_it, data_iteration=data_it,
                                        normalization=norm, **result), ensure_ascii=False, allow_nan=False),
                  flush=True)
            if wm_it == data_it:
                policy = Policy(mdp.obs_dim, goals.spec.dim, hist=config.hist, obs_adapter=config.obs_adapter,
                                squash_action=False).to(args.device).eval()
                policy.load_state_dict(check['pi'])
                result = evaluate_windows(model, mdp, motion, goals, batch, config.gamma, policy)
                print('闭环', json.dumps(dict(model_iteration=wm_it, **result),
                                          ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
