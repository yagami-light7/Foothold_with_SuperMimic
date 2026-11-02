"""课程固定点评估：物理判据与preflight一致，逐步聚合留在GPU，最后才读取标量。

完整奖励/相位/观测契约诊断仍使用preflight_signals默认模式；此处不重复契约断言或逐帧trace。
"""

import time
import torch

from tasks.foot_placement.first_step_metrics import summarize_first_steps
from tasks.foot_placement.heading import heading_error
from tasks.foot_placement.rewards import normalized_episode_return


@torch.inference_mode()
def evaluate_walk(env, wrapped, policy, steps):
    n, device = env.num_envs, env.device
    totals = {}

    def add(key, value):
        value = value.float().sum()
        if key not in totals:
            totals[key] = torch.zeros((), device=device)
        totals[key] += value

    alive = torch.ones(n, dtype=torch.bool, device=device)
    air, touchdown, support = torch.zeros_like(alive), torch.zeros_like(alive), torch.ones_like(alive)
    streak = torch.zeros(n, dtype=torch.long, device=device)
    best = streak.clone()
    cycle_counts = torch.zeros(2, device=device)
    successes = torch.zeros_like(cycle_counts)
    start_counts = torch.zeros(2, 2, device=device)
    start_successes = torch.zeros_like(start_counts)
    max_heading = torch.zeros((), device=device)
    episode_reward = torch.zeros(n, device=device)
    episode_reference = torch.zeros_like(episode_reward)
    episode_steps = torch.zeros_like(streak)
    first_sums, first_counts = {}, {}
    initial_counts = env.goal_sampler.initial_left_is_swing.sum().item()
    obs = wrapped.get_observations()
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)
    started = time.monotonic()
    finite = torch.ones((), dtype=torch.bool, device=device)
    for _ in range(steps):
        s = env.goal_sampler
        swing, phase = s.left_is_swing.clone(), s.phase.clone()
        first = s.completed_half_cycles == 0
        started_left, reference = s.initial_left_is_swing.clone(), s.reference_heading_w.clone()
        lt, rt = s.left_target_pos_w.clone(), s.right_target_pos_w.clone()
        add("nominal_speed", 2 * s.gait_frequencies * s.step_lengths)
        actions = policy(obs)
        obs, rewards, dones, _ = wrapped.step(actions)
        finite &= torch.isfinite(rewards).all() & torch.isfinite(obs['policy']).all()
        valid = ~dones.bool()
        alive &= ~env.reset_terminated
        add("terminated", env.reset_terminated)
        add("truncated", env.reset_time_outs)
        add("valid", valid)
        add("reward", rewards)
        add("normalized", (rewards / env.reward_reference.clamp_min(1e-8)).clamp(0, 1))
        episode_reward += rewards
        episode_reference += env.reward_reference
        episode_steps += 1
        add("normalized_return", normalized_episode_return(episode_reward, episode_reference,
                                                            episode_steps, env.max_episode_length) * dones.bool())
        add("episodes", dones)
        episode_reward[dones.bool()] = 0
        episode_reference[dones.bool()] = 0
        episode_steps[dones.bool()] = 0
        for key, value in env.extras.get('log', {}).items():
            if key.startswith('Startup/'):
                first_sums[key] = first_sums.get(key, torch.zeros((), device=device)) + value.sum()
                first_counts[key] = first_counts.get(key, 0) + value.numel()
        for name, value in env.reward_terms.items():
            add("reward/" + name, value * env.step_dt)
        add("upright", -env.robot.data.projected_gravity_b.torch[:, 2])
        error_heading = heading_error(env.robot.data.root_quat_w.torch, reference).abs()
        max_heading = torch.maximum(max_heading, torch.where(valid, error_heading, 0).max())
        velocity = env.robot.data.root_lin_vel_w.torch
        add("forward", (reference.cos() * velocity[:, 0] + reference.sin() * velocity[:, 1]) * valid)
        lp, _, rp, _ = env._get_foot_poses()
        lf, rf = env._get_ground_contact_forces()
        swing_pos, target = torch.where(swing[:, None], lp, rp), torch.where(swing[:, None], lt, rt)
        local = torch.where(swing, 2 * phase, 2 * phase - 1)
        swing_fz, stance_fz = torch.where(swing, lf[:, 2], rf[:, 2]), torch.where(swing, rf[:, 2], lf[:, 2])
        xy = (swing_pos[:, :2] - target[:, :2]).norm(dim=-1)
        lift = swing_pos[:, 2] - target[:, 2]
        middle, late = valid & (local > .2) & (local < .8), valid & (local >= .8)
        cfg = env.reward_cfg
        air |= middle & (swing_fz < cfg.contact_threshold) & (lift > max(.01, .6 * cfg.swing_height))
        touchdown |= late & (swing_fz > cfg.contact_threshold) & (xy < .03)
        support &= ~middle | (stance_fz > cfg.contact_threshold)
        contact = torch.stack((lf[:, 2], rf[:, 2]), dim=1) > cfg.contact_threshold
        add("both_air", ~contact.any(-1) & valid)
        interior = valid & (local > .1) & (local < .9)
        stance_height = torch.where(swing, rp[:, 2] - rt[:, 2], lp[:, 2] - lt[:, 2])
        add("hop", interior & ((stance_fz <= cfg.contact_threshold)
                                | (stance_height > getattr(cfg, 'support_height_tolerance', .025))))
        add("interior", interior)
        state = env.flat_reward_state()
        speeds = torch.stack([state[side + '_velocity'][:, :2].norm(dim=-1)
                            for side in ('left', 'right')], dim=1)
        contacting = contact & valid[:, None]
        add("slip", speeds * contacting)
        add("contacts", contacting)
        add("late_xy", xy * late)
        add("late", late)
        add("post_start_xy", xy * late * ~first)
        add("post_start_samples", late & ~first)
        switched = (swing != s.left_is_swing) & valid
        ok = air & touchdown & support
        streak = torch.where(switched, torch.where(ok, streak + 1, 0), streak)
        best = torch.maximum(best, streak)
        streak[~valid] = 0
        for side, mask in enumerate((swing, ~swing)):
            ended = switched & mask
            cycle_counts[side] += ended.sum()
            successes[side] += (ended & ok).sum()
            for start, start_mask in enumerate((started_left, ~started_left)):
                start_counts[start, side] += (ended & start_mask).sum()
                start_successes[start, side] += (ended & start_mask & ok).sum()
        clear = switched | ~valid
        air[clear], touchdown[clear], support[clear] = False, False, True

    if not finite.item():
        raise RuntimeError("评估出现非有限奖励或观测")
    rollout_seconds = time.monotonic() - started
    values = {key: value.item() for key, value in totals.items()}
    scale = n * steps
    result = dict(
        evaluation_mode="fast_physical", num_envs=n, steps=steps, observation_dim=obs['policy'].shape[-1],
        rollout_wall_seconds=rollout_seconds,
        initial_swing_side=env.cfg.initial_swing_side,
        initial_start_counts_left_right=[int(initial_counts), n - int(initial_counts)],
        reward_profile=getattr(env.reward_cfg, 'profile', 'legacy'),
        mean_reward=values['reward'] / scale, normalized_reward=values['normalized'] / scale,
        normalized_return=values['normalized_return'] / values['episodes'] if values['episodes'] else None,
        normalized_return_completed_episodes=int(values['episodes']),
        terminated=int(values['terminated']), truncated=int(values['truncated']),
        survival_fraction=alive.float().mean().item(), mean_upright=values['upright'] / scale,
        max_abs_heading_error_rad=max_heading.item(),
        mean_reference_forward_velocity=values['forward'] / max(values['valid'], 1),
        command_nominal_forward_speed_m_s=values['nominal_speed'] / scale,
        both_air_fraction=values['both_air'] / max(values['valid'], 1),
        stance_hop_fraction=values['hop'] / max(values['interior'], 1),
        contact_foot_slip_speed_m_s=values['slip'] / max(values['contacts'], 1),
        late_xy_error_m=values['late_xy'] / max(values['late'], 1),
        late_xy_samples=int(values['late']),
        post_start_late_xy_error_m=values['post_start_xy'] / max(values['post_start_samples'], 1),
        half_cycle_success_left_right=(successes / cycle_counts.clamp_min(1)).tolist(),
        completed_half_cycles_left_right=cycle_counts.long().tolist(),
        consecutive_8_success_fraction=((best >= 8) & alive).float().mean().item(),
        by_initial_swing={side: dict(completed_half_cycles_left_right=start_counts[i].long().tolist(),
            half_cycle_success_left_right=(start_successes[i] / start_counts[i].clamp_min(1)).tolist())
            for i, side in enumerate(('left', 'right')) if start_counts[i].sum().item() > 0},
        first_step=summarize_first_steps({k: v.item() for k, v in first_sums.items()}, first_counts),
        reward_components={k.split('/', 1)[1]: v / scale for k, v in values.items() if k.startswith('reward/')})
    return result
