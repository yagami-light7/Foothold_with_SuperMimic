"""Reproducible rollout, timing and reward diagnostics (no optimizer updates)."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.gait_cli import add_gait_arguments, apply_gait_overrides

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=["zero", "random", "lift", "policy", "policy_sampled"], default="zero")
parser.add_argument("--stage", default="legacy")
parser.add_argument("--ablation", choices=["none", "legacy_reward", "legacy_scale", "no_smoothing", "targets_only", "rewards_only"], default="none")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--std", type=float, default=0.35)
parser.add_argument("--knee", type=float, default=0.1)
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--step_length_range", type=float, nargs=2, metavar=("MIN_M", "MAX_M"))
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--task-profile", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--trace-reward-terms", action="store_true",
                    help="在本地逐步trace中记录奖励分项和本步相位，用于比较起步与稳态；默认关闭")
parser.add_argument("--fast", action="store_true", help="快速确定性物理评估；不重复契约断言或生成逐帧trace")
add_gait_arguments(parser)
add_launcher_args(parser)
parser.set_defaults(device="cuda:0")
args = parser.parse_args()

import torch
from isaaclab.utils import to_dict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, check_rsl_rl_version, handle_deprecated_rsl_rl_cfg
from agents.rsl_rl_ppo_cfg import S54FootPlacementPPORunnerCfg
from tasks.foot_placement.foot_placement_env import S54FootPlacementEnv
from tasks.foot_placement.foot_placement_env_cfg import S54FootPlacementEnvCfg
from tasks.foot_placement.rewards import compute_foot_placement_reward, normalized_episode_return
from tasks.foot_placement.heading import heading_error, yaw_from_quat
from tasks.foot_placement.gait import advance_phase
from agents.checkpoint_compat import checkpoint_task_config, load_s54_checkpoint
from configs.s54_flat_reward import FlatWalkRewardCfg


def main():
    if args.steps < 1 or args.num_envs < 1:
        raise ValueError("steps and num_envs must be positive")
    cfg = S54FootPlacementEnvCfg()
    agent = S54FootPlacementPPORunnerCfg()
    if args.stage != "legacy":
        from configs.s54_curriculum import configure_stage
        configure_stage(cfg, agent, args.stage, args.ablation)
    if args.mode.startswith("policy") and args.checkpoint is not None and args.ablation == "none":
        data = checkpoint_task_config(args.checkpoint)
        if data.get("curriculum_stage", args.stage) == args.stage:
            from configs.s54_curriculum import restore_task_config
            # 独立评估使用实际训练参数，不能被profile默认的3cm覆盖。
            restore_task_config(cfg, data)
    from configs.walk_training import apply_task_profile
    apply_task_profile(cfg, args.task_profile)
    apply_gait_overrides(cfg, args)
    if args.step_length_range is not None:
        low, high = args.step_length_range
        if not (math.isfinite(low) and math.isfinite(high) and 0 <= low <= high):
            raise ValueError("水平步长范围必须有限且0 <= MIN_M <= MAX_M")
        cfg.step_length_range = (low, high)
        cfg.command_groups = None
        cfg.command_resample_half_cycles = 0
    cfg.seed = agent.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = agent.device = args.device
    if args.mode == "lift" and args.stage == "legacy":
        cfg.action_scale = max(args.knee, 0.1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with launch_simulation(cfg.sim, args):
        env = S54FootPlacementEnv(cfg)
        try:
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
            policy = None
            if args.mode.startswith("policy"):
                if args.checkpoint is None:
                    raise ValueError("policy modes require --checkpoint")
                runner = create_rsl_rl_runner(wrapped, handle_deprecated_rsl_rl_cfg(agent, check_rsl_rl_version()))
                load_s54_checkpoint(runner, args.checkpoint, map_location=args.device)
                policy = runner.get_inference_policy(device=str(env.device))
            if args.fast:
                if args.mode != 'policy' or policy is None or args.trace_reward_terms:
                    raise ValueError('--fast仅支持确定性policy，完整trace请使用默认模式')
                from scripts.quick_walk_eval import evaluate_walk
                result = evaluate_walk(env, wrapped, policy, args.steps)
                result.update(env_cfg=to_dict(cfg), agent_cfg=to_dict(agent), args=vars(args))
                args.output.write_text(json.dumps(result, default=str, indent=2))
                print(json.dumps({k: v for k, v in result.items() if k not in ('env_cfg', 'agent_cfg', 'args')}, indent=2), flush=True)
                return
            actions = torch.zeros(env.num_envs, 27, device=env.device)
            reward_sum = torch.zeros(env.num_envs, device=env.device)
            episode_return = torch.zeros_like(reward_sum)
            episode_reference = torch.zeros_like(reward_sum)
            episode_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
            completed_normalized_returns = []
            normalized_step_sum = 0.0
            alive = torch.zeros_like(reward_sum)
            totals = {key: 0.0 for key in ("terminated", "truncated", "swing_air", "touchdown", "late_xy_error", "late_samples", "action_square", "forward_velocity", "post_start_xy", "post_start_samples")}
            term_sums = {}
            log_snapshots = []
            no_fall = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
            cycle_air = torch.zeros_like(no_fall)
            cycle_touchdown = torch.zeros_like(no_fall)
            cycle_support = torch.ones_like(no_fall)
            cycle_totals = [0, 0]
            cycle_successes = [0, 0]
            # 分别统计左脚起步/右脚起步的表现，避免混合平均掩盖其中一种起步失败。
            start_cycle_totals = [[0, 0], [0, 0]]
            start_cycle_successes = [[0, 0], [0, 0]]
            initial_left = env.goal_sampler.initial_left_is_swing
            initial_start_counts = [int(initial_left.sum().item()), int((~initial_left).sum().item())]
            reset_start_counts = [0, 0]
            both_air_samples = 0
            contact_speed_sum = contact_speed_samples = 0.0
            vertical_speed_square_sum = 0.0
            stance_hop_samples = stance_interior_samples = 0
            consecutive = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
            best_consecutive = torch.zeros_like(consecutive)
            filtered_square_sum = 0.0
            height_sum = 0.0
            upright_sum = 0.0
            lateral_sum = 0.0
            yaw_rate_sum = 0.0
            minimum_height = float("inf")
            max_reward_error = 0.0
            heading_sum = 0.0
            heading_count = 0
            max_heading_error = 0.0
            reference_forward_sum = 0.0
            command_length_sum = 0.0
            command_length_count = 0
            command_speed_sum = 0.0
            frequency_sums = {name: 0.0 for name in ("left", "right", "effective")}
            frequency_min = {name: float("inf") for name in ("left", "right")}
            frequency_max = {name: -float("inf") for name in ("left", "right")}
            duration_sums = {name: 0.0 for name in ("left", "right")}
            cycle_failure_counts = {name: [0, 0] for name in ("air", "touchdown", "support")}
            max_phase_error = 0.0
            command_length_min = float("inf")
            command_length_max = -float("inf")
            reference_lateral_sum = 0.0
            max_lateral_displacement = 0.0
            episode_start_pos = env.robot.data.root_pos_w.torch.clone()
            trace = []
            first_sums, first_counts = {}, {}
            obs = wrapped.get_observations()
            if str(env.device).startswith('cuda'):
                torch.cuda.synchronize(env.device)
            started = time.monotonic()
            with torch.inference_mode():
                for step in range(args.steps):
                    s = env.goal_sampler
                    phase = s.phase.clone()
                    first_swing = s.completed_half_cycles == 0
                    left_frequency = s.left_gait_frequencies.clone()
                    right_frequency = s.right_gait_frequencies.clone()
                    commanded_length = s.step_lengths.clone()
                    command_speed_sum += (2 * s.gait_frequencies * commanded_length).sum().item()
                    frequency_sums["effective"] += s.gait_frequencies.sum().item()
                    for side, values in (("left", left_frequency), ("right", right_frequency)):
                        frequency_sums[side] += values.sum().item()
                        frequency_min[side] = min(frequency_min[side], values.min().item())
                        frequency_max[side] = max(frequency_max[side], values.max().item())
                        duration_sums[side] += (0.5 / values).sum().item()
                    command_length_sum += commanded_length.sum().item()
                    command_length_count += env.num_envs
                    command_length_min = min(command_length_min, commanded_length.min().item())
                    command_length_max = max(command_length_max, commanded_length.max().item())
                    # 保存本步使用的参考；step末尾可能局部reset，不能拿新参考重算旧奖励。
                    reference_heading = s.reference_heading_w.clone()
                    swing = s.left_is_swing.clone()
                    episode_started_left = s.initial_left_is_swing.clone()
                    local = torch.where(swing, 2 * phase, 2 * phase - 1)
                    left_target = s.left_target_pos_w.clone()
                    right_target = s.right_target_pos_w.clone()
                    prev_actions = env.prev_actions.clone()
                    actions.zero_()
                    if args.mode == "random":
                        actions.normal_(0, args.std)
                    elif args.mode == "lift":
                        knee = args.knee * torch.sin(torch.pi * local).clamp_min(0)
                        # Same coordinated hip/knee/ankle pattern as the step test.
                        scales = getattr(env, "action_scales", torch.full_like(actions, cfg.action_scale))
                        for start, mask in ((0, swing), (6, ~swing)):
                            actions[:, start + 2] = torch.where(mask, -knee / 2 / scales[:, start + 2], 0)
                            actions[:, start + 3] = torch.where(mask, knee / scales[:, start + 3], 0)
                            actions[:, start + 4] = torch.where(mask, -knee / 2 / scales[:, start + 4], 0)
                    elif policy is not None:
                        actions = policy(obs, stochastic_output=args.mode == "policy_sampled")
                    applied = actions.clamp(-1, 1).clone()
                    filtered = (1 - cfg.action_smoothing) * env.actions + cfg.action_smoothing * applied
                    filtered_square_sum += filtered.square().mean(-1).sum().item()
                    obs, rewards, dones, _ = wrapped.step(actions)
                    assert torch.isfinite(rewards).all()
                    assert torch.isfinite(obs["policy"]).all()
                    terms = env.reward_terms
                    terminated = env.reset_terminated
                    truncated = env.reset_time_outs
                    valid = ~dones.bool()
                    episode_return += rewards
                    episode_reference += env.reward_reference
                    episode_steps += 1
                    normalized_step_sum += (rewards / env.reward_reference.clamp_min(1e-8)).clamp(0, 1).sum().item()
                    ended = dones.bool()
                    if ended.any():
                        completed_normalized_returns.extend(normalized_episode_return(
                            episode_return[ended], episode_reference[ended], episode_steps[ended],
                            env.max_episode_length).tolist())
                        episode_return[ended] = 0
                        episode_reference[ended] = 0
                        episode_steps[ended] = 0
                    no_fall &= ~terminated
                    height = env.robot.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
                    height_sum += height.sum().item()
                    upright_sum += (-env.robot.data.projected_gravity_b.torch[:, 2]).sum().item()
                    lateral_sum += env.robot.data.root_lin_vel_b.torch[:, 1].abs().sum().item()
                    yaw_rate_sum += env.robot.data.root_ang_vel_b.torch[:, 2].abs().sum().item()
                    if valid.any():
                        minimum_height = min(minimum_height, height[valid].min().item())
                        error_heading = heading_error(env.robot.data.root_quat_w.torch, reference_heading)
                        heading_sum += error_heading[valid].abs().sum().item()
                        heading_count += int(valid.sum().item())
                        max_heading_error = max(max_heading_error, error_heading[valid].abs().max().item())
                        # 将世界速度投影到固定参考方向，分别统计前进和侧向运动。
                        velocity = env.robot.data.root_lin_vel_w.torch
                        c, sn = reference_heading.cos(), reference_heading.sin()
                        vx_ref = c * velocity[:, 0] + sn * velocity[:, 1]
                        vy_ref = -sn * velocity[:, 0] + c * velocity[:, 1]
                        reference_forward_sum += vx_ref[valid].sum().item()
                        reference_lateral_sum += vy_ref[valid].abs().sum().item()
                        delta = env.robot.data.root_pos_w.torch - episode_start_pos
                        lateral_displacement = -sn * delta[:, 0] + c * delta[:, 1]
                        max_lateral_displacement = max(max_lateral_displacement,
                                                       lateral_displacement[valid].abs().max().item())
                    episode_start_pos[~valid] = env.robot.data.root_pos_w.torch[~valid]
                    torch.testing.assert_close(s.reference_heading_w[valid], reference_heading[valid])
                    # 正常步进和换脚都不能改变episode内的命令；只有reset环境允许重采样。
                    # 渐变命令只允许在换脚边界更新；原有episode固定命令断言保留。
                    unchanged = valid
                    if cfg.command_resample_half_cycles or cfg.command_warmup_half_cycles:
                        unchanged = valid & (swing == s.left_is_swing)
                    torch.testing.assert_close(s.step_lengths[unchanged], commanded_length[unchanged], rtol=0, atol=0)
                    torch.testing.assert_close(s.left_gait_frequencies[unchanged], left_frequency[unchanged], rtol=0, atol=0)
                    torch.testing.assert_close(s.right_gait_frequencies[unchanged], right_frequency[unchanged], rtol=0, atol=0)
                    lp, lq, rp, rq = env._get_foot_poses()
                    lf, rf = env._get_ground_contact_forces()
                    physical_state = env.flat_reward_state() if isinstance(env.reward_cfg, FlatWalkRewardCfg) else None
                    if physical_state is not None:
                        # step已推进相位；首步加权重算必须使用产生本次奖励的旧半周期标志。
                        physical_state['first_swing'] = first_swing
                    expected, _ = compute_foot_placement_reward(
                        lp, rp, left_target, right_target, lf, rf, phase, swing,
                        env.robot.data.projected_gravity_b.torch,
                        env.actions, prev_actions, env.step_dt, env.reward_cfg,
                        swing_has_lifted=env._reward_swing_has_lifted,
                        root_quat_w=env.robot.data.root_quat_w.torch,
                        reference_heading_w=reference_heading,
                        root_yaw_rate=env.robot.data.root_ang_vel_b.torch[:, 2],
                        flat_state=physical_state,
                    )
                    if valid.any():
                        error = (expected[valid] - rewards[valid]).abs().max().item()
                        max_reward_error = max(max_reward_error, error)
                        assert error < 1e-5, "Reward/phase/action history mismatch"
                    reset_phase = torch.where(s.left_is_swing, 0., .5)
                    phase_expected = torch.where(valid, advance_phase(phase, env.step_dt, left_frequency, right_frequency), reset_phase)
                    max_phase_error = max(max_phase_error, (s.phase - phase_expected).abs().max().item())
                    torch.testing.assert_close(s.phase, phase_expected)
                    torch.testing.assert_close(s.left_is_swing, s.phase < .5)
                    reset_start_counts[0] += int(((~valid) & s.initial_left_is_swing).sum().item())
                    reset_start_counts[1] += int(((~valid) & ~s.initial_left_is_swing).sum().item())
                    if cfg.observe_gait_frequency:
                        if cfg.gait_frequency_obs_dim == 2:
                            torch.testing.assert_close(obs["policy"][:, 87:89], torch.stack(
                                (s.left_gait_frequencies, s.right_gait_frequencies), dim=-1))
                        else:
                            torch.testing.assert_close(obs["policy"][:, 87:88], s.gait_frequencies[:, None])
                    torch.testing.assert_close(obs["policy"][:, 60:87], env.actions)
                    angle = 2 * torch.pi * s.phase
                    torch.testing.assert_close(obs["policy"][:, -2:], torch.stack((angle.cos(), angle.sin()), dim=-1))
                    reward_sum += rewards
                    alive += valid.float()
                    totals["terminated"] += terminated.sum().item()
                    totals["truncated"] += truncated.sum().item()
                    totals["action_square"] += applied.square().mean(-1).sum().item()
                    totals["forward_velocity"] += env.robot.data.root_lin_vel_b.torch[:, 0].sum().item()
                    swing_pos = torch.where(swing[:, None], lp, rp)
                    target = torch.where(swing[:, None], left_target, right_target)
                    late = (local >= 0.8) & valid
                    middle = (local > 0.2) & (local < 0.8) & valid
                    swing_fz = torch.where(swing, lf[:, 2], rf[:, 2])
                    stance_fz = torch.where(swing, rf[:, 2], lf[:, 2])
                    lift_height = swing_pos[:, 2] - target[:, 2]
                    cycle_air |= middle & (swing_fz < cfg.reward_cfg.contact_threshold) & (lift_height > max(0.01, 0.6 * cfg.reward_cfg.swing_height))
                    cycle_touchdown |= late & (swing_fz > cfg.reward_cfg.contact_threshold) & ((swing_pos[:, :2] - target[:, :2]).norm(dim=-1) < 0.03)
                    cycle_support &= ~middle | (stance_fz > cfg.reward_cfg.contact_threshold)
                    contact = torch.stack((lf[:,2],rf[:,2]),dim=1) > cfg.reward_cfg.contact_threshold
                    both_air_samples += int((~contact.any(-1) & valid).sum().item())
                    interior = (local>.1) & (local<.9) & valid
                    stance_height = torch.where(swing,rp[:,2]-right_target[:,2],lp[:,2]-left_target[:,2])
                    stance_hop_samples += int((interior & ((stance_fz<=cfg.reward_cfg.contact_threshold)
                        | (stance_height>getattr(cfg.reward_cfg,'support_height_tolerance',.025)))).sum().item())
                    stance_interior_samples += int(interior.sum().item())
                    physical = env.flat_reward_state()
                    speeds = torch.stack((physical['left_velocity'][:,:2].norm(dim=-1),
                                          physical['right_velocity'][:,:2].norm(dim=-1)),dim=1)
                    selected = contact & valid[:,None]
                    contact_speed_sum += (speeds*selected.float()).sum().item()
                    contact_speed_samples += selected.sum().item()
                    vertical_speed_square_sum += physical['root_vz'][valid].square().sum().item()
                    totals["swing_air"] += (middle & (swing_fz < cfg.reward_cfg.contact_threshold)
                                            & (stance_fz > cfg.reward_cfg.contact_threshold) & (lift_height > .01)).sum().item()
                    totals["touchdown"] += (late & (swing_fz > cfg.reward_cfg.contact_threshold)
                                            & ((swing_pos[:,:2]-target[:,:2]).norm(dim=-1)<.03)).sum().item()
                    switched = (swing != s.left_is_swing) & valid
                    cycle_ok = cycle_air & cycle_touchdown & cycle_support
                    consecutive = torch.where(switched, torch.where(cycle_ok, consecutive + 1, 0), consecutive)
                    consecutive = torch.where(valid, consecutive, 0)
                    best_consecutive = torch.maximum(best_consecutive, consecutive)
                    for side, mask in enumerate((swing, ~swing)):
                        ended = switched & mask
                        cycle_totals[side] += int(ended.sum().item())
                        cycle_successes[side] += int((ended & cycle_air & cycle_touchdown & cycle_support).sum().item())
                        for start_side, start_mask in enumerate((episode_started_left, ~episode_started_left)):
                            start_cycle_totals[start_side][side] += int((ended & start_mask).sum().item())
                            start_cycle_successes[start_side][side] += int((ended & start_mask & cycle_ok).sum().item())
                        for name, ok in (("air", cycle_air), ("touchdown", cycle_touchdown), ("support", cycle_support)):
                            cycle_failure_counts[name][side] += int((ended & ~ok).sum().item())
                    clear = switched | ~valid
                    assert not env._swing_has_lifted[clear].any()
                    cycle_air[clear] = False
                    cycle_touchdown[clear] = False
                    cycle_support[clear] = True
                    totals["late_xy_error"] += ((swing_pos[:, :2] - target[:, :2]).norm(dim=-1) * late).sum().item()
                    totals["late_samples"] += late.sum().item()
                    totals['post_start_xy'] += ((swing_pos[:, :2] - target[:, :2]).norm(dim=-1) * late * ~first_swing).sum().item()
                    totals['post_start_samples'] += (late & ~first_swing).sum().item()
                    for name, value in terms.items():
                        term_sums[name] = term_sums.get(name, 0.0) + value.mean().item() * env.step_dt
                    log_snapshots.append(env.extras["log"])
                    for key, value in env.extras['log'].items():
                        if key.startswith('Startup/'):
                            first_sums[key] = first_sums.get(key, 0.) + value.sum().item()
                            first_counts[key] = first_counts.get(key, 0) + value.numel()
                    record = {"step":step,"reward":rewards.mean().item(),"terminated":terminated.sum().item(),"truncated":truncated.sum().item(),"phase":s.phase[0].item(),"left_z":lp[0,2].item(),"right_z":rp[0,2].item(),"left_fz":lf[0,2].item(),"right_fz":rf[0,2].item(),
                                  "heading_error_rad": heading_error(env.robot.data.root_quat_w.torch, reference_heading)[0].item(),
                                  "root_x": env.robot.data.root_pos_w.torch[0,0].item(),
                                  "root_y": env.robot.data.root_pos_w.torch[0,1].item(),
                                  "root_yaw_rad": yaw_from_quat(env.robot.data.root_quat_w.torch)[0].item(),
                                  "command_step_length_m": commanded_length[0].item(),
                                  "left_gait_frequency_hz": left_frequency[0].item(),
                                  "right_gait_frequency_hz": right_frequency[0].item(),
                                  "reset": bool(dones[0].item())}
                    if args.trace_reward_terms:
                        # 奖励来自step推进相位之前；这里单独保存旧相位，避免画出错位曲线。
                        # 分项取环境0，与足端/接触trace一致；总reward仍保留原来的环境均值。
                        record.update(
                            phase_before=phase[0].item(), local_phase_before=local[0].item(),
                            left_is_swing_before=bool(swing[0].item()),
                            reward_env0=rewards[0].item(), reward_reference_env0=env.reward_reference[0].item(),
                            reward_components_env0={name:value[0].item()*env.step_dt for name,value in terms.items()},
                            swing_lift_m=lift_height[0].item(),
                            swing_xy_error_m=(swing_pos[0,:2]-target[0,:2]).norm().item(),
                            stance_height_m=stance_height[0].item(),
                            swing_fz=swing_fz[0].item(), stance_fz=stance_fz[0].item(),
                            root_vz_m_s=physical['root_vz'][0].item(),
                        )
                    trace.append(record)
            n = env.num_envs * args.steps
            rollout_seconds = time.monotonic() - started
            from tasks.foot_placement.first_step_metrics import summarize_first_steps
            result = {"mode":args.mode,"stage":args.stage,"num_envs":env.num_envs,"steps":args.steps,
                "rollout_wall_seconds": rollout_seconds,
                "first_step": summarize_first_steps(first_sums, first_counts),
                "initial_swing_side": cfg.initial_swing_side,
                "initial_start_counts_left_right": initial_start_counts,
                "reset_start_counts_left_right": reset_start_counts,
                "reward_profile": getattr(cfg.reward_cfg,'profile','legacy'),
                "both_air_fraction": both_air_samples/max(alive.sum().item(),1),
                "stance_hop_fraction": stance_hop_samples/max(stance_interior_samples,1),
                "contact_foot_slip_speed_m_s": contact_speed_sum/max(contact_speed_samples,1),
                "root_vertical_velocity_rms_m_s": (vertical_speed_square_sum/max(alive.sum().item(),1))**.5,
                "by_initial_swing": {
                    name: {"completed_half_cycles_left_right": start_cycle_totals[i],
                           "half_cycle_success_left_right": [ok / max(total, 1) for ok, total in zip(
                               start_cycle_successes[i], start_cycle_totals[i])]}
                    for i, name in enumerate(("left", "right")) if sum(start_cycle_totals[i]) > 0},
                "mean_reward":reward_sum.mean().item()/args.steps,
                "normalized_reward": normalized_step_sum / n,
                "normalized_return": (sum(completed_normalized_returns) / len(completed_normalized_returns)
                                       if completed_normalized_returns else None),
                "normalized_return_completed_episodes": len(completed_normalized_returns),
                "terminated":totals['terminated'],"truncated":totals['truncated'],
                "swing_air_ratio":totals['swing_air']/n,"touchdown_ratio":totals['touchdown']/n,
                "late_xy_error_m":totals['late_xy_error']/max(totals['late_samples'],1),
                "post_start_late_xy_error_m":totals['post_start_xy']/max(totals['post_start_samples'],1),
                "action_rms":(totals['action_square']/n)**0.5,
                "filtered_action_rms":(filtered_square_sum/n)**0.5,
                "survival_fraction":no_fall.float().mean().item(),
                "mean_root_height":height_sum/n,"minimum_valid_root_height":minimum_height,
                "mean_upright":upright_sum/n,"mean_abs_lateral_velocity":lateral_sum/n,
                "mean_abs_yaw_rate":yaw_rate_sum/n,
                "mean_abs_heading_error_rad": heading_sum / max(heading_count, 1),
                "max_abs_heading_error_rad": max_heading_error,
                "heading_valid_samples": heading_count,
                "mean_reference_forward_velocity": reference_forward_sum / max(heading_count, 1),
                "command_step_length_mean_m": command_length_sum / max(command_length_count, 1),
                "command_step_length_min_m": command_length_min,
                "command_step_length_max_m": command_length_max,
                "command_nominal_forward_speed_m_s": command_speed_sum / max(command_length_count, 1),
                "command_effective_gait_frequency_mean_hz": frequency_sums['effective'] / n,
                **{f"command_{side}_gait_frequency_mean_hz": frequency_sums[side] / n for side in ("left", "right")},
                **{f"command_{side}_gait_frequency_min_hz": frequency_min[side] for side in ("left", "right")},
                **{f"command_{side}_gait_frequency_max_hz": frequency_max[side] for side in ("left", "right")},
                **{f"command_{side}_swing_duration_mean_s": duration_sums[side] / n for side in ("left", "right")},
                "half_cycle_failure_counts_left_right": cycle_failure_counts,
                "observation_dim": obs['policy'].shape[-1],
                "mean_abs_reference_lateral_velocity": reference_lateral_sum / max(heading_count, 1),
                "max_abs_reference_lateral_displacement_m": max_lateral_displacement,
                "completed_half_cycles_left_right":cycle_totals,
                "successful_half_cycles_left_right":cycle_successes,
                "half_cycle_success_left_right":[ok/max(count,1) for ok,count in zip(cycle_successes,cycle_totals)],
                "consecutive_8_success_fraction":((best_consecutive >= 8) & no_fall).float().mean().item(),
                "best_consecutive_half_cycles":best_consecutive.tolist(),
                "mean_forward_velocity":totals['forward_velocity']/n,
                "max_reward_alignment_error":max_reward_error,
                "max_phase_alignment_error": max_phase_error,
                "distinct_log_objects":len({id(x) for x in log_snapshots}),
                "reward_components":{key:value/args.steps for key,value in term_sums.items()},
                "totals":totals,"env_cfg":to_dict(cfg),"agent_cfg":to_dict(agent),"args":vars(args)}
            if str(env.device).startswith("cuda"):
                free, total = torch.cuda.mem_get_info(env.device)
                result["gpu_memory_used_bytes"] = total - free
                result["gpu_memory_total_bytes"] = total
            args.output.write_text(json.dumps(result, default=str, indent=2))
            args.output.with_suffix('.jsonl').write_text('\n'.join(json.dumps(x) for x in trace)+'\n')
            print(json.dumps({k:v for k,v in result.items() if k not in ('env_cfg','agent_cfg','args','totals')},indent=2),flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    main()
