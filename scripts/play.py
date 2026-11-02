
"""回放 S54 PPO，支持目标足印可视化及零动作对照。"""

import argparse
import json
import math
import sys
from pathlib import Path

# 直接运行 scripts/play.py 时，Python 优先搜索 scripts/，找不到同级之外的 tasks/。
# 根据脚本位置加入项目根目录，回放不再依赖外部设置 PYTHONPATH，也不依赖启动目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import add_launcher_args, launch_simulation
from scripts.gait_cli import add_gait_arguments, apply_gait_overrides

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=96)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--stage", choices=["legacy", "stand", "lift", "step", "walk", "straight"], default=None)
parser.add_argument("--show-goals", action=argparse.BooleanOptionalAction, default=True,
                    help="绘制目标足印和摆动高度参考；--no-show-goals 可关闭")
parser.add_argument("--policy-only", action="store_true", help="全部环境执行策略，不设置零动作对照组")
parser.add_argument("--step-length-range", "--step_length_range", type=float, nargs=2)
parser.add_argument("--task-profile", type=Path, help="显式覆盖回放任务，例如关闭扰动或设置60秒episode")
add_gait_arguments(parser)

add_launcher_args(parser)
parser.set_defaults(device="cuda:0")
args=parser.parse_args()

import torch
import gymnasium as gym

import tasks.foot_placement as s54_task

from agents.rsl_rl_ppo_cfg import S54FootPlacementPPORunnerCfg
from tasks.foot_placement.foot_placement_env_cfg import (
    S54FootPlacementEnvCfg,
)
from tasks.foot_placement.foot_placement_env import (
    S54FootPlacementEnv,
)
from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)
from configs.s54_curriculum import configure_stage, restore_task_config
from tasks.foot_placement.visualization import FootstepVisualizer
from agents.checkpoint_compat import checkpoint_task_config, load_s54_checkpoint


def main():
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    if args.num_envs < 1 or (not args.policy_only and (args.num_envs < 2 or args.num_envs % 2)):
        raise ValueError("对照模式要求偶数个环境且 >= 2；--policy-only 允许任意正数个环境")
    if args.steps <= 0:
        raise ValueError("--steps must be greater than zero")

    env_cfg = S54FootPlacementEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device

    agent_cfg = S54FootPlacementPPORunnerCfg()
    saved_task = checkpoint_task_config(checkpoint)
    if args.stage is not None:
        configure_stage(env_cfg, agent_cfg, args.stage)
        env_cfg.observe_gait_frequency = saved_task["observe_gait_frequency"]
        env_cfg.gait_frequency_obs_dim = saved_task["gait_frequency_obs_dim"]
    else:
        restore_task_config(env_cfg, saved_task)
    from configs.walk_training import apply_task_profile
    apply_task_profile(env_cfg, args.task_profile)
    apply_gait_overrides(env_cfg, args)
    if args.step_length_range is not None:
        low, high = args.step_length_range
        if not (math.isfinite(low) and math.isfinite(high) and 0 <= low <= high):
            raise ValueError("步长范围必须满足0 <= 下限 <= 上限且有限")
        env_cfg.step_length_range = (low, high)
        env_cfg.command_groups = None
        env_cfg.command_resample_half_cycles = 0
    agent_cfg.device = args.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        agent_cfg, check_rsl_rl_version()
    )

    half = args.num_envs if args.policy_only else args.num_envs // 2

    with launch_simulation(env_cfg.sim, args):
        env = gym.make(s54_task.TASK_ID, cfg=env_cfg)

        try:
            task_env = env.unwrapped
            if not isinstance(task_env, S54FootPlacementEnv):
                raise TypeError("Expected an S54FootPlacementEnv")

            wrapped = RslRlVecEnvWrapper(
                task_env, clip_actions=agent_cfg.clip_actions
            )

            runner = create_rsl_rl_runner(
                wrapped, agent_cfg, log_dir=None
            )
            load_s54_checkpoint(runner, checkpoint, map_location=args.device)

            policy = runner.get_inference_policy(
                device=str(wrapped.device)
            )

            obs = wrapped.get_observations()
            # 无可视化后端时不创建 marker，保持 headless 回放开销不变。
            goal_visualizer = FootstepVisualizer(task_env) if args.show_goals and task_env.sim.has_active_visualizers() else None

            reward_sum = torch.zeros(
                args.num_envs, device=wrapped.device
            )
            done_count = torch.zeros_like(reward_sum)
            air_count = torch.zeros_like(reward_sum)
            touchdown_count = torch.zeros_like(reward_sum)

            # Separate falls from normal episode timeouts.
            fall_count = torch.zeros_like(reward_sum)
            timeout_count = torch.zeros_like(reward_sum)

            # Tracking error accumulated during late swing.
            late_xy_error_sum = torch.zeros_like(reward_sum)
            late_xy_error_count = torch.zeros_like(reward_sum)

            # Magnitude of policy actions.
            action_square_sum = torch.zeros_like(reward_sum)


            print("\n===== Policy Evaluation =====")
            print("Checkpoint:", checkpoint)
            print("Curriculum stage:", env_cfg.curriculum_stage)
            if env_cfg.step_length_range is None:
                print(f"实际目标偏移 [前向, 横向] = [{env_cfg.step_length:.3f}, ±{env_cfg.feet_distance:.3f}] m")
                nominal_length = env_cfg.step_length
            else:
                low, high = env_cfg.step_length_range
                nominal_length = (low + high) / 2
                print(f"实际前向偏移均匀采样 [{low:.3f}, {high:.3f}] m，脚间距 {env_cfg.feet_distance:.3f} m；每episode固定。")
            s = task_env.goal_sampler
            print(f"目标抬脚高度 = {env_cfg.reward_cfg.swing_height:.3f} m；观测维度 = {env_cfg.observation_space}")
            print(f"左右节奏均值 = {s.left_gait_frequencies.mean().item():.3f}/{s.right_gait_frequencies.mean().item():.3f} Hz")
            print(f"左右摆动时长均值 = {(0.5/s.left_gait_frequencies).mean().item():.3f}/{(0.5/s.right_gait_frequencies).mean().item():.3f} s")
            print(f"理想交替落脚的平均名义前向速度 ≈ {(2*s.gait_frequencies*s.step_lengths).mean().item():.3f} m/s")
            print("Policy envs:", half)
            print("Zero-action envs:", args.num_envs - half)
            print("接触统计：中摆动期离地>1cm且另一脚接地；末摆动期接地且XY误差<3cm。")
            print("两项ratio均为控制帧占比，不是每步成功率；统计不依赖奖励项名称。")
            if goal_visualizer is not None:
                print("足印图例：青=左脚，橙=右脚；亮=摆动，暗=支撑；箭头=目标脚尖朝向；紫点=摆动高度参考。")

            completed_steps = 0

            with torch.inference_mode():
                for step in range(args.steps):
                    if not task_env.sim.is_running():
                        break
                    if goal_visualizer is not None:
                        # 先画当前目标，再推进物理；相位切换后下次循环自动显示新足印。
                        goal_visualizer.update()

                    sampler = task_env.goal_sampler

                    # Save the goal that will be used to compute this step's reward.
                    phase_before = sampler.phase.clone()
                    left_swing_before = sampler.left_is_swing.clone()

                    swing_target = torch.where(
                        left_swing_before[:, None],
                        sampler.left_target_pos_w,
                        sampler.right_target_pos_w,
                    ).clone()

                    local_phase = torch.where(
                        left_swing_before,
                        phase_before * 2.0,
                        (phase_before - 0.5) * 2.0,
                    ).clamp(0.0, 1.0)

                    policy_actions = policy(obs)

                    assert policy_actions.shape == (
                        args.num_envs, 27
                    )
                    assert torch.isfinite(policy_actions).all()

                    actions = policy_actions.clone()

                    # Second half: zero-action control group.
                    actions[half:] = 0.0

                    obs, rewards, dones, _ = wrapped.step(actions)
                    completed_steps += 1

                    # Isaac Lab distinguishes actual termination and timeout.
                    terminated = task_env.reset_terminated.clone()
                    truncated = task_env.reset_time_outs.clone()

                    fall_count += terminated.float()
                    timeout_count += truncated.float()

                    # Match the wrapper's optional action clipping.
                    clip_actions = wrapped.clip_actions
                    applied_actions = actions
                    if clip_actions is not None:
                        applied_actions = actions.clamp(
                            -clip_actions, clip_actions
                        )
                    action_square_sum += applied_actions.square().mean(dim=-1)

                    # Read actual sole positions after Newton has stepped.
                    left_pos, _, right_pos, _ = task_env._get_foot_poses()

                    swing_pos = torch.where(
                        left_swing_before[:, None],
                        left_pos,
                        right_pos,
                    )

                    xy_error = torch.linalg.vector_norm(
                        swing_pos[:, :2] - swing_target[:, :2],
                        dim=-1,
                    )

                    # Avoid measuring the reset pose against a previous goal.
                    valid = ~(terminated | truncated)

                    # Only evaluate placement accuracy near the end of swing.
                    late_swing = (local_phase >= 0.8) & valid

                    late_xy_error_sum += torch.where(
                        late_swing,
                        xy_error,
                        torch.zeros_like(xy_error),
                    )
                    late_xy_error_count += late_swing.float()

                    assert torch.isfinite(rewards).all()

                    reward_sum += rewards
                    done_count += dones.float()

                    # v2已经移除swing_air/touchdown奖励；指标应由实际接触计算，
                    # 不能依赖奖励字典或奖励权重。判据与preflight_signals保持一致。
                    left_force, right_force = task_env._get_ground_contact_forces()
                    swing_fz = torch.where(
                        left_swing_before, left_force[:, 2], right_force[:, 2]
                    )
                    stance_fz = torch.where(
                        left_swing_before, right_force[:, 2], left_force[:, 2]
                    )
                    contact_threshold = env_cfg.reward_cfg.contact_threshold
                    middle_swing = (local_phase > 0.2) & (local_phase < 0.8) & valid
                    lift_height = swing_pos[:, 2] - swing_target[:, 2]
                    air_count += (middle_swing & (swing_fz < contact_threshold)
                                  & (stance_fz > contact_threshold) & (lift_height > 0.01)).float()
                    touchdown_count += (late_swing & (swing_fz > contact_threshold)
                                        & (xy_error < 0.03)).float()

                    policy.reset(dones)

                    if (step + 1) % 100 == 0:
                        progress = (f"step={step + 1:04d} "
                                    f"policy_reward={rewards[:half].mean().item():.4f}")
                        # 全策略模式没有对照环境，不对空切片求均值，避免打印无意义的nan。
                        if not args.policy_only:
                            progress += f" baseline_reward={rewards[half:].mean().item():.4f}"
                        print(progress, flush=True)
            def summarize(name: str, start: int, end: int) -> None:
                avg_reward = (
                    reward_sum[start:end].mean().item()
                    / completed_steps
                )
                avg_dones = done_count[start:end].mean().item()
                air_ratio = (
                    air_count[start:end].mean().item()
                    / completed_steps
                )
                touchdown_ratio = (
                    touchdown_count[start:end].mean().item()
                    / completed_steps
                )
                falls = fall_count[start:end].mean().item()
                timeouts = timeout_count[start:end].mean().item()

                late_samples = late_xy_error_count[start:end].sum().item()
                late_error = (
                    late_xy_error_sum[start:end].sum().item() / late_samples
                    if late_samples > 0 else None
                )

                action_rms = torch.sqrt(
                    action_square_sum[start:end].mean() / completed_steps
                ).item()

                print(f"\n{name}")
                print(f"  Reward/step:       {avg_reward:.5f}")
                print(f"  Resets/env:        {avg_dones:.2f}")
                print(f"  Swing-air ratio:   {air_ratio:.3f}")
                print(f"  Touchdown ratio:   {touchdown_ratio:.3f}")
                print(f"  Falls/env:         {falls:.2f}")
                print(f"  Timeouts/env:      {timeouts:.2f}")
                if late_error is None:
                    print("  Late swing XY err: N/A (no late-swing samples)")
                else:
                    print(f"  Late swing XY err: {late_error:.4f} m")
                print(f"  Action RMS:        {action_rms:.4f}")

            print("\n===== Evaluation Summary =====")
            if completed_steps:
                summarize("PPO Policy", 0, half)
                if not args.policy_only:
                    summarize("Zero-action Baseline", half, args.num_envs)
            if goal_visualizer is not None:
                print(f"目标可视化更新次数：{goal_visualizer.updates}；每环境14条足印线段和3个参考点。")

            print("\n回放完成；步态质量请查看独立评估指标。")
        
        finally:
            env.close()

if __name__ == "__main__":
    main()


