
"""Train S54 Foot Placement with Isaac Lab 3.0 + RSL-RL."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import signal
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.gait_cli import add_gait_arguments, apply_gait_overrides

from isaaclab.app import add_launcher_args, launch_simulation


parser = argparse.ArgumentParser(
    description="S54 Foot Placement PPO training"
)

parser.add_argument(
    "--num_envs",
    type=int,
    default=16,
)
parser.add_argument(
    "--max_iterations",
    type=int,
    default=3,
)
parser.add_argument(
    "--run_name",
    type=str,
    default="smoke",
)
parser.add_argument("--stage", choices=["legacy", "stand", "lift", "step", "walk", "straight"], default="legacy")
parser.add_argument("--ablation", choices=["none", "legacy_reward", "legacy_scale", "no_smoothing", "targets_only", "rewards_only"], default="none")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--log_root", type=Path)
parser.add_argument("--normalization_warmup_steps", type=int, default=0)
parser.add_argument("--normalization_prior_rollouts", type=int, default=0,
                    help="For straight actor transfer, cap normalizer history to N rollouts; 0 keeps old count")
parser.add_argument("--learning_rate", type=float)
parser.add_argument("--step_length_range", "--step-length-range", type=float, nargs=2, metavar=("MIN_M", "MAX_M"),
                    help="每个episode均匀采样水平步长，单位m；显式覆盖续训checkpoint的固定步长")
parser.add_argument("--validate_only", action="store_true", help="仅初始化并验证checkpoint与配置，执行0次PPO更新")
parser.add_argument("--task-profile", type=Path, help="恢复checkpoint后显式应用的混合命令/抗扰动JSON")
parser.add_argument("--output_dir", type=Path, help="Explicit new run directory (must not already exist)")
parser.add_argument("--metrics_jsonl", action="store_true", help="Stream PPO curves to the long-run supervisor")
parser.add_argument("--num_steps_per_env", type=int)
parser.add_argument("--num_mini_batches", type=int)
parser.add_argument("--save_interval", "--save-interval", type=int,
                    help="每N次PPO更新保存checkpoint；正整数，未指定时沿用PPO配置")
restart = parser.add_mutually_exclusive_group()
restart.add_argument("--resume", type=Path, help="Continue the same stage, including optimizer state")
restart.add_argument("--warm_start", type=Path, help="Transfer actor only; reset critic, optimizer and iteration")
add_gait_arguments(parser)

add_launcher_args(parser)
parser.set_defaults(device="cuda:0")
args_cli = parser.parse_args()
if args_cli.save_interval is not None and args_cli.save_interval < 1:
    parser.error("save_interval必须为正整数")


# Import training components after parsing launcher arguments.
import gymnasium as gym
import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.modules.normalization import EmpiricalNormalization

import tasks.foot_placement as s54_task

from agents.rsl_rl_ppo_cfg import (
    S54FootPlacementPPORunnerCfg,
)

from tasks.foot_placement.foot_placement_env_cfg import (
    S54FootPlacementEnvCfg,
)
from tasks.foot_placement.foot_placement_env import (
    S54FootPlacementEnv,
)

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab.utils import to_dict
from configs.s54_curriculum import configure_stage, restore_task_config
from agents.checkpoint_compat import checkpoint_task_config, load_s54_checkpoint
from agents.vec_env import S54RslRlVecEnvWrapper


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    env_cfg = S54FootPlacementEnvCfg()
    agent_cfg = S54FootPlacementPPORunnerCfg()
    configure_stage(env_cfg, agent_cfg, args_cli.stage, args_cli.ablation)
    if args_cli.resume is not None:
        if not args_cli.resume.is_file():
            raise FileNotFoundError(args_cli.resume)
        saved_task = checkpoint_task_config(args_cli.resume)
        if saved_task:
            if "curriculum_stage" in saved_task and saved_task["curriculum_stage"] != args_cli.stage:
                raise ValueError("Stage changes require --warm_start, not --resume")
            # 同阶段续训恢复保存的任务参数，避免后来修改默认reward权重悄悄改变实验。
            restore_task_config(env_cfg, saved_task)
    # 所有覆盖完成后、导出env_cfg.json之前同步维度，避免文件写103而实际网络用105。
    from configs.walk_training import apply_task_profile
    apply_task_profile(env_cfg, args_cli.task_profile)
    apply_gait_overrides(env_cfg, args_cli)
    if args_cli.step_length_range is not None:
        low, high = args_cli.step_length_range
        if args_cli.stage != "straight" or not (math.isfinite(low) and math.isfinite(high) and 0 <= low <= high):
            raise ValueError("随机步长要求stage=straight且0 <= MIN_M <= MAX_M，范围必须有限")
        # 先恢复控制/奖励，再只覆盖用户明确指定的水平目标范围。
        env_cfg.step_length_range = (low, high)
        env_cfg.command_groups = None
        env_cfg.command_resample_half_cycles = 0
    if args_cli.num_steps_per_env is not None:
        agent_cfg.num_steps_per_env = args_cli.num_steps_per_env
    if args_cli.num_mini_batches is not None:
        agent_cfg.algorithm.num_mini_batches = args_cli.num_mini_batches
    if agent_cfg.num_steps_per_env < 1 or agent_cfg.algorithm.num_mini_batches < 1:
        raise ValueError("rollout length and mini-batch count must be positive")
    if (args_cli.num_envs * agent_cfg.num_steps_per_env) % agent_cfg.algorithm.num_mini_batches:
        raise ValueError("rollout sample count must be divisible by num_mini_batches")
    if args_cli.learning_rate is not None:
        if not 0 < args_cli.learning_rate < 0.01:
            raise ValueError("learning_rate must be in (0, 0.01)")
        agent_cfg.algorithm.learning_rate = args_cli.learning_rate
    if args_cli.max_iterations < 1 or args_cli.num_envs < 1:
        raise ValueError("num_envs and max_iterations must be positive")
    if args_cli.normalization_prior_rollouts < 0:
        raise ValueError("normalization_prior_rollouts must be nonnegative")
    env_cfg.seed = agent_cfg.seed = args_cli.seed
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    agent_cfg.device = args_cli.device
    agent_cfg.max_iterations = args_cli.max_iterations
    # 仅覆盖保存频率，不改变PPO更新次数、模型或优化器恢复方式。
    if args_cli.save_interval is not None:
        agent_cfg.save_interval = args_cli.save_interval

    # Adapt model configuration to installed RSL-RL version.
    installed_version = check_rsl_rl_version()
    print("[INFO] RSL-RL version:", installed_version)

    agent_cfg = handle_deprecated_rsl_rl_cfg(
        agent_cfg,
        installed_version,
    )
    if not isinstance(agent_cfg, RslRlOnPolicyRunnerCfg):
        raise TypeError("S54 training requires an on-policy PPO runner config")

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    log_root = args_cli.log_root or PROJECT_ROOT / "logs" / "rsl_rl"
    log_dir = args_cli.output_dir or (
        log_root
        / agent_cfg.experiment_name
        / f"{timestamp}_{args_cli.run_name}"
    )
    log_dir.mkdir(parents=True, exist_ok=args_cli.output_dir is None)

    logger_cfg = {"class_name": "scripts.training_log_writer:TensorboardJsonWriter"}
    for name, cfg in (("env_cfg", env_cfg), ("agent_cfg", agent_cfg)):
        data = to_dict(cfg)
        if name == "agent_cfg" and args_cli.metrics_jsonl:
            data["logger"] = logger_cfg
        (log_dir / f"{name}.json").write_text(json.dumps(data, default=str, indent=2))
    versions = {"python": platform.python_version()}
    for package in ("torch", "warp-lang", "newton", "rsl-rl-lib", "isaaclab"):
        versions[package] = importlib.metadata.version(package)
    sources = {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for folder in ("agents", "configs", "tasks", "scripts")
               for p in (PROJECT_ROOT / folder).rglob("*.py")}
    manifest = {"args": vars(args_cli), "versions": versions, "source_sha256": sources,
                "samples_per_update": args_cli.num_envs * agent_cfg.num_steps_per_env}
    (log_dir / "manifest.json").write_text(json.dumps(manifest, default=str, indent=2))

    print("\n===== S54 配置校验（不训练） =====" if args_cli.validate_only else "\n===== S54 PPO Training =====")
    print("Task:", s54_task.TASK_ID)
    print("Curriculum stage:", args_cli.stage, "ablation:", args_cli.ablation)
    print("Environments:", args_cli.num_envs)
    print("Iterations:", 0 if args_cli.validate_only else agent_cfg.max_iterations)
    print("Rollout length:", agent_cfg.num_steps_per_env)
    print("Checkpoint save interval:", agent_cfg.save_interval)
    print("Device:", agent_cfg.device)
    print("Learning rate:", agent_cfg.algorithm.learning_rate, "schedule:", agent_cfg.algorithm.schedule)
    print("Action scales:", env_cfg.action_scale_by_joint or env_cfg.action_scale)
    print("Action smoothing alpha:", env_cfg.action_smoothing)
    print("Fixed reset heading:", env_cfg.fixed_heading_targets)
    print("Heading / yaw-rate reward weights:", env_cfg.reward_cfg.heading_w, env_cfg.reward_cfg.yaw_rate_w)
    print("Gait frequency / step length / swing height:", env_cfg.gait_frequency,
          env_cfg.step_length, env_cfg.reward_cfg.swing_height)
    print("实际水平步长范围[m]（None表示固定）：", env_cfg.step_length_range)
    print("Observation dimension:", env_cfg.observation_space)
    print("左右节奏范围[Hz]:", env_cfg.left_gait_frequency_range, env_cfg.right_gait_frequency_range)
    print("Log directory:", log_dir)

    with launch_simulation(env_cfg.sim, args_cli):
        env = None

        try:
            # Create the registered DirectRLEnv.
            env = gym.make(
                s54_task.TASK_ID,
                cfg=env_cfg,
            )
            task_env = env.unwrapped
            if not isinstance(task_env, S54FootPlacementEnv):
                raise TypeError("Expected an S54FootPlacementEnv")

            # The wrapper also performs the initial env.reset().
            wrapped_env = S54RslRlVecEnvWrapper(
                task_env,
                clip_actions=agent_cfg.clip_actions,
            )

            # Isaac Lab 3.0 helper creates the PPO Runner.
            runner = create_rsl_rl_runner(
                wrapped_env,
                agent_cfg,
                log_dir=str(log_dir),
            )
            if args_cli.metrics_jsonl:
                runner.cfg["logger"] = logger_cfg
            restart_path = args_cli.resume or args_cli.warm_start
            if restart_path is not None:
                if not restart_path.is_file():
                    raise FileNotFoundError(restart_path)
                old_manifest = restart_path.parent / "manifest.json"
                if args_cli.resume is not None and old_manifest.is_file():
                    old_stage = json.loads(old_manifest.read_text())["args"].get("stage", "legacy")
                    if old_stage != args_cli.stage:
                        raise ValueError("Stage changes require --warm_start, not --resume")
                load_cfg = None if args_cli.resume is not None else {"actor": True}
                migration = load_s54_checkpoint(runner, restart_path, load_cfg=load_cfg, map_location=args_cli.device)
                manifest["restart"] = {"checkpoint": str(restart_path.resolve()),
                                       "sha256": hashlib.sha256(restart_path.read_bytes()).hexdigest(),
                                       "mode": "full_resume" if args_cli.resume else "actor_only",
                                       "observation_migration": migration}
                if args_cli.resume is not None:
                    # RSL 5.5 stores the index of the last completed update.
                    runner.current_learning_iteration += 1
                elif args_cli.stage == "straight":
                    algorithm = runner.alg
                    if not isinstance(algorithm, PPO):
                        raise TypeError("Straight actor transfer requires PPO")
                    normalizer = algorithm.actor.obs_normalizer
                    if isinstance(normalizer, EmpiricalNormalization):
                        preview = wrapped_env.get_observations()
                        with torch.no_grad():
                            initial_actions = algorithm.actor(preview, stochastic_output=False).clone()
                        normalizer_count = normalizer.get_buffer("count")
                        old_count = int(normalizer_count.item())
                        new_count = (min(old_count, args_cli.normalization_prior_rollouts
                                         * args_cli.num_envs * agent_cfg.num_steps_per_env)
                                     if args_cli.normalization_prior_rollouts else old_count)
                        # 默认0保留旧统计；显式设为正数时，仅折算历史样本数。
                        # 均值/方差保持不变，所以“刚加载时”的policy函数不变。
                        normalizer_count.fill_(new_count)
                        with torch.no_grad():
                            rebased_actions = algorithm.actor(preview, stochastic_output=False)
                        # 直接检查动作完全一致，不能只凭“没有修改网络权重”来推断一致。
                        torch.testing.assert_close(rebased_actions, initial_actions, rtol=0, atol=0)
                        manifest["actor_normalization_prior"] = {"old_count": old_count, "new_count": new_count,
                                                                   "mean_std_preserved": True,
                                                                   "max_initial_action_change": (rebased_actions - initial_actions).abs().max().item()}
                        print("Actor normalization history rebased:", old_count, "->", new_count)

            algorithm = runner.alg
            if not isinstance(algorithm, PPO):
                raise TypeError("S54 requires PPO")
            # load()会恢复优化器里的学习率，所以显式覆盖必须在load之后执行。
            if args_cli.learning_rate is not None:
                algorithm.learning_rate = args_cli.learning_rate
                for group in algorithm.optimizer.param_groups:
                    group["lr"] = args_cli.learning_rate
            agent_cfg.algorithm.learning_rate = algorithm.learning_rate
            agent_snapshot = to_dict(agent_cfg)
            if args_cli.metrics_jsonl:
                agent_snapshot["logger"] = logger_cfg
            (log_dir / "agent_cfg.json").write_text(json.dumps(agent_snapshot, default=str, indent=2))
            manifest["effective_learning_rate"] = algorithm.learning_rate
            manifest["loaded_learning_iteration"] = runner.current_learning_iteration
            print("实际PPO学习率：", algorithm.learning_rate, flush=True)

            if args_cli.validate_only:
                # 只检查初始化与完整状态恢复，绝不调用learn或保存新训练模型。
                sample = wrapped_env.get_observations()
                with torch.no_grad():
                    output = algorithm.actor(sample, stochastic_output=False)
                assert output.shape == (args_cli.num_envs, 27) and torch.isfinite(output).all()
                manifest["status"] = "validated_no_training"
                manifest["optimizer_updates"] = 0
                manifest["sampled_step_length_min_m"] = task_env.goal_sampler.step_lengths.min().item()
                manifest["sampled_step_length_max_m"] = task_env.goal_sampler.step_lengths.max().item()
                manifest["observation_dim"] = sample["policy"].shape[-1]
                manifest["left_frequency_min_max_hz"] = [task_env.goal_sampler.left_gait_frequencies.min().item(),
                                                          task_env.goal_sampler.left_gait_frequencies.max().item()]
                manifest["right_frequency_min_max_hz"] = [task_env.goal_sampler.right_gait_frequencies.min().item(),
                                                           task_env.goal_sampler.right_gait_frequencies.max().item()]
                (log_dir / "manifest.json").write_text(json.dumps(manifest, default=str, indent=2))
                print("配置及checkpoint校验通过，PPO更新次数=0。", flush=True)
                return

            warmup_steps = args_cli.normalization_warmup_steps
            if warmup_steps < 0:
                raise ValueError("normalization_warmup_steps must be nonnegative")
            if restart_path is None and warmup_steps:
                algorithm = runner.alg
                if not isinstance(algorithm, PPO):
                    raise TypeError("S54 normalization warmup requires PPO")
                observations = []
                with torch.no_grad():
                    for _ in range(warmup_steps):
                        obs = wrapped_env.get_observations()
                        observations.append(obs["policy"].clone())
                        actions = algorithm.actor(obs, stochastic_output=True)
                        wrapped_env.step(actions)
                    batch = TensorDict({"policy": torch.cat(observations)},
                                       batch_size=[warmup_steps * args_cli.num_envs])
                    algorithm.actor.update_normalization(batch)
                    algorithm.critic.update_normalization(batch)
                wrapped_env.reset()
                print("Normalization warmup observations:", warmup_steps * args_cli.num_envs)
            manifest["normalization_warmup_steps"] = warmup_steps if restart_path is None else 0

            print("\nPPO Runner initialized")
            print("Observation groups:", agent_cfg.obs_groups)
            print("Action count:", wrapped_env.num_actions)

            # Rollout -> GAE -> PPO updates.
            saved_task = json.loads(json.dumps(to_dict(env_cfg), default=str))

            def interrupt_training(signum, frame):
                raise KeyboardInterrupt(f"Received signal {signum}")

            previous_term = signal.signal(signal.SIGTERM, interrupt_training)
            try:
                runner.learn(
                    num_learning_iterations=agent_cfg.max_iterations,
                    init_at_random_ep_len=agent_cfg.init_at_random_ep_len,
                )
            except KeyboardInterrupt:
                runner.save(str(log_dir / "model_interrupted.pt"),
                            infos={"env_cfg": saved_task, "stage": args_cli.stage})
                runner.logger.stop_logging_writer()
                manifest["status"] = "interrupted"
                (log_dir / "manifest.json").write_text(json.dumps(manifest, default=str, indent=2))
                print("Interrupted checkpoint saved:", log_dir / "model_interrupted.pt", flush=True)
                raise
            finally:
                signal.signal(signal.SIGTERM, previous_term)

            # Save an explicit final checkpoint so even
            # a very short smoke run has an output artifact.
            checkpoint = log_dir / "model_final.pt"
            runner.save(str(checkpoint), infos={"env_cfg": saved_task, "stage": args_cli.stage})
            manifest["status"] = "complete"
            if str(args_cli.device).startswith("cuda"):
                free, total = torch.cuda.mem_get_info(args_cli.device)
                manifest["gpu_memory_used_bytes"] = total - free
                manifest["gpu_memory_total_bytes"] = total
            (log_dir / "manifest.json").write_text(json.dumps(manifest, default=str, indent=2))

            if not checkpoint.is_file():
                raise RuntimeError(
                    "PPO checkpoint was not created"
                )

            print("\n===== Training Complete =====")
            print("Checkpoint:", checkpoint)
            print("Size:", checkpoint.stat().st_size, "bytes")
            print("S54 PPO smoke training PASSED")

        finally:
            if env is not None:
                env.close()


if __name__ == "__main__":
    main()
