
"""Integrated smoke test: S54 DirectRLEnv + GoalSampler."""

import sys
from pathlib import Path

# 支持直接运行测试文件，无需额外设置PYTHONPATH。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)

add_launcher_args(parser)
parser.set_defaults(device="cuda:0")
args_cli = parser.parse_args()

import torch

from tasks.foot_placement.foot_placement_env_cfg import (
    S54FootPlacementEnvCfg,
)
from tasks.foot_placement.foot_placement_env import (
    S54FootPlacementEnv,
)


def main():
    cfg = S54FootPlacementEnvCfg()
    cfg.initial_swing_side = "left"  # 此回归固定检查原来的左脚起步目标。
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device

    with launch_simulation(cfg.sim, args_cli):
        env = S54FootPlacementEnv(cfg)

        try:
            obs, _ = env.reset()
            assert isinstance(obs["policy"], torch.Tensor)
            sampler = env.goal_sampler

            # Validate sole-frame geometry.
            left_sole_pos, _, right_sole_pos, _ = (
                env._get_foot_poses()
            )

            body_pos = env.robot.data.body_pos_w.torch

            left_link_z = body_pos[:, env.left_foot_id, 2]
            right_link_z = body_pos[:, env.right_foot_id, 2]

            left_sole_z = left_sole_pos[:, 2]
            right_sole_z = right_sole_pos[:, 2]

            print("\n===== Sole Frame Calibration =====")
            print("Left link Z:", left_link_z[0].item())
            print("Left sole Z:", left_sole_z[0].item())
            print("Right link Z:", right_link_z[0].item())
            print("Right sole Z:", right_sole_z[0].item())

            assert torch.isfinite(left_sole_pos).all()
            assert torch.isfinite(right_sole_pos).all()

            # Preliminary sanity check for a robot standing
            # on a flat ground plane at world z = 0.
            assert (left_sole_z.abs() < 0.05).all()
            assert (right_sole_z.abs() < 0.05).all()

            print("Sole frame calibration test PASSED")

            actions = torch.zeros(
                (env.num_envs, 27),
                device=env.device,
            )

            assert obs["policy"].shape == (env.num_envs, 103)
            assert torch.all(sampler.phase == 0.0)
            assert torch.all(sampler.left_is_swing)

            print("\n===== Initial Goal =====")
            print("Phase:", sampler.phase[0].item())
            print("Left swing:", sampler.left_is_swing[0].item())
            print("Goal:", obs["policy"][0, -16:].tolist())

            initial_goal = obs["policy"][:, -16:]

            # Left foot target: [0.12, 0.19, 0.0]
            expected_left = initial_goal.new_tensor(
                [0.12, 0.19, 0.0]
            )

            assert torch.allclose(
                initial_goal[:, 0:3],
                expected_left.expand(env.num_envs, -1),
                atol=1e-3,
            ), "Initial left foot goal is incorrect"

            # Right stance foot target: [0, 0, 0]
            assert torch.allclose(
                initial_goal[:, 7:10],
                torch.zeros_like(initial_goal[:, 7:10]),
                atol=1e-3,
            ), "Initial right stance foot goal is incorrect"

            print("Initial goal coordinate test PASSED")

            # 26 policy steps = 0.52 seconds.
            for _ in range(26):
                obs, rewards, terminated, truncated, _ = (
                    env.step(actions)
                )

                assert isinstance(obs["policy"], torch.Tensor)
                assert not terminated.any()
                assert not truncated.any()
                assert torch.isfinite(obs["policy"]).all()
                assert torch.isfinite(rewards).all()

            # We should now be in the right-swing half-cycle.
            assert torch.all(sampler.phase > 0.5)
            assert not torch.any(sampler.left_is_swing)

            print("\n===== After 0.52 s =====")
            print("Phase:", sampler.phase[0].item())
            print("Left swing:", sampler.left_is_swing[0].item())
            print("Goal:", obs["policy"][0, -16:].tolist())

            # Trigger one timeout to verify partial autoreset.
            # Requires at least three parallel environments.
            if env.num_envs >= 3:
                env.episode_length_buf[2] = (
                    env.max_episode_length - 1
                )

                old_phase_0 = sampler.phase[0].clone()

                obs, rewards, terminated, truncated, _ = (
                    env.step(actions)
                )

                assert isinstance(obs["policy"], torch.Tensor)
                assert truncated[2].item()
                assert sampler.phase[2].item() == 0.0
                assert sampler.left_is_swing[2].item()

                # Environment 0 continues; only env 2 resets.
                assert sampler.phase[0] > old_phase_0
                assert not sampler.left_is_swing[0].item()

                expected_phase_obs = torch.tensor(
                    [1.0, 0.0],
                    device=env.device,
                )

                assert torch.allclose(
                    obs["policy"][2, -2:],
                    expected_phase_obs,
                    atol=1e-5,
                )

                print("\n===== Partial Autoreset =====")
                print("Env 0 phase:", sampler.phase[0].item())
                print("Env 2 phase:", sampler.phase[2].item())
                print(
                    "Env 2 left swing:",
                    sampler.left_is_swing[2].item(),
                )

            print("\nS54 Goal Integration Test PASSED")

        finally:
            env.close()


if __name__ == "__main__":
    main()
