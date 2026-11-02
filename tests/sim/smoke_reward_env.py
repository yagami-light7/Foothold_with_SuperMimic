
"""Integrated smoke test for S54 foot-placement rewards."""

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
parser.add_argument("--steps", type=int, default=110)

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
    cfg.initial_swing_side = "left"  # 固定相位断言；随机起步由独立测试覆盖。
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device

    with launch_simulation(cfg.sim, args_cli):
        env = S54FootPlacementEnv(cfg)

        try:
            obs, _ = env.reset()

            assert isinstance(obs["policy"], torch.Tensor)
            assert obs["policy"].shape == (
                env.num_envs, 103
            )

            actions = torch.zeros(
                env.num_envs,
                27,
                device=env.device,
            )

            last_swing = (
                env.goal_sampler.left_is_swing.clone()
            )

            print("\n===== Reward Integration =====")
            print("Environments:", env.num_envs)
            print("Policy dt:", env.step_dt)

            for step in range(args_cli.steps):
                actions.zero_()

                # Small action perturbation in environment 0.
                # 0.05 * action_scale(0.1) = 0.005 rad.
                if step == 30:
                    actions[0, 0] = 0.05

                # Capture the phase used for this transition.
                reward_phase = (
                    env.goal_sampler.phase[0].item()
                )

                obs, rewards, terminated, truncated, _ = (
                    env.step(actions)
                )

                assert isinstance(obs["policy"], torch.Tensor)
                assert obs["policy"].shape == (
                    env.num_envs, 103
                )
                assert rewards.shape == (env.num_envs,)

                assert torch.isfinite(
                    obs["policy"]
                ).all()

                assert torch.isfinite(rewards).all()
                assert not terminated.any()
                assert not truncated.any()

                # The action-rate penalty should activate
                # when we introduce a new action.
                if step == 30:
                    assert (
                        env.reward_terms["action_rate"][0]
                        < 0.0
                    )
                    print(
                        "Action-rate penalty test PASSED"
                    )

                current_swing = (
                    env.goal_sampler.left_is_swing
                )

                if torch.any(current_swing != last_swing):
                    print(
                        f"Swing switch: "
                        f"reward_phase={reward_phase:.3f} "
                        f"next_phase="
                        f"{env.goal_sampler.phase[0].item():.3f}"
                    )

                last_swing = current_swing.clone()

                if step % 25 == 0:
                    left_f, right_f = (
                        env._get_ground_contact_forces()
                    )

                    print(
                        f"\nstep={step:03d} "
                        f"reward={rewards.mean().item():.5f} "
                        f"phase="
                        f"{env.goal_sampler.phase[0].item():.3f}"
                    )

                    print(
                        "Contact Fz:",
                        round(left_f[:, 2].mean().item(), 2),
                        round(right_f[:, 2].mean().item(), 2),
                    )

                    for name, value in env.reward_terms.items():
                        print(
                            f"  {name}: "
                            f"{value.mean().item():.5f}"
                        )

            # Verify the action history is cleared by
            # partial autoreset.
            if env.num_envs >= 3:
                env._swing_has_lifted[[0, 2]] = True
                env.episode_length_buf[2] = (
                    env.max_episode_length - 1
                )

                actions.zero_()

                _, rewards, _, truncated, _ = (
                    env.step(actions)
                )

                assert truncated[2].item()
                assert torch.all(
                    env.prev_actions[2] == 0.0
                )
                assert env.goal_sampler.phase[2].item() == 0.0
                assert not env._swing_has_lifted[2].item()
                assert env._swing_has_lifted[0].item()

                print("\nReward history reset test PASSED")

            print("\nS54 Reward Integration Test PASSED")

        finally:
            env.close()


if __name__ == "__main__":
    main()
