
"""Smoke test for S54 DirectRLEnv with Newton MJWarp."""

import sys
from pathlib import Path

# 支持直接运行测试文件，无需额外设置PYTHONPATH。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from isaaclab.app import (
    add_launcher_args,
    launch_simulation,
)

parser = argparse.ArgumentParser(
    description="S54 DirectRLEnv smoke test"
)

parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
)

parser.add_argument(
    "--steps",
    type=int,
    default=300,
)

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

    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device

    with launch_simulation(cfg.sim, args_cli):
        env = S54FootPlacementEnv(cfg)

        try:
            obs, extras = env.reset()
            assert isinstance(obs["policy"], torch.Tensor)

            print("\n===== Initial Reset =====")
            print("Observation:", obs["policy"].shape)
            print("Action space:", env.action_space)
            print("Policy dt:", env.step_dt)

            assert obs["policy"].shape == (
                args_cli.num_envs, 103
            )

            for step in range(args_cli.steps):
                # Zero action = validated standing pose.
                actions = torch.zeros(
                    args_cli.num_envs,
                    27,
                    device=env.device,
                )

                obs, rewards, terminated, truncated, extras = (
                    env.step(actions)
                )

                assert isinstance(obs["policy"], torch.Tensor)
                assert obs["policy"].shape == (
                    args_cli.num_envs, 103
                )

                assert torch.isfinite(obs["policy"]).all()
                assert torch.isfinite(rewards).all()

                if step % 50 == 0:
                    root_z = (
                        env.robot.data.root_pos_w.torch[:, 2]
                        - env.scene.env_origins[:, 2]
                    )

                    print(
                        f"step={step:04d} "
                        f"t={step * env.step_dt:.2f}s "
                        f"mean_root_z={root_z.mean().item():.4f} "
                        f"mean_reward={rewards.mean().item():.4f} "
                        f"terminated={terminated.sum().item()} "
                        f"truncated={truncated.sum().item()}"
                    )

            print("\nS54 DirectRLEnv smoke test PASSED")

        finally:
            env.close()


if __name__ == "__main__":
    main()
