
"""Test ground contact forces in the S54 RL environment."""

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

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=100)

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
            env.reset()

            print("\n===== Contact Sensor =====")
            print("Sensors:", env.foot_contact.sensor_names)
            print(
                "Ground filters:",
                env.foot_contact.filter_object_names,
            )

            assert len(env.foot_contact.filter_object_names) > 0

            actions = torch.zeros(
                (env.num_envs, 27),
                device=env.device,
            )

            # First let the robot settle.
            for _ in range(args_cli.steps):
                _, _, terminated, _, _ = env.step(actions)

                assert not terminated.any(), (
                    "Robot unexpectedly fell during contact test"
                )

            left_force, right_force = (
                env._get_ground_contact_forces()
            )

            assert left_force.shape == (env.num_envs, 3)
            assert right_force.shape == (env.num_envs, 3)

            assert torch.isfinite(left_force).all()
            assert torch.isfinite(right_force).all()

            left_fz = left_force[:, 2]
            right_fz = right_force[:, 2]
            total_fz = left_fz + right_fz

            print("\n===== Ground Reaction Forces =====")
            print("Left Fz mean:", left_fz.mean().item(), "N")
            print("Right Fz mean:", right_fz.mean().item(), "N")
            print("Total Fz mean:", total_fz.mean().item(), "N")

            print("Left Fz min/max:",
                  left_fz.min().item(),
                  left_fz.max().item())

            print("Right Fz min/max:",
                  right_fz.min().item(),
                  right_fz.max().item())

            # Loose checks for the known S54 flat-ground stand.
            assert (left_fz > 50.0).all()
            assert (right_fz > 50.0).all()
            assert (total_fz > 500.0).all()
            assert (total_fz < 850.0).all()

            print("\nS54 Foot Contact Test PASSED")

        finally:
            env.close()


if __name__ == "__main__":
    main()
