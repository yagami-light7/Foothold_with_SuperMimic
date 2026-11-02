
"""Phase-resolved reward diagnostics for S54 + Newton."""

import argparse
import math

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=200)
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

            actions = torch.zeros(
                env.num_envs, 27, device=env.device
            )

            term_names = [
                "swing_xy",
                "stance_xy",
                "clearance",
                "swing_air",
                "stance_contact",
                "touchdown",
                "upright",
                "action_rate",
            ]

            # Five bins over one normalized swing phase [0, 1].
            bins = [
                {name: [] for name in term_names}
                for _ in range(5)
            ]

            rewards_all = []
            reset_count = 0

            print("\n===== Reward Diagnostics =====")

            for step in range(args_cli.steps):
                # Capture the phase used to calculate this reward.
                phase = env.goal_sampler.phase[0].item()
                local_phase = (phase % 0.5) * 2.0
                bin_id = min(int(local_phase * 5), 4)

                actions.zero_()

                # Keep environments 1..N-1 as zero-action
                # standing baselines.
                #
                # In environment 0 only, apply a small
                # sinusoidal perturbation to the left knee.
                if step >= 10:
                    actions[0, 3] = (
                        0.1
                        * math.sin(
                            2.0 * math.pi * step / 50.0
                        )
                    )

                obs, rewards, terminated, truncated, _ = (
                    env.step(actions)
                )

                assert isinstance(obs["policy"], torch.Tensor)
                assert torch.isfinite(obs["policy"]).all()
                assert torch.isfinite(rewards).all()

                reset_count += (
                    terminated.sum().item()
                    + truncated.sum().item()
                )

                # Record environment 1 (zero-action baseline)
                # when parallel environments are available.
                baseline_id = 1 if env.num_envs > 1 else 0

                for name in term_names:
                    value = env.reward_terms[name][
                        baseline_id
                    ].item()
                    bins[bin_id][name].append(value)

                rewards_all.append(
                    rewards[baseline_id].item()
                )

                if step % 25 == 0:
                    left_f, right_f = (
                        env._get_ground_contact_forces()
                    )

                    print(
                        f"step={step:03d} "
                        f"phase={phase:.3f} "
                        f"local={local_phase:.3f} "
                        f"reward={rewards[baseline_id].item():.5f} "
                        f"Fz_L={left_f[baseline_id, 2].item():.1f} "
                        f"Fz_R={right_f[baseline_id, 2].item():.1f}"
                    )

            print("\n===== Reward by Local Gait Phase =====")

            for i, data in enumerate(bins):
                print(f"\nPhase bin [{i/5:.1f}, {(i+1)/5:.1f})")

                for name in term_names:
                    values = data[name]
                    mean = (
                        sum(values) / len(values)
                        if values else float("nan")
                    )
                    print(f"  {name:16s}: {mean:.5f}")

            mean_reward = (
                sum(rewards_all) / len(rewards_all)
            )

            print("\n===== Summary =====")
            print("Mean standing reward:", mean_reward)
            print("Reset events:", reset_count)
            print("Policy dt:", env.step_dt)

            if reset_count == 0:
                print("Standing baseline stable")

            print("\nReward diagnostics PASSED")

        finally:
            env.close()


if __name__ == "__main__":
    main()
