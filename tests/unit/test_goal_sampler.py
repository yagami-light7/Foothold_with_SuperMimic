"""Standalone test for phase-based foot-placement goals."""

import sys
from pathlib import Path

# 支持直接运行测试文件，无需额外设置PYTHONPATH。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from tasks.foot_placement.goal_sampler import (
    FootPlacementGoalSampler,
)


def main():
    num_envs = 4
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    sampler = FootPlacementGoalSampler(
        num_envs=num_envs,
        device=device,
        gait_frequency=1.0,
        step_length=0.12,
        feet_distance=0.19,
    )

    # Synthetic standing foot poses.
    left_pos = torch.tensor(
        [[0.0, 0.095, 0.061]],
        device=device,
    ).expand(num_envs, -1).clone()

    right_pos = torch.tensor(
        [[0.0, -0.095, 0.061]],
        device=device,
    ).expand(num_envs, -1).clone()

    identity_quat = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0]],
        device=device,
    ).expand(num_envs, -1).clone()

    sampler.reset(
        left_pos, identity_quat,
        right_pos, identity_quat,
    )

    initial_obs = sampler.get_observation(
        left_pos, identity_quat,
        right_pos, identity_quat,
    )

    assert initial_obs.shape == (num_envs, 16)
    assert torch.allclose(
        initial_obs[0, :3],
        torch.tensor([0.12, 0.19, 0.0], device=device),
        atol=1e-5,
    )

    print("t=0.00")
    print("  phase:", sampler.phase[0].item())
    print("  left swing:", sampler.left_is_swing[0].item())
    print("  left target:", initial_obs[0, :3].tolist())

    # Cross the half-cycle boundary.
    for _ in range(26):
        sampler.step(
            0.02,
            left_pos, identity_quat,
            right_pos, identity_quat,
        )

    switched_obs = sampler.get_observation(
        left_pos, identity_quat,
        right_pos, identity_quat,
    )

    assert not sampler.left_is_swing[0].item()
    assert torch.allclose(
        switched_obs[0, 7:10],
        torch.tensor([0.12, -0.19, 0.0], device=device),
        atol=1e-5,
    )

    print("\nt=0.52")
    print("  phase:", sampler.phase[0].item())
    print("  left swing:", sampler.left_is_swing[0].item())
    print("  right target:", switched_obs[0, 7:10].tolist())

    # Reset only environment 2.
    sampler.reset(
        left_pos, identity_quat,
        right_pos, identity_quat,
        env_ids=torch.tensor([2], device=device),
    )

    assert sampler.phase[2].item() == 0.0
    assert sampler.left_is_swing[2].item()
    assert not sampler.left_is_swing[0].item()

    print("\nPartial reset test PASSED")
    print("FootPlacementGoalSampler test PASSED")


if __name__ == "__main__":
    main()
