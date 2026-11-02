
"""Smoke test for the S54 actor observation contract."""

import sys
from pathlib import Path

# 支持直接运行测试文件，无需额外设置PYTHONPATH。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from tasks.foot_placement.observations import (
    build_actor_observation,
)
from tasks.foot_placement.spec import (
    ACTION_DIM,
    ACTOR_OBS_DIM,
    GOAL_DIM,
    JOINT_NAMES,
    get_joint_indices,
)


def main():
    num_envs = 16

    gravity = torch.zeros(num_envs, 3)
    gravity[:, 2] = -1.0

    angular_vel = torch.zeros(num_envs, 3)
    joint_pos = torch.zeros(num_envs, 27)
    joint_vel = torch.zeros(num_envs, 27)
    last_action = torch.zeros(num_envs, ACTION_DIM)

    goal = torch.zeros(num_envs, GOAL_DIM)

    # Left foot target
    goal[:, 0:3] = torch.tensor([0.12, 0.19, 0.0])
    goal[:, 6] = 1.0  # xyzw quaternion

    # Right foot target
    goal[:, 13] = 1.0  # xyzw quaternion

    # Phase = 0
    goal[:, 14] = 1.0

    obs = build_actor_observation(
        projected_gravity=gravity,
        base_ang_vel=angular_vel,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        last_action=last_action,
        goal_obs=goal,
    )

    assert obs.shape == (num_envs, ACTOR_OBS_DIM)
    assert len(get_joint_indices(JOINT_NAMES)) == ACTION_DIM

    print("Observation shape:", obs.shape)
    print("Action dimension:", ACTION_DIM)
    print("Goal dimension:", GOAL_DIM)
    print("Canonical joint order verified")
    print("Observation contract test PASSED")


if __name__ == "__main__":
    main()
