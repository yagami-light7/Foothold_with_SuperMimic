
"""Unit tests for the S54 foot-placement reward."""

import sys
from pathlib import Path

# 支持直接运行测试文件，无需额外设置PYTHONPATH。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from tasks.foot_placement.rewards import (
    FootRewardCfg,
    compute_foot_placement_reward,
)


def main():
    device = "cpu"
    cfg = FootRewardCfg()

    n = 4

    left = torch.tensor(
        [[0.0, 0.095, 0.0055]],
        device=device,
    ).repeat(n, 1)

    right = torch.tensor(
        [[0.0, -0.095, 0.0055]],
        device=device,
    ).repeat(n, 1)

    left_target = left.clone()
    left_target[:, 0] += 0.12

    right_target = right.clone()

    left_force = torch.zeros(n, 3, device=device)
    right_force = torch.zeros(n, 3, device=device)
    right_force[:, 2] = 330.0

    gravity = torch.tensor(
        [[0.0, 0.0, -1.0]],
        device=device,
    ).repeat(n, 1)

    actions = torch.zeros(n, 27, device=device)

    def evaluate(
        left_pos,
        left_fz,
        right_fz,
        phase_value,
    ):
        lf = left_force.clone()
        rf = right_force.clone()

        lf[:, 2] = left_fz
        rf[:, 2] = right_fz

        phase = torch.full(
            (n,), phase_value, device=device
        )

        return compute_foot_placement_reward(
            left_pos_w=left_pos,
            right_pos_w=right,
            left_target_w=left_target,
            right_target_w=right_target,
            left_force_w=lf,
            right_force_w=rf,
            phase=phase,
            left_is_swing=torch.ones(
                n, dtype=torch.bool, device=device
            ),
            projected_gravity_b=gravity,
            actions=actions,
            prev_actions=actions,
            dt=0.02,
            cfg=cfg,
        )

    # Test 1: XY tracking improves near the goal.
    far_reward, far_terms = evaluate(
        left, 330.0, 330.0, 0.45
    )

    near = left.clone()
    near[:, 0] += 0.12

    near_reward, near_terms = evaluate(
        near, 330.0, 330.0, 0.45
    )

    assert torch.all(
        near_terms["swing_xy"] > far_terms["swing_xy"]
    )
    assert torch.all(near_reward > far_reward)

    print("Swing XY tracking test PASSED")

    # Test 2: Mid-swing clearance rewards lifting.
    lifted = left.clone()
    lifted[:, 2] += cfg.swing_height

    _, ground_terms = evaluate(
        left, 0.0, 330.0, 0.25
    )

    _, lifted_terms = evaluate(
        lifted, 0.0, 330.0, 0.25
    )

    assert torch.all(
        lifted_terms["clearance"]
        > ground_terms["clearance"]
    )

    print("Swing clearance test PASSED")

    # Test 3: Swing air reward requires no contact.
    _, contact_terms = evaluate(
        left, 330.0, 330.0, 0.25
    )

    _, air_terms = evaluate(
        left, 0.0, 330.0, 0.25
    )

    assert torch.all(
        air_terms["swing_air"] > contact_terms["swing_air"]
    )

    print("Swing air-time test PASSED")

    # Test 4: Stance foot must retain ground contact.
    _, support_terms = evaluate(
        left, 0.0, 330.0, 0.25
    )

    _, no_support_terms = evaluate(
        left, 0.0, 0.0, 0.25
    )

    assert torch.all(
        support_terms["stance_contact"]
        > no_support_terms["stance_contact"]
    )

    print("Stance contact test PASSED")

    # Test 5: Late touchdown near the target.
    _, good_touchdown = evaluate(
        near, 330.0, 330.0, 0.45
    )

    _, bad_touchdown = evaluate(
        left, 330.0, 330.0, 0.45
    )

    assert torch.all(
        good_touchdown["touchdown"]
        > bad_touchdown["touchdown"]
    )

    print("Touchdown test PASSED")

    # Tensor validation.
    assert near_reward.shape == (n,)
    assert torch.isfinite(near_reward).all()

    print("\nS54 Foot Placement Reward test PASSED")


if __name__ == "__main__":
    main()
