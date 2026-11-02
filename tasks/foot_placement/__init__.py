
"""Gymnasium registration for the S54 Foot Placement task."""

import gymnasium as gym

TASK_ID = "Isaac-S54-FootPlacement-Direct-v0"

if TASK_ID not in gym.registry:
    gym.register(
        id = TASK_ID,
        entry_point=(
            "tasks.foot_placement.foot_placement_env:"
            "S54FootPlacementEnv"
        ),
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                "tasks.foot_placement."
                "foot_placement_env_cfg:"
                "S54FootPlacementEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                "agents.rsl_rl_ppo_cfg:"
                "S54FootPlacementPPORunnerCfg"
            ),
            "default_agent": "rsl_rl",
        }
    )
