"""RSL-RL PPO configuration for S54 Foot Placement."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

@configclass
class S54FootPlacementPPORunnerCfg(RslRlOnPolicyRunnerCfg):

    seed = 42
    device = "cuda:0"

    experiment_name = "s54_foot_placement"
    run_name = "smoke"

    # 16 environment * 32 steps = 512 samples / update
    num_steps_per_env = 32
    max_iterations = 3
    save_interval = 250

    # Keep episode lengths deterministic for first debug run.
    init_at_random_ep_len = False

    # Both networks initially use the same 103D observations.
    obs_groups = {
        "actor":["policy"],
        "critic":["policy"]
    }

    clip_actions = 1.0
    logger = "tensorboard"
    check_for_nan = True

    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=(
            RslRlMLPModelCfg.GaussianDistributionCfg(
                init_std=0.35,
            )
        )
    )

    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef = 1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,

        entropy_coef=0.005,

        num_learning_epochs=4,
        num_mini_batches=4,

        learning_rate=3.0e-4,
        schedule="adaptive",

        gamma=0.995,
        lam=0.95,

        desired_kl=0.01,
        max_grad_norm=1.0,
    )
