"""Explicit S54 flat-ground curriculum; stages advance only after evaluation."""

from dataclasses import replace
from typing import TYPE_CHECKING

from tasks.foot_placement.rewards import FootRewardCfg, reward_config_from_dict
from tasks.foot_placement.spec import JOINT_NAMES, actor_observation_dim
from configs.walk_training import PROFILE_DEFAULTS

if TYPE_CHECKING:
    from agents.rsl_rl_ppo_cfg import S54FootPlacementPPORunnerCfg
    from tasks.foot_placement.foot_placement_env_cfg import S54FootPlacementEnvCfg

STAGES = ("legacy", "stand", "lift", "step", "walk", "straight")
ABLATIONS = ("none", "legacy_reward", "legacy_scale", "no_smoothing", "targets_only", "rewards_only")
TASK_FIELDS = ("curriculum_stage", "action_scale", "action_scale_by_joint", "action_smoothing",
            "gait_frequency", "gait_frequency_range", "observe_gait_frequency",
            "gait_frequency_obs_dim", "left_gait_frequency", "right_gait_frequency",
            "left_gait_frequency_range", "right_gait_frequency_range",
            "step_length", "step_length_range", "feet_distance", "retain_stance_targets",
            "planar_targets", "flat_ground_height",
            "fixed_heading_targets",
            "termination_height", "min_upright_projection", "episode_length_s", *PROFILE_DEFAULTS)


def configure_stage(env: "S54FootPlacementEnvCfg", agent: "S54FootPlacementPPORunnerCfg", stage: str,
                    ablation: str = "none") -> None:
    if stage not in STAGES:
        raise ValueError(f"Unknown curriculum stage {stage!r}; choose {STAGES}")
    if stage == "legacy":
        return
    if ablation not in ABLATIONS:
        raise ValueError(f"Unknown ablation {ablation!r}")
    env.curriculum_stage = stage
    env.step_length = {"stand": 0.0, "lift": 0.0, "step": 0.03, "walk": 0.06, "straight": 0.03}[stage]
    env.step_length_range = None
    env.gait_frequency = 0.75
    # 课程默认使用固定步频；随机范围后续由实验参数指定。
    env.gait_frequency_range = None
    # 原有课程默认使用103维观测，新实验再显式启用。
    env.observe_gait_frequency = False
    env.gait_frequency_obs_dim = 2
    env.left_gait_frequency = env.right_gait_frequency = None
    env.left_gait_frequency_range = env.right_gait_frequency_range = None
    sync_observation_space(env)

    env.retain_stance_targets = True
    env.planar_targets = True
    env.flat_ground_height = 0.0
    env.fixed_heading_targets = False
    env.action_smoothing = 0.2
    scales = []
    for name in JOINT_NAMES:
        if name.startswith("leg_"):
            index = int(name.split("_joint")[0][-1])
            scales.append({1: 0.1, 2: 0.15, 3: 0.3, 4: 0.6, 5: 0.3, 6: 0.1}[index])
        else:
            scales.append(0.05)
    env.action_scale_by_joint = tuple(scales)
    env.reward_cfg = FootRewardCfg(
        swing_xy_w=4.0, stance_xy_w=0.5, clearance_w=4.0,
        swing_air_w=3.0, stance_contact_w=0.5, touchdown_w=2.0,
        upright_w=0.5, action_rate_w=0.05,
        swing_height=0.02 if stage in ("lift", "step") else 0.03,
        relative_clearance=True, unloading_w=1.0, normalize_action_rate=True,
        require_stance_support=True, require_liftoff_for_touchdown=True,
    )
    if stage == "lift":
        env.reward_cfg = replace(env.reward_cfg, swing_xy_w=1.0)
    if stage == "stand":
        env.reward_cfg = replace(env.reward_cfg, swing_height=0.0, clearance_w=0.0,
                                swing_air_w=0.0, unloading_w=0.0, touchdown_w=0.0)
    if stage == "straight":
        # 动作权限和足端奖励权重沿用step；注意上方straight的抬脚目标为0.03m，step为0.02m。
        # 水平偏移仍为0.03m，均与sampler的默认0.12m相互独立；回放以checkpoint配置为准。
        env.fixed_heading_targets = True
        env.reward_cfg = replace(env.reward_cfg, heading_w=1.0, heading_sigma=0.35,
                                yaw_rate_w=0.5, yaw_rate_sigma=0.30)
    agent.num_steps_per_env = 64
    agent.actor.distribution_cfg = type(agent.actor).GaussianDistributionCfg(init_std=0.15)
    agent.algorithm.learning_rate = 1e-4
    agent.algorithm.schedule = "fixed"
    agent.algorithm.entropy_coef = 0.002
    agent.algorithm.num_learning_epochs = 4
    agent.algorithm.num_mini_batches = 4
    agent.init_at_random_ep_len = False
    if ablation == "legacy_reward":
        env.reward_cfg = replace(FootRewardCfg(), swing_height=env.reward_cfg.swing_height)
    elif ablation == "legacy_scale":
        env.action_scale_by_joint = None
    elif ablation == "no_smoothing":
        env.action_smoothing = 1.0
    elif ablation in ("targets_only", "rewards_only"):
        # 单因素消融：区分“目标方向变了”和“方向奖励变了”各自带来的影响。
        if stage != "straight":
            raise ValueError("Direction ablations require stage=straight")
        if ablation == "targets_only":
            env.reward_cfg = replace(env.reward_cfg, heading_w=0.0, yaw_rate_w=0.0)
        else:
            env.fixed_heading_targets = False


def restore_task_config(env: "S54FootPlacementEnvCfg", saved: dict) -> None:
    """从保存的配置恢复任务参数，用于回放和续训。"""

    # 旧配置没有这些字段时，恢复旧任务的行为。
    env.fixed_heading_targets = saved.get("fixed_heading_targets", False)
    env.step_length_range = None
    env.gait_frequency_range = None
    
    # 旧配置缺少此字段时，恢复103维观测模式。
    env.observe_gait_frequency = False
    # 已保存的104维任务没有dim字段；按原来的单频率观测恢复。
    env.gait_frequency_obs_dim = 1 if saved.get("observe_gait_frequency", False) else 2
    env.left_gait_frequency = env.right_gait_frequency = None
    env.left_gait_frequency_range = env.right_gait_frequency_range = None

    for name in TASK_FIELDS:
        if name in saved:
            value = saved[name]

            # JSON 会把元组保存为列表，读取后转回配置约定的元组。
            if name in (
                "action_scale_by_joint",
                "step_length_range",
                "gait_frequency_range",
                "left_gait_frequency_range",
                "right_gait_frequency_range",
            ) and value is not None:
                value = tuple(value)

            setattr(env, name, value)

    if "reward_cfg" in saved:
        env.reward_cfg = reward_config_from_dict(saved["reward_cfg"])
    # 旧checkpoint没有新字段；恢复时明确关闭新功能，避免继承构造前的临时profile。
    for name, default in PROFILE_DEFAULTS.items():
        if name not in saved:
            setattr(env, name, default)
    sync_observation_space(env)


def sync_observation_space(env: "S54FootPlacementEnvCfg") -> int:
    """在导出配置、建立Gym空间、创建网络之前统一计算输入维度。"""
    env.observation_space = actor_observation_dim(env.observe_gait_frequency, env.gait_frequency_obs_dim)
    return env.observation_space
