"""S54 foot-placement RL environment configuration."""

import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass, replace
from isaaclab.sensors import ContactSensorCfg
from configs.s54_robot_cfg import S54_ROBOT_CFG
from configs.s54_physics_cfg import (
    S54_SIM_CFG,
    CONTROL_DECIMATION,
)
from tasks.foot_placement.spec import(
    ACTION_DIM,
    ACTOR_OBS_DIM
)
from tasks.foot_placement.rewards import FootRewardCfg, FlatWalkRewardCfg

# 仿真场景
@configclass
class S54SceneCfg(InteractiveSceneCfg):
    """One S54 robot per cloned environment."""

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    robot:ArticulationCfg = replace(
        S54_ROBOT_CFG,
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0)
    )

    foot_contact:ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/leg_[lr]6_link",
        filter_shape_prim_expr=["/World/Ground/.*"],
        update_period=0.0,
        history_length=0,
    )

# 强化学习任务
@configclass
class S54FootPlacementEnvCfg(DirectRLEnvCfg):
    """Minimal RL environment for the KUAVO S54."""

    seed = 42

    # 500 Hz physics / 10 = 50 Hz policy.
    decimation = CONTROL_DECIMATION
    episode_length_s = 10.0

    # Gymnasium spaces.
    action_space = ACTION_DIM
    observation_space = ACTOR_OBS_DIM
    state_space = 0

    # Start with small joint position offsets.
    action_scale = 0.1  # rad 动作缩放
    action_scale_by_joint: tuple[float, ...] | None = None
    action_smoothing: float = 1.0
    curriculum_stage: str = "legacy"
    # 默认gait频率:1.0Hz
    gait_frequency: float = 1.0
    # 每个环境每次reset独立选择起始摆动脚；相位从该侧半周期起点开始。
    # 旧checkpoint恢复时回退left，可用CLI显式覆盖为random。
    initial_swing_side: str = "random"
    # 随机步频范围，单位 Hz。
    # None表示固定步频，否则每个episode重置时采样一次共同步频。
    gait_frequency_range: tuple[float,float] | None = None

    # f_L/f_R是本侧半周期的等效频率，摆动时长分别为1/(2*f_L/R)。
    # None沿用共同步频；每侧可选择固定值或独立随机范围。
    left_gait_frequency: float | None = None
    right_gait_frequency: float | None = None
    left_gait_frequency_range: tuple[float, float] | None = None
    right_gait_frequency_range: tuple[float, float] | None = None

    # 新任务观测统一使用左右两维命令；1仅用于恢复旧104维格式。
    # 默认关闭，兼容已有103维checkpoint。
    observe_gait_frequency: bool = False
    gait_frequency_obs_dim: int = 2

    step_length: float = 0.12
    step_length_range: tuple[float, float] | None = None  # m；None为固定步长，否则每个episode独立均匀采样。
    # 混合任务及抗扰动默认关闭，旧checkpoint恢复后维持原来的行为。
    command_groups: list[dict] | None = None
    command_resample_half_cycles: int = 0
    command_max_length_change: float = 0.02
    command_max_frequency_change: float = 0.10
    command_warmup_half_cycles: int = 0
    observation_noise_scale: float = 0.0
    reset_joint_position_noise: float = 0.0
    reset_root_velocity_noise: float = 0.0
    body_mass_scale_range: tuple[float, float] | None = None
    foot_friction_range: tuple[float, float] | None = None
    push_interval_s: tuple[float, float] | None = None
    push_velocity_max: float = 0.0
    feet_distance: float = 0.19
    retain_stance_targets: bool = False
    planar_targets: bool = False
    flat_ground_height: float | None = None
    fixed_heading_targets: bool = False  # 足印是否沿reset记录的根部朝向生成；旧阶段默认关闭。
    reward_cfg: FootRewardCfg | FlatWalkRewardCfg = FootRewardCfg()

    # Temporary termination thresholds.
    termination_height = 0.55 # 跌倒阈值
    min_upright_projection = 0.5

    sim = replace(S54_SIM_CFG)

    scene:S54SceneCfg = S54SceneCfg(
        num_envs=1,
        env_spacing=3.0,
        replicate_physics=True,
        filter_collisions=True,
    )
