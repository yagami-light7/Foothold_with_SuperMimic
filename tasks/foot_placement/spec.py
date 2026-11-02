"""KUAVO S54 foot-placement task specifications."""

# Robot dimension
NUM_JOINTS = 27

# Policy output
ACTION_DIM = 27

# Foot-placement goal:
# left position 3 + left quaternion 4
# + right position 3 + right quaternion 4
# + gait phase 2
GOAL_DIM = 16

# Actor:
# projected gravity 3
# + base angular velocity 3
# + joint position 27
# + joint velocity 27
# + previous action 27
# + foot-placement goal 16
ACTOR_OBS_DIM = 103

# 节奏命令插在身体/动作历史之后、16维Goal之前；旧103维索引保持可迁移。
BODY_OBS_DIM = 87
GAIT_COMMAND_DIM = 2


def actor_observation_dim(observe_frequency: bool, frequency_dim: int = GAIT_COMMAND_DIM) -> int:
    """103为旧任务；104仅兼容单步频旧格式；新节奏任务统一使用105。"""
    if frequency_dim not in (1, 2):
        raise ValueError("gait_frequency_obs_dim只能为1（旧格式）或2（左右命令）")
    return ACTOR_OBS_DIM + (frequency_dim if observe_frequency else 0)

# Canonical joint order from the S54 configuration.
JOINT_NAMES = (
    "leg_l1_joint",
    "leg_l2_joint",
    "leg_l3_joint",
    "leg_l4_joint",
    "leg_l5_joint",
    "leg_l6_joint",
    "leg_r1_joint",
    "leg_r2_joint",
    "leg_r3_joint",
    "leg_r4_joint",
    "leg_r5_joint",
    "leg_r6_joint",
    "waist_yaw_joint",
    "zarm_l1_joint",
    "zarm_l2_joint",
    "zarm_l3_joint",
    "zarm_l4_joint",
    "zarm_l5_joint",
    "zarm_l6_joint",
    "zarm_l7_joint",
    "zarm_r1_joint",
    "zarm_r2_joint",
    "zarm_r3_joint",
    "zarm_r4_joint",
    "zarm_r5_joint",
    "zarm_r6_joint",
    "zarm_r7_joint",
)

assert len(JOINT_NAMES) == NUM_JOINTS


# 建立关节映射，防止神经网络输出和关节顺序不一致
def get_joint_indices(sim_joint_names):
    """Map canonical joint order to simulation indices."""
    sim_joint_names = list(sim_joint_names)

    # 检查是否存在重复关节
    if len(sim_joint_names) != len(set(sim_joint_names)):
        raise ValueError("Duplicated joint names in simulation")

    # 检查是否缺少关节
    missing = set(JOINT_NAMES) - set(sim_joint_names)

    if missing:
        raise ValueError(f"Missing S54 joints: {sorted(missing)}")

    # 返回列表索引
    return [sim_joint_names.index(name) for name in JOINT_NAMES]

