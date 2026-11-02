"""Reusable KUAVO S54 articulation configuration."""

from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils

from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from pathlib import Path

# usd 
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 外观图层复用SuperMimic的灰阶材质；几何、碰撞、关节和惯量沿用原始USD。
USD_PATH = PROJECT_ROOT / "assets" / "biped_s54_colored.usda"

# Kp、Kd、N*m
JOINT_PARAMS = {
    "leg_[lr]1_joint": (100.0, 5.0, 88.9),
    "leg_[lr]2_joint": (100.0, 5.0, 49.7),
    "leg_[lr]3_joint": (80.0, 5.0, 92.4),
    "leg_[lr]4_joint": (80.0, 6.0, 196.0),
    "leg_[lr]5_joint": (30.0, 7.5, 39.9),
    "leg_[lr]6_joint": (50.0, 7.5, 39.9),

    "waist_yaw_joint": (40.0, 7.0, 71.4),

    "zarm_[lr]1_joint": (15.0, 6.0, 46.2),
    "zarm_[lr]2_joint": (30.0, 3.0, 52.5),
    "zarm_[lr]3_joint": (15.0, 3.0, 39.9),
    "zarm_[lr]4_joint": (30.0, 6.0, 52.5),
    "zarm_[lr][567]_joint": (15.0, 3.0, 9.87),
}

# Preserve the gains that worked in the stand test.
KP_SCALE = 30.0
KD_SCALE = 30.0 ** 0.5


# 此处填入已经验证成功的 S54 机器人配置：
# 1. USD 模型路径
# 2. 初始基座姿态
# 3. 27 个关节的站立目标角度
# 4. 分关节 PD 参数
# 5. 力矩限幅
# 6. Newton Articulation 配置
S54_ROBOT_CFG = ArticulationCfg(
    prim_path="/World/S54",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(USD_PATH),
        fix_root_link=False,  # 自由基座
        articulation_props=NewtonArticulationCfg(
            self_collision_enabled=True, 
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.969025),
        # from kuavo-ros-opensource
        joint_pos={
            "leg_[lr][126]_joint": 0.0,
            "leg_[lr]3_joint": -0.10,     # 髋 pitch
            "leg_[lr]4_joint": 0.25,      # 膝 pitch
            "leg_[lr]5_joint": -0.15,     # 踝 pitch

            "waist_yaw_joint": 0.0,

            "zarm_[lr][134567]_joint": 0.0,
            "zarm_l2_joint": 0.075,      # 左臂略向外展开
            "zarm_r2_joint": -0.075,     # 右臂略向外展开
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            # Actual PD(Based on real test)
            stiffness={
                        name: values[0] * KP_SCALE
                for name, values in JOINT_PARAMS.items()
            },
            damping={
                        name: values[1] * KD_SCALE
                        for name, values in JOINT_PARAMS.items()
                    },
            joint_effort_limit={
                name: values[2]
                for name, values in JOINT_PARAMS.items()
            }
        )
    },
)
