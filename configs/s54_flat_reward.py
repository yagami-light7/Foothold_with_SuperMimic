"""S54平地奖励v2：参考Mind Your Steps的类别，数值按本机控制接口设置。

此文件只依赖标准库，SwanLab监督器校验JSON时不会加载Torch/CUDA。
旧奖励仍可用于重放旧checkpoint；新实验通过profile显式选择本配置。
"""

from dataclasses import dataclass


@dataclass
class FlatWalkRewardCfg:
    profile: str = "flat_walk_v2"
    tracking_xy_w: float = 4.0
    tracking_yaw_w: float = 0.5
    # 默认1保持旧checkpoint奖励；试训可只增强摆动脚，避免支撑脚高分掩盖落点误差。
    swing_xy_multiplier: float = 1.0
    first_swing_xy_multiplier: float = 1.0  # 仅reset后的第一个半周期生效。
    swing_xy_coarse_weight: float = 0.0  # 默认关闭；混入宽距离核，让20–30cm误差仍有学习信号。
    swing_xy_coarse_sharpness: float = 25.0
    swing_phase_w: float = 4.0
    stance_slip_w: float = 4.0
    stance_hop_w: float = 4.0
    body_height_w: float = 10.0
    body_vz_w: float = 0.5
    posture_w: float = 0.5
    action_rate_w: float = 0.05
    heading_w: float = 1.0
    yaw_rate_w: float = 0.5

    xy_sharpness: float = 100.0
    yaw_sharpness: float = 25.0
    posture_sharpness: float = 4.0
    heading_sigma: float = 0.35
    yaw_rate_sigma: float = 0.30
    swing_height: float = 0.03  # m：窗口内充分抬脚的高度；不追踪整条Z正弦轨迹。
    swing_window_half_width: float = 0.10  # 完整phase中的半窗，中心为0.25/0.75。
    contact_threshold: float = 20.0  # N，地面滤波后的足端法向力。
    support_height_tolerance: float = 0.025  # m：正常足掌滚动时足底中心可升高约2.4cm。
    hop_height_scale: float = 0.02  # m：超过容差后的支撑脚离地惩罚尺度。
    hop_velocity_tolerance: float = 0.05  # m/s：支撑脚竖直速度死区。
    hop_velocity_scale: float = 0.50  # m/s；二次惩罚截断，限制偶发碰撞的数值幅度。
