"""朝向计算工具：四元数统一使用 Isaac Lab / Newton 的 xyzw 顺序，角度单位为弧度。"""

import torch


def yaw_from_quat(quaternion: torch.Tensor) -> torch.Tensor:
    """从 [..., 4] 四元数提取绕世界竖直轴的 yaw，返回范围为 [-π, π]。"""
    x, y, z, w = quaternion.unbind(-1)
    # 这里的顺序是 x、y、z、w，不能按部分旧代码中的 wxyz 顺序解包。
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))


def quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    """生成只含 yaw 的四元数，不把机身或支撑脚的 roll/pitch 带进平地足印。"""
    zero = torch.zeros_like(yaw)
    return torch.stack((zero, zero, torch.sin(yaw / 2), torch.cos(yaw / 2)), dim=-1)


def heading_error(quaternion: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """计算“当前朝向 - 参考朝向”的最短有符号角度误差，单位为弧度。"""
    difference = yaw_from_quat(quaternion) - reference
    # 将误差包回 [-π, π]：例如当前 -179°、目标 179°，误差应为 2°，而非 -358°。
    return torch.atan2(difference.sin(), difference.cos())
