"""左右摆动节奏与相位计算；纯张量函数，不依赖仿真后端。"""

import math

import torch


def frequency_range(value, name: str):
    """统一检查频率范围；频率f对应本侧摆动时长1/(2f)。"""
    if value is None:
        return None
    low, high = value
    if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high):
        raise ValueError(f"{name}必须满足0 < 下限 <= 上限，且均为有限数")
    return float(low), float(high)


def effective_frequency(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """完整左右循环频率：1 / (左侧时长 + 右侧时长)。"""
    # 等频时直接返回原值，保留旧固定频率的浮点计算结果。
    return torch.where(left == right, left, 2 * left * right / (left + right))


def advance_phase(phase: torch.Tensor, dt: float,
                  left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """推进相位；调用方保证dt*最大频率<0.5，单步至多跨一个换脚边界。"""
    is_left = phase < 0.5
    current = torch.where(is_left, left, right).to(dtype=phase.dtype)
    following = torch.where(is_left, right, left).to(dtype=phase.dtype)
    boundary = torch.where(is_left, 0.5, 1.0)
    ordinary = phase + dt * current
    # 过边界后，剩余的是时间，必须按下一侧频率重新转换为相位增量。
    remaining_time = dt - (boundary - phase) / current
    corrected = boundary + remaining_time * following
    # 左右相同时沿用旧算式，避免边界处无意义的浮点差异。
    crosses_asymmetric = (ordinary >= boundary) & (left != right)
    return torch.remainder(torch.where(crosses_asymmetric, corrected, ordinary), 1.0)
