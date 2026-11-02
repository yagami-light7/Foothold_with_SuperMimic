"""Observation construction for S54 foot-placement RL."""

import torch

from .spec import ACTOR_OBS_DIM

def build_actor_observation(
        projected_gravity:torch.Tensor,
        base_ang_vel:torch.Tensor,
        joint_pos:torch.Tensor,
        joint_vel:torch.Tensor,
        last_action:torch.Tensor,
        goal_obs:torch.Tensor,
        gait_frequency:torch.Tensor |None = None,
        gait_frequencies: torch.Tensor | None = None,
) -> torch.Tensor:
    """构建103/104/105维观测；新任务传入[N, 2]的左右节奏命令。"""
    parts = (
        projected_gravity,
        base_ang_vel,
        joint_pos,
        joint_vel,
        last_action,
    )
    
    expected_dims = (3, 3, 27, 27, 27)

    num_envs = parts[0].shape[0]

    if gait_frequency is not None and gait_frequencies is not None:
        raise ValueError("单步频旧格式与左右节奏命令不能同时传入")
    # 可选的一维步频用于兼容旧104维格式。
    if gait_frequency is not None:
        parts += (gait_frequency,)
        expected_dims += (1,)
    if gait_frequencies is not None:
        parts += (gait_frequencies,)
        expected_dims += (2,)

    # Goal始终放在最后，便于继续使用末尾16维获取足印目标。
    parts += (goal_obs,)
    expected_dims += (16,)

    for tensor, expected_dim in zip(parts, expected_dims):
        # 检查张量维数 (num_envs, expected_dim)是否=2
        if tensor.ndim != 2:
            raise ValueError(
                f"Expected a 2D tensor, got {tensor.shape}"
            )
        # 检查张量形状
        if tensor.shape != (num_envs, expected_dim):
            raise ValueError(
                f"Expected {(num_envs, expected_dim)}, "
                f"got {tensor.shape}"
            )

    obs =torch.cat(parts, dim=-1)

    # ACTOR_OBS_DIM=103表示基础观测；启用步频命令后增加1维。
    expected_obs_dim = ACTOR_OBS_DIM + (
        1 if gait_frequency is not None else 0
    ) + (2 if gait_frequencies is not None else 0)

    if obs.shape != (num_envs, expected_obs_dim):
        raise RuntimeError(
            f"观测形状错误：预期 {(num_envs, expected_obs_dim)}，"
            f"实际 {tuple(obs.shape)}"
        )

    if not torch.isfinite(obs).all():
        raise RuntimeError("Observation contains NaN or Inf")

    return obs
