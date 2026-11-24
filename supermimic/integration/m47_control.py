"""F2 上身起步目标与动作平滑分项的公共计算。"""

import torch


def upper_joint_indices(names):
    indices = [i for i, name in enumerate(names) if name == 'waist_yaw_joint' or name.startswith('zarm_')]
    if len(indices) != 9:
        raise ValueError('M4.7 应包含一腰、八臂共九个上身关节')
    return indices


def blend_startup_target(target, reset_q, tick, upper_mask, ticks):
    # 两端都在硬限位内。到达结束帧时精确恢复原目标，并保留下肢指令。
    u = (tick.to(target.dtype) / ticks).clamp(0., 1.)
    beta = (u.square() * (3. - 2. * u)).unsqueeze(-1)
    return torch.where((tick < ticks).unsqueeze(-1) & upper_mask, torch.lerp(reset_q, target, beta), target)


def upper_action_cost(action, previous, previous2, valid, indices):
    delta = action[..., indices] - 2. * previous[..., indices] + previous2[..., indices]
    return torch.where(valid, delta.square().mean(-1), 0.)
