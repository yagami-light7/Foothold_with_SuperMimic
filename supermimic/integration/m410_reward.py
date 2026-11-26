"""M4.10 根部跟踪与分关节姿态惩罚的共享定义。"""

import torch


ROOT_TERMS = ('root_xy', 'root_height', 'root_heading')
HUBER_TERMS = ('shoulder_huber', 'elbow_huber', 'waist_huber')
WHOLE_BODY_TERMS = ROOT_TERMS + HUBER_TERMS
SHOULDER_JOINTS = tuple(f'zarm_{side}{i}_joint' for side in 'lr' for i in (1, 2, 3))
POSTURE_JOINTS = (SHOULDER_JOINTS, ('zarm_l4_joint', 'zarm_r4_joint'), ('waist_yaw_joint',))

ROOT_CONFIG = {
    'root_xy': .30, 'root_xy_tolerance': .03, 'root_xy_sigma': .10,
    'root_height': .15, 'root_height_tolerance': .02, 'root_height_sigma': .06,
    'root_heading': .15, 'root_heading_tolerance': .08726646259971647, 'root_heading_sigma': .25,
}
HUBER_CONFIG = {
    'shoulder_huber': .20, 'shoulder_huber_tolerance': .20, 'shoulder_huber_sigma': .50,
    'elbow_huber': .10, 'elbow_huber_tolerance': .25, 'elbow_huber_sigma': .50,
    'waist_huber': .05, 'waist_huber_tolerance': .20, 'waist_huber_sigma': .50,
}
GROUPS = {'a': 'a_root_only', 'b': 'b_huber_prior_only', 'c': 'c_root_huber'}
GROUP_REWARDS = {'a': ROOT_CONFIG, 'b': HUBER_CONFIG, 'c': ROOT_CONFIG | HUBER_CONFIG}


def heading_delta(rotation, reference):
    # 世界水平面内比较两条前向轴，直接得到最短有符号角。
    x, y = rotation[..., :2, 0].unbind(-1)
    rx, ry = reference[..., :2, 0].unbind(-1)
    sine, cosine = y * rx - x * ry, x * rx + y * ry
    # 前向轴竖直时航向无定义，避免 atan2(0, 0) 的反向传播。
    valid = sine.square() + cosine.square() > 1.e-12
    return torch.atan2(sine.masked_fill(~valid, 0.), cosine.masked_fill(~valid, 1.))


def dead_zone_huber(error, tolerance, sigma):
    excess = (error.abs() - tolerance).clamp_min(0.)
    scaled = excess / sigma
    quadratic = scaled.clamp_max(1.)
    return .5 * quadratic.square() + (scaled - quadratic), excess


def root_errors(position, rotation, reference_position, reference_rotation):
    delta = position - reference_position
    return (torch.linalg.vector_norm(delta[..., :2], dim=-1),
            delta[..., 2].abs(), heading_delta(rotation, reference_rotation).abs())


def tolerance_metric_names(term):
    return f'{term}_excess_rad', f'{term}_outside_ratio'
