"""Roban 足部航向、接触代理与法向角速度。"""

from dataclasses import dataclass
from functools import cache
import math

import mujoco
import numpy as np
import torch

from ..roban import FOOT_BODY_NAMES, foot_geometry, model_contract
from .m410_reward import dead_zone_huber


TERMS = ('stance_yaw_huber', 'stance_angular_slip')
PARAMETERS = {
    'stance_yaw_huber_tolerance': math.radians(3.),
    'stance_yaw_huber_sigma': math.radians(10.),
    'stance_angular_slip_tolerance': .10,
    'stance_angular_slip_sigma': .50,
}
WEIGHTS = {'a': (0., 0.), 'b': (.25, 0.), 'c': (.25, .10)}
EPS = 1.e-6


@cache
def foot_axes():
    # Roban 左右脚均由小半径后跟圆柱及大半径前掌圆柱界定前向。
    model, _ = model_contract()
    axes, points = [], []
    for foot, bounds in zip(FOOT_BODY_NAMES, foot_geometry()):
        body = model.body(foot).id
        active = (model.geom_contype > 0) | (model.geom_conaffinity > 0)
        cylinders = np.flatnonzero((model.geom_bodyid == body) & active
                                   & (model.geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER))
        if len(cylinders) != 2:
            raise ValueError(f'{foot} 需要两枚足底圆柱来识别前掌与后跟')
        heel, toe = sorted(cylinders, key=lambda g: model.geom_size[g, 0])
        direction = model.geom_pos[toe] - model.geom_pos[heel]
        if np.linalg.norm(direction[:2]) <= EPS or abs(direction[2]) > EPS:
            raise ValueError(f'{foot} 的前掌和后跟不在同一足底平面')
        direction /= np.linalg.norm(direction)
        axes.append(direction.tolist())
        points.append([[bounds['heel'], model.geom_pos[heel, 1], bounds['sole'][2]],
                       [bounds['toe'], model.geom_pos[toe, 1], bounds['sole'][2]]])
    return axes, points


def wrapped(angle):
    return torch.atan2(angle.sin(), angle.cos())


def projected_heading(rotation, axis):
    return (rotation @ axis[..., None]).squeeze(-1)[..., :2]


def heading_error(actual, reference, axis):
    u, r = projected_heading(actual, axis), projected_heading(reference, axis)
    # atan2 对向量尺度不敏感。退化投影用确定的 (0, 1) 及零贡献。
    valid = (u.square().sum(-1) > EPS**2) & (r.square().sum(-1) > EPS**2)
    cross = r[..., 0]*u[..., 1] - r[..., 1]*u[..., 0]
    dot = (u*r).sum(-1)
    return torch.atan2(torch.where(valid, cross, 0.), torch.where(valid, dot, 1.)), valid


def euler_angles(rotation, axis):
    f = (rotation @ axis[..., None]).squeeze(-1)
    yaw = torch.atan2(f[..., 1], f[..., 0])
    pitch = torch.atan2(-f[..., 2], f[..., :2].norm(dim=-1))
    roll = torch.atan2(rotation[..., 2, 1], rotation[..., 2, 2])
    return torch.stack((yaw, pitch, roll), dim=-1)


def rotation_error(actual, reference):
    trace = (actual * reference).sum((-1, -2))
    return ((trace-1.)*.5).clamp(-1., 1.).acos()


@dataclass(frozen=True)
class StanceSignals:
    yaw: torch.Tensor
    yaw_valid: torch.Tensor
    omega_error: torch.Tensor
    plan: torch.Tensor
    contact: torch.Tensor
    quality: torch.Tensor

    def raw_rewards(self, cfg):
        yaw, yaw_excess = dead_zone_huber(self.yaw, cfg.stance_yaw_huber_tolerance, cfg.stance_yaw_huber_sigma)
        angular, angular_excess = dead_zone_huber(self.omega_error, cfg.stance_angular_slip_tolerance,
                                                cfg.stance_angular_slip_sigma)
        stance = (self.plan & self.yaw_valid).to(yaw.dtype)
        # 沿用旧 Slip 的最低足底角代理。权重不反传，避免给抬脚减小惩罚的直接梯度。
        load = self.plan * self.contact.detach()
        mean = lambda value, mask: (value*mask).sum(-1)/(mask.sum(-1) + EPS)
        return dict(stance_yaw_huber=-self.quality*mean(yaw, stance),
                    stance_angular_slip=-self.quality*mean(angular, load)), (yaw_excess, angular_excess)


def stance_signals(rotation, angular, ref, ref_w, axis, soft_contact):
    yaw, valid = heading_error(rotation, ref['foot_R'][:, 0], axis)
    return StanceSignals(yaw, valid, angular[..., 2] - ref_w[..., 2],
                         ref['contact'][:, 0] & ref['contact_valid'][:, 0], soft_contact, ref['quality'][:, 0])
