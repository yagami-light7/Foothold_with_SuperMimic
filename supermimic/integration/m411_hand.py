"""M4.11 双手末端在躯干局部坐标系中的跟踪奖励。"""

from pathlib import Path

import torch

from .m410_reward import dead_zone_huber


GROUPS = {'a': 'a_continue_baseline', 'b': 'b_hand_huber'}
SOURCE = Path('outputs/m4_10/c_root_huber/checkpoints/iter_9000.pt')
TORSO_BODY = 'base_link'
HAND_BODIES = ('l_handball', 'r_handball')
HAND_CONFIG = {'hand_huber': .10, 'hand_huber_tolerance': .03, 'hand_huber_sigma': .10}
HAND_SIDES = ('left', 'right')


def torso_local_points(position, rotation, points):
    return (rotation.transpose(-1, -2)[:, None] @ (points - position[:, None]).unsqueeze(-1)).squeeze(-1)


def hand_position_huber(delta, quality, tolerance, sigma):
    distance = torch.linalg.vector_norm(delta, dim=-1)
    penalty, excess = dead_zone_huber(distance, tolerance, sigma)
    return -quality * penalty.mean(dim=-1), distance, excess
