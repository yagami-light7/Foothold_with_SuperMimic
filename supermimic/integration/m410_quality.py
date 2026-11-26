"""统一以原 F2 时钟度量真实轨迹的全身姿态和运动。"""

import numpy as np
import torch

from ..mdp import unflatten_state
from ..motion import CONTROL_DT
from ..roban import JOINT_NAMES
from .m410_reward import HUBER_CONFIG, HUBER_TERMS, POSTURE_JOINTS, root_errors
from .m411_hand import HAND_BODIES, torso_local_points


def rms(value):
    return value.square().mean().sqrt().item()


def angle(a, b):
    trace = (a * b).sum(dim=(-1, -2))
    return ((trace - 1.) * .5).clamp(-1., 1.).acos()


def local_upper_points(state, kin):
    # 依次取双肩、双肘、双手在躯干局部坐标系中的位置。
    names = ('zarm_l1_link', 'zarm_r1_link', 'zarm_l4_link', 'zarm_r4_link', *HAND_BODIES)
    body = kin.forward(*state, all_bodies=True)[0]
    points = body[:, [kin.body_names.index(name) for name in names]]
    return torso_local_points(*state[:2], points)


@torch.no_grad()
def whole_body_metrics(trace, mdp, *, start=1, end=None):
    device = mdp.motion.q.device
    rows = slice(start, None if end is None else end + 1)
    take = lambda key: torch.as_tensor(trace[key][rows], device=device)
    frames = take('frame').long()
    if not len(frames):
        return {}
    state = unflatten_state(take('state'))
    p, rotation, v, w, q, _ = state
    f2 = mdp.motion.at_time(frames.to(torch.float64) * CONTROL_DT)
    reference = tuple(f2[key] for key in ('root_p', 'root_R', 'root_v', 'root_w', 'q', 'qd'))
    xy, z, heading = root_errors(p, rotation, reference[0], reference[1])
    error = q - reference[4]
    roll = torch.atan2(rotation[:, 2, 1], rotation[:, 2, 2]).rad2deg()
    pitch = (-rotation[:, 2, 0]).clamp(-1., 1.).asin().rad2deg()
    contact = take('contact').bool()
    foot = take('foot')
    if 'foot_R' in trace and 'foot_velocity' in trace:
        foot_rot, foot_speed = take('foot_R'), take('foot_velocity')
    else:
        _, foot_rot, foot_speed = mdp.motion.feet(state)
    relative = p[:, None] - foot
    support = contact & f2['contact']
    relative_error = relative - (reference[0][:, None] - f2['foot_p'])
    forward = reference[1][:, :2, 0]
    forward = torch.nn.functional.normalize(forward, dim=-1)
    forward_offset = (relative_error[..., :2] * forward[:, None]).sum(-1)
    metrics = dict(root_xy_rmse_m=rms(xy), root_z_rmse_m=rms(z),
                   root_xyz_rmse_m=rms(torch.linalg.vector_norm(p - reference[0], dim=-1)),
                   root_heading_rmse_deg=rms(heading.rad2deg()),
                   root_roll_pitch_rms_deg=[rms(roll), rms(pitch)],
                   root_roll_pitch_peak_deg=[roll.abs().max().item(), pitch.abs().max().item()],
                   root_speed_rms_m_s=rms(torch.linalg.vector_norm(v, dim=-1)),
                   root_angular_speed_rms_rad_s=rms(torch.linalg.vector_norm(w, dim=-1)),
                   root_linear_acceleration_rms_m_s2=(rms(torch.linalg.vector_norm(v.diff(dim=0) / CONTROL_DT, dim=-1))
                                                       if len(v) > 1 else None),
                   root_height_range_m=[p[:, 2].min().item(), p[:, 2].max().item()],
                   root_support_offset_xyz_m=relative[contact].mean(0).tolist() if contact.any() else None,
                   root_support_forward_error_m=forward_offset[support].mean().item() if support.any() else None,
                   foot_orientation_rmse_deg=rms(angle(foot_rot, f2['foot_R']).rad2deg()),
                   support_slide_rms_m_s=(rms(torch.linalg.vector_norm(foot_speed[contact][:, :2], dim=-1))
                                           if contact.any() else None),
                   arm_rmse_rad=rms(error[:, mdp.upper_ids]))
    leg_groups = {'hip': (1, 2, 3), 'knee': (4,), 'ankle': (5, 6)}
    for group, numbers in leg_groups.items():
        joints = [JOINT_NAMES.index(f'leg_{side}{number}_joint') for side in 'lr' for number in numbers]
        metrics[f'{group}_rmse_rad'] = rms(error[:, joints])
        if group == 'knee':
            metrics['knee_signed_mean_error_rad'] = error[:, joints].mean(0).tolist()
    metrics['upper_joint_error'] = {}
    for term, names in zip(HUBER_TERMS, POSTURE_JOINTS):
        group = term.removesuffix('_huber')
        indices = mdp.posture_groups[term]
        absolute = error[:, indices].abs()
        margin = torch.minimum(q[:, indices] - mdp.kin.q_lo[indices], mdp.kin.q_hi[indices] - q[:, indices])
        excess = (absolute - HUBER_CONFIG[term + '_tolerance']).clamp_min(0.)
        metrics[f'{group}_rmse_rad'] = rms(absolute)
        metrics[f'{group}_max_near_limit_ratio'] = (margin <= .02).float().mean(0).max().item()
        for j, name in enumerate(names):
            metrics['upper_joint_error'][name] = dict(rms_rad=rms(absolute[:, j]),
                p95_rad=torch.quantile(absolute[:, j], .95).item(), max_rad=absolute[:, j].max().item(),
                tolerance_excess_rad=excess[:, j].mean().item(), outside_tolerance=(excess[:, j] > 0.).float().mean().item(),
                near_limit=(margin[:, j] <= .02).float().mean().item())

    kin = mdp.kin
    actual, desired = local_upper_points(state, kin), local_upper_points(reference, kin)
    offsets = actual - desired
    dx = offsets.norm(dim=-1)
    u = torch.nn.functional.normalize(actual[:, 2:4] - actual[:, :2], dim=-1)
    uref = torch.nn.functional.normalize(desired[:, 2:4] - desired[:, :2], dim=-1)
    hands = actual[:, 4:]
    spread = (torch.quantile(hands, .95, dim=0) - torch.quantile(hands, .05, dim=0))
    # 零时滞相关按左右手取值，静止或太短的轨迹保留缺失状态。
    correlations = []
    for a, b in zip(hands[:, :, 0].T, desired[:, 4:, 0].T):
        a, b = a - a.mean(), b - b.mean()
        scale = a.norm() * b.norm()
        correlations.append((a @ b / scale).item() if scale > 1.e-8 else None)
    metrics.update(elbow_local_rmse_m=[rms(x) for x in dx[:, 2:4].T],
                   hand_local_rmse_m=[rms(x) for x in dx[:, 4:].T],
                   hand_local_axis_rmse_m=offsets[:, 4:].square().mean(0).sqrt().tolist(),
                   hand_fore_aft_span_m=spread[:, 0].tolist(),
                   hand_lateral_span_m=spread[:, 1].tolist(),
                   hand_lateral_abs_p95_m=torch.quantile(hands[:, :, 1].abs(), .95, dim=0).tolist(),
                   upper_arm_direction_rmse_deg=[rms(x) for x in (u * uref).sum(-1).clamp(-1.,1.).acos().rad2deg().T],
                   hand_reference_phase_correlation=correlations)
    return metrics


def whole_body_rank(row, *, hand_tracking=False):
    gait = row['aggregate']
    if not gait['valid_for_best'] or gait['complete_tracking_rate'] < 1.:
        return None
    body = gait['whole_body']
    control = gait['upper_smoothness']['after_startup']
    # 完整四步候选按所在实验的验收尺度换算量纲后等权比较。
    foot, arm, elbow, accel, jerk = (.01, .30, .05, 30., 2000.) if hand_tracking else (.025, .4, .1, 40., 3000.)
    values = (gait['foot_position_rmse_m'] / foot, body['root_xy_rmse_m'] / .05,
              body['root_z_rmse_m'] / .03, body['root_heading_rmse_deg'] / 5., body['arm_rmse_rad'] / arm,
              body['elbow_max_near_limit_ratio'] / elbow, control['acceleration_rms_rad_s2'] / accel,
              control['jerk_rms_rad_s3'] / jerk)
    if hand_tracking:
        values += (float(np.mean(body['hand_local_rmse_m'])) / .05,
                   float(np.mean(body['upper_arm_direction_rmse_deg'])) / 15.)
    return float(np.mean(values)), -row['iteration']
