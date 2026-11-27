"""Roban 根部倾斜定义与独立姿态、空间构型诊断。"""

import torch

from ..mdp import unflatten_state
from ..roban import JOINT_NAMES, ROOT_BODY
from .m411_hand import HAND_BODIES, torso_local_points
from .m51_foot import rotation_error


WAIST = JOINT_NAMES.index('waist_yaw_joint')
ELBOWS = ('zarm_l4_link', 'zarm_r4_link')
SHOULDERS = ('zarm_l1_link', 'zarm_r1_link')


def heading_components(rotation):
    # 正常用局部 X 的水平投影。接近竖直时用局部 Y 推定航向。
    fx, fy = rotation[..., :2, 0].unbind(-1)
    fallback = fx.square() + fy.square() < 1e-8
    x = torch.where(fallback, rotation[..., 1, 1], fx)
    y = torch.where(fallback, -rotation[..., 0, 1], fy)
    inv_norm = (x.square() + y.square()).clamp_min(1e-12).rsqrt()
    return x * inv_norm, y * inv_norm


def heading(rotation):
    c, s = heading_components(rotation)
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, zero, s, c, zero, zero, zero, one), -1).reshape(rotation.shape)


def tilt_error(rotation, reference):
    # 训练只对矩阵做逐元素运算。低精度输入先升到 FP32，无需 acos。
    def up(matrix):
        if matrix.dtype in (torch.float16, torch.bfloat16):
            matrix = matrix.float()
        c, s = heading_components(matrix)
        x, y, z = matrix[..., :, 2].unbind(-1)
        return torch.stack((c*x+s*y, c*y-s*x, z), -1)

    return .5 * (up(rotation) - up(reference)).square().sum(-1)


def tilt_angle(rotation, reference):
    return 2 * (tilt_error(rotation, reference) * .5).clamp(0., 1.).sqrt().asin()


def euler(rotation):
    # ZYX 约定，返回 Roll、Pitch、Yaw。仅供测量与绘图。
    roll = torch.atan2(rotation[..., 2, 1], rotation[..., 2, 2])
    pitch = torch.atan2(-rotation[..., 2, 0], rotation[..., :2, 0].norm(dim=-1))
    c, s = heading_components(rotation)
    return torch.stack((roll, pitch, torch.atan2(s, c)), -1)


def rms(value, dim=0):
    return value.square().mean(dim=dim).sqrt()


def python(value):
    return value.detach().cpu().tolist()


def angle_error(rotation, reference):
    delta = euler(rotation) - euler(reference)
    return torch.atan2(delta.sin(), delta.cos())


def distribution(values):
    angles = values.rad2deg()
    low, high = torch.quantile(angles, angles.new_tensor([.05, .95]), dim=0)
    return dict(mean_deg=python(angles.mean(0)), rms_deg=python(rms(angles)),
                p05_deg=python(low), p95_deg=python(high), range90_deg=python(high-low))


def correlation(actual, target):
    actual, target = actual-actual.mean(0), target-target.mean(0)
    denom = actual.norm(dim=0) * target.norm(dim=0)
    score = ((actual*target).sum(0) / denom.clamp_min(1e-12)).cpu().tolist()
    return [v if ok else None for v, ok in zip(score, (denom > 1e-8).cpu().tolist())]


def orientation_stats(state, ref, torso, ref_torso):
    root = state[1]
    target = ref[1]
    angles, expected = euler(root), euler(target)
    delta = angle_error(root, target).rad2deg()
    tilt = tilt_angle(root, target).rad2deg()
    chest = tilt_angle(torso, ref_torso).rad2deg()
    return dict(angles=distribution(angles), reference=distribution(expected),
                rpy_rmse_deg=python(rms(delta)), rpy_bias_deg=python(delta.mean(0)),
                rp_phase=correlation(angles[:, :2], expected[:, :2]),
                tilt_rms_deg=python(rms(tilt)), so3_rms_deg=python(rms(rotation_error(root, target).rad2deg())),
                omega_rms_rad_s=python(rms((state[3]-ref[3]).norm(dim=-1))),
                torso_tilt_rms_deg=python(rms(chest)),
                root_torso_tilt_rms_deg=python(rms(tilt_angle(root, torso).rad2deg())),
                waist_rms_rad=python(rms(state[4][..., WAIST]-ref[4][..., WAIST])))


def keypoints(state, kin):
    positions, rotations = kin.forward(*state)[:2]
    indices = kin.tracked_names.index
    chest = indices(ROOT_BODY)
    hand, elbow, shoulder = ([indices(name) for name in names] for names in (HAND_BODIES, ELBOWS, SHOULDERS))
    return dict(hand=positions[:, hand], hand_R=rotations[:, hand], elbow=positions[:, elbow],
                shoulder=positions[:, shoulder], torso_p=positions[:, chest], torso_R=rotations[:, chest])


def posture_signals(state, ref, kin):
    actual, target = (keypoints(s, kin) for s in (state, ref))
    signals = dict(root_rpy=euler(state[1]).rad2deg(), reference_rpy=euler(ref[1]).rad2deg(),
                   root_tilt=tilt_angle(state[1], ref[1]).rad2deg(),
                   torso_tilt=tilt_angle(actual['torso_R'], target['torso_R']).rad2deg(),
                   angular_velocity=(state[3]-ref[3]).norm(dim=-1),
                   root_xy=(state[0]-ref[0])[:, :2].norm(dim=-1),
                   hand_orientation=rotation_error(actual['hand_R'], target['hand_R']).rad2deg())
    for label in ('hand', 'elbow'):
        point, expected = actual[label], target[label]
        signals[label+'_world_m'] = (point-expected).norm(dim=-1)
        body = torso_local_points(state[0], state[1], point)
        body_target = torso_local_points(ref[0], ref[1], expected)
        signals[label+'_root_local_m'] = (body-body_target).norm(dim=-1)
        local = torso_local_points(actual['torso_p'], actual['torso_R'], point)
        local_target = torso_local_points(target['torso_p'], target['torso_R'], expected)
        signals[label+'_torso_local_m'] = (local-local_target).norm(dim=-1)
    for label, proximal, distal in (('upper_arm', 'shoulder', 'elbow'), ('forearm', 'elbow', 'hand')):
        v = actual[distal]-actual[proximal]
        w = target[distal]-target[proximal]
        for suffix, a, b in (('world', v, w),
                             ('local', v@actual['torso_R'], w@target['torso_R'])):
            cross = torch.linalg.cross(a, b, dim=-1).norm(dim=-1)
            dot = (a*b).sum(-1)
            signals[f'{label}_{suffix}_deg'] = torch.atan2(cross, dot).rad2deg()
    return signals, actual['torso_R'], target['torso_R']


def trace_states(trace, motion, end=None):
    state = torch.as_tensor(trace['state'][:end], device=motion.q.device)
    frames = torch.as_tensor(trace['frame'][:end], device=motion.q.device, dtype=torch.long)
    return unflatten_state(state), motion.ref_state(frames)


@torch.no_grad()
def posture_metrics(state, ref, kin, *, segments=None):
    values, chest, target_chest = posture_signals(state, ref, kin)
    masks = {'all': slice(None)} | (segments or {})
    return {stage: orientation_stats(tuple(x[select] for x in state), tuple(x[select] for x in ref),
                                     chest[select], target_chest[select])
            | {key+'_rms': python(rms(value[select])) for key, value in values.items()
               if key not in ('root_rpy', 'reference_rpy')}
            for stage, select in masks.items() if len(chest[select])}


@torch.no_grad()
def counterfactual(state, ref, kin):
    p, rotation, v, w, q, qd = state
    ref_p, ref_R = ref[:2]
    tilt_swapped = heading(rotation) @ (heading(ref_R).mT @ ref_R)
    cases = (state, (p, tilt_swapped, v, w, q, qd), (p, ref_R, v, w, q, qd),
             (ref_p, ref_R, v, w, q, qd), ref)
    return {f'D{i}': posture_metrics(candidate, ref, kin)['all'] for i, candidate in enumerate(cases)}
