"""从独立 Newton 闭环轨迹计算足部方向和实际承重指标。"""

import numpy as np
import torch

from ..mdp import unflatten_state
from ..motion import CONTROL_DT
from ..roban import JOINT_NAMES
from .m410_evaluation import mean_metrics
from .m51_foot import euler_angles, foot_axes, rotation_error, wrapped


def read_guide_heading(path):
    with np.load(path, allow_pickle=False) as clip:
        return clip['diag__guide_root_yaw']


def rms(values):
    return float(np.sqrt(np.square(values).mean())) if values.size else None


def magnitudes(values):
    if not len(values):
        return None
    return dict(rms=rms(values), p95=float(np.quantile(np.abs(values), .95)), max=float(np.abs(values).max()))


@torch.no_grad()
def trace_signals(trace, mdp, guide):
    device = mdp.default_q.device
    tensor = lambda value: torch.as_tensor(value, device=device)
    frames = tensor(trace['frame']).long()
    motion = mdp.motion
    actual_R, ref_R = tensor(trace['foot_R']), motion.foot_R[frames]
    axis, points = (mdp.default_q.new_tensor(x) for x in foot_axes())
    world, ref_world = euler_angles(actual_R, axis), euler_angles(ref_R, axis)
    actual_w = (tensor(trace['foot_angular_velocity']) if 'foot_angular_velocity' in trace
                else mdp.kin.forward(*unflatten_state(tensor(trace['state'])))[3][:, motion.foot_indices])
    ref_w = motion.body_w[frames][:, motion.foot_indices]
    p, v = tensor(trace['foot']), tensor(trace['foot_velocity'])
    offsets = points-motion.sole_offsets[:, None]
    displacement = (actual_R[..., None, :, :] @ offsets[..., None]).squeeze(-1)
    point_v = v[..., None, :] + torch.linalg.cross(actual_w[..., None, :], displacement, dim=-1)
    heading = tensor(guide)[frames, None]
    data = dict(world=world.rad2deg(), ref_world=ref_world.rad2deg(),
                local=wrapped(world[..., 0]-heading).rad2deg(),
                ref_local=wrapped(ref_world[..., 0]-heading).rad2deg(),
                error=wrapped(world-ref_world).rad2deg(), so3=rotation_error(actual_R, ref_R).rad2deg(),
                omega=actual_w[..., 2], ref_omega=ref_w[..., 2],
                speed=v[..., :2].norm(dim=-1), ref_speed=motion.foot_v[frames][..., :2].norm(dim=-1),
                heel_speed=point_v[..., 0, :2].norm(dim=-1), toe_speed=point_v[..., 1, :2].norm(dim=-1),
                heel_xy=(p+displacement[..., 0, :])[..., :2], toe_xy=(p+displacement[..., 1, :])[..., :2],
                proxy=(((mdp.cfg.contact_height-mdp._sole_clearance(p, actual_R))/mdp.cfg.contact_height_std).sigmoid()),
                planned=motion.contact[frames], force=tensor(trace['force'])[..., 2])
    return {key: value.cpu().numpy() for key, value in data.items()}


def _segments(mask, yaw, speed):
    values = []
    for side in range(mask.shape[1]):
        edges = np.flatnonzero(np.diff(np.r_[False, mask[:, side], False]))
        for begin, end in edges.reshape(-1, 2):
            angle = np.unwrap(np.deg2rad(yaw[begin:end, side]))
            delta = np.rad2deg(angle-angle[0])
            values.append(dict(load_s=(end-begin)*CONTROL_DT,
                               yaw_total_deg=float(np.abs(np.diff(delta)).sum()),
                               yaw_drift_deg=float(np.abs(delta).max()),
                               distance_mm=float(speed[begin:end, side].sum()*CONTROL_DT*1000)))
    return values


def _support(signals, mask):
    segments = _segments(mask, signals['world'][..., 0], signals['speed'])
    result = {key: magnitudes(signals[key][mask]) for key in ('omega', 'speed', 'heel_speed', 'toe_speed', 'force')}
    result.update(samples=int(mask.sum()), segments=len(segments), load_time_s=float(mask.sum()*CONTROL_DT),
                  segment_mean=mean_metrics(segments),
                  segment_max={key: max(row[key] for row in segments) for key in segments[0]} if segments else None)
    return result


def _orientation(signal, mask):
    return dict(yaw_rms_deg=rms(signal['error'][..., 0][mask]),
                pitch_rms_deg=rms(signal['error'][..., 1][mask]),
                roll_rms_deg=rms(signal['error'][..., 2][mask]),
                so3_rms_deg=rms(signal['so3'][mask]))


def _toe_pose(angles):
    if not len(angles):
        return {}
    # 左脚负偏角、右脚正偏角朝向身体内侧。
    toe_in = angles * np.array([-1., 1.])
    circular = np.rad2deg(np.angle(np.exp(1j*np.deg2rad(angles)).mean(0)))
    return dict(signed_mean_deg=circular.tolist(), p05_deg=np.quantile(angles, .05, axis=0).tolist(),
                p95_deg=np.quantile(angles, .95, axis=0).tolist(), toe_in_gt_3deg_ratio=(toe_in > 3.).mean(0).tolist())


@torch.no_grad()
def contact_metrics(trace, row, mdp, guide):
    signal = trace_signals(trace, mdp, guide)
    # Reset 帧的接触力是占位值，正式承重统计从第一个物理步开始。
    signal = {key: value[1:] for key, value in signal.items()}
    if not len(signal['planned']):
        return {}
    support, plan = np.asarray(trace['contact'][1:], bool), signal['planned']
    loaded = support & plan
    count = support.sum(-1)
    result = {key: _orientation(signal, mask) for key, mask in
              (('all', np.ones_like(plan)), ('swing', ~plan), ('planned_stance', plan), ('loaded_stance', loaded))}
    result['side_yaw_rmse_deg'] = [rms(signal['error'][:, side, 0]) for side in range(2)]
    result['local_actual'] = _toe_pose(signal['local'])
    result['local_reference'] = _toe_pose(signal['ref_local'])
    result['world_actual'] = _toe_pose(signal['world'][..., 0])
    result['world_reference'] = _toe_pose(signal['ref_world'][..., 0])
    for phase, mask in (('stance', plan), ('swing', ~plan)):
        result[f'{phase}_toe_in_gt_3deg_ratio'] = [
            float(((sign*signal['local'][:, side] > 3.) & mask[:, side]).sum()/max(1, mask[:, side].sum()))
            for side, sign in enumerate((-1, 1))]
    result['proxy_agreement'] = float(((signal['proxy'] >= .5) == support).mean())
    result['plan_load_fraction'] = float(loaded.sum()/max(1, plan.sum()))
    result['unexpected_planned_unload'] = 1.-result['plan_load_fraction']
    result.update({key: _support(signal, mask) for key, mask in
                   (('support', loaded), ('single', loaded & (count[:, None] == 1)),
                    ('double', loaded & (count[:, None] == 2)))})
    reference = dict(signal, world=signal['ref_world'], speed=signal['ref_speed'])
    result['reference_stance'] = dict(omega=magnitudes(signal['ref_omega'][plan]),
                                      speed=magnitudes(signal['ref_speed'][plan]),
                                      segment_mean=mean_metrics(_segments(plan, reference['world'][..., 0],
                                                                          reference['speed'])))
    joints = [JOINT_NAMES.index(f'leg_{side}{i}_joint') for side in 'lr' for i in (3, 4, 5, 6)]
    q = trace['state'][1:, 13:13+len(JOINT_NAMES)][:, joints]
    q_ref = mdp.motion.q[trace['frame'][1:]][:, joints].cpu().numpy()
    result['hip_knee_ankle'] = {JOINT_NAMES[j]: dict(mean_deg=float(np.rad2deg(q[:, i].mean())),
                    ref_mean_deg=float(np.rad2deg(q_ref[:, i].mean())),
                    rmse_deg=float(np.rad2deg(rms(q[:, i]-q_ref[:, i]))))
                               for i, j in enumerate(joints)}
    angles = []
    for footstep, side in zip(row['footsteps'], mdp.motion.touchdown_side.tolist()):
        when = footstep['actual_touchdown_s']
        if when is None:
            continue
        actual = signal['world'][round(when/CONTROL_DT)-1, side, 0]
        frame = round(footstep['planned_touchdown_s']/CONTROL_DT)
        axis = mdp.default_q.new_tensor(foot_axes()[0][side])
        target = float(euler_angles(mdp.motion.foot_R[frame, side], axis)[0].rad2deg())
        angles.append((actual-target+180.) % 360.-180.)
    result['touchdown_yaw_rmse_deg'] = rms(np.asarray(angles))
    return result
