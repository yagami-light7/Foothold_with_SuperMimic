"""平地奖励：8项，可独立测试；S54适配，不声称逐项复现论文。"""

import torch

from configs.s54_flat_reward import FlatWalkRewardCfg
from .heading import heading_error, yaw_from_quat


def swing_xy_weight(cfg, phase, first_swing=None):
    """首步标志来自独立半周期计数，不依赖相位值或命令重采样开关。"""
    if first_swing is None:
        if cfg.first_swing_xy_multiplier != 1.0:
            raise ValueError("首步奖励需要first_swing标志")
        first_swing = torch.zeros_like(phase, dtype=torch.bool)
    return cfg.swing_xy_multiplier * torch.where(first_swing, cfg.first_swing_xy_multiplier, 1.)


def flat_reward_reference(phase, left_is_swing, dt, cfg, first_swing=None):
    local = torch.where(left_is_swing, phase * 2, (phase - .5) * 2).clamp(0, 1)
    window = (local - .5).abs() <= 2 * cfg.swing_window_half_width
    # 惩罚的最大值为0；仅用于日志归一化，不改变交给PPO的奖励。
    weight = swing_xy_weight(cfg, phase, first_swing)
    upper = (cfg.tracking_xy_w * .5 * (1 + weight * local.square()) + cfg.tracking_yaw_w
             + cfg.swing_phase_w * window.float() + cfg.posture_w + cfg.heading_w + cfg.yaw_rate_w)
    return dt * upper


def compute_flat_walk_reward(left_pos, right_pos, left_target, right_target,
                             left_force, right_force, phase, left_is_swing,
                             gravity, actions, previous_actions, dt, cfg: FlatWalkRewardCfg,
                             state, root_quat, reference_heading, root_yaw_rate):
    """state中速度为足底参考点的世界速度；已补偿脚踝角速度×参考点偏移。"""
    if state is None or root_quat is None or reference_heading is None or root_yaw_rate is None:
        raise ValueError("flat_walk_v2需要足底速度、姿态及机身状态")
    mask = left_is_swing[:, None]
    swing_pos, stance_pos = torch.where(mask,left_pos,right_pos), torch.where(mask,right_pos,left_pos)
    swing_target = torch.where(mask,left_target,right_target)
    stance_target = torch.where(mask,right_target,left_target)
    velocities = torch.stack((state['left_velocity'],state['right_velocity']),dim=1)
    stance_velocity = torch.where(mask,state['right_velocity'],state['left_velocity'])
    contacts = torch.stack((left_force[:,2],right_force[:,2]),dim=1) > cfg.contact_threshold
    swing_contact = torch.where(left_is_swing,contacts[:,0],contacts[:,1])
    stance_contact = torch.where(left_is_swing,contacts[:,1],contacts[:,0])
    local = torch.where(left_is_swing,phase*2,(phase-.5)*2).clamp(0,1)
    swing_window = (local-.5).abs() <= 2*cfg.swing_window_half_width
    interior = (local > .1) & (local < .9)  # 换脚前后留出接触转移窗口。
    stance_height = (stance_pos[:,2]-stance_target[:,2]).clamp_min(0)
    support = stance_contact & (stance_height <= cfg.support_height_tolerance)

    swing_error = (swing_pos[:,:2]-swing_target[:,:2]).square().sum(-1)
    stance_error = (stance_pos[:,:2]-stance_target[:,:2]).square().sum(-1)
    weight = swing_xy_weight(cfg, phase, state.get('first_swing'))
    swing_score = torch.exp(-cfg.xy_sharpness*swing_error)
    if cfg.swing_xy_coarse_weight:
        # 只拓宽摆动脚XY，不增加站立/接触常数奖励；混合后的最大值仍为1。
        swing_score = ((1 - cfg.swing_xy_coarse_weight) * swing_score
                       + cfg.swing_xy_coarse_weight * torch.exp(-cfg.swing_xy_coarse_sharpness*swing_error))
    tracking_xy = cfg.tracking_xy_w*.5*(weight*local.square()*swing_score
                                                   + torch.exp(-cfg.xy_sharpness*stance_error))
    left_yaw_error = heading_error(state['left_quat'],yaw_from_quat(state['left_target_quat']))
    right_yaw_error = heading_error(state['right_quat'],yaw_from_quat(state['right_target_quat']))
    tracking_yaw = cfg.tracking_yaw_w*.5*(torch.exp(-cfg.yaw_sharpness*left_yaw_error.square())
                                         + torch.exp(-cfg.yaw_sharpness*right_yaw_error.square()))
    # 全身腾空/支撑脚离地时，不能靠把摆动脚送到目标继续领取跟踪奖励。
    tracking = (tracking_xy+tracking_yaw)*support.float()
    clearance = ((swing_pos[:,2]-swing_target[:,2])/cfg.swing_height).clamp(0,1)
    swing_phase = cfg.swing_phase_w*swing_window.float()*(~swing_contact).float()*support.float()*clearance

    # 任一接地脚的滑动都惩罚；腾空时不会靠此项获益，因为另有离地惩罚和跟踪门控。
    stance_slip = -cfg.stance_slip_w*(velocities[:,:,:2].square().sum(-1)*contacts.float()).sum(-1)
    height_cost = ((stance_height-cfg.support_height_tolerance).clamp_min(0)/cfg.hop_height_scale).square().clamp_max(4)
    vz_cost = ((stance_velocity[:,2].abs()-cfg.hop_velocity_tolerance).clamp_min(0)/cfg.hop_velocity_scale).square().clamp_max(4)
    missing_support = interior & ~stance_contact
    both_air = ~contacts.any(-1)
    stance_hop = -cfg.stance_hop_w*(height_cost+vz_cost+missing_support.float()+both_air.float())
    body_motion = (-cfg.body_height_w*(state['root_height']-state['nominal_height']).square()
                   -cfg.body_vz_w*state['root_vz'].square())
    posture = cfg.posture_w*(-gravity[:,2]).clamp(0,1)*torch.exp(
        -cfg.posture_sharpness*state['upper_joint_offsets'].square().sum(-1))
    action_rate = -cfg.action_rate_w*(actions-previous_actions).square().mean(-1)
    heading = (cfg.heading_w*torch.exp(-(heading_error(root_quat,reference_heading)/cfg.heading_sigma).square())
               +cfg.yaw_rate_w*torch.exp(-(root_yaw_rate/cfg.yaw_rate_sigma).square()))
    terms = dict(tracking=tracking,swing_phase=swing_phase,stance_slip=stance_slip,stance_hop=stance_hop,
                 body_motion=body_motion,posture=posture,action_rate=action_rate,heading=heading)
    return dt*sum(terms.values(),torch.zeros_like(phase)),terms
