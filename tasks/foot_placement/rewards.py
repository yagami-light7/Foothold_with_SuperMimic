
"""First-stage S54 foot-placement reward.

Inputs and outputs are batched PyTorch tensors.
This is an initial reward design, not an exact
reimplementation of the original paper.
"""

from dataclasses import dataclass

import torch
from .heading import heading_error
from configs.s54_flat_reward import FlatWalkRewardCfg


def reward_reference_upper(phase, left_is_swing, dt, cfg, first_swing=None):
    """当前非负奖励权重下的逐相位参考上界；仅用于记录，不改变PPO奖励。"""
    if isinstance(cfg, FlatWalkRewardCfg):
        from .flat_rewards import flat_reward_reference
        return flat_reward_reference(phase, left_is_swing, dt, cfg, first_swing)
    local = torch.where(left_is_swing, phase * 2, (phase - .5) * 2).clamp(0, 1)
    height = torch.sin(torch.pi * local)
    clearance_max = torch.ones_like(phase)
    if cfg.relative_clearance:
        baseline = torch.exp(-cfg.z_sharpness * (cfg.swing_height * height).square())
        clearance_max = ((1 - baseline) / (1 - baseline).clamp_min(.05)).clamp(0, 1)
    middle = ((local > .2) & (local < .8)).float()
    late = (local >= .8).float()
    # 动作变化惩罚的最好值为0；相位窗口外的接触项不计入分母。
    upper = (cfg.swing_xy_w * local.square() + cfg.stance_xy_w
             + cfg.clearance_w * height.square() * clearance_max
             + (cfg.swing_air_w + cfg.unloading_w) * middle
             + cfg.stance_contact_w + cfg.touchdown_w * late
             + cfg.upright_w + cfg.heading_w + cfg.yaw_rate_w)
    return dt * upper


def normalized_episode_return(reward_sum, reference_sum, steps, maximum_steps):
    """[0,1]回报=奖励质量×存活时长比例；短暂高分后跌倒会被扣分。"""
    quality = (reward_sum / reference_sum.clamp_min(1e-8)).clamp(0, 1)
    survival = (steps.float() / maximum_steps).clamp(0, 1)
    return quality * survival


def episode_return_metrics(reward_sum, reference_sum, steps, maximum_steps):
    """保留每个完成episode的分数，交给RSL-RL按episode数量求均值。"""
    # 不能先对同一结束帧求mean：1个失败与上千个超时会因此获得相同权重。
    return {
        "Episode/normalized_return": normalized_episode_return(
            reward_sum, reference_sum, steps, maximum_steps),
        "Episode/normalized_quality": (reward_sum / reference_sum.clamp_min(1e-8)).clamp(0, 1),
        "Episode/survival_fraction": (steps.float() / maximum_steps).clamp(0, 1),
    }


@dataclass
class FootRewardCfg:
    # Tracking weights
    swing_xy_w: float = 4.0
    stance_xy_w: float = 2.0
    clearance_w: float = 2.0

    # Contact weights
    swing_air_w: float = 1.0
    stance_contact_w: float = 0.5
    touchdown_w: float = 2.0

    # Stability and regularization
    upright_w: float = 1.0
    action_rate_w: float = 0.05

    # Tracking sharpness
    xy_sharpness: float = 100.0
    z_sharpness: float = 400.0
    # 可选宽核只塑造接近目标的信号；默认0保持原奖励，严格落脚判据不放宽。
    xy_coarse_weight: float = 0.0
    xy_coarse_sharpness: float = 25.0

    # Gait configuration
    swing_height: float = 0.06
    contact_threshold: float = 20.0
    touchdown_xy_threshold: float = 0.05

    # Opt-in curriculum terms; False/zero preserves the legacy reward.
    relative_clearance: bool = False
    unloading_w: float = 0.0
    nominal_foot_load: float = 330.0
    normalize_action_rate: bool = False
    require_stance_support: bool = False
    require_liftoff_for_touchdown: bool = False

    # 可选方向跟随奖励：默认权重为0，保证旧阶段/旧checkpoint的奖励不变。
    # 权重表示奖励率，最后统一乘控制步长dt；这里不重复乘dt。
    heading_w: float = 0.0
    heading_sigma: float = 0.35  # 朝向误差尺度，rad，约20°；误差等于此值时得分为exp(-1)。
    yaw_rate_w: float = 0.0
    yaw_rate_sigma: float = 0.30  # 偏航角速度误差尺度，rad/s，注意不是角度。


def reward_config_from_dict(data):
    """保存的profile决定奖励版本；旧checkpoint没有profile时保持原奖励。"""
    if data.get('profile') == 'flat_walk_v2':
        return FlatWalkRewardCfg(**data)
    if 'profile' in data:
        raise ValueError(f"未知奖励profile：{data['profile']}")
    return FootRewardCfg(**data)


def compute_foot_placement_reward(
    left_pos_w: torch.Tensor,
    right_pos_w: torch.Tensor,
    left_target_w: torch.Tensor,
    right_target_w: torch.Tensor,
    left_force_w: torch.Tensor,
    right_force_w: torch.Tensor,
    phase: torch.Tensor,
    left_is_swing: torch.Tensor,
    projected_gravity_b: torch.Tensor,
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    dt: float,
    cfg: FootRewardCfg | FlatWalkRewardCfg,
    swing_has_lifted: torch.Tensor | None = None,
    root_quat_w: torch.Tensor | None = None,
    reference_heading_w: torch.Tensor | None = None,
    root_yaw_rate: torch.Tensor | None = None,
    flat_state: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute reward and per-component diagnostics.

    Positions: [N, 3]
    Forces: [N, 3]
    Phase and swing mask: [N]
    Actions: [N, 27]

    Returns:
        reward: [N]
        terms: dict of [N] tensors
    """

    if isinstance(cfg, FlatWalkRewardCfg):
        from .flat_rewards import compute_flat_walk_reward
        return compute_flat_walk_reward(left_pos_w,right_pos_w,left_target_w,right_target_w,
            left_force_w,right_force_w,phase,left_is_swing,projected_gravity_b,actions,prev_actions,
            dt,cfg,flat_state,root_quat_w,reference_heading_w,root_yaw_rate)

    # --------------------------------------------------
    # 1. Select swing / stance foot by the desired phase.
    # --------------------------------------------------

    mask = left_is_swing.unsqueeze(-1)

    swing_pos = torch.where(
        mask, left_pos_w, right_pos_w
    )
    stance_pos = torch.where(
        mask, right_pos_w, left_pos_w
    )

    swing_target = torch.where(
        mask, left_target_w, right_target_w
    )
    stance_target = torch.where(
        mask, right_target_w, left_target_w
    )

    swing_force = torch.where(
        mask, left_force_w, right_force_w
    )
    stance_force = torch.where(
        mask, right_force_w, left_force_w
    )

    # --------------------------------------------------
    # 2. Normalize the current half-cycle to [0, 1].
    # --------------------------------------------------

    local_phase = torch.where(
        left_is_swing,
        phase * 2.0,
        (phase - 0.5) * 2.0,
    ).clamp(0.0, 1.0)

    # --------------------------------------------------
    # 3. Foot XY tracking.
    # --------------------------------------------------

    swing_xy_err_sq = torch.sum(
        (swing_pos[:, :2] - swing_target[:, :2]).square(),
        dim=-1,
    )

    stance_xy_err_sq = torch.sum(
        (stance_pos[:, :2] - stance_target[:, :2]).square(),
        dim=-1,
    )

    # Increase the importance of reaching the goal
    # toward the end of the swing phase.
    xy_score = torch.exp(-cfg.xy_sharpness * swing_xy_err_sq)
    if cfg.xy_coarse_weight:
        if not (0 <= cfg.xy_coarse_weight <= 1 and cfg.xy_coarse_sharpness > 0):
            raise ValueError("XY宽核权重需在[0,1]且sharpness为正")
        xy_score = ((1 - cfg.xy_coarse_weight) * xy_score
                    + cfg.xy_coarse_weight * torch.exp(-cfg.xy_coarse_sharpness * swing_xy_err_sq))
    swing_xy = (
        cfg.swing_xy_w
        * local_phase.square()
        * xy_score
    )

    stance_xy = (
        cfg.stance_xy_w
        * torch.exp(-cfg.xy_sharpness * stance_xy_err_sq)
    )

    # --------------------------------------------------
    # 4. Swing-foot height tracking.
    # --------------------------------------------------

    height_profile = torch.sin(
        torch.pi * local_phase
    )

    desired_swing_z = (
        swing_target[:, 2]
        + cfg.swing_height * height_profile
    )

    z_err_sq = (
        swing_pos[:, 2] - desired_swing_z
    ).square()

    # Focus clearance reward on mid-swing.
    clearance_score = torch.exp(-cfg.z_sharpness * z_err_sq)
    if cfg.relative_clearance:
        standing_score = torch.exp(
            -cfg.z_sharpness * (desired_swing_z - swing_target[:, 2]).square()
        )
        clearance_score = (
            (clearance_score - standing_score)
            / (1.0 - standing_score).clamp_min(0.05)
        ).clamp(-1.0, 1.0)
    clearance = cfg.clearance_w * height_profile.square() * clearance_score

    # --------------------------------------------------
    # 5. Ground contact rewards.
    # --------------------------------------------------

    swing_contact = (
        swing_force[:, 2] > cfg.contact_threshold
    )
    stance_contact = (
        stance_force[:, 2] > cfg.contact_threshold
    )

    mid_swing = (
        (local_phase > 0.2)
        & (local_phase < 0.8)
    )

    late_swing = local_phase >= 0.8

    swing_air = (
        cfg.swing_air_w
        * (mid_swing & ~swing_contact).float()
    )

    # Dense precursor to binary no-contact reward during the swing window.
    unloading = (
        cfg.unloading_w * mid_swing.float()
        * (1.0 - swing_force[:, 2].clamp_min(0.0) / cfg.nominal_foot_load).clamp(0.0, 1.0)
    )

    stance_support = (
        cfg.stance_contact_w
        * stance_contact.float()
    )

    touchdown = (
        cfg.touchdown_w
        * (
            late_swing
            & swing_contact
            & (
                swing_xy_err_sq
                < cfg.touchdown_xy_threshold ** 2
            )
        ).float()
    )
    if cfg.require_liftoff_for_touchdown:
        if swing_has_lifted is None:
            raise ValueError("Touchdown curriculum requires swing liftoff history")
        touchdown = touchdown * swing_has_lifted.float()
    if cfg.require_stance_support:
        support_gate = stance_contact.float()
        clearance = clearance * support_gate
        swing_air = swing_air * support_gate
        unloading = unloading * support_gate

    # --------------------------------------------------
    # 6. Body stabilization and action smoothness.
    # --------------------------------------------------

    upright = (
        cfg.upright_w
        * (-projected_gravity_b[:, 2]).clamp(0.0, 1.0)
    )

    action_change = (actions - prev_actions).square()
    action_rate = (
        -cfg.action_rate_w
        * (action_change.mean(dim=-1) if cfg.normalize_action_rate else action_change.sum(dim=-1))
    )

    # --------------------------------------------------
    # 7. Combine components.
    # --------------------------------------------------

    terms = {
        "swing_xy": swing_xy,
        "stance_xy": stance_xy,
        "clearance": clearance,
        "swing_air": swing_air,
        "stance_contact": stance_support,
        "touchdown": touchdown,
        "upright": upright,
        "action_rate": action_rate,
        "unloading": unloading,
    }

    if cfg.heading_w:
        if root_quat_w is None or reference_heading_w is None or cfg.heading_sigma <= 0:
            raise ValueError("Heading reward needs root quaternion, reset reference and positive sigma")
        error = heading_error(root_quat_w, reference_heading_w)
        # 累计朝向越接近reset参考，得分越高；wrap避免±π边界造成虚假的巨大误差。
        terms["heading"] = cfg.heading_w * torch.exp(-(error / cfg.heading_sigma).square())
    if cfg.yaw_rate_w:
        if root_yaw_rate is None or cfg.yaw_rate_sigma <= 0:
            raise ValueError("Yaw-rate reward needs yaw rate and positive sigma")
        # 当前只训练直行，因此目标偏航角速度为0。
        # 未来加入转向指令时，应改为跟踪“实际yaw_rate - 指令yaw_rate”。
        terms["yaw_rate"] = cfg.yaw_rate_w * torch.exp(-(root_yaw_rate / cfg.yaw_rate_sigma).square())

    # Reward is integrated over one 50 Hz policy step.
    reward = dt * sum(terms.values(), torch.zeros_like(phase))

    return reward, terms
