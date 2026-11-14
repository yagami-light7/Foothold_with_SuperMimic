"""Analytic MDP heads shared by the simulator and the world model (paper Appendix A).

State tuple ``S = (p, R, v, w, q, qd)``: base_link position, rotation matrix, world linear/angular
velocity of the base_link link frame, joint positions and velocities.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path

import torch

from .kinematics import S54Kinematics
from .motion import CONTROL_DT, MotionLib
from .rotations import mat_to_6d, quat_to_mat, mat_to_quat, rot_angle_sq, yaw_mat

from .robot_contract import NUM_J, XI_DIM, TRACKED_BODIES, JOINT_OFFSET, COM_OFFSET, HEAD_JOINT_NAMES, JOINT_NAMES, FOOT_BODY_NAMES

STATE_DIM = 3 + 4 + 3 + 3 + 2 * NUM_J
# Legacy/independent-Foot interface. Motion training selects MDP.obs_dim=103.
OBS_DIM = 3 * NUM_J + 3 + 3 + 6
MOTION_OBS_DIM = OBS_DIM + 4
MOTION_OBS_MODES = ('legacy_v1', 'root_xy_heading_v2')
LEGACY_REWARD_FIELDS = (
    'body_pos_std', 'body_ori_std', 'body_linvel_std', 'body_angvel_std',
    'root_pos_std', 'root_ori_std', 'w_action_rate', 'w_joint_limit',
    'w_head_position', 'w_head_velocity', 'soft_limit_factor', 'term_height', 'term_tilt',
)

# Table A.1 sensor noise (uniform half-widths), plus the reference-relative orientation
OBS_NOISE = torch.cat(
    (
        torch.full((NUM_J,), 0.01),
        torch.full((NUM_J,), 0.5),
        torch.full((3,), 0.2),
        torch.full((3,), 0.05),
        torch.zeros(NUM_J),
        torch.full((6,), 0.05),
    )
)
# normalization of the obs before the policy (qd and angular velocity are large)
OBS_SCALE = torch.cat(
    (torch.ones(NUM_J), torch.full((NUM_J,), 0.1), torch.full((3,), 0.25), torch.ones(3), torch.ones(NUM_J), torch.ones(6))
)
MAX_STEP_REWARD = (4 * 1.0 + 2 * 0.5) * CONTROL_DT


@dataclass
class RewardCfg:
    body_pos_std: float = 0.3
    body_ori_std: float = 0.4
    body_linvel_std: float = 1.0
    body_angvel_std: float = 3.14
    root_pos_std: float = 0.3
    root_ori_std: float = 0.4
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0
    # Heads are neutral-filled, and absent from the 14 tracked bodies. Without
    # this term any constant head angle inside the limits has the same reward.
    w_head_position: float = -0.5
    w_head_velocity: float = -0.005
    soft_limit_factor: float = 0.9
    term_height: float = 0.25
    term_tilt: float = 0.8
    # Legacy defaults preserve the reward saved by existing checkpoints.
    w_body_pos: float = 1.0
    w_body_ori: float = 1.0
    w_body_linvel: float = 1.0
    w_body_angvel: float = 1.0
    w_root_pos: float = 0.5
    w_root_ori: float = 0.5
    # 默认关闭，避免旧 checkpoint 评估时自动加入新奖励
    w_foot_pos: float = 0.0
    w_foot_ori: float = 0.0
    foot_pos_std: float = 0.15   # m
    foot_ori_std: float = 0.3    # rad
    root_position_mode: str = 'xyz'

    def __post_init__(self):
        if self.root_position_mode not in ('xy', 'xyz'):
            raise ValueError('root_position_mode must be xy or xyz')
        penalties = ('w_action_rate', 'w_joint_limit', 'w_head_position', 'w_head_velocity')
        for name, value in asdict(self).items():
            if name == 'root_position_mode':
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'Nonfinite/non-numeric reward parameter: {name}')
            if name.endswith('_std') and value <= 0:
                raise ValueError(f'{name} must be positive')
            if name.startswith('w_') and ((name in penalties and value > 0) or (name not in penalties and value < 0)):
                raise ValueError(f'Wrong reward weight sign: {name}')
        if self.positive_weight_sum <= 0 or not 0 < self.soft_limit_factor <= 1:
            raise ValueError('Positive reward sum and valid soft_limit_factor required')
        if self.term_height <= 0 or not 0 < self.term_tilt < 2:
            raise ValueError('Invalid termination thresholds')

    @property
    def positive_weight_sum(self):
        return (self.w_body_pos + self.w_body_ori + self.w_body_linvel + self.w_body_angvel
                + self.w_root_pos + self.w_root_ori + self.w_foot_pos + self.w_foot_ori)

    @classmethod
    def tracking_v2(cls):
        """Requested table, plus the two retained velocity terms (weight 1 each)."""
        return cls(w_root_pos=1., root_pos_std=.20, root_position_mode='xy',
                   w_root_ori=.75, root_ori_std=.30,
                   w_foot_pos=.50, foot_pos_std=.08, w_foot_ori=.25, foot_ori_std=.20)


def resolve_motion_reward(value=None):
    defaults = asdict(RewardCfg.tracking_v2())
    if isinstance(value, str):
        value = json.loads(value) if value.lstrip().startswith('{') else json.loads(Path(value).read_text())
    if value is not None and not isinstance(value, dict):
        raise ValueError('Motion reward config must be a JSON object or JSON file')
    defaults.update(value or {})
    return RewardCfg(**defaults)


def motion_reward_contract(cfg):
    legacy = (cfg.root_position_mode == 'xyz' and
              (cfg.w_body_pos, cfg.w_body_ori, cfg.w_body_linvel, cfg.w_body_angvel,
               cfg.w_root_pos, cfg.w_root_ori, cfg.w_foot_pos, cfg.w_foot_ori) == (1., 1., 1., 1., .5, .5, 0., 0.))
    config = asdict(cfg)
    return dict(version=1 if legacy else 2,
                config={k: config[k] for k in LEGACY_REWARD_FIELDS} if legacy else config,
                reward='tracked_bodies_root_head_v1' if legacy else 'weighted_root_feet_tracking_v2')

def flatten_state(S) -> torch.Tensor:
    p, R, v, w, q, qd = S
    return torch.cat((p, mat_to_quat(R), v, w, q, qd), -1)


def unflatten_state(x: torch.Tensor):
    return x[..., 0:3], quat_to_mat(x[..., 3:7]), x[..., 7:10], x[..., 10:13], x[..., 13:13+NUM_J], x[..., 13+NUM_J:STATE_DIM]


def invariant_features(S) -> torch.Tensor:
    """Heading- and position-invariant state features: height, tilt (6D), body-frame velocities, joints."""
    p, R, v, w, q, qd = S
    H = yaw_mat(R)
    Rt = R.transpose(-1, -2)
    return torch.cat(
        (
            p[..., 2:3],
            mat_to_6d(H.transpose(-1, -2) @ R),
            (Rt @ v.unsqueeze(-1)).squeeze(-1),
            (Rt @ w.unsqueeze(-1)).squeeze(-1),
            q,
            qd,
        ),
        -1,
    )


INV_DIM = 1 + 6 + 3 + 3 + 2 * NUM_J


class MDP:
    is_foot = False
    max_step_reward = MAX_STEP_REWARD

    def contract(self):
        value = dict(**motion_reward_contract(self.cfg), mode='motion', obs_dim=self.obs_dim,
                     critic_state_dim=self.critic_dim, control_dt=CONTROL_DT)
        if self.observation_mode != 'legacy_v1':
            value.update(version=2, observation_mode=self.observation_mode,
                         root_feedback='reference_minus_actual_xy_in_actual_yaw; sin_cos_reference_minus_actual_yaw')
        return value

    def __init__(self, kin: S54Kinematics, motion: MotionLib, default_q: torch.Tensor, cfg: RewardCfg = RewardCfg(),
                 observation_mode='legacy_v1'):
        if observation_mode not in MOTION_OBS_MODES:
            raise ValueError('Unknown Motion observation mode')
        self.motion = motion
        self._init_robot(kin, default_q, cfg)
        self.observation_mode = observation_mode
        if observation_mode == 'root_xy_heading_v2':
            self.obs_dim = MOTION_OBS_DIM
            self.obs_noise = torch.cat((self.obs_noise, default_q.new_zeros(4)))
            self.obs_scale = torch.cat((self.obs_scale, default_q.new_ones(4)))
        self.foot_indices = [kin.tracked_names.index(n) for n in FOOT_BODY_NAMES]
        self.max_step_reward = CONTROL_DT * cfg.positive_weight_sum
        assert motion.q.shape[-1] == NUM_J and motion.body_p.shape[-2] == len(TRACKED_BODIES)

    def _init_robot(self, kin, default_q, cfg):
        """Robot-only proprioception and limits shared with independent foot tasks."""
        self.kin, self.cfg = kin, cfg
        self.obs_dim = OBS_DIM
        assert default_q.numel() == NUM_J
        assert kin.tracked_names == TRACKED_BODIES
        self.critic_dim = INV_DIM + CRITIC_EXTRA
        self.default_q = default_q
        mid, half = (kin.q_lo + kin.q_hi) / 2, (kin.q_hi - kin.q_lo) / 2
        self.soft_lo, self.soft_hi = mid - half * cfg.soft_limit_factor, mid + half * cfg.soft_limit_factor
        dev = default_q.device
        self.obs_noise = OBS_NOISE.to(dev)
        self.obs_scale = OBS_SCALE.to(dev)
        self.head_indices = torch.tensor([JOINT_NAMES.index(n) for n in HEAD_JOINT_NAMES], device=dev)

    # ---------------- observation ----------------
    def obs(self, S, a_prev, xi, f, reference_R=None) -> torch.Tensor:
        """Legacy proprioception, optionally followed by local XY and heading feedback."""
        p, R, v, w, q, qd = S
        Rt = R.transpose(-1, -2)
        dq0 = xi[..., JOINT_OFFSET]
        g_b = -R[..., 2, :]  # R^T (0,0,-1)
        ref_R = self.motion.root_R[f] if reference_R is None else reference_R
        rel = mat_to_6d(Rt @ ref_R)
        base = torch.cat((q - self.default_q - dq0, qd, (Rt @ w.unsqueeze(-1)).squeeze(-1), g_b, a_prev, rel), -1)
        if getattr(self, 'observation_mode', 'legacy_v1') == 'legacy_v1':
            return base
        Ht = yaw_mat(R).transpose(-1, -2)
        position_error = (Ht @ (self.motion.root_p[f] - p).unsqueeze(-1)).squeeze(-1)[..., :2]
        heading = Ht @ yaw_mat(ref_R)
        # sin/cos avoids an angle discontinuity at +/-pi.
        heading_error = torch.stack((heading[..., 1, 0], heading[..., 0, 0]), -1)
        return torch.cat((base, position_error, heading_error), -1)

    def add_noise(self, o: torch.Tensor) -> torch.Tensor:
        return o + (2 * torch.rand_like(o) - 1) * self.obs_noise

    # ---------------- reward / termination ----------------
    def reference_bodies(self, S, f):
        """Legacy body anchor. Terrain tasks can explicitly select world targets."""
        p, R = S[:2]
        rp, rR = self.motion.root_p[f], self.motion.root_R[f]
        dR = yaw_mat(R @ rR.transpose(-1, -2))
        anchor = torch.cat((p[..., :2], rp[..., 2:3]), -1)
        ref_bp = anchor.unsqueeze(-2) + ((dR.unsqueeze(-3) @
            (self.motion.body_p[f] - rp.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1))
        return ref_bp, dR.unsqueeze(-3) @ self.motion.body_R[f]

    def reward(self, S, f, a, a_prev, terms: bool = False, *, command=None, ep_step=None):
        """Per-step reward (already multiplied by dt) comparing S with reference frame f."""
        c, m = self.cfg, self.motion
        p, R, v, w, q, qd = S
        bp, bR, bv, bw = self.kin.forward(p, R, v, w, q, qd)
        rp, rR = m.root_p[f], m.root_R[f]
        ref_bp, ref_bR = self.reference_bodies(S, f)
        e_pos = ((ref_bp - bp) ** 2).sum(-1).mean(-1)
        e_ori = rot_angle_sq(ref_bR, bR).mean(-1)
        e_lv = ((m.body_v[f] - bv) ** 2).sum(-1).mean(-1)
        e_av = ((m.body_w[f] - bw) ** 2).sum(-1).mean(-1)
        e_root_xy = (rp[..., :2] - p[..., :2]).square().sum(-1)
        e_rp = e_root_xy if c.root_position_mode == 'xy' else ((rp - p) ** 2).sum(-1)
        e_ro = rot_angle_sq(rR, R)
        # 每只脚的世界系位置误差，保留全局轨迹偏差
        e_fp = (
            bp[..., self.foot_indices, :]
            - m.body_p[f][..., self.foot_indices, :]
        ).square().sum(-1)

        # 每只脚的世界系姿态误差：SO(3) 旋转夹角的平方
        e_fo = rot_angle_sq(
            bR[..., self.foot_indices, :, :],
            m.body_R[f][..., self.foot_indices, :, :],
        )

        r_fp = torch.exp(-e_fp / c.foot_pos_std**2).mean(-1)
        r_fo = torch.exp(-e_fo / c.foot_ori_std**2).mean(-1)

        root_position_reward = c.w_root_pos * torch.exp(-e_rp / c.root_pos_std**2)
        root_orientation_reward = c.w_root_ori * torch.exp(-e_ro / c.root_ori_std**2)

        r_track = (
            c.w_body_pos * torch.exp(-e_pos / c.body_pos_std**2)
            + c.w_body_ori * torch.exp(-e_ori / c.body_ori_std**2)
            + c.w_body_linvel * torch.exp(-e_lv / c.body_linvel_std**2)
            + c.w_body_angvel * torch.exp(-e_av / c.body_angvel_std**2)
            + root_position_reward
            + root_orientation_reward
            + c.w_foot_pos * r_fp
            + c.w_foot_ori * r_fo
        )

        r_rate = ((a - a_prev) ** 2).sum(-1)
        r_lim = (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1)
        e_head = (q[..., self.head_indices] - m.q[f][..., self.head_indices]).square().sum(-1)
        e_head_velocity = (qd[..., self.head_indices] - m.qd[f][..., self.head_indices]).square().sum(-1)
        r_head = c.w_head_position * e_head + c.w_head_velocity * e_head_velocity
        r = (r_track + c.w_action_rate * r_rate + c.w_joint_limit * r_lim + r_head) * CONTROL_DT
        if terms:
            return r, {
                "track": r_track * CONTROL_DT,
                "e_body_pos": e_pos,
                "e_body_ori": e_ori,
                "e_root_pos": e_rp,
                "e_root_xy": e_root_xy,
                "e_root_ori": e_ro,
                "root_position_reward": root_position_reward * CONTROL_DT,
                "root_orientation_reward": root_orientation_reward * CONTROL_DT,
                "action_rate": r_rate,
                "joint_limit": r_lim,
                "head_position_error": e_head,
                "head_velocity_error": e_head_velocity,
                "head_penalty": r_head * CONTROL_DT,
                "e_foot_pos": e_fp.mean(-1),
                "e_foot_ori": e_fo.mean(-1),
                "foot_position_reward": c.w_foot_pos * r_fp * CONTROL_DT,
                "foot_orientation_reward": c.w_foot_ori * r_fo * CONTROL_DT,
            }
        return r

    def termination(self, S, f, *, command=None, ep_step=None) -> torch.Tensor:
        p, R = S[0], S[1]
        rp, rR = self.motion.root_p[f], self.motion.root_R[f]
        e_z = (rp[..., 2] - p[..., 2]).abs()
        e_tilt = (rR[..., 2, 2] - R[..., 2, 2]).abs()
        return (e_z > self.cfg.term_height) | (e_tilt > self.cfg.term_tilt)

    def soft_termination(self, S, f, tz: float = 0.02, tt: float = 0.04, *, command=None, ep_step=None) -> torch.Tensor:
        """Differentiable termination probability for imagined rollouts."""
        p, R = S[0], S[1]
        rp, rR = self.motion.root_p[f], self.motion.root_R[f]
        e_z = (rp[..., 2] - p[..., 2]).abs()
        e_tilt = (rR[..., 2, 2] - R[..., 2, 2]).abs()
        alive = (1 - torch.sigmoid((e_z - self.cfg.term_height) / tz)) * (1 - torch.sigmoid((e_tilt - self.cfg.term_tilt) / tt))
        return 1 - alive

    def critic_features(self, S, f, xi, *, command=None, ep_step=None) -> torch.Tensor:
        """Critic input: invariant state, goal, sim context, and robot-vs-reference errors."""
        m = self.motion
        p, R, v, w, q, qd = S
        H = yaw_mat(R)
        Ht = H.transpose(-1, -2)
        rp, rR = m.root_p[f], m.root_R[f]
        bp = self.kin.forward(p, R, v, w, q, qd)[0]
        ref_bp, _ = self.reference_bodies(S, f)
        e_body = (Ht.unsqueeze(-3) @ (ref_bp - bp).unsqueeze(-1)).squeeze(-1).flatten(-2)
        e_root = (Ht @ (rp - p).unsqueeze(-1)).squeeze(-1)
        rel = mat_to_6d(R.transpose(-1, -2) @ rR)
        return torch.cat((invariant_features(S), m.q[f] - q, e_body, e_root, rel, xi), -1)


CRITIC_EXTRA = NUM_J + len(TRACKED_BODIES) * 3 + 3 + 6 + XI_DIM


def normalize_xi(xi: torch.Tensor) -> torch.Tensor:
    return torch.cat(((xi[..., :1] - 0.75) / 0.45, xi[..., JOINT_OFFSET] / 0.01, xi[..., COM_OFFSET] / 0.05), -1)
