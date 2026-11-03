"""Motion-independent S54 foot tracking heads shared by Newton and BPTT.

Only command time determines contact/event masks. Actual foot pose and velocity
always come from differentiable FK, never from contact sensors or a Motion CSV.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math

import torch

from .foot_trajectory import query, command_time
from .mdp import MDP, INV_DIM, OBS_DIM, invariant_features
from .motion import CONTROL_DT
from .rotations import mat_to_6d, quat_to_mat, so3_exp, yaw_mat
from .s54 import FOOT_BODY_NAMES, XI_DIM

FOOT_CRITIC_EXTRA = 2 * (3 + 6 + 3) + 1 + 2 + 2 + 2 + XI_DIM


@dataclass(frozen=True)
class FootTrackingCfg:
    command_mode: str = 'alternating'
    episode_steps: int = 100
    domain_randomization: bool = False
    position_std: float = 0.08
    velocity_std: float = 0.5
    orientation_std: float = 0.35
    touchdown_std: float = 0.05
    base_height_std: float = 0.08
    base_xy_std: float = 0.15
    upright_std: float = 0.2
    w_swing_position: float = 1.0
    w_swing_velocity: float = 0.25
    w_touchdown: float = 0.5
    w_stance_position: float = 1.0
    w_stance_velocity: float = 0.25
    w_foot_orientation: float = 0.25
    w_upright: float = 1.0
    w_base_height: float = 0.5
    w_base_xy: float = 0.5
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0
    w_posture: float = -0.05
    w_joint_velocity: float = -0.001
    soft_limit_factor: float = 0.9
    term_height: float = 0.25
    term_tilt: float = 0.5  # 1 - Rzz, i.e. 60 degrees
    reset_joint_noise: float = 0.005
    reset_xy_noise: float = 0.005
    reset_tilt_noise: float = 0.01
    reset_yaw_noise: float = 0.05
    reset_velocity_noise: float = 0.01

    def __post_init__(self):
        if self.command_mode not in ('standing', 'alternating'):
            raise ValueError('Invalid foot command mode')
        if not isinstance(self.episode_steps, int) or self.episode_steps < 2:
            raise ValueError('episode_steps must be an integer >= 2')
        for name, value in asdict(self).items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f'Nonfinite task parameter: {name}')
            if name.endswith('_std') or name in ('term_height', 'term_tilt'):
                if value <= 0:
                    raise ValueError(f'{name} must be positive')
            if name.startswith('reset_') and value < 0:
                raise ValueError(f'{name} must be nonnegative')
            if name.startswith('w_'):
                penalty = name in ('w_action_rate', 'w_joint_limit', 'w_posture', 'w_joint_velocity')
                if (penalty and value > 0) or (not penalty and value < 0):
                    raise ValueError(f'Wrong reward weight sign: {name}')
        if not 0 < self.soft_limit_factor <= 1 or not 0 < self.term_tilt < 2:
            raise ValueError('Invalid soft joint limit factor or termination tilt')


def resolve_foot_task(value=None):
    if isinstance(value, str):
        with open(value) as stream:
            value = json.load(stream)
    return FootTrackingCfg(**(value or {}))


def assert_task_checkpoint(checkpoint, mdp):
    if not mdp.is_foot:
        return  # Preserve legacy Motion checkpoints and their evaluation convention.
    if checkpoint.get('task_contract') != mdp.contract():
        raise ValueError('Foot MDP contract mismatch: goal-only/other-task checkpoints cannot resume this task')


class FootTrackingMDP(MDP):
    is_foot = True

    def __init__(self, kin, default_q, goals, cfg=FootTrackingCfg()):
        if not goals.is_foot or goals.command_mode != cfg.command_mode:
            raise ValueError('Foot MDP and GoalSource command modes must agree')
        self.motion = None
        self._init_robot(kin, default_q, cfg)
        # S54 neutral knees/elbows are legal hard-limit endpoints. A symmetric
        # soft interval would penalize the commanded model-neutral standing pose.
        self.soft_lo = torch.minimum(self.soft_lo, default_q)
        self.soft_hi = torch.maximum(self.soft_hi, default_q)
        self.critic_dim = INV_DIM + FOOT_CRITIC_EXTRA
        self.foot_indices = [kin.tracked_names.index(n) for n in FOOT_BODY_NAMES]
        p = default_q.new_zeros(1, 3)
        R = torch.eye(3, device=p.device, dtype=p.dtype)[None]
        feet = kin.forward(p, R, p, p, default_q[None], default_q.new_zeros(1, len(default_q)))[0][:, self.foot_indices]
        self.nominal_height = float((p.new_tensor(goals.ankle_height) - feet[0, :, 2]).max())
        self.root_xy_offset = -feet[0, :, :2].mean(0).detach()
        c = cfg
        positive = (max(c.w_swing_position+c.w_swing_velocity, c.w_stance_position+c.w_stance_velocity)
                    + c.w_foot_orientation+c.w_upright+c.w_base_height+c.w_base_xy)
        if c.command_mode != 'standing':
            positive += c.w_touchdown
        if positive <= 0:
            raise ValueError('At least one positive task reward is required')
        self.max_step_reward = positive * CONTROL_DT

    def contract(self):
        return dict(version=1, mode='foot_tracking', config=asdict(self.cfg), obs_dim=OBS_DIM,
                    critic_state_dim=self.critic_dim, control_dt=CONTROL_DT,
                    reset='s54_model_neutral_grounded_v1', reward='inverse_quadratic_fk_v1')

    def nominal_state(self, n):
        q = self.default_q[None].expand(n, -1).clone()
        p = q.new_zeros(n, 3); p[:, 2] = self.nominal_height
        R = torch.eye(3, device=q.device, dtype=q.dtype)[None].expand(n, -1, -1).clone()
        return p, R, torch.zeros_like(p), torch.zeros_like(p), q, torch.zeros_like(q)

    def perturb_reset(self, S):
        p, R, v, w, q, qd = S
        c = self.cfg
        noise = lambda x, scale: (2*torch.rand_like(x)-1)*scale
        dp = torch.cat((noise(p[:, :2], c.reset_xy_noise), torch.zeros_like(p[:, 2:])), -1)
        rotation = noise(p, c.reset_tilt_noise)
        rotation = torch.cat((rotation[:, :2], noise(p[:, 2:], c.reset_yaw_noise)), -1)
        return (p+dp, so3_exp(rotation)@R, noise(v,c.reset_velocity_noise), noise(w,c.reset_velocity_noise),
                (q+noise(q,c.reset_joint_noise)).clamp(self.kin.q_lo,self.kin.q_hi), noise(qd,c.reset_velocity_noise))

    def obs(self, S, a_prev, xi, f=None, reference_R=None):
        if reference_R is None:
            raise ValueError('Foot observations require the fixed command heading')
        return super().obs(S, a_prev, xi, f, reference_R=reference_R)

    def _reference(self, command, ep_step):
        if command is None or ep_step is None:
            raise ValueError('Foot task requires its saved command and integer execution step')
        return query(command, command_time(ep_step,CONTROL_DT,command.initial_pos.dtype)[:,None])

    def _base_target(self, ref, command):
        heading = quat_to_mat(command.heading)[..., :2, :2]
        return ref.position[:, 0, :, :2].mean(-2)+(heading@self.root_xy_offset[:, None]).squeeze(-1)

    def reward(self, S, f, a, a_prev, terms=False, *, command=None, ep_step=None):
        c = self.cfg
        p, R, v, w, q, qd = S
        bp, bR, bv, _ = self.kin.forward(*S)
        bp, bR, bv = bp[:, self.foot_indices], bR[:, self.foot_indices], bv[:, self.foot_indices]
        ref = self._reference(command, ep_step)
        target, target_R = ref.position[:, 0], quat_to_mat(ref.quaternion[:, 0])
        stance = ref.contact[:, 0].to(p.dtype); swing = 1-stance
        e_p = (bp-target).square().sum(-1)
        e_v = (bv-ref.velocity[:, 0]).square().sum(-1)
        e_ori = .5*(bR-target_R).square().sum((-1,-2))
        quality = lambda error, std: 1/(1+error/std**2)
        swing_p = (swing*quality(e_p,c.position_std)).mean(-1)
        swing_v = (swing*quality(e_v,c.velocity_std)).mean(-1)
        stance_p = (stance*quality(e_p,c.position_std)).mean(-1)
        stance_v = (stance*quality(bv.square().sum(-1),c.velocity_std)).mean(-1)
        orientation = quality(e_ori,c.orientation_std).mean(-1)
        # Reward each scheduled landing only on its first crossed control interval.
        t = command_time(ep_step,CONTROL_DT,p.dtype)[:,None]
        valid = torch.arange(command.foot.shape[1],device=p.device)[None]<command.lengths[:,None]
        previous_t = command_time(ep_step-1,CONTROL_DT,p.dtype)[:,None]
        landing = valid & (command.touchdown > previous_t) & (command.touchdown <= t)
        actual_landing = bp.gather(1,command.foot[...,None].expand(-1,-1,3))
        landing_error = (actual_landing-command.target_pos).square().sum(-1)
        touchdown = (quality(landing_error,c.touchdown_std)*landing).sum(-1)/landing.sum(-1).clamp_min(1)
        foot = (c.w_swing_position*swing_p+c.w_swing_velocity*swing_v+c.w_stance_position*stance_p
                +c.w_stance_velocity*stance_v+c.w_foot_orientation*orientation+c.w_touchdown*touchdown)
        upright = quality((1-R[:,2,2]).clamp_min(0),c.upright_std)
        height = quality((p[:,2]-self.nominal_height).square(),c.base_height_std)
        balance = quality((p[:,:2]-self._base_target(ref,command)).square().sum(-1),c.base_xy_std)
        action_rate = (a-a_prev).square().sum(-1)
        limit = (torch.relu(q-self.soft_hi)+torch.relu(self.soft_lo-q)).sum(-1)
        posture = (q-self.default_q).square().sum(-1)
        joint_velocity = qd.square().mean(-1)
        track = foot+c.w_upright*upright+c.w_base_height*height+c.w_base_xy*balance
        reward = (track+c.w_action_rate*action_rate+c.w_joint_limit*limit+c.w_posture*posture
                  +c.w_joint_velocity*joint_velocity)*CONTROL_DT
        if not terms:
            return reward
        return reward, dict(swing_tracking=swing_p, swing_velocity=swing_v, touchdown_accuracy=touchdown,
                            touchdown_events=landing.sum(-1), stance_stability=stance_p, stance_velocity=stance_v,
                            foot_orientation=orientation, base_upright=upright, base_height=height, base_balance=balance,
                            action_rate=action_rate, joint_limit=limit, posture=posture,
                            foot_position_error=e_p.mean(-1), foot_velocity_error=e_v.mean(-1),
                            foot_objective=foot*CONTROL_DT, track=track*CONTROL_DT,
                            head_position_error=(q[:,self.head_indices]-self.default_q[self.head_indices]).square().sum(-1),
                            head_penalty=torch.zeros_like(reward))

    def termination(self, S, f=None, *, command=None, ep_step=None):
        p, R = S[:2]
        finite = torch.stack([x.flatten(1).isfinite().all(-1) for x in S]).all(0)
        return (~finite | ((p[:,2]-self.nominal_height).abs()>self.cfg.term_height)
                | ((1-R[:,2,2])>self.cfg.term_tilt))

    def soft_termination(self, S, f=None, tz=.02, tt=.04, *, command=None, ep_step=None):
        p, R = S[:2]
        alive = ((1-torch.sigmoid(((p[:,2]-self.nominal_height).abs()-self.cfg.term_height)/tz))
                 *(1-torch.sigmoid((1-R[:,2,2]-self.cfg.term_tilt)/tt)))
        return 1-alive

    def critic_features(self, S, f, xi, *, command=None, ep_step=None):
        ref = self._reference(command,ep_step)
        p,R = S[:2]
        bp,bR,bv,_ = self.kin.forward(*S)
        bp,bR,bv = bp[:,self.foot_indices],bR[:,self.foot_indices],bv[:,self.foot_indices]
        Ht = yaw_mat(R).transpose(-1,-2)
        pos = (Ht[:,None]@(ref.position[:,0]-bp)[...,None]).squeeze(-1).flatten(1)
        vel = (Ht[:,None]@(ref.velocity[:,0]-bv)[...,None]).squeeze(-1).flatten(1)
        ori = mat_to_6d(bR.transpose(-1,-2)@quat_to_mat(ref.quaternion[:,0])).flatten(1)
        base_xy = (self._base_target(ref,command)-p[:,:2])
        base_xy = (Ht[:,:2,:2]@base_xy[...,None]).squeeze(-1)
        return torch.cat((invariant_features(S),pos,ori,vel,self.nominal_height-p[:,2:3],base_xy,
                          ref.contact[:,0].to(p.dtype),ref.swing_phase[:,0],xi),-1)
