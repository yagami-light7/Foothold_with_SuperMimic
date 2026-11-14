"""Goal contracts shared by collection, replay, imagination and evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json

import torch

from .foot_trajectory import AlternatingConfig, query, sample_alternating, command_time
from .motion import CONTROL_DT
from .rotations import mat_to_6d, mat_to_quat, quat_to_mat, yaw_mat
from .robot_contract import FOOT_BODY_NAMES, GOAL_FRAME_DIM

FOOT_FRAME_DIM = 23


@dataclass(frozen=True)
class GoalSpec:
    mode: str = 'motion'
    frames: int = 10
    dt: float | None = None

    def __post_init__(self):
        if self.mode not in ('motion', 'foot_trajectory','reference_foot', 'roban_foot') or not isinstance(self.frames, int) or self.frames < 1:
            raise ValueError('Invalid goal mode or number of future samples')
        dt = (CONTROL_DT if self.mode == 'motion' else 0.1) if self.dt is None else self.dt
        if not 0 < dt < float('inf'):
            raise ValueError('goal_dt must be finite and positive')
        if self.mode == 'motion' and abs(dt - CONTROL_DT) > 1e-9:
            raise ValueError('Motion mode preserves its original control-frame spacing (0.02s)')
        object.__setattr__(self, 'dt', dt)

    @property
    def frame_dim(self):
        return GOAL_FRAME_DIM if self.mode == 'motion' else FOOT_FRAME_DIM

    @property
    def dim(self):
        return self.frames * self.frame_dim

    def contract(self):
        value = dict(version=3 if self.mode=='roban_foot' else 2 if self.mode=='reference_foot' else 1, **asdict(self), frame_dim=self.frame_dim, dim=self.dim,
                    coordinates='reference_heading' if self.mode == 'motion' else 'current_root_yaw',
                    rotation='6d_columns', quaternion='wxyz',
                    observation_reference='proprio_only_sole_goals' if self.mode == 'roban_foot' else 'motion_root' if self.mode == 'motion' else 'fixed_command_heading')
        if self.mode == 'roban_foot':
            value.update(foot_frame='model_collision_sole',stance='fixed_at_planned_touchdown',
                         contact='planned_schedule; truth_validity_is_privileged')
        return value

    def assert_checkpoint(self, checkpoint, foot_config=None):
        saved = checkpoint.get('goal_contract')
        if saved is None:
            a = checkpoint.get('args', {})
            # Absence of a contract is accepted only for legacy Motion weights.
            if a.get('goal_mode', 'motion') != 'motion':
                raise ValueError('Foot trajectory checkpoint is missing goal_contract')
            saved = GoalSpec('motion', a.get('goal_frames', 10)).contract()
        if saved != self.contract():
            raise ValueError(f'Checkpoint goal contract mismatch: saved={saved}, requested={self.contract()}')
        if self.mode == 'foot_trajectory' and foot_config is not None:
            if checkpoint.get('args', {}).get('foot_config') != json.loads(json.dumps(asdict(foot_config))):
                raise ValueError('Checkpoint foot sampling configuration differs; resume with the saved configuration')


def add_goal_args(parser):
    parser.add_argument('--goal_mode', choices=['motion', 'foot_trajectory','reference_foot','roban_foot'], default='motion')
    parser.add_argument('--goal_dt', type=float, default=None, help='Future sample spacing: motion=0.02, foot=0.1 seconds')
    parser.add_argument('--foot_config', default=None, help='AlternatingConfig JSON; motion mode does not use it')


def resolve_goal_config(args):
    spec = GoalSpec(args.goal_mode, args.goal_frames, args.goal_dt)
    value = args.foot_config
    if isinstance(value, str):
        with open(value) as stream:
            value = json.load(stream)
    cfg = AlternatingConfig(**(value or {}))
    # JSON-compatible form is also what checkpoint comparisons see on resume.
    args.goal_dt = spec.dt
    args.foot_config = json.loads(json.dumps(asdict(cfg)))
    return spec, cfg


def foot_goal(S, command, ep_step, spec):
    """23 channels/sample: [left(p3,R6,c,phase), right(...), delta_t]."""
    p, R = S[:2]
    offsets64 = torch.arange(1, spec.frames + 1, device=p.device, dtype=torch.float64) * spec.dt
    times = (command_time(ep_step,CONTROL_DT,torch.float64)[:,None]+offsets64[None]).to(p.dtype)
    offsets = offsets64.to(p.dtype)
    ref = query(command, times)
    return encode_foot_samples(S,ref,offsets)


def encode_foot_samples(S,ref,offsets):
    """Shared 23D encoding; the reference source does not enter the Actor."""
    p,R = S[:2]
    Ht = yaw_mat(R).transpose(-1, -2)[:, None, None]
    local_p = (Ht @ (ref.position - p[:, None, None]).unsqueeze(-1)).squeeze(-1)
    local_R = mat_to_6d(Ht @ quat_to_mat(ref.quaternion))
    feet = torch.cat((local_p, local_R, ref.contact[..., None].to(p.dtype), ref.swing_phase[..., None]), -1)
    return torch.cat((feet.flatten(-2), offsets[None, :, None].expand(p.shape[0], -1, -1)), -1).flatten(1)


class GoalSource:
    def __init__(self, motion, kin, spec=GoalSpec(), foot_config=AlternatingConfig(), command_mode='alternating'):
        self.motion, self.kin, self.spec, self.foot_config = motion, kin, spec, foot_config
        self.is_foot = spec.mode != 'motion'
        if command_mode not in ('alternating', 'standing', 'curriculum'):
            raise ValueError('Expected alternating or standing foot commands')
        self.command_mode = command_mode
        self.curriculum = None
        self.fixed_command = None
        if self.is_foot:
            # Extract the existing grounding implementation without changing its geometry.
            from .foot_geometry import FootGeometry
            from .robot_contract import model_contract
            geometry = FootGeometry(model_contract()[0])
            import numpy as np
            self.ankle_height = [0.003 - geometry.local_low(side, np.eye(3)) for side in (0, 1)]
            self.foot_indices = [kin.tracked_names.index(n) for n in FOOT_BODY_NAMES]

    def new_command(self, S, generator=None, frames=None):
        if not self.is_foot:
            return None
        feet = self.kin.forward(*S)[0][:, self.foot_indices].clone()
        feet[..., 2] = feet.new_tensor(self.ankle_height)
        heading = mat_to_quat(yaw_mat(S[1]))
        initial_quat = heading[:, None].expand(-1, 2, -1).clone()
        if self.command_mode == 'curriculum':
            if self.curriculum is None: raise ValueError('Curriculum sampler is required')
            return self.curriculum.plan(feet, initial_quat, heading, self.command_duration, self.fixed_command)
        command = sample_alternating(feet, initial_quat, heading, self.foot_config, generator)
        if self.command_mode == 'standing':
            # Reuse the generator's existing zero-length stationary command support.
            command.lengths.zero_()
        return command

    def goal(self, S, f, ep_step=None, command=None):
        if not self.is_foot:
            return self.motion.goal(f, self.spec.frames)
        if ep_step is None or command is None:
            raise ValueError('Foot goal requires a command snapshot and command execution step')
        return foot_goal(S, command, ep_step, self.spec)

    def obs(self, mdp, S, a_prev, xi, f, command=None):
        reference_R = None
        if self.is_foot:
            if command is None:
                raise ValueError('Foot observation requires a command snapshot')
            reference_R = quat_to_mat(command.heading)
            while reference_R.ndim < S[1].ndim:
                reference_R = reference_R.unsqueeze(-3)
        return mdp.obs(S, a_prev, xi, f, reference_R=reference_R)
