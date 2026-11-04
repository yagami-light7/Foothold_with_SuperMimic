"""Stateless batched foot trajectories. SI units; quaternions are wxyz.

Coordinates are in a fixed inertial frame (the simulator world minus its fixed
environment origin). No robot pose or mutable clock participates in query().
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math

import torch

from .rotations import quat_slerp, quat_to_mat


@dataclass
class FootstepCommand:
    initial_pos: torch.Tensor       # (B,2,3), left then right, ankle origins
    initial_quat: torch.Tensor      # (B,2,4)
    foot: torch.Tensor              # (B,M), 0=left, 1=right
    target_pos: torch.Tensor        # (B,M,3)
    target_quat: torch.Tensor       # (B,M,4)
    liftoff: torch.Tensor           # (B,M), seconds from command start
    touchdown: torch.Tensor         # (B,M)
    height: torch.Tensor            # (B,M), apex above higher endpoint
    lengths: torch.Tensor           # (B,), valid prefix, padded events ignored
    heading: torch.Tensor           # (B,4), fixed upright observation reference

    # Optional immutable schedule labels: [start_s,length,landings_per_s,yaw_rate,task_id].
    # Geometry remains the source of truth for reward/goal; labels are telemetry only.
    labels: torch.Tensor | None = None

    def tensors(self):
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self,f.name) is not None}

    def clone(self):
        return FootstepCommand(**{k: v.detach().clone() for k, v in self.tensors().items()})

    def index(self, indices):
        if isinstance(indices, int):
            indices = slice(indices, indices + 1)
        return FootstepCommand(**{k: v[indices] for k, v in self.tensors().items()})

    @staticmethod
    def cat(commands):
        return FootstepCommand(**{f.name: torch.cat([getattr(c, f.name) for c in commands]) for f in fields(FootstepCommand) if getattr(commands[0],f.name) is not None})

    def assign(self, indices, command):
        """Update live environment slots only; replay owns detached snapshots."""
        for k, v in self.tensors().items():
            v[indices] = getattr(command, k)

    def repeat(self, count):
        """Time-major copies for flattened (horizon,batch) reward/critic calls."""
        return FootstepCommand(**{k: v.unsqueeze(0).expand((count,) + v.shape).flatten(0, 1)
                                  for k, v in self.tensors().items()})

    def validate(self):
        if not self.initial_pos.is_floating_point():
            raise ValueError('Positions must have a floating point dtype')
        B, M = self.foot.shape
        expected = dict(initial_pos=(B, 2, 3), initial_quat=(B, 2, 4), foot=(B, M),
                        target_pos=(B, M, 3), target_quat=(B, M, 4), liftoff=(B, M),
                        touchdown=(B, M), height=(B, M), lengths=(B,), heading=(B, 4))
        if self.labels is not None: expected['labels']=(B,M,5)
        if M < 1:
            raise ValueError('Use at least one padded event, with lengths=0 for stationary commands')
        for k, v in self.tensors().items():
            if v.shape != expected[k] or v.device != self.initial_pos.device or not torch.isfinite(v).all():
                raise ValueError(f'Invalid command field {k}')
            if v.is_floating_point() and v.dtype != self.initial_pos.dtype:
                raise ValueError('Command floating point dtypes must agree')
        if self.foot.dtype != torch.long or self.lengths.dtype != torch.long:
            raise ValueError('foot and lengths must be int64')
        valid = torch.arange(M, device=self.foot.device)[None] < self.lengths[:, None]
        if ((self.lengths < 0) | (self.lengths > M)).any():
            raise ValueError('Invalid command length')
        if (valid & ((self.foot < 0) | (self.foot > 1) | (self.liftoff < 0)
                     | (self.touchdown <= self.liftoff) | (self.height < 0))).any():
            raise ValueError('Invalid foot, timing or height')
        if (valid[:, 1:] & (self.liftoff[:, 1:] < self.liftoff[:, :-1])).any():
            raise ValueError('Events must be ordered by liftoff')
        for side in (0, 1):
            mask = valid & (self.foot == side)
            ends = torch.where(mask, self.touchdown, 0).cummax(-1).values
            previous = torch.cat((ends[:, :1] * 0, ends[:, :-1]), -1)
            if (mask & (self.liftoff < previous)).any():
                raise ValueError('Overlapping swings of the same foot')
        for q in (self.initial_quat, self.heading):
            if (q.norm(dim=-1) - 1).abs().max() > 1e-4:
                raise ValueError('Initial/heading quaternion must be unit length')
        if (valid & ((self.target_quat.norm(dim=-1) - 1).abs() > 1e-4)).any():
            raise ValueError('Target quaternion must be unit length')
        return self


@dataclass
class FootTrajectory:
    position: torch.Tensor          # (B,Q,2,3)
    quaternion: torch.Tensor        # (B,Q,2,4)
    velocity: torch.Tensor          # (B,Q,2,3)
    acceleration: torch.Tensor      # (B,Q,2,3)
    contact: torch.Tensor           # (B,Q,2), bool; both true = double support
    swing_phase: torch.Tensor       # (B,Q,2), zero during stance


def _quintic(u):
    return (u**3 * (10 + u * (-15 + 6 * u)),
            30 * u**2 * (1 - u)**2, 60 * u * (1 - u) * (1 - 2 * u))


def command_time(steps, dt, dtype):
    """Integer control ticks -> seconds, rounded once in the trajectory dtype.

    Multiplying float32 ticks by float32 0.02 can place tick 40 just BELOW a
    supplied float32 touchdown=0.8. Use the same conversion in every MDP head.
    """
    return (steps.to(torch.float64)*dt).to(dtype)


def _gather(table, indices):
    tail = table.shape[2:]
    return table.gather(1, indices.reshape(indices.shape + (1,) * len(tail)).expand(indices.shape + tail))


def query(command: FootstepCommand, times: torch.Tensor) -> FootTrajectory:
    """Query arbitrary (B,Q) times; before start/after end hold initial/final stance.

    Commands must be validated at creation, outside compiled rollout functions.
    All selection is tensor based, so query is suitable for torch.compile.
    """
    c = command
    if times.ndim != 2 or times.shape[0] != c.foot.shape[0]:
        raise ValueError('times must have shape (B,Q)')
    B, M = c.foot.shape
    ids = torch.arange(1, M + 1, device=c.foot.device)[None].expand(B, -1)
    valid = ids <= c.lengths[:, None]
    outputs = []
    for side in (0, 1):
        is_side = valid & (c.foot == side)
        selected = torch.where(is_side[:, None] & (times[:, :, None] >= c.liftoff[:, None]), ids[:, None], 0).amax(-1)
        previous = torch.where(is_side, ids, 0).cummax(-1).values
        previous = torch.cat((ids[:, :1] * 0, previous[:, :-1]), 1)
        previous = _gather(torch.cat((ids[:, :1] * 0, previous), 1), selected)
        positions = torch.cat((c.initial_pos[:, side:side+1], c.target_pos), 1)
        quats = torch.cat((c.initial_quat[:, side:side+1], c.target_quat), 1)
        p0, p1 = _gather(positions, previous), _gather(positions, selected)
        q0, q1 = _gather(quats, previous), _gather(quats, selected)
        start = _gather(torch.cat((c.liftoff[:, :1] * 0, c.liftoff), 1), selected)
        end = _gather(torch.cat((torch.ones_like(c.touchdown[:, :1]), c.touchdown), 1), selected)
        height = _gather(torch.cat((c.height[:, :1] * 0, c.height), 1), selected)
        duration = (end - start).clamp_min(torch.finfo(times.dtype).eps)
        u = ((times - start) / duration).clamp(0, 1)
        s, ds, dds = _quintic(u)
        dp = p1 - p0
        pos = p0 + s[..., None] * dp
        vel = ds[..., None] / duration[..., None] * dp
        acc = dds[..., None] / duration[..., None]**2 * dp
        # Two quintics give a C2 apex, including zero vertical speed at mid-swing.
        apex = torch.maximum(p0[..., 2], p1[..., 2]) + height
        ascending = u < 0.5
        z0 = torch.where(ascending, p0[..., 2], apex)
        z1 = torch.where(ascending, apex, p1[..., 2])
        sz, dz, ddz = _quintic(torch.where(ascending, 2 * u, 2 * u - 1))
        pos = torch.cat((pos[..., :2], (z0 + sz * (z1 - z0))[..., None]), -1)
        vel = torch.cat((vel[..., :2], (dz * (z1 - z0) * 2 / duration)[..., None]), -1)
        acc = torch.cat((acc[..., :2], (ddz * (z1 - z0) * 4 / duration**2)[..., None]), -1)
        swinging = (selected > 0) & (times < end)
        outputs.append((pos, quat_slerp(q0, q1, s), vel, acc, ~swinging,
                        torch.where(swinging, u, 0)))
    return FootTrajectory(*(torch.stack([o[i] for o in outputs], 2) for i in range(6)))


@dataclass(frozen=True)
class AlternatingConfig:
    num_steps: int = 32
    first_foot: int = 0
    initial_support: float = 0.1
    step_length: tuple[float, float] = (0.08, 0.20)
    swing_time: tuple[float, float] = (0.35, 0.55)
    double_support: tuple[float, float] = (0.08, 0.15)
    height: tuple[float, float] = (0.04, 0.08)
    yaw_offset: tuple[float, float] = (-0.05, 0.05)

    def __post_init__(self):
        if (not isinstance(self.num_steps, int) or self.num_steps < 1 or self.first_foot not in (0, 1)
                or not math.isfinite(self.initial_support) or self.initial_support < 0):
            raise ValueError('Invalid step count, first foot or initial support')
        for name in ('step_length', 'swing_time', 'double_support', 'height', 'yaw_offset'):
            bounds = getattr(self, name)
            if len(bounds) != 2 or not all(math.isfinite(v) for v in bounds) or bounds[0] > bounds[1]:
                raise ValueError(f'Invalid sampling range: {name}')
        if self.swing_time[0] <= 0 or self.double_support[0] < 0 or self.height[0] < 0:
            raise ValueError('Swing time must be positive, support/height nonnegative')


def sample_alternating(initial_pos, initial_quat, heading, cfg=AlternatingConfig(), generator=None):
    """Sample a flat-ground command once; step width is inherited from initial feet.

    Input feet should already be grounded and yaw-only. Heading is fixed during
    an episode; individual landing yaw offsets do not change the travel axis.
    """
    B, M = initial_pos.shape[0], cfg.num_steps
    def uniform(bounds):
        return torch.rand(B, M, device=initial_pos.device, dtype=initial_pos.dtype,
                          generator=generator) * (bounds[1] - bounds[0]) + bounds[0]
    swing, support = uniform(cfg.swing_time), uniform(cfg.double_support)
    cycle = swing + support
    liftoff = cfg.initial_support + torch.cat((cycle[:, :1] * 0, cycle.cumsum(-1)[:, :-1]), 1)
    foot = ((torch.arange(M, device=initial_pos.device) + cfg.first_foot) % 2).expand(B, -1).clone()
    distance = uniform(cfg.step_length).cumsum(-1)
    forward = quat_to_mat(heading)[..., :, 0]
    pos = initial_pos.gather(1, foot[..., None].expand(-1, -1, 3)) + distance[..., None] * forward[:, None]
    yaw0 = torch.atan2(forward[:, 1], forward[:, 0])
    yaw = yaw0[:, None] + uniform(cfg.yaw_offset)
    zero = torch.zeros_like(yaw)
    quat = torch.stack((torch.cos(yaw / 2), zero, zero, torch.sin(yaw / 2)), -1)
    return FootstepCommand(initial_pos.clone(), initial_quat.clone(), foot, pos, quat, liftoff,
                           liftoff + swing, uniform(cfg.height), torch.full((B,), M, device=foot.device),
                           heading.clone()).validate()
