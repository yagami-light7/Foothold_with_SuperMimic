"""Fixed-size GPU replay buffer of simulator transitions, laid out as (time, env)."""

from __future__ import annotations

import torch
from collections import Counter

from .mdp import STATE_DIM, XI_DIM, NUM_J
from .foot_trajectory import FootstepCommand

FIELDS = {
    "s": (STATE_DIM, torch.float32),
    "s1": (STATE_DIM, torch.float32),
    "a": (NUM_J, torch.float32),
    "a_prev": (NUM_J, torch.float32),
    "xi": (XI_DIM, torch.float32),
    "r": ((), torch.float32),
    "f": ((), torch.long),
    "ep_step": ((), torch.int32),
    "done": ((), torch.bool),
    "term": ((), torch.bool),
    "timeout": ((), torch.bool),
}


class ReplayBuffer:
    def __init__(self, T: int, N: int, device: str, foot_commands: bool = False):
        self.T, self.N, self.device = T, N, device
        self.fields = dict(FIELDS)
        if foot_commands:
            self.fields['command_id'] = ((), torch.long)
        self.command_store = {}
        self.command_refs = Counter()
        self.data = {}
        for k, (shape, dt) in self.fields.items():
            shp = (T, N) + ((shape,) if isinstance(shape, int) else shape)
            self.data[k] = torch.zeros(shp, dtype=dt, device=device)
        self.ptr, self.size = 0, 0

    def add(self, tr: dict):
        if 'command_id' in self.fields:
            # The transition owns a snapshot taken BEFORE env.step auto-resets.
            # Store once per command, rather than copying a whole plan per row.
            if self.size == self.T:
                for cid, count in Counter(self.data['command_id'][self.ptr].cpu().tolist()).items():
                    self.command_refs[cid] -= count
                    if self.command_refs[cid] == 0:
                        del self.command_refs[cid], self.command_store[cid]
            ids = tr['command_id'].cpu().tolist()
            for i, cid in enumerate(ids):
                if cid not in self.command_store:
                    self.command_store[cid] = tr['command'].index(i).clone()
                self.command_refs[cid] += 1
        for k in self.fields:
            self.data[k][self.ptr] = tr[k]
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)

    def commands(self, ids: torch.Tensor) -> FootstepCommand:
        """Resolve immutable plans outside compiled imagination, in sample order."""
        if ids.ndim != 1:
            raise ValueError('Expected one command ID per imagination start')
        commands=[self.command_store[cid] for cid in ids.cpu().tolist()]
        if len({type(c) for c in commands}) != 1: raise ValueError('Cannot mix paired and quintic command schemas')
        return type(commands[0]).cat(commands)

    def _row(self, age: torch.Tensor) -> torch.Tensor:
        """Map age-ordered time index (0 = oldest) to the storage row."""
        return (self.ptr - self.size + age) % self.T

    def sample_windows(self, B: int, k: int):
        """k-step windows; returns dict of (B, k, ...) tensors and a validity mask (B, k)."""
        t0 = torch.randint(0, self.size - k + 1, (B,), device=self.device)
        env = torch.randint(0, self.N, (B,), device=self.device)
        ages = t0.unsqueeze(-1) + torch.arange(k, device=self.device)
        rows = self._row(ages)
        e = env.unsqueeze(-1).expand_as(rows)
        out = {key: self.data[key][rows, e] for key in ("s", "s1", "a", "xi", "done")}
        # Explicit evaluation/manual resets can occur without a terminal row in
        # replay. Integer episode ticks must also be consecutive; otherwise the
        # WM would learn an impossible teleport across the reset boundary.
        ticks = self.data['ep_step'][rows, e]
        continuous = (~out['done'][:, :-1]) & (ticks[:, 1:] == ticks[:, :-1] + 1)
        mask = torch.cumprod(torch.cat((torch.ones(B, 1, device=self.device), continuous.float()), -1), -1)
        out["mask"] = mask
        return out

    def sample_starts(self, B: int, hist: int):
        """Start points for imagination plus the indices of their observation history (oldest first)."""
        age = torch.randint(0, self.size, (B,), device=self.device)
        env = torch.randint(0, self.N, (B,), device=self.device)
        row = self._row(age)
        ep_step = self.data["ep_step"][row, env].long()
        back = torch.arange(hist - 1, -1, -1, device=self.device)  # hist-1 ... 0
        back = torch.minimum(back.unsqueeze(0), ep_step.unsqueeze(-1))  # stay inside the episode
        back = torch.minimum(back, age.unsqueeze(-1))  # stay inside the buffer
        hrows = self._row(age.unsqueeze(-1) - back)
        henv = env.unsqueeze(-1).expand_as(hrows)
        keys = ('s', 'a_prev', 'xi', 'f', 'ep_step')
        if 'command_id' in self.fields:
            keys += ('command_id',)
        start = {k: self.data[k][row, env] for k in keys}
        hist_d = {k: self.data[k][hrows, henv] for k in keys}
        return start, hist_d

    def sample_mc_returns(self, B: int, window: int, gamma: float, recent: int):
        """Observed discounted returns ending at a physical termination only.

        Timeouts, replay edges and explicit reset gaps are censored, not assigned
        a zero terminal value. Targets are recent behavior-policy returns; this
        is an auxiliary value calibration source, not exact on-policy targets.
        """
        low = max(0, self.size - recent) if recent > 0 else 0
        age = torch.randint(low, self.size, (B,), device=self.device)
        env = torch.randint(0, self.N, (B,), device=self.device)
        ages = age[:, None] + torch.arange(window, device=self.device)[None]
        valid = ages < self.size
        rows = self._row(ages.clamp_max(self.size - 1))
        ee = env[:, None].expand_as(rows)
        ticks = self.data['ep_step'][rows, ee]
        done = self.data['done'][rows, ee]
        continuous = valid[:, 1:] & (~done[:, :-1]) & (ticks[:, 1:] == ticks[:, :-1] + 1)
        alive = torch.cumprod(torch.cat((torch.ones(B, 1, device=self.device), continuous.float()), 1), 1) * valid
        physical = self.data['term'][rows, ee] & alive.bool()
        found = physical.any(1)
        discounts = gamma ** torch.arange(window, device=self.device, dtype=torch.float32)
        returns = (self.data['r'][rows, ee] * alive * discounts[None]).sum(1)
        first = self._row(age)
        keys = ('s', 'a_prev', 'xi', 'f', 'ep_step')
        if 'command_id' in self.fields: keys += ('command_id',)
        out = {k: self.data[k][first[found], env[found]] for k in keys}
        out['mc_return'] = returns[found].detach()
        return out
