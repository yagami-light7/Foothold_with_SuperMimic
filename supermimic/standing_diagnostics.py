"""Paired, open-loop Newton/WM diagnostics; no optimizers or training changes."""
import json
import re

import numpy as np
import torch

from .mdp import normalize_xi, unflatten_state
from .motion import CONTROL_DT
from .rotations import so3_log
from .s54 import FOOT_BODY_NAMES


def evaluation_steps(duration_s, default_steps, dt):
    if duration_s is None:
        return default_steps
    if not np.isfinite(duration_s):
        raise ValueError('Evaluation duration must be finite')
    steps = round(duration_s / dt)
    if steps < 1 or not np.isclose(steps * dt, duration_s):
        raise ValueError('Evaluation duration must be a positive multiple of CONTROL_DT')
    return steps


def stop_reasons(S, mdp, terminated, timeout):
    """Physical termination takes precedence when it coincides with the time limit."""
    finite = torch.stack([x.flatten(1).isfinite().all(-1) for x in S]).all(0)
    height = (S[0][:, 2] - mdp.nominal_height).abs() > mdp.cfg.term_height
    tilt = 1 - S[1][:, 2, 2] > mdp.cfg.term_tilt
    reasons = []
    for i in range(len(terminated)):
        if bool(terminated[i]):
            reasons.append('nonfinite' if not finite[i] else
                           '+'.join(name for name, flag in [('height', height[i]), ('tilt', tilt[i])] if flag)
                           or 'physical_other')
        else:
            reasons.append('timeout' if timeout[i] else 'running')
    return reasons


def aggregate_ground_contacts(shape_body, body_labels, shape0, shape1, forces, num_envs):
    """Match Newton SensorContact's force-on-shape0 sign; ignore non-ground pairs."""
    counts = np.zeros((num_envs, 2), dtype=np.int64)
    force = np.zeros((num_envs, 2, 3))
    other = np.zeros(num_envs, dtype=np.int64)
    for j, (s0, s1) in enumerate(zip(shape0, shape1)):
        if s0 < 0 or s1 < 0:
            continue
        b0, b1 = int(shape_body[s0]), int(shape_body[s1])
        if (b0 < 0) == (b1 < 0):
            continue
        body = max(b0, b1)
        label = body_labels[body]
        match = re.search(r'/env_(\d+)/', label)
        if match is None:
            raise ValueError(f'Cannot identify environment of contact body {label}')
        env = int(match.group(1))
        name = label.split('/')[-1]
        if name in FOOT_BODY_NAMES:
            side = FOOT_BODY_NAMES.index(name)
            counts[env, side] += 1
            force[env, side] += forces[j, :3] * (1 if body == b0 else -1)
        else:
            other[env] += 1
    return counts, force, other


def ground_contact_subset(shape_body, body_env, shape0, shape1, env_ids):
    """Select ground contacts on-device before copying arrays to CPU."""
    valid = (shape0 >= 0) & (shape1 >= 0)
    b0, b1 = shape_body[shape0.clamp_min(0)], shape_body[shape1.clamp_min(0)]
    body = torch.maximum(b0, b1)
    return valid & ((b0 < 0) != (b1 < 0)) & torch.isin(body_env[body.clamp_min(0)], env_ids)


class NewtonGroundContacts:
    """Last solver-substep contacts, not commanded contact masks or frame averages.

    Pinned SolverMuJoCo.update_contacts exports efc.force into the extended
    Contacts.force spatial array. rigid_contact_force is unused in this version.
    """
    def __init__(self, env):
        import warp as wp
        self.wp, self.env, self.backend = wp, env, env.physics
        self.contacts = self.backend.get_contacts()
        if self.contacts is None:
            raise RuntimeError('Newton did not expose a contact buffer')
        if self.contacts.force is None:
            self.contacts.force = wp.zeros(self.contacts.rigid_contact_max + self.contacts.soft_contact_max,
                                           dtype=wp.spatial_vector, device=env.device)
        model = self.backend.get_model()
        self.shape_body = wp.to_torch(model.shape_body).cpu().numpy()
        self.labels = model.body_label
        self._shape_body = wp.to_torch(model.shape_body).long()
        body_env = []
        for label in self.labels:
            match = re.search(r'/env_(\d+)/', label)
            body_env.append(int(match.group(1)) if match else -1)
        self._body_env = torch.tensor(body_env, device=env.device, dtype=torch.long)

    def read(self, env_ids=None):
        wp, c = self.wp, self.contacts
        self.backend._solver.update_contacts(c, self.backend.get_state_0())
        count = int(wp.to_torch(c.rigid_contact_count).sum())
        s0 = wp.to_torch(c.rigid_contact_shape0)[:count].long()
        s1 = wp.to_torch(c.rigid_contact_shape1)[:count].long()
        force = wp.to_torch(c.force)[:count]
        if env_ids is not None:
            ids = torch.as_tensor(env_ids, device=s0.device, dtype=torch.long)
            mask = ground_contact_subset(self._shape_body, self._body_env, s0, s1, ids)
            s0, s1, force = s0[mask], s1[mask], force[mask]
        return aggregate_ground_contacts(self.shape_body, self.labels,
                s0.cpu().numpy(), s1.cpu().numpy(), force.cpu().numpy(), self.env.N)


def valid_windows(lengths, horizon, stride=1):
    """Indices include a terminal endpoint, never cross an episode boundary."""
    return [(t, e) for e, length in enumerate(lengths)
            for t in range(0, int(length) - horizon + 1, stride)]


@torch.no_grad()
def paired_predictions(wm, mdp, command, trace, device, horizons=(1, 4, 8), stride=1):
    """Every WM rollout starts at an observed Newton state and reuses its actions.

    Use the SAME windows at every horizon. Reward is the raw finite-horizon sum,
    without Critic, bootstrap, soft survival, or discount on either side.
    """
    if 'task_contract' in trace and json.loads(str(trace['task_contract'])) != mdp.contract():
        raise ValueError('Recorded task contract differs from the diagnostic MDP')
    windows = valid_windows(trace['episode_lengths'], max(horizons), stride)
    if not windows:
        raise ValueError('No complete paired windows in this recording')
    starts = torch.tensor([t for t, e in windows], device=device)
    envs = torch.tensor([e for t, e in windows], device=device)
    states = torch.as_tensor(trace['state'], device=device)
    actions = torch.as_tensor(trace['action'], device=device)
    previous = torch.as_tensor(trace['a_prev'], device=device)
    steps = torch.as_tensor(trace['command_step'], device=device)
    xi = torch.as_tensor(trace['xi'], device=device)[envs]
    cmd = command.index(envs) if command is not None else None
    S = unflatten_state(states[starts, envs])
    previous_action = previous[starts, envs]
    rsum = torch.zeros(len(windows), device=device); psum = torch.zeros_like(rsum)
    absolute_reward_error = torch.zeros_like(rsum)
    metrics = {}; rows = []; sync = 0.
    for j in range(1, max(horizons) + 1):
        action = actions[starts+j-1, envs]
        S = wm.step(S, action, normalize_xi(xi))
        true = unflatten_state(states[starts+j, envs])
        tick = steps[starts+j, envs]
        frame=None
        if not mdp.is_foot:
            if cmd is not None and hasattr(cmd,'start_frame'): frame=cmd.start_frame+tick
            elif 'reference_frame' in trace: frame=torch.as_tensor(trace['reference_frame'],device=device)[starts+j,envs]
            else: raise ValueError('Motion prediction check requires saved reference frame indices')
        actual_r = mdp.reward(true, frame, action, previous_action, command=cmd, ep_step=tick)
        pred_r = mdp.reward(S, frame, action, previous_action, command=cmd, ep_step=tick)
        saved_r = torch.as_tensor(trace['reward'], device=device)[starts+j-1, envs]
        sync = max(sync, float((saved_r-actual_r).abs().max()))
        rsum += actual_r; psum += pred_r
        absolute_reward_error += (pred_r-actual_r).abs()
        previous_action = action
        if j not in horizons:
            continue
        delta = [S[k]-true[k] for k in (0, 2, 3, 4, 5)]
        angle = so3_log(true[1].transpose(-1, -2)@S[1])
        height = delta[0][:, 2]
        pitch_error=(torch.asin((-S[1][:,2,0]).clamp(-1,1))-torch.asin((-true[1][:,2,0]).clamp(-1,1)))*180/np.pi
        if hasattr(mdp,'foot_indices'): foot_indices=mdp.foot_indices
        else:
            from .s54 import FOOT_BODY_NAMES
            foot_indices=[mdp.kin.tracked_names.index(n) for n in FOOT_BODY_NAMES]
        feet_error=mdp.kin.forward(*S)[0][:,foot_indices]-mdp.kin.forward(*true)[0][:,foot_indices]
        reward_bias = (psum-rsum)/(j*mdp.max_step_reward)
        def rmse(x): return float(x.square().mean().sqrt())
        result = dict(windows=len(windows), base_height_rmse_m=rmse(height),
                      root_pitch_rmse_deg=rmse(pitch_error),feet_position_rmse_m=rmse(feet_error.norm(dim=-1)),
                      base_height_bias_m=float(height.mean()), orientation_rmse_deg=rmse(angle.norm(dim=-1))*180/np.pi,
                      base_height_abs_error_p95_m=float(height.abs().quantile(.95)),
                      base_height_positive_bias_fraction=float((height>0).float().mean()),
                      rotation_vector_bias_deg=(angle.mean(0)*180/np.pi).cpu().tolist(),
                      linear_velocity_rmse_m_s=rmse(delta[1]), linear_velocity_bias_m_s=delta[1].mean(0).cpu().tolist(),
                      angular_velocity_rmse_rad_s=rmse(delta[2]), angular_velocity_bias_rad_s=delta[2].mean(0).cpu().tolist(),
                      joint_position_rmse_rad=rmse(delta[3]), joint_velocity_rmse_rad_s=rmse(delta[4]),
                      per_joint_position_rmse_rad=delta[3].square().mean(0).sqrt().cpu().tolist(),
                      per_joint_velocity_rmse_rad_s=delta[4].square().mean(0).sqrt().cpu().tolist(),
                      paired_sim_reward=float(rsum.mean()/(j*mdp.max_step_reward)),
                      paired_imagined_reward=float(psum.mean()/(j*mdp.max_step_reward)),
                      paired_reward_bias=float(reward_bias.mean()), paired_reward_mae=float(reward_bias.abs().mean()),
                      per_step_reward_mae=float(absolute_reward_error.mean()/(j*mdp.max_step_reward)),
                      endpoint_reward_bias=float((pred_r-actual_r).mean()/mdp.max_step_reward))
        metrics[str(j)] = result
        for i, (t, e) in enumerate(windows):
            rows.append(dict(start=t, command_start_step=int(steps[t,e]), env=e, horizon=j, true_height=float(true[0][i, 2]),
                             predicted_height=float(S[0][i, 2]), height_bias_m=float(height[i]),
                             orientation_error_deg=float(angle[i].norm()*180/np.pi),
                             simulation_reward=float(rsum[i]/(j*mdp.max_step_reward)),
                             imagined_reward=float(psum[i]/(j*mdp.max_step_reward))))
    if sync > 1e-5:
        raise ValueError(f'Recorded reward does not match restored command/time/MDP: max error {sync}')
    phases = {}
    for name, lower, upper in [('0_to_0.5s',0.,.5),('0.5_to_1.2s',.5,1.2),('after_1.2s',1.2,float('inf'))]:
        phases[name] = {}
        for h in horizons:
            part = [r for r in rows if r['horizon']==h and lower <= r['command_start_step']*CONTROL_DT < upper]
            if part:
                phases[name][str(h)] = dict(windows=len(part),
                    height_bias_m=float(np.mean([r['height_bias_m'] for r in part])),
                    height_rmse_m=float(np.sqrt(np.mean([r['height_bias_m']**2 for r in part]))),
                    orientation_rmse_deg=float(np.sqrt(np.mean([r['orientation_error_deg']**2 for r in part]))),
                    paired_reward_bias=float(np.mean([r['imagined_reward']-r['simulation_reward'] for r in part])))
    return dict(protocol='observed Newton starts, recorded open-loop actions, matched command/ticks; no optimizer',
                horizons=metrics, phases=phases, replay_reward_sync_max_error=sync), rows
