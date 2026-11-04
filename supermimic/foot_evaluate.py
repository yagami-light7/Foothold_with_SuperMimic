"""Bounded independent foot evaluation, optionally with complete control/contact traces."""
import json
from pathlib import Path

import numpy as np
import torch

from .foot_mdp import assert_task_checkpoint
from .mdp import OBS_DIM, flatten_state
from .networks import Policy
from .rotations import mat_to_quat
from .s54 import JOINT_NAMES, ROBOT_ID, assert_checkpoint
from .control import bound_action
from .motion import CONTROL_DT
from .standing_diagnostics import evaluation_steps, stop_reasons, NewtonGroundContacts
from . import networks


@torch.no_grad()
def evaluate_foot(env, checkpoint, goals, mdp, record=None, device='cuda:0', *,
                  duration_s=None, seed=None, reset_noise=False, controller='policy', diagnostics=False,
                  ankle_offset_rad=0.):
    assert_checkpoint(checkpoint)
    goals.spec.assert_checkpoint(checkpoint, goals.foot_config)
    assert_task_checkpoint(checkpoint, mdp)
    if controller not in ('policy', 'default_pd', 'ankle_pd'):
        raise ValueError('Unknown controller')
    if not np.isfinite(ankle_offset_rad) or abs(ankle_offset_rad)>.08:
        raise ValueError('Diagnostic ankle offset must be finite and <=0.08rad')
    cfg = checkpoint['args']; hist_len = cfg['hist']
    networks.AMP_DTYPE = torch.bfloat16 if cfg.get('amp') == 'bf16' else None
    pi = Policy(OBS_DIM, goals.spec.dim, hist=hist_len,obs_adapter=cfg.get('obs_adapter','identity')).to(device)
    pi.load_state_dict(checkpoint['pi']); pi.eval()
    seed = cfg.get('seed', 0) if seed is None else seed
    # Evaluation horizon is a protocol override, never a change to the saved MDP contract.
    horizon = evaluation_steps(duration_s, mdp.cfg.episode_steps, CONTROL_DT)
    old_horizon, old_noise = env.max_episode_steps, env.reset_noise
    env.max_episode_steps, env.reset_noise = horizon, reset_noise
    try:
        torch.manual_seed(seed)
        ids = torch.arange(env.N, device=device)
        env.done_eval[:] = False; env.failed_eval[:] = False; env.reset(ids)
        commands = {f'command_{k}': v.cpu().numpy() for k, v in env.command.tensors().items()} if record else {}
        returns = torch.zeros(env.N, device=device); steps = torch.zeros_like(returns); error = torch.zeros_like(returns)
        history = None
        trace = {k: [] for k in ('state', 'root_p', 'root_quat', 'root_v', 'root_w', 'q', 'qd',
                                 'command_step', 'alive', 'reference_foot_p', 'actual_foot_p', 'actual_foot_R',
                                 'contact_count', 'contact_force_w', 'other_ground_contacts')}
        transitions = {k: [] for k in ('action', 'a_prev', 'pd_target', 'reward', 'term', 'timeout')}
        reward_terms = {}
        contacts = None
        contact_status = 'NOT_REQUESTED'
        if diagnostics:
            try:
                contacts = NewtonGroundContacts(env)
                contact_status = 'last_substep_solver_contacts_and_world_forces'
            except (AttributeError, RuntimeError, NotImplementedError) as exc:
                contact_status = f'UNAVAILABLE: {exc}'
        reasons = ['running'] * env.N

        def capture(S, alive, initial=False):
            ref = mdp._reference(env.command, env.ep_step)
            bp, bR, _, _ = mdp.kin.forward(*S)
            counts = np.full((env.N, 2), -1, dtype=np.int64)
            forces = np.full((env.N, 2, 3), np.nan); other = np.full(env.N, -1, dtype=np.int64)
            if contacts is not None and not initial:
                counts, forces, other = contacts.read()
            values = dict(state=flatten_state(S), root_p=S[0], root_quat=mat_to_quat(S[1]),
                          root_v=S[2], root_w=S[3], q=S[4], qd=S[5], command_step=env.ep_step,
                          alive=alive, reference_foot_p=ref.position[:, 0],
                          actual_foot_p=bp[:, mdp.foot_indices], actual_foot_R=bR[:, mdp.foot_indices],
                          contact_count=counts, contact_force_w=forces, other_ground_contacts=other)
            for k, v in values.items():
                trace[k].append(v.detach().cpu().clone().numpy() if isinstance(v, torch.Tensor) else v.copy())

        if record:
            capture(env.state(), ~env.done_eval, initial=True)
        while not bool(env.done_eval.all()):
            S = env.state(); alive = ~env.done_eval.clone()
            obs = goals.obs(mdp, S, env.a_prev, env.xi, env.f, env.command)
            history = (obs[:, None].repeat(1, hist_len, 1) if history is None
                       else torch.cat((history[:, 1:], obs[:, None]), 1))
            action = (pi(history, goals.goal(S, env.f, env.ep_step, env.command)) if controller == 'policy'
                      else torch.zeros_like(env.a_prev))
            if controller=='ankle_pd':
                for name in ('leg_l5_joint','leg_r5_joint'):
                    j=JOINT_NAMES.index(name);action[:,j]=ankle_offset_rad/env.alpha[j]
            action = bound_action(action)
            pd_target = env.pd_target(action)
            out = env.step(action)
            returns += torch.where(alive, out['r'], 0.)
            error += torch.where(alive, out['terms']['foot_position_error'], 0.)
            steps += alive.float()
            stopped = stop_reasons(env.state(), mdp, out['term'], out['timeout'])
            for i in (alive & out['done']).nonzero().flatten().tolist():
                reasons[i] = stopped[i]
            if record:
                for k, value in dict(action=action, a_prev=out['a_prev'], pd_target=pd_target,
                                     reward=out['r'], term=out['term'], timeout=out['timeout']).items():
                    transitions[k].append(value.detach().cpu().numpy().copy())
                for k, value in out['terms'].items():
                    reward_terms.setdefault('reward_' + k, []).append(value.detach().cpu().numpy().copy())
                capture(env.state(), ~env.done_eval)
        normal = returns / (horizon * mdp.max_step_reward)
        names = [f'{mdp.cfg.command_mode}_{i}' for i in range(env.N)]
        result = dict(task=mdp.cfg.command_mode, controller=controller, seed=seed, reset_noise=reset_noise,
                      diagnostic_ankle_offset_rad=ankle_offset_rad if controller=='ankle_pd' else None,
                      duration_s=horizon*CONTROL_DT, evaluation_steps=horizon,
                      norm_return=float(normal.mean()), norm_return_framew=float(normal.mean()),
                      n_failed=int(env.failed_eval.sum()), n_timeout=reasons.count('timeout'),
                      stop_reasons=reasons, episode_lengths=steps.int().cpu().tolist(),
                      failed=[names[i] for i in env.failed_eval.nonzero().flatten().tolist()],
                      per_episode=dict(zip(names, normal.cpu().tolist())), mean_episode_length=float(steps.mean()),
                      normalized_sim_reward=float((returns/steps.clamp_min(1)/mdp.max_step_reward).mean()),
                      foot_position_rmse_m=float((error/steps.clamp_min(1)).mean().sqrt()),
                      actual_contact_status=contact_status, motion_reference_required=False)
        if record:
            target = Path(record); target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, robot=ROBOT_ID, joint_names=np.array(JOINT_NAMES), quat='wxyz',
                                control_dt=CONTROL_DT, names=np.array(names), xi=env.xi.cpu().numpy(),
                                episode_lengths=steps.int().cpu().numpy(), evaluation_protocol=json.dumps(result),
                                goal_contract=json.dumps(goals.spec.contract()), task_contract=json.dumps(mdp.contract()),
                                **commands, **{k: np.stack(v) for k, v in {**trace, **transitions, **reward_terms}.items()})
        return result
    finally:
        env.max_episode_steps, env.reset_noise = old_horizon, old_noise
