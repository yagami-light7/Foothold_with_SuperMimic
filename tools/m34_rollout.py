"""Independent Newton FK gate, PD baseline, and frozen M3.4 policy evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import add_launcher_args, launch_simulation

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--dataset', default='data/s54_flat_dataset/dataset_export')
p.add_argument('--split', choices=['train', 'test'], default='train')
p.add_argument('--checkpoint')
p.add_argument('--out', required=True)
p.add_argument('--ids', nargs='+')
p.add_argument('--seeds', nargs='+', type=int, default=[0])
p.add_argument('--reset_noise', action='store_true', help='Training reset perturbations; report separately from nominal')
p.add_argument('--fk_only', action='store_true')
p.add_argument('--raw_comparison', help='One train clip ID: explicit GMR-Raw PD comparison only')
p.add_argument('--max_steps', type=int, help='Diagnostic truncation only; never counts as survival success')
add_launcher_args(p)
args = p.parse_args()
if args.split == 'test' and not args.checkpoint:
    p.error('Test references are for frozen policy evaluation only')
if args.raw_comparison and (args.checkpoint or args.split != 'train' or args.ids):
    p.error('Raw comparison is a standalone single-source PD diagnostic, never a policy input')
if args.max_steps is not None and args.max_steps < 1:
    p.error('max_steps must be positive')

import numpy as np
import torch
from supermimic.m34 import motion_library, dataset_contract, sha256
from supermimic.kinematics import S54Kinematics
from supermimic.env import TrackingSim, make_sim
from supermimic.goals import GoalSource, GoalSpec
from supermimic.mdp import MDP, OBS_DIM, RewardCfg
from supermimic.networks import Policy
from supermimic.control import bound_action
from supermimic.rotations import quat_to_mat, mat_to_quat, so3_log, so3_exp
from supermimic.s54 import (default_joint_pos_list, JOINT_NAMES, NUM_J,
                            FOOT_BODY_NAMES, assert_checkpoint)
from supermimic.standing_diagnostics import NewtonGroundContacts
from supermimic.transfer import physical_contract


@torch.no_grad()
def fk_gate(env, kin, motion):
    """Compare all 30 link poses and world-origin velocities against Newton.

    State writes occur only in this reset/FK gate, never in a rollout step.
    """
    torch.manual_seed(54)
    errors = torch.zeros(4, device=env.device)
    ids = torch.tensor([0], device=env.device)
    frames = [int(motion.clip_start[c]+motion.clip_len[c]//2) for c in range(len(motion.names))]
    for t in range(len(frames)+16):
        env.reset(ids, torch.tensor([frames[t % len(frames)]], device=env.device))
        if t >= len(frames):
            S = list(env.state())
            S[0] = torch.randn_like(S[0])*.2 + torch.tensor([0, 0, 1.2], device=env.device)
            S[1] = so3_exp(torch.randn(1, 3, device=env.device))
            S[2], S[3] = torch.randn_like(S[2]), torch.randn_like(S[3])
            S[4] = kin.q_lo+(kin.q_hi-kin.q_lo)*torch.rand_like(S[4])
            S[5] = torch.randn_like(S[5])
            quat = mat_to_quat(S[1])
            env.robot.write_root_link_pose_to_sim_index(root_pose=torch.cat((S[0], quat[:, 1:], quat[:, :1]), -1), env_ids=ids)
            env.robot.write_root_link_velocity_to_sim_index(root_velocity=torch.cat((S[2], S[3]), -1), env_ids=ids)
            env.robot.write_joint_position_to_sim_index(position=S[4][:, env.sim_to_canonical], env_ids=ids)
            env.robot.write_joint_velocity_to_sim_index(velocity=S[5][:, env.sim_to_canonical], env_ids=ids)
            env.refresh()
        S = env.state()
        actual = kin.forward(*S, all_bodies=True)
        body_ids = [env.robot.body_names.index(n) for n in kin.body_names]
        poses = env.robot.data.body_link_pose_w.torch[:, body_ids]
        vels = env.robot.data.body_link_vel_w.torch[:, body_ids]
        expected = (poses[..., :3]-env.scene.env_origins[:, None],
                    quat_to_mat(torch.cat((poses[..., 6:7], poses[..., 3:6]), -1)),
                    vels[..., :3], vels[..., 3:])
        errors = torch.maximum(errors, torch.stack([(a-b).abs().max() for a, b in zip(actual, expected)]))
    limits = torch.tensor([1e-4, 1e-3, 1e-3, 1e-3], device=env.device)
    if not (errors < limits).all():
        raise RuntimeError(f'Newton FK/velocity gate failed: {errors.tolist()}')
    return dict(status='PASS', reference_poses=len(frames), random_poses=16,
                all_bodies=30, max_errors=errors.tolist(), physical_contract=physical_contract())


@torch.no_grad()
def rollout(env, kin, m, mdp, goals, pi, hist_len, clip, seed, out_dir):
    torch.manual_seed(seed)
    env.done_eval.zero_(); env.failed_eval.zero_()
    env.reset_noise = args.reset_noise
    env.reset(torch.tensor([0], device=env.device), m.clip_start[clip:clip+1])
    contacts = NewtonGroundContacts(env)
    feet = [kin.tracked_names.index(n) for n in FOOT_BODY_NAMES]
    history = None
    trace = {k: [] for k in ('root_p', 'root_quat', 'q', 'f', 'alive')}
    metrics = []
    slip_sum = 0.; stance_count = 0; any_contacts = 0; other_contacts = 0
    torque_peak = None; saturation_sum = 0.; torque_count = 0; torque_source = None
    previous_action = torch.zeros(1, NUM_J, device=env.device)
    requested_steps = int(m.clip_len[clip])-1
    limit = min(requested_steps, args.max_steps or requested_steps)
    status = 'diagnostic_truncation'
    for tick in range(limit):
        S = env.state()
        for key, value in zip(('root_p', 'root_quat', 'q', 'f', 'alive'),
                              (S[0], mat_to_quat(S[1]), S[4], env.f, ~env.done_eval)):
            trace[key].append(value.cpu().clone())
        if pi is None:
            # Targets at t+dt, identical to the physics step's reference clock.
            action = (m.q[env.f+1]-env.default_q)/env.alpha
        else:
            obs = goals.obs(mdp, S, env.a_prev, env.xi, env.f, env.command)
            history = obs[:, None].repeat(1, hist_len, 1) if history is None else torch.cat((history[:, 1:], obs[:, None]), 1)
            action = bound_action(pi(history, goals.goal(S, env.f, env.ep_step, env.command)))
        transition = env.step(action)
        S1 = env.state(); f = env.f
        if not all(torch.isfinite(x).all() for x in S1):
            status = 'newton_nonfinite'; break
        bp, _, bv, _ = kin.forward(*S1)
        counts, forces, other = contacts.read()
        contact = torch.as_tensor(counts[0] > 0, device=env.device)
        ref_stance = m.body_v[f][:, feet, :2].norm(dim=-1)[0] < .05
        stance = contact & ref_stance
        slip_sum += float((bv[:, feet, :2].norm(dim=-1)[0]*stance).sum())
        stance_count += int(stance.sum()); any_contacts += int((counts > 0).sum()); other_contacts += int(other.sum())
        orientation = so3_log(m.root_R[f].transpose(-1, -2) @ S1[1]).norm(dim=-1)
        metrics.append([float((S1[0]-m.root_p[f]).square().sum()), float(orientation.square().mean()),
                        float((S1[4]-m.q[f]).square().mean()),
                        float((bp[:, feet]-m.body_p[f][:, feet]).square().sum(-1).mean()),
                        float(transition['r'].mean()), float((action-previous_action).square().mean())])
        previous_action = action
        # These are actuator-reported efforts. Implicit-actuator estimates are
        # not claimed as solver-measured torques; missing data stays null.
        data = env.robot.data
        for attr in ('joint_applied_torque', 'applied_torque', 'joint_effort'):
            if hasattr(data, attr):
                raw = getattr(data, attr)
                raw = raw.torch if hasattr(raw, 'torch') else raw
                torque = raw[:, env.canonical_to_sim]
                torque_source = 'actuator_reported:'+attr+'; implicit actuator may estimate solver effort'
                from supermimic.robot import PARAMETERS
                effort = torch.tensor([PARAMETERS[n]['effort'] for n in JOINT_NAMES], device=env.device)
                peak = float(torque.abs().max())
                torque_peak = max(torque_peak or 0., peak)
                saturation_sum += float((torque.abs() >= effort*.99).sum()); torque_count += torque.numel()
                break
        if bool(transition['term'][0]):
            dz = float((S1[0][:, 2]-m.root_p[f, 2]).abs())
            tilt = float((S1[1][:, 2, 2]-m.root_R[f, 2, 2]).abs())
            status = 'tracking_height' if dz > mdp.cfg.term_height else 'tracking_tilt' if tilt > mdp.cfg.term_tilt else 'tracking_other'
            break
        if bool(transition['timeout'][0]):
            status = 'reference_complete'; break
    S = env.state()
    for key, value in zip(('root_p', 'root_quat', 'q', 'f', 'alive'),
                          (S[0], mat_to_quat(S[1]), S[4], env.f, ~env.done_eval)):
        trace[key].append(value.cpu().clone())
    target = out_dir/f'{m.names[clip]}_seed{seed}.npz'
    np.savez_compressed(target, joint_names=np.array(JOINT_NAMES), control_dt=.02,
                        provenance='Newton MJWarp physics rollout; reset writes only at episode start',
                        **{k: torch.stack(v).numpy() for k, v in trace.items()})
    x = np.asarray(metrics)
    row = dict(id=m.names[clip], mode=m.modes[clip], split=m.split, seed=seed,
               extrapolation=m.modes[clip] == 'backward', reset_noise=args.reset_noise,
               controller='policy' if pi else 'reference_pd', status=status,
               episode_steps=len(x), episode_s=len(x)*.02, reference_steps=requested_steps,
               survival_fraction=len(x)/requested_steps, survived=status == 'reference_complete',
               failure_frame=int(env.f[0]) if status.startswith('tracking_') else None,
               ground_foot_contact_samples=any_contacts, nonfoot_ground_contacts=other_contacts,
               stance_slip_m_s=slip_sum/stance_count if stance_count else None,
               stance_slip_definition='horizontal foot link-origin speed, actual ground contact AND reference link speed <0.05 m/s; proxy, not contact-patch slip',
               stance_slip_samples=stance_count, actuator_reported_torque_peak_Nm=torque_peak,
               actuator_reported_torque_saturation_rate=saturation_sum/torque_count if torque_count else None,
               torque_source=torque_source,
               record=str(target))
    if len(x):
        row.update(root_position_rmse_m=float(np.sqrt(x[:, 0].mean())),
                   root_orientation_rmse_rad=float(np.sqrt(x[:, 1].mean())),
                   joint_rmse_rad=float(np.sqrt(x[:, 2].mean())), foot_position_rmse_m=float(np.sqrt(x[:, 3].mean())),
                   reward_sum=float(x[:, 4].sum()), normalized_return=float(x[:, 4].sum()/(requested_steps*mdp.max_step_reward)),
                   action_delta_rms=float(np.sqrt(x[:, 5].mean())))
    print(json.dumps(row), flush=True)
    return row


def main():
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f'Preserve existing diagnostics: {out}')
    out.mkdir(parents=True)
    report = dict(status='initializing', split=args.split, rows=[])
    try:
        device = 'cuda:0'
        kin = S54Kinematics(device)
        if args.raw_comparison:
            from supermimic.m34_comparison import raw_library
            motion = raw_library(args.dataset, kin, device, args.raw_comparison)
        else:
            motion = motion_library(args.dataset, kin, device, args.split, args.ids)
        report['split'] = motion.split
        ck = torch.load(args.checkpoint, map_location=device, weights_only=False) if args.checkpoint else None
        if ck:
            assert_checkpoint(ck)
            if ck['data_contract'] != dataset_contract(args.dataset):
                raise ValueError('Checkpoint training dataset contract mismatch')
            if ck['physical_contract'] != physical_contract():
                raise ValueError('Checkpoint physics contract mismatch')
        config = ck['args'] if ck else {}
        spec = GoalSpec('motion', config.get('goal_frames', 10), config.get('goal_dt', .02))
        if ck: spec.assert_checkpoint(ck)
        goals = GoalSource(motion, kin, spec)
        mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=device), RewardCfg(**config.get('reward_config', {})))
        pi = None
        history = config.get('hist', 10)
        if ck:
            import supermimic.networks as networks
            networks.AMP_DTYPE = torch.bfloat16 if config.get('amp') == 'bf16' else None
            pi = Policy(OBS_DIM, spec.dim, hist=history, obs_adapter=config.get('obs_adapter', 'identity')).to(device)
            pi.load_state_dict(ck['pi']); pi.eval(); pi.requires_grad_(False)
            report.update(checkpoint=args.checkpoint, checkpoint_sha256=sha256(args.checkpoint), iteration=ck['iteration'])
        sim_cfg = make_sim(device, args)
        with launch_simulation(sim_cfg, args):
            env = TrackingSim(sim_cfg, 1, motion, mdp, eval_mode=True, goals=goals)
            report['fk_gate'] = fk_gate(env, kin, motion)
            report['data_validation'] = motion.validation
            if not args.fk_only:
                for c in range(len(motion.names)):
                    for seed in args.seeds:
                        row = rollout(env, kin, motion, mdp, goals, pi, history, c, seed, out)
                        report['rows'].append(row)
                        (out/'results.json').write_text(json.dumps(report, indent=2))
            report['status'] = 'completed'
    except BaseException as exc:
        report.update(status='environment_or_contract_error', error=repr(exc))
        raise
    finally:
        (out/'results.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
