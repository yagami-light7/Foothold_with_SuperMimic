"""Bounded Standing diagnosis: paired controllers, true contacts, open-loop WM audit.

No training/IK/optimizer updates. Reuses the checkpoint's exact MDP and evaluator.
"""

# Support both direct execution and python -m from the repository layout.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_root = str(_ToolPath(__file__).resolve().parents[2])
if _tool_root not in _tool_sys.path:
    _tool_sys.path.insert(0, _tool_root)

import argparse
import csv
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from isaaclab.app import add_launcher_args, launch_simulation
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', default='results/s54/standing_diagnosis')
    p.add_argument('--durations', nargs='+', type=float, default=[2, 5, 10])
    p.add_argument('--seeds', nargs='+', type=int, default=[54, 55, 56])
    p.add_argument('--num_envs', type=int, default=1)
    p.add_argument('--analyze_only', action='store_true', help='Recompute plots and WM metrics from saved nominal NPZs; no simulation')
    add_launcher_args(p)
    args = p.parse_args()
    if not 1 <= args.num_envs <= 4 or not all(0 < d <= 10 for d in args.durations) or len(args.seeds) > 10:
        p.error('Diagnosis is bounded to 1–4 envs, <=10s, <=10 seeds')
    return args

import numpy as np
import torch
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from supermimic import networks
from supermimic.env import TrackingSim, make_sim
from supermimic.foot_evaluate import evaluate_foot
from supermimic.foot_mdp import FootTrackingMDP, resolve_foot_task, assert_task_checkpoint
from supermimic.foot_trajectory import AlternatingConfig, FootstepCommand
from supermimic.goals import GoalSource, GoalSpec
from supermimic.kinematics import S54Kinematics
from supermimic.networks import WorldModel
from supermimic.s54 import default_joint_pos_list, JOINT_NAMES, assert_checkpoint
from supermimic.standing_diagnostics import paired_predictions


def write_csv(path, rows):
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def audit_trace(path, mdp):
    with np.load(path, allow_pickle=False) as z:
        data = dict(z)
    length = int(data['episode_lengths'][0]); n = length + 1
    time = np.arange(n) * .02
    q, qd = data['q'][:n, 0], data['qd'][:n, 0]
    action = data['action'][:length, 0]; target = data['pd_target'][:length, 0]
    euler = Rotation.from_quat(data['root_quat'][:n, 0][:, [1, 2, 3, 0]]).as_euler('xyz', degrees=True)
    foot_error = np.linalg.norm(data['actual_foot_p'][:n, 0] - data['reference_foot_p'][:n, 0], axis=-1)
    count = data['contact_count'][:n, 0]; force = data['contact_force_w'][:n, 0]
    def sample(i):
        t = max(0, i-1)
        return dict(step=i, time_s=i*.02, height_m=float(data['root_p'][i, 0, 2]),
                    roll_pitch_deg=euler[i, :2].tolist(), q_abs_max_rad=float(abs(q[i]).max()),
                    knees_rad=q[i, [JOINT_NAMES.index('leg_l4_joint'), JOINT_NAMES.index('leg_r4_joint')]].tolist(),
                    foot_error_m=foot_error[i].tolist(), contact_count=count[i].tolist(),
                    contact_force_z_N=force[i, :, 2].tolist(),
                    normalized_reward=float(data['reward'][t, 0]/mdp.max_step_reward),
                    reward_terms={k.removeprefix('reward_'): float(data[k][t, 0]) for k in data if k.startswith('reward_')})
    summary = dict(episode_steps=length, final=sample(length),
                   samples=[sample(i) for i in sorted(set(min(length, v) for v in (1, 25, 50, 75, 90, 95, length)))],
                   max_abs_q_rad=float(abs(q).max()), max_abs_qd_rad_s=float(abs(qd).max()),
                   max_abs_action=float(abs(action).max()), max_action_change=float(abs(np.diff(action, axis=0)).max()),
                   action_change_rms=float(np.sqrt(np.mean(np.diff(action, axis=0)**2))),
                   max_abs_pd_target_rad=float(abs(target).max()),
                   reference_world_drift_m=float(abs(data['reference_foot_p'][:n]-data['reference_foot_p'][0]).max()),
                   feet_ground_contact_fraction=(count[1:] > 0).mean(0).tolist(),
                   feet_loaded_fraction=(force[1:, :, 2] > 1.).mean(0).tolist(),
                   min_total_ground_force_z_N=float(force[1:, :, 2].sum(-1).min()),
                   max_other_ground_contacts=int(data['other_ground_contacts'][1:n, 0].max()))
    # Find the dangerous interval explicitly; a good episode average can hide it.
    low = data['root_p'][1:n, 0, 2] < mdp.nominal_height-.08
    summary['height_drop_over_8cm'] = dict(steps=int(low.sum()),
        normalized_reward_mean=float((data['reward'][:length, 0][low]/mdp.max_step_reward).mean()) if low.any() else None,
        foot_rmse_m=float(np.sqrt(np.mean(foot_error[1:][low]**2))) if low.any() else None)
    rows = []
    for i in range(n):
        row = dict(step=i, time_s=time[i], root_height_m=float(data['root_p'][i, 0, 2]),
                   root_roll_deg=euler[i, 0], root_pitch_deg=euler[i, 1])
        for field in ('root_p', 'root_v', 'root_w'):
            row.update({f'{field}_{axis}': float(data[field][i, 0, j]) for j, axis in enumerate('xyz')})
        for j, joint in enumerate(JOINT_NAMES):
            row.update({f'{joint}_q': q[i, j], f'{joint}_qd': qd[i, j]})
            if i < length:
                row.update({f'{joint}_action': action[i, j], f'{joint}_pd_target': target[i, j]})
        for side, name in enumerate(('left', 'right')):
            row[f'{name}_contact_count'] = count[i, side]
            for j, axis in enumerate('xyz'):
                row[f'{name}_foot_{axis}'] = data['actual_foot_p'][i, 0, side, j]
                row[f'{name}_contact_force_{axis}_N'] = force[i, side, j]
        if i > 0:
            row['normalized_reward'] = data['reward'][i-1, 0]/mdp.max_step_reward
            for k in data:
                if k.startswith('reward_'): row[k] = data[k][i-1, 0]
        rows.append(row)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.with_suffix('.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    fig, axes = plt.subplots(5, 2, figsize=(15, 17))
    axes = axes.flatten(); ts = time[1:]
    axes[0].plot(time, data['root_p'][:n, 0, 2]); axes[0].axhline(mdp.nominal_height-mdp.cfg.term_height, ls='--', c='r')
    axes[0].set(ylabel='m', title='Root height; dashed = termination threshold')
    axes[1].plot(time, euler[:, :2], label=['roll', 'pitch']); axes[1].set(ylabel='deg', title='Root roll / pitch')
    for ax, values, title, unit in [(axes[2], q, 'Joint position (29 joints)', 'rad'),
                                    (axes[3], qd, 'Joint velocity (29 joints)', 'rad/s')]:
        ax.plot(time, values, alpha=.45, lw=.7); ax.set(title=title, ylabel=unit)
    for ax, values, title, unit in [(axes[4], action, 'Action (29 joints)', 'scaled'),
                                    (axes[5], target, 'Applied PD target (29 joints)', 'rad')]:
        ax.plot(time[:-1], values, alpha=.45, lw=.7); ax.set(title=title, ylabel=unit)
    axes[6].plot(time, foot_error, label=['left', 'right']); axes[6].set(title='World foot position error', ylabel='m')
    axes[7].plot(time, data['actual_foot_p'][:n, 0, :, 2], label=['left', 'right'])
    axes[7].axhline(data['reference_foot_p'][0, 0, 0, 2], ls='--'); axes[7].set(title='Ankle height', ylabel='m')
    axes[8].plot(time, force[:, :, 2], label=['left Fz', 'right Fz'])
    ax2 = axes[8].twinx(); ax2.step(time, (count > 0).sum(-1), c='gray', alpha=.5)
    ax2.set(ylabel='feet contacting ground', ylim=(-.1, 2.1)); axes[8].set(title='Actual last-substep contact forces', ylabel='N')
    axes[9].plot(ts, data['reward'][:length, 0]/mdp.max_step_reward, label='total / max')
    for k in ('base_height', 'base_upright', 'stance_stability', 'foot_orientation'):
        axes[9].plot(ts, data['reward_'+k][:length, 0], label=k)
    axes[9].set(title='Reward / individual quality factors')
    for ax in axes:
        ax.set_xlabel('time (s)'); ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=8)
    fig.suptitle(path.stem); fig.tight_layout(); fig.savefig(path.with_suffix('.png'), dpi=140); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for side, name in enumerate(('left', 'right')):
        for j, axis in enumerate('xyz'):
            ax = axes[side, j]
            ax.plot(time, data['actual_foot_p'][:n, 0, side, j], label='actual')
            ax.plot(time, data['reference_foot_p'][:n, 0, side, j], '--', label='reference')
            ax.set(title=f'{name} ankle {axis}', xlabel='time (s)', ylabel='m'); ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(path.with_name(path.stem+'_feet.png'), dpi=140); plt.close(fig)
    return summary, data


def plot_wm(rows, output):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for h in (1, 4, 8):
        subset = [r for r in rows if r['horizon'] == h and r['env'] == 0]
        t = np.array([r['start'] for r in subset])*.02
        axes[0].plot(t, [r['height_bias_m'] for r in subset], label=f'{h} steps')
        axes[1].plot(t, [r['orientation_error_deg'] for r in subset], label=f'{h} steps')
        axes[2].plot(t, [r['imagined_reward']-r['simulation_reward'] for r in subset], label=f'{h} steps')
    for ax, title in zip(axes, ['WM - Newton height (m)', 'Orientation error (deg)', 'Paired reward bias (normalized)']):
        ax.set(title=title, xlabel='window start (s)'); ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(output, dpi=150); plt.close(fig)


def audit_perturbed_models(report, out, wm, mdp, device):
    """Evaluate already-recorded perturbed trajectories, without more simulation."""
    duration = max(r['duration_s'] for r in report['evaluations'])
    results = []
    for evaluation in report['evaluations']:
        if evaluation['duration_s'] != duration:
            continue
        controller, seed = evaluation['controller'], evaluation['seed']
        with np.load(out/f'{controller}_{duration:g}s_seed{seed}.npz', allow_pickle=False) as z:
            trace = dict(z)
        command = FootstepCommand(**{k.removeprefix('command_'): torch.as_tensor(v, device=device)
                                    for k, v in trace.items() if k.startswith('command_') and k != 'command_step'})
        prediction, rows = paired_predictions(wm, mdp, command, trace, device)
        results.append(dict(controller=controller, seed=seed, **prediction))
        write_csv(out/f'wm_{controller}_seed{seed}.csv', rows)
    report['paired_perturbed_models'] = results


def main(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = args.device if args.analyze_only else 'cuda:0'; torch.backends.cuda.matmul.allow_tf32 = True
    ck = torch.load(args.checkpoint, map_location=dev, weights_only=False); saved = ck['args']
    if saved.get('goal_mode') != 'foot_trajectory': raise ValueError('Standing diagnosis requires a Foot checkpoint')
    cfg = resolve_foot_task(saved['foot_task_config'])
    if cfg.command_mode != 'standing': raise ValueError('Standing command required')
    kin = S54Kinematics(dev); spec = GoalSpec('foot_trajectory', saved['goal_frames'], saved['goal_dt'])
    goals = GoalSource(None, kin, spec, AlternatingConfig(**saved['foot_config']), command_mode='standing')
    mdp = FootTrackingMDP(kin, torch.tensor(default_joint_pos_list(), device=dev), goals, cfg)
    assert_checkpoint(ck); spec.assert_checkpoint(ck, goals.foot_config); assert_task_checkpoint(ck, mdp)
    wm = WorldModel().to(dev); wm.load_state_dict(ck['wm']); wm.eval()
    networks.AMP_DTYPE = torch.bfloat16 if saved.get('amp') == 'bf16' and str(dev).startswith('cuda') else None
    if args.analyze_only:
        report = json.loads((out/'summary.json').read_text())
        for controller in ('policy', 'default_pd'):
            summary, trace = audit_trace(out/f'nominal_{controller}.npz', mdp)
            report['nominal'][controller]['trace'] = summary
            command = FootstepCommand(**{k.removeprefix('command_'): torch.as_tensor(v, device=dev)
                                        for k, v in trace.items() if k.startswith('command_') and k != 'command_step'})
            prediction, rows = paired_predictions(wm, mdp, command, trace, dev)
            report['paired_world_model'][controller] = prediction
            write_csv(out/f'wm_{controller}.csv', rows); plot_wm(rows, out/f'wm_{controller}.png')
        report['analysis_device'] = dev
        audit_perturbed_models(report, out, wm, mdp, dev)
        (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
        return
    sim_cfg = make_sim(dev, args)
    report = dict(checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_iteration=ck['iteration'],
                  device=torch.cuda.get_device_name(), optimizer_updates=0, reward_changed=False,
                  evaluations=[], nominal={}, paired_world_model={}, task_contract=mdp.contract())
    with launch_simulation(sim_cfg, args):
        env = TrackingSim(sim_cfg, args.num_envs, None, mdp, eval_mode=True, goals=goals)
        for controller in ('policy', 'default_pd'):
            path = out/f'nominal_{controller}.npz'
            result = evaluate_foot(env, ck, goals, mdp, path, dev, duration_s=10., seed=54,
                                   controller=controller, diagnostics=True)
            summary, trace = audit_trace(path, mdp)
            report['nominal'][controller] = dict(evaluation=result, trace=summary)
            command = FootstepCommand(**{k.removeprefix('command_'): torch.as_tensor(v, device=dev)
                                        for k, v in trace.items() if k.startswith('command_') and k != 'command_step'})
            prediction, rows = paired_predictions(wm, mdp, command, trace, dev)
            report['paired_world_model'][controller] = prediction
            write_csv(out/f'wm_{controller}.csv', rows); plot_wm(rows, out/f'wm_{controller}.png')
            print(json.dumps(dict(controller=controller, nominal=result, wm=prediction)), flush=True)
        for duration in args.durations:
            for seed in args.seeds:
                for controller in ('policy', 'default_pd'):
                    path = out/f'{controller}_{duration:g}s_seed{seed}.npz'
                    result = evaluate_foot(env, ck, goals, mdp, path, dev, duration_s=duration,
                                           seed=seed, reset_noise=True, controller=controller, diagnostics=True)
                    report['evaluations'].append(result)
                    print(json.dumps(result), flush=True)
        # These must match exactly: same seed gives each controller the same initial state/command.
        pairing = []
        for duration in args.durations:
            for seed in args.seeds:
                with np.load(out/f'policy_{duration:g}s_seed{seed}.npz') as a, np.load(out/f'default_pd_{duration:g}s_seed{seed}.npz') as b:
                    error = float(abs(a['state'][0]-b['state'][0]).max())
                    command_error = float(abs(a['command_initial_pos']-b['command_initial_pos']).max())
                    pairing.append(dict(duration_s=duration, seed=seed, initial_state_max_error=error, command_max_error=command_error))
        report['controller_pairing'] = pairing
        assert all(r['initial_state_max_error'] < 1e-6 and r['command_max_error'] == 0 for r in pairing)
        audit_perturbed_models(report, out, wm, mdp, dev)
    (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    # One row per environment, fall vs normal timeout stays explicit.
    rows = []
    for result in report['evaluations']:
        for e, (length, reason) in enumerate(zip(result['episode_lengths'], result['stop_reasons'])):
            rows.append(dict(controller=result['controller'], duration_s=result['duration_s'], seed=result['seed'], env=e,
                             episode_steps=length, elapsed_s=length*.02, stop_reason=reason,
                             physical_fall=reason != 'timeout', normalized_reward=result['normalized_sim_reward']))
    write_csv(out/'evaluation.csv', rows)
    print(f'Diagnosis written to {out}', flush=True)


if __name__ == '__main__': main(parse_args())
