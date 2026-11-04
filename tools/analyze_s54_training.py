"""Read-only summary and exported curves for an S54 curriculum training run."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run_dir', required=True)
    p.add_argument('--out_dir', required=True)
    args = p.parse_args()
    run, out = Path(args.run_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wm_only = json.loads((run/'args.json').read_text()).get('wm_only_iters', 0)
    with (run/'train_log.csv').open() as f:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    events = [json.loads(line) for line in (run/'curriculum_metrics.jsonl').read_text().splitlines()]
    by_iteration = {int(r['iteration']): r for r in rows}
    evaluations = []
    evaluation_wall = 0.
    for path in sorted(run.glob('eval_[0-9]*.json')):
        result = json.loads(path.read_text())
        iteration = int(path.stem[5:])
        evaluation_wall += result['wall_seconds']
        for row in result['rows']:
            if row['status'] != 'evaluated': continue
            evaluations.append(dict(iteration=iteration, wall_h=by_iteration[iteration]['wall_h'],
                kind=result['kind'], success=row['success_rate'], physical=row['physical_termination_rate'],
                timeout=row['timeout_rate'], tracking_failure=row['goal_tracking_failure_rate'],
                foot_rmse=row['foot_rmse'], mean_survival_s=statistics.mean(row['episode_lengths'])*.02,
                protocol_seconds=row['duration'], speed=row['actual_speed'], yaw_mae=row['yaw_rate_error'],
                physical_value_bias=row.get('critic_mc_physical_bias'), source=str(path)))
    status = json.loads((run/'training_status.json').read_text())
    summary = dict(run=str(run), status=status, final_logged_metrics=rows[-1],
        finite_training_csv=all(math.isfinite(v) for r in rows for v in r.values()),
        evaluation_count=len(evaluations), evaluation_seconds=evaluation_wall,
        evaluation_fraction=evaluation_wall/status['wall_seconds'],
        actual_transitions=int(rows[-1]['sim_steps']),
        effective_transitions_s=rows[-1]['sim_steps']/status['wall_seconds'],
        stage_values=sorted({e['values']['Curriculum/Stage'] for e in events if 'Curriculum/Stage' in e['values']}),
        executed_distribution={k: v for k, v in events[-1]['values'].items() if k.startswith('Commands/Executed/task_')},
        ranges={k: dict(min=min(r[k] for r in rows), max=max(r[k] for r in rows))
                for k in ('wm_loss','wm_err8','v_loss','pi_gradnorm','sim_rew')}, hourly_medians=[], evaluations=evaluations)
    for hour in range(8):
        part = [r for r in rows if hour <= r['wall_h'] < hour+1 and r['iteration'] > wm_only]
        if part:
            summary['hourly_medians'].append(dict(hour=hour+1, **{
                k: statistics.median(r[k] for r in part)
                for k in ('wm_loss','wm_err8','v_loss','pi_gradnorm','sim_rew','ep_len','imag_survival','it_per_s')}))
    for key, lo, hi in [('initial_actor',1000,5000),('late_actor',30000,40000)]:
        part = [e['values'] for e in events if lo < e['iteration'] <= hi]
        summary[key] = {}
        for field in ('actor_bootstrap_mean','actor_reward_return_mean','actor_reward_gradnorm',
                      'actor_bootstrap_gradnorm','critic_real_mc_loss','imagined_action_clip_fraction'):
            values = [e['Optimization/'+field] for e in part if 'Optimization/'+field in e]
            if values: summary[key][field] = statistics.median(values)
    summary['first_norm_freeze_possible_h'] = next((r['wall_h'] for r in rows if r['sim_steps'] >= 2e8), None)
    (out/'training_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (out/'evaluations.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(evaluations[0]))
        writer.writeheader(); writer.writerows(evaluations)

    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 4, figsize=(19, 8), constrained_layout=True)
    time_h = np.array([r['wall_h'] for r in rows])

    def curve(ax, key, label, positive=False):
        y = np.array([r[key] for r in rows])
        mask = (y > 0) if positive else np.ones(len(y), dtype=bool)
        x, y = time_h[mask], y[mask]
        raw, = ax.plot(x, y, alpha=.16, lw=.6)
        width = 21
        if len(y) >= width:
            ax.plot(x[width//2:-(width//2)], np.convolve(y,np.ones(width)/width,'valid'),
                    color=raw.get_color(), label=label, lw=1.5)

    ax = axes[0,0]
    curve(ax,'wm_loss','WM loss',True); curve(ax,'wm_err8','WM error H8',True)
    ax.set_yscale('log'); ax.set_title('World-model error (normalized)'); ax.legend()
    ax = axes[0,1]
    curve(ax,'v_loss','Imagined Critic MSE',True)
    ax.set_yscale('log'); ax.set_title('Critic loss'); ax.legend()
    ax = axes[0,2]
    actor = [e for e in events if 'Optimization/actor_bootstrap_mean' in e['values']]
    xx = [by_iteration[e['iteration']]['wall_h'] for e in actor]
    ax.plot(xx,[e['values']['Optimization/actor_bootstrap_mean'] for e in actor],lw=.8,label='Terminal value contribution')
    ax.plot(xx,[e['values']['Optimization/actor_reward_return_mean'] for e in actor],lw=.8,label='H8 reward contribution')
    ax.set_title('Actor objective components'); ax.legend()
    ax = axes[0,3]
    curve(ax,'sim_rew','Simulation reward'); curve(ax,'imag_rew','Imagined reward')
    ax.set_title('Per-step normalized reward (unpaired starts)'); ax.legend()
    xx = [e['wall_h'] for e in evaluations]
    ax = axes[1,0]
    for key,label in [('success','Tracking success'),('physical','Physical termination'),('timeout','Timeout')]:
        ax.plot(xx,[e[key] for e in evaluations],'.-',label=label,alpha=.85)
    ax.axhline(.95,ls='--',c='k',lw=.8,label='Standing gate .95')
    ax.set_ylim(-.03,1.07); ax.set_title('Independent evaluation; 8 envs'); ax.legend(fontsize=8)
    ax = axes[1,1]
    ax.plot(xx,[e['foot_rmse']*100 for e in evaluations],'.-',label='Foot RMSE')
    ax.axhline(4.5,ls='--',c='k',lw=.8,label='Gate 4.5 cm')
    ax.set_title('Foot trajectory RMSE (cm)'); ax.legend()
    ax = axes[1,2]
    ax.plot(xx,[e['mean_survival_s'] for e in evaluations],'.-',label='Time to termination / timeout')
    ax.plot(xx,[e['protocol_seconds'] for e in evaluations],'--',label='Protocol duration')
    ax.set_title('Survival duration (s); not tracking success'); ax.legend(fontsize=8)
    ax = axes[1,3]
    ax.plot(xx,[e['speed'] for e in evaluations],'.-',label='Mean speed')
    ax.axhline(.025,ls='--',c='k',lw=.8,label='Standing gate .025 m/s per env')
    ax.set_title('Residual base speed (m/s)'); ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel('Wall clock (h)'); ax.grid(alpha=.2); ax.set_xlim(0,8)
    stages = ', '.join(f'C{int(s)}' for s in summary['stage_values'])
    fig.suptitle(f'S54 curriculum: {stages} | {status["iteration"]:,} iterations | '
                 f'{status["wall_seconds"]/3600:.3f} h',fontsize=16)
    fig.savefig(out/'training_curves.png',dpi=170)
    fig.savefig(out/'training_curves.pdf')
    plt.close(fig)
    print(json.dumps({k:summary[k] for k in ['evaluation_count','evaluation_fraction','stage_values','finite_training_csv','first_norm_freeze_possible_h']},indent=2))


if __name__ == '__main__': main()
