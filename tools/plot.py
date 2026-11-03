"""Plot chronological training stages without changing their logs.

Real ckpt_*_eval.json results fill missing CSV rows at matching training times.
Missing evaluations are reported, never interpolated.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ("#2a78d6", "#16876e", "#8955a4", "#b56b27")
INK, INK2, GRID, SURF, REF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#8a8984"


def read(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return list(csv.DictReader(stream))


def load_run(path):
    path = Path(path)
    config_path = path / 'args.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    training = {int(r['iteration']): r for r in read(path / 'train_log.csv')}
    evaluations = {int(r['iteration']): r for r in read(path / 'eval.csv')}
    for result_path in sorted(path.glob('ckpt_*_eval.json')):
        iteration = int(result_path.name.split('_')[1])
        if iteration in evaluations or iteration not in training:
            continue
        result = json.loads(result_path.read_text())
        evaluations[iteration] = dict(iteration=iteration,
                                      train_wall_h=training[iteration]['wall_h'],
                                      norm_return=result['norm_return'],
                                      n_failed=result['n_failed'])
    checkpoints = {int(p.stem.split('_')[1]) for p in path.glob('ckpt_*.pt')}
    return dict(path=str(path), name=path.name, config=config,
                training=[training[k] for k in sorted(training)],
                evaluations=[evaluations[k] for k in sorted(evaluations)],
                missing_evaluations=sorted(checkpoints - set(evaluations)))


def style(ax, title, ylabel, x_axis):
    ax.set_facecolor(SURF)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_xlabel('training iteration' if x_axis == 'iteration' else
                  'wall-clock training time (h), 1 GPU', color=INK2, fontsize=9)
    ax.grid(True, color=GRID, lw=.8)
    for name in ('top', 'right'):
        ax.spines[name].set_visible(False)
    for name in ('left', 'bottom'):
        ax.spines[name].set_color('#c9c8c2')
    ax.tick_params(colors=INK2, labelsize=8)


def coordinates(rows, time_column, x_axis):
    # Resume logs already contain cumulative wall_h; never add another offset.
    return [int(row['iteration']) if x_axis == 'iteration' else
            float(row[time_column]) for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dirs', nargs='+', help='Run directories, earliest first')
    parser.add_argument('--out', default=None)
    parser.add_argument('--target', type=float, default=.82)
    parser.add_argument('--x-axis', choices=('wall', 'iteration'), default='wall')
    parser.add_argument('--paper-reference', action='store_true')
    args = parser.parse_args()
    if args.paper_reference and args.x_axis != 'wall':
        parser.error('--paper-reference requires --x-axis wall')
    runs = [load_run(path) for path in args.run_dirs]
    counts = {len(run['config'].get('clips') or []) for run in runs}
    clip_count = next(iter(counts)) if len(counts) == 1 else 0
    clip_label = f'{clip_count} clips' if clip_count else 'recorded clips'
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor=SURF)
    style(axes[0], f'(a) Evaluation normalized return, {clip_label}', 'normalized return', args.x_axis)
    style(axes[1], f'(b) Failed clips in evaluation ({clip_label})', 'clips terminated early', args.x_axis)
    style(axes[2], '(c) Training-episode normalized return', 'mean normalized return per episode', args.x_axis)
    all_returns, all_failures = [], []
    for index, run in enumerate(runs):
        color = COLORS[index % len(COLORS)]
        evaluations, training = run['evaluations'], run['training']
        if evaluations:
            x = coordinates(evaluations, 'train_wall_h', args.x_axis)
            returns = [float(row['norm_return']) for row in evaluations]
            failures = [int(row['n_failed']) for row in evaluations]
            all_returns += returns
            all_failures += failures
            label = f"{run['name']} ({len(evaluations)} eval)"
            axes[0].plot(x, returns, color=color, lw=2, marker='o', ms=4.5,
                         mec=SURF, mew=1.2, zorder=3, label=label)
            axes[1].plot(x, failures, color=color, lw=2, marker='o', ms=4.5,
                         mec=SURF, mew=1.2, zorder=3, label=run['name'])
            axes[0].annotate(f'{returns[-1]:.3f}', (x[-1], returns[-1]), xytext=(-4, 10),
                             textcoords='offset points', fontsize=8, color=color, ha='center')
        if training:
            x = coordinates(training, 'wall_h', args.x_axis)
            y = [float(row['ep_ret_norm']) for row in training]
            # Keep smoothing inside each stage, respecting changed rewards.
            smoothed = [sum(y[max(0, i - 10):i + 1]) / len(y[max(0, i - 10):i + 1])
                        for i in range(len(y))]
            axes[2].plot(x, smoothed, color=color, lw=2, zorder=3, label=run['name'])
        print(f"{run['name']}: {len(training)} training rows, {len(evaluations)} evaluation points")
        if run['missing_evaluations']:
            print('  Missing checkpoint evaluations:', ', '.join(map(str, run['missing_evaluations'])))

    for previous in runs[:-1]:
        if previous['training']:
            row = previous['training'][-1]
            boundary = coordinates([row], 'wall_h', args.x_axis)[0]
            for ax in axes:
                ax.axvline(boundary, color=REF, ls='--', lw=1, alpha=.8)
            axes[0].text(boundary, .98, f" resume @ {int(row['iteration'])}",
                         transform=axes[0].get_xaxis_transform(), va='top', fontsize=7, color=INK2)

    axes[0].axhline(args.target, color=REF, lw=1, ls='--', zorder=1)
    axes[0].text(.01, args.target + .01, f'target {args.target}',
                 transform=axes[0].get_yaxis_transform(), fontsize=8, color=INK2)
    if args.paper_reference:
        axes[0].plot([1.2], [.82], marker='D', ms=7, color=REF, mec=SURF, zorder=4)
        axes[0].annotate('paper: 0.82 at 1.2 h (H100)', (1.2, .82), xytext=(-60, -22),
                         textcoords='offset points', fontsize=8, color=INK2)
    axes[0].set_ylim(min(0., min(all_returns, default=0.)), max(1., max(all_returns, default=1.)))
    axes[1].set_ylim(-.5, max(clip_count, max(all_failures, default=0), 1) + .5)
    axes[2].set_ylim(0, 1)
    for ax in axes:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, loc='best', framealpha=.9)

    protocols = [(r['config'].get('motion_dir'), r['config'].get('reward_config')) for r in runs]
    changed = any(protocol != protocols[0] for protocol in protocols[1:])
    if changed:
        fig.text(.5, .02, 'Reference/reward settings changed between stages; lines and smoothing stay within each stage.',
                 ha='center', fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, .05 if changed else 0, 1, 1))
    default_name = 'success_curve_combined.png' if len(runs) > 1 else 'success_curve.png'
    output = Path(args.out) if args.out else Path(args.run_dirs[-1]) / default_name
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, facecolor=SURF)
    plt.close(fig)
    metadata = dict(x_axis=args.x_axis, protocol_changed=changed, stages=[dict(
        run_dir=run['path'], training_rows=len(run['training']),
        training_iterations=[int(r['iteration']) for r in run['training']],
        evaluation_iterations=[int(r['iteration']) for r in run['evaluations']],
        missing_evaluations=run['missing_evaluations'],
        motion_dir=run['config'].get('motion_dir'), reward_config=run['config'].get('reward_config'),
        clips=run['config'].get('clips')) for run in runs])
    output.with_suffix('.data.json').write_text(json.dumps(metadata, indent=2))
    print('wrote', output)


if __name__ == '__main__':
    main()
