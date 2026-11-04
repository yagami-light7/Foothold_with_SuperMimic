"""Plot existing SuperMimic CSVs and checkpoint evaluations; runs no simulation/training."""

# Support both direct execution and python -m from the repository layout.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_root = str(_ToolPath(__file__).resolve().parents[2])
if _tool_root not in _tool_sys.path:
    _tool_sys.path.insert(0, _tool_root)

import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', type=Path, required=True)
    p.add_argument('--evaluations', nargs='*', type=Path, default=[], help='diagnose_s54_standing evaluation.csv files')
    p.add_argument('--out', type=Path, default=Path('results/s54/standing_training_curves.png'))
    a = p.parse_args()
    fig, axes = plt.subplots(3, 2, figsize=(14, 11)); axes = axes.flatten()
    for run in a.runs:
        with (run/'train_log.csv').open() as f: rows = list(csv.DictReader(f))
        x = [int(r['iteration']) for r in rows]
        for ax, key in zip(axes[:4], ['wm_loss', 'wm_err8', 'pi_gradnorm', 'ep_len']):
            values = np.array([float(r[key]) for r in rows])
            if key == 'ep_len': values[np.array([float(r['completed_episodes']) for r in rows]) == 0] = np.nan
            ax.plot(x, values, marker='.', label=run.name)
            ax.set(title=key, xlabel='iteration')
        for key, style in [('sim_rew', '-'), ('imag_rew', '--')]:
            axes[4].plot(x, [float(r[key]) for r in rows], style, label=run.name+':'+key)
    axes[4].set(title='Training reward means (different batches; descriptive only)', xlabel='iteration')
    for path in a.evaluations:
        with path.open() as f: rows = list(csv.DictReader(f))
        for ctrl in sorted(set(r['controller'] for r in rows)):
            selected = [r for r in rows if r['controller'] == ctrl]
            axes[5].scatter([float(r['duration_s']) for r in selected], [float(r['elapsed_s']) for r in selected],
                            marker='x' if ctrl == 'policy' else 'o', alpha=.7, label=path.parent.name+':'+ctrl)
    axes[5].plot([2, 5, 10], [2, 5, 10], 'k--', label='normal timeout')
    axes[5].set(title='Evaluation lifetime (inspect CSV stop_reason)', xlabel='protocol duration (s)', ylabel='actual lifetime (s)')
    for ax in axes:
        ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=7)
    fig.tight_layout(); a.out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(a.out, dpi=160); plt.close(fig)
    print(a.out)


if __name__ == '__main__': main()
