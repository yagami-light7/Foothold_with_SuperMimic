"""Container-side bounded M4 preparation/training/eval job; never tunes on test."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def call(*command):
    print('EXEC:', ' '.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)


def train_args(config, run, smoke=False, resume=None):
    cfg = dict(config)
    eval_every = cfg.pop('eval_every', 1000)
    eval_seeds = cfg.pop('eval_seeds', [0, 1, 2])
    if type(eval_every) is not int or eval_every < 0:
        raise ValueError('eval_every must be a nonnegative integer')
    if not isinstance(eval_seeds, list) or not eval_seeds or any(type(s) is not int or s < 0 for s in eval_seeds) or len(set(eval_seeds)) != len(eval_seeds):
        raise ValueError('eval_seeds must contain distinct nonnegative integer seeds')
    if smoke:
        cfg.update(num_envs=16, buffer_steps=64, warmup_steps=32, wm_batch=64,
                   wm_batches=1, pi_batch=32, horizon=8, iterations=30,
                   wm_only_iters=10, ckpt_every=30, log_every=10, wall_seconds=300)
    args = [sys.executable, '-u', 'supermimic/train.py', '--m34_dataset', cfg.pop('dataset'),
            '--log_dir', str(run), '--goal_mode', 'motion', '--no_compile',
            '--m34_eval_every', str(0 if smoke else eval_every),
            '--m34_eval_seeds', *map(str, eval_seeds)]
    for k, v in cfg.items():
        args += ['--'+k, str(v)]
    if resume: args += ['--resume', resume]
    return args


def check_smoke(run):
    import torch
    from supermimic.s54 import assert_checkpoint
    rows = list(csv.DictReader((run/'train_log.csv').open()))
    if not rows or int(rows[-1]['iteration']) != 30:
        raise RuntimeError('Incomplete smoke')
    keys = ('wm_loss', 'wm_err8', 'pi_loss', 'v_loss', 'imag_rew', 'sim_rew', 'pi_gradnorm')
    if not all(math.isfinite(float(row[k])) for row in rows for k in keys):
        raise RuntimeError('Nonfinite smoke training metrics')
    if not any(float(row['pi_gradnorm']) > 0 for row in rows):
        raise RuntimeError('No BPTT Actor gradient')
    ck = torch.load(run/'last.pt', map_location='cpu', weights_only=False)
    assert_checkpoint(ck)
    if ck['iteration'] != 30 or not all(k in ck for k in ('opt_wm', 'opt_pi', 'opt_v')):
        raise RuntimeError('Incomplete resumable checkpoint')
    if not all(torch.isfinite(x).all() for model in ('wm', 'pi', 'critic') for x in ck[model].values()):
        raise RuntimeError('Nonfinite checkpoint')


def report(run, config, checkpoint, evals):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader((run/'train_log.csv').open()))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, keys in zip(axes.flat, [('wm_loss', 'wm_err8'), ('sim_rew', 'imag_rew'),
                                    ('pi_gradnorm',), ('ep_len',)]):
        for key in keys:
            ax.plot([int(r['iteration']) for r in rows], [float(r[key]) for r in rows], label=key)
        ax.set_xlabel('iteration'); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(run/'training_curves.png'); plt.close(fig)
    lines = ['# S54 M4.2 bounded training results', '',
             f'Run: `{run}`. Frozen final checkpoint: `{checkpoint}`.', '',
             'Original SuperMimic Motion Encoder + shared decoder, differentiable WM, BPTT Actor and Critic.',
             'WM labels are real Newton transitions from replay; no reference transition supervision.',
             'One unified policy over 14 train clips. Test arrays are loaded only by independent frozen evaluation processes.',
             f'Independent train/test evaluation every {config.get("eval_every", 1000)} iterations; no test feedback enters optimization or checkpoint selection.',
             'Native 50 Hz / 29 joints / 30 bodies; strict hashes, all-frame FK and random Newton FK gate.',
             'Straight emphasis 80/10/10 then 1/3 per class; equal clips within a class, adaptive failure bins within a clip.',
             'No separate standing curriculum has been added in this initial bounded run.', '',
             'Budget/config: `pipeline_config.json`. Curve: `training_curves.png`.',
             'This short run is a pipeline baseline; it does not establish convergence or stable tracking.', '',
             '| split / reset | mode | trials | survival | mean duration (s) | root RMSE (m) | foot RMSE (m) | normalized return |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for split, result in evals.items():
        groups = sorted({(r['mode'], r['reset_noise']) for r in result['rows']})
        for mode, noisy in groups:
            group = [r for r in result['rows'] if r['mode'] == mode and r['reset_noise'] == noisy]
            avg = lambda k: float(np.mean([r[k] for r in group]))
            reset = 'perturbed' if noisy else 'nominal'
            lines.append(f'| {split}/{reset} | {mode} | {len(group)} | {avg("survived"):.3f} | {avg("episode_s"):.3f} | {avg("root_position_rmse_m"):.4f} | {avg("foot_position_rmse_m"):.4f} | {avg("normalized_return"):.4f} |')
    lines += ['', 'All row metrics, termination frames/reasons, contacts, slip and torque availability are in `evals/*/attempt_*/*/results.json` (legacy: `eval_*/results.json`).',
              f'Evaluation seeds: {config.get("eval_seeds", [0, 1, 2])}. Seed 0 is nominal; nonzero seeds use the training reset perturbation distribution (separate reports).',
              'Forward/Turn/Spin tests are same-template parameter generalization, not disjoint-source generalization.',
              'Backward is cross-motion-mode extrapolation. Test results never select checkpoints or sampler weights.',
              'The precommitted final checkpoint is evaluated; no best checkpoint is chosen from test performance.',
              'Foot RMSE is absolute world-space link-origin error. Stance slip is the contacting link-origin horizontal-speed proxy, not contact-patch slip.',
              'Contact measurements are last solver substep. Actuator-reported efforts can be implicit-controller estimates; missing data is null.',
              'Videos render recorded Newton states; they do not drive the physical rollout.',
              'Ablations and longer convergence study are not completed by this initial bounded run.']
    (run/'S54_M4_2_RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True); p.add_argument('--config', required=True); p.add_argument('--resume')
    args = p.parse_args()
    if not args.run.startswith('s54_') or '/' in args.run: p.error('Use a fresh s54_* run name')
    run = Path('logs')/args.run
    run.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text())
    if (run/'pipeline_config.json').exists(): raise FileExistsError('Pipeline run already exists')
    (run/'pipeline_config.json').write_text(json.dumps(config, indent=2))
    # Preserve code used by this run, including uncommitted integration work.
    source = run/'source'; source.mkdir()
    for directory in ('supermimic', 'tools', 'configs'):
        shutil.copytree(directory, source/directory, ignore=shutil.ignore_patterns('__pycache__'))
    (run/'launch.json').write_text(json.dumps(dict(source_snapshot=str(source),
        git_sha=os.environ.get('S54_SOURCE_GIT_SHA', 'unavailable-in-container')), indent=2))
    status = dict(status='preflight', started=time.time(), config=args.config)
    def update(stage):
        status['stage'] = stage
        if status['status'] not in ('completed', 'failed'): status['status'] = 'running'
        (run/'pipeline_status.json').write_text(json.dumps(status, indent=2))
    rollout = [sys.executable, '-u', 'tools/m34_rollout.py', '--dataset', config['dataset']]
    try:
        update('cpu_contract_tests'); call(sys.executable, 'tools/test_s54_m34.py')
        update('pd_baseline')
        call(*rollout, '--out', run/'pd')
        for clean_id in ('straight_flat_20', 'turn_flat_11', 'spin_flat_02'):
            raw_command = [*rollout, '--out', run/('pd_raw_'+clean_id), '--raw_comparison', clean_id]
            # A raw reference failure is diagnostic, not a reason to reject
            # the clean train set. m34_rollout retains its error JSON.
            print('RAW COMPARISON:', clean_id, flush=True)
            subprocess.run(list(map(str, raw_command)), check=False)
        call(sys.executable, 'tools/report_s54_m34_pd.py', '--run_dir', run)
        update('smoke')
        smoke = Path(str(run)+'_smoke')
        call(*train_args(config, smoke, smoke=True)); check_smoke(smoke)
        update('training')
        call(*train_args(config, run, resume=args.resume))
        # Reuse a completed boundary evaluation instead of evaluating the same
        # final model twice. Non-boundary stops receive one final evaluation.
        update('frozen_evaluation')
        iteration = json.loads((run/'training_status.json').read_text())['iteration']
        interval = config.get('eval_every', 1000)
        evaluation_dir = run/'evals'/(f'iter_{iteration:06d}' if interval and iteration % interval == 0 else f'final_{iteration:06d}')
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = evaluation_dir/'checkpoint.pt'
        if not checkpoint.exists(): os.link(run/'last.pt', checkpoint)
        from supermimic.m34_evaluation import evaluate_checkpoint
        evaluations = evaluate_checkpoint(checkpoint, config['dataset'], evaluation_dir, config.get('eval_seeds', [0, 1, 2]))
        update('rendering')
        pd_rows = json.loads((run/'pd/results.json').read_text())['rows']
        for mode in ('straight', 'turn', 'spin'):
            row = next(r for r in pd_rows if r['mode'] == mode)
            call(sys.executable, 'tools/render_s54_motion_prior.py', '--records', row['record'],
                 '--labels', f'Newton PD {mode}', '--out', run/f'pd_{mode}.mp4', '--seconds', 2)
        for split, result in evaluations.items():
            for row in result['rows']:
                if row['seed'] != 0: continue
                call(sys.executable, 'tools/render_s54_motion_prior.py', '--records', row['record'],
                     '--labels', f'Newton {split} {row["id"]}', '--out', run/f'{split}_{row["id"]}.mp4', '--seconds', 2)
        report(run, config, checkpoint, evaluations)
        status['status'] = 'completed'; update('completed')
    except BaseException as exc:
        status.update(status='failed', error=repr(exc)); update(status['stage']); raise


if __name__ == '__main__': main()
