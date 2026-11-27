"""独立 CUDA Newton 的 M5.1 周期评估和后续终评入口。"""

from argparse import ArgumentParser
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

if __name__ == '__main__':
    os.environ['SUPERMIMIC_ROBOT'] = 'roban'

import numpy as np
import torch

from .m50_evaluation import (DEFAULT_SEEDS, HOLDOUT_SEEDS, INVALID_TERMINATIONS, PERTURB_SEEDS,
                             TRACE_FIELDS, mean_metrics, scene_for, summarize, trial, trial_metrics, trial_ok)
from .m48_evaluation import load_archive, save_archive
from .m50_reference import MANIFEST, REFERENCE_ROOT, REPO, read_manifest
from .m51_metrics import contact_metrics, read_guide_heading
from .m51_task import GROUPS, OUTPUT, SOURCE


def evaluate_cases(args, checkpoint, cases, *, split='train', seeds=DEFAULT_SEEDS, resets=None,
                   arrays=None, prefix='', measure=None):
    resets = {} if resets is None else resets
    for item in read_manifest(args.library)[split]:
        clip = item['clip_id']
        rows = cases.setdefault(clip, {})
        pending = [seed for seed in seeds if str(seed) not in rows]
        if not pending:
            continue
        root = args.reference_root if split == 'train' else REPO
        guide = read_guide_heading(root/item['file_path'])
        with scene_for(args, checkpoint, item, split) as (env, actor, sensors, solver):
            for seed in pending:
                trace, row = trial(env, actor, sensors, solver, seed, perturb=seed not in DEFAULT_SEEDS)
                initial = trace['state'][0]
                np.testing.assert_allclose(initial, resets.setdefault((clip, seed), initial), atol=1e-7, rtol=0.)
                if row['termination'] in INVALID_TERMINATIONS:
                    save_archive(args.out/'evaluation/fault.npz',
                                 dict(checkpoint=str(checkpoint), clip=clip, seed=seed, reason=row['fault_detail']),
                                 {k: v for k, v in trace.items() if isinstance(v, np.ndarray)})
                    raise FloatingPointError(f'{checkpoint.name} {clip}/{seed} 无效 Newton 转移')
                trial_metrics(trace, row, env.mdp)
                row['stance'] = contact_metrics(trace, row, env.mdp, guide)
                if measure is not None:
                    row.update(measure(trace, row, env.mdp))
                if arrays is not None:
                    row['prefix'] = f'{prefix}_{clip}_s{seed}_'
                    fields = (*TRACE_FIELDS, 'foot_angular_velocity')
                    arrays.update({row['prefix']+key: trace[key] for key in fields})
                rows[str(seed)] = row
                print(f'{checkpoint.name} {clip}/{seed} {row["consecutive_planned_touchdowns"]}/'
                      f'{len(row["footsteps"])}，{row["termination"]}', flush=True)
        torch.cuda.empty_cache()


def summarize_foot(cases, seeds):
    summary = summarize(cases, seeds)
    for clip, metrics in summary.items():
        rows = [cases[clip][str(seed)] for seed in seeds]
        metrics['observed']['stance'] = mean_metrics([row['stance'] for row in rows])
        metrics['completed']['stance'] = mean_metrics([row['stance'] for row in rows if trial_ok(row)])
        if all('posture' in row for row in rows):
            metrics['observed']['posture'] = mean_metrics([row['posture'] for row in rows])
            metrics['completed']['posture'] = mean_metrics([row['posture'] for row in rows if trial_ok(row)])
            metrics['matched_touchdowns'] = sum(row['matched_touchdowns'] for row in rows)
    return summary


def rank(nominal, perturb=None, *, contact_first=True):
    def success(group):
        complete = [row['complete']/row['attempts'] for row in group.values()]
        survival = [row['survives']/row['attempts'] for row in group.values()]
        return float(np.mean(complete)), min(complete), float(np.mean(survival)), min(survival)

    quality = [row['completed'] if row['complete'] == row['attempts'] else row['observed']
               for row in nominal.values()]
    foot, root = [-float(np.mean([row[key] for row in quality])) for key in
                  ('foot_position_rmse_m', 'root_xy_rmse_m')]
    measured = [row['stance'] for row in quality]

    def lower(values):
        return -float(np.mean([x if x is not None else np.inf for x in values]))

    load = float(np.mean([s['plan_load_fraction'] for s in measured]))
    angular = lower([(s['support']['omega'] or {}).get('rms') for s in measured])
    slip = lower([(s['support']['segment_mean'] or {}).get('distance_mm') for s in measured])
    yaw = lower([s['planned_stance']['yaw_rms_deg'] for s in measured])
    arm = lower([row['body']['arm_rmse_rad'] for row in quality])
    jerk = lower([row['upper_smoothness']['after_startup']['jerk_rms_rad_s3'] for row in quality])
    contact = (angular, slip, yaw, foot) if contact_first else (foot, angular, slip, yaw)
    return success(nominal) + (success(perturb) if perturb else ()) + (load, *contact, root, arm, jerk)


def periodic(args):
    out = args.out/'evaluation'
    out.mkdir(parents=True, exist_ok=True)
    path = out/'results.json'
    info = json.loads(path.read_text()) if path.is_file() else dict(milestones={})
    ck = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    iteration, cfg = ck['iteration'], ck['args']
    if not cfg.get('m51_group') or iteration < 500 or iteration % 500:
        raise ValueError('周期评估需要 M5.1 每 500 轮冻结的本组联合权重')
    if args.out.name != GROUPS[cfg['m51_group']]:
        raise ValueError('检查点与 M5.1 组目录不一致')
    key = str(iteration)
    frozen = args.out/'checkpoints'/f'iter_{iteration:04d}.pt'
    if not frozen.samefile(args.checkpoint):
        raise ValueError('周期评估需要同一冻结检查点')
    if key not in info['milestones']:
        cases = {}
        evaluate_cases(args, frozen, cases)
        info['milestones'][key] = dict(iteration=iteration, checkpoint=str(frozen), rows=cases,
                                       nominal=summarize_foot(cases, DEFAULT_SEEDS))
    ranks = {'best_contact_stability': lambda row: rank(row['nominal']),
             'best_multi_reference': lambda row: rank(row['nominal'], contact_first=False)}
    publish_periodic(out, info, ranks)
    print({c: r['complete'] for c, r in info['milestones'][key]['nominal'].items()}, flush=True)


def publish_periodic(out, info, ranks):
    path = out/'results.json'
    with TemporaryDirectory(dir=out) as directory:
        tmp = Path(directory)
        for alias, ranker in ranks.items():
            row = max(info['milestones'].values(), key=ranker)
            info[alias] = row['iteration']
            temporary = tmp/f'{alias}.pt'
            temporary.hardlink_to(Path(row['checkpoint']))
            temporary.replace(out.parent/'checkpoints'/temporary.name)
        (tmp/path.name).write_text(json.dumps(info, ensure_ascii=False, allow_nan=False, indent=2)+'\n')
        (tmp/path.name).replace(path)


def final(args):
    path = args.out/'cuda_evaluation.npz'
    info, arrays = load_archive(path)
    cases = info.setdefault('cases', {})
    resets = {}
    for entry in cases.values():
        for table in ('seen', 'heldout'):
            for clip, rows in entry.get(table, {}).items():
                for seed, row in rows.items():
                    initial = arrays[row['prefix']+'state'][0]
                    np.testing.assert_allclose(initial, resets.setdefault((clip, int(seed)), initial), atol=1e-7, rtol=0.)
    by_group = {}
    for group, folder in GROUPS.items():
        out = args.out/folder
        if (out/'train.exitcode').read_text().strip() != '0':
            raise ValueError(f'M5.1-{group.upper()} 正式训练尚未完成')
        periodic_rows = json.loads((out/'evaluation/results.json').read_text())
        candidates = sorted({periodic_rows[key] for key in ('best_contact_stability', 'best_multi_reference')} | {2000})
        by_group[group] = {f'{group}{it}': out/'checkpoints'/f'iter_{it:04d}.pt' for it in candidates}
    sources = {'m50': SOURCE} | {key: path for options in by_group.values() for key, path in options.items()}
    try:
        for name, source in sources.items():
            entry = cases.setdefault(name, dict(checkpoint=str(source), seen={}))
            if entry['checkpoint'] != str(source):
                raise ValueError('继续复评需要保持候选路径一致')
            evaluate_cases(args, source, entry['seen'], seeds=DEFAULT_SEEDS+PERTURB_SEEDS, resets=resets,
                           arrays=arrays, prefix=name)
            entry['nominal'] = summarize_foot(entry['seen'], DEFAULT_SEEDS)
            entry['perturb'] = summarize_foot(entry['seen'], PERTURB_SEEDS)
            save_archive(path, info, arrays)
        winners = {key: max(options, key=lambda name: rank(cases[name]['nominal'], cases[name]['perturb']))
                   for key, options in by_group.items()}
        if info.setdefault('winners', winners) != winners:
            raise ValueError('检查点已选定，留出试次仅用于验证')
        for name in ('m50', *winners.values()):
            entry, source = cases[name], sources[name]
            evaluate_cases(args, source, entry['seen'], seeds=HOLDOUT_SEEDS, resets=resets, arrays=arrays, prefix=name)
            evaluate_cases(args, source, entry.setdefault('heldout', {}), split='heldout',
                           seeds=DEFAULT_SEEDS+PERTURB_SEEDS, resets=resets, arrays=arrays, prefix=name)
            entry['holdout_reset'] = summarize_foot(entry['seen'], HOLDOUT_SEEDS)
            entry['unseen_nominal'] = summarize_foot(entry['heldout'], DEFAULT_SEEDS)
            entry['unseen_perturb'] = summarize_foot(entry['heldout'], PERTURB_SEEDS)
    finally:
        save_archive(path, info, arrays)
    print('各组入选权重', info['winners'], flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('periodic', 'final'))
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    parser.add_argument('--library', type=Path, default=MANIFEST)
    parser.add_argument('--reference-root', type=Path, default=REFERENCE_ROOT)
    args = parser.parse_args()
    torch.set_num_threads(2)
    (periodic if args.mode == 'periodic' else final)(args)


if __name__ == '__main__':
    main()
