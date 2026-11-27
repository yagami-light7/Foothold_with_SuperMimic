"""M5.2 周期独立 Newton 与检查点筛选、历史回归和预锁定种子验证。"""

from argparse import ArgumentParser
from itertools import chain
import json
import os
from pathlib import Path

if __name__ == '__main__':
    os.environ['SUPERMIMIC_ROBOT'] = 'roban'

import numpy as np
import torch

from ..kinematics import RobotKinematics
from ..motion import CONTROL_DT
from ..roban import JOINT_NAMES
from .m48_evaluation import load_archive, save_archive
from .m50_evaluation import DEFAULT_SEEDS, HOLDOUT_SEEDS, PERTURB_SEEDS, TRACE_FIELDS, trial_metrics
from .m50_reference import MANIFEST, REFERENCE_ROOT, REPO, load_clip, read_manifest
from .m51_evaluation import evaluate_cases, publish_periodic, summarize_foot
from .m51_metrics import contact_metrics, read_guide_heading
from .m52_audit import frozen_task, trace_posture
from .m52_orientation import counterfactual, trace_states
from .m52_task import GROUPS, OUTPUT, SOURCE, TRANSFER


NEW_SEEDS = (1301, 1302, 1303, 1304, 1305, 1306)
SELECT = DEFAULT_SEEDS+PERTURB_SEEDS
ALIASES = ('best_contact_stability', 'best_multi_reference')


def measure_posture(trace, row, mdp, *, diagnose=True):
    posture = trace_posture(trace, mdp.motion, mdp.kin)
    result = dict(posture=posture)
    if diagnose and row['seed'] == DEFAULT_SEEDS[0]:
        result['counterfactual'] = counterfactual(*trace_states(trace, mdp.motion), mdp.kin)
    return result


def rank(nominal, perturb=None, *, tilt_first=False):
    def outcome(group, key):
        rates = [row[key]/row['attempts'] for row in group.values()]
        return float(np.mean(rates)), min(rates)

    quality = [row['observed'] for row in nominal.values()]
    posture = [row['posture']['all'] for row in quality]
    contact = [row['stance'] for row in quality]
    average = lambda values: float(np.mean([x if x is not None else np.inf for x in values]))
    load = float(np.mean([row['plan_load_fraction'] for row in contact]))
    angular = average([(row['support']['omega'] or {}).get('rms') for row in contact])
    yaw = average([row['planned_stance']['yaw_rms_deg'] for row in contact])
    sole_rotation = average([row['all']['so3_rms_deg'] for row in contact])
    foot = average([row['foot_position_rmse_m'] for row in quality])
    slide = average([(row['support']['segment_mean'] or {}).get('distance_mm') for row in contact])
    accepted_contact = (load >= .97 and angular <= .20 and yaw <= 5. and sole_rotation <= 6. and foot <= .010)
    spatial = (-average([row['torso_tilt_rms_deg'] for row in posture]),
               -average([np.mean(row['hand_world_m_rms']) for row in posture]))
    tilt = -average([row['tilt_rms_deg'] for row in posture])
    root = (-average([row['root_xy_rmse_m'] for row in quality]),
            -average([row['body']['root_z_rmse_m'] for row in quality]),
            -average([row['body']['root_heading_rmse_deg'] for row in quality]))
    local = (-average([row['body']['arm_rmse_rad'] for row in quality]),
             -average([row['upper_smoothness']['after_startup']['jerk_rms_rad_s3'] for row in quality]))
    contact_score = (load, -angular, -yaw, -sole_rotation, -slide, -foot)
    successful = outcome(nominal, 'complete') + (outcome(perturb, 'complete') if perturb else ())
    survives = outcome(nominal, 'survives') + (outcome(perturb, 'survives') if perturb else ())
    # 两种周期候选均优先完整步态和足部阈值，再分别保留承重与倾斜表现最好的权重。
    quality_score = (tilt, *spatial, *contact_score) if tilt_first else (*contact_score, tilt, *spatial)
    return successful + survives + (accepted_contact, *quality_score, *root, *local)


def periodic(args):
    out = args.out/'evaluation'
    path = out/'results.json'
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads(path.read_text()) if path.is_file() else dict(milestones={})
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    group, iteration = saved['args']['m52_group'], saved['iteration']
    assert args.out.name == GROUPS[group] and saved['transfer_report']['strategy'] == TRANSFER
    frozen = args.out/'checkpoints'/f'iter_{iteration:04d}.pt'
    if iteration not in (500, 1000, 1500, 2000) or not frozen.samefile(args.checkpoint):
        raise ValueError('周期评估需要该组正式里程碑的冻结检查点')
    key = str(iteration)
    if key not in info['milestones']:
        rows = {}
        evaluate_cases(args, frozen, rows, measure=measure_posture)
        info['milestones'][key] = dict(iteration=iteration, checkpoint=str(frozen), rows=rows,
                                     nominal=summarize_foot(rows, DEFAULT_SEEDS))
    ranks = dict(zip(ALIASES, (lambda row: rank(row['nominal']),
                              lambda row: rank(row['nominal'], tilt_first=True)), strict=True))
    publish_periodic(out, info, ranks)
    for clip, row in info['milestones'][key]['nominal'].items():
        print(f'{clip} {row["complete"]}/{row["attempts"]}，'
              f'Tilt RMSE {row["observed"]["posture"]["all"]["tilt_rms_deg"]:.2f}°', flush=True)


def paired_windows(args, cases, arrays, split, seeds):
    table = 'seen' if split == 'train' else 'heldout'
    saved = torch.load(SOURCE, map_location='cpu', mmap=True, weights_only=False)
    kin = RobotKinematics('cpu')
    end = {}
    for item in read_manifest(args.library)[split]:
        clip = item['clip_id']
        records = {key: value[table][clip] for key, value in cases.items()}
        motion = load_clip(item, kin, 'cpu', args.reference_root, split)
        mdp = frozen_task(kin, motion, saved)
        root = args.reference_root if split == 'train' else REPO
        guide = read_guide_heading(root/item['file_path'])
        end[clip] = {}
        for seed in map(str, seeds):
            rows = [bundle[seed] for bundle in records.values()]
            steps = min(row['steps'] for row in rows)
            end[clip][seed] = steps*CONTROL_DT
            for row in rows:
                row.pop('common', None)
                if steps == row['steps']:
                    continue
                data = {key: arrays[row['prefix']+key] for key in (*TRACE_FIELDS, 'foot_angular_velocity')}
                trace = {key: value[:steps+int(len(value) == len(data['frame']))] for key, value in data.items()}
                # 缩短窗口只用于误差对照，原完整率、终止和事件判据保留。
                events = [dict(event) if (event['actual_touchdown_s'] or 0.) <= steps*CONTROL_DT+1e-9
                          else dict(event, actual_touchdown_s=None) for event in row['footsteps']]
                sample = dict(steps=steps, footsteps=events)
                if steps:
                    r, _ = trace_states(trace, motion)
                    f = trace['frame'][1:]
                    foot = trace['foot'][1:]-motion.foot_p[f].numpy()
                    root_xy = r[0][1:, :2]-motion.root_p[f, :2]
                    clamp = trace['clamp_mask']
                    sample.update(foot_position_rmse_m=float(np.square(foot).sum(-1).mean()**.5),
                                  root_xy_rmse_m=float(root_xy.square().sum(-1).mean().sqrt()),
                                  per_joint_clamp=dict(zip(JOINT_NAMES, clamp.mean(0).tolist())),
                                  clamp_ratio=float(clamp.mean()),
                                  contact_agreement=float((trace['contact'][1:] == trace['planned_contact'][1:]).mean()))
                    trial_metrics(trace, sample, mdp)
                    sample['stance'] = contact_metrics(trace, sample, mdp, guide)
                else:
                    sample.update(foot_position_rmse_m=None, root_xy_rmse_m=None, stance={})
                sample.update(measure_posture(trace, row, mdp, diagnose=False))
                row['common'] = {key: value for key, value in sample.items() if key not in ('steps', 'footsteps')}
    return end


def compare_summary(rows, seeds):
    return summarize_foot({clip: {seed: row | row.get('common', {}) for seed, row in entries.items()}
                           for clip, entries in rows.items()}, seeds)


def final(args):
    archive = args.out/'cuda_evaluation.npz'
    info, arrays = load_archive(archive)
    cases = info['cases']
    resets = {}
    for case in cases.values():
        for table in ('seen', 'heldout'):
            for clip, rows in case.get(table, {}).items():
                for seed, row in rows.items():
                    initial = arrays[row['prefix']+'state'][0]
                    np.testing.assert_allclose(initial, resets.setdefault((clip, int(seed)), initial), atol=1e-7, rtol=0.)
    by_group = {}
    for group, folder in GROUPS.items():
        directory = args.out/folder
        if (directory/'train.exitcode').read_text().strip() != '0':
            raise ValueError(f'M5.2-{group.upper()} 正式训练尚未完成')
        prior = json.loads((directory/'evaluation/results.json').read_text())
        iterations = sorted({prior[key] for key in ALIASES} | {2000})
        by_group[group] = {f'{group}{it}': directory/'checkpoints'/f'iter_{it:04d}.pt' for it in iterations}
    sources = {'m51': SOURCE} | dict(chain.from_iterable(group.items() for group in by_group.values()))
    twins = dict(arrays=arrays, resets=resets, measure=measure_posture)
    try:
        for name, source in sources.items():
            case = cases.setdefault(name, dict(checkpoint=str(source), seen={}))
            if case['checkpoint'] != str(source):
                raise ValueError('候选的冻结权重路径发生改变')
            evaluate_cases(args, source, case['seen'], seeds=SELECT, prefix=name, **twins)
            save_archive(archive, info, arrays)
        info['selection_windows_s'] = paired_windows(args, cases, arrays, 'train', SELECT)
        for case in cases.values():
            case['nominal'] = summarize_foot(case['seen'], DEFAULT_SEEDS)
            case['perturb'] = summarize_foot(case['seen'], PERTURB_SEEDS)

        def selection(name):
            rows = cases[name]['seen']
            return rank(compare_summary(rows, DEFAULT_SEEDS), compare_summary(rows, PERTURB_SEEDS))

        winners = {group: max(options, key=selection) for group, options in by_group.items()}
        if info.setdefault('winners', winners) != winners:
            raise ValueError('留出结果不参与重新选模')
        final_names = ('m51', *winners.values())
        info['selection_order'] = sorted(final_names, key=selection, reverse=True)
        for name in final_names:
            case = cases[name]
            evaluate_cases(args, sources[name], case['seen'], seeds=HOLDOUT_SEEDS, prefix=name, **twins)
            evaluate_cases(args, sources[name], case.setdefault('heldout', {}), split='heldout',
                           seeds=SELECT+NEW_SEEDS, prefix=name, **twins)
        finalists = {name: cases[name] for name in final_names}
        info['heldout_windows_s'] = {
            'seen': paired_windows(args, finalists, arrays, 'train', HOLDOUT_SEEDS),
            'heldout': paired_windows(args, finalists, arrays, 'heldout', SELECT+NEW_SEEDS)}
        for case in finalists.values():
            case['historical_reset'] = summarize_foot(case['seen'], HOLDOUT_SEEDS)
            case['historical_unseen_nominal'] = summarize_foot(case['heldout'], DEFAULT_SEEDS)
            case['historical_unseen_perturb'] = summarize_foot(case['heldout'], PERTURB_SEEDS)
            case['new_unseen_perturb'] = summarize_foot(case['heldout'], NEW_SEEDS)
        info['complete'] = True
    finally:
        save_archive(archive, info, arrays)
    print('三组选定权重', info['winners'], '预锁定新种子', NEW_SEEDS, flush=True)


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
