"""沿用 M4.8 的配对 CUDA 口径，复评 M4.9 的等预算、末轮和 Best。"""

from argparse import ArgumentParser, Namespace
import json
import os
from pathlib import Path

if __name__ == '__main__':
    os.environ['SUPERMIMIC_ROBOT'] = 'roban'

import numpy as np
import torch

from .m48_evaluation import FIELDS, SEEDS, comparison, evaluate, load_archive, movement_metrics, save_archive
from ..motion import CONTROL_DT
from ..roban import ELBOW_JOINT_NAMES, JOINT_NAMES


ROOT = Path('outputs/m4_9')
GROUPS = {'a': 'a_long_baseline', 'b': 'b_original_prior', 'c': 'c_soft_prior'}
HISTORICAL = Path('outputs/m4_8/cuda_evaluation.npz')
HISTORICAL_CASES = {'m43_best': 'm43_best', 'm48c_best': 'c_best', 'm48c_last': 'c_last'}


def catalogue():
    checkpoints = {}
    for group, name in GROUPS.items():
        folder = ROOT / name
        summary = json.loads((folder / 'evaluation/results.json').read_text())
        for i in (5000, int(summary['best_iteration']), 10000):
            path = folder / 'checkpoints' / f'iter_{i}.pt'
            checkpoints[f'{group}_{i}'] = (str(path), i)
        if group == 'c':
            # 在后半程名义全部成功的里程碑中选双肘较差一侧近限位比例最低的候选。
            acceptable = [r for r in summary['milestones'].values() if r['iteration'] >= 5000
                          and r['aggregate']['valid_for_best'] and r['aggregate']['complete_tracking_rate'] == 1.
                          and r['aggregate']['elbows']]
            if acceptable:
                posture = min(acceptable, key=lambda r: max(v['near_limit_ratio']
                                                           for v in r['aggregate']['elbows'].values()))
                i = posture['iteration']
                path = folder / 'checkpoints' / f'iter_{i}.pt'
                checkpoints.setdefault(f'{group}_{i}', (str(path), i))
    return checkpoints


def include_history(path):
    if path.exists():
        return
    saved, arrays = load_archive(HISTORICAL)
    manifest = {key: value for key, value in saved.items() if key not in ('cases', 'comparison')}
    manifest['cases'] = {name: dict(saved['cases'][key], source_archive=str(HISTORICAL))
                         for name, key in HISTORICAL_CASES.items()}
    prefixes = tuple(row['prefix'] for case in manifest['cases'].values()
                     for phase in ('full', 'probe') for row in case[phase].values())
    subset = {key: value for key, value in arrays.items()
              if key in ('joint_limits', 'effort') or key.startswith(prefixes)}
    save_archive(path, manifest, subset)


def posture_metrics(windows, reference, start):
    if not windows or min(end for _, end in windows) < start + 4:
        return {}
    n = len(JOINT_NAMES)
    errors, speeds = [], []
    for row, end in windows:
        frames = row['frame'][start:end + 1]
        state = row['state'][start:end + 1].astype(np.float64)
        errors.append(state[:, 13:13 + n] - reference[frames])
        speeds.append(state[:, 13 + n:])
    error, speed = np.concatenate(errors), np.concatenate(speeds)
    arms = [JOINT_NAMES.index(name) for name in JOINT_NAMES if name.startswith('zarm_')]
    waist = JOINT_NAMES.index('waist_yaw_joint')
    elbows = [JOINT_NAMES.index(name) for name in ELBOW_JOINT_NAMES]
    # 逐条轨迹差分，避免把试次首尾拼接成加速度峰值。
    arm_acc = np.concatenate([np.diff(x[:, arms], axis=0) / CONTROL_DT for x in speeds])
    arm_jerk = np.concatenate([np.diff(x[:, arms], n=2, axis=0) / CONTROL_DT**2 for x in speeds])
    return dict(arm_rmse_rad=float(np.sqrt(np.square(error[:, arms]).mean())),
                waist_rmse_rad=float(np.sqrt(np.square(error[:, waist]).mean())),
                elbow_velocity_rms_rad_s=np.sqrt(np.square(speed[:, elbows]).mean(0)).tolist(),
                arm_velocity_rms=float(np.sqrt(np.square(speed[:, arms]).mean())),
                arm_acceleration_rms=float(np.sqrt(np.square(arm_acc).mean())),
                arm_acceleration_p95=float(np.quantile(np.abs(arm_acc), .95)),
                arm_acceleration_p99=float(np.quantile(np.abs(arm_acc), .99)),
                arm_acceleration_max=float(np.abs(arm_acc).max()),
                arm_jerk_rms=float(np.sqrt(np.square(arm_jerk).mean())))


def summarize(path, reference):
    from .m4_reference import F2MotionReference
    from ..kinematics import RobotKinematics

    info, data = load_archive(path)
    required = set(catalogue()) | set(HISTORICAL_CASES)
    seeds = set(map(str, SEEDS))
    if set(info['cases']) != required or any(set(v.get(phase, {})) != seeds
                                            for v in info['cases'].values() for phase in ('full', 'probe')):
        raise ValueError('闭环与启动探针齐备后才能计算最终对照')
    ref = F2MotionReference(reference, RobotKinematics('cpu')).q.numpy()
    data['reference_joint_position'] = ref
    combined = comparison(info, data)
    start = round(.20 / CONTROL_DT)

    def windows(name, endpoints):
        source = info['cases'][name]['full']
        fields = (*FIELDS, 'hand_fore_aft')
        return [({key: data[source[str(seed)]['prefix'] + key] for key in fields}, end)
                for seed, end in endpoints.items()]

    for split in combined.values():
        shared = {seed: round(end / CONTROL_DT) for seed, end in split['common_end_s'].items()}
        baseline = set(split['cases']['m48c_best']['gait']['full'])
        for name, row in split['cases'].items():
            row['steady'].update(posture_metrics(windows(name, shared), ref, start))
            successful = row['gait']['full']
            full = {seed: info['cases'][name]['full'][str(seed)]['steps'] for seed in successful}
            picked = windows(name, full)
            # 成功组使用完整持续段，并保存与历史控制基线通过同一种子的子集。
            stats = movement_metrics(picked, data['joint_limits'], start)
            stats.update(posture_metrics(picked, ref, start))
            row['successful_steady'] = dict(seeds=successful, metrics=stats)
            paired = {seed: end for seed, end in full.items() if seed in baseline}
            row['paired_successful_steady'] = dict(seeds=list(paired),
                candidate=movement_metrics(windows(name, paired), data['joint_limits'], start),
                baseline=movement_metrics(windows('m48c_best', paired), data['joint_limits'], start))
    info['comparison'] = combined
    save_archive(path, info, data)


def main():
    checkpoints = catalogue()
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--m4_reference', required=True, type=Path)
    parser.add_argument('--out', type=Path, default=ROOT / 'cuda_evaluation.npz')
    parser.add_argument('--cases', nargs='+', choices=checkpoints, default=list(checkpoints))
    parser.add_argument('--phase', choices=('all', 'full', 'probe'), default='all')
    parser.add_argument('--summarize', action='store_true', help='复算共同窗口及成功完整持续段的指标')
    cli = parser.parse_args()
    torch.set_num_threads(2)
    if cli.summarize:
        summarize(cli.out, cli.m4_reference)
        return
    include_history(cli.out)
    args = Namespace(m4_reference=cli.m4_reference, device='cuda:0', visualizer=None)
    for phase in (('full', 'probe') if cli.phase == 'all' else (cli.phase,)):
        evaluate(args, cli.out, cli.cases, phase, checkpoints=checkpoints)


if __name__ == '__main__':
    main()
