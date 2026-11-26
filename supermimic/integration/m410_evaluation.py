"""配对复评 M4.10 的冻结权重及历史对照。"""

from argparse import ArgumentParser, Namespace
import json
import os
from pathlib import Path

if __name__ == '__main__':
    os.environ['SUPERMIMIC_ROBOT'] = 'roban'

import numpy as np
import torch

from .m410_reward import GROUPS
from .m48_evaluation import (FIELDS, INVALID_TERMINATIONS, SEEDS, comparison, evaluate,
                            load_archive, movement_metrics, save_archive)
from ..motion import CONTROL_DT


ROOT = Path('outputs/m4_10')
HISTORY = Path('outputs/m4_9/cuda_evaluation.npz')
HISTORICAL_CASES = {'m43_best': 'm43_best', 'm48c_best': 'm48c_best', 'm49a_9000': 'a_9000',
                    'm49c_8500': 'c_8500', 'm49c_9500': 'c_9500'}
EXTRA_FIELDS = ('foot_R', 'foot_velocity')


def catalogue():
    checkpoints = {}
    for group, name in GROUPS.items():
        folder = ROOT / name
        summary = json.loads((folder / 'evaluation/results.json').read_text())
        best, whole = summary['best_iteration'], summary.get('best_whole_body_iteration')
        # 同一轮的步态与全身最佳只复评一次。
        rounds = (5000, best, 10000) if whole is None else (5000, best, whole, 10000)
        for i in dict.fromkeys(rounds):
            path = folder / 'checkpoints' / f'iter_{i}.pt'
            checkpoints[f'{group}_{i}'] = (str(path), i)
    return checkpoints


def include_history(path):
    if path.exists():
        return
    saved, arrays = load_archive(HISTORY)
    result = {key: value for key, value in saved.items() if key not in ('cases', 'comparison')}
    result['cases'] = {}
    subset = {key: arrays[key] for key in ('joint_limits', 'effort')}
    for name, key in HISTORICAL_CASES.items():
        case = dict(saved['cases'][key], source_archive=str(HISTORY))
        for phase in ('full', 'probe'):
            rows = {}
            for seed, row in case[phase].items():
                old = row['prefix']
                prefix = f'{name}_{phase}_s{seed}_'
                rows[seed] = dict(row, prefix=prefix)
                subset.update({prefix+k.removeprefix(old): v for k, v in arrays.items() if k.startswith(old)})
            case[phase] = rows
        result['cases'][name] = case
    save_archive(path, result, subset)


def mean_metrics(rows):
    values = [v for v in rows if v is not None]
    if not values:
        return None
    first = values[0]
    if isinstance(first, dict):
        return {key: mean_metrics([row.get(key) for row in values]) for key in first}
    if isinstance(first, list):
        return [mean_metrics(group) for group in zip(*values)]
    return float(np.mean(values))


def archive_trace(data, row):
    prefix = row['prefix']
    keys = (*FIELDS, *EXTRA_FIELDS, 'hand_fore_aft')
    return {key: data[prefix+key] for key in keys if prefix+key in data}


def landing_errors(row, trace, motion):
    targets = motion.at_time(motion.touchdown_time)['foot_p'].numpy()
    result = []
    for step, (event, side) in enumerate(zip(row['footsteps'], motion.touchdown_side.tolist())):
        measured = event['actual_touchdown_s']
        if measured is not None:
            xy = trace['foot'][round(measured / CONTROL_DT), side, :2]
            result.append(float(np.linalg.norm(xy - targets[step, side, :2])))
    return result


def summarize(path, reference):
    from ..kinematics import RobotKinematics
    from ..networks import Critic
    from .m410_preflight import ankle_readback, bptt, replay_case, select
    from .m410_quality import whole_body_metrics
    from .m43_wm_audit import make_policy, make_wm, task_for
    from .m49_evaluation import posture_metrics

    info, data = load_archive(path)
    seeds = set(map(str, SEEDS))
    if set(info['cases']) != set(catalogue()) | set(HISTORICAL_CASES) or any(
            set(case.get(phase, {})) != seeds for case in info['cases'].values() for phase in ('full', 'probe')):
        raise ValueError('闭环与启动探针齐备后才能计算最终统计')
    c_summary = json.loads((ROOT / GROUPS['c'] / 'evaluation/results.json').read_text())
    c_best = c_summary['best_iteration']
    ck = torch.load(info['cases'][f'c_{c_best}']['checkpoint'], map_location='cpu', mmap=True, weights_only=False)
    kin = RobotKinematics('cpu')
    task = task_for(ck, kin, reference, 'cpu')
    _, motion, _, mdp = task
    if motion.validation != ck['data_contract']:
        raise ValueError('复算使用的 F2 与训练数据契约不同')
    joints = motion.q.numpy()
    data['reference_joint_position'] = joints
    result = comparison(info, data)
    start = round(.20 / CONTROL_DT)
    for split, summary in result.items():
        shared = {int(seed): round(end / CONTROL_DT) for seed, end in summary['common_end_s'].items()}
        baseline = set(summary['cases']['m48c_best']['gait']['full'])
        for name, entry in summary['cases'].items():
            case = info['cases'][name]
            full = case['full']
            valid = entry['gait']['valid_seeds']
            traces = {seed: archive_trace(data, full[str(seed)]) for seed in valid}
            # 单条试次量化保留原有效终点，配对窗口逐种子按最早终止截取。
            rows = {}
            scores = []
            all_landing_errors = []
            for seed in valid:
                trial = full[str(seed)]
                rows[seed] = whole_body_metrics(traces[seed], mdp)
                errors = landing_errors(trial, traces[seed], motion)
                trial['landing_xy_error_m'] = errors
                trial['whole_body'] = rows[seed]
                all_landing_errors.extend(errors)
                # M4.3 按归档中的实际下发目标计分，其起步方式与 M4.10 不同。
                scores.append(replay_case(task, data, trial, split == 'perturb',
                                          verify_target=case['startup_steps'] == mdp.startup_steps))
            entry['body_observed'] = mean_metrics(list(rows.values()))
            entry['gait']['landing_xy_rmse_m'] = (float(np.sqrt(np.mean(np.square(all_landing_errors))))
                                                if all_landing_errors else None)
            entry['gait']['right_ankle_clamp_ratio'] = mean_metrics(
                [full[str(seed)]['per_joint_clamp']['leg_r5_joint'] for seed in valid])
            entry['common_body'] = mean_metrics([whole_body_metrics(traces[seed], mdp, start=start, end=end)
                                                 for seed, end in shared.items() if end >= start])
            entry['steady'].update(posture_metrics([(traces[seed], end) for seed, end in shared.items()], joints, start))
            good = entry['gait']['full']
            windows = [(traces[seed], full[str(seed)]['steps']) for seed in good]
            steady = movement_metrics(windows, data['joint_limits'], start)
            steady.update(posture_metrics(windows, joints, start))
            entry['successful_steady'] = dict(seeds=good, metrics=steady)
            entry['successful_body'] = mean_metrics([whole_body_metrics(traces[seed], mdp, start=start) for seed in good])
            paired = [seed for seed in good if seed in baseline]
            baseline_rows = info['cases']['m48c_best']['full']
            entry['paired_successful_steady'] = dict(seeds=paired,
                candidate=movement_metrics([(traces[seed], full[str(seed)]['steps']) for seed in paired],
                                           data['joint_limits'], start),
                baseline=movement_metrics([(archive_trace(data, baseline_rows[str(seed)]), baseline_rows[str(seed)]['steps'])
                                           for seed in paired], data['joint_limits'], start))
            if scores:
                combined = {key: torch.cat([row['rewards'][key] for row in scores])
                            for key in scores[0]['rewards']}
                entry['replay_reward'] = {key: value.mean().item() for key, value in combined.items()}
                entry['ankle'] = ankle_readback(mdp, scores)
    info['comparison'] = result

    # 在组合组自身的真实起点复核 Root 与 Huber 方向，避免拿预检的旧网络推断。
    info['trained_bptt'] = {}
    for name in dict.fromkeys(('c_5000', f'c_{c_best}', 'c_10000')):
        case = info['cases'][name]
        ck = torch.load(case['checkpoint'], map_location='cpu', mmap=True, weights_only=False)
        task = task_for(ck, kin, reference, 'cpu')
        args, _, goals, mdp = task
        pi, wm = make_policy(ck, mdp, goals, 'cpu'), make_wm(ck, 'cpu')
        critic = Critic(mdp.critic_dim + goals.spec.dim).eval().requires_grad_(False)
        critic.load_state_dict(ck['critic'])
        entries = [replay_case(task, data, row, int(seed) >= 1100) for seed, row in case['full'].items()
                   if row['steps'] and row['termination'] not in INVALID_TERMINATIONS]
        audit = {}
        for phase in ('all', 'startup', 'single_support', 'touchdown'):
            batch = select(entries, args.horizon, phase, 24 if phase == 'all' else 16)
            if len(batch[0]):
                audit[phase] = bptt(task, batch, (pi, wm, critic))
        info['trained_bptt'][name] = audit
        print(name, '已复算 BPTT', {phase: entry['starts'] for phase, entry in audit.items()}, flush=True)
    save_archive(path, info, data)


def main():
    checkpoints = catalogue()
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--m4_reference', required=True, type=Path)
    parser.add_argument('--out', type=Path, default=ROOT / 'cuda_evaluation.npz')
    parser.add_argument('--cases', nargs='+', choices=checkpoints, default=list(checkpoints))
    parser.add_argument('--phase', choices=('all', 'full', 'probe'), default='all')
    parser.add_argument('--summarize', action='store_true', help='从已归档的真实轨迹复算姿态、控制和梯度')
    cli = parser.parse_args()
    torch.set_num_threads(2)
    if cli.summarize:
        summarize(cli.out, cli.m4_reference)
        return
    include_history(cli.out)
    args = Namespace(m4_reference=cli.m4_reference, device='cuda:0', visualizer=None)
    for phase in (('full', 'probe') if cli.phase == 'all' else (cli.phase,)):
        evaluate(args, cli.out, cli.cases, phase, checkpoints=checkpoints, extra_fields=EXTRA_FIELDS)


if __name__ == '__main__':
    main()
