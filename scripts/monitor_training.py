"""只读监督正在进行的训练：定时汇总已有日志，不启动eval、不修改训练参数。

示例：python3 scripts/monitor_training.py --run-dir logs/walk/<实验目录>
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


CHINA = timezone(timedelta(hours=8))
KEYS = (
    'Reward/total', 'Reward/normalized', 'Episode/normalized_return',
    'Episode/normalized_quality', 'Episode/survival_fraction', 'Train/mean_episode_length',
    'Termination/terminated_fraction', 'Termination/truncated_fraction',
    'Startup/left_success', 'Startup/right_success', 'Startup/left_xy_error_m',
    'Startup/right_xy_error_m', 'Tracking/heading_error_rad', 'Tracking/yaw_rate_abs',
    'Loss/value', 'Loss/surrogate', 'Loss/learning_rate', 'Policy/mean_std',
    'Perf/total_fps', 'Perf/collection_time', 'Perf/learning_time',
    'Reward/tracking', 'Reward/swing_phase', 'Reward/stance_slip', 'Reward/stance_hop',
    'Reward/body_motion', 'Reward/posture', 'Reward/action_rate', 'Reward/heading',
)


def read_rows(run_dir: Path):
    """正在写入的最后一行可能不完整；只接受已经写完换行的记录。"""
    rows = []
    for source in sorted((run_dir / 'training').glob('*/metrics.jsonl')):
        for line in source.read_text().splitlines(keepends=True):
            if line.endswith('\n'):
                rows.append(json.loads(line))
    return rows


def summarize(rows):
    """均值按PPO更新统计；首步/episode样本数未导出，不能据此声称事件加权成功率。"""
    result = {}
    for key in KEYS:
        values = [row['metrics'][key] for row in rows if key in row['metrics']]
        if values:
            result[key] = dict(mean=sum(values) / len(values), min=min(values),
                               max=max(values), updates_with_metric=len(values))
    return result


def snapshot(run_dir: Path, previous_row_time: float, origin: float):
    state = json.loads((run_dir / 'state.json').read_text())
    launcher = json.loads((run_dir / 'launcher.json').read_text())
    rows = read_rows(run_dir)
    window = [row for row in rows if row['wall_time'] > previous_row_time]
    now = time.time()
    envs, steps = state['num_envs'], launcher['rollout_steps']
    groups = launcher['adaptive_plan_config']['phases'][0]['profile']['command_groups']
    from configs.walk_training import group_counts
    breakdown = {}
    for group, count in zip(groups, group_counts(groups, envs)):
        name = group['name']
        metrics = {}
        for key in ('normalized_reward', 'reward', 'terminated_fraction'):
            values = [row['metrics'][f'Group/{name}/{key}'] for row in window
                      if f'Group/{name}/{key}' in row['metrics']]
            if values:
                metrics[key] = sum(values) / len(values)
        metrics['environments'] = count
        metrics['terminated_count'] = round(metrics.get('terminated_fraction', 0) * len(window) * count * steps)
        breakdown[name] = metrics
    nonfinite = [dict(iteration=row['iteration'], key=key) for row in window
                 for key, value in row['metrics'].items() if not math.isfinite(value)]
    coverage = state.get('rollout_coverage', {})
    counts = {kind: round(sum(row['metrics'].get(f'Termination/{kind}_fraction', 0)
                             * envs * steps for row in window)) for kind in ('terminated', 'truncated')}
    gpu = {}
    gpu_sources = sorted((run_dir / 'training').glob('*_gpu.jsonl'))
    if gpu_sources:
        complete = [line for line in gpu_sources[-1].read_text().splitlines(keepends=True) if line.endswith('\n')]
        if complete:
            gpu = json.loads(complete[-1])
    latest = rows[-1]['wall_time'] if rows else previous_row_time
    result = dict(
        checked_at=datetime.fromtimestamp(now, CHINA).isoformat(),
        monitoring_minutes=(now - origin) / 60, training_wall_minutes=state.get('wall_budget_seconds_used', 0) / 60,
        status=state['status'], updates=state['updates'], samples=state['samples'],
        new_updates=len(window), last_row_time=latest, log_age_seconds=now - latest,
        checkpoint=state.get('checkpoint'), swanlab_url=state.get('swanlab_url'),
        evaluation_enabled=state.get('evaluation_enabled'),
        evaluation_directory_exists=(run_dir / 'evaluations').exists(),
        coverage={key: value for key, value in coverage.items() if key != 'cells'},
        metrics=summarize(window), groups=breakdown, termination_counts=counts,
        nonfinite_metrics=nonfinite, gpu=gpu,
        note='训练包含随机动作；首步和episode指标为按更新均值，缺原始事件计数。不能替代确定性物理eval。',
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--minutes', type=float, default=30.)
    parser.add_argument('--interval-seconds', type=float, default=300.)
    parser.add_argument('--started-at', help='带时区ISO时间；默认从脚本启动时计时')
    args = parser.parse_args()
    if args.minutes <= 0 or args.interval_seconds <= 0:
        parser.error('minutes和interval-seconds必须为正数')
    if args.started_at:
        started = datetime.fromisoformat(args.started_at)
        if started.tzinfo is None:
            parser.error('started-at必须带时区')
        origin = started.timestamp()
    else:
        origin = time.time()
    # 使用monotonic安排等待，避免时钟校正导致长时间漏检或重复检查。
    origin_monotonic = time.monotonic() - (time.time() - origin)
    duration = args.minutes * 60
    args.output_dir.mkdir(parents=True, exist_ok=False)
    previous = 0.
    schedule = [0.] + [min(i * args.interval_seconds, duration)
                       for i in range(1, math.ceil(duration / args.interval_seconds) + 1)]
    for index, offset in enumerate(schedule):
        # 启动前已经错过的周期不伪造历史快照；本次基线后等待下一个真实检查点。
        if index and origin_monotonic + offset < time.monotonic():
            continue
        while time.monotonic() < origin_monotonic + offset:
            time.sleep(min(10., origin_monotonic + offset - time.monotonic()))
        result = snapshot(args.run_dir, previous, origin)
        previous = result['last_row_time']
        destination = args.output_dir / f'check_{round(offset / 60):02d}m.json'
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        with (args.output_dir / 'checks.jsonl').open('a') as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + '\n')
        metrics = result['metrics']
        print(json.dumps(dict(check=destination.name, checked_at=result['checked_at'],
            monitor_minutes=round(result['monitoring_minutes'], 2), status=result['status'],
            updates=result['updates'], new_updates=result['new_updates'],
            reward=metrics.get('Reward/total'), episode_return=metrics.get('Episode/normalized_return'),
            terminated=result['termination_counts']['terminated'], truncated=result['termination_counts']['truncated'],
            corner=result['groups'].get('corner'), coverage=result['coverage']), ensure_ascii=False), flush=True)
    print('监督结束；没有启动eval、修改训练参数或延长训练。', flush=True)


if __name__ == '__main__':
    # 与其他scripts入口一致，直接运行时也能定位项目内configs包。
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
