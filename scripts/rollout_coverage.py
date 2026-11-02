"""用训练已有的日志统计实际分区transition；不额外运行仿真或eval。"""

from __future__ import annotations

import json
import math

from configs.walk_training import coverage_cells, group_counts


def record_rollout_coverage(supervisor, phase, metrics):
    """每个PPO更新检查15格都有样本，同时保留逐格累计数供复现实验。"""
    from scripts.train_4060 import write_json
    cells = coverage_cells(phase, supervisor.state['num_envs'])
    if not cells:
        return {}
    steps = supervisor.args.rollout_steps
    state = supervisor.state
    summary = state.setdefault('rollout_coverage', dict(updates=0, incomplete_updates=0, cells={}))
    counts = {}
    # 另外核对基础组和固定角点；它们不与15格重复计数。
    groups = phase['profile']['command_groups']
    other_groups = [dict(name=g['name'], step_length_range=g['step_length_range'],
                         frequency_range=g['frequency_range'], environments=n)
                    for g, n in zip(groups, group_counts(groups, state['num_envs']))
                    if g['name'] not in {c['name'] for c in cells}]
    for cell in cells + other_groups:
        name = cell['name']
        key = f'Group/{name}/command_count'
        value = metrics.get(key)
        # 缺日志直接失败，不能把配置里的期望配额伪装成真实覆盖率。
        if value is None or not math.isfinite(value) or not 0 <= value <= cell['environments'] + 1e-4:
            raise RuntimeError(f'rollout覆盖日志缺失或非法：{key}={value}')
        counts[name] = round(value * steps)
        previous = summary['cells'].get(name, {})
        summary['cells'][name] = dict(**cell, transitions=previous.get('transitions', 0) + counts[name])
    complete = all(counts[c['name']] == c['environments'] * steps for c in cells + other_groups)
    summary['updates'] += 1
    summary['incomplete_updates'] += int(not complete)
    summary['last_update'] = state['updates']
    minimum = min(counts[c['name']] for c in cells)
    summary['last_min_cell_transitions'] = minimum
    coverage = sum(counts[c['name']] > 0 for c in cells) / len(cells)
    summary['last_cell_fraction'] = coverage
    # 本地每次更新逐格记录，云端只保留两条聚合曲线。
    with (supervisor.directory / 'coverage.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(update=state['updates'], transitions=counts,
                                     all_commands_in_bounds=complete)) + '\n')
    write_json(supervisor.directory / 'coverage.json', summary)
    if not complete:
        raise RuntimeError('rollout存在区间外命令或区间缺样，详见coverage.jsonl；停止以免静默漏采')
    return {'Coverage/cell_fraction': coverage, 'Coverage/min_cell_transitions': minimum}
