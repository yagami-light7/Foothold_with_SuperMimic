"""基于同条件物理评估推荐checkpoint；与“已学会全部任务”的资格判定分开。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def assess_retention(summary: dict[str, Any], baseline: dict[str, Any], phase: dict[str, Any]) -> dict[str, Any]:
    """先守住存活与支撑质量，再允许有限回退并比较相对XY；不读取奖励分数。"""
    cfg = phase['checkpoint_selection']
    old = {tuple(p['case']): p['metrics'] for p in baseline['points']}
    current = {tuple(p['case']): p['metrics'] for p in summary['points']}
    cases = [tuple(c) for c in phase['required_cases']]
    checks, survivals, ratios, successes = {}, [], [], []
    for case in cases:
        label = '/'.join(f'{x:g}' for x in case)
        if case not in old or case not in current:
            checks[label + '/evaluated'] = False
            continue
        before, after = old[case], current[case]
        survival = after['survival_fraction']
        error, reference_error = after['late_xy_error_m'], before['late_xy_error_m']
        finite = all(math.isfinite(x) for x in (survival, error, reference_error,
                                                *after['half_cycle_success_left_right']))
        checks[label + '/finite'] = finite
        if not finite:
            continue
        checks[label + '/survival'] = survival >= cfg['minimum_survival']
        tolerance = max(cfg['absolute_xy_tolerance_m'], reference_error * cfg['max_xy_regression_fraction'])
        checks[label + '/xy_retained'] = error <= reference_error + tolerance
        checks[label + '/success_retained'] = all(
            a >= b - cfg['max_success_drop']
            for a, b in zip(after['half_cycle_success_left_right'], before['half_cycle_success_left_right']))
        for key, maximum in (('both_air_fraction', .02), ('stance_hop_fraction', .02),
                             ('contact_foot_slip_speed_m_s', .10)):
            value = after.get(key, float('inf'))
            checks[label + '/' + key] = math.isfinite(value) and value <= maximum
        # 首步已有能力也要保留；缺失测量不会被当成0误差或成功。
        for side in ('left', 'right'):
            initial = before.get('first_step', {}).get(side, {})
            final = after.get('first_step', {}).get(side, {})
            old_xy, new_xy = initial.get('late_xy_error_m'), final.get('late_xy_error_m')
            old_rate, new_rate = initial.get('success_rate'), final.get('success_rate')
            measured = (initial.get('count', 0) >= 4 and final.get('count', 0) >= 4
                        and final.get('xy_measured_count', 0) >= 4
                        and all(isinstance(x, (int, float)) and math.isfinite(x)
                                for x in (old_xy, new_xy, old_rate, new_rate)))
            checks[label + '/' + side + '_first_measured'] = measured
            if measured:
                first_tolerance = max(cfg['absolute_xy_tolerance_m'], old_xy * cfg['max_xy_regression_fraction'])
                checks[label + '/' + side + '_first_xy'] = new_xy <= old_xy + first_tolerance
                checks[label + '/' + side + '_first_success'] = new_rate >= old_rate - cfg['max_success_drop']
        survivals.append(survival)
        ratios.append(error / max(reference_error, .003))
        successes.extend(after['half_cycle_success_left_right'])
    complete = len(survivals) == len(cases)
    # 非有限或缺失输入不生成看似有效的数值分数；JSON中保持可审查。
    return dict(eligible=complete and all(checks.values()), checks=checks,
                minimum_survival=min(survivals) if complete else None,
                relative_xy_score=sum(ratios) / len(ratios) if complete else None,
                mean_success=sum(successes) / len(successes) if complete else None)


def update_retention_selection(supervisor, phase, summary):
    """只保存推荐路径，保留原模型与任务元数据位置；训练仍从最新权重继续。"""
    if not phase.get('checkpoint_selection'):
        return
    from scripts.train_4060 import write_json
    state = supervisor.state
    baseline = state.get('baseline_evaluation')
    if baseline is None:
        raise RuntimeError('checkpoint筛选需要先完成同条件基线评估')
    assessment = assess_retention(summary, baseline, phase)
    best = state.get('retention_best')
    selected = False
    if assessment['eligible']:
        source = Path(summary['checkpoint'])
        if not source.is_file():
            raise FileNotFoundError(source)
        previous = best['assessment'] if best else None
        selected = (previous is None
                    or assessment['minimum_survival'] > previous['minimum_survival'] + 1e-8
                    or (assessment['minimum_survival'] >= previous['minimum_survival'] - 1e-8
                        and assessment['relative_xy_score'] <= previous['relative_xy_score']
                        - phase['checkpoint_selection']['minimum_score_improvement']))
        if selected:
            best = dict(checkpoint=str(source.resolve()), updates=summary['updates'], phase=phase['name'],
                        assessment=assessment, evaluation=summary,
                        qualification='retention_only_not_full_task_success')
            state['retention_best'] = best
    state['retention_last_assessment'] = dict(checkpoint=summary['checkpoint'],
                                             updates=summary['updates'], **assessment)
    write_json(supervisor.directory / 'best_checkpoint.json', dict(
        best=best, latest_evaluation=state['retention_last_assessment'],
        note='推荐路径基于已完成的短评估；不是全部任务达标证明。原路径保留env_cfg/agent_cfg/manifest，训练权重不自动回滚。'))
    supervisor.log({'Retention/candidate_accepted': int(assessment['eligible']),
                    'Retention/best_updated': int(selected)}, state['updates'])
    print(f"[模型筛选] retained={assessment['eligible']} | best_updated={selected} | "
          f"recommended={best['checkpoint'] if best else '暂无合格候选'}", flush=True)
    supervisor.persist()
