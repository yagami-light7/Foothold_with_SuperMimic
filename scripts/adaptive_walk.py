"""按实际评估晋级的12小时专项课程；复用原监督器的进程、预算和SwanLab。"""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

from configs.walk_training import adaptive_gate, clean_profile, coverage_cells, group_counts


def select_phase(plan, state):
    """阶段保存在state里；不能从累计更新数推断晋级。"""
    index = int(state.get("adaptive_phase_index", 0))
    if not 0 <= index < len(plan["phases"]):
        raise ValueError("恢复的adaptive_phase_index超出计划")
    return index, plan["phases"][index]


def final_checkpoint_qualified(plan, state):
    """历史上通过过不代表最新权重也通过；保留已通过模型的独立路径。"""
    final = state.get("adaptive_qualified", {}).get(plan["phases"][-1]["name"], {})
    return bool(final.get("checkpoint") and final["checkpoint"] == state.get("checkpoint"))


def record_evaluation(plan, state, passed, checkpoint, full=False):
    """失败清零连续通过次数；满足最少更新和连续通过条件后才晋级。"""
    index, phase = select_phase(plan, state)
    state["adaptive_pass_streak"] = state.get("adaptive_pass_streak", 0) + 1 if passed else 0
    if not passed:
        state["adaptive_full_passed"] = False
    elif full:
        state["adaptive_full_passed"] = True
    phase_updates = state["updates"] - state.get("adaptive_phase_start_updates", 0)
    qualified = (passed and phase_updates >= phase["minimum_updates"]
                 and state["adaptive_pass_streak"] >= phase["passes_required"]
                 and state.get("adaptive_full_passed", False))
    if passed:
        state.setdefault("adaptive_passed_checkpoints", {})[phase["name"]] = str(checkpoint)
    if qualified:
        state.setdefault("adaptive_qualified", {})[phase["name"]] = {
            "checkpoint": str(checkpoint), "updates": state["updates"]}
        if index + 1 < len(plan["phases"]):
            state["adaptive_phase_index"] = index + 1
            state["adaptive_phase_start_updates"] = state["updates"]
            state["adaptive_pass_streak"] = 0
            state["adaptive_full_passed"] = False
            return True
    return False


def preview_adaptive(supervisor):
    """预览只输出计划；不生成profile文件、不创建目录、不连接SwanLab。"""
    args, state = supervisor.args, supervisor.state
    plan = args.adaptive_plan_config
    evaluations_enabled = plan.get('evaluation', {}).get('enabled', True)
    first_index, _ = select_phase(plan, state)
    plans = []
    for index, phase in enumerate(plan["phases"]):
        if index < first_index:
            continue
        target = supervisor.directory / "training" / f"{state['chunk'] + len(plans):05d}_{phase['name']}"
        profile = supervisor.directory / "profiles" / f"train_{phase['name']}.json"
        command = supervisor.train_command("straight", int(args.num_envs), args.chunk_iterations, target,
                                           learning_rate=phase["learning_rate"])
        command += ["--task-profile", str(profile)]
        if state.get("checkpoint"):
            command += ["--resume", state["checkpoint"]]
        plans.append({"phase": phase["name"], "conditional_on_previous_gates": index > first_index,
                      "checkpoint_selection": phase.get('checkpoint_selection') if evaluations_enabled else None,
                      "coverage_cells": coverage_cells(phase, int(args.num_envs)),
                      "minimum_updates": phase["minimum_updates"], "passes_required": phase["passes_required"],
                      "max_updates_this_process": args.chunk_iterations, "profile": phase["profile"],
                      "groups": [{"name": g["name"], "count": n} for g, n in zip(
                          phase["profile"]["command_groups"], group_counts(phase["profile"]["command_groups"], int(args.num_envs)))],
                      "command": command})
    return {"dry_run": True, "output_dir": str(supervisor.directory), "max_hours": args.max_hours,
            "evaluation": plan.get('evaluation', {}), "eval_envs": args.eval_envs, "eval_steps": args.eval_steps,
            "swanlab_detail": getattr(args,'swanlab_detail','full'),
            "remaining_budget_hours": max(0., args.max_hours - state.get("wall_budget_seconds_used", 0) / 3600),
            "mode": args.mode, "resume_swanlab_id": state.get("swanlab_id"),
            "advancement": ("minimum updates AND consecutive evaluation passes; failures hold phase"
                            if evaluations_enabled else "disabled: single-phase training only, no qualification"),
            "plans": plans}


def _case_name(case):
    length, left, right = case
    return f"step{round(length * 1000):03d}_L{round(left * 1000):04d}_R{round(right * 1000):04d}"


def _evaluate(supervisor, phase, checkpoint, chunk, tag="", force_full=False):
    from scripts.train_4060 import ROOT, write_json
    args, state = supervisor.args, supervisor.state
    plan = args.adaptive_plan_config
    settings = plan.get('evaluation', {})
    if not settings.get('enabled', True):
        raise RuntimeError('本轮evaluation.enabled=false，禁止启动评估子进程')
    # 每阶段首次评估就扫完整网格，随后低频扫；晋级前必须至少一次完整通过。
    full = (force_full or phase["name"] not in state.get("adaptive_evaluations", {})
            or state["chunk"] % plan["full_eval_every_chunks"] == 0)
    required = {tuple(c) for c in phase["required_cases"]}
    if full:
        required.update(tuple(c) for c in phase.get("full_cases", []))
    # 最大角点始终记录；桥接阶段不要求它通过，否则无法先学会较容易的组合。
    cases = sorted(required | ({(.40, 2., 2.)} if settings.get('monitor_max_corner', True) else set()))
    prefix = f"{chunk:05d}_{phase['name']}" + (f"_{tag}" if tag else "")
    profiles = supervisor.directory / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    base_profile = clean_profile()
    if phase.get("robust", False):
        base_profile.update(copy.deepcopy(phase["profile"]))
        base_profile["command_groups"] = None
        base_profile["command_resample_half_cycles"] = 0
    # 本阶段评估使用同一奖励版本；旧checkpoint作为新奖励起点时也显式覆盖。
    if phase["profile"].get("reward_cfg"):
        base_profile["reward_cfg"] = copy.deepcopy(phase["profile"]["reward_cfg"])
    # 新课程的评估也要覆盖两种起步方式；旧schema=2实验快照未声明时仍按左脚复现。
    base_profile["initial_swing_side"] = phase["profile"].get("initial_swing_side", "left")
    points, all_passed = [], True

    def evaluate(case, suffix="", seconds=None, warmup=0):
        length, left, right = case
        label = _case_name(case) + suffix
        destination = supervisor.directory / "evaluations" / f"{prefix}_{label}.json"
        profile = copy.deepcopy(base_profile)
        profile["command_warmup_half_cycles"] = warmup
        if seconds is not None:
            profile["episode_length_s"] = seconds
        profile_file = profiles / f"eval_{prefix}_{label}.json"
        write_json(profile_file, profile)
        steps = args.eval_steps if seconds is None else round(seconds / .02)
        command = [str(args.isaac_python), str(ROOT / "scripts/preflight_signals.py"),
                   "--stage", "straight", "--mode", "policy", "--checkpoint", str(checkpoint),
                   "--num_envs", str(args.eval_envs), "--steps", str(steps),
                   "--seed", str(args.seed if settings.get('fixed_seed', False)
                                 else args.seed + state.get("adaptive_eval_counter", 0) % 3),
                   "--output", str(destination), "--task-profile", str(profile_file),
                   "--step_length_range", str(length), str(length),
                   "--left-gait-frequency", str(left), "--right-gait-frequency", str(right), "--viz", "none"]
        if settings.get('fast', False):
            command += ['--fast']
        code, _, _ = supervisor.run_child(command, destination.with_suffix(".log"), timeout=900)
        if code:
            if supervisor.stopping or time.monotonic() >= supervisor.deadline:
                return None
            raise RuntimeError(f"专项评估失败，见{destination.with_suffix('.log')}")
        result = json.loads(destination.read_text())
        passed, checks = adaptive_gate(result)
        if phase.get('require_first_step', False):
            for side in ('left', 'right'):
                first = result.get('first_step', {}).get(side, {})
                checks[f'first_{side}_samples_ge_4'] = first.get('count', 0) >= 4
                checks[f'first_{side}_success_ge_80pct'] = (first.get('success_rate') or 0.) >= .8
                xy = first.get('late_xy_error_m')
                checks[f'first_{side}_xy_le_3cm'] = xy is not None and xy <= .03
            passed = all(checks.values())
        write_json(destination.with_suffix(".gate.json"), {"passed": passed, "checks": checks,
            "checkpoint": str(checkpoint), "phase": phase["name"], "full": full})
        values = {f"Eval/adaptive/{label}/{k}": v for k, v in result.items()
                  if isinstance(v, (int, float)) and math.isfinite(v)}
        for side, value in zip(("left", "right"), result["half_cycle_success_left_right"]):
            values[f"Eval/adaptive/{label}/success_{side}"] = value
        for start, metrics in result.get("by_initial_swing", {}).items():
            for side, value in zip(("left", "right"), metrics["half_cycle_success_left_right"]):
                values[f"Eval/adaptive/{label}/{start}_start_success_{side}"] = value
        values[f"Eval/adaptive/{label}/passed"] = int(passed)
        supervisor.log(values, state["updates"])
        print(f"[EVAL {phase['name']}/{label}] survival={result['survival_fraction']:.1%} | "
              f"L/R={result['half_cycle_success_left_right']} | xy={result['late_xy_error_m']:.3f}m | "
              f"terminated={result['terminated']:.0f} | truncated={result['truncated']:.0f} | passed={passed}", flush=True)
        points.append({"label": label, "case": list(case), "passed": passed, "checks": checks,
                       "result": str(destination),
                       "required": case in required or suffix != '',
                       "metrics": {k: result[k] for k in ('survival_fraction','half_cycle_success_left_right',
                           'late_xy_error_m','both_air_fraction','stance_hop_fraction',
                           'contact_foot_slip_speed_m_s','by_initial_swing','first_step',
                           'post_start_late_xy_error_m') if k in result}})
        if result.get('first_step'):
            print(f"[首步 {label}] {result['first_step']}", flush=True)
        return passed

    for case in cases:
        if supervisor.stopping or time.monotonic() >= supervisor.deadline:
            return None
        passed = evaluate(case)
        if passed is None:
            return None
        if case in required:
            all_passed &= passed
    # 最后阶段必须包含一次真正60秒episode，以及逐渐加速后的60秒行走。
    # 每次晋级评估都检查长episode，不让20秒结果冒充长期稳定。
    if phase.get("robust", False):
        for suffix, warmup in (("_continuous60s", 0), ("_warmup60s", 4)):
            if supervisor.stopping or time.monotonic() >= supervisor.deadline:
                return None
            passed = evaluate((.4, 2., 2.), suffix, 60., warmup)
            if passed is None:
                return None
            all_passed &= passed
    state["adaptive_eval_counter"] = state.get("adaptive_eval_counter", 0) + 1
    summary = {"phase": phase["name"], "passed": all_passed, "full": full,
               "checkpoint": str(checkpoint), "updates": state["updates"], "points": points}
    required_metrics = [p['metrics'] for p in points if p['required']]
    if required_metrics:
        condensed = {
            'Eval/survival_min':min(m['survival_fraction'] for m in required_metrics),
            'Eval/left_success_min':min(m['half_cycle_success_left_right'][0] for m in required_metrics),
            'Eval/right_success_min':min(m['half_cycle_success_left_right'][1] for m in required_metrics),
            'Eval/late_xy_max_m':max(m['late_xy_error_m'] for m in required_metrics)}
        corner = next((p['metrics'] for p in points if p['case'] == [.4, 2., 2.]), None)
        if corner is not None:
            condensed['Eval/corner_survival_fraction'] = corner['survival_fraction']
            condensed['Eval/corner_xy_error_m'] = corner['late_xy_error_m']
        for source,destination in (('both_air_fraction','both_air_max_fraction'),
                                   ('stance_hop_fraction','stance_hop_max_fraction'),
                                   ('contact_foot_slip_speed_m_s','stance_slip_max_m_s')):
            values=[m[source] for m in required_metrics if source in m]
            if values:
                condensed['Eval/'+destination]=max(values)
        for side in ('left','right'):
            scores=[min(m['by_initial_swing'][side]['half_cycle_success_left_right'])
                    for m in required_metrics if side in m.get('by_initial_swing',{})]
            if scores:
                condensed[f'Eval/{side}_start_success_min']=min(scores)
            first = [m['first_step'][side] for m in required_metrics if side in m.get('first_step', {})]
            rates = [m['success_rate'] for m in first if m.get('success_rate') is not None]
            errors = [m['late_xy_error_m'] for m in first if m.get('late_xy_error_m') is not None]
            if rates:
                condensed[f'Eval/first_{side}_success_min'] = min(rates)
            if errors:
                condensed[f'Eval/first_{side}_xy_max_m'] = max(errors)
        supervisor.log(condensed,state['updates'])
        summary['dashboard']=condensed
    write_json(supervisor.directory / "evaluations" / f"{prefix}.gate.json", summary)
    return summary


def pilot_iterations(requested, updates_left, seconds_left, seconds_per_update):
    """给启动/收尾留35秒，并按最近实测速度缩小末尾训练段；不是额外训练时间。"""
    return min(requested, updates_left, max(0, int((seconds_left - 35) / max(seconds_per_update, .1))))


def write_pilot_comparison(supervisor, final):
    from scripts.train_4060 import write_json
    baseline = supervisor.state.get('baseline_evaluation')
    if baseline is None:
        return
    before = {tuple(p['case']): p for p in baseline['points']}
    rows = []
    for point in final['points']:
        old = before.get(tuple(point['case']))
        if old:
            rows.append(dict(case=point['case'], before=old['metrics'], after=point['metrics'],
                             late_xy_change_m=point['metrics']['late_xy_error_m'] - old['metrics']['late_xy_error_m']))
    write_json(supervisor.directory / 'pilot_comparison.json', dict(
        baseline_checkpoint=baseline['checkpoint'], final_checkpoint=final['checkpoint'],
        updates=supervisor.state['updates'], points=rows,
        note='同seed/同命令下的物理趋势检查；奖励与episode定义以本轮profile为准，不能直接用跨实验总分排名。'))


def run_adaptive(supervisor):
    from scripts.train_4060 import write_json, format_training_progress
    args, state = supervisor.args, supervisor.state
    plan = args.adaptive_plan_config
    state.setdefault("adaptive_phase_index", 0)
    state.setdefault("adaptive_phase_start_updates", 0)
    settings = plan.get('evaluation', {})
    evaluations_enabled = settings.get('enabled', True)
    state['evaluation_enabled'] = evaluations_enabled
    reserve = settings.get('reserve_seconds', 0) if evaluations_enabled else 0
    # 纯训练模式在启动、周期、末尾均不调用eval，也不占用末尾预留时间。
    if evaluations_enabled and settings.get('baseline', False) and not state.get('baseline_evaluation') and not supervisor.stopping:
        _, phase = select_phase(plan, state)
        state.setdefault('baseline_checkpoint', state['checkpoint'])
        state['status'] = 'baseline_evaluating'
        supervisor.persist()
        baseline = _evaluate(supervisor, phase, state['baseline_checkpoint'], state['chunk'],
                             tag='baseline', force_full=True)
        if baseline is None:
            state['status'] = 'stopped' if supervisor.stopping else 'budget_exhausted'
            supervisor.persist()
            return
        state['baseline_evaluation'] = baseline
        supervisor.persist()
    # 新方案按存活/能力保持推荐模型，旧课程不启用；基线也参与，防止所有新候选都退化。
    _, initial_phase = select_phase(plan, state)
    if (evaluations_enabled and initial_phase.get('checkpoint_selection')
            and not state.get('retention_best') and state.get('baseline_evaluation')):
        from scripts.retention_selection import update_retention_selection
        update_retention_selection(supervisor, initial_phase, state['baseline_evaluation'])
    while not supervisor.stopping and time.monotonic() < supervisor.deadline and state["updates"] < args.max_iterations:
        index, phase = select_phase(plan, state)
        cells = coverage_cells(phase, state['num_envs'])  # 在启动GPU前拒绝环境配额为0的区间。
        chunk = state["chunk"]
        directory = supervisor.directory / "training" / f"{chunk:05d}_{phase['name']}"
        profile = supervisor.directory / "profiles" / f"train_{phase['name']}.json"
        profile.parent.mkdir(parents=True, exist_ok=True)
        write_json(profile, phase["profile"])
        iterations = min(args.chunk_iterations, args.max_iterations - state["updates"])
        training_seconds = supervisor.deadline - reserve - time.monotonic()
        if reserve:
            iterations = pilot_iterations(iterations, args.max_iterations - state['updates'], training_seconds,
                                          state.get('adaptive_seconds_per_update', 5.))
            if iterations < 1:
                break
        command = supervisor.train_command("straight", state["num_envs"], iterations, directory,
                                           learning_rate=phase["learning_rate"])
        command += ["--task-profile", str(profile)]
        if state.get("checkpoint"):
            command += ["--resume", state["checkpoint"]]
        state.update(status="training", active_training_dir=str(directory), active_training_stage="straight", chunk=chunk + 1)
        supervisor.persist()
        qualification = (f"pass_streak={state.get('adaptive_pass_streak', 0)}/{phase['passes_required']}"
                         if evaluations_enabled else 'eval=关闭，无自动资格判定')
        print(f"[训练阶段 {index + 1}/{len(plan['phases'])}] {phase['name']} | "
              f"phase_updates={state['updates'] - state['adaptive_phase_start_updates']} | {qualification}", flush=True)
        if cells:
            print(f"[采样覆盖] {len(cells)}格，每rollout每格至少"
                  f"{min(c['environments'] for c in cells) * args.rollout_steps}条transition | "
                  f"eval={'开启' if evaluations_enabled else '关闭'}", flush=True)
        chunk_updates = 0

        def on_row(row):
            nonlocal chunk_updates
            chunk_updates += 1
            state["updates"] += 1
            state["samples"] += state["num_envs"] * args.rollout_steps
            values = {f"straight/{k}": v for k, v in row["metrics"].items()}
            if cells:
                from scripts.rollout_coverage import record_rollout_coverage
                values.update(record_rollout_coverage(supervisor, phase, row['metrics']))
            remaining = max(0., supervisor.deadline - time.monotonic())
            values.update({"Progress/environment_samples": state["samples"],
                           "Progress/command_phase": index + 1,
                           "Progress/phase_updates": state["updates"] - state["adaptive_phase_start_updates"],
                           "Progress/remaining_wall_hours": remaining / 3600})
            supervisor.log(values, state["updates"])
            if chunk_updates == 1 or state["updates"] % args.print_every == 0 or chunk_updates == iterations:
                print(format_training_progress(phase["name"], state["updates"], args.max_iterations, row["metrics"])
                      + f" | remaining={remaining / 3600:.2f}h", flush=True)
            supervisor.persist()

        started = time.monotonic()
        code, rows, _ = supervisor.run_child(command, directory.with_suffix(".log"), directory / "metrics.jsonl", on_row,
                                            timeout=training_seconds if reserve else None)
        if rows:
            state['adaptive_seconds_per_update'] = (time.monotonic() - started) / len(rows)
            state['ppo_seconds'] = state.get('ppo_seconds', 0) + sum(
                r['metrics'].get('Perf/collection_time', 0) + r['metrics'].get('Perf/learning_time', 0) for r in rows)
        checkpoint = directory / ("model_final.pt" if code == 0 else "model_interrupted.pt")
        if checkpoint.is_file():
            state["checkpoint"] = str(checkpoint)
            state["checkpoint_stage"] = "straight"
            state.pop("active_training_dir", None)
            state.pop("active_training_stage", None)
        supervisor.persist()
        reached_training_deadline = bool(reserve and time.monotonic() >= supervisor.deadline - reserve)
        if code and not (reached_training_deadline and checkpoint.is_file() and rows):
            if supervisor.stopping or time.monotonic() >= supervisor.deadline:
                break
            raise RuntimeError(f"专项训练失败，见{directory.with_suffix('.log')}")
        if not rows or not checkpoint.is_file():
            raise RuntimeError("专项训练未产生metrics或checkpoint")
        if supervisor.sdk is not None:
            supervisor.sdk.config.update({f"adaptive_{phase['name']}": {
                "profile": phase["profile"], "ppo": json.loads((directory / "agent_cfg.json").read_text()),
                "source_sha256": json.loads((directory / "manifest.json").read_text())["source_sha256"]}})
        if supervisor.stopping or time.monotonic() >= supervisor.deadline:
            break
        if not evaluations_enabled:
            continue
        state["status"] = "evaluating"
        supervisor.persist()
        summary = _evaluate(supervisor, phase, checkpoint, chunk)
        if summary is None:
            break
        write_pilot_comparison(supervisor, summary)
        if phase.get('checkpoint_selection'):
            from scripts.retention_selection import update_retention_selection
            update_retention_selection(supervisor, phase, summary)
        state.setdefault("adaptive_evaluations", {})[phase["name"]] = summary
        advanced = record_evaluation(plan, state, summary["passed"], checkpoint, summary["full"])
        supervisor.log({"Curriculum/phase": index + 1,
                        "Curriculum/evaluation_passed": int(summary["passed"]),
                        "Curriculum/full_grid": int(summary["full"]),
                        "Curriculum/promoted": int(advanced),
                        "Curriculum/latest_checkpoint_final_qualified": int(final_checkpoint_qualified(plan, state)),
                        "Curriculum/all_stages_ever_qualified": int(all(
                            p["name"] in state.get("adaptive_qualified", {}) for p in plan["phases"]))}, state["updates"])
        print(f"[课程评估] passed={summary['passed']} | 连续通过={state.get('adaptive_pass_streak', 0)} | "
              + ("晋级下一阶段" if advanced else "保持当前阶段"), flush=True)
        supervisor.persist()
    state["status"] = "stopped" if supervisor.stopping else "budget_exhausted"
    state["all_stages_ever_qualified"] = evaluations_enabled and all(
        p["name"] in state.get("adaptive_qualified", {}) for p in plan["phases"])
    state["all_adaptive_gates_passed"] = state["all_stages_ever_qualified"] and final_checkpoint_qualified(plan, state)
    supervisor.persist()
