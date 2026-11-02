"""按PPO更新数切换步长/节奏分布；只依赖标准库，供SwanLab监督器使用。"""

from __future__ import annotations

import json
import math
from pathlib import Path


def load_command_schedule(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if (data.get("schema_version") != 1 or data.get("length_unit") != "m"
            or data.get("frequency_unit") != "Hz"):
        raise ValueError("课程要求schema_version=1、length_unit=m、frequency_unit=Hz")
    phases = data.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("课程phases必须是非空列表")
    result = []
    names = set()
    for phase in phases:
        name, updates = phase.get("name"), phase.get("updates")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("课程名称必须非空且不重复")
        if isinstance(updates, bool) or not isinstance(updates, int) or updates < 1:
            raise ValueError("课程updates必须是正整数，单位是PPO更新次数")
        normalized = {"name": name, "updates": updates}
        for field in ("step_length_range", "frequency_range"):
            bounds = phase.get(field)
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise ValueError(f"{name}.{field}必须包含下限和上限")
            if any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(x) for x in bounds):
                raise ValueError(f"{name}.{field}必须是有限数")
            low, high = map(float, bounds)
            if low < 0 or low > high or (field == "frequency_range" and low == 0):
                raise ValueError(f"{name}.{field}范围不合法")
            normalized[field] = [low, high]
        names.add(name)
        result.append(normalized)
    return result


def command_phase_at(phases: list[dict], updates: int, tail: str) -> dict:
    """使用本次实验累计更新数选阶段；hold-last让末阶段持续到总时间预算。"""
    start = 0
    for index, phase in enumerate(phases):
        end = start + phase["updates"]
        last = index == len(phases) - 1
        if updates < end or (last and tail == "hold-last"):
            return {**phase, "index": index, "start_update": start,
                    "remaining_updates": None if last and tail == "hold-last" else end - updates}
        start = end
    raise ValueError("课程更新次数已完成")


def apply_command_phase(args, phase: dict) -> None:
    """只切换命令分布，保持任务stage=straight及105维网络不变。"""
    args.step_length_range = phase["step_length_range"]
    args.gait_frequency = args.gait_frequency_range = None
    args.left_gait_frequency = args.right_gait_frequency = None
    # 两侧范围相同，但每个episode左右独立采样，延续已有任务的语义。
    args.left_gait_frequency_range = phase["frequency_range"]
    args.right_gait_frequency_range = phase["frequency_range"]
    args.observe_gait_frequency = True


def restore_command_schedule(args, previous: dict) -> None:
    """恢复保存的计划内容，不依赖原JSON文件仍然存在，也不允许静默更换计划。"""
    old = previous.get("command_schedule_config")
    current = getattr(args, "command_schedule_config", None)
    if current is not None and current != old:
        raise ValueError("resume-dir不能改变已有实验的命令课程；请用resume-checkpoint建立新实验")
    args.command_schedule_config = old
    old_tail = previous.get("schedule_tail", "stop")
    if args.schedule_tail is not None and args.schedule_tail != old_tail:
        raise ValueError("resume-dir不能改变schedule-tail；请用resume-checkpoint建立新实验")
    args.schedule_tail = old_tail


def configure_command_schedule(args) -> None:
    """统一新实验与恢复实验的课程预算，避免12000次上限提前结束第四阶段。"""
    if args.schedule_tail is None:
        args.schedule_tail = "stop"
    phases = getattr(args, "command_schedule_config", None)
    if not phases:
        return
    total = sum(phase["updates"] for phase in phases)
    if args.start_stage != "straight":
        raise ValueError("步长/节奏课程要求start-stage=straight")
    if args.schedule_tail == "stop":
        args.max_iterations = total
    elif args.max_iterations <= total:
        raise ValueError("hold-last要求max-iterations大于课程总更新数，以保留补训预算")
    args.observe_gait_frequency = True
