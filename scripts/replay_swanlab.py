"""从本地PPO日志补传SwanLab曲线；不启动仿真、训练或eval。

示例：python3 scripts/replay_swanlab.py --run-dir logs/walk/<实验目录> --dry-run
实际上传请使用安装了SwanLab的.venv-swanlab/bin/python。
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.swanlab_metrics import compact_metrics


def read_metadata(path: Path) -> dict:
    """伴随文件可缺省，但已存在的损坏文件不能静默忽略。"""
    value = json.loads(path.read_text()) if path.is_file() else {}
    if not isinstance(value, dict):
        raise ValueError(f"配置必须是JSON对象：{path}")
    return value


def metric_snapshot(run_dir: Path, fallback_stage: str) -> list[tuple[Path, int, str]]:
    """固定本次读取的字节边界；训练仍在追加时也只补传启动时已有的完整行。"""
    result = []
    for path in sorted((run_dir / "training").glob("*/metrics.jsonl")):
        stage = read_metadata(path.parent / "env_cfg.json").get("curriculum_stage", fallback_stage)
        result.append((path, path.stat().st_size, stage))
    if not result:
        raise ValueError(f"没有找到training/*/metrics.jsonl：{run_dir}")
    return result


def metric_rows(sources: list[tuple[Path, int, str]]) -> Iterator[tuple[int, dict, str]]:
    """按chunk顺序累计本轮Step，不直接采用包含旧checkpoint历史的iteration编号。"""
    step = 0
    for path, size, stage in sources:
        with path.open("rb") as stream:
            line_number = 0
            while stream.tell() < size:
                line = stream.readline(size - stream.tell())
                line_number += 1
                if not line.endswith(b"\n"):
                    break  # 末尾未完成的记录留待下次补传，不制造缺失的数值。
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    metrics = row["metrics"]
                    if not isinstance(metrics, dict):
                        raise ValueError("metrics必须是JSON对象")
                    for key, value in metrics.items():
                        if (not isinstance(key, str) or isinstance(value, bool)
                                or not isinstance(value, (int, float)) or not math.isfinite(value)):
                            raise ValueError(f"指标不是有限标量：{key}")
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"日志损坏：{path}:{line_number}：{exc}") from exc
                step += 1
                yield step, row, stage


def upload_values(metrics: dict, stage: str, detail: str) -> dict:
    # 与监督器共用筛选器和命名，补传后不会多出一套曲线。
    values = {f"{stage}/{key}": value for key, value in metrics.items()}
    return compact_metrics(metrics) if detail == "compact" else values


def make_plan(args: argparse.Namespace) -> tuple[dict, list[tuple[Path, int, str]]]:
    run_dir = args.run_dir.expanduser().resolve()
    launcher = read_metadata(run_dir / "launcher.json")
    state = read_metadata(run_dir / "state.json")
    run_id = None if args.new_run else state.get("swanlab_id")
    if not args.new_run and not run_id:
        raise ValueError("state.json缺少swanlab_id；使用--new-run新建补传实验")
    processed = state.get("updates", 0)
    if isinstance(processed, bool) or not isinstance(processed, int) or processed < 0:
        raise ValueError("state.json的updates必须为非负整数")
    start = args.start_step if args.start_step is not None else (1 if args.new_run else processed + 1)
    if start < 1 or (args.end_step is not None and args.end_step < start):
        raise ValueError("要求start-step>=1，且end-step>=start-step")
    sources = metric_snapshot(run_dir, launcher.get("start_stage", "straight"))
    total = selected = 0
    first = last = None
    keys: set[str] = set()
    # 先完整校验快照，再连接云端，避免解析到半途才发现损坏日志。
    for step, row, stage in metric_rows(sources):
        total = step
        if step < start or (args.end_step is not None and step > args.end_step):
            continue
        values = upload_values(row["metrics"], stage, args.detail)
        if values:
            selected += 1
            first = step if first is None else first
            last = step
            keys.update(values)
    manifests = [read_metadata(path.parent / "manifest.json") for path, _, _ in sources]
    # 新补传任务可正常完成；原实验若被中断，则继续保留aborted语义。
    finished = bool(manifests) and all(item.get("status") == "complete" for item in manifests)
    return dict(
        run_dir=str(run_dir), target="new_run" if args.new_run else "existing_run",
        run_id=run_id, project=args.project or launcher.get("project") or "S54-Locomotion",
        workspace=args.workspace if args.workspace is not None else launcher.get("workspace"),
        name=run_dir.name + "_recovered", detail=args.detail,
        supervisor_updates=processed, local_updates=total, records_to_submit=selected,
        first_step=first, last_step=last, metric_keys=sorted(keys),
        finish_state="success" if args.new_run or finished else "aborted",
        sources=[dict(path=str(path), snapshot_bytes=size) for path, size, _ in sources],
    ), sources


def replay(args: argparse.Namespace, sdk: Any = None) -> dict:
    plan, sources = make_plan(args)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run or not plan["records_to_submit"]:
        return plan
    if sdk is None:
        sdk = importlib.import_module("swanlab")
    os.environ["SWANLAB_DISABLE_GIT"] = "True"
    options = dict(
        project=plan["project"], workspace=plan["workspace"], mode="online", public=False,
        log_dir=str(Path(plan["run_dir"]) / "swanlog_replay"),
        settings=sdk.Settings(interactive=False, probe={
            "hardware": False, "monitor": False, "git": False, "requirements": False}),
    )
    if args.new_run:
        options.update(name=plan["name"], config={"metric_recovery": plan})
    else:
        # must防止run id或账号不匹配时意外创建另一条实验；不重写原训练config。
        options.update(id=plan["run_id"], resume="must")
    run = sdk.init(**options)
    print("SwanLab:", run.url, flush=True)
    submitted = 0
    try:
        for step, row, stage in metric_rows(sources):
            if step < plan["first_step"] or step > plan["last_step"]:
                continue
            values = upload_values(row["metrics"], stage, args.detail)
            if not values:
                continue
            sdk.log(values, step=step)
            submitted += 1
            if submitted == 1 or submitted % args.print_every == 0:
                print(f"[补传] 已提交{submitted}/{plan['records_to_submit']}条，Step={step}", flush=True)
        if submitted != plan["records_to_submit"]:
            raise RuntimeError("本地日志在补传期间被修改，实际记录数与预检不符")
    except BaseException:
        sdk.finish(state="aborted")
        raise
    sdk.finish(state=plan["finish_state"])
    print(f"补传提交及SDK收尾完成：{submitted}条。请在云端核对最后Step；网络失败可从本地补传日志继续sync。",
          flush=True)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="原训练实验根目录")
    parser.add_argument("--start-step", type=int, help="包含此Step；默认从state.json的updates+1开始")
    parser.add_argument("--end-step", type=int, help="包含此Step；默认读到当前日志快照末尾")
    parser.add_argument("--new-run", action="store_true", help="新建完整恢复曲线，默认从Step 1开始")
    parser.add_argument("--project", help="默认沿用launcher.json中的SwanLab项目")
    parser.add_argument("--workspace", help="默认沿用原工作空间")
    parser.add_argument("--detail", choices=("compact", "full"), default="compact")
    parser.add_argument("--print-every", type=int, default=250)
    parser.add_argument("--dry-run", action="store_true", help="只校验和预览，不导入SDK或联网")
    args = parser.parse_args()
    if args.print_every < 1:
        parser.error("print-every必须为正整数")
    try:
        replay(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
