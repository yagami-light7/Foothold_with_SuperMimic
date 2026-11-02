"""GPU-sized, resumable S54 curriculum supervisor with live SwanLab curves.

The supervisor has its own small Python environment. Simulation subprocesses
always use Isaac Lab's Python, so installing a logger cannot change CUDA/Torch.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.gait_cli import (GAIT_FIELDS, add_gait_arguments, frequency_corners,
                            gait_cli_args, validate_gait_arguments)
from scripts.command_curriculum import (apply_command_phase, command_phase_at,
                                       configure_command_schedule, load_command_schedule,
                                       restore_command_schedule)
from configs.walk_training import load_adaptive_plan
STAGES = ("stand", "lift", "step", "walk")


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False))
    temporary.replace(path)


def gpu_snapshot() -> dict[str, Any]:
    fields = "name,memory.total,memory.used,utilization.gpu,power.draw,temperature.gpu"
    result = subprocess.run(["nvidia-smi", "-i", "0", f"--query-gpu={fields}",
                            "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True, timeout=10)
    values = [value.strip() for value in result.stdout.strip().split(",")]
    keys = ("name", "total_mib", "used_mib", "util_percent", "power_w", "temperature_c")
    snapshot: dict[str, Any] = {"name": values[0]}
    for key, value in zip(keys[1:], values[1:]):
        try:
            snapshot[key] = float(value)
        except ValueError:
            pass
    if "total_mib" not in snapshot or "used_mib" not in snapshot:
        raise RuntimeError(f"Cannot read GPU memory: {result.stdout}")
    return snapshot


def summarize_probe(rows: list[dict], hardware: list[dict], returncode: int,
                    num_envs: int, budget_mib: float) -> dict:
    # Exclude the first two updates (graph/kernel warmup).
    steady = rows[2:] or rows[-1:]
    fps = [r["metrics"]["Perf/total_fps"] for r in steady if "Perf/total_fps" in r["metrics"]]
    peak = max((r["used_mib"] for r in hardware), default=float("inf"))
    active = [r for r in hardware if not steady or "wall_time" not in steady[0]
            or r.get("wall_time", 0) >= steady[0]["wall_time"] - 1]
    return {"num_envs": num_envs, "returncode": returncode,
            "eligible": returncode == 0 and bool(fps) and peak <= budget_mib,
            "median_fps": statistics.median(fps) if fps else 0,
            "peak_memory_mib": peak if math.isfinite(peak) else None,
            "mean_gpu_util_percent": statistics.mean(r.get("util_percent", 0) for r in hardware) if hardware else 0,
            "steady_gpu_util_percent": statistics.mean(r.get("util_percent", 0) for r in active) if active else 0,
            "peak_power_w": max((r.get("power_w", 0) for r in hardware), default=0)}


def select_probe(probes: list[dict]) -> dict:
    good = [p for p in probes if p["eligible"]]
    if not good:
        raise RuntimeError("No GPU probe completed within the VRAM budget; inspect benchmark logs")
    # Prefer more real environments while retaining >=90% of best throughput.
    fastest = max(p["median_fps"] for p in good)
    return max((p for p in good if p["median_fps"] >= fastest * 0.9), key=lambda p: p["num_envs"])


def evaluate_gate(stage: str, result: dict) -> tuple[bool, dict[str, bool]]:
    checks = {
        "survival_ge_0.95": result["survival_fraction"] >= 0.95,
        "upright_ge_0.95": result["mean_upright"] >= 0.95,
        "root_height_ge_0.85": result["minimum_valid_root_height"] >= 0.85,
        "lateral_speed_le_0.10": result["mean_abs_lateral_velocity"] <= 0.10,
        "yaw_rate_le_0.15": result["mean_abs_yaw_rate"] <= 0.15,
    }
    if stage == "stand":
        checks["forward_drift_le_0.05"] = abs(result["mean_forward_velocity"]) <= 0.05
    else:
        checks["both_feet_half_cycle_success_ge_0.80"] = min(result["half_cycle_success_left_right"]) >= 0.8
        checks["eight_alternating_half_cycles_ge_0.80"] = result["consecutive_8_success_fraction"] >= 0.8
        checks["late_xy_error_le_0.03"] = result["late_xy_error_m"] <= 0.03
        if stage in ("step", "walk"):
            checks["forward_speed_positive"] = result["mean_forward_velocity"] >= {"step": 0.02, "walk": 0.04}[stage]
        if stage == "straight":
            # 速度沿reset参考方向计算，避免机器人一边转身、一边沿body x前进也被判为直行。
            checks["reference_forward_speed_ge_0.02"] = result["mean_reference_forward_velocity"] >= 0.02
            checks["max_heading_error_le_10deg"] = result["max_abs_heading_error_rad"] <= math.radians(10)
            checks["heading_samples_present"] = result["heading_valid_samples"] > 0
            if "command_nominal_forward_speed_m_s" in result:
                # 步长扩大后，不能仍靠很慢地挪动通过前进门槛。
                checks["forward_speed_ge_half_nominal"] = (result["mean_reference_forward_velocity"]
                                                           >= 0.5 * result["command_nominal_forward_speed_m_s"])
    return all(checks.values()), checks


def format_training_progress(stage: str, updates: int, maximum: int, metrics: dict) -> str:
    parts = [f"[TRAIN {stage}] update={updates}/{maximum}"]
    for key, label, spec in (
        ("Reward/total", "reward/step", ".5f"),
        ("Episode/normalized_return", "normalized_return", ".3f"),
        ("Reward/normalized", "normalized_reward", ".3f"),
        ("Train/mean_reward", "reward/episode", ".2f"),
        ("Train/mean_episode_length", "episode_steps", ".1f"),
        ("Termination/terminated_fraction", "fall/step", ".3%"),
        ("Termination/truncated_fraction", "timeout/step", ".3%"),
        ("Policy/mean_std", "std", ".3f"),
        ("Tracking/heading_error_rad", "heading_rad", ".3f"),
        ("Tracking/yaw_rate_abs", "yaw_rad/s", ".3f"),
        ("Command/step_length_mean_m", "step_m", ".3f"),
        ("Command/left_gait_frequency_mean_hz", "left_hz", ".3f"),
        ("Command/right_gait_frequency_mean_hz", "right_hz", ".3f"),
        ("Perf/total_fps", "samples/s", ".0f"),
    ):
        if key in metrics:
            parts.append(f"{label}={metrics[key]:{spec}}")
    return " | ".join(parts)


class Supervisor:
    def __init__(self, args: argparse.Namespace, directory: Path, state: dict):
        self.args = args
        self.directory = directory
        self.state = state
        self.stages = tuple(state.get("stages", STAGES))
        self.stopping = False
        self.sdk: Any = None
        self.started_monotonic = time.monotonic()
        # 命令课程恢复时继续累计已使用的预算，避免每次resume又额外训练12小时。
        self.previous_budget_seconds = (float(state.get("wall_budget_seconds_used", 0))
                                       if (getattr(args, "command_schedule_config", None)
                                           or getattr(args, "adaptive_plan_config", None)) else 0.)
        self.deadline = self.started_monotonic + max(0., args.max_hours * 3600 - self.previous_budget_seconds)
        self.hardware_tick = int(state.get("hardware_tick", 0))
        self.environment = os.environ.copy()
        # Never leak the logger's isolated site-packages into Isaac Lab.
        self.environment["PYTHONPATH"] = str(ROOT)
        self.environment["PYTHONUNBUFFERED"] = "1"

    def persist(self) -> None:
        if (getattr(self.args, "command_schedule_config", None)
                or getattr(self.args, "adaptive_plan_config", None)):
            self.state["wall_budget_seconds_used"] = (self.previous_budget_seconds
                + max(0., time.monotonic() - self.started_monotonic))
        self.state["hardware_tick"] = self.hardware_tick
        write_json(self.directory / "state.json", self.state)

    def log(self, values: dict, step: int) -> None:
        if self.sdk is not None:
            if getattr(self.args,'swanlab_detail','full') == 'compact':
                from scripts.swanlab_metrics import compact_metrics
                values = compact_metrics(values)
            if values:
                self.sdk.log(values, step=step)

    def initialize_swanlab(self) -> None:
        self.sdk = importlib.import_module("swanlab")
        if self.args.mode == "online":
            # SDK login uses SWANLAB_API_KEY or ~/.swanlab/.netrc; no key is put in argv/logs.
            key_file = Path(os.environ.get("SWANLAB_ROOT", str(Path.home() / ".swanlab"))) / ".netrc"
            if not os.environ.get("SWANLAB_API_KEY") and not key_file.is_file():
                raise RuntimeError("SwanLab login missing. Run .venv-swanlab/bin/swanlab login locally first.")
            if not self.sdk.login():
                raise RuntimeError("SwanLab authentication failed; check local login/network")
        os.environ["SWANLAB_DISABLE_GIT"] = "True"
        run = self.sdk.init(project=self.args.project, workspace=self.args.workspace,
                            name=self.directory.name, mode=self.args.mode,
                            log_dir=str(self.directory / "swanlog"), public=False,
                            id=self.state.get("swanlab_id"), resume="allow",
                            settings=self.sdk.Settings(interactive=False, probe={
                                "hardware": False, "monitor": False, "git": False, "requirements": False}),
                            config={"launcher": json.loads(json.dumps(vars(self.args), default=str)),
                                    "selected_envs": self.state["num_envs"],
                                    "gpu": gpu_snapshot(),
                                    "evaluation": ("disabled: training only, no baseline/periodic/final evaluation"
                                                   if getattr(self.args, "adaptive_plan_config", None)
                                                   and not self.args.adaptive_plan_config.get('evaluation', {}).get('enabled', True) else
                                                   "fixed points + periodic full grid, two passes + full-grid pass; final 60s episodes"
                                                   if getattr(self.args, "adaptive_plan_config", None) else
                                                   "deterministic, 20 simulated seconds, phase-boundary diagnostics; advance by update count"
                                                   if getattr(self.args, "command_schedule_config", None)
                                                   else "deterministic, 20 simulated seconds, two successive passes"),
                                    "curriculum": list(self.stages), "benchmark": self.state.get("benchmark")})
        self.state["swanlab_id"] = run.id
        self.state["swanlab_url"] = run.url if self.args.mode == "online" else None
        self.persist()
        print("SwanLab:", self.state["swanlab_url"] or self.args.mode, flush=True)

    def run_child(self, command: list[str], log: Path, metrics: Path | None = None,
                  on_row: Callable[[dict], None] | None = None,
                  timeout: float | None = None) -> tuple[int, list[dict], list[dict]]:
        started = time.monotonic()
        rows: list[dict] = []
        hardware: list[dict] = []
        offset = 0
        tail = ""
        stop_sent: float | None = None
        log.parent.mkdir(parents=True, exist_ok=True)
        print("Running:", " ".join(command), flush=True)
        print("Detailed log:", log, flush=True)

        def read_metrics() -> None:
            nonlocal offset, tail
            if metrics is None or not metrics.is_file():
                return
            with metrics.open() as stream:
                stream.seek(offset)
                text = tail + stream.read()
                offset = stream.tell()
            lines = text.split("\n")
            tail = lines.pop()
            for line in lines:
                if not line:
                    continue
                row = json.loads(line)
                if not all(math.isfinite(v) for v in row["metrics"].values()):
                    raise RuntimeError("Non-finite PPO metrics; stop and inspect checkpoint/log")
                rows.append(row)
                if on_row is not None:
                    on_row(row)

        with log.open("w") as output, (log.parent / f"{log.stem}_gpu.jsonl").open("w", buffering=1) as gpu_log:
            process = subprocess.Popen(command, cwd=ROOT, env=self.environment, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            self.state["active_child_pid"] = process.pid
            self.persist()
            try:
                while process.poll() is None:
                    now = time.monotonic()
                    if stop_sent is None and (self.stopping or now >= self.deadline or
                                               (timeout is not None and now - started >= timeout)):
                        # SIGINT is handled by train.py, which saves model_interrupted.pt.
                        os.killpg(process.pid, signal.SIGINT)
                        stop_sent = now
                    if stop_sent is not None and now - stop_sent > 60:
                        os.killpg(process.pid, signal.SIGKILL)
                    snapshot = gpu_snapshot()
                    snapshot["seconds"] = now - started
                    snapshot["wall_time"] = time.time()
                    hardware.append(snapshot)
                    gpu_log.write(json.dumps(snapshot) + "\n")
                    self.hardware_tick += 1
                    self.log({"Hardware/" + key: value for key, value in snapshot.items()
                              if isinstance(value, (int, float)) and key not in ("seconds", "wall_time")}, self.hardware_tick)
                    read_metrics()
                    time.sleep(1)
                read_metrics()
                return process.wait(), rows, hardware
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                self.state.pop("active_child_pid", None)
                self.persist()

    def train_command(self, stage: str, envs: int, iterations: int, directory: Path,
                      learning_rate: float | None = None) -> list[str]:
        command = [str(self.args.isaac_python), str(ROOT / "scripts/train.py"),
                "--stage", stage, "--num_envs", str(envs), "--max_iterations", str(iterations),
                "--num_steps_per_env", str(self.args.rollout_steps),
                "--num_mini_batches", str(self.args.mini_batches),
                "--ablation", self.args.ablation,
                "--normalization_prior_rollouts", str(self.args.normalization_prior_rollouts),
                "--seed", str(self.args.seed), "--metrics_jsonl", "--output_dir", str(directory), "--viz", "none"]
        if self.args.step_length_range is not None and stage == "straight":
            command += ["--step_length_range", *(str(value) for value in self.args.step_length_range)]
        if getattr(self.args, "save_interval", None) is not None:
            command += ["--save_interval", str(self.args.save_interval)]
        rate = self.args.learning_rate if learning_rate is None else learning_rate
        if rate is not None:
            command += ["--learning_rate", str(rate)]
        command += gait_cli_args(self.args)
        return command

    def benchmark(self) -> None:
        gpu = gpu_snapshot()
        budget = min(gpu["total_mib"] * self.args.target_vram, gpu["total_mib"] - self.args.reserve_mib)
        probes = []
        for envs in sorted(set(self.args.probe_envs)):
            if self.stopping or time.monotonic() >= self.deadline:
                break
            valid = [p for p in probes if p["eligible"]]
            if len(valid) >= 2:
                previous, latest = valid[-2:]
                slope = max(0, (latest["peak_memory_mib"] - previous["peak_memory_mib"]) /
                            (latest["num_envs"] - previous["num_envs"]))
                predicted = latest["peak_memory_mib"] + slope * (envs - latest["num_envs"])
                if predicted > budget:
                    print(f"Skip {envs} environments: predicted {predicted:.0f} MiB exceeds {budget:.0f} MiB budget", flush=True)
                    break
            directory = self.directory / "benchmark" / f"envs_{envs}"
            command = self.train_command("lift", envs, self.args.probe_iterations, directory)
            code, rows, hardware = self.run_child(command, directory.parent / f"envs_{envs}.log",
                                                  directory / "metrics.jsonl", timeout=self.args.probe_timeout)
            result = summarize_probe(rows, hardware, code, envs, budget)
            probes.append(result)
            write_json(self.directory / "benchmark.json", {"gpu": gpu, "budget_mib": budget, "probes": probes})
            print("Probe:", json.dumps(result), flush=True)
            if code != 0 or (result["peak_memory_mib"] or 0) > budget:
                break
        if self.stopping or time.monotonic() >= self.deadline:
            self.state["status"] = "stopped" if self.stopping else "budget_exhausted"
            self.stopping = True
            self.persist()
            return
        chosen = select_probe(probes)
        self.state["num_envs"] = chosen["num_envs"]
        self.state["benchmark"] = {"gpu": gpu, "budget_mib": budget, "probes": probes, "selected": chosen}
        self.state["status"] = "benchmarked"
        self.persist()
        print("Selected environments:", chosen["num_envs"], flush=True)

    def curriculum(self) -> None:
        if getattr(self.args, "adaptive_plan_config", None):
            from scripts.adaptive_walk import run_adaptive
            run_adaptive(self)
            return
        while (not self.stopping and time.monotonic() < self.deadline and
               self.state["updates"] < self.args.max_iterations and self.state["stage_index"] < len(self.stages)):
            index = self.state["stage_index"]
            stage = self.stages[index]
            phases = getattr(self.args, "command_schedule_config", None)
            phase_count = len(phases) if phases else 0
            command_phase = None
            if phases:
                command_phase = command_phase_at(phases, self.state["updates"], self.args.schedule_tail)
                apply_command_phase(self.args, command_phase)
                if self.state.get("command_phase_index") != command_phase["index"]:
                    # 不同难度的eval不能拼成“连续两次通过”。固定更新计划照常推进。
                    self.state["pass_streak"] = 0
                    self.state.pop("last_stage_qualified", None)
                self.state["command_phase_index"] = command_phase["index"]
                self.state["command_phase_name"] = command_phase["name"]
                self.state["command_phase_updates"] = self.state["updates"] - command_phase["start_update"]
                print(f"[课程 {command_phase['index'] + 1}/{phase_count}] {command_phase['name']} | "
                      f"步长[m]={command_phase['step_length_range']} | "
                      f"左右频率各自采样[Hz]={command_phase['frequency_range']} | "
                      f"本阶段已更新={self.state['command_phase_updates']}", flush=True)
            chunk = self.state["chunk"]
            label = command_phase["name"] if command_phase else stage
            directory = self.directory / "training" / f"{chunk:05d}_{label}"
            iterations = min(self.args.chunk_iterations, self.args.max_iterations - self.state["updates"])
            if command_phase and command_phase["remaining_updates"] is not None:
                # 前三个阶段严格在3000次边界保存并换档；第四阶段hold-last不受3000次截断。
                iterations = min(iterations, command_phase["remaining_updates"])
            command = self.train_command(stage, self.state["num_envs"], iterations, directory)
            if self.state.get("checkpoint"):
                option = "--resume" if self.state.get("checkpoint_stage") == stage else "--warm_start"
                command += [option, self.state["checkpoint"]]
            self.state["status"] = "training"
            self.state["active_training_dir"] = str(directory)
            self.state["active_training_stage"] = stage
            self.state["chunk"] += 1  # Reserve a unique output path before launching, including on crashes.
            self.persist()

            chunk_updates = 0

            def on_row(row: dict) -> None:
                nonlocal chunk_updates
                chunk_updates += 1
                self.state["updates"] += 1
                self.state["samples"] += self.state["num_envs"] * self.args.rollout_steps
                values = {f"{stage}/{key}": value for key, value in row["metrics"].items()}
                values.update({"Progress/environment_samples": self.state["samples"],
                               "Progress/stage": index, "Progress/environments": self.state["num_envs"]})
                remaining_seconds = max(0., self.deadline - time.monotonic())
                values["Progress/remaining_wall_hours"] = remaining_seconds / 3600
                if command_phase:
                    phase_updates = self.state["updates"] - command_phase["start_update"]
                    self.state["command_phase_updates"] = phase_updates
                    values.update({"Progress/command_phase": command_phase["index"] + 1,
                                   "Progress/phase_updates": phase_updates,
                                   "Progress/phase_target_updates": command_phase["updates"]})
                self.log(values, self.state["updates"])
                if (chunk_updates == 1 or self.state["updates"] % self.args.print_every == 0
                        or chunk_updates == iterations):
                    progress = format_training_progress(stage, self.state["updates"],
                                                        self.args.max_iterations, row["metrics"])
                    if command_phase:
                        progress += (f" | phase={command_phase['index'] + 1}/{phase_count}"
                                     f" | phase_updates={self.state['command_phase_updates']}/{command_phase['updates']}")
                    print(progress + f" | remaining={remaining_seconds / 3600:.2f}h", flush=True)
                self.persist()

            code, rows, _ = self.run_child(command, directory.parent / f"{directory.name}.log",
                                           directory / "metrics.jsonl", on_row)
            checkpoint = directory / ("model_final.pt" if code == 0 else "model_interrupted.pt")
            if checkpoint.is_file():
                self.state["checkpoint"] = str(checkpoint)
                self.state["checkpoint_stage"] = stage
                self.state.pop("active_training_dir", None)
                self.state.pop("active_training_stage", None)
            self.persist()
            if code != 0:
                if self.stopping or time.monotonic() >= self.deadline:
                    break
                raise RuntimeError(f"Training failed ({code}); inspect {directory.parent / (directory.name + '.log')}")
            if not rows or not checkpoint.is_file():
                raise RuntimeError("Training returned no metrics/checkpoint")
            if self.sdk is not None:
                self.sdk.config.update({f"stage_{label}": {
                    "env": json.loads((directory / "env_cfg.json").read_text()),
                    "ppo": json.loads((directory / "agent_cfg.json").read_text()),
                    "source_sha256": json.loads((directory / "manifest.json").read_text())["source_sha256"],
                }})
            if self.stopping or time.monotonic() >= self.deadline:
                break
            # chunk保存checkpoint与上传曲线照常执行，只有独立eval按间隔运行。
            interval = getattr(self.args, "eval_every_chunks", 1)
            if self.state["chunk"] % interval and self.state["updates"] < self.args.max_iterations:
                continue
            self.state["status"] = "evaluating"
            self.persist()
            evaluation = self.directory / "evaluations" / f"{chunk:05d}_{label}.json"
            command = [str(self.args.isaac_python), str(ROOT / "scripts/preflight_signals.py"),
                        "--stage", stage, "--mode", "policy", "--checkpoint", str(checkpoint),
                        "--num_envs", str(self.args.eval_envs), "--steps", str(self.args.eval_steps),
                        "--seed", str(self.args.seed),
                        "--ablation", self.args.ablation,
                        "--output", str(evaluation), "--viz", "none"]
            evaluation_base = command.copy()
            command += gait_cli_args(self.args)
            code, _, _ = self.run_child(command, evaluation.with_suffix(".log"), timeout=900)
            if code != 0:
                if self.stopping or time.monotonic() >= self.deadline:
                    break
                raise RuntimeError(f"Evaluation failed: {evaluation.with_suffix('.log')}")
            result = json.loads(evaluation.read_text())
            passed, checks = evaluate_gate(stage, result)
            if self.args.step_length_range is not None and stage == "straight":
                # 随机平均成绩可能掩盖大步长失败；6/9/12cm等固定点必须逐档通过。
                low, high = self.args.step_length_range
                sweep_complete = True
                for length in sorted(set((low, (low + high) / 2, high))):
                    if self.stopping or time.monotonic() >= self.deadline:
                        sweep_complete = False
                        break
                    label = f"step_{round(length * 1000):03d}mm"
                    point_file = evaluation.with_name(f"{evaluation.stem}_{label}.json")
                    point_command = command.copy()
                    point_command[point_command.index("--output") + 1] = str(point_file)
                    point_command += ["--step_length_range", str(length), str(length)]
                    point_code, _, _ = self.run_child(point_command, point_file.with_suffix(".log"), timeout=900)
                    if point_code != 0:
                        if self.stopping or time.monotonic() >= self.deadline:
                            sweep_complete = False
                            break
                        raise RuntimeError(f"固定步长评估失败：{point_file.with_suffix('.log')}")
                    point = json.loads(point_file.read_text())
                    point_passed, point_checks = evaluate_gate(stage, point)
                    checks[f"{label}_all_gates"] = point_passed
                    write_json(point_file.with_suffix(".gate.json"), {"passed": point_passed,
                               "checks": point_checks, "checkpoint": checkpoint})
                    point_metrics = {f"Eval/{stage}/{label}/{key}": value for key, value in point.items()
                                     if isinstance(value, (int, float)) and math.isfinite(value)}
                    for side, value in zip(("left", "right"), point["half_cycle_success_left_right"]):
                        point_metrics[f"Eval/{stage}/{label}/half_cycle_success_{side}"] = value
                    point_metrics[f"Eval/{stage}/{label}/passed"] = int(point_passed)
                    self.log(point_metrics, self.state["updates"])
                    print(f"[EVAL {label}] survival={point['survival_fraction']:.1%} | "
                          f"left/right={point['half_cycle_success_left_right']} | "
                          f"heading={math.degrees(point['max_abs_heading_error_rad']):.1f}deg | "
                          f"passed={point_passed}", flush=True)
                if not sweep_complete:
                    break
                passed = all(checks.values())
            if getattr(self.args, "eval_frequency_corners", False) and stage == "straight":
                # 平均分不能掩盖左右快慢组合失败；角点同时采用本次最大步长。
                for left, right in frequency_corners(result["env_cfg"]):
                    if self.stopping or time.monotonic() >= self.deadline:
                        break
                    label = f"cadence_L{round(left*1000):04d}_R{round(right*1000):04d}mHz"
                    point_file = evaluation.with_name(f"{evaluation.stem}_{label}.json")
                    point_args = argparse.Namespace(**vars(self.args))
                    point_args.left_gait_frequency = point_args.right_gait_frequency = None
                    point_args.left_gait_frequency_range = (left, left)
                    point_args.right_gait_frequency_range = (right, right)
                    point_command = evaluation_base.copy() + gait_cli_args(point_args)
                    point_command[point_command.index("--output") + 1] = str(point_file)
                    if self.args.step_length_range is not None:
                        high = self.args.step_length_range[1]
                        point_command += ["--step_length_range", str(high), str(high)]
                    point_code, _, _ = self.run_child(point_command, point_file.with_suffix(".log"), timeout=900)
                    if point_code != 0:
                        if self.stopping or time.monotonic() >= self.deadline:
                            break
                        raise RuntimeError(f"左右节奏角点评估失败：{point_file.with_suffix('.log')}")
                    point = json.loads(point_file.read_text())
                    point_passed, point_checks = evaluate_gate(stage, point)
                    checks[f"{label}_all_gates"] = point_passed
                    write_json(point_file.with_suffix(".gate.json"),
                               {"passed": point_passed, "checks": point_checks, "checkpoint": checkpoint})
                    point_metrics = {f"Eval/{stage}/{label}/{key}": value for key, value in point.items()
                                     if isinstance(value, (int, float)) and math.isfinite(value)}
                    for side, value in zip(("left", "right"), point["half_cycle_success_left_right"]):
                        point_metrics[f"Eval/{stage}/{label}/half_cycle_success_{side}"] = value
                    point_metrics[f"Eval/{stage}/{label}/passed"] = int(point_passed)
                    self.log(point_metrics, self.state["updates"])
                    print(f"[EVAL {label}] left/right={point['half_cycle_success_left_right']} | passed={point_passed}", flush=True)
                if self.stopping or time.monotonic() >= self.deadline:
                    break
                passed = all(checks.values())
            print(f"[EVAL {stage}] survival={result['survival_fraction']:.1%} | "
                  f"upright={result['mean_upright']:.3f} | "
                  f"min_height={result['minimum_valid_root_height']:.3f}m | "
                  f"forward_speed={result['mean_forward_velocity']:.3f}m/s | "
                  f"late_xy={result['late_xy_error_m']:.3f}m | "
                  f"terminated={result['terminated']:.0f} | truncated={result['truncated']:.0f}", flush=True)
            self.state["pass_streak"] = self.state.get("pass_streak", 0) + 1 if passed else 0
            metrics = {f"Eval/{stage}/{key}": value for key, value in result.items()
                       if isinstance(value, (int, float)) and math.isfinite(value)}
            metrics[f"Eval/{stage}/passed"] = int(passed)
            for side, value in zip(("left", "right"), result["half_cycle_success_left_right"]):
                metrics[f"Eval/{stage}/half_cycle_success_{side}"] = value
            self.log(metrics, self.state["updates"])
            write_json(evaluation.with_suffix(".gate.json"), {"passed": passed, "checks": checks,
                       "streak": self.state["pass_streak"], "checkpoint": checkpoint,
                       "command_phase": command_phase})
            if command_phase:
                self.state.setdefault("command_phase_evaluations", {})[command_phase["name"]] = {
                    "passed": passed, "checkpoint": str(checkpoint), "evaluation": str(evaluation),
                    "updates": self.state["updates"]}
            print(f"Evaluation {stage}: passed={passed}, streak={self.state['pass_streak']}; {checks}", flush=True)
            if passed:
                # Preserve the exact candidate independently of later overwritten latest-state pointers.
                self.state.setdefault("best_passed", {})[stage] = str(checkpoint)
                if command_phase:
                    self.state.setdefault("best_passed_by_command_phase", {})[command_phase["name"]] = str(checkpoint)
            if self.state["pass_streak"] >= 2 and not command_phase:
                if self.args.continue_after_pass and index == len(self.stages) - 1:
                    # 用户要求按12小时预算持续训练；最后阶段通过也保留最佳模型并继续。
                    if self.state["pass_streak"] == 2:
                        print("最后阶段连续通过两次，已记录模型，继续运行至时间预算。", flush=True)
                    self.state["last_stage_qualified"] = True
                else:
                    self.state["stage_index"] += 1
                    self.state["pass_streak"] = 0
                    print("Stage passed twice:", stage, flush=True)
            self.persist()
        self.state["status"] = ("all_gates_passed" if self.state["stage_index"] == len(self.stages)
                                else "stopped" if self.stopping else "budget_exhausted")
        phases = getattr(self.args, "command_schedule_config", None)
        if phases:
            self.state["schedule_minimum_updates_completed"] = self.state["updates"] >= sum(p["updates"] for p in phases)
            if not self.state["schedule_minimum_updates_completed"] and not self.stopping:
                print("总时间预算已到，保存当前模型；课程最低更新目标尚未全部完成。", flush=True)
        self.persist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-python", type=Path, default=Path("/mnt/sage_data/workspace/IsaacLab_3_0/.venv/bin/python"))
    parser.add_argument("--num-envs", default="auto", help="auto or a positive integer; explicit value skips probes")
    parser.add_argument("--probe-envs", type=int, nargs="+", default=[256, 512, 1024, 2048, 3072, 3456, 4096])
    parser.add_argument("--probe-iterations", type=int, default=5)
    parser.add_argument("--probe-timeout", type=float, default=300)
    parser.add_argument("--target-vram", type=float, default=0.85)
    parser.add_argument("--reserve-mib", type=int, default=768)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--mini-batches", type=int, default=4)
    parser.add_argument("--save-interval", "--save_interval", type=int,
                        help="每N次PPO更新保存checkpoint；正整数，未指定时沿用PPO配置")
    parser.add_argument("--max-hours", type=float, default=12)
    parser.add_argument("--max-iterations", type=int, default=20000, help="Total updates across curriculum stages")
    parser.add_argument("--chunk-iterations", type=int, default=100,
                        help="单个训练子进程最多执行的PPO更新次数；是否评估由课程配置控制")
    parser.add_argument("--print-every", type=int, default=10, help="Print a compact training summary every N updates")
    parser.add_argument("--eval-envs", type=int, default=64)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", default="S54-Locomotion")
    parser.add_argument("--workspace")
    parser.add_argument("--mode", choices=("online", "offline"), default="online")
    parser.add_argument('--swanlab-detail',choices=('compact','full'),default='compact',
                        help='compact只上传核心曲线；完整诊断仍写入本地metrics.jsonl')
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-dir", type=Path)
    parser.add_argument("--start-stage", choices=(*STAGES, "straight"), default="stand")
    parser.add_argument("--ablation", choices=("none", "legacy_reward", "legacy_scale", "no_smoothing", "targets_only", "rewards_only"), default="none")
    parser.add_argument("--normalization-prior-rollouts", type=int, default=0)
    parser.add_argument("--step-length-range", type=float, nargs=2, metavar=("MIN_M", "MAX_M"))
    parser.add_argument("--command-schedule", type=Path,
                        help="按PPO更新次数切换步长/左右频率范围的JSON计划，长度单位必须为m")
    parser.add_argument("--adaptive-plan", type=Path,
                        help="按物理评估晋级的混合命令课程JSON；与command-schedule互斥")
    parser.add_argument("--schedule-tail", choices=("stop", "hold-last"), default=None,
                        help="stop完成计划即停；hold-last让最后阶段持续到总时间预算")
    parser.add_argument("--dry-run", action="store_true", help="只校验参数并打印各阶段命令，不创建目录或上传")
    add_gait_arguments(parser)
    parser.add_argument("--eval-frequency-corners", action="store_true",
                        help="额外评估左右节奏范围角点，全部通过才判定新任务通过")
    parser.add_argument("--eval-every-chunks", type=int, default=1,
                        help="每N个训练chunk执行一次完整eval，训练日志与checkpoint仍逐chunk保存")
    parser.add_argument("--learning-rate", type=float, help="显式设置PPO学习率，完整恢复后仍生效")
    parser.add_argument("--continue-after-pass", action="store_true", help="最后阶段通过后继续到时间/更新预算")
    parser.add_argument("--warm-start", type=Path, help="Actor checkpoint, with matching S54 architecture")
    parser.add_argument("--resume-checkpoint", type=Path, help="从同阶段checkpoint完整恢复actor/critic/优化器，建立新实验")
    parser.add_argument("--benchmark-only", action="store_true", help="Probe actual PPO work; no long training/upload")
    args = parser.parse_args()
    args.max_hours_explicit = any(value == "--max-hours" or value.startswith("--max-hours=")
                                  for value in sys.argv[1:])
    args.command_schedule_config = load_command_schedule(args.command_schedule) if args.command_schedule else None
    args.adaptive_plan_config = load_adaptive_plan(args.adaptive_plan) if args.adaptive_plan else None
    if args.adaptive_plan_config and args.command_schedule_config:
        parser.error("adaptive-plan与command-schedule不能同时使用")
    if (args.command_schedule_config or args.adaptive_plan_config) and (args.step_length_range is not None
            or any(getattr(args, name) is not None for name in GAIT_FIELDS[:-1])):
        parser.error("课程已管理命令分布，不能同时传入步长/频率覆盖")
    if args.adaptive_plan_config and (args.start_stage != "straight" or args.warm_start):
        parser.error("adaptive-plan要求start-stage=straight，使用resume-checkpoint完整续训")
    try:
        validate_gait_arguments(args)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.isaac_python.is_file():
        parser.error("Isaac Lab Python does not exist")
    if not (0 < args.target_vram < 1) or args.reserve_mib < 256 or args.max_hours <= 0:
        parser.error("target-vram must be (0,1), reserve-mib >=256, max-hours >0")
    if min(args.max_iterations, args.chunk_iterations, args.rollout_steps, args.mini_batches,
           args.eval_envs, args.probe_iterations, args.probe_timeout, args.print_every, *args.probe_envs) < 1:
        parser.error("iteration counts, environments and timeout must be positive")
    if args.eval_steps < 1000:
        parser.error("eval-steps must be >=1000 (20 seconds, including eight alternating half-cycles)")
    if args.eval_every_chunks < 1:
        parser.error("eval-every-chunks必须为正整数")
    if args.num_envs != "auto" and (not args.num_envs.isdecimal() or int(args.num_envs) < 1):
        parser.error("num-envs must be auto or a positive integer")
    if args.benchmark_only and args.num_envs != "auto":
        args.probe_envs = [int(args.num_envs)]
        args.num_envs = "auto"
    env_counts = args.probe_envs if args.num_envs == "auto" else [int(args.num_envs)]
    if any(n * args.rollout_steps % args.mini_batches for n in env_counts):
        parser.error("num-envs * rollout-steps must be divisible by mini-batches")
    if args.resume_dir and (args.output_dir or args.warm_start or args.resume_checkpoint):
        parser.error("resume-dir cannot be combined with output-dir or an initial checkpoint")
    if args.warm_start and args.resume_checkpoint:
        parser.error("warm-start和resume-checkpoint只能选一个")
    for checkpoint in (args.warm_start, args.resume_checkpoint):
        if checkpoint and not checkpoint.is_file():
            parser.error(f"checkpoint does not exist: {checkpoint}")
    if args.step_length_range is not None:
        low, high = args.step_length_range
        if args.start_stage != "straight" or not (math.isfinite(low) and math.isfinite(high) and 0 <= low <= high):
            parser.error("随机步长要求start-stage=straight且范围有限、0 <= MIN_M <= MAX_M")
    if args.learning_rate is not None and not (0 < args.learning_rate < 0.01):
        parser.error("learning-rate必须在(0,0.01)内")
    if args.normalization_prior_rollouts < 0:
        parser.error("normalization-prior-rollouts must be nonnegative")
    if args.save_interval is not None and args.save_interval < 1:
        parser.error("save-interval必须为正整数")
    return args


def main() -> None:
    args = parse_args()
    if args.resume_dir:
        directory = args.resume_dir.resolve()
        state = json.loads((directory / "state.json").read_text())
        previous = json.loads((directory / "launcher.json").read_text())
        saved_adaptive = previous.get("adaptive_plan_config")
        if args.adaptive_plan_config is not None and args.adaptive_plan_config != saved_adaptive:
            raise ValueError("resume-dir不能改变自适应课程；请用resume-checkpoint建立新实验")
        args.adaptive_plan_config = saved_adaptive
        restore_command_schedule(args, previous)
        if args.command_schedule_config or args.adaptive_plan_config:
            args.start_stage = "straight"
            if args.num_envs != "auto" and int(args.num_envs) != state["num_envs"]:
                raise ValueError("resume-dir不能改变命令课程的环境数量")
            args.num_envs = str(state["num_envs"])
            if not args.max_hours_explicit:
                args.max_hours = previous["max_hours"]
        if (args.step_length_range is not None
                and tuple(args.step_length_range) != tuple(previous.get("step_length_range") or ())):
            raise ValueError("resume-dir不能改变已有实验的步长分布；请用resume-checkpoint建立新实验")
        for name in (*GAIT_FIELDS, "initial_swing_side"):
            current, old = getattr(args, name), previous.get(name)
            if current is not None and current != old:
                raise ValueError(f"resume-dir不能改变已有实验的{name}；请用resume-checkpoint建立新实验")
            setattr(args, name, old)
        args.eval_frequency_corners = args.eval_frequency_corners or previous.get("eval_frequency_corners", False)
        args.eval_every_chunks = previous.get("eval_every_chunks", args.eval_every_chunks)
        # Keep task, sample accounting and cloud identity consistent on resume.
        for key in ("seed", "rollout_steps", "mini_batches", "project", "workspace", "mode"):
            setattr(args, key, previous[key])
        args.ablation = previous.get("ablation", "none")
        args.normalization_prior_rollouts = previous.get("normalization_prior_rollouts", 0)
        # 未显式指定时继承原实验的保存间隔；允许恢复时只调整保存频率。
        if args.save_interval is None:
            args.save_interval = previous.get("save_interval")
        args.step_length_range = previous.get("step_length_range")
        args.learning_rate = args.learning_rate if args.learning_rate is not None else previous.get("learning_rate")
        args.continue_after_pass = args.continue_after_pass or previous.get("continue_after_pass", False)
    else:
        directory = (args.output_dir or ROOT / "logs" / "long_runs" /
                     datetime.now().strftime("%Y-%m-%d_%H-%M-%S_s54_4060")).resolve()
        # 新直行任务独立微调；无需再次执行已经通过的站立、抬脚阶段。
        stages: list[str] = ["straight"] if args.start_stage == "straight" else list(STAGES)
        state = {"stage_index": stages.index(args.start_stage), "stages": stages, "updates": 0, "samples": 0,
                 "chunk": 0, "pass_streak": 0, "status": "initializing"}
        if args.warm_start:
            state["checkpoint"] = str(args.warm_start.resolve())
        if args.resume_checkpoint:
            # 新实验只改变目标分布；继承完整PPO状态，避免无意退回actor-only迁移。
            state["checkpoint"] = str(args.resume_checkpoint.resolve())
            state["checkpoint_stage"] = args.start_stage
    configure_command_schedule(args)
    if args.adaptive_plan_config:
        args.observe_gait_frequency = True
    if args.dry_run:
        if args.num_envs == "auto":
            raise ValueError("dry-run请显式指定num-envs，不执行GPU探测")
        preview = Supervisor(args, directory, state.copy())
        if args.adaptive_plan_config:
            from scripts.adaptive_walk import preview_adaptive
            print(json.dumps(preview_adaptive(preview), ensure_ascii=False, indent=2))
            return
        plans = []
        updates = state["updates"]
        checkpoint = state.get("checkpoint")
        phases = args.command_schedule_config
        while updates < args.max_iterations:
            phase = command_phase_at(phases, updates, args.schedule_tail) if phases else None
            if phase:
                apply_command_phase(args, phase)
            count = min(args.chunk_iterations, args.max_iterations - updates)
            if phase and phase["remaining_updates"] is not None:
                count = min(count, phase["remaining_updates"])
            label = phase["name"] if phase else args.start_stage
            target = directory / "training" / f"{state['chunk'] + len(plans):05d}_{label}"
            command = preview.train_command(args.start_stage, int(args.num_envs), count, target)
            if checkpoint:
                option = "--warm_start" if not plans and args.warm_start else "--resume"
                command += [option, checkpoint]
            plans.append({"phase": label, "phase_minimum_updates": phase["updates"] if phase else count,
                          "start_update": updates, "max_updates_this_process": count, "command": command})
            checkpoint = str(target / "model_final.pt")
            updates += count
        print(json.dumps({"dry_run": True, "output_dir": str(directory), "max_hours": args.max_hours,
                          "remaining_budget_hours": max(0., args.max_hours - state.get("wall_budget_seconds_used", 0) / 3600),
                          "schedule_tail": args.schedule_tail, "mode": args.mode,
                          "resume_swanlab_id": state.get("swanlab_id"), "plans": plans}, ensure_ascii=False, indent=2))
        return
    if not args.resume_dir:
        directory.mkdir(parents=True, exist_ok=False)
    supervisor = Supervisor(args, directory, state)

    def stop(signum, frame) -> None:
        supervisor.stopping = True
        print("Stop requested; saving the training checkpoint...", flush=True)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    # Prevent accidental competing supervisors writing the same resume directory.
    import fcntl
    failure = None
    with (directory / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if state.get("active_child_pid"):
            process_path = Path(f"/proc/{state['active_child_pid']}/cmdline")
            if process_path.exists() and str(ROOT).encode() in process_path.read_bytes():
                raise RuntimeError(f"An earlier simulation is still running (PID {state['active_child_pid']}); stop it before resuming")
            state.pop("active_child_pid", None)
        if state.get("active_training_dir"):
            active = Path(state["active_training_dir"])
            candidates = list(active.glob("model_*.pt"))
            if candidates:
                state["checkpoint"] = str(max(candidates, key=lambda p: p.stat().st_mtime))
                state["checkpoint_stage"] = state["active_training_stage"]
        write_json(directory / "launcher.json", vars(args))
        lock.write(str(os.getpid()))
        lock.flush()
        write_json(directory / "process.json", {"pid": os.getpid(), "started": datetime.now().isoformat()})
        try:
            if "num_envs" not in state:
                if args.num_envs == "auto":
                    supervisor.benchmark()
                else:
                    state["num_envs"] = int(args.num_envs)
                    supervisor.persist()
            if not args.benchmark_only and not supervisor.stopping:
                supervisor.initialize_swanlab()
                supervisor.curriculum()
        except Exception:
            failure = "Training supervisor failed; inspect the local logs."
            state["status"] = "failed"
            supervisor.persist()
            raise
        finally:
            if supervisor.sdk is not None:
                supervisor.sdk.finish(state="crashed" if failure else "aborted" if supervisor.stopping else "success",
                                      error=failure)
    print("Run directory:", directory, flush=True)
    print("Status:", state["status"], "Checkpoint:", state.get("checkpoint"), flush=True)


if __name__ == "__main__":
    main()
