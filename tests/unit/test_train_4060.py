"""CPU checks for resource selection, curriculum gates and the live metric stream."""

import sys
from pathlib import Path

# 支持直接运行测试文件，无需额外设置PYTHONPATH。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import shlex
import subprocess
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from scripts.train_4060 import Supervisor, evaluate_gate, main, select_probe, summarize_probe
from scripts.training_log_writer import TensorboardJsonWriter


class TrainingSupervisorTests(unittest.TestCase):
    @unittest.skipUnless((PROJECT_ROOT / 'scripts/history_train/train_walk_3h.sh').is_file(),
                         '历史入口只在本地归档，不随Git上传；当前入口另有覆盖')
    def test_walk_launcher_forwards_explicit_checkpoint_without_starting_training(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temp:
            # 带空格的自选路径也必须原样传递，不能被脚本默认checkpoint覆盖。
            checkpoint = Path(temp) / 'custom model.pt'
            checkpoint.write_text('dry-run fixture')
            output = Path(temp) / 'new_walk_run'
            result = subprocess.run(['bash', str(root / 'scripts/history_train/train_walk_3h.sh'),
                                     '--checkpoint', str(checkpoint), '--output-dir', str(output),
                                     '--dry-run'], cwd=temp, text=True, capture_output=True, check=True)
            line = next(line for line in result.stdout.splitlines() if line.startswith('运行命令: '))
            command = shlex.split(line.removeprefix('运行命令: '))
            self.assertEqual(command[command.index('--resume-checkpoint') + 1], str(checkpoint))
            self.assertEqual(command[command.index('--output-dir') + 1], str(output))
            self.assertEqual(command[command.index('--chunk-iterations') + 1], '50000')
            self.assertEqual(command[command.index('--mode') + 1], 'online')
            self.assertNotIn('--warm-start', command)
            self.assertFalse(output.exists())

            # 不允许静默忽略自选checkpoint；同一实验恢复与换模型是两种互斥操作。
            conflict = subprocess.run(['bash', str(root / 'scripts/history_train/train_walk_3h.sh'),
                                       '--checkpoint', str(checkpoint), '--resume-dir', temp,
                                       '--dry-run'], cwd=temp, text=True, capture_output=True)
            self.assertEqual(conflict.returncode, 2)
            self.assertIn('resume-dir不能与checkpoint并用', conflict.stderr)
            missing = subprocess.run(['bash', str(root / 'scripts/history_train/train_walk_3h.sh'),
                                      '--checkpoint', str(Path(temp) / 'missing.pt'), '--dry-run'],
                                     cwd=temp, text=True, capture_output=True)
            self.assertEqual(missing.returncode, 2)
            self.assertIn('checkpoint不存在', missing.stderr)

    def test_vram_budget_and_throughput(self):
        def probe(envs, fps, memory):
            rows = [{"metrics": {"Perf/total_fps": value}} for value in (1, 2, fps, fps)]
            return summarize_probe(rows, [{"used_mib": memory}], 0, envs, 6900)
        probes = [probe(1024, 30000, 2200), probe(2048, 45000, 4300),
                  probe(3072, 44000, 6400), probe(4096, 48000, 7800)]
        self.assertEqual(select_probe(probes)["num_envs"], 3072)
        probes[2] = probe(3072, 35000, 6400)
        self.assertEqual(select_probe(probes)["num_envs"], 2048)

    def test_standing_or_falling_cannot_pass_walk(self):
        result = {"survival_fraction": 1.0, "mean_upright": 0.999,
                  "minimum_valid_root_height": 0.96, "mean_abs_lateral_velocity": 0.0,
                  "mean_abs_yaw_rate": 0.0, "mean_forward_velocity": 0.0,
                  "half_cycle_success_left_right": [0.0, 0.0],
                  "consecutive_8_success_fraction": 0.0, "late_xy_error_m": 0.06}
        self.assertTrue(evaluate_gate("stand", result)[0])
        self.assertFalse(evaluate_gate("lift", result)[0])
        self.assertFalse(evaluate_gate("walk", result)[0])
        result.update(mean_forward_velocity=0.10, half_cycle_success_left_right=[0.95, 0.95],
                      consecutive_8_success_fraction=0.95, late_xy_error_m=0.01)
        self.assertTrue(evaluate_gate("walk", result)[0])
        result["survival_fraction"] = 0.2
        self.assertFalse(evaluate_gate("walk", result)[0])

    def test_straight_gate_rejects_drift_and_missing_samples(self):
        result = {"survival_fraction": 1., "mean_upright": .999,
                  "minimum_valid_root_height": .96, "mean_abs_lateral_velocity": .05,
                  "mean_abs_yaw_rate": .1, "mean_reference_forward_velocity": .04,
                  "half_cycle_success_left_right": [1., 1.],
                  "consecutive_8_success_fraction": 1., "late_xy_error_m": .01,
                  "max_abs_heading_error_rad": .1, "heading_valid_samples": 1000}
        self.assertTrue(evaluate_gate("straight", result)[0])
        result["max_abs_heading_error_rad"] = .3
        self.assertFalse(evaluate_gate("straight", result)[0])
        result.update(max_abs_heading_error_rad=0., heading_valid_samples=0)
        self.assertFalse(evaluate_gate("straight", result)[0])

    def test_writer_flush_and_distinct_time_axes(self):
        with tempfile.TemporaryDirectory() as temp:
            writer = TensorboardJsonWriter(temp)
            writer.add_scalar("Reward/total", 0.1, 5)
            writer.add_scalar("Perf/total_fps", 1000, 5)
            writer.add_scalar("Train/mean_reward/time", 3.0, 300)
            writer.add_scalar("Reward/total", 0.2, 6)
            writer.stop()
            writer.stop()
            rows = [json.loads(line) for line in (Path(temp) / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([row["iteration"] for row in rows], [5, 6])
            self.assertEqual(rows[0]["metrics"], {"Reward/total": 0.1, "Perf/total_fps": 1000.0})
            self.assertTrue(list(Path(temp).glob("events.out.tfevents.*")))

    def _mock_curriculum(self, directory, continue_after_pass, random_range=None, fail_upper=False,
                         eval_interval=1, cadence=False):
        """用假子进程检查调度，不启动仿真、训练或云端日志。"""
        args = Namespace(max_hours=1, max_iterations=3, chunk_iterations=1,
                               isaac_python=Path("/unused/python"), rollout_steps=64, mini_batches=4,
                               ablation="none", normalization_prior_rollouts=0, seed=42,
                               print_every=1, eval_envs=2, eval_steps=1000,
                               step_length_range=random_range, learning_rate=1e-4,
                               continue_after_pass=continue_after_pass, eval_every_chunks=eval_interval,
                               eval_frequency_corners=cadence)
        if cadence:
            args.left_gait_frequency_range = args.right_gait_frequency_range = (.65, .85)
        state = {"stages": ["straight"], "stage_index": 0, "chunk": 0, "updates": 0,
                 "samples": 0, "num_envs": 2, "pass_streak": 0}
        supervisor = Supervisor(args, directory, state)
        result = {"survival_fraction": 1., "mean_upright": .999,
                  "minimum_valid_root_height": .96, "mean_abs_lateral_velocity": .01,
                  "mean_abs_yaw_rate": .05, "mean_forward_velocity": .14,
                  "mean_reference_forward_velocity": .14,
                  "half_cycle_success_left_right": [1., 1.], "consecutive_8_success_fraction": 1.,
                  "late_xy_error_m": .01, "max_abs_heading_error_rad": .05,
                  "heading_valid_samples": 2000, "terminated": 0, "truncated": 4,
                  "env_cfg": {"gait_frequency": .75, "left_gait_frequency_range": [.65, .85],
                              "right_gait_frequency_range": [.65, .85]}}

        def child(command, log, metrics=None, on_row=None, timeout=None):
            if "--output_dir" in command:
                output = Path(command[command.index("--output_dir") + 1])
                output.mkdir(parents=True)
                (output / "model_final.pt").write_text("mock")
                row = {"metrics": {}}
                assert on_row is not None
                on_row(row)
                return 0, [row], []
            output = Path(command[command.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            point = dict(result)
            if "--step_length_range" in command:
                length = float(command[command.index("--step_length_range") + 1])
                point["command_nominal_forward_speed_m_s"] = 1.5 * length
                if fail_upper and length > .11:
                    point["mean_abs_yaw_rate"] = 1.
            if cadence and "--left-gait-frequency-range" in command:
                left_index, right_index = command.index("--left-gait-frequency-range"), command.index("--right-gait-frequency-range")
                left = list(map(float, command[left_index+1:left_index+3]))
                right = list(map(float, command[right_index+1:right_index+3]))
                if left == [.65, .65] and right == [.85, .85]:
                    point["mean_abs_yaw_rate"] = 1.
            output.write_text(json.dumps(point))
            return 0, [], []

        with patch.object(supervisor, "run_child", side_effect=child), redirect_stdout(StringIO()):
            supervisor.curriculum()
        return state

    def test_continue_after_pass_uses_update_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            state = self._mock_curriculum(Path(temp), True)
            self.assertEqual(state["updates"], 3)
            self.assertEqual(state["status"], "budget_exhausted")
            self.assertTrue(state["last_stage_qualified"])
        with tempfile.TemporaryDirectory() as temp:
            state = self._mock_curriculum(Path(temp), False)
            self.assertEqual(state["updates"], 2)
            self.assertEqual(state["status"], "all_gates_passed")

    def test_failed_upper_step_cannot_pass_random_range(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            state = self._mock_curriculum(directory, True, (.06, .12), fail_upper=True)
            gate = json.loads((directory / "evaluations/00000_straight.gate.json").read_text())
            self.assertFalse(gate["passed"])
            self.assertTrue(gate["checks"]["step_060mm_all_gates"])
            self.assertFalse(gate["checks"]["step_120mm_all_gates"])
            self.assertNotIn("best_passed", state)

    def test_resume_directory_rejects_changed_command_distribution(self):
        # 原3cm实验不能被同名续训偷偷改成6–12cm；新分布必须建立独立实验。
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "state.json").write_text("{}")
            (directory / "launcher.json").write_text(json.dumps({"step_length_range": None}))
            argv = ["train_4060.py", "--resume-dir", temp, "--start-stage", "straight",
                    "--step-length-range", ".06", ".12"]
            with patch("sys.argv", argv), self.assertRaisesRegex(ValueError, "resume-dir"):
                main()

    def test_less_frequent_eval_keeps_all_training_updates(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            state = self._mock_curriculum(directory, True, eval_interval=4)
            self.assertEqual(state['updates'], 3)
            self.assertEqual(len(list((directory/'evaluations').glob('*.gate.json'))), 1)
            self.assertEqual(len(list((directory/'training').glob('*/model_final.pt'))), 3)

    def test_failed_asymmetric_corner_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            state = self._mock_curriculum(directory, True, eval_interval=4, cadence=True)
            gate = json.loads((directory/'evaluations/00002_straight.gate.json').read_text())
            self.assertFalse(gate['passed'])
            self.assertFalse(gate['checks']['cadence_L0650_R0850mHz_all_gates'])
            self.assertEqual(len(list((directory/'evaluations').glob('*cadence*.gate.json'))), 4)
            self.assertNotIn('best_passed', state)

    def test_resume_rejects_changed_cadence_distribution(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory/'state.json').write_text('{}')
            (directory/'launcher.json').write_text(json.dumps({'left_gait_frequency_range': [.65, .85]}))
            argv = ['train_4060.py', '--resume-dir', temp, '--start-stage', 'straight',
                    '--left-gait-frequency-range', '.7', '.9']
            with patch('sys.argv', argv), self.assertRaisesRegex(ValueError, 'resume-dir'):
                main()


if __name__ == "__main__":
    unittest.main()
