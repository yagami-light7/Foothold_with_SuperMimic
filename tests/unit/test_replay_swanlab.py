"""补传的Step对齐、断尾和原实验恢复回归；不联网或启动仿真。"""

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.replay_swanlab import make_plan, metric_rows, metric_snapshot, replay


class ReplaySwanlabTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.state = {"updates": 1, "swanlab_id": "original-run", "status": "training"}
        (self.run / "state.json").write_text(json.dumps(self.state))
        (self.run / "launcher.json").write_text(json.dumps({
            "project": "S54-Locomotion", "workspace": "test-user", "start_stage": "straight"}))
        self.args = Namespace(run_dir=self.run, start_step=None, end_step=None,
                              new_run=False, project=None, workspace=None,
                              detail="compact", print_every=250, dry_run=False)
        self.path = self.add_chunk("00000_retention", [40658, 40659, 40660])
        self.sdk = Mock()
        self.sdk.init.return_value = SimpleNamespace(url="https://example.invalid/original-run")

    def add_chunk(self, name, iterations):
        directory = self.run / "training" / name
        directory.mkdir(parents=True)
        (directory / "env_cfg.json").write_text(json.dumps({"curriculum_stage": "straight"}))
        (directory / "manifest.json").write_text(json.dumps({"status": "interrupted"}))
        path = directory / "metrics.jsonl"
        rows = [dict(iteration=iteration, wall_time=1000 + index, metrics={
            "Reward/total": .12, "Episode/normalized_return": .8,
            "Loss/learning_rate": 2e-5, "Control/joint": .9})
            for index, iteration in enumerate(iterations)]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    def test_original_step_axis_skips_processed_rows(self):
        plan, sources = make_plan(self.args)
        self.assertEqual((plan["local_updates"], plan["supervisor_updates"]), (3, 1))
        self.assertEqual((plan["first_step"], plan["last_step"], plan["records_to_submit"]), (2, 3, 2))
        self.assertEqual([step for step, _, _ in metric_rows(sources)], [1, 2, 3])
        self.assertIn("Train/return_0to1", plan["metric_keys"])
        self.assertNotIn("straight/Control/joint", plan["metric_keys"])

    def test_resume_original_without_mutating_training_state(self):
        before = {path: path.read_bytes() for path in self.run.rglob("*") if path.is_file()}
        with redirect_stdout(StringIO()):
            replay(self.args, sdk=self.sdk)
        options = self.sdk.init.call_args.kwargs
        self.assertEqual(options["id"], "original-run")
        self.assertEqual(options["resume"], "must")
        self.assertNotIn("config", options)
        self.assertEqual([call.kwargs["step"] for call in self.sdk.log.call_args_list], [2, 3])
        self.assertEqual(self.sdk.log.call_args_list[0].args[0]["Train/reward_per_step"], .12)
        self.sdk.finish.assert_called_once_with(state="aborted")
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_new_run_replays_all_rows_without_old_id(self):
        self.args.new_run = True
        with redirect_stdout(StringIO()):
            replay(self.args, sdk=self.sdk)
        options = self.sdk.init.call_args.kwargs
        self.assertNotIn("id", options)
        self.assertNotIn("resume", options)
        self.assertTrue(options["name"].endswith("_recovered"))
        self.assertEqual([call.kwargs["step"] for call in self.sdk.log.call_args_list], [1, 2, 3])
        self.sdk.finish.assert_called_once_with(state="success")

    def test_frozen_snapshot_ignores_partial_tail_and_newly_appended_records(self):
        line = json.dumps({"iteration": 40661, "metrics": {"Reward/total": .1}})
        with self.path.open("a") as stream:
            stream.write(line[:-3])
        sources = metric_snapshot(self.run, "straight")
        with self.path.open("a") as stream:
            stream.write(line[-3:] + "\n")
        self.assertEqual(len(list(metric_rows(sources))), 3)
        self.assertEqual(len(list(metric_rows(metric_snapshot(self.run, "straight")))), 4)

    def test_multiple_chunks_continue_axis_despite_old_iteration_indices(self):
        self.add_chunk("00001_retention", [40661, 40662])
        plan, _ = make_plan(self.args)
        self.assertEqual((plan["local_updates"], plan["first_step"], plan["last_step"]), (5, 2, 5))

    def test_explicit_range_and_full_diagnostics(self):
        self.args.start_step = self.args.end_step = 3
        self.args.detail = "full"
        plan, _ = make_plan(self.args)
        self.assertEqual(plan["records_to_submit"], 1)
        self.assertIn("straight/Control/joint", plan["metric_keys"])

    def test_dry_run_never_imports_sdk_or_creates_output(self):
        self.args.dry_run = True
        with patch("scripts.replay_swanlab.importlib.import_module") as load_sdk, redirect_stdout(StringIO()):
            replay(self.args)
        load_sdk.assert_not_called()
        self.assertFalse((self.run / "swanlog_replay").exists())

    def test_no_missing_rows_does_not_open_cloud_run(self):
        self.state["updates"] = 3
        (self.run / "state.json").write_text(json.dumps(self.state))
        with redirect_stdout(StringIO()):
            plan = replay(self.args, sdk=self.sdk)
        self.assertEqual(plan["records_to_submit"], 0)
        self.sdk.init.assert_not_called()

    def test_invalid_complete_record_fails_before_cloud_init(self):
        for text in ('{"metrics":', '{"metrics": {"Reward/total": NaN}}'):
            with self.subTest(text=text):
                self.path.write_text(text + "\n")
                with self.assertRaisesRegex(ValueError, "日志损坏"), redirect_stdout(StringIO()):
                    replay(self.args, sdk=self.sdk)
                self.sdk.init.assert_not_called()

    def test_missing_run_id_requires_explicit_new_run(self):
        (self.run / "state.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "new-run"):
            make_plan(self.args)
        self.args.new_run = True
        plan, _ = make_plan(self.args)
        self.assertEqual(plan["first_step"], 1)


if __name__ == "__main__":
    unittest.main()
