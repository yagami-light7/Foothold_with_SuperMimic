"""CPU regressions for clip isolation, launcher safety and SwanLab CSV following."""

# Support both direct execution and python -m from the repository layout.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_root = str(_ToolPath(__file__).resolve().parents[2])
if _tool_root not in _tool_sys.path:
    _tool_sys.path.insert(0, _tool_root)

import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'tools'))
from supermimic.reference_selection import select_reference_files
from tools.ops.sync_swanlab import METRICS, read_latest_run, sync_run


HEADER = ['iteration', 'wall_h', 'sim_steps', 'wm_loss', 'wm_err8', 'pi_loss', 'v_loss',
          'imag_rew', 'pi_gradnorm', 'imag_survival', 'sim_rew', 'ep_len', 'ep_ret_norm',
          'term_frac', 'it_per_s', 'lr']


def training_csv(steps):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=HEADER, lineterminator='\n')
    writer.writeheader()
    for step in steps:
        writer.writerow({key: step if key == 'iteration' else 0.25 for key in HEADER})
    return buffer.getvalue()


class Recorder:
    def __init__(self):
        self.rows = []
        self.state = None

    def init(self, **kwargs):
        self.config = kwargs

    def log(self, values, step):
        self.rows.append((step, values))

    def finish(self, state):
        self.state = state


class SelectionAndSyncTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='s54_launcher_')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def references(self):
        root = self.root / 'references with spaces'
        root.mkdir()
        for name in ['dance1_subject1', 'walk1_subject1', 'walk2_subject1']:
            (root / f'{name}.csv').write_text('placeholder')
            (root / f'{name}.json').write_text('{}')
        for directory in ['raw', '.staging', 'rejected']:
            (root / directory).mkdir()
            (root / directory / 'not_a_training_clip.csv').write_text('placeholder')
        return root

    def sync_args(self, follow=False):
        (self.root / 'args.json').write_text(json.dumps({'iterations': 25, 'clips': ['dance1_subject1']}))
        return argparse.Namespace(run_dir=self.root, container=None, session=None, startup_timeout=1,
                                  poll_seconds=.001, experiment_name='test_run', project='SuperMimic',
                                  mode='offline', workspace=None, follow=follow, completion_marker=True)

    def test_selection_deduplicates_and_excludes_nested_csvs(self):
        root = self.references()
        names = [Path(p).stem for p in select_reference_files(root, ['walk*', 'walk1_subject1.csv'])]
        self.assertEqual(names, ['walk1_subject1', 'walk2_subject1'])
        self.assertEqual(len(select_reference_files(root)), 3)

    def test_every_requested_pattern_must_match(self):
        with self.assertRaises(FileNotFoundError):
            select_reference_files(self.references(), ['dance1_subject1', 'does_not_exist'])

    def test_missing_metadata_is_rejected(self):
        root = self.references()
        (root / 'dance1_subject1.json').unlink()
        with self.assertRaises(FileNotFoundError):
            select_reference_files(root, ['dance1_subject1'])

    def test_clip_paths_are_rejected(self):
        with self.assertRaises(ValueError):
            select_reference_files(self.references(), ['raw/not_a_training_clip'])

    def test_root_entry_dry_run_preserves_paths_and_creates_no_run(self):
        run = f's54_launcher_test_{uuid.uuid4().hex}'
        log_path = REPO / 'logs' / run
        result = subprocess.run(['bash', str(REPO / 'tools/train/run_train.sh'), '--dry-run', 'train', run,
                                 '--motion-dir', str(self.references()), '--clips', 'dance1_subject1',
                                 'walk1_subject1', '--swanlab-name', 'Dance and Walk'],
                                cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Selected 2 clips: dance1_subject1, walk1_subject1', result.stdout)
        self.assertIn('--clips dance1_subject1 walk1_subject1', result.stdout)
        self.assertIn('Dance\\ and\\ Walk', result.stdout)
        self.assertIn('SwanLab (online)', result.stdout)
        self.assertFalse(log_path.exists())

    def test_all_clips_and_cloud_disable(self):
        result = subprocess.run(['bash', str(REPO / 'tools/train/run_train.sh'), '--dry-run', 'smoke',
                                 's54_launcher_test_all', '--motion-dir', str(self.references()),
                                 '--all-clips', '--no-swanlab'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Selected 3 clips', result.stdout)
        self.assertIn('--iterations 30', result.stdout)
        self.assertIn('SwanLab: disabled', result.stdout)

    def test_foot_goal_options_reach_training_without_launch(self):
        config = self.root / 'foot sampler.json'
        config.write_text('{}')
        result = subprocess.run(['bash', str(REPO / 'tools/train/run_train.sh'), '--dry-run', 'train',
                                 's54_foot_launcher_test', '--motion-dir', str(self.references()),
                                 '--clips', 'dance1_subject1', '--no-swanlab', '--goal-mode', 'foot_trajectory',
                                 '--goal-dt', '0.1', '--foot-config', str(config)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--goal_mode foot_trajectory', result.stdout)
        self.assertIn('--goal_dt 0.1', result.stdout)
        self.assertIn('--foot_config', result.stdout)
        self.assertFalse((REPO / 'logs/s54_foot_launcher_test').exists())

    def test_foot_standing_dry_run_needs_no_motion_directory(self):
        result = subprocess.run(['bash',str(REPO/'tools/train/run_train.sh'),'--dry-run','train','s54_standing_dry_test',
                                 '--goal-mode','foot_trajectory','--motion-dir',str(self.root/'missing_motion'),
                                 '--foot-task-config','configs/s54/standing/s54_standing_balance.json',
                                 '--check-foot-gradients','--no-swanlab'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('Motion directory and clip selection are unused',result.stdout)
        self.assertIn('--foot_task_config configs/s54/standing/s54_standing_balance.json',result.stdout)
        self.assertIn('--check_foot_gradients',result.stdout)
        self.assertNotIn('--clips dance1_subject1',result.stdout)

    def test_resume_uses_total_iteration_override_without_creating_logs(self):
        # The launcher only checks the path here; a real Torch checkpoint is
        # validated by train.py. Keep this fixture outside all user run folders.
        (REPO / 'logs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='s54_resume_fixture_', dir=REPO / 'logs') as tmp:
            checkpoint = Path(tmp) / 'last.pt'
            checkpoint.touch()
            run = f's54_resume_test_{uuid.uuid4().hex}'
            result = subprocess.run(['bash', str(REPO / 'tools/train/run_train.sh'), '--dry-run', 'resume', run,
                                     str(checkpoint.relative_to(REPO)), '--iterations', '20000',
                                     '--motion-dir', str(self.references()), '--clips',
                                     'dance1_subject1', 'walk1_subject1'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--iterations 20000', result.stdout)
            self.assertIn('--resume ' + str(checkpoint.relative_to(REPO)), result.stdout)
            self.assertIn('--wm_only_iters 1000', result.stdout)
            self.assertIn('--lr_decay_iters 40000', result.stdout)
            self.assertFalse((REPO / 'logs' / run).exists())

    def test_invalid_clip_fails_dry_run_before_creating_logs(self):
        result = subprocess.run(['bash', str(REPO / 'tools/train/run_train.sh'), '--dry-run', 'train',
                                 's54_launcher_test_missing', '--motion-dir', str(self.references()),
                                 '--clips', 'missing'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('No reference matches', result.stderr)

    def container_preflight(self, state, mount=None, dry_run=False):
        """Exercise the real launcher with Docker isolated; stop at the VRAM guard."""
        bin_dir = self.root / 'bin'
        bin_dir.mkdir()
        state_path = self.root / 'container_state'
        state_path.write_text(state)
        calls_path = self.root / 'docker_calls.jsonl'
        fake_docker = bin_dir / 'docker'
        fake_docker.write_text(
            '#!/usr/bin/env python3\n'
            'import json, sys\nfrom pathlib import Path\n'
            f'state = Path({str(state_path)!r})\n'
            f'calls = Path({str(calls_path)!r})\n'
            'args = sys.argv[1:]\n'
            'with calls.open("a") as f: f.write(json.dumps(args) + "\\n")\n'
            'if args[0] == "inspect":\n'
            '    fmt = args[2]\n'
            f'    if "Mounts" in fmt: print({str(mount or REPO)!r})\n'
            '    elif "State.Status" in fmt: print(state.read_text())\n'
            '    elif "State.Running" in fmt: print("true" if state.read_text() == "running" else "false")\n'
            '    else: raise SystemExit(2)\n'
            'elif args == ["start", "supermimic_s54"]: state.write_text("running")\n'
            'elif args[0] == "exec" and "--query-gpu=memory.free" in args: print("0")\n'
            'else: raise SystemExit("Unexpected Docker operation: " + repr(args))\n'
        )
        fake_docker.chmod(0o755)
        env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ['PATH'],
                   CONTAINER_NAME='supermimic_s54')
        command = ['bash', str(REPO / 'tools/train/run_train.sh')]
        if dry_run:
            command.append('--dry-run')
        command += ['train', f's54_preflight_{uuid.uuid4().hex}', '--motion-dir',
                    str(self.references()), '--clips', 'dance1_subject1', '--no-swanlab']
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        calls = [json.loads(line) for line in calls_path.read_text().splitlines()] if calls_path.exists() else []
        return result, calls

    def test_stopped_container_is_started_before_gpu_check(self):
        result, calls = self.container_preflight('exited')
        self.assertIn(['start', 'supermimic_s54'], calls)
        self.assertIn('Starting existing container', result.stdout)
        self.assertIn('Insufficient GPU memory', result.stderr)
        self.assertEqual(result.returncode, 1)

    def test_running_container_is_reused_without_restart(self):
        result, calls = self.container_preflight('running')
        self.assertFalse(any(call[0] == 'start' for call in calls))
        self.assertIn('Insufficient GPU memory', result.stderr)

    def test_wrong_mount_cannot_trigger_container_start(self):
        result, calls = self.container_preflight('exited', mount='/tmp/unrelated_project')
        self.assertFalse(any(call[0] == 'start' for call in calls))
        self.assertIn('Wrong container mount', result.stderr)

    def test_dry_run_does_not_start_stopped_container(self):
        result, calls = self.container_preflight('exited', dry_run=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, [])

    def test_job_runner_preserves_failure_through_tee(self):
        for code in (0, 7):
            with self.subTest(exit_code=code):
                run_dir = f'logs/s54_test_job_{code}'
                result = subprocess.run(['bash', str(REPO / 'tools/train/run_s54_job.sh'), run_dir,
                                         sys.executable, '-c', f'print("trainer output"); raise SystemExit({code})'],
                                        cwd=self.root, capture_output=True, text=True)
                self.assertEqual(result.returncode, code)
                self.assertEqual(int((self.root / run_dir / 'train.exitcode').read_text()), code)
                self.assertIn('trainer output', (self.root / run_dir / 'stdout.log').read_text())

    def test_partial_line_is_deferred_and_older_run_is_excluded(self):
        path = self.root / 'train_log.csv'
        text = training_csv([50]) + training_csv([10])
        path.write_text(text + '20,partial')
        _, rows = read_latest_run(path)
        self.assertEqual([row['iteration'] for row in rows], ['10'])

    def test_sync_records_actual_clips_and_deduplicates_steps(self):
        args = self.sync_args()
        (self.root / 'train_log.csv').write_text(training_csv([10, 10, 20]))
        recorder = Recorder()
        sync_run(args, recorder)
        self.assertEqual([step for step, _ in recorder.rows], [10, 20])
        self.assertEqual(set(recorder.rows[0][1]), set(METRICS.values()))
        self.assertEqual(recorder.config['config']['clips'], ['dance1_subject1'])
        self.assertEqual(recorder.state, 'success')

    def test_resume_keeps_cloud_id_and_uploads_optional_head_metrics(self):
        args = self.sync_args()
        lines = training_csv([10100]).splitlines()
        (self.root / 'train_log.csv').write_text(
            lines[0] + ',head_rms_deg,head_penalty_norm\n' + lines[1] + ',12.5,-0.01\n')
        recorder = Recorder()
        with patch.dict(os.environ, {'SWANLAB_RUN_ID': 'existing_run', 'SWANLAB_RESUME': 'must'}):
            sync_run(args, recorder)
        self.assertEqual(recorder.config['id'], 'existing_run')
        self.assertEqual(recorder.config['resume'], 'must')
        self.assertEqual(recorder.rows[0][0], 10100)
        self.assertEqual(recorder.rows[0][1]['Head/RMSErrorDeg'], 12.5)
        self.assertEqual(recorder.rows[0][1]['Head/NormalizedPenalty'], -.01)
        status = json.loads((self.root / 'swanlab_sync_status.json').read_text())
        self.assertEqual(status['run_id'], 'existing_run')

    def test_completion_marker_finishes_when_last_iteration_was_not_logged(self):
        args = self.sync_args(follow=True)
        (self.root / 'train_log.csv').write_text(training_csv([10, 20]))
        (self.root / 'train.exitcode').write_text('0\n')
        recorder = Recorder()
        sync_run(args, recorder)
        status = json.loads((self.root / 'swanlab_sync_status.json').read_text())
        self.assertEqual((status['status'], status['last_step']), ('finished', 20))

    def test_failed_training_still_uploads_final_rows_then_finishes_crashed(self):
        args = self.sync_args(follow=True)
        (self.root / 'train_log.csv').write_text(training_csv([10]))
        (self.root / 'train.exitcode').write_text('7\n')
        recorder = Recorder()
        sync_run(args, recorder)
        self.assertEqual([step for step, _ in recorder.rows], [10])
        self.assertEqual(recorder.state, 'crashed')
        status=json.loads((self.root / 'swanlab_sync_status.json').read_text())
        self.assertEqual(status['status'], 'finished_training_failed')
        self.assertIn('Training exit 7',status['error'])

    def test_new_appended_run_does_not_mix_cloud_experiments(self):
        args = self.sync_args(follow=True)
        path = self.root / 'train_log.csv'
        path.write_text(training_csv([10]))
        recorder = Recorder()
        def append_run(_):
            path.write_text(path.read_text() + training_csv([1]))
        with patch('tools.ops.sync_swanlab.time.sleep', side_effect=append_run):
            with self.assertRaisesRegex(RuntimeError, 'refusing to mix'):
                sync_run(args, recorder)
        self.assertEqual([step for step, _ in recorder.rows], [10])
        self.assertEqual(recorder.state, 'crashed')


if __name__ == '__main__':
    unittest.main(verbosity=2)
