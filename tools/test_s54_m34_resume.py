"""CPU orchestration checks: resume avoids PD/fresh smoke but keeps frozen eval."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/'tools'))
import run_s54_m34 as pipeline


class ResumePipelineTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='s54_resume_pipeline_')
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        original = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, original)
        for name in ('supermimic', 'tools', 'configs'):
            Path(name).mkdir()
        self.config = dict(dataset='data/export', iterations=10000, eval_every=1000, eval_seeds=[0, 1, 2])
        Path('configs/train.json').write_text(json.dumps(self.config))
        self.checkpoint = Path('logs/s54_old/last.pt')
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_bytes(b'old checkpoint fixture')
        self.commands = []
        self.fail_cuda = False

    def child(self, *command):
        command = list(map(str, command))
        self.commands.append(command)
        if 'tools/check_s54_cuda.py' in command and self.fail_cuda:
            raise subprocess.CalledProcessError(1, command)
        if 'supermimic/train.py' in command:
            target = Path(command[command.index('--log_dir')+1])
            target.mkdir(parents=True, exist_ok=True)
            (target/'last.pt').write_bytes(b'new checkpoint fixture')
            (target/'training_status.json').write_text(json.dumps(dict(iteration=2000)))
        if 'tools/m34_rollout.py' in command:
            target = Path(command[command.index('--out')+1])
            target.mkdir(parents=True)
            (target/'results.json').write_text(json.dumps(dict(rows=[
                dict(mode=mode, record='diagnostic.npz') for mode in ('straight', 'turn', 'spin')])))

    def execute(self, resume=True):
        argv = ['run_s54_m34.py', '--run', 's54_resumed', '--config', 'configs/train.json']
        if resume:
            argv += ['--resume', str(self.checkpoint)]
        evaluations = {split: dict(rows=[dict(seed=0, id=split+'_reference', record='rollout.npz')])
                       for split in ('train', 'test')}
        with patch.object(sys, 'argv', argv), patch.object(pipeline, 'call', side_effect=self.child), \
             patch.object(pipeline, 'check_smoke') as smoke, patch.object(pipeline, 'report') as report, \
             patch.object(pipeline.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as raw, \
             patch('supermimic.m34_evaluation.evaluate_checkpoint', return_value=evaluations) as evaluate:
            pipeline.main()
        return smoke, report, raw, evaluate

    def test_resume_skips_pd_and_smoke_keeps_checkpoint_and_evaluation(self):
        smoke, report, raw, evaluate = self.execute()
        smoke.assert_not_called(); raw.assert_not_called(); report.assert_called_once()
        self.assertFalse(any('tools/m34_rollout.py' in c for c in self.commands))
        self.assertFalse(any('tools/report_s54_m34_pd.py' in c for c in self.commands))
        trains = [c for c in self.commands if 'supermimic/train.py' in c]
        self.assertEqual(len(trains), 1)
        command = trains[0]
        self.assertEqual(command[command.index('--resume')+1], str(self.checkpoint))
        self.assertEqual(command[command.index('--iterations')+1], '10000')
        self.assertEqual(command[command.index('--m34_eval_every')+1], '1000')
        evaluate.assert_called_once()
        self.assertEqual(evaluate.call_args.args[-1], [0, 1, 2])
        rendered = [c for c in self.commands if 'tools/render_s54_motion_prior.py' in c]
        self.assertEqual(len(rendered), 2)  # train and test, no missing-PD failure
        self.assertFalse(Path('logs/s54_resumed/pd').exists())
        self.assertFalse(Path('logs/s54_resumed_smoke').exists())
        state = json.loads(Path('logs/s54_resumed/pipeline_status.json').read_text())
        self.assertEqual(state['status'], 'completed')
        self.assertEqual(state['resume_checkpoint'], str(self.checkpoint.resolve()))
        self.assertEqual(self.checkpoint.read_bytes(), b'old checkpoint fixture')

    def test_fresh_training_keeps_pd_comparisons_and_smoke(self):
        smoke, report, raw, evaluate = self.execute(resume=False)
        smoke.assert_called_once(); report.assert_called_once(); evaluate.assert_called_once()
        self.assertEqual(raw.call_count, 3)
        self.assertTrue(any('tools/m34_rollout.py' in c for c in self.commands))
        trains = [c for c in self.commands if 'supermimic/train.py' in c]
        self.assertEqual(len(trains), 2)
        self.assertEqual(trains[0][trains[0].index('--iterations')+1], '30')
        self.assertFalse(any('--resume' in c for c in trains))

    def test_failed_cuda_is_reported_before_cpu_pd_or_training(self):
        self.fail_cuda = True
        with self.assertRaises(subprocess.CalledProcessError):
            self.execute()
        self.assertEqual(len(self.commands), 1)
        self.assertIn('tools/check_s54_cuda.py', self.commands[0])
        state = json.loads(Path('logs/s54_resumed/pipeline_status.json').read_text())
        self.assertEqual((state['status'], state['stage']), ('failed', 'cuda_preflight'))
        self.assertFalse(Path('logs/s54_resumed/last.pt').exists())

    def test_missing_checkpoint_does_not_create_run(self):
        self.checkpoint.unlink()
        with self.assertRaises(SystemExit) as result:
            self.execute()
        self.assertEqual(result.exception.code, 2)
        self.assertFalse(Path('logs/s54_resumed').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
