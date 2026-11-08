"""CPU orchestration regression; Newton itself is exercised by m34_rollout."""
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from supermimic.m34_evaluation import evaluate_checkpoint
from run_s54_m34 import train_args


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.latest = self.root/'last.pt'; self.latest.write_bytes(b'frozen weights')
        self.checkpoint = self.root/'checkpoint.pt'; self.checkpoint.hardlink_to(self.latest)
        self.output = self.root/'evals/iter_001000'
        self.calls = []

    def child(self, command, **kwargs):
        self.calls.append(command)
        target = Path(command[command.index('--out')+1]); target.mkdir(parents=True)
        (target/'results.json').write_text(json.dumps(dict(status='completed',
            checkpoint_sha256=hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
            rows=[dict(split=command[command.index('--split')+1], reset_noise='--reset_noise' in command)])))
        return subprocess.CompletedProcess(command, 0)

    def test_frozen_train_test_protocol_reuse_and_parent_state(self):
        state = random.getstate()
        with patch('supermimic.m34_evaluation.subprocess.run', side_effect=self.child):
            result = evaluate_checkpoint(self.checkpoint, self.root, self.output, [0, 1, 2])
            self.assertEqual(set(result), {'train', 'test'})
            self.assertEqual(len(self.calls), 4)
            for command in self.calls:
                self.assertEqual(command[command.index('--checkpoint')+1], str(self.checkpoint))
                seeds = command[command.index('--seeds')+1:]
                self.assertEqual(seeds, ['1', '2', '--reset_noise'] if '--reset_noise' in command else ['0'])
            # Atomic latest replacement cannot modify the frozen hard link.
            temp = self.root/'new.pt'; temp.write_bytes(b'new weights'); temp.replace(self.latest)
            again = evaluate_checkpoint(self.checkpoint, self.root, self.output, [0, 1, 2])
            self.assertEqual(result, again); self.assertEqual(len(self.calls), 4)
        self.assertEqual(random.getstate(), state)
        self.assertEqual(self.checkpoint.read_bytes(), b'frozen weights')

    def test_failed_attempt_is_preserved_and_retried_separately(self):
        with patch('supermimic.m34_evaluation.subprocess.run', side_effect=subprocess.CalledProcessError(7, ['newton'])):
            with self.assertRaises(subprocess.CalledProcessError):
                evaluate_checkpoint(self.checkpoint, self.root, self.output, [0])
        self.assertEqual(json.loads((self.output/'status.json').read_text())['status'], 'failed')
        self.assertFalse((self.output/'summary.json').exists())
        with patch('supermimic.m34_evaluation.subprocess.run', side_effect=self.child):
            evaluate_checkpoint(self.checkpoint, self.root, self.output, [0])
        self.assertTrue((self.output/'attempt_001/train_nominal.log').exists())
        self.assertTrue((self.output/'attempt_002/train_nominal.log').exists())

    def test_protocol_or_checkpoint_change_cannot_reuse_results(self):
        with patch('supermimic.m34_evaluation.subprocess.run', side_effect=self.child):
            evaluate_checkpoint(self.checkpoint, self.root, self.output, [0])
            with self.assertRaises(ValueError): evaluate_checkpoint(self.checkpoint, self.root, self.output, [0, 1])
            self.checkpoint.write_bytes(b'changed')
            with self.assertRaises(ValueError): evaluate_checkpoint(self.checkpoint, self.root, self.output, [0])

    def test_changed_checkpoint_during_evaluation_is_rejected(self):
        def changed(command, **kwargs):
            result = self.child(command, **kwargs)
            self.checkpoint.write_bytes(b'changed during evaluation')
            return result
        with patch('supermimic.m34_evaluation.subprocess.run', side_effect=changed):
            with self.assertRaises(RuntimeError): evaluate_checkpoint(self.checkpoint, self.root, self.output, [0])
        self.assertFalse((self.output/'summary.json').exists())

    def test_pipeline_forwards_interval_seeds_and_disables_smoke_eval(self):
        config = dict(dataset='data/example', iterations=10000, eval_every=1000, eval_seeds=[0, 5, 7])
        command = train_args(config, self.root)
        self.assertEqual(command[command.index('--m34_eval_every')+1], '1000')
        self.assertEqual(command[command.index('--m34_eval_seeds')+1:command.index('--m34_eval_seeds')+4], ['0', '5', '7'])
        self.assertNotIn('--eval_every', command); self.assertNotIn('--eval_seeds', command)
        smoke = train_args(config, self.root, smoke=True)
        self.assertEqual(smoke[smoke.index('--m34_eval_every')+1], '0')
        self.assertEqual(config['iterations'], 10000)

    def test_invalid_schedule_rejected_before_launch(self):
        base = dict(dataset='data/example', eval_every=1000, eval_seeds=[0, 1, 2])
        for change in ({'eval_every': -1}, {'eval_every': 1000.5}, {'eval_every': True},
                       {'eval_seeds': []}, {'eval_seeds': [0, 0]}, {'eval_seeds': [-1]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                train_args(base | change, self.root)


if __name__ == '__main__': unittest.main(verbosity=2)
