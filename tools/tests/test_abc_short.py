"""CPU checks for short-run accounting and truthful evaluation upload payloads."""
import json
from contextlib import nullcontext
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from tools.train import run_s54_abc_short as batch
from tools.ops.upload_eval_swanlab import validate
from tools.ops.swanlab_jobs import PROXY_VARIABLES, tmux_proxy_options
from tools.ops import sync_eval_swanlab as queue


class ShortBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='s54_abc_short_test_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.profile = json.loads((batch.ROOT/'configs/motion_prior/stage2/abc_short_3090.json').read_text())

    def checkpoint(self, samples=None):
        directory = self.root/'dummy_A'
        directory.mkdir()
        train = self.profile['train']
        final = dict(iteration=train['iterations'],
            sim_steps=self.profile['comparison']['same_stage2_real_transitions_target'] if samples is None else samples,
            stage_stats=dict(actor_updates=self.profile['comparison']['same_actor_updates_target']),
            policy_contract=dict(decoder_trainable=False),
            wm={'weight': torch.zeros(1)}, critic={'weight': torch.zeros(1)},
            pi={'enc.weight': torch.ones(1), 'dec.weight': torch.zeros(1)})
        warm = dict(iteration=train['wm_only_iters'], pi={'enc.weight': torch.zeros(1), 'dec.weight': torch.zeros(1)})
        torch.save(final, directory/'last.pt')
        torch.save(warm, directory/'wm_calibrated.pt')
        return directory

    def test_audit_checks_saved_samples_updates_and_encoder_change(self):
        directory = self.checkpoint()
        result = batch.audit_training(directory, self.profile, self.root/'unused_teacher.pt')
        self.assertTrue(result['decoder_unchanged'])
        self.assertTrue(result['foot_encoder_updated'])
        self.assertEqual(result['real_transitions'], 5251072)

    def test_group_names_share_batch_timestamp_and_teacher_source(self):
        profile_path=batch.ROOT/'configs/motion_prior/stage2/abc_short_3090.json'
        entries=batch.plans(profile_path,self.profile,'s54_abc_short_test','python',self.root/'teacher_v1.pt')
        for entry in entries:
            command=entry['launcher']
            name=command[command.index('--experiment-name')+1]
            self.assertEqual(name,f'{entry["group"]}_test_seed54_walking')
            if entry['group']=='A':self.assertNotIn('--source',command)
            else:self.assertEqual(command[command.index('--source')+1],str(self.root/'teacher_v1.pt'))

    def test_stage2_uses_foot_goals_without_motion_clips(self):
        from supermimic.goals import GoalSpec
        task = json.loads((batch.ROOT/'configs/motion_prior/stage2/stage2_walking.json').read_text())
        self.assertEqual(task['command_mode'], 'alternating')
        self.assertEqual(self.profile['training_data']['reference_clips'], [])
        train = self.profile['train']
        foot = GoalSpec('foot_trajectory', train['goal_frames'], train['goal_dt'])
        self.assertEqual(foot.dim, 230)
        self.assertEqual(GoalSpec('motion', 10).dim, 730)
        entries = batch.plans(batch.ROOT/'configs/motion_prior/stage2/abc_short_3090.json',
                              self.profile, 's54_abc_short_test', 'python', self.root/'teacher.pt')
        for entry in entries:
            cmd = entry['training_command']
            self.assertEqual(cmd[cmd.index('--goal_mode')+1], 'foot_trajectory')
            self.assertNotIn('--clips', cmd)
            self.assertNotIn('--reference_manifest', cmd)

    def test_audit_rejects_unequal_budget_even_if_launcher_succeeded(self):
        directory = self.checkpoint(samples=5250048)
        with self.assertRaises(AssertionError):
            batch.audit_training(directory, self.profile, self.root/'unused_teacher.pt')

    def test_eval_payload_keeps_physical_failures_and_missing_contacts_explicit(self):
        directory = self.root/'logs/s54_dummy_A'
        directory.mkdir(parents=True)
        # Authentication can fail before any SDK status exists.
        (directory/'swanlab_run_id').write_text('dummy\n')
        (directory/'swanlab_experiment_name').write_text('A_test_seed54_walking\n')
        rows = []
        for duration in self.profile['evaluation']['durations_s']:
            for seed in self.profile['evaluation']['seeds']:
                physical = seed == 54
                rows.append(dict(duration_s=duration, seed=seed, norm_return=.2 if physical else .8,
                    n_failed=int(physical), n_timeout=int(not physical), mean_episode_length=25 if physical else 50*duration,
                    metrics=dict(foot_position_rmse_m=.04, root_pitch_max_abs_deg=11., root_roll_max_abs_deg=3.,
                                 root_height_deviation_max_m=.02, actual_loaded_stance_slip_m_s=None),
                    wm_paired=dict(status='NOT_AVAILABLE')))
        with patch.object(batch, 'ROOT', self.root):
            payload = batch.eval_bundle('s54_dummy_A', dict(rows=rows, eval_real_steps=100), self.profile)
        validate(payload)
        metrics = payload['rows'][0]['metrics']
        self.assertEqual(payload['rows'][0]['step'], 2000)
        self.assertEqual(metrics['Eval/Foot10s/PhysicalTerminations'], 1)
        self.assertEqual(metrics['Eval/Foot10s/Timeouts'], 2)
        self.assertAlmostEqual(metrics['Eval/Foot10s/NormalizedReturn'], .6)
        self.assertNotIn('Eval/Foot10s/actual_loaded_stance_slip_m_s', metrics)
        self.assertEqual(metrics['Eval/Foot10s/WM8/AvailableEpisodes'], 0)
        self.assertFalse(any('SuccessRate' in key for key in metrics))

    def test_tmux_explicitly_replaces_stale_or_removed_proxy_variables(self):
        options = tmux_proxy_options({'HTTPS_PROXY': 'http://127.0.0.1:7897'})
        self.assertEqual(options[::2], ['-e']*len(PROXY_VARIABLES))
        values = dict(v.split('=', 1) for v in options[1::2])
        self.assertEqual(values['HTTPS_PROXY'], 'http://127.0.0.1:7897')
        self.assertEqual(values['http_proxy'], '')

    def test_upload_launch_failure_keeps_bundle_without_raising(self):
        directory = self.root/'logs/s54_dummy_A'
        (directory/'evaluation').mkdir(parents=True)
        payload = dict(target=dict(run_id='original', name='A_original'), rows=[])
        with patch.object(batch, 'eval_bundle', return_value=payload), \
             patch.object(batch.subprocess, 'run', side_effect=[SimpleNamespace(returncode=1),
                 subprocess.CalledProcessError(1, 'tmux')]):
            batch.queue_evaluation_upload(directory, {}, self.profile)
        self.assertEqual(json.loads((directory/'evaluation/eval_bundle.json').read_text()), payload)
        self.assertEqual(json.loads((directory/'evaluation/queue_status.json').read_text())['status'], 'launch_failed')

    def test_recovery_reuses_completed_groups_and_cloud_failure_does_not_abort_batch(self):
        directory = self.root/'logs/s54_batch'
        directory.mkdir(parents=True)
        (directory/'shared_inputs.json').write_text('{}')
        entries = [dict(group=g, run='s54_batch_'+g, launcher=[]) for g in 'ABC']
        for entry in entries:
            (self.root/'logs'/entry['run']/'evaluation').mkdir(parents=True)
            (self.root/'logs'/entry['run']/'evaluation/summary.json').write_text('{}')
        audit = dict(initial_foot_encoder_hash='same', decoder_hash='same')
        # The queue handles an unreachable backend independently. The worker
        # must still finish all equal-budget local comparisons, without reruns.
        with patch.object(batch, 'ROOT', self.root), \
             patch('tools.visualization.policy_preview.gpu_slot', return_value=nullcontext()), \
             patch.object(batch.signal, 'signal'), \
             patch.object(batch, 'digest', return_value=self.profile['teacher']['sha256']), \
             patch.object(batch, 'completed_training') as training, \
             patch.object(batch, 'completed_evaluation', return_value={}) as evaluation, \
             patch.object(batch, 'audit_training', return_value=audit), \
             patch.object(batch, 'queue_evaluation_upload') as upload, \
             patch.object(batch.subprocess, 'Popen') as process:
            batch.run_worker(SimpleNamespace(python='python', reuse_completed=True), directory, self.profile, entries)
        self.assertEqual(training.call_count, 3)
        self.assertEqual(evaluation.call_count, 3)
        self.assertEqual(upload.call_count, 3)
        process.assert_not_called()
        state = json.loads((directory/'batch_status.json').read_text())
        self.assertEqual(state['completed_groups'], list('ABC'))
        self.assertEqual(state['status'], 'completed')
        self.assertEqual(state['uploads'], dict(A='pending', B='pending', C='pending'))

    def test_training_upload_failure_prevents_eval_until_retry(self):
        directory = self.root/'run'
        directory.mkdir()
        payload = dict(target=dict(run_id='same-id', name='A_original',
                                  project='Foothold_with_SuperMimic', workspace='yagami_light'))
        with patch.object(queue, 'execute', side_effect=RuntimeError('network offline')) as execute:
            with self.assertRaisesRegex(RuntimeError, 'network offline'):
                queue.sync_once(directory, directory/'eval_bundle.json', payload, lambda *_: None)
        self.assertEqual(execute.call_count, 1)
        self.assertIn('tools/ops/sync_swanlab.py', execute.call_args.args[0])
        self.assertFalse((directory/'final_sync.exitcode').exists())

    def test_eval_upload_runs_only_after_training_finished(self):
        directory = self.root/'run'
        directory.mkdir()
        (directory/'swanlab_sync_status.json').write_text('{"status":"finished"}')
        (directory/'final_sync.exitcode').write_text('0\n')
        (directory/'upload_status.json').write_text('{"status":"verified"}')
        with patch.object(queue, 'execute') as execute:
            queue.sync_once(directory, directory/'eval_bundle.json', {'target': {}}, lambda *_: None)
        self.assertEqual(execute.call_count, 1)
        self.assertIn('tools/ops/upload_eval_swanlab.py', execute.call_args.args[0])

    def test_failed_auth_empty_sdk_directory_does_not_block_real_cache_drain(self):
        directory = self.root/'run'
        (directory/'swanlab/run-failed-auth').mkdir(parents=True)
        valid = directory/'swanlab/run-valid'
        valid.mkdir()
        (valid/'run-original.swanlab').write_bytes(b'record')
        (directory/'upload_status.json').write_text('{"status":"verified"}')
        payload = dict(target=dict(run_id='original', name='A_original',
                                  project='Foothold_with_SuperMimic', workspace='yagami_light'))
        with patch.object(queue, 'execute') as execute:
            queue.sync_once(directory, directory/'eval_bundle.json', payload, lambda *_: None)
        self.assertEqual(execute.call_count, 3)  # CSV, valid SDK cache, Eval
        self.assertIn(str(valid), execute.call_args_list[1].args[0])
        self.assertEqual((directory/'final_sync.exitcode').read_text().strip(), '0')
        self.assertTrue((directory/'swanlab/run-failed-auth').is_dir())


if __name__ == '__main__':
    unittest.main()
