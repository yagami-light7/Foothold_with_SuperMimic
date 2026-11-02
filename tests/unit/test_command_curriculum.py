"""四阶段课程调度的CPU回归；模拟子进程，不运行仿真、PPO或SwanLab上传。"""

import sys
from pathlib import Path

# 支持直接运行测试文件，无需额外设置PYTHONPATH。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import copy
import json
import subprocess
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from scripts.command_curriculum import (apply_command_phase, command_phase_at,
                                       configure_command_schedule, load_command_schedule,
                                       restore_command_schedule)
from scripts.train_4060 import Supervisor

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / 'configs/walk_4stage_12h.json'


class CommandCurriculumTests(unittest.TestCase):
    def setUp(self):
        self.phases = load_command_schedule(PLAN)

    def test_exact_boundaries_and_unbounded_last_phase(self):
        for updates, expected in ((0, 0), (2999, 0), (3000, 1), (5999, 1),
                                  (6000, 2), (8999, 2), (9000, 3), (12000, 3), (16000, 3)):
            phase = command_phase_at(self.phases, updates, 'hold-last')
            self.assertEqual(phase['index'], expected)
            if expected == 3:
                self.assertIsNone(phase['remaining_updates'])
            else:
                self.assertEqual(phase['remaining_updates'], (expected + 1) * 3000 - updates)
        self.assertEqual(command_phase_at(self.phases, 11999, 'stop')['remaining_updates'], 1)
        with self.assertRaises(ValueError):
            command_phase_at(self.phases, 12000, 'stop')

    def test_budget_modes_and_frequency_overrides(self):
        args = Namespace(command_schedule_config=self.phases, schedule_tail='hold-last',
                         start_stage='straight', max_iterations=50000)
        configure_command_schedule(args)
        self.assertEqual(args.max_iterations, 50000)
        phase = command_phase_at(self.phases, 6000, 'hold-last')
        apply_command_phase(args, phase)
        self.assertEqual(args.step_length_range, [.05, .4])
        self.assertEqual(args.left_gait_frequency_range, [.5, 1.8])
        self.assertEqual(args.right_gait_frequency_range, [.5, 1.8])
        self.assertIsNone(args.left_gait_frequency)
        self.assertTrue(args.observe_gait_frequency)
        args.max_iterations = 12000
        with self.assertRaisesRegex(ValueError, 'hold-last'):
            configure_command_schedule(args)
        args.schedule_tail = 'stop'
        configure_command_schedule(args)
        self.assertEqual(args.max_iterations, 12000)

    def test_invalid_units_and_ranges_are_rejected(self):
        original = json.loads(PLAN.read_text())
        for field, value in (('length_unit', 'cm'), ('frequency_unit', 'steps/s')):
            data = copy.deepcopy(original)
            data[field] = value
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / 'plan.json'
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    load_command_schedule(path)
        for key, value in (('updates', 0), ('updates', True), ('step_length_range', [.3, .05]),
                           ('frequency_range', [0, 2]), ('frequency_range', [.5, float('nan')])):
            data = copy.deepcopy(original)
            data['phases'][0][key] = value
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / 'plan.json'
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    load_command_schedule(path)

    def test_resume_restores_snapshot_and_rejects_changed_plan(self):
        previous = {'command_schedule_config': self.phases, 'schedule_tail': 'hold-last'}
        args = Namespace(command_schedule_config=None, schedule_tail=None)
        restore_command_schedule(args, previous)
        self.assertEqual(args.command_schedule_config, self.phases)
        self.assertEqual(args.schedule_tail, 'hold-last')
        args.command_schedule_config = copy.deepcopy(self.phases)
        args.command_schedule_config[1]['step_length_range'][1] = .35
        with self.assertRaisesRegex(ValueError, 'resume-dir'):
            restore_command_schedule(args, previous)

    def test_resume_preserves_used_wall_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            args = Namespace(max_hours=12, command_schedule_config=self.phases)
            state = {'wall_budget_seconds_used': 3 * 3600}
            with patch('scripts.train_4060.time.monotonic', return_value=100.):
                supervisor = Supervisor(args, Path(temp), state)
            self.assertEqual(supervisor.deadline, 100 + 9 * 3600)
            with patch('scripts.train_4060.time.monotonic', return_value=160.):
                supervisor.persist()
            self.assertEqual(state['wall_budget_seconds_used'], 3 * 3600 + 60)
            with patch('scripts.train_4060.time.monotonic', return_value=1000.):
                resumed = Supervisor(args, Path(temp), state)
            self.assertEqual(resumed.deadline, 1000 + 9 * 3600 - 60)

    def test_four_phases_full_resume_and_failed_eval_does_not_block(self):
        phases = copy.deepcopy(self.phases)
        for phase in phases:
            phase['updates'] = 2
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            source = directory / 'initial.pt'
            source.write_text('fixture; no model is executed')
            args = Namespace(max_hours=12, command_schedule_config=phases, schedule_tail='hold-last',
                             max_iterations=11, chunk_iterations=50000,
                             isaac_python=Path('/unused/python'), rollout_steps=64, mini_batches=4,
                             ablation='none', normalization_prior_rollouts=0, seed=42, print_every=1,
                             eval_envs=2, eval_steps=1000, eval_every_chunks=1, eval_frequency_corners=True,
                             step_length_range=None, learning_rate=1e-4, continue_after_pass=True)
            state = {'stages': ['straight'], 'stage_index': 0, 'chunk': 0, 'updates': 0,
                     'samples': 0, 'num_envs': 2, 'pass_streak': 0,
                     'checkpoint': str(source), 'checkpoint_stage': 'straight'}
            supervisor = Supervisor(args, directory, state)
            train_commands = []
            seen_metrics = []

            def child(command, log, metrics=None, on_row=None, timeout=None):
                if '--output_dir' in command:
                    train_commands.append(command)
                    output = Path(command[command.index('--output_dir') + 1])
                    output.mkdir(parents=True)
                    (output / 'model_final.pt').write_text('fixture')
                    count = int(command[command.index('--max_iterations') + 1])
                    rows = [{'metrics': {'Reward/total': .1}} for _ in range(count)]
                    assert on_row is not None
                    for row in rows:
                        on_row(row)
                    return 0, rows, []
                # 故意让所有阶段eval失败：固定3000次计划仍应切换，不能误报通过。
                result = {'survival_fraction': .1, 'mean_upright': .9,
                          'minimum_valid_root_height': .8, 'mean_abs_lateral_velocity': .2,
                          'mean_abs_yaw_rate': .2, 'mean_forward_velocity': .1,
                          'mean_reference_forward_velocity': .1, 'half_cycle_success_left_right': [0., 0.],
                          'consecutive_8_success_fraction': 0., 'late_xy_error_m': .1,
                          'max_abs_heading_error_rad': .3, 'heading_valid_samples': 100,
                          'terminated': 4, 'truncated': 0,
                          'env_cfg': {'left_gait_frequency_range': args.left_gait_frequency_range,
                                      'right_gait_frequency_range': args.right_gait_frequency_range}}
                output = Path(command[command.index('--output') + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(result))
                return 0, [], []

            with patch.object(supervisor, 'run_child', side_effect=child), \
                    patch.object(supervisor, 'log', side_effect=lambda values, step: seen_metrics.append((step, values))), \
                    redirect_stdout(StringIO()):
                supervisor.curriculum()
            self.assertEqual([int(c[c.index('--max_iterations') + 1]) for c in train_commands], [2, 2, 2, 5])
            for i, command in enumerate(train_commands):
                self.assertNotIn('--warm_start', command)
                expected = source if i == 0 else directory / 'training' / f'{i-1:05d}_walk_{i}' / 'model_final.pt'
                self.assertEqual(command[command.index('--resume') + 1], str(expected))
                self.assertEqual(float(command[command.index('--step_length_range') + 2]), phases[i]['step_length_range'][1])
                self.assertEqual(float(command[command.index('--left-gait-frequency-range') + 2]), phases[i]['frequency_range'][1])
            training = [(step, v) for step, v in seen_metrics if 'Progress/command_phase' in v]
            self.assertEqual([step for step, _ in training], list(range(1, 12)))
            self.assertEqual([v['Progress/command_phase'] for _, v in training], [1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 4])
            self.assertEqual(state['command_phase_updates'], 5)
            self.assertTrue(state['schedule_minimum_updates_completed'])
            self.assertEqual(state['samples'], 11 * 2 * 64)
            self.assertEqual(len(state['command_phase_evaluations']), 4)
            self.assertTrue(all(not r['passed'] for r in state['command_phase_evaluations'].values()))
            self.assertNotIn('best_passed', state)

    def test_launcher_custom_checkpoint_preview_creates_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'custom checkpoint.pt'
            source.write_text('preview fixture')
            output = Path(temp) / 'new_run'
            command = ['bash', str(ROOT / 'scripts/train_walk_12h.sh'), '--checkpoint', str(source),
                       '--output-dir', str(output), '--dry-run']
            result = subprocess.run(command, cwd=temp, text=True, capture_output=True, check=True)
            preview = json.loads(result.stdout)
            self.assertEqual(preview['max_hours'], 12)
            self.assertEqual(preview['schedule_tail'], 'hold-last')
            self.assertEqual(len(preview['plans']), 4)
            self.assertEqual([p['max_updates_this_process'] for p in preview['plans']], [3000, 3000, 3000, 41000])
            first = preview['plans'][0]['command']
            self.assertEqual(first[first.index('--resume') + 1], str(source))
            self.assertEqual(preview['mode'], 'online')
            self.assertFalse(output.exists())

    def test_hard_region_plan_preserves_resume_and_targets_the_difficult_range(self):
        # 通过真实shell入口检查新实验的范围和完整恢复语义，不执行GPU/PPO/上传。
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'custom checkpoint.pt'
            source.write_text('preview fixture')
            output = Path(temp) / 'new_run'
            result = subprocess.run(['bash', str(ROOT / 'scripts/train_walk_12h.sh'),
                                     '--adaptive-plan', str(ROOT / 'configs/walk_hard_region.json'),
                                     '--hours', '3', '--chunk-iterations', '250',
                                     '--checkpoint', str(source), '--output-dir', str(output),
                                     '--dry-run'], cwd=temp, text=True, capture_output=True, check=True)
            preview = json.loads(result.stdout)
            self.assertEqual(preview['max_hours'], 3)
            self.assertEqual(preview['mode'], 'online')
            self.assertEqual([p['phase'] for p in preview['plans']], ['bridge', 'frontier', 'hard', 'robust'])
            first = preview['plans'][0]['command']
            self.assertEqual(first[first.index('--resume') + 1], str(source))
            self.assertNotIn('--warm_start', first)
            self.assertEqual(first[first.index('--num_envs') + 1], '2048')
            self.assertEqual(first[first.index('--max_iterations') + 1], '250')
            self.assertIn('--task-profile', first)
            self.assertEqual(preview['plans'][-1]['profile']['command_groups'][-1]['fraction'], .2)
            self.assertEqual(sum(g['count'] for g in preview['plans'][0]['groups']), 2048)
            self.assertFalse(output.exists())

    def test_resume_preview_starts_mid_phase_and_keeps_cloud_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            source = directory / 'checkpoint.pt'
            source.write_text('preview fixture')
            state = {'stages': ['straight'], 'stage_index': 0, 'chunk': 2, 'updates': 4500,
                     'samples': 4500 * 2048 * 64, 'num_envs': 2048, 'pass_streak': 0,
                     'checkpoint': str(source), 'checkpoint_stage': 'straight',
                     'wall_budget_seconds_used': 4 * 3600, 'swanlab_id': 'unchanged_id'}
            previous = {'command_schedule_config': self.phases, 'schedule_tail': 'hold-last',
                        'max_hours': 12, 'step_length_range': None, 'seed': 42,
                        'rollout_steps': 64, 'mini_batches': 4, 'project': 'test', 'workspace': None,
                        'mode': 'offline', 'observe_gait_frequency': True}
            (directory / 'state.json').write_text(json.dumps(state))
            (directory / 'launcher.json').write_text(json.dumps(previous))
            for script, chunk in (('train_walk_12h.sh', 1500), ('train_walk_retention_12h.sh', 1500)):
                with self.subTest(script=script):
                    result = subprocess.run(['bash', str(ROOT / 'scripts' / script),
                                             '--resume-dir', temp, '--dry-run'], cwd=temp,
                                            text=True, capture_output=True, check=True)
                    preview = json.loads(result.stdout)
                    self.assertEqual(preview['remaining_budget_hours'], 8)
                    self.assertEqual(preview['resume_swanlab_id'], 'unchanged_id')
                    self.assertEqual(preview['mode'], 'offline')
                    self.assertEqual(preview['plans'][0]['phase'], 'walk_2')
                    self.assertEqual(preview['plans'][0]['max_updates_this_process'], chunk)
            # dry-run不能把已保存的state/launcher或SwanLab身份覆盖掉。
            self.assertEqual(json.loads((directory / 'state.json').read_text()), state)
            self.assertEqual(json.loads((directory / 'launcher.json').read_text()), previous)


if __name__ == '__main__':
    unittest.main()
