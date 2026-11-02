"""落点/首步专项的兼容性、归一化、随机起步和预算；CPU不启动训练或上传。"""

import copy
import json
import subprocess
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch

from configs.s54_flat_reward import FlatWalkRewardCfg
from configs.walk_training import load_adaptive_plan
from scripts.adaptive_walk import _evaluate, pilot_iterations, run_adaptive
from scripts.train_4060 import Supervisor
from scripts.swanlab_metrics import compact_metrics
from tasks.foot_placement.first_step_metrics import FirstStepMetrics, summarize_first_steps
from tasks.foot_placement.rewards import compute_foot_placement_reward, reward_reference_upper
from tests.unit import test_flat_rewards, test_initial_swing

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / 'configs/walk_precision_1h.json'


class WalkPrecisionTests(unittest.TestCase):
    def test_only_swing_xy_changes_and_reference_matches_first_step(self):
        original = test_flat_rewards.FlatRewardTests().fixture()
        _, old_terms = compute_foot_placement_reward(**original)
        for first, expected_tracking in ((False, 3.5), (True, 4.0)):
            data = copy.deepcopy(original)
            data['cfg'] = FlatWalkRewardCfg(swing_xy_multiplier=2., first_swing_xy_multiplier=1.5)
            data['flat_state']['first_swing'] = torch.tensor([first])
            reward, terms = compute_foot_placement_reward(**data)
            self.assertAlmostEqual(terms['tracking'].item(), expected_tracking)
            for name in old_terms:
                if name != 'tracking':
                    torch.testing.assert_close(terms[name], old_terms[name], rtol=0, atol=0)
            reference = reward_reference_upper(data['phase'], data['left_is_swing'], .02, data['cfg'],
                                               torch.tensor([first]))
            torch.testing.assert_close(reward, reference)  # 完美状态恰好达到本相位上界。
        with self.assertRaises(ValueError):
            reward_reference_upper(torch.tensor([.25]), torch.tensor([True]), .02,
                                   FlatWalkRewardCfg(first_swing_xy_multiplier=1.5))

    def test_half_cycle_counter_independent_of_command_resampling_and_partial_reset(self):
        from tasks.foot_placement.goal_sampler import FootPlacementGoalSampler
        poses = test_initial_swing.InitialSwingTests().poses(4)
        sampler = FootPlacementGoalSampler(4, 'cpu', initial_swing_side='right',
                                          left_gait_frequency=.5, right_gait_frequency=1.5)
        sampler.reset(*poses)
        self.assertTrue((sampler.completed_half_cycles == 0).all())
        for _ in range(17):
            sampler.step(.02, *poses)
        self.assertEqual(sampler.completed_half_cycles.tolist(), [1, 1, 1, 1])
        self.assertTrue((sampler._command_half_cycles == 0).all())
        sampler.reset(*poses, env_ids=torch.tensor([1, 3]))
        self.assertEqual(sampler.completed_half_cycles.tolist(), [1, 0, 1, 0])

    def test_coarse_xy_restores_distant_signal_without_weakening_support_gate(self):
        from configs.walk_training import validate_profile
        data = test_flat_rewards.FlatRewardTests().fixture()
        data['left_pos_w'][:, 0] -= .30
        _, narrow = compute_foot_placement_reward(**data)
        data['cfg'] = FlatWalkRewardCfg(swing_xy_coarse_weight=.25)
        _, coarse = compute_foot_placement_reward(**data)
        self.assertGreater(coarse['tracking'].item(), narrow['tracking'].item())
        data['right_force_w'].zero_()
        _, hopping = compute_foot_placement_reward(**data)
        self.assertEqual(hopping['tracking'].item(), 0.)
        for override in ({'swing_xy_coarse_weight': 1.1}, {'swing_xy_coarse_sharpness': 0}):
            with self.assertRaises(ValueError):
                validate_profile({'reward_cfg': {'profile': 'flat_walk_v2', **override}})

    def test_first_metrics_include_fall_and_skip_unmeasured_xy(self):
        m = FirstStepMetrics(4, 'cpu')
        first, valid = torch.ones(4, dtype=torch.bool), torch.tensor([True, True, True, False])
        z, support = torch.zeros(4), torch.full((4,), 600.)
        m.observe(first, torch.full((4,), .5), z, support, torch.full((4,), .03), z,
                  valid, 20., .03)
        # 环境1末段误差大，环境2中途失去支撑，环境3提前摔倒，没有XY样本。
        m.support[2] = False
        m.observe(first, torch.full((4,), .9), support, support, z,
                  torch.tensor([.01, .08, .01, 0.]), valid, 20., .03)
        logs = m.finish(first, valid, torch.tensor([True, False, True, False]))
        self.assertEqual(logs['Startup/left_success'].tolist(), [1., 0.])
        self.assertEqual(logs['Startup/right_success'].tolist(), [0., 0.])
        result = summarize_first_steps({k: v.sum().item() for k, v in logs.items()},
                                       {k: v.numel() for k, v in logs.items()})
        self.assertEqual(result['right']['count'], 2)
        self.assertEqual(result['right']['xy_measured_count'], 1)
        m.reset(torch.tensor([0, 3]))
        self.assertEqual(m.xy_count.tolist(), [0., 1., 1., 0.])
        self.assertTrue(m.support[0])

    def test_pilot_evaluation_has_fixed_cases_seed_first_step_gates(self):
        plan = load_adaptive_plan(PLAN)
        with tempfile.TemporaryDirectory() as tmp:
            commands = []

            def child(command, log, timeout=None):
                commands.append(command)
                result: dict[str, Any] = dict(survival_fraction=1., half_cycle_success_left_right=[.9, .9],
                    consecutive_8_success_fraction=.9, late_xy_error_m=.02, mean_upright=.99,
                    max_abs_heading_error_rad=.05, mean_reference_forward_velocity=.9,
                    command_nominal_forward_speed_m_s=1., terminated=0, truncated=1,
                    first_step={side: dict(count=12, success_rate=.9, late_xy_error_m=.02)
                                for side in ('left', 'right')})
                result['first_step']['right']['success_rate'] = .5
                output = Path(command[command.index('--output') + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(result))
                return 0, [], []

            supervisor = SimpleNamespace(directory=Path(tmp), state=dict(chunk=1, updates=0, adaptive_eval_counter=2),
                stopping=False, deadline=float('inf'), run_child=child, log=lambda *args: None,
                args=Namespace(adaptive_plan_config=plan, eval_envs=16, eval_steps=1000, seed=42,
                               isaac_python=Path('/unused/python')))
            with redirect_stdout(StringIO()):
                summary = _evaluate(supervisor, plan['phases'][0], Path('source.pt'), 0, tag='baseline')
            assert summary is not None
            self.assertFalse(summary['passed'])
            self.assertEqual(len(commands), 3)
            for command in commands:
                self.assertIn('--fast', command)
                self.assertEqual(command[command.index('--seed') + 1], '42')
                profile = json.loads(Path(command[command.index('--task-profile') + 1]).read_text())
                self.assertEqual(profile['episode_length_s'], 10.)  # 短训练episode不能替代连续行走评估。
                self.assertIsNone(profile['command_groups'])
            self.assertAlmostEqual(summary['dashboard']['Eval/first_right_success_min'], .5)

    def test_budget_caps_tail_and_dry_run_has_no_side_effects(self):
        self.assertEqual(pilot_iterations(500, 1000, 1035, 5.), 200)
        self.assertEqual(pilot_iterations(500, 1000, 34, 5.), 0)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'uncreated'
            checkpoint = Path(tmp) / 'selected.pt'
            checkpoint.write_text('dry-run')
            result = subprocess.run(['bash', str(ROOT / 'scripts/train_walk_12h.sh'), '--dry-run',
                                     '--adaptive-plan', str(PLAN), '--hours', '1', '--chunk-iterations', '500',
                                     '--eval-envs', '16', '--eval-steps', '1000',
                                     '--output-dir', str(output), '--checkpoint', str(checkpoint)],
                                    capture_output=True, text=True, check=True)
            preview = json.loads(result.stdout)
            self.assertFalse(output.exists())
            self.assertEqual(preview['max_hours'], 1.)
            self.assertEqual(preview['eval_envs'], 16)
            self.assertTrue(preview['evaluation']['baseline'])
            self.assertEqual(preview['plans'][0]['command'][-1], str(checkpoint))
            self.assertEqual(preview['plans'][0]['groups'], [dict(name='retain', count=410),
                dict(name='frontier', count=1229), dict(name='stride', count=409)])

    def test_invalid_evaluation_settings_and_compact_startup_metrics(self):
        plan = json.loads(PLAN.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'plan.json'
            for bad in ({'typo': True}, {'fast': 'true'}, {'reserve_seconds': -1}):
                file.write_text(json.dumps({**plan, 'evaluation': bad}))
                with self.assertRaises(ValueError):
                    load_adaptive_plan(file)
        raw = {'straight/Startup/right_success': .8, 'Eval/first_left_xy_max_m': .04,
               'Eval/adaptive/case/first_left_xy': .1}
        self.assertEqual(compact_metrics(raw), {'Startup/right_success': .8, 'Eval/first_left_xy_max_m': .04})

    def test_baseline_and_final_comparison_share_budget_without_promoting_baseline(self):
        plan = load_adaptive_plan(PLAN)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = Namespace(max_hours=1, adaptive_plan_config=plan, command_schedule_config=None,
                isaac_python=Path('/unused/python'), max_iterations=2, chunk_iterations=500,
                rollout_steps=64, mini_batches=4, ablation='none', normalization_prior_rollouts=0,
                seed=42, print_every=10, step_length_range=None, learning_rate=1e-4)
            state: dict[str, Any] = dict(stages=['straight'], stage_index=0, chunk=0, updates=0, samples=0, num_envs=2048,
                         checkpoint=str(root/'initial.pt'), checkpoint_stage='straight')
            supervisor = Supervisor(args, root, state)
            summary = dict(passed=False, full=True, checkpoint='evaluated.pt',
                points=[dict(case=[.3, 1.5, 1.5], metrics=dict(late_xy_error_m=.04))])

            def child(command, log, metrics=None, on_row=None, timeout=None):
                assert timeout is not None
                self.assertLess(timeout, 3600 - 180)
                directory = Path(command[command.index('--output_dir') + 1])
                directory.mkdir(parents=True)
                (directory/'model_final.pt').write_text('fixture')
                rows = [dict(metrics={'Perf/collection_time': 3., 'Perf/learning_time': .2}) for _ in range(2)]
                assert on_row is not None
                for row in rows:
                    on_row(row)
                return 0, rows, []

            with patch.object(supervisor, 'run_child', side_effect=child), redirect_stdout(StringIO()), \
                 patch('scripts.adaptive_walk._evaluate', return_value=summary) as evaluate:
                run_adaptive(supervisor)
            self.assertEqual(evaluate.call_count, 2)
            self.assertEqual(evaluate.call_args_list[0].kwargs['tag'], 'baseline')
            self.assertEqual(state['updates'], 2)
            self.assertAlmostEqual(state['ppo_seconds'], 6.4)
            self.assertNotIn('adaptive_qualified', state)
            self.assertTrue((root/'pilot_comparison.json').is_file())


if __name__ == '__main__':
    unittest.main()
