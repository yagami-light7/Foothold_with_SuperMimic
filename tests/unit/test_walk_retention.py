"""半小时续训与能力保持筛选；不用CUDA或SwanLab，防止再推荐角点退化的最终模型。"""

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

from configs.walk_training import coverage_cells, group_counts, load_adaptive_plan
from scripts.adaptive_walk import _evaluate, run_adaptive
from scripts.retention_selection import assess_retention, update_retention_selection
from scripts.rollout_coverage import record_rollout_coverage
from scripts.swanlab_metrics import compact_metrics
from scripts.train_4060 import Supervisor

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / 'configs/walk_retention_30m.json'


def evaluation(phase, checkpoint='initial.pt', updates=0) -> dict[str, Any]:
    """初始角点不够准，但有稳定支撑；资格未通过也能作为能力保持参照。"""
    return dict(phase=phase['name'], checkpoint=checkpoint, updates=updates, full=True, passed=False,
        points=[dict(case=case, required=True, metrics=dict(
            survival_fraction=1., late_xy_error_m=.12 if case[0] == .4 else .03,
            half_cycle_success_left_right=[.1, .1] if case[0] == .4 else [.9, .9],
            both_air_fraction=.005, stance_hop_fraction=.005, contact_foot_slip_speed_m_s=.05,
            first_step={side: dict(count=10, xy_measured_count=10, success_rate=.8, late_xy_error_m=.02)
                        for side in ('left', 'right')})) for case in phase['required_cases']])


class WalkRetentionTests(unittest.TestCase):
    def setUp(self):
        self.plan = load_adaptive_plan(PLAN)
        # 筛选器是可选工具；本轮正式计划关闭eval，不启用筛选。
        self.phase = copy.deepcopy(self.plan['phases'][0])
        self.phase['checkpoint_selection'] = dict(minimum_survival=.95, max_xy_regression_fraction=.2,
            absolute_xy_tolerance_m=.003, max_success_drop=.05, minimum_score_improvement=.005)
        self.baseline = evaluation(self.phase)

    def test_baseline_eligible_without_claiming_full_task_success(self):
        result = assess_retention(self.baseline, self.baseline, self.phase)
        self.assertTrue(result['eligible'])
        self.assertEqual(result['relative_xy_score'], 1.)
        self.assertFalse(self.baseline['passed'])

    def test_corner_survival_outweighs_better_xy_and_reward(self):
        candidate = copy.deepcopy(self.baseline)
        for point in candidate['points']:
            point['metrics']['late_xy_error_m'] *= .7
        candidate['points'][-1]['metrics']['survival_fraction'] = .9375
        candidate['points'][-1]['metrics']['normalized_reward'] = 1.
        result = assess_retention(candidate, self.baseline, self.phase)
        self.assertFalse(result['eligible'])
        self.assertFalse(result['checks']['0.4/2/2/survival'])

    def test_reject_common_task_first_step_or_support_regression(self):
        variants = [('late_xy_error_m', .05), ('half_cycle_success_left_right', [.8, .9]),
                    ('both_air_fraction', .03), ('stance_hop_fraction', .03),
                    ('contact_foot_slip_speed_m_s', .11)]
        for key, value in variants:
            with self.subTest(key=key):
                candidate = copy.deepcopy(self.baseline)
                candidate['points'][0]['metrics'][key] = value
                self.assertFalse(assess_retention(candidate, self.baseline, self.phase)['eligible'])
        for field, value in [('late_xy_error_m', .04), ('success_rate', .7),
                             ('xy_measured_count', 0), ('late_xy_error_m', None)]:
            with self.subTest(field=field, value=value):
                candidate = copy.deepcopy(self.baseline)
                candidate['points'][0]['metrics']['first_step']['right'][field] = value
                self.assertFalse(assess_retention(candidate, self.baseline, self.phase)['eligible'])

    def test_missing_or_nonfinite_evaluations_never_select(self):
        candidate = copy.deepcopy(self.baseline)
        candidate['points'].pop()
        self.assertFalse(assess_retention(candidate, self.baseline, self.phase)['eligible'])
        candidate = copy.deepcopy(self.baseline)
        candidate['points'][0]['metrics']['late_xy_error_m'] = float('nan')
        result = assess_retention(candidate, self.baseline, self.phase)
        self.assertFalse(result['eligible'])
        self.assertIsNone(result['relative_xy_score'])
        json.dumps(result, allow_nan=False)

    def test_selected_path_preserves_metadata_and_rejected_final_does_not_replace_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = root / 'original' / 'model_final.pt'; initial.parent.mkdir(); initial.write_bytes(b'baseline')
            trained = root / 'training' / 'model_final.pt'; trained.parent.mkdir(); trained.write_bytes(b'candidate')
            (initial.parent / 'env_cfg.json').write_text('{}')
            baseline = evaluation(self.phase, str(initial))
            supervisor = SimpleNamespace(directory=root, state=dict(updates=0, baseline_evaluation=baseline),
                                         log=lambda *args: None, persist=lambda: None)
            with redirect_stdout(StringIO()):
                update_retention_selection(supervisor, self.phase, baseline)
                better = evaluation(self.phase, str(trained), 250)
                for point in better['points']:
                    point['metrics']['late_xy_error_m'] *= .95
                supervisor.state['updates'] = 250
                update_retention_selection(supervisor, self.phase, better)
                worse = copy.deepcopy(better); worse['updates'] = 500
                rejected = root / 'rejected.pt'; rejected.write_bytes(b'failed_candidate')
                worse['checkpoint'] = str(rejected)
                worse['points'][-1]['metrics']['survival_fraction'] = 0.
                supervisor.state['updates'] = 500
                update_retention_selection(supervisor, self.phase, worse)
            saved = json.loads((root / 'best_checkpoint.json').read_text())
            self.assertEqual(saved['best']['checkpoint'], str(trained))
            self.assertEqual(saved['best']['updates'], 250)
            self.assertEqual(saved['latest_evaluation']['updates'], 500)
            self.assertFalse(saved['latest_evaluation']['eligible'])
            self.assertEqual(initial.read_bytes(), b'baseline')
            self.assertEqual(saved['best']['qualification'], 'retention_only_not_full_task_success')

    def test_current_entrypoint_pretrained_source_reward_budget_and_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'uncreated'
            result = subprocess.run(['bash', str(ROOT / 'scripts/train_walk_retention_12h.sh'),
                '--dry-run', '--output-dir', str(output)], capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            self.assertFalse(output.exists())
            self.assertEqual(data['max_hours'], 12.)
            self.assertEqual(data['remaining_budget_hours'], 12.)
            self.assertFalse(data['evaluation']['enabled'])
            self.assertFalse(data['evaluation']['baseline'])
            self.assertEqual(data['evaluation']['reserve_seconds'], 0)
            step = data['plans'][0]
            self.assertEqual(step['max_updates_this_process'], 50000)
            self.assertEqual(step['command'][-1], str(ROOT / 'pretrained/s54_walk_12h_20261102/model.pt'))
            self.assertEqual(step['command'][step['command'].index('--learning_rate') + 1], '2e-05')
            self.assertEqual(step['profile']['episode_length_s'], 10.)
            self.assertEqual(sum(g['count'] for g in step['groups']), 2048)
            self.assertEqual(len(step['coverage_cells']), 15)
            self.assertEqual(min(c['environments'] for c in step['coverage_cells']), 41)
            self.assertEqual(step['groups'][-1], {'name':'corner', 'count':205})
            old_profile = json.loads((ROOT / 'configs/walk_precision_1h.json').read_text())['phases'][0]['profile']
            old_profile['reward_cfg']['first_swing_xy_multiplier'] = 2.
            self.assertEqual(step['profile']['reward_cfg'], old_profile['reward_cfg'])
            self.assertIsNone(step['checkpoint_selection'])

    def test_selection_configuration_validation(self):
        mutations = [lambda p: p['evaluation'].update(baseline=False),
                     lambda p: p['evaluation'].update(fixed_seed=False),
                     lambda p: p['phases'][0]['required_cases'].pop(),
                     lambda p: p['phases'][0]['checkpoint_selection'].update(minimum_survival=1.1),
                     lambda p: p['phases'][0]['checkpoint_selection'].update(minimum_score_improvement=0)]
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'plan.json'
            for mutate in mutations:
                plan = copy.deepcopy(self.plan)
                plan['phases'][0] = copy.deepcopy(self.phase)
                plan['evaluation'].update(enabled=True, baseline=True, fixed_seed=True)
                mutate(plan)
                file.write_text(json.dumps(plan))
                with self.assertRaises(ValueError):
                    load_adaptive_plan(file)

    def test_corner_always_evaluated_and_in_compact_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            commands = []
            def child(command, log, timeout=None):
                commands.append(command)
                length = float(command[command.index('--step_length_range') + 1])
                metrics = dict(survival_fraction=0. if length == .4 else 1.,
                    half_cycle_success_left_right=[.9, .9], consecutive_8_success_fraction=.9,
                    late_xy_error_m=.02, mean_upright=.99, max_abs_heading_error_rad=.05,
                    mean_reference_forward_velocity=1., command_nominal_forward_speed_m_s=1.,
                    terminated=1 if length == .4 else 0, truncated=0)
                output = Path(command[command.index('--output') + 1])
                output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(metrics))
                return 0, [], []
            supervisor = SimpleNamespace(directory=Path(tmp), state=dict(chunk=1, updates=250), stopping=False,
                deadline=float('inf'), run_child=child, log=lambda *args: None,
                args=Namespace(adaptive_plan_config=self.plan, eval_envs=16, eval_steps=1000, seed=42,
                               isaac_python=Path('/unused/python')))
            supervisor.args.adaptive_plan_config = copy.deepcopy(self.plan)
            supervisor.args.adaptive_plan_config['evaluation']['enabled'] = True
            with redirect_stdout(StringIO()):
                result = _evaluate(supervisor, self.phase, Path('candidate.pt'), 0)
            assert result is not None
            self.assertEqual(len(commands), 4)
            self.assertFalse(result['passed'])
            cloud = compact_metrics(result['dashboard'])
            self.assertEqual(cloud['Eval/corner_survival_fraction'], 0.)
            self.assertEqual(cloud['Eval/corner_xy_error_m'], .02)

    def test_baseline_and_final_selection_run_inside_existing_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = root / 'initial.pt'; initial.write_text('fixture')
            plan = copy.deepcopy(self.plan)
            plan['phases'][0] = copy.deepcopy(self.phase)
            plan['evaluation'].update(enabled=True, baseline=True, fixed_seed=True, reserve_seconds=180)
            args = Namespace(max_hours=.5, adaptive_plan_config=plan, command_schedule_config=None,
                isaac_python=Path('/unused/python'), max_iterations=2, chunk_iterations=250, rollout_steps=64,
                mini_batches=4, ablation='none', normalization_prior_rollouts=0, seed=42, print_every=10,
                step_length_range=None, learning_rate=1e-4)
            state: dict[str, Any] = dict(stages=['straight'], stage_index=0, chunk=0, updates=0, samples=0,
                num_envs=2048, checkpoint=str(initial), checkpoint_stage='straight')
            supervisor = Supervisor(args, root, state)
            def evaluate(supervisor, phase, checkpoint, chunk, **kwargs):
                summary = evaluation(phase, str(checkpoint), state['updates'])
                if state['updates']:
                    summary['points'][-1]['metrics']['survival_fraction'] = 0.
                return summary
            def child(command, log, metrics=None, on_row=None, timeout=None):
                assert timeout is not None and on_row is not None
                self.assertLess(timeout, 1800 - 180)
                directory = Path(command[command.index('--output_dir') + 1]); directory.mkdir(parents=True)
                (directory / 'model_final.pt').write_text('candidate')
                metrics = {'Perf/collection_time': 3., 'Perf/learning_time': .2,
                           **{f'Group/{g["name"]}/command_count': n for g,n in zip(
                               self.phase['profile']['command_groups'],
                               group_counts(self.phase['profile']['command_groups'], 2048))}}
                rows = [dict(metrics=metrics) for _ in range(2)]
                for row in rows: on_row(row)
                return 0, rows, []
            with patch.object(supervisor, 'run_child', side_effect=child), \
                 patch('scripts.adaptive_walk._evaluate', side_effect=evaluate), redirect_stdout(StringIO()):
                run_adaptive(supervisor)
            self.assertEqual(state['updates'], 2)
            self.assertEqual(state['retention_best']['checkpoint'], str(initial))
            self.assertFalse(state['retention_last_assessment']['eligible'])
            self.assertNotEqual(state['checkpoint'], str(initial))
            self.assertTrue((root / 'best_checkpoint.json').is_file())

    def test_training_only_never_calls_eval_or_selector_and_keeps_whole_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = Namespace(max_hours=.5, adaptive_plan_config=self.plan, command_schedule_config=None,
                isaac_python=Path('/unused/python'), max_iterations=2, chunk_iterations=50000, rollout_steps=64,
                mini_batches=4, ablation='none', normalization_prior_rollouts=0, seed=42, print_every=10,
                step_length_range=None, learning_rate=1e-4)
            state: dict[str, Any] = dict(stages=['straight'], stage_index=0, chunk=0, updates=0, samples=0,
                num_envs=2048, checkpoint='original.pt', checkpoint_stage='straight')
            supervisor = Supervisor(args, root, state)
            def child(command, log, metrics=None, on_row=None, timeout=None):
                self.assertIsNone(timeout)  # 仅全局1800s截止，无eval预留。
                assert on_row is not None
                directory = Path(command[command.index('--output_dir') + 1]); directory.mkdir(parents=True)
                (directory / 'model_final.pt').write_text('fixture')
                values = {f'Group/{g["name"]}/command_count': n for g,n in zip(
                    self.phase['profile']['command_groups'], group_counts(self.phase['profile']['command_groups'],2048))}
                rows = [dict(metrics=values) for _ in range(2)]
                for row in rows: on_row(row)
                return 0, rows, []
            with patch.object(supervisor, 'run_child', side_effect=child), \
                 patch('scripts.adaptive_walk._evaluate') as evaluate_mock, \
                 patch('scripts.retention_selection.update_retention_selection') as selector, redirect_stdout(StringIO()):
                run_adaptive(supervisor)
            evaluate_mock.assert_not_called(); selector.assert_not_called()
            self.assertFalse((root / 'evaluations').exists())
            self.assertFalse((root / 'best_checkpoint.json').exists())
            self.assertFalse(state['all_adaptive_gates_passed'])
            self.assertEqual(state['rollout_coverage']['incomplete_updates'], 0)
            self.assertEqual(state['rollout_coverage']['last_min_cell_transitions'], 2624)
            self.assertEqual(state['samples'], 262144)
            self.assertFalse(state['evaluation_enabled'])

    def test_grid_holes_zero_quota_warmup_and_duplicate_cells_are_rejected(self):
        self.assertEqual(len(coverage_cells(self.phase, 2048)),15)
        with self.assertRaisesRegex(ValueError,'环境数不足'):
            coverage_cells(self.phase,16)
        for mutation in (
                lambda p: p['profile']['command_groups'][1].update(step_length_range=[.24,.30]),
                lambda p: p['profile']['command_groups'][1].update(coupled=False),
                lambda p: p['profile'].update(command_warmup_half_cycles=4),
                lambda p: p['profile']['command_groups'].append(p['profile']['command_groups'][1])):
            phase = copy.deepcopy(self.phase); mutation(phase)
            with self.assertRaises(ValueError): coverage_cells(phase,2048)

    def test_sampler_covers_every_cell_through_rollout_and_partial_resets(self):
        import torch
        from tasks.foot_placement.goal_sampler import FootPlacementGoalSampler
        groups = self.phase['profile']['command_groups']
        torch.manual_seed(42)
        sampler = FootPlacementGoalSampler(2048,'cpu',command_groups=groups,initial_swing_side='random')
        left = torch.tensor([[0., .095, .061]]).repeat(2048,1)
        right = left.clone(); right[:,1] *= -1
        quat = torch.tensor([[0.,0.,0.,1.]]).repeat(2048,1)
        poses = (left,quat,right,quat)
        sampler.reset(*poses)
        ids = sampler.command_group_ids.clone()
        before = sampler.step_lengths.clone()
        totals = torch.zeros(len(groups))
        for step in range(64):
            if step % 8 == 0:
                sampler.reset(*poses,env_ids=torch.arange(0,2048,3))
            sampler.step(.02,*poses)
            mask = sampler.commands_in_group_bounds()
            self.assertTrue(mask.all())
            totals += torch.bincount(ids[mask],minlength=len(groups))
            torch.testing.assert_close(sampler.command_group_ids,ids)
        self.assertEqual(totals.tolist(),[64*n for n in group_counts(groups,2048)])
        untouched = torch.arange(2048)%3 != 0
        torch.testing.assert_close(sampler.step_lengths[untouched],before[untouched])
        # 审计实际命令，不能只核对静态的组标签。
        sampler.step_lengths[0] = 99.
        sampler.left_gait_frequencies[-1] = float('nan')
        mask = sampler.commands_in_group_bounds()
        self.assertFalse(mask[0]); self.assertFalse(mask[-1])
        sampler.left_gait_frequencies[-1] = 1.9
        self.assertFalse(sampler.commands_in_group_bounds()[-1])

    def test_actual_coverage_missing_or_outside_samples_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            groups = self.phase['profile']['command_groups']
            values = {f'Group/{g["name"]}/command_count': n for g,n in zip(groups,group_counts(groups,2048))}
            supervisor = SimpleNamespace(directory=Path(tmp),state=dict(num_envs=2048,updates=1),
                                         args=Namespace(rollout_steps=64))
            result = record_rollout_coverage(supervisor,self.phase,values)
            self.assertEqual(compact_metrics(result), {'Coverage/cell_fraction':1.,'Coverage/min_cell_transitions':2624})
            totals = json.loads((Path(tmp)/'coverage.json').read_text())
            self.assertEqual(sum(c['transitions'] for c in totals['cells'].values()),131072)
            self.assertEqual(totals['cells']['corner']['transitions'],13120)
            invalid = dict(values); invalid.pop('Group/corner/command_count')
            with self.assertRaisesRegex(RuntimeError,'日志缺失'):
                record_rollout_coverage(supervisor,self.phase,invalid)
            invalid = dict(values); invalid['Group/grid_d25_30_f10_12/command_count'] = 40.
            with self.assertRaisesRegex(RuntimeError,'区间外命令'):
                record_rollout_coverage(supervisor,self.phase,invalid)


if __name__ == '__main__':
    unittest.main()
