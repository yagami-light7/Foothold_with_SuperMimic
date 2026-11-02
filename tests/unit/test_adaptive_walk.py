"""混合命令、评估晋级和断点恢复；CPU测试不启动仿真或上传。"""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout
from io import StringIO
from typing import Any
from dataclasses import replace
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from configs.walk_training import (PROFILE_DEFAULTS, adaptive_gate, clean_profile,
                                   group_counts, load_adaptive_plan, validate_profile)
from scripts.adaptive_walk import record_evaluation, select_phase
from scripts.train_4060 import Supervisor
from tasks.foot_placement.goal_sampler import FootPlacementGoalSampler

PLAN = ROOT / 'configs/walk_hard_region.json'


class AdaptiveWalkTests(unittest.TestCase):
    def setUp(self):
        self.plan = load_adaptive_plan(PLAN)

    def sampler(self, **kwargs):
        n = kwargs.pop('num_envs', 100)
        sampler = FootPlacementGoalSampler(n, 'cpu', **kwargs)
        left = torch.tensor([[0., .095, .061]]).repeat(n, 1)
        right = left.clone(); right[:, 1] *= -1
        quat = torch.tensor([[0., 0., 0., 1.]]).repeat(n, 1)
        poses = (left, quat, right, quat)
        sampler.reset(*poses)
        return sampler, poses

    def test_persistent_groups_partial_reset_and_coupled_cadence(self):
        profile = self.plan['phases'][0]['profile']
        sampler, poses = self.sampler(command_groups=profile['command_groups'])
        self.assertEqual(torch.bincount(sampler.command_group_ids).tolist(), [20, 35, 40, 5])
        self.assertEqual(group_counts(profile['command_groups'], 2048), [410, 717, 819, 102])
        ids_before = sampler.command_group_ids.clone()
        untouched = sampler.step_lengths[:20].clone()
        for _ in range(20):
            sampler.reset(*poses, env_ids=torch.arange(20, 100))
            torch.testing.assert_close(sampler.command_group_ids, ids_before)
            torch.testing.assert_close(sampler.step_lengths[:20], untouched)
            torch.testing.assert_close(sampler.left_gait_frequencies[20:], sampler.right_gait_frequencies[20:])
            self.assertTrue((sampler.step_lengths[95:] == .4).all())
        self.assertTrue((sampler.left_gait_frequencies[:20] != sampler.right_gait_frequencies[:20]).any())

    def test_default_features_preserve_rng_and_legacy_commands(self):
        torch.manual_seed(42)
        sampler, poses = self.sampler()
        rng = torch.random.get_rng_state().clone()
        sampler.reset(*poses)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        for _ in range(100):
            sampler.step(.02, *poses)
        torch.testing.assert_close(sampler.step_lengths, torch.full((100,), .12))
        torch.testing.assert_close(sampler.left_gait_frequencies, torch.ones(100))

    def test_warmup_only_changes_commands_at_boundary_and_reaches_requested_maximum(self):
        sampler, poses = self.sampler(num_envs=4, step_length=.4, gait_frequency=2., command_warmup_half_cycles=4)
        torch.testing.assert_close(sampler.step_lengths, torch.full((4,), .12))
        previous = sampler.step_lengths.clone()
        switches = 0
        for _ in range(200):
            swing = sampler.left_is_swing.clone()
            sampler.step(.02, *poses)
            switched = swing != sampler.left_is_swing
            if switched.any():
                switches += 1
            else:
                torch.testing.assert_close(sampler.step_lengths, previous)
            previous = sampler.step_lengths.clone()
        self.assertGreaterEqual(switches, 4)
        torch.testing.assert_close(sampler.step_lengths, torch.full((4,), .4))
        torch.testing.assert_close(sampler.left_gait_frequencies, torch.full((4,), 2.))

    def test_bounded_resampling_keeps_group_and_phase_progress(self):
        groups = self.plan['phases'][2]['profile']['command_groups']
        sampler, poses = self.sampler(command_groups=groups, command_resample_half_cycles=2)
        changes = 0
        for _ in range(100):
            swing = sampler.left_is_swing.clone()
            old = torch.stack((sampler.step_lengths, sampler.left_gait_frequencies), -1).clone()
            phase = sampler.phase.clone()
            left, right = sampler.left_gait_frequencies.clone(), sampler.right_gait_frequencies.clone()
            from tasks.foot_placement.gait import advance_phase
            expected = advance_phase(phase, .02, left, right)
            sampler.step(.02, *poses)
            new = torch.stack((sampler.step_lengths, sampler.left_gait_frequencies), -1)
            delta = (new-old).abs()
            torch.testing.assert_close(sampler.phase, expected)
            self.assertTrue((delta[:, 0] <= .020001).all())
            self.assertTrue((delta[:, 1] <= .100001).all())
            self.assertTrue((delta[swing == sampler.left_is_swing] == 0).all())
            changes += int((delta > 0).any())
            torch.testing.assert_close(sampler.left_gait_frequencies[20:], sampler.right_gait_frequencies[20:])
            self.assertTrue((sampler.step_lengths[-20:] == .4).all())
        self.assertGreater(changes, 0)

    def test_profile_validation_rejects_unknown_nonfinite_and_unsafe_parameters(self):
        for bad in ({'typo': 1}, {'push_velocity_max': .1}, {'command_resample_half_cycles': 1.5},
                    {'observation_noise_scale': float('nan')}, {'body_mass_scale_range': [0, 1]},
                    {'reward_cfg': {'xy_coarse_weight': 1.1}}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_profile(bad)
        self.assertEqual(validate_profile(clean_profile())['command_groups'], None)

    def test_reward_default_unchanged_and_optional_coarse_kernel_is_bounded(self):
        from tests.unit.test_preflight import evaluate
        from tasks.foot_placement.rewards import FootRewardCfg, reward_reference_upper
        original = FootRewardCfg()
        reward, terms = evaluate(original)
        zero, zero_terms = evaluate(replace(original, xy_coarse_weight=0.))
        torch.testing.assert_close(reward, zero, atol=0, rtol=0)
        for name in terms:
            torch.testing.assert_close(terms[name], zero_terms[name], atol=0, rtol=0)
        widened = replace(original, xy_coarse_weight=.25)
        broad, broad_terms = evaluate(widened)
        self.assertGreater(broad_terms['swing_xy'].item(), terms['swing_xy'].item())
        upper = reward_reference_upper(torch.tensor([.25]),torch.tensor([True]),.02,widened)
        self.assertTrue((broad <= upper).all())
        # 旧奖励的宽核开关仍可复现；新课程显式切换到简化平地奖励。
        self.assertTrue(all(p['profile']['reward_cfg']['profile'] == 'flat_walk_v2' for p in self.plan['phases']))

    def test_profile_roundtrip_and_explicit_fixed_command_clear_saved_groups(self):
        from agents.rsl_rl_ppo_cfg import S54FootPlacementPPORunnerCfg
        from configs.s54_curriculum import configure_stage, restore_task_config
        from configs.walk_training import apply_task_profile
        from isaaclab.utils import to_dict
        from scripts.gait_cli import apply_gait_overrides
        from tasks.foot_placement.foot_placement_env_cfg import S54FootPlacementEnvCfg
        cfg = S54FootPlacementEnvCfg()
        configure_stage(cfg, S54FootPlacementPPORunnerCfg(), 'straight')
        apply_task_profile(cfg, self.plan['phases'][-1]['profile'])
        cfg.observe_gait_frequency = True
        saved = to_dict(cfg)
        restored = S54FootPlacementEnvCfg()
        restore_task_config(restored, saved)
        self.assertEqual(restored.command_groups, cfg.command_groups)
        self.assertEqual(restored.body_mass_scale_range, [.97,1.03])
        explicit = Namespace(left_gait_frequency=2.,right_gait_frequency=2.)
        apply_gait_overrides(restored,explicit)
        self.assertIsNone(restored.command_groups)
        self.assertEqual(restored.command_resample_half_cycles,0)
        self.assertEqual(restored.observation_space,105)
        # 旧模型没有新字段时显式关闭，不能误继承临时扰动配置。
        legacy = {k:v for k,v in saved.items() if k not in PROFILE_DEFAULTS}
        restore_task_config(restored,legacy)
        self.assertIsNone(restored.body_mass_scale_range)
        self.assertEqual(restored.push_velocity_max,0)

    def test_failure_holds_phase_even_after_many_updates(self):
        state = {'updates': 20000}
        for _ in range(10):
            self.assertFalse(record_evaluation(self.plan, state, False, 'failed.pt', True))
        self.assertEqual(select_phase(self.plan, state)[0], 0)
        self.assertNotIn('adaptive_qualified', state)

    def test_promotion_requires_minimum_updates_streak_and_full_grid_pass(self):
        state = {'updates': 1000}
        self.assertFalse(record_evaluation(self.plan, state, True, 'first.pt', True))
        state['updates'] = 2000
        self.assertFalse(record_evaluation(self.plan, state, False, 'fail.pt', True))
        self.assertFalse(record_evaluation(self.plan, state, True, 'quick1.pt'))
        self.assertFalse(record_evaluation(self.plan, state, True, 'quick2.pt'))
        self.assertTrue(record_evaluation(self.plan, state, True, 'full.pt', True))
        self.assertEqual(state['adaptive_phase_index'], 1)
        self.assertEqual(state['adaptive_phase_start_updates'], 2000)
        self.assertEqual(state['adaptive_pass_streak'], 0)
        self.assertFalse(state['adaptive_full_passed'])

    def test_gate_uses_independent_physical_success_and_speed(self):
        good = dict(survival_fraction=1., half_cycle_success_left_right=[.9,.9],
                    consecutive_8_success_fraction=.9, late_xy_error_m=.02, mean_upright=.99,
                    max_abs_heading_error_rad=.1, mean_reference_forward_velocity=.9,
                    command_nominal_forward_speed_m_s=1.6)
        self.assertTrue(adaptive_gate(good)[0])
        for field, bad in [('half_cycle_success_left_right', [.99,.79]), ('survival_fraction', .94),
                           ('late_xy_error_m', .031), ('mean_reference_forward_velocity', .1)]:
            result = {**good, field: bad, 'normalized_return': 1.}
            self.assertFalse(adaptive_gate(result)[0])

    def test_adaptive_budget_resume_and_saved_phase_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = Namespace(max_hours=12, adaptive_plan_config=self.plan)
            state: dict[str, Any] = {'wall_budget_seconds_used': 4*3600}
            with patch('scripts.train_4060.time.monotonic', return_value=100.):
                supervisor = Supervisor(args, root, state)
            self.assertEqual(supervisor.deadline, 100+8*3600)
            with patch('scripts.train_4060.time.monotonic', return_value=160.):
                supervisor.persist()
            self.assertEqual(state['wall_budget_seconds_used'], 4*3600+60)
            source = root / 'saved.pt'; source.write_text('fixture')
            state.update(stages=['straight'],stage_index=0,chunk=4,updates=4500,samples=4500*2048*64,
                         num_envs=2048,checkpoint=str(source),checkpoint_stage='straight',
                         adaptive_phase_index=1,adaptive_phase_start_updates=3000,swanlab_id='same-cloud-run')
            previous = dict(adaptive_plan_config=self.plan,command_schedule_config=None,schedule_tail='stop',
                            max_hours=12,seed=42,rollout_steps=64,mini_batches=4,project='test',workspace=None,
                            mode='offline',observe_gait_frequency=True)
            (root/'state.json').write_text(json.dumps(state))
            (root/'launcher.json').write_text(json.dumps(previous))
            result = subprocess.run(['bash',str(ROOT/'scripts/train_walk_retention_12h.sh'),
                                     '--resume-dir',tmp,'--dry-run'],capture_output=True,text=True,check=True)
            preview = json.loads(result.stdout)
            self.assertEqual(preview['plans'][0]['phase'], 'frontier')
            self.assertEqual(preview['mode'],'offline')
            self.assertEqual(preview['resume_swanlab_id'],'same-cloud-run')
            self.assertAlmostEqual(preview['remaining_budget_hours'],8-1/60)
            self.assertEqual(json.loads((root/'state.json').read_text()),state)
            changed=copy.deepcopy(self.plan)
            changed['phases'][0]['minimum_updates'] += 1
            alternate=root/'different.json';alternate.write_text(json.dumps(changed))
            rejected=subprocess.run(['bash',str(ROOT/'scripts/train_walk_retention_12h.sh'),'--resume-dir',tmp,
                                     '--adaptive-plan',str(alternate),'--dry-run'],capture_output=True,text=True)
            self.assertNotEqual(rejected.returncode,0)
            self.assertIn('resume-dir不能改变自适应课程',rejected.stderr)

    def test_final_qualification_belongs_to_evaluated_checkpoint(self):
        from scripts.adaptive_walk import final_checkpoint_qualified
        state: dict[str,Any] = dict(updates=10000,adaptive_phase_index=3,checkpoint='passed.pt')
        self.assertFalse(record_evaluation(self.plan,state,True,'passed.pt',True))
        self.assertFalse(record_evaluation(self.plan,state,True,'passed.pt'))
        self.assertTrue(final_checkpoint_qualified(self.plan,state))
        state['checkpoint']='newer_unevaluated.pt'
        self.assertFalse(final_checkpoint_qualified(self.plan,state))
        self.assertEqual(state['adaptive_qualified']['robust']['checkpoint'],'passed.pt')

    def test_failed_evaluations_do_not_change_real_training_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            args=Namespace(max_hours=12,adaptive_plan_config=self.plan,command_schedule_config=None,
                           isaac_python=Path('/unused/python'),max_iterations=6,chunk_iterations=2,
                           rollout_steps=64,mini_batches=4,ablation='none',normalization_prior_rollouts=0,
                           seed=42,print_every=1,step_length_range=None,learning_rate=1e-4,
                           eval_envs=2,eval_steps=1000)
            state=dict(stages=['straight'],stage_index=0,chunk=0,updates=0,samples=0,num_envs=2,
                       checkpoint=str(root/'initial.pt'),checkpoint_stage='straight')
            supervisor=Supervisor(args,root,state)
            commands=[]
            def child(command, log, metrics=None, on_row=None, timeout=None):
                commands.append(command)
                output=Path(command[command.index('--output_dir')+1]);output.mkdir(parents=True)
                (output/'model_final.pt').write_text('fixture')
                rows=[{'metrics':{'Reward/total':.1}}]*2
                assert on_row is not None
                for row in rows:
                    on_row(row)
                return 0,rows,[]
            with patch.object(supervisor,'run_child',side_effect=child), redirect_stdout(StringIO()), \
                 patch('scripts.adaptive_walk._evaluate',return_value=dict(passed=False,full=True)):
                supervisor.curriculum()
            self.assertEqual(len(commands),3)
            self.assertEqual(state['adaptive_phase_index'],0)
            self.assertNotIn('adaptive_qualified',state)
            for i,command in enumerate(commands):
                self.assertIn('--resume',command)
                if i:
                    self.assertIn(f'{i-1:05d}_bridge/model_final.pt',command[command.index('--resume')+1])
            self.assertEqual(state['updates'],6)
            self.assertEqual(state['samples'],6*2*64)

    def test_corner_monitor_and_final_continuous_episode_evaluation(self):
        from scripts.adaptive_walk import _evaluate
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            commands=[]
            def child(command, log, timeout=None):
                commands.append(command)
                length=float(command[command.index('--step_length_range')+1])
                left=float(command[command.index('--left-gait-frequency')+1])
                right=float(command[command.index('--right-gait-frequency')+1])
                corner=length==.4 and left==2. and right==2.
                result=dict(survival_fraction=0. if corner else 1.,
                            half_cycle_success_left_right=[.9,.9],consecutive_8_success_fraction=.9,
                            late_xy_error_m=.02,mean_upright=.99,max_abs_heading_error_rad=.1,
                            mean_reference_forward_velocity=1.6,
                            command_nominal_forward_speed_m_s=2*length*left,
                            terminated=1 if corner else 0,truncated=0)
                output=Path(command[command.index('--output')+1]);output.parent.mkdir(parents=True,exist_ok=True)
                output.write_text(json.dumps(result))
                return 0,[],[]
            supervisor=SimpleNamespace(directory=root,state=dict(chunk=1,updates=1000),stopping=False,
                                       deadline=float('inf'),run_child=child,log=lambda *args:None,
                                       args=Namespace(adaptive_plan_config=self.plan,eval_envs=2,eval_steps=1000,
                                                      seed=42,isaac_python=Path('/unused/python')))
            with redirect_stdout(StringIO()):
                summary=_evaluate(supervisor,self.plan['phases'][0],root/'model.pt',0)
                assert summary is not None
                self.assertTrue(summary['passed'])  # 桥接阶段最大角点只监控，不阻断较容易任务。
                commands.clear()
                summary=_evaluate(supervisor,self.plan['phases'][-1],root/'model.pt',1)
            assert summary is not None
            self.assertFalse(summary['passed'])
            profiles=[json.loads(Path(c[c.index('--task-profile')+1]).read_text()) for c in commands]
            continuous=[(c,p) for c,p in zip(commands,profiles) if p['episode_length_s']==60.]
            self.assertEqual(len(continuous),2)
            self.assertEqual({p['command_warmup_half_cycles'] for _,p in continuous},{0,4})
            for command,profile in continuous:
                self.assertEqual(command[command.index('--steps')+1],'3000')
                self.assertIsNone(profile['command_groups'])
                self.assertEqual(profile['command_resample_half_cycles'],0)
                self.assertEqual(profile['push_velocity_max'],.06)


if __name__ == '__main__':
    unittest.main()
