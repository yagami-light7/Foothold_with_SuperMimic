"""随机起步的相位、目标、局部reset和兼容性；纯CPU，不启动仿真。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import unittest
from argparse import Namespace
from typing import Any

import torch
from configs.walk_training import adaptive_gate, load_adaptive_plan, validate_profile
from configs.s54_curriculum import restore_task_config
from scripts.gait_cli import apply_gait_overrides, gait_cli_args
from tasks.foot_placement.foot_placement_env_cfg import S54FootPlacementEnvCfg
from tasks.foot_placement.goal_sampler import FootPlacementGoalSampler


class InitialSwingTests(unittest.TestCase):
    def poses(self, n):
        left = torch.tensor([[0., .095, 0.]]).repeat(n, 1)
        right = torch.tensor([[0., -.095, 0.]]).repeat(n, 1)
        quat = torch.tensor([[0., 0., 0., 1.]]).repeat(n, 1)
        return left, quat, right, quat

    def test_fixed_left_and_right_goals_phase_and_rng(self):
        poses = self.poses(8)
        for mode in ('left', 'right'):
            sampler = FootPlacementGoalSampler(8, 'cpu', initial_swing_side=mode, retain_stance_targets=True)
            rng = torch.random.get_rng_state().clone()
            sampler.reset(*poses)
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            expected_left = mode == 'left'
            self.assertTrue((sampler.left_is_swing == expected_left).all())
            torch.testing.assert_close(sampler.phase, torch.full((8,), 0. if expected_left else .5))
            obs = sampler.get_observation(*poses)
            moving, stationary = (slice(0,3), slice(7,10)) if expected_left else (slice(7,10),slice(0,3))
            target = torch.tensor([[.12,.19 if expected_left else -.19,0.]]).repeat(8,1)
            torch.testing.assert_close(obs[:,moving],target)
            torch.testing.assert_close(obs[:,stationary],torch.zeros(8,3))
            torch.testing.assert_close(obs[:,-2],torch.full((8,),1. if expected_left else -1.))
            torch.testing.assert_close(obs[:,-1],torch.zeros(8),atol=1e-6,rtol=0)

    def test_random_sampling_reproducible_and_reselected_each_reset(self):
        sequences = []
        for _ in range(2):
            torch.manual_seed(42)
            sampler = FootPlacementGoalSampler(1024,'cpu',initial_swing_side='random')
            sampler.reset(*self.poses(1024))
            first = sampler.initial_left_is_swing.clone()
            self.assertTrue(450 < int(first.sum()) < 575)
            torch.testing.assert_close(sampler.phase,torch.where(first,0.,.5))
            sampler.reset(*self.poses(1024))
            second = sampler.initial_left_is_swing.clone()
            self.assertGreater(int((first != second).sum()),400)
            sequences.append((first,second))
        for a,b in zip(*sequences):
            torch.testing.assert_close(a,b)

    def test_partial_reset_preserves_other_environments(self):
        torch.manual_seed(7)
        poses = self.poses(32)
        sampler = FootPlacementGoalSampler(32,'cpu',initial_swing_side='random',
                                           step_length_range=(.1,.2),left_gait_frequency=.5,right_gait_frequency=1.5)
        sampler.reset(*poses)
        for _ in range(10):
            sampler.step(.02,*poses)
        names = ('phase','left_is_swing','initial_left_is_swing','step_lengths',
                 'left_target_pos_w','right_target_pos_w','reference_heading_w')
        before = {name:getattr(sampler,name).clone() for name in names}
        ids = torch.tensor([1,7,19])
        mask = torch.ones(32,dtype=torch.bool);mask[ids]=False
        sampler.reset(*poses,env_ids=ids)
        for name in names:
            torch.testing.assert_close(getattr(sampler,name)[mask],before[name][mask],atol=0,rtol=0)
        torch.testing.assert_close(sampler.phase[ids],torch.where(sampler.left_is_swing[ids],0.,.5))
        torch.testing.assert_close(sampler.left_is_swing,sampler.phase<.5)

    def test_right_start_uses_right_frequency_then_switches_to_left(self):
        from tasks.foot_placement.gait import advance_phase
        poses = self.poses(4)
        sampler = FootPlacementGoalSampler(4,'cpu',initial_swing_side='right',
                                           left_gait_frequency=.5,right_gait_frequency=1.5)
        sampler.reset(*poses)
        for step in range(17):
            expected = advance_phase(sampler.phase,.02,sampler.left_gait_frequencies,sampler.right_gait_frequencies)
            sampler.step(.02,*poses)
            torch.testing.assert_close(sampler.phase,expected,atol=0,rtol=0)
            if step < 16:
                self.assertFalse(sampler.left_is_swing.any())
        self.assertTrue(sampler.left_is_swing.all())
        torch.testing.assert_close(sampler.initial_left_is_swing,torch.zeros(4,dtype=torch.bool))

    def test_checkpoint_compatibility_cli_and_training_profiles(self):
        cfg = S54FootPlacementEnvCfg()
        self.assertEqual(cfg.initial_swing_side,'random')
        restore_task_config(cfg,{'observe_gait_frequency':True,'gait_frequency_obs_dim':2})
        self.assertEqual(cfg.initial_swing_side,'left')  # 旧模型复现保持原起步方式。
        args = Namespace(initial_swing_side='random')
        apply_gait_overrides(cfg,args)
        self.assertEqual(cfg.initial_swing_side,'random')
        self.assertEqual(cfg.observation_space,105)
        self.assertEqual(gait_cli_args(args),['--initial-swing-side','random'])
        restore_task_config(cfg,{'initial_swing_side':'right'})
        self.assertEqual(cfg.initial_swing_side,'right')
        plan=load_adaptive_plan(ROOT/'configs/walk_hard_region.json')
        self.assertTrue(all(p['profile']['initial_swing_side']=='random' for p in plan['phases']))

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            FootPlacementGoalSampler(1,'cpu',initial_swing_side='invalid')
        with self.assertRaises(ValueError):
            validate_profile({'initial_swing_side':'invalid'})

    def test_random_start_gate_does_not_hide_failed_right_start(self):
        result: dict[str,Any] = dict(survival_fraction=1.,half_cycle_success_left_right=[.9,.9],
                    consecutive_8_success_fraction=.9,late_xy_error_m=.01,mean_upright=.99,
                    max_abs_heading_error_rad=.05,mean_reference_forward_velocity=.5,
                    command_nominal_forward_speed_m_s=.6,initial_swing_side='random',
                    by_initial_swing={'left':{'half_cycle_success_left_right':[1.,1.]},
                                      'right':{'half_cycle_success_left_right':[.7,.9]}})
        self.assertFalse(adaptive_gate(result)[0])
        result['by_initial_swing']['right']['half_cycle_success_left_right']=[.9,.9]
        self.assertTrue(adaptive_gate(result)[0])


if __name__ == '__main__':
    unittest.main()
