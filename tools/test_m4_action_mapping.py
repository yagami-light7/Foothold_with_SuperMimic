"""M4.2 动作、可微限位和 EPA 异常分流的 CPU 回归测试。"""

from enum import IntFlag
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ['SUPERMIMIC_ROBOT'] = 'roban'
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch

from supermimic.buffer import ReplayBuffer
from supermimic.integration.m4_task import F2GoalSpec, M4EpaEvaluationError, M4TrackingSim, build_m4_task
from supermimic.kinematics import RobotKinematics
from supermimic.networks import Policy, WorldModel
from supermimic.roban import JOINT_NAMES, NUM_J, XI_DIM, actuator_parameters, name_mapping


class Overflow(IntFlag):
    NEFC = 1
    EPA_HORIZON = 256
    ITERATIONS = 512


class M4ActionMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.kin = RobotKinematics('cpu')
        cls.config = json.loads((REPO / 'configs/roban/m4/m4_2.json').read_text())
        cls.args = SimpleNamespace(**cls.config, reference_fps=None, roban_reward_config={},
                                   body_prior_weight=0., wm_use_terrain=False)
        cls.mdp = cls.build_task(torch.zeros(8, NUM_J))
        cls.q0 = cls.mdp.default_q
        cls.sim = object.__new__(M4TrackingSim)
        cls.sim.mdp = cls.mdp

    @classmethod
    def build_task(cls, reference_joints, *, unclipped=False):
        def indices(frame, frames, dt):
            return frame[:, None].expand(-1, frames)

        ref = SimpleNamespace(q=reference_joints, indices=indices, duration=5.2)
        args = SimpleNamespace(**vars(cls.args), m4_unclipped_pd_target=unclipped)
        with patch('supermimic.integration.m4_task.F2MotionReference', return_value=ref):
            return build_m4_task(args, cls.kin, ['unused'], 'cpu')[2]

    def test_decoder_and_noise_keep_linear_output(self):
        policy = Policy(69, 320, hist=10)
        last = policy.dec[-1]
        self.assertIsInstance(last, torch.nn.Linear)
        with torch.no_grad():
            last.weight.zero_()
            last.bias.copy_(torch.linspace(-25., 25., NUM_J))
        obs, goal = torch.zeros(2, 10, 69), torch.zeros(2, 320)
        raw = policy(obs, goal)
        self.assertEqual(raw.shape, (2, NUM_J))
        torch.testing.assert_close(raw, last.bias.expand_as(raw), rtol=0, atol=0)
        with patch('torch.randn_like', return_value=torch.ones_like(raw)):
            executed, logits = policy(obs, goal, action_noise=.1, return_logits=True)
        torch.testing.assert_close(executed, raw + .1, rtol=0, atol=0)
        torch.testing.assert_close(logits, raw, rtol=0, atol=0)
        self.assertLess(float(executed.min().detach()), -20.)
        self.assertGreater(float(executed.max().detach()), 20.)

    def test_center_scale_and_joint_order(self):
        params = actuator_parameters()
        self.assertEqual(self.kin.joint_names, JOINT_NAMES)
        a = torch.eye(NUM_J)
        expected = torch.tensor([self.config['m4_standing_pose'][name] for name in JOINT_NAMES])
        scale = torch.tensor([.35 * params[name]['effort'] / params[name]['stiffness'] for name in JOINT_NAMES])
        self.sim.a_prev = torch.zeros_like(a)
        torch.testing.assert_close(self.sim.pd_target(torch.zeros_like(a)), expected.expand_as(a))
        torch.testing.assert_close(self.sim.pd_target(a), (expected + scale * a).clamp(self.kin.q_lo, self.kin.q_hi))
        torch.testing.assert_close(self.mdp.alpha, scale)
        sim_names = JOINT_NAMES[::-1]
        canonical_to_sim, sim_to_canonical = name_mapping(sim_names)
        self.assertEqual([sim_names[i] for i in canonical_to_sim], JOINT_NAMES)
        torch.testing.assert_close(a[:, sim_to_canonical][:, canonical_to_sim], a)

    def test_reference_independence_and_checkpoints(self):
        other = self.build_task(torch.full((8, NUM_J), .3))
        a = torch.linspace(-4., 4., NUM_J).repeat(2, 1)
        f = torch.tensor([0, 7])
        torch.testing.assert_close(self.mdp.wm_action(a, f), other.wm_action(a, f.flip(0)))
        torch.testing.assert_close(self.mdp._reward_pd_target(a, f), other._reward_pd_target(a, f.flip(0)))
        self.mdp.assert_checkpoint_task({'task_contract': other.contract()})
        with self.assertRaisesRegex(ValueError, 'joint_range'):
            self.mdp.assert_checkpoint_task({'task_contract': {'action_mapping': 'joint_range'}})
        self.assertEqual(self.mdp.obs_dim, 69)
        self.assertEqual(F2GoalSpec('roban_foot', 10, .1).dim + self.mdp.critic_dim, 970)

    def test_replay_and_imagined_target_match_newton_and_reward(self):
        a = torch.linspace(-30., 30., NUM_J).reshape(1, -1)
        self.sim.a_prev = torch.zeros_like(a)
        target = self.sim.pd_target(a)
        replay = ReplayBuffer(4, 1, 'cpu')
        row = {key: value[0].clone() for key, value in replay.data.items()}
        row.update(a=a, f=torch.tensor([3]))
        replay.add(row)
        window = replay.sample_windows(1, 1, include_frames=True)
        torch.testing.assert_close(window['a'][:, 0], a, rtol=0, atol=0)
        a_eff = self.mdp.wm_action(window['a'], window['f'])[:, 0]
        torch.testing.assert_close(self.q0 + self.mdp.alpha * a_eff, target)
        torch.testing.assert_close(self.mdp._reward_pd_target(a, row['f'] + 1), target)
        self.assertTrue(((target >= self.kin.q_lo) & (target <= self.kin.q_hi)).all())
        self.assertTrue(self.mdp.target_clip_mask(a).any())

    def test_unclipped_target_and_wm_use_same_raw_action_beyond_limits(self):
        mdp = self.build_task(torch.zeros(8, NUM_J), unclipped=True)
        a = torch.linspace(-70., 70., NUM_J).repeat(2, 1).requires_grad_()
        sim = object.__new__(M4TrackingSim)
        sim.mdp, sim.a_prev = mdp, torch.zeros_like(a)
        target = sim.pd_target(a)
        torch.testing.assert_close(target, mdp.default_q + mdp.alpha * a)
        torch.testing.assert_close(target, mdp._reward_pd_target(a, None))
        self.assertIs(mdp.wm_action(a, None), a)
        self.assertTrue(mdp.target_out_of_range_mask(a).any())
        self.assertFalse(mdp.target_clip_mask(a).any())
        target_grad, = torch.autograd.grad(target.sum(), a, retain_graph=True)
        torch.testing.assert_close(target_grad, mdp.alpha.expand_as(a))
        torque = ((mdp.kp * (target - mdp.default_q)) / mdp.effort).clamp(-1., 1.)
        saturated = torque.abs() == 1.
        self.assertTrue(saturated.any())
        torque_grad, = torch.autograd.grad(torque.sum(), a)
        torch.testing.assert_close(torque_grad[saturated], torch.zeros_like(torque_grad[saturated]))
        with self.assertRaisesRegex(ValueError, '裁剪契约'):
            mdp.assert_checkpoint_task({'task_contract': self.mdp.contract()})

    def test_bptt_gradient_follows_clamped_target(self):
        inside = torch.full((NUM_J,), .08)
        outside = (self.kin.q_hi - self.q0) / self.mdp.alpha + 1.
        actions = torch.stack((inside, outside)).requires_grad_()
        a_eff = self.mdp.wm_action(actions, None)
        expected = torch.stack((torch.ones_like(inside), torch.zeros_like(outside)))
        grad, = torch.autograd.grad(a_eff.sum(), actions, retain_graph=True)
        torch.testing.assert_close(grad, expected)
        torch.manual_seed(13)
        wm = WorldModel(hidden=(32, 32))
        with torch.no_grad():
            wm.net[-1].weight.normal_(std=.03)
        wm.requires_grad_(False)
        zeros = torch.zeros(2, 3)
        state = (zeros, torch.eye(3).repeat(2, 1, 1), zeros, zeros, self.q0.expand(2, -1), torch.zeros_like(actions))
        for _ in range(4):
            state = wm.step(state, a_eff, torch.zeros(2, XI_DIM))
        grad, = torch.autograd.grad(state[4].sum(), actions)
        self.assertTrue(grad.isfinite().all())
        self.assertGreater(float(grad[0].norm()), 0.)
        torch.testing.assert_close(grad[1], torch.zeros_like(outside), rtol=0, atol=0)

    def epa_fixture(self):
        sim = object.__new__(M4TrackingSim)
        sim.mdp = self.mdp
        sim.eval_mode = False
        sim.dynamics = None
        sim.epa_resets = sim.joint_resets = 0
        sim._reported_joint_position = sim._reported_joint_speed = False
        sim._overflow_type = Overflow
        sim._epa_overflow = int(Overflow.EPA_HORIZON)
        sim._fatal_overflow = ~int(Overflow.EPA_HORIZON | Overflow.ITERATIONS)
        sim._solver_overflow = torch.tensor([0, sim._epa_overflow], dtype=torch.int32)
        sim._solver_nefc = torch.tensor([0, 14])
        sim._njmax = 1024
        sim.f = torch.tensor([51, 158])
        vec = torch.zeros(2, 3)
        return sim, (vec, torch.eye(3).repeat(2, 1, 1), vec, vec,
                     self.q0.repeat(2, 1), torch.zeros(2, NUM_J))

    def test_epa_isolated_in_training_and_reported_in_evaluation(self):
        sim, state = self.epa_fixture()
        with patch('builtins.print'):
            valid = sim.validate_physics_step(state, state, state[4])
        self.assertEqual(valid.tolist(), [True, False])
        self.assertEqual(sim.epa_resets, 1)
        self.assertEqual(sim.joint_resets, 0)
        sim.eval_mode = True
        with self.assertRaises(M4EpaEvaluationError):
            sim.validate_physics_step(state, state, state[4])

    def test_epa_with_other_overflow_or_abnormal_state_still_stops(self):
        sim, state = self.epa_fixture()
        sim._solver_overflow[1] = int(Overflow.ITERATIONS)
        self.assertIsNone(sim.validate_physics_step(state, state, state[4]))
        sim._solver_overflow[1] = int(Overflow.NEFC)
        with self.assertRaises(FloatingPointError):
            sim.validate_physics_step(state, state, state[4])
        for flag in (Overflow.NEFC, Overflow.ITERATIONS):
            with self.subTest(flag=flag):
                sim._solver_overflow[1] = int(Overflow.EPA_HORIZON | flag)
                with self.assertRaises(FloatingPointError) as caught:
                    sim.validate_physics_step(state, state, state[4])
                self.assertNotIsInstance(caught.exception, M4EpaEvaluationError)
        sim._solver_overflow[1] = sim._epa_overflow
        for index, val in ((0, 1.), (2, float('nan')), (3, 200.), (4, 10.), (5, 400.)):
            with self.subTest(index=index):
                after = tuple(x.clone() for x in state)
                after[index][1, 0] = val
                with self.assertRaises(FloatingPointError):
                    sim.validate_physics_step(state, after, state[4])
        self.assertEqual(sim.epa_resets, 0)

    def test_new_control_step_clears_only_old_iteration_warning(self):
        sim, state = self.epa_fixture()
        sim._solver_overflow[:] = torch.tensor([int(Overflow.ITERATIONS), int(Overflow.NEFC | Overflow.ITERATIONS)])
        with patch('supermimic.env.TrackingSim.step') as step:
            sim.step(torch.zeros_like(state[4]))
        step.assert_called_once()
        self.assertEqual(sim._solver_overflow.tolist(), [0, int(Overflow.NEFC)])


if __name__ == '__main__':
    unittest.main()
