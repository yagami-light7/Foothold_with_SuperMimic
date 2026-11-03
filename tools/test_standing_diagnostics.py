"""Checks for matched WM windows, actual-contact attribution and evaluation protocol."""
import sys
import json
from dataclasses import asdict
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from supermimic.standing_diagnostics import (aggregate_ground_contacts, evaluation_steps,
                                          stop_reasons, valid_windows, paired_predictions)
from supermimic.mdp import flatten_state
from supermimic.foot_evaluate import evaluate_foot
from supermimic.s54 import checkpoint_contract
from test_foot_tracking import setup
from run_s54_standing_controlled import training_command


class DiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.goals, cls.mdp = setup()

    def test_horizons_and_nonfinite_durations(self):
        self.assertEqual([evaluation_steps(d, 100, .02) for d in (2, 5, 10)], [100, 250, 500])
        self.assertEqual(evaluation_steps(None, 100, .02), 100)
        for value in (-1., .031, float('nan'), float('inf')):
            with self.assertRaises(ValueError): evaluation_steps(value, 100, .02)

    def test_physical_termination_at_timeout_is_a_fall(self):
        S = list(self.mdp.nominal_state(2)); S[0][0, 2] -= .3
        self.assertEqual(stop_reasons(S, self.mdp, torch.tensor([True, False]), torch.tensor([True, True])),
                         ['height', 'timeout'])

    def test_contact_force_sign_and_environment(self):
        # Shape0 is ground in one pair, the robot in the other.
        labels = ['/World/envs/env_0/Robot/leg_l6_link', '/World/envs/env_1/Robot/leg_r6_link',
                  '/World/envs/env_1/Robot/base_link']
        count, force, other = aggregate_ground_contacts(np.array([-1, 0, 1, 2]), labels,
                     [0, 2, 0, 1], [1, 0, 3, 2], np.array([[0, 0, -12], [0, 0, 15], [0, 0, -2], [0, 0, 99]]), 2)
        np.testing.assert_array_equal(count, [[1, 0], [0, 1]])
        self.assertEqual(force[0, 0, 2], 12); self.assertEqual(force[1, 1, 2], 15)
        np.testing.assert_array_equal(other, [0, 1])

    def test_windows_include_terminal_but_never_cross_it(self):
        self.assertEqual(valid_windows([8, 9, 3], 8), [(0, 0), (0, 1), (1, 1)])

    def test_training_is_staged_bounded_and_pilot_requires_calibration(self):
        profile = json.loads((Path(__file__).resolve().parents[1]/'configs/s54_standing_3090.json').read_text())
        command = training_command(profile, 'python', 55, 's54_test')
        self.assertEqual(command[command.index('--iterations')+1], '200')
        self.assertEqual(command[command.index('--wm_only_iters')+1], '200')
        with self.assertRaises(ValueError): training_command(profile, 'python', 55, 's54_test', 'pilot')
        command = training_command(profile, 'python', 55, 's54_test', 'pilot', 'logs/s54_calibration/last.pt')
        self.assertEqual(command[command.index('--iterations')+1], '600')
        self.assertIn('--resume', command)
        profile['max_wall_minutes'] = 31
        with self.assertRaises(ValueError): training_command(profile, 'python', 55, 's54_test')

    def test_paired_oracle_has_zero_error_and_checks_command_ticks(self):
        S = self.mdp.nominal_state(2); command = self.goals.new_command(S)
        state = flatten_state(S).numpy()
        action = np.zeros((9, 2, 29), dtype=np.float32)
        trace = dict(state=np.repeat(state[None], 10, axis=0), action=action, a_prev=action.copy(),
                     command_step=np.repeat(np.arange(17, 27)[:, None], 2, axis=1),
                     xi=np.array([[1.]+[0.]*32]*2, dtype=np.float32), episode_lengths=[8, 9],
                     reward=np.full((9, 2), self.mdp.max_step_reward, dtype=np.float32))
        class Oracle:
            def step(self, S, a, xi): return S
        report, rows = paired_predictions(Oracle(), self.mdp, command, trace, 'cpu')
        self.assertEqual(report['replay_reward_sync_max_error'], 0)
        self.assertEqual(len(rows), 9)
        for h in report['horizons'].values():
            self.assertEqual(h['windows'], 3); self.assertEqual(h['base_height_rmse_m'], 0)
            self.assertEqual(h['paired_reward_bias'], 0)
            self.assertAlmostEqual(h['paired_sim_reward'], 1., places=6)
        wrong_reward = dict(trace, reward=np.zeros_like(trace['reward']))
        with self.assertRaises(ValueError): paired_predictions(Oracle(), self.mdp, command, wrong_reward, 'cpu')
        wrong_task = dict(trace, task_contract=np.array(json.dumps({'version': -1})))
        with self.assertRaises(ValueError): paired_predictions(Oracle(), self.mdp, command, wrong_task, 'cpu')

    def test_evaluator_overrides_protocol_and_restores_environment(self):
        mdp, goals = self.mdp, self.goals
        class FakePolicy:
            def __init__(self, *a, **kw): pass
            def to(self, device): return self
            def load_state_dict(self, state): pass
            def eval(self): pass
        class FakeEnv:
            N = 1; max_episode_steps = 100; reset_noise = False
            done_eval = torch.zeros(1, dtype=torch.bool); failed_eval = done_eval.clone()
            xi = torch.tensor([[1.]+[0.]*32]); a_prev = torch.zeros(1, 29)
            def reset(self, ids):
                self.S = mdp.nominal_state(1); self.command = goals.new_command(self.S)
                self.ep_step = torch.zeros(1, dtype=torch.long); self.f = self.ep_step.clone()
            def state(self): return self.S
            def pd_target(self, a): return a
            def step(self, a):
                self.ep_step += 1; self.f += 1
                timeout = self.ep_step >= self.max_episode_steps
                self.done_eval |= timeout
                r, terms = mdp.reward(self.S, None, a, a, True, command=self.command, ep_step=self.ep_step)
                return dict(r=r, terms=terms, done=timeout, term=torch.zeros_like(timeout), timeout=timeout, a_prev=a)
        env = FakeEnv()
        ck = dict(robot_contract=checkpoint_contract(), goal_contract=goals.spec.contract(),
                  task_contract=mdp.contract(), args=dict(hist=2, goal_mode='foot_trajectory',
                  goal_frames=10, goal_dt=.1, foot_config=json.loads(json.dumps(asdict(goals.foot_config)))), pi={})
        with patch('supermimic.foot_evaluate.Policy', FakePolicy):
            result = evaluate_foot(env, ck, goals, mdp, device='cpu', duration_s=.04, controller='default_pd', reset_noise=True)
        self.assertEqual(result['stop_reasons'], ['timeout']); self.assertEqual(result['episode_lengths'], [2])
        self.assertAlmostEqual(result['norm_return'], 1., places=6)
        self.assertEqual(env.max_episode_steps, 100); self.assertFalse(env.reset_noise)
        self.assertEqual(mdp.cfg.episode_steps, 100)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
