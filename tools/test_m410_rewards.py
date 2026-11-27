"""检查 M4.10 奖励公式、历史兼容与隔离 CUDA 采集。"""

import csv
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

os.environ['SUPERMIMIC_ROBOT'] = 'roban'
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from supermimic.control import bound_bootstrap
from supermimic.integration.m410_quality import whole_body_rank
from supermimic.integration.m410_reward import (GROUPS, GROUP_REWARDS, HUBER_CONFIG, HUBER_TERMS,
                                                POSTURE_JOINTS, ROOT_CONFIG, ROOT_TERMS, WHOLE_BODY_TERMS,
                                                dead_zone_huber, heading_delta)
from supermimic.integration.m43_wm_audit import task_for
from supermimic.kinematics import RobotKinematics
from supermimic.mdp import unflatten_state
from supermimic.motion import CONTROL_DT
from supermimic.roban import JOINT_NAMES
from supermimic.roban_task import FootRewardConfig


BASE = REPO / 'outputs/m4_9/a_long_baseline'
REFERENCE = Path('/m4_reference.npz')
if not REFERENCE.is_file():
    REFERENCE = REPO.parent / 'Humanoid_Online_Motion_Generator/outputs/m3_7_1/forward/F2/reference.npz'


def checkpoint(path):
    return torch.load(path, map_location='cpu', mmap=True, weights_only=False)


def check_configs():
    directory = REPO / 'configs/roban/m4'
    base = json.loads((directory / 'm4_9_a.json').read_text())
    effective = json.loads((BASE / 'config/train.json').read_text())
    assert all(effective[key] == value for key, value in base.items())
    assert (effective['lr_decay_iters'], effective['seed'], effective['body_prior_weight']) == (250000, 0, 0.)
    for group, weights in GROUP_REWARDS.items():
        config = json.loads((directory / f'm4_10_{group}.json').read_text())
        assert config == dict(base, roban_reward_config=base['roban_reward_config'] | weights)
    return effective


def yaw_matrix(theta):
    c, s = theta.cos(), theta.sin()
    zero, one = torch.zeros_like(theta), torch.ones_like(theta)
    return torch.stack((c, -s, zero, s, c, zero, zero, zero, one), -1).unflatten(-1, (3, 3))


class Rewards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        old = checkpoint(BASE / 'checkpoints/best.pt')
        cls.args, cls.motion, cls.goals, cls.mdp = task_for(old, RobotKinematics('cpu'), REFERENCE, 'cpu')
        cls.frame = torch.tensor([8, 73, 130])
        cls.anchor = cls.motion.anchor(torch.zeros_like(cls.frame))
        cls.reference = cls.mdp._reference(cls.frame, cls.anchor)
        cls.state = tuple(cls.reference[name][:, 0] for name in ('root_p', 'root_R', 'root_v', 'root_w', 'q', 'qd'))
        cls.cfg = cls.mdp.cfg

    def setUp(self):
        self.mdp.cfg = self.cfg
        self.addCleanup(setattr, self.mdp, 'cfg', self.cfg)

    def test_effective_configs(self):
        check_configs()
        expected = dict(swing_position=1., swing_orientation=.25, swing_clearance=.75, stance_position=1.,
                        contact_schedule=.25, touchdown=.50, slip=.10, upright=.50, survival=.20,
                        joint_limit=2., torque=.01, collision=1., self_collision=1., action_rate=.05,
                        upper_action_smoothness_weight=.02, body_prior_weight=0., soft_body_prior_weight=0.)
        for name, value in expected.items():
            self.assertEqual(getattr(self.cfg, name), value, name)

    def test_root_plateau_and_recovery(self):
        self.mdp.cfg = replace(self.cfg, **ROOT_CONFIG)
        rotation_error = torch.tensor([0., .03, .2], requires_grad=True)
        delta = torch.tensor([[0., 0., 0.], [.01, .01, .01], [.11, .01, .07]], requires_grad=True)
        state = (self.state[0] + delta, yaw_matrix(rotation_error) @ self.state[1], *self.state[2:])
        rewards, _ = self.mdp.whole_body_terms(state, self.reference)
        for value in rewards.values():
            torch.testing.assert_close(value[:2], torch.ones(2), atol=0, rtol=0)
            self.assertLess(float(value[2].detach()), 1.)
        dp, dyaw = torch.autograd.grad(sum(rewards.values()).sum(), (delta, rotation_error))
        torch.testing.assert_close(dp[:2], torch.zeros_like(dp[:2]), atol=0, rtol=0)
        torch.testing.assert_close(dyaw[:2], torch.zeros_like(dyaw[:2]), atol=0, rtol=0)
        self.assertTrue(bool((dp[2] < 0.).all() and dyaw[2] < 0.))
        errors = torch.linspace(0., .4, 30)
        reference = {key: self.reference[key][:1] for key in ('root_p', 'root_R')}
        for term, delta_map in zip(ROOT_TERMS, (lambda e: (e, 0.), lambda e: (0., e), lambda e: (0., 0.))):
            x, z = delta_map(errors)
            p = self.state[0][:1].expand(len(errors), -1).clone()
            p[:, 0] += x
            p[:, 2] += z
            r = self.state[1][:1]
            if term == 'root_heading':
                r = yaw_matrix(errors) @ r
            value = self.mdp.whole_body_terms((p, r, *self.state[2:]), reference)[0][term]
            self.assertTrue(bool((value.diff() <= 1.e-6).all()))

    def test_heading_wrap_and_vertical(self):
        theta = torch.tensor([math.pi-.001, -math.pi+.001, math.pi-.002], dtype=torch.float64, requires_grad=True)
        ref = yaw_matrix(torch.tensor([-math.pi+.001, math.pi-.001, -math.pi+.001], dtype=theta.dtype))
        heading = heading_delta(yaw_matrix(theta), ref)
        torch.testing.assert_close(heading, theta.new_tensor([-.002, .002, -.003]), atol=1.e-12, rtol=0)
        self.assertTrue(bool(torch.isfinite(torch.autograd.grad(heading.sum(), theta)[0]).all()))
        vertical = torch.zeros((1, 3, 3), requires_grad=True)
        degenerate = heading_delta(vertical, torch.eye(3).unsqueeze(0))
        torch.testing.assert_close(torch.autograd.grad(degenerate.sum(), vertical)[0], torch.zeros_like(vertical))

    def test_huber_boundary_and_large_error(self):
        sigma, tolerance = .5, .2
        error = torch.tensor([-.1, .0, .1, .7-1.e-7, .7, .7+1.e-7, 100.], dtype=torch.float64, requires_grad=True)
        penalty, excess = dead_zone_huber(error, tolerance, sigma)
        grad = torch.autograd.grad(penalty.sum(), error)[0]
        torch.testing.assert_close(penalty[:3], error.new_zeros(3), atol=0, rtol=0)
        torch.testing.assert_close(grad, error.new_tensor([0., 0., 0., 2., 2., 2., 2.]), atol=1.e-6, rtol=0)
        self.assertTrue(bool(torch.isfinite(penalty).all() and (excess >= 0.).all()))

    def test_named_groups_quality_weight_and_dt(self):
        mdp = self.mdp
        mdp.cfg = replace(self.cfg, **GROUP_REWARDS['c'])
        names = tuple(tuple(JOINT_NAMES[i] for i in mdp.posture_groups[term]) for term in HUBER_TERMS)
        self.assertEqual(names, POSTURE_JOINTS)
        target = self.state[4].clone()
        for i, term in enumerate(HUBER_TERMS):
            target[:, mdp.posture_groups[term][0]] += (-1.)**i * (getattr(mdp.cfg, term + '_tolerance') + .75)
        q = target.requires_grad_()
        quality = torch.tensor([0., .5, 1.])
        reference = dict(self.reference, quality=quality[:, None])
        a = torch.zeros_like(q)
        with patch.object(mdp, '_reference', return_value=reference):
            reward, terms = mdp.reward((*self.state[:4], q, self.state[5]), self.frame, a, a, True,
                                       command=self.anchor, ep_step=self.frame, a_prev2=a, pd_target=self.state[4])
        root = sum(terms[f'{term}_contribution'] for term in ROOT_TERMS)
        torch.testing.assert_close(root, quality.new_full((3,), .012), atol=1.e-8, rtol=0)
        huber = 0.
        for i, term in enumerate(HUBER_TERMS):
            value = terms[f'{term}_contribution']
            n = len(POSTURE_JOINTS[i])
            scale = getattr(mdp.cfg, term) * CONTROL_DT / n
            torch.testing.assert_close(value, -quality * scale, atol=1.e-8, rtol=0)
            d = torch.autograd.grad(value.sum(), q, retain_graph=True)[0]
            expected = torch.zeros_like(q)
            expected[:, mdp.posture_groups[term][0]] = (-1.)**(i+1) * quality * scale / .5
            torch.testing.assert_close(d, expected, atol=1.e-8, rtol=0)
            huber = huber + value
        torch.testing.assert_close(reward, terms['baseline_reward'] + root + huber)

    def test_disabled_baseline_on_saved_newton(self):
        mdp = self.mdp
        for field in WHOLE_BODY_TERMS:
            self.assertNotIn(field, mdp.cfg.contract())
        self.assertAlmostEqual(mdp.max_step_reward, .089)
        self.assertAlmostEqual(CONTROL_DT * replace(self.cfg, **ROOT_CONFIG).positive_sum, .101)
        self.assertAlmostEqual(CONTROL_DT * replace(self.cfg, **HUBER_CONFIG).positive_sum, .089)
        with np.load(BASE.parent / 'cuda_evaluation.npz', allow_pickle=False) as archive:
            row = json.loads(archive['metadata'].item())['cases']['a_9000']['full']['1001']
            prefix = row['prefix']
            f = torch.from_numpy(archive[prefix+'frame'][1:]).long()
            raw = torch.from_numpy(archive[prefix+'raw_action'])
            target = torch.from_numpy(archive[prefix+'target'])
            state = unflatten_state(torch.from_numpy(archive[prefix+'state'][1:]))
        prev = torch.cat((torch.zeros_like(raw[:1]), raw[:-1]))
        prev2 = torch.cat((torch.zeros_like(raw[:1]), prev[:-1]))
        context = dict(command=mdp.motion.anchor(torch.zeros_like(f)), ep_step=f, a_prev2=prev2, pd_target=target)
        baseline, terms = mdp.reward(state, f, raw, prev, True, **context)
        self.assertAlmostEqual(float(baseline.sum()), row['tracking_return'], places=4)
        for key, value in row['reward_means'].items():
            self.assertAlmostEqual(float(terms[f'{key}_contribution'].mean()), value, places=6)
        mdp.cfg = replace(self.cfg, **GROUP_REWARDS['c'])
        _, expanded = mdp.reward(state, f, raw, prev, True, **context)
        torch.testing.assert_close(expanded['baseline_reward'], baseline, atol=0, rtol=0)

    def test_old_priors_and_negative_targets(self):
        with self.assertRaises(ValueError):
            FootRewardConfig(**HUBER_CONFIG, body_prior_weight=.02)
        with self.assertRaises(ValueError):
            FootRewardConfig(root_xy=.30, root_xy_sigma=0.)
        values = torch.tensor([-5., -.5, 0., 2.])
        torch.testing.assert_close(bound_bootstrap(values, None), values)
        torch.testing.assert_close(bound_bootstrap(values, 1.), torch.tensor([-5., -.5, 0., 1.]))

    def test_whole_body_rank_gait_gate(self):
        aggregate = dict(valid_for_best=True, complete_tracking_rate=2/3, foot_position_rmse_m=.025,
                         whole_body=dict(root_xy_rmse_m=.05, root_z_rmse_m=.03, root_heading_rmse_deg=5.,
                                         arm_rmse_rad=.4, elbow_max_near_limit_ratio=.1),
                         upper_smoothness=dict(after_startup=dict(acceleration_rms_rad_s2=40., jerk_rms_rad_s3=3000.)))
        row = dict(iteration=500, aggregate=aggregate)
        self.assertIsNone(whole_body_rank(row))
        aggregate['complete_tracking_rate'] = 1.
        self.assertEqual(whole_body_rank(row), (1., -500))


def smoke():
    torch.set_num_threads(2)
    check_configs()
    previous = checkpoint(BASE / 'checkpoints/initial.pt')
    with TemporaryDirectory(dir='/tmp', prefix='m410-smoke-') as scratch:
        for group, name in GROUPS.items():
            out = Path(scratch) / name
            command = [sys.executable, 'supermimic/train.py', '--roban_config', f'configs/roban/m4/m4_10_{group}.json',
                       '--m4_reference', str(REFERENCE), '--device', 'cuda:0', '--fresh_start', '--m410_smoke',
                       '--m410_group', group, '--log_dir', str(out / 'checkpoints')]
            run = subprocess.run(command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=240)
            required = ('M4 completed | iteration=2 |', 'M4.10 Smoke 覆盖步龄 0/5/10/11')
            if run.returncode or not all(s in run.stdout for s in required):
                raise RuntimeError(run.stdout)
            if any(s in run.stdout for s in ('CUDA error', 'Warning: EPA horizon', 'nefc overflow')):
                raise RuntimeError(run.stdout)
            initial, last = (checkpoint(out / 'checkpoints' / f'{tag}.pt') for tag in ('initial', 'last'))
            assert (last['iteration'], last['sim_steps']) == (2, 312)
            assert initial['iteration'] == initial['sim_steps'] == 0 and initial['transfer_report'] is None
            assert all(not initial[k]['state'] and last[k]['state'] for k in ('opt_wm', 'opt_pi', 'opt_v'))
            for key in ('wm', 'pi', 'critic'):
                assert all(torch.equal(t, previous[key][n]) for n, t in initial[key].items())
                assert any(not torch.equal(t, last[key][n]) for n, t in initial[key].items() if n.endswith('.weight'))
                assert all(torch.isfinite(t).all() for t in last[key].values())
            for key in ('goal_contract', 'policy_contract', 'imagination_contract', 'data_contract', 'physical_contract'):
                assert initial[key] == previous[key], key
            assert last['critic']['norm.mean'].numel() == 970
            assert initial['args']['reward_config'] == previous['args']['reward_config'] | GROUP_REWARDS[group]
            if 'S54_SOURCE_GIT_SHA' in os.environ:
                assert initial['args']['git_commit'] == os.environ['S54_SOURCE_GIT_SHA']
                assert initial['args']['git_worktree'] == os.environ['M410_SOURCE_GIT_STATUS']
            with (out / 'logs/train.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                assert all(float(row[key]) == 0. for key in ('epa_resets', 'invalid_transitions', 'physics_exceptions'))
                assert max(float(row[key]) for key in ('reward_sync_max_error', 'goal_sync_max_error')) <= 1.e-5
                assert all(math.isfinite(float(row[key])) for key in ('wm_loss', 'pi_loss', 'v_loss', 'total_reward'))
            row = rows[-1]
            used = {term: float(row[f'term_{term}_actor_gradnorm']) for term in WHOLE_BODY_TERMS}
            assert all(math.isfinite(g) and (g == 0. if term not in GROUP_REWARDS[group] else g >= 0.)
                       for term, g in used.items())
            for family in (ROOT_TERMS, HUBER_TERMS):
                if any(term in GROUP_REWARDS[group] for term in family):
                    assert max(used[term] for term in family) > 0., used
            columns = ('sim_steps', 'wm_gradnorm', 'pi_gradnorm', 'critic_gradnorm', 'wm_foot_rmse_8',
                       'wm_root_position_rmse_8', 'wm_upper_joint_rmse_8', 'goal_sync_max_error', 'reward_sync_max_error',
                       'baseline_reward', 'total_reward', 'critic_target_min', 'critic_target_negative_ratio',
                       'wm_effective_action_rms', 'near_limit_zarm_l1_joint', 'clamp_leg_l5_joint')
            assert all(math.isfinite(float(row[key])) for key in columns)
            assert all(float(row[key]) > 0 for key in ('wm_gradnorm', 'pi_gradnorm', 'critic_gradnorm'))
            print('M410_SMOKE', json.dumps(dict(group=group, gpu=initial['args']['training_gpu'],
                   versions=initial['args']['environment_versions'], terms=used,
                   stats={key: float(row[key]) for key in columns}), ensure_ascii=False), flush=True)
            del initial, last

        # 用 C 的短训权重经过正式周期入口，确保 Best 的排序与快照恢复可用。
        evaluate = [sys.executable, '-m', 'supermimic.integration.m43_evaluation', '--out', str(out),
                    '--m4_reference', str(REFERENCE), '--device', 'cuda:0', '--m410-periodic']
        completed = subprocess.run(evaluate, cwd=REPO, text=True, capture_output=True, timeout=240)
        if completed.returncode:
            raise RuntimeError(completed.stdout + completed.stderr)
        summary = json.loads((out / 'evaluation/results.json').read_text())
        aggregate = summary['milestones']['2']['aggregate']
        assert aggregate['valid_for_best'] and aggregate['fatal_errors'] == 0
        assert summary['best_iteration'] == 2
        assert aggregate['whole_body']['root_heading_rmse_deg'] is not None
        if aggregate['complete_tracking_rate'] < 1.:
            assert 'best_whole_body_iteration' not in summary
        alias = out / 'checkpoints/best.pt'
        milestone = out / 'checkpoints/iter_2.pt'
        assert alias.samefile(milestone)
        alias.unlink()
        subprocess.run(evaluate, cwd=REPO, check=True, text=True, capture_output=True, timeout=30)
        assert alias.samefile(milestone)
        assert json.loads((out / 'evaluation/results.json').read_text()) == summary
        print('M410_PERIODIC 三条有效轨迹，Best 恢复与全身候选门槛通过。', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--smoke']:
        smoke()
    else:
        unittest.main()
