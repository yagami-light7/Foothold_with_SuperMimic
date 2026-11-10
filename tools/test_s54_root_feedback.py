"""Motion reward/feedback regression tests; no simulator or test references used."""
from dataclasses import asdict
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from supermimic.foot_mdp import assert_task_checkpoint
from supermimic.kinematics import S54Kinematics
from supermimic.mdp import MDP, OBS_DIM, MOTION_OBS_DIM, RewardCfg, resolve_motion_reward
from supermimic.networks import Policy, WorldModel
from supermimic.rotations import so3_exp
from supermimic.s54 import NUM_J, XI_DIM, default_joint_pos_list
from supermimic.transfer import policy_contract, assert_policy_checkpoint
from run_s54_m34 import train_args


class MotionFeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.kin = S54Kinematics('cpu')
        cls.q0 = torch.tensor(default_joint_pos_list())

    def make(self, cfg=None, mode='root_xy_heading_v2'):
        p = torch.tensor([[0., 0., 1.05]])
        R = torch.eye(3)[None]
        q = self.q0[None].clone()
        S = (p, R, torch.zeros_like(p), torch.zeros_like(p), q, torch.zeros_like(q))
        bp, bR, bv, bw = self.kin.forward(*S)
        m = SimpleNamespace(root_p=p.clone(), root_R=R.clone(), q=q.clone(), qd=S[5].clone(),
                            body_p=bp, body_R=bR, body_v=bv, body_w=bw)
        mdp = MDP(self.kin, m, self.q0, cfg or RewardCfg.tracking_v2(), observation_mode=mode)
        return mdp, S, torch.zeros(1, dtype=torch.long), torch.zeros(1, NUM_J)

    def observation(self, mdp, S, f, action):
        xi = torch.zeros(1, XI_DIM); xi[:, 0] = 1.
        return mdp.obs(S, action, xi, f)

    def test_requested_weights_scales_and_dynamic_bound(self):
        cfg = RewardCfg.tracking_v2()
        self.assertEqual((cfg.w_body_pos, cfg.w_body_ori, cfg.w_body_linvel, cfg.w_body_angvel,
                          cfg.w_root_pos, cfg.w_root_ori, cfg.w_foot_pos, cfg.w_foot_ori),
                         (1., 1., 1., 1., 1., .75, .5, .25))
        self.assertEqual((cfg.body_pos_std, cfg.body_ori_std, cfg.root_pos_std,
                          cfg.root_ori_std, cfg.foot_pos_std, cfg.foot_ori_std), (.3, .4, .2, .3, .08, .2))
        mdp, S, f, a = self.make(cfg)
        self.assertAlmostEqual(mdp.max_step_reward, .13)
        self.assertLessEqual(float(mdp.reward(S, f, a, a)), mdp.max_step_reward)

    def test_root_xy_kernel_ignores_z_and_tracks_xy(self):
        mdp, S, f, a = self.make()
        _, original = mdp.reward(S, f, a, a, terms=True)
        vertical = tuple(x.clone() for x in S); vertical[0][:, 2] += .1
        _, elevated = mdp.reward(vertical, f, a, a, terms=True)
        self.assertTrue(torch.equal(original['root_position_reward'], elevated['root_position_reward']))
        horizontal = tuple(x.clone() for x in S); horizontal[0][:, 0] += .2
        _, shifted = mdp.reward(horizontal, f, a, a, terms=True)
        self.assertAlmostEqual(float(shifted['root_position_reward']), .02 * math.exp(-1), places=7)
        self.assertLess(float(shifted['foot_position_reward']), float(original['foot_position_reward']))

    def test_feet_use_global_reference_and_average_both_sides(self):
        mdp, S, f, a = self.make()
        _, terms = mdp.reward(S, f, a, a, terms=True)
        self.assertAlmostEqual(float(terms['foot_position_reward']), .5*.02, places=7)
        shifted = tuple(x.clone() for x in S); shifted[0][:, 0] += .08
        _, terms = mdp.reward(shifted, f, a, a, terms=True)
        self.assertAlmostEqual(float(terms['foot_position_reward']), .5*.02*math.exp(-1), places=6)
        turned = list(x.clone() for x in S)
        turned[1] = so3_exp(torch.tensor([[0., 0., .2]]))
        _, terms = mdp.reward(tuple(turned), f, a, a, terms=True)
        self.assertAlmostEqual(float(terms['foot_orientation_reward']), .25*.02*math.exp(-1), places=6)

    def test_feedback_is_reference_minus_actual_in_actual_heading(self):
        mdp, S, f, a = self.make()
        baseline = self.observation(mdp, S, f, a)
        self.assertEqual(baseline.shape, (1, MOTION_OBS_DIM))
        self.assertTrue(torch.allclose(baseline[:, 99:], torch.tensor([[0., 0., 0., 1.]]), atol=1e-6))
        changed = list(x.clone() for x in S)
        changed[0][:, 0] += 1.
        moved = self.observation(mdp, tuple(changed), f, a)
        self.assertTrue(torch.equal(baseline[:, :99], moved[:, :99]))
        self.assertTrue(torch.allclose(moved[:, 99:101], torch.tensor([[-1., 0.]])))
        changed[1] = so3_exp(torch.tensor([[0., 0., math.pi/2]]))
        moved = self.observation(mdp, tuple(changed), f, a)
        self.assertTrue(torch.allclose(moved[:, 99:], torch.tensor([[0., 1., -1., 0.]]), atol=1e-6))
        self.assertEqual(mdp.add_noise(moved).shape, moved.shape)

    def test_heading_encoding_is_continuous_across_pi(self):
        mdp, S, f, a = self.make()
        values = []
        for angle in (math.pi-1e-4, math.pi+1e-4):
            changed = list(x.clone() for x in S)
            changed[1] = so3_exp(torch.tensor([[0., 0., angle]]))
            values.append(self.observation(mdp, tuple(changed), f, a)[:, 101:103])
        self.assertLess(float((values[0]-values[1]).abs().max()), 3e-4)

    def test_batch_history_dimensions_and_differentiable_feedback_reward(self):
        mdp, S, f, a = self.make()
        p = (S[0] + torch.tensor([[.03, .02, 0.]])).requires_grad_()
        rotation = torch.tensor([[.01, -.02, .12]], requires_grad=True)
        q = (S[4] + .01).requires_grad_()
        state = (p, so3_exp(rotation), S[2], S[3], q, S[5])
        obs = self.observation(mdp, state, f, a)
        reward = mdp.reward(state, f, a, a)
        loss = reward.sum() + .01 * obs[:, 99:].square().sum()
        loss.backward()
        for tensor in (p, rotation, q):
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(float(tensor.grad.abs().sum()), 0.)
        history_state = tuple(x[:, None].repeat(1, 3, *([1]*(x.ndim-1))) for x in S)
        xi = torch.zeros(1, 3, XI_DIM)
        actual = mdp.obs(history_state, a[:, None].repeat(1, 3, 1), xi, f[:, None].repeat(1, 3))
        self.assertEqual(actual.shape, (1, 3, MOTION_OBS_DIM))

    def test_policy_forward_and_contract_reject_legacy_resume(self):
        mdp, S, f, a = self.make()
        pi = Policy(mdp.obs_dim, 730, hist=10)
        output = pi(self.observation(mdp, S, f, a)[:, None].repeat(1, 10, 1), torch.zeros(1, 730))
        self.assertEqual(output.shape, (1, 29))
        self.assertEqual(pi.dec[0].in_features, 1062)
        contract = policy_contract(10, obs_dim=103, observation_mode='root_xy_heading_v2')
        assert_policy_checkpoint({'policy_contract': contract}, 10, 'identity', True,
                                 obs_dim=103, observation_mode='root_xy_heading_v2')
        with self.assertRaises(ValueError):
            assert_policy_checkpoint({'policy_contract': policy_contract(10)}, 10, 'identity', True,
                                     obs_dim=103, observation_mode='root_xy_heading_v2')
        with self.assertRaises(ValueError):
            assert_task_checkpoint({}, mdp)

    def test_bptt_reaches_actor_through_frozen_world_model(self):
        torch.manual_seed(54103)
        mdp, S, f, a = self.make()
        pi = Policy(mdp.obs_dim, 730, hist=2)
        wm = WorldModel()
        # A small deterministic nonconstant dynamics map, not reference labels
        # or a simulator claim. Zero-initialized WM would hide action gradients.
        with torch.no_grad():
            wm.net[-1].weight.normal_(0, 1e-4)
        wm.requires_grad_(False)
        xi = torch.zeros(1, XI_DIM); xi[:, 0] = 1.
        xi_n = xi.clone(); xi_n[:, 0] = (xi_n[:, 0]-.75)/.45
        history = self.observation(mdp, S, f, a)[:, None].repeat(1, 2, 1)
        goal = torch.zeros(1, 730)
        objective = 0.
        for _ in range(3):
            action = pi(history, goal).clamp(-4, 4)
            S = wm.step(S, action, xi_n)
            objective = objective + mdp.reward(S, f, action, a).mean()
            obs = mdp.obs(S, action, xi, f)
            history = torch.cat((history[:, 1:], obs[:, None]), 1)
            a = action
        (-objective).backward()
        gradients = [p.grad for p in pi.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.)
        self.assertTrue(all(p.grad is None for p in wm.parameters()))

    def test_orientation_adapter_keeps_new_feedback_channels(self):
        torch.manual_seed(54104)
        pi = Policy(103, 730, hist=2, obs_adapter='mask_reference_orientation')
        obs = torch.randn(1, 2, 103)
        z = torch.zeros(1, 32)
        baseline = pi.decode(obs, z)
        orientation_changed = obs.clone(); orientation_changed[..., 93:99] += 3.
        feedback_changed = obs.clone(); feedback_changed[..., 99:103] += 3.
        self.assertTrue(torch.equal(baseline, pi.decode(orientation_changed, z)))
        self.assertFalse(torch.equal(baseline, pi.decode(feedback_changed, z)))

    def test_legacy_reward_and_contract_remain_compatible(self):
        mdp, S, f, a = self.make(RewardCfg(), 'legacy_v1')
        self.assertEqual(self.observation(mdp, S, f, a).shape[-1], OBS_DIM)
        self.assertAlmostEqual(mdp.max_step_reward, .1)
        self.assertEqual(mdp.contract()['version'], 1)
        original = Path('logs/s54_m34_run02/source/supermimic/mdp.py')
        if not original.exists():
            self.skipTest('Historical source snapshot not installed')
        spec = importlib.util.spec_from_file_location('supermimic._reward_v1_snapshot', original)
        module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        old = module.MDP(self.kin, mdp.motion, self.q0)
        self.assertEqual(mdp.contract(), old.contract())
        changed = tuple(x.clone() for x in S); changed[0][:, :2] += .2
        self.assertTrue(torch.equal(mdp.reward(changed, f, a, a), old.reward(changed, f, a, a)))

    def test_config_reaches_training_as_valid_json(self):
        config = json.loads(Path('configs/s54_m34_4060.json').read_text())
        args = train_args(config, Path('logs/s54_test'))
        raw = args[args.index('--motion_reward_config')+1]
        cfg = resolve_motion_reward(raw)
        self.assertEqual(cfg, RewardCfg.tracking_v2())
        self.assertEqual(args[args.index('--motion_obs_mode')+1], 'root_xy_heading_v2')
        for invalid in ({'foot_pos_std': 0}, {'w_foot_ori': -1}, {'w_root_pos': float('nan')}):
            with self.assertRaises(ValueError):
                resolve_motion_reward(invalid)


if __name__ == '__main__':
    unittest.main(verbosity=2)
