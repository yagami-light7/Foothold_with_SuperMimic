"""Regressions for solver startup motion and the previously unobserved head DOFs."""

# Support both direct execution and python -m from the repository layout.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_root = str(_ToolPath(__file__).resolve().parents[2])
if _tool_root not in _tool_sys.path:
    _tool_sys.path.insert(0, _tool_root)

from pathlib import Path
import sys
import unittest
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from supermimic.reference_quality import root_motion_report, validate_root_motion
from supermimic.s54 import default_joint_pos_list, HEAD_JOINT_NAMES, JOINT_NAMES, NUM_J
from supermimic.kinematics import S54Kinematics
from supermimic.motion import MotionLib
from supermimic.mdp import MDP, RewardCfg, CONTROL_DT
from tools.data.generate_s54_reference import settle_first_frame

REPO = Path(__file__).resolve().parents[2]
DANCE_REFERENCE=REPO/'logs/s54_motion/eval_refs/dance1_subject1.csv'
if not DANCE_REFERENCE.is_file():
    DANCE_REFERENCE=REPO/'data/retarget_data/reference_startup_fixed/dance1_subject1.csv'


def stationary():
    x = np.zeros((61, 36)); x[:, 2] = 1.; x[:, 6] = 1.
    return x


class StartupTests(unittest.TestCase):
    def test_stationary_passes(self):
        self.assertEqual(validate_root_motion(stationary())['status'], 'PASS')

    def test_teleport_at_start_is_rejected(self):
        x = stationary(); x[0, 0] = -2.
        with self.assertRaisesRegex(ValueError, 'unsafe reference root motion'):
            validate_root_motion(x)

    def test_quaternion_sign_flip_does_not_invent_angular_velocity(self):
        x = stationary(); x[::2, 3:7] *= -1
        self.assertEqual(validate_root_motion(x)['max_angular_speed_rad_s'], 0.)

    def test_abrupt_start_rotation_is_rejected(self):
        x = stationary(); x[0, 5:7] = [np.sin(np.pi/6), np.cos(np.pi/6)]
        self.assertEqual(root_motion_report(x)['status'], 'FAIL')

    def test_invalid_root_samples_fail_closed(self):
        for invalid in (np.zeros((0, 7)), stationary()[:1], np.zeros((3, 6))):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                validate_root_motion(invalid)
        x = stationary(); x[0, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_root_motion(x)

    def test_invalid_quaternion_is_rejected(self):
        x = stationary(); x[0, 3:7] = 0
        with self.assertRaisesRegex(ValueError, 'unit xyzw'):
            validate_root_motion(x)

    def test_invalid_fps_is_rejected(self):
        for fps in (0, -30., np.nan):
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                validate_root_motion(stationary(), fps)

    def test_real_bad_aiming_certificate_does_not_bypass_root_gate(self):
        path = REPO/'logs/s54_motion/eval_refs/aiming1_subject1.csv'
        if not path.is_file():self.skipTest('Historical aiming CSV is absent; synthetic unsafe-root tests still run')
        with self.assertRaisesRegex(ValueError, 'unsafe reference root motion'):
            MotionLib([str(path)], S54Kinematics(), 'cpu')

    def test_real_dance_is_not_rejected(self):
        x = np.loadtxt(DANCE_REFERENCE, delimiter=',')
        self.assertEqual(validate_root_motion(x)['status'], 'PASS')

    def test_warmup_exports_settled_root_not_first_solver_iterate(self):
        class Solver:
            calls = 0
            def retarget(self, frame, offset_to_ground=False):
                self.calls += 1
                x = np.zeros(36); x[3] = 1.; x[0] = min(self.calls, 3)
                # Arm null-space drift is replaced by the separate arm IK.
                x[20] = self.calls * .01
                return x
        solver = Solver()
        pose, report = settle_first_frame(solver, {})
        self.assertGreaterEqual(report['calls'], 5)
        self.assertEqual(pose[0], 3.)

    def test_warmup_fails_closed_if_root_never_settles(self):
        class Solver:
            calls = 0
            def retarget(self, frame, offset_to_ground=False):
                self.calls += 1
                x = np.zeros(36); x[3] = 1.; x[0] = self.calls
                return x
        with self.assertRaisesRegex(ValueError, 'did not settle'):
            settle_first_frame(Solver(), {}, max_calls=5)


class HeadRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kin = S54Kinematics()
        cls.motion = MotionLib([str(DANCE_REFERENCE)], cls.kin, 'cpu')
        cls.heads = [JOINT_NAMES.index(n) for n in HEAD_JOINT_NAMES]

    def gradient(self, cfg, position, velocity=(0., 0.)):
        mdp = MDP(self.kin, self.motion, torch.tensor(default_joint_pos_list()), cfg)
        state = list(self.motion.ref_state(torch.tensor([0])))
        q = state[4].clone(); q[:, self.heads] = torch.tensor(position)
        q.requires_grad_(); state[4] = q
        qd = state[5].clone(); qd[:, self.heads] = torch.tensor(velocity)
        qd.requires_grad_(); state[5] = qd
        a = torch.zeros(1, NUM_J)
        reward, terms = mdp.reward(tuple(state), torch.tensor([0]), a, a, terms=True)
        gradients = torch.autograd.grad(reward.sum(), (q, qd))
        return reward.detach(), [v[:, self.heads] for v in gradients], terms

    def test_old_reward_was_blind_to_constant_head_angle(self):
        cfg = RewardCfg(w_head_position=0., w_head_velocity=0.)
        neutral, _, _ = self.gradient(cfg, (0., 0.))
        tilted, gradients, _ = self.gradient(cfg, (.7, .3))
        torch.testing.assert_close(neutral, tilted, atol=1e-7, rtol=0)
        torch.testing.assert_close(gradients[0], torch.zeros(1, 2), atol=1e-7, rtol=0)

    def test_new_head_gradient_points_back_to_neutral(self):
        _, gradients, terms = self.gradient(RewardCfg(), (.7, -.3))
        expected = 2 * RewardCfg().w_head_position * CONTROL_DT * torch.tensor([[.7, -.3]])
        torch.testing.assert_close(gradients[0], expected, atol=1e-7, rtol=0)
        self.assertLess(float(terms['head_penalty'][0]), 0)

    def test_neutral_head_has_no_added_penalty(self):
        _, _, terms = self.gradient(RewardCfg(), (0., 0.))
        self.assertEqual(float(terms['head_penalty'][0]), 0.)

    def test_head_velocity_is_damped(self):
        _, gradients, _ = self.gradient(RewardCfg(), (0., 0.), (1., -1.))
        self.assertLess(float(gradients[1][0, 0]), 0.)
        self.assertGreater(float(gradients[1][0, 1]), 0.)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
