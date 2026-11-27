"""检查 M4.3 在 F2 全时钟上的奖励与梯度。"""

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

os.environ['SUPERMIMIC_ROBOT'] = 'roban'
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch

from supermimic.integration.m4_task import build_m4_task
from supermimic.kinematics import RobotKinematics
from supermimic.mdp import NUM_J, flatten_state, unflatten_state
from supermimic.roban_task import long_tail_position


def build_reward_task(swing_aware=True):
    config = json.loads((REPO / 'configs/roban/m4/m4_2.json').read_text())
    config.update(reference_fps=None, wm_use_terrain=False, body_prior_weight=0.,
                  roban_reward_config={'swing_aware': int(swing_aware)})
    path = os.environ.get('M4_REFERENCE', str(REPO.parent / 'Humanoid_Online_Motion_Generator'
                                              / 'outputs/m3_7_1/forward/F2/reference.npz'))
    args = SimpleNamespace(**config)
    return build_m4_task(args, RobotKinematics('cpu'), [path], 'cpu')[2]


class M43RewardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.mdp = build_reward_task()
        cls.motion = cls.mdp.motion

    def reference(self, frames):
        f = torch.as_tensor(frames)
        command = self.motion.anchor(torch.zeros_like(f))
        ref = self.motion.sample(f[:, None], command)
        return f, command, ref

    def foot_terms(self, frames, offset, velocity=None):
        f, command, ref = self.reference(frames)
        fp = (ref['foot_p'][:,0] + offset).requires_grad_()
        if velocity is None:
            velocity = torch.zeros_like(fp)
        return fp, self.mdp._foot_reward_terms(fp, ref['foot_R'][:,0], velocity, ref, f, command)[0]

    def test_swing_tracks_both_near_and_far_in_world(self):
        offset = torch.tensor([[[.15,0.,0.], [.10,0.,0.]], [[.40,0.,0.], [.10,0.,0.]],
                               [[0.,0.,0.], [0.,0.,0.]]])
        fp, terms = self.foot_terms([60, 60, 60], offset)
        grad, = torch.autograd.grad(terms['swing_position'].sum(), fp)
        self.assertLess(float(grad[0,0,0]), -.5)
        self.assertLess(float(grad[1,0,0]), -.1)
        self.assertEqual(torch.count_nonzero(grad[:,1]).item(), 0)
        torch.testing.assert_close(terms['swing_position'][2], torch.tensor(.5))
        torch.testing.assert_close(grad[2], torch.zeros_like(grad[2]))

    def test_clearance_encourages_lift_and_keeps_trajectory_height(self):
        offset = torch.zeros(5, 2, 3)
        offset[:3,0,2] = torch.tensor([-.030, .025, 0.])
        offset[3,0,2] = -.03
        offset[4,0,2] = -.03
        fp, terms = self.foot_terms([60, 60, 60, 40, 80], offset)
        grad, = torch.autograd.grad(terms['swing_clearance'].sum(), fp)
        self.assertGreater(float(grad[0,0,2]), 2.)
        self.assertLess(float(grad[1,0,2]), -2.)
        torch.testing.assert_close(terms['swing_clearance'][2:], torch.tensor([.5, 0., 0.]))
        torch.testing.assert_close(grad[2:], torch.zeros_like(grad[2:]))

    def test_grounded_stance_and_speed_do_not_hide_contact(self):
        offset = torch.zeros(3, 2, 3)
        offset[:,1,2] = .006
        velocity = torch.zeros_like(offset)
        velocity[:,1,0] = torch.tensor([0.,.25,.80])
        fp, terms = self.foot_terms([60, 60, 60], offset, velocity)
        grad, = torch.autograd.grad(terms['contact_schedule'].sum(), fp)
        self.assertLess(float(grad[0,1,2]), -5.)
        torch.testing.assert_close(terms['contact_schedule'], terms['contact_schedule'][0].expand(3))
        self.assertLess(float(terms['slip'][2].detach()), float(terms['slip'][1].detach()) - .5)

    def test_landing_gate_tracks_phase_and_soft_support(self):
        offset = torch.zeros(3, 2, 3)
        offset[0,0,2] = -.025
        offset[1,0,2] = .007
        offset[:2,0,0] = -.17
        fp, terms = self.foot_terms([70, 84, 30], offset)
        grad, = torch.autograd.grad(terms['touchdown'].sum(), fp)
        self.assertGreater(float(grad[0,0,2]), .5)
        self.assertLess(float(grad[1,0,2]), -.2)
        self.assertGreater(float(grad[0,0,0]), .03)
        self.assertGreater(float(grad[1,0,0]), .01)
        self.assertEqual(float(terms['touchdown'][2].detach()), 0.)
        self.assertLess(float(self.mdp.landing_window.diff(dim=0).abs().max()), .25)
        self.assertAlmostEqual(float(self.mdp.landing_window[79,0]), .99275, places=4)
        self.assertEqual(float(self.mdp.landing_window[80,0]), 1.)
        self.assertEqual(float(self.mdp.landing_window[86,0]), 0.)

    def test_real_and_imagined_share_finite_reward_over_entire_clip(self):
        f = torch.arange(self.motion.num_frames)
        command = self.motion.anchor(torch.zeros_like(f))
        state = unflatten_state(flatten_state(self.motion.ref_state(f)))
        state = tuple(t.requires_grad_() for t in state)
        actions = torch.zeros(len(f), NUM_J, requires_grad=True)
        last = torch.zeros_like(actions)
        real = self.mdp.reward(state, f, actions, last, command=command, ep_step=f)
        imagined, terms = self.mdp.reward(state, f, actions, last, True, command=command, ep_step=f)
        torch.testing.assert_close(real, imagined, rtol=0, atol=0)
        contribution = sum(v for k,v in terms.items() if k.endswith('_contribution'))
        torch.testing.assert_close(contribution, real, rtol=0, atol=0)
        grad = torch.autograd.grad(real.sum(), (*state, actions))
        self.assertTrue(real.isfinite().all() and all(x.isfinite().all() for x in grad))
        self.assertLess(float(max(g.abs().max() for g in grad)), 2.)
        self.assertLessEqual(float(real.detach().max()), self.mdp.max_step_reward)
        self.assertNotIn('swing_clearance', build_reward_task(False).contract()['config'])


def save_figure(out):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    mdp = build_reward_task()
    fig, axes = plt.subplots(3, 1, figsize=(9,10), layout='constrained')
    distance = torch.linspace(0., .5, 501, requires_grad=True)
    curves = [('M4.2 Gaussian, sigma 0.08 m', (-distance.square()/.08**2).exp())]
    curves += [(f'Long tail, sigma {s:.2f} m', long_tail_position(distance.square(), s)) for s in (.08,.12,.16)]
    for label, value in curves:
        grad, = torch.autograd.grad(value.sum(), distance)
        axes[0].plot(distance.detach(), value.detach(), label=label)
        axes[1].plot(distance.detach(), -grad, label=label)
    for index, ax in enumerate(axes[:2]):
        ax.set(xlim=(0,.5), xlabel='Foot position error (m)',
               ylabel=('Position score' if index==0 else 'Restoring derivative (1/m)'))
        ax.axvspan(.15,.40,color='#6689a6',alpha=.14)
        ax.grid(alpha=.25)
    axes[0].legend()
    axes[1].set_yscale('symlog', linthresh=.01)
    time = mdp.motion.control_time
    weights = [('Swing position', mdp.cfg.swing_position*mdp.swing_envelope),
               ('Swing clearance', mdp.cfg.swing_clearance*mdp.swing_envelope),
               ('Landing', mdp.cfg.touchdown*mdp.landing_window),
               ('Stance position', mdp.cfg.stance_position*mdp.motion.contact)]
    for label, value in weights:
        axes[2].plot(time, value[:,0]/2, label=label)
    axes[2].set(xlim=(0,5.2), xlabel='F2 time (s)', ylabel='Left foot gain / step / dt')
    axes[2].legend(ncol=2)
    axes[2].grid(alpha=.25)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=135)
    plt.close(fig)


def contact_audit(path):
    """用已有牛顿接触力记录按实际支撑状态核对足底代理。"""
    import numpy as np

    mdp = build_reward_task()
    rows = []
    with np.load(path) as record:
        for seed in record['seeds']:
            prefix = f'i5000_s{seed}_'

            def read(key):
                return torch.from_numpy(record[prefix + key][1:])

            h = mdp._sole_clearance(read('foot'), read('foot_R'))
            speed = read('foot_velocity').norm(dim=-1)
            time = read('time')[:,None]
            true = read('force')[...,2] >= 10.
            first = torch.tensor([[True, False]])
            old = torch.sigmoid((.015-h)/.01) * torch.exp(-speed.square()/.25**2)
            new = torch.sigmoid((mdp.cfg.contact_height-h)/mdp.cfg.contact_height_std)
            masks = dict(all=torch.ones_like(true), double=true.all(-1, keepdim=True).expand_as(true),
                         first_swing=(time >= .8) & (time < 1.6) & first,
                         loaded_sliding=true & (speed > .25), unloaded=~true,
                         first_touchdown=((time-1.6).abs() <= .10 + 1e-8) & first)
            rows.append((true, old, new, masks))
    for name in rows[0][3]:
        true, old, new = (torch.cat([row[i][row[3][name]] for row in rows]) for i in range(3))
        metrics = [(float(((p >= .5) == true).float().mean()),
                    float((p-true.float()).square().mean())) for p in (old,new)]
        print(f'{name}: n={len(true)}, force+={float(true.float().mean()):.3f}, '
              f'accuracy={metrics[0][0]:.3f}/{metrics[1][0]:.3f}, '
              f'Brier={metrics[0][1]:.3f}/{metrics[1][1]:.3f}')


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--figure':
        save_figure(Path(sys.argv[2]))
    elif len(sys.argv) == 3 and sys.argv[1] == '--contact-audit':
        contact_audit(Path(sys.argv[2]))
    else:
        unittest.main()
