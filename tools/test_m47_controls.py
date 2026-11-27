"""验证 M4.7 命令、动作历史及隔离 CUDA 冒烟。"""
import argparse
from copy import copy
from dataclasses import replace
import csv
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
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))

import numpy as np
import torch

from supermimic.buffer import ReplayBuffer
from supermimic.env import TrackingSim
from supermimic.integration.m4_task import M4TrackingSim
from supermimic.integration.m47_control import blend_startup_target, upper_action_cost, upper_joint_indices
from supermimic.motion import CONTROL_DT
from supermimic.roban import ELBOW_JOINT_NAMES, JOINT_NAMES, XI_DIM
from supermimic.roban_task import UPPER_POSE_JOINTS
from tools.test_m43_reward import build_reward_task

UPPER=upper_joint_indices(JOINT_NAMES)
LEGS=[i for i in range(len(JOINT_NAMES)) if i not in UPPER]


class M47Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mdp=build_reward_task()

    def test_same_mapping_at_reset_and_completion(self):
        mdp=self.mdp
        lo,hi=mdp.kin.q_lo,mdp.kin.q_hi
        reset=mdp.motion.q[0].clamp(lo,hi)
        a=torch.linspace(-100.,100.,len(JOINT_NAMES)).expand(11,-1)
        nominal=(mdp.default_q+mdp.alpha*a).clamp(lo,hi)
        mask=torch.zeros_like(reset,dtype=torch.bool)
        mask[UPPER]=True
        for steps in (5,10):
            with self.subTest(steps=steps):
                target=blend_startup_target(nominal,reset,torch.arange(11),mask,steps)
                torch.testing.assert_close(target[0,UPPER],reset[UPPER],rtol=0,atol=0)
                torch.testing.assert_close(target[steps:],nominal[steps:],rtol=0,atol=0)
                torch.testing.assert_close(target[:,LEGS],nominal[:,LEGS],rtol=0,atol=0)
                u = torch.arange(1, steps, dtype=target.dtype)[:, None] / steps
                beta = 3 * u.square() - 2 * u.pow(3)
                expected = (1 - beta) * reset[UPPER] + beta * nominal[1:steps, UPPER]
                torch.testing.assert_close(target[1:steps, UPPER], expected, rtol=0, atol=3e-7)
                self.assertTrue(bool(((target>=lo-1e-7) & (target<=hi+1e-7)).all()))
                effective=(target-mdp.default_q)/mdp.alpha
                torch.testing.assert_close(mdp.default_q+mdp.alpha*effective,target,atol=3e-7,rtol=0)

    def test_smooth_cost_masks_and_channels(self):
        raw=torch.ones((4,len(JOINT_NAMES)))
        checked=torch.tensor([True,True,False,False])
        for a,b,c in ((raw,raw,raw),(3*raw,2*raw,raw)):
            torch.testing.assert_close(upper_action_cost(a,b,c,checked,UPPER),torch.zeros(4))
        low=2.*raw
        low[:,LEGS]=100.
        torch.testing.assert_close(upper_action_cost(low,raw,0.*raw,checked,UPPER),torch.zeros(4))
        self.assertTrue(torch.equal(upper_action_cost(raw,-raw,raw,checked,UPPER),torch.tensor([16.,16.,0.,0.])))

    def test_bptt_retains_both_previous_gradients(self):
        trajectories=torch.tensor([[.0],[2.],[.4]],dtype=torch.float64).expand(-1,len(JOINT_NAMES)).clone().requires_grad_()
        def cost(x):
            return upper_action_cost(x[2],x[1],x[0],torch.tensor(True),UPPER)
        grad,=torch.autograd.grad(cost(trajectories),trajectories)
        delta=trajectories[2,UPPER]-2.*trajectories[1,UPPER]+trajectories[0,UPPER]
        expected=torch.tensor([[1.],[-2.],[1.]])*2.*delta.detach()/len(UPPER)
        torch.testing.assert_close(grad[:,UPPER],expected)
        torch.testing.assert_close(grad[:,LEGS],torch.zeros_like(grad[:,LEGS]))
        epsilon=1.e-5
        direction=torch.zeros_like(trajectories)
        direction[0,UPPER[0]]=1.
        numerical=(cost(trajectories+epsilon*direction)-cost(trajectories-epsilon*direction))/(2.*epsilon)
        torch.testing.assert_close(numerical,grad[0,UPPER[0]],rtol=2e-7,atol=1e-10)

    def test_reward_uses_dt_once_and_defaults_match(self):
        from copy import copy

        mdp=self.mdp
        s=copy(mdp)
        s.cfg=replace(mdp.cfg,upper_action_smoothness_weight=.02)
        post=torch.tensor([1,2,3,4])
        command=mdp.motion.anchor(torch.zeros_like(post))
        state=mdp.motion.ref_state(post)
        a=mdp.default_q.new_ones((4,len(JOINT_NAMES)))
        p=-.5*a
        pp=.25*a
        arguments=dict(command=command,ep_step=post)
        expected,t0=mdp.reward(state,post,a,p,True,**arguments)
        zero,t1=mdp.reward(state,post,a,p,True,a_prev2=pp,**arguments)
        out,t2=s.reward(state,post,a,p,True,a_prev2=pp,**arguments)
        torch.testing.assert_close(expected,zero,rtol=0,atol=0)
        torch.testing.assert_close(out-zero,-CONTROL_DT*.02*torch.tensor([0.,0.,2.25**2,2.25**2]),atol=5e-9,rtol=0)
        for k in ('foot_objective','action_rate_contribution','torque_contribution'):
            torch.testing.assert_close(t0[k],t2[k],atol=0,rtol=0)

    def test_reset_preserves_other_episode_history(self):
        mdp=self.mdp
        env=M4TrackingSim.__new__(M4TrackingSim)
        env.mdp=mdp
        env.a_prev2=torch.full((2,len(JOINT_NAMES)),2.)
        env.startup_q=torch.zeros_like(env.a_prev2)
        env.failed_eval=env.done_eval=torch.zeros(2,dtype=torch.bool)
        env.state=lambda:mdp.motion.ref_state(torch.tensor([0,0]))
        with patch.object(TrackingSim,'reset'):
            env.reset(torch.tensor([1]))
        torch.testing.assert_close(env.a_prev2[:,0],torch.tensor([2.,0.]))
        torch.testing.assert_close(env.startup_q[1],mdp.motion.q[0])

    def test_replay_start_keeps_both_previous_actions_through_wrap(self):
        buffer=ReplayBuffer(3,1,'cpu',action_context=True)
        for age,tick in enumerate((0,1,2,0)):
            row={k: torch.zeros((1,)+((d,) if isinstance(d,int) else d),dtype=t)
                 for k,(d,t) in buffer.fields.items()}
            row['ep_step'][:]=tick
            row['f'][:]=tick
            row['done'][:]=tick==2
            row['a'][:]=age+3.
            row['a_prev'][:]=age+2. if tick else 0.
            row['a_prev2'][:]=age+1. if tick>=2 else 0.
            row['a_exec'][:]=age+7.
            row['startup_q'][:]=1. if age<3 else 2.
            buffer.add(row)
        with patch.object(buffer,'_sample_indices',return_value=(torch.tensor([1]),torch.tensor([0]))):
            start,_=buffer.sample_starts(1,10)
            window=buffer.sample_windows(1,2,include_reward=True)
        self.assertEqual(start['ep_step'].item(),2)
        self.assertEqual(start['a_prev2'][0,0].item(),3.)
        self.assertEqual(start['startup_q'][0,0].item(),1.)
        self.assertEqual(window['a_exec'][0,0,0].item(),9.)
        self.assertEqual(window['mask'].tolist(),[[1.,0.]])


class M49Posture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = build_reward_task()
        cls.base.cfg = replace(cls.base.cfg, upper_action_smoothness_weight=.02)
        cls.tolerance = {name: .25 for name in UPPER_POSE_JOINTS}

    def task(self, **prior):
        mdp = copy(self.base)
        mdp.cfg = replace(self.base.cfg, **prior)
        tolerance = mdp.cfg.soft_body_prior_tolerance
        mdp.soft_prior_tolerance = mdp.pose_tolerance(tolerance) if mdp.cfg.soft_body_prior_weight else None
        return mdp

    def fixture(self):
        f = torch.tensor([20, 77, 120])
        command = self.base.motion.anchor(torch.zeros_like(f))
        state = self.base.motion.ref_state(f)
        a = torch.linspace(-.3,.5,len(JOINT_NAMES)).expand(len(f),-1)
        history = dict(command=command, ep_step=f, a_prev2=a*.6, pd_target=self.base.standing_pd_target(a))
        return f, state, a, history

    def test_zero_weight_keeps_reward_contract_and_actor_inputs(self):
        disabled = self.task(soft_body_prior_weight=0., soft_body_prior_tolerance=self.tolerance)
        f, state, action, history = self.fixture()
        old = self.base.reward(state, f, action, action*.7, True, **history)
        new = disabled.reward(state, f, action, action*.7, True, **history)
        torch.testing.assert_close(new, old, rtol=0, atol=0)
        self.assertEqual(disabled.contract(), self.base.contract())
        enabled = self.task(soft_body_prior_weight=.02, soft_body_prior_tolerance=self.tolerance)
        xi = state[0].new_zeros(len(f), XI_DIM)
        obs = enabled.goals.obs(enabled, state, action, xi, f, history['command'])
        torch.testing.assert_close(obs, self.base.goals.obs(self.base, state, action, xi, f, history['command']), rtol=0, atol=0)
        self.assertEqual(obs.shape[-1], 69)
        torch.testing.assert_close(enabled.goals.goal(state, f, f, history['command']),
                                   self.base.goals.goal(state, f, f, history['command']), rtol=0, atol=0)
        self.assertEqual(enabled.goals.spec.dim, 320)

    def test_zero_tolerance_matches_original_values_and_gradients(self):
        original = self.task(body_prior_weight=.02)
        soft = self.task(soft_body_prior_weight=.02, soft_body_prior_tolerance=dict.fromkeys(UPPER_POSE_JOINTS,0.))
        f, state, action, history = self.fixture()
        q = (state[4] + .11*action).requires_grad_()
        ref = (*state[:4], q, state[5])
        ro, o = original.reward(ref, f, action, action*.7, True, **history)
        rs, s = soft.reward(ref, f, action, action*.7, True, **history)
        torch.testing.assert_close(rs, ro, rtol=0, atol=0)
        torch.testing.assert_close(s['soft_body_prior'], o['body_prior'], rtol=0, atol=0)
        gradients = tuple(torch.autograd.grad(v.sum(), q)[0]
                          for v in (o['body_prior_contribution'], s['soft_body_prior_contribution']))
        torch.testing.assert_close(*gradients, rtol=0, atol=0)
        self.assertEqual(original.cfg.positive_sum, soft.cfg.positive_sum)

    def test_dead_zone_symmetry_and_continuous_slope(self):
        mdp = self.task(soft_body_prior_weight=.02, soft_body_prior_tolerance=self.tolerance)
        j = JOINT_NAMES.index(ELBOW_JOINT_NAMES[0])
        x = torch.tensor([-.61,-.31,-.25,-.10,.10,.25,.250001,.31,.61], dtype=torch.float64)
        q = torch.zeros((len(x),len(JOINT_NAMES)), dtype=torch.float64)
        q[:,j] = x
        q.requires_grad_()
        arm, waist, residual = mdp.posture_prior_terms(q, torch.zeros_like(q), mdp.soft_prior_tolerance)
        grad, = torch.autograd.grad(arm.sum(), q)
        torch.testing.assert_close(residual[2:6,j], torch.zeros(4,dtype=q.dtype))
        torch.testing.assert_close(grad[2:6,j], torch.zeros(4,dtype=q.dtype))
        self.assertLess(abs(float(grad[6,j])), 1e-5)
        torch.testing.assert_close(arm[:2], arm[-2:].flip(0))
        torch.testing.assert_close(grad[:2,j], -grad[-2:,j].flip(0))
        self.assertLess(float(arm[-1].detach()), float(arm[-2].detach()))
        torch.testing.assert_close(waist, q.new_full((len(x),), .3))

    def test_real_prior_scaling_gate_and_extreme_elbows(self):
        mdp = self.task(soft_body_prior_weight=.02, soft_body_prior_tolerance=self.tolerance)
        f, state, a, history = self.fixture()
        q = state[4].clone()
        l,r = (JOINT_NAMES.index(n) for n in ELBOW_JOINT_NAMES)
        q[2,l], q[2,r] = mdp.kin.q_hi[l], mdp.kin.q_lo[r]
        s = (*state[:4], q, state[5])
        ref = mdp._reference(f, history['command'])
        ref['quality'] = torch.tensor([[1.],[0.],[.5]])
        with patch.object(mdp, '_reference', return_value=ref):
            total, terms = mdp.reward(s, f, a, a*.7, True, **history)
        with patch.object(self.base, '_reference', return_value=ref):
            without = self.base.reward(s, f, a, a*.7, **history)
        torch.testing.assert_close(total - without, terms['soft_body_prior_contribution'], atol=4e-9, rtol=0)
        torch.testing.assert_close(terms['soft_body_prior_contribution'], .02*CONTROL_DT*terms['soft_body_prior'])
        torch.testing.assert_close(terms['soft_body_prior_arm'][:2], torch.tensor([.7,0.]))
        torch.testing.assert_close(terms['soft_body_prior_waist'], torch.tensor([.3,0.,.15]))
        self.assertLess(float(terms['soft_body_prior'][2]), .5)
        self.assertGreater(float(terms['soft_body_prior_dead_zone_violation_rad'][2]), 0.)

    def test_rejects_conflicting_prior_or_bad_joint_mapping(self):
        with self.assertRaises(ValueError):
            self.task(body_prior_weight=.02, soft_body_prior_weight=.02, soft_body_prior_tolerance=self.tolerance)
        with self.assertRaises(ValueError):
            self.task(soft_body_prior_weight=.02, soft_body_prior_tolerance={'elbow':.25})
        with self.assertRaises(ValueError):
            self.task(soft_body_prior_weight=.02, soft_body_prior_tolerance=dict(self.tolerance,zarm_r4_joint=-.25))

    def test_best_ranks_survival_and_gait_before_posture(self):
        from supermimic.integration.m43_evaluation import gait_rank

        score = dict(survival_rate=1., complete_tracking_rate=1., consecutive_planned_touchdowns=4.,
                     first_touchdown_rate=1., foot_position_rmse_m=.03, root_xy_rmse_m=.05,
                     root_orientation_rmse_deg=10., elbows={'l': {'near_limit_ratio': .8}})
        best = dict(iteration=5000, aggregate=score)
        fall = dict(iteration=5500, aggregate=dict(score, survival_rate=2/3, foot_position_rmse_m=.01))
        missed = dict(iteration=6000, aggregate=dict(score, complete_tracking_rate=2/3,
                       elbows={'l': {'near_limit_ratio': .05}}))
        better_pose = dict(iteration=4500, aggregate=dict(score, elbows={'l': {'near_limit_ratio': .10}}))
        self.assertGreater(gait_rank(best, m49=True), gait_rank(fall, m49=True))
        self.assertGreater(gait_rank(best, m49=True), gait_rank(missed, m49=True))
        self.assertGreater(gait_rank(better_pose, m49=True), gait_rank(best, m49=True))


def check_m43_baseline(reference):
    from supermimic.integration.m42v3_ac import episode_input, gradient, load_trace, objective
    from supermimic.integration.m43_wm_audit import make_policy, make_wm, task_for
    from supermimic.kinematics import RobotKinematics
    from supermimic.mdp import unflatten_state
    from supermimic.networks import Critic

    root = REPO / 'outputs/m4_3'
    ck = torch.load(root / 'checkpoints/best.pt', map_location='cpu', mmap=True, weights_only=False)
    assert ck['iteration'] == 4500
    kin = RobotKinematics('cpu')
    args, motion, goals, mdp = task_for(ck, kin, reference, 'cpu')
    explicit = dict(ck, args=dict(ck['args'], m47_startup_steps=0, upper_action_smoothness_weight=0.))
    cfg, zero_motion, zero_goals, zero_mdp = task_for(explicit, kin, reference, 'cpu')
    assert not (mdp.action_history_enabled or zero_mdp.action_history_enabled)
    assert ck['physical_contract'] == mdp.physics_contract(args) == zero_mdp.physics_contract(cfg)
    assert ck['data_contract'] == motion.validation == zero_motion.validation
    assert (motion.num_frames, motion.duration) == (261, 5.2)
    with np.load(root / 'evaluation/rollouts.npz', allow_pickle=False) as archive:
        trial = load_trace(archive, 4500, 1001, 'cpu')
        applied = torch.from_numpy(archive['i4500_s1001_target'])

    f, _, command, xi, goal, prev, hist, features = episode_input(motion, mdp, goals, trial, args.hist)
    raw = trial['action']
    prev2 = torch.cat((torch.zeros_like(prev[:1]), prev[:-1]))
    state, after = trial['state'][:-1], unflatten_state(trial['state'][1:])
    reward = mdp.reward(after, f+1, raw, prev, command=command, ep_step=f+1)
    torch.testing.assert_close(reward, trial['reward'], rtol=0, atol=1e-6)
    torch.testing.assert_close(mdp.standing_pd_target(raw), applied, rtol=0, atol=1e-6)
    assert max(features.values()) < 1e-5, features
    obs = zero_goals.obs(zero_mdp, unflatten_state(state), prev, xi, f, command)
    torch.testing.assert_close(obs, goals.obs(mdp, unflatten_state(state), prev, xi, f, command), rtol=0, atol=0)

    # 历史 M4.3 路径和显式关闭两项的路径使用同一批真实起点、权重及随机状态。
    selected = torch.tensor([0, 1, 2, 5, 9, 10, 11, 40, 65, 90, 120, 172, 200, 230])
    batch = tuple(value[selected] for value in (state, prev, f, xi, hist, goal))
    context = (prev2[selected], unflatten_state(state[:1])[4].expand_as(batch[1]))
    pi, wm = make_policy(ck, mdp, goals, 'cpu'), make_wm(ck, 'cpu')
    critic = Critic(970).eval().requires_grad_(False)
    critic.load_state_dict(ck['critic'])
    limit = ck['imagination_contract']['bootstrap_upper_bound']
    old = objective(batch, pi, critic, wm, motion, mdp, goals, args, limit, 4801)
    new = objective(batch, pi, critic, wm, zero_motion, zero_mdp, zero_goals, cfg, limit, 4801, action_context=context)
    for key in ('A', 'B', 'C', 'bootstrap', 'actions', 'effective'):
        torch.testing.assert_close(new[key], old[key], rtol=0, atol=0)
    params = tuple(pi.parameters())
    old_grad, new_grad = gradient(old['A'], params), gradient(new['A'], params)
    torch.testing.assert_close(new_grad, old_grad, rtol=1e-6, atol=1e-7)
    print('M4.3 固定轨迹回归通过：观测/目标最大误差', features,
          '实测奖励最大误差', float((reward-trial['reward']).abs().max()),
          '16 步 BPTT 损失差', float((new['A']-old['A']).detach()),
          'Actor 梯度最大差', float((new_grad-old_grad).abs().max()), flush=True)


def check_m49_configs():
    directory = REPO / 'configs/roban/m4'
    base = json.loads((directory/'m4_8_c.json').read_text())
    a,b,c = (json.loads((directory/f'm4_9_{key}.json').read_text()) for key in 'abc')
    soft = {name:c[name] for name in ('soft_body_prior_weight','soft_body_prior_tolerance')}
    assert a == dict(base, iterations=10000)
    assert b == dict(a, body_prior_weight=.02)
    assert c == dict(a, **soft) and soft['soft_body_prior_weight'] > 0.
    old = json.loads((REPO/'outputs/m4_8/c_startup_smoothness/config/train.json').read_text())
    assert all(value == old[name] for name,value in base.items())
    assert old['lr_decay_iters'] == 250000 and old['lr'] == .0002 and old['critic_lr'] is None
    return soft


def smoke(reference, steps, *, m48=False, m49=False):
    torch.set_num_threads(2)
    checkpoints = []
    ablation = m48 or m49
    if m49:
        selected = check_m49_configs()
        designs = [('a_long_baseline', 10, .02, 0., 0.), ('b_original_prior', 10, .02, .02, 0.),
                   ('c_soft_prior', 10, .02, 0., selected['soft_body_prior_weight'])]
    elif m48:
        check_m43_baseline(reference)
        designs = [('a_startup_only', 10, 0., 0., 0.), ('b_smoothness_only', 0, .02, 0., 0.),
                   ('c_startup_smoothness', 10, .02, 0., 0.)]
    else:
        designs = [('a', steps, 0., 0., 0.), ('b', steps, .02, 0., 0.), ('c', steps, .02, .02, 0.)]
    transitions = 312 if ablation else 168
    phase = 'm49' if m49 else 'm48' if m48 else 'm47'
    with TemporaryDirectory(dir='/tmp', prefix=f'{phase}-smoke-') as temp:
        # 每组用同一随机种子，按本轮设计单独配置启动步数与奖励权重。
        for name, startup, smooth, body, soft in designs:
            root = Path(temp)/name
            config = f'm4_{phase[-1]}_{name[0]}' if ablation else 'm4_3'
            args = [sys.executable, 'supermimic/train.py', '--roban_config', f'configs/roban/m4/{config}.json',
                    '--m4_reference', str(reference), '--device', 'cuda:0', '--fresh_start',
                    f'--{phase}_smoke', '--log_dir', str(root/'checkpoints')]
            if ablation:
                args += [f'--{phase}_group', name[0]]
            else:
                args += ['--m47_startup_steps', str(startup), '--upper_action_smoothness_weight', str(smooth),
                         '--body_prior_weight', str(body)]
            result=subprocess.run(args,cwd=REPO,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=200)
            if result.returncode or 'M4 completed | iteration=2 |' not in result.stdout:
                raise RuntimeError(result.stdout)
            if any(s in result.stdout for s in ('CUDA error','Warning: EPA horizon','nefc overflow')):
                raise RuntimeError(result.stdout)
            if ablation and f'M4.{phase[-1]} Smoke 覆盖步龄 0/5/10/11' not in result.stdout:
                raise RuntimeError(result.stdout)
            initial,last=(torch.load(root/'checkpoints'/f'{tag}.pt',map_location='cpu',mmap=True,weights_only=False)
                          for tag in ('initial','last'))
            assert (initial['iteration'],initial['sim_steps'],initial['transfer_report'])==(0,0,None)
            assert (last['iteration'],last['sim_steps'])==(2,transitions) and last['rng']
            assert all(not initial[k]['state'] and last[k]['state'] for k in ('opt_pi','opt_wm','opt_v'))
            assert all(any(not torch.equal(initial[k][n],last[k][n]) for n in initial[k] if n.endswith('.weight'))
                       for k in ('pi','wm','critic'))
            assert all(torch.isfinite(t).all() for k in ('pi','wm','critic') for t in last[k].values())
            assert initial['imagination_contract']['startup_steps']==startup
            assert initial['task_contract']['wm_action']=='standing_equivalent_of_executed_pd_target'
            assert initial['policy_contract']['obs_dim']==69 and initial['critic']['norm.mean'].numel()==970
            assert initial['args']['upper_action_smoothness_weight']==smooth and initial['args']['body_prior_weight']==body
            assert initial['args']['soft_body_prior_weight']==soft
            with (root/'logs/train.csv').open() as log:
                rows=list(csv.DictReader(log))
            for item in rows:
                assert all(float(item[k])==0. for k in ('epa_resets','invalid_transitions','physics_exceptions'))
                assert float(item['reward_sync_max_error'])<=1.e-5 and float(item['goal_sync_max_error'])<=1.e-5
                assert all(math.isfinite(float(item[k])) for k in ('wm_loss', 'pi_loss', 'v_loss', 'pi_gradnorm'))
            values=tuple(float(rows[-1][k]) for k in ('wm_gradnorm','pi_gradnorm','critic_gradnorm',
                         'term_upper_action_smoothness_actor_gradnorm','term_body_prior_actor_gradnorm'))
            assert all(math.isfinite(x) for x in values)
            assert min(values[:3])>0 and (values[3]>0)==bool(smooth) and (values[4]>0)==bool(body)
            if ablation:
                for key in ('term_foot_actor_gradnorm', 'actor_bootstrap_gradnorm', 'total_reward'):
                    assert math.isfinite(float(rows[-1][key]))
                assert float(rows[-1]['term_foot_actor_gradnorm']) > 0
            if m49:
                soft_gradient = float(rows[-1]['term_soft_body_prior_actor_gradnorm'])
                assert math.isfinite(soft_gradient) and (soft_gradient > 0) == bool(soft)
                row = rows[-1]
                for k in ('soft_body_prior_contribution', 'body_prior_contribution', 'soft_body_prior_arm',
                          'soft_body_prior_waist', 'soft_body_prior_dead_zone_violation_rad', 'body_prior_arm', 'body_prior_waist'):
                    assert math.isfinite(float(row[k]))
                assert (float(row['body_prior_contribution']) > 0.) == bool(body)
                assert (float(row['soft_body_prior_contribution']) > 0.) == bool(soft)
                assert (float(row['soft_body_prior_dead_zone_violation_rad']) > 0.) == bool(soft)
                assert all(math.isfinite(float(v)) and abs(float(v)) <= 1.000001 for k,v in row.items()
                           if k.startswith('grad_cosine_'))
                for joint in ELBOW_JOINT_NAMES:
                    qs = [float(row[f'raw_{joint}_{k}']) for k in ('p05','p50','p95')]
                    assert all(math.isfinite(x) for x in qs) and qs == sorted(qs)
                    assert all(0. <= float(row[f'{k}_{joint}']) <= 1. for k in ('clamp','near_limit'))
                assert initial['args']['git_commit'] == os.environ['S54_SOURCE_GIT_SHA']
                assert initial['args']['git_worktree'] == os.environ['M49_SOURCE_GIT_STATUS']
                values += (soft_gradient,)
            checkpoints.append(initial)
            labels = '梯度 WM/Actor/Critic/Smooth/Body' + ('/Soft' if m49 else '')
            print(name,transitions,labels,values,
                  '实测回放重算误差',float(rows[-1]['reward_sync_max_error']),flush=True)
        comparison = 'm4_8/c_startup_smoothness' if m49 else 'm4_3'
        old=torch.load(f'outputs/{comparison}/checkpoints/initial.pt',map_location='cpu',mmap=True,weights_only=False)
        for ck in checkpoints:
            for name in ('pi','wm','critic'):
                assert ck[name].keys()==old[name].keys()
                assert all(torch.equal(v,old[name][k]) for k,v in ck[name].items())
            for key in ('goal_contract', 'policy_contract', 'physical_contract', 'data_contract'):
                assert ck[key] == old[key], key
        if ablation:
            ignored = {'roban_config', 'log_dir', f'{phase}_group', 'm47_startup_steps', 'upper_action_smoothness_weight',
                       'body_prior_weight', 'reward_config', 'soft_body_prior_weight', 'soft_body_prior_tolerance'}
            baseline = {k: v for k, v in checkpoints[0]['args'].items() if k not in ignored}
            for ck, design in zip(checkpoints, designs):
                assert {k: v for k, v in ck['args'].items() if k not in ignored} == baseline
                reward = dict(ck['args']['reward_config'])
                expected_reward = dict(old['args']['reward_config'])
                if m49:
                    assert ck['imagination_contract'] == old['imagination_contract']
                    expected_reward['body_prior_weight'] = design[3]
                    if design[4]:
                        expected_reward.update(selected)
                else:
                    del reward['upper_action_smoothness_weight']
                assert reward == expected_reward

            # 用 C 组两轮权重走一次正式周期评估入口，验证 Best 和轻量指标落盘。
            command = [sys.executable, '-m', 'supermimic.integration.m43_evaluation', '--out', str(root),
                       '--m4_reference', str(reference), '--device', 'cuda:0', f'--{phase}-periodic']
            completed = subprocess.run(command, cwd=REPO, capture_output=True, text=True, timeout=200)
            if completed.returncode:
                raise RuntimeError(completed.stdout + completed.stderr)
            directory = root / 'evaluation'
            summary = json.loads((directory / 'results.json').read_text())
            assert summary['best_iteration'] == 2
            assert (root / 'checkpoints/best.pt').samefile(root / 'checkpoints/iter_2.pt')
            assert {p.name for p in directory.iterdir()} == {'results.json'}
            result = summary['milestones']['2']
            assert result['aggregate']['valid_for_best'] and result['aggregate']['fatal_errors'] == 0
            for row in result['per_reset']:
                assert all('continuous_support_s' in event for event in row['footsteps'])
                assert row['upper_smoothness']['first_elbow_target_error_rad'] == 0.
                assert math.isfinite(row['upper_smoothness']['after_startup']['pd_target_diff2_rms_rad'])
                if m49:
                    assert set(row['elbows']) == set(ELBOW_JOINT_NAMES)
            best = root / 'checkpoints/best.pt'
            best.unlink()
            subprocess.run(command, cwd=REPO, check=True, capture_output=True, text=True, timeout=30)
            assert best.samefile(root / 'checkpoints/iter_2.pt')
            assert json.loads((directory / 'results.json').read_text()) == summary
            print(f'M4.{phase[-1]} 三种子轻量周期评估通过，控制指标及中断后 Best 恢复已核对。', flush=True)
    print(f'三组随机初始张量与 outputs/{comparison} 的 Initial 逐项相等，临时输出已清理。',flush=True)


if __name__=='__main__':
    if '--smoke' not in sys.argv:
        unittest.main()
    else:
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--smoke',action='store_true')
        stage = parser.add_mutually_exclusive_group()
        stage.add_argument('--m48', action='store_true', help='验证 M4.8 三组正式配置的隔离短训')
        stage.add_argument('--m49', action='store_true', help='验证 M4.9 三组正式配置的隔离短训')
        parser.add_argument('--reference',type=Path,default=Path('/m4_reference.npz'))
        parser.add_argument('--steps',type=int,choices=(5,10),default=5)
        options=parser.parse_args()
        smoke(options.reference,options.steps,m48=options.m48,m49=options.m49)

