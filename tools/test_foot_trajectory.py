"""CPU trajectory, goal, replay and no-update BPTT regressions (no Isaac Lab)."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from supermimic.foot_trajectory import AlternatingConfig, FootstepCommand, query, sample_alternating
from supermimic.goals import GoalSource, GoalSpec, foot_goal, resolve_goal_config
from supermimic.rotations import axis_angle_to_mat, mat_to_quat, quat_to_mat, so3_exp, yaw_mat
MOTION_DIR = 'data/retarget_data/reference_startup_fixed'


def command(B=2, dtype=torch.float64):
    p = torch.tensor([[[0, .1, .06], [0, -.1, .06]]], dtype=dtype).repeat(B, 1, 1)
    q = torch.tensor([[[1, 0, 0, 0], [1, 0, 0, 0]]], dtype=dtype).repeat(B, 1, 1)
    target = p.clone(); target[:, 0, 0] = .2; target[:, 1, 0] = .4
    yaw = torch.tensor([[.8, 1.4]], dtype=dtype).expand(B, -1)
    z = torch.zeros_like(yaw)
    tq = torch.stack((torch.cos(yaw/2), z, z, torch.sin(yaw/2)), -1)
    times = lambda x: torch.tensor([x], dtype=dtype).repeat(B, 1)
    return FootstepCommand(p, q, torch.tensor([[0, 1]]).repeat(B, 1), target, tq,
                           times([.2, 1.]), times([.8, 1.6]), times([.1, .12]),
                           torch.full((B,), 2), q[:, 0].clone()).validate()


class TrajectoryTests(unittest.TestCase):
    def test_endpoints_apex_and_contact_boundaries(self):
        c = command(1)
        t = torch.tensor([[-.1, 0, .2, .5, .8, 1., 1.3, 1.6, 10]], dtype=torch.float64)
        r = query(c, t)
        expected = torch.tensor([[[1,1], [1,1], [0,1], [0,1], [1,1], [1,0], [1,0], [1,1], [1,1]]], dtype=torch.bool)
        self.assertTrue(torch.equal(r.contact, expected))
        torch.testing.assert_close(r.position[0, 2], c.initial_pos[0])
        torch.testing.assert_close(r.position[0, 4, 0], c.target_pos[0, 0])
        torch.testing.assert_close(r.position[0, -1], c.target_pos[0])
        self.assertAlmostEqual(float(r.position[0, 3, 0, 2]), .16)
        self.assertAlmostEqual(float(r.position[0, 6, 1, 2]), .18)
        for i in (0, 1, 2, 4, 5, 7, 8):
            self.assertLess(float(r.velocity[:, i].abs().max()), 1e-12)
            self.assertLess(float(r.acceleration[:, i].abs().max()), 1e-11)
        torch.testing.assert_close(r.quaternion.norm(dim=-1), torch.ones_like(r.contact, dtype=t.dtype))
        # Mid-swing orientation is halfway along the shortest rotation.
        self.assertAlmostEqual(float(r.quaternion[0, 3, 0, 0]), float(torch.cos(torch.tensor(.2, dtype=t.dtype))))

    def test_c2_continuity_and_derivatives(self):
        c = command(1)
        for event in (.2, .5, .8, 1., 1.3, 1.6):
            eps = 1e-6
            r = query(c, torch.tensor([[event-eps, event, event+eps]], dtype=torch.float64))
            self.assertLess(float((r.position[:, 2] - r.position[:, 0]).abs().max()), 2e-5)
            self.assertLess(float((r.velocity[:, 2] - r.velocity[:, 0]).abs().max()), 2e-4)
            self.assertLess(float((r.acceleration[:, 2] - r.acceleration[:, 0]).abs().max()), .003)
            # Quaternion sign is canonical in this fixture; check zero boundary angular speed.
            if event in (.2, .8, 1., 1.6):
                self.assertLess(float((r.quaternion[:, 2] - r.quaternion[:, 0]).abs().max()) / (2*eps), 1e-6)
        t = torch.tensor([[.34, .66, 1.18, 1.44]], dtype=torch.float64)
        eps = 1e-6
        r, lo, hi = query(c, t), query(c, t-eps), query(c, t+eps)
        torch.testing.assert_close((hi.position-lo.position)/(2*eps), r.velocity, atol=1e-8, rtol=1e-7)
        torch.testing.assert_close((hi.velocity-lo.velocity)/(2*eps), r.acceleration, atol=1e-7, rtol=1e-7)

    def test_batch_unsorted_queries_and_stance_lock(self):
        c = command(3)
        c.target_pos[1, :, 0] += .5
        t = torch.tensor([[1.4, .3, 9., .3], [.9, 1.2, -.2, .6], [0., .1, 1.5, .8]], dtype=torch.float64)
        result = query(c, t)
        for b in range(3):
            one = query(c.index(b), t[b:b+1])
            for k in vars(one):
                torch.testing.assert_close(getattr(result, k)[b:b+1], getattr(one, k))
        r = query(c, torch.tensor([[.21, .4, .79]], dtype=t.dtype).expand(3, -1))
        torch.testing.assert_close(r.position[:, :, 1], c.initial_pos[:, None, 1].expand(-1, 3, -1))

    def test_explicit_non_alternating_sequence_and_padding(self):
        c = command(2)
        c.foot[:, 1] = 0
        c.lengths[1] = 0
        c.validate()
        r = query(c, torch.tensor([[.9, 1., 2.], [.9, 1., 2.]], dtype=torch.float64))
        torch.testing.assert_close(r.position[0, :2, 0], c.target_pos[0, :1].expand(2, -1))
        torch.testing.assert_close(r.position[0, -1, 0], c.target_pos[0, 1])
        torch.testing.assert_close(r.position[0, :, 1], c.initial_pos[0, 1].expand(3, -1))
        torch.testing.assert_close(r.position[1], c.initial_pos[1].expand(3, -1, -1))
        self.assertTrue(r.contact[1].all())

    def test_sampling_reproducible_and_alternating(self):
        c = command(4)
        cfg = AlternatingConfig(num_steps=6)
        a = sample_alternating(c.initial_pos, c.initial_quat, c.heading, cfg, torch.Generator().manual_seed(54))
        b = sample_alternating(c.initial_pos, c.initial_quat, c.heading, cfg, torch.Generator().manual_seed(54))
        for k in a.tensors():
            torch.testing.assert_close(getattr(a, k), getattr(b, k))
        self.assertTrue(torch.equal(a.foot[0], torch.tensor([0,1,0,1,0,1])))
        self.assertGreater(float(a.target_pos[:, :, 0].std(0).max()), 0)
        r = query(a, torch.linspace(0, 5, 501, dtype=c.initial_pos.dtype)[None].expand(4,-1))
        self.assertTrue(r.contact.any(-1).all())
        torch.testing.assert_close(a.target_pos[..., 1], c.initial_pos.gather(1, a.foot[..., None].expand(-1,-1,3))[...,1])

    def test_invalid_commands_and_quaternion_sign(self):
        c = command(1); c.touchdown[0, 0] = .1
        with self.assertRaises(ValueError): c.validate()
        c = command(1); c.foot[0, 1] = 0; c.liftoff[0, 1] = .6
        with self.assertRaises(ValueError): c.validate()
        c = command(1); d = c.clone(); d.target_quat *= -1
        t = torch.linspace(0, 2, 101, dtype=c.initial_pos.dtype)[None]
        torch.testing.assert_close(quat_to_mat(query(c,t).quaternion), quat_to_mat(query(d,t).quaternion))

    def test_compile_query_and_goal(self):
        c = command(2, torch.float32)
        t = torch.tensor([[.4, 1.3], [.7, 1.1]])
        compiled = torch.compile(query, backend='eager', fullgraph=True)
        torch.testing.assert_close(compiled(c, t).position, query(c, t).position)
        S = (torch.zeros(2,3), torch.eye(3).expand(2,-1,-1))
        fn = torch.compile(foot_goal, backend='eager', fullgraph=True)
        torch.testing.assert_close(fn(S,c,torch.zeros(2,dtype=torch.long),GoalSpec('foot_trajectory')),
                                   foot_goal(S,c,torch.zeros(2,dtype=torch.long),GoalSpec('foot_trajectory')))


class GoalTests(unittest.TestCase):
    def test_world_local_roundtrip_and_equivariance(self):
        c = command(2)
        yaw = torch.tensor([.4, -.6], dtype=torch.float64)
        R = axis_angle_to_mat(torch.tensor([0.,0.,1.], dtype=yaw.dtype), yaw)
        p = torch.tensor([[.1,.2,.8],[-.3,.1,.9]], dtype=yaw.dtype)
        S, step, spec = (p,R), torch.tensor([8,13]), GoalSpec('foot_trajectory')
        g = foot_goal(S,c,step,spec).reshape(2,10,23)
        ref = query(c, step[:,None]*.02 + torch.arange(1,11,dtype=p.dtype)[None]*.1)
        local = torch.stack((g[...,:3],g[...,11:14]),2)
        recovered = (yaw_mat(R)[:,None,None] @ local[...,None]).squeeze(-1)+p[:,None,None]
        torch.testing.assert_close(recovered, ref.position, atol=1e-8, rtol=1e-8)
        J = axis_angle_to_mat(torch.tensor([0.,0.,1.],dtype=p.dtype), torch.tensor(.7,dtype=p.dtype))
        shift = p.new_tensor([3.,-2.,.4])
        moved = c.clone()
        for name in ('initial_pos','target_pos'):
            setattr(moved,name,(J@getattr(c,name)[...,None]).squeeze(-1)+shift)
        for name in ('initial_quat','target_quat','heading'):
            setattr(moved,name,mat_to_quat(J@quat_to_mat(getattr(c,name))))
        transformed = foot_goal(((J@p[...,None]).squeeze(-1)+shift,J@R),moved,step,spec)
        torch.testing.assert_close(transformed,g.flatten(1),atol=2e-8,rtol=2e-8)
        # Moving the robot changes local targets, while reconstructing the same world command.
        p2 = p + p.new_tensor([.2,-.1,0])
        g2 = foot_goal((p2,R),c,step,spec).reshape(2,10,23)
        self.assertFalse(torch.allclose(g2,g))
        rec2=(yaw_mat(R)[:,None]@g2[...,:3,None]).squeeze(-1)+p2[:,None]
        torch.testing.assert_close(rec2,ref.position[:,:,0],atol=1e-8,rtol=1e-8)

    def test_goal_keeps_state_gradients(self):
        c=command(2)
        p=torch.zeros(2,3,dtype=torch.float64,requires_grad=True)
        rv=torch.tensor([[.1,.05,.3],[.05,.1,-.2]],dtype=p.dtype,requires_grad=True)
        g=foot_goal((p,so3_exp(rv)),c,torch.tensor([4,8]),GoalSpec('foot_trajectory'))
        (g*torch.linspace(.3,1,g.shape[-1],dtype=p.dtype)).sum().backward()
        for grad in (p.grad,rv.grad):
            self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.norm()),0)

    def test_contracts_and_defaults(self):
        self.assertEqual(GoalSpec().dim,730)
        spec=GoalSpec('foot_trajectory');self.assertEqual(spec.dim,230);self.assertEqual(spec.dt,.1)
        GoalSpec().assert_checkpoint({'args':{'goal_frames':10}})
        with self.assertRaises(ValueError): spec.assert_checkpoint({'args':{'goal_frames':10}})
        with self.assertRaises(ValueError): GoalSpec('motion',dt=.1)
        with self.assertRaises(ValueError): GoalSpec('foot_trajectory',dt=float('nan'))
        a=argparse.Namespace(goal_mode='foot_trajectory',goal_frames=10,goal_dt=None,foot_config=None)
        spec,cfg=resolve_goal_config(a)
        spec.assert_checkpoint({'goal_contract':spec.contract(),'args':vars(a)},cfg)
        bad=spec.contract();bad['version']=2
        with self.assertRaises(ValueError):spec.assert_checkpoint({'goal_contract':bad})


class ReplayTests(unittest.TestCase):
    def test_snapshots_ring_reclamation_and_history(self):
        from supermimic.buffer import ReplayBuffer,FIELDS
        replay=ReplayBuffer(3,2,'cpu',foot_commands=True)
        live=command(2,torch.float32)
        def add(ids,steps):
            tr={}
            for k,(shape,dtype) in FIELDS.items():
                shape=(shape,) if isinstance(shape,int) else shape
                tr[k]=torch.zeros((2,)+shape,dtype=dtype)
            tr.update(command_id=torch.tensor(ids),command=live.clone(),ep_step=torch.tensor(steps))
            replay.add(tr)
        add([0,1],[0,0]);saved=replay.commands(torch.tensor([0])).clone()
        add([0,1],[1,1])
        live.target_pos[0]+=10  # reset to a different command before its next transition
        add([2,1],[0,2]);add([2,1],[1,3])
        torch.testing.assert_close(replay.commands(torch.tensor([0])).target_pos,saved.target_pos)
        add([2,3],[2,0]);self.assertNotIn(0,replay.command_store)
        starts,hist=replay.sample_starts(128,4)
        self.assertTrue((hist['command_id']==starts['command_id'][:,None]).all())
        self.assertTrue((hist['ep_step']<=starts['ep_step'][:,None]).all())
        self.assertTrue((hist['ep_step']>=0).all())
        self.assertEqual(replay.command_refs,Counter(replay.data['command_id'].flatten().tolist()))
        resolved=replay.commands(starts['command_id'])
        t=starts['ep_step'][:,None].float()*.02+.1
        for i in range(10):
            one=replay.command_store[int(starts['command_id'][i])]
            torch.testing.assert_close(query(resolved.index(i),t[i:i+1]).position,query(one,t[i:i+1]).position)
        # Motion replay retains its original field allocation.
        self.assertEqual(set(ReplayBuffer(2,1,'cpu').data),set(FIELDS))


class S54IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from supermimic.kinematics import S54Kinematics
        from supermimic.motion import MotionLib
        from supermimic.mdp import MDP
        from supermimic.s54 import default_joint_pos_list
        cls.kin=S54Kinematics()
        cls.motion=MotionLib.from_dir(MOTION_DIR,cls.kin,'cpu')
        cls.mdp=MDP(cls.kin,cls.motion,torch.tensor(default_joint_pos_list()))
        cls.frames=cls.motion.clip_start[:1].expand(2)+torch.tensor([100,130])
        cls.S=cls.motion.ref_state(cls.frames)
        cls.goals=GoalSource(cls.motion,cls.kin,GoalSpec('foot_trajectory'))

    def test_grounding_and_motion_observation_isolation(self):
        from supermimic.s54 import NUM_J,XI_DIM
        from supermimic.mdp import OBS_DIM
        source=self.goals;c=source.new_command(self.S)
        self.assertTrue((c.initial_pos[...,2]>0).all())
        a=torch.zeros(2,NUM_J);xi=torch.zeros(2,XI_DIM)
        o=source.obs(self.mdp,self.S,a,xi,self.frames,c)
        self.assertEqual(o.shape,(2,OBS_DIM))
        torch.testing.assert_close(o,source.obs(self.mdp,self.S,a,xi,self.frames+40,c))
        hist=tuple(x[:,None].expand((-1,3)+x.shape[1:]) for x in self.S)
        oh=source.obs(self.mdp,hist,a[:,None].expand(-1,3,-1),xi[:,None].expand(-1,3,-1),self.frames[:,None].expand(-1,3),c)
        torch.testing.assert_close(oh,o[:,None].expand(-1,3,-1))
        legacy=GoalSource(self.motion,self.kin)
        torch.testing.assert_close(legacy.goal(self.S,self.frames),self.motion.goal(self.frames,10))
        torch.testing.assert_close(legacy.obs(self.mdp,self.S,a,xi,self.frames),self.mdp.obs(self.S,a,xi,self.frames))

    def test_full_policy_critic_and_bptt_without_optimizer(self):
        from supermimic.mdp import OBS_DIM,NUM_J,XI_DIM,INV_DIM,CRITIC_EXTRA,normalize_xi
        from supermimic.networks import Policy,WorldModel,Critic
        torch.manual_seed(54)
        source=self.goals;c=source.new_command(self.S)
        pi=Policy(OBS_DIM,source.spec.dim,hist=2);wm=WorldModel();critic=Critic(INV_DIM+CRITIC_EXTRA+source.spec.dim)
        a=torch.zeros(2,NUM_J);xi=torch.zeros(2,XI_DIM);xi[:,0]=1;xn=normalize_xi(xi)
        S=self.S;f=self.frames;step=torch.tensor([10,13])
        o=source.obs(self.mdp,S,a,xi,f,c);g=source.goal(S,f,step,c)
        pi.obs_norm.update(o);pi.goal_norm.update(g)
        hist=o[:,None].repeat(1,2,1)
        with torch.no_grad(): self.assertEqual(pi(hist,g).shape,(2,29))
        wm.in_norm.update(wm.inputs(S,a,xn));wm.out_norm.update(wm.targets(S,self.motion.ref_state(f+1)))
        for module in (wm,critic):
            for p in module.parameters():p.requires_grad_(False)
        J=torch.zeros(2)
        for _ in range(3):
            act=pi(hist,g);S=wm.step(S,act,xn);f=f+1;step=step+1
            J=J+self.mdp.reward(S,f,act,a);a=act
            o=source.obs(self.mdp,S,a,xi,f,c);hist=torch.cat((hist[:,1:],o[:,None]),1)
            g=source.goal(S,f,step,c)
        X=torch.cat((self.mdp.critic_features(S,f,xn),g),-1)
        self.assertEqual(X.shape,(2,414))
        loss=-(J+.99**3*critic(X)).mean();loss.backward()
        grads=[p.grad for p in pi.parameters() if p.grad is not None]
        self.assertTrue(torch.isfinite(loss));self.assertTrue(all(torch.isfinite(x).all() for x in grads))
        self.assertGreater(float(sum(x.abs().sum() for x in grads)),0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion_dir',default='data/retarget_data/reference_startup_fixed')
    parser.add_argument('--out',default='results/s54/foot_trajectory_cpu.json')
    args,remaining=parser.parse_known_args();MOTION_DIR=args.motion_dir
    torch.set_num_threads(4)
    result=unittest.main(argv=[sys.argv[0]]+remaining,exit=False,verbosity=2).result
    report=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                status='PASS' if result.wasSuccessful() else 'FAIL',training_started=False,
                goal_frame_dim=23,goal_dim=230,obs_dim=99,action_dim=29,critic_dim=414)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));sys.exit(not result.wasSuccessful())
