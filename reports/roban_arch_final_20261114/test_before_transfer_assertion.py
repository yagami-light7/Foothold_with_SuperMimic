"""CPU contract/gradient tests. Newton tests use tools/roban_rollout.py separately."""
import os
os.environ['SUPERMIMIC_ROBOT']='roban'
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import copy
import json
import tempfile
import unittest

import mujoco
import numpy as np
import torch

from supermimic.roban import (NUM_J,XI_DIM,JOINT_NAMES,GOAL_FRAME_DIM,model_contract,default_joint_pos_list,
                             checkpoint_contract,foot_geometry,name_mapping,REPO)
from supermimic.kinematics import RobotKinematics
from supermimic.motion_reference import MotionReferenceProvider
from supermimic.roban_task import FootRewardConfig,RobanFootGoals,RobanFootMDP
from supermimic.goals import GoalSpec
from supermimic.mdp import flatten_state,unflatten_state
from supermimic.networks import Policy,Critic,WorldModel
from supermimic.terrain import FlatTerrainProvider,PillarTerrainProvider,HeightMapTerrainProvider
from supermimic.rotations import so3_exp,mat_to_quat,yaw_mat
from supermimic.roban_runtime import initialize_compatible
from supermimic.transfer import physical_contract,policy_contract
from supermimic.buffer import ReplayBuffer

torch.set_num_threads(2)


class RobanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(22)
        cls.tmp=tempfile.TemporaryDirectory(prefix='roban_tests_')
        cls.kin=RobotKinematics()
        model,_=model_contract();T=121
        p=np.tile(model.qpos0[:3],(T,1));p[:,0]=np.arange(T)*.002
        q=np.tile(default_joint_pos_list(),(T,1))
        q[:,JOINT_NAMES.index('zarm_l1_joint')]=.1*np.sin(np.arange(T)*.02)
        contact=np.tile([1,0],(T,1));contact[40:80]=[0,1]
        path=Path(cls.tmp.name)/'clip.npz'
        np.savez(path,root_pos=p,root_quat=np.tile([0.,0.,0.,1.],(T,1)),joint_pos=q,
                 joint_names=np.array(JOINT_NAMES),fps=50.,quaternion_order='xyzw',quality=np.ones(T),
                 contact=contact,contact_valid=np.ones((T,2),dtype=bool))
        cls.path=path;cls.motion=MotionReferenceProvider([path],cls.kin)
        cls.f=torch.tensor([3,45]);cls.S=cls.motion.ref_state(cls.f)
        cls.spec=GoalSpec('roban_foot',4,.1)
        cls.goals=RobanFootGoals(cls.motion,cls.kin,cls.spec)
        cls.command=cls.motion.anchor(cls.f)
        cls.a=torch.zeros(2,NUM_J);cls.xi=torch.zeros(2,XI_DIM)
        cls.default=torch.tensor(default_joint_pos_list())

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def task(self,**kwargs):
        return RobanFootMDP(self.kin,self.motion,self.default,self.goals,**kwargs)

    def test_01_actor_shape_dtype_device(self):
        mdp=self.task();o=mdp.obs(self.S,self.a,self.xi,self.f)
        g=self.goals.goal(self.S,self.f,torch.zeros_like(self.f),self.command)
        self.assertEqual(o.shape,(2,3*NUM_J+6));self.assertEqual(g.shape,(2,4*23))
        pi=Policy(o.shape[-1],g.shape[-1],hist=3)
        action=pi(o[:,None].expand(-1,3,-1),g)
        self.assertEqual(action.shape,(2,NUM_J));self.assertEqual(action.dtype,torch.float32)
        self.assertEqual(action.device.type,'cpu');self.assertTrue(torch.isfinite(action).all())

    def test_02_actor_body_reference_and_privilege_isolation(self):
        mdp=self.task();g=self.goals.goal(self.S,self.f,self.f*0,self.command)
        o=mdp.obs(self.S,self.a,self.xi,self.f)
        pi=Policy(mdp.obs_dim,g.shape[-1],hist=1);before=pi(o[:,None],g)
        backup=self.motion.q.clone()
        try:
            self.motion.q[:,self.task().upper_ids]+=1
            changed=self.goals.goal(self.S,self.f,self.f*0,self.command)
            torch.testing.assert_close(g,changed,rtol=0,atol=0)
            o1=mdp.obs(self.S,self.a,self.xi+99,self.f+1)
            torch.testing.assert_close(o,o1,rtol=0,atol=0)
            torch.testing.assert_close(before,pi(o1[:,None],changed),rtol=0,atol=0)
        finally:self.motion.q.copy_(backup)

    def test_03_critic_privilege_and_value_gradients(self):
        no=self.task();yes=self.task(critic_use_body_reference=True)
        X=no.critic_features(self.S,self.f,self.xi,command=self.command)
        Y=yes.critic_features(self.S,self.f,self.xi,command=self.command)
        self.assertEqual(Y.shape[-1]-X.shape[-1],self.spec.frames*GOAL_FRAME_DIM)
        backup=self.motion.q.clone()
        try:
            self.motion.q[:,yes.upper_ids]+=.2
            self.assertFalse(torch.equal(Y,yes.critic_features(self.S,self.f,self.xi,command=self.command)))
            torch.testing.assert_close(X,no.critic_features(self.S,self.f,self.xi,command=self.command),rtol=0,atol=0)
        finally:self.motion.q.copy_(backup)
        critic=Critic(Y.shape[-1]);loss=(critic(Y)-torch.tensor([.3,-.1])).square().mean();loss.backward()
        self.assertGreater(sum(float(p.grad.square().sum()) for p in critic.parameters()),0)

    def test_04_reward_scale_and_body_prior_bound(self):
        mdp=self.task(cfg=FootRewardConfig(body_prior_weight=.03))
        r,terms=mdp.reward(self.S,self.f,self.a,self.a,True,command=self.command)
        self.assertTrue(torch.isfinite(r).all());self.assertTrue((r<=mdp.max_step_reward+1e-6).all())
        self.assertTrue((terms['body_prior_contribution']<=.03*.02+1e-8).all())
        for key in ('swing_position','swing_orientation','stance_position','contact_schedule','touchdown','upright','survival'):
            self.assertTrue(((terms[key]>=0)&(terms[key]<=1+1e-6)).all(),key)
        changed=list(self.S);changed[0]=self.S[0]+self.S[0].new_tensor([.1,0,0])
        _,worse=mdp.reward(tuple(changed),self.f,self.a,self.a,True,command=self.command)
        self.assertTrue((worse['foot_objective']<terms['foot_objective']).all())

    def test_05_zero_prior_and_quality_gate(self):
        off=self.task();r0=off.reward(self.S,self.f,self.a,self.a,command=self.command)
        backup=self.motion.q.clone();qual=self.motion.quality.clone()
        try:
            self.motion.q[:]=100
            torch.testing.assert_close(r0,off.reward(self.S,self.f,self.a,self.a,command=self.command),rtol=0,atol=0)
            on=self.task(cfg=FootRewardConfig(body_prior_weight=.03))
            self.motion.quality.zero_()
            torch.testing.assert_close(r0,on.reward(self.S,self.f,self.a,self.a,command=self.command),rtol=0,atol=0)
        finally:self.motion.q.copy_(backup);self.motion.quality.copy_(qual)

    def test_06_terrain_switch_batch_and_void(self):
        flat=FlatTerrainProvider(self.kin);wm=WorldModel(hidden=(32,32));wt=WorldModel(hidden=(32,32),terrain_provider=flat)
        self.assertEqual(wt.in_dim-wm.in_dim,225)
        features=flat.features(self.S)
        self.assertEqual(features.relative_heights.shape,(2,3,25));self.assertTrue(features.valid_mask.all())
        self.assertTrue((features.relative_heights==0).all())
        provider=PillarTerrainProvider(self.kin,[[0,0,.1,.1,.5]])
        h,valid,edge=provider.surface(torch.tensor([[0.,0.],[.5,0.]]))
        self.assertEqual(valid.tolist(),[True,False]);self.assertEqual(h[0],.5);self.assertLess(edge[1],0)

    def test_07_terrain_predicted_position_gradient(self):
        provider=HeightMapTerrainProvider(self.kin,[[0.,1.],[0.,1.]],[[1,1],[1,1]],[0.,0.],1.)
        xy=torch.tensor([[.3,.4]],requires_grad=True);h,valid,_=provider.surface(xy)
        h.sum().backward();torch.testing.assert_close(xy.grad,torch.tensor([[1.,0.]]))
        p=self.S[0].clone().requires_grad_();S=(p,)+self.S[1:]
        pillar=PillarTerrainProvider(self.kin,[[0,0,.1,.1,.5]])
        initial=pillar(S)
        shifted=(p+torch.tensor([3.,0.,0.]),)+S[1:]
        self.assertFalse(torch.equal(initial,pillar(shifted)))

    def test_08_world_model_prediction_and_bptt(self):
        wm=WorldModel(hidden=(32,32),terrain_provider=FlatTerrainProvider(self.kin))
        target=list(self.S);target[4]=target[4]+.01
        target=tuple(target);wm.out_norm.update(wm.targets(self.S,target))
        opt=torch.optim.Adam(wm.parameters(),lr=.01)
        wm.in_norm.update(wm.inputs(self.S,self.a,self.xi))
        loss=wm.state_error(wm.step(self.S,self.a,self.xi),target).square().mean()
        opt.zero_grad();loss.backward();self.assertTrue(torch.isfinite(loss));opt.step()
        for p in wm.parameters():p.requires_grad_(False)
        pi=Policy(3*NUM_J+6,self.spec.dim,hist=1);mdp=self.task()
        S=self.S;previous=self.a;objective=0
        for t in range(3):
            f=self.f+t;g=self.goals.goal(S,f,self.f*0+t,self.command)
            a=pi(mdp.obs(S,previous)[:,None],g);S=wm.step(S,a,self.xi)
            objective=objective+mdp.reward(S,f+1,a,previous,command=self.command).mean();previous=a
        (-objective).backward()
        grads=[p.grad for p in pi.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in grads));self.assertGreater(sum(float(g.square().sum()) for g in grads),0)
        self.assertTrue(all(p.grad is None or not p.requires_grad for p in wm.parameters()))

    def test_09_reference_clock_heading_anchor(self):
        frames=self.motion.indices(self.f,self.spec.frames,self.spec.dt)
        torch.testing.assert_close(frames[:,0],self.f+5)
        with self.assertRaises(ValueError):self.motion.indices(self.f,3,.03)
        g=self.goals.goal(self.S,self.f,self.f*0,self.command)
        A=so3_exp(torch.tensor([[0.,0.,.8],[0.,0.,.8]]));t=torch.tensor([[2.,3.,0.],[2.,3.,0.]])
        S=((A@self.S[0][...,None]).squeeze(-1)+t,A@self.S[1],(A@self.S[2][...,None]).squeeze(-1),(A@self.S[3][...,None]).squeeze(-1))+self.S[4:]
        anchor=self.motion.anchor(self.f,A,t)
        moved=self.goals.goal(S,self.f,self.f*0,anchor)
        torch.testing.assert_close(g,moved,atol=2e-5,rtol=1e-5)
        sampled=self.motion.sample(frames,self.command)
        torch.testing.assert_close(sampled['q'],self.motion.q[frames])
        last=self.motion.indices(torch.tensor([119]),10,.1)
        self.assertTrue((last==120).all())

    def test_10_mujoco_joint_body_fk_alignment(self):
        model,rows=model_contract();data=mujoco.MjData(model)
        for seed in range(4):
            rng=np.random.default_rng(seed);data.qpos[:]=model.qpos0
            q=rng.uniform(np.array([r['range'][0] for r in rows])*.15,np.array([r['range'][1] for r in rows])*.15)
            for r,x in zip(rows,q):data.qpos[r['qposadr']]=x
            rv=torch.tensor(rng.normal(size=(1,3))*.2,dtype=torch.float32);R=so3_exp(rv)
            data.qpos[3:7]=mat_to_quat(R)[0].numpy()
            v=torch.tensor(rng.normal(size=(1,3)),dtype=torch.float32)
            w=torch.tensor(rng.normal(size=(1,3)),dtype=torch.float32)
            qd=torch.tensor(rng.normal(size=(1,NUM_J)),dtype=torch.float32)
            data.qvel[:3]=v[0].numpy();data.qvel[3:6]=(R[0].T@w[0]).numpy()
            for row in rows:data.qvel[row['dofadr']]=qd[0,row['canonical_index']]
            mujoco.mj_forward(model,data)
            bp,bR,bv,bw=self.kin.forward(torch.tensor(data.qpos[:3],dtype=torch.float32)[None],R,v,w,
                                     torch.tensor(q,dtype=torch.float32)[None],qd)
            ids=[model.body(n).id for n in self.kin.tracked_names]
            np.testing.assert_allclose(bp[0].numpy(),data.xpos[ids],atol=1e-5,rtol=1e-5)
            np.testing.assert_allclose(bR[0].numpy(),data.xmat[ids].reshape(-1,3,3),atol=1e-5,rtol=1e-5)
            for k,b in enumerate(ids):
                jp=np.zeros((3,model.nv));jr=jp.copy();mujoco.mj_jacBody(model,data,jp,jr,b)
                np.testing.assert_allclose(bv[0,k].numpy(),jp@data.qvel,atol=1e-4,rtol=1e-4)
                np.testing.assert_allclose(bw[0,k].numpy(),jr@data.qvel,atol=1e-4,rtol=1e-4)
        canonical,inverse=name_mapping(list(reversed(JOINT_NAMES)))
        self.assertEqual([canonical[i] for i in inverse],list(range(NUM_J)))
        self.assertEqual(NUM_J,len(rows));self.assertEqual(len(foot_geometry()),2)

    def test_11_checkpoint_optimizer_and_transfer(self):
        mdp=self.task();pi=Policy(mdp.obs_dim,self.spec.dim,hist=1);wm=WorldModel(hidden=(32,32));critic=Critic(mdp.critic_dim+self.spec.dim)
        opt=torch.optim.Adam(pi.parameters());g=self.goals.goal(self.S,self.f,self.f*0,self.command)
        pi(mdp.obs(self.S,self.a)[:,None],g).square().mean().backward();opt.step()
        ck=dict(robot_contract=checkpoint_contract(),goal_contract=self.spec.contract(),task_contract=mdp.contract(),
                args=dict(hist=1),physical_contract=physical_contract(obs_dim=mdp.obs_dim),
                policy_contract=policy_contract(1,obs_dim=mdp.obs_dim,observation_mode=mdp.observation_mode),
                pi=pi.state_dict(),wm=wm.state_dict(),critic=critic.state_dict(),opt_pi=opt.state_dict())
        path=Path(self.tmp.name)/'checkpoint.pt';torch.save(ck,path);loaded=torch.load(path,weights_only=False)
        pi2=Policy(mdp.obs_dim,self.spec.dim,hist=1);pi2.load_state_dict(loaded['pi']);opt2=torch.optim.Adam(pi2.parameters());opt2.load_state_dict(loaded['opt_pi'])
        torch.testing.assert_close(pi(mdp.obs(self.S,self.a)[:,None],g),pi2(mdp.obs(self.S,self.a)[:,None],g),rtol=0,atol=0)
        self.assertEqual(len(opt.state),len(opt2.state))
        wt=WorldModel(hidden=(32,32),terrain_provider=FlatTerrainProvider(self.kin))
        report=initialize_compatible(loaded,wt,pi2,loaded['physical_contract'],hist=1)
        self.assertIn('expanded_zero_terrain_columns',[r['status'] for r in report['mapping']])
        torch.testing.assert_close(wm.step(self.S,self.a,self.xi)[0],wt.step(self.S,self.a,self.xi)[0],rtol=0,atol=0)
        bad=copy.deepcopy(loaded);bad['robot_contract']={'robot':'kuavo_s54'}
        with self.assertRaises(ValueError):initialize_compatible(bad,wt,pi2,loaded['physical_contract'],hist=1)
        with self.assertRaises(ValueError):GoalSpec('roban_foot',5,.1).assert_checkpoint(loaded)

    def test_12_unlabelled_contacts_are_masked(self):
        with np.load(self.path) as z: payload={k:z[k] for k in z.files if k not in ('contact','contact_valid')}
        path=Path(self.tmp.name)/'unlabelled.npz';np.savez(path,**payload)
        motion=MotionReferenceProvider([path],self.kin)
        self.assertFalse(motion.contact_valid.any())
        goals=RobanFootGoals(motion,self.kin,self.spec);mdp=RobanFootMDP(self.kin,motion,self.default,goals)
        _,terms=mdp.reward(self.S,self.f,self.a,self.a,True,command=motion.anchor(self.f))
        self.assertTrue((terms['contact_schedule']==0).all());self.assertTrue((terms['touchdown']==0).all())

    def test_13_real_replay_rewards_and_episode_mask(self):
        mdp=self.task();buffer=ReplayBuffer(4,2,'cpu',foot_commands=True)
        for i in range(4):
            f=self.f+i;S=self.motion.ref_state(f);S1=self.motion.ref_state(f+1)
            r=mdp.reward(S1,f+1,self.a,self.a,command=self.command,ep_step=f*0+i+1)
            restored=unflatten_state(flatten_state(S1))
            torch.testing.assert_close(r,mdp.reward(restored,f+1,self.a,self.a,command=self.command,ep_step=f*0+i+1),atol=1e-6,rtol=1e-5)
            buffer.add(dict(s=flatten_state(S),s1=flatten_state(S1),a=self.a,a_prev=self.a,xi=self.xi,r=r,f=f,
                ep_step=f*0+i,done=torch.full((2,),i==1),term=torch.full((2,),i==1),timeout=torch.zeros(2,dtype=torch.bool),command=self.command,command_id=torch.tensor([0,1])))
        w=buffer.sample_windows(8,4)
        self.assertTrue((w['mask'][:,:2]==1).all());self.assertTrue((w['mask'][:,2:]==0).all())
        restored=buffer.commands(torch.tensor([1,0]))
        torch.testing.assert_close(restored.start_frame,self.command.start_frame.flip(0))

    def test_14_ablation_configs_consumed_and_paired(self):
        configs=[json.loads((REPO/f'configs/roban/E{i}.json').read_text()) for i in range(3)]
        keys={'critic_use_body_reference','body_prior_weight'}
        self.assertEqual({k:v for k,v in configs[0].items() if k not in keys},{k:v for k,v in configs[1].items() if k not in keys})
        self.assertEqual({k:v for k,v in configs[1].items() if k not in keys},{k:v for k,v in configs[2].items() if k not in keys})
        import argparse
        from supermimic.roban_runtime import add_roban_args
        # Parse defaults with the actual Trainer option schema (AST avoids launching physics).
        import ast
        tree=ast.parse((REPO/'supermimic/train.py').read_text())
        parser=argparse.ArgumentParser()
        for node in tree.body:
            if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='parser' and node.value.func.attr=='add_argument':
                call=node.value
                flags=[ast.literal_eval(x) for x in call.args]
                if flags[0].startswith('--'):parser.add_argument(*flags)
        from supermimic.goals import add_goal_args
        add_goal_args(parser)
        add_roban_args(parser,['--roban_config',str(REPO/'configs/roban/E1.json')])
        args=parser.parse_args([])
        self.assertTrue(args.critic_use_body_reference);self.assertEqual(args.body_prior_weight,0.)

    def test_15_resampling_30hz_and_clip_boundaries(self):
        with np.load(self.path) as z: payload={k:z[k] for k in z.files}
        payload['fps']=np.array(30.)
        path=Path(self.tmp.name)/'thirty.npz';np.savez(path,**payload)
        motion=MotionReferenceProvider([path,self.path],self.kin)
        self.assertEqual(motion.clip_len.tolist(),[201,121])
        torch.testing.assert_close(motion.root_v[:201,0],torch.full((201,),.06),atol=1e-5,rtol=1e-4)
        frames=motion.indices(torch.tensor([198,318]),3,.02)
        self.assertEqual(frames.tolist(),[[199,200,200],[319,320,321]])
        self.assertTrue(torch.isfinite(motion.qd).all());self.assertTrue(torch.isfinite(motion.root_w).all())

    def test_16_body_prior_active_gradient_and_swing_gradient(self):
        mdp=self.task(cfg=FootRewardConfig(body_prior_weight=.03))
        q=self.S[4].clone();q[:,mdp.upper_ids]+=.15;q[:,JOINT_NAMES.index('leg_l4_joint')]+=.08
        q.requires_grad_();S=self.S[:4]+(q,)+self.S[5:]
        _,terms=mdp.reward(S,self.f,self.a,self.a,True,command=self.command)
        prior_grad=torch.autograd.grad(terms['body_prior_contribution'].sum(),q,retain_graph=True)[0]
        foot_grad=torch.autograd.grad(terms['foot_objective'].sum(),q)[0]
        self.assertTrue(torch.isfinite(prior_grad).all());self.assertGreater(float(prior_grad[:,mdp.upper_ids].norm()),0)
        leg_ids=[i for i,n in enumerate(JOINT_NAMES) if n.startswith('leg_')]
        self.assertEqual(float(prior_grad[:,leg_ids].abs().sum()),0.)
        self.assertGreater(float(foot_grad[:,leg_ids].norm()),0.)
        self.assertLess(float((terms['body_prior_contribution'].max()/terms['foot_objective'].min()).detach()),.05)

    def test_17_clock_mismatch_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.goals.goal(self.S,self.f+1,self.f*0,self.command)
        with self.assertRaises(RuntimeError):
            self.task().reward(self.S,self.f,self.a,self.a,command=self.command,ep_step=torch.ones_like(self.f))


if __name__=='__main__':unittest.main(verbosity=2)
