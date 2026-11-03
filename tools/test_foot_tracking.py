"""Independent foot MDP preflight; optional learned-WM Actor-gradient audit."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from supermimic.foot_mdp import FootTrackingCfg,FootTrackingMDP,assert_task_checkpoint
from supermimic.goals import GoalSource,GoalSpec
from supermimic.foot_trajectory import query
from supermimic.kinematics import S54Kinematics
from supermimic.mdp import OBS_DIM,normalize_xi
from supermimic.rotations import so3_exp
from supermimic.s54 import default_joint_pos_list,NUM_J,XI_DIM,assert_checkpoint


def setup(mode='standing',device='cpu',dtype=torch.float32):
    kin=S54Kinematics(device,dtype=dtype)
    cfg=FootTrackingCfg(command_mode=mode)
    goals=GoalSource(None,kin,GoalSpec('foot_trajectory'),command_mode=mode)
    mdp=FootTrackingMDP(kin,torch.tensor(default_joint_pos_list(),device=device,dtype=dtype),goals,cfg)
    return kin,goals,mdp


class FootMDPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kin,cls.goals,cls.mdp=setup()

    def data(self):
        S=self.mdp.nominal_state(2)
        return S,self.goals.new_command(S),torch.zeros(2,NUM_J),torch.ones(2,dtype=torch.long)

    def reward(self,S,c,a,step,mdp=None):
        return (mdp or self.mdp).reward(S,None,a,a,True,command=c,ep_step=step)

    def test_no_motion_dependency_and_neutral_reward(self):
        S,c,a,step=self.data();r,t=self.reward(S,c,a,step)
        self.assertIsNone(self.mdp.motion);self.assertIsNone(self.goals.motion)
        self.assertTrue(torch.equal(c.lengths,torch.zeros(2,dtype=torch.long)))
        torch.testing.assert_close(r,torch.full_like(r,self.mdp.max_step_reward),atol=1e-7,rtol=1e-6)
        self.assertTrue((t['joint_limit']==0).all())
        self.assertTrue(query(c,torch.tensor([[0.,10.,1000.]]).expand(2,-1)).contact.all())
        # Deliberately invalid human-motion indices cannot affect any foot head.
        huge=torch.full((2,),10**12,dtype=torch.long)
        torch.testing.assert_close(r,self.mdp.reward(S,huge,a,a,command=c,ep_step=step))
        xi=torch.zeros(2,XI_DIM)
        obs=self.goals.obs(self.mdp,S,a,xi,huge,c)
        self.assertEqual(obs.shape,(2,99))
        self.assertEqual(self.mdp.critic_features(S,huge,xi,command=c,ep_step=step).shape,(2,135))
        self.assertFalse(self.mdp.termination(S,huge).any())
        with self.assertRaises(ValueError):self.mdp.obs(S,a,xi,huge)

    def test_tracking_stance_upright_action_and_limit_response(self):
        S,c,a,step=self.data();r,t=self.reward(S,c,a,step)
        shifted=list(S);shifted[0]=S[0]+S[0].new_tensor([.06,0,0])
        r2,t2=self.reward(tuple(shifted),c,a,step)
        self.assertTrue((r2<r).all());self.assertTrue((t2['stance_stability']<t['stance_stability']).all())
        moving=list(S);moving[2]=S[2]+.2
        _,mv=self.reward(tuple(moving),c,a,step)
        self.assertTrue((mv['stance_velocity']<t['stance_velocity']).all())
        tilted=list(S);rv=S[0].new_tensor([[.15,0,0]]).expand(2,-1);tilted[1]=so3_exp(rv)@S[1]
        _,tt=self.reward(tuple(tilted),c,a,step)
        self.assertTrue((tt['base_upright']<t['base_upright']).all())
        rate=self.mdp.reward(S,None,a+1,a,command=c,ep_step=step)
        self.assertTrue((rate<r).all())
        outside=list(S);outside[4]=self.mdp.kin.q_hi[None].expand(2,-1)+.05
        _,lim=self.reward(tuple(outside),c,a,step)
        self.assertTrue((lim['joint_limit']>0).all())
        fallen=list(S);fallen[0]=S[0].clone();fallen[0][:,2]-=.4
        self.assertTrue(self.mdp.termination(tuple(fallen)).all())
        self.assertTrue((self.mdp.soft_termination(tuple(fallen))>.99).all())

    def test_swing_and_touchdown_are_time_specific(self):
        _,goals,mdp=setup('alternating')
        S=mdp.nominal_state(2);c=goals.new_command(S);a=torch.zeros(2,NUM_J)
        # A simple scheduled event with an endpoint exactly on a control tick.
        c.lengths[:]=1;c.foot[:,0]=0;c.liftoff[:,0]=.2;c.touchdown[:,0]=.8
        c.target_pos[:,0]=c.initial_pos[:,0]+torch.tensor([.1,0.,0.])
        c.target_quat[:,0]=c.initial_quat[:,0];c.validate()
        total=torch.zeros(2,dtype=torch.long)
        for i in (39,40,41):
            _,terms=mdp.reward(S,None,a,a,True,command=c,ep_step=torch.full((2,),i))
            total+=terms['touchdown_events']
            if i==40:self.assertTrue((terms['touchdown_accuracy']>0).all())
            else:self.assertTrue((terms['touchdown_accuracy']==0).all())
        self.assertTrue(torch.equal(total,torch.ones_like(total)))
        future=goals.goal(S,None,torch.full((2,),30),c).reshape(2,10,23)
        current=mdp._reference(c,torch.full((2,),40))
        torch.testing.assert_close(future[:,1,9],current.contact[:,0,0].float())
        _,swing=mdp.reward(S,None,a,a,True,command=c,ep_step=torch.full((2,),25))
        self.assertTrue((swing['swing_tracking']>0).all())
        self.assertTrue((swing['swing_tracking']<.5).all())
        # Landing at the target improves touchdown accuracy.
        moved=list(S);moved[0]=S[0]+torch.tensor([.1,0,0])
        _,before=mdp.reward(S,None,a,a,True,command=c,ep_step=torch.full((2,),40))
        _,after=mdp.reward(tuple(moved),None,a,a,True,command=c,ep_step=torch.full((2,),40))
        self.assertTrue((after['touchdown_accuracy']>before['touchdown_accuracy']).all())

    def test_time_major_batch_and_single_queries_match(self):
        S,c,a,step=self.data();c.initial_pos[1,:,0]+=.03
        states=tuple(x.unsqueeze(0).expand((3,)+x.shape).flatten(0,1) for x in S)
        steps=torch.tensor([[1,3],[7,9],[15,20]])
        cmds=c.repeat(3);act=a.repeat(3,1)
        batch=self.mdp.reward(states,None,act,act,command=cmds,ep_step=steps.flatten())
        expected=torch.cat([self.mdp.reward(S,None,a,a,command=c,ep_step=steps[i]) for i in range(3)])
        torch.testing.assert_close(batch,expected)
        ref=query(cmds,steps.flatten()[:,None]*.02)
        self.assertEqual(ref.position.shape,(6,1,2,3))

    def test_fk_reward_gradient_matches_finite_difference(self):
        _,goals,mdp=setup(dtype=torch.float64)
        S=list(mdp.nominal_state(1));c=goals.new_command(tuple(S))
        S[0]=S[0]+S[0].new_tensor([.025,-.01,0]);S[0].requires_grad_()
        a=torch.zeros(1,NUM_J,dtype=torch.float64);step=torch.tensor([1])
        _,terms=mdp.reward(tuple(S),None,a,a,True,command=c,ep_step=step)
        gradient=torch.autograd.grad(terms['foot_objective'].sum(),S[0])[0]
        eps=1e-6
        values=[]
        for sign in (-1,1):
            copy=list(S);dp=torch.zeros_like(S[0]);dp[:,0]=sign*eps;copy[0]=S[0].detach()+dp
            _,t=mdp.reward(tuple(copy),None,a,a,True,command=c,ep_step=step);values.append(t['foot_objective'])
        torch.testing.assert_close(gradient[:,0],(values[1]-values[0])/(2*eps),atol=1e-7,rtol=1e-5)
        self.assertGreater(float(gradient.norm()),0)

    def test_contract_rejects_goal_only_checkpoint(self):
        with self.assertRaises(ValueError):assert_task_checkpoint({},self.mdp)
        assert_task_checkpoint({'task_contract':self.mdp.contract()},self.mdp)
        with self.assertRaises(ValueError):FootTrackingCfg(position_std=0)

    def test_leg_fk_gradient_and_compiled_heads(self):
        from supermimic.s54 import JOINT_NAMES
        S,c,a,step=self.data();state=list(S)
        joint=JOINT_NAMES.index('leg_l3_joint')
        state[4]=S[4].clone();state[4][:,joint]=.02;state[4].requires_grad_()
        _,terms=self.mdp.reward(tuple(state),None,a,a,True,command=c,ep_step=step)
        gradient=torch.autograd.grad(terms['foot_objective'].sum(),state[4])[0]
        self.assertTrue(torch.isfinite(gradient).all());self.assertGreater(float(gradient[:,joint].abs().sum()),0)
        fn=torch.compile(self.mdp.reward,backend='eager',fullgraph=True)
        expected=self.mdp.reward(tuple(state),None,a,a,command=c,ep_step=step)
        torch.testing.assert_close(fn(tuple(state),None,a,a,command=c,ep_step=step),expected)
        critic=torch.compile(self.mdp.critic_features,backend='eager',fullgraph=True)
        xi=torch.zeros(2,XI_DIM)
        torch.testing.assert_close(critic(S,None,xi,command=c,ep_step=step),
                                   self.mdp.critic_features(S,None,xi,command=c,ep_step=step))


def learned_gradient_audit(checkpoint_path,device):
    from supermimic.networks import Policy,WorldModel
    ck=torch.load(checkpoint_path,map_location=device,weights_only=False);assert_checkpoint(ck)
    a=ck['args'];kin=S54Kinematics(device)
    cfg=FootTrackingCfg(**a['foot_task_config'])
    from supermimic.foot_trajectory import AlternatingConfig
    spec=GoalSpec(a['goal_mode'],a['goal_frames'],a['goal_dt'])
    goals=GoalSource(None,kin,spec,AlternatingConfig(**a['foot_config']),cfg.command_mode)
    mdp=FootTrackingMDP(kin,torch.tensor(default_joint_pos_list(),device=device),goals,cfg)
    spec.assert_checkpoint(ck,goals.foot_config);assert_task_checkpoint(ck,mdp)
    wm=WorldModel().to(device);wm.load_state_dict(ck['wm'])
    pi=Policy(OBS_DIM,spec.dim,hist=a['hist']).to(device);pi.load_state_dict(ck['pi'])
    for p in wm.parameters():p.requires_grad_(False)
    torch.manual_seed(54)
    nominal=mdp.nominal_state(4);command=goals.new_command(nominal)
    S=mdp.perturb_reset(nominal);prev=torch.zeros(4,NUM_J,device=device)
    xi=torch.zeros(4,XI_DIM,device=device);xi[:,0]=1;xn=normalize_xi(xi)
    step=torch.zeros(4,dtype=torch.long,device=device)
    obs=goals.obs(mdp,S,prev,xi,None,command);history=obs[:,None].repeat(1,a['hist'],1)
    objective=torch.zeros(4,device=device)
    for t in range(a['horizon']):
        action=pi(history,goals.goal(S,None,step,command));S=wm.step(S,action,xn);step=step+1
        _,terms=mdp.reward(S,None,action,prev,True,command=command,ep_step=step)
        # No action penalties, base/posture terms, critic or survival in this audit.
        objective=objective+a['gamma']**t*terms['foot_objective'];prev=action
        obs=goals.obs(mdp,S,action,xi,None,command)
        history=torch.cat((history[:,1:],obs[:,None]),1)
    (-objective.mean()).backward()
    grads=[p.grad for p in pi.parameters() if p.grad is not None]
    norm=torch.stack([g.float().square().sum() for g in grads]).sum().sqrt()
    assert torch.isfinite(norm) and norm>0,'Pure foot reward did not reach Actor through learned WM'
    return dict(checkpoint=str(checkpoint_path),horizon=a['horizon'],pure_foot_actor_gradnorm=float(norm),
                mean_foot_objective=float(objective.detach().mean()),wm_frozen=True,optimizer_updates=0,
                direct_action_cost_included=False,critic_included=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint');parser.add_argument('--device',default='cpu')
    parser.add_argument('--out',default='results/s54/foot_tracking_preflight.json')
    args,rest=parser.parse_known_args();torch.set_num_threads(4)
    result=unittest.main(argv=[sys.argv[0]]+rest,verbosity=2,exit=False).result
    report=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
                failures=len(result.failures),errors=len(result.errors),motion_reference_required=False)
    if args.checkpoint and result.wasSuccessful():report['learned_wm_gradient']=learned_gradient_audit(args.checkpoint,args.device)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));sys.exit(not result.wasSuccessful())
