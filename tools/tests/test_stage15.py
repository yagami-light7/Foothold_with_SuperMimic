import unittest
import torch
from supermimic.stage15 import split_frames,fit_goal_norm,goal_floor,tensor_hash
from supermimic.networks import Policy,RunningNorm
from supermimic.joint_models import frozen

class Stage15Tests(unittest.TestCase):
    def test_fit_excludes_evaluation_windows(self):
        tr,va=split_frames(13066);self.assertFalse(bool(torch.isin(tr,va).any()))
        for start in (0,3000,6000):
            self.assertFalse(bool(((tr>=max(0,start-10))&(tr<=start+510)).any()))
        self.assertGreater(len(tr),8000)
    def test_reference_goal_time_alignment_and_no_joint_leak(self):
        from supermimic.reference_foot import ReferenceFootGoalSource,ReferenceFootCommand
        from supermimic.goals import GoalSpec
        from supermimic.foot_trajectory import FootTrajectory
        from types import SimpleNamespace
        from supermimic.rotations import mat_to_quat
        source=ReferenceFootGoalSource.__new__(ReferenceFootGoalSource);source.spec=GoalSpec('reference_foot',10,.02)
        captured=[]
        def query(command,times):
            captured.append(times)
            return FootTrajectory(torch.zeros(2,10,2,3),torch.tensor([1.,0.,0.,0.]).expand(2,10,2,4),
                torch.zeros(2,10,2,3),torch.zeros(2,10,2,3),torch.ones(2,10,2,dtype=torch.bool),torch.zeros(2,10,2))
        source.query=query
        S=(torch.zeros(2,3),torch.eye(3).expand(2,3,3),torch.zeros(2,3),torch.zeros(2,3),torch.zeros(2,29),torch.zeros(2,29))
        cmd=ReferenceFootCommand(torch.tensor([50,100]),torch.zeros(2,dtype=torch.long),mat_to_quat(S[1]),torch.zeros(2,2,3))
        g=source.goal(S,torch.tensor([57,111]),torch.tensor([7,11]),cmd)
        expected=(torch.tensor([7,11])[:,None]+torch.arange(1,11)[None])*.02
        self.assertTrue(torch.allclose(captured[0],expected.double()))
        self.assertEqual(g.shape,(2,230))
        changed=(*S[:4],torch.randn_like(S[4])*100,torch.randn_like(S[5])*100)
        self.assertTrue(torch.equal(g,source.goal(changed,torch.tensor([57,111]),torch.tensor([7,11]),cmd)))
    def test_causal_heading_has_no_reference_root_dependency(self):
        from supermimic.stage15 import CausalHeadingGoals
        from supermimic.reference_foot import ReferenceFootCommand
        from supermimic.rotations import so3_exp
        from types import SimpleNamespace
        source=CausalHeadingGoals.__new__(CausalHeadingGoals)
        source.heading_bias=0.;source.reference_tilt=(0.,0.)
        g=torch.zeros(2,230).view(2,10,23)
        for idx in (3,14):g[:,:,idx]=1.;g[:,:,idx+3]=1.
        g[:,:,9]=1.;g[:,:,20]=1.
        source.goal=lambda *args:g.flatten(1)
        class Forbidden:
            def __getattr__(self,key):raise AssertionError('Motion reference leaked: '+key)
        source.motion=Forbidden()
        mdp=SimpleNamespace(obs=lambda S,a,xi,f,reference_R:S[1].transpose(-1,-2)@reference_R)
        R=so3_exp(torch.tensor([[0.,0.,0.],[0.,0.,1.2]]))
        S=(torch.zeros(2,3),R,torch.zeros(2,3),torch.zeros(2,3),torch.zeros(2,29),torch.zeros(2,29))
        cmd=ReferenceFootCommand(torch.tensor([0,0]),torch.zeros(2,dtype=torch.long),torch.zeros(2,4),torch.zeros(2,2,3))
        result=source.obs(mdp,S,None,None,torch.tensor([10,10]),cmd)
        self.assertTrue(torch.allclose(result,torch.eye(3).expand(2,3,3),atol=1e-6))
    def test_norm_floor_constant_rotations(self):
        norm=RunningNorm(230);x=torch.zeros(128,230);fit_goal_norm(norm,x)
        self.assertTrue(torch.allclose(norm.std(),goal_floor(),atol=1e-6))
        y=x.clone();y[:,3]=1.;self.assertLess(float(norm(y).abs().max()),7)
    def test_frozen_decoder_preserves_input_gradient_and_mask(self):
        p=Policy(99,230,hist=10,obs_adapter='mask_reference_orientation');o=torch.randn(2,10,99)
        z=torch.randn(2,32,requires_grad=True);before=tensor_hash(p.dec.state_dict())
        with frozen(p.dec):
            a=p.decode(o,z);a.sum().backward()
            self.assertGreater(float(z.grad.norm()),0);self.assertTrue(all(v.grad is None for v in p.dec.parameters()))
        q=o.clone();q[:,:,93:]=torch.randn_like(q[:,:,93:])*1000
        self.assertTrue(torch.equal(p.decode(o,z),p.decode(q,z)))
        self.assertEqual(before,tensor_hash(p.dec.state_dict()))
if __name__=='__main__':unittest.main()
