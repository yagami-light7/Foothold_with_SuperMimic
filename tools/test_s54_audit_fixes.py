"""Regression: explicit resets must split WM windows; subset contacts stay exact."""
from pathlib import Path
import sys,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from supermimic.buffer import ReplayBuffer
from supermimic.standing_diagnostics import ground_contact_subset,aggregate_ground_contacts

class AuditFixTests(unittest.TestCase):
    def buffer(self,ticks,dones=None):
        b=ReplayBuffer(len(ticks),1,'cpu');b.size=len(ticks);b.ptr=0
        b.data['ep_step'][:,0]=torch.tensor(ticks)
        if dones is not None:b.data['done'][:,0]=torch.tensor(dones)
        return b

    def test_evaluation_reset_without_done_cuts_multi_step_target(self):
        b=self.buffer([10,11,0,1])
        torch.testing.assert_close(b.sample_windows(1,4)['mask'],torch.tensor([[1.,1.,0.,0.]]))
        self.assertFalse(b.data['done'].any()) # no fake physical termination was inserted

    def test_regular_ticks_and_terminal_boundary(self):
        b=self.buffer([10,11,12,13]);self.assertTrue(b.sample_windows(1,4)['mask'].all())
        b=self.buffer([10,11,0,1],[False,True,False,False])
        torch.testing.assert_close(b.sample_windows(1,4)['mask'],torch.tensor([[1.,1.,0.,0.]]))
        self.assertTrue(b.sample_windows(1,1)['mask'].all())

    def test_ring_wrap_uses_chronological_ticks(self):
        b=self.buffer([0,0,0,0]);b.ptr=2
        b.data['ep_step'][:,0]=torch.tensor([0,1,10,11])
        torch.testing.assert_close(b.sample_windows(1,4)['mask'],torch.tensor([[1.,1.,0.,0.]]))

    def mc(self,b,starts,window=8,gamma=.5):
        with patch('torch.randint',side_effect=[torch.tensor(starts),torch.zeros(len(starts),dtype=torch.long)]):
            return b.sample_mc_returns(len(starts),window,gamma,recent=b.size)

    def test_real_return_includes_terminal_reward_and_stops_before_reset(self):
        b=self.buffer([0,1,2,0,1],[False,False,True,False,False])
        b.data['r'][:,0]=torch.tensor([1.,2.,3.,100.,100.])
        b.data['term'][2,0]=True
        out=self.mc(b,[0,1,2,3])
        torch.testing.assert_close(out['mc_return'],torch.tensor([2.75,3.5,3.]))
        self.assertEqual(out['ep_step'].tolist(),[0,1,2])

    def test_timeout_reset_gap_and_unfinished_tail_are_censored(self):
        b=self.buffer([0,1,0,1,0,1],[False,True,False,False,False,True])
        b.data['r'][:,0]=1.
        b.data['timeout'][1,0]=True;b.data['term'][5,0]=True
        out=self.mc(b,[0,1,2,3,4,5])
        # The timeout and the explicit reset without done cannot borrow a later termination.
        torch.testing.assert_close(out['mc_return'],torch.tensor([1.5,1.]))
        b.data['term'].zero_()
        self.assertEqual(len(self.mc(b,[4,5])['mc_return']),0)

    def test_real_return_crosses_ring_wrap_and_prioritizes_physical_end(self):
        b=self.buffer([2,0,0,1],[True,False,False,False]);b.ptr=2
        b.data['r'][:,0]=torch.tensor([3.,100.,1.,2.])
        b.data['term'][0,0]=True;b.data['timeout'][0,0]=True
        torch.testing.assert_close(self.mc(b,[0,1,2])['mc_return'],torch.tensor([2.75,3.5,3.]))

    def test_contact_subset_preserves_ground_attribution_and_force_sign(self):
        sb=torch.tensor([-1,0,1,2]);be=torch.tensor([0,1,1])
        s0=torch.tensor([0,2,0,1,-1,0]);s1=torch.tensor([1,0,3,2,1,0])
        forces=np.array([[0,0,-12],[0,0,15],[0,0,-2],[0,0,99],[0,0,99],[0,0,99]],dtype=float)
        labels=['/World/envs/env_0/Robot/leg_l6_link','/World/envs/env_1/Robot/leg_r6_link','/World/envs/env_1/Robot/base_link']
        full=aggregate_ground_contacts(sb.numpy(),labels,s0.numpy(),s1.numpy(),forces,2)
        for env in (0,1):
            mask=ground_contact_subset(sb,be,s0,s1,torch.tensor([env]))
            filtered=aggregate_ground_contacts(sb.numpy(),labels,s0[mask].numpy(),s1[mask].numpy(),forces[mask.numpy()],2)
            for before,after in zip(full,filtered):
                np.testing.assert_array_equal(before[env],after[env]);self.assertTrue((after[1-env]==0).all())
        self.assertFalse(ground_contact_subset(sb,be,s0,s1,torch.empty(0,dtype=torch.long)).any())

if __name__=='__main__':unittest.main()
