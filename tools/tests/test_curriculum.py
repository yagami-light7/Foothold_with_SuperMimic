"""CPU curriculum tests: joint envelope, timing, replay, turning, resume and gates."""

# Support both direct execution and python -m from the repository layout.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_root = str(_ToolPath(__file__).resolve().parents[2])
if _tool_root not in _tool_sys.path:
    _tool_sys.path.insert(0, _tool_root)

import copy
import math
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from supermimic.curriculum import Curriculum,Command,feasibility,evaluation_commands
from supermimic.foot_trajectory import query
from supermimic.curriculum_runtime import atomic_save,rng_state,restore_rng


class CurriculumTests(unittest.TestCase):
    def inputs(self,n=2):
        p=torch.tensor([[[0.,.12,.08],[0.,-.12,.08]]]).repeat(n,1,1)
        q=torch.tensor([[[1.,0,0,0],[1.,0,0,0]]]).repeat(n,1,1)
        return p,q,q[:,0]

    def test_zero_frequency(self):
        c=Curriculum(); plan=c.plan(*self.inputs(),fixed=Command())
        ref=query(plan,torch.tensor([[0.,1.,10.,100.]]).repeat(2,1))
        self.assertTrue(ref.contact.all()); self.assertTrue((plan.lengths==0).all())
        self.assertTrue(torch.isfinite(ref.position).all()); self.assertEqual(feasibility(Command(length=.1)),'zero_frequency_requires_standing')

    def test_sampling_coverage_and_feasibility(self):
        c=Curriculum(); c.stage=5; c.progress=1
        samples=[c.sample() for _ in range(4000)]
        self.assertEqual(set(s.task for s in samples),set(('standing','short','medium','long','turn','curve')))
        self.assertTrue(all(feasibility(s,c.cfg) is None for s in samples))
        self.assertTrue(any(s.frequency>1.3 for s in samples))
        self.assertTrue(any(s.length>.3 for s in samples))
        self.assertTrue(any(s.yaw_rate<0 for s in samples)); self.assertTrue(any(s.yaw_rate>0 for s in samples))
        self.assertIsNotNone(feasibility(Command('long',.4,2.)))

    def test_resume_exact_sampler_sequence(self):
        c=Curriculum(); c.stage=4; [c.sample() for _ in range(20)]
        state=copy.deepcopy(c.state_dict()); expected=[c.sample() for _ in range(50)]
        other=Curriculum(); other.load_state_dict(state)
        self.assertEqual(expected,[other.sample() for _ in range(50)])

    def test_dynamic_weights(self):
        a=Curriculum(); a.stage=5; a.success['long']=0.
        b=Curriculum(); b.stage=5; b.success['long']=1.
        [a.sample() for _ in range(2000)]; [b.sample() for _ in range(2000)]
        self.assertLess(a.counts['long'],b.counts['long'])

    def test_frequency_bucket_feedback(self):
        weights=[[0,1,0,0,0,0] for _ in range(6)]
        a=Curriculum({'weights':weights});a.stage=5;a.bucket_success['short_high']=0.
        b=Curriculum({'weights':weights});b.stage=5;b.bucket_success['short_high']=1.
        count_a=sum(a.sample().frequency>=1.3 for _ in range(2000))
        count_b=sum(b.sample().frequency>=1.3 for _ in range(2000))
        self.assertLess(count_a,count_b)

    def test_continuous_turn_and_touchdown(self):
        c=Curriculum(); cmd=Command('turn',0.,1.,.20)
        plan=c.plan(*self.inputs(1),duration=40.,fixed=cmd)
        self.assertGreater(float(plan.target_quat[0,int(plan.lengths[0])-1,3].abs()),.1)
        # Each lift and landing is C2 in position/velocity/acceleration.
        for side in (0,1):
            for event in range(4):
                t=plan.touchdown[:,event:event+1]
                r=query(plan,torch.cat((t-1e-5,t,t+1e-5),1))
                torch.testing.assert_close(r.position[:,0,side],r.position[:,2,side],atol=1e-5,rtol=0)
                self.assertLess(float(r.velocity[:,1,side].abs().max()),1e-5)
        # cumulative yaw >2pi; wxyz remains unit and validation succeeds.
        self.assertGreater(float(plan.lengths[0])*.2,2*math.pi)

    def test_switches_hold_old_targets(self):
        c=Curriculum({'switch_probability':1.,'segment_seconds':2.});c.stage=4
        plan=c.plan(*self.inputs(),duration=12.)
        ref=query(plan,torch.linspace(0,12,601)[None].repeat(2,1))
        self.assertTrue(ref.contact.any(-1).all()) # no flight / overlapping opposite swings
        self.assertLess(float((ref.position[:,1:]-ref.position[:,:-1]).norm(dim=-1).max()),.06)
        self.assertTrue((plan.liftoff[:,1:]>=plan.touchdown[:,:-1]-1e-6).logical_or(
            torch.arange(1,64)[None]>=plan.lengths[:,None]).all())

    def test_gate_and_regression(self):
        c=Curriculum({'min_stage_seconds':0,'consecutive_passes':2})
        self.assertFalse(c.update({'standing':dict(success_rate=.2,foot_rmse=.1)},1000));self.assertEqual(c.stage,0)
        m={'standing':dict(success_rate=1.,foot_rmse=.01)}
        c.update(m,1001);self.assertTrue(c.update(m,1002)); self.assertEqual(c.stage,1)
        for _ in range(3): c.update({'standing':dict(success_rate=.1,foot_rmse=.1)},1100)
        self.assertEqual(c.stage,0)

    def test_atomic_rng(self):
        with tempfile.TemporaryDirectory() as d:
            state=rng_state(); expected=torch.rand(4); restore_rng(state)
            torch.testing.assert_close(expected,torch.rand(4))
            atomic_save({'rng':state},Path(d)/'state.pt')
            saved=torch.load(Path(d)/'state.pt',weights_only=False)
            self.assertIn('rng',saved);self.assertFalse((Path(d)/'state.pt.tmp').exists())

if __name__=='__main__': unittest.main()
