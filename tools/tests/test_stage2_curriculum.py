import copy,json,unittest
from pathlib import Path
import torch
from supermimic.stage2_curriculum import Stage2Curriculum,STAGES,evaluation_cases,build
from supermimic.curriculum import Command,feasibility

CFG=json.loads(Path('configs/s54/stage2/curriculum_3090_8h.json').read_text())
class Stage2Tests(unittest.TestCase):
    def test_distribution_feasibility_and_signs(self):
        sampler=Stage2Curriculum(CFG)
        for stage in range(5):
            sampler.stage=stage;sampler.progress=1.;sampler.apply_stage()
            commands=[sampler.sample() for _ in range(1000)]
            self.assertTrue(all(feasibility(c,sampler.cfg) is None for c in commands))
            self.assertTrue(any(c.task=='standing' for c in commands))
            if stage>=2:
                self.assertTrue(any(c.yaw_rate<0 for c in commands));self.assertTrue(any(c.yaw_rate>0 for c in commands))
        self.assertIsNotNone(feasibility(Command('long',.4,2.),sampler.cfg))
    def test_zero_frequency(self):
        c=Stage2Curriculum(CFG)
        self.assertIsNone(feasibility(Command(),c.cfg))
        self.assertIsNotNone(feasibility(Command('short',.1,0.),c.cfg))
    def test_resume_exact_rng_dynamic_stage(self):
        c=Stage2Curriculum(CFG);c.stage=4;c.progress=.7;c.apply_stage();c.sample()
        d=Stage2Curriculum(CFG);d.load_state_dict(copy.deepcopy(c.state_dict()))
        self.assertEqual([c.sample() for _ in range(100)],[d.sample() for _ in range(100)])
    def test_real_full_gate_only(self):
        c=Stage2Curriculum(CFG)
        rows=[dict(command=vars(cmd),success_rate=1.,survival_rate=1.) for _,cmd in evaluation_cases(0)]
        report=dict(complete=True,rows=rows)
        self.assertFalse(c.update_evaluation(report,1000,False));self.assertEqual(c.stage,0)
        self.assertTrue(c.update_evaluation(report,1100,True));self.assertEqual(c.stage,1)
        report['complete']=False;self.assertFalse(c.update_evaluation(report,2000,True))
    def test_immutable_plan_clocks_and_contract(self):
        c=Stage2Curriculum(CFG);p=torch.tensor([[[0.,.1,.1],[0.,-.1,.1]]]);q=torch.tensor([[[1.,0,0,0],[1.,0,0,0]]])
        cmd=c.plan(p,q,q[:,0],12.,Command('short',.04,.95));cmd.validate()
        saved=cmd.clone();cmd.target_pos.add_(1.);self.assertFalse(torch.equal(cmd.target_pos,saved.target_pos))
        self.assertEqual(len(STAGES),5)
        valid=saved.touchdown[0,:saved.lengths[0]]-saved.liftoff[0,:saved.lengths[0]]
        self.assertTrue(torch.allclose(valid,torch.full_like(valid,.72/.95),atol=1e-5))
    @unittest.skipUnless(Path(CFG['phase_a_checkpoint']).is_file(),'Local Phase A integration asset required')
    def test_transferred_goal_contract_and_gradient(self):
        from supermimic.stage15 import tensor_hash
        torch.set_num_threads(4);parts=build(CFG,'cpu');mdp=parts['mdp'];g=parts['goals'];pi=parts['policy']
        S=mdp.nominal_state(2);cmd=g.new_command(S);f=torch.zeros(2,dtype=torch.long)
        goal=g.goal(S,f,f,cmd);self.assertEqual(goal.shape,(2,230))
        self.assertTrue(torch.allclose(goal.view(2,10,23)[:,:,-1],torch.arange(1,11)[None]*.02,atol=1e-7))
        obs=g.obs(mdp,S,torch.zeros(2,29),torch.zeros(2,33),f,cmd)
        history=obs[:,None].repeat(1,10,1);before=tensor_hash(pi.dec.state_dict())
        action=pi(history,goal);action.square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in pi.enc.parameters()))
        self.assertTrue(all(p.grad is None for p in pi.dec.parameters()));self.assertEqual(before,tensor_hash(pi.dec.state_dict()))
        self.assertIsNone(g.motion);self.assertIsNone(mdp.motion)
if __name__=='__main__':unittest.main()
