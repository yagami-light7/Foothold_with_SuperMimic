"""Evaluation bookkeeping fixtures (not physical success measurements)."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from supermimic.curriculum import Curriculum
from supermimic.curriculum_runtime import evaluate

class EvaluationTests(unittest.TestCase):
    def run_case(self,fail=False,offset=0.,deadline=None,whole=False):
        n=2
        class Env:
            N=n; device='cpu';eval_mode=False;max_episode_steps=500;reset_noise=False
            def __init__(self):
                self.done_eval=torch.zeros(n,dtype=torch.bool);self.failed_eval=self.done_eval.clone()
                self.ep_step=torch.zeros(n,dtype=torch.long);self.a_prev=torch.zeros(n,29);self.xi=torch.zeros(n,1);self.f=self.ep_step.clone()
                self.command=SimpleNamespace(foot=torch.zeros(n,1,dtype=torch.long),lengths=torch.zeros(n,dtype=torch.long),
                    touchdown=torch.ones(n,1)*999.,target_pos=torch.zeros(n,1,3))
                self.command.index=lambda ids:self.command
            def reset(self,ids):self.ep_step.zero_()
            def state(self):
                return (torch.zeros(n,3),torch.eye(3).repeat(n,1,1),torch.zeros(n,3),torch.zeros(n,3),torch.zeros(n,29),torch.zeros(n,29))
            def step(self,a):
                self.ep_step+=1
                self.failed_eval |= fail & (self.ep_step>=30)
                self.done_eval |= self.failed_eval | (self.ep_step>=self.max_episode_steps)
                return {}
        env=Env();c=Curriculum()
        goals=SimpleNamespace(fixed_command=None,obs=lambda *a:torch.zeros(n,99),goal=lambda *a:torch.zeros(n,230))
        kin=SimpleNamespace(forward=lambda *s:(torch.zeros(n,2,3)+offset,None,torch.zeros(n,2,3),None))
        mdp=SimpleNamespace(kin=kin,foot_indices=[0,1],_reference=lambda *a:SimpleNamespace(position=torch.zeros(n,1,2,3)))
        class Pi(torch.nn.Module):
            def forward(self,*a):return torch.zeros(n,29)
        import itertools
        from contextlib import nullcontext
        timer=patch('supermimic.curriculum_runtime.time.monotonic',side_effect=itertools.count()) if deadline is not None else nullcontext()
        with patch('supermimic.curriculum_runtime.NewtonGroundContacts',side_effect=RuntimeError('fixture')),timer:
            report=evaluate(env,Pi(),goals,mdp,c,2,deadline=float('inf') if deadline is None else deadline)
        return report if whole else report['rows'][0]

    def test_timeout_and_tracking_are_distinct(self):
        r=self.run_case();self.assertEqual(r['success_rate'],1.);self.assertEqual(r['physical_termination_rate'],0.)
        self.assertEqual(r['timeout_rate'],1.);self.assertEqual(r['duration'],5.)
        r=self.run_case(offset=.1);self.assertEqual(r['success_rate'],0.);self.assertEqual(r['goal_tracking_failure_rate'],1.)
        self.assertEqual(r['physical_termination_rate'],0.)

    def test_short_survival_is_not_success(self):
        r=self.run_case(fail=True);self.assertEqual(r['success_rate'],0.);self.assertEqual(r['timeout_rate'],0.)
        self.assertEqual(r['physical_termination_rate'],1.)

    def test_incomplete_evaluation_is_unknown_not_failed(self):
        report=self.run_case(deadline=10,whole=True);row=report['rows'][0]
        self.assertEqual(row['status'],'incomplete');self.assertIsNone(row['success_rate'])
        self.assertEqual(row['physical_termination_rate'],0.)
        self.assertEqual(row['timeout_rate'],0.);self.assertEqual(row['incomplete_rate'],1.)
        self.assertEqual(report['metrics'],{});self.assertEqual(report['bucket_metrics'],{})
        self.assertTrue(all(0<t<250 for t in row['episode_lengths']))
        self.assertGreater(report['minimum_next_eval_seconds'],10.)

if __name__=='__main__':unittest.main()
