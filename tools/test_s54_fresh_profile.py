"""Fresh start profile, resume arguments, and exact 1000-iteration WM gate."""
import ast
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.s54_training_profile import profile_args

ROOT=Path(__file__).resolve().parents[1]

class FreshProfileTests(unittest.TestCase):
    def test_production_profile_is_fresh_and_wm1000(self):
        args=profile_args(ROOT/'configs/s54_curriculum_fresh_3090.json')
        self.assertIn('--fresh_start',args)
        self.assertEqual(args[args.index('--wm_only_iters')+1],'1000')
        self.assertNotIn('--resume',args);self.assertNotIn('--init_weights',args)
        self.assertEqual(args[args.index('--horizon')+1],'8')

    def test_resume_keeps_actual_overrides_without_fresh_flag(self):
        with tempfile.TemporaryDirectory() as d:
            saved=Path(d)/'args.json';saved.write_text(json.dumps({'num_envs':1024,'wm_batch':4096,'wm_only_iters':1000}))
            args=profile_args(ROOT/'configs/s54_curriculum_fresh_3090.json',saved,True)
            self.assertNotIn('--fresh_start',args)
            self.assertEqual(args[args.index('--num_envs')+1],'1024')
            self.assertEqual(args[args.index('--wm_batch')+1],'4096')

    def test_profile_cannot_load_weights(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'profile.json'
            for key in ('resume','init_weights','log_dir'):
                p.write_text(json.dumps({'train':{'fresh_start':True,'wm_only_iters':1000,key:'old.pt'}}))
                with self.assertRaises(ValueError):profile_args(p)

    def test_actual_trainer_rejects_checkpoint_with_fresh_start(self):
        tree=ast.parse((ROOT/'supermimic/train.py').read_text())
        guard=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
                   and 'args.fresh_start' in ast.unparse(n.test))
        code=compile(ast.Module(body=[guard],type_ignores=[]),'fresh guard','exec')
        for resume,init in [('old.pt',None),(None,'old.pt')]:
            with self.assertRaises(ValueError):exec(code,{'args':SimpleNamespace(fresh_start=True,resume=resume,init_weights=init)})
        exec(code,{'args':SimpleNamespace(fresh_start=True,resume=None,init_weights=None)})

    def test_actual_actor_and_critic_gate(self):
        tree=ast.parse((ROOT/'supermimic/train.py').read_text())
        guard=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
                   and ast.unparse(n.test)=='it >= max(args.wm_only_iters, recovery_until)')
        expression=compile(ast.Expression(body=guard.test),'WM gate','eval')
        ns={'args':SimpleNamespace(wm_only_iters=1000),'recovery_until':0}
        for it in (0,199,999):self.assertFalse(eval(expression,ns|{'it':it}))
        self.assertTrue(eval(expression,ns|{'it':1000})) # logged iteration1001

if __name__=='__main__':unittest.main()
