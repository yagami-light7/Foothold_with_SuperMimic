"""Regression for real/imagined action consistency and valid foot bootstrap bounds."""

# Support both direct execution and python -m from the repository layout.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_root = str(_ToolPath(__file__).resolve().parents[2])
if _tool_root not in _tool_sys.path:
    _tool_sys.path.insert(0, _tool_root)

import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from supermimic.control import ACTION_LIMIT,bound_action,bound_bootstrap,return_upper_bound

class ControlTests(unittest.TestCase):
    def test_actual_imagination_step_uses_real_action_bound(self):
        path=Path(__file__).resolve().parents[2]/'supermimic/train.py'
        tree=ast.parse(path.read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_imag_step')
        stepper=Mock(return_value=('state',))
        ns=dict(pi=lambda *a:torch.tensor([[50.,-50.]]),bound_action=bound_action,ACTION_LIMIT=ACTION_LIMIT,
                wm=SimpleNamespace(step=stepper),mdp=SimpleNamespace(add_noise=lambda x:x),torch=torch,
                goals=SimpleNamespace(obs=lambda *a:torch.zeros(1,2),goal=lambda *a:torch.zeros(1,3)))
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
        out=ns['_imag_step'](('state',),torch.zeros(1,2,2),None,torch.tensor([0]),None,None,None,torch.tensor([0]),None)
        torch.testing.assert_close(out[1],torch.tensor([[20.,-20.]]))
        torch.testing.assert_close(stepper.call_args.args[1],out[1])
        self.assertEqual(float(out[-1]),1.)

    def test_clamp_preserves_valid_actions_and_gradients(self):
        x=torch.tensor([-100.,-2.,3.,100.],requires_grad=True)
        y=bound_action(x);y.sum().backward()
        torch.testing.assert_close(y,torch.tensor([-20.,-2.,3.,20.]))
        torch.testing.assert_close(x.grad,torch.tensor([0.,1.,1.,0.]))

    def test_overestimated_bootstrap_is_bounded_without_changing_reward(self):
        upper=return_upper_bound(.08,.99)
        self.assertAlmostEqual(upper,8.)
        values=torch.tensor([4.,120.],requires_grad=True)
        bounded=bound_bootstrap(values,upper)
        torch.testing.assert_close(bounded,torch.tensor([4.,8.]))
        bounded.sum().backward();torch.testing.assert_close(values.grad,torch.tensor([1.,0.]))
        self.assertIs(bound_bootstrap(values,None),values)

    def test_negative_reward_cannot_be_hidden_by_arbitrarily_large_bootstrap(self):
        # Probe reproduces the observed failure: negative immediate return + huge positive V.
        r=torch.full((8,),-1.);gamma=.99
        value=bound_bootstrap(torch.tensor(1e6),return_upper_bound(.08,gamma))
        objective=(gamma**torch.arange(8)*r).sum()+gamma**8*value
        self.assertLess(float(objective),0.)
        self.assertGreaterEqual(float(-objective),-8.)

if __name__=='__main__':unittest.main()
