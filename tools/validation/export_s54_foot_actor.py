"""Export only obs-history + Foot Goal -> bounded action, with no WM/Teacher/data loader."""

# Support both direct execution and python -m from the repository layout.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_root = str(_ToolPath(__file__).resolve().parents[2])
if _tool_root not in _tool_sys.path:
    _tool_sys.path.insert(0, _tool_root)

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from supermimic import networks
from supermimic.networks import Policy
from supermimic.s54 import assert_checkpoint,checkpoint_contract
from supermimic.goals import GoalSpec
from supermimic.control import bound_action


class ActorOnly(torch.nn.Module):
    def __init__(self,policy):super().__init__();self.policy=policy
    def forward(self,obs_history,foot_goal):return bound_action(self.policy(obs_history,foot_goal))


def export_actor(checkpoint,target):
    assert_checkpoint(checkpoint);a=checkpoint['args']
    if a.get('goal_mode')!='foot_trajectory':raise ValueError('Deploy only direct Quintic Foot actors, not reference-conditioned bridge interfaces')
    spec=GoalSpec(a['goal_mode'],a['goal_frames'],a['goal_dt']);spec.assert_checkpoint(checkpoint)
    networks.AMP_DTYPE=None
    pi=Policy(99,spec.dim,hist=a['hist'],obs_adapter=a.get('obs_adapter','identity'))
    pi.load_state_dict(checkpoint['pi'],strict=True);pi.eval();wrapper=ActorOnly(pi).eval()
    sample=(torch.zeros(1,a['hist'],99),torch.zeros(1,spec.dim))
    with torch.no_grad():
        scripted=torch.jit.trace(wrapper,sample,check_trace=True)
        varied=(torch.randn(3,a['hist'],99),torch.randn(3,spec.dim))
        torch.testing.assert_close(scripted(*varied),wrapper(*varied),rtol=1e-5,atol=1e-6)
    target=Path(target)
    if target.exists():raise ValueError('Export target exists')
    target.parent.mkdir(parents=True,exist_ok=True);scripted.save(str(target))
    report=dict(robot_contract=checkpoint_contract(),goal_contract=spec.contract(),
        input_shapes=['B x '+str(a['hist'])+' x 99','B x '+str(spec.dim)],output_shape='B x 29',
        action='canonical S54 PD target offset; clipped to [-20,20]',obs_adapter=pi.obs_adapter,
        teacher_required=False,world_model_required=False,motion_reference_required=False,prior_bank_required=False,
        normalization='Frozen Actor observation/goal statistics embedded',dtype='float32',variable_batch_test='PASS')
    target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();torch.set_num_threads(4)
    print(json.dumps(export_actor(torch.load(a.checkpoint,map_location='cpu',weights_only=False),a.out),indent=2))


if __name__=='__main__':main()
