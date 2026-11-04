"""Verify finite parameter updates, full recovery state and saved command curriculum."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from supermimic.curriculum import Curriculum
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--before',required=True)
p.add_argument('--out',required=True);a=p.parse_args()
run=Path(a.run);before=torch.load(a.before,map_location='cpu',weights_only=False)
after=torch.load(run/'latest.pt',map_location='cpu',weights_only=False)
required=['wm','pi','critic','opt_wm','opt_pi','opt_v','rng','curriculum','step_acc','goal_contract','task_contract']
assert all(k in after for k in required);assert int((run/'train.exitcode').read_text())==0
updates={}
for name in ('wm','pi'):
    keys=[k for k in after[name] if k.endswith(('.weight','.bias'))]
    assert all(torch.isfinite(after[name][k]).all() for k in keys)
    delta=sum(float((after[name][k]-before[name][k]).square().sum()) for k in keys)**.5
    updates[name]=delta;assert delta>0, f'{name} parameters did not change'
assert after['iteration']>0
c=Curriculum(after['curriculum']['cfg']);c.load_state_dict(after['curriculum'])
other=Curriculum(after['curriculum']['cfg']);other.load_state_dict(after['curriculum'])
assert [c.sample() for _ in range(10)]==[other.sample() for _ in range(10)]
result=dict(status='PASS',iteration=after['iteration'],parameter_l2_change=updates,
            optimizer_steps={k:max(float(v['step']) for v in after[k]['state'].values()) for k in ('opt_wm','opt_pi','opt_v')},
            curriculum_stage=after['curriculum']['stage'],wall_seconds=after['wall_time'],
            normalization='included in model state_dict',full_state_keys=required,
            physical_mastery_claimed=False,notes='Stage transition fixture is separately unit tested; real eval does not pass C0.')
Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
