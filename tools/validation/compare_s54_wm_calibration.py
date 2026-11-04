"""Compare two WMs on identical recorded Newton windows; never starts a simulator."""

# Support both direct execution and python -m from the repository layout.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_root = str(_ToolPath(__file__).resolve().parents[2])
if _tool_root not in _tool_sys.path:
    _tool_sys.path.insert(0, _tool_root)

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from supermimic import networks
from supermimic.motion_prior_runtime import reconstruct
from supermimic.networks import WorldModel
from supermimic.foot_trajectory import FootstepCommand
from supermimic.standing_diagnostics import paired_predictions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',required=True);p.add_argument('--after',required=True)
    p.add_argument('--evaluation',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();out=Path(a.out)
    if out.exists():p.error('Output protected; select a new result path')
    checkpoints=[torch.load(path,map_location='cuda:0',weights_only=False) for path in (a.before,a.after)]
    before,after=checkpoints
    networks.AMP_DTYPE=torch.bfloat16 if before['args']['amp']=='bf16' else None
    kin,motion,goals,mdp,pi=reconstruct(before,'cuda:0')
    frozen={key:all(torch.equal(before[key][k],after[key][k]) for k in before[key]) for key in ('pi','critic')}
    if not all(frozen.values()) or after['opt_pi']['state'] or after['opt_v']['state']:
        raise RuntimeError('WM-only calibration changed Actor/normalization/Critic or created their optimizer state')
    report=json.loads((Path(a.evaluation)/'summary.json').read_text());rows=[]
    limits=json.loads(Path('configs/motion_prior/stage2/curriculum_bounded.json').read_text())['wm_gate']
    models=[]
    for ck in checkpoints:
        wm=WorldModel().cuda();wm.load_state_dict(ck['wm']);wm.eval();models.append(wm)
    # Longest duration per seed avoids counting identical short prefixes repeatedly.
    longest=max(r['duration_s'] for r in report['rows'])
    for record in report['rows']:
        if record['duration_s']!=longest:continue
        path=Path(record['record'])
        with np.load(path,allow_pickle=False) as z:trace=dict(z)
        command=FootstepCommand(**{k.removeprefix('command_'):torch.as_tensor(v,device='cuda:0')
            for k,v in trace.items() if k.startswith('command_') and k!='command_step'})
        results=[paired_predictions(model,mdp,command,trace,'cuda:0',stride=1)[0] for model in models]
        rows.append(dict(seed=record['seed'],trace=str(path),trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            before=results[0],after=results[1]))
    worst={side:{k:max(row[side]['horizons']['8'][k] for row in rows) for k in limits} for side in ('before','after')}
    passed={side:all(worst[side][k]<v for k,v in limits.items()) for side in worst}
    result=dict(protocol='identical Newton records/actions/xi/commands/ticks, no new simulation',frozen=frozen,
        actor_critic_optimizer_empty=True,wm_changed=any(not torch.equal(before['wm'][k],after['wm'][k]) for k in before['wm']),
        limits=limits,worst_seed_horizon8=worst,passed=passed,rows=rows,
        standing_success_unchanged='Actor AND all normalization identical; WM calibration is not policy training')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
