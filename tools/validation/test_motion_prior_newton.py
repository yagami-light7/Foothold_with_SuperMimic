"""One-environment Standing curriculum-gate integration; no optimizer or training."""

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
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from isaaclab.app import add_launcher_args,launch_simulation

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
add_launcher_args(p);a=p.parse_args()


def main():
    import torch
    from supermimic import networks
    from supermimic.motion_prior_runtime import reconstruct
    from supermimic.networks import WorldModel
    from supermimic.curriculum import Curriculum
    from supermimic.curriculum_runtime import evaluate
    from supermimic.env import TrackingSim,make_sim
    out=Path(a.out)
    if out.exists():p.error('Existing result protected')
    ck=torch.load(a.checkpoint,map_location='cuda:0',weights_only=False)
    networks.AMP_DTYPE=torch.bfloat16 if ck['args'].get('amp')=='bf16' else None
    kin,motion,goals,mdp,pi=reconstruct(ck,'cuda:0')
    if not mdp.is_foot or mdp.cfg.command_mode!='standing':p.error('Use an independent Standing checkpoint')
    wm=WorldModel().cuda();wm.load_state_dict(ck['wm']);wm.eval()
    cur=Curriculum(json.loads(Path('configs/motion_prior/stage2/curriculum_bounded.json').read_text()))
    versions=[v._version for m in (pi,wm) for v in m.parameters()]
    sim=make_sim('cuda:0',a)
    with launch_simulation(sim,a):
        env=TrackingSim(sim,1,motion,mdp,goals=goals,eval_mode=False)
        old=(env.eval_mode,env.max_episode_steps,env.reset_noise,pi.training)
        report=evaluate(env,pi,goals,mdp,cur,10,deadline=time.monotonic()+30,wm=wm)
        assert old==(env.eval_mode,env.max_episode_steps,env.reset_noise,pi.training)
        assert versions==[v._version for m in (pi,wm) for v in m.parameters()]
        assert all(v.grad is None for m in (pi,wm) for v in m.parameters())
        assert report['wm_metrics'] is not None,'Missing matched WM diagnosis'
        assert all(k in report['metrics']['standing'] for k in cur.cfg['posture_gate'])
        changed=cur.update(report['metrics'],100,wm_metrics=report['wm_metrics'])
        report.update(integration_status='PASS',curriculum_stage=cur.stage,promoted=changed,
            parameters_unchanged=True,environment_protocol_restored=True,
            standing_success=report['metrics']['standing']['success_rate'])
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
    print(out)


if __name__=='__main__':main()
