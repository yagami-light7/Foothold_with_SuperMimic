"""Newton foot-MDP preflight: no Motion CSV, no optimization, no walking claim."""

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
from isaaclab.app import add_launcher_args,launch_simulation
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--num_envs',type=int,default=4);p.add_argument('--steps',type=int,default=120)
p.add_argument('--command_mode',choices=['standing','alternating'],default='alternating')
p.add_argument('--out',default='results/s54/foot_tracking_newton_preflight.json')
add_launcher_args(p);args=p.parse_args()
if not 1<=args.num_envs<=4 or not 100<=args.steps<=200:p.error('Use 1–4 envs and 100–200 steps')
import torch
from supermimic.buffer import ReplayBuffer
from supermimic.env import TrackingSim,make_sim
from supermimic.foot_mdp import FootTrackingMDP,FootTrackingCfg
from supermimic.foot_trajectory import command_time,query
from supermimic.goals import GoalSource,GoalSpec
from supermimic.kinematics import S54Kinematics
from supermimic.mdp import unflatten_state
from supermimic.s54 import default_joint_pos_list,NUM_J,FOOT_BODY_NAMES


def main():
    torch.manual_seed(54);torch.set_num_threads(4);dev='cuda:0'
    kin=S54Kinematics(dev);cfg=FootTrackingCfg(command_mode=args.command_mode)
    goals=GoalSource(None,kin,GoalSpec('foot_trajectory'),command_mode=cfg.command_mode)
    mdp=FootTrackingMDP(kin,torch.tensor(default_joint_pos_list(),device=dev),goals,cfg)
    sim=make_sim(dev,args);max_goal=max_reward=max_fk=0.;events=resets=0;contacts=set()
    with launch_simulation(sim,args):
        env=TrackingSim(sim,args.num_envs,None,mdp,goals=goals,reset_noise=False)
        replay=ReplayBuffer(48,args.num_envs,dev,foot_commands=True)
        action=torch.zeros(args.num_envs,NUM_J,device=dev)
        with torch.no_grad():
            for i in range(args.steps):
                S=env.state();step=env.ep_step.clone();c=env.command.clone()
                goal=goals.goal(S,None,step,c)
                ref=query(c,command_time(step,.02,torch.float32)[:,None])
                contacts.update(tuple(row) for row in ref.contact[:,0].cpu().tolist())
                out=env.step(action);replay.add(out)
                restored=replay.commands(out['command_id'])
                S0,S1=unflatten_state(out['s']),unflatten_state(out['s1'])
                regenerated=goals.goal(S0,torch.full_like(out['f'],10**12),out['ep_step'],restored)
                reward=mdp.reward(S1,None,out['a'],out['a_prev'],command=restored,ep_step=out['ep_step']+1)
                max_goal=max(max_goal,float((goal-regenerated).abs().max()))
                max_reward=max(max_reward,float((reward-out['r']).abs().max()))
                # Time-major layout used by BPTT must preserve each sampled plan.
                future_steps=(step[None]+torch.arange(1,9,device=dev)[:,None]).flatten()
                imagined=query(restored.repeat(8),command_time(future_steps,.02,torch.float32)[:,None])
                expected=torch.cat([query(c,command_time(step+t,.02,torch.float32)[:,None]).position for t in range(1,9)])
                torch.testing.assert_close(imagined.position,expected,atol=1e-7,rtol=1e-7)
                assert torch.isfinite(reward).all()
                resets+=int(out['done'].sum());events+=int(out['terms']['touchdown_events'].sum())
                if bool(out['done'].any()):assert (env.ep_step[out['done']]==0).all()
                pose=env.robot.data.body_link_pose_w.torch[:,[env.robot.body_names.index(n) for n in FOOT_BODY_NAMES],:3]
                actual=pose-env.scene.env_origins[:,None]
                fk=kin.forward(*env.state())[0][:,mdp.foot_indices]
                max_fk=max(max_fk,float((fk-actual).norm(dim=-1).max()))
                if (i+1)%40==0:print(f'Foot MDP Newton {i+1}/{args.steps}: resets={resets}, touchdown_events={events}',flush=True)
        assert max_goal<1e-5 and max_reward<1e-5 and max_fk<1e-4
        assert resets>0
        if args.command_mode=='alternating':assert events>0 and len(contacts)==3
    report=dict(status='PASS',num_envs=args.num_envs,steps=args.steps,command_mode=args.command_mode,
                motion_reference_required=False,optimizer_updates=0,max_goal_sync_error=max_goal,
                max_reward_sync_error=max_reward,max_foot_fk_alignment_m=max_fk,
                reset_transitions=resets,touchdown_events=events,observed_contacts=sorted(contacts))
    target=Path(args.out);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
