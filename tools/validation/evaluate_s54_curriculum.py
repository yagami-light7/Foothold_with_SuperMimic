"""Standalone bounded Newton evaluation; run only when the training GPU is idle."""

# Support both direct execution and python -m from the repository layout.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_root = str(_ToolPath(__file__).resolve().parents[2])
if _tool_root not in _tool_sys.path:
    _tool_sys.path.insert(0, _tool_root)

import argparse
from pathlib import Path
import sys
import json
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from isaaclab.app import add_launcher_args,launch_simulation
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',required=True);p.add_argument('--num_envs',type=int,default=8)
p.add_argument('--stage',type=int,choices=range(6),help='Independent evaluation coverage override; does not change training state')
p.add_argument('--full',action='store_true');p.add_argument('--out',required=True)
add_launcher_args(p);args=p.parse_args()
import torch
from supermimic.curriculum import Curriculum
from supermimic.curriculum_runtime import evaluate
from supermimic.env import TrackingSim,make_sim
from supermimic.foot_mdp import FootTrackingMDP,resolve_foot_task,assert_task_checkpoint
from supermimic.goals import GoalSource,GoalSpec
from supermimic.foot_trajectory import AlternatingConfig
from supermimic.kinematics import S54Kinematics
from supermimic.networks import Policy,Critic
from supermimic.mdp import OBS_DIM
from supermimic.s54 import default_joint_pos_list,assert_checkpoint
from supermimic import networks

ck=torch.load(args.checkpoint,map_location='cuda:0',weights_only=False);cfg=ck['args'];dev='cuda:0'
assert_checkpoint(ck)
kin=S54Kinematics(dev); task=resolve_foot_task(cfg['foot_task_config'])
goals=GoalSource(None,kin,GoalSpec(cfg['goal_mode'],cfg['goal_frames'],cfg['goal_dt']),
                 AlternatingConfig(**cfg['foot_config']),command_mode=task.command_mode)
goals.spec.assert_checkpoint(ck,goals.foot_config)
c=Curriculum(ck['curriculum']['cfg']); c.load_state_dict(ck['curriculum'])
if args.stage is not None: c.stage=args.stage
goals.curriculum=c;goals.command_duration=12.
mdp=FootTrackingMDP(kin,torch.tensor(default_joint_pos_list(),device=dev),goals,task)
assert_task_checkpoint(ck,mdp)
pi=Policy(OBS_DIM,goals.spec.dim,hist=cfg['hist']).to(dev);pi.load_state_dict(ck['pi'])
critic=Critic(mdp.critic_dim+goals.spec.dim).to(dev);critic.load_state_dict(ck['critic'])
networks.AMP_DTYPE=torch.bfloat16 if cfg['amp']=='bf16' else None
sim=make_sim(dev,args)
with launch_simulation(sim,args):
    env=TrackingSim(sim,args.num_envs,None,mdp,goals=goals)
    report=evaluate(env,pi,goals,mdp,c,cfg['hist'],args.full,deadline=time.monotonic()+90,rich=args.full,critic=critic)
Path(args.out).write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
