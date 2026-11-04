"""CPU checkpoint probe: no simulator, optimizer updates or training changes.

Nominal grounded starts are probes, not held-out Newton validation. Compare raw
policy actions and critic values against the known real-action and reward bounds.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from supermimic import networks
from supermimic.curriculum import Curriculum
from supermimic.foot_mdp import FootTrackingMDP,resolve_foot_task
from supermimic.foot_trajectory import AlternatingConfig
from supermimic.goals import GoalSource,GoalSpec
from supermimic.kinematics import S54Kinematics
from supermimic.mdp import OBS_DIM,normalize_xi
from supermimic.networks import Critic,Policy,WorldModel
from supermimic.s54 import NUM_J,XI_DIM,default_joint_pos_list

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('checkpoints',nargs='+');p.add_argument('--out',required=True);args=p.parse_args()
torch.set_num_threads(2);networks.AMP_DTYPE=None
results=[]
for path in args.checkpoints:
    ck=torch.load(path,map_location='cpu',weights_only=False);a=ck['args']
    task=resolve_foot_task(a['foot_task_config']);kin=S54Kinematics('cpu')
    goals=GoalSource(None,kin,GoalSpec(a['goal_mode'],a['goal_frames'],a['goal_dt']),
                     AlternatingConfig(**a['foot_config']),command_mode=task.command_mode)
    c=Curriculum(ck['curriculum']['cfg']);c.load_state_dict(ck['curriculum'])
    goals.curriculum=c;goals.command_duration=12.
    mdp=FootTrackingMDP(kin,torch.tensor(default_joint_pos_list()),goals,task)
    wm=WorldModel();wm.load_state_dict(ck['wm'])
    pi=Policy(OBS_DIM,goals.spec.dim,hist=a['hist']);pi.load_state_dict(ck['pi'])
    critic=Critic(mdp.critic_dim+goals.spec.dim);critic.load_state_dict(ck['critic'])
    torch.manual_seed(4254);S=mdp.perturb_reset(mdp.nominal_state(8));cmd=goals.new_command(mdp.nominal_state(8))
    xi=torch.zeros(8,XI_DIM);xi[:,0]=1.;xi_n=normalize_xi(xi)
    step=torch.zeros(8,dtype=torch.long);prev=torch.zeros(8,NUM_J)
    obs=goals.obs(mdp,S,prev,xi,step,cmd);history=obs[:,None].repeat(1,a['hist'],1)
    row=dict(checkpoint=path,iteration=ck['iteration'],max_step_reward=mdp.max_step_reward,
             valid_discounted_return_upper_bound=mdp.max_step_reward/(1-a['gamma']),steps=[])
    with torch.no_grad():
        for j in range(a['horizon']+1):
            g=goals.goal(S,step,step,cmd)
            X=torch.cat((mdp.critic_features(S,step,xi_n,command=cmd,ep_step=step),g),-1)
            V=critic(X);raw=pi(history,g)
            row['steps'].append(dict(tick=j,value_mean=float(V.mean()),value_min=float(V.min()),value_max=float(V.max()),
                raw_action_absmax=float(raw.abs().max()),raw_action_outside_real_clip_fraction=float((raw.abs()>20).float().mean())))
            if j==a['horizon']:break
            S=wm.step(S,raw,xi_n);step+=1
            reward=mdp.reward(S,step,raw,prev,command=cmd,ep_step=step)
            row['steps'][-1]['imagined_reward_mean']=float(reward.mean())
            obs=goals.obs(mdp,S,raw,xi,step,cmd)
            history=torch.cat((history[:,1:],obs[:,None]),1);prev=raw
    results.append(row);print(json.dumps(row),flush=True)
Path(args.out).write_text(json.dumps(results,indent=2)+'\n')
