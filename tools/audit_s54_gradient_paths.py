"""Deterministic CPU/float64 BPTT audit; never opens another GPU simulator.

Grounded perturbed starts are probes, not a replacement for paired Newton data.
"""
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from supermimic import networks
from supermimic.control import bound_action,bound_bootstrap,return_upper_bound
from supermimic.foot_mdp import FootTrackingMDP,resolve_foot_task
from supermimic.foot_trajectory import AlternatingConfig,FootstepCommand
from supermimic.goals import GoalSource,GoalSpec
from supermimic.kinematics import S54Kinematics
from supermimic.mdp import OBS_DIM,normalize_xi
from supermimic.networks import Critic,Policy,WorldModel
from supermimic.s54 import NUM_J,XI_DIM,default_joint_pos_list

p=argparse.ArgumentParser(description=__doc__);p.add_argument('checkpoints',nargs='+');p.add_argument('--out',required=True);a=p.parse_args()
torch.set_num_threads(2);networks.AMP_DTYPE=None;results=[]
for path in a.checkpoints:
 ck=torch.load(path,map_location='cpu',weights_only=False);args=ck['args'];dt=torch.float64
 kin=S54Kinematics('cpu',dtype=dt);cfg=resolve_foot_task(args['foot_task_config'])
 goals=GoalSource(None,kin,GoalSpec(args['goal_mode'],args['goal_frames'],args['goal_dt']),AlternatingConfig(**args['foot_config']),cfg.command_mode)
 mdp=FootTrackingMDP(kin,torch.tensor(default_joint_pos_list(),dtype=dt),goals,cfg)
 wm=WorldModel();wm.load_state_dict(ck['wm']);wm.double().requires_grad_(False)
 pi=Policy(OBS_DIM,goals.spec.dim,hist=args['hist']);pi.load_state_dict(ck['pi']);pi.double()
 critic=Critic(mdp.critic_dim+goals.spec.dim);critic.load_state_dict(ck['critic']);critic.double().requires_grad_(False)
 n=4;H=args['horizon'];nominal=mdp.nominal_state(n);torch.manual_seed(4254);initial=mdp.perturb_reset(nominal)
 feet=kin.forward(*nominal)[0][:,mdp.foot_indices].detach();feet[...,2]=feet.new_tensor(goals.ankle_height)
 q=torch.zeros(n,2,4,dtype=dt);q[...,0]=1.;heading=q[:,0].clone();M=64
 cmd=FootstepCommand(feet,q,torch.zeros(n,M,dtype=torch.long),torch.zeros(n,M,3,dtype=dt),
     torch.cat((torch.ones(n,M,1,dtype=dt),torch.zeros(n,M,3,dtype=dt)),-1),
     torch.zeros(n,M,dtype=dt),torch.ones(n,M,dtype=dt),torch.zeros(n,M,dtype=dt),torch.zeros(n,dtype=torch.long),heading).validate()
 xi=torch.zeros(n,XI_DIM,dtype=dt);xi[:,0]=1.;xn=normalize_xi(xi);upper=return_upper_bound(mdp.max_step_reward,args['gamma'])
 def rollout():
  S=initial;prev=torch.zeros(n,NUM_J,dtype=dt);step=torch.zeros(n,dtype=torch.long)
  obs=goals.obs(mdp,S,prev,xi,step,cmd);hist=obs[:,None].repeat(1,args['hist'],1)
  alive=torch.ones(n,dtype=dt);rewardJ=torch.zeros(n,dtype=dt);footJ=rewardJ.clone();actions=[];wmz=[]
  for t in range(H):
   g=goals.goal(S,step,step,cmd);action=bound_action(pi(hist,g));actions.append(action)
   wmz.append(wm.in_norm(wm.inputs(S,action,xn)))
   S=wm.step(S,action,xn);step=step+1
   r,terms=mdp.reward(S,step,action,prev,terms=True,command=cmd,ep_step=step)
   rewardJ=rewardJ+args['gamma']**t*alive*r
   footJ=footJ+args['gamma']**t*alive.detach()*terms['foot_objective']
   alive=alive*(1-mdp.soft_termination(S,step,command=cmd,ep_step=step))
   obs=goals.obs(mdp,S,action,xi,step,cmd);hist=torch.cat((hist[:,1:],obs[:,None]),1);prev=action
  g=goals.goal(S,step,step,cmd);X=torch.cat((mdp.critic_features(S,step,xn,command=cmd,ep_step=step),g),-1)
  raw=critic(X);boot=args['gamma']**H*alive*bound_bootstrap(raw,upper)
  diag=dict(raw_value_mean=float(raw.detach().mean()),raw_value_max=float(raw.detach().max()),
    raw_action_absmax=float(torch.stack(actions).detach().abs().max()),critic_z_absmax=float(critic.norm(X).detach().abs().max()),
    wm_input_z_absmax=float(torch.stack(wmz).detach().abs().max()),reward_return=float(rewardJ.detach().mean()),bootstrap=float(boot.detach().mean()))
  return -rewardJ.mean(),-boot.mean(),-footJ.mean(),diag
 params=tuple(pi.parameters());r,b,f,diag=rollout()
 def grad(x):return torch.autograd.grad(x,params,retain_graph=True,allow_unused=True)
 gr,gb,gf=grad(r),grad(b),grad(f);gt=grad(r+b)
 def norm(gs):return float(sum(g.square().sum() for g in gs if g is not None).sqrt())
 dot=float(sum((x*y).sum() for x,y in zip(gr,gb) if x is not None and y is not None))
 identity=float(sum(((x if x is not None else torch.zeros_like(z))+(y if y is not None else torch.zeros_like(z))-z).square().sum()
                    for x,y,z in zip(gr,gb,gt) if z is not None).sqrt())
 bias=pi.dec[-1].bias;analytic=gt[-1]
 idx=int(analytic.abs().argmax());eps=1e-5
 with torch.no_grad():bias[idx]+=eps
 rp,bp,_,_=rollout();plus=float((rp+bp).detach())
 with torch.no_grad():bias[idx]-=2*eps
 rm,bm,_,_=rollout();minus=float((rm+bm).detach())
 with torch.no_grad():bias[idx]+=eps
 fd=(plus-minus)/(2*eps);ag=float(analytic[idx])
 diag.update(checkpoint=path,iteration=ck['iteration'],obs_dim=OBS_DIM,goal_dim=goals.spec.dim,action_dim=NUM_J,critic_dim=mdp.critic_dim+goals.spec.dim,
   wm_parameters=sum(p.numel() for p in wm.parameters()),actor_parameters=sum(p.numel() for p in pi.parameters()),critic_parameters=sum(p.numel() for p in critic.parameters()),
   reward_gradnorm=norm(gr),bootstrap_gradnorm=norm(gb),pure_foot_gradnorm=norm(gf),total_gradnorm=norm(gt),
   gradient_sum_error=identity,gradient_cosine=dot/max(norm(gr)*norm(gb),1e-30),
   bias_index=idx,bias_analytic_gradient=ag,bias_finite_difference=fd,finite_difference_relative_error=abs(ag-fd)/max(abs(ag),abs(fd),1e-10),
   wm_or_critic_parameter_grad_leak=any(p.grad is not None for p in tuple(wm.parameters())+tuple(critic.parameters())))
 results.append(diag);print(json.dumps(diag),flush=True)
Path(a.out).write_text(json.dumps(results,indent=2)+'\n')
