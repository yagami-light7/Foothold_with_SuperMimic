"""在 M4.6 的真实回放起点上标定上身二阶差分奖励。"""
from argparse import ArgumentParser
from dataclasses import replace
import os
from pathlib import Path
from time import perf_counter

if __name__ == '__main__':
    os.environ['SUPERMIMIC_ROBOT'] = 'roban'

import numpy as np
import torch

from .m42v3_ac import cosine, gradient, objective
from .m43_wm_audit import make_policy, make_wm, task_for, write_json
from .m47_startup_audit import CHECKPOINTS, UPPER
from ..kinematics import RobotKinematics
from ..mdp import XI_DIM, unflatten_state
from ..motion import CONTROL_DT
from ..networks import Critic
from ..roban import JOINT_NAMES

LABELS = {'m43':'m43_best','w002':'w002_final','w005':'w005_best'}
WEIGHTS = (.01,.02,.05,.10)
PHASES = ('initial_support', 'swing', 'touchdown', 'perturb_recovery')
LEGS = [j for j in range(len(JOINT_NAMES)) if j not in UPPER]


def phases(f, force, motion, perturb):
    time = f[:,None]*CONTROL_DT
    lift, land = motion.liftoff_time, motion.touchdown_time
    return dict(initial_support=(time[:,0]<lift[0]) & (force[:,:,2]>=10.).all(-1),
                swing=((time>=lift) & (time<land)).any(-1),
                touchdown=((time-land).abs()<=.100001).any(-1),
                perturb_recovery=(time[:,0]<lift[0]) if perturb else torch.zeros_like(f,dtype=torch.bool))


@torch.no_grad()
def read_episode(archive, prefix, task, *, perturb=False):
    args, motion, goals, mdp = task
    read = lambda key: torch.as_tensor(archive[prefix+key],device=motion.q.device)
    raw = read('action')
    n = len(raw)
    # F2 冻结评估无随机化，上一步及上两步均来自同一条真实动作序列。
    zero = torch.zeros_like(raw[:1])
    prev = torch.cat((zero,raw[:-1]))
    prev2 = torch.cat((zero,prev[:-1]))
    state_full, frame = read('state'), read('frame').long()
    f, f1 = frame[:-1], frame[1:]
    state = state_full[:-1]
    command = motion.anchor(torch.zeros_like(f))
    xi = state.new_zeros(n,XI_DIM)
    xi[:,0] = 1.
    ss = unflatten_state(state)
    obs = goals.obs(mdp, ss, prev, xi, f, command)
    goal = goals.goal(ss,f,f,command)
    anchor_pose = state_full[0,13:13+len(JOINT_NAMES)].expand_as(raw)
    target = read('target')
    # 重算历史实际下发目标的奖励，保留真实 Replay 口径。
    _, terms = mdp.reward(unflatten_state(state_full[1:]),f1,raw,prev,True,command=command,ep_step=f1,
                          a_prev2=prev2,pd_target=target)
    values = torch.stack((-terms['upper_action_smoothness'],terms['foot_objective'],
                          -terms['action_rate_contribution'],-terms['torque_contribution']),-1)
    selection = phases(f,read('force')[:-1],motion,perturb)
    history_index = (f[:,None]+torch.arange(1-args.hist,1,device=f.device)).clamp_min(0)
    starts = (state,prev,f,xi,obs[history_index],goal,prev2,anchor_pose)
    allowed = f+args.horizon <= motion.num_frames-1
    return values, selection, starts, allowed, f>=2


def metric_stats(rows):
    cost, foot, rate, torque = rows.T
    if not len(cost):
        return {}
    avg = cost.mean()
    penalty = {}
    for w in WEIGHTS:
        contribution = w*CONTROL_DT*cost
        penalty[str(w)]=dict(mean=float(contribution.mean()),
                            foot_fraction=float(contribution.mean()/foot.mean().abs().clamp_min(1e-12)),
                            action_rate_fraction=float(contribution.mean()/rate.mean().clamp_min(1e-12)),
                            torque_fraction=float(contribution.mean()/torque.mean().clamp_min(1e-12)))
    return dict(n=len(cost),mean=float(avg),rms=float(cost.square().mean().sqrt()),
                p95=float(torch.quantile(cost,.95)),p99=float(torch.quantile(cost,.99)),max=float(cost.max()),
                foot_mean=float(foot.mean()),rate_mean=float(rate.mean()),torque_mean=float(torque.mean()), weights=penalty)


def gradient_stats(args,task,ck,batch,pi,wm,critic):
    _,motion,goals,mdp = task
    # 平滑系数先置为 1。各候选的分项梯度可按正权重精确线性缩放。
    packet=objective(batch[:6],pi,critic,wm,motion,mdp,goals,args,
                     ck['imagination_contract']['bootstrap_upper_bound'],4701,action_context=batch[6:])
    terms={'foot':packet['C'],'body':packet['direct']['body_prior'],'smooth_unit':packet['direct']['upper_action_smoothness']}
    params=tuple(pi.parameters())
    cg, action = {}, {}
    for key, obj in terms.items():
        cg[key]=gradient(obj,params)
        dg=torch.autograd.grad(obj,packet['actions'],retain_graph=True,allow_unused=True)
        actions=torch.stack([torch.zeros_like(a) if g is None else g.detach() for a,g in zip(packet['actions'],dg)])
        if not (cg[key].isfinite().all() and actions.isfinite().all()):
            raise FloatingPointError(f'上身平滑分项梯度非有限：{key}')
        action[key]=dict(upper=float(actions[...,UPPER].square().mean().sqrt()),
                         lower=float(actions[...,LEGS].square().mean().sqrt()))
    norms={k:float(g.norm()) for k,g in cg.items()}
    return dict(n=len(batch[0]),norm=norms,action_rms=action,
                cosine={f'{a}_{b}':cosine(cg[a],cg[b]) for a,b in
                        (('foot','smooth_unit'),('body','smooth_unit'),('foot','body'))},
                smooth_to_foot={str(w):w*norms['smooth_unit']/max(norms['foot'],1e-12) for w in WEIGHTS})


def run(args):
    torch.set_num_threads(2)
    report=dict(archive=str(args.archive),candidate_weights=WEIGHTS,body_weight=.02,actors={})
    kin=RobotKinematics(args.device)
    started=perf_counter()
    with np.load(args.archive,allow_pickle=False) as archive:
        for name, source in LABELS.items():
            path, expected=CHECKPOINTS[name]
            ck=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            assert ck['iteration']==expected
            task=task_for(ck,kin,args.m4_reference,args.device)
            cfg,motion,goals,mdp=task
            mdp.cfg=replace(mdp.cfg,body_prior_weight=.02,upper_action_smoothness_weight=1.)
            pi, wm=make_policy(ck,mdp,goals,args.device),make_wm(ck,args.device)
            critic=Critic(mdp.critic_dim+goals.spec.dim).to(args.device).eval().requires_grad_(False)
            critic.load_state_dict(ck['critic'])
            entries=[read_episode(archive,f'{split}_{source}_s{seed}_',task,perturb=split=='perturb')
                     for split,seeds in (('nominal',(1001,1002,1003)),('perturb',(1101,1102,1103,1104,1105,1106)))
                     for seed in seeds]
            result=report['actors'][name]=dict(checkpoint=path,iteration=expected,phases={})
            for phase in ('all',*PHASES):
                choose=lambda p,m: torch.ones_like(m) if phase=='all' else p[phase]
                values=torch.cat([v[choose(p,valid)&valid] for v,p,_,_,valid in entries])
                partial=[tuple(t[choose(p,valid)&allowed] for t in s) for _,p,s,allowed,valid in entries]
                batch=tuple(torch.cat(column) for column in zip(*partial))
                i=torch.linspace(0,len(batch[0])-1,min(args.starts,len(batch[0])),device=args.device).round().long()
                grad=gradient_stats(cfg,task,ck,tuple(t[i] for t in batch),pi,wm,critic)
                result['phases'][phase]=dict(replay=metric_stats(values),bptt=grad)
                print(name,phase,grad['n'],grad['smooth_to_foot'],flush=True)
            write_json(args.output,report)
    ratios={str(w):[item['bptt']['smooth_to_foot'][str(w)] for r in report['actors'].values()
                    for item in r['phases'].values()] for w in WEIGHTS}
    report.update(elapsed_s=perf_counter()-started,
                  ratios={w:dict(min=min(x),median=float(np.median(x)),p90=float(np.quantile(x,.90)),max=max(x))
                          for w,x in ratios.items()})
    write_json(args.output,report)


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--m4_reference',type=Path,required=True)
    parser.add_argument('--archive',type=Path,default=Path('outputs/m4_6/cuda_retest.npz'))
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--starts',type=int,default=20)
    parser.add_argument('--output',type=Path,default=Path('outputs/m4_7/smoothness_preflight/calibration.json'))
    run(parser.parse_args())

