"""用冻结权重和匹配 Newton 状态审计上半身奖励及其梯度。"""

import argparse
from copy import copy
import os
from pathlib import Path

if __name__ == '__main__':
    os.environ['SUPERMIMIC_ROBOT'] = 'roban'

import numpy as np
import torch

from ..control import return_upper_bound
from ..kinematics import RobotKinematics
from ..mdp import normalize_xi, unflatten_state
from ..networks import Critic
from .m42v3_ac import cosine, episode_input, gradient, load_trace, objective, sample_starts
from .m42v3_wm import select_windows
from .m43_wm_audit import collect_samples, make_policy, make_wm, restore, slope_metrics, task_for, write_json
from .m44_wm_audit import view_trials
from .m4_task import build_m4_task
from .m45_upper import ARMS, UPPER


def reward_heads(mdp, state, frame, a, previous):
    anchor = mdp.motion.anchor(torch.zeros_like(frame))
    return mdp.reward(state, frame, a, previous, terms=True, command=anchor, ep_step=frame)


def check_reward(task, original, traces):
    _, motion, goals, mdp = task
    arm, waist, reward, quality, foot, derivative = [], [], [], [], [], []
    residual, gain_min = 0., float('inf')
    for run in traces:
        f, state, command, xi, g, prev, *_ = episode_input(motion, mdp, goals, run, 10)
        original_input = episode_input(original[1], original[3], original[2], run, 10)
        torch.testing.assert_close(g, original_input[4], atol=0, rtol=0)
        assert g.shape[-1] == 320 and run['observation'].shape[-1] == 69
        assert mdp.critic_features(state, f, normalize_xi(xi), command=command, ep_step=f).shape[-1] == 650
        state = unflatten_state(run['state'][1:])
        state = (*state[:4], state[4].detach().requires_grad_(True), state[5])
        frame, action = f+1, run['action']
        r, terms = reward_heads(mdp, state, frame, action, prev)
        baseline = reward_heads(original[3], state, frame, action, prev)[0]
        arm_error = (state[4][:, ARMS] - motion.q[frame][:, ARMS]).square().mean(-1).sqrt()
        waist_error = (state[4][:, 0]-motion.q[frame, 0]).abs()
        improved = state[4].clone()
        improved[:, ARMS] = torch.lerp(improved[:, ARMS], motion.q[frame][:, ARMS], .1)
        improved_terms = reward_heads(mdp, (*state[:4], improved, state[5]), frame, action, prev)[1]
        value = terms['body_prior_contribution']
        gain_min = min(gain_min, float((improved_terms['body_prior_contribution']-value).detach().min()))
        direct = torch.autograd.grad(value.sum(), state[4])[0][:, ARMS].norm(dim=-1)
        residual = max(residual, float((r-baseline-value).detach().abs().max()))
        arm.append(arm_error.detach()); waist.append(waist_error.detach()); reward.append(value.detach())
        quality.append(motion.quality[frame]); foot.append(terms['foot_objective'].detach())
        derivative.append(direct[arm_error >= .5])
    a, w, r, q, f, g = map(torch.cat, (arm, waist, reward, quality, foot, derivative))
    assert residual < 1e-7 and gain_min >= -1e-8 and len(g) and g.isfinite().all() and (g > 0).all()
    return dict(steps=len(a), arm_rms=float(a.square().mean().sqrt()), waist_rms=float(w.square().mean().sqrt()),
                body_reward=float(r.mean()), foot_reward=float(f.mean()), quality=[float(q.mean()), *torch.quantile(q, q.new_tensor([.1,.9])).tolist()],
                min_arm_reward_gain=gain_min, joint_grad_large_error_n=len(g), joint_grad_large_error_median=float(g.median()),
                reward_difference_residual=residual)


def bptt(checkpoint, task, wm, traces):
    cfg, motion, goals, mdp = task
    pi = make_policy(checkpoint, mdp, goals, cfg.device)
    critic = Critic(970).to(cfg.device).eval().requires_grad_(False)
    critic.load_state_dict(checkpoint['critic'])
    assert pi.obs_norm.mean.shape[-1] == 69 and pi.goal_norm.mean.shape[-1] == 320
    runs = [(x, episode_input(motion, mdp, goals, x, cfg.hist)) for x in traces]
    batch = sample_starts(runs, motion.num_frames-1, cfg.horizon, 72, 4505, cfg.device)
    groups = {'all': torch.ones_like(batch[2],dtype=torch.bool), 'initial': batch[2] < 40,
              'first_swing': (batch[2] >= 40)&(batch[2] < 80), 'later': batch[2] >= 80}
    report = {}
    for name, mask in groups.items():
        if not mask.any():
            continue
        graph = objective(tuple(t[mask] for t in batch), pi, critic, wm, motion, mdp, goals, cfg,
                          return_upper_bound(mdp.max_step_reward, cfg.gamma), 4505)
        scores = dict(body=graph['direct']['body_prior'], foot=graph['C'], total=graph['A'], critic=graph['bootstrap'])
        gs = {k: gradient(v, tuple(pi.parameters())) for k,v in scores.items()}
        paths = {}
        for k in ('body', 'foot'):
            actions = graph['actions']
            vals = torch.autograd.grad(scores[k], actions, retain_graph=True, allow_unused=True)
            path = torch.stack([torch.zeros_like(a) if v is None else v.detach() for a,v in zip(actions,vals)])
            legs = [i for i in range(path.shape[-1]) if i not in UPPER]
            paths[k] = [float(path[..., ch].square().mean().sqrt()) for ch in (UPPER,legs)]
        assert all(v.isfinite().all() for v in gs.values()) and gs['body'].norm() > 0
        report[name] = dict(starts=int(mask.sum()), norms={k: float(v.norm()) for k,v in gs.items()},
                            cosine_foot=cosine(gs['body'],gs['foot']), cosine_critic=cosine(gs['body'],gs['critic']),
                            combined_norm=float((gs['body']+gs['foot']).norm()), action_rms_upper_lower=paths)
    return report


@torch.no_grad()
def prediction(task, model, batch):
    cfg, motion, _, mdp = task
    state = unflatten_state(batch['start'])
    report = {}
    for k in range(8):
        f, a, prev = (batch[name][:,k] for name in ('frame','action','prior'))
        state = model.step(state, mdp.wm_action(a,f), normalize_xi(batch['xi'][:,k]))
        if k not in (0,3,7):
            continue
        real = unflatten_state(batch['next_state'][:,k])
        dq = state[4]-real[4]
        est = reward_heads(mdp,state,f+1,a,prev)[1]['body_prior_contribution']
        actual = reward_heads(mdp,real,f+1,a,prev)[1]['body_prior_contribution']
        report[str(k+1)] = dict(n=len(a), arm_rad=float(dq[:, ARMS].square().mean().sqrt()),
                                waist_rad=float(dq[:,0].square().mean().sqrt()),
                                foot_m=float((motion.feet(state)[0]-motion.feet(real)[0]).square().sum(-1).mean().sqrt()),
                                body_reward_rmse=float((est-actual).square().mean().sqrt()))
    return report


def local_test(task, models, samples):
    from isaaclab.app import launch_simulation

    from .m4_dynamics import SolverReadback, fault_masks
    from .m4_task import M4TrackingSim, make_m4_sim

    cfg, motion, goals, mdp = task
    n = len(samples['state'])
    f, a, prev = (samples[k] for k in ('frame','action','previous'))
    rng = torch.Generator(device=a.device).manual_seed(4505)
    random = torch.randn(3,*a.shape,generator=rng,device=a.device)
    random[1,:,UPPER] = 0.
    random[2,:,[i for i in range(a.shape[-1]) if i not in UPPER]] = 0.
    direction = torch.nn.functional.normalize(random,dim=-1)
    d, eps = len(direction), .15
    sim = make_m4_sim(cfg.device,cfg)
    with torch.no_grad(), launch_simulation(sim,cfg):
        env = M4TrackingSim(sim,2*d*n,motion,mdp,eval_mode=True,goals=goals,gamma=cfg.gamma)
        probe = SolverReadback(env)
        s = restore(env,probe,samples,2*d)
        action = torch.cat((a+eps*direction,a-eps*direction)).flatten(0,1)
        after = env.advance_physics(env.pd_target(action))
        bad = torch.stack(fault_masks(s,after,probe,mdp)[:3]).any(0).view(2*d,n)
        valid = ~(bad[:d] | bad[d:])
        heads = reward_heads(mdp,after,(f+1).repeat(2*d),action,prev.repeat(2*d,1))[1]
    report = {}
    for name, model in models.items():
        v = a.clone().requires_grad_(True)
        pred = model.step(unflatten_state(samples['state']),mdp.wm_action(v,f),normalize_xi(samples['xi']))
        est = reward_heads(mdp,pred,f+1,v,prev)[1]
        report[name] = {}
        for term in ('foot_objective','body_prior_contribution'):
            y = heads[term].view(2*d,n)
            truth = (y[:d]-y[d:])/(2*eps)/mdp.max_step_reward
            jac = torch.autograd.grad(est[term].sum(),v,retain_graph=True)[0].detach()/mdp.max_step_reward
            derivative = (jac*direction).sum(-1)
            report[name][term] = {label: slope_metrics(derivative[i],truth[i],valid[i],f,torch.ones_like(f,dtype=torch.bool))
                                  for i,label in enumerate(('mixed','legs','upper'))}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m4_reference', required=True,type=Path)
    parser.add_argument('--archive', type=Path,default=Path('outputs/m4_4_unclipped/evaluation/rollouts.npz'))
    parser.add_argument('--output', type=Path,required=True)
    parser.add_argument('--post', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(4)
    kin = RobotKinematics(args.device)
    sources = {'m43_best':'outputs/m4_3/checkpoints/best.pt','m43_final':'outputs/m4_3/checkpoints/iter_5000.pt'}
    if args.post:
        sources.update(m45_best='outputs/m4_5/checkpoints/best.pt',m45_final='outputs/m4_5/checkpoints/iter_5000.pt')
    report, tasks, models, windows = {}, {}, {}, {}
    with np.load(args.archive, allow_pickle=False) as data:
        for name, path in sources.items():
            ck = torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            old = task_for(ck,kin,args.m4_reference,args.device)
            cfg = copy(old[0])
            cfg.body_prior_weight = .10
            motion, goals, mdp, _ = build_m4_task(cfg,kin,[str(args.m4_reference)],args.device)
            task = (cfg,motion,goals,mdp)
            tasks[name], models[name] = task, make_wm(ck,args.device)
            view = view_trials(data,name,int(ck['iteration']))
            traces = [load_trace(view,int(ck['iteration']),int(s),args.device) for s in view['seeds']]
            row = report[name] = {'iteration':int(ck['iteration']), 'bptt':bptt(ck,task,models[name],traces)}
            if not args.post:
                row['real'] = check_reward(task,old,traces)
            windows[name] = select_windows(view,int(ck['iteration']),view['seeds'],cfg.hist,args.device)
        cfg, motion, _, mdp = tasks['m43_final']
        shared = collect_samples(data,'m43_final',motion,mdp,10,args.device)
        for name in models:
            report[name]['own_prediction'] = prediction(tasks[name],models[name],windows[name])
            report[name]['shared_prediction'] = prediction(tasks[name],models[name],windows['m43_final'])
    if torch.device(args.device).type == 'cuda':
        report['local_shared'] = local_test(tasks['m43_final'],models,shared)
    write_json(args.output,report)
    for name in models:
        print(name, report[name]['bptt']['all'], report[name].get('real', ''), flush=True)


if __name__ == '__main__':
    main()
