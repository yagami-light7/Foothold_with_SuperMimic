"""用同一真实 Newton 轨迹审计 M4.2 的 Critic 与 Actor BPTT。"""

import argparse
import copy
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

if __name__ == '__main__':
    os.environ['SUPERMIMIC_ROBOT'] = 'roban'

import numpy as np
import torch

from ..control import bound_bootstrap
from ..kinematics import RobotKinematics
from ..mdp import flatten_state, normalize_xi, unflatten_state
from ..motion import CONTROL_DT
from ..networks import Critic, Policy, WorldModel
from ..roban import GOAL_FRAME_DIM, XI_DIM, assert_checkpoint
from .m4_task import build_m4_task


FOOT = ('swing_position', 'swing_orientation', 'stance_position', 'contact_schedule', 'touchdown')
POSTURE = ('upright', 'survival')
REGULARIZATION = ('joint_limit', 'torque', 'action_rate')


def load_trace(data, iteration, seed, device):
    prefix = f'i{iteration}_s{seed}_'
    keys = ('state', 'observation', 'goal', 'xi', 'frame', 'action', 'reward', 'term', 'timeout', 'force')
    run = {key: torch.as_tensor(data[prefix + key], device=device) for key in keys}
    run['reason'] = str(data[prefix + 'termination'].item())
    t = len(run['action'])
    assert t + 1 == len(run['state']) == len(run['frame']) and t == len(run['observation'])
    assert torch.equal(run['frame'], torch.arange(t + 1, device=device))
    return run


def episode_input(motion, mdp, goals, run, hist):
    frames = run['frame'][:-1]
    state = unflatten_state(run['state'][:-1])
    command = motion.anchor(torch.zeros_like(frames))
    action = run['action']
    previous = torch.cat((torch.zeros_like(action[:1]), action[:-1]))
    xi = run['xi'][:-1]
    goal = goals.goal(state, frames, frames, command)
    obs = goals.obs(mdp, state, previous, xi, frames, command)
    observations = dict(
        obs_max_delta=float((obs - run['observation']).abs().max()),
        goal_max_delta=float((goal - run['goal']).abs().max()),
    )
    # 从保存的真实 Actor 输入重建历史，重置前缺失的历史由第零帧填充。
    history_index = (frames[:, None] + torch.arange(1-hist, 1, device=xi.device)).clamp_min(0)
    histories = run['observation'][history_index]
    return frames, state, command, xi, goal, previous, histories, observations


def verify_features(motion, mdp, goals, critic, pi, run, expected_obs):
    frames, state, command, xi, goal, previous, histories, result = expected_obs
    xi_n = normalize_xi(xi)
    chart = mdp.critic_features(state, frames, xi_n, command=command, ep_step=frames)
    state_dim = run['state'].shape[-1]
    assert chart.shape[-1] + goal.shape[-1] == critic.net[0].in_features == 970
    assert histories.shape[1:] == (pi.hist, 69) and goal.shape[-1] == 320
    torch.testing.assert_close(chart[:, :state_dim], flatten_state(state))
    torch.testing.assert_close(chart[:, state_dim:state_dim + XI_DIM], xi_n)
    body = chart[:, state_dim + XI_DIM:].view(-1, goals.spec.frames, GOAL_FRAME_DIM)
    wanted = (frames[:, None] + 5 * torch.arange(1, 11, device=frames.device)).clamp_max(motion.num_frames-1)
    # 维度 15:36 是未来关节位置，36:57 是乘 0.1 后的关节速度。
    torch.testing.assert_close(body[..., 15:36], motion.q[wanted])
    torch.testing.assert_close(body[..., 36:57], .1 * motion.qd[wanted])
    full = goal.view(-1, 10, 32)
    torch.testing.assert_close(full[..., 22], (wanted - frames[:, None]).float() * CONTROL_DT)
    # 只改变 Critic 的未来 q 与上下文，Actor 观测和动作必须保持相同。
    old = motion.q
    try:
        motion.q = old + 0.03125
        changed_obs = goals.obs(mdp, state, previous, xi + .2, frames, command)
        changed_goal = goals.goal(state, frames, frames, command)
        changed_chart = mdp.critic_features(state, frames, normalize_xi(xi+.2), command=command, ep_step=frames)
        torch.testing.assert_close(changed_obs, goals.obs(mdp, state, previous, xi, frames, command), rtol=0, atol=0)
        torch.testing.assert_close(changed_goal, goal, rtol=0, atol=0)
        assert not torch.equal(changed_chart, chart)
    finally:
        motion.q = old
    action = pi(histories, run['goal'])
    result.update(actors_input=[10, 69, 320], critic_input=[state_dim, XI_DIM, 10 * GOAL_FRAME_DIM, 320],
                  logged_action_max_delta=float((action - run['action']).abs().max()),
                  critic_norm_count=float(critic.norm.count),
                  critic_xi_distinct_rows=int(xi_n.unique(dim=0).shape[0]),
                  critic_xi_mean_error=float((xi_n - critic.norm.mean[state_dim:state_dim + XI_DIM]).abs().max()))
    return result


def compute_returns(rewards, gamma):
    result = torch.empty_like(rewards)
    value = rewards.new_zeros(())
    for i in range(len(rewards)-1, -1, -1):
        value = rewards[i] + gamma * value
        result[i] = value
    return result


def agreement(value, mc):
    if len(value) == 0:
        return {'n': 0}
    v, r = value.double(), mc.double()
    dv, dr = v-v.mean(), r-r.mean()
    pearson = None
    if dv.norm() > 1e-8 and dr.norm() > 1e-8:
        pearson = float((dv@dr) / (dv.norm()*dr.norm()))
    return dict(n=len(v), mean_v=float(v.mean()), mean_mc=float(r.mean()), bias=float((v-r).mean()),
                rmse=float((v-r).square().mean().sqrt()), pearson=pearson,
                range_v=[float(v.min()), float(v.max())])


def value_by_phase(critic, trajectories, motion, goals, mdp, gamma):
    runs = []
    first_liftoff = float(motion.liftoff_time[0])
    first_touchdown = float(motion.touchdown_time[0])
    for run, item in trajectories:
        f, s, command, xi, g, *_ = item
        input_v = torch.cat((mdp.critic_features(s, f, normalize_xi(xi), command=command, ep_step=f), g), -1)
        v = critic(input_v)
        t = f * CONTROL_DT
        mc = compute_returns(run['reward'], gamma)
        # 跨参考终点与硬终止均以未来真实回报为零。故障中止的轨迹无完整 MC 标签。
        assert bool(run['term'][-1] | run['timeout'][-1]), run['reason']
        phases = dict(all=torch.ones_like(f, dtype=torch.bool),
                      initial_support=t < first_liftoff-1e-5,
                      true_double_support=(run['force'][:-1, :, 2] >= 10.).all(-1),
                      first_swing=(t >= first_liftoff-1e-5) & (t < first_touchdown-1e-5),
                      near_first_touchdown=(t-first_touchdown).abs() <= .10001,
                      near_fall=(f >= len(f)-10) & run['term'][-1],
                      near_reference_end=(t >= motion.duration-.20001) & run['timeout'][-1])
        norm = critic.norm(input_v).abs().flatten()
        runs.append((v, mc, phases, norm))
    normalized = torch.cat([row[3] for row in runs])
    return {name: agreement(torch.cat([v[mask[name]] for v, _, mask, _ in runs]),
                            torch.cat([mc[mask[name]] for _, mc, mask, _ in runs]))
            for name in runs[0][2]} | {'normalized_input_z99': float(torch.quantile(normalized, .99))}


def sample_starts(trajectories, clip_end, horizon, batch, seed, device):
    entries = []
    for run, item in trajectories:
        frames, _, _, xi, goal, previous, histories, _ = item
        can_start = frames + horizon <= clip_end
        entries.append(tuple(x[can_start] for x in (run['state'][:-1], previous, frames, xi, histories, goal)))
    all_rows = tuple(torch.cat([entry[i] for entry in entries]) for i in range(len(entries[0])))
    index = torch.randperm(len(all_rows[0]), generator=torch.Generator(device=device).manual_seed(seed),
                          device=device)[:batch]
    return tuple(x[index] for x in all_rows)


def inverse_position_terms(mdp, state, frames, command):
    """只在诊断图里换 3 个位置核，保持事件掩码及原权重。"""
    ref = mdp._reference(frames, command)
    foot = mdp.motion.feet(state)[0]
    sigma2 = mdp.cfg.foot_position_std**2
    error_swing = (foot-ref['foot_p'][:, 0]).square().sum(-1)
    error_stance = (foot-ref['stance_p'][:, 0]).square().sum(-1)
    contact = ref['contact'][:, 0]
    motion = mdp.motion
    previous = torch.maximum(frames-1, motion.clip_start[motion.clip_id[frames]])
    event = motion.contact[frames] & ~motion.contact[previous] & motion.contact_valid[frames] & motion.contact_valid[previous]
    expressions = dict(swing_position=(~contact / (1+error_swing/sigma2)).mean(-1),
                       stance_position=(contact / (1+error_stance/sigma2)).mean(-1),
                       touchdown=(event / (1+error_swing/sigma2)).mean(-1))
    return {name: raw * getattr(mdp.cfg, name) * CONTROL_DT for name, raw in expressions.items()}


def position_kernel_signal(trajectories, motion, mdp):
    # 在真实状态上比较位置核对足底平移误差的局部导数绝对值。
    distances = {name: [] for name in ('first_swing', 'all_planned_touchdown', 'first_touchdown')}
    for run, _ in trajectories:
        frames = run['frame'][1:]
        command = motion.anchor(torch.zeros_like(frames))
        reference = mdp._reference(frames, command)
        foot = motion.feet(unflatten_state(run['state'][1:]))[0]
        distance = (foot-reference['foot_p'][:, 0]).norm(dim=-1)
        t = frames * CONTROL_DT
        side = int(motion.touchdown_side[0])
        first_swing = (t >= float(motion.liftoff_time[0])-1e-5) & (t < float(motion.touchdown_time[0])-1e-5)
        previous = torch.maximum(frames-1, motion.clip_start[motion.clip_id[frames]])
        event = motion.contact[frames] & ~motion.contact[previous] & motion.contact_valid[frames] & motion.contact_valid[previous]
        first_window = (t-float(motion.touchdown_time[0])).abs() < CONTROL_DT / 2
        distances['first_swing'].append(distance[first_swing, side])
        distances['all_planned_touchdown'].append(distance[event])
        distances['first_touchdown'].append(distance[first_window & event[:, side], side])
    out = {}
    for name, group in distances.items():
        d = torch.cat(group)
        if not len(d):
            out[name] = {'n': 0}
            continue
        x = d.square()/mdp.cfg.foot_position_std**2
        deriv = 2*d/mdp.cfg.foot_position_std**2
        exp = deriv * (-x).exp()
        inverse = deriv/(1+x).square()
        out[name] = dict(n=len(d), distance_p50=float(torch.quantile(d,.5)), distance_p90=float(torch.quantile(d,.9)),
                         exp_derivative_p50=float(torch.quantile(exp,.5)), inverse_derivative_p50=float(torch.quantile(inverse,.5)),
                         exp_below_1e_4=float((exp<1e-4).float().mean()))
    return out


def objective(batch, pi, critic, wm, motion, mdp, goals, args, value_upper, seed, *, action_context=None):
    # 复用正式实现的 WM、观测噪声、奖励与软终止。批次步法与 train.py 对齐。
    torch.manual_seed(seed)
    flat, previous, f0, xi, history, g = batch
    state = unflatten_state(flat)
    history = mdp.add_noise(history)
    xi_n = normalize_xi(xi)
    command = motion.anchor(torch.zeros_like(f0))
    H = args.horizon
    states, actions, previous_actions, frames, effective = [], [], [], [], []
    previous_two, targets = [], []
    if action_context is not None:
        previous2, startup_q = action_context
    for offset in range(H):
        f = f0 + offset
        a = pi(history, g)
        # 保留独立的图节点，便于区分 WM 路径与原动作直接进入奖励的梯度。
        if action_context is not None:
            target = mdp.executed_target(a, startup_q, f)
            eff = mdp.effective_action(target).view_as(a)
            targets.append(target)
            previous_two.append(previous2)
            previous2 = previous
        else:
            eff = mdp.wm_action(a, f).view_as(a)
        state = wm.step(state, eff, xi_n)
        f1 = f + 1
        obs = mdp.add_noise(goals.obs(mdp, state, a, xi, f1, command))
        history = torch.cat((history[:, 1:], obs[:, None]), 1)
        g = goals.goal(state, f1, f1, command)
        frames.append(f1)
        previous_actions.append(previous)
        actions.append(a)
        effective.append(eff)
        states.append(state)
        previous = a
    b = len(f0)
    stack = lambda xs: torch.stack(xs).flatten(0, 1)
    flat_s = tuple(stack([entry[i] for entry in states]) for i in range(6))
    flat_f = stack(frames)
    all_commands = command.repeat(H)
    history_inputs = dict(a_prev2=stack(previous_two),pd_target=stack(targets)) if action_context is not None else {}
    reward, terms = mdp.reward(flat_s, flat_f, stack(actions), stack(previous_actions), terms=True,
                               command=all_commands, ep_step=flat_f, **history_inputs)
    d = mdp.soft_termination(flat_s, flat_f, command=all_commands, ep_step=flat_f)
    d = torch.where(flat_f >= motion.clip_end[motion.clip_id[flat_f]], 1., d).view(H, b)
    alive = torch.cumprod(1-d, 0)
    c_t = torch.cat((torch.ones_like(alive[:1]), alive[:-1]), 0)
    disc = args.gamma ** torch.arange(H, device=reward.device, dtype=torch.float32).unsqueeze(-1)
    reward = reward.view(H, b)
    weighted = disc * c_t
    fixed_weighted = weighted.detach()
    reward_loss = -(weighted * reward).sum(0).mean()
    f_final = frames[-1]
    features = mdp.critic_features(state, f_final, xi_n, command=command, ep_step=f_final)
    terminal_value = critic(torch.cat((features, g), -1))
    bootstrap = -((args.gamma**H)*alive[-1]*bound_bootstrap(terminal_value, value_upper)).mean()
    direct = {name.removesuffix('_contribution'): -(fixed_weighted*value.view(H, b)).sum(0).mean()
              for name,value in terms.items() if name.endswith('_contribution')}
    foot_names = FOOT + (('swing_clearance',) if 'swing_clearance' in direct else ())
    # 长尾核对照只用于原有指数奖励，新奖励直接报告真实计算图。
    shaped = ({} if 'swing_clearance' in direct else
              {name: -(fixed_weighted*term.view(H,b)).sum(0).mean()
               for name,term in inverse_position_terms(mdp, flat_s, flat_f, all_commands).items()})
    shaped_foot = sum(shaped.get(name, direct[name]) for name in foot_names)
    survival_gate = -(disc * (c_t-c_t.detach()) * reward.detach()).sum(0).mean()
    foot = sum(direct[name] for name in foot_names)
    return dict(A=reward_loss+bootstrap, B=reward_loss, C=foot, shaped_foot=shaped_foot,
                bootstrap=bootstrap, survival_gate=survival_gate, direct=direct, shaped=shaped, actions=actions,
                effective=effective, clip=mdp.target_clip_mask(torch.stack(actions).detach()),
                foot_rmse=terms['foot_rmse'].mean(), alive=alive[-1].mean(), terminal_value=terminal_value.mean(),
                joint_position=flat_s[4], reward_frames=flat_f, fixed_weighted=fixed_weighted,
                coarse_gradient=dict(foot=foot, upright_survival=sum(direct[name] for name in POSTURE),
                                     regularization=sum(direct[name] for name in REGULARIZATION),
                                     other=sum(direct[name] for name in direct if name not in foot_names+POSTURE+REGULARIZATION)))


def gradient(loss, params):
    if not loss.requires_grad:
        return torch.cat([p.new_zeros(p.numel()) for p in params])
    result = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([p.new_zeros(p.numel()) if g is None else g.detach().flatten()
                      for p, g in zip(params, result)])


def cosine(a, b):
    n = a.norm()*b.norm()
    return float((a@b)/n) if n > 1e-12 else None


def update_policy(base, ck, params, grad, lr, batch, critic, wm, motion, mdp, goals, args, value_upper, seed,
                  *, return_policy=False):
    # A/B/C 三个分支都从同一快照与同一 Adam 状态出发，只执行一次训练式更新。
    pi = copy.deepcopy(base)
    opt = torch.optim.Adam(pi.parameters(), lr=lr, betas=(.9, .95))
    opt.load_state_dict(copy.deepcopy(ck['opt_pi']))
    for group in opt.param_groups:
        group['lr'] = lr
    for p, item in zip(pi.parameters(), grad.split([p.numel() for p in params])):
        p.grad = item.reshape_as(p).clone()
    torch.nn.utils.clip_grad_norm_(pi.parameters(), 1., error_if_nonfinite=True)
    torch.nn.utils.clip_grad_value_(pi.parameters(), .1)
    opt.step()
    if return_policy:
        return pi
    with torch.no_grad():
        outcome = objective(batch,pi,critic,wm,motion,mdp,goals,args,value_upper,seed)
    return summarize_loss(outcome)


def summarize_loss(item):
    return {name: float(item[name].detach()) for name in ('A', 'B','C','foot_rmse','alive','terminal_value')} | {
        'clamp': float(item['clip'].float().mean()),
        'loss_bootstrap': float(item['bootstrap'].detach())}


def gradient_audit(batch, pi, critic, wm, motion, mdp, goals, args, ck, seed, single_updates=True):
    value_upper = ck['imagination_contract']['bootstrap_upper_bound']
    params = tuple(pi.parameters())
    calc = objective(batch, pi, critic, wm, motion, mdp, goals, args, value_upper, seed)
    selection = {k: calc[k] for k in ('A','B','C','bootstrap','survival_gate','shaped_foot')} | calc['coarse_gradient']
    grads = {key: gradient(v, params) for key,v in selection.items()}
    finer = {}
    for name, value in calc['direct'].items():
        g = gradient(value, params)
        finer[name] = dict(norm=float(g.norm()), cosine_vs_bootstrap=cosine(g,grads['bootstrap']))
        if name in calc['shaped']:
            alt = gradient(calc['shaped'][name], params)
            finer[name].update(inverse_norm=float(alt.norm()), cosine_original_inverse=cosine(g,alt))
    groups = ('bootstrap','survival_gate','foot','upright_survival','regularization','other')
    sum_grad = sum(grads[k] for k in groups)
    verify_delta = float((grads['A']-sum_grad).norm()/grads['A'].norm().clamp_min(1e-9))
    if verify_delta > .005:
        raise AssertionError(f'Actor 梯度分解残差过大 {verify_delta}')
    norms = {k: float(g.norm()) for k,g in grads.items()}
    finite = {k: bool(g.isfinite().all()) for k,g in grads.items()}
    if not all(finite.values()):
        raise FloatingPointError(f'Actor 审计收到非有限梯度 {finite}')
    cosines = {a+'__'+b: cosine(grads[a],grads[b])
               for i,a in enumerate(groups) for b in groups[i+1:]}
    cosines.update(A_B=cosine(grads['A'],grads['B']), A_C=cosine(grads['A'],grads['C']),
                   B_C=cosine(grads['B'],grads['C']), C_inverse=cosine(grads['C'],grads['shaped_foot']))
    clip = calc['clip']
    unclipped_pd = getattr(mdp, 'unclipped_pd_target', False)
    outside = mdp.target_out_of_range_mask(torch.stack(calc['actions']).detach()) if unclipped_pd else clip
    clip_summary = {}
    # 原始动作梯度还有动作变化与历史路径。有效动作梯度乘 Clamp 局部雅可比得到动力学路径。
    for key in ('A','C'):
        ga = torch.stack(torch.autograd.grad(calc[key],calc['actions'],retain_graph=True))
        effective = torch.stack(torch.autograd.grad(calc[key],calc['effective'],retain_graph=True))
        local = torch.where(clip,0.,effective)
        def rms(g, mask):
            return float(g[mask].square().mean().sqrt()) if mask.any() else None
        clip_summary[key] = dict(raw_rms_clipped=rms(ga,clip), raw_rms_unclipped=rms(ga,~clip),
                                 dynamics_rms_clipped=rms(local,clip), dynamics_rms_unclipped=rms(local,~clip))
        if unclipped_pd:
            clip_summary[key].update(raw_rms_out_of_range=rms(ga,outside),
                                     dynamics_rms_out_of_range=rms(local,outside),
                                     dynamics_rms_in_range=rms(local,~outside),
                                     count_out_of_range=int(outside.sum()), count_actions=outside.numel())
    updates = None
    baseline = summarize_loss(calc)
    del calc
    if single_updates:
        a = ck['args']
        it = ck['iteration']
        mult = .03 + .97*.5*(1+math.cos(math.pi*it/a['lr_decay_iters'])) if it < a['lr_decay_iters'] else .03
        lr = mult*(a['actor_lr'] if a['actor_lr'] is not None else a['lr'])
        updates = {key: update_policy(pi,ck,params,grads[key],lr,batch,critic,wm,motion,mdp,goals,args,value_upper,seed)
                   for key in ('A','B','C')}
    return dict(grad_norm=norms, grad_finite=finite, cosines=cosines, terms=finer, decomposition_delta=verify_delta,
                clip_gradient=clip_summary, starts=len(batch[0]), horizon=args.horizon,
                before_update=baseline, after_adam_step=updates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m4_reference', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--seed', type=int, default=381)
    parser.add_argument('--no_updates', action='store_true', help='跳过冻结网络的单次 Adam 更新诊断')
    cli = parser.parse_args()
    dev = torch.device(cli.device)
    if dev.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        torch.set_num_threads(min(6,os.cpu_count() or 1))
    for name in ('early','last'):
        ck = torch.load(cli.checkpoints / f'{name}.pt', map_location='cpu', weights_only=False)
        assert_checkpoint(ck)
        args = SimpleNamespace(**ck['args'])
        assert args.m4_action_mapping == 'standing_offset' and not args.no_critic and args.amp == 'none'
        assert args.paired_latent_weight == 0 and not args.freeze_decoder
        kin = RobotKinematics(dev)
        motion, goals, mdp, terrain = build_m4_task(args,kin,[str(cli.m4_reference)],dev)
        mdp.assert_checkpoint_task(ck)
        goals.spec.assert_checkpoint(ck)
        assert terrain is None and mdp.physics_contract(args) == ck['physical_contract']
        pi = Policy(mdp.obs_dim,goals.spec.dim,hist=args.hist,obs_adapter=args.obs_adapter).to(dev).eval()
        critic = Critic(mdp.critic_dim+goals.spec.dim).to(dev).eval().requires_grad_(False)
        wm = WorldModel().to(dev).eval().requires_grad_(False)
        for model, key in ((pi,'pi'),(critic,'critic'),(wm,'wm')):
            model.load_state_dict(ck[key])
        it = ck['iteration']
        with np.load(cli.trace, allow_pickle=False) as data:
            with torch.no_grad():
                rows = [load_trace(data,it,int(seed),dev) for seed in data['seeds']]
                traj = [(run,episode_input(motion,mdp,goals,run,args.hist)) for run in rows]
                features = verify_features(motion,mdp,goals,critic,pi,*traj[0])
                phases = value_by_phase(critic,traj,motion,goals,mdp,args.gamma)
                kernel = position_kernel_signal(traj,motion,mdp)
                print('AC_VALUE',it,json.dumps(dict(features=features, value=phases, position_kernel=kernel),ensure_ascii=False,allow_nan=False),
                      flush=True)
                batch = sample_starts(traj,motion.num_frames-1,args.horizon,cli.batch,cli.seed,dev)
        result = gradient_audit(batch,pi,critic,wm,motion,mdp,goals,args,ck,cli.seed,not cli.no_updates)
        print('AC_GRADIENT',it,json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
