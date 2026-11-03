"""SuperMimic training (Algorithm A.1): collect -> fit world model -> BPTT policy + critic update."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from isaaclab.app import add_launcher_args, launch_simulation
from supermimic.goals import GoalSource, add_goal_args, resolve_goal_config
from supermimic.foot_mdp import FootTrackingMDP, resolve_foot_task, assert_task_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--iterations", type=int, default=30)
parser.add_argument("--buffer_steps", type=int, default=64)
parser.add_argument("--utd", type=float, default=0.4, help="updates per vectorized simulator step")
parser.add_argument("--horizon", type=int, default=8)
parser.add_argument("--goal_frames", type=int, default=10)
parser.add_argument("--hist", type=int, default=10)
parser.add_argument("--wm_window", type=int, default=8)
parser.add_argument("--wm_batches", type=int, default=1)
parser.add_argument("--wm_batch", type=int, default=64)
parser.add_argument("--pi_batch", type=int, default=32)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--action_noise", type=float, default=0.1)
parser.add_argument("--critic_epochs", type=int, default=5)
parser.add_argument("--critic_minibatches", type=int, default=4)
parser.add_argument("--no_critic", action="store_true")
parser.add_argument("--warmup_steps", type=int, default=32, help="sim steps before any update")
parser.add_argument("--wm_only_iters", type=int, default=10, help="world-model-only iterations at the start")
parser.add_argument("--lr_decay_iters", type=int, default=250_000)
parser.add_argument("--ckpt_every", type=int, default=30)
parser.add_argument("--log_every", type=int, default=10)
parser.add_argument("--motion_dir", type=str, default=os.path.join(REPO, "data", "retarget_data", "reference_startup_fixed"))
parser.add_argument("--clips", nargs="+", help="Clip names or quoted glob patterns; omit to train all top-level CSVs")
parser.add_argument("--log_dir", type=str, default=os.path.join(REPO, "logs", "s54_smoke"))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16"], help="autocast dtype for MLP matmuls")
parser.add_argument("--no_compile", action="store_true", help="eager PyTorch for a short smoke; same WM/BPTT algorithm")
parser.add_argument('--foot_task_config', default=None, help='Independent foot MDP JSON, including standing/alternating mode')
parser.add_argument('--check_foot_gradients', action='store_true', help='Measure pure foot reward Actor gradients and real/replay time consistency')
add_goal_args(parser)
add_launcher_args(parser)
args = parser.parse_args()
try:
    goal_spec, foot_config = resolve_goal_config(args)
    if args.goal_mode != 'foot_trajectory' and (args.foot_task_config or args.check_foot_gradients):
        raise ValueError('Foot task configuration/gradient checks require --goal_mode foot_trajectory')
    task_cfg = resolve_foot_task(args.foot_task_config) if args.goal_mode == 'foot_trajectory' else None
    if task_cfg is not None and args.horizon >= task_cfg.episode_steps:
        raise ValueError('Foot episode_steps must be greater than the imagination horizon')
except (ValueError, TypeError, OSError) as exc:
    parser.error(str(exc))
from pathlib import Path
log_path = Path(args.log_dir).resolve()
if not log_path.is_relative_to(Path(REPO)/"logs") or not log_path.name.startswith("s54_"):
    parser.error("S54 logs must be under this repository logs/s54_*; existing G1 logs are protected")
existing_training = ((log_path/'args.json').exists() or (log_path/'train_log.csv').exists()
                     or any(log_path.glob('*.pt')))
if not args.resume and existing_training:
    parser.error("Training artifacts already exist: use a fresh s54_* run or --resume last.pt")
if args.buffer_steps < max(args.wm_window, args.hist) or args.warmup_steps < args.wm_window:
    parser.error("buffer_steps >= max(wm_window,hist) and warmup_steps >= wm_window required")
if args.pi_batch * args.horizon < args.critic_minibatches:
    parser.error("Empty critic minibatches: increase pi_batch*horizon")

from supermimic.reference_selection import select_reference_files
motion_files = []
if task_cfg is None:
    try:
        motion_files = select_reference_files(args.motion_dir, args.clips)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
else:
    args.motion_dir = None
    args.foot_task_config = asdict(task_cfg)
# Record the resolved names, including when all clips or a glob was requested.
args.clips = [Path(path).stem for path in motion_files]
print(f"Training clips ({len(args.clips)}): {', '.join(args.clips)}" if task_cfg is None else
      f'Independent S54 foot task: {task_cfg.command_mode}; no Motion reference loaded', flush=True)

import torch
import torch.nn as nn

from supermimic.buffer import ReplayBuffer
from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import S54Kinematics
from supermimic.mdp import CRITIC_EXTRA, INV_DIM, MDP, OBS_DIM, RewardCfg, invariant_features, normalize_xi, unflatten_state
from supermimic.motion import MotionLib
import supermimic.networks as networks
from supermimic.networks import Critic, Policy, WorldModel
from supermimic.s54 import (default_joint_pos_list, NUM_J, checkpoint_contract, assert_checkpoint)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(args.seed)
networks.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
dev = "cuda:0"
args.reward_config = asdict(RewardCfg() if task_cfg is None else task_cfg)
os.makedirs(args.log_dir, exist_ok=True)
json.dump(vars(args), open(os.path.join(args.log_dir, "args.json"), "w"), indent=2)

kin = S54Kinematics(dev)
motion = MotionLib(motion_files, kin, dev) if task_cfg is None else None
default_q = torch.tensor(default_joint_pos_list(), device=dev)
goals = GoalSource(motion, kin, goal_spec, foot_config,
                   command_mode='alternating' if task_cfg is None else task_cfg.command_mode)
mdp = (MDP(kin, motion, default_q) if task_cfg is None else FootTrackingMDP(kin, default_q, goals, task_cfg))
reward_scale = mdp.max_step_reward
K, Hh, HZ = args.goal_frames, args.hist, args.horizon
GOAL_DIM = goal_spec.dim
CRITIC_IN = mdp.critic_dim + GOAL_DIM

wm = WorldModel().to(dev)
pi = Policy(OBS_DIM, GOAL_DIM, hist=Hh).to(dev)
critic = Critic(CRITIC_IN).to(dev)
opt_wm = torch.optim.Adam(wm.parameters(), lr=args.lr)
opt_pi = torch.optim.Adam(pi.parameters(), lr=args.lr, betas=(0.9, 0.95))
opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr)
start_it, wall_offset, sim_steps = 0, 0.0, 0
if args.resume:
    ck = torch.load(args.resume, map_location=dev)
    assert_checkpoint(ck)
    goal_spec.assert_checkpoint(ck, foot_config)
    assert_task_checkpoint(ck, mdp)
    if "critic" not in ck: raise ValueError("Resume requires last.pt with critic and optimizers")
    wm.load_state_dict(ck["wm"]), pi.load_state_dict(ck["pi"]), critic.load_state_dict(ck["critic"])
    opt_wm.load_state_dict(ck["opt_wm"]), opt_pi.load_state_dict(ck["opt_pi"]), opt_v.load_state_dict(ck["opt_v"])
    start_it, wall_offset, sim_steps = ck["iteration"], ck["wall_time"], ck["sim_steps"]
    if args.iterations <= start_it:
        raise ValueError(f'Resume checkpoint is at iteration {start_it}; --iterations must be larger '
                         f'(it is the total target, not additional iterations; got {args.iterations})')
    print(f'Resuming iteration {start_it} -> {args.iterations}; model and optimizer states restored. '
          f'Replay buffer will be refilled; WM-only phase is based on the absolute iteration.', flush=True)


# per-component floors on the error normalization (pos, rot, lin vel, ang vel, q, qd increments)
ERR_FLOOR = torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3), torch.full((3,), 2e-2), torch.full((3,), 5e-2),
                       torch.full((NUM_J,), 3e-3), torch.full((NUM_J,), 5e-2))).to(dev)  # fmt: skip


def _wm_rollout(s, a, xi):
    S = unflatten_state(s[:, 0])
    preds = []
    for j in range(args.wm_window):
        S = wm.step(S, a[:, j], normalize_xi(xi[:, j]))
        preds.append(S)
    return tuple(torch.stack([P[i] for P in preds], 1) for i in range(6))  # (B, k, ...)


def _wm_err(S_hat, s1, mask):
    std = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
    e = (wm.state_error(S_hat, unflatten_state(s1)) / std) ** 2  # (B, k, D)
    e = e.mean(-1)
    return (e * mask).sum() / s1.shape[0], e[:, -1]


def _imag_step(S, o_hist, a_prev, f, xi, xi_n, g, ep_step, command):
    a = pi(o_hist, g)
    S1 = wm.step(S, a, xi_n)
    f1 = f + 1
    step1 = ep_step + 1
    o1 = mdp.add_noise(goals.obs(mdp, S1, a, xi, f1, command))
    o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
    return S1, a, o_hist, f1, goals.goal(S1, f1, step1, command), step1


def _reward_term(S, f, a, a_prev, ep_step, command):
    if mdp.is_foot:
        r, terms = mdp.reward(S,f,a,a_prev,terms=True,command=command,ep_step=ep_step)
        foot_reward = terms['foot_objective']
    else:
        r = mdp.reward(S,f,a,a_prev)
        foot_reward = torch.zeros_like(r)
    return r, mdp.soft_termination(S,f,command=command,ep_step=ep_step), foot_reward


def _critic_input(S, f, xi_n, g, ep_step, command):
    return torch.cat((mdp.critic_features(S, f, xi_n,command=command,ep_step=ep_step), g), -1)


compile_fn = (lambda f: f) if args.no_compile else torch.compile
wm_rollout = compile_fn(_wm_rollout)
wm_err = compile_fn(_wm_err)
imag_step = compile_fn(_imag_step)
reward_term = compile_fn(_reward_term)
critic_input_c = compile_fn(_critic_input)


def lr_at(it: int) -> float:
    if it >= args.lr_decay_iters:
        return args.lr * 0.03
    return args.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * it / args.lr_decay_iters)))


def set_requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters():
        p.requires_grad_(flag)


# ------------------------------------------------------------------------------------------------
def wm_update():
    w = buffer.sample_windows(args.wm_batch, args.wm_window)
    S_hat = wm_rollout(w["s"], w["a"], w["xi"])
    loss, e_last = wm_err(S_hat, w["s1"], w["mask"])
    opt_wm.zero_grad(set_to_none=True)
    if not torch.isfinite(loss): raise FloatingPointError("Nonfinite WM/Policy loss")
    loss.backward()
    nn.utils.clip_grad_norm_(wm.parameters(), 1.0, error_if_nonfinite=True)
    opt_wm.step()
    m = w["mask"][:, -1]
    return loss.detach() / args.wm_window, (e_last.detach() * m).sum() / m.sum().clamp(min=1)


def sample_valid_starts(B: int):
    starts, hists, n = [], [], 0
    attempts = 0
    while n < B:
        attempts += 1
        if attempts > 1000: raise RuntimeError("Replay contains no valid imagination starts; reduce horizon or use longer clips")
        s, h = buffer.sample_starts(2 * B, Hh)
        f = s["f"]
        ok = (s['ep_step']+HZ <= task_cfg.episode_steps if mdp.is_foot else
              f + HZ <= motion.clip_end[motion.clip_id[f]])
        idx = ok.nonzero(as_tuple=False).squeeze(-1)[: B - n]
        starts.append({k: v[idx] for k, v in s.items()}), hists.append({k: v[idx] for k, v in h.items()})
        n += len(idx)
    cat = lambda ds: {k: torch.cat([d[k] for d in ds]) for k in ds[0]}  # noqa: E731
    return cat(starts), cat(hists)


PROFILE = bool(int(os.environ.get("SM_PROFILE", "0")))


def policy_update():
    if PROFILE:
        torch.cuda.synchronize(); tq0 = time.time()
    s, h = sample_valid_starts(args.pi_batch)
    S = unflatten_state(s["s"])
    xi, f, a_prev = s["xi"], s["f"], s["a_prev"]
    ep_step = s['ep_step']
    command = buffer.commands(s['command_id']) if goals.is_foot else None
    xi_n = normalize_xi(xi)
    with torch.no_grad():
        o_hist = goals.obs(mdp, unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"], command)
    o_hist = mdp.add_noise(o_hist)
    set_requires_grad(wm, False)
    set_requires_grad(critic, False)
    B = args.pi_batch
    g0 = goals.goal(S, f, ep_step, command)
    g = g0
    S0 = S
    Ss, As, Aps, Fs, Gs, Steps = [], [], [], [], [], []
    for t in range(HZ):
        Aps.append(a_prev)
        S, a_prev, o_hist, f, g, ep_step = imag_step(S, o_hist, a_prev, f, xi, xi_n, g, ep_step, command)
        Ss.append(S), As.append(a_prev), Fs.append(f), Gs.append(g)
        Steps.append(ep_step)
    flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731  (H*B, ...)
    S_all = tuple(flat([St[i] for St in Ss]) for i in range(6))
    f_all = flat(Fs)
    command_all = command.repeat(HZ) if command is not None else None
    r, d, foot_reward = reward_term(S_all, f_all, flat(As), flat(Aps), flat(Steps), command_all)
    r, d = r.view(HZ, B), d.view(HZ, B)
    alive = torch.cumprod(1 - d, 0)  # c_{t+1}
    c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
    disc = args.gamma ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
    J = (disc * c_t * r).sum(0)
    c = alive[-1]
    r_sum = (c_t * r).detach().sum(0)
    rews, dones = r.detach(), d.detach()
    if not args.no_critic:
        J = J + (args.gamma**HZ) * c * critic(_critic_input(S, f, xi_n, g, ep_step, command))
    loss = -J.mean()
    foot_gn = torch.zeros((), device=dev)
    if mdp.is_foot and args.check_foot_gradients:
        # Exclude direct action costs, posture, base terms, critic and survival derivatives.
        foot_loss = -(disc*c_t.detach()*foot_reward.view(HZ,B)).sum(0).mean()
        fg = torch.autograd.grad(foot_loss,tuple(pi.parameters()),retain_graph=True,allow_unused=True)
        foot_gn = torch.stack([g.detach().float().square().sum() for g in fg if g is not None]).sum().sqrt()
        if not torch.isfinite(foot_gn): raise FloatingPointError('Nonfinite pure foot Actor gradient')
    if PROFILE:
        torch.cuda.synchronize(); tq1 = time.time()
    opt_pi.zero_grad(set_to_none=True)
    if not torch.isfinite(loss): raise FloatingPointError("Nonfinite WM/Policy loss")
    loss.backward()
    if PROFILE:
        torch.cuda.synchronize(); tq2 = time.time()
    gn = nn.utils.clip_grad_norm_(pi.parameters(), 1.0, error_if_nonfinite=True)
    nn.utils.clip_grad_value_(pi.parameters(), 0.1)
    if torch.isfinite(gn):
        opt_pi.step()
    set_requires_grad(wm, True)
    set_requires_grad(critic, True)
    # ---------------- critic: GAE(lambda) targets over the imagined transitions ----------------
    v_loss = torch.zeros(())
    if not args.no_critic:
        with torch.no_grad():
            S_c = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0).flatten(0, 1) for i in range(6))
            f_c = torch.cat((s["f"].unsqueeze(0), torch.stack(Fs)), 0).flatten()
            g_c = torch.cat((g0.unsqueeze(0), torch.stack(Gs)), 0).flatten(0, 1)
            xi_c = xi_n.unsqueeze(0).expand(HZ + 1, -1, -1).flatten(0, 1)
            step_c = torch.cat((s['ep_step'].unsqueeze(0),torch.stack(Steps)),0).flatten()
            command_c = command.repeat(HZ+1) if command is not None else None
            X = critic_input_c(S_c, f_c, xi_c, g_c, step_c, command_c).view(HZ + 1, B, -1)
        critic.norm.update(X)
        with torch.no_grad():
            V = critic(X)
            R = V[-1]
            targets = []
            for t in range(HZ - 1, -1, -1):
                R = rews[t] + args.gamma * (1 - dones[t]) * ((1 - args.lam) * V[t + 1] + args.lam * R)
                targets.append(R)
            targets = torch.stack(targets[::-1])  # (H, B)
        Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
        n = Xf.shape[0]
        mb = n // args.critic_minibatches
        for _ in range(args.critic_epochs):
            perm = torch.randperm(n, device=dev)
            for i in range(args.critic_minibatches):
                idx = perm[i * mb : (i + 1) * mb]
                v_loss = ((critic(Xf[idx]) - Yf[idx]) ** 2).mean()
                opt_v.zero_grad(set_to_none=True)
                if not torch.isfinite(v_loss): raise FloatingPointError("Nonfinite critic loss")
                v_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0, error_if_nonfinite=True)
                opt_v.step()
    if PROFILE:
        torch.cuda.synchronize()
        print(f"[prof] imagine fwd {1e3*(tq1-tq0):.0f} ms | bwd {1e3*(tq2-tq1):.0f} ms | critic {1e3*(time.time()-tq2):.0f} ms", flush=True)
    return loss.detach(), v_loss.detach(), (r_sum / HZ / reward_scale).mean(), gn.detach(), c.mean().detach(), foot_gn


# ------------------------------------------------------------------------------------------------
sim_cfg = make_sim(dev, args)
with launch_simulation(sim_cfg, args):
    env = TrackingSim(sim_cfg, args.num_envs, motion, mdp, gamma=args.gamma, goals=goals)
    buffer = ReplayBuffer(args.buffer_steps, args.num_envs, dev, foot_commands=goals.is_foot)
    obs_hist = torch.zeros(args.num_envs, Hh, OBS_DIM, device=dev)

    @torch.no_grad()
    def collect(n_steps: int, stats: dict):
        global obs_hist, sim_steps
        for _ in range(n_steps):
            S = env.state()
            o = mdp.add_noise(goals.obs(mdp, S, env.a_prev, env.xi, env.f, env.command))
            fresh = env.ep_step == 0
            obs_hist = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(obs_hist), torch.cat((obs_hist[:, 1:], o.unsqueeze(1)), 1))
            g = goals.goal(S, env.f, env.ep_step, env.command)
            pi.obs_norm.update(o)
            pi.goal_norm.update(g)
            a = pi(obs_hist, g)
            a = torch.clamp(a + args.action_noise * torch.randn_like(a), -20, 20)
            out = env.step(a)
            buffer.add(out)
            S0, S1 = unflatten_state(out["s"]), unflatten_state(out["s1"])
            wm.in_norm.update(wm.inputs(S0, out["a"], normalize_xi(out["xi"])))
            wm.out_norm.update(wm.targets(S0, S1))
            sim_steps += args.num_envs
            stats["r"].append(out["r"].mean() / reward_scale)
            stats["head_error"].append(out['terms']['head_position_error'].mean())
            stats["head_penalty"].append(out['terms']['head_penalty'].mean() / reward_scale)
            if mdp.is_foot and args.check_foot_gradients:
                restored = goals.goal(S0,out['f'],out['ep_step'],out['command'])
                recomputed = mdp.reward(S1,out['f']+1,out['a'],out['a_prev'],
                                        command=out['command'],ep_step=out['ep_step']+1)
                goal_error = (g-restored).abs().max()
                reward_error = (out['r']-recomputed).abs().max()
                if goal_error > 1e-5 or reward_error > 1e-5:
                    raise RuntimeError('Real/replay foot command or reward timing mismatch')
                stats['goal_sync'].append(goal_error)
                stats['reward_sync'].append(reward_error)
            if "ep_len" in out:
                stats["ep_len"].append(out["ep_len"].mean())
                stats["ep_ret"].append(out["ep_ret_norm"].mean())
                stats["term"].append(out["term"].float().sum())
                stats["done"].append(out["done"].float().sum())

    stats = {"r": [], "head_error": [], "head_penalty": [], "ep_len": [], "ep_ret": [], "term": [], "done": [], 'goal_sync': [], 'reward_sync': []}
    # Replay is deliberately not checkpointed. Refill it on resume as well as on
    # the first run; otherwise sampling the fresh empty buffer fails immediately.
    collect(args.warmup_steps, stats)
    log_f = open(os.path.join(args.log_dir, "train_log.csv"), "a")
    if log_f.tell() == 0:
        log_f.write("iteration,wall_h,sim_steps,wm_loss,wm_err8,pi_loss,v_loss,imag_rew,pi_gradnorm,imag_survival,sim_rew,ep_len,ep_ret_norm,term_frac,it_per_s,lr,head_rms_deg,head_penalty_norm,foot_actor_gradnorm,goal_sync_max_error,reward_sync_max_error,completed_episodes\n")
    t_start = time.time() - wall_offset
    t_log = time.time()
    step_acc = 0.0
    agg = {"wm": [], "wm8": [], "pi": [], "v": [], "ir": [], "gn": [], "surv": [], 'fgn': []}
    for it in range(start_it, args.iterations):
        lr = lr_at(it)
        for o in (opt_wm, opt_pi, opt_v):
            for gparam in o.param_groups:
                gparam["lr"] = lr
        step_acc += 1.0 / args.utd
        n = int(step_acc)
        step_acc -= n
        prof = PROFILE and it % 20 == 0 and it > args.wm_only_iters + 5
        if prof:
            torch.cuda.synchronize(); tp0 = time.time()
        collect(n, stats)
        if prof:
            torch.cuda.synchronize(); tp1 = time.time()
        for _ in range(args.wm_batches):
            l, e8 = wm_update()
        if prof:
            torch.cuda.synchronize(); tp2 = time.time()
        agg["wm"].append(l), agg["wm8"].append(e8)
        if it >= args.wm_only_iters:
            pl, vl, ir, gn, surv, fgn = policy_update()
            if prof:
                torch.cuda.synchronize(); tp3 = time.time()
                print(f"[prof] collect {1e3*(tp1-tp0):.0f} ms ({n} steps) | wm {1e3*(tp2-tp1):.0f} ms | policy+critic {1e3*(tp3-tp2):.0f} ms", flush=True)
            agg["pi"].append(pl), agg["v"].append(vl), agg["ir"].append(ir), agg["gn"].append(gn), agg["surv"].append(surv)
            agg['fgn'].append(fgn)
        if (it + 1) % args.log_every == 0:
            mean = lambda x: float(sum(float(v) for v in x) / max(len(x), 1))  # noqa: E731
            ips = args.log_every / (time.time() - t_log)
            t_log = time.time()
            wall_h = (time.time() - t_start) / 3600
            term_frac = float(sum(float(v) for v in stats["term"]) / max(sum(float(v) for v in stats["done"]), 1))
            head_rms_deg = math.sqrt(mean(stats['head_error']) / 2) * 180 / math.pi
            row = [it + 1, wall_h, sim_steps, mean(agg["wm"]), mean(agg["wm8"]), mean(agg["pi"]), mean(agg["v"]), mean(agg["ir"]),
                   mean(agg["gn"]), mean(agg["surv"]), mean(stats["r"]), mean(stats["ep_len"]), mean(stats["ep_ret"]), term_frac, ips, lr,
                   head_rms_deg, mean(stats['head_penalty']), mean(agg['fgn']),
                   max((float(x) for x in stats['goal_sync']),default=0.),
                   max((float(x) for x in stats['reward_sync']),default=0.),
                   sum(float(x) for x in stats['done'])]  # fmt: skip
            log_f.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row) + "\n")
            log_f.flush()
            print(f"it {it + 1} | {wall_h:.3f} h | wm {row[3]:.4f} e8 {row[4]:.4f} | pi {row[5]:.3f} v {row[6]:.4f} imag_r {row[7]:.3f} surv {row[9]:.3f} "
                  f"| sim_r {row[10]:.3f} ep_len {row[11]:.1f} ep_ret {row[12]:.3f} term {term_frac:.2f} | head {head_rms_deg:.1f}deg | actor_grad {row[8]:.3g} foot_grad {row[18]:.3g} | {ips:.2f} it/s", flush=True)  # fmt: skip
            stats = {k: [] for k in stats}
            agg = {k: [] for k in agg}
        if (it + 1) % args.ckpt_every == 0 or it + 1 == args.iterations:
            ck = {"robot_contract": checkpoint_contract(), "goal_contract": goal_spec.contract(), "wm": wm.state_dict(), "pi": pi.state_dict(), "critic": critic.state_dict(), "opt_wm": opt_wm.state_dict(),
                  "opt_pi": opt_pi.state_dict(), "opt_v": opt_v.state_dict(), "iteration": it + 1,
                  "wall_time": time.time() - t_start, "sim_steps": sim_steps, "args": vars(args),
                  "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
            ck['task_contract'] = mdp.contract() if mdp.is_foot else None
            last = os.path.join(args.log_dir, "last.pt")
            torch.save(ck, last + ".tmp")
            os.replace(last + ".tmp", last)
            small = {k: ck[k] for k in ("robot_contract", "goal_contract", "task_contract", "pi", "wm", "iteration", "wall_time", "sim_steps", "args", "saved_at")}
            path = os.path.join(args.log_dir, f"ckpt_{it + 1:06d}.pt")
            torch.save(small, path + ".tmp")
            os.replace(path + ".tmp", path)
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            runtime_stats = {
                "iteration": it + 1,
                "gpu": torch.cuda.get_device_name(),
                "torch_peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
                "torch_peak_reserved_MiB": torch.cuda.max_memory_reserved() / 2**20,
                "device_used_at_checkpoint_MiB": (total_bytes - free_bytes) / 2**20,
                "device_total_MiB": total_bytes / 2**20,
                "note": "Torch peaks exclude Warp; device usage is a checkpoint snapshot, not a peak.",
            }
            with open(os.path.join(args.log_dir, "runtime_stats.json"), "w") as f:
                json.dump(runtime_stats, f, indent=2)
