"""SuperMimic training (Algorithm A.1): collect -> fit world model -> BPTT policy + critic update."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
import sys
import time
import signal
import subprocess
import random
import numpy as np

PROCESS_START = time.monotonic()
STOP_REQUESTED = False
def request_stop(*_):
    global STOP_REQUESTED
    STOP_REQUESTED = True

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
parser.add_argument('--gradient_audit_every',type=int,default=0,help='Periodically split reward/bootstrap Actor gradients; 0 disables')
parser.add_argument('--actor_lr', type=float, default=None, help='Actor LR; default inherits --lr')
parser.add_argument('--critic_real_batch',type=int,default=0,help='Optional real physical-termination return anchors; 0 disables')
parser.add_argument('--critic_real_window',type=int,default=512)
parser.add_argument('--critic_real_recent_steps',type=int,default=512)
parser.add_argument('--critic_real_weight',type=float,default=0.25)
parser.add_argument('--critic_lr', type=float, default=None, help='Critic LR; default inherits --lr')
parser.add_argument('--recovery_wm_iters', type=int, default=20, help='WM-only updates after replay refill on resume')
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
parser.add_argument('--keep_checkpoints',type=int,default=None,help='Optional maximum number of periodic ckpt_*.pt files; named checkpoints retained')
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
parser.add_argument('--curriculum_config', help='Enable the bounded curriculum (JSON)')
parser.add_argument('--max_device_memory_mib',type=float,default=None,help='Optional device-memory stop threshold at telemetry intervals')
parser.add_argument('--fresh_start', action='store_true', help='Assert fresh models/progress; forbid resume and transfer')
parser.add_argument('--init_weights', help='Explicit transfer of compatible WM/Actor weights; fresh optimizers/critic')
parser.add_argument('--init_from_motion', help='Cross-task Stage 1 initialization; never a resume')
parser.add_argument('--init_from_bridge', help='Initialize Quintic Foot RL from an optional paired Motion-reward bridge')
parser.add_argument('--init_stage2_wm',help='D/F: override Bridge WM with common Stage 1 WM for fair Stage 2 comparison')
parser.add_argument('--transfer_components', choices=['wm','wm_decoder'],default='wm_decoder')
parser.add_argument('--obs_adapter', choices=['identity','mask_reference_orientation'],default='identity')
parser.add_argument('--freeze_decoder',action='store_true')
parser.add_argument('--train_decoder',action='store_true',help='Explicitly finetune D during direct Foot RL; bridge always freezes D')
parser.add_argument('--reference_manifest',help='Hashed clip-level train/held-out manifest')
parser.add_argument('--manifest_split',choices=['train','heldout'],default='train')
parser.add_argument('--prior_config',help='Optional upper-body prior config; default off')
parser.add_argument('--wall_seconds',type=float,help='Safe wall-clock stop, including initialization; finite bounded experiments')
parser.add_argument('--wm_calibration',action='store_true',help='WM-only paired control: freeze Actor normalization and save initial models')
parser.add_argument('--reward_term_logging',action='store_true',help='Persist per-term mean/std and weighted prior contributions locally')
parser.add_argument('--paired_latent_weight',type=float,default=0.,help='Optional same-clip latent alignment; never applies to random Quintic goals')
add_goal_args(parser)
add_launcher_args(parser)
args = parser.parse_args()
from supermimic.config_paths import resolve_config_path
for config_key in ('foot_task_config','foot_config','prior_config','curriculum_config'):
    config_value=getattr(args,config_key,None)
    if isinstance(config_value,str):setattr(args,config_key,str(resolve_config_path(config_value)))
try:
    goal_spec, foot_config = resolve_goal_config(args)
    if args.goal_mode != 'foot_trajectory' and (args.foot_task_config or args.check_foot_gradients):
        raise ValueError('Foot task configuration/gradient checks require --goal_mode foot_trajectory')
    task_cfg = resolve_foot_task(args.foot_task_config) if args.goal_mode == 'foot_trajectory' else None
    if args.goal_mode=='reference_foot' and (args.foot_task_config or args.curriculum_config):
        raise ValueError('Paired bridge uses Motion reward/reset and its own reference-foot goal')
    if args.init_from_motion and args.goal_mode=='motion': raise ValueError('init_from_motion is a cross-task Foot initializer')
    if args.init_from_bridge and args.goal_mode!='foot_trajectory': raise ValueError('Bridge initialization targets Quintic Foot RL')
    if args.init_stage2_wm and not args.init_from_bridge:raise ValueError('Stage 2 WM override requires explicit Bridge initialization')
    if sum(bool(x) for x in (args.resume,args.init_weights,args.init_from_motion,args.init_from_bridge))>1:
        raise ValueError('Choose exactly one resume/initialization source')
    if args.freeze_decoder and args.train_decoder: raise ValueError('Conflicting Decoder update flags')
    if args.goal_mode=='reference_foot' and args.train_decoder: raise ValueError('Sequential bridge freezes Decoder; joint Motion training is not enabled')
    if args.init_from_motion and args.transfer_components=='wm_decoder': args.obs_adapter='mask_reference_orientation'
    if args.init_from_motion and args.transfer_components=='wm_decoder' and not args.train_decoder: args.freeze_decoder=True
    if args.init_from_bridge:
        args.obs_adapter='mask_reference_orientation'
        if not args.train_decoder: args.freeze_decoder=True
    if args.goal_mode=='reference_foot':
        if not (args.init_from_motion or args.resume): raise ValueError('Paired bridge requires Stage 1 initialization or same-task resume')
        if args.goal_frames!=10: raise ValueError('The paired Motion encoder contract currently requires K=10')
        args.freeze_decoder=True;args.obs_adapter='mask_reference_orientation'
    if not math.isfinite(args.paired_latent_weight) or args.paired_latent_weight<0 or (args.paired_latent_weight and args.goal_mode!='reference_foot'):
        raise ValueError('Latent alignment requires paired reference data and a nonnegative weight')
    if args.wall_seconds is not None and (not math.isfinite(args.wall_seconds) or args.wall_seconds<=0): raise ValueError('Invalid wall_seconds')
    if args.keep_checkpoints is not None and args.keep_checkpoints<1:raise ValueError('keep_checkpoints must be positive')
    if args.wm_calibration and args.wm_only_iters<args.iterations:
        raise ValueError('WM calibration must not update Actor/Critic')
    if args.curriculum_config and (task_cfg is None or task_cfg.command_mode != 'curriculum'):
        raise ValueError('Curriculum requires command_mode=curriculum foot task')
    if task_cfg is not None and task_cfg.command_mode == 'curriculum' and not args.curriculum_config:
        raise ValueError('Curriculum task requires --curriculum_config')
    if args.lr<=0 or any(v is not None and v<=0 for v in (args.actor_lr,args.critic_lr)):
        raise ValueError('Learning rates must be positive')
    if args.critic_real_batch<0 or args.critic_real_window<1 or args.critic_real_recent_steps<1 or args.critic_real_weight<0:
        raise ValueError('Invalid real Critic anchor configuration')
    if args.critic_real_batch and task_cfg is None: raise ValueError('Real value anchors currently require the foot task')
    if args.gradient_audit_every<0: raise ValueError('gradient_audit_every must be nonnegative')
    if args.recovery_wm_iters<0: raise ValueError('recovery_wm_iters must be nonnegative')
    if not 0<=args.gamma<1: raise ValueError('gamma must be in [0,1)')
    if args.max_device_memory_mib is not None and args.max_device_memory_mib<=0:
        raise ValueError('max_device_memory_mib must be positive')
    if args.fresh_start and (args.resume or args.init_weights or args.init_from_motion or args.init_from_bridge):
        raise ValueError('--fresh_start cannot load --resume or --init_weights')
    if args.resume and args.init_weights: raise ValueError('Choose resume OR init_weights')
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
from supermimic.motion_prior_data import load_manifest
motion_files = []
data_contract=None
if task_cfg is None:
    try:
        if args.reference_manifest:
            motion_files,data_contract=load_manifest(args.reference_manifest,args.manifest_split)
        else: motion_files = select_reference_files(args.motion_dir, args.clips)
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

from supermimic.control import ACTION_LIMIT, bound_action, bound_bootstrap, return_upper_bound
from supermimic.buffer import ReplayBuffer
from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import S54Kinematics
from supermimic.mdp import CRITIC_EXTRA, INV_DIM, MDP, OBS_DIM, RewardCfg, invariant_features, normalize_xi, unflatten_state
from supermimic.motion import MotionLib
import supermimic.networks as networks
from supermimic.networks import Critic, Policy, WorldModel
from supermimic.transfer import (physical_contract,policy_contract,assert_policy_checkpoint,initialize_from_motion)
from supermimic.s54 import (default_joint_pos_list, NUM_J, checkpoint_contract, assert_checkpoint)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(args.seed)
random.seed(args.seed); np.random.seed(args.seed)
from supermimic.curriculum import Curriculum, TASKS
from supermimic.curriculum_runtime import atomic_save, rng_state, restore_rng, emit, evaluate
curriculum = Curriculum(json.load(open(args.curriculum_config))) if args.curriculum_config else None
signal.signal(signal.SIGTERM, request_stop)
signal.signal(signal.SIGINT, request_stop)
networks.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
dev = "cuda:0"
args.reward_config = asdict(RewardCfg() if task_cfg is None else task_cfg)
os.makedirs(args.log_dir, exist_ok=True)


kin = S54Kinematics(dev)
motion = MotionLib(motion_files, kin, dev) if task_cfg is None else None
default_q = torch.tensor(default_joint_pos_list(), device=dev)
goals = GoalSource(motion, kin, goal_spec, foot_config,
                   command_mode='alternating' if task_cfg is None else task_cfg.command_mode)
mdp = (MDP(kin, motion, default_q) if task_cfg is None else FootTrackingMDP(kin, default_q, goals, task_cfg))
if args.goal_mode=='reference_foot':
    from supermimic.reference_foot import ReferenceFootGoalSource,PairedMotionMDP
    goals=ReferenceFootGoalSource(motion,kin,goal_spec,foot_config)
    mdp=PairedMotionMDP(kin,motion,default_q,goals,RewardCfg())
if args.prior_config:
    if task_cfg is None: raise ValueError('Dynamic prior adapter currently requires independent Foot RL')
    from supermimic.upper_prior import PriorFootMDP,resolve_prior
    prior_cfg=resolve_prior(args.prior_config)
    args.prior_config=asdict(prior_cfg)
    mdp=PriorFootMDP(kin,default_q,goals,task_cfg,prior_cfg)
if curriculum:
    goals.curriculum = curriculum
    goals.command_duration = task_cfg.episode_steps * .02 + goal_spec.frames * goal_spec.dt + 1
reward_scale = mdp.max_step_reward
value_upper = return_upper_bound(reward_scale,args.gamma) if mdp.is_foot else None
imagination_contract=dict(version=1,action_clip=ACTION_LIMIT,bootstrap_upper_bound=value_upper)
actor_lr=args.lr if args.actor_lr is None else args.actor_lr
critic_lr=args.lr if args.critic_lr is None else args.critic_lr
K, Hh, HZ = args.goal_frames, args.hist, args.horizon
GOAL_DIM = goal_spec.dim
CRITIC_IN = mdp.critic_dim + GOAL_DIM

wm = WorldModel().to(dev)
if args.goal_mode=='reference_foot':
    from supermimic.networks import DualEncoderPolicy
    pi=DualEncoderPolicy(hist=Hh).to(dev)
else:
    pi = Policy(OBS_DIM, GOAL_DIM, hist=Hh,obs_adapter=args.obs_adapter).to(dev)
critic = Critic(CRITIC_IN).to(dev)
opt_wm = torch.optim.Adam(wm.parameters(), lr=args.lr)
opt_pi = torch.optim.Adam(pi.parameters(), lr=actor_lr, betas=(0.9, 0.95))
opt_v = torch.optim.Adam(critic.parameters(), lr=critic_lr)
start_it, wall_offset, sim_steps = 0, 0.0, 0
supervision_steps=0
actor_updates=0
imagined_motion_reward_steps=0
imagined_foot_reward_steps=0
physics_metadata=physical_contract()
transfer_report=None
git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
launch_metadata=json.loads((log_path/'launch.json').read_text()) if (log_path/'launch.json').exists() else {}
source_snapshot=launch_metadata.get('source_snapshot')
if args.resume:
    ck = torch.load(args.resume, map_location=dev, weights_only=False)
    assert_checkpoint(ck)
    if ck.get('imagination_contract') not in (None,imagination_contract):
        raise ValueError('Imagination action/value contract mismatch')
    if ck.get('imagination_contract') is None:
        print('Legacy checkpoint: applying real/imagined action consistency and valid foot return bound; '
              'network dimensions and physical task unchanged',flush=True)
    goal_spec.assert_checkpoint(ck, foot_config)
    assert_task_checkpoint(ck, mdp)
    assert_policy_checkpoint(ck,Hh,args.obs_adapter,not args.freeze_decoder)
    if ck.get('physical_contract',physics_metadata)!=physics_metadata: raise ValueError('Resume physics contract mismatch')
    if ck.get('data_contract',data_contract)!=data_contract: raise ValueError('Resume data manifest mismatch')
    if "critic" not in ck: raise ValueError("Resume requires last.pt with critic and optimizers")
    wm.load_state_dict(ck["wm"]), pi.load_state_dict(ck["pi"]), critic.load_state_dict(ck["critic"])
    opt_wm.load_state_dict(ck["opt_wm"]), opt_pi.load_state_dict(ck["opt_pi"]), opt_v.load_state_dict(ck["opt_v"])
    transfer_report=ck.get('transfer_report')
    supervision_steps=ck.get('stage_stats',{}).get('supervision_steps',0)
    actor_updates=ck.get('stage_stats',{}).get('actor_updates',0)
    imagined_motion_reward_steps=ck.get('stage_stats',{}).get('imagined_motion_reward_steps',0)
    imagined_foot_reward_steps=ck.get('stage_stats',{}).get('imagined_foot_reward_steps',0)
    if curriculum:
        if 'curriculum' not in ck: raise ValueError('Full curriculum resume requires curriculum/RNG state')
        curriculum.load_state_dict(ck['curriculum'])
    start_it, wall_offset, sim_steps = ck["iteration"], ck["wall_time"], ck["sim_steps"]
    if args.iterations <= start_it:
        raise ValueError(f'Resume checkpoint is at iteration {start_it}; --iterations must be larger '
                         f'(it is the total target, not additional iterations; got {args.iterations})')
    print(f'Resuming iteration {start_it} -> {args.iterations}; model and optimizer states restored. '
          f'Replay buffer will be refilled; WM-only phase is based on the absolute iteration.', flush=True)

if args.init_weights:
    transfer = torch.load(args.init_weights, map_location=dev, weights_only=False)
    assert_checkpoint(transfer)
    goal_spec.assert_checkpoint(transfer)  # representation must match; sampling/task may change explicitly
    if transfer['args']['hist'] != args.hist: raise ValueError('Transfer requires matching Actor history')
    if transfer.get('policy_contract',{}).get('obs_adapter','identity')!=args.obs_adapter:
        raise ValueError('Compatible Actor transfer requires matching observation semantics')
    if transfer.get('physical_contract',physics_metadata)!=physics_metadata:raise ValueError('Actor transfer physics mismatch')
    wm.load_state_dict(transfer['wm']); pi.load_state_dict(transfer['pi'])
    pi.goal_norm.mean.zero_(); pi.goal_norm.var.fill_(1); pi.goal_norm.count.zero_()
    print('TRANSFER: WM+Actor (goal normalization reset); fresh Critic/optimizers/replay/curriculum. Source iteration',
          transfer['iteration'], flush=True)

if args.init_from_motion:
    transfer=torch.load(args.init_from_motion,map_location=dev,weights_only=False)
    transfer_report=initialize_from_motion(transfer,wm,pi,hist=Hh,components=args.transfer_components)
    transfer_report['source_path']=str(Path(args.init_from_motion).resolve())
if args.init_from_bridge:
    from supermimic.transfer import initialize_from_bridge
    transfer=torch.load(args.init_from_bridge,map_location=dev,weights_only=False)
    transfer_report=initialize_from_bridge(transfer,wm,pi,hist=Hh)
    transfer_report['source_path']=str(Path(args.init_from_bridge).resolve())
    if args.init_stage2_wm:
        common=torch.load(args.init_stage2_wm,map_location=dev,weights_only=False)
        common_report=initialize_from_motion(common,wm,pi,hist=Hh,components='wm')
        transfer_report['stage2_wm_override']=dict(source_path=str(Path(args.init_stage2_wm).resolve()),
            source_iteration=common['iteration'],reason='Common Stage1 WM across B/C/D/F/G; discard Bridge-adapted WM')
        transfer_report['mapping']=[row for row in transfer_report['mapping'] if not str(row.get('target','')).startswith('wm.')]+[
            dict(row,reason='Common Stage1 WM override after Bridge initialization') for row in common_report['mapping'] if str(row.get('target','')).startswith('wm.')]
if transfer_report:
    (log_path/'transfer_report.json').write_text(json.dumps(transfer_report,indent=2))
    print('Cross-task initialization:',json.dumps({k:v for k,v in transfer_report.items() if k!='mapping'}),flush=True)
if args.freeze_decoder:
    for parameter in pi.dec.parameters(): parameter.requires_grad_(False)
if args.goal_mode=='reference_foot':
    for parameter in pi.motion_enc.parameters():parameter.requires_grad_(False)
actor_parameters=tuple(p for p in pi.parameters() if p.requires_grad)

args_temp=log_path/'args.json.tmp'
args_temp.write_text(json.dumps(vars(args),indent=2))
os.replace(args_temp,log_path/'args.json')


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
    raw_a = pi(o_hist, g)
    a = bound_action(raw_a)
    S1 = wm.step(S, a, xi_n)
    f1 = f + 1
    step1 = ep_step + 1
    o1 = mdp.add_noise(goals.obs(mdp, S1, a, xi, f1, command))
    o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
    return S1, a, o_hist, f1, goals.goal(S1, f1, step1, command), step1, (raw_a.abs()>ACTION_LIMIT).float().mean()


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


def policy_update(audit_gradients=False):
    global supervision_steps
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
    wm.zero_grad(set_to_none=True)  # Actor gradients may traverse WM inputs, never its parameters.
    set_requires_grad(wm, False)
    set_requires_grad(critic, False)
    B = args.pi_batch
    audit_freeze=args.check_foot_gradients or audit_gradients or args.goal_mode=='reference_foot'
    wm_versions=tuple(p._version for p in wm.parameters()) if audit_freeze else None
    decoder_versions=tuple(p._version for p in pi.dec.parameters()) if audit_freeze and args.freeze_decoder else None
    g0 = goals.goal(S, f, ep_step, command)
    g = g0
    S0 = S
    Ss, As, Aps, Fs, Gs, Steps = [], [], [], [], [], []
    clip_fractions=[]
    for t in range(HZ):
        Aps.append(a_prev)
        S, a_prev, o_hist, f, g, ep_step, clip_fraction = imag_step(S, o_hist, a_prev, f, xi, xi_n, g, ep_step, command)
        Ss.append(S), As.append(a_prev), Fs.append(f), Gs.append(g)
        Steps.append(ep_step); clip_fractions.append(clip_fraction)
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
    reward_J = J
    raw_v = torch.zeros_like(J); bootstrap = torch.zeros_like(J)
    if not args.no_critic:
        raw_v = critic(_critic_input(S, f, xi_n, g, ep_step, command))
        bootstrap = (args.gamma**HZ) * c * bound_bootstrap(raw_v,value_upper)
        J = J + bootstrap
    loss = -J.mean()
    latent_loss=torch.zeros((),device=dev)
    if args.paired_latent_weight:
        with torch.no_grad(): target_latent=pi.encode_motion(motion.goal(f_all,K))
        latent_loss=(pi.encode(flat(Gs))-target_latent).square().mean()
        loss=loss+args.paired_latent_weight*latent_loss
        supervision_steps+=HZ*B
    foot_gn = torch.zeros((), device=dev)
    if mdp.is_foot and (args.check_foot_gradients or audit_gradients):
        # Exclude direct action costs, posture, base terms, critic and survival derivatives.
        foot_loss = -(disc*c_t.detach()*foot_reward.view(HZ,B)).sum(0).mean()
        fg = torch.autograd.grad(foot_loss,actor_parameters,retain_graph=True,allow_unused=True)
        foot_gn = torch.stack([g.detach().float().square().sum() for g in fg if g is not None]).sum().sqrt()
        if not torch.isfinite(foot_gn): raise FloatingPointError('Nonfinite pure foot Actor gradient')
    gradient_diagnostics={}
    if audit_gradients:
        params=actor_parameters
        rg=torch.autograd.grad(-reward_J.mean(),params,retain_graph=True,allow_unused=True)
        bg=(torch.autograd.grad(-bootstrap.mean(),params,retain_graph=True,allow_unused=True)
            if bootstrap.requires_grad else tuple(None for _ in params))
        def grad_norm(gs):
            return sum((g.detach().float().square().sum() for g in gs if g is not None),torch.zeros((),device=dev)).sqrt()
        rn,bn=grad_norm(rg),grad_norm(bg)
        dot=sum(((x.detach().float()*y.detach().float()).sum() for x,y in zip(rg,bg) if x is not None and y is not None),torch.zeros((),device=dev))
        gradient_diagnostics=dict(actor_reward_gradnorm=rn,actor_bootstrap_gradnorm=bn,
            actor_grad_cosine=dot/(rn*bn).clamp_min(1e-20),pure_foot_gradnorm=foot_gn)
        if mdp.is_foot:
            _,heads=mdp.reward(S_all,f_all,flat(As),flat(Aps),True,command=command_all,ep_step=flat(Steps))
            objectives={k:v for k,v in heads.items() if k.endswith('_contribution')}
            objectives.update(base_height=heads['base_height']*task_cfg.w_base_height*.02,
                              base_upright=heads['base_upright']*task_cfg.w_upright*.02,
                              foot=heads['foot_objective'])
            for key,values in objectives.items():
                objective=-(disc*c_t.detach()*values.view(HZ,B)).sum(0).mean()
                gradients=torch.autograd.grad(objective,params,retain_graph=True,allow_unused=True)
                gradient_diagnostics['term_'+key+'_actor_gradnorm']=grad_norm(gradients)
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
    if audit_freeze:
        if tuple(p._version for p in wm.parameters())!=wm_versions or any(p.grad is not None for p in wm.parameters()):
            raise RuntimeError('Actor update mutated WM parameters/gradients')
        if decoder_versions is not None and (tuple(p._version for p in pi.dec.parameters())!=decoder_versions
                or any(p.grad is not None for p in pi.dec.parameters())):
            raise RuntimeError('Frozen Decoder changed during Encoder adaptation')
    set_requires_grad(wm, True)
    set_requires_grad(critic, True)
    # ---------------- critic: GAE(lambda) targets over the imagined transitions ----------------
    v_loss = torch.zeros(())
    target_mean=torch.zeros((),device=dev); target_max=target_mean
    real_loss_log=[];real_count=0
    if not args.no_critic:
        with torch.no_grad():
            real_X=real_Y=None
            if args.critic_real_batch:
                real=buffer.sample_mc_returns(args.critic_real_batch,args.critic_real_window,args.gamma,args.critic_real_recent_steps)
                real_count=len(real['s'])
                if real_count:
                    real_S=unflatten_state(real['s']);real_c=buffer.commands(real['command_id'])
                    real_g=goals.goal(real_S,real['f'],real['ep_step'],real_c)
                    real_X=_critic_input(real_S,real['f'],normalize_xi(real['xi']),real_g,real['ep_step'],real_c)
                    real_Y=real['mc_return']
            S_c = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0).flatten(0, 1) for i in range(6))
            f_c = torch.cat((s["f"].unsqueeze(0), torch.stack(Fs)), 0).flatten()
            g_c = torch.cat((g0.unsqueeze(0), torch.stack(Gs)), 0).flatten(0, 1)
            xi_c = xi_n.unsqueeze(0).expand(HZ + 1, -1, -1).flatten(0, 1)
            step_c = torch.cat((s['ep_step'].unsqueeze(0),torch.stack(Steps)),0).flatten()
            command_c = command.repeat(HZ+1) if command is not None else None
            X = critic_input_c(S_c, f_c, xi_c, g_c, step_c, command_c).view(HZ + 1, B, -1)
        critic.norm.update(X)
        if real_X is not None: critic.norm.update(real_X)
        with torch.no_grad():
            V = bound_bootstrap(critic(X),value_upper)
            R = V[-1]
            targets = []
            for t in range(HZ - 1, -1, -1):
                R = rews[t] + args.gamma * (1 - dones[t]) * ((1 - args.lam) * V[t + 1] + args.lam * R)
                targets.append(R)
            targets = torch.stack(targets[::-1])  # (H, B)
            target_mean=targets.mean(); target_max=targets.max()
        Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
        n = Xf.shape[0]
        mb = n // args.critic_minibatches
        for _ in range(args.critic_epochs):
            perm = torch.randperm(n, device=dev)
            real_perm=torch.randperm(real_count,device=dev) if real_count else None
            for i in range(args.critic_minibatches):
                idx = perm[i * mb : (i + 1) * mb]
                v_loss = ((critic(Xf[idx]) - Yf[idx]) ** 2).mean()
                total_v_loss=v_loss
                if real_count:
                    ri=real_perm[i::args.critic_minibatches]
                    if len(ri):
                        real_v_loss=((critic(real_X[ri])-real_Y[ri])**2).mean()
                        real_loss_log.append(real_v_loss.detach())
                        total_v_loss=total_v_loss+args.critic_real_weight*real_v_loss
                opt_v.zero_grad(set_to_none=True)
                if not torch.isfinite(total_v_loss): raise FloatingPointError("Nonfinite critic loss")
                total_v_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0, error_if_nonfinite=True)
                opt_v.step()
    if PROFILE:
        torch.cuda.synchronize()
        print(f"[prof] imagine fwd {1e3*(tq1-tq0):.0f} ms | bwd {1e3*(tq2-tq1):.0f} ms | critic {1e3*(time.time()-tq2):.0f} ms", flush=True)
    diagnostics=dict(**gradient_diagnostics,paired_latent_loss=latent_loss.detach(),critic_real_samples=torch.tensor(float(real_count),device=dev),
        **({'critic_real_mc_loss':torch.stack(real_loss_log).mean()} if real_loss_log else {}),critic_raw_value_mean=raw_v.detach().mean(),critic_raw_value_max=raw_v.detach().max(),
        critic_target_mean=target_mean,critic_target_max=target_max,
        actor_bootstrap_mean=bootstrap.detach().mean(),actor_reward_return_mean=reward_J.detach().mean(),
        imagined_action_clip_fraction=torch.stack(clip_fractions).mean().detach())
    return loss.detach(), v_loss.detach(), (r_sum / HZ / reward_scale).mean(), gn.detach(), c.mean().detach(), foot_gn, diagnostics


# ------------------------------------------------------------------------------------------------
sim_cfg = make_sim(dev, args)
with launch_simulation(sim_cfg, args):
    env = TrackingSim(sim_cfg, args.num_envs, motion, mdp, gamma=args.gamma, goals=goals)
    buffer = ReplayBuffer(args.buffer_steps, args.num_envs, dev, foot_commands=goals.is_foot)
    obs_hist = torch.zeros(args.num_envs, Hh, OBS_DIM, device=dev)

    collection_frames=0; collection_seconds=0.
    calibration_coverage=dict(transitions=0,after_1_2s=0,pitch_min_deg=0.,pitch_max_deg=0.,
        max_abs_action=0.,loaded_foot_samples=0,contact_edge_samples=0)
    if args.wm_calibration:
        from supermimic.standing_diagnostics import NewtonGroundContacts
        calibration_contacts=NewtonGroundContacts(env)

    @torch.no_grad()
    def collect(n_steps: int, stats: dict):
        global obs_hist, sim_steps, collection_frames, collection_seconds
        collect_started=time.monotonic()
        for _ in range(n_steps):
            S = env.state()
            o = mdp.add_noise(goals.obs(mdp, S, env.a_prev, env.xi, env.f, env.command))
            fresh = env.ep_step == 0
            obs_hist = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(obs_hist), torch.cat((obs_hist[:, 1:], o.unsqueeze(1)), 1))
            g = goals.goal(S, env.f, env.ep_step, env.command)
            if not args.wm_calibration:
                pi.obs_norm.update(o)
                pi.goal_norm.update(g)
            a = pi(obs_hist, g)
            a = bound_action(a + args.action_noise * torch.randn_like(a))
            out = env.step(a)
            if args.wm_calibration:
                observed=unflatten_state(out['s1'])
                pitch=torch.asin((-observed[1][:,2,0]).clamp(-1,1))*180/math.pi
                counts,forces,_=calibration_contacts.read()
                calibration_coverage['transitions']+=args.num_envs
                calibration_coverage['after_1_2s']+=int((out['ep_step']>=60).sum())
                calibration_coverage['pitch_min_deg']=min(calibration_coverage['pitch_min_deg'],float(pitch.min()))
                calibration_coverage['pitch_max_deg']=max(calibration_coverage['pitch_max_deg'],float(pitch.max()))
                calibration_coverage['max_abs_action']=max(calibration_coverage['max_abs_action'],float(a.abs().max()))
                calibration_coverage['loaded_foot_samples']+=int((forces[:,:,2]>1).sum())
                calibration_coverage['contact_edge_samples']+=int(((counts==0)|(counts<=2)).any(-1).sum())
            if args.goal_mode=='reference_foot' and not torch.equal(out['command'].start_frame+out['ep_step'],out['f']):
                raise RuntimeError('Paired reference frame and immutable command clock diverged')
            if args.reward_term_logging:
                record=dict(sim_steps=sim_steps+args.num_envs,terms={
                    k:dict(mean=float(v.float().mean()),std=float(v.float().std(unbiased=False)))
                    for k,v in out['terms'].items()})
                with (log_path/'reward_terms.jsonl').open('a') as sink: sink.write(json.dumps(record)+'\n')
            if not torch.isfinite(out["s1"]).all(): raise FloatingPointError("Nonfinite Newton state; refusing to poison normalization/replay")
            buffer.add(out)
            if curriculum: curriculum.observe(out['command'],out['ep_step'])
            S0, S1 = unflatten_state(out["s"]), unflatten_state(out["s1"])
            wm.in_norm.update(wm.inputs(S0, out["a"], normalize_xi(out["xi"])))
            wm.out_norm.update(wm.targets(S0, S1))
            sim_steps += args.num_envs
            stats["r"].append(out["r"].mean() / reward_scale)
            stats["head_error"].append(out['terms']['head_position_error'].mean())
            stats["head_penalty"].append(out['terms']['head_penalty'].mean() / reward_scale)
            if goals.is_foot and (args.check_foot_gradients or args.goal_mode=='reference_foot'):
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

        collection_frames+=n_steps*args.num_envs
        collection_seconds+=time.monotonic()-collect_started

    t_start = time.monotonic() - wall_offset - (time.monotonic()-PROCESS_START)
    def elapsed(): return time.monotonic()-t_start
    current_it = start_it
    step_acc = float(ck.get('step_acc',0.)) if args.resume else 0.
    if args.resume and 'rng' in ck: restore_rng(ck['rng'])

    def save_checkpoint(label=None):
        ckpt = {"robot_contract": checkpoint_contract(), "goal_contract": goal_spec.contract(),
                "checkpoint_schema_version":2,
                "task_contract": mdp.contract(),
                "physical_contract":physics_metadata,"policy_contract":policy_contract(Hh,args.obs_adapter,not args.freeze_decoder),
                "data_contract":data_contract,"git_sha":git_sha,"transfer_report":transfer_report,
                "source_snapshot":source_snapshot,
                "stage_stats":dict(stage='bridge' if args.goal_mode=='reference_foot' else 'stage1' if args.goal_mode=='motion' else 'stage2',
                    current_stage_real_steps=sim_steps,upstream_real_steps=transfer_report.get('source_total_real_steps',transfer_report.get('source_sim_steps',0)) if transfer_report else 0,
                    upstream_wall_seconds=transfer_report.get('source_total_wall_seconds',0.) if transfer_report else 0.,
                    supervision_steps=supervision_steps,actor_updates=actor_updates,
                    imagined_motion_reward_steps=imagined_motion_reward_steps,
                    imagined_foot_reward_steps=imagined_foot_reward_steps),
                "wm":wm.state_dict(), "pi":pi.state_dict(), "critic":critic.state_dict(),
                "opt_wm":opt_wm.state_dict(), "opt_pi":opt_pi.state_dict(), "opt_v":opt_v.state_dict(),
                "iteration":current_it, "wall_time":elapsed(), "sim_steps":sim_steps, "args":vars(args),
                "saved_at":time.strftime('%Y-%m-%d %H:%M:%S'), "rng":rng_state(), "step_acc":step_acc,
                "replay_restore":"fresh episode reset + refill; WM-only recovery warmup",
                "curriculum":curriculum.state_dict() if curriculum else None,
                "imagination_contract":imagination_contract}
        finite=all(torch.isfinite(value).all() for model in (wm,pi,critic)
                   for value in model.state_dict().values() if value.is_floating_point())
        if not finite:
            atomic_save(ckpt,log_path/'failure_nonfinite.pt')
            print('Nonfinite model/normalization state saved separately; previous latest preserved',flush=True)
            return
        atomic_save(ckpt, log_path/'last.pt')
        if curriculum:
            # latest is an atomic hard-link alias of the full last.pt.
            temp=log_path/'latest.pt.tmp'
            if temp.exists(): temp.unlink()
            os.link(log_path/'last.pt',temp); os.replace(temp,log_path/'latest.pt')
        if label: atomic_save(ckpt,log_path/(label+'.pt'))
        atomic_save(ckpt,log_path/f'ckpt_{current_it:06d}.pt')
        if curriculum:
            for path in sorted(log_path.glob('ckpt_*.pt'))[:-curriculum.cfg['keep_checkpoints']]: path.unlink()
        elif args.keep_checkpoints is not None:
            for path in sorted(log_path.glob('ckpt_*.pt'))[:-args.keep_checkpoints]:path.unlink()
        free,total=torch.cuda.mem_get_info()
        (log_path/'runtime_stats.json').write_text(json.dumps(dict(iteration=current_it,gpu=torch.cuda.get_device_name(),
            torch_peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
            torch_peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
            device_used_at_checkpoint_MiB=(total-free)/2**20,device_total_MiB=total/2**20,
            wall_seconds=elapsed(),note='Torch peaks exclude Warp; device snapshot includes Warp'),indent=2))
        if args.wm_calibration:
            (log_path/'calibration_coverage.json').write_text(json.dumps(calibration_coverage,indent=2))

    stats = {"r": [], "head_error": [], "head_penalty": [], "ep_len": [], "ep_ret": [], "term": [], "done": [], 'goal_sync': [], 'reward_sync': []}
    # Replay is deliberately not checkpointed. Refill it on resume as well as on
    # the first run; otherwise sampling the fresh empty buffer fails immediately.
    # Fresh replay after resume: refill, then a short WM-only recovery before Actor updates.
    recovery_until = start_it + (args.recovery_wm_iters if args.resume else 0)
    log_f = open(os.path.join(args.log_dir, "train_log.csv"), "a")
    if log_f.tell() == 0:
        log_f.write("iteration,wall_h,sim_steps,wm_loss,wm_err8,pi_loss,v_loss,imag_rew,pi_gradnorm,imag_survival,sim_rew,ep_len,ep_ret_norm,term_frac,it_per_s,lr,head_rms_deg,head_penalty_norm,foot_actor_gradnorm,goal_sync_max_error,reward_sync_max_error,completed_episodes\n")
    t_log = time.time()
    agg = {"wm": [], "wm8": [], "pi": [], "v": [], "ir": [], "gn": [], "surv": [], 'fgn': []}
    diag_agg={}
    try:
        if args.wm_calibration:save_checkpoint('calibration_before')
        collect(args.warmup_steps, stats)
        for it in range(start_it, args.iterations):
            if STOP_REQUESTED or (args.wall_seconds is not None and time.monotonic()-PROCESS_START>=args.wall_seconds) or (curriculum and elapsed() >= curriculum.cfg['budget_seconds']-45):
                print('Safe stop requested / wall-clock budget reserve reached',flush=True); break
            lr = lr_at(it)
            for o,scale in ((opt_wm,1.),(opt_pi,actor_lr/args.lr),(opt_v,critic_lr/args.lr)):
                for gparam in o.param_groups:
                    gparam["lr"] = lr*scale
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
            if it >= max(args.wm_only_iters,recovery_until):
                audit_gradients=args.gradient_audit_every>0 and (it+1)%args.gradient_audit_every==0
                pl, vl, ir, gn, surv, fgn, diagnostics = policy_update(audit_gradients)
                actor_updates+=1
                if mdp.is_foot:imagined_foot_reward_steps+=args.pi_batch*HZ
                else:imagined_motion_reward_steps+=args.pi_batch*HZ
                for key,value in diagnostics.items(): diag_agg.setdefault(key,[]).append(value)
                if prof:
                    torch.cuda.synchronize(); tp3 = time.time()
                    print(f"[prof] collect {1e3*(tp1-tp0):.0f} ms ({n} steps) | wm {1e3*(tp2-tp1):.0f} ms | policy+critic {1e3*(tp3-tp2):.0f} ms", flush=True)
                agg["pi"].append(pl), agg["v"].append(vl), agg["ir"].append(ir), agg["gn"].append(gn), agg["surv"].append(surv)
                if args.check_foot_gradients or audit_gradients: agg['fgn'].append(fgn)
            if (it + 1) % args.log_every == 0:
                mean = lambda x: float(sum(float(v) for v in x) / max(len(x), 1))  # noqa: E731
                ips = args.log_every / (time.time() - t_log)
                t_log = time.time()
                wall_h = elapsed() / 3600
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
                if diag_agg:
                    with (log_path/'gradient_terms.jsonl').open('a') as sink:
                        sink.write(json.dumps(dict(iteration=it+1,**{k:mean(v) for k,v in diag_agg.items()}))+'\n')
                    if not curriculum:
                        emit(log_path,it+1,{
                            **{f'Optimization/{k}':mean(v) for k,v in diag_agg.items()},
                            'Stage/ActorUpdates':actor_updates,
                            'Stage/RealTransitions':sim_steps,
                            'Stage/ImaginedMotionRewardTransitions':imagined_motion_reward_steps,
                            'Stage/ImaginedFootRewardTransitions':imagined_foot_reward_steps,
                            'Performance/CollectionFPS':collection_frames/max(collection_seconds,1e-9),
                            'Performance/PeakTorchMiB':torch.cuda.max_memory_allocated()/2**20})
                        diag_agg={}
                print(f"it {it + 1} | {wall_h:.3f} h | wm {row[3]:.4f} e8 {row[4]:.4f} | pi {row[5]:.3f} v {row[6]:.4f} imag_r {row[7]:.3f} surv {row[9]:.3f} "
                      f"| sim_r {row[10]:.3f} ep_len {row[11]:.1f} ep_ret {row[12]:.3f} term {term_frac:.2f} | head {head_rms_deg:.1f}deg | actor_grad {row[8]:.3g} foot_grad {row[18]:.3g} | {ips:.2f} it/s", flush=True)  # fmt: skip
                stats = {k: [] for k in stats}
                agg = {k: [] for k in agg}
            current_it = it+1
            if curriculum:
                if current_it % args.log_every == 0:
                    gpu_stats={}
                    try:
                        gpu_sample=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used',
                            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=2,check=True)
                        utilization,memory=gpu_sample.stdout.splitlines()[0].split(',')
                        gpu_stats={'Performance/GPUUtilization':float(utilization),'Performance/DeviceUsedMiB':float(memory)}
                        memory_limit=getattr(args,'max_device_memory_mib',None)
                        if memory_limit is not None and float(memory)>=memory_limit:
                            STOP_REQUESTED=True
                            print(f'Device memory {memory} MiB reached configured {memory_limit} MiB; safe save/stop requested',flush=True)
                    except (OSError,ValueError,subprocess.SubprocessError) as exc:
                        print(f'GPU telemetry unavailable: {exc}',flush=True)
                    status_temp=log_path/'training_status.json.tmp'
                    status_temp.write_text(json.dumps(dict(status='running',iteration=current_it,wall_seconds=elapsed(),
                        stage=curriculum.stage,budget_seconds=curriculum.cfg['budget_seconds'])))
                    os.replace(status_temp,log_path/'training_status.json')
                    emit(log_path,current_it,curriculum.scalars() | gpu_stats | {
                        **{f'Optimization/{k}':float(sum(v)/len(v)) for k,v in diag_agg.items()},
                        'Optimization/ActorLR':opt_pi.param_groups[0]['lr'],
                        'Optimization/CriticLR':opt_v.param_groups[0]['lr'],
                        'Performance/PeakTorchMiB':torch.cuda.max_memory_allocated()/2**20,
                        'Performance/CollectionFPS':collection_frames/max(collection_seconds,1e-9)})
                    diag_agg={}
                remaining=curriculum.cfg['budget_seconds']-elapsed()
                final_due=120 < remaining < 300 and not getattr(curriculum,'final_evaluated',False)
                due=elapsed()-curriculum.last_eval>=curriculum.cfg['eval_seconds'] or final_due
                available=elapsed()*curriculum.cfg['eval_fraction']-curriculum.eval_spent
                pending_full=curriculum.passes>=curriculum.cfg['consecutive_passes']-1
                full_reserve=[45,90,150,210,270,270][curriculum.stage]
                full=(pending_full and remaining>600 and available>=full_reserve) or (final_due and available>=120)
                evaluation_minimum=getattr(curriculum,'eval_min_seconds',10.)
                allowance=min(300. if full else max(90.,evaluation_minimum),available,max(0.,remaining-60))
                if pending_full and not full: due=False  # accumulate the 5% quota for a complete stage gate
                if due and current_it % args.log_every == 0 and allowance>=evaluation_minimum and remaining>120 and not STOP_REQUESTED:
                    # Full evaluation only after a successful fast gate, at most on stage transitions.
                    report=evaluate(env,pi,goals,mdp,curriculum,Hh,full,
                                    deadline=time.monotonic()+allowance,rich=final_due,critic=critic,wm=wm)
                    obs_hist.zero_()
                    if final_due and full: curriculum.final_evaluated=True
                    curriculum.eval_spent+=report['wall_seconds']; curriculum.last_eval=elapsed()
                    stage_before=curriculum.stage
                    complete=not any(r['status'] in ('budget_skipped','incomplete') or r.get('incomplete_rate',0)>0 for r in report['rows'])
                    curriculum.eval_min_seconds=report['minimum_next_eval_seconds']
                    changed=curriculum.update(report['metrics'],elapsed(),allow_promotion=full,bucket_metrics=report['bucket_metrics'],wm_metrics=report.get('wm_metrics')) if complete else False
                    (log_path/f'eval_{current_it:06d}.json').write_text(json.dumps(report,indent=2))
                    scalars={f'Evaluation/{task}/{key}':value for task,m in report['metrics'].items() for key,value in m.items()}
                    for i,row in enumerate(report['rows']):
                        if row['status']=='evaluated':
                            scalars.update({f'Evaluation/Command{i}/{k}':v for k,v in row.items() if isinstance(v,(int,float))})
                    emit(log_path,current_it,scalars | curriculum.scalars())
                    for name,tasks in [('standing',['standing']),('walking',['short','medium','long']),
                                       ('overall',[t for t,w in zip(TASKS,
                                                                 curriculum.cfg['weights'][stage_before]) if w>0])]:
                        if not complete or not all(t in report['metrics'] for t in tasks): continue
                        # Score is actual success first, then tracking error. No WM-loss selection.
                        score=min(report['metrics'][t]['success_rate'] for t in tasks)-.1*max(report['metrics'][t]['foot_rmse'] for t in tasks)
                        if name=='overall':
                            if report['metrics'].get('standing',{}).get('success_rate',0)<curriculum.cfg['standing_threshold']: continue
                            score += stage_before
                        if score>curriculum.best.get(name,-float('inf')):
                            curriculum.best[name]=score; save_checkpoint('best_'+name)
                    if curriculum.failures>=3:
                        print(f'Curriculum C{curriculum.stage} stalled for {curriculum.failures} complete evaluations; '
                              'difficulty reduced; no forced promotion',flush=True)
                    if changed:
                        save_checkpoint(f'stage_C{stage_before}_end')
                        print(f'Curriculum transition C{stage_before} -> C{curriculum.stage}',flush=True)
                if elapsed()-curriculum.last_save>=curriculum.cfg['checkpoint_seconds']:
                    curriculum.last_save=elapsed(); save_checkpoint()
            if current_it == args.wm_only_iters:
                label='wm_warmup_complete' if args.goal_mode=='motion' else 'wm_calibrated'
                save_checkpoint(label)
                print(f'WM-only phase complete at iteration {current_it}; saved {label}.pt; '
                      'Actor/Critic updates begin next iteration; prediction accuracy requires independent evaluation',flush=True)
            elif current_it % args.ckpt_every == 0 or current_it == args.iterations:
                save_checkpoint()
        if curriculum and not STOP_REQUESTED:
            remaining=curriculum.cfg['budget_seconds']-elapsed()
            allowance=min(180.,elapsed()*curriculum.cfg['eval_fraction']-curriculum.eval_spent)
            if remaining>allowance+60 and allowance>=35:
                final_report=evaluate(env,pi,goals,mdp,curriculum,Hh,full=True,
                                      deadline=time.monotonic()+allowance,critic=critic,wm=wm)
                curriculum.eval_spent+=final_report['wall_seconds']
                (log_path/'eval_final.json').write_text(json.dumps(final_report,indent=2))
    finally:
        # Signals set a flag; exceptions also leave the last complete optimizer state.
        save_checkpoint()
        log_f.close()
        final_status='error' if sys.exc_info()[0] is not None else 'stopped' if STOP_REQUESTED or current_it<args.iterations else 'completed'
        (log_path/'training_status.json').write_text(json.dumps(dict(status=final_status,iteration=current_it,wall_seconds=elapsed(),
            stop_requested=STOP_REQUESTED,budget_seconds=curriculum.cfg['budget_seconds'] if curriculum else None)))
