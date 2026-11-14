"""Checkpoint evaluation (paper Appendix F.1) on all motions of the library.

Each motion plays once from its first frame with friction 1.0, no domain randomization, no
observation or action noise. A motion fails if the rollout triggers the training termination.
Normalized return is the undiscounted reward sum divided by its maximum attainable value.

Run as a watcher: ``python supermimic/evaluate.py --run_dir logs/<run> --watch``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", type=str, required=True)
parser.add_argument("--watch", action="store_true")
parser.add_argument("--ckpt", type=str, default=None, help="evaluate one checkpoint")
parser.add_argument("--record", type=str, default=None, help="save trajectories (npz) for rendering")
parser.add_argument('--out_json', help='Explicit single-checkpoint result path (preserve historical eval JSON)')
parser.add_argument("--motion_dir", type=str, default=None, help="defaults to this run's saved reference directory")
parser.add_argument("--clips", nargs="+", help="defaults to the clips recorded in args.json")
parser.add_argument("--every", type=int, default=1, help="evaluate every n-th checkpoint")
parser.add_argument('--num_envs',type=int,default=4,help='Independent foot evaluation episodes; Motion still uses one env/clip')
parser.add_argument('--duration_s', type=float, help='Foot evaluation time limit, independent of checkpoint task contract')
parser.add_argument('--seeds', nargs='+', type=int, help='Foot evaluation seeds; use --reset_noise to test distinct initial states')
parser.add_argument('--reset_noise', action='store_true', help='Foot eval: configured reset perturbations only; no observation/action noise')
parser.add_argument('--diagnostics', action='store_true', help='Foot records: actual Newton ground contacts and forces')
parser.add_argument('--controller', choices=['policy', 'default_pd'], default='policy')
parser.add_argument('--reference_manifest',help='Optional hashed train/held-out reference manifest')
parser.add_argument('--manifest_split',choices=['train','heldout'])
add_launcher_args(parser)
args = parser.parse_args()
saved_config = json.loads(open(os.path.join(args.run_dir, 'args.json')).read())
args.motion_dir = args.motion_dir or saved_config.get('motion_dir') or os.path.join(REPO, 'library/motion_reference/kuavo_s54/retarget_data/reference_startup_fixed')
from supermimic.reference_selection import select_reference_files
foot_run = saved_config.get('goal_mode','motion') == 'foot_trajectory'
if not foot_run and (args.duration_s is not None or args.seeds or args.reset_noise or args.diagnostics or args.controller != 'policy'):
    parser.error('Duration, seed and controller protocol options currently apply to Foot evaluation only')
motion_files = []
if not foot_run:
    try:
        manifest=args.reference_manifest or saved_config.get('reference_manifest')
        if manifest:
            from supermimic.motion_prior_data import load_manifest
            motion_files,_=load_manifest(manifest,args.manifest_split or saved_config.get('manifest_split','train'))
        else: motion_files = select_reference_files(args.motion_dir, args.clips or saved_config.get('clips'))
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
elif not 1 <= args.num_envs <= 16:
    parser.error('Use 1–16 foot evaluation environments')

import numpy as np
import torch

import supermimic.networks as networks
from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import S54Kinematics
from supermimic.mdp import MDP, RewardCfg
from supermimic.motion import MotionLib
from supermimic.networks import Policy
from supermimic.s54 import default_joint_pos_list, JOINT_NAMES, ROBOT_ID, assert_checkpoint
from supermimic.goals import GoalSource, GoalSpec
from supermimic.foot_trajectory import AlternatingConfig
from supermimic.foot_mdp import FootTrackingMDP,resolve_foot_task,assert_task_checkpoint
from supermimic.foot_evaluate import evaluate_foot

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda:0"
kin = S54Kinematics(dev)
motion = MotionLib(motion_files, kin, dev) if not foot_run else None
# Historical checkpoints used no head penalty. Keep their reported metrics
# comparable; newly trained checkpoints record the corrected reward config.
reward_config = saved_config.get('reward_config', dict(w_head_position=0., w_head_velocity=0.))
goal_spec = GoalSpec(saved_config.get('goal_mode', 'motion'), saved_config.get('goal_frames', 10), saved_config.get('goal_dt'))
foot_config = AlternatingConfig(**(saved_config.get('foot_config') or {}))
task_cfg = resolve_foot_task(saved_config.get('foot_task_config')) if foot_run else None
goals = GoalSource(motion, kin, goal_spec, foot_config,command_mode=task_cfg.command_mode if foot_run else 'alternating')
default_q = torch.tensor(default_joint_pos_list(),device=dev)
mdp = (FootTrackingMDP(kin,default_q,goals,task_cfg) if foot_run else MDP(
    kin,motion,default_q,RewardCfg(**reward_config),
    observation_mode=saved_config.get('motion_obs_mode', 'legacy_v1')))
if goal_spec.mode=='reference_foot':
    from supermimic.reference_foot import ReferenceFootGoalSource,PairedMotionMDP
    goals=ReferenceFootGoalSource(motion,kin,goal_spec,foot_config)
    mdp=PairedMotionMDP(kin,motion,default_q,goals,RewardCfg(**reward_config))
if saved_config.get('prior_config'):
    from supermimic.upper_prior import PriorFootMDP,resolve_prior
    mdp=PriorFootMDP(kin,default_q,goals,task_cfg,resolve_prior(saved_config['prior_config']))
C = args.num_envs if foot_run else len(motion.names)


@torch.no_grad()
def evaluate(env: TrackingSim, ck: dict, record: str | None = None) -> dict:
    if foot_run:
        seeds = args.seeds or [ck['args'].get('seed',0)]
        results = []
        for seed in seeds:
            target = record
            if record and len(seeds) > 1:
                from pathlib import Path
                path = Path(record)
                target = str(path.with_name(f'{path.stem}_seed{seed}{path.suffix}'))
            results.append(evaluate_foot(env,ck,goals,mdp,target,dev,duration_s=args.duration_s,seed=seed,
                                        reset_noise=args.reset_noise,controller=args.controller,diagnostics=args.diagnostics))
        if len(results) == 1:
            return results[0]
        aggregate = {k: float(np.mean([r[k] for r in results]))
                     for k in ('norm_return','norm_return_framew','mean_episode_length','normalized_sim_reward','foot_position_rmse_m')}
        aggregate.update(n_failed=sum(r['n_failed'] for r in results), n_timeout=sum(r['n_timeout'] for r in results),
                         failed=[f'seed{r["seed"]}:{name}' for r in results for name in r['failed']], per_seed=results)
        return aggregate
    assert_checkpoint(ck)
    assert_task_checkpoint(ck,mdp)
    goal_spec.assert_checkpoint(ck, foot_config)
    a_ = ck["args"]
    K, Hh = a_["goal_frames"], a_["hist"]
    networks.AMP_DTYPE = torch.bfloat16 if a_.get("amp", "none") == "bf16" else None
    if goal_spec.mode=='reference_foot':
        from supermimic.networks import DualEncoderPolicy
        pi=DualEncoderPolicy(hist=Hh).to(dev)
    else:pi = Policy(mdp.obs_dim, goal_spec.dim, hist=Hh,obs_adapter=a_.get('obs_adapter','identity')).to(dev)
    pi.load_state_dict(ck["pi"])
    pi.eval()
    # Re-evaluating a checkpoint uses the same sampled foot commands.
    torch.manual_seed(a_.get('seed', 0))
    ids = torch.arange(C, device=dev)
    env.done_eval[:] = False
    env.failed_eval[:] = False
    env.reset(ids, motion.clip_start.clone())
    recorded_commands = ({'command_' + k: v.cpu().numpy() for k, v in env.command.tensors().items()}
                         if goals.is_foot and record else {})
    clip_end = motion.clip_end.clone()
    ret = torch.zeros(C, device=dev)
    head_penalty_ret = torch.zeros_like(ret)
    head_error_sum = torch.zeros_like(ret)
    steps = torch.zeros(C, device=dev)
    hist = None
    traj = {"root_p": [], "root_quat": [], "q": [], "f": [], "alive": []}
    if goals.is_foot:
        traj['command_step'] = []
    while not bool(env.done_eval.all()):
        env.f = torch.minimum(env.f, clip_end - 1)  # done clips may keep stepping while others finish
        S = env.state()
        o = goals.obs(mdp, S, env.a_prev, env.xi, env.f, env.command)
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        a = pi(hist, goals.goal(S, env.f, env.ep_step, env.command))
        alive = ~env.done_eval.clone()
        if record:
            from supermimic.rotations import mat_to_quat

            traj["root_p"].append(S[0].cpu()), traj["root_quat"].append(mat_to_quat(S[1]).cpu())
            traj["q"].append(S[4].cpu()), traj["f"].append(env.f.cpu()), traj["alive"].append(alive.cpu())
            if goals.is_foot:
                traj['command_step'].append(env.ep_step.cpu().clone())
        out = env.step(a)
        ret += torch.where(alive, out["r"], 0.0)
        head_penalty_ret += torch.where(alive, out['terms']['head_penalty'], 0.0)
        head_error_sum += torch.where(alive, out['terms']['head_position_error'], 0.0)
        steps += alive.float()
    norm = ret / ((motion.clip_len - 1).float() * mdp.max_step_reward)
    # Removing just the new head cost preserves a common reward diagnostic;
    # the actual rollout and all terminations remain unchanged.
    no_head_cost = (ret - head_penalty_ret) / ((motion.clip_len - 1).float() * mdp.max_step_reward)
    head_rms = (head_error_sum / (2 * steps.clamp_min(1))).sqrt() * (180 / np.pi)
    res = {
        "norm_return": norm.mean().item(),
        "norm_return_framew": (ret.sum() / ((motion.clip_len - 1).sum() * mdp.max_step_reward)).item(),
        "n_failed": int(env.failed_eval.sum().item()),
        "failed": [motion.names[i] for i in env.failed_eval.nonzero().flatten().tolist()],
        "per_clip": {motion.names[i]: round(norm[i].item(), 4) for i in range(C)},
        "norm_return_without_head_penalty": no_head_cost.mean().item(),
        "per_clip_without_head_penalty": {motion.names[i]: round(no_head_cost[i].item(), 4) for i in range(C)},
        "head_rms_error_deg": {motion.names[i]: head_rms[i].item() for i in range(C)},
    }
    if record:
        # Preserve the actual terminal state, including a single-clip early failure.
        # Rendering can freeze this state and label termination without fabricating a rollout.
        S = env.state()
        traj['root_p'].append(S[0].cpu()); traj['root_quat'].append(mat_to_quat(S[1]).cpu())
        traj['q'].append(S[4].cpu()); traj['f'].append(torch.minimum(env.f,clip_end).cpu())
        traj['alive'].append((~env.done_eval).cpu())
        if goals.is_foot:
            traj['command_step'].append(env.ep_step.cpu().clone())
        os.makedirs(os.path.dirname(os.path.abspath(record)), exist_ok=True)
        np.savez_compressed(record, robot=ROBOT_ID, joint_names=np.array(JOINT_NAMES), control_dt=0.02, quat="wxyz", names=np.array(motion.names), goal_contract=json.dumps(goal_spec.contract()), **recorded_commands, **{k: torch.stack(v).numpy() for k, v in traj.items()},
                            ref_root_p=motion.root_p.cpu().numpy(), ref_root_R=motion.root_R.cpu().numpy(), ref_q=motion.q.cpu().numpy(),
                            clip_start=motion.clip_start.cpu().numpy(), clip_len=motion.clip_len.cpu().numpy())  # fmt: skip
    return res


sim_cfg = make_sim(dev, args)
with launch_simulation(sim_cfg, args):
    env = TrackingSim(sim_cfg, C, motion, mdp, eval_mode=True, goals=goals)
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location=dev)
        res = evaluate(env, ck, args.record)
        result_path = args.out_json or args.ckpt.replace('.pt', '_eval.json')
        os.makedirs(os.path.dirname(os.path.abspath(result_path)), exist_ok=True)
        json.dump(res, open(result_path, 'w'), indent=2)
        print(json.dumps(res, indent=2))
        sys.exit(0)
    out_csv = os.path.join(args.run_dir, "eval.csv")
    done = set()
    if os.path.exists(out_csv):
        done = {int(l.split(",")[0]) for l in open(out_csv).read().strip().splitlines()[1:]}
    else:
        open(out_csv, "w").write("iteration,train_wall_h,saved_at,eval_at,norm_return,norm_return_framew,n_failed,failed\n")
    while True:
        ckpts = sorted(glob.glob(os.path.join(args.run_dir, "ckpt_*.pt")))
        todo = [c for c in ckpts if int(os.path.basename(c)[5:11]) not in done]
        todo = [c for c in todo if (int(os.path.basename(c)[5:11]) // 2000) % args.every == 0]
        for c in todo:
            it = int(os.path.basename(c)[5:11])
            ck = torch.load(c, map_location=dev)
            t0 = time.time()
            res = evaluate(env, ck)
            line = f"{it},{ck['wall_time'] / 3600:.4f},{ck['saved_at']},{time.strftime('%Y-%m-%d %H:%M:%S')},{res['norm_return']:.4f},{res['norm_return_framew']:.4f},{res['n_failed']},{'|'.join(res['failed'])}\n"
            open(out_csv, "a").write(line)
            json.dump(res, open(c.replace(".pt", "_eval.json"), "w"), indent=1)
            print(f"[eval] it {it} ({ck['wall_time'] / 3600:.2f} h): norm return {res['norm_return']:.4f}, failed {res['n_failed']}/{C} ({time.time() - t0:.0f}s)", flush=True)
            done.add(it)
        if not args.watch:
            break
        time.sleep(60)
