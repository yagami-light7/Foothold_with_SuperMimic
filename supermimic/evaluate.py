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
parser.add_argument("--motion_dir", type=str, default=None, help="defaults to this run's saved reference directory")
parser.add_argument("--clips", nargs="+", help="defaults to the clips recorded in args.json")
parser.add_argument("--every", type=int, default=1, help="evaluate every n-th checkpoint")
add_launcher_args(parser)
args = parser.parse_args()
saved_config = json.loads(open(os.path.join(args.run_dir, 'args.json')).read())
args.motion_dir = args.motion_dir or saved_config.get('motion_dir') or os.path.join(REPO, 'data/retarget_data/reference_startup_fixed')
from supermimic.reference_selection import select_reference_files
try:
    motion_files = select_reference_files(args.motion_dir, args.clips or saved_config.get('clips'))
except (ValueError, FileNotFoundError) as exc:
    parser.error(str(exc))

import numpy as np
import torch

import supermimic.networks as networks
from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import S54Kinematics
from supermimic.mdp import MAX_STEP_REWARD, OBS_DIM, MDP, RewardCfg
from supermimic.motion import MotionLib
from supermimic.networks import Policy
from supermimic.s54 import default_joint_pos_list, GOAL_FRAME_DIM, JOINT_NAMES, ROBOT_ID, assert_checkpoint

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda:0"
kin = S54Kinematics(dev)
motion = MotionLib(motion_files, kin, dev)
# Historical checkpoints used no head penalty. Keep their reported metrics
# comparable; newly trained checkpoints record the corrected reward config.
reward_config = saved_config.get('reward_config', dict(w_head_position=0., w_head_velocity=0.))
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev), RewardCfg(**reward_config))
C = len(motion.names)


@torch.no_grad()
def evaluate(env: TrackingSim, ck: dict, record: str | None = None) -> dict:
    assert_checkpoint(ck)
    a_ = ck["args"]
    K, Hh = a_["goal_frames"], a_["hist"]
    networks.AMP_DTYPE = torch.bfloat16 if a_.get("amp", "none") == "bf16" else None
    pi = Policy(OBS_DIM, GOAL_FRAME_DIM * K, hist=Hh).to(dev)
    pi.load_state_dict(ck["pi"])
    pi.eval()
    ids = torch.arange(C, device=dev)
    env.done_eval[:] = False
    env.failed_eval[:] = False
    env.reset(ids, motion.clip_start.clone())
    clip_end = motion.clip_end.clone()
    ret = torch.zeros(C, device=dev)
    head_penalty_ret = torch.zeros_like(ret)
    head_error_sum = torch.zeros_like(ret)
    steps = torch.zeros(C, device=dev)
    hist = None
    traj = {"root_p": [], "root_quat": [], "q": [], "f": [], "alive": []}
    while not bool(env.done_eval.all()):
        env.f = torch.minimum(env.f, clip_end - 1)  # done clips may keep stepping while others finish
        S = env.state()
        o = mdp.obs(S, env.a_prev, env.xi, env.f)
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        a = pi(hist, motion.goal(env.f, K))
        alive = ~env.done_eval.clone()
        if record:
            from supermimic.rotations import mat_to_quat

            traj["root_p"].append(S[0].cpu()), traj["root_quat"].append(mat_to_quat(S[1]).cpu())
            traj["q"].append(S[4].cpu()), traj["f"].append(env.f.cpu()), traj["alive"].append(alive.cpu())
        out = env.step(a)
        ret += torch.where(alive, out["r"], 0.0)
        head_penalty_ret += torch.where(alive, out['terms']['head_penalty'], 0.0)
        head_error_sum += torch.where(alive, out['terms']['head_position_error'], 0.0)
        steps += alive.float()
    norm = ret / ((motion.clip_len - 1).float() * MAX_STEP_REWARD)
    # Removing just the new head cost preserves a common reward diagnostic;
    # the actual rollout and all terminations remain unchanged.
    no_head_cost = (ret - head_penalty_ret) / ((motion.clip_len - 1).float() * MAX_STEP_REWARD)
    head_rms = (head_error_sum / (2 * steps.clamp_min(1))).sqrt() * (180 / np.pi)
    res = {
        "norm_return": norm.mean().item(),
        "norm_return_framew": (ret.sum() / ((motion.clip_len - 1).sum() * MAX_STEP_REWARD)).item(),
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
        os.makedirs(os.path.dirname(os.path.abspath(record)), exist_ok=True)
        np.savez_compressed(record, robot=ROBOT_ID, joint_names=np.array(JOINT_NAMES), control_dt=0.02, quat="wxyz", names=np.array(motion.names), **{k: torch.stack(v).numpy() for k, v in traj.items()},
                            ref_root_p=motion.root_p.cpu().numpy(), ref_root_R=motion.root_R.cpu().numpy(), ref_q=motion.q.cpu().numpy(),
                            clip_start=motion.clip_start.cpu().numpy(), clip_len=motion.clip_len.cpu().numpy())  # fmt: skip
    return res


sim_cfg = make_sim(dev, args)
with launch_simulation(sim_cfg, args):
    env = TrackingSim(sim_cfg, C, motion, mdp, eval_mode=True)
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location=dev)
        res = evaluate(env, ck, args.record)
        json.dump(res, open(args.ckpt.replace(".pt", "_eval.json"), "w"), indent=2)
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
