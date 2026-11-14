"""Independent frozen-policy evaluation through real Newton dynamics only."""
import os
os.environ['SUPERMIMIC_ROBOT']='roban'
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from isaaclab.app import add_launcher_args,launch_simulation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path,help='New report directory; existing results are protected')
    p.add_argument('--num_envs',type=int,default=4)
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--seed',type=int,default=0)
    add_launcher_args(p);args=p.parse_args()
    if not 1<=args.num_envs<=16 or not 1<=args.steps<=1000:p.error('Use 1–16 environments and 1–1000 steps')
    if args.out.exists():p.error('Evaluation output already exists; choose a new directory')
    import numpy as np
    import torch
    from supermimic.roban import assert_checkpoint,NUM_J,FOOT_BODY_NAMES,foot_geometry
    from supermimic.roban_runtime import build_task,select_roban_files
    from supermimic.kinematics import RobotKinematics
    from supermimic.networks import Policy
    from supermimic.mdp import flatten_state,unflatten_state
    from supermimic.env import TrackingSim,make_sim
    from supermimic.control import bound_action
    torch.manual_seed(args.seed);dev='cuda:0'
    ck=torch.load(args.checkpoint,map_location=dev,weights_only=False);assert_checkpoint(ck)
    config=SimpleNamespace(**ck['args'])
    kin=RobotKinematics(dev)
    files=select_roban_files(config.motion_dir,config.clips)
    motion,goals,mdp,_=build_task(config,kin,files,dev)
    goals.spec.assert_checkpoint(ck)
    if ck['task_contract']!=mdp.contract():raise ValueError('Evaluation task/terrain differs from checkpoint')
    pi=Policy(mdp.obs_dim,goals.spec.dim,hist=config.hist,obs_adapter='identity').to(dev)
    pi.load_state_dict(ck['pi']);pi.eval()
    args.out.mkdir(parents=True)
    returns=torch.zeros(args.num_envs,device=dev);lengths=returns.clone();square_error=returns.clone()
    saved={k:[] for k in ('state','action','reference_foot_p','actual_foot_p','reward','alive')}
    max_fk=max_sync=0.;observed_force=False;contact_status='unavailable'
    sim=make_sim(dev,args)
    with launch_simulation(sim,args):
        env=TrackingSim(sim,args.num_envs,motion,mdp,eval_mode=True,reset_noise=False,goals=goals)
        frames=motion.clip_start[torch.arange(args.num_envs,device=dev)%len(motion.names)]
        env.reset(torch.arange(args.num_envs,device=dev),frames)
        env.max_episode_steps=args.steps
        # State writes are allowed only at the initial reset. Any attempt to
        # animate reference poses during evaluation is an immediate test failure.
        def forbid_state_write(*a,**kw): raise RuntimeError('Evaluation attempted a non-reset root/joint state write')
        for method in ('write_root_link_pose_to_sim_index','write_root_link_velocity_to_sim_index',
                       'write_joint_position_to_sim_index','write_joint_velocity_to_sim_index'):
            setattr(env.robot,method,forbid_state_write)
        try:
            from supermimic.standing_diagnostics import NewtonGroundContacts
            contacts=NewtonGroundContacts(env)
            contact_status='last_substep_solver_forces'
        except (AttributeError,RuntimeError,NotImplementedError,ValueError) as exc:
            contacts=None;contact_status=f'unavailable: {exc}'
        history=None
        with torch.no_grad():
            for _ in range(args.steps):
                S=env.state();alive=~env.done_eval.clone()
                o=goals.obs(mdp,S,env.a_prev,env.xi,env.f,env.command)
                history=o[:,None].expand(-1,config.hist,-1).clone() if history is None else torch.cat((history[:,1:],o[:,None]),1)
                goal=goals.goal(S,env.f,env.ep_step,env.command)
                a=bound_action(pi(history,goal))
                out=env.step(a)
                if not torch.isfinite(out['s1']).all():raise FloatingPointError('Nonfinite Newton rollout')
                S1=unflatten_state(out['s1'])
                reward=mdp.reward(S1,out['f']+1,a,out['a_prev'],command=out['command'],ep_step=out['ep_step']+1)
                max_sync=max(max_sync,float((reward-out['r']).abs().max()))
                actual,R,_=motion.feet(S1)
                ref=motion.sample((out['f']+1)[:,None],out['command'])['foot_p'][:,0]
                ids=[env.robot.body_names.index(n) for n in FOOT_BODY_NAMES]
                poses=env.robot.data.body_link_pose_w.torch
                linkp=poses[:,ids,:3]-env.scene.env_origins[:,None]
                from supermimic.rotations import quat_to_mat
                quat=poses[:,ids,3:7][...,[3,0,1,2]]
                realR=quat_to_mat(quat)
                offsets=linkp.new_tensor([f['sole'] for f in foot_geometry()])
                soles=linkp+(realR@offsets[...,None]).squeeze(-1)
                max_fk=max(max_fk,float((soles-actual).norm(dim=-1).max()))
                if contacts is not None:
                    counts,forces,_=contacts.read()
                    observed_force=observed_force or bool(np.any(np.linalg.norm(forces,axis=-1)>1e-3))
                returns+=out['r']*alive;lengths+=alive;square_error+=(actual-ref).square().sum(-1).mean(-1)*alive
                for key,value in dict(state=out['s1'],action=a,reference_foot_p=ref,actual_foot_p=actual,reward=out['r'],alive=alive).items():
                    saved[key].append(value.cpu().numpy())
                if env.done_eval.all():break
        failed=int(env.failed_eval.sum())
        normalized=returns/(args.steps*mdp.max_step_reward)
    if max_fk>1e-4 or max_sync>1e-5:raise AssertionError(f'Newton FK/reward mismatch: {max_fk}, {max_sync}')
    report=dict(status='PASS',scope='short physical chain test; untrained policy may fall',
        backend='Isaac Lab 3 Newton/MJWarp',checkpoint=str(args.checkpoint.resolve()),
        num_envs=args.num_envs,requested_steps=args.steps,executed_steps=len(saved['state']),
        episode_lengths=lengths.cpu().tolist(),norm_return=float(normalized.mean()),n_failed=failed,
        foot_rmse_m=float((square_error/lengths.clamp_min(1)).mean().sqrt()),
        max_sole_fk_error_m=max_fk,max_real_replay_reward_error=max_sync,
        actor_obs_dim=mdp.obs_dim,actor_goal_dim=goals.spec.dim,actor_uses_whole_body_reference=False,
        state_writes_after_reset=0,contact_status=contact_status,nonzero_contact_force_observed=observed_force,
        reference_validation=motion.validation)
    np.savez_compressed(args.out/'trajectory.npz',**{k:np.stack(v) for k,v in saved.items()})
    (args.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='reference_validation'},indent=2))


if __name__=='__main__':main()
