"""Atomic runtime state, bounded serial Newton evaluation and local telemetry."""
from __future__ import annotations
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from .curriculum import evaluation_commands, feasibility, Command
from .control import bound_action
from .motion import CONTROL_DT
from .standing_diagnostics import NewtonGroundContacts


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(s):
    random.setstate(s['python']); np.random.set_state(s['numpy']); torch.set_rng_state(s['torch'].cpu())
    if s['cuda']: torch.cuda.set_rng_state_all([v.cpu() for v in s['cuda']])


def atomic_save(value, path):
    path=Path(path); temp=path.with_name(path.name+'.tmp')
    with temp.open('wb') as f:
        torch.save(value,f); f.flush(); os.fsync(f.fileno())
    os.replace(temp,path)
    fd=os.open(path.parent,os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def emit(path, iteration, values):
    with (Path(path)/'curriculum_metrics.jsonl').open('a') as f:
        f.write(json.dumps(dict(iteration=iteration,values=values),allow_nan=False)+'\n'); f.flush()


@torch.no_grad()
def evaluate(env, pi, goals, mdp, curriculum, hist, full=False, deadline=float('inf'), rich=False, critic=None):
    """Reuse the training scene sequentially. Reset training episodes afterwards.

    Freeze policy normalization, no replay insertion, fixed independent seeds.
    Timeout is survival evidence only; tracking gates also have to pass.
    """
    saved_rng=rng_state(); sampler_state=curriculum.state_dict(); training=pi.training
    old=(env.eval_mode,env.max_episode_steps,env.reset_noise)
    rows=[]; metrics={}; started=time.monotonic()
    try:
        env.eval_mode=True; env.max_episode_steps=500 if full else 250; env.reset_noise=True; pi.eval()
        try: contacts=NewtonGroundContacts(env); contact_status='last_substep_solver_contacts'
        except (AttributeError,RuntimeError,NotImplementedError) as exc:
            contacts=None; contact_status='UNAVAILABLE: '+str(exc)
        for cmd in evaluation_commands(curriculum.stage,rich):
            why=feasibility(cmd,curriculum.cfg)
            if why:
                rows.append(dict(command=vars(cmd),status='unreachable_envelope',reason=why)); continue
            if time.monotonic()+2 >= deadline:
                rows.append(dict(command=vars(cmd),status='budget_skipped')); continue
            command_started=time.monotonic()
            n=min(8,env.N); ids=torch.arange(n,device=env.device)
            goals.fixed_command=cmd; seed=8541 if not full else 9541
            torch.manual_seed(seed); curriculum.rng.seed(seed)
            env.done_eval.zero_(); env.failed_eval.zero_(); env.reset(ids)
            history=None
            error=torch.zeros(n,device=env.device); steps=error.clone(); yawerr=error.clone(); velerr=error.clone()
            speed_sum=error.clone(); yaw_sum=error.clone()
            slip=error.clone(); contact_mismatch=error.clone(); contact_samples=error.clone(); stance_samples=error.clone()
            touch_error=error.clone(); touch_count=error.clone(); step_error=error.clone(); step_count=error.clone()
            prev_contact=torch.ones(n,2,dtype=torch.bool,device=env.device)
            last_landing=torch.full((n,2),float('nan'),device=env.device); freq_error=error.clone(); freq_count=error.clone()
            previous_landing_pos=None
            discounted_return=error.clone(); initial_values=None
            if critic is not None:
                initial=tuple(v[:n] for v in env.state()); cc=env.command.index(ids)
                from .mdp import normalize_xi
                gg=goals.goal(initial,env.f[:n],env.ep_step[:n],cc)
                xx=mdp.critic_features(initial,env.f[:n],normalize_xi(env.xi[:n]),command=cc,ep_step=env.ep_step[:n])
                initial_values=critic(torch.cat((xx,gg),-1))
            while not bool(env.done_eval[:n].all()):
                if time.monotonic() >= deadline: break
                alive=~env.done_eval[:n].clone(); S=tuple(v[:n] for v in env.state()); command=env.command.index(ids)
                obs=goals.obs(mdp,S,env.a_prev[:n],env.xi[:n],env.f[:n],command)
                history=obs[:,None].repeat(1,hist,1) if history is None else torch.cat((history[:,1:],obs[:,None]),1)
                actions=torch.zeros_like(env.a_prev)
                actions[:n]=bound_action(pi(history,goals.goal(S,env.f[:n],env.ep_step[:n],command)))
                out=env.step(actions)
                if 'r' in out:
                    discounted_return+=torch.where(alive,env.gamma**steps*out['r'][:n],0.)
                state=tuple(v[:n] for v in env.state()); bp,_,bv,_=mdp.kin.forward(*state)
                feet=bp[:n,mdp.foot_indices]; velocities=bv[:n,mdp.foot_indices]
                ref=mdp._reference(command,env.ep_step[:n])
                e=(feet-ref.position[:n,0]).square().sum(-1).mean(-1)
                error+=torch.where(alive,e,0); steps+=alive
                yawerr+=torch.where(alive,(state[3][:n,2]-cmd.yaw_rate).abs(),0)
                speed_sum+=torch.where(alive,state[2][:n,:2].norm(dim=-1),0)
                yaw_sum+=torch.where(alive,state[3][:n,2],0)
                # Speed magnitude allows a curved travel heading; no world-x assumption.
                velerr+=torch.where(alive,(state[2][:n,:2].norm(dim=-1)-cmd.length*cmd.frequency).abs(),0)
                t=env.ep_step[:n].to(feet.dtype)*CONTROL_DT
                valid=torch.arange(env.command.foot.shape[1],device=feet.device)[None]<env.command.lengths[:n,None]
                landing=valid & (env.command.touchdown[:n] > (t-CONTROL_DT)[:,None]+1e-7) & (env.command.touchdown[:n] <= t[:,None]+1e-7)
                actual=feet.gather(1,env.command.foot[:n,:,None].expand(-1,-1,3))
                dist=(actual-env.command.target_pos[:n]).norm(dim=-1)
                touch_error+=(dist*landing*alive[:,None]).sum(-1); touch_count+=(landing*alive[:,None]).sum(-1)
                if contacts is not None:
                    counts,_,_=contacts.read(env_ids=ids); actual_contact=torch.as_tensor(counts[:n]>0,device=feet.device)
                    stance=actual_contact & alive[:,None]
                    slip+=(velocities[...,:2].norm(dim=-1)*stance).sum(-1); stance_samples+=stance.sum(-1)
                    contact_mismatch+=((actual_contact!=ref.contact[:n,0]) & alive[:,None]).sum(-1)
                    contact_samples+=alive*2
                    landed=actual_contact & ~prev_contact & alive[:,None]
                    previous_times=last_landing.clone()
                    for side in (0,1):
                        elapsed=t-last_landing[:,side]
                        ok=landed[:,side] & torch.isfinite(elapsed) & (elapsed>.2)
                        freq_error+=torch.where(ok,(2/elapsed.clamp_min(.01)-cmd.frequency).abs(),0); freq_count+=ok
                        last_landing[:,side]=torch.where(landed[:,side],t,last_landing[:,side])
                    if previous_landing_pos is None: previous_landing_pos=feet.clone()
                    for side in (0,1):
                        # Same-foot stride is two alternating step lengths.
                        ok=landed[:,side] & previous_times[:,side].isfinite() & (t-previous_times[:,side]>.2)
                        displacement=(feet[:,side,:2]-previous_landing_pos[:,side,:2]).norm(dim=-1)/2
                        step_error+=torch.where(ok,(displacement-cmd.length).abs(),0); step_count+=ok
                        previous_landing_pos[:,side]=torch.where(landed[:,side,None],feet[:,side],previous_landing_pos[:,side])
                    prev_contact=actual_contact
            rmse=(error/steps.clamp_min(1)).sqrt(); yr=yawerr/steps.clamp_min(1); ve=velerr/steps.clamp_min(1)
            completed=env.done_eval[:n] & ~env.failed_eval[:n] & (steps>=env.max_episode_steps)
            yaw_tolerance=.08 if cmd.yaw_rate==0 else .04+.25*abs(cmd.yaw_rate)
            speed_tolerance=max(.025,.35*cmd.length*cmd.frequency)
            tracking=(rmse<=curriculum.cfg['rmse_threshold']) & (yr<yaw_tolerance) & (ve<speed_tolerance)
            if cmd.frequency>0:
                tracking &= (touch_count>=2) & (touch_error/touch_count.clamp_min(1)<.07)
                # Scheduled touchdown alone cannot prove actual stepping.
                if contacts is None: tracking &= False
                else:
                    tracking &= ((freq_count>=1) & (freq_error/freq_count.clamp_min(1)<max(.15,.25*cmd.frequency))
                                 & (step_count>=1) & (step_error/step_count.clamp_min(1)<.03+.25*cmd.length))
            success=completed & tracking
            complete=bool(env.done_eval[:n].all())
            row=dict(command=vars(cmd),status='evaluated' if complete else 'incomplete',duration=env.max_episode_steps*CONTROL_DT,
                     evaluation_complete=complete,episode_lengths=steps.int().cpu().tolist(),
                     command_wall_seconds=time.monotonic()-command_started,
                     success_rate=float(success.float().mean()) if complete else None,
                     success_rate_lower_bound=float(success.float().mean()),foot_rmse=float(rmse.mean()),
                     physical_termination_rate=float(env.failed_eval[:n].float().mean()),
                     timeout_rate=float(completed.float().mean()),goal_tracking_failure_rate=float((completed & ~tracking).float().mean()),
                     incomplete_rate=float((~env.done_eval[:n]).float().mean()),
                     yaw_rate_error=float(yr.mean()),speed_error=float(ve.mean()),
                     actual_speed=float((speed_sum/steps.clamp_min(1)).mean()),
                     actual_yaw_rate=float((yaw_sum/steps.clamp_min(1)).mean()),
                     touchdown_error=float(touch_error.sum()/touch_count.sum().clamp_min(1)),
                     touchdown_events=int(touch_count.sum()),actual_contact_status=contact_status)
            if contacts is not None:
                row.update(contact_mismatch=float(contact_mismatch.sum()/contact_samples.sum().clamp_min(1)),
                           stance_slip=float(slip.sum()/stance_samples.sum().clamp_min(1)),
                           frequency_error=float(freq_error.sum()/freq_count.sum().clamp_min(1)),
                           frequency_measurements=int(freq_count.sum()),
                           step_length_error=float(step_error.sum()/step_count.sum().clamp_min(1)),
                           step_length_measurements=int(step_count.sum()))
            row['discounted_episode_return']=discounted_return.cpu().tolist()
            if initial_values is not None:
                row['critic_initial_values']=initial_values.cpu().tolist()
                physically_ended=env.failed_eval[:n]
                row['critic_mc_physical_samples']=int(physically_ended.sum())
                row['critic_mc_physical_bias']=float((initial_values-discounted_return)[physically_ended].mean()) if bool(physically_ended.any()) else None
            rows.append(row)
        # Worst representative command gates each task, so easy cells cannot hide failure.
        for task in {r['command']['task'] for r in rows if r['status']=='evaluated'}:
            part=[r for r in rows if r['command']['task']==task and r['status']=='evaluated']
            metrics[task]=dict(success_rate=min(r['success_rate'] for r in part),foot_rmse=max(r['foot_rmse'] for r in part))
        # A deadline-truncated protocol cannot promote a stage based on a subset.
        for row in rows:
            if row['status'] in ('budget_skipped','incomplete'):
                metrics.pop(row['command']['task'],None)
        buckets={}
        for row in rows:
            if row['status']!='evaluated': continue
            bucket=Command(**row['command']).bucket
            value=buckets.setdefault(bucket,dict(success_rate=1.,foot_rmse=0.))
            value['success_rate']=min(value['success_rate'],row['success_rate'])
            value['foot_rmse']=max(value['foot_rmse'],row['foot_rmse'])
        estimated=0.
        measured=[r for r in rows if 'command_wall_seconds' in r]
        for row in rows:
            if row['status']=='incomplete':
                estimated+=row['command_wall_seconds']*(500 if full else 250)/max(max(row['episode_lengths']),1)
            elif row['status']=='budget_skipped':
                estimated+=max((r['command_wall_seconds'] for r in measured),default=30.)
            elif row['status']=='evaluated': estimated+=row['command_wall_seconds']
        minimum_next=min(300.,max(10.,estimated*1.15)) if any(r['status'] in ('budget_skipped','incomplete') for r in rows) else 10.
        return dict(minimum_next_eval_seconds=minimum_next,bucket_metrics=buckets,kind='full' if full else 'fast',rows=rows,metrics=metrics,wall_seconds=time.monotonic()-started)
    finally:
        goals.fixed_command=None; env.eval_mode,env.max_episode_steps,env.reset_noise=old
        curriculum.load_state_dict(sampler_state)
        env.done_eval.zero_(); env.failed_eval.zero_(); env.reset(torch.arange(env.N,device=env.device))
        restore_rng(saved_rng); pi.train(training)
