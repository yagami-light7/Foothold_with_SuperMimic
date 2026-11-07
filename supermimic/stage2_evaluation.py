"""Serial, fixed-seed Newton evaluation of generated goals, without MotionLib."""
import copy,json,time,math
from pathlib import Path
import numpy as np
import torch
from .stage2_curriculum import evaluation_cases
from .curriculum import feasibility
from .curriculum_runtime import rng_state,restore_rng
from .standing_diagnostics import NewtonGroundContacts,stop_reasons
from .control import bound_action
from .mdp import flatten_state
from .foot_trajectory import command_time

@torch.no_grad()
def evaluate(env,parts,cfg,out,seconds=5.,deadline=float('inf'),*,seed=8541,record_diagnostics=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    pi,mdp,g,sampler=[parts[k] for k in ('policy','mdp','goals','curriculum')]
    saved_rng=rng_state();saved_sampler=copy.deepcopy(sampler.state_dict())
    old=(env.eval_mode,env.reset_noise,env.max_episode_steps,pi.training)
    n=min(cfg['eval_envs'],env.N);ids=torch.arange(n,device=env.device);steps_max=round(seconds/.02)
    started=time.monotonic();rows=[];newton_transitions=0
    try:
        env.eval_mode=True;env.reset_noise=True;env.max_episode_steps=steps_max;pi.eval()
        contacts=NewtonGroundContacts(env)
        for label,cmd in evaluation_cases(sampler.stage):
            if time.monotonic()>=deadline:break
            why=feasibility(cmd,sampler.cfg)
            if why:raise ValueError('Infeasible independent evaluation command '+label+': '+why)
            torch.manual_seed(seed);sampler.rng.seed(seed);g.fixed_command=cmd
            env.done_eval.zero_();env.failed_eval.zero_();env.reset(ids)
            cc=env.command.index(ids).clone();initial=env.state()[0][:n].clone()
            prev_contact=torch.ones(n,2,dtype=torch.bool,device=env.device)
            history=None;count=torch.zeros(n,device=env.device);errors=count.clone();rewards=count.clone()
            yaw_error=count.clone();speed_error=count.clone();slip=count.clone();stance_samples=count.clone()
            contact_match=count.clone();contact_samples=count.clone();landings=count.clone();touch_error=count.clone();touch_count=count.clone()
            max_roll=count.clone();max_pitch=count.clone();max_height=count.clone();action_delta=count.clone()
            foot_peak=count.clone();velocity=count.clone();net_yaw=count.clone();reasons=['incomplete']*n
            landing_side=torch.zeros(n,2,device=env.device);last_landing=torch.full_like(landing_side,-1000.)
            frequency_error=count.clone();frequency_count=count.clone()
            trace={k:[] for k in ('state','reference_foot_p','reference_foot_quat','reference_contact','reference_phase','action','reward')}
            measured_contacts=None
            if record_diagnostics:
                trace.update({k:[] for k in ('command_step','contact_count','contact_force_w','contact_valid','goal','a_prev')})
            def capture():
                S=env.state();ref=mdp._reference(cc,env.ep_step[:n])
                for k,v in dict(state=flatten_state(tuple(x[:n] for x in S)),reference_foot_p=ref.position[:,0],
                        reference_foot_quat=ref.quaternion[:,0],reference_contact=ref.contact[:,0],reference_phase=ref.swing_phase[:,0]).items():
                    trace[k].append(v[0].cpu().numpy().copy())
                if record_diagnostics:
                    trace['command_step'].append(int(env.ep_step[0]))
                    trace['contact_count'].append(np.zeros(2,dtype=np.int64) if measured_contacts is None else measured_contacts[0][0].copy())
                    trace['contact_force_w'].append(np.zeros((2,3)) if measured_contacts is None else measured_contacts[1][0].copy())
                    trace['contact_valid'].append(measured_contacts is not None)
            capture()
            for step in range(steps_max):
                if time.monotonic()>=deadline:break
                alive=~env.done_eval[:n].clone();S=tuple(x[:n] for x in env.state())
                obs=g.obs(mdp,S,env.a_prev[:n],env.xi[:n],env.f[:n],cc)
                history=obs[:,None].repeat(1,cfg['hist'],1) if history is None else torch.cat((history[:,1:],obs[:,None]),1)
                goal=g.goal(S,env.f[:n],env.ep_step[:n],cc)
                a=bound_action(pi(history,goal));actions=torch.zeros_like(env.a_prev);actions[:n]=a
                previous=env.a_prev[:n].clone();tr=env.step(actions);newton_transitions+=env.N;S1=tuple(x[:n] for x in env.state())
                p,_,v,_=mdp.kin.forward(*S1);feet=p[:,mdp.foot_indices];fv=v[:,mdp.foot_indices]
                ref=mdp._reference(cc,env.ep_step[:n]);R=S1[1]
                errors+=torch.where(alive,(feet-ref.position[:,0]).square().sum(-1).mean(-1),0.)
                rewards+=torch.where(alive,tr['r'][:n],0.);count+=alive
                pitch=torch.asin((-R[:,2,0]).clamp(-1,1)).abs()*180/math.pi
                roll=torch.atan2(R[:,2,1],R[:,2,2]).abs()*180/math.pi
                max_pitch=torch.maximum(max_pitch,torch.where(alive,pitch,0.));max_roll=torch.maximum(max_roll,torch.where(alive,roll,0.))
                max_height=torch.maximum(max_height,torch.where(alive,(S1[0][:,2]-mdp.nominal_height).abs(),0.))
                speed=S1[2][:,:2].norm(dim=-1);velocity+=torch.where(alive,speed,0.)
                speed_error+=torch.where(alive,(speed-cmd.length*cmd.frequency).abs(),0.)
                yaw_error+=torch.where(alive,(S1[3][:,2]-cmd.yaw_rate).abs(),0.)
                net_yaw+=torch.where(alive,S1[3][:,2]*.02,0.)
                action_delta+=torch.where(alive,(a-previous).square().mean(-1).sqrt(),0.)
                foot_peak=torch.maximum(foot_peak,torch.where(alive,(feet[:,:,2]-cc.initial_pos[:,:,2]).max(-1).values,0.))
                raw_counts,forces,_=contacts.read(env_ids=ids)
                measured_contacts=(raw_counts,forces)
                # Require loaded contact; collision proximity alone is not support.
                actual=torch.as_tensor((raw_counts[:n]>0)&(np.abs(forces[:n,:,2])>1.),device=env.device)
                landed=(actual&~prev_contact)&alive[:,None]&((step-last_landing)>.15/.02)
                interval=(step-last_landing)*.02;valid_frequency=landed&(last_landing>=0)
                frequency_error+=(torch.where(valid_frequency,(2/interval.clamp_min(.02)-cmd.frequency).abs(),0.)).sum(-1)
                frequency_count+=valid_frequency.sum(-1);last_landing=torch.where(landed,float(step),last_landing)
                landing_side+=landed;landings+=landed.sum(-1);prev_contact=actual
                stance=actual&alive[:,None];slip+=(fv[...,:2].norm(dim=-1)*stance).sum(-1);stance_samples+=stance.sum(-1)
                contact_match+=((actual==ref.contact[:,0])&alive[:,None]).sum(-1);contact_samples+=alive*2
                t=command_time(env.ep_step[:n],.02,cc.touchdown.dtype)[:,None]
                valid=torch.arange(cc.foot.shape[1],device=env.device)[None]<cc.lengths[:,None]
                landing=valid&(cc.touchdown>t-.02+1e-7)&(cc.touchdown<=t+1e-7)&alive[:,None]
                actual_p=feet.gather(1,cc.foot[...,None].expand(-1,-1,3))
                touch_error+=((actual_p-cc.target_pos).norm(dim=-1)*landing).sum(-1);touch_count+=landing.sum(-1)
                ended=alive&tr['done'][:n]
                native=stop_reasons(S1,mdp,tr['term'][:n],tr['timeout'][:n])
                for i in range(n):
                    if ended[i]:reasons[i]=native[i]
                if bool(alive[0]):
                    if record_diagnostics:
                        trace['goal'].append(goal[0].cpu().numpy().copy());trace['a_prev'].append(previous[0].cpu().numpy().copy())
                    trace['action'].append(a[0].cpu().numpy().copy());trace['reward'].append(float(tr['r'][0]));capture()
                if record_diagnostics and (step+1)%100==0:
                    print(f'[Newton] {label}: {step+1}/{steps_max} control steps; {int((~env.done_eval[:n]).sum())}/{n} active',flush=True)
                if bool(env.done_eval[:n].all()):break
            complete=bool(env.done_eval[:n].all());survived=env.done_eval[:n]&~env.failed_eval[:n]&(count>=steps_max)
            rmse=(errors/count.clamp_min(1)).sqrt();touch=touch_error/touch_count.clamp_min(1)
            posture=(max_pitch<=cfg['pitch_roll_max_deg'])&(max_roll<=cfg['pitch_roll_max_deg'])&(max_height<=cfg['height_deviation_max'])
            tracking=(rmse<=(cfg['standing_rmse_threshold'] if cmd.frequency==0 else cfg['foot_rmse_threshold']))&posture
            displacement=(env.state()[0][:n,:2]-initial[:,:2]).norm(dim=-1)
            if cmd.frequency==0:tracking&=(velocity/count.clamp_min(1)<.035)
            else:
                tracking&=(touch_count>=2)&(touch<=cfg['touchdown_threshold'])&(landing_side.min(-1).values>=1)
                tracking&=(frequency_count>=1)&(frequency_error/frequency_count.clamp_min(1)<=max(.2,.3*cmd.frequency))
                tracking&=(foot_peak>=.5*sampler.cfg['clearance'])
                if cmd.length>0:
                    target=mdp._reference(cc,torch.full_like(env.ep_step[:n],steps_max)).position[:,0,:,:2].mean(1)-cc.initial_pos[:,:,:2].mean(1)
                    direction=target/target.norm(dim=-1,keepdim=True).clamp_min(1e-6)
                    forward=((env.state()[0][:n,:2]-initial[:,:2])*direction).sum(-1)
                    tracking&=(forward>=.6*target.norm(dim=-1))&(speed_error/count.clamp_min(1)<=max(.025,.5*cmd.length*cmd.frequency))
                if cmd.yaw_rate!=0:tracking&=(net_yaw*math.copysign(1,cmd.yaw_rate)>=.6*abs(cmd.yaw_rate)*seconds)&(yaw_error/count.clamp_min(1)<.04+.3*abs(cmd.yaw_rate))
            success=survived&tracking
            row=dict(label=label,command=vars(cmd),complete=complete,duration_s=seconds,seed=seed,environment_seeds='fixed vector RNG, env indices 0..n-1',
                episode_lengths=count.int().cpu().tolist(),stop_reasons=reasons,survival_rate=float(survived.float().mean()),success_rate=float(success.float().mean()),
                physical_termination_rate=float(env.failed_eval[:n].float().mean()),timeout_rate=float(survived.float().mean()),
                goal_tracking_failure_rate=float((survived&~tracking).float().mean()),foot_rmse=float(rmse.mean()),
                norm_return=float((rewards/(steps_max*mdp.max_step_reward)).mean()),touchdown_error=float(touch_error.sum()/touch_count.sum().clamp_min(1)) if touch_count.sum()>0 else None,
                actual_contact_landings=int(landings.sum()),actual_displacement=float(displacement.mean()),actual_speed=float((velocity/count.clamp_min(1)).mean()),
                step_frequency_error_hz=float(frequency_error.sum()/frequency_count.sum()) if frequency_count.sum()>0 else None,
                speed_error=float((speed_error/count.clamp_min(1)).mean()),yaw_rate_error=float((yaw_error/count.clamp_min(1)).mean()),
                root_roll_max_deg=float(max_roll.max()),root_pitch_max_deg=float(max_pitch.max()),root_height_deviation_max_m=float(max_height.max()),
                contact_accuracy=float(contact_match.sum()/contact_samples.sum().clamp_min(1)),stance_slip_mps=float(slip.sum()/stance_samples.sum().clamp_min(1)),
                action_delta_rms=float((action_delta/count.clamp_min(1)).mean()),foot_lift_max_m=float(foot_peak.max()),
                upper_body_naturalness='NOT EVALUATED: no full-body reference in Stage2')
            if record_diagnostics:
                row['per_environment']=[dict(index=i,survived=bool(survived[i]),success=bool(success[i]),foot_rmse=float(rmse[i]),
                    touchdown_error=float(touch[i]) if touch_count[i]>0 else None,loaded_landings_per_foot=landing_side[i].cpu().tolist(),
                    step_frequency_error_hz=float(frequency_error[i]/frequency_count[i]) if frequency_count[i]>0 else None,
                    mean_horizontal_speed=float(velocity[i]/count[i].clamp_min(1)),speed_error=float(speed_error[i]/count[i].clamp_min(1)),
                    root_roll_max_deg=float(max_roll[i]),root_pitch_max_deg=float(max_pitch[i]),height_deviation_max_m=float(max_height[i])) for i in range(n)]
            row.update(recorded_env_index=0,evaluation_num_envs=n)
            extra={} if not record_diagnostics else dict(xi=env.xi[0].cpu().numpy(),
                **{'command_'+k:v[0].cpu().numpy() for k,v in cc.tensors().items()})
            record=out/(label+'.npz');np.savez_compressed(record,**{k:np.asarray(v) for k,v in trace.items()},control_dt=.02,command=json.dumps(vars(cmd)),summary=json.dumps(row),**extra)
            row['record']=str(record);rows.append(row)
            print('[Newton] '+json.dumps({k:row[k] for k in ('label','episode_lengths','survival_rate','success_rate','foot_rmse','stop_reasons')}),flush=True)
        report=dict(kind='full' if seconds>=10 else 'fast',stage=sampler.stage,rows=rows,wall_seconds=time.monotonic()-started,
            complete=len(rows)==len(evaluation_cases(sampler.stage)) and all(r['complete'] for r in rows),
            actual_contacts='Newton last solver substep, |Fz|>1N, landing debounce .15s',newton_transitions=newton_transitions,no_motion_reference=True,normalization='Frozen, no eval adaptation')
        (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False));return report
    finally:
        g.fixed_command=None;env.eval_mode,env.reset_noise,env.max_episode_steps=old[:3];pi.train(old[3])
        sampler.load_state_dict(saved_sampler);env.done_eval.zero_();env.failed_eval.zero_();env.reset(torch.arange(env.N,device=env.device));restore_rng(saved_rng)
