"""Finite independent Newton evaluation plus matched 1/4/8-step WM diagnosis.

Motion/paired evaluations use a fixed clip prefix, not training reset-bin means.
Foot deployment reconstructs no MotionLib. Every record retains real contacts.
"""

# Support both direct execution and python -m from the repository layout.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_root = str(_ToolPath(__file__).resolve().parents[2])
if _tool_root not in _tool_sys.path:
    _tool_sys.path.insert(0, _tool_root)

import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from isaaclab.app import add_launcher_args,launch_simulation
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
p.add_argument('--manifest');p.add_argument('--split',choices=['train','heldout'],default='train')
p.add_argument('--durations',nargs='+',type=float,default=[2,5,10]);p.add_argument('--seeds',nargs='+',type=int,default=[54,55,56])
p.add_argument('--reset_noise',action='store_true');p.add_argument('--controller',choices=['policy','default_pd','ankle_pd'],default='policy')
p.add_argument('--ankle_offset_rad',type=float,default=0.,help='Diagnostic ankle PD setpoint offset, <=0.08rad; no model or reward change')
p.add_argument('--start_seconds',nargs='+',type=float,default=[0.],help='Motion clip offsets; distinct windows are not independent held-out clips')
add_launcher_args(p);a=p.parse_args()
if len(a.seeds)>5 or len(a.durations)>3 or not all(0<d<=10 for d in a.durations):p.error('Use <=5 seeds and <=3 durations, each <=10s')
if len(a.start_seconds)>3 or not all(0<=offset<=200 for offset in a.start_seconds):p.error('Use <=3 motion offsets in [0,200]s')

import numpy as np
import torch
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from supermimic import networks
from supermimic.motion_prior_runtime import reconstruct
from supermimic.env import TrackingSim,make_sim
from supermimic.foot_evaluate import evaluate_foot
from supermimic.foot_trajectory import FootstepCommand,query,command_time
from supermimic.reference_foot import ReferenceFootCommand,ReferenceFootGoalSource
from supermimic.goals import GoalSpec
from supermimic.motion import CONTROL_DT
from supermimic.mdp import flatten_state,unflatten_state
from supermimic.networks import WorldModel
from supermimic.control import bound_action
from supermimic.standing_diagnostics import NewtonGroundContacts,paired_predictions
from supermimic.s54 import JOINT_NAMES,FOOT_BODY_NAMES,COM_BODY


@torch.no_grad()
def motion_episode(env,pi,mdp,goals,seed,duration,clip,path,start_seconds=0.):
    torch.manual_seed(seed);env.done_eval.zero_();env.failed_eval.zero_();env.reset_noise=a.reset_noise
    env.max_episode_steps=round(duration/CONTROL_DT)
    start=mdp.motion.clip_start[clip:clip+1]+round(start_seconds/CONTROL_DT)
    if int(start[0])+env.max_episode_steps>=int(mdp.motion.clip_end[clip]):
        raise ValueError('Evaluation window would reach clip end instead of the requested duration')
    env.reset(torch.tensor([0],device=env.device),start)
    try:contacts=NewtonGroundContacts(env);contact_status='last_substep_solver_contacts_and_world_forces'
    except (RuntimeError,AttributeError,NotImplementedError) as exc:contacts=None;contact_status='UNAVAILABLE: '+str(exc)
    initial=env.command.clone() if goals.is_foot else None
    values={k:[] for k in ('state','command_step','reference_frame','action','a_prev','reward','contact_count','contact_force_w','pd_target')}
    history=None;ret=0.;reason='running'
    while not bool(env.done_eval.all()):
        S=env.state();obs=goals.obs(mdp,S,env.a_prev,env.xi,env.f,env.command)
        history=obs[:,None].repeat(1,pi.hist,1) if history is None else torch.cat((history[:,1:],obs[:,None]),1)
        action=bound_action(pi(history,goals.goal(S,env.f,env.ep_step,env.command))) if a.controller=='policy' else torch.zeros_like(env.a_prev)
        values['state'].append(flatten_state(S).cpu().numpy());values['command_step'].append(env.ep_step.cpu().numpy().copy())
        values['reference_frame'].append(env.f.cpu().numpy().copy())
        values['action'].append(action.cpu().numpy());values['pd_target'].append(env.pd_target(action).cpu().numpy())
        out=env.step(action);ret+=float(out['r'][0])
        values['a_prev'].append(out['a_prev'].cpu().numpy());values['reward'].append(out['r'].cpu().numpy())
        counts,forces,_=contacts.read() if contacts else (np.full((1,2),-1),np.full((1,2,3),np.nan),None)
        values['contact_count'].append(counts);values['contact_force_w'].append(forces)
        if bool(out['done'][0]):
            if out['term'][0]:
                height=abs(float(env.state()[0][0,2]-mdp.motion.root_p[env.f][0,2]))
                reason='height' if height>mdp.cfg.term_height else 'tilt'
            else:reason='timeout'
    values['state'].append(flatten_state(env.state()).cpu().numpy())
    values['command_step'].append(env.ep_step.cpu().numpy().copy());values['reference_frame'].append(env.f.cpu().numpy().copy())
    # Contact arrays are aligned with post-action states; unknown initial frame is explicit.
    values['contact_count'].insert(0,np.full((1,2),-1));values['contact_force_w'].insert(0,np.full((1,2,3),np.nan))
    length=len(values['action'])
    arrays={k:np.stack(v) for k,v in values.items()}
    np.savez_compressed(path,**arrays,xi=env.xi.cpu().numpy(),episode_lengths=np.array([length]),task_contract=json.dumps(mdp.contract()),
        **({'command_'+k:v.cpu().numpy() for k,v in initial.tensors().items()} if initial else {}))
    return dict(seed=seed,clip=mdp.motion.names[clip],start_seconds=start_seconds,duration_s=duration,episode_lengths=[length],stop_reasons=[reason],
        n_failed=int(env.failed_eval.sum()),n_timeout=int(reason=='timeout'),actual_contact_status=contact_status,
        norm_return=ret/(env.max_episode_steps*mdp.max_step_reward),normalized_sim_reward=ret/(length*mdp.max_step_reward))


@torch.no_grad()
def analyze(path,mdp,goals,wm,reference_goals=None):
    with np.load(path,allow_pickle=False) as z:data=dict(z)
    device=next(wm.parameters()).device;steps=len(data['action']);S=unflatten_state(torch.as_tensor(data['state'][:,0],device=device))
    bp,bR,bv,_=mdp.kin.forward(*S);feet=[mdp.kin.tracked_names.index(n) for n in FOOT_BODY_NAMES]
    ids=torch.as_tensor(data['command_step'][:,0],device=device)
    command=None
    if goals.is_foot:
        cls=ReferenceFootCommand if goals.spec.mode=='reference_foot' else FootstepCommand
        command=cls(**{k.removeprefix('command_'):torch.as_tensor(v,device=device) for k,v in data.items() if k.startswith('command_') and k!='command_step'})
    if mdp.is_foot:ref=query(command,command_time(ids,CONTROL_DT,torch.float32)[None])
    elif goals.is_foot:ref=goals.query(command,command_time(ids,CONTROL_DT,torch.float64)[None])
    else:
        frame=torch.as_tensor(data['reference_frame'][:,0],device=device)
        refpos=mdp.motion.body_p[frame][:,feet];refR=mdp.motion.body_R[frame][:,feet];refvel=mdp.motion.body_v[frame][:,feet]
        ref=None
    if ref is not None:
        refpos=ref.position[0];refR=__import__('supermimic.rotations',fromlist=['quat_to_mat']).quat_to_mat(ref.quaternion[0]);refvel=ref.velocity[0]
    from supermimic.rotations import so3_log,mat_to_quat
    pos=(bp[:,feet]-refpos).norm(dim=-1);vel=(bv[:,feet]-refvel).norm(dim=-1)
    ori=so3_log(refR.transpose(-1,-2)@bR[:,feet]).norm(dim=-1)*180/np.pi
    upper=[i for i,n in enumerate(JOINT_NAMES) if not n.startswith('leg_')]
    loaded=torch.as_tensor(data['contact_force_w'][1:,0,:,2]>1.,device=device)
    slip=bv[1:,feet,:2].norm(dim=-1)
    euler=Rotation.from_quat(mat_to_quat(S[1]).cpu().numpy()[:,[1,2,3,0]]).as_euler('xyz',degrees=True)
    metrics=dict(foot_position_rmse_m=float(pos[1:].square().mean().sqrt()),foot_velocity_rmse_m_s=float(vel[1:].square().mean().sqrt()),
        foot_orientation_rmse_deg=float(ori[1:].square().mean().sqrt()),root_height_min_m=float(S[0][:,2].min()),
        root_pitch_max_abs_deg=float(abs(euler[:,1]).max()),root_roll_max_abs_deg=float(abs(euler[:,0]).max()),
        actual_loaded_stance_slip_m_s=float((slip*loaded).sum()/loaded.sum().clamp_min(1)),
        loaded_foot_samples=int(loaded.sum()),upper_velocity_rms_rad_s=float(S[5][:,upper].square().mean().sqrt()),
        upper_acceleration_rms_rad_s2=float(((S[5][1:,upper]-S[5][:-1,upper])/CONTROL_DT).square().mean().sqrt()),
        upper_body_reference_error='NOT_APPLICABLE: random Quintic has no arbitrary full-body reference')
    if mdp.is_foot:metrics['root_height_deviation_max_m']=float((S[0][:,2]-mdp.nominal_height).abs().max())
    if mdp.is_foot:
        from supermimic.rotations import quat_to_mat
        from supermimic.upper_prior import command_features
        forward=quat_to_mat(command.heading)[0,:,0]
        metrics['actual_mean_forward_speed_m_s']=float((S[2][1:]*forward).sum(-1).mean())
        metrics['actual_mean_yaw_rate_rad_s']=float(S[3][1:,2].mean())
        metrics['plan_mean_forward_speed_m_s']=float(command_features(command,ids[:1])[0][0])
        count=int(command.lengths[0]);td=command.touchdown[0,:count]
        occurred=(td>0)&(td<=steps*CONTROL_DT+1e-7)
        event=torch.where(occurred)[0];tick=(td[event]/CONTROL_DT-1e-6).ceil().long().clamp(1,steps)
        side=command.foot[0,event]
        landing_error=(bp[tick,torch.as_tensor(feet,device=device)[side]]-command.target_pos[0,event]).norm(dim=-1)
        metrics['planned_touchdown_events']=len(event)
        metrics['scheduled_touchdown_rmse_m']=float(landing_error.square().mean().sqrt()) if len(event) else None
        metrics['plan_mean_step_frequency_hz']=(float(1/(command.liftoff[0,1:count]-command.liftoff[0,:count-1]).mean()) if count>1 else 0.)
        metrics['plan_mean_forward_step_spacing_m']=(float(((command.target_pos[0,1:count]-command.target_pos[0,:count-1])*forward).sum(-1).mean()) if count>1 else 0.)
        finite_force=np.isfinite(data['contact_force_w'][1:,0]).all()
        metrics['actual_contact_measurement_available']=bool(finite_force)
        # Exclude the unmeasured initial frame. A loaded edge is diagnostic;
        # contact chatter alone does not establish actual stepping success.
        edges=(~loaded[:-1])&loaded[1:]
        metrics['actual_loaded_touchdown_edges']=int(edges.sum()) if finite_force else None
        if not finite_force:metrics['actual_loaded_stance_slip_m_s']=None
    if not mdp.is_foot:
        frame=torch.as_tensor(data['reference_frame'][:,0],device=device)
        metrics['upper_body_pose_rmse_rad']=float((S[4][:,upper]-mdp.motion.q[frame][:,upper]).square().mean().sqrt())
        metrics['upper_body_reference_error']='same clip/time paired reference'
        metrics['body_position_world_rmse_m']=float((bp-mdp.motion.body_p[frame]).square().sum(-1).mean().sqrt())
        metrics['body_orientation_world_rmse_deg']=float(so3_log(mdp.motion.body_R[frame].transpose(-1,-2)@bR).square().sum(-1).mean().sqrt()*180/np.pi)
        metrics['head_pose_rmse_rad']=float((S[4][:,-2:]-mdp.motion.q[frame][:,-2:]).square().mean().sqrt())
        metrics['joint_pose_rmse_rad']=float((S[4]-mdp.motion.q[frame]).square().mean().sqrt())
        metrics['joint_velocity_rmse_rad_s']=float((S[5]-mdp.motion.qd[frame]).square().mean().sqrt())
        metrics['joint_limit_violation_fraction']=float(((S[4]<mdp.kin.q_lo-1e-4)|(S[4]>mdp.kin.q_hi+1e-4)).float().mean())
        metrics['root_orientation_tracking_rmse_deg']=float(so3_log(mdp.motion.root_R[frame].transpose(-1,-2)@S[1]).square().sum(-1).mean().sqrt()*180/np.pi)
        metrics['root_height_tracking_max_m']=float((S[0][:,2]-mdp.motion.root_p[frame,2]).abs().max())
        arms=[j for j,n in enumerate(JOINT_NAMES) if n.startswith('zarm_')]
        metrics['arm_pose_rmse_rad']=float((S[4][:,arms]-mdp.motion.q[frame][:,arms]).square().mean().sqrt())
        metrics['arm_velocity_rmse_rad_s']=float((S[5][:,arms]-mdp.motion.qd[frame][:,arms]).square().mean().sqrt())
        metrics['arm_motion_std_rad']=float(S[4][:,arms].std(0).mean())
        metrics['reference_arm_motion_std_rad']=float(mdp.motion.q[frame][:,arms].std(0).mean())
        if reference_goals is not None:
            measured=torch.as_tensor(data['contact_force_w'][1:,0,:,2]>1.,device=device)
            planned=reference_goals.contact[frame[1:]]
            metrics['measured_contact_accuracy']=float((measured==planned).float().mean())
    elif getattr(mdp,'prior',None) is not None:
        targets,velocities,_,matched=mdp.prior.targets(command.repeat(len(ids)),ids)
        if matched.any():
            metrics['upper_body_pose_rmse_rad']=float((S[4][:,upper][matched]-targets[matched]).square().mean().sqrt())
            metrics['upper_body_velocity_error_rmse_rad_s']=float((S[5][:,upper][matched]-velocities[matched]).square().mean().sqrt())
            metrics['upper_body_reference_error']='matched velocity/phase upper-only template; no lower-body/root trajectory reference'
    if ref is not None:
        contact=torch.as_tensor(data['contact_count'][1:,0]>0,device=device)
        metrics['contact_state_accuracy']=float((contact==ref.contact[0,1:]).float().mean())
    try:prediction,rows=paired_predictions(wm,mdp,command,data,device,stride=4)
    except ValueError as exc:prediction=dict(status='NOT_AVAILABLE',reason=str(exc));rows=[]
    fig,axes=plt.subplots(3,2,figsize=(13,10));axes=axes.flatten();t=np.arange(steps+1)*CONTROL_DT
    axes[0].plot(t,S[0][:,2].cpu());axes[0].set_title('Root height (m)')
    axes[1].plot(t,euler[:,:2],label=['roll','pitch']);axes[1].legend();axes[1].set_title('Root roll / pitch (deg)')
    axes[2].plot(t,pos.cpu(),label=['left','right']);axes[2].legend();axes[2].set_title('Foot position error (m)')
    axes[3].plot(t,data['contact_force_w'][:,0,:,2],label=['left','right']);axes[3].legend();axes[3].set_title('Actual ground force z (N), last substep')
    axes[4].plot(t[:-1],data['action'][:,0],lw=.7);axes[4].set_title('Actions (29 joints)')
    axes[5].plot(t,S[5][:,upper].cpu(),lw=.7);axes[5].set_title('Upper joint velocities (rad/s)')
    for ax in axes:ax.set_xlabel('s');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(path.with_suffix('.png'),dpi=130);plt.close(fig)
    return metrics,prediction


def main():
    started=time.monotonic()
    out=Path(a.out)
    if out.exists():p.error('Output exists; choose a fresh evaluation directory')
    out.mkdir(parents=True);device='cuda:0';torch.backends.cuda.matmul.allow_tf32=True
    checkpoint=torch.load(a.checkpoint,map_location=device,weights_only=False)
    networks.AMP_DTYPE=torch.bfloat16 if checkpoint['args'].get('amp')=='bf16' else None
    kin,motion,goals,mdp,pi=reconstruct(checkpoint,device,a.manifest,a.split)
    if a.controller=='ankle_pd' and not mdp.is_foot:p.error('Ankle PD diagnostic requires an independent Foot task')
    if mdp.is_foot and mdp.cfg.command_mode=='curriculum':
        from supermimic.curriculum import Curriculum,Command
        goals.curriculum=Curriculum(checkpoint['curriculum']['cfg']);goals.curriculum.load_state_dict(checkpoint['curriculum'])
        goals.fixed_command=Command();goals.command_duration=12
    wm=WorldModel().to(device);wm.load_state_dict(checkpoint['wm']);wm.eval()
    reference_goals=(ReferenceFootGoalSource(motion,kin,GoalSpec('reference_foot'),goals.foot_config) if motion is not None else None)
    rows=[];sim=make_sim(device,a)
    with launch_simulation(sim,a):
        env=TrackingSim(sim,1,motion,mdp,goals=goals,eval_mode=True)
        for duration in a.durations:
            for seed in a.seeds:
                clips=range(len(motion.names)) if motion is not None else [None]
                for clip,start_seconds in ((c,s) for c in clips for s in (a.start_seconds if motion is not None else [0.])):
                    name=motion.names[clip] if clip is not None else mdp.cfg.command_mode
                    suffix=f'_offset{start_seconds:g}' if start_seconds else ''
                    path=out/f'{name}_{duration:g}s_seed{seed}{suffix}.npz'
                    result=(evaluate_foot(env,checkpoint,goals,mdp,path,device,duration_s=duration,seed=seed,
                               reset_noise=a.reset_noise,controller=a.controller,diagnostics=True,ankle_offset_rad=a.ankle_offset_rad) if mdp.is_foot else
                            motion_episode(env,pi,mdp,goals,seed,duration,clip,path,start_seconds))
                    metrics,prediction=analyze(path,mdp,goals,wm,reference_goals)
                    row=dict(**result,metrics=metrics,wm_paired=prediction,record=str(path));rows.append(row)
                    (out/'partial_summary.json').write_text(json.dumps(dict(status='INCOMPLETE_UNTIL_SUMMARY',rows=rows),indent=2)+'\n')
                    print(json.dumps(dict(seed=seed,clip=name,duration=duration,steps=result['episode_lengths'],
                        stop=result['stop_reasons'],foot_rmse=metrics['foot_position_rmse_m'],
                        wm8={k:v for k,v in prediction.get('horizons',{}).get('8',{}).items()
                             if k in ('base_height_rmse_m','root_pitch_rmse_deg','feet_position_rmse_m','paired_reward_bias')})),flush=True)
    gates=dict(provenance='actual_newton',all_timeouts=all(r['n_failed']==0 and r['n_timeout']==1 for r in rows),
               evaluated_durations=a.durations,evaluated_seeds=a.seeds,passed=False)
    wm8=[r['wm_paired']['horizons']['8'] for r in rows if 'horizons' in r['wm_paired']]
    if wm8:gates['wm_metrics']={k:max(m[k] for m in wm8) for k in ('base_height_rmse_m','root_pitch_rmse_deg','feet_position_rmse_m','paired_reward_mae')}
    # Explicit predeclared Standing + WM gate, not a reward or training-loss threshold.
    gates['passed']=(mdp.is_foot and mdp.cfg.command_mode=='standing' and a.reset_noise and len(set(a.seeds))>=3 and set(a.durations)=={2,5,10}
        and gates['all_timeouts'] and len(wm8)==len(rows) and all(r['metrics']['root_pitch_max_abs_deg']<10 and r['metrics']['root_roll_max_abs_deg']<10
            and r['metrics']['root_height_deviation_max_m']<.05 and r['metrics']['foot_position_rmse_m']<.035
            and r['metrics']['foot_orientation_rmse_deg']<10 and r['metrics']['actual_contact_measurement_available']
            and r['metrics']['actual_loaded_stance_slip_m_s']<.02
            and r['metrics']['loaded_foot_samples']>=1.6*sum(r['episode_lengths']) for r in rows)
        and gates['wm_metrics']['base_height_rmse_m']<.01 and gates['wm_metrics']['root_pitch_rmse_deg']<1)
    report=dict(checkpoint=a.checkpoint,split=a.split,reset_noise=a.reset_noise,wall_seconds=time.monotonic()-started,
                eval_real_steps=sum(sum(r['episode_lengths']) for r in rows),rows=rows,standing_gates=gates,
                teacher_required_at_deployment=False,motion_reference_required_at_deployment=goals.spec.mode=='reference_foot',long_training_authorized=False)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(out/'summary.json')


if __name__=='__main__':main()
