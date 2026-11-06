"""Read-only Newton trace metrics for joint experiments; no optimizer import."""
import json
from pathlib import Path
import numpy as np
import torch
from .mdp import flatten_state,unflatten_state
from .motion import CONTROL_DT
from .control import bound_action
from .standing_diagnostics import NewtonGroundContacts,paired_predictions
from .rotations import so3_log,yaw_mat,mat_to_quat,quat_to_mat
from .foot_trajectory import FootstepCommand,query,command_time
from .s54 import JOINT_NAMES,FOOT_BODY_NAMES


def paired_summary(wm,mdp,command,data,device,stride):
    summary,windows=paired_predictions(wm,mdp,command,data,device,horizons=(1,4,8,16,32),stride=stride)
    if not isinstance(summary,dict) or set(summary.get('horizons',{}))!={'1','4','8','16','32'}:
        raise ValueError('Unexpected paired WM diagnostic contract')
    return summary


@torch.no_grad()
def motion_episode(env,policy,mdp,goals,seed,duration,offset,path):
    torch.manual_seed(seed);env.done_eval.zero_();env.failed_eval.zero_();env.reset_noise=False
    start=round(offset/CONTROL_DT)+int(mdp.motion.clip_start[0])
    H=min(round(duration/CONTROL_DT),int(mdp.motion.clip_end[0])-start)
    env.max_episode_steps=H;env.reset(torch.tensor([0],device=env.device),torch.tensor([start],device=env.device))
    contact=NewtonGroundContacts(env)
    trace={k:[] for k in ['state','command_step','reference_frame','action','a_prev','reward','term','timeout','pd_target','contact_count','contact_force_w']}
    history=None;reason='running';total=0.
    while not bool(env.done_eval.all()):
        S=env.state();o=goals.obs(mdp,S,env.a_prev,env.xi,env.f)
        history=o[:,None].repeat(1,policy.hist,1) if history is None else torch.cat((history[:,1:],o[:,None]),1)
        action=bound_action(policy(history,goals.goal(S,env.f,env.ep_step)))
        for key,value in [('state',flatten_state(S)),('command_step',env.ep_step),('reference_frame',env.f),('action',action),('pd_target',env.pd_target(action))]:trace[key].append(value.cpu().numpy().copy())
        out=env.step(action);total+=float(out['r'][0])
        for k in ['reward','a_prev','term','timeout']:trace[k].append(out['r' if k=='reward' else k].cpu().numpy().copy())
        count,force,_=contact.read();trace['contact_count'].append(count);trace['contact_force_w'].append(force)
        if bool(out['done'][0]):
            if bool(out['term'][0]):
                dz=float((env.state()[0][0,2]-mdp.motion.root_p[env.f][0,2]).abs())
                reason='height' if dz>mdp.cfg.term_height else 'tilt'
            else:reason='clip_end' if int(env.f[0])>=int(mdp.motion.clip_end[0]) else 'timeout'
    for key,value in [('state',flatten_state(env.state())),('command_step',env.ep_step),('reference_frame',env.f)]:trace[key].append(value.cpu().numpy().copy())
    trace['contact_count'].insert(0,np.full((1,2),-1));trace['contact_force_w'].insert(0,np.full((1,2,3),np.nan))
    n=len(trace['action']);np.savez_compressed(path,**{k:np.stack(v) for k,v in trace.items()},xi=env.xi.cpu().numpy(),
        episode_lengths=np.array([n]),task_contract=json.dumps(mdp.contract()),control_dt=CONTROL_DT)
    return dict(interface='motion',seed=seed,offset_s=offset,duration_s=H*CONTROL_DT,episode_lengths=[n],stop_reasons=[reason],
        n_failed=int(reason in ('height','tilt','nonfinite')),n_timeout=int(reason=='timeout'),n_clip_end=int(reason=='clip_end'),
        norm_return=total/(H*mdp.max_step_reward))


@torch.no_grad()
def analyze_trace(path,mdp,goals,wm):
    with np.load(path,allow_pickle=False) as z:data=dict(z)
    device=next(wm.parameters()).device;S=unflatten_state(torch.as_tensor(data['state'][:,0],device=device))
    bp,bR,bv,_=mdp.kin.forward(*S);feet=[mdp.kin.tracked_names.index(n) for n in FOOT_BODY_NAMES]
    upper=[i for i,n in enumerate(JOINT_NAMES) if not n.startswith('leg_')]
    loaded=torch.as_tensor(data['contact_force_w'][1:,0,:,2]>1,device=device)
    metrics=dict(root_roll_max_abs_deg=float(torch.atan2(S[1][:,2,1],S[1][:,2,2]).abs().max()*180/np.pi),
        root_pitch_max_abs_deg=float(torch.asin((-S[1][:,2,0]).clamp(-1,1)).abs().max()*180/np.pi),
        upper_velocity_rms_rad_s=float(S[5][:,upper].square().mean().sqrt()),
        upper_acceleration_rms_rad_s2=float(((S[5][1:,upper]-S[5][:-1,upper])/CONTROL_DT).square().mean().sqrt()),
        loaded_foot_samples=int(loaded.sum()),actual_contact_available=bool(np.isfinite(data['contact_force_w'][1:]).all()),
        loaded_foot_slip_proxy_m_s=float((bv[1:,feet,:2].norm(dim=-1)*loaded).sum()/loaded.sum().clamp_min(1)),
        action_rms=float(np.sqrt(np.mean(data['action']**2))),action_clip_fraction=float(np.mean(abs(data['action'])>=20)))
    command=None;steps=len(data['action']);ids=torch.as_tensor(data['command_step'][:,0],device=device)
    if mdp.is_foot:
        command=FootstepCommand(**{k[8:]:torch.as_tensor(v,device=device) for k,v in data.items() if k.startswith('command_') and k!='command_step'})
        ref=query(command,command_time(ids,CONTROL_DT,torch.float32)[None])
        p,R,v=ref.position[0],quat_to_mat(ref.quaternion[0]),ref.velocity[0]
        metrics.update(foot_position_rmse_m=float((bp[1:,feet]-p[1:]).square().sum(-1).mean().sqrt()),
            foot_velocity_rmse_m_s=float((bv[1:,feet]-v[1:]).square().sum(-1).mean().sqrt()),
            foot_orientation_rmse_deg=float(so3_log(R[1:].transpose(-1,-2)@bR[1:,feet]).norm(dim=-1).square().mean().sqrt()*180/np.pi),
            root_height_deviation_max_m=float((S[0][:,2]-mdp.nominal_height).abs().max()),
            contact_state_accuracy=float((loaded==ref.contact[0,1:]).float().mean()),
            actual_loaded_touchdown_edges=int((~loaded[:-1]&loaded[1:]).sum()),
            upper_body_pose_rmse_rad=None,upper_body_pose_status='NOT EVALUATED: no paired whole-body reference for random Quintic')
        forward=quat_to_mat(command.heading)[0,:,0]
        metrics['actual_forward_speed_m_s']=float((S[2][1:]*forward).sum(-1).mean())
        metrics['actual_yaw_rate_rad_s']=float(S[3][1:,2].mean())
        from .upper_prior import command_features
        metrics['planned_forward_speed_m_s']=float(command_features(command,ids[:1])[0][0])
        count=int(command.lengths[0]);td=command.touchdown[0,:count];events=torch.where((td>0)&(td<=steps*CONTROL_DT+1e-7))[0]
        metrics['scheduled_touchdown_events']=len(events)
        if len(events):
            ticks=(td[events]/CONTROL_DT-1e-6).ceil().long().clamp(1,steps);side=command.foot[0,events]
            error=(bp[ticks,torch.as_tensor(feet,device=device)[side]]-command.target_pos[0,events]).norm(dim=-1)
            metrics['scheduled_touchdown_rmse_m']=float(error.square().mean().sqrt())
        else:metrics['scheduled_touchdown_rmse_m']=None
    else:
        frame=torch.as_tensor(data['reference_frame'][:,0],device=device);m=mdp.motion
        dR=yaw_mat(S[1]@m.root_R[frame].transpose(-1,-2));anchor=torch.cat((S[0][:,:2],m.root_p[frame,2:3]),-1)
        reference=anchor[:,None]+(dR[:,None]@(m.body_p[frame]-m.root_p[frame,None])[...,None]).squeeze(-1)
        metrics.update(body_position_aligned_rmse_m=float((bp[1:]-reference[1:]).square().sum(-1).mean().sqrt()),
            body_orientation_aligned_rmse_deg=float(so3_log((dR[:,None]@m.body_R[frame])[1:].transpose(-1,-2)@bR[1:]).norm(dim=-1).square().mean().sqrt()*180/np.pi),
            root_xy_world_rmse_m=float((S[0][1:,:2]-m.root_p[frame][1:,:2]).square().sum(-1).mean().sqrt()),
            root_height_tracking_rmse_m=float((S[0][1:,2]-m.root_p[frame][1:,2]).square().mean().sqrt()),
            upper_body_pose_rmse_rad=float((S[4][1:,upper]-m.q[frame][1:,upper]).square().mean().sqrt()),
            joint_pose_rmse_rad=float((S[4][1:]-m.q[frame][1:]).square().mean().sqrt()))
    if steps>=32:
        # Same observed starts/actions at all horizons; cap compute via stride.
        prediction=paired_summary(wm,mdp,command,data,device,stride=max(1,(steps-32)//32))
    else:prediction=dict(status='NOT EVALUATED',reason='Physical episode shorter than a 32-step common paired window')
    return metrics,prediction
