"""Matched Teacher / Stage15 evaluation; original Newton and unchanged physical guards."""
import argparse,json,sys,time,math,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))

def run(args):
    import numpy as np
    import torch
    from supermimic.stage15 import build,seed_all,tensor_hash
    from supermimic.joint_runtime import json_write,digest
    from supermimic.mdp import flatten_state,unflatten_state
    from supermimic.s54 import JOINT_NAMES
    from supermimic.rotations import so3_log
    from supermimic.foot_trajectory import command_time
    from supermimic.control import bound_action
    from supermimic import networks
    from tools.visualization.policy_preview import gpu_slot
    from tools.validation.evaluate_s54_reference_foot_probe import reference_window
    cfg=json.loads(Path(args.config).read_text());out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise FileExistsError('Evaluation already exists')
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    protocol=dict(kind='Stage15_matched_Newton',phase=ck['phase'],checkpoint=str(Path(args.checkpoint).resolve()),
        checkpoint_sha256=digest(args.checkpoint),teacher_sha256=cfg['teacher_sha256'],
        offsets_s=args.offsets,seconds=args.seconds,seed=args.seed,gate=cfg['gate'],
        oracle_latent_diagnostic_only=args.oracle_latent,
        normalization='Saved training fit; no eval update, clipping or recalibration',
        physical_guards='Union of original Motion + Foot guards; timeout is not failure',
        actor_inputs='Foot Goal230D + proprioceptive history; oracle mode explicitly excluded from acceptance',
        observation_contract=ck['contract']['observation'],
        holdout_walking='NOT EVALUATED: no independent qualified Walking clip')
    json_write(out/'protocol.json',protocol);start_time=time.monotonic()
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Evaluation time budget')));signal.alarm(600)
    with gpu_slot():
        seed_all(args.seed);torch.set_num_threads(4);networks.AMP_DTYPE=torch.bfloat16
        parts=build(ROOT/cfg['teacher'],cfg,'cuda:0',ck)
        mp,fp,motion,mg,mdp,paired,foot_mdp,wm=[parts[k] for k in ('mp','fp','motion','mg','mdp','goals','guard','wm')]
        dev='cuda:0';policies={'motion':mp,'foot':fp}
        for pi in policies.values():pi.eval().requires_grad_(False)
        before={name:tensor_hash(pi.state_dict()) for name,pi in policies.items()}
        ev=out/'evaluation'
        for name in policies:(ev/name).mkdir(parents=True,exist_ok=True)
        reports={name:[] for name in policies};initial={};latent={}
        windows=[reference_window(offset,args.seconds,int(motion.clip_start[0]),int(motion.clip_end[0]),lookahead_s=.2) for offset in args.offsets]
        @torch.no_grad()
        def episode(env,controller,offset,start,H):
            goals=mg if controller=='motion' else paired;pi=policies[controller]
            env.goals=goals;env.max_episode_steps=H;env.reset_noise=False
            torch.manual_seed(args.seed);env.done_eval.zero_();env.failed_eval.zero_()
            env.reset(torch.tensor([0],device=dev),torch.tensor([start],device=dev))
            S=env.state();initial[(controller,offset)]=flatten_state(S).cpu().numpy().copy()
            from supermimic.standing_diagnostics import NewtonGroundContacts
            contacts=NewtonGroundContacts(env)
            command=paired.new_command(S,frames=env.f)
            path=ev/controller/f'{controller}_{args.seconds:g}s_seed{args.seed}_offset{offset:g}.npz'
            trace={k:[] for k in ('state','command_step','reference_frame','action','a_prev','reward',
                'term','timeout','pd_target','contact_count','contact_force_w','reference_foot_p',
                'reference_foot_quat','reference_contact','reference_phase','goal','latent')}
            counts=np.full((1,2),-1);force=np.full((1,2,3),np.nan)
            def capture():
                S=env.state();ref=paired.query(command,command_time(env.ep_step,.02,torch.float64)[:,None])
                values=dict(state=flatten_state(S),command_step=env.ep_step,reference_frame=env.f,
                    reference_foot_p=ref.position[:,0],reference_foot_quat=ref.quaternion[:,0],
                    reference_contact=ref.contact[:,0],reference_phase=ref.swing_phase[:,0])
                for k,v in values.items():trace[k].append(v.detach().cpu().numpy().copy())
                trace['contact_count'].append(counts.copy());trace['contact_force_w'].append(force.copy())
            capture();history=None;reason='running';native={};total=0.;step=0;max_input_z=0.
            while step<H:
                S=env.state();obs=goals.obs(mdp,S,env.a_prev,env.xi,env.f,env.command if controller=='foot' else None)
                history=obs[:,None].repeat(1,pi.hist,1) if history is None else torch.cat((history[:,1:],obs[:,None]),1)
                goal=goals.goal(S,env.f,env.ep_step,env.command if controller=='foot' else None)
                if goal.shape!=(1,730 if controller=='motion' else 230) or not torch.isfinite(goal).all():
                    raise ValueError('Invalid Actor Goal shape/value')
                z=mp.encode(motion.goal(env.f,10)) if args.oracle_latent and controller=='foot' else pi.encode(goal)
                action=bound_action(pi.decode(history,z))
                if not torch.isfinite(action).all():raise FloatingPointError('Nonfinite Actor action')
                max_input_z=max(max_input_z,float(pi.goal_norm(goal).abs().max()))
                if step==0:latent[(controller,offset)]=z.cpu().numpy().copy()
                for k,v in dict(goal=goal,latent=z,action=action,a_prev=env.a_prev,pd_target=env.pd_target(action)).items():
                    trace[k].append(v.detach().cpu().numpy().copy())
                transition=env.step(action);step+=1;S1=env.state();total+=float(transition['r'][0])
                finite=all(bool(torch.isfinite(v).all()) for v in S1)
                mterm=bool(transition['term'][0]);fterm=bool(foot_mdp.termination(S1)[0])
                physical=not finite or mterm or fterm
                timeout=bool(transition['timeout'][0]);counts,force,_=contacts.read()
                trace['reward'].append(transition['r'].cpu().numpy().copy())
                trace['term'].append(np.array([physical]));trace['timeout'].append(np.array([timeout]))
                capture()
                if step%100==0:
                    print(f'[Newton] {controller} offset={offset:g}s: {step}/{H} control steps',flush=True)
                if physical or timeout:
                    flags=[]
                    if not finite:flags.append('nonfinite')
                    if mterm:
                        if bool((S1[0][0,2]-motion.root_p[env.f][0,2]).abs()>mdp.cfg.term_height):flags.append('motion_height')
                        if bool((motion.root_R[env.f][0,2,2]-S1[1][0,2,2]).abs()>mdp.cfg.term_tilt):flags.append('motion_tilt')
                    if fterm:
                        if bool((S1[0][0,2]-foot_mdp.nominal_height).abs()>foot_mdp.cfg.term_height):flags.append('foot_height')
                        if bool(1-S1[1][0,2,2]>foot_mdp.cfg.term_tilt):flags.append('foot_tilt')
                    native=dict(motion_physical=mterm,foot_physical=fterm,nonfinite=not finite)
                    reason='+'.join(flags) if physical else 'timeout';break
            arrays={k:np.stack(v) for k,v in trace.items()}
            np.savez_compressed(path,**arrays,xi=env.xi.cpu().numpy(),episode_lengths=np.array([step]),
                control_dt=.02,task_contract=json.dumps(mdp.contract()),
                **{'command_'+k:v.cpu().numpy() for k,v in command.tensors().items()})
            metrics=measure(arrays)
            from supermimic.standing_diagnostics import paired_predictions
            wm_diagnostic={'status':'NOT EVALUATED: fewer than 32 real steps'}
            if step>=32:
                wm_diagnostic,_=paired_predictions(wm,mdp,None,dict(arrays,xi=env.xi.cpu().numpy(),
                    episode_lengths=np.array([step])),dev,horizons=(1,4,8,16,32),stride=max(1,(step-32)//16))
            row=dict(interface=controller,seed=args.seed,offset_s=offset,duration_s=args.seconds,
                episode_lengths=[step],stop_reasons=[reason],n_failed=int(reason!='timeout'),n_timeout=int(reason=='timeout'),
                norm_return=total/(H*mdp.max_step_reward),prefix_normalized_return=total/(step*mdp.max_step_reward),
                metrics=metrics,wm_paired=wm_diagnostic,native_termination=native,max_abs_normalized_goal=max_input_z,
                trace=str(path),record=str(path),actor_goal_dim=goal.shape[-1],controller=controller)
            print(json.dumps({k:row[k] for k in ('controller','offset_s','norm_return','episode_lengths','stop_reasons')}),flush=True)
            return row

        @torch.no_grad()
        def measure(trace):
            S=unflatten_state(torch.as_tensor(trace['state'][:,0],device=dev))
            frames=torch.as_tensor(trace['reference_frame'][:,0],device=dev)
            p,R,v,w=mdp.kin.forward(*S);feet=paired.foot_indices
            upper=[i for i,n in enumerate(JOINT_NAMES) if not n.startswith('leg_')]
            refp=motion.body_p[frames][:,feet];refR=motion.body_R[frames][:,feet]
            refc=paired.contact[frames]
            loaded=torch.as_tensor(trace['contact_force_w'][1:,0,:,2]>1,device=dev)
            err=(p[:,feet]-refp).norm(dim=-1)
            metrics=dict(foot_position_rmse_m=float(err[1:].square().mean().sqrt()),
                foot_orientation_rmse_deg=float(so3_log(refR[1:].transpose(-1,-2)@R[1:,feet]).norm(dim=-1).square().mean().sqrt()*180/np.pi),
                foot_velocity_rmse_m_s=float((v[1:,feet]-motion.body_v[frames][1:,feet]).square().sum(-1).mean().sqrt()),
                root_xy_world_rmse_m=float((S[0][1:,:2]-motion.root_p[frames][1:,:2]).square().sum(-1).mean().sqrt()),
                root_height_tracking_rmse_m=float((S[0][1:,2]-motion.root_p[frames][1:,2]).square().mean().sqrt()),
                root_pitch_max_abs_deg=float(torch.asin((-S[1][:,2,0]).clamp(-1,1)).abs().max()*180/np.pi),
                root_roll_max_abs_deg=float(torch.atan2(S[1][:,2,1],S[1][:,2,2]).abs().max()*180/np.pi),
                upper_body_pose_rmse_rad=float((S[4][1:,upper]-motion.q[frames][1:,upper]).square().mean().sqrt()),
                joint_pose_rmse_rad=float((S[4][1:]-motion.q[frames][1:]).square().mean().sqrt()),
                upper_velocity_rms_rad_s=float(S[5][1:,upper].square().mean().sqrt()),
                upper_acceleration_rms_rad_s2=float(((S[5][1:,upper]-S[5][:-1,upper])/.02).square().mean().sqrt()),
                contact_state_accuracy=float((loaded==refc[1:]).float().mean()),
                loaded_foot_slip_proxy_m_s=float((v[1:,feet,:2].norm(dim=-1)*loaded).sum()/loaded.sum().clamp_min(1)),
                action_rms=float(np.sqrt(np.mean(trace['action']**2))),
                action_clip_fraction=float(np.mean(np.abs(trace['action'])>=20)),
                latent_saturation_fraction=float(np.mean(np.abs(trace['latent'])>.95)))
            edges=(~refc[:-1])&refc[1:]
            metrics['reference_touchdown_events']=int(edges.sum())
            metrics['reference_touchdown_rmse_m']=float(err[1:][edges].square().mean().sqrt()) if bool(edges.any()) else None
            return metrics

        from isaaclab.app import launch_simulation
        from supermimic.env import TrackingSim,make_sim
        sim=make_sim(dev,args)
        with launch_simulation(sim,args):
            env=TrackingSim(sim,1,motion,mdp,eval_mode=True,reset_noise=False,goals=mg)
            for offset,(start,H) in zip(args.offsets,windows):
                for name in policies:
                    reports[name].append(episode(env,name,offset,start,H))
                    json_write(out/'partial.json',dict(status='INCOMPLETE_UNTIL_SUMMARY',reports=reports))
                gap=float(np.max(np.abs(initial['motion',offset]-initial['foot',offset])))
                if gap>1e-6:raise RuntimeError('Controller initial physical states differ')
        for name,pi in policies.items():
            if before[name]!=tensor_hash(pi.state_dict()):raise RuntimeError('Evaluation changed weights/norms')
            json_write(ev/name/'summary.json',dict(interface=name,iteration=ck['iteration'],checkpoint=str(Path(args.checkpoint).resolve()),
                full=False,rows=reports[name],goal_source='reference_foot' if name=='foot' else 'motion'))
        comparisons=[]
        for m,f in zip(reports['motion'],reports['foot']):
            survived=f['n_failed']==0 and f['episode_lengths'][0]==round(args.seconds/.02)
            mm,fm=m['metrics'],f['metrics']
            passed=survived and f['norm_return']>=m['norm_return']-cfg['gate']['max_nr_drop'] and fm['foot_position_rmse_m']<=mm['foot_position_rmse_m']+cfg['gate']['foot_rmse_margin_m'] and fm['upper_body_pose_rmse_rad']<=mm['upper_body_pose_rmse_rad']+cfg['gate']['upper_pose_margin_rad']
            comparisons.append(dict(offset_s=f['offset_s'],survived=survived,quality_gate=passed,
                initial_state_max_gap=float(np.max(np.abs(initial['motion',f['offset_s']]-initial['foot',f['offset_s']])))))
        result=dict(protocol=protocol,rows=reports,comparisons=comparisons,weights_and_norms_unchanged=True,
            actual_newton_transitions=sum(r['episode_lengths'][0] for rows in reports.values() for r in rows),
            wall_seconds=time.monotonic()-start_time,
            gate_passed=not args.oracle_latent and len(comparisons)==3 and args.seconds==10 and all(r['quality_gate'] for r in comparisons))
        if any(r['initial_state_max_gap']>1e-6 for r in comparisons):raise RuntimeError('Initial states differ')
        json_write(out/'summary.json',result)
        json_write(out/'status.json',dict(status='COMPLETED',gate_passed=result['gate_passed']))
        print(json.dumps(dict(gate_passed=result['gate_passed'],comparisons=comparisons)),flush=True)
    signal.alarm(0)

def main():
    from isaaclab.app import add_launcher_args
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/motion_prior/stage15/lafan1_skill_transfer_3090.json')
    p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--offsets',type=float,nargs='+',default=[0.,60.,120.]);p.add_argument('--seconds',type=float,default=10.)
    p.add_argument('--seed',type=int,default=54);p.add_argument('--oracle-latent',action='store_true')
    add_launcher_args(p);a=p.parse_args()
    if not 0<a.seconds<=10 or len(a.offsets)>3:p.error('Bounded <=3 windows of <=10 seconds')
    run(a)
if __name__=='__main__':main()
