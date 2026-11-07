"""Isolated generated-foot Stage2; original Foot BPTT/WM math, frozen Decoder."""
import argparse,copy,json,math,os,signal,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))

def main():
    from isaaclab.app import add_launcher_args
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--num-envs',type=int);p.add_argument('--resume');add_launcher_args(p);a=p.parse_args()
    import numpy as np,torch
    from supermimic import networks
    from supermimic.stage15 import seed_all,tensor_hash
    from supermimic.stage2_curriculum import build,Stage2Updates,JointReplay,STAGES
    from supermimic.stage2_evaluation import evaluate
    from supermimic.curriculum_runtime import atomic_save,rng_state,restore_rng
    from supermimic.joint_runtime import json_write,digest
    from supermimic.control import bound_action
    from tools.visualization.policy_preview import gpu_slot
    cfg=json.loads(Path(a.config).read_text())
    if a.num_envs:cfg['num_envs']=a.num_envs
    if a.smoke:cfg.update(iterations=8,wm_only_iters=2,log_every=1,checkpoint_every=4,wall_seconds=600,resume_wm_iters=1)
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'latest.pt').exists() and not a.resume:raise FileExistsError('Existing Stage2 checkpoint; explicit --resume required')
    for key,value in [('hist',10),('horizon',32),('wm_window',8),('goal_frames',10),('goal_dt',.02)]:
        if cfg[key]!=value:raise ValueError('Unverified '+key)
    if cfg['wall_seconds']>28800:raise ValueError('8h hard limit')
    json_write(out/'args.json',cfg);started=time.monotonic();stopping=[False];stop_reason='iteration_limit'
    signal.signal(signal.SIGTERM,lambda *_:stopping.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *_:stopping.__setitem__(0,True))
    counts=dict(newton=0,newton_eval=0,wm=0,actor=0,critic=0);first=it=0;previous_wall=0.;history=None;last={};first_audit=None
    best={};eval_log=[];env=None;status='STARTING';last_save=0.;last_eval=0.;eval_spent=0.;low_survival_evals=0
    saved=torch.load(a.resume,map_location='cpu',weights_only=False) if a.resume else None
    process_started_unix=time.time();run_started_unix=saved.get('run_started_unix',process_started_unix) if saved else process_started_unix
    with gpu_slot():
        seed_all(cfg['seed']);torch.set_num_threads(4);networks.AMP_DTYPE=torch.bfloat16
        if '3090' not in torch.cuda.get_device_name():raise RuntimeError('This release requires the actual RTX 3090; do not mislabel other GPUs')
        parts=build(cfg,'cuda:0',saved);pi,wm,critic,goals,mdp,curr=[parts[k] for k in ('policy','wm','critic','goals','mdp','curriculum')]
        norms=lambda:tensor_hash({k:v for k,v in pi.state_dict().items() if k.startswith(('goal_norm.','obs_norm.'))})
        frozen_hash=tensor_hash(pi.dec.state_dict());norm_hash=norms();encoder_hash=tensor_hash(pi.enc.state_dict());wm_hash=tensor_hash(wm.state_dict())
        opts={'foot':torch.optim.Adam(pi.enc.parameters(),lr=cfg['lr_actor'],betas=(.9,.95)),
              'wm':torch.optim.Adam(wm.parameters(),lr=cfg['lr_wm']),
              'critic_foot':torch.optim.Adam(critic.parameters(),lr=cfg['lr_critic'])}
        if saved:
            if {k:v for k,v in saved['config'].items() if k!='iterations'}!={k:v for k,v in cfg.items() if k!='iterations'}:
                raise ValueError('Resume requires same training profile (iteration cap can increase)')
            for k,v in opts.items():v.load_state_dict(saved['optimizers'][k])
            counts=copy.deepcopy(saved['counts']);first=it=saved['iteration'];previous_wall=saved['wall_seconds']
            best=saved['best'];eval_log=saved['eval_log'];last_eval=saved['last_eval'];eval_spent=saved['eval_spent'];restore_rng(saved['rng'])
        initial_actor=counts['actor'];B=JointReplay(cfg['buffer_steps'],cfg['num_envs'],'cuda:0','foot')
        updates=Stage2Updates(parts,opts,B,cfg)
        transfer=dict(parts['transfer'],decoder_hash=frozen_hash,normalizer_hash=norm_hash,encoder_hash=encoder_hash,wm_hash=wm_hash,
            optimizer_initialization='restored Stage2 only' if saved else 'fresh Adam for E_f / WM / independent Foot Critic',
            trainable_actor_parameters=sum(p.numel() for p in pi.enc.parameters()),frozen_decoder_parameters=sum(p.numel() for p in pi.dec.parameters()))
        json_write(out/'transfer_report.json',transfer)
        elapsed=lambda:previous_wall+time.monotonic()-started
        def checkpoint(tag=None):
            nonlocal last_save
            if tensor_hash(pi.dec.state_dict())!=frozen_hash or norms()!=norm_hash:raise RuntimeError('Decoder/Policy norm drift')
            state=dict(stage2_contract=parts['contract'],config=cfg,iteration=it,wall_seconds=elapsed(),counts=counts,
                run_started_unix=run_started_unix,
                policy=pi.state_dict(),wm=wm.state_dict(),critic=critic.state_dict(),optimizers={k:v.state_dict() for k,v in opts.items()},
                curriculum=copy.deepcopy(curr.state_dict()),rng=rng_state(),best=best,eval_log=eval_log,last_eval=last_eval,eval_spent=eval_spent,
                replay_restore='Replay not serialized: refill warmup_steps on grounded resets, then resume_wm_iters WM-only; simulator solver warm state is fresh',
                environment_snapshot=None if env is None else dict(state=tuple(x.detach().cpu() for x in env.state()),ep_step=env.ep_step.cpu(),command=env.command.clone()),
                stop_reason=stop_reason)
            atomic_save(state,out/'latest.pt')
            if tag:
                path=out/(tag+'.pt');tmp=path.with_suffix('.pt.linktmp');tmp.unlink(missing_ok=True);os.link(out/'latest.pt',tmp);tmp.replace(path)
            if tag and tag.startswith('ckpt_'):
                for old in sorted(out.glob('ckpt_[0-9]*.pt'))[:-4]:old.unlink()
            last_save=elapsed()
        def heartbeat(state,error=None):
            json_write(out/'status.json',dict(status=state,iteration=it,stage=STAGES[curr.stage],wall_seconds=elapsed(),
                first_started_unix=run_started_unix,process_started_unix=process_started_unix,updated_unix=time.time(),
                budget_seconds=cfg['wall_seconds'],counts=counts,stop_reason=stop_reason,error=error,pid=os.getpid(),gpu=torch.cuda.get_device_name(),latest=str(out/'latest.pt')))
        @torch.no_grad()
        def collect(steps):
            nonlocal history
            start=time.monotonic();rewards=[];terms=[];timeouts=[];ep=[];ret=[];err=[];maxgoal=0.
            for _ in range(steps):
                S=env.state();obs=goals.obs(mdp,S,env.a_prev,env.xi,env.f,env.command)
                history=obs[:,None].repeat(1,cfg['hist'],1) if history is None else torch.cat((history[:,1:],obs[:,None]),1)
                reset=env.ep_step==0;history[reset]=obs[reset,None].expand(-1,cfg['hist'],-1)
                g=goals.goal(S,env.f,env.ep_step,env.command)
                if g.shape!=(cfg['num_envs'],230) or not torch.isfinite(g).all():raise FloatingPointError('Invalid generated Goal')
                maxgoal=max(maxgoal,float(pi.goal_norm(g).abs().max()))
                action=bound_action(pi(history,g)+torch.randn(cfg['num_envs'],29,device='cuda:0')*cfg['action_noise'])
                tr=env.step(action);B.add(tr);counts['newton']+=cfg['num_envs'];curr.observe(tr['command'],tr['ep_step'])
                rewards.append(float(tr['r'].mean()/mdp.max_step_reward));terms.append(float(tr['term'].float().mean()));timeouts.append(float((tr['timeout']&~tr['term']).float().mean()))
                err.append(float(tr['terms']['foot_position_error'].mean().sqrt()))
                if 'ep_len' in tr:ep.extend(tr['ep_len'].cpu().tolist());ret.extend(tr['ep_ret_norm'].cpu().tolist())
            return dict(sim_reward=sum(rewards)/len(rewards),termination_fraction=sum(terms)/len(terms),timeout_fraction=sum(timeouts)/len(timeouts),
                collection_fps=steps*cfg['num_envs']/(time.monotonic()-start),foot_rmse=sum(err)/len(err),max_abs_normalized_goal=maxgoal,
                **(dict(episode_length=sum(ep)/len(ep),episode_return=sum(ret)/len(ret)) if ep else {}))
        def run_eval(seconds,tag,cap=180):
            nonlocal history,last_eval,eval_spent,low_survival_evals
            directory=out/'evaluation'/tag
            report=evaluate(env,parts,cfg,directory,seconds,min(time.monotonic()+cap,time.monotonic()+max(0,cfg['wall_seconds']-elapsed()-80)))
            counts['newton_eval']=counts.get('newton_eval',0)+report['newton_transitions']
            history=None;eval_spent+=report['wall_seconds'];last_eval=elapsed()
            if report['complete']:
                rows=report['rows'];standing=next(r for r in rows if r['label']=='standing')
                walking=next(r for r in rows if r['label']=='short_walk')
                # Lexicographic true Newton success, survival, tracking/return score; no WM loss.
                candidates={'standing':[standing],'walking':[r for r in rows if r['command']['length']>0], 'overall':rows}
                for label,rr in candidates.items():
                    if not rr:continue
                    score=(sum(r['success_rate'] for r in rr)/len(rr),sum(r['survival_rate'] for r in rr)/len(rr),
                           -sum(r['foot_rmse'] for r in rr)/len(rr),sum(r['norm_return'] for r in rr)/len(rr))
                    if label not in best or tuple(best[label]['score'])<score:
                        best[label]=dict(score=score,evaluation=str(directory/'summary.json'),seconds=seconds,iteration=it,qualified=seconds>=10 and score[0]>=cfg['success_threshold'])
                        checkpoint('best_'+label)
                low_survival_evals=low_survival_evals+1 if max(standing['survival_rate'],walking['survival_rate'])==0 and max(max(standing['episode_lengths']),max(walking['episode_lengths']))<100 else 0
                changed=curr.update_evaluation(report,elapsed(),full=seconds>=10)
                if changed:checkpoint('stage_'+STAGES[curr.stage]+'_'+str(it))
                last.update(eval_standing_survival=standing['survival_rate'],eval_standing_success=standing['success_rate'],eval_walking_success=walking['success_rate'],eval_walking_survival=walking['survival_rate'],eval_foot_rmse=walking['foot_rmse'])
            eval_log.append(dict(path=str(directory/'summary.json'),iteration=it,wall_seconds=elapsed(),complete=report['complete']))
            return report
        try:
            heartbeat('INITIALIZING')
            from supermimic.env import TrackingSim,make_sim
            from isaaclab.app import launch_simulation
            simcfg=make_sim('cuda:0',a)
            with launch_simulation(simcfg,a):
                env=TrackingSim(simcfg,cfg['num_envs'],None,mdp,gamma=cfg['gamma'],eval_mode=False,max_episode_steps=500,reset_noise=True,goals=goals)
                # No Reference is loaded; all standing/walking episodes begin with the same grounded S54 model pose.
                print('[Stage2] Phase A transfer verified; no MotionLib / Motion Encoder / Motion Goal. D frozen; Goal230 dt=.02.',flush=True)
                initial_states=env.state();assert not bool(mdp.termination(initial_states).any())
                json_write(out/'reset_audit.json',dict(initial_physical_termination_count=int(mdp.termination(initial_states).sum()),nominal_height=float(mdp.nominal_height),reset=mdp.contract()['reset'],domain_randomization=False))
                last=collect(cfg['warmup_steps']);heartbeat('RUNNING')
                checkpoint('initial')
                # Baseline is independent, deterministic Newton, not a training-reward claim.
                run_eval(5.,'baseline_'+str(first),cap=120)
                last.update(collect(cfg['warmup_steps']));history=None
                limit=first+4 if a.resume and a.smoke else cfg['iterations']
                for next_it in range(first+1,limit+1):
                    tick=time.monotonic()
                    if stopping[0]:stop_reason='signal';break
                    if elapsed()>=cfg['wall_seconds']-90:stop_reason='wall_budget';break
                    it=next_it
                    last.update(collect(cfg['rollout_steps']))
                    for _ in range(cfg['wm_batches']):last.update(updates.wm_update());counts['wm']+=1
                    if it>cfg['wm_only_iters'] and (not saved or it>first+cfg['resume_wm_iters']):
                        ramp=min(1.,(counts['actor']+1)/cfg['actor_ramp_iterations'])
                        opts['foot'].param_groups[0]['lr']=cfg['lr_actor']*ramp
                        stats=updates.actor_critic('foot',audit=counts['actor']==initial_actor);last.update(stats)
                        if counts['actor']==initial_actor:first_audit=stats
                        counts['actor']+=1;counts['critic']+=cfg['critic_epochs']*cfg['critic_minibatches']
                    if not a.smoke and elapsed()-last_eval>=cfg['eval_seconds'] and eval_spent<cfg['wall_seconds']*cfg['eval_fraction']-100 and cfg['wall_seconds']-elapsed()>300:
                        full=curr.passes>=1 and elapsed()-curr.stage_started>=cfg['min_stage_seconds']
                        run_eval(10. if full else 5.,'it_'+str(it)+('_full' if full else '_fast'))
                    if not a.smoke and elapsed()>cfg['watchdog_no_survival_seconds'] and low_survival_evals>=3:
                        stop_reason='watchdog_persistent_failure_to_survive';checkpoint('diagnostic_stop');break
                    if not a.smoke and elapsed()>cfg['watchdog_no_progress_seconds'] and len(eval_log)>=3:
                        evaluations=[json.loads(Path(x['path']).read_text()) for x in eval_log[-3:]]
                        baseline=json.loads(Path(eval_log[0]['path']).read_text())
                        base_rmse=sum(r['foot_rmse'] for r in baseline['rows'])/len(baseline['rows'])
                        no_success=all(all(r['success_rate']==0 for r in e['rows']) for e in evaluations)
                        no_tracking_improvement=all(sum(r['foot_rmse'] for r in e['rows'])/len(e['rows'])>.9*base_rmse for e in evaluations)
                        if no_success and no_tracking_improvement:
                            stop_reason='watchdog_no_task_progress';checkpoint('diagnostic_stop');break
                    if not a.smoke and elapsed()>900 and len(eval_log)>=3:
                        evaluations=[json.loads(Path(x['path']).read_text()) for x in eval_log[-2:]]
                        baseline=json.loads(Path(eval_log[0]['path']).read_text())
                        base_rmse=sum(r['foot_rmse'] for r in baseline['rows'])/len(baseline['rows'])
                        if all(e['complete'] and sum(r['foot_rmse'] for r in e['rows'])/len(e['rows'])>2*base_rmse for e in evaluations):
                            stop_reason='watchdog_severe_tracking_regression';checkpoint('diagnostic_stop');break
                    last.update(iteration=it,wall_seconds=elapsed(),stage=curr.stage,curriculum_progress=curr.progress,**counts,
                        peak_torch_mib=torch.cuda.max_memory_allocated()/2**20,iterations_per_second=1/(time.monotonic()-tick))
                    if any(isinstance(v,float) and not math.isfinite(v) for v in last.values()):raise FloatingPointError('Nonfinite telemetry')
                    if it%cfg['log_every']==0 or it==limit:
                        last['task_samples']=dict(curr.counts);last['command_buckets']=dict(curr.hist);last['rejected_commands']=dict(curr.rejected);last['executed_commands']=dict(curr.executed)
                        with (out/'train_metrics.jsonl').open('a') as f:f.write(json.dumps(last,allow_nan=False)+'\n');f.flush()
                        print(f"it {it} | stage C{curr.stage} | {elapsed()/3600:.3f}h | wm {last['wm_loss']:.4f} e8 {last['wm_err8']:.4f} | sim {last['sim_reward']:.3f} foot {last['foot_rmse']:.4f} | pi {last.get('actor_loss',0):.3f} critic {last.get('critic_loss',0):.4f} Ef_grad {last.get('actor_gradnorm',0):.3g} | {last['iterations_per_second']:.2f} it/s | {last['peak_torch_mib']:.0f}MiB",flush=True);heartbeat('RUNNING')
                    if it%cfg['checkpoint_every']==0 or elapsed()-last_save>=cfg['checkpoint_seconds']:checkpoint('ckpt_'+str(it).zfill(6))
                if a.smoke:run_eval(5.,'post_update_'+str(it),cap=120)
                elif cfg['wall_seconds']-elapsed()>200 and not stopping[0]:run_eval(10.,'final_'+str(it),cap=120)
                status='COMPLETED' if stop_reason in ('iteration_limit','wall_budget') else 'STOPPED'
        except BaseException as error:
            stop_reason=type(error).__name__+': '+str(error);status='FAILED';traceback.print_exc();raise
        finally:
            checkpoint('last');heartbeat(status)
            assertions=dict(decoder_unchanged=tensor_hash(pi.dec.state_dict())==frozen_hash,policy_normalizers_unchanged=norms()==norm_hash,
                foot_encoder_updated=tensor_hash(pi.enc.state_dict())!=encoder_hash,wm_updated=tensor_hash(wm.state_dict())!=wm_hash,
                foot_actor_nonzero_gradient=bool(first_audit and first_audit['latent_gradnorm']>0 and first_audit['pure_foot_gradnorm']>0),
                actor_did_not_update_wm=bool(first_audit and first_audit['delta_wm_actor']==0))
            audit=dict(assertions=assertions,first_actor_audit=first_audit,counts=counts,wall_seconds=elapsed(),gpu=torch.cuda.get_device_name(),
                peak_torch_mib=torch.cuda.max_memory_allocated()/2**20,smoke=a.smoke,resume=bool(a.resume),no_motion_reference=True,stop_reason=stop_reason)
            json_write(out/('resume_smoke_audit.json' if saved and a.smoke else 'smoke_audit.json' if a.smoke else 'training_audit.json'),audit)
            json_write(out/'training_summary.json',dict(status=status,iteration=it,counts=counts,best=best,evaluations=eval_log,wall_seconds=elapsed(),stop_reason=stop_reason))
        if a.smoke and not all(assertions.values()):raise RuntimeError('Technical smoke assertions failed: '+str(assertions))
if __name__=='__main__':main()
