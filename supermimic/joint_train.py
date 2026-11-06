"""Same-process Motion/Foot Appendix-B training, with bounded serial evaluation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    import argparse,csv,json,os,random,signal,subprocess,time,traceback,math
    from datetime import datetime,timezone
    from isaaclab.app import add_launcher_args,launch_simulation
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',required=True);p.add_argument('--run_dir',required=True);p.add_argument('--resume')
    p.add_argument('--smoke_stop_iteration',type=int)
    add_launcher_args(p);a=p.parse_args()
    import numpy as np,torch
    from dataclasses import asdict
    from supermimic import networks
    from supermimic.joint_runtime import build,contract,assert_resume,json_write,profile_hash
    from supermimic.joint_env import JointScene
    from supermimic.joint_updates import JointUpdates
    from supermimic.env import make_sim
    from supermimic.joint_replay import JointReplay
    from supermimic.mdp import unflatten_state,normalize_xi
    from supermimic.control import bound_action
    from supermimic.curriculum_runtime import atomic_save,rng_state,restore_rng
    started=time.monotonic();cfg=json.loads(Path(a.profile).read_text());directory=Path(a.run_dir).resolve()
    if directory.parent!=ROOT/'logs' or not directory.name.startswith('s54_joint_ab1_'):raise ValueError('Protected output path')
    if a.smoke_stop_iteration and cfg['mode']!='smoke':raise ValueError('Smoke-only interruption option')
    if not a.resume and (directory/'last.pt').exists():raise FileExistsError('Existing training protected')
    directory.mkdir(parents=True,exist_ok=True);(directory/'trainer.pid').write_text(str(os.getpid())+'\n')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    torch.manual_seed(cfg['seed']);random.seed(cfg['seed']);np.random.seed(cfg['seed'])
    networks.AMP_DTYPE=torch.bfloat16 if cfg['amp']=='bf16' else None
    stopped=False
    def request_stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    objects=build(cfg,'cuda:0');pi,wm,critics,opts=[objects[k] for k in ('policy','wm','critics','optimizers')]
    mdps,goals=objects['mdps'],objects['goals'];expected=contract(cfg,objects)
    json_write(directory/'transfer_mapping.json',objects['mapping'])
    counts={n:cfg['num_envs_'+n] for n in ('motion','foot')}
    buffers={n:JointReplay(cfg['buffer_steps'],counts[n],'cuda:0',interface=n) for n in counts}
    updates=JointUpdates(pi,wm,critics,opts,mdps,goals,buffers,cfg)
    counters={k:0 for k in ['real_motion','real_foot','wm_updates','wm_windows_motion','wm_windows_foot',
        'wm_valid_steps_motion','wm_valid_steps_foot','actor_motion','actor_foot','critic_motion','critic_foot',
        'imagined_motion','imagined_foot','recovery_real_motion','recovery_real_foot','eval_real_motion','eval_real_foot','events']}
    current=0;step_acc=0.;wall_offset=0.;histories={};baseline=None;best_foot=None;retention_failures=0
    eval_history=[];status='RUNNING';reason=None;last_eval=0.;last_full=0.;last_save=0.
    instability_streak={name:0 for name in counts}
    recovery_until=0;resume_ck=None
    if a.resume:
        resume_ck=torch.load(a.resume,map_location='cuda:0',weights_only=False);assert_resume(resume_ck,expected)
        pi.load_canonical(resume_ck['policy']);wm.load_state_dict(resume_ck['wm'])
        for n in critics:critics[n].load_state_dict(resume_ck['critics'][n])
        for n in opts:opts[n].load_state_dict(resume_ck['optimizers'][n])
        current=resume_ck['iteration'];step_acc=resume_ck['step_acc'];wall_offset=resume_ck['wall_seconds'];counters.update(resume_ck['counters'])
        histories=resume_ck['histories'];baseline=resume_ck['motion_baseline'];best_foot=resume_ck['best_foot_score']
        retention_failures=resume_ck['retention_failures'];last_eval=resume_ck['last_eval'];last_full=resume_ck['last_full']
        eval_history=resume_ck['eval_history'];recovery_until=current+cfg['recovery_wm_iters']
        instability_streak=resume_ck.get('instability_streak',instability_streak)
        events_path=directory/'interface_metrics.jsonl'
        if events_path.exists():
            for line in events_path.read_text().splitlines(keepends=True):
                if line.endswith('\n'):counters['events']=max(counters['events'],json.loads(line)['event_step'])
    elapsed=lambda:wall_offset+time.monotonic()-started
    def emit(values):
        counters['events']+=1
        with (directory/'interface_metrics.jsonl').open('a') as f:
            f.write(json.dumps(dict(iteration=current,event_step=counters['events'],wall_seconds=elapsed(),
                values=dict(values,**{'Progress/Iteration':current,'Progress/WallHours':elapsed()/3600,'Counters/events':counters['events']})),allow_nan=False)+'\n');f.flush()
    def record(**extra):
        json_write(directory/'training_status.json',dict(status=status,iteration=current,wall_seconds=elapsed(),
            wall_budget_seconds=cfg['wall_seconds'],reason=reason,counters=counters,**extra))
    sim=make_sim('cuda:0',a)
    with launch_simulation(sim,a):
        scene=JointScene(sim,counts,objects['motion'],mdps,goals,cfg['gamma'])
        if resume_ck:
            scene.restore(resume_ck['environment']);restore_rng(resume_ck['rng'])
            del resume_ck
        else:
            for n in counts:histories[n]=torch.zeros(counts[n],cfg['hist'],99,device='cuda:0')
        metrics={n:{} for n in counts};collected=0;collection_seconds=0.;agg={}
        def observe(key,value):agg.setdefault(key,[]).append(float(value))
        @torch.no_grad()
        def collect(steps,recovery=False):
            nonlocal collected,collection_seconds
            begin=time.monotonic()
            for index in range(steps):
                S=scene.base.state();actions={};goal_before={}
                for n,ids in scene.ids.items():
                    local=tuple(v[ids] for v in S);cmd=scene.command(n)
                    o=mdps[n].add_noise(goals[n].obs(mdps[n],local,scene.base.a_prev[ids],scene.base.xi[ids],scene.base.f[ids],cmd))
                    fresh=scene.base.ep_step[ids]==0
                    histories[n]=torch.where(fresh[:,None,None],o[:,None].expand(-1,cfg['hist'],-1),torch.cat((histories[n][:,1:],o[:,None]),1))
                    goal=goals[n].goal(local,scene.base.f[ids],scene.base.ep_step[ids],cmd);policy=getattr(pi,n)
                    policy.obs_norm.update(o);policy.goal_norm.update(goal);goal_before[n]=goal
                    actions[n]=bound_action(policy(histories[n],goal)+cfg['action_noise']*torch.randn_like(scene.base.a_prev[ids]))
                out=scene.step(actions)
                for n,tr in out.items():
                    buffers[n].add(tr);counters['real_'+n]+=counts[n]
                    if recovery:counters['recovery_real_'+n]+=counts[n]
                    wm.in_norm.update(wm.inputs(unflatten_state(tr['s']),tr['a'],normalize_xi(tr['xi'])))
                    wm.out_norm.update(wm.targets(unflatten_state(tr['s']),unflatten_state(tr['s1'])))
                    observe(n+'/sim_rew',tr['r'].mean()/mdps[n].max_step_reward)
                    observe(n+'/physical_count',tr['term'].sum());observe(n+'/done_count',tr['done'].sum())
                    observe(n+'/clip_end_count',(tr['clip_end']&~tr['term']).sum())
                    if 'ep_len' in tr:
                        observe(n+'/ep_len',tr['ep_len'].mean());observe(n+'/ep_ret_norm',tr['ep_ret_norm'].mean())
                    if n=='foot' and index==0:
                        g=goals[n].goal(unflatten_state(tr['s']),tr['f'],tr['ep_step'],tr['command'])
                        r=mdps[n].reward(unflatten_state(tr['s1']),tr['f']+1,tr['a'],tr['a_prev'],command=tr['command'],ep_step=tr['ep_step']+1)
                        ge=float((g-goal_before[n]).abs().max());re=float((r-tr['r']).abs().max())
                        if ge>1e-5 or re>1e-5:raise RuntimeError('Foot replay/command/reset clock mismatch')
                        observe('foot/goal_sync_error',ge);observe('foot/reward_sync_error',re)
                collected+=sum(counts.values())
            collection_seconds+=time.monotonic()-begin

        def save(label='last',selection=None):
            nonlocal last_save
            # Refuse to publish a corrupt checkpoint as last/best.
            for module in [pi,wm,*critics.values()]:
                if not all(torch.isfinite(v).all() for v in module.state_dict().values() if v.is_floating_point()):raise FloatingPointError('Nonfinite checkpoint; previous last retained')
            value=dict(joint_contract=expected,profile=cfg,iteration=current,wall_seconds=elapsed(),step_acc=step_acc,
                counters=dict(counters),policy=pi.canonical_state(),wm=wm.state_dict(),critics={n:v.state_dict() for n,v in critics.items()},
                optimizers={n:o.state_dict() for n,o in opts.items()},histories=histories,environment=scene.checkpoint(),rng=rng_state(),
                replay_meta={n:dict(contract=b.contract(),size=b.size,ptr=b.ptr,stored_commands=len(b.command_store)) for n,b in buffers.items()},
                resolved_foot_config=json.loads(json.dumps(asdict(goals['foot'].foot_config))),motion_baseline=baseline,best_foot_score=best_foot,
                retention_failures=retention_failures,last_eval=last_eval,last_full=last_full,eval_history=eval_history,
                selection=selection,
                instability_streak=instability_streak,
                source_snapshot=json.loads((directory/'source_snapshot/manifest.json').read_text()) if (directory/'source_snapshot/manifest.json').exists() else None)
            target=directory/(label+'.pt');atomic_save(value,target)
            if label=='last':last_save=elapsed()
            json_write(directory/'runtime_stats.json',dict(iteration=current,gpu=torch.cuda.get_device_name(),
                peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
                device_used_snapshot_MiB=(torch.cuda.mem_get_info()[1]-torch.cuda.mem_get_info()[0])/2**20,
                collection_fps=collected/max(collection_seconds,1e-9),wall_seconds=elapsed(),counters=counters))
            return target

        def child_eval(interface,out,full=False,video=False):
            remaining=cfg['wall_seconds']-elapsed()-45
            timeout=min(cfg['full_evaluation_timeout'] if full else cfg['evaluation_timeout'],remaining)
            if timeout<30:return None
            torch.cuda.synchronize()
            command=[sys.executable,'-u',str(ROOT/'supermimic/joint_evaluate.py'),'--checkpoint',str(directory/'last.pt'),
                     '--interface',interface,'--out',str(out),'--viz','none']+(['--full'] if full else [])+(['--video'] if video else [])
            # Main loop is blocked and CUDA is synchronized. This child is
            # scheduled serially, never concurrently with trainer GPU work.
            with (directory/'evaluation.log').open('a') as stream:
                stream.write(' '.join(command)+'\n');stream.flush()
                proc=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                record(phase='serial_evaluation',evaluation_pid=proc.pid)
                deadline=time.monotonic()+timeout
                while proc.poll() is None:
                    if stopped or time.monotonic()>=deadline:
                        os.killpg(proc.pid,signal.SIGTERM)
                        try:proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                        if stopped:
                            out.mkdir(parents=True,exist_ok=True)
                            json_write(out/'interrupted.json',dict(status='NOT EVALUATED',reason='trainer safe-stop request'))
                            return None
                        raise RuntimeError('Bounded '+interface+' evaluation timed out; see evaluation.log')
                    time.sleep(1)
                code=proc.returncode
            if code:raise RuntimeError(interface+' evaluator failed '+str(code)+'; see evaluation.log')
            report=json.loads((out/'summary.json').read_text());counters['eval_real_'+interface]+=report['eval_real_steps']
            return report

        def evaluate(full=False,video=False):
            nonlocal baseline,best_foot,retention_failures,last_eval,last_full,status,reason
            save();stamp=f'iteration_{current:06d}_'+str(time.time_ns());root=directory/'evaluation'/stamp
            root.mkdir(parents=True,exist_ok=True)
            values={};reports={}
            for name in (('motion',) if full else ('motion','foot')):
                if stopped:break
                report=child_eval(name,root/name,full,video)
                if report is None:continue
                reports[name]=report
                for row in report['rows']:
                    prefix='Eval/'+name.capitalize()+'/'+('Full/' if full else f'{row["duration_s"]:g}s/')
                    suffix='/offset'+str(row.get('offset_s',0)) if name=='motion' else '/seed'+str(row['seed'])
                    values[prefix+'NormalizedReturn'+suffix]=row['norm_return']
                    values[prefix+'PhysicalTermination'+suffix]=row['n_failed']
                    for k,v in row['metrics'].items():
                        if isinstance(v,(float,int)) and math.isfinite(v):values[prefix+k+suffix]=v
                    for horizon,pair in row['wm_paired'].get('horizons',{}).items():
                        for key in ('base_height_rmse_m','root_pitch_rmse_deg','feet_position_rmse_m','paired_reward_mae'):
                            values[prefix+'WM'+horizon+'/'+key+suffix]=pair[key]
                if name=='motion':
                    score=sum(r['norm_return'] for r in report['rows'])/len(report['rows'])
                    if full:
                        values['Eval/Motion/FullRetentionThreshold']=.820197-.05
                        bad=score<.820197-.05
                    else:
                        if baseline is None:
                            baseline=dict(mean=score,rows=[dict(offset=r['offset_s'],return_=r['norm_return']) for r in report['rows']],path=str(root/'motion/summary.json'))
                        values['Eval/Motion/TeacherSameProbeReturn']=baseline['mean']
                        bad=score<baseline['mean']-cfg['motion_retention_drop'] or any(r['n_failed'] for r in report['rows'])
                    retention_failures=retention_failures+1 if bad else 0
                    values['Gate/MotionRetentionFailures']=retention_failures
                    if not bad:save('last_good_motion',dict(evaluation=str(root/'motion/summary.json'),mean_return=score,full=full))
                    if retention_failures>=cfg['motion_retention_patience']:
                        status='STOPPED_BY_GATE';reason='Motion forgetting relative to identical-protocol baseline'
                else:
                    survival={}
                    for duration in cfg['foot_eval_durations']:
                        rows=[r for r in report['rows'] if r['duration_s']==duration]
                        rate=sum(r['n_failed']==0 and r['episode_lengths'][0]>=round(duration/.02) for r in rows)/len(rows)
                        survival[duration]=rate;values[f'Eval/Foot/{duration:g}s/SurvivalRate']=rate
                    rows=[r for r in report['rows'] if r['duration_s']==10]
                    speed=sum(r['metrics']['actual_forward_speed_m_s'] for r in rows)/len(rows)
                    values['Gate/FootProgress']=int(survival[10]>=2/3 and speed>=-.01)
                    score=100*survival[10]+10*survival[5]+survival[2]+.01*sum(r['norm_return'] for r in rows)/len(rows)
                    if best_foot is None or score>best_foot:
                        best_foot=score;save('best_foot',dict(evaluation=str(root/'foot/summary.json'),score=score,qualified_foot_progress=values['Gate/FootProgress']))
            if full:last_full=elapsed()
            else:last_eval=elapsed()
            eval_history.append(dict(iteration=current,full=full,path=str(root),interfaces=list(reports)))
            json_write(root/'summary.json',dict(iteration=current,full=full,metrics=values,reports={n:str(root/n/'summary.json') for n in reports}))
            with (directory/'wm_pair_metrics.jsonl').open('a') as f:f.write(json.dumps(dict(iteration=current,path=str(root),full=full))+'\n')
            emit(values);save();record(phase='training')

        code=1
        try:
            record(phase='initializing')
            if not a.resume:
                save('initial');save()
                if cfg.get('evaluate_initial',cfg['mode']=='formal'):evaluate(video=cfg['mode']=='formal')
            collect(cfg['warmup_steps'],recovery=bool(a.resume))
            begin_iteration=current;begin_loop=elapsed();agg={}
            while current<cfg['iterations']:
                if stopped:status='STOPPED_BY_REQUEST';reason='SIGINT/SIGTERM';break
                if status=='STOPPED_BY_GATE':break
                if elapsed()>=cfg['wall_seconds']-cfg['final_reserve_seconds']:
                    status='COMPLETED_BY_WALL_CAP';reason='Training stopped with reserved final eval/checkpoint time';break
                if a.smoke_stop_iteration and current>=a.smoke_stop_iteration:
                    status='STOPPED_BY_REQUEST';reason='Bounded smoke interruption for actual resume test';break
                scale=.03+.97*.5*(1+math.cos(math.pi*min(current,cfg['lr_decay_iters'])/cfg['lr_decay_iters']))
                for opt in opts.values():
                    for group in opt.param_groups:group['lr']=group['initial_lr']*scale
                step_acc+=1/cfg['utd'];n=int(step_acc);step_acc-=n;collect(n)
                for _ in range(cfg['wm_batches']):
                    result=updates.wm_update();counters['wm_updates']+=1
                    for name in counts:
                        for key,value in result[name].items():
                            if key.startswith('wm_'):observe(name+'/'+key,value)
                        counters['wm_windows_'+name]+=result[name]['sampled_windows'];counters['wm_valid_steps_'+name]+=result[name]['valid_prediction_steps']
                if current>=max(cfg['wm_only_iters'],recovery_until):
                    for name in ('foot','motion'):
                        result=updates.actor_critic(name,audit=(current+1)%cfg['gradient_audit_every']==0 or cfg['mode']=='smoke')
                        counters['actor_'+name]+=1;counters['critic_'+name]+=cfg['critic_epochs']*cfg['critic_minibatches']
                        counters['imagined_'+name]+=cfg['pi_batch_'+name]*cfg['horizon']
                        for key,value in result.items():observe(name+'/'+key,value)
                        unstable=result['actor_gradnorm']>cfg.get('max_actor_gradnorm',1000) or result['critic_loss']>cfg.get('max_critic_loss',10000)
                        instability_streak[name]=instability_streak[name]+1 if unstable else 0
                        if instability_streak[name]>=cfg.get('unstable_update_patience',10):
                            status='STOPPED_BY_GATE';reason='Sustained '+name+' Actor/Critic instability; thresholds fixed in profile'
                        if 'delta_decoder' in result:
                            with (directory/'gradient_audit.jsonl').open('a') as f:f.write(json.dumps(dict(iteration=current+1,interface=name,**result),allow_nan=False)+'\n')
                        if status=='STOPPED_BY_GATE':break
                current+=1
                if current%cfg['log_every']==0 or current==cfg['iterations']:
                    data={key:sum(v)/len(v) for key,v in agg.items()}
                    for name in counts:
                        data[name+'/term_frac']=sum(agg.get(name+'/physical_count',[0]))/max(1,sum(agg.get(name+'/done_count',[0])))
                    data.update({'Performance/CollectionFPS':collected/max(collection_seconds,1e-9),
                        'Performance/IterationsPerSecond':(current-begin_iteration)/max(elapsed()-begin_loop,1e-9),
                        'Performance/PeakTorchMiB':torch.cuda.max_memory_allocated()/2**20,
                        **{'Counters/'+k:v for k,v in counters.items()},
                        **{'LR/'+n:opt.param_groups[0]['lr'] for n,opt in opts.items()}})
                    emit(data)
                    # JSONL holds sparse diagnostics; CSV keeps stable core columns.
                    row={'iteration':current,'wall_h':elapsed()/3600}
                    core=['sim_rew','ep_ret_norm','ep_len','term_frac','wm_loss','wm_err8','actor_loss','critic_loss','imagined_reward','actor_gradnorm','pure_foot_gradnorm','latent_gradnorm']
                    for name in counts:
                        for key in core:row[name+'_'+key]=data.get(name+'/'+key,'')
                    exists=(directory/'train_log.csv').exists()
                    with (directory/'train_log.csv').open('a',newline='') as f:
                        writer=csv.DictWriter(f,fieldnames=list(row))
                        if not exists:writer.writeheader()
                        writer.writerow(row);f.flush()
                    print(f'joint it {current} | {elapsed()/3600:.3f} h | '+ ' | '.join(f'{name}: wm {data.get(name+"/wm_loss",0):.4g} sim {data.get(name+"/sim_rew",0):.3f} ep {data.get(name+"/ep_len",0):.1f} grad {data.get(name+"/actor_gradnorm",0):.3g}' for name in counts)+f' | {data["Performance/IterationsPerSecond"]:.2f} it/s',flush=True)
                    agg={};record(phase='training')
                if current%cfg['checkpoint_every']==0 or elapsed()-last_save>=cfg['checkpoint_seconds']:
                    save();save(f'ckpt_{current:06d}')
                    old=sorted(directory.glob('ckpt_*.pt'))
                    for path in old[:-cfg['keep_checkpoints']]:path.unlink()
                if cfg['mode']=='formal' and elapsed()-last_eval>=cfg['evaluation_seconds'] and cfg['wall_seconds']-elapsed()>cfg['evaluation_timeout']*2+60:
                    evaluate()
                if cfg['mode']=='formal' and elapsed()-last_full>=cfg['full_evaluation_seconds'] and cfg['wall_seconds']-elapsed()>cfg['full_evaluation_timeout']+60:
                    evaluate(full=True,video=True)
            if status=='RUNNING':status='COMPLETED_BY_ITER_CAP'
            if cfg['mode']=='formal' and status.startswith('COMPLETED') and cfg['wall_seconds']-elapsed()>90:
                evaluate(video=True)
                if status.startswith('COMPLETED') and cfg['wall_seconds']-elapsed()>cfg['full_evaluation_timeout']+45:evaluate(full=True)
            if stopped:status='STOPPED_BY_REQUEST';reason='SIGINT/SIGTERM'
            code=0 if status.startswith('COMPLETED') else 42 if status=='STOPPED_BY_GATE' else 130
        except BaseException as error:
            status='CRASHED';reason=type(error).__name__+': '+str(error)
            (directory/('fault_'+str(time.time_ns())+'.log')).write_text(traceback.format_exc())
            traceback.print_exc();code=1
        finally:
            try:
                emit({'Counters/'+key:value for key,value in counters.items()});save();record(phase='terminal')
            except BaseException:
                (directory/'checkpoint_save_failed.log').write_text(traceback.format_exc());status='CRASHED';code=1;record(phase='checkpoint_failure')
            (directory/'train.exitcode').write_text(str(code)+'\n')
            print('Joint training terminal:',status,reason,'iteration',current,flush=True)
        raise SystemExit(code)


if __name__=='__main__':main()
