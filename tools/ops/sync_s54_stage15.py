"""Core-only Stage15 replay; CPU/tmux, immutable local source and API readback."""
import argparse,fcntl,json,math,os,subprocess,sys,time,uuid
from pathlib import Path

def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def events(directory):
    rows=[];last=0
    for phase in ('a','b'):
        path=directory/f'phase_{phase}/phase_{phase}_metrics.jsonl'
        if not path.exists():continue
        offset=last
        for line in path.read_text().splitlines():
            r=json.loads(line);step=offset+r['iteration'];last=max(last,step)
            names=({'train_latent_mse':'A/TrainLatentMSE','val_latent_mse':'A/ValidationLatentMSE',
                    'val_latent_cosine':'A/ValidationCosine','val_saturation':'A/LatentSaturation','actor_gradnorm':'A/EncoderGradNorm'} if phase=='a' else
                   {'sim_reward':'B/SimulationMotionReward','imagined_reward':'B/ImaginedMotionReward',
                    'wm_loss':'B/WorldModelLoss','wm_err8':'B/WorldModelError8','actor_gradnorm':'B/EncoderGradNorm','critic_loss':'B/CriticLoss'})
            values={v:r[k] for k,v in names.items() if k in r}
            values['Resource/PeakTorchMiB']=r['peak_torch_mib'];values['Progress/PhaseIteration']=r['iteration']
            rows.append(dict(step=step,values=values))
    values={}
    for phase,label in [('eval_a','PhaseA'),('eval_b','PhaseB')]:
        path=directory/phase/'summary.json'
        if not path.exists():continue
        r=json.loads(path.read_text())
        for role,name in [('motion','Teacher'),('foot',label)]:
            if role=='motion' and any(k.startswith('Eval/Teacher/') for k in values):continue
            rr=r['rows'][role];prefix=f'Eval/{name}/'
            values[prefix+'Survival10s']=sum(x['n_failed']==0 and x['episode_lengths'][0]==500 for x in rr)/len(rr)
            values[prefix+'MotionReturn']=sum(x['norm_return'] for x in rr)/len(rr)
            for field,metric in [('FootRMSE','foot_position_rmse_m'),('UpperPoseRMSE','upper_body_pose_rmse_rad')]:
                values[prefix+field]=sum(x['metrics'][metric] for x in rr)/len(rr)
    alignment=directory/'latent_alignment_posthoc.json'
    if alignment.exists():
        records=json.loads(alignment.read_text())['rows']
        for phase,label in [('eval_a','PhaseA'),('eval_b','PhaseB')]:
            selected=[r['latent_mse'] for r in records if r['phase']==phase]
            if selected:values[f'Eval/{label}/LatentAlignmentMSE']=sum(selected)/len(selected)
    if values:rows.append(dict(step=last+1,values=values))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();d=a.run_dir.resolve()
    dest=d/'swanlab';dest.mkdir(exist_ok=True)
    with (dest/'upload.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (d/'status.json').exists() or json.loads((d/'status.json').read_text())['status'] not in ('COMPLETED','BLOCKED_PHASE_A','FAILED'):
            time.sleep(10)
        import swanlab
        cfg=json.loads((d/'profile.json').read_text());ip=dest/'identity.json'
        identity=json.loads(ip.read_text()) if ip.exists() else dict(run_id=uuid.uuid4().hex[:16],project=cfg['swanlab_project'],workspace=cfg['swanlab_workspace'],name=d.name+'_LAFAN1_STAGE15')
        write(ip,identity);ev=events(d);write(dest/'core_events.json',ev)
        if not ev:raise RuntimeError('No formal measurements to upload; smoke intentionally excluded')
        swanlab.merge_settings(swanlab.Settings(interactive=False,probe=swanlab.Settings.Probe(monitor=False)))
        def status(state,**kw):write(dest/'status.json',dict(status=state,**identity,**kw))
        for attempt in range(3):
            try:
                cache=dest/f'offline_attempt{attempt}';status('offline_replay',attempt=attempt)
                swanlab.init(project=identity['project'],workspace=identity['workspace'],id=identity['run_id'],
                    name=identity['name'],mode='offline',log_dir=str(cache),config={k:cfg[k] for k in
                    ('teacher','teacher_sha256','seed','hist','horizon','wm_window','goal_dt','phase_a_iterations','phase_a_batch','phase_a_lr','num_envs','pi_batch','wm_batch','wm_batches','phase_b_iterations','phase_b_budget_s')})
                for row in ev:swanlab.log(row['values'],step=row['step'])
                swanlab.finish();status('syncing',attempt=attempt)
                for path in cache.glob('run-*'):
                    if any(path.glob('run-*.swanlab')):subprocess.run([sys.executable,'-m','swanlab','sync',str(path),'--project',identity['project'],'--workspace',identity['workspace'],'--id',identity['run_id']],check=True,timeout=300)
                run=swanlab.Api().run(f'{identity["workspace"]}/{identity["project"]}/{identity["run_id"]}')
                expected={}
                for row in ev:
                    for key,value in row['values'].items():expected[key]=(row['step'],value)
                result=run.metrics(keys=list(expected),sample=1500,ignore_timestamp=True)
                actual={x['key']:{int(p['index']):float(p['data']) for p in x.get('metrics',[])} for x in result.get('list',[])}
                checks={k:dict(step=s,expected=v,observed=actual.get(k,{}).get(s),verified=math.isclose(actual.get(k,{}).get(s,float('nan')),v,rel_tol=1e-6,abs_tol=1e-7)) for k,(s,v) in expected.items()}
                write(dest/'cloud_readback.json',checks)
                if not all(c['verified'] for c in checks.values()):raise RuntimeError('Cloud metrics readback incomplete')
                status('VERIFIED',url=run.url,metric_count=len(checks),records=len(ev));print(run.url,flush=True);return
            except Exception as error:
                status('RETRY_PENDING',attempt=attempt,error=type(error).__name__+': '+str(error));print(type(error).__name__,str(error),flush=True)
                time.sleep(10)
        status('FAILED',reason='Three bounded sync attempts failed; all offline data retained');raise RuntimeError('Stage15 upload failed')
if __name__=='__main__':main()
