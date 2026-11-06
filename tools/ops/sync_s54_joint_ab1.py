"""Formal joint-only SwanLab follower. Local JSONL is the durable authority.

Runs in its own CPU/tmux session and retries independently of the trainer.
Final core-only offline replay plus CLI sync repairs network interruptions.
Full diagnostics remain local; the cloud allowlist is deliberately closed.
"""
from pathlib import Path
import argparse,json,math,os,re,subprocess,sys,time,fcntl,uuid
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
def json_write(path,value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


NAMES={'sim_rew':'Reward/Simulator','imagined_reward':'Reward/Imagined','ep_ret_norm':'Reward/EpisodeReturn',
       'ep_len':'Episode/Length','term_frac':'Episode/TerminationFraction','actor_loss':'Actor/Loss','critic_loss':'Critic/Loss',
       'wm_loss':'WorldModel/Loss','wm_err8':'WorldModel/Error8','actor_gradnorm':'Actor/GradNorm',
       'pure_foot_gradnorm':'Actor/FootGradNorm','latent_gradnorm':'Actor/LatentGradNorm'}

TRAIN_KEYS=('sim_rew','imagined_reward','ep_ret_norm','ep_len','term_frac',
            'wm_loss','wm_err8','actor_loss','critic_loss','actor_gradnorm')
PROGRESS_KEYS=('Progress/Iteration','Progress/WallHours',
               'Performance/CollectionFPS','Performance/PeakTorchMiB')
CORE_KEYS=frozenset(
    [name+'/'+NAMES[key] for name in ('Motion','Foot') for key in TRAIN_KEYS]
    +list(PROGRESS_KEYS)
    +['Eval/Motion/10s/NormalizedReturn','Eval/Motion/Full/NormalizedReturn']
    +[f'Eval/Foot/{duration}s/SurvivalRate' for duration in (2,5,10)]
    +['Eval/Foot/10s/'+key for key in
      ('NormalizedReturn','foot_position_rmse_m','actual_forward_speed_m_s')])
EVAL_MEAN=re.compile(
    r'^(Eval/Motion/(?:10s|Full)/NormalizedReturn)/offset[^/]+$'
    r'|^(Eval/Foot/10s/(?:NormalizedReturn|foot_position_rmse_m|actual_forward_speed_m_s))/seed[^/]+$')


def finite(value):
    return isinstance(value,(float,int)) and not isinstance(value,bool) and math.isfinite(value)


def values(event):
    """32 dashboard series; equal-weight means over the fixed eval probes.

    Short and full Motion evaluations retain separate series. Foot error and
    speed describe the observed prefix, including failed episodes, not success.
    No new diagnostic can leak to the cloud just by being added to JSONL.
    """
    result={};groups={}
    for key,value in event['values'].items():
        if not finite(value):continue
        match=EVAL_MEAN.fullmatch(key)
        if match:
            groups.setdefault(match.group(1) or match.group(2),[]).append(value)
            continue
        if key.startswith(('motion/','foot/')):
            name,suffix=key.split('/',1)
            if suffix not in TRAIN_KEYS:continue
            key=name.capitalize()+'/'+NAMES[suffix]
        if key in CORE_KEYS:result[key]=value
    result.update({key:sum(group)/len(group) for key,group in groups.items()})
    return result


def verification_targets(events):
    """Verify the terminal progress and latest core losses after cloud sync."""
    required=('Progress/Iteration','Progress/WallHours','Motion/WorldModel/Loss','Foot/WorldModel/Loss')
    targets={}
    for event in events:
        for key,value in values(event).items():
            if key in required:targets[key]=(event['event_step'],value)
    if set(targets)!=set(required):raise RuntimeError('Missing local core verification metrics')
    return targets


def rows(path):
    if not path.exists():return []
    return [json.loads(line) for line in path.read_text().splitlines(keepends=True) if line.endswith('\n')]


def iteration_rows(events):
    """One cloud record per learner iteration; combine train/eval/final events."""
    merged={}
    for event in events:
        iteration=event['iteration']
        payload=values(event)
        payload['Progress/Iteration']=iteration
        merged.setdefault(iteration,{}).update(payload)
    return [dict(iteration=iteration,event_step=iteration,values=payload)
            for iteration,payload in sorted(merged.items())]


def export_core(d,project):
    """New, clean dashboard from formal local data; never starts a learner."""
    cfg=json.loads((d/'profile.json').read_text())
    if cfg['mode']!='formal':raise ValueError('Smoke must never be uploaded')
    if not (d/'train.exitcode').exists():raise RuntimeError('Core export requires completed final evaluation/save')
    if project==cfg['swanlab_project']:raise ValueError('Use a new project to avoid existing project-wide charts')
    import swanlab
    swanlab.merge_settings(swanlab.Settings(interactive=False,probe=swanlab.Settings.Probe(monitor=False)))
    dest=d/'swanlab_core_export';dest.mkdir(exist_ok=True)
    with (dest/'export.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity_path=dest/'identity.json'
        if identity_path.exists():identity=json.loads(identity_path.read_text())
        else:
            stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
            identity=dict(run_id=uuid.uuid4().hex[:16],project=project,workspace=cfg['swanlab_workspace'],
                experiment_name='S54_JOINT_AB1_CORE_'+stamp+'_seed'+str(cfg['seed']),
                source_run=d.name,source_cloud_run_id=(d/'swanlab_run_id').read_text().strip())
            json_write(identity_path,identity)
        if identity['project']!=project:raise ValueError('Core export identity/project mismatch')
        def record(status,**extra):
            json_write(dest/'status.json',dict(identity,status=status,updated_at=datetime.now(timezone.utc).isoformat(),**extra))
        try:
            record('building_core_replay')
            events=iteration_rows(rows(d/'interface_metrics.jsonl'))
            json_write(dest/'core_events.json',events)
            core_config={k:cfg[k] for k in ('seed','num_envs_motion','num_envs_foot','horizon','wm_window',
                'hist','goal_frames','motion_goal_dt','foot_goal_dt','wm_batch','wm_batches',
                'pi_batch_motion','pi_batch_foot','wm_only_iters','utd','lr_wm','lr_motion_encoder',
                'lr_decoder','lr_foot_encoder','lr_critic','gamma','lam','action_noise')}
            core_config.update(source_run=d.name,source_cloud_run_id=identity['source_cloud_run_id'],
                record_kind='Formal training core-only replay; no new training',
                logged_every_iterations=cfg['log_every'],x_axis='Actual learner iteration',metric_count=len(CORE_KEYS))
            terminal=json.loads((d/'training_status.json').read_text())
            swanlab.init(project=project,workspace=identity['workspace'],id=identity['run_id'],
                name=identity['experiment_name'],config=core_config,log_dir=str(dest/'offline'),mode='offline')
            for event in events:swanlab.log(event['values'],step=event['iteration'])
            swanlab.finish(state='success' if terminal['status'].startswith('COMPLETED') else 'aborted')
            record('syncing',metric_count=len(CORE_KEYS),logged_iterations=len(events),last_iteration=events[-1]['iteration'])
            for cache in sorted((dest/'offline').glob('run-*')):
                if any(cache.glob('run-*.swanlab')):
                    subprocess.run([sys.executable,'-m','swanlab','sync',str(cache),'--project',project,
                        '--workspace',identity['workspace'],'--id',identity['run_id']],check=True,timeout=300)
            run=swanlab.Api().run(f'{identity["workspace"]}/{project}/{identity["run_id"]}')
            response=run.metrics(keys=sorted(CORE_KEYS),sample=1500,ignore_timestamp=True)
            observed={item['key']:{int(point['index']):float(point['data']) for point in item.get('metrics',[])}
                      for item in response.get('list',[])}
            expected={}
            for event in events:
                for key,value in event['values'].items():expected[key]=(event['iteration'],value)
            checks={key:dict(iteration=step,expected=value,observed=observed.get(key,{}).get(step),
                            verified=observed.get(key,{}).get(step)==value)
                    for key,(step,value) in expected.items()}
            json_write(dest/'cloud_readback.json',checks)
            if not all(item['verified'] for item in checks.values()):raise RuntimeError('Core cloud metrics not all verified')
            record('verified',url=run.url,metric_count=len(checks),logged_iterations=len(events),
                   last_iteration=events[-1]['iteration'],terminal_training_status=terminal['status'])
            (dest/'sync.exitcode').write_text('0\n');print(run.url,flush=True)
        except BaseException as error:
            record('failed',error=type(error).__name__+': '+str(error));(dest/'sync.exitcode').write_text('1\n')
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run_dir',type=Path,required=True)
    p.add_argument('--export-core',action='store_true',help='Replay completed formal run to a new clean dashboard')
    p.add_argument('--project',default='Foothold_with_SuperMimic_Core');a=p.parse_args()
    d=a.run_dir.resolve();cfg=json.loads((d/'profile.json').read_text())
    if cfg['mode']!='formal':raise ValueError('Smoke / non-formal runs must never be uploaded')
    if a.export_core:return export_core(d,a.project)
    import swanlab
    swanlab.merge_settings(swanlab.Settings(interactive=False,probe=swanlab.Settings.Probe(monitor=False)))
    identity=(d/'swanlab_run_id').read_text().strip();name=(d/'swanlab_experiment_name').read_text().strip()
    state=dict(run_id=identity,experiment_name=name,project=cfg['swanlab_project'],workspace=cfg['swanlab_workspace'])
    json_write(d/'swanlab_core_policy.json',dict(version=1,metric_count=len(CORE_KEYS),keys=sorted(CORE_KEYS),
        evaluation_aggregation='Equal-weight mean across fixed offsets/seeds, separate short/full Motion protocols',
        local_authority='interface_metrics.jsonl and evaluation/**/summary.json (unfiltered)',
        existing_cloud_charts='Retained; installed SDK cannot hide or delete already-created charts. Use WebUI to hide them.'))
    def record(status,**kw):
        state.update(status=status,updated_at=datetime.now(timezone.utc).isoformat(),**kw);json_write(d/'swanlab_sync_status.json',state)
    def init(mode,cache):
        return swanlab.init(project=cfg['swanlab_project'],workspace=cfg['swanlab_workspace'],id=identity,
            resume='allow' if mode=='online' else None,name=name,config=cfg,log_dir=str(cache),mode=mode)
    delay=5
    with (d/'uploader.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            initialized=False
            try:
                record('connecting');init('online',d/'swanlab');initialized=True
                seen=0
                while True:
                    events=rows(d/'interface_metrics.jsonl')
                    for event in events[seen:]:
                        payload=values(event)
                        if payload:swanlab.log(payload,step=event['event_step'])
                    seen=len(events)
                    record('following',local_events=seen,last_iteration=events[-1]['iteration'] if events else 0,
                           upload_policy='core_only_v1',max_metric_count=len(CORE_KEYS))
                    if (d/'train.exitcode').exists():break
                    time.sleep(10)
                terminal=json.loads((d/'training_status.json').read_text())
                swanlab.finish(state='success' if terminal['status'].startswith('COMPLETED') else 'aborted');initialized=False
                if not events:
                    record('finished_no_metrics',terminal_training_status=terminal['status'])
                    (d/'final_sync.exitcode').write_text('1\n');return
                # Rebuild all authoritative events offline, without cloud SDK
                # backfill restrictions, then drain exactly this immutable spool.
                record('final_core_replay');init('offline',d/'swanlab_final_core_replay');initialized=True
                events=rows(d/'interface_metrics.jsonl')
                for event in events:
                    payload=values(event)
                    if payload:swanlab.log(payload,step=event['event_step'])
                swanlab.finish(state='success' if terminal['status'].startswith('COMPLETED') else 'aborted');initialized=False
                for cache in sorted((d/'swanlab_final_core_replay').glob('run-*')):
                    if not any(cache.glob('run-*.swanlab')):continue
                    subprocess.run([sys.executable,'-m','swanlab','sync',str(cache),'--project',cfg['swanlab_project'],
                        '--workspace',cfg['swanlab_workspace'],'--id',identity],check=True,timeout=300)
                api=swanlab.Api();experiment=api.run(f'{cfg["swanlab_workspace"]}/{cfg["swanlab_project"]}/{identity}')
                targets=verification_targets(events)
                response=experiment.metrics(keys=list(targets),sample=1500,ignore_timestamp=True)
                observed={item['key']:{int(point['index']):float(point['data']) for point in item.get('metrics',[])} for item in response.get('list',[])}
                final_event=events[-1]
                for key,(step,value) in targets.items():
                    if observed.get(key,{}).get(step)!=value:raise RuntimeError('Final cloud core metric not verified: '+key)
                json_write(d/'swanlab_cloud_final_readback.json',response)
                record('verified',local_events=len(events),last_iteration=final_event['iteration'],url=experiment.url,
                       terminal_training_status=terminal['status'],note='unique event step; Progress/Iteration records actual learner iteration')
                (d/'final_sync.exitcode').write_text('0\n')
                subprocess.run([str(ROOT/'../../IsaacLab_3_0/.venv/bin/python'),'-B','tools/validation/report_s54_joint_ab1.py','--run_dir',str(d)],check=False)
                return
            except Exception as error:
                record('retry_pending',error=type(error).__name__+': '+str(error),retry_seconds=delay)
                print('SwanLab upload failed; durable local metrics preserved:',type(error).__name__,str(error),flush=True)
                if initialized:
                    try:swanlab.finish(state='aborted')
                    except Exception:pass
                time.sleep(delay);delay=min(300,delay*2)


if __name__=='__main__':main()
