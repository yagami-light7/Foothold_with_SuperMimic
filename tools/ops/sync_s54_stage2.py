"""CPU-only, core-metric SwanLab follower; durable local source and final sync."""
import argparse,fcntl,json,math,os,subprocess,sys,time,uuid
from pathlib import Path
CORE={'stage':'Curriculum/Stage','curriculum_progress':'Curriculum/Progress','sim_reward':'Train/SimulationReward',
      'imagined_reward':'Train/ImaginedReward','foot_rmse':'Train/FootRMSE','episode_length':'Train/EpisodeLength',
      'termination_fraction':'Train/PhysicalTerminationFraction','wm_loss':'WM/Loss','wm_err8':'WM/Error8',
      'actor_loss':'Actor/Loss','critic_loss':'Critic/Loss','actor_gradnorm':'Actor/EncoderGradNorm',
      'latent_gradnorm':'Actor/LatentGradNorm','collection_fps':'Resource/NewtonFPS','iterations_per_second':'Resource/IterationsPerSecond',
      'peak_torch_mib':'Resource/PeakTorchMiB','newton':'Progress/NewtonTransitions','wall_seconds':'Progress/WallSeconds',
      'eval_standing_survival':'Eval/StandingSurvival','eval_standing_success':'Eval/StandingSuccess',
      'eval_walking_survival':'Eval/WalkingSurvival','eval_walking_success':'Eval/WalkingSuccess','eval_foot_rmse':'Eval/WalkingFootRMSE'}
def write(p,v):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();d=a.run_dir.resolve();dest=d/'swanlab';dest.mkdir(exist_ok=True)
 for k in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):os.environ.pop(k,None)
 import swanlab
 swanlab.merge_settings(swanlab.Settings(interactive=False,probe=swanlab.Settings.Probe(monitor=False)))
 with (dest/'uploader.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);cfg=json.loads((d/'profile.json').read_text());ip=dest/'identity.json'
  identity=json.loads(ip.read_text()) if ip.exists() else dict(run_id=uuid.uuid4().hex[:16],project=cfg['swanlab_project'],workspace=cfg['swanlab_workspace'],name=d.name)
  identity['url']=f'https://swanlab.cn/@{identity["workspace"]}/{identity["project"]}/runs/{identity["run_id"]}/chart';write(ip,identity)
  last=-1;expected={};attempt=0
  while True:
   try:
    cache=dest/f'cache_{attempt}'
    swanlab.init(project=identity['project'],workspace=identity['workspace'],id=identity['run_id'],resume='allow',name=identity['name'],
      mode='online',log_dir=str(cache),config={k:cfg[k] for k in ('seed','num_envs','hist','horizon','wm_window','goal_dt','rollout_steps','wm_only_iters','wm_batch','wm_batches','pi_batch','lr_actor','lr_wm','lr_critic','wall_seconds')})
    write(dest/'status.json',dict(status='FOLLOWING',**identity,core_metric_limit=len(CORE)))
    while True:
     path=d/'train_metrics.jsonl'
     if path.exists():
      for line in path.read_text().splitlines():
       try:r=json.loads(line)
       except json.JSONDecodeError:continue
       step=r['iteration']
       if step<=last:continue
       values={v:r[k] for k,v in CORE.items() if k in r and isinstance(r[k],(float,int)) and math.isfinite(r[k])}
       swanlab.log(values,step=step);last=step
       for k,v in values.items():expected[k]=(step,v)
     sp=d/'status.json'
     if sp.exists() and json.loads(sp.read_text())['status'] in ('COMPLETED','STOPPED','FAILED'):break
     time.sleep(10)
    swanlab.finish()
    # Force synchronization of the durable local stream after the worker exits.
    write(dest/'status.json',dict(status='FINAL_SYNC',**identity,last_iteration=last))
    for folder in cache.glob('run-*'):
     if any(folder.glob('run-*.swanlab')):subprocess.run([sys.executable,'-m','swanlab','sync',str(folder),'--project',identity['project'],'--workspace',identity['workspace'],'--id',identity['run_id']],check=True,timeout=300)
    if not expected:raise RuntimeError('No formal core measurements available')
    run=swanlab.Api().run(f'{identity["workspace"]}/{identity["project"]}/{identity["run_id"]}')
    result=run.metrics(keys=list(expected),sample=1500,ignore_timestamp=True)
    actual={x['key']:{int(y['index']):float(y['data']) for y in x.get('metrics',[])} for x in result.get('list',[])}
    checks={k:dict(expected=v,iteration=s,observed=actual.get(k,{}).get(s),verified=math.isclose(actual.get(k,{}).get(s,float('nan')),v,rel_tol=1e-6,abs_tol=1e-7)) for k,(s,v) in expected.items()}
    write(dest/'cloud_readback.json',checks)
    if not all(x['verified'] for x in checks.values()):raise RuntimeError('Incomplete cloud readback')
    write(dest/'status.json',dict(status='VERIFIED',**identity,last_iteration=last,metric_count=len(expected)));return
   except Exception as e:
    print(type(e).__name__+': '+str(e),flush=True)
    try:swanlab.finish()
    except Exception:pass
    attempt+=1;last=-1;expected={};write(dest/'status.json',dict(status='RETRY_PENDING',attempt=attempt,error=str(e),**identity))
    # Independent tmux process retries after an outage; local JSONL is authoritative.
    time.sleep(30)
if __name__=='__main__':main()
