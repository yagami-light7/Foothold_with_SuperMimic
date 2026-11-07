"""CPU supervisor: bounded preflight then one GPU training worker, never parallel sims."""
import argparse,fcntl,json,os,signal,subprocess,sys,time,traceback,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];PY=Path('/home/rm/workspace/IsaacLab_3_0/.venv/bin/python')
sys.path.insert(0,str(ROOT))

def write(p,v):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();d=a.run_dir.resolve()
 cfg=json.loads((d/'profile.json').read_text());child=[None];stopping=[False]
 def stop(*_):
  stopping[0]=True
  if child[0] and child[0].poll() is None:child[0].send_signal(signal.SIGTERM)
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 def worker(name,script,args,budget):
  if stopping[0]:raise RuntimeError('Stop requested')
  command=[str(PY),'-u',script,*map(str,args)];print('[PIPELINE] '+name+' '+str(command),flush=True)
  with (d/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(name=name,argv=command,budget=budget,started_at=time.time()))+'\n')
  child[0]=subprocess.Popen(command,cwd=ROOT)
  try:code=child[0].wait(timeout=budget)
  except subprocess.TimeoutExpired:
   child[0].send_signal(signal.SIGTERM)
   try:code=child[0].wait(timeout=20)
   except subprocess.TimeoutExpired:child[0].kill();code=child[0].wait()
   write(d/(name+'_timeout.json'),dict(timeout_seconds=budget,exitcode=code));raise TimeoutError(name+' bounded process budget exceeded')
  (d/(name+'.exitcode')).write_text(str(code)+'\n')
  if code:raise RuntimeError(name+' failed, exit '+str(code))
 with (d/'pipeline.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  (d/'pipeline.pid').write_text(str(os.getpid())+'\n');failed=False
  try:
   for n in (16,64):
    audit=d/f'smoke{n}/smoke_audit.json'
    if not audit.exists():worker(f'smoke{n}','tools/train/run_s54_stage2_curriculum.py',['--config',d/'profile.json','--out',d/f'smoke{n}','--smoke','--num-envs',n],580)
    data=json.loads(audit.read_text())
    if not all(data['assertions'].values()) or data['counts']['actor']<1:raise RuntimeError('Technical preflight failed: '+str(audit))
   if not (d/'smoke64/resume_smoke_audit.json').exists():
    worker('resume_smoke','tools/train/run_s54_stage2_curriculum.py',['--config',d/'profile.json','--out',d/'smoke64','--smoke','--num-envs',64,'--resume',d/'smoke64/latest.pt'],580)
   if not all(json.loads((d/'smoke64/resume_smoke_audit.json').read_text())['assertions'].values()):raise RuntimeError('Resume smoke failed')
   if not (d/'normalization_audit.json').exists():worker('norm_audit','tools/validation/audit_s54_stage2.py',['--config',d/'profile.json','--out',d/'normalization_audit.json'],60)
   # Preserve the exact implementation/config used by this run; never touch old experiments.
   snapshot=d/'code_snapshot'
   if not snapshot.exists():
    snapshot.mkdir();sources=['supermimic/stage2_curriculum.py','supermimic/stage2_evaluation.py','tools/train/run_s54_stage2_curriculum.py','tools/train/stage2_pipeline.py','tools/train/run_s54_stage2_curriculum.sh','tools/ops/sync_s54_stage2.py','tools/validation/audit_s54_stage2.py','tools/validation/report_s54_stage2.py','tools/visualization/render_s54_stage2.py','supermimic/foot_mdp.py','supermimic/foot_trajectory.py','supermimic/joint_updates.py','supermimic/stage15_updates.py']
    import hashlib
    manifest={}
    for name in sources:
     target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target);manifest[name]=hashlib.sha256(target.read_bytes()).hexdigest()
    write(d/'source_manifest.json',manifest)
   previews=sorted((d/'smoke64/evaluation').glob('post_update_*/short_walk.npz'))
   if previews and not (d/'videos/smoke64_preview.mp4').exists():worker('preview','tools/visualization/render_s54_stage2.py',['--record',previews[-1],'--out',d/'videos/smoke64_preview.mp4'],120)
   write(d/'preflight_release.json',dict(technical_gates='PASS',foot_tracking_quality='NOT YET QUALIFIED',source='Phase A only',gpu='RTX 3090',released_at=time.time(),num_envs=cfg['num_envs']))
   from tools.validation.report_s54_stage2 import generate
   generate(d)
   args=['--config',d/'profile.json','--out',d]
   previous=json.loads((d/'status.json').read_text()).get('wall_seconds',0) if a.resume else 0
   if a.resume:args+=['--resume',d/'latest.pt']
   worker('training','tools/train/run_s54_stage2_curriculum.py',args,max(1,cfg['wall_seconds']-previous-30))
  except BaseException as e:
   failed=True
   traceback.print_exc();write(d/'pipeline_error.json',dict(error=type(e).__name__+': '+str(e),time=time.time()))
   status=json.loads((d/'status.json').read_text()) if (d/'status.json').exists() else {}
   if status.get('status') not in ('COMPLETED','STOPPED','FAILED'):write(d/'status.json',dict(status,status='FAILED',error=str(e)))
  finally:
   # The train process has exited before this renderer can acquire the GPU lock.
   try:
    summary=json.loads((d/'training_summary.json').read_text()) if (d/'training_summary.json').exists() else {}
    best=summary.get('best',{}).get('walking')
    if best:
     evaluation=json.loads(Path(best['evaluation']).read_text());row=next(r for r in evaluation['rows'] if r['label']=='short_walk')
     video=d/f'videos/best_walking_it{best["iteration"]:06d}.mp4'
     if not video.exists() and not stopping[0]:worker('best_video','tools/visualization/render_s54_stage2.py',['--record',row['record'],'--out',video],120)
   except Exception:traceback.print_exc()
   from tools.validation.report_s54_stage2 import generate
   generate(d)
  if failed:raise SystemExit(1)
if __name__=='__main__':main()
