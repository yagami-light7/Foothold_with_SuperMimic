"""Read-only CPU resource/summary monitor; safe alongside the one Newton worker."""
import argparse,datetime,fcntl,json,subprocess,time,os,signal
from pathlib import Path
from zoneinfo import ZoneInfo
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
def watchdog(d,status):
 """Track new independent evaluations, not prefix RMSE or WM loss alone."""
 profile=json.loads((d/'profile.json').read_text());sp=d/'monitor_state.json'
 state=json.loads(sp.read_text()) if sp.exists() else dict(last_evaluation=None,last_progress=0.,stage=0,best=None,no_survival=0)
 reason=None
 for path in sorted((d/'evaluation').glob('*/summary.json'),key=lambda p:p.stat().st_mtime):
  report=json.loads(path.read_text())
  if not report['complete']:continue
  if state.get('last_evaluation_mtime',0)>=path.stat().st_mtime:continue
  rows=report['rows'];success=sum(r['success_rate'] for r in rows)/len(rows);survival=sum(r['survival_rate'] for r in rows)/len(rows);rmse=sum(r['foot_rmse'] for r in rows)/len(rows)
  score=(success,survival,rmse);best=state['best']
  improved=(best is None or state['stage']!=report['stage'] or success>best[0]+.01 or survival>best[1]+.01 or (survival>=best[1]-.1 and rmse<.95*best[2]))
  if improved:state.update(best=score,last_progress=status['wall_seconds'])
  required=[r for r in rows if r['label'] in ('standing','short_walk')]
  state['no_survival']=state['no_survival']+1 if required and all(r['survival_rate']==0 for r in required) else 0
  state.update(last_evaluation=str(path),last_evaluation_mtime=path.stat().st_mtime,stage=report['stage'],latest_success=success)
 if status['wall_seconds']>=profile['watchdog_no_survival_seconds'] and state['no_survival']>=3:reason='3 successive independent evaluations: no 5s standing/walking survival'
 if status['wall_seconds']-state['last_progress']>=profile['watchdog_no_progress_seconds'] and state.get('latest_success',1)==0:reason='No independent-task success and no meaningful survival/RMSE improvement for 1h'
 temp=sp.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(sp)
 if reason and not (d/'external_watchdog.json').exists():
  info=dict(reason=reason,status=status,progress=state);(d/'external_watchdog.json').write_text(json.dumps(info,indent=2))
  pid=int((d/'pipeline.pid').read_text());cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
  if b'stage2_pipeline.py' not in cmd or str(d).encode() not in cmd:raise RuntimeError('Watchdog PID no longer matches this experiment')
  os.kill(pid,signal.SIGTERM);print('Safe stop: '+reason,flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();d=a.run_dir.resolve()
 from tools.validation.report_s54_stage2 import generate
 with (d/'monitor.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  while True:
   try:
    text=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True,timeout=10)
    fields=text.strip().split(', ')
    row=dict(time=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),gpu=fields[0],total_mib=float(fields[1]),used_mib=float(fields[2]),utilization_percent=float(fields[3]),power_w=float(fields[4]))
    with (d/'gpu_telemetry.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    generate(d)
    status=json.loads((d/'status.json').read_text())
    if status['status']=='RUNNING':watchdog(d,status)
    if status['status'] in ('COMPLETED','FAILED','STOPPED'):
     end=time.monotonic()+200
     while not (d/'launcher.exitcode').exists() and time.monotonic()<end:time.sleep(5)
     generate(d);break
   except Exception as e:print(type(e).__name__+': '+str(e),flush=True)
   time.sleep(30)
if __name__=='__main__':main()
