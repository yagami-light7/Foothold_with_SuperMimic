"""Stage2-only Newton evaluation -> immutable NPZ trajectories -> reference MP4s.

The CPU supervisor exits each GPU worker before starting the renderer. No
optimizer, training resume, curriculum promotion or normalization update occurs.
"""
import argparse,datetime,json,math,os,shlex,signal,subprocess,sys,time
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))

def parser(worker=False):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--checkpoint',type=Path,required=True)
 p.add_argument('--seconds',type=float,default=10.,help='Requested Newton duration AND video duration for each command, 0.02..30s')
 p.add_argument('--num-envs',type=int,default=4,help='1..16 matched physical environments; video shows env0, JSON includes every env')
 p.add_argument('--seed',type=int,default=8541)
 p.add_argument('--stage',type=int,choices=range(5),help='Independent evaluation coverage override; never promotes training')
 p.add_argument('--out',type=Path,help='Fresh output directory; defaults beside the checkpoint with a timestamp')
 p.add_argument('--render',action='store_true',help='Render each saved command trajectory to MP4 after Newton exits')
 p.add_argument('--fps',type=int,default=25)
 p.add_argument('--process-budget-seconds',type=float,default=600.)
 p.add_argument('--tmux',action='store_true',help='Detach evaluation/rendering into its own tmux session with persistent stdout')
 p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
 if worker:
  from isaaclab.app import add_launcher_args
  add_launcher_args(p)
 return p

def validate(a):
 a.checkpoint=a.checkpoint.resolve()
 if not a.checkpoint.is_file():raise FileNotFoundError(a.checkpoint)
 if not math.isfinite(a.seconds) or not .02<=a.seconds<=30 or not math.isclose(a.seconds/.02,round(a.seconds/.02),abs_tol=1e-7):raise ValueError('--seconds must be a multiple of .02s, between .02 and 30s')
 if not 1<=a.num_envs<=16:raise ValueError('--num-envs must be 1..16')
 if not 1<=a.fps<=60:raise ValueError('--fps must be 1..60')
 if not math.isfinite(a.process_budget_seconds) or a.process_budget_seconds<=0:raise ValueError('Invalid process budget')
 stamp=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
 a.out=(a.out or a.checkpoint.parent/f'eval_render_{stamp}').resolve()
 return stamp

def write(path,value):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def launch_tmux(a,stamp):
 if a.out.exists() or a.out.with_suffix('.stdout.log').exists():raise FileExistsError('Existing evaluation protected: '+str(a.out))
 a.out.parent.mkdir(parents=True,exist_ok=True)
 command=[sys.executable,'-u',str(Path(__file__).resolve()),'--checkpoint',str(a.checkpoint),'--out',str(a.out),
          '--seconds',str(a.seconds),'--num-envs',str(a.num_envs),'--seed',str(a.seed),'--fps',str(a.fps),'--process-budget-seconds',str(a.process_budget_seconds)]
 if a.render:command.append('--render')
 if a.stage is not None:command+=['--stage',str(a.stage)]
 session='s54_stage2_eval_'+stamp;log=a.out.with_suffix('.stdout.log');exitfile=a.out.with_suffix('.exitcode')
 script=f'cd {shlex.quote(str(ROOT))}\nset -o pipefail\n{shlex.join(command)} 2>&1 | tee {shlex.quote(str(log))}\neval_exit=$?\nprintf "%s\\n" "$eval_exit" > {shlex.quote(str(exitfile))}\nexit "$eval_exit"'
 subprocess.run(['tmux','new-session','-d','-s',session,'bash -c '+shlex.quote(script)],check=True)
 subprocess.run(['tmux','set-option','-t',session+':0','remain-on-exit','on'],check=True)
 print(f'Started {session}\nAttach: tmux attach -t {session}\nLog: tail -f {log}\nResults: {a.out}',flush=True)

def supervisor(a):
 if a.out.exists():raise FileExistsError('Existing evaluation protected: '+str(a.out))
 a.out.mkdir(parents=True);start=time.monotonic();deadline=start+a.process_budget_seconds
 env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR='/tmp/s54_stage2_mpl',MUJOCO_GL='egl')
 phases=[];active=[None]
 def stop(*_):
  if active[0] and active[0].poll() is None:os.killpg(active[0].pid,signal.SIGTERM)
  raise KeyboardInterrupt('Evaluation stop requested')
 signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
 def phase(name,command):
  remaining=deadline-time.monotonic()
  if remaining<=0:raise TimeoutError('Evaluation/render total process budget exhausted')
  phases.append(dict(name=name,argv=command));write(a.out/'commands.json',phases)
  write(a.out/'status.json',dict(status='RUNNING',phase=name,wall_seconds=time.monotonic()-start))
  print('[Stage2 preview] '+name,flush=True)
  # run_phase persists stdout and streams progress into the terminal/tmux log.
  from tools.visualization.policy_preview import run_phase
  run_phase(command,a.out/(name+'.log'),env,deadline)
 try:
  cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker','--checkpoint',str(a.checkpoint),'--out',str(a.out),
       '--seconds',str(a.seconds),'--num-envs',str(a.num_envs),'--seed',str(a.seed)]
  if a.stage is not None:cmd+=['--stage',str(a.stage)]
  phase('newton',cmd)
  summary=json.loads((a.out/'evaluation/summary.json').read_text())
  if not summary['complete']:raise RuntimeError('Incomplete Newton evaluation; partial traces retained')
  if a.render:
   for row in summary['rows']:
    phase('render_'+row['label'],[sys.executable,'-u',str(ROOT/'tools/visualization/render_s54_stage2.py'),'--record',row['record'],
         '--out',str(a.out/'videos'/(row['label']+'.mp4')),'--fps',str(a.fps)])
  write(a.out/'status.json',dict(status='COMPLETED',wall_seconds=time.monotonic()-start,summary=str(a.out/'evaluation/summary.json'),videos=a.render))
  print(f'Evaluation: {a.out}/evaluation/summary.json\nTrajectories: {a.out}/evaluation/*.npz\nVideos: {a.out}/videos/*.mp4',flush=True)
 except BaseException as error:
  write(a.out/'status.json',dict(status='FAILED',wall_seconds=time.monotonic()-start,error=type(error).__name__+': '+str(error)));raise

def worker(a):
 import torch
 from supermimic import networks
 from supermimic.stage15 import seed_all,tensor_hash
 from supermimic.stage2_curriculum import build
 from supermimic.stage2_evaluation import evaluate
 from supermimic.joint_runtime import digest
 from tools.visualization.policy_preview import gpu_slot
 from isaaclab.app import launch_simulation
 from supermimic.env import TrackingSim,make_sim
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
 if ck.get('stage2_contract',{}).get('kind')!='s54_stage2_curriculum_v1':raise ValueError('Expected a Stage2 curriculum checkpoint')
 cfg=ck['config'];stage=ck['curriculum']['stage'] if a.stage is None else a.stage
 protocol=dict(checkpoint=str(a.checkpoint),checkpoint_sha256=digest(a.checkpoint),iteration=ck['iteration'],trained_stage=ck['curriculum']['stage'],
  evaluation_stage=stage,seconds=a.seconds,num_envs=a.num_envs,seed=a.seed,goal='Generated Quintic Foot230, dt=.02',
  normalization='Exact saved Phase A-compatible normalization; no refitting/clipping',
  success_protocol='Existing stage2_evaluation v1, unchanged; disputed speed/contact gates retained for comparability',
  record='NPZ stores env0 plus per-environment summary metrics, actual contacts, immutable command and Goal230',
  actor='Foot Encoder + frozen Shared Decoder, no Motion Reference',training_or_promotion=False)
 write(a.out/'protocol.json',protocol)
 with gpu_slot():
  seed_all(a.seed);torch.set_num_threads(4);networks.AMP_DTYPE=torch.bfloat16 if cfg['amp']=='bf16' else None
  parts=build(cfg,'cuda:0',ck)
  for key in ('policy','wm','critic'):parts[key].eval().requires_grad_(False)
  before={key:tensor_hash(parts[key].state_dict()) for key in ('policy','wm','critic')}
  parts['curriculum'].stage=stage;parts['curriculum'].apply_stage();parts['goals'].command_duration=max(12.,a.seconds+1.)
  eval_cfg=dict(cfg,eval_envs=a.num_envs)
  sim=make_sim('cuda:0',a)
  with launch_simulation(sim,a):
   scene=TrackingSim(sim,a.num_envs,None,parts['mdp'],gamma=cfg['gamma'],eval_mode=True,max_episode_steps=round(a.seconds/.02),reset_noise=True,goals=parts['goals'])
   print(f'[Newton] checkpoint iteration={ck["iteration"]}; C{stage}; {a.num_envs} env; {a.seconds:g}s per command',flush=True)
   report=evaluate(scene,parts,eval_cfg,a.out/'evaluation',a.seconds,seed=a.seed,record_diagnostics=True)
  checks={key+'_unchanged':tensor_hash(parts[key].state_dict())==value for key,value in before.items()}
  write(a.out/'inference_audit.json',checks)
  if not all(checks.values()):raise RuntimeError('Evaluation mutated a saved model/normalizer')
  print(json.dumps({r['label']:{k:r[k] for k in ('foot_rmse','survival_rate','success_rate')} for r in report['rows']},indent=2),flush=True)

def main():
 a=parser('--worker' in sys.argv).parse_args();stamp=validate(a)
 if a.worker:
  if a.tmux:raise ValueError('Worker cannot launch tmux')
  worker(a)
 elif a.tmux:launch_tmux(a,stamp)
 else:supervisor(a)
if __name__=='__main__':main()
