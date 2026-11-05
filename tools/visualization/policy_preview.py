"""One duration for a real Newton Motion rollout and its reference-overlay video."""
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def reference_window(clip_frames,seconds,start=0.,dt=.02):
    if not math.isfinite(seconds) or not math.isfinite(start) or seconds<=0 or start<0:
        raise ValueError('Use finite positive seconds and a nonnegative reference start')
    offset=round(start/dt)
    if abs(offset*dt-start)>1e-8:raise ValueError('Reference start must align to a 0.02s control step')
    steps=math.floor(seconds/dt+1e-8)
    remaining=int(clip_frames)-1-offset
    if steps<1 or remaining<1:raise ValueError('Requested interval contains no valid reference control step')
    return offset,min(steps,remaining)


@contextmanager
def gpu_slot():
    """Serialize previews with S54 jobs; desktop display clients may stay alive."""
    lock_path=ROOT/'logs/.s54_curriculum_gpu.lock';lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another S54 job holds the GPU lock') from None
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name',
                                          '--format=csv,noheader'],text=True)
        displays={'todesk_session','xorg','xwayland','gnome-shell'}
        busy=[line for line in processes.splitlines() if line.strip() and
              Path(line.split(',',1)[1].strip()).name.lower() not in displays]
        if busy:raise RuntimeError('GPU compute job active; preview not started: '+ '; '.join(busy))
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
                                         '--format=csv,noheader,nounits'],text=True).splitlines()[0])
        if free<4096:raise RuntimeError('Preview needs at least 4096 MiB free GPU memory')
        yield


def preview_plan(a):
    from supermimic.kinematics import S54Kinematics
    from supermimic.motion import MotionLib, CONTROL_DT
    from supermimic.motion_prior_data import load_manifest
    from supermimic.reference_selection import select_reference_files
    checkpoint=Path(a.checkpoint).resolve() if a.checkpoint else (Path(a.run_dir)/'last.pt').resolve()
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    run=Path(a.run_dir).resolve() if a.run_dir else checkpoint.parent
    saved=json.loads((run/'args.json').read_text())
    if saved.get('goal_mode','motion')!='motion':raise ValueError('This reference preview requires a Motion checkpoint')
    manifest=a.reference_manifest or saved.get('reference_manifest')
    split=a.manifest_split or saved.get('manifest_split','train')
    if manifest:files,contract=load_manifest(manifest,split)
    else:files=select_reference_files(saved['motion_dir'],a.clips or saved.get('clips'));contract=None
    motion=MotionLib(files,S54Kinematics('cpu'),'cpu')
    if len(motion.names)!=1:raise ValueError('Use a single-clip reference manifest for a synchronized policy preview')
    if a.clips and a.clips!=motion.names:raise ValueError('Requested clip differs from the selected reference manifest')
    offset,steps=reference_window(int(motion.clip_len[0]),a.seconds,a.start)
    output=Path(a.output).resolve();artifacts=output.with_name(output.stem+'_recording')
    if output.exists() or artifacts.exists():raise FileExistsError('Existing preview protected; choose a fresh --out path')
    duration=steps*CONTROL_DT
    trajectory=artifacts/'trajectory.npz'
    evaluate=[sys.executable,'-u',str(ROOT/'supermimic/evaluate.py'),'--run_dir',str(run),
        '--ckpt',str(checkpoint),'--duration_s',str(duration),'--start_seconds',str(offset*CONTROL_DT),
        '--record',str(trajectory),'--out_json',str(artifacts/'evaluation.json'),'--viz','none']
    if manifest:evaluate+=['--reference_manifest',str(Path(manifest).resolve()),'--manifest_split',split]
    render=[sys.executable,'-u',str(ROOT/'tools/visualization/render.py'),str(trajectory),str(output),
        '--seconds',str(duration),'--fps',str(a.fps),'--width',str(a.width),'--height',str(a.height),
        '--title',a.title,'--global_ghost' if a.global_ghost else '--align_reference']
    return dict(requested_seconds=a.seconds,effective_seconds=duration,reference_start_seconds=offset*CONTROL_DT,
        reference_end_seconds=(int(motion.clip_len[0])-1)*CONTROL_DT,clip=motion.names[0],
        source_checkpoint=str(checkpoint),reference_contract=contract,output=str(output),artifacts=str(artifacts),
        evaluate_command=evaluate,render_command=render)


def run_phase(command,log_path,env,deadline):
    with log_path.open('w') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        viewer=subprocess.Popen(['tail','--pid='+str(child.pid),'-n','+1','-F',str(log_path)])
        try:
            last_output=time.monotonic();last_size=0
            while True:
                remaining=deadline-time.monotonic()
                if remaining<=0:raise subprocess.TimeoutExpired(command,0)
                try:code=child.wait(timeout=min(5,remaining));break
                except subprocess.TimeoutExpired:
                    now=time.monotonic();size=log_path.stat().st_size
                    if size!=last_size:last_output=now;last_size=size
                    elif now-last_output>=15:
                        print(f'[Preview/{log_path.stem}] process alive (PID {child.pid}); '
                              f'waiting for progress output; log: {log_path}',flush=True)
                        last_output=now
            if code:raise RuntimeError(f'Preview phase exited {code}; see {log_path}')
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            # Drain the final lines before ending the terminal follower.
            try:viewer.wait(timeout=2)
            except subprocess.TimeoutExpired:viewer.terminate();viewer.wait()


def run_preview(a):
    if not math.isfinite(a.process_budget_s) or a.process_budget_s<=0:raise ValueError('Process budget must be finite and positive')
    plan=preview_plan(a)
    print(f"{plan['clip']}: requested {plan['requested_seconds']:g}s; Newton and video target "
          f"{plan['effective_seconds']:g}s from reference t={plan['reference_start_seconds']:g}s "
          f"(reference ends at {plan['reference_end_seconds']:g}s).",flush=True)
    print('Reference overlay: '+('world trajectory' if a.global_ghost else 'XY/yaw aligned to policy'),flush=True)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',
        WARP_CACHE_PATH=str(ROOT/'.cache/warp'),MPLCONFIGDIR='/tmp/s54_matplotlib',
        MUJOCO_GL=os.environ.get('MUJOCO_GL','egl'),PATH=str(ROOT/'.cache/bin')+os.pathsep+os.environ.get('PATH',''))
    import shutil
    if not shutil.which('ffmpeg',path=env['PATH']):raise RuntimeError('FFmpeg is required for MP4 output')
    with gpu_slot():
        artifacts=Path(plan['artifacts']);artifacts.mkdir(parents=True)
        (artifacts/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
        started=time.monotonic();deadline=started+a.process_budget_s
        status=dict(status='running',started_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        (artifacts/'status.json').write_text(json.dumps(status,indent=2)+'\n')
        try:
            status['phase']='newton'
            (artifacts/'status.json').write_text(json.dumps(status,indent=2)+'\n')
            print('[Preview] Phase 1/2: execute the policy in Newton and save the trajectory',flush=True)
            run_phase(plan['evaluate_command'],artifacts/'newton.log',env,deadline)
            status['phase']='render'
            (artifacts/'status.json').write_text(json.dumps(status,indent=2)+'\n')
            print('[Preview] Phase 2/2: render the saved trajectory and reference to MP4',flush=True)
            run_phase(plan['render_command'],artifacts/'render.log',env,deadline)
            status.update(status='finished',phase='complete')
        except BaseException as exc:
            status.update(status='failed',error=str(exc));raise
        finally:
            status['wall_seconds']=time.monotonic()-started
            (artifacts/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    print('Video: '+plan['output'],flush=True)
    return plan
