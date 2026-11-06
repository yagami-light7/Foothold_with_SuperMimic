"""tmux launcher for tested, isolated, bounded S54 Motion/Foot joint runs."""
from pathlib import Path
import argparse,fcntl,json,os,re,shlex,shutil,signal,subprocess,sys,time,uuid
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.train.run_s54_motion_prior import default_python,snapshot_source
from tools.ops.swanlab_jobs import tmux_proxy_options
from supermimic.joint_runtime import validate_config,json_write,digest


def command(profile,run,python,resume=None,stop=None):
    result=[python,'-u',str(ROOT/'supermimic/joint_train.py'),'--profile',str(profile),'--run_dir',str(ROOT/'logs'/run),'--viz','none']
    if resume:result+=['--resume',str(resume)]
    if stop:result+=['--smoke_stop_iteration',str(stop)]
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['dry-run','smoke','train','resume','status','attach','stop','_worker'])
    p.add_argument('--profile',type=Path,default=Path('configs/motion_prior/joint_ab1/joint_ab1_3090_12h.json'))
    p.add_argument('--run');p.add_argument('--python',default=default_python());p.add_argument('--gates',type=Path)
    p.add_argument('--smoke_stop_iteration',type=int)
    a=p.parse_args();os.chdir(ROOT)
    if not a.run and a.mode in ('resume','status','attach','stop','_worker'):p.error('--run required')
    a.run=a.run or 's54_joint_ab1_'+('smoke_' if a.mode=='smoke' else '')+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')+'_seed54'
    if not re.fullmatch('s54_joint_ab1_[A-Za-z0-9_.-]+',a.run):raise ValueError('Use isolated s54_joint_ab1_ prefix')
    d=ROOT/'logs'/a.run
    if a.mode=='attach':os.execvp('tmux',['tmux','attach','-t','='+a.run])
    if a.mode=='status':
        for name in ('training_status.json','runtime_stats.json','swanlab_sync_status.json'):
            if (d/name).exists():print(name+'\n'+(d/name).read_text())
        return
    if a.mode=='stop':
        pid=int((d/'trainer.pid').read_text());cmd=subprocess.check_output(['ps','-p',str(pid),'-o','args='],text=True)
        if 'supermimic/joint_train.py' not in cmd or str(d) not in cmd:raise RuntimeError('Stale PID: refusing unrelated signal')
        os.kill(pid,signal.SIGTERM);print('Safe stop requested');return
    resume=a.mode=='resume' or a.mode=='_worker' and (d/'resume_request.json').exists()
    profile=d/'profile.json' if a.mode in ('resume','_worker') else a.profile.resolve()
    cfg=json.loads(profile.read_text());validate_config(cfg)
    if a.mode=='smoke' and cfg['mode']!='smoke':raise ValueError('Smoke requires an explicit small smoke profile; it is never uploaded')
    if a.mode=='train' and cfg['mode']!='formal':raise ValueError('train requires frozen formal profile')
    cmd=command(profile,a.run,a.python,d/'last.pt' if resume else None,a.smoke_stop_iteration)
    if a.mode=='dry-run':
        print(shlex.join(cmd));print(json.dumps(cfg,indent=2));print('No GPU work / output / SwanLab experiment started.');return
    # Reject duplicate/resolved resumes before writing receipts or clearing any
    # terminal files. A live run must remain entirely untouched by this command.
    if a.mode=='resume':
        if subprocess.run(['tmux','has-session','-t','='+a.run],capture_output=True).returncode==0:
            dead=subprocess.check_output(['tmux','list-panes','-s','-t','='+a.run,'-F','#{pane_dead}'],text=True).splitlines()
            if any(v!='1' for v in dead):raise RuntimeError('Cannot resume while a pane is live')
        if not (d/'last.pt').exists():raise FileNotFoundError('No complete joint checkpoint')
        import torch
        previous=torch.load(d/'last.pt',map_location='cpu',weights_only=False)
        if previous['iteration']>=cfg['iterations']:raise RuntimeError('Iteration cap already reached; no new sampling allowed')
        if previous['wall_seconds']>=cfg['wall_seconds']-cfg['final_reserve_seconds']:raise RuntimeError('Wall budget already exhausted; no new sampling allowed')
        del previous
    if a.mode=='_worker':
        with (d/'worker.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            from tools.visualization.policy_preview import gpu_slot
            with gpu_slot():
                process=subprocess.Popen(cmd,cwd=ROOT,start_new_session=True)
                (d/'launcher_child.pid').write_text(str(process.pid)+'\n')
                stopping=False
                def stop(*_):
                    nonlocal stopping
                    stopping=True
                    if process.poll() is None:process.send_signal(signal.SIGTERM)
                signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
                # Formal uploader is a separate CPU/tmux job, and uses direct
                # networking by default so it has no laptop/SSH proxy dependency.
                if cfg['mode']=='formal':
                    session=a.run+'_upload'
                    job=shlex.join([str(ROOT/'.venv-swanlab/bin/python'),'-u','tools/ops/sync_s54_joint_ab1.py','--run_dir',str(d)])
                    env=dict(os.environ)
                    for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):env[key]=''
                    if subprocess.run(['tmux','has-session','-t','='+session],capture_output=True).returncode:
                        try:subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(ROOT),*tmux_proxy_options(env),job+' >> '+shlex.quote(str(d/'swanlab_sync.log'))+' 2>&1'],check=True)
                        except subprocess.SubprocessError as e:(d/'upload_launch_failed.log').write_text(str(e))
                start=time.monotonic()
                offset=0
                if resume:
                    import torch
                    offset=torch.load(d/'last.pt',map_location='cpu',weights_only=False)['wall_seconds']
                while process.poll() is None:
                    if time.monotonic()-start+offset>=cfg['wall_seconds']:
                        stop()
                        try:process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            # Evaluator has its own process group. Kill only a
                            # currently attested joint child before hard exit.
                            state=json.loads((d/'training_status.json').read_text()) if (d/'training_status.json').exists() else {}
                            ep=state.get('evaluation_pid')
                            if ep:
                                check=subprocess.run(['ps','-p',str(ep),'-o','args='],capture_output=True,text=True).stdout
                                if 'joint_evaluate.py' in check and str(d) in check:
                                    try:os.killpg(ep,signal.SIGTERM)
                                    except ProcessLookupError:pass
                            os.killpg(process.pid,signal.SIGKILL);process.wait()
                            json_write(d/'watchdog.json',dict(status='HARD_TIMEOUT',previous_atomic_checkpoint_retained=True))
                        break
                    time.sleep(2)
                code=process.wait();(d/'launcher.exitcode').write_text(str(code)+'\n')
                if not (d/'train.exitcode').exists():
                    (d/'train.exitcode').write_text(str(code)+'\n')
                    json_write(d/'training_status.json',dict(status='CRASHED',reason='trainer exited before terminal status',exitcode=code))
                print('Joint launcher exit:',code,flush=True)
                subprocess.run([a.python,'-B','tools/validation/report_s54_joint_ab1.py','--run_dir',str(d)],check=False)
                raise SystemExit(code)
    from tools.visualization.policy_preview import gpu_slot
    with gpu_slot():
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
        if free<16000 and cfg['mode']=='formal':raise RuntimeError('Shared GPU insufficiently free for validated formal run')
    if resume:
        if not (d/'last.pt').exists():raise FileNotFoundError('No complete joint checkpoint')
        (d/'resume_request.json').write_text(json.dumps(dict(time=time.time())))
        stamp=str(time.time_ns())
        for name in ('train.exitcode','launcher.exitcode','training_status.json'):
            if (d/name).exists():shutil.copyfile(d/name,d/(name+'.before_resume_'+stamp))
        (d/'train.exitcode').unlink(missing_ok=True)
        (d/'launcher.exitcode').unlink(missing_ok=True)
    else:
        if d.exists():raise FileExistsError('Existing experimental directory protected')
        result=ROOT/'results/s54'/a.run
        if result.exists():raise FileExistsError('Existing experimental results protected')
        if cfg['mode']=='formal':
            if not a.gates:raise ValueError('Formal launch needs actual CPU/Newton/scale/resume gate evidence')
            gates=json.loads(a.gates.read_text())
            if not gates.get('passed') or gates.get('num_envs_per_interface')!=cfg['num_envs_motion']:raise ValueError('Missing passed preflight at this scale')
            for key,h in gates['source_hashes'].items():
                if digest(ROOT/key)!=h:raise ValueError('Source changed after preflight: '+key)
        d.mkdir();result.mkdir(parents=True)
        json_write(result/'artifacts.json',dict(run=a.run,training_directory=str(d),evaluation_directory=str(d/'evaluation'),
            report='docs/experiment/S54_JOINT_AB1_RUN_REPORT.md' if cfg['mode']=='formal' else str(d/'run_report.md')))
        shutil.copyfile(profile,d/'profile.json')
        if a.gates:shutil.copyfile(a.gates,d/'preflight_gates.json')
        json_write(d/'args.json',cfg);snapshot_source(d)
        for doc in ['docs/prompt/Codex_Prompt_S54_AppendixB_JointTraining_3090_Overnight.md','docs/experiment/S54_AppendixB_JointMotionFoot_ExperimentPlan.md']:
            dest=d/'source_snapshot'/doc;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/doc,dest)
        manifest=json.loads((d/'source_snapshot/manifest.json').read_text())
        manifest['git_sha']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        for path in (d/'source_snapshot/docs').rglob('*.md'):manifest['files'][str(path.relative_to(d/'source_snapshot'))]=digest(path)
        json_write(d/'source_snapshot/manifest.json',manifest)
        json_write(d/'launch.json',dict(command=command(d/'profile.json',a.run,a.python,stop=a.smoke_stop_iteration),profile=cfg,
            source_manifest_sha256=digest(d/'source_snapshot/manifest.json'),teacher_sha256=cfg['teacher_sha256'],
            launched_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()))
        json_write(d/'hardware_before.json',dict(gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.free,driver_version','--format=csv,noheader'],text=True)))
        if cfg['mode']=='formal':
            (d/'swanlab_run_id').write_text(uuid.uuid4().hex[:16]+'\n')
            (d/'swanlab_experiment_name').write_text('S54_JOINT_AB1_'+a.run.removeprefix('s54_joint_ab1_')+'\n')
    job=shlex.join(['env','PYTHONDONTWRITEBYTECODE=1','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=4',a.python,'-u',str(Path(__file__).resolve()),'_worker','--run',a.run,'--python',a.python]+(['--smoke_stop_iteration',str(a.smoke_stop_iteration)] if a.smoke_stop_iteration else []))
    body='set -o pipefail\n'+job+' 2>&1 | tee -a '+shlex.quote(str(d/'stdout.log'))
    session=subprocess.run(['tmux','has-session','-t','='+a.run],capture_output=True).returncode==0
    if session:
        if not resume:raise RuntimeError('Live tmux session already exists')
        dead=subprocess.check_output(['tmux','list-panes','-s','-t','='+a.run,'-F','#{pane_dead}'],text=True).splitlines()
        if any(v!='1' for v in dead):raise RuntimeError('Cannot resume while a pane is live')
        tm=['tmux','new-window','-d','-P','-F','#{window_id}','-t','='+a.run,'-n','resume']
    else:tm=['tmux','new-session','-d','-P','-F','#{window_id}','-s',a.run]
    window=subprocess.check_output([*tm,'-c',str(ROOT),*tmux_proxy_options(),'bash -lc '+shlex.quote(body)],text=True).strip()
    subprocess.run(['tmux','set-option','-w','-t',window,'remain-on-exit','on'],check=True)
    subprocess.run(['tmux','select-window','-t',window],check=True)
    print('Started:',a.run,'mode:',cfg['mode'],'SwanLab:',cfg['mode']=='formal')
    print('tmux attach -t '+a.run);print('tail -F '+str(d/'stdout.log'))


if __name__=='__main__':main()
