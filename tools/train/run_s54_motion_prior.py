"""Run bounded stage checks or explicitly requested Stage 1 Walking training.

The default keeps the original 16-env/200-iteration/300-second limits.
--stage1-train permits a separate, time-limited Motion training profile.
Stage 2 ablation limits remain unchanged. SwanLab upload is optional.
"""

# Support both direct execution and python -m from the repository layout.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_root = str(_ToolPath(__file__).resolve().parents[2])
if _tool_root not in _tool_sys.path:
    _tool_sys.path.insert(0, _tool_root)

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[2]
from supermimic.config_paths import resolve_config_path


def default_python():
    explicit=os.environ.get('S54_PYTHON')
    if explicit:return explicit
    for path in (ROOT.parents[1]/'IsaacLab_3_0/.venv/bin/python',
                 Path('/mnt/sage_data/workspace/IsaacLab_3_0/.venv/bin/python')):
        if path.is_file():return str(path)
    return sys.executable


def snapshot_source(target):
    """Retain dirty/untracked implementation bytes, not only the base Git SHA."""
    paths=sorted(set(list((ROOT/'supermimic').glob('*.py'))+
                 list((ROOT/'configs').rglob('*.json'))+
                 list((ROOT/'tools').rglob('*.py'))+
                 list((ROOT/'tools').rglob('*.sh'))+[ROOT/'tools/layout_migration.json']))
    hashes={}
    for source in paths:
        relative=source.relative_to(ROOT)
        destination=target/'source_snapshot'/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,destination)
        hashes[str(relative)]=hashlib.sha256(destination.read_bytes()).hexdigest()
    contract=dict(version=1,files=hashes,git_status=subprocess.check_output(
        ['git','status','--short'],cwd=ROOT,text=True))
    manifest=target/'source_snapshot/manifest.json'
    manifest.write_text(json.dumps(contract,indent=2)+'\n')
    return dict(manifest=str(manifest.relative_to(ROOT)),sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())


def command_for(a):
    profile=json.loads(resolve_config_path(ROOT/a.profile).read_text());train=dict(profile['train'])
    extended=getattr(a,'stage1_train',False)
    if extended and a.stage!='stage1':raise ValueError('--stage1-train only applies to Motion Stage 1')
    max_envs,max_iters,max_wall=(1024,20000,7200) if extended else (16,200,300)
    if train['hist']!=10 or not 1<=train['num_envs']<=max_envs or not 1<=train['iterations']<=max_iters:
        raise ValueError(f'Profile requires hist=10, <={max_envs} envs and <={max_iters} iterations')
    wall=profile['wall_seconds']
    if not 0<wall<=max_wall: raise ValueError(f'This mode is limited to {max_wall} seconds')
    # The process budget includes initialization, checkpoint saving and shutdown.
    train_wall=max(1,wall-30)
    if not re.fullmatch(r's54_[A-Za-z0-9_.-]+',a.run): raise ValueError('Use a fresh s54_* run name')
    if a.stage=='stage1':
        if a.source: raise ValueError('Stage 1 starts fresh')
        if extended and (train.get('freeze_decoder') or train.get('prior_config') or train.get('curriculum_config')):
            raise ValueError('Stage 1 trains the Motion encoder and Decoder without Foot curriculum/prior')
        train.update(goal_mode='motion',reference_manifest=a.manifest,manifest_split='train')
    else:
        train.update(goal_mode='foot_trajectory',foot_task_config=f'configs/motion_prior/stage2/stage2_{a.task}.json',
                     foot_config='configs/motion_prior/stage2/conservative_footsteps.json',goal_dt=.1,
                     obs_adapter='mask_reference_orientation',freeze_decoder=True,check_foot_gradients=True)
        if a.stage=='bridge':
            train.update(goal_mode='reference_foot',reference_manifest=a.manifest,manifest_split='train')
            for key in ('foot_task_config','check_foot_gradients'):train.pop(key,None)
            if not a.source and not a.resume: raise ValueError('Bridge requires an explicit Stage 1 source')
            if a.source:train['init_from_motion']=a.source
            train.update(no_critic=True,paired_latent_weight=0.)
        elif a.group in ('B','C','G'):
            if not a.source and not a.resume: raise ValueError('WM/Decoder transfer requires --source Stage1/last.pt')
            if a.source:train.update(init_from_motion=a.source,transfer_components='wm' if a.group=='B' else 'wm_decoder')
        elif a.group in ('D','F'):
            if not a.source and not a.resume: raise ValueError('D/F require an explicit optional Bridge checkpoint')
            if a.source:train['init_from_bridge']=a.source
            if not a.resume:
                if not getattr(a,'wm_source',None):raise ValueError('D/F require --wm-source with the SAME Stage 1 WM used by B/C/G')
                train['init_stage2_wm']=a.wm_source
        elif a.source:raise ValueError('Group A is random initialization')
        if a.group in ('F','G'):train['prior_config']='configs/motion_prior/prior/prior_matched.json'
        if a.stage=='stage2' and a.task=='curriculum':
            train.update(foot_task_config='configs/motion_prior/stage2/stage2_curriculum_task.json',curriculum_config='configs/motion_prior/stage2/curriculum_bounded.json')
    train.update(seed=a.seed,log_dir='logs/'+a.run,wall_seconds=train_wall,goal_frames=10,viz='none')
    if a.resume:
        if a.source:raise ValueError('Resume is separate from cross-task initialization')
        saved=json.loads((ROOT/a.resume).parent.joinpath('args.json').read_text())
        for key in ('obs_adapter','freeze_decoder','goal_mode','foot_task_config','foot_config','prior_config','reference_manifest','manifest_split','no_critic','paired_latent_weight','wm_calibration'):
            if key in saved and saved[key] is not None: train[key]=saved[key]
        # Dict arguments are represented by immutable resolved JSON snapshots.
        for key in ('foot_task_config','foot_config','prior_config'):
            if isinstance(train.get(key),dict):train[key]=str(ROOT/'logs'/a.run/(key+'_resolved.json'))
        train['resume']=a.resume
        train['iterations']=a.resume_iterations
        train.pop('fresh_start',None)
        if not 1<=a.resume_iterations<=max_iters:raise ValueError(f'Resume total target must be <={max_iters}')
    command=[a.python,'-u','supermimic/train.py']
    for key,value in train.items():
        if isinstance(value,bool):
            if value:command.append('--'+key)
        else:command.extend(['--'+key,str(value)])
    return command,profile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['stage1','bridge','stage2'],required=True)
    p.add_argument('--group',choices=list('ABCDFG'),default='C')
    p.add_argument('--task',choices=['standing','walking','curriculum'],default='standing')
    p.add_argument('--manifest',default='results/s54/motion_prior_stage0_verified/manifest.json')
    p.add_argument('--profile',default='configs/motion_prior/stage1/stage1_smoke.json')
    p.add_argument('--source');p.add_argument('--seed',type=int,default=54);p.add_argument('--run')
    p.add_argument('--wm-source',help='D/F: common Stage 1 checkpoint; exclude Bridge WM adaptation from Stage 2 initialization')
    p.add_argument('--resume');p.add_argument('--resume_iterations',type=int,default=24)
    p.add_argument('--python',default=default_python())
    p.add_argument('--execute',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--stage1-train',action='store_true',help='Explicit Stage 1 only: <=1024 envs, <=20000 iterations, <=2h process budget')
    p.add_argument('--stream-log',action='store_true',help='Mirror the persisted stdout/stderr to the attached terminal')
    p.add_argument('--experiment-name',help='SwanLab name; stable ID is generated and retained independently')
    p.add_argument('--swanlab-mode',choices=['offline','online'],help='Optional independent tmux uploader; local CSV/JSON always persist')
    a=p.parse_args()
    a.run=a.run or f's54_mp_{a.stage}_{a.group}_seed{a.seed}_{datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")}'
    try:command,profile=command_for(a)
    except (ValueError,OSError) as exc:p.error(str(exc))
    print(shlex.join(command),flush=True)
    if not a.execute or a.dry_run:return
    target=ROOT/'logs'/a.run
    if target.exists():p.error('Existing run protected; choose a fresh name')
    if a.source and not (ROOT/a.source).is_file():p.error('Source checkpoint missing')
    if a.stage in ('stage1','bridge'):
        sys.path.insert(0,str(ROOT))
        from supermimic.motion_prior_data import load_manifest
        load_manifest(ROOT/a.manifest)
    target.mkdir(parents=True)
    (target/'train_profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    label=a.group if a.stage=='stage2' else a.stage
    experiment=a.experiment_name or f'{label}_{datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")}_seed{a.seed}_{a.task if a.stage=="stage2" else "motion"}'
    (target/'swanlab_experiment_name').write_text(experiment+'\n')
    (target/'swanlab_run_id').write_text(uuid.uuid4().hex[:16]+'\n')
    source_snapshot=snapshot_source(target)
    if a.resume:
        saved=json.loads((ROOT/a.resume).parent.joinpath('args.json').read_text())
        for key in ('foot_task_config','foot_config','prior_config'):
            if isinstance(saved.get(key),dict):
                target.joinpath(key+'_resolved.json').write_text(json.dumps(saved[key],indent=2))
    (target/'launch.json').write_text(json.dumps(dict(command=command,profile=profile,stage=a.stage,group=a.group,seed=a.seed,
        git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_snapshot=source_snapshot),indent=2))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',
             WARP_CACHE_PATH=str(ROOT/'.cache/warp'),MPLCONFIGDIR='/tmp/s54_matplotlib')
    started=time.monotonic()
    with (target/'stdout.log').open('w') as stdout:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=subprocess.STDOUT,start_new_session=True)
        viewer=(subprocess.Popen(['tail','--pid='+str(process.pid),'-n','+1','-F',str(target/'stdout.log')])
                if a.stream_log else None)
        def stop_child(*_):
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        signal.signal(signal.SIGINT,stop_child);signal.signal(signal.SIGTERM,stop_child)
        (target/'trainer.pid').write_text(str(process.pid)+'\n')
        if a.swanlab_mode:
            uploader_env=dict(env,SWANLAB_MODE=a.swanlab_mode)
            # tmux servers retain their own environment; pass mode in the job
            # itself, otherwise an existing server can silently select online.
            job=shlex.join(['env','SWANLAB_MODE='+a.swanlab_mode,
                'SWANLAB_PYTHON='+os.environ.get('SWANLAB_PYTHON',str(ROOT/'.venv-swanlab/bin/python')),
                'bash','tools/train/run_s54_curriculum_8h.sh','_sync',a.run])+' >> '+shlex.quote(str(target/'swanlab_sync.log'))+' 2>&1'
            try:subprocess.run(['tmux','new-session','-d','-s',a.run+'_upload','-c',str(ROOT),job],cwd=ROOT,env=uploader_env,check=True,timeout=5)
            except (OSError,subprocess.SubprocessError) as exc:
                (target/'uploader_launch_error.txt').write_text(str(exc)+'\n')
                print('Uploader launch failed; local metrics retained:',exc,file=sys.stderr,flush=True)
        deadline=started+profile['wall_seconds']
        try:code=process.wait(timeout=max(.01,deadline-15-time.monotonic()))
        except (subprocess.TimeoutExpired,KeyboardInterrupt):
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=max(.01,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            code=124
        if viewer is not None:
            try:viewer.wait(timeout=3)
            except subprocess.TimeoutExpired:viewer.terminate();viewer.wait()
    (target/'train.exitcode.tmp').write_text(str(code)+'\n');(target/'train.exitcode.tmp').replace(target/'train.exitcode')
    (target/'launcher_exit.json').write_text(json.dumps(dict(exit_code=code,process_wall_seconds=time.monotonic()-started,
        process_budget_seconds=profile['wall_seconds'],experiment_name=experiment)))
    print(target,'exit',code);sys.exit(code)


if __name__=='__main__':main()
