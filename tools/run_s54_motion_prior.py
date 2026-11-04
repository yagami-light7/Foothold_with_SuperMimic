"""Dry-run / execute one finite Stage 1, optional bridge, or fair Stage 2 ablation.

Never starts long training, uploads logs, or overwrites a run. The bounded
profile uses the existing SuperMimic loop, not a separate learning algorithm.
"""
import argparse
from datetime import datetime
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

ROOT=Path(__file__).resolve().parents[1]


def snapshot_source(target):
    """Retain dirty/untracked implementation bytes, not only the base Git SHA."""
    paths=sorted(list((ROOT/'supermimic').glob('*.py'))+
                 list((ROOT/'configs/motion_prior').glob('*.json'))+
                 list((ROOT/'tools').glob('*motion_prior*.py')))
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
    profile=json.loads((ROOT/a.profile).read_text());train=dict(profile['train'])
    if train['hist']!=10 or not 1<=train['num_envs']<=16 or not 1<=train['iterations']<=200:
        raise ValueError('Bounded profiles require common hist=10, <=16 envs and <=200 iterations')
    wall=profile['wall_seconds']
    if not 0<wall<=300: raise ValueError('This launcher is limited to 5 minutes')
    if not re.fullmatch(r's54_[A-Za-z0-9_.-]+',a.run): raise ValueError('Use a fresh s54_* run name')
    if a.stage=='stage1':
        if a.source: raise ValueError('Stage 1 starts fresh')
        train.update(goal_mode='motion',reference_manifest=a.manifest,manifest_split='train')
    else:
        train.update(goal_mode='foot_trajectory',foot_task_config=f'configs/motion_prior/stage2_{a.task}.json',
                     foot_config='configs/motion_prior/conservative_footsteps.json',goal_dt=.1,
                     obs_adapter='mask_reference_orientation',freeze_decoder=True,check_foot_gradients=True)
        if a.stage=='bridge':
            train.update(goal_mode='reference_foot',reference_manifest=a.manifest,manifest_split='train')
            for key in ('foot_task_config','check_foot_gradients'):train.pop(key,None)
            if not a.source and not a.resume: raise ValueError('Bridge requires an explicit Stage 1 source')
            if a.source:train['init_from_motion']=a.source
        elif a.group in ('B','C','G'):
            if not a.source and not a.resume: raise ValueError('WM/Decoder transfer requires --source Stage1/last.pt')
            if a.source:train.update(init_from_motion=a.source,transfer_components='wm' if a.group=='B' else 'wm_decoder')
        elif a.group in ('D','F'):
            if not a.source and not a.resume: raise ValueError('D/F require an explicit optional Bridge checkpoint')
            if a.source:train['init_from_bridge']=a.source
        elif a.source:raise ValueError('Group A is random initialization')
        if a.group in ('F','G'):train['prior_config']='configs/motion_prior/prior_matched.json'
        if a.stage=='stage2' and a.task=='curriculum':
            train.update(foot_task_config='configs/motion_prior/stage2_curriculum_task.json',curriculum_config='configs/motion_prior/curriculum_bounded.json')
    train.update(seed=a.seed,log_dir='logs/'+a.run,wall_seconds=wall,goal_frames=10,viz='none')
    if a.resume:
        if a.source:raise ValueError('Resume is separate from cross-task initialization')
        saved=json.loads((ROOT/a.resume).parent.joinpath('args.json').read_text())
        for key in ('obs_adapter','freeze_decoder','goal_mode','foot_task_config','foot_config','prior_config','reference_manifest','manifest_split'):
            if key in saved and saved[key] is not None: train[key]=saved[key]
        # Dict arguments are represented by immutable resolved JSON snapshots.
        for key in ('foot_task_config','foot_config','prior_config'):
            if isinstance(train.get(key),dict):train[key]=str(ROOT/'logs'/a.run/(key+'_resolved.json'))
        train['resume']=a.resume
        train['iterations']=a.resume_iterations
        if not 1<=a.resume_iterations<=200:raise ValueError('Resume total target must be <=200')
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
    p.add_argument('--profile',default='configs/motion_prior/stage1_smoke.json')
    p.add_argument('--source');p.add_argument('--seed',type=int,default=54);p.add_argument('--run')
    p.add_argument('--resume');p.add_argument('--resume_iterations',type=int,default=24)
    p.add_argument('--python',default=os.environ.get('S54_PYTHON','/mnt/sage_data/workspace/IsaacLab_3_0/.venv/bin/python'))
    p.add_argument('--execute',action='store_true');p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    a.run=a.run or f's54_mp_{a.stage}_{a.group}_seed{a.seed}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
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
    with (target/'stdout.log').open('w') as stdout:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=process.wait(timeout=profile['wall_seconds']+90)
        except (subprocess.TimeoutExpired,KeyboardInterrupt):
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=45)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            code=124
    (target/'launcher_exit.json').write_text(json.dumps(dict(exit_code=code)))
    print(target,'exit',code);sys.exit(code)


if __name__=='__main__':main()
