#!/usr/bin/env python3
"""Durable native-IsaacLab T1.5 jobs in tmux, with exact commands/exit codes."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def training_command(cfg, python):
    cmd=[python,'-u','supermimic/train.py']
    for name,value in cfg.items():
        if value is True:cmd.append('--'+name)
        elif value is not False and value is not None:
            cmd+=['--'+name]
            cmd+=list(map(str,value)) if isinstance(value,list) else [str(value)]
    return cmd


def sync_command(run_dir):
    return [str(ROOT/'.venv-swanlab/bin/python'),'-u','tools/sync_swanlab.py',
        '--run_dir',str(run_dir),'--project','SuperMimic-S54-Stair','--mode','online',
        '--core_only','--follow','--completion_marker']


def launch(command,out,kind):
    out=Path(out).resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Job output must be inside this workspace')
    env=os.environ.copy()
    tmux=shutil.which('tmux')
    if tmux is None:
        tmux=str(ROOT/'.cache/tmux/usr/bin/tmux')
        if not Path(tmux).exists():raise FileNotFoundError('tmux required; local binary .cache/tmux/usr/bin/tmux not found')
        env['LD_LIBRARY_PATH']=str(ROOT/'.cache/tmux/usr/lib/x86_64-linux-gnu')+':'+env.get('LD_LIBRARY_PATH','')
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    session=f's54_t15_{kind}_{stamp}'
    socket=f'/tmp/s54_t15_{os.getuid()}.sock'
    jobs=out.parent/'t15_jobs';jobs.mkdir(parents=True,exist_ok=True)
    script=jobs/f'{session}.sh'
    target_log=(out/('stdout.log' if kind=='train' else 'swanlab_stdout.log')
                if kind in ['train','sync'] else out.parent/(out.name+'.log'))
    # Do not start a second writer/uploader for the same run while it is alive.
    for path in jobs.glob('*.json'):
        previous=json.loads(path.read_text())
        if previous.get('command')==command or previous.get('stdout')==str(target_log):
            alive=subprocess.run([tmux,'-S',previous['socket'],'has-session','-t',previous['session']],
                                 env=env,capture_output=True)
            if alive.returncode==0:raise RuntimeError(f"Job already running: {previous['session']}")
    if kind in ['train','sync']:
        out.mkdir(parents=True,exist_ok=True)
        log=out/('stdout.log' if kind=='train' else 'swanlab_stdout.log')
        exit_file=out/('train.exitcode' if kind=='train' else 'swanlab.exitcode')
        # A fresh launch/resume must not inherit an earlier completion marker.
        if exit_file.exists():exit_file.unlink()
    else:
        out.parent.mkdir(parents=True,exist_ok=True)
        if out.exists():raise FileExistsError(out)
        log=out.parent/(out.name+'.log');exit_file=out.parent/(out.name+'.exitcode')
    script.write_text('#!/usr/bin/env bash\nset -u\ncd '+shlex.quote(str(ROOT))+'\n'+
        shlex.join(command)+' >> '+shlex.quote(str(log))+' 2>&1\n'+
        'job_code=$?\nprintf "%s\\n" "$job_code" > '+shlex.quote(str(exit_file))+'\nexit "$job_code"\n')
    metadata=dict(session=session,socket=socket,command=command,stdout=str(log),exitcode=str(exit_file),script=str(script))
    (jobs/f'{session}.json').write_text(json.dumps(metadata,indent=2)+'\n')
    subprocess.run([tmux,'-S',socket,'new-session','-d','-s',session,'bash '+shlex.quote(str(script))],env=env,check=True,cwd=ROOT)
    print(json.dumps(metadata,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python',default=os.environ.get('ISAAC_PYTHON','/mnt/sage_data/workspace/Toolbox/IsaacLab_3_0/.venv/bin/python'))
    sub=parser.add_subparsers(dest='phase',required=True)
    train=sub.add_parser('train');train.add_argument('--config',required=True)
    train.add_argument('--resume');train.add_argument('--iterations',type=int)
    train.add_argument('--wall-seconds',type=float)
    train.add_argument('--with-swanlab',action='store_true')
    train.add_argument('--dry-run',action='store_true')
    roll=sub.add_parser('rollout');roll.add_argument('--out',required=True)
    roll.add_argument('rollout_args',nargs=argparse.REMAINDER)
    sync=sub.add_parser('sync');sync.add_argument('--run-dir',required=True)
    args=parser.parse_args()
    if args.phase=='train':
        cfg=json.loads(Path(args.config).read_text())
        if args.resume:cfg.pop('init_from_flat',None);cfg['resume']=args.resume
        if args.iterations:cfg['iterations']=args.iterations
        if args.wall_seconds:cfg['wall_seconds']=args.wall_seconds
        from supermimic.stair_training import read_design
        read_design(cfg['stair_config'])
        cmd=training_command(cfg,args.python)
        if args.dry_run:
            print(shlex.join(cmd))
            if args.with_swanlab:print(shlex.join(sync_command(cfg['log_dir'])))
            return
        launch(cmd,cfg['log_dir'],'train')
        if args.with_swanlab:launch(sync_command(cfg['log_dir']),cfg['log_dir'],'sync')
    elif args.phase=='rollout':
        rest=args.rollout_args
        if rest and rest[0]=='--':rest=rest[1:]
        launch([args.python,'-u','tools/t15_stair_rollout.py',*rest,'--out',args.out],args.out,'rollout')
    else:
        launch(sync_command(args.run_dir),args.run_dir,'sync')


if __name__=='__main__':main()
