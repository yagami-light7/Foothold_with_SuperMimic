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
    jobs=ROOT/'results/s54_stair_t15_20261110/jobs';jobs.mkdir(parents=True,exist_ok=True)
    script=jobs/f'{session}.sh'
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
    roll=sub.add_parser('rollout');roll.add_argument('--out',required=True)
    roll.add_argument('rollout_args',nargs=argparse.REMAINDER)
    sync=sub.add_parser('sync');sync.add_argument('--run-dir',required=True)
    args=parser.parse_args()
    if args.phase=='train':
        cfg=json.loads(Path(args.config).read_text())
        if args.resume:cfg.pop('init_from_flat',None);cfg['resume']=args.resume
        if args.iterations:cfg['iterations']=args.iterations
        cmd=[args.python,'-u','supermimic/train.py']
        for name,value in cfg.items():
            if value is True:cmd.append('--'+name)
            elif value is not False and value is not None:
                cmd.extend(['--'+name,str(value)])
        launch(cmd,cfg['log_dir'],'train')
    elif args.phase=='rollout':
        rest=args.rollout_args
        if rest and rest[0]=='--':rest=rest[1:]
        launch([args.python,'-u','tools/t15_stair_rollout.py',*rest,'--out',args.out],args.out,'rollout')
    else:
        launch([str(ROOT/'.venv-swanlab/bin/python'),'-u','tools/sync_swanlab.py',
            '--run_dir',args.run_dir,'--project','SuperMimic-S54-Stair','--mode','online',
            '--core_only','--follow','--completion_marker'],args.run_dir,'sync')


if __name__=='__main__':main()
