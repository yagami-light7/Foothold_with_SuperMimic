"""Write a reproducible A/B/C/D/F/G bounded command matrix; runs nothing."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage1',required=True);p.add_argument('--bridge',required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[54,55,56]);p.add_argument('--out',required=True)
    p.add_argument('--task',choices=['standing','walking','curriculum'],default='standing')
    a=p.parse_args();target=Path(a.out)
    if target.exists():p.error('Output exists; choose a fresh command manifest')
    rows=[]
    for seed in a.seeds:
        for group in 'ABCDFG':
            run=f's54_mp_ablation_{group}_seed{seed}_{a.task}'
            command=[sys.executable,'tools/run_s54_motion_prior.py','--stage','stage2','--group',group,
                     '--seed',str(seed),'--task',a.task,'--run',run,'--dry-run']
            source=a.stage1 if group in 'BCG' else a.bridge if group in 'DF' else None
            if source:command+=['--source',source]
            result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,check=True)
            rows.append(dict(group=group,seed=seed,run=run,source=source,command=shlex.join(command),
                resolved_training_command=result.stdout.strip(),hist=10,
                budget='Same 12 Stage2 updates, 2 envs, 16 warmup vector steps, horizon8, 240s max',
                evaluation='Same 2/5/10s, seeds54/55/56, reset noise, diagnostics',
                decoder_update='frozen in all groups, including random-D controls A/B',
                status='DRY_RUN_ONLY'))
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(dict(rows=rows,stage15_optional=True,action_mse='NOT_IMPLEMENTED_OPTIONAL',
        accounting='Report Stage2-only real steps and upstream Stage1/Bridge real steps separately'),indent=2)+'\n')
    print(target)


if __name__=='__main__':main()
