"""Serial fresh-start capacity checks. Benchmark warmup is explicitly shortened.

Production profile always has 1000 WM-only iterations; these tests use four to
exercise both WM and Actor memory paths without starting a long experiment.
"""

# Support both direct execution and python -m from the repository layout.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_root = str(_ToolPath(__file__).resolve().parents[2])
if _tool_root not in _tool_sys.path:
    _tool_sys.path.insert(0, _tool_root)

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
CASES=[(1024,2048,1024,256),(2048,4096,2048,512),(2048,4096,4096,1024)]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--large',action='store_true');args=parser.parse_args()
    cases=[(4096,3072,4096,1024)] if args.large else CASES
    output=ROOT/('results/s54/fresh_capacity_large.json' if args.large else 'results/s54/fresh_capacity.json')
    results=[]
    for envs,buffer,wm,pi in cases:
        name=f's54_fresh_capacity_{envs}_{buffer}_{wm}_{pi}'
        path=ROOT/'logs'/name
        if not path.exists():
            cmd=['bash','tools/train/run_s54_curriculum_8h.sh','fresh',name,
                 '--num_envs',str(envs),'--buffer_steps',str(buffer),'--wm_batch',str(wm),
                 '--pi_batch',str(pi),'--warmup_steps','32','--wm_only_iters','4',
                 '--iterations','16','--log_every','4','--ckpt_every','16']
            subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,S54_NO_UPLOAD='1'),check=True)
        start=time.monotonic();samples=[]
        while not (path/'train.exitcode').exists():
            if time.monotonic()-start>480:
                subprocess.run(['bash','tools/train/run_s54_curriculum_8h.sh','stop',name],cwd=ROOT,check=True)
                raise TimeoutError('Capacity test exceeded 8 minutes; safe stop requested')
            sample=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True)
            util,mem=map(float,sample.splitlines()[0].split(','));samples.append((util,mem))
            if mem>21500:
                subprocess.run(['bash','tools/train/run_s54_curriculum_8h.sh','stop',name],cwd=ROOT,check=True)
                raise RuntimeError('Device usage exceeded 21500MiB; safe stop requested')
            time.sleep(2)
        code=int((path/'train.exitcode').read_text())
        item=dict(run=name,num_envs=envs,buffer_steps=buffer,wm_batch=wm,pi_batch=pi,exit_code=code)
        if code:
            item['error']=(path/'stdout.log').read_text()[-1500:]
            results.append(item);output.write_text(json.dumps(results,indent=2))
            raise RuntimeError(f'Capacity test failed: {name}; no larger test started')
        rows=list(csv.DictReader((path/'train_log.csv').open()))
        actor=[r for r in rows if int(r['iteration'])>=12]
        wm_rows=[r for r in rows if int(r['iteration'])<=4]
        events=[json.loads(line) for line in (path/'curriculum_metrics.jsonl').read_text().splitlines()]
        ips=statistics.median(float(r['it_per_s']) for r in actor)
        item.update(actor_iterations_s=ips,wm_phase_iterations_s=float(wm_rows[0]['it_per_s']),
            effective_transitions_s=envs*2.5*ips,
            collection_fps=events[-1]['values']['Performance/CollectionFPS'],
            device_memory_peak_sampled_mib=max(m for u,m in samples),
            gpu_util_mean=statistics.mean(u for u,m in samples),
            **json.loads((path/'runtime_stats.json').read_text()))
        results.append(item);output.write_text(json.dumps(results,indent=2))
        print(json.dumps(item),flush=True)
        for _ in range(15):
            active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
            if not active.strip():break
            time.sleep(1)

if __name__=='__main__':main()
