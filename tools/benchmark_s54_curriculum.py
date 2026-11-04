"""Bounded serial tmux benchmarks. Refuses busy GPUs; no formal 8h run."""
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
ROOT=Path(__file__).resolve().parents[1]


def main():
    result=[]
    for n,wm,pi in [(128,256,64),(256,256,64),(512,256,64),(512,512,128)]:
        run=f's54_curriculum_bench_{n}_{wm}_{pi}'
        path=ROOT/'logs'/run
        if not path.exists():
            cmd=['bash','tools/run_s54_curriculum_8h.sh','start',run,'--num_envs',str(n),
                 '--warmup_steps','32','--iterations','24','--wm_only_iters','2',
                 '--wm_batch',str(wm),'--pi_batch',str(pi),'--log_every','4','--ckpt_every','24',
                 '--init_weights','logs/s54_standing_3090_calibration_seed54/last.pt']
            subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,S54_NO_UPLOAD='1'),check=True)
        util=[]; memory=[]; started=time.monotonic()
        while not (path/'train.exitcode').exists():
            if time.monotonic()-started>180:
                subprocess.run(['bash','tools/run_s54_curriculum_8h.sh','stop',run],cwd=ROOT,check=True)
                raise TimeoutError('Benchmark took >180s; safe stop requested')
            out=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True)
            u,m=out.splitlines()[0].split(',');util.append(float(u));memory.append(float(m))
            time.sleep(2)
        code=int((path/'train.exitcode').read_text())
        if code: raise RuntimeError(f'{run} failed: {code}; inspect stdout.log')
        rows=list(csv.DictReader((path/'train_log.csv').open()))[2:]
        events=[json.loads(line) for line in (path/'curriculum_metrics.jsonl').read_text().splitlines()]
        ips=statistics.median(float(r['it_per_s']) for r in rows)
        result.append(dict(run=run,num_envs=n,wm_batch=wm,pi_batch=pi,iterations_s=ips,
                    effective_transitions_s=ips*2.5*n,collection_fps=events[-1]['values']['Performance/CollectionFPS'],
                    gpu_util_mean=statistics.mean(util) if util else None,
                    device_memory_peak_sampled_mib=max(memory) if memory else None,
                    **json.loads((path/'runtime_stats.json').read_text())))
        (ROOT/'results/s54/curriculum_benchmark.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result[-1]),flush=True)
        # Let simulator teardown release its CUDA context before the next admission check.
        for _ in range(10):
            active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
            if not active.strip(): break
            time.sleep(1)

if __name__=='__main__': main()
