"""Explicit, capped B repair after completed causal-heading diagnosis."""
import argparse,json,subprocess,time,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];P='/home/rm/workspace/IsaacLab_3_0/.venv/bin/python'
def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    baseline=a.baseline.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'status.json').exists():raise FileExistsError('Fresh directory required')
    start=time.monotonic()
    def status(value,**kw):
        t=out/'status.json.tmp';t.write_text(json.dumps(dict(status=value,wall_seconds=time.monotonic()-start,**kw),indent=2));t.replace(out/'status.json')
    status('WAITING_FOR_DIAGNOSIS')
    while json.loads((baseline/'status.json').read_text())['status'] not in ('BLOCKED_PHASE_A','FAILED','COMPLETED'):
        if time.monotonic()-start>900:raise TimeoutError('Diagnosis wait budget')
        time.sleep(5)
    if not (baseline/'diagnostic_masked_oracle/summary.json').exists():raise RuntimeError('Run completed heading-interface diagnostic first')
    cfg=ROOT/'configs/motion_prior/stage15/lafan1_skill_transfer_bounded_b_3090.json';shutil.copyfile(cfg,out/'profile.json')
    reuse={key:str(baseline/key) for key in ('phase_a','eval_a','smoke_b','diagnostic_masked_oracle')}
    (out/'baseline_source.json').write_text(json.dumps(dict(read_only_reuse=reuse,new_phase_a_training=False,
        rationale='A survives3/3, Motion NR>.65 but misses quality gate. Interface diagnosis completed. Test bounded original Motion-reward BPTT; keep all physical/final quality gates.'),indent=2))
    for name,target in reuse.items():(out/name).symlink_to(target,target_is_directory=True)
    def worker(name,script,args,budget):
        status(name);cmd=[P,'-u','-B',script,'--config',str(out/'profile.json'),*map(str,args)]
        with (out/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(name=name,argv=cmd,budget=budget))+'\n')
        print('[BOUNDED B] '+name,flush=True);r=subprocess.run(cmd,cwd=ROOT,timeout=budget)
        (out/(name+'.exitcode')).write_text(str(r.returncode)+'\n')
        if r.returncode:raise RuntimeError(name+' failed')
    try:
        worker('phase_b','tools/train/run_s54_stage15_phase_b.py',['--phase-a',baseline/'phase_a/phase_a_best.pt','--gate',baseline/'eval_a/summary.json','--out',out/'phase_b','--diagnostic-bounded'],690)
        worker('eval_b','tools/validation/evaluate_s54_stage15.py',['--checkpoint',out/'phase_b/phase_b_last.pt','--out',out/'eval_b'],620)
        summary=json.loads((out/'eval_b/summary.json').read_text());last=out/'phase_b/phase_b_last.pt';best=out/'phase_b/phase_b_best.pt'
        shutil.copyfile(last,best.with_suffix('.tmp'));best.with_suffix('.tmp').replace(best)
        (out/'phase_b/best_selection.json').write_text(json.dumps(dict(candidates=1,qualification=summary['gate_passed'],
            source='eval_b/summary.json',criterion='Measured 10s Newton quality, no WM loss selection'),indent=2))
        status('COMPLETED',phase_b_gate=summary['gate_passed'],phase_a_quality_gate=False,kind='bounded_diagnostic_repair')
    except BaseException as e:status('FAILED',error=type(e).__name__+': '+str(e));raise
if __name__=='__main__':main()
