"""CPU orchestrator: serial GPU workers, strict Phase A gate and durable logs."""
import argparse,json,subprocess,sys,time,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PYTHON='/home/rm/workspace/IsaacLab_3_0/.venv/bin/python'

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--config',default='configs/motion_prior/stage15/lafan1_skill_transfer_3090.json');a=p.parse_args()
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True);cfg=json.loads(Path(a.config).read_text())
    if (out/'status.json').exists():raise FileExistsError('Fresh directory required; existing experiments are immutable')
    shutil.copyfile(a.config,out/'profile.json');start=time.monotonic()
    def status(state,**kw):
        tmp=out/'status.json.tmp';tmp.write_text(json.dumps(dict(status=state,wall_seconds=time.monotonic()-start,**kw),indent=2));tmp.replace(out/'status.json')
    def worker(name,script,argv,budget):
        status(name);cmd=[PYTHON,'-u','-B',script,'--config',str(out/'profile.json'),*map(str,argv)]
        with (out/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(name=name,argv=cmd,budget=budget))+'\n')
        print('[PIPELINE] '+name,flush=True)
        result=subprocess.run(cmd,cwd=ROOT,timeout=budget)
        (out/(name+'.exitcode')).write_text(str(result.returncode)+'\n')
        if result.returncode:raise RuntimeError(name+' failed; inspect stdout.log')
    try:
        worker('smoke_a','tools/train/run_s54_stage15_phase_a.py',['--out',out/'smoke_a','--smoke'],120)
        worker('smoke_newton','tools/validation/evaluate_s54_stage15.py',['--checkpoint',out/'smoke_a/phase_a_best.pt','--out',out/'smoke_newton','--offsets',0,'--seconds',.2],620)
        worker('smoke_b','tools/train/run_s54_stage15_phase_b.py',['--phase-a',out/'smoke_a/phase_a_best.pt','--gate',out/'smoke_newton/summary.json','--out',out/'smoke_b','--smoke'],620)
        worker('phase_a','tools/train/run_s54_stage15_phase_a.py',['--out',out/'phase_a'],cfg['phase_a_budget_s']+90)
        worker('eval_a','tools/validation/evaluate_s54_stage15.py',['--checkpoint',out/'phase_a/phase_a_best.pt','--out',out/'eval_a'],620)
        gate=json.loads((out/'eval_a/summary.json').read_text())
        if not gate['gate_passed']:
            worker('diagnostic_masked_oracle','tools/validation/evaluate_s54_stage15.py',['--checkpoint',out/'phase_a/phase_a_best.pt','--out',out/'diagnostic_masked_oracle','--oracle-latent'],620)
            status('BLOCKED_PHASE_A',reason='Phase A Newton gate failed; no formal Phase B or Stage 2 started')
            return
        worker('phase_b','tools/train/run_s54_stage15_phase_b.py',['--phase-a',out/'phase_a/phase_a_best.pt','--gate',out/'eval_a/summary.json','--out',out/'phase_b'],cfg['phase_b_budget_s']+90)
        worker('eval_b','tools/validation/evaluate_s54_stage15.py',['--checkpoint',out/'phase_b/phase_b_last.pt','--out',out/'eval_b'],620)
        b_result=json.loads((out/'eval_b/summary.json').read_text())
        # Only one bounded formal B candidate: select by observed Newton score,
        # never by imagined reward. Keep its qualification separate from rank.
        shutil.copyfile(out/'phase_b/phase_b_last.pt',out/'phase_b/phase_b_best.pt.tmp')
        (out/'phase_b/phase_b_best.pt.tmp').replace(out/'phase_b/phase_b_best.pt')
        (out/'phase_b/best_selection.json').write_text(json.dumps(dict(
            candidates=1,source='eval_b/summary.json',qualified=b_result['gate_passed'],
            mean_motion_return=sum(r['norm_return'] for r in b_result['rows']['foot'])/3),indent=2))
        status('COMPLETED',phase_b_gate=b_result['gate_passed'])
    except BaseException as e:
        status('FAILED',error=type(e).__name__+': '+str(e));raise
if __name__=='__main__':main()
