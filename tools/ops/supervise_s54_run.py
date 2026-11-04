"""Bounded log supervisor. Stops only on nonfinite data or sustained severe collapse.

Does not edit code, change success metrics, send external messages, or restart
experiments in an uncontrolled loop. A stop always goes through the safe runner.
"""

# Support both direct execution and python -m from the repository layout.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_root = str(_ToolPath(__file__).resolve().parents[2])
if _tool_root not in _tool_sys.path:
    _tool_sys.path.insert(0, _tool_root)

import argparse,csv,json,math,re,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def verdict(rows):
    for row in rows[-3:]:
        try:
            if any(not math.isfinite(float(v)) for v in row.values() if v not in (None,'')):
                return 'NaN/Inf in completed training log row'
        except ValueError:continue
    if len(rows)<3:return None
    last=rows[-3:]
    if all(float(r['v_loss'])>1e6 for r in last):return 'Critic loss >1e6 for three consecutive log intervals'
    if all(float(r['v_loss'])>1000 and float(r['sim_rew'])<0 and 0<float(r['ep_len'])<25 for r in last):
        return 'Sustained Critic loss >1000, negative real reward, and <0.5s completed episodes'
    return None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True)
    p.add_argument('--seconds',type=float,default=1800);a=p.parse_args()
    run=(ROOT/a.run).resolve()
    if not run.is_relative_to(ROOT/'logs') or not re.fullmatch(r's54_[A-Za-z0-9_.-]+',run.name):p.error('Expected logs/s54_RUN')
    start=time.monotonic();last=-1;state=dict(status='watching',run=run.name,duration_seconds=a.seconds)
    def save():
        state['elapsed_seconds']=time.monotonic()-start
        temp=run/'supervision_status.json.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(run/'supervision_status.json')
    while time.monotonic()-start<a.seconds:
        log=run/'train_log.csv';rows=[]
        if log.exists():
            lines=[l for l in log.read_text().splitlines(keepends=True) if l.endswith('\n')]
            rows=[r for r in csv.DictReader(lines) if r.get('completed_episodes') is not None]
        if rows:
            current=int(rows[-1]['iteration'])
            if current!=last:
                state.update(iteration=current,last_metrics=rows[-1]);print(json.dumps(state),flush=True);last=current
            reason=verdict(rows)
            if reason and not (run/'train.exitcode').exists():
                print('SAFE STOP: '+reason,flush=True)
                result=subprocess.run(['bash','tools/train/run_s54_curriculum_8h.sh','stop',run.name],cwd=ROOT,capture_output=True,text=True)
                state.update(status='stop_requested' if result.returncode==0 else 'stop_failed',reason=reason,
                    stop_command_exit=result.returncode,stop_output=result.stdout+result.stderr);save();return
        if (run/'train.exitcode').exists():
            state.update(status='trainer_exited',exit_code=int((run/'train.exitcode').read_text()));save();return
        save();time.sleep(20)
    state['status']='observation_complete';save()

if __name__=='__main__':main()
