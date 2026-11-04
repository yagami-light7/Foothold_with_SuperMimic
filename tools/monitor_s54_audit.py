"""Read-only bounded log supervision. Never signals or edits the trainer."""
import argparse,csv,json,math,time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--seconds',type=float,default=300);p.add_argument('--out',required=True);a=p.parse_args()
run=Path(a.run);out=Path(a.out);start=time.monotonic();seen=-1
while True:
    raw=(run/'train_log.csv').read_text().splitlines(keepends=True)
    rows=list(csv.DictReader([s for s in raw if s.endswith('\n')]))
    if rows:
        row=rows[-1];it=int(row['iteration'])
        if it!=seen:
            event=dict(observed_wall_s=time.monotonic()-start,iteration=it,
                values={k:float(row[k]) for k in ('wall_h','wm_loss','wm_err8','pi_loss','v_loss','pi_gradnorm','imag_rew','sim_rew','ep_len','term_frac')})
            event['nonfinite']=not all(math.isfinite(x) for x in event['values'].values())
            event['exit_code']=(run/'train.exitcode').read_text().strip() if (run/'train.exitcode').exists() else None
            with out.open('a') as f:f.write(json.dumps(event)+'\n')
            print(json.dumps(event),flush=True);seen=it
    if time.monotonic()-start>=a.seconds or (run/'train.exitcode').exists():break
    time.sleep(20)
