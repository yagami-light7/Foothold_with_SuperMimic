"""Fail closed unless a real Newton smoke has finite training logs and S54 checkpoint."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from supermimic.s54 import assert_checkpoint


def main():
    p=argparse.ArgumentParser();p.add_argument('--run_dir',type=Path,default=Path('logs/s54_smoke'));p.add_argument('--iteration',type=int,default=30)
    p.add_argument('--require_eval',action='store_true');a=p.parse_args()
    rows=list(csv.DictReader((a.run_dir/'train_log.csv').open()))
    if not rows or int(rows[-1]['iteration'])!=a.iteration:raise ValueError('Incomplete smoke iterations')
    for row in rows:
        for name in ('wm_loss','wm_err8','pi_loss','v_loss','imag_rew','sim_rew','pi_gradnorm'):
            if not math.isfinite(float(row[name])):raise ValueError(f'Nonfinite {name}: {row}')
    if not any(float(r['pi_gradnorm'])>0 for r in rows):raise ValueError('No nonzero policy gradient recorded')
    ck=torch.load(a.run_dir/f'ckpt_{a.iteration:06d}.pt',map_location='cpu',weights_only=True);assert_checkpoint(ck)
    assert ck['iteration']==a.iteration
    for group in ('wm','pi'):
        assert all(torch.isfinite(v).all() for v in ck[group].values()),f'Nonfinite {group} checkpoint'
    last=torch.load(a.run_dir/'last.pt',map_location='cpu',weights_only=True)
    assert_checkpoint(last)
    assert last['iteration']==a.iteration
    assert all(torch.isfinite(v).all() for v in last['critic'].values()),'Nonfinite critic checkpoint'
    assert all(k in last for k in ('opt_wm','opt_pi','opt_v')),'Missing resume optimizer state'
    if a.require_eval:
        result=json.loads((a.run_dir/f'ckpt_{a.iteration:06d}_eval.json').read_text())
        assert all(k in result for k in ('norm_return','norm_return_framew','n_failed','failed','per_clip'))
        assert all(math.isfinite(result[k]) for k in ('norm_return','norm_return_framew'))
        assert result['n_failed']==len(result['failed'])
        assert result['per_clip'] and all(math.isfinite(v) for v in result['per_clip'].values())
        assert set(result['failed']).issubset(result['per_clip'])
    print(f'PASS finite logs and S54 checkpoint at iteration {a.iteration}; no performance threshold asserted')


if __name__=='__main__':main()
