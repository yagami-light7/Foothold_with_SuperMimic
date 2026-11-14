#!/usr/bin/env python3
"""Upload T1.5 core scalar diagnostics to the existing run; no media/artifacts."""
import argparse
import json
from pathlib import Path
from statistics import mean

from sync_swanlab import load_swanlab


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path('results/s54_stair_t15_20261110'))
    parser.add_argument('--run-dir',type=Path,default=Path('logs/s54_stair_t15_smoke'))
    args=parser.parse_args()
    metrics={}
    for case,label in [('flat_source','FlatPolicy_Flat'),('stair_zero_shot','FlatPolicy_Stair'),('pd_attempt01','PD_Stair')]:
        report=json.loads((args.results/case/'summary.json').read_text())
        if report['status']!='completed':raise ValueError('Incomplete independent evaluation')
        rows=report['rows']
        for protocol,selected in [('Nominal',[r for r in rows if r['seed']==0]),('Perturbed',[r for r in rows if r['seed']!=0])]:
            for name,key in [('NormalizedReturn','normalized_return'),('RawRewardSum','raw_reward_sum'),
                             ('EpisodeSeconds','episode_s'),('CompletionFraction','completion_rate'),
                             ('SurvivalRate','survived'),('YawRMSE_rad','yaw_rmse_rad')]:
                metrics[f'Eval/{label}/{protocol}/{name}']=mean(float(r[key]) for r in selected)
    run_id=(args.run_dir/'swanlab_run_id').read_text().strip()
    swanlab=load_swanlab('online',authenticate=True,core_only=True)
    swanlab.init(project='SuperMimic-S54-Stair',id=run_id,resume='allow',name=args.run_dir.name,
                 log_dir=str(args.run_dir/'swanlab'),mode='online')
    swanlab.log(metrics,step=32)
    swanlab.finish()
    (args.results/'swanlab_eval_scalars.json').write_text(json.dumps(dict(status='uploaded',run_id=run_id,
        metrics=metrics,media_uploaded=False,formal_training='on hold by user'),indent=2)+'\n')
    print(f'Uploaded {len(metrics)} scalar diagnostics; no GIF/video uploaded.')


if __name__=='__main__':main()
