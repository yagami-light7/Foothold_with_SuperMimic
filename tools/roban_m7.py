"""M7 PMT dataset integration, real collisions, reference playback and PD eval."""
import os
os.environ['SUPERMIMIC_ROBOT']='roban'
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    from isaaclab.app import add_launcher_args
    pre=argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config',type=Path)
    config_args,_=pre.parse_known_args()
    config={} if config_args.config is None else json.loads(config_args.config.read_text())
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path)
    p.add_argument('--mode',choices=['alignment','collision','playback','pd'],required='mode' not in config)
    p.add_argument('--sample_mode',choices=['static_pillars','dynamic_three_pillars'],default='static_pillars')
    p.add_argument('--dataset_root',type=Path,default=os.environ.get('PMT_DATASET_ROOT','/mnt/sage_data/workspace/Foothold/PMT/TCRS/outputs/roban_pillar_dataset'))
    p.add_argument('--seed',type=int,default=44)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--experiment',choices=['double_support','initial_hold','four_step'],default='four_step')
    p.add_argument('--hold_seconds',type=float,default=5.)
    p.add_argument('--safe_frames',type=int,default=2)
    p.add_argument('--trust_offline_body_prior',action='store_true')
    add_launcher_args(p)
    fields={action.dest:action for action in p._actions}
    unknown=set(config)-set(fields)
    if unknown:p.error(f'Unknown M7 config fields: {sorted(unknown)}')
    for name,value in config.items():
        if fields[name].choices is not None and value not in fields[name].choices:p.error(f'Invalid configured {name}')
    p.set_defaults(**config)
    args=p.parse_args()
    if args.out.exists():p.error('Output exists; choose a new experiment directory')
    if args.hold_seconds<=0 or args.safe_frames<1:p.error('Invalid test duration/recycling guard')
    args.out.mkdir(parents=True)
    status=dict(status='RUNNING',mode=args.mode,sample_mode=args.sample_mode,formal_training=False)
    (args.out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    try:
        import numpy as np
        import torch
        np.random.seed(args.seed);torch.manual_seed(args.seed)
        from supermimic.integration.pmt_dataset_loader import load_catalog
        samples=load_catalog(args.dataset_root,mode=args.sample_mode,seed=args.seed)
        if len(samples)!=1:raise ValueError('Physics evaluation requires exactly one authoritative scene')
        sample=samples[0]
        (args.out/'dataset_alignment_report.json').write_text(json.dumps(sample.alignment,indent=2)+'\n')
        if args.mode=='alignment':report=sample.alignment
        elif args.mode=='collision':
            from supermimic.integration.collision_validation import run
            report=run(sample,args,args.out)
        else:
            from supermimic.integration.pd_tracking_eval import run
            report=run(sample,args,args.out)
        status.update(status='COMPLETE',result=report['status'])
        print(json.dumps({k:v for k,v in report.items() if k not in ('tests','rows')},indent=2),flush=True)
    except Exception as exc:
        status.update(status='ERROR',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:(args.out/'status.json').write_text(json.dumps(status,indent=2)+'\n')


if __name__=='__main__':main()
