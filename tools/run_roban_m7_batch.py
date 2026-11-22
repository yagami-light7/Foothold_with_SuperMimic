"""Run the five M7 stages in separate processes; continue on measured failures."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--dataset_root',type=Path,default=os.environ.get('PMT_DATASET_ROOT','/mnt/sage_data/workspace/Foothold/PMT/TCRS/outputs/roban_pillar_dataset'))
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    tasks=[('static_collision','collision','static_pillars','four_step'),
           ('dynamic_collision','collision','dynamic_three_pillars','four_step'),
           ('static_kinematic_playback','playback','static_pillars','four_step'),
           ('dynamic_kinematic_playback','playback','dynamic_three_pillars','four_step'),
           ('pd_0_double_support','pd','static_pillars','double_support'),
           ('pd_1_initial_hold','pd','static_pillars','initial_hold'),
           ('pd_2_static_four_step','pd','static_pillars','four_step'),
           ('pd_3_dynamic_four_step','pd','dynamic_three_pillars','four_step')]
    results=[]
    for name,mode,sample,experiment in tasks:
        print('START',name,flush=True)
        command=[sys.executable,'tools/roban_m7.py','--mode',mode,'--sample_mode',sample,'--experiment',experiment,
                 '--dataset_root',str(args.dataset_root),'--out',str(args.out/name)]
        with (args.out/f'{name}.log').open('w') as log:
            code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
        metrics=args.out/name/'evaluation_metrics.json'
        result=json.loads(metrics.read_text()) if metrics.exists() else {'status':'ERROR'}
        row=dict(test=name,exit_code=code,result=result['status'],metrics=str(metrics))
        results.append(row)
        (args.out/'batch_summary.json').write_text(json.dumps(results,indent=2)+'\n')
        print('FINISH',json.dumps(row),flush=True)


if __name__=='__main__':main()
