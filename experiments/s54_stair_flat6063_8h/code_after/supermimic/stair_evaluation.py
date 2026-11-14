"""Frozen Newton evaluations on designated TRAIN actions only; no optimizer feedback."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
import subprocess
import sys
import time

from .m34 import sha256,dataset_contract


def write_json(path, value):
    path=Path(path)
    temp=path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def scalar_metrics(references):
    result={}
    for name, report in references.items():
        for label, rows in [('Nominal',[r for r in report['rows'] if r['seed']==0]),
                            ('Perturbed',[r for r in report['rows'] if r['seed']!=0])]:
            if not rows:continue
            prefix=f'Eval/{name}/{label}'
            for out_key,key in [('NormalizedReturn','normalized_return'),('RawRewardSum','raw_reward_sum'),
                                ('SurvivalRate','survived'),('CompletionRate','completion_rate'),
                                ('EpisodeSeconds','episode_s'),('YawRMSE_rad','yaw_rmse_rad')]:
                result[f'{prefix}/{out_key}']=mean(float(row[key]) for row in rows)
            result[f'{prefix}/RootRMSE_m']=mean(sum(x*x for x in r['root_xyz_rmse_m'])**.5 for r in rows)
            result[f'{prefix}/FootRMSE_m']=mean(mean(r['left_right_foot_position_rmse_m']) for r in rows)
    return result


def evaluate_checkpoint(checkpoint, cfg, output, seeds=(0,1,2), *, video=False, flat_ids=None, timeout_seconds=None):
    checkpoint=Path(checkpoint).resolve();output=Path(output).resolve()
    seeds=list(seeds)
    if not seeds or len(set(seeds))!=len(seeds) or any(type(s) is not int or s<0 for s in seeds):
        raise ValueError('Distinct nonnegative evaluation seeds required')
    if flat_ids is None:flat_ids=['straight_flat_18','turn_flat_10','spin_flat_02']
    identity=dict(kind='s54_stair_t15',checkpoint_sha256=sha256(checkpoint),seeds=seeds,
                  flat_ids=flat_ids,video=video,stair_npz_sha256=sha256(cfg['stair_npz']),
                  flat_dataset_contract=dataset_contract(cfg['flat_dataset']),
                  terrain_sha256=sha256(Path(cfg['terrain'])/'terrain_mesh.npz'))
    deadline=time.monotonic()+(timeout_seconds if timeout_seconds is not None else cfg.get('eval_timeout_seconds',300))
    output.mkdir(parents=True,exist_ok=True)
    if (output/'summary.json').exists():
        previous=json.loads((output/'summary.json').read_text())
        if previous.get('status')!='completed' or any(previous.get(k)!=v for k,v in identity.items()):
            raise ValueError('Evaluation directory belongs to a different checkpoint/protocol')
        return previous
    attempt=output/f'attempt_{len(list(output.glob("attempt_*")))+1:03d}'
    attempt.mkdir()
    state=dict(status='running',checkpoint=str(checkpoint),attempt=str(attempt),**identity)
    write_json(output/'status.json',state)
    reports={}
    try:
        for name,reference in [('StairTrain','stair'),*[(f'FlatTrain_{c}','flat') for c in flat_ids]]:
            target=attempt/name
            command=[sys.executable,'-u','tools/t15_stair_rollout.py','--mode','policy',
                '--reference',reference,'--checkpoint',str(checkpoint),'--bundle',cfg['bundle'],
                '--stair-npz',cfg['stair_npz'],'--terrain',cfg['terrain'],
                '--flat-dataset',cfg['flat_dataset'],'--contacts','newton',
                '--seeds',*map(str,seeds),'--out',str(target)]
            if reference=='flat':command+=['--flat-id',name.removeprefix('FlatTrain_')]
            if not video:command+=['--no-video']
            print('EVAL:', ' '.join(command),flush=True)
            with (attempt/f'{name}.log').open('w') as sink:
                subprocess.run(command,stdout=sink,stderr=subprocess.STDOUT,check=True,
                               timeout=max(.1,deadline-time.monotonic()))
            report=json.loads((target/'summary.json').read_text())
            if report['status']!='completed' or report['checkpoint_sha256']!=identity['checkpoint_sha256']:
                raise RuntimeError('Incomplete evaluation or checkpoint mismatch')
            if sorted(r['seed'] for r in report['rows'])!=sorted(seeds):
                raise RuntimeError('Missing or duplicate evaluation seeds')
            reports[name]=report
        if sha256(checkpoint)!=identity['checkpoint_sha256']:
            raise RuntimeError('Checkpoint changed during evaluation')
        flat_scores=[mean(r['normalized_return'] for r in report['rows'])
                     for name,report in reports.items() if name.startswith('FlatTrain_')]
        summary=dict(status='completed',**identity,references=reports,scalars=scalar_metrics(reports),
            stair_score=mean(r['normalized_return'] for r in reports['StairTrain']['rows']),
            flat_retention_passed=bool(flat_scores) and min(flat_scores)>=.85,
            selection_split='experiment_train; four-step stair is not held-out generalization')
        write_json(output/'summary.json',summary)
        state['status']='completed'
        return summary
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));raise
    finally:write_json(output/'status.json',state)
