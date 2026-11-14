#!/usr/bin/env python3
"""Prepare, gate, launch and evaluate the Flat-6063 -> four-step stair experiment."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BASE=Path('experiments/s54_stair_flat6063_8h')


def read(path):return json.loads(Path(path).read_text())


def cpu_check():
    import torch
    from supermimic.goals import GoalSpec
    from supermimic.kinematics import S54Kinematics
    from supermimic.m34 import sha256
    from supermimic.networks import Policy,WorldModel
    from supermimic.transfer import physical_contract
    from supermimic.stair_training import read_design,contract,library,initialize_flat,sample_frames
    torch.set_num_threads(4)
    design=read_design(BASE/'design.json',require_smoke=False)
    cfg=read(BASE/'train.json')
    ck=torch.load(cfg['init_from_flat'],map_location='cpu',weights_only=False)
    kin=S54Kinematics('cpu');motion=library(design,kin,'cpu')
    wm=WorldModel();pi=Policy(103,730,hist=10)
    physics=physical_contract(obs_dim=103)
    physics['solver']={**physics['solver'],'use_mujoco_contacts':False}
    physics['terrain_task']=contract(design)
    transfer=initialize_flat(ck,wm,pi,physics,GoalSpec('motion',10,.02),10,
        normalizer_policy=design.get('normalizer_policy','frozen_source'))
    # Every tensor, including normalizer effective counts, must be copied exactly.
    for name,model in [('wm',wm),('pi',pi)]:
        if any(not torch.equal(v,ck[name][k]) for k,v in model.state_dict().items()):
            raise AssertionError('Flat initialization changed a source tensor')
    class Sampler:pass
    sampler=Sampler();sampler.motion=motion;sampler.device='cpu'
    sampler.bin_G=torch.zeros(motion.num_bins);sampler.G_max=13.
    torch.manual_seed(54);ratios=[]
    for it in [0,1000,6000]:
        sampler.stair_iteration=it
        f,bins=sample_frames(sampler,20000,design)
        is_stair=motion.clip_id[f]==motion.names.index('stair_up_4')
        actual=float(is_stair.float().mean())
        expected=next(s['stair_fraction'] for s in design['sampling_stages'] if it<s['until_iteration'])
        if abs(actual-expected)>.02:raise AssertionError('Sampling group ratio differs')
        if not ((f>=motion.clip_start[motion.clip_id[f]])&(f<motion.clip_end[motion.clip_id[f]])).all():
            raise AssertionError('Sampler produced a terminal/out-of-clip frame')
        ratios.append(dict(iteration=it,stair_fraction=actual,expected=expected))
    if 'stair_up_2' in motion.names or len(motion.names)!=15:
        raise AssertionError('Wrong training inventory')
    mesh=sha256(Path(design['terrain'])/'terrain_mesh.npz')
    if mesh!=sha256(Path(design['bundle'])/'terrain/terrain_mesh.npz'):
        raise AssertionError('Terrain and reference contract differ')
    report=dict(status='PASS',gpu_training_started=False,source_iteration=ck['iteration'],
        source_sha256=sha256(cfg['init_from_flat']),data_contract=contract(design),
        training_clips=motion.names,flat_test_opened=False,stair_frames=int(motion.clip_len[-1]),
        byte_exact_initialization=True,transfer=transfer,sampling=ratios,all_body_fk=motion.validation,
        remaining='New physical Smoke + independent Smoke checkpoint evaluation must pass before 8h training')
    (BASE/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','source_iteration','training_clips','stair_frames','byte_exact_initialization','remaining']},indent=2))


def smoke_gate():
    import torch
    from supermimic.m34 import sha256
    from supermimic.stair_evaluation import evaluate_checkpoint,write_json
    from supermimic.stair_training import read_design,contract
    cfg=read(BASE/'smoke.json');run=Path(cfg['log_dir'])
    design=read_design(BASE/'smoke_design.json')
    report=dict(passed=False,data_contract=contract(design),checks={},purpose='technical stability gate, not stair convergence')
    try:
        status=read(run/'training_status.json')
        if status['status']!='completed' or status['iteration']!=cfg['iterations']:
            raise ValueError('Smoke did not complete the requested iterations')
        if (run/'failure_physics_transition.pt').exists():raise ValueError('Smoke recorded extreme physics')
        with (run/'train_log.csv').open() as stream:rows=list(csv.DictReader(stream))
        keys=['wm_loss','wm_err8','pi_loss','v_loss','imag_rew','pi_gradnorm','sim_rew','reward_sync_max_error']
        if not rows or any(not math.isfinite(float(r[k])) for r in rows for k in keys):
            raise ValueError('Missing/nonfinite Smoke metrics')
        actor=[r for r in rows if int(r['iteration'])>cfg['wm_only_iters']]
        report['tracking_diagnostics']=dict(min_mean_real_reward=min(float(r['sim_rew']) for r in rows),
            wm8_final=float(rows[-1]['wm_err8']),
            max_abs_imagined_reward=max((abs(float(r['imag_rew'])) for r in actor),default=None),
            max_critic_loss=max((float(r['v_loss']) for r in actor),default=None),
            interpretation='Finite tracking losses are diagnostics; negative returns do not imply a Newton environment error')
        report['checks']=dict(finite_training_metrics=True,
            actor_gradients=bool(actor) and all(float(r['pi_gradnorm'])>0 for r in actor),
            reward_sync=max(float(r['reward_sync_max_error']) for r in rows)<1e-5)
        ckpath=run/'latest.pt';ck=torch.load(ckpath,map_location='cpu',weights_only=False)
        if ck['data_contract']!=report['data_contract']:raise ValueError('Smoke checkpoint contract differs')
        if not all(torch.isfinite(v).all() for model in ['wm','pi','critic'] for v in ck[model].values()):
            raise ValueError('Nonfinite Smoke weights/statistics')
        source=torch.load(design['source_checkpoint'],map_location='cpu',weights_only=False)
        if design.get('normalizer_policy')=='source_online':
            report['checks']['source_normalizer_counts_retained']=all(
                float(ck[model][key])>=float(value) for model in ['wm','pi']
                for key,value in source[model].items() if key.endswith('_norm.count'))
        else:
            report['checks']['source_normalizers_preserved']=all(torch.equal(v,source[model][key])
                for model in ['wm','pi'] for key,v in ck[model].items() if '_norm.' in key)
        if not all(report['checks'].values()):raise ValueError('Smoke stability checks failed; see checks')
        evaluation=evaluate_checkpoint(ckpath,design,run/'evals'/f'iter_{ck["iteration"]:06d}')
        report['checks']['flat_retention']=evaluation['flat_retention_passed']
        report['smoke_checkpoint_sha256']=sha256(ckpath)
        report['evaluation']=str(run/'evals'/f'iter_{ck["iteration"]:06d}'/'summary.json')
        report['passed']=all(report['checks'].values())
        if not report['passed']:raise ValueError('Independent flat retention check failed')
    except BaseException as exc:
        report['error']=repr(exc);raise
    finally:
        write_json(read(BASE/'design.json')['required_smoke_gate'],report)
        print(json.dumps(report,indent=2))
    # The live Smoke uploader may have finished before this evaluation. Resume
    # that same SwanLab run once to drain the scalar-only evaluation series.
    marker=run/'swanlab.exitcode'
    deadline=time.monotonic()+60
    while not marker.exists():
        if time.monotonic()>deadline:raise TimeoutError('Smoke uploader is still active; retry gate after it exits')
        time.sleep(2)
    subprocess.run([str(ROOT/'.venv-swanlab/bin/python'),'tools/sync_swanlab.py',
        '--run_dir',str(run),'--project','SuperMimic-S54-Stair','--core_only'],check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['check','night','_night','smoke','gate','train','resume','eval','view'])
    parser.add_argument('--checkpoint')
    parser.add_argument('--out')
    parser.add_argument('--video',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--wall-seconds',type=float,help='Total cumulative budget, including checkpoint elapsed time')
    args=parser.parse_args();os.chdir(ROOT)
    if args.phase=='night':
        if args.dry_run:
            print('tmux: CPU check -> 64-iteration Smoke + core SwanLab -> stability/independent eval gate -> 8h training + core SwanLab')
            return
        from t15_launch import launch
        launch([sys.executable,'-u','tools/run_s54_stair.py','_night'],
               'logs/s54_stair_flat6063_night_v3','night')
        return
    if args.phase=='_night':
        cpu_check()
        subprocess.run([sys.executable,'tools/run_s54_stair.py','smoke'],check=True)
        marker=Path(read(BASE/'smoke.json')['log_dir'])/'train.exitcode'
        deadline=time.monotonic()+1020
        while not marker.exists():
            if time.monotonic()>deadline:raise TimeoutError('Smoke completion marker not received; formal training NOT launched')
            time.sleep(2)
        if int(marker.read_text())!=0:
            log=marker.parent/'stdout.log'
            tail=log.read_text()[-2000:] if log.exists() else 'No trainer log was written'
            raise RuntimeError(f'Smoke failed; formal training NOT launched. Actual trainer log: {log}\n{tail}')
        smoke_gate()
        subprocess.run([sys.executable,'tools/run_s54_stair.py','train'],check=True)
        print('Smoke passed; 8h training and scalar SwanLab uploader launched in separate tmux sessions.',flush=True)
        return
    if args.phase=='check':cpu_check();return
    if args.phase=='gate':smoke_gate();return
    if args.phase=='view':
        subprocess.run([sys.executable,'tools/view_s54_stair_reference.py'],check=True);return
    if args.phase=='eval':
        if not args.checkpoint or not args.out:parser.error('eval requires --checkpoint and --out')
        from supermimic.stair_evaluation import evaluate_checkpoint
        report=evaluate_checkpoint(args.checkpoint,read(BASE/'design.json'),args.out,video=args.video)
        print(json.dumps(report['scalars'],indent=2));return
    cfg_path=BASE/('smoke.json' if args.phase=='smoke' else 'train.json')
    cfg=read(cfg_path)
    if args.phase!='resume' and any(Path(cfg['log_dir']).glob('*.pt')):
        parser.error('Run already has checkpoints; use resume or choose a new log_dir in the config')
    command=[sys.executable,'tools/t15_launch.py','--python',sys.executable,
             'train','--config',str(cfg_path),'--with-swanlab']
    if args.phase=='resume':
        checkpoint=args.checkpoint or str(Path(cfg['log_dir'])/'latest.pt')
        if not Path(checkpoint).is_file():parser.error(f'Checkpoint missing: {checkpoint}')
        command+=['--resume',checkpoint]
    if args.wall_seconds:command+=['--wall-seconds',str(args.wall_seconds)]
    if args.dry_run:command+=['--dry-run']
    subprocess.run(command,check=True)


if __name__=='__main__':main()
