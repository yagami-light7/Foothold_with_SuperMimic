"""Bounded original-Trainer + frozen-Newton smoke, never a formal training job."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix',default='roban_arch_smoke')
    p.add_argument('--experiments',nargs='+',choices=['E0','E1','E2','terrain_smoke'],default=['E0','E1','E2','terrain_smoke'])
    args=p.parse_args()
    if not args.prefix.startswith('roban_') or '/' in args.prefix or len(set(args.experiments))!=len(args.experiments):
        p.error('Use a unique roban_* prefix and unique experiments')
    report_dir=ROOT/'reports'/args.prefix
    paths={e:ROOT/'logs'/f'{args.prefix}_{e}' for e in args.experiments}
    if report_dir.exists() or any(path.exists() for path in paths.values()):
        p.error('Smoke outputs already exist; choose a new prefix')
    report_dir.mkdir(parents=True)
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Newton smoke requires CUDA; run through the existing kit-less image')
    report=dict(status='RUNNING',gpu=torch.cuda.get_device_name(),experiments={},formal_training=False)
    def run(label,argv):
        with (report_dir/f'{label}.log').open('w') as sink:
            subprocess.run([sys.executable,*argv],cwd=ROOT,stdout=sink,stderr=subprocess.STDOUT,check=True,timeout=300)
    try:
        for experiment,path in paths.items():
            print('Running',experiment,flush=True)
            run(experiment,['supermimic/train.py','--roban_config',f'configs/roban/{experiment}.json','--log_dir',str(path)])
            ck=torch.load(path/'last.pt',map_location='cpu',weights_only=False)
            rows=list(csv.DictReader((path/'train_log.csv').open()))
            required=['wm_loss','wm_err8','pi_loss','v_loss','pi_gradnorm','imag_survival']
            if len(rows)!=4 or any(not __import__('math').isfinite(float(r[k])) for r in rows for k in required):
                raise AssertionError('Missing/nonfinite training metrics')
            if not any(float(r['pi_gradnorm'])>0 for r in rows[1:]):raise AssertionError('Actor did not receive BPTT gradients')
            if not any(float(r['foot_actor_gradnorm'])>0 for r in rows[1:]):raise AssertionError('Primary foot reward did not reach Actor')
            if not ck['opt_pi']['state'] or not ck['opt_wm']['state'] or not ck['opt_v']['state']:
                raise AssertionError('Optimizer state was not saved')
            for key in ('pi','wm','critic'):
                if any(not torch.isfinite(v).all() for v in ck[key].values() if v.is_floating_point()):
                    raise AssertionError('Nonfinite checkpoint')
            summary=dict(status='PASS',iterations=ck['iteration'],sim_steps=ck['sim_steps'],
                         wm_loss=float(rows[-1]['wm_loss']),critic_loss=float(rows[-1]['v_loss']),
                         actor_gradnorm=float(rows[-1]['pi_gradnorm']),
                         foot_actor_gradnorm=float(rows[-1]['foot_actor_gradnorm']),
                         obs_dim=ck['args']['obs_dim'],goal_dim=ck['goal_contract']['dim'],
                         critic_input=ck['critic']['norm.mean'].numel(),wm_input=ck['wm']['in_norm.mean'].numel(),
                         wm_output=ck['wm']['out_norm.mean'].numel(),
                         source_contact_fraction=ck['data_contract']['trusted_contact_fraction'],
                         prior_quality_mean=ck['data_contract']['body_prior_quality_mean'])
            report['experiments'][experiment]=summary
            print(experiment,summary,flush=True)
            if experiment=='E1':
                step=lambda c:float(next(iter(c['opt_pi']['state'].values()))['step'])
                before=step(ck)
                run('resume',['supermimic/train.py','--roban_config','configs/roban/E1.json','--log_dir',str(path),
                              '--resume',str(path/'last.pt'),'--iterations','6','--recovery_wm_iters','0'])
                resumed=torch.load(path/'last.pt',map_location='cpu',weights_only=False)
                if resumed['iteration']!=6 or step(resumed)!=before+2:raise AssertionError('Optimizer progress did not resume')
                if torch.equal(ck['pi']['dec.8.weight'],resumed['pi']['dec.8.weight']):raise AssertionError('Actor parameters did not update after resume')
                report['resume']=dict(status='PASS',iteration=6,adam_actor_step_before=before,adam_actor_step_after=step(resumed))
                run('newton',['tools/roban_rollout.py','--checkpoint',str(path/'last.pt'),
                              '--out',str(report_dir/'newton'),'--num_envs','4','--steps','100'])
                report['newton']=json.loads((report_dir/'newton/summary.json').read_text())
        report['status']='PASS'
    except Exception as exc:
        report.update(status='FAIL',error=repr(exc));raise
    finally:
        (report_dir/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print('Smoke report:',report_dir/'summary.json',flush=True)


if __name__=='__main__':main()
