"""Audit real joint smoke artifacts on CPU before permitting a formal launch."""
from pathlib import Path
import argparse,json,math,subprocess,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from supermimic.joint_runtime import digest,json_write


def check_run(name,scale=None):
    d=ROOT/'logs'/name;cfg=json.loads((d/'profile.json').read_text())
    status=json.loads((d/'training_status.json').read_text())
    assert cfg['mode']=='smoke' and status['status']=='COMPLETED_BY_ITER_CAP',name
    assert int((d/'train.exitcode').read_text())==0
    assert not (d/'swanlab_run_id').exists(),'Non-formal run must never upload'
    if scale:assert cfg['num_envs_motion']==cfg['num_envs_foot']==scale
    ck=torch.load(d/'last.pt',map_location='cpu',weights_only=False);c=ck['counters']
    assert ck['iteration']==cfg['iterations'] and ck['iteration']==status['iteration']
    assert c==status['counters']
    assert c['real_motion']==c['real_foot']>0 and c['actor_motion']==c['actor_foot']>0
    assert c['wm_updates']==cfg['iterations']*cfg['wm_batches']
    for n in ('motion','foot'):
        assert c['wm_windows_'+n]==c['wm_updates']*cfg['wm_batch']//2
        assert c['critic_'+n]==c['actor_'+n]*cfg['critic_epochs']*cfg['critic_minibatches']
        assert c['imagined_'+n]==c['actor_'+n]*cfg['pi_batch_'+n]*cfg['horizon']
        schema=ck['replay_meta'][n]['contract']
        assert schema['interface']==n and 'start_frame' in schema['fields']
    assert not any(k.startswith('dec.') for k in ck['policy']['foot'])
    for component in [ck['wm'],*ck['policy'].values(),*ck['critics'].values()]:
        assert all(torch.isfinite(v).all() for v in component.values() if v.is_floating_point())
    for n,opt in ck['optimizers'].items():
        key='wm_updates' if n=='wm' else 'actor_'+n if n in ('motion','foot') else n
        assert {int(v['step']) for v in opt['state'].values()}=={c[key]},n
    audit=[json.loads(line) for line in (d/'gradient_audit.jsonl').read_text().splitlines()]
    for name in ('motion','foot'):
        rows=[r for r in audit if r['interface']==name];assert rows
        for r in rows:
            assert r['delta_wm_actor']==0 and r['delta_encoder']>0 and r['latent_gradnorm']>0
            assert math.isfinite(r['actor_gradnorm']) and math.isfinite(r['critic_loss'])
            assert r['delta_decoder']==0 if name=='foot' else r['delta_decoder']>0
    return dict(run=str(d.relative_to(ROOT)),status=status['status'],iteration=ck['iteration'],
                counters=c,checkpoint_sha256=digest(d/'last.pt'),gradient_audits=len(audit),
                runtime=json.loads((d/'runtime_stats.json').read_text()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ledger',type=Path,required=True)
    p.add_argument('--resume_run',required=True);p.add_argument('--release_run',required=True)
    p.add_argument('--tests_log',type=Path,required=True);p.add_argument('--video_summary',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    ledger=json.loads(a.ledger.read_text());tests=a.tests_log.read_text()
    assert 'Ran 54 tests' in tests and tests.rstrip().endswith('OK')
    proofs={}
    for key,n in [('one',1),('scale256',256),('scale512',512)]:
        proofs[key]=check_run(ledger['runs'][key]['run'],n)
    proofs['release512']=check_run(a.release_run,512);proofs['resume16']=check_run(a.resume_run,16)
    d=ROOT/proofs['resume16']['run'];before=[json.loads(v.read_text()) for v in d.glob('training_status.json.before_resume_*')]
    assert any(v['status']=='STOPPED_BY_REQUEST' and v['iteration']==4 for v in before)
    assert proofs['resume16']['iteration']==8
    assert proofs['resume16']['counters']['recovery_real_motion']==2048
    assert proofs['resume16']['counters']['recovery_real_foot']==2048
    evaluations={}
    for n,expected in [('motion',3),('foot',9)]:
        path=next(d.glob('evaluation/*/'+n+'/summary.json'));v=json.loads(path.read_text())
        assert v['normalization_unchanged'] and len(v['rows'])==expected
        for r in v['rows']:
            assert r['episode_lengths'][0]>32 and math.isfinite(r['norm_return'])
            assert set(r['wm_paired']['horizons'])=={'1','4','8','16','32'}
            for values in r['wm_paired']['horizons'].values():
                assert all(math.isfinite(values[k]) for k in ['base_height_rmse_m','root_pitch_rmse_deg','feet_position_rmse_m','paired_reward_mae'])
        evaluations[n]=str(path.relative_to(ROOT))
    video=json.loads(a.video_summary.read_text());assert video['preview'] and Path(video['preview']).is_file()
    if Path(video['preview']).suffix=='.gif':
        from PIL import Image
        with Image.open(video['preview']) as im:assert im.is_animated and im.n_frames>1
    else:
        probe=json.loads(subprocess.check_output(['ffprobe','-v','quiet','-print_format','json','-show_format','-show_streams',video['preview']],text=True))
        assert float(probe['format']['duration'])>0
        assert any(stream['codec_type']=='video' for stream in probe['streams'])
    files=sorted(set(list((ROOT/'supermimic').glob('*.py'))+list((ROOT/'configs').rglob('*.json'))+
        list((ROOT/'tools').rglob('*.py'))+list((ROOT/'tools').rglob('*.sh'))+[ROOT/'tools/layout_migration.json']))
    source={str(p.relative_to(ROOT)):digest(p) for p in files}
    formal=json.loads((ROOT/'configs/motion_prior/joint_ab1/joint_ab1_3090_12h.json').read_text())
    release=json.loads((ROOT/proofs['release512']['run']/'profile.json').read_text())
    for key in ['num_envs_motion','num_envs_foot','hist','goal_frames','motion_goal_dt','foot_goal_dt','horizon','wm_window','wm_batch','wm_batches','pi_batch_motion','pi_batch_foot','buffer_steps','warmup_steps','utd','gamma','lam','action_noise','critic_epochs','critic_minibatches','amp','seed','prior_config','paired_latent_weight','lr_wm','lr_motion_encoder','lr_decoder','lr_foot_encoder','lr_critic']:
        assert release[key]==formal[key],key
    value=dict(passed=True,num_envs_per_interface=512,cpu_tests=54,cpu_log=str(a.tests_log),
        proofs=proofs,evaluations=evaluations,video_summary=str(a.video_summary),video=video['preview'],
        preserved_failed_preflight=dict(run='s54_joint_ab1_smoke_eval_resume_v2_20261106_021910_seed54',
            reason='paired_predictions tuple was incorrectly treated as summary dict; fixed and regression-tested'),
        limitations='Engineering gates only; initial Foot survival fails and no convergence/holdout claim. Resume refills Replay; Newton contact warm state is fresh.',
        source_hashes=source)
    json_write(a.out,value);print('PASS: formal preflight; evidence:',a.out)


if __name__=='__main__':main()
