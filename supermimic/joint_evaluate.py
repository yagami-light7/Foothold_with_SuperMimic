"""Serialized child evaluator: trainer is idle, no training/normalizer updates."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    import argparse,json,time,subprocess,shutil
    from isaaclab.app import add_launcher_args,launch_simulation
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--interface',choices=['motion','foot'],required=True);p.add_argument('--full',action='store_true')
    p.add_argument('--video',action='store_true')
    add_launcher_args(p);a=p.parse_args()
    import numpy as np,torch
    from supermimic import networks
    from supermimic.joint_runtime import project,json_write
    from supermimic.motion_prior_runtime import reconstruct
    from supermimic.env import TrackingSim,make_sim
    from supermimic.foot_evaluate import evaluate_foot
    from supermimic.joint_evaluation import motion_episode,analyze_trace
    out=Path(a.out)
    if out.exists():raise FileExistsError('Evaluation output already exists')
    out.mkdir(parents=True)
    started=time.monotonic();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    joint=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cfg=joint['profile'];ck=project(joint,a.interface)
    networks.AMP_DTYPE=torch.bfloat16 if cfg['amp']=='bf16' else None
    kin,motion,goals,mdp,policy=reconstruct(ck,'cuda:0')
    wm=networks.WorldModel().to('cuda:0');wm.load_state_dict(ck['wm']);wm.eval()
    before={k:v.clone() for k,v in policy.state_dict().items()};rows=[]
    sim=make_sim('cuda:0',a)
    with launch_simulation(sim,a):
        env=TrackingSim(sim,1,motion,mdp,goals=goals,eval_mode=True)
        if a.interface=='motion':
            cases=[(54,float(motion.clip_end[0])*.02,0)] if a.full else [(54,cfg['motion_probe_duration'],offset) for offset in cfg['motion_probe_offsets']]
        else:cases=[(seed,d,0) for d in cfg['foot_eval_durations'] for seed in cfg['foot_eval_seeds']]
        for seed,duration,offset in cases:
            path=out/f'{a.interface}_{duration:g}s_seed{seed}_offset{offset:g}.npz'
            result=(motion_episode(env,policy,mdp,goals,seed,duration,offset,path) if a.interface=='motion' else
                    evaluate_foot(env,ck,goals,mdp,path,device='cuda:0',duration_s=duration,seed=seed,reset_noise=True,diagnostics=True))
            metrics,wm_pair=analyze_trace(path,mdp,goals,wm)
            row=dict(**result,metrics=metrics,wm_paired=wm_pair,trace=str(path));rows.append(row)
            json_write(out/'partial.json',dict(status='INCOMPLETE_UNTIL_SUMMARY',rows=rows))
            print(json.dumps(dict(interface=a.interface,seed=seed,duration=duration,offset=offset,
                norm_return=row['norm_return'],steps=row['episode_lengths'],stop=row['stop_reasons'])),flush=True)
    assert all(torch.equal(v,policy.state_dict()[k]) for k,v in before.items()),'Evaluation changed Actor/norm'
    preview=None;preview_status='NOT EVALUATED'
    if a.video and rows:
        row=rows[0] if a.interface=='motion' else next(r for r in rows if r['duration_s']==10 and r['seed']==54)
        seconds=min(10.,row['episode_lengths'][0]*.02);records=[row['trace']];labels=[a.interface+' Newton policy']
        if a.interface=='motion':
            from supermimic.mdp import flatten_state
            with np.load(row['trace'],allow_pickle=False) as z:frames=torch.as_tensor(z['reference_frame'][:,0],device='cuda:0')
            ref=out/'reference_visualization_only.npz'
            np.savez_compressed(ref,state=flatten_state(mdp.motion.ref_state(frames))[:,None].cpu().numpy(),visualization_only=True)
            records=[str(ref),*records];labels=['LAFAN1 kinematic reference',*labels]
        extension='.mp4' if shutil.which('ffmpeg') else '.gif'
        preview=str(out/('preview_'+a.interface+extension))
        render=[sys.executable,str(ROOT/'tools/visualization/render_s54_motion_prior.py'),'--records',*records,
                '--labels',*labels,'--out',preview,'--seconds',str(seconds),'--fps','25']
        try:
            subprocess.run(render,check=True,timeout=120);preview_status='recorded states only; capped to actual physical execution'
        except (subprocess.SubprocessError,OSError) as error:preview_status='NOT EVALUATED: '+str(error);preview=None
    report=dict(interface=a.interface,iteration=joint['iteration'],checkpoint=str(a.checkpoint),full=a.full,
        rows=rows,wall_seconds=time.monotonic()-started,eval_real_steps=sum(sum(r['episode_lengths']) for r in rows),normalization_unchanged=True,
        preview=preview,preview_status=preview_status,
        protocol='single-env Newton, fixed seeds/starts; Motion no noise/domain randomization; Foot configured reset noise only')
    json_write(out/'summary.json',report);print(out/'summary.json',flush=True)


if __name__=='__main__':main()
