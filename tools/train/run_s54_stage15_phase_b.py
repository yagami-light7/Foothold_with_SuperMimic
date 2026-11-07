"""Bounded reference-only Motion BPTT; E_f is the only Actor optimizer owner."""
import argparse,json,sys,time,signal,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))

def main():
    from isaaclab.app import add_launcher_args
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/motion_prior/stage15/lafan1_skill_transfer_3090.json')
    p.add_argument('--phase-a',required=True);p.add_argument('--gate',required=True);p.add_argument('--out',required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--resume')
    p.add_argument('--diagnostic-bounded',action='store_true',help='Explicit post-diagnosis repair trial, <=400 updates / 600s; never marks failed A gate as passed')
    add_launcher_args(p);a=p.parse_args()
    import torch,numpy as np,random
    from supermimic import networks
    from supermimic.stage15 import build,seed_all,atomic_save,checkpoint,tensor_hash,split_frames
    from supermimic.stage15_updates import Stage15Replay,Stage15Updates
    from supermimic.joint_runtime import json_write
    from supermimic.mdp import unflatten_state,normalize_xi
    from supermimic.control import bound_action
    from tools.visualization.policy_preview import gpu_slot
    cfg=json.loads(Path(a.config).read_text());out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'phase_b_last.pt').exists() and not a.resume:raise FileExistsError('No overwrite; use fresh directory or explicit resume')
    gate=json.loads(Path(a.gate).read_text())
    if not a.smoke:
        if gate['protocol']['checkpoint_sha256']!=__import__('supermimic.joint_runtime',fromlist=['digest']).digest(a.phase_a):
            raise RuntimeError('Phase A checkpoint / evaluation mismatch')
        if a.diagnostic_bounded:
            rows=gate['rows']['foot']
            if (cfg['phase_b_iterations']>400 or cfg['phase_b_budget_s']>600 or not cfg.get('diagnostic_bounded_trial')
                    or gate['protocol']['oracle_latent_diagnostic_only'] or len(rows)!=3
                    or any(r['n_failed'] or r['episode_lengths']!=[500] or r['norm_return']<.65 for r in rows)):
                raise RuntimeError('Bounded repair trial requires diagnosed, finite 3/3 10s walking and hard resource caps')
        elif not gate['gate_passed']:
            raise RuntimeError('Phase A Newton quality gate has not passed; diagnose first')
    ck=torch.load(a.resume or a.phase_a,map_location='cpu',weights_only=False);start=time.monotonic()
    with gpu_slot():
        seed_all(cfg['seed']);torch.set_num_threads(4);networks.AMP_DTYPE=torch.bfloat16
        parts=build(ROOT/cfg['teacher'],cfg,'cuda:0',ck);fp,wm=parts['fp'],parts['wm'];mdp=parts['paired']
        teacher_hash=tensor_hash(parts['mp'].state_dict());norm_hash=tensor_hash(fp.goal_norm.state_dict())
        fp_hash=tensor_hash(fp.enc.state_dict());wm_hash=tensor_hash(wm.state_dict())
        critic=networks.Critic(365).to('cuda:0')
        opts={'motion':torch.optim.Adam(fp.enc.parameters(),lr=cfg['lr_actor'],betas=(.9,.95)),
              'wm':torch.optim.Adam(wm.parameters(),lr=cfg['lr_wm']),
              'critic_motion':torch.optim.Adam(critic.parameters(),lr=cfg['lr_critic'])}
        counts=dict(newton=0,wm=0,actor=0,critic=0);previous_wall=0.;first=0
        if a.resume:
            critic.load_state_dict(ck['critic'],strict=True)
            for k,v in opts.items():v.load_state_dict(ck['optimizers'][k])
            counts=ck['counts'];previous_wall=ck['wall_seconds'];first=ck['iteration']
            torch.set_rng_state(ck['rng']['torch']);torch.cuda.set_rng_state_all(ck['rng']['cuda']);np.random.set_state(ck['rng']['numpy']);random.setstate(ck['rng']['python'])
        actor_before=counts['actor']
        # The physical guards become stricter by retaining the original Foot guard too.
        original_termination=mdp.termination
        def physical(S,f,**kw):
            finite=torch.isfinite(__import__('supermimic.mdp',fromlist=['flatten_state']).flatten_state(S)).all(-1)
            return original_termination(S,f,**kw)|parts['guard'].termination(S)|(~finite)
        mdp.termination=physical
        N=16 if a.smoke else cfg['num_envs'];B=Stage15Replay(cfg['buffer_steps'],N,'cuda:0')
        if a.smoke:cfg=dict(cfg,wm_batch=32,wm_batches=1,pi_batch=8,wm_only_iters=0)
        updates=Stage15Updates(parts,critic,opts,B,cfg)
        train,_=split_frames(parts['motion'].num_frames);train=train.to('cuda:0')
        # Keep complete training episodes outside the evaluation intervals.
        valid=train.clone();keep=torch.ones_like(valid,dtype=torch.bool)
        for off in (0,3000,6000):keep &= ~((valid+500>=off-10)&(valid<=off+510))
        train=valid[keep];train=train[train+510<parts['motion'].num_frames]
        from supermimic.env import TrackingSim,make_sim
        from isaaclab.app import launch_simulation
        sim=make_sim('cuda:0',a);history=None;last={};audit=None;it=first
        def save():
            atomic_save(checkpoint(parts,cfg,'B_SMOKE' if a.smoke else ('B_DIAGNOSTIC' if a.diagnostic_bounded else 'B'),it,opts['motion'],normalization=ck['normalization'],
                critic=critic.state_dict(),optimizers={k:v.state_dict() for k,v in opts.items()},counts=counts,
                wall_seconds=previous_wall+time.monotonic()-start,
                replay_restore='Not serialized. Refill at least warmup_steps with fresh reference resets; WM-only 20 iterations after resume; simulator contact solver state is fresh'),out/'phase_b_last.pt')
            history_path=out/f'phase_b_{it:06d}.pt'
            if not history_path.exists():__import__('os').link(out/'phase_b_last.pt',history_path)
            for old in sorted(out.glob('phase_b_[0-9]*.pt'))[:-2]:old.unlink()
        stopping=[False]
        signal.signal(signal.SIGTERM,lambda *_:stopping.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *_:stopping.__setitem__(0,True))
        try:
            with launch_simulation(sim,a):
                env=TrackingSim(sim,N,parts['motion'],mdp,eval_mode=False,reset_noise=False,max_episode_steps=500,goals=parts['goals'])
                def sample_frames(self,n):
                    frames=train[torch.randint(len(train),(n,),device='cuda:0')]
                    return frames,self.motion.frame_bin[frames]
                env.sample_frames=types.MethodType(sample_frames,env)
                env.reset(torch.arange(N,device='cuda:0'))
                @torch.no_grad()
                def collect(steps):
                    nonlocal history
                    collection_start=time.monotonic()
                    rs=[];ts=[]
                    for _ in range(steps):
                        S=env.state();obs=parts['goals'].obs(mdp,S,env.a_prev,env.xi,env.f,env.command)
                        history=obs[:,None].repeat(1,10,1) if history is None else torch.cat((history[:,1:],obs[:,None]),1)
                        reset=env.ep_step==0;history[reset]=obs[reset,None].expand(-1,10,-1)
                        g=parts['goals'].goal(S,env.f,env.ep_step,env.command)
                        action=bound_action(fp(history,g)+torch.randn(N,29,device='cuda:0')*cfg['action_noise'])
                        tr=env.step(action);B.add(tr);counts['newton']+=N
                        # Keep pretrained WM scales fixed during this bounded transfer.
                        rs.append(float(tr['r'].mean()/mdp.max_step_reward));ts.append(float(tr['term'].float().mean()))
                    return dict(sim_reward=sum(rs)/len(rs),termination_fraction=sum(ts)/len(ts),
                                collection_fps=steps*N/(time.monotonic()-collection_start))
                collect(max(cfg['warmup_steps'],cfg['hist']+cfg['wm_window']))
                limit=(first+22 if a.resume else 2) if a.smoke else cfg['phase_b_iterations']
                for it in range(first+1,limit+1):
                    iteration_start=time.monotonic()
                    last=collect(cfg['rollout_steps'])
                    for _ in range(cfg['wm_batches']):last.update(updates.wm_update());counts['wm']+=1
                    if it>cfg['wm_only_iters'] and (not a.resume or it>first+20):
                        stats=updates.actor_critic(audit=(counts['actor']==actor_before));last.update(stats)
                        if counts['actor']==actor_before:audit=stats
                        counts['actor']+=1;counts['critic']+=cfg['critic_epochs']*cfg['critic_minibatches']
                    last.update(iteration=it,wall_seconds=previous_wall+time.monotonic()-start,**counts,
                                peak_torch_mib=torch.cuda.max_memory_allocated()/2**20,
                                iterations_per_second=1/(time.monotonic()-iteration_start))
                    if it%cfg['log_every']==0 or a.smoke or it==limit:
                        with (out/'phase_b_metrics.jsonl').open('a') as f:f.write(json.dumps(last)+'\n')
                        print('[B] '+json.dumps(last),flush=True)
                    if it%cfg['checkpoint_every']==0:save()
                    if stopping[0] or last['wall_seconds']>=cfg['phase_b_budget_s']:break
        finally:save()
        assertions=dict(teacher_and_decoder_unchanged=tensor_hash(parts['mp'].state_dict())==teacher_hash,
            goal_norm_unchanged=tensor_hash(fp.goal_norm.state_dict())==norm_hash,
            obs_norm_matches_teacher=tensor_hash(fp.obs_norm.state_dict())==tensor_hash(parts['mp'].obs_norm.state_dict()),
            foot_encoder_updated=tensor_hash(fp.enc.state_dict())!=fp_hash,wm_supervised_updated=tensor_hash(wm.state_dict())!=wm_hash)
        json_write(out/'phase_b_audit.json',dict(assertions,first_actor_audit=audit,counts=counts,smoke=a.smoke,
            wall_seconds=time.monotonic()-start,gpu=torch.cuda.get_device_name(),peak_torch_mib=torch.cuda.max_memory_allocated()/2**20))
        if counts['actor']>actor_before and not all(assertions.values()):raise RuntimeError('Stage15 B gradient/freeze audit failed')
if __name__=='__main__':main()
