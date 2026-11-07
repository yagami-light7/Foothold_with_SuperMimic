"""Bounded latent distillation. Only the fresh Foot Encoder has an optimizer."""
import argparse,json,sys,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/motion_prior/stage15/lafan1_skill_transfer_3090.json')
    p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--resume');a=p.parse_args()
    import torch
    from supermimic import networks
    from supermimic.stage15 import seed_all,build,dataset,checkpoint,atomic_save,tensor_hash
    from supermimic.joint_runtime import json_write
    from tools.visualization.policy_preview import gpu_slot
    cfg=json.loads(Path(a.config).read_text());out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'phase_a_last.pt').exists() and not a.resume:raise FileExistsError('Choose a fresh experiment directory')
    json_write(out/'profile.json',cfg);start=time.monotonic()
    with gpu_slot():
        seed_all(cfg['seed']);torch.set_num_threads(4);networks.AMP_DTYPE=torch.bfloat16
        saved=torch.load(a.resume,map_location='cpu',weights_only=False) if a.resume else None
        previous_wall=saved.get('wall_seconds',0.) if saved else 0.
        elapsed=lambda:previous_wall+time.monotonic()-start
        parts=build(ROOT/cfg['teacher'],cfg,'cuda:0',saved);fp=parts['fp'];mp=parts['mp']
        frozen_hash=tensor_hash(mp.state_dict());wm_hash=tensor_hash(parts['wm'].state_dict())
        X,Y,V,Z,meta=dataset(parts,cfg);json_write(out/'normalization.json',meta)
        if saved and meta['sha256']!=saved['normalization']['sha256']:raise ValueError('Training normalization changed at resume')
        norm_hash=tensor_hash(fp.goal_norm.state_dict());obs_hash=tensor_hash(fp.obs_norm.state_dict())
        opt=torch.optim.Adam(fp.enc.parameters(),lr=cfg['phase_a_lr']);initial=tensor_hash(fp.enc.state_dict())
        first=0
        if saved:
            opt.load_state_dict(saved['optimizer']);first=saved['iteration']
            torch.set_rng_state(saved['rng']['torch']);torch.cuda.set_rng_state_all(saved['rng']['cuda'])
            __import__('numpy').random.set_state(saved['rng']['numpy']);__import__('random').setstate(saved['rng']['python'])
        limit=first+3 if a.smoke else cfg['phase_a_iterations'];best=float('inf');records=[]
        for it in range(first+1,limit+1):
            ids=torch.randint(len(X),(cfg['phase_a_batch'],),device='cuda');z=fp.encode(X[ids]);loss=(z-Y[ids]).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite latent loss')
            opt.zero_grad(set_to_none=True);loss.backward()
            gn=torch.nn.utils.clip_grad_norm_(fp.enc.parameters(),1.,error_if_nonfinite=True);opt.step()
            if it==first+1 or it%cfg['log_every']==0 or it==limit:
                with torch.no_grad():
                    val=torch.cat([fp.encode(v) for v in V.split(512)]);mse=float((val-Z).square().mean())
                    cosine=float(torch.nn.functional.cosine_similarity(val,Z).mean())
                    sat=float((val.abs()>.95).float().mean())
                row=dict(iteration=it,wall_seconds=elapsed(),train_latent_mse=float(loss.detach()),
                    val_latent_mse=mse,val_latent_cosine=cosine,val_saturation=sat,actor_gradnorm=float(gn),
                    peak_torch_mib=torch.cuda.max_memory_allocated()/2**20)
                records.append(row)
                with (out/'phase_a_metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(f'[A] {it}/{limit} train={row["train_latent_mse"]:.6f} val={mse:.6f} cos={cosine:.4f} sat={sat:.3f} grad={float(gn):.3g} {row["wall_seconds"]:.1f}s',flush=True)
                if mse<best:
                    best=mse;atomic_save(checkpoint(parts,cfg,'A',it,opt,normalization=meta,wall_seconds=row['wall_seconds']),out/'phase_a_best.pt')
            if it%cfg['checkpoint_every']==0 or it==limit or elapsed()>=cfg['phase_a_budget_s']:
                atomic_save(checkpoint(parts,cfg,'A',it,opt,normalization=meta,wall_seconds=elapsed()),out/'phase_a_last.pt')
            if elapsed()>=cfg['phase_a_budget_s']:break
        assertions=dict(teacher_unchanged=tensor_hash(mp.state_dict())==frozen_hash,
            wm_unchanged=tensor_hash(parts['wm'].state_dict())==wm_hash,
            fixed_goal_norm=tensor_hash(fp.goal_norm.state_dict())==norm_hash,
            fixed_obs_norm=tensor_hash(fp.obs_norm.state_dict())==obs_hash,
            foot_encoder_changed=tensor_hash(fp.enc.state_dict())!=initial,
            only_foot_encoder_has_grad=all(v.grad is None for v in mp.parameters()) and all(v.grad is None for v in parts['wm'].parameters()))
        json_write(out/'phase_a_audit.json',dict(assertions,iterations=it,distillation_samples=it*cfg['phase_a_batch'],
            resumed_from_iteration=first,new_updates=it-first,
            newton_transitions=0,best_validation_mse=best,wall_seconds=elapsed(),
            mean_updates_per_second=it/elapsed(),smoke=a.smoke))
        if not all(assertions.values()):raise RuntimeError('Freeze/normalization audit failed')
        print(out/'phase_a_best.pt',flush=True)
if __name__=='__main__':main()
