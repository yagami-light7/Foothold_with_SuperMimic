"""Post-hoc scoring of recorded Foot-only rollouts; never changes a policy."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
def main():
    import torch,numpy as np
    from supermimic.stage15 import build
    from supermimic.rotations import mat_to_quat
    from supermimic.mdp import unflatten_state
    from supermimic.reference_foot import ReferenceFootCommand
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();d=a.run_dir.resolve()
    torch.set_num_threads(4);cfg=json.loads((d/'profile.json').read_text());ck=torch.load(d/'phase_a/phase_a_best.pt',map_location='cpu',weights_only=False)
    parts=build(ROOT/cfg['teacher'],cfg,'cpu',ck);records=[]
    with torch.no_grad():
        for phase in ('eval_a','eval_b'):
            if not (d/phase/'summary.json').exists():continue
            ev=json.loads((d/phase/'summary.json').read_text())
            for row in ev['rows']['foot']:
                with np.load(row['record']) as z:t=dict(z)
                gf=torch.tensor(t['goal'][:,0]);frames=torch.tensor(t['reference_frame'][:-1,0])
                zf=torch.tensor(t['latent'][:,0]);zm=parts['mp'].encode(parts['motion'].goal(frames,10))
                norm=parts['fp'].goal_norm(gf).view(-1,10,23)
                state=t['state'][:,0];rf=t['reference_frame'][:,0];ref=parts['motion'].root_p[rf].numpy()
                yaw=lambda q:np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
                angle=np.angle(np.exp(1j*(yaw(state[:,3:7])-yaw(mat_to_quat(parts['motion'].root_R[rf]).numpy()))))
                records.append(dict(phase=phase,offset_s=row['offset_s'],steps=len(gf),
                    latent_mse=float((zf-zm).square().mean()),latent_cosine=float(torch.nn.functional.cosine_similarity(zf,zm).mean()),
                    initial_latent_mse=float((zf[0]-zm[0]).square().mean()),
                    rotation_max_abs_normalized=float(norm[:,:,[3,4,5,6,7,8,14,15,16,17,18,19]].abs().max()),
                    position_max_abs_normalized=float(norm[:,:,[0,1,2,11,12,13]].abs().max()),
                    root_distance_travelled_m=float(np.linalg.norm(np.diff(state[:,:2],axis=0),axis=1).sum()),
                    reference_distance_travelled_m=float(np.linalg.norm(np.diff(ref[:,:2],axis=0),axis=1).sum()),
                    root_final_xy_error_m=float(np.linalg.norm(state[-1,:2]-ref[-1,:2])),
                    action_delta_rms_per_step=float(np.sqrt(np.mean((t['action']-t['a_prev'])**2))),
                    root_yaw_rmse_deg=float(np.sqrt(np.mean(angle**2))*180/np.pi)))
                if cfg.get('causal_heading'):
                    S=unflatten_state(torch.tensor(t['state'][:-1,0]));ap=torch.tensor(t['a_prev'][:,0])
                    xi=torch.tensor(t['xi']).repeat(len(frames),1)
                    command=ReferenceFootCommand(**{k[8:]:torch.tensor(v) for k,v in t.items() if k.startswith('command_') and k!='command_step'}).repeat(len(frames))
                    causal=parts['goals'].obs(parts['mdp'],S,ap,xi,frames,command)
                    teacher=parts['mdp'].obs(S,ap,xi,frames)
                    first=causal[:,-6:].reshape(-1,3,2)[:,:,0]
                    desired=(S[1]@first[:,:,None]).squeeze(-1)
                    true=parts['motion'].root_R[frames][:,:,0]
                    err=torch.atan2(desired[:,1],desired[:,0])-torch.atan2(true[:,1],true[:,0])
                    err=torch.atan2(err.sin(),err.cos())
                    records[-1].update(goal_heading_yaw_error_rmse_deg=float(err.square().mean().sqrt()*180/np.pi),
                        goal_heading_yaw_error_bias_deg=float(err.mean()*180/np.pi),
                        normalized_reference_orientation_gap_rms=float(((causal[:,-6:]-teacher[:,-6:])/parts['mp'].obs_norm.std()[-6:]).square().mean().sqrt()))
    result=dict(note='Diagnostic only. Recorded bf16 Foot latents vs CPU fp32 Teacher labels at EXACT same reference frames; no policy inputs/weights/norms changed. This is not an extra Newton evaluation.',rows=records)
    (d/'latent_alignment_posthoc.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
