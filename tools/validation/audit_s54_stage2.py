"""CPU-only deterministic Stage2 transfer/distribution audit, no normalization fitting."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
def main():
 import torch
 from supermimic.stage2_curriculum import build
 from supermimic.foot_trajectory import query,command_time
 from supermimic.rotations import quat_to_mat,yaw_mat
 from supermimic.joint_runtime import json_write
 from supermimic.stage15 import tensor_hash
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 cfg=json.loads(Path(a.config).read_text());torch.set_num_threads(4);torch.manual_seed(cfg['seed'])
 parts=build(cfg,'cpu');pi=parts['policy'];g=parts['goals'];cur=parts['curriculum'];mdp=parts['mdp'];rows=[]
 with torch.no_grad():
  for stage in range(5):
   cur.stage=stage;cur.progress=1.;cur.apply_stage();S=mdp.nominal_state(64);cmd=g.new_command(S)
   ticks=(torch.linspace(0,9.5,64)/.02).long();ref=mdp._reference(cmd,ticks)
   pos=ref.position[:,0].mean(1);pos[:,2]=mdp.nominal_height;R=yaw_mat(quat_to_mat(ref.quaternion[:,0,0]))
   pos[:,:2]+=torch.einsum('nij,j->ni',R[:,:2,:2],mdp.root_xy_offset)
   states=(pos,R,*S[2:]);goal=g.goal(states,ticks,ticks,cmd);norm=pi.goal_norm(goal).reshape(64,10,23);z=pi.encode(goal)
   rot=norm[:,:,[3,4,5,6,7,8,14,15,16,17,18,19]].abs()
   obs=g.obs(mdp,states,torch.zeros(64,29),torch.zeros(64,33),ticks,cmd)
   rows.append(dict(stage=stage,max_abs=float(norm.abs().max()),p99=float(torch.quantile(norm.abs().flatten(),.99)),rotation_max=float(rot.max()),rotation_p99=float(torch.quantile(rot.flatten(),.99)),fraction_abs_over_6=float((norm.abs()>6).float().mean()),latent_saturation=float((z.abs()>.95).float().mean()),obs_finite=bool(torch.isfinite(obs).all()),goal230=list(goal.shape)))
   if not torch.isfinite(goal).all() or not torch.isfinite(obs).all() or rot.max()>=15:raise RuntimeError('Invalid/outlier Quintic goal distribution')
 source=torch.load(cfg['phase_a_checkpoint'],map_location='cpu',weights_only=False)
 teacher=torch.load(source['config']['teacher'],map_location='cpu',weights_only=False)
 # Teacher joint checkpoint uses canonical shared WM; equality is all tensors, not just dimensions.
 wm_equal=tensor_hash(source['wm'])==tensor_hash(teacher['wm'])
 if not wm_equal:raise ValueError('Phase A WM is not the corresponding Teacher WM')
 result=dict(method='Frozen Phase A norms. 64 generated plans/stage, t=0..9.5s, root follows planned support center; not a physical reachability certificate',rows=rows,transfer=parts['transfer'],contract=parts['contract'],teacher_wm_exact=wm_equal)
 json_write(a.out,result);print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
