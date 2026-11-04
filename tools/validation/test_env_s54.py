"""Strict S54 Newton smoke: canonical roundtrip, native-FK alignment, targets and rollout."""

# Support both direct execution and python -m from the repository layout.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_root = str(_ToolPath(__file__).resolve().parents[2])
if _tool_root not in _tool_sys.path:
    _tool_sys.path.insert(0, _tool_root)

import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from isaaclab.app import add_launcher_args, launch_simulation
p=argparse.ArgumentParser();p.add_argument('--num_envs',type=int,default=1);p.add_argument('--steps',type=int,default=60)
p.add_argument('--motion_dir',default='data/retarget_data/reference');p.add_argument('--out',default=None)
add_launcher_args(p);args=p.parse_args()
if not 1<=args.num_envs<=16:p.error('S54 initial smoke is limited to 1–16 envs')
if not 20<=args.steps<=100:p.error('Use 20–100 steps')
import torch
import warp as wp
from newton import ShapeFlags
from supermimic.env import TrackingSim,make_sim
from supermimic.kinematics import S54Kinematics
from supermimic.motion import MotionLib
from supermimic.mdp import MDP,MAX_STEP_REWARD
from supermimic.s54 import TRACKED_BODIES,default_joint_pos_list,NUM_J
from supermimic.rotations import quat_to_mat,so3_log


def main():
    dev='cuda:0';kin=S54Kinematics(dev);motion=MotionLib.from_dir(args.motion_dir,kin,dev)
    mdp=MDP(kin,motion,torch.tensor(default_joint_pos_list(),device=dev))
    sim_cfg=make_sim(dev,args);report={}
    with launch_simulation(sim_cfg,args):
        env=TrackingSim(sim_cfg,args.num_envs,motion,mdp,eval_mode=True)
        ids=torch.arange(args.num_envs,device=dev);frames=motion.clip_start[ids%len(motion.names)]+100
        env.reset(ids,frames);env.refresh();S=env.state();ref=motion.ref_state(frames)
        report['roundtrip']={n:float((x-y).abs().max()) for n,x,y in zip(['p','R','v','w','q','qd'],S,ref)}
        assert max(report['roundtrip'].values())<1e-4,report
        def alignment():
            S=env.state();bp,bR,bv,bw=kin.forward(*S);d=env.robot.data
            body_ids=[env.robot.body_names.index(b) for b in TRACKED_BODIES]
            pose=d.body_link_pose_w.torch[:,body_ids];vel=d.body_link_vel_w.torch[:,body_ids]
            xyz=pose[:,:,:3]-env.scene.env_origins[:,None,:];q=pose[:,:,3:7]
            R=quat_to_mat(torch.cat((q[...,3:4],q[...,:3]),-1))
            out=dict(position_m=float((bp-xyz).norm(dim=-1).max()),
                     rotation_rad=float(so3_log(bR.transpose(-1,-2)@R).norm(dim=-1).max()),
                     linear_velocity=float((bv-vel[...,:3]).norm(dim=-1).max()),
                     angular_velocity=float((bw-vel[...,3:]).norm(dim=-1).max()))
            assert out['position_m']<1e-4 and out['rotation_rad']<1e-3,out
            assert out['linear_velocity']<1e-3 and out['angular_velocity']<1e-3,out
            return out
        report['reset_fk']=alignment()
        backend=env.physics;model=backend.get_model()
        shape_body=wp.to_torch(model.shape_body)
        collidable=(wp.to_torch(model.shape_flags).long() & int(ShapeFlags.COLLIDE_SHAPES))!=0
        ground=int(((shape_body<0)&collidable).sum())
        report['world_shapes']=ground;assert ground==1,f'Expected exactly one ground; got {ground}'
        report['gravity']=list(sim_cfg.gravity);assert sim_cfg.gravity[2]<0
        contacts=backend.get_contacts()
        if contacts is None:raise RuntimeError('Pinned MJWarp backend did not allocate contact buffer')
        feet=set();peak_contacts=0
        rewards=[];qs=S[4].clone()
        for t in range(args.steps):
            f=torch.minimum(env.f+1,motion.clip_end[motion.clip_id[env.f]])
            act=(motion.q[f]-env.default_q)/env.alpha
            out=env.step(act);rewards.append(float(out['r'].mean()/MAX_STEP_REWARD))
            assert all(torch.isfinite(x).all() for x in env.state())
            assert torch.isfinite(out['r']).all()
            # Pinned manager uses this exact signature for contact sensors. Export here
            # even when no sensor is enabled, so a smoke cannot pass with disabled contacts.
            backend._solver.update_contacts(contacts,backend.get_state_0())
            count=int(wp.to_torch(contacts.rigid_contact_count).sum());peak_contacts=max(peak_contacts,count)
            s0=wp.to_torch(contacts.rigid_contact_shape0)[:count].cpu().tolist()
            s1=wp.to_torch(contacts.rigid_contact_shape1)[:count].cpu().tolist()
            for left,right in zip(s0,s1):
                if left<0 or right<0:continue
                b0,b1=int(shape_body[left]),int(shape_body[right])
                if b0<0 or b1<0:
                    b=max(b0,b1)
                    if b>=0:
                        name=model.body_label[b].split('/')[-1]
                        if name in ('leg_l6_link','leg_r6_link'):feet.add(name)
            if t in (0,args.steps-1): print('step',t,'normalized reward',rewards[-1],flush=True)
        report['step_fk']=alignment();report['joint_motion_max_rad']=float((env.state()[4]-qs).abs().max())
        assert report['joint_motion_max_rad']>1e-5,'Joint targets did not affect state'
        report['mean_normalized_reward']=sum(rewards)/len(rewards)
        report['failed_envs']=int(env.failed_eval.sum());report['cuda_peak_MiB']=torch.cuda.max_memory_allocated()/2**20
        report['peak_contacts']=peak_contacts;report['feet_with_ground_contact']=sorted(feet)
        assert peak_contacts>0 and len(feet)==2,'Both feet must make actual ground contact during smoke'
        # Drive values actually applied by the articulation, in simulator order.
        report['stiffness']=env.robot.data.joint_stiffness.torch[0].tolist()
        report['armature']=env.robot.data.joint_armature.torch[0].tolist()
        assert min(report['stiffness'])>0 and min(report['armature'])>0
        report['root_height_after_steps_m']=env.state()[0][:,2].tolist()
        # Gravity and foot contact forces must be checked on the backend. These runtime
        # attributes are exported by the pinned IsaacLab/Newton version.
        print('backend',type(backend).__name__,flush=True)
    out=Path(args.out or f'results/s54/newton_{args.num_envs}env.json');out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
