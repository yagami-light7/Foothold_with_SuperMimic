#!/usr/bin/env python3
"""One-step Newton replay of saved reset failures. No training or geometry edits."""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ.pop('DISPLAY',None)
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import warp as wp
from isaaclab.app import launch_simulation
from supermimic.kinematics import S54Kinematics
from supermimic.goals import GoalSource
from supermimic.mdp import unflatten_state,resolve_motion_reward
from supermimic.rotations import mat_to_quat
from supermimic.s54 import default_joint_pos_list,COM_OFFSET
from supermimic.stair_training import library
from supermimic.stair_runtime import StairTrackingMDP,make_stair_sim,make_stair_env


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',default='logs/s54_stair_flat6063_smoke_v3/failure_physics_transition.pt')
    parser.add_argument('--out',required=True)
    parser.add_argument('--conditioned',action='store_true')
    parser.add_argument('--plane',action='store_true')
    parser.add_argument('--mesh-flat',action='store_true')
    a=parser.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    device='cuda:0';torch.backends.cuda.matmul.allow_tf32=False
    cfg=json.loads(Path('experiments/s54_stair_flat6063_8h/design.json').read_text())
    capture=torch.load(a.capture,map_location=device,weights_only=False)
    kin=S54Kinematics(device);m=library(cfg,kin,device)
    mdp=StairTrackingMDP(kin,m,torch.tensor(default_joint_pos_list(),device=device),
        resolve_motion_reward('experiments/s54_stair_flat6063_8h/reward.json'),observation_mode='root_xy_heading_v2')
    sim=make_stair_sim(device,a);report=dict(capture=a.capture,optimization_updates=0,rows=[])
    with torch.no_grad(),launch_simulation(sim):
        if a.plane:
            from supermimic.env import TrackingSim
            env=TrackingSim(sim,len(capture['f']),m,mdp,eval_mode=True,goals=GoalSource(m,kin))
        else:env=make_stair_env(sim,len(capture['f']),m,mdp,cfg['terrain'],mixed_flat=True,
            goals=GoalSource(m,kin),flat_collider='mesh' if a.mesh_flat else 'box')
        model=env.physics.get_model();bodies=wp.to_torch(model.shape_body)
        static=(bodies<0).nonzero().flatten()
        report['static_shapes']={name:wp.to_torch(getattr(model,name))[static].tolist()
            for name in ['shape_type','shape_scale','shape_transform','shape_gap','shape_margin'] if hasattr(model,name)}
        ids=torch.arange(env.N,device=device);S=unflatten_state(capture['s'])
        com=env.robot.data.body_com_pose_b.torch[:,env.torso_id,:3].clone()
        env.xi.copy_(capture['xi']);env._apply_friction()
        env.robot.set_coms_index(coms=(com+env.xi[:,COM_OFFSET]).unsqueeze(1),body_ids=[env.torso_id])
        from supermimic.stair_reset import TerrainResetValidator
        validator=TerrainResetValidator(kin,cfg['terrain'])
        cases=[('saved_reset',0.),('diagnostic_reset_lift_8cm',.08)]
        if a.conditioned:cases.append(('conditioned_original_noise',0.))
        for label,lift in cases:
            env.reset(ids,capture['f'])
            current=(validator.condition(S,m.ref_state(capture['f'])) if label=='conditioned_original_noise' else S)
            p=current[0].clone();p[:,2]+=lift;q=mat_to_quat(current[1])
            env.robot.write_root_link_pose_to_sim_index(root_pose=torch.cat((p,q[:,1:],q[:,:1]),-1),env_ids=ids)
            env.robot.write_root_link_velocity_to_sim_index(root_velocity=torch.cat((current[2],current[3]),-1),env_ids=ids)
            env.robot.write_joint_position_to_sim_index(position=current[4][:,env.sim_to_canonical],env_ids=ids)
            env.robot.write_joint_velocity_to_sim_index(velocity=current[5][:,env.sim_to_canonical],env_ids=ids)
            env.refresh();env.a_prev.copy_(capture['a_prev'])
            transition=env.step(capture['a']);s1=unflatten_state(transition['s1'])
            model=env.physics.get_model();contacts=env.physics.get_contacts()
            n=int(wp.to_torch(contacts.rigid_contact_count).sum())
            sh0=wp.to_torch(contacts.rigid_contact_shape0)[:n].long()
            sh1=wp.to_torch(contacts.rigid_contact_shape1)[:n].long()
            worlds=wp.to_torch(model.shape_world).long()
            cross=((worlds[sh0]>=0)&(worlds[sh1]>=0)&(worlds[sh0]!=worlds[sh1])).sum()
            row=dict(case=label,root_speed=s1[2].norm(dim=-1).tolist(),root_angvel=s1[3].norm(dim=-1).tolist(),
                joint_speed=s1[5].abs().amax(-1).tolist(),normalized_reward=(transition['r']/.13).tolist(),
                contacts=n,cross_world_contacts=int(cross),state_finite=bool(torch.isfinite(transition['s1']).all()))
            report['rows'].append(row);print(json.dumps(row),flush=True)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
