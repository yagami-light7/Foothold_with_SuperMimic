#!/usr/bin/env python3
"""T1.5 independent Newton kinematics / PD / frozen SuperMimic rollouts."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.pop('DISPLAY', None)  # Select EGL before IsaacLab/pyglet imports.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import warp as wp
from PIL import Image, ImageDraw

from supermimic.kinematics import S54Kinematics
from supermimic.m34 import motion_library, sha256
from supermimic.stair_data import load_stairs
from supermimic.stair_runtime import StairTrackingMDP, make_stair_env, make_stair_sim
from supermimic.mdp import MDP, RewardCfg, flatten_state, unflatten_state, normalize_xi
from supermimic.goals import GoalSpec, GoalSource
from supermimic.rotations import mat_to_quat, quat_to_mat, so3_log, so3_exp
from supermimic.s54 import JOINT_NAMES, FOOT_BODY_NAMES, default_joint_pos_list, assert_checkpoint
from supermimic.control import bound_action


class Video:
    def __init__(self, path, width=960, height=640):
        self.count = 0
        self.path = path
        self.log = path.with_suffix('.ffmpeg.log').open('w')
        self.process = subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-n',
            '-f','rawvideo','-pixel_format','rgb24','-video_size',f'{width}x{height}',
            '-framerate','50','-i','pipe:0','-an','-c:v','libx264','-preset','fast',
            '-crf','20','-pix_fmt','yuv420p',str(path)], stdin=subprocess.PIPE, stderr=self.log)

    def add(self, rgb, caption):
        image = Image.fromarray(rgb[..., :3]); draw = ImageDraw.Draw(image)
        draw.rectangle((0,0,image.width,44),fill='black')
        draw.text((12,10),caption,fill='white')
        self.process.stdin.write(np.asarray(image).tobytes()); self.count += 1

    def close(self):
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError(f'Video encoding failed: {self.path}')
        self.log.close()
        actual = int(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-count_frames','-show_entries','stream=nb_read_frames','-of','default=nw=1:nk=1',str(self.path)]))
        if actual != self.count:
            raise RuntimeError(f'Missing video frames: expected {self.count}, decoded {actual}')


@torch.no_grad()
def fk_gate(env, kin, motion):
    """Same all-body/velocity + random-pose gate as tools/m34_rollout.py."""
    torch.manual_seed(54)
    ids = torch.tensor([0],device=env.device)
    errors = torch.zeros(4,device=env.device)
    samples = [0, int(motion.clip_len[0])//2, int(motion.clip_len[0])-1]
    body_ids = [env.robot.body_names.index(n) for n in kin.body_names]
    for index in range(len(samples)+16):
        env.reset(ids, motion.clip_start[:1]+samples[index % len(samples)])
        if index >= len(samples):
            S = list(env.state())
            S[0] = torch.randn_like(S[0])*.2+S[0].new_tensor([0,0,1.2])
            S[1] = so3_exp(torch.randn(1,3,device=env.device))
            S[2], S[3] = torch.randn_like(S[2]), torch.randn_like(S[3])
            S[4] = kin.q_lo+(kin.q_hi-kin.q_lo)*torch.rand_like(S[4]); S[5] = torch.randn_like(S[5])
            quaternion = mat_to_quat(S[1])
            env.robot.write_root_link_pose_to_sim_index(root_pose=torch.cat((S[0],quaternion[:,1:],quaternion[:,:1]),-1),env_ids=ids)
            env.robot.write_root_link_velocity_to_sim_index(root_velocity=torch.cat(S[2:4],-1),env_ids=ids)
            env.robot.write_joint_position_to_sim_index(position=S[4][:,env.sim_to_canonical],env_ids=ids)
            env.robot.write_joint_velocity_to_sim_index(velocity=S[5][:,env.sim_to_canonical],env_ids=ids)
            env.refresh()
        actual = kin.forward(*env.state(),all_bodies=True)
        poses=env.robot.data.body_link_pose_w.torch[:,body_ids]
        velocity=env.robot.data.body_link_vel_w.torch[:,body_ids]
        expected=(poses[...,:3]-env.scene.env_origins[:,None],
            quat_to_mat(torch.cat((poses[...,6:7],poses[...,3:6]),-1)),velocity[...,:3],velocity[...,3:])
        errors=torch.maximum(errors,torch.stack([(a-b).abs().max() for a,b in zip(actual,expected)]))
    if not (errors < errors.new_tensor([1e-4,1e-3,1e-3,1e-3])).all():
        raise RuntimeError(f'Newton all-body FK failed: {errors.tolist()}')
    return dict(status='PASS',all_bodies=30,reference_poses=3,random_poses=16,max_errors=errors.tolist())


@torch.no_grad()
def wm_audit(wm, states, actions, xi):
    if wm is None:
        return None
    report={}
    states=torch.stack(states); actions=torch.stack(actions)
    for horizon in [1,4,8]:
        if len(actions)<horizon:
            report[str(horizon)]=dict(samples=0);continue
        starts=torch.arange(0,len(actions)-horizon+1,4,device=states.device)
        predicted=unflatten_state(states[starts])
        for j in range(horizon):
            predicted=wm.step(predicted,actions[starts+j],normalize_xi(xi.expand(len(starts),-1)))
        error=wm.state_error(predicted,unflatten_state(states[starts+horizon]))
        report[str(horizon)]=dict(samples=len(starts),root_position_rmse_m=float(error[:,:3].square().sum(-1).mean().sqrt()),
            root_orientation_rmse_rad=float(error[:,3:6].square().sum(-1).mean().sqrt()),
            joint_position_rmse_rad=float(error[:,12:41].square().mean().sqrt()),
            normalized_state70_rmse=float((error/wm.out_norm.std()).square().mean().sqrt()))
    return report


@torch.no_grad()
def main(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    report=dict(status='initializing',mode=args.mode,reference=args.reference,seeds=args.seeds,
        contact_pipeline=args.contacts,physics='Newton MJWarp',control_dt=.02,physics_substeps=4,
        reference_anchor='fixed exported world; no drift realignment',rows=[])
    (out/'status.json').write_text(json.dumps(report,indent=2)+'\n')
    video=None
    try:
        from isaaclab.app import launch_simulation
        from supermimic.env import TrackingSim,make_sim
        from supermimic.networks import Policy,WorldModel
        from supermimic.standing_diagnostics import NewtonGroundContacts
        from supermimic.robot import PARAMETERS
        torch.backends.cuda.matmul.allow_tf32=False
        device='cuda:0';kin=S54Kinematics(device)
        motion=(load_stairs(args.bundle,kin,device,ids=['stair_up_4'],npz_path=args.stair_npz) if args.reference=='stair' else
            motion_library(args.flat_dataset,kin,device,'train',ids=[args.flat_id]))
        checkpoint=torch.load(args.checkpoint,map_location=device,weights_only=False) if args.checkpoint else None
        config=checkpoint['args'] if checkpoint else {}
        if checkpoint:assert_checkpoint(checkpoint)
        reward=RewardCfg(**config.get('reward_config',{}))
        # Fixed evaluation convention across source/fine-tuned policies, with a
        # separate legacy reward to contextualize historical flat returns.
        from dataclasses import replace
        reward=replace(reward,root_position_mode='xyz')
        spec=GoalSpec('motion',config.get('goal_frames',10),config.get('goal_dt',.02))
        if checkpoint:spec.assert_checkpoint(checkpoint)
        goals=GoalSource(motion,kin,spec)
        mdp=StairTrackingMDP(kin,motion,torch.tensor(default_joint_pos_list(),device=device),reward,
            observation_mode=config.get('motion_obs_mode','root_xy_heading_v2'))
        legacy=MDP(kin,motion,mdp.default_q,RewardCfg(**config.get('reward_config',{})),observation_mode=mdp.observation_mode)
        policy=wm=None;history_length=config.get('hist',10)
        if checkpoint:
            import supermimic.networks as networks
            networks.AMP_DTYPE=torch.bfloat16 if config.get('amp')=='bf16' else None
            policy=Policy(mdp.obs_dim,spec.dim,hist=history_length).to(device)
            policy.load_state_dict(checkpoint['pi']);policy.eval().requires_grad_(False)
            wm=WorldModel().to(device);wm.load_state_dict(checkpoint['wm']);wm.eval().requires_grad_(False)
            report.update(checkpoint_sha256=sha256(args.checkpoint),checkpoint_iteration=checkpoint['iteration'])
        report.update(reward_config=asdict(reward),task_contract=mdp.contract(),max_step_reward=mdp.max_step_reward,
            goal_dim=spec.dim,obs_dim=mdp.obs_dim,reference_sha256=motion.rows[0]['motion_content_sha256'])
        cfg=make_stair_sim(device,args,contact_pipeline=args.contacts,render=not args.no_video)
        report['collision_decimation']=cfg.physics.collision_decimation
        if not args.no_video:
            from isaaclab_visualizers.newton import NewtonGLVisualizerCfg
            cfg.visualizer_cfgs=[NewtonGLVisualizerCfg(headless=True,window_width=960,window_height=640,streaming_view=False)]
        with launch_simulation(cfg):
            if args.reference=='stair':
                env=make_stair_env(cfg,1,motion,mdp,args.terrain,goals=goals)
            else:
                env=TrackingSim(cfg,1,motion,mdp,eval_mode=True,reset_noise=False,goals=goals)
            env.sim.set_camera_view(eye=(2.4,-4.2,2.7),target=(-.9,0.,.85))
            report['newton_fk']=fk_gate(env,kin,motion)
            f0=int(motion.clip_start[0]);frames=int(motion.clip_len[0]);feet=mdp.foot_indices
            all_f=torch.arange(f0,f0+frames,device=device)
            full_goal=motion.goal(all_f,spec.frames).view(frames,spec.frames,-1)
            report['goal_root_z_delta_max_m']=float(full_goal[...,2].abs().max())
            if policy is not None:
                g=full_goal.reshape(frames,-1);no_z=full_goal.clone();no_z[...,2]=0.
                report['encoder_z_sensitivity_latent_l2_mean']=float((policy.encode(g)-policy.encode(no_z.reshape_as(g))).norm(dim=-1).mean())
            if not args.no_video:
                viewer=env.sim._visualizers[0]._viewer
                for side,index in enumerate(feet):
                    path=motion.body_p[f0:f0+frames,index].cpu().numpy()
                    viewer.log_lines(f'reference_foot_{side}',wp.array(path[:-1],dtype=wp.vec3,device=device),
                        wp.array(path[1:],dtype=wp.vec3,device=device),(.1,.95,.2) if side==0 else (.95,.35,.05))
            body_ids=[env.robot.body_names.index(name) for name in kin.body_names]
            for seed in args.seeds:
                torch.manual_seed(seed)
                env.done_eval.zero_();env.failed_eval.zero_();env.reset_noise=seed!=0
                env.reset(torch.tensor([0],device=device),motion.clip_start[:1])
                history=None;states=[];actions=[];metrics=[];torques=[];contacts_trace=[];frame_rows=[]
                saved={k:[] for k in ['root_p','root_quat','q','f','alive']}
                status='reference_complete' if args.mode=='kinematic' else 'running'
                if not args.no_video:
                    video=Video(out/f'{args.mode}_{args.reference}_seed{seed}.mp4')
                contact_sensor=NewtonGroundContacts(env) if args.mode!='kinematic' else None
                for tick in range(frames):
                    if args.mode=='kinematic':
                        env.reset(torch.tensor([0],device=device),motion.clip_start[:1]+tick)
                    S=env.state();f=env.f.clone();states.append(flatten_state(S)[0].clone())
                    for key,value in zip(saved,(S[0],mat_to_quat(S[1]),S[4],f,~env.done_eval)):
                        saved[key].append(value.cpu().numpy().copy())
                    if video is not None:
                        env.sim.render();rgb=env.sim._visualizers[0].render_rgb_array()
                        mask=motion.contacts[f][0].tolist() if hasattr(motion,'contacts') else ['proxy','proxy']
                        video.add(rgb,f'{args.mode.upper()} | {args.reference} | t={tick*.02:.2f}s | L/R ref contact={mask} | seed={seed}')
                    if args.mode=='kinematic':
                        pose=env.robot.data.body_link_pose_w.torch[:,body_ids]
                        expected=kin.forward(*motion.ref_state(f),all_bodies=True)
                        position=float((pose[...,:3]-expected[0]).abs().max())
                        orientation=float(so3_log(expected[1].transpose(-1,-2)@quat_to_mat(torch.cat((pose[...,6:7],pose[...,3:6]),-1))).norm(dim=-1).max())
                        frame_rows.append(dict(frame=tick,time_s=tick*.02,newton_body_position_max_error_m=position,newton_body_orientation_max_error_rad=orientation))
                        continue
                    if tick==frames-1 or status!='running':break
                    obs=goals.obs(mdp,S,env.a_prev,env.xi,f,env.command)
                    history=obs[:,None].repeat(1,history_length,1) if history is None else torch.cat((history[:,1:],obs[:,None]),1)
                    a=((motion.q[f+1]-env.default_q)/env.alpha if args.mode=='pd' else bound_action(policy(history,goals.goal(S,f,env.ep_step,env.command))))
                    actions.append(a[0].clone());previous=env.a_prev.clone()
                    transition=env.step(a);S1=env.state();f1=env.f
                    bp,bR,bv,_=kin.forward(*S1)
                    counts,forces,other=contact_sensor.read()
                    contacts_trace.append(dict(frame=tick+1,foot_contact_count=counts[0].tolist(),foot_force_w_N=forces[0].tolist(),nonfoot_contact_count=int(other[0])))
                    contact=motion.contacts[f1][0] if hasattr(motion,'contacts') else (motion.body_v[f1][:,feet,:2].norm(dim=-1)[0]<.05)
                    e_root=(S1[0]-motion.root_p[f1])[0];yaw=torch.atan2(S1[1][:,1,0],S1[1][:,0,0])-torch.atan2(motion.root_R[f1][:,1,0],motion.root_R[f1][:,0,0])
                    e_yaw=torch.atan2(torch.sin(yaw),torch.cos(yaw))[0]
                    foot_error=(bp[:,feet]-motion.body_p[f1][:,feet]).square().sum(-1)[0]
                    foot_angle=so3_log(motion.body_R[f1][:,feet].transpose(-1,-2)@bR[:,feet]).square().sum(-1)[0]
                    slip=(bv[:,feet,:2].norm(dim=-1)[0]*contact).sum()/contact.sum().clamp_min(1)
                    metrics.append(torch.cat((e_root.square(),e_yaw.square()[None],foot_error,foot_angle,
                        transition['r'],legacy.reward(S1,f1,a,previous),slip[None],(a-previous).square().mean()[None],S1[5].abs().max()[None])).cpu().numpy())
                    for name in ['joint_applied_torque','applied_torque','joint_effort']:
                        if hasattr(env.robot.data,name):
                            value=getattr(env.robot.data,name);value=value.torch if hasattr(value,'torch') else value
                            torques.append(value[0,env.canonical_to_sim].cpu().numpy().copy());break
                    if bool(transition['term'][0]):status='tracking_height' if abs(float(e_root[2]))>mdp.cfg.term_height else 'tracking_tilt'
                    elif bool(transition['timeout'][0]):status='reference_complete'
                    if not all(torch.isfinite(x).all() for x in S1):raise FloatingPointError('Nonfinite Newton state')
                if video is not None:video.close();video=None
                if args.mode=='kinematic' and (len(states)!=frames or max(r['newton_body_position_max_error_m'] for r in frame_rows)>1e-4):
                    raise RuntimeError('Kinematic full-frame/all-body FK failed')
                np.savez_compressed(out/f'{args.mode}_{args.reference}_seed{seed}.npz',control_dt=.02,joint_names=np.array(JOINT_NAMES),
                    provenance='KINEMATIC state writes; not physics evidence' if args.mode=='kinematic' else 'Newton physics; state writes at reset only',
                    state=torch.stack(states).cpu().numpy(),actions=torch.stack(actions).cpu().numpy() if actions else np.empty((0,29)),
                    **{k:np.stack(v) for k,v in saved.items()})
                row=dict(seed=seed,status=status,frames_saved=len(states),reference_frames=frames,
                    survived=status=='reference_complete' and args.mode!='kinematic',
                    kinematic_only=args.mode=='kinematic',episode_steps=len(actions),episode_s=len(actions)*.02,
                    completion_rate=len(actions)/(frames-1) if actions else None,
                    failure_frame=len(actions) if status.startswith('tracking_') else None)
                if metrics:
                    x=np.stack(metrics)
                    row.update(normalized_return=float(x[:,8].sum()/((frames-1)*mdp.max_step_reward)),
                        raw_reward_sum=float(x[:,8].sum()),legacy_raw_reward_sum=float(x[:,9].sum()),
                        legacy_normalized_return=float(x[:,9].sum()/((frames-1)*legacy.max_step_reward)),
                        root_xyz_rmse_m=np.sqrt(x[:,:3].mean(0)).tolist(),yaw_rmse_rad=float(np.sqrt(x[:,3].mean())),
                        left_right_foot_position_rmse_m=np.sqrt(x[:,4:6].mean(0)).tolist(),
                        left_right_foot_orientation_rmse_rad=np.sqrt(x[:,6:8].mean(0)).tolist(),
                        stance_slip_link_origin_xy_speed_m_s=float(x[:,10].mean()),action_delta_rms=float(np.sqrt(x[:,11].mean())),
                        peak_joint_velocity_rad_s=float(x[:,12].max()),wm_prediction=wm_audit(wm,states,actions,env.xi))
                    if torques:
                        effort=np.array([PARAMETERS[n]['effort'] for n in JOINT_NAMES]);tau=np.stack(torques)
                        row.update(actuator_reported_torque_peak_Nm=float(abs(tau).max()),
                            actuator_reported_torque_saturation_rate=float((abs(tau)>=.99*effort).mean()),
                            torque_source=f'actuator data.{name}; implicit estimate, not claimed solver-measured')
                        np.savez_compressed(out/f'torques_seed{seed}.npz',torque_Nm=tau,limits_Nm=effort)
                    np.save(out/f'metrics_seed{seed}.npy',x)
                for label,rows in [('kinematic',frame_rows),('contacts',contacts_trace)]:
                    if rows:(out/f'{label}_seed{seed}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                report['rows'].append(row);print(json.dumps(row),flush=True)
                (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        report['status']='completed'
    except BaseException as exc:
        report.update(status='FAILED',error=repr(exc));raise
    finally:
        if video is not None:video.close()
        (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        (out/'status.json').write_text(json.dumps(dict(status=report['status'],error=report.get('error')),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['kinematic','pd','policy'],required=True)
    parser.add_argument('--reference',choices=['stair','flat'],default='stair')
    parser.add_argument('--bundle',default='/mnt/sage_data/workspace/Foothold/PMT/TCRS/outputs/s54_stair_t14/newton_import')
    parser.add_argument('--stair-npz',help='Relocated four-step NPZ; validated against the original manifest')
    parser.add_argument('--terrain',default='assets/terrains/stairs_straight_4_v1')
    parser.add_argument('--flat-dataset',default='data/s54_flat_dataset/dataset_export')
    parser.add_argument('--flat-id',default='straight_flat_18')
    parser.add_argument('--checkpoint')
    parser.add_argument('--contacts',choices=['newton','mujoco'],default='newton')
    parser.add_argument('--seeds',nargs='+',type=int,default=[0])
    parser.add_argument('--no-video',action='store_true')
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    if args.mode=='policy' and not args.checkpoint:parser.error('Frozen policy evaluation requires checkpoint')
    main(args)
