"""Newton/MJWarp foot-command preview and replay smoke; never runs an optimizer.

Blue=left, orange=right; dots=reference path, boxes=landing ankle poses,
large spheres=current reference, white tips=reference forward axes. Physics
uses existing Motion PD targets, while a fresh Policy verifies foot Goal input.
This is an interface preview, not evidence of learned foot tracking.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

# Pyglet resolves its window/display classes on first import. Select EGL before
# Isaac Lab or Newton imports them, even when the host has a DISPLAY variable.
if '--headless' in sys.argv:
    os.environ.setdefault('PYGLET_HEADLESS', 'true')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import add_launcher_args, launch_simulation
from supermimic.goals import GoalSource, add_goal_args, resolve_goal_config

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--num_envs',type=int,default=1)
p.add_argument('--steps',type=int,default=100)
p.add_argument('--goal_frames',type=int,default=10)
p.add_argument('--motion_dir',default='data/retarget_data/reference_startup_fixed')
p.add_argument('--seed',type=int,default=54)
p.add_argument('--out',default='results/s54/foot_trajectory_newton.json')
p.add_argument('--video',help='Optional MP4 from newton_gl; requires --viz newton_gl')
add_goal_args(p);add_launcher_args(p)
if '--headless' not in p._option_string_actions:
    p.add_argument('--headless',action='store_true',help='Render Newton GL without a desktop window')
p.set_defaults(goal_mode='foot_trajectory')
args=p.parse_args()
if not 1<=args.num_envs<=4 or not 20<=args.steps<=200:
    p.error('Use 1–4 environments and 20–200 steps; this tool does not train')
if args.goal_mode!='foot_trajectory':p.error('Use tools/test_env_s54.py for Motion smoke')
spec,cfg=resolve_goal_config(args)

import numpy as np
import torch
from supermimic.buffer import ReplayBuffer
from supermimic.env import TrackingSim,make_sim
from supermimic.foot_trajectory import query
from supermimic.kinematics import S54Kinematics
from supermimic.mdp import MDP,OBS_DIM,unflatten_state
from supermimic.motion import MotionLib,CONTROL_DT
from supermimic.networks import Policy
from supermimic.rotations import quat_to_mat
from supermimic.s54 import default_joint_pos_list


def markers():
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers,VisualizationMarkersCfg
    def material(color):return sim_utils.PreviewSurfaceCfg(diffuse_color=color)
    colors=[(.1,.5,1.),(1.,.4,.08)]
    def group(name,prototypes):
        return VisualizationMarkers(VisualizationMarkersCfg(prim_path='/Visuals/FootTrajectory/'+name,markers=prototypes))
    path=group('path',{str(i):sim_utils.SphereCfg(radius=.008,visual_material=material(c)) for i,c in enumerate(colors)})
    landing=group('landing',{str(i):sim_utils.CuboidCfg(size=(.09,.05,.008),visual_material=material(c)) for i,c in enumerate(colors)})
    current=group('current',{str(i):sim_utils.SphereCfg(radius=.023,visual_material=material(c)) for i,c in enumerate(colors)})
    tips=group('heading',{'tip':sim_utils.SphereCfg(radius=.009,visual_material=material((1.,1.,1.)))})
    return path,landing,current,tips


def main():
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    dev='cuda:0'
    kin=S54Kinematics(dev);motion=MotionLib.from_dir(args.motion_dir,kin,dev)
    mdp=MDP(kin,motion,torch.tensor(default_joint_pos_list(),device=dev))
    goals=GoalSource(motion,kin,spec,cfg)
    sim_cfg=make_sim(dev,args)
    from isaaclab.visualizers import VisualizerCfg
    sim_cfg.default_visualizer_cfg=VisualizerCfg(eye=(2.5,-2.5,1.8),lookat=(.4,0.,.5))
    if 'newton_gl' in (args.visualizer or ''):
        from isaaclab_visualizers.newton import NewtonGLVisualizerCfg
        sim_cfg.visualizer_cfgs=[NewtonGLVisualizerCfg(headless=args.headless,window_width=960,window_height=720,
                                                     eye=(2.5,-2.5,1.8),lookat=(.4,0.,.5))]
    pi=Policy(OBS_DIM,spec.dim,hist=2).to(dev).eval()
    report=dict(backend='Newton/MJWarp',num_envs=args.num_envs,steps=args.steps,
                goal_contract=spec.contract(),obs_dim=OBS_DIM,action_dim=29,
                control='existing Motion PD targets; untrained Policy forward only',training_started=False)
    encoder=None
    try:
        with launch_simulation(sim_cfg,args):
            # A short timeout deliberately exercises reset -> old replay command recovery.
            env=TrackingSim(sim_cfg,args.num_envs,motion,mdp,max_episode_steps=80,reset_noise=False,goals=goals)
            ids=torch.arange(args.num_envs,device=dev)
            env.reset(ids,motion.clip_start[ids%len(motion.names)]+100)
            replay=ReplayBuffer(48,args.num_envs,dev,foot_commands=True)
            active_visualizers=list(env.sim.visualizers)
            if args.video and not active_visualizers:
                raise RuntimeError('--video requires a successfully initialized visualizer')
            marker_groups=markers() if active_visualizers else None
            history=None;resets=0;max_error=0.;video_frames=0
            trace={k:[] for k in ('reference_position','reference_quaternion','contact','command_id','command_step')}
            with torch.no_grad():
                for step in range(args.steps):
                    S=env.state();f=env.f.clone();elapsed=env.ep_step.clone();c=env.command.clone()
                    g=goals.goal(S,f,elapsed,c)
                    o=goals.obs(mdp,S,env.a_prev,env.xi,f,c)
                    fresh=elapsed==0
                    history=(o[:,None].repeat(1,2,1) if history is None else
                             torch.where(fresh[:,None,None],o[:,None].expand(-1,2,-1),torch.cat((history[:,1:],o[:,None]),1)))
                    action=pi(history,g)
                    assert g.shape==(args.num_envs,spec.dim)
                    assert action.shape==(args.num_envs,29) and torch.isfinite(action).all()
                    ref=query(c,elapsed[:,None].float()*CONTROL_DT)
                    for k,v in dict(reference_position=ref.position[:,0],reference_quaternion=ref.quaternion[:,0],
                                    contact=ref.contact[:,0],command_id=env.command_id,command_step=elapsed).items():
                        trace[k].append(v.detach().cpu().clone())
                    if marker_groups:
                        center=S[0][0].cpu().tolist()
                        env.sim.set_camera_view((center[0]+2.5,center[1]-2.5,1.8),(center[0]+.4,center[1],.5))
                        path,landing,current,tips=marker_groups
                        # Dense environment-major arrays preserve viewer world selection.
                        timeline=elapsed[:,None].float()*CONTROL_DT+torch.linspace(0,2,160,device=dev)[None]
                        preview=query(c,timeline)
                        origin=env.scene.env_origins
                        path.visualize(translations=(preview.position+origin[:,None,None]).reshape(-1,3),
                                       marker_indices=[0,1]*(160*args.num_envs))
                        event=torch.arange(cfg.num_steps,device=dev)[None].expand(args.num_envs,-1)
                        upcoming=(event<c.lengths[:,None]) & (c.touchdown>=timeline[:,:1])
                        selected=torch.where(upcoming,event,cfg.num_steps).sort(-1).values[:,:min(4,cfg.num_steps)]
                        selected=selected.clamp(max=cfg.num_steps-1)
                        landing_pos=torch.cat((c.initial_pos,c.target_pos.gather(1,selected[...,None].expand(-1,-1,3))),1)
                        landing_q=torch.cat((c.initial_quat,c.target_quat.gather(1,selected[...,None].expand(-1,-1,4))),1).reshape(-1,4)
                        landing_indices=torch.cat((ids.new_tensor([0,1])[None].expand(args.num_envs,-1),c.foot.gather(1,selected)),1)
                        landing.visualize(translations=(landing_pos+origin[:,None]).reshape(-1,3),
                                          orientations=landing_q[:,[1,2,3,0]],marker_indices=landing_indices.flatten())
                        current.visualize(translations=(ref.position[:,0]+origin[:,None]).reshape(-1,3),
                                          marker_indices=[0,1]*args.num_envs)
                        forward=quat_to_mat(ref.quaternion[:,0])[..., :,0]
                        tips.visualize(translations=(ref.position[:,0]+origin[:,None]+.055*forward).reshape(-1,3))
                        env.sim.render()
                        if args.video:
                            rgb=active_visualizers[0].render_rgb_array()
                            if rgb is None:raise RuntimeError('Newton viewer did not return RGB')
                            rgb=np.ascontiguousarray(rgb[...,:3],dtype=np.uint8)
                            if encoder is None:
                                from PIL import Image
                                video=Path(args.video);video.parent.mkdir(parents=True,exist_ok=True)
                                Image.fromarray(rgb).save(video.with_suffix('.png'))
                                h,w=rgb.shape[:2]
                                encoder=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
                                    '-s',f'{w}x{h}','-r','50','-i','-','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2',
                                    '-c:v','libx264','-pix_fmt','yuv420p',str(video)],stdin=subprocess.PIPE)
                            encoder.stdin.write(rgb.tobytes());video_frames+=1
                    # No learned controller/IK: reuse the existing simulator smoke control.
                    pd_action=(motion.q[f+1]-env.default_q)/env.alpha
                    out=env.step(pd_action);replay.add(out)
                    recovered=replay.commands(out['command_id'])
                    replay_g=goals.goal(unflatten_state(out['s']),out['f'],out['ep_step'],recovered)
                    max_error=max(max_error,float((g-replay_g).abs().max()))
                    torch.testing.assert_close(g,replay_g,atol=1e-6,rtol=1e-6)
                    # Next imagined query advances the saved execution step, not Motion f.
                    S1=unflatten_state(out['s1'])
                    future=goals.goal(S1,f+1,elapsed+1,c)
                    replay_future=goals.goal(S1,out['f']+1,out['ep_step']+1,recovered)
                    torch.testing.assert_close(future,replay_future,atol=1e-6,rtol=1e-6)
                    assert all(torch.isfinite(x).all() for x in env.state())
                    resets+=int(out['done'].sum())
                    if (step+1)%20==0:
                        print(f'Newton foot smoke {step+1}/{args.steps}: resets={resets}, replay_goal_error={max_error:.3g}',flush=True)
            report.update(status='PASS',reset_transitions=resets,replay_goal_max_error=max_error,
                          live_replay_commands=len(replay.command_store),video_frames=video_frames,
                          visualizers=[type(v).__name__ for v in active_visualizers],
                          torch_peak_MiB=torch.cuda.max_memory_allocated()/2**20)
            assert resets>0,'Reset/replay path was not exercised'
            if marker_groups:
                assert all(group.count>0 for group in marker_groups)
                report['marker_counts']=[group.count for group in marker_groups]
            outpath=Path(args.out);outpath.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(outpath.with_suffix('.npz'),**{k:torch.stack(v).numpy() for k,v in trace.items()})
    finally:
        if encoder is not None:
            encoder.stdin.close()
            if encoder.wait()!=0:raise RuntimeError('ffmpeg failed')
    Path(args.out).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
