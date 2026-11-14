#!/usr/bin/env python3
"""Open a Newton GL window for the four-step reference. KINEMATICS ONLY, no training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=ROOT/'results/s54_stair_t15_20261110/data/newton_import')
    parser.add_argument('--terrain',type=Path,default=ROOT/'assets/terrains/stairs_straight_4_v1')
    parser.add_argument('--stair-npz',type=Path,default=ROOT/'data/s54_stair_dataset/stair_up_4.npz')
    parser.add_argument('--speed',type=float,default=1.,help='Playback speed; 1 = native 50 Hz, .5 = half speed')
    parser.add_argument('--loops',type=int,default=0,help='0 loops until the window closes')
    parser.add_argument('--start-frame',type=int,default=0)
    parser.add_argument('--check-only',action='store_true',help='CPU-only import/FK/geometry check, no viewer or GPU context')
    args=parser.parse_args()
    if not 0<args.speed<=10 or args.loops<0:parser.error('Require 0 < speed <= 10 and loops >= 0')
    import numpy as np
    import torch
    from supermimic.stair_data import load_stairs
    from supermimic.kinematics import S54Kinematics
    from supermimic.m34 import sha256
    mesh_hash=sha256(args.terrain/'terrain_mesh.npz')
    if mesh_hash!=sha256(args.bundle/'terrain/terrain_mesh.npz'):
        raise ValueError('Viewer terrain differs from the exported reference mesh')
    if args.check_only:
        motion=load_stairs(args.bundle,S54Kinematics('cpu'),'cpu',ids=['stair_up_4'],npz_path=args.stair_npz)
        print(json.dumps(dict(status='PASS',motion=motion.names[0],frames=int(motion.clip_len[0]),
            fps=50,joints=29,bodies=30,mesh_sha256=mesh_hash,validation=motion.validation,
            state_writes='kinematic playback only',optimizer_updates=0),indent=2))
        return
    if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
        parser.error('No DISPLAY available for a Newton GL window. Use tools/t15_stair_rollout.py --mode kinematic to export MP4 on a headless host.')
    # Keep DISPLAY and select windowed GL consistently before importing pyglet.
    from isaaclab.app import launch_simulation
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg
    from supermimic.stair_runtime import make_stair_sim,make_stair_env,StairTrackingMDP
    from supermimic.mdp import RewardCfg
    from supermimic.goals import GoalSource
    from supermimic.s54 import FOOT_BODY_NAMES,default_joint_pos_list
    import warp as wp
    from pyglet.window import key
    from pyglet.event import EVENT_HANDLED

    device='cuda:0'
    kin=S54Kinematics(device)
    motion=load_stairs(args.bundle,kin,device,ids=['stair_up_4'],npz_path=args.stair_npz)
    count=int(motion.clip_len[0]);start=int(motion.clip_start[0])
    if not 0<=args.start_frame<count:parser.error(f'start-frame must be in [0,{count-1}]')
    mdp=StairTrackingMDP(kin,motion,torch.tensor(default_joint_pos_list(),device=device),RewardCfg(),
                         observation_mode='root_xy_heading_v2')
    cfg=make_stair_sim(device,args,render=True)
    cfg.visualizer_cfgs=[NewtonGLVisualizerCfg(headless=False,window_width=1280,window_height=800,
                                             streaming_view=False)]
    with torch.no_grad(),launch_simulation(cfg):
        env=make_stair_env(cfg,1,motion,mdp,args.terrain,goals=GoalSource(motion,kin))
        env.sim.set_camera_view(eye=(2.4,-4.2,2.7),target=(-.9,0.,.85))
        viewer=env.sim._visualizers[0]._viewer
        playback=dict(frame=args.start_frame,playing=True,requested=None,loops=0)
        ids=torch.tensor([0],device=device)
        for side,name in enumerate(FOOT_BODY_NAMES):
            points=motion.body_p[:,kin.tracked_names.index(name)].cpu().numpy()
            color=(.1,.95,.2) if side==0 else (.95,.35,.05)
            viewer.log_lines(f'reference_{name}',wp.array(points[:-1],dtype=wp.vec3,device=device),
                wp.array(points[1:],dtype=wp.vec3,device=device),color)
        with np.load(args.stair_npz,allow_pickle=False) as payload:
            sequence=json.loads(payload['footstep_sequence_json'].item())
        for foot,color in [('left',(.1,.95,.2)),('right',(.95,.35,.05))]:
            positions=np.asarray([g['sole_center_world'] for g in sequence['goals'] if g['foot']==foot],dtype=np.float32)
            delta=np.array([.035,0.,0.],dtype=np.float32)
            starts=np.concatenate([positions-delta,positions-delta[[1,0,2]]])
            ends=np.concatenate([positions+delta,positions+delta[[1,0,2]]])
            viewer.log_lines(f'footsteps_{foot}',wp.array(starts,dtype=wp.vec3,device=device),
                wp.array(ends,dtype=wp.vec3,device=device),color)

        def on_key_press(symbol,modifiers):
            if symbol==key.SPACE:playback['playing']=not playback['playing']
            elif symbol==key.R:playback['requested']=0
            elif symbol==key.RIGHT:playback.update(playing=False,requested=min(count-1,playback['frame']+1))
            elif symbol==key.LEFT:playback.update(playing=False,requested=max(0,playback['frame']-1))
            else:return None
            return EVENT_HANDLED

        def panel(imgui):
            frame=playback['frame'];contact=motion.contacts[start+frame].tolist()
            imgui.text('KINEMATIC REFERENCE ONLY / NO TRAINING')
            imgui.text(f'Frame {frame}/{count-1} | t={frame*.02:.2f}s | speed={args.speed:g}x')
            imgui.text(f'Reference contact: L={int(contact[0])} R={int(contact[1])}')
            imgui.text('Space: pause | Left/Right: frame | R: restart')

        viewer.renderer.window.push_handlers(on_key_press=on_key_press)
        viewer.register_ui_callback(panel,position='side')
        print('KINEMATIC ONLY: 436 frames, 50 Hz, fixed world anchor; no physics steps or optimizer updates.',flush=True)
        print('Space pause/resume; Left/Right single frame; R restart; close window or Ctrl+C to exit.',flush=True)
        last=None;deadline=time.monotonic()+.02/args.speed
        try:
            while viewer.is_running():
                frame=playback['frame']
                if frame!=last:
                    # State writes are intentional and confined to this reference
                    # viewer. PD/policy rollouts use the independent physics tool.
                    env.reset(ids,motion.clip_start[:1]+frame)
                    contact=motion.contacts[start+frame].tolist()
                    viewer.renderer.window.set_caption(f'S54 | KINEMATIC ONLY | {frame}/{count-1} | t={frame*.02:.2f}s | L/R={contact}')
                    last=frame
                env.sim.render()
                if playback['requested'] is not None:
                    playback['frame']=playback['requested'];playback['requested']=None
                    deadline=time.monotonic()+.02/args.speed
                elif playback['playing'] and time.monotonic()>=deadline:
                    if frame==count-1:
                        playback['loops']+=1
                        if args.loops and playback['loops']>=args.loops:break
                        playback['frame']=0
                    else:playback['frame']+=1
                    deadline=max(deadline+.02/args.speed,time.monotonic())
                time.sleep(.001)
        except KeyboardInterrupt:
            pass


if __name__=='__main__':main()
