"""Execute a saved S54 foot policy in Newton, with a live viewer or a video/GIF.

Loads the saved robot/goal/task contracts and Actor normalization. No optimizer,
WM rollout, Motion reference controller, action noise or normalization updates.
Timeout means the requested duration elapsed; this viewer does not score success.
"""
from __future__ import annotations

# Support both direct execution and python -m from the repository layout.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_root = str(_ToolPath(__file__).resolve().parents[2])
if _tool_root not in _tool_sys.path:
    _tool_sys.path.insert(0, _tool_root)


import argparse
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

if '--headless' in sys.argv:
    os.environ.setdefault('PYGLET_HEADLESS', 'true')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from isaaclab.app import add_launcher_args, launch_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--loop', action='store_true', help='Repeat episodes until stopped')
    parser.add_argument('--seconds', type=float, default=10.)
    parser.add_argument('--seed', type=int, default=9541)
    parser.add_argument('--step_length', type=float, default=0., help='Metres per alternating step')
    parser.add_argument('--step_frequency', type=float, default=0., help='Alternating landings/s; 0 = standing')
    parser.add_argument('--yaw_rate', type=float, default=0., help='rad/s; positive turns left')
    parser.add_argument('--nominal_reset', action='store_true', help='Disable the saved reset perturbations')
    parser.add_argument('--no_realtime', action='store_true', help='Skip realtime pacing')
    parser.add_argument('--viser_port', type=int, default=8080)
    parser.add_argument('--viser_host', default='127.0.0.1', help='Use 0.0.0.0 for LAN access')
    parser.add_argument('--follow_camera', action='store_true', help='Follow root every step; otherwise Viser permits free orbit')
    parser.add_argument('--video', help='Optional GIF, or MP4 with ffmpeg; requires --viz newton_gl')
    parser.add_argument('--out', help='Write playback measurements as JSON')
    add_launcher_args(parser)
    if '--headless' not in parser._option_string_actions:
        parser.add_argument('--headless', action='store_true')
    parser.set_defaults(visualizer='newton_gl')
    args = parser.parse_args()
    if not 1 <= args.episodes <= 100 or not math.isfinite(args.seconds) or not 0 < args.seconds <= 60:
        parser.error('Use 1–100 episodes and a finite duration in (0,60] seconds')
    selected = args.visualizer
    selected = selected.split(',') if isinstance(selected, str) else (selected or [])
    if args.video and 'newton_gl' not in selected:
        parser.error('--video requires --viz newton_gl')
    gif = args.video and Path(args.video).suffix.lower() == '.gif'
    if args.video and not gif and shutil.which('ffmpeg') is None:
        parser.error('MP4 requires ffmpeg; use --video output.gif on this server')
    if 'viser' in selected:
        if not 1 <= args.viser_port <= 65535:
            parser.error('viser_port must be 1–65535')
        try:
            with socket.socket() as probe: probe.bind((args.viser_host, args.viser_port))
        except OSError as exc:
            parser.error(f'Viser bind address/port unavailable: {exc}')
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    import numpy as np
    import torch
    from supermimic import networks
    from supermimic.control import bound_action
    from supermimic.curriculum import Command, Curriculum, feasibility
    from supermimic.env import TrackingSim, make_sim
    from supermimic.foot_mdp import FootTrackingMDP, resolve_foot_task, assert_task_checkpoint
    from supermimic.foot_trajectory import AlternatingConfig
    from supermimic.goals import GoalSource, GoalSpec
    from supermimic.kinematics import S54Kinematics
    from supermimic.mdp import OBS_DIM
    from supermimic.motion import CONTROL_DT
    from supermimic.networks import Policy
    from supermimic.s54 import default_joint_pos_list, assert_checkpoint
    from supermimic.standing_diagnostics import evaluation_steps, stop_reasons

    # Check before creating a CUDA context; never start a second GPU simulation.
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    if any(line.strip() and line.strip() != str(os.getpid()) for line in active.splitlines()):
        parser.error('GPU compute process exists; run playback when the GPU is idle')
    torch.set_num_threads(4)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    assert_checkpoint(ck)
    cfg = ck['args']
    if cfg.get('goal_mode') != 'foot_trajectory' or not ck.get('curriculum'):
        parser.error('Expected an S54 foot curriculum checkpoint')
    dev = args.device or 'cuda:0'
    kin = S54Kinematics(dev)
    task = resolve_foot_task(cfg['foot_task_config'])
    goals = GoalSource(None, kin, GoalSpec(cfg['goal_mode'], cfg['goal_frames'], cfg['goal_dt']),
                       AlternatingConfig(**cfg['foot_config']), task.command_mode)
    goals.spec.assert_checkpoint(ck, goals.foot_config)
    curriculum = Curriculum(ck['curriculum']['cfg'])
    curriculum.load_state_dict(ck['curriculum'])
    goals.curriculum = curriculum
    horizon = evaluation_steps(args.seconds, task.episode_steps, CONTROL_DT)
    goals.command_duration = horizon * CONTROL_DT + goals.spec.frames * goals.spec.dt + 1.
    length, freq, yaw = args.step_length, args.step_frequency, args.yaw_rate
    task_name = ('standing' if freq == 0 else 'turn' if length == 0 and yaw != 0 else
                 'curve' if yaw != 0 else 'short' if length <= .08 else 'medium' if length <= .2 else 'long')
    command = Command(task_name, length, freq, yaw)
    why = feasibility(command, curriculum.cfg)
    if why:
        parser.error(f'Command rejected by the saved feasibility envelope: {why}')
    goals.fixed_command = command
    mdp = FootTrackingMDP(kin, torch.tensor(default_joint_pos_list(), device=dev), goals, task)
    assert_task_checkpoint(ck, mdp)
    pi = Policy(OBS_DIM, goals.spec.dim, hist=cfg['hist']).to(dev)
    pi.load_state_dict(ck['pi']); pi.eval()
    networks.AMP_DTYPE = torch.bfloat16 if cfg.get('amp') == 'bf16' else None
    report = dict(checkpoint=str(Path(args.checkpoint).resolve()), iteration=ck['iteration'],
                  trained_stage=curriculum.stage, command=vars(command), controller='saved_actor',
                  reset_noise=not args.nominal_reset, seed=args.seed, episodes=[], video_frames=0)
    print(json.dumps({k: v for k, v in report.items() if k != 'episodes'}), flush=True)
    if curriculum.stage == 0 and freq > 0:
        print('UNTRAINED COMMAND: this checkpoint only trained Standing (C0)', flush=True)
    del ck  # Keep only the Actor on GPU; optimizer/WM/Critic are not needed for play.
    sim_cfg = make_sim(dev, args)
    if 'newton_gl' in selected:
        from isaaclab_visualizers.newton import NewtonGLVisualizerCfg
        sim_cfg.visualizer_cfgs = [NewtonGLVisualizerCfg(
            headless=args.headless, window_width=960, window_height=720,
            eye=(2.5, -2.5, 1.8), lookat=(0., 0., .65))]
    if 'viser' in selected:
        from isaaclab_visualizers.viser import ViserVisualizerCfg
        sim_cfg.visualizer_cfgs = list(sim_cfg.visualizer_cfgs or []) + [ViserVisualizerCfg(
            port=args.viser_port, bind_address=args.viser_host, display_address='localhost',
            open_browser=False, share=False, eye=(1.8, -1.8, 1.4), lookat=(0., 0., .65))]
    encoder = None; gif_frames = []
    try:
        with launch_simulation(sim_cfg, args), torch.no_grad():
            env = TrackingSim(sim_cfg, 1, None, mdp, gamma=cfg.get('gamma', .99),
                              eval_mode=True, max_episode_steps=horizon, goals=goals)
            env.reset_noise = not args.nominal_reset
            visualizers = list(env.sim.visualizers)
            if args.video and not visualizers:
                raise RuntimeError('No active visualizer for video')
            for episode in itertools.count() if args.loop else range(args.episodes):
                if not env.sim.is_running(): break
                torch.manual_seed(args.seed + episode)
                env.done_eval.zero_(); env.failed_eval.zero_()
                env.reset(torch.arange(1, device=dev))
                history = None; reward = error = speed = yaw_error = 0.; steps = 0
                reason = 'viewer_closed'
                while not bool(env.done_eval.all()) and env.sim.is_running():
                    frame_started = time.monotonic()
                    if env.sim.consume_reset_request():
                        reason = 'user_reset'; break
                    S = env.state()
                    obs = goals.obs(mdp, S, env.a_prev, env.xi, env.f, env.command)
                    history = (obs[:, None].repeat(1, cfg['hist'], 1) if history is None else
                               torch.cat((history[:, 1:], obs[:, None]), 1))
                    action = bound_action(pi(history, goals.goal(S, env.f, env.ep_step, env.command)))
                    if not torch.isfinite(action).all():
                        raise FloatingPointError('Nonfinite Actor action')
                    out = env.step(action)
                    S = env.state()
                    if not all(torch.isfinite(v).all() for v in S):
                        raise FloatingPointError('Nonfinite Newton state')
                    steps += 1; reward += float(out['r'][0])
                    error += float(out['terms']['foot_position_error'][0])
                    speed += float(S[2][0, :2].norm()); yaw_error += abs(float(S[3][0, 2]) - yaw)
                    if visualizers:
                        if 'viser' not in selected or args.follow_camera:
                            center = (S[0][0] + env.scene.env_origins[0]).cpu().tolist()
                            env.sim.set_camera_view((center[0]+2.5, center[1]-2.5, 1.8), (center[0], center[1], .65))
                        env.sim.render()
                        if args.video:
                            rgb = visualizers[0].render_rgb_array()
                            if rgb is None:
                                raise RuntimeError('Newton GL did not return RGB')
                            rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
                            if gif and (steps-1) % 5 == 0:
                                from PIL import Image, ImageDraw
                                frame = Image.fromarray(rgb).resize((640, 480))
                                ImageDraw.Draw(frame).text((12, 12), f'Actor iter {report["iteration"]} | episode {episode+1} | {steps*CONTROL_DT:.2f}s', fill='white')
                                gif_frames.append(frame)
                            elif not gif and encoder is None:
                                video = Path(args.video); video.parent.mkdir(parents=True, exist_ok=True)
                                h, w = rgb.shape[:2]
                                encoder = subprocess.Popen([
                                    'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                                    '-s', f'{w}x{h}', '-r', '50', '-i', '-', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(video)], stdin=subprocess.PIPE)
                            if not gif: encoder.stdin.write(rgb.tobytes())
                            report['video_frames'] += 1
                    if bool(out['done'][0]):
                        reason = stop_reasons(S, mdp, out['term'], out['timeout'])[0]
                    if visualizers and not args.no_realtime and ('viser' in selected or not args.headless):
                        delay = CONTROL_DT - (time.monotonic() - frame_started)
                        if delay > 0: time.sleep(delay)
                measurement = dict(episode=episode+1, seed=args.seed+episode, reason=reason,
                    seconds=steps*CONTROL_DT, physical_termination=bool(env.failed_eval[0]),
                    timeout=reason == 'timeout', normalized_sim_reward=reward/max(steps, 1)/mdp.max_step_reward,
                    foot_rmse_m=math.sqrt(error/max(steps, 1)), mean_speed_m_s=speed/max(steps, 1),
                    yaw_rate_mae=yaw_error/max(steps, 1), tracking_success='not_scored_by_viewer')
                report['episodes'].append(measurement)
                print(json.dumps(measurement), flush=True)
                if reason == 'viewer_closed': break
    finally:
        if encoder is not None:
            encoder.stdin.close()
            if encoder.wait() != 0: raise RuntimeError('ffmpeg failed')
        if gif_frames:
            video = Path(args.video); video.parent.mkdir(parents=True, exist_ok=True)
            gif_frames[0].save(video, save_all=True, append_images=gif_frames[1:], duration=100, loop=0)
            report['gif_frames'] = len(gif_frames)
        if args.out:
            target = Path(args.out); target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
