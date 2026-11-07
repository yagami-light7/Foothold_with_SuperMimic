"""Render paired, recorded Newton evaluations; no policy rollout or training.

Left: Motion policy with its world-space LAFAN1 reference robot.
Right: Foot policy with command foot meshes, paths and contact schedule.
Foot has no whole-body reference. A terminated panel is held and labelled.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def read_json(path):
    return json.loads(Path(path).read_text())


def select_evaluation(run_dir, evaluation=None):
    """Require a complete same-checkpoint pair, never mix different iterations."""
    candidates = [Path(evaluation)] if evaluation else list((Path(run_dir)/'evaluation').glob('*'))
    pairs = []
    for directory in candidates:
        paths = [directory/name/'summary.json' for name in ('motion', 'foot')]
        if not all(p.is_file() for p in paths):
            continue
        motion, foot = map(read_json, paths)
        if (motion.get('interface') != 'motion' or foot.get('interface') != 'foot'
                or motion.get('full') or foot.get('full')
                or motion['iteration'] != foot['iteration']
                or motion['checkpoint'] != foot['checkpoint']):
            continue
        pairs.append((int(motion['iteration']), directory.name, directory, motion, foot))
    if not pairs:
        raise ValueError('No complete Motion/Foot evaluation pair from the same checkpoint')
    _, _, directory, motion, foot = max(pairs, key=lambda x: x[:2])
    return directory, motion, foot


def select_row(summary, *, seed, duration=None, offset=None):
    matches = [row for row in summary['rows'] if row['seed'] == seed
               and (duration is None or math.isclose(row['duration_s'], duration))
               and (offset is None or math.isclose(row.get('offset_s', 0), offset))]
    if len(matches) != 1:
        raise ValueError(f'Expected one {summary["interface"]} recording for seed={seed}, '
                         f'duration={duration}, offset={offset}; found {len(matches)}')
    return matches[0]


def recording_path(row, directory):
    recorded = Path(row.get('trace', row.get('record', '')))
    local = directory/recorded.name
    if local.is_file():
        return local
    if recorded.is_file():
        return recorded
    raise FileNotFoundError(recorded)


def playback_index(seconds, steps, dt):
    if seconds < 0 or steps < 1 or dt <= 0:
        raise ValueError('Invalid replay time or recording interval')
    return min(int(math.floor(seconds/dt + .5)), steps), seconds >= steps*dt-1e-9


def resolve_ffmpeg(explicit=None):
    if explicit:
        candidate = shutil.which(explicit)
        if not candidate:
            raise FileNotFoundError('--ffmpeg is not an executable: '+explicit)
        return candidate
    if executable := shutil.which('ffmpeg'):
        return executable
    # Use the repository's existing SDK environment; do not install packages.
    for executable in sorted((ROOT/'.venv-swanlab/lib').glob(
            'python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64*')):
        if os.access(executable, os.X_OK):
            return str(executable)
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError('ffmpeg unavailable; pass --ffmpeg /path/to/ffmpeg') from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', '--run-dir', type=Path, required=True)
    parser.add_argument('--eval-dir', type=Path, help='Default: latest complete paired evaluation')
    parser.add_argument('--out', type=Path, required=True, help='Fresh .mp4 filename')
    parser.add_argument('--seconds', type=float, default=10., help='Video duration, capped to recorded Motion/reference interval')
    parser.add_argument('--seed', type=int, default=54, help='Foot evaluation seed; Motion uses its recorded seed 54')
    parser.add_argument('--motion-offset', type=float, default=0.)
    parser.add_argument('--foot-duration', type=float, default=10., help='Select the existing 2/5/10-second Foot test')
    parser.add_argument('--stop-at-first-end', action='store_true', help='End both panels when either physical recording ends')
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--width', type=int, default=1280, help='Total width of both panels')
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--ffmpeg', help='Optional explicit executable; a bundled encoder is detected automatically')
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0 or not 1 <= args.fps <= 50:
        parser.error('Use finite positive seconds and 1–50 fps')
    if args.width < 960 or args.width % 4 or args.height < 480 or args.height % 2:
        parser.error('Use width >=960 divisible by 4 and even height >=480')
    output = args.out.resolve()
    sidecar, preview = output.with_suffix('.json'), output.with_suffix('.png')
    if output.suffix.lower() != '.mp4':
        parser.error('--out must be .mp4')
    if any(path.exists() for path in (output, sidecar, preview)):
        parser.error('Output or sidecar exists; choose a fresh output name')
    directory, motion_summary, foot_summary = select_evaluation(args.run_dir, args.eval_dir)
    paired_reference = foot_summary.get('goal_source') == 'reference_foot'
    motion_row = select_row(motion_summary, seed=args.seed if paired_reference else 54, offset=args.motion_offset)
    foot_row = select_row(foot_summary, seed=args.seed, duration=args.foot_duration,
                          offset=args.motion_offset if paired_reference else None)
    motion_path = recording_path(motion_row, directory/'motion')
    foot_path = recording_path(foot_row, directory/'foot')
    profile = read_json(args.run_dir/'profile.json')
    manifest = Path(profile['reference_manifest'])
    if not manifest.is_absolute():
        manifest = ROOT/manifest
    encoder_path = resolve_ffmpeg(args.ffmpeg)

    os.environ.setdefault('MUJOCO_GL', 'egl')
    import mujoco
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from tools.visualization.render import load_recording, two_robot_xml, apply_render_style, set_pose, REFERENCE_RGBA
    from tools.visualization.policy_preview import gpu_slot
    from supermimic.foot_trajectory import FootstepCommand, query, command_time
    from supermimic.rotations import quat_to_mat
    from supermimic.s54 import MJCF_PATH, SOURCE_MJCF, FOOT_BODY_NAMES

    torch.set_num_threads(4)
    started = time.monotonic()
    motion = load_recording(motion_path, str(manifest))
    with np.load(foot_path, allow_pickle=False) as archive:
        foot = dict(archive)
    x = foot['state'][:, 0]
    steps = int(foot_row['episode_lengths'][0])
    dt = float(foot['control_dt'])
    if x.shape != (steps+1, 71) or not np.isfinite(x).all() or not math.isclose(dt, .02):
        raise ValueError('Expected one finite S54 Newton Foot episode at dt=.02')
    if not np.allclose(np.linalg.norm(x[:, 3:7], axis=1), 1, atol=1e-4):
        raise ValueError('Invalid recorded root quaternion')
    if paired_reference:
        # This explicit diagnostic has an original whole-body source, available
        # to visualization only; the recorded Foot Actor still received 230D.
        foot_motion = load_recording(foot_path, str(manifest))
        target_p = foot['reference_foot_p'][:, 0]
        target_q = foot['reference_foot_quat'][:, 0]
        target_contact = foot['reference_contact'][:, 0]
    else:
        command = FootstepCommand(**{k[8:]: torch.as_tensor(v) for k, v in foot.items()
                                    if k.startswith('command_') and k != 'command_step'})
        ticks = torch.as_tensor(foot['command_step'][:, 0])
        reference = query(command, command_time(ticks, dt, torch.float32)[None])
        target_p, target_q = reference.position[0].numpy(), reference.quaternion[0].numpy()
        target_contact = reference.contact[0].numpy()
        if not np.allclose(target_p, foot['reference_foot_p'][:, 0], atol=1e-6):
            raise ValueError('Saved foot command differs from recorded reference')
    target_R = quat_to_mat(torch.as_tensor(target_q)).numpy()
    msteps = int(motion_row['episode_lengths'][0]); mdt = float(motion['control_dt'])
    seconds = min(args.seconds, msteps*mdt)
    if args.stop_at_first_end:
        seconds = min(seconds, steps*dt)
    count = int(math.ceil(seconds*args.fps-1e-9))
    if count < 1:
        raise ValueError('No frames in requested interval')
    width, height = args.width//2, args.height
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
        title_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
    except OSError:
        font = title_font = ImageFont.load_default()

    def foot_xml():
        root = ET.parse(MJCF_PATH).getroot()
        root.find('compiler').set('meshdir', str(SOURCE_MJCF.parent.parent/'meshes'))
        apply_render_style(root, width, height)
        world = root.find('worldbody')
        for side, name in enumerate(FOOT_BODY_NAMES):
            body = ET.SubElement(world, 'body', name=f'command_foot_{side}', mocap='true')
            for source in root.find(f'.//body[@name="{name}"]').findall('geom'):
                if source.get('type') != 'mesh':
                    continue
                geom = deepcopy(source)
                geom.attrib.pop('name', None)
                geom.attrib.update(material='render_reference', rgba=REFERENCE_RGBA, contype='0', conaffinity='0')
                body.append(geom)
        ET.SubElement(world, 'geom', name='render_floor', type='plane', size='0 0 .01',
                      material='groundplane', contype='0', conaffinity='0')
        return ET.tostring(root, encoding='unicode')

    def camera():
        cam = mujoco.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance = 135, -12, 4.0
        return cam

    def aim(cam, actual, reference_xy):
        center = (actual[:2]+reference_xy)*.5
        cam.lookat[:] = [center[0], center[1], .8]
        cam.distance = max(4.0, 3.0+1.4*np.linalg.norm(actual[:2]-reference_xy))

    def line(scene, first, last, color, pixels=2.):
        if np.linalg.norm(last-first) < 1e-8:
            return
        if scene.ngeom >= scene.maxgeom:
            raise RuntimeError('Too many trajectory markers')
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3),
                           np.eye(3).flatten(), np.array(color, dtype=np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, pixels,
                            np.asarray(first, dtype=np.float64), np.asarray(last, dtype=np.float64))
        scene.ngeom += 1

    def label(image, title, info, legend, ended=None):
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, width, 88), fill='#152332')
        draw.text((12, 9), title, font=title_font, fill='#f4f7fa')
        draw.multiline_text((12, 40), info, font=font, fill='#d6e7f1', spacing=4)
        draw.rectangle((0, height-62, width, height), fill='#152332')
        draw.multiline_text((12, height-55), legend, font=font, fill='#c6eaf7', spacing=4)
        if ended:
            draw.rectangle((8, height-112, width-8, height-70), fill='#85252a')
            draw.text((15, height-101), ended, font=font, fill='white')
        return np.asarray(image)

    print(f'Evaluation iteration {motion_summary["iteration"]}: {directory}', flush=True)
    print(f'Motion: {msteps*mdt:g}s; Foot: {steps*dt:g}s ({foot_row["stop_reasons"][0]}). '
          f'Video: {count/args.fps:g}s / {count} frames. Rendering saved states only.', flush=True)
    print(('References: full Walking robots in both panels; Foot reference is visualization-only. ' if paired_reference else
           'References: full Motion robot; Foot command meshes/path/contact schedule. ')+
          'Terminated panels hold their final state and reference with an explicit label.', flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f'.{output.stem}.{os.getpid()}.partial.mp4')
    option = mujoco.MjvOption(); option.geomgroup[3] = 0
    first_frame = None
    try:
        with (gpu_slot() if os.environ['MUJOCO_GL'] == 'egl' else nullcontext()), ExitStack() as stack:
            mm = mujoco.MjModel.from_xml_string(two_robot_xml(width, height))
            fm = mujoco.MjModel.from_xml_string(two_robot_xml(width, height) if paired_reference else foot_xml())
            md, fd = mujoco.MjData(mm), mujoco.MjData(fm)
            mr = stack.enter_context(mujoco.Renderer(mm, height, width))
            fr = stack.enter_context(mujoco.Renderer(fm, height, width))
            mc, fc = camera(), camera()
            mocap_ids = [] if paired_reference else [int(fm.body_mocapid[fm.body(f'command_foot_{side}').id]) for side in (0, 1)]
            ff = subprocess.Popen([encoder_path, '-n', '-loglevel', 'error', '-f', 'rawvideo',
                '-pix_fmt', 'rgb24', '-s', f'{args.width}x{height}', '-r', str(args.fps), '-i', '-',
                '-an', '-c:v', 'libx264', '-threads', '2', '-pix_fmt', 'yuv420p', '-crf', '20',
                '-movflags', '+faststart', str(temporary)], stdin=subprocess.PIPE)
            try:
                next_progress = time.monotonic()+5
                for frame in range(count):
                    t = frame/args.fps
                    mi, _ = playback_index(t, msteps, mdt)
                    fi, ended = playback_index(t, steps, dt)
                    R = np.empty(9)
                    mujoco.mju_quat2Mat(R, motion['root_quat'][mi, 0].astype(float))
                    set_pose(mm, md, motion['root_p'][mi, 0], R.reshape(3, 3), motion['q'][mi, 0])
                    ref_frame = int(motion['reference_frame'][mi, 0])
                    rp = motion['ref_root_p'][ref_frame]
                    set_pose(mm, md, rp, motion['ref_root_R'][ref_frame], motion['ref_q'][ref_frame], prefix='ref_')
                    mujoco.mj_forward(mm, md)
                    aim(mc, motion['root_p'][mi, 0], rp[:2])
                    mr.update_scene(md, camera=mc, scene_option=option)
                    left = label(Image.fromarray(mr.render()), 'Motion Tracking',
                        f't={mi*mdt:.2f}s | iteration {motion_summary["iteration"]}\n'
                        f'Walking reference offset {args.motion_offset:g}s | NR {motion_row["norm_return"]:.3f}',
                        'Gray: Newton policy | Cyan: full-body reference\nReference follows its original world trajectory')

                    mujoco.mju_quat2Mat(R, x[fi, 3:7].astype(float))
                    set_pose(fm, fd, x[fi, :3], R.reshape(3, 3), x[fi, 13:42])
                    if paired_reference:
                        rf = int(foot_motion['reference_frame'][fi, 0])
                        set_pose(fm, fd, foot_motion['ref_root_p'][rf], foot_motion['ref_root_R'][rf],
                                 foot_motion['ref_q'][rf], prefix='ref_')
                    else:
                        fd.mocap_pos[mocap_ids] = target_p[fi]
                        fd.mocap_quat[mocap_ids] = target_q[fi]
                    mujoco.mj_forward(fm, fd)
                    aim(fc, x[fi, :3], target_p[fi, :, :2].mean(0))
                    fr.update_scene(fd, camera=fc, scene_option=option)
                    samples = np.unique(np.linspace(0, steps, min(steps+1, 140)).astype(int))
                    for side, color in enumerate(((.15, .65, 1., 1.), (1., .55, .15, 1.))):
                        path = target_p[samples, side]
                        for a, b in zip(path[:-1], path[1:]):
                            line(fr.scene, a, b, color)
                        actual = fd.xpos[fm.body(FOOT_BODY_NAMES[side]).id]
                        line(fr.scene, actual, target_p[fi, side], (1., .9, .25, 1.), 3.)
                        line(fr.scene, target_p[fi, side], target_p[fi, side]+.12*target_R[fi, side, :, 0], color, 4.)
                    contacts = target_contact[fi].tolist()
                    phase = ' | '.join(f'{name}: {"stance" if contact else "swing"}'
                                       for name, contact in zip(('L', 'R'), contacts))
                    right = label(Image.fromarray(fr.render()), 'Foot Encoder | Walking Foot Goal' if paired_reference else 'Foot Trajectory Tracking',
                        f't={fi*dt:.2f}s | seed {args.seed} | {phase}\n'
                        f'Test {args.foot_duration:g}s | recorded {steps*dt:.2f}s | NR {foot_row["norm_return"]:.3f}',
                        ('Cyan: Walking reference (visualization only)\nBlue/orange: L/R foot goals | Yellow: tracking error' if paired_reference else
                         'Cyan: command feet | Blue/orange: L/R target paths\nYellow: actual-to-target error | No full-body reference'),
                        (f'{"TERMINATED" if foot_row["n_failed"] else "ENDED"}: '
                         f'{foot_row["stop_reasons"][0].replace("motion_","M:").replace("foot_","F:")} '
                         f'at {steps*dt:.2f}s; held') if ended else None)
                    combined = np.concatenate((left, right), axis=1)
                    combined[:, width-1:width+1] = [24, 35, 48]
                    if first_frame is None:
                        first_frame = Image.fromarray(combined.copy())
                    ff.stdin.write(combined.tobytes())
                    if time.monotonic() >= next_progress:
                        print(f'[Render] {frame+1}/{count} frames ({(frame+1)/count:.0%})', flush=True)
                        next_progress = time.monotonic()+5
            finally:
                ff.stdin.close()
                code = ff.wait(timeout=60)
            if code:
                raise RuntimeError(f'Video encoder exited {code}')
        os.link(temporary, output)  # Atomic publication, refuses any existing destination.
        temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)
    first_frame.save(preview)
    sidecar.write_text(json.dumps(dict(render_only=True,simulation_started=False,
        run_dir=str(args.run_dir.resolve()),evaluation_dir=str(directory.resolve()),iteration=motion_summary['iteration'],
        motion_record=str(motion_path.resolve()),foot_record=str(foot_path.resolve()),reference_manifest=str(manifest),
        motion_reference='Full-body LAFAN1 world trajectory',
        foot_reference=('Walking FK foot goal; full-body source is visualization only' if paired_reference else
                        'Saved command foot poses/path/contact schedule; no whole-body reference'),
        requested_seconds=args.seconds,video_seconds=count/args.fps,fps=args.fps,frames=count,
        size=[args.width,height],motion_record_seconds=msteps*mdt,foot_record_seconds=steps*dt,
        foot_stop_reason=foot_row['stop_reasons'][0],terminated_panel='Final physical pose and target held with label',
        ffmpeg=encoder_path,wall_seconds=time.monotonic()-started),indent=2)+'\n')
    print(f'Wrote {output}\nPreview: {preview}\nMetadata: {sidecar}', flush=True)


if __name__ == '__main__':
    main()
