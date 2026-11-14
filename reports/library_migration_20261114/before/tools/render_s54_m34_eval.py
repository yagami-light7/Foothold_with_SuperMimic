"""Render every recorded Newton evaluation state against its global reference.

Rendering replays saved physics states; it never steps or modifies the evaluator.
The cyan reference retains its original world position and heading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from render import set_pose, two_robot_xml
from supermimic.m34 import entries, load
from supermimic.s54 import JOINT_NAMES


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rotation(quaternion):
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=np.float64))
    return matrix.reshape(3, 3)


def probe_video(path, expected_frames, fps):
    result = subprocess.run([
        'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_read_frames,nb_frames,r_frame_rate,width,height,duration',
        '-of', 'json', str(path),
    ], check=True, capture_output=True, text=True)
    stream, = json.loads(result.stdout)['streams']
    actual = int(stream['nb_read_frames'])
    numerator, denominator = map(int, stream['r_frame_rate'].split('/'))
    if actual != expected_frames or numerator / denominator != fps:
        raise RuntimeError(f'Incomplete video: expected {expected_frames} frames at {fps} FPS, got {stream}')
    return stream


def render_one(row, reference, output, model, renderer, width, height, iteration):
    record = Path(row['record'])
    with np.load(record, allow_pickle=False) as archive:
        data = dict(archive)
    if list(data['joint_names']) != JOINT_NAMES or float(data['control_dt']) != .02:
        raise ValueError(f'Recording contract mismatch: {record}')
    q = data['q'][:, 0]
    p = data['root_p'][:, 0]
    quat = data['root_quat'][:, 0]
    frames = data['f'][:, 0]
    indices = frames - frames[0]
    count = len(q)
    if not np.array_equal(indices, np.arange(count)) or count != row['episode_steps'] + 1:
        raise ValueError('Missing/repeated physics states or inconsistent episode length')
    if row['survived'] and count != len(reference['joint_pos']):
        raise ValueError('A completed reference must include every frame, including the terminal state')
    if q.shape != (count, len(JOINT_NAMES)) or not all(np.isfinite(a).all() for a in (q, p, quat)):
        raise ValueError('Invalid recorded poses')
    root_index = list(reference['body_names']).index(str(reference['root_body_name']))
    ref_p = reference['body_pos_w'][indices, root_index]
    ref_quat = reference['body_quat_w'][indices, root_index]
    ref_q = reference['joint_pos'][indices]
    fps = 50
    target = output / f"{row['id']}_full.mp4"
    temporary = output / f"{row['id']}_full.partial.mp4"
    if target.exists() or temporary.exists():
        raise FileExistsError(target)
    sim_data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation = 135, -16
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    encoder = subprocess.Popen([
        'ffmpeg', '-n', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{width}x{height}', '-r', str(fps), '-i', '-', '-an',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(temporary),
    ], stdin=subprocess.PIPE)
    try:
        for i in range(count):
            set_pose(model, sim_data, p[i], rotation(quat[i]), q[i])
            set_pose(model, sim_data, ref_p[i], rotation(ref_quat[i]), ref_q[i], prefix='ref_')
            mujoco.mj_forward(model, sim_data)
            # Keep both original world-space poses visible, including drift.
            separation = float(np.linalg.norm(p[i, :2] - ref_p[i, :2]))
            camera.lookat[:] = [*(.5 * (p[i, :2] + ref_p[i, :2])), .85]
            camera.distance = max(4.2, 3.2 + 1.4 * separation)
            renderer.update_scene(sim_data, camera=camera, scene_option=options)
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            error = float(np.linalg.norm(p[i] - ref_p[i]))
            text = (f"{row['id']} | checkpoint {iteration} | frame {i + 1}/{count}\n"
                    f"t={i * .02:.2f}s | normalized return={row['normalized_return']:.4f} | root error={error:.3f}m")
            draw.rectangle((8, 8, width - 8, 48), fill='#152332')
            draw.multiline_text((12, 12), text, fill='white')
            draw.rectangle((8, height - 43, width - 8, height - 8), fill='#152332')
            draw.multiline_text((12, height - 39),
                                f"Gray: recorded Newton policy | Cyan: {row['split']} reference (original global pose)\n"
                                '50 FPS; every recorded state; no reference alignment', fill='white')
            encoder.stdin.write(np.asarray(frame).tobytes())
            if i == count // 2:
                frame.save(output / f"{row['id']}_preview.jpg", quality=92)
            if (i + 1) % 250 == 0 or i + 1 == count:
                print(f"RENDER {row['id']}: {i + 1}/{count}", flush=True)
    finally:
        encoder.stdin.close()
        code = encoder.wait()
    if code:
        raise RuntimeError(f'ffmpeg failed with exit status {code}')
    probe = probe_video(temporary, count, fps)
    temporary.replace(target)
    return dict(
        id=row['id'], split=row['split'], seed=row['seed'], iteration=iteration,
        normalized_return=row['normalized_return'], root_position_rmse_m=row['root_position_rmse_m'],
        source_record=str(record), source_record_sha256=sha256(record),
        video=str(target), video_sha256=sha256(target),
        source_frames=count, encoded_frames=int(probe['nb_read_frames']),
        fps=fps, duration_s=count / fps, video_probe=probe,
        all_recorded_frames=True, includes_initial_and_terminal_states=True,
        camera='shared world-space camera; midpoint of policy/reference roots; no alignment',
        provenance='Replay of actual independent Newton/MJWarp rollout, with dataset reference overlay',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True, help='Completed per-split results.json')
    parser.add_argument('--dataset', type=Path, default=Path('data/s54_flat_dataset/dataset_export'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=540)
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.width % 2 or args.height % 2:
        parser.error('Positive, even video dimensions required')
    results = json.loads(args.evaluation.read_text())
    if results['status'] != 'completed' or results['split'] not in ('train', 'test'):
        raise ValueError('Use a completed train or test evaluation')
    split = results['split']
    meta, manifest, selected = entries(args.dataset, split)
    selected = {entry['id']: entry for entry in selected}
    if set(selected) != {row['id'] for row in results['rows']}:
        raise ValueError(f'Expected all {split} references')
    args.out.mkdir(parents=True, exist_ok=False)
    status = dict(status='running', checkpoint_sha256=results['checkpoint_sha256'],
                  evaluation=str(args.evaluation.resolve()), iteration=results['iteration'], split=split, videos=[])
    try:
        model = mujoco.MjModel.from_xml_string(two_robot_xml(args.width, args.height))
        with mujoco.Renderer(model, args.height, args.width) as renderer:
            for row in results['rows']:
                reference = load(args.dataset, selected[row['id']], meta, manifest)
                video = render_one(row, reference, args.out, model, renderer,
                                   args.width, args.height, results['iteration'])
                video['reference_file'] = selected[row['id']]['path']
                video['reference_sha256'] = manifest['file_sha256'][video['reference_file']]
                status['videos'].append(video)
                (args.out / 'video_manifest.json').write_text(json.dumps(status, indent=2) + '\n')
        status['status'] = 'completed'
    except BaseException as exc:
        status.update(status='failed', error=repr(exc))
        raise
    finally:
        (args.out / 'video_manifest.json').write_text(json.dumps(status, indent=2) + '\n')
    print(f"PASS: all {len(status['videos'])} {split} videos contain every recorded frame", flush=True)


if __name__ == '__main__':
    main()
