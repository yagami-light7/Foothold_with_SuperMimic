"""Compose separate, synchronized train/test grids from complete rollout videos."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess


def probe(path, count=False):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0']
    if count:
        command.append('-count_frames')
    command += ['-show_entries', 'stream=width,height,r_frame_rate,nb_frames,nb_read_frames,duration',
                '-of', 'json', str(path)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    stream, = json.loads(result.stdout)['streams']
    return stream


def compose(manifest_path, out, split, columns, width, height):
    manifest = json.loads(manifest_path.read_text())
    if manifest['status'] != 'completed':
        raise ValueError(f'Incomplete source videos: {manifest_path}')
    videos = manifest['videos']
    if not videos or any(v['split'] != split for v in videos):
        raise ValueError('Train and test must remain in separate grids')
    fps = 50
    longest = max(v['encoded_frames'] for v in videos)
    rows = math.ceil(len(videos) / columns)
    header = 64
    tile_w, tile_h = width // columns, (height - header) // rows
    if width != columns * tile_w or height != rows * tile_h + header or tile_w % 2 or tile_h % 2:
        raise ValueError('Grid dimensions must divide into even-sized tiles')
    work = out / f'{split}_work'
    work.mkdir()
    font = subprocess.check_output(['fc-match', '-f', '%{file}', 'DejaVu Sans'], text=True).strip()
    inputs, filters, layout = [], [], []
    for i, video in enumerate(videos):
        source = Path(video['video']).resolve()
        info = probe(source)
        if (video['source_frames'] != video['encoded_frames'] or
                int(info['nb_frames']) != video['encoded_frames'] or
                Fraction(info['r_frame_rate']) != fps or video['fps'] != fps):
            raise ValueError(f'Incomplete recording or unexpected frame rate: {source}')
        inputs += ['-threads', '1', '-i', str(source)]
        title = work / f'tile_{i:02d}.txt'
        title.write_text(f"{video['id']} | R={video['normalized_return']:.4f}")
        frames = video['encoded_frames']
        filters.append(
            f'[{i}:v]setpts=N/({fps}*TB),'
            f'scale={tile_w}:{tile_h}:flags=lanczos:force_original_aspect_ratio=decrease:force_divisible_by=2,'
            f'pad={tile_w}:{tile_h}:(ow-iw)/2:(oh-ih)/2:color=0x152332,setsar=1,'
            f'tpad=stop_mode=clone:stop={longest-frames},trim=end_frame={longest},'
            'drawbox=x=0:y=0:w=iw:h=32:color=0x152332:t=fill,'
            f'drawtext=fontfile={font}:textfile={title.resolve()}:expansion=none:'
            'fontsize=18:fontcolor=white:x=8:y=8,'
            'drawbox=x=0:y=ih-25:w=iw:h=25:color=0x152332:t=fill,'
            f"drawtext=fontfile={font}:text='PLAYING':fontsize=14:fontcolor=0x8ce6c9:"
            f"x=8:y=h-20:enable='lt(n,{frames})',"
            f"drawtext=fontfile={font}:text='ENDED - last frame held':fontsize=14:fontcolor=0xffcd80:"
            f"x=8:y=h-20:enable='gte(n,{frames})',"
            f'drawbox=x=0:y=0:w=iw:h=ih:color=0x4b596b:t=1[v{i}]'
        )
        layout.append(f'{(i % columns)*tile_w}_{(i // columns)*tile_h}')
    label = 'TRAIN' if split == 'train' else 'EVAL / TEST'
    header_text = work / 'header.txt'
    header_text.write_text(f'{label} | {len(videos)} references | checkpoint {manifest["iteration"]} | 50 FPS | t=%{{pts:hms}}')
    legend = work / 'legend.txt'
    legend.write_text('Gray: Newton rollout | Cyan: original global reference | Short clips hold their final frame')
    filters.append(
        ''.join(f'[v{i}]' for i in range(len(videos))) +
        f'xstack=inputs={len(videos)}:layout={"|".join(layout)}:fill=0x152332:shortest=1,'
        f'pad={width}:{height}:0:{header}:color=0x152332,'
        f'drawtext=fontfile={font}:textfile={header_text.resolve()}:'
        'fontsize=24:fontcolor=white:x=12:y=8,'
        f'drawtext=fontfile={font}:textfile={legend.resolve()}:expansion=none:'
        'fontsize=16:fontcolor=0xc2d2e5:x=12:y=39,format=yuv420p[grid]'
    )
    graph = work / 'filters.txt'
    graph.write_text(';\n'.join(filters))
    target = out / ('train_grid.mp4' if split == 'train' else 'eval_grid.mp4')
    temporary = target.with_suffix('.partial.mp4')
    if target.exists() or temporary.exists():
        raise FileExistsError(target)
    command = ['ffmpeg', '-n', '-hide_banner', '-loglevel', 'warning',
               '-filter_complex_threads', '4', *inputs,
               '-filter_complex_script', str(graph), '-map', '[grid]', '-an',
               '-vsync', '0', '-frames:v', str(longest), '-c:v', 'libx264',
               '-threads', '4', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', '-progress', str(work / 'progress.txt'), '-nostats', str(temporary)]
    (work / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    print(f'COMPOSE {split}: {len(videos)} panels, {longest} frames, {width}x{height}', flush=True)
    with (work / 'ffmpeg.log').open('w') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    info = probe(temporary, count=True)
    if (int(info['nb_read_frames']) != longest or Fraction(info['r_frame_rate']) != fps or
            abs(float(info['duration']) - longest/fps) > 1/fps):
        raise RuntimeError(f'Incomplete grid video: {info}')
    temporary.replace(target)
    preview = out / f'{split}_grid_preview.jpg'
    subprocess.run(['ffmpeg', '-n', '-loglevel', 'error', '-ss', '12', '-i', str(target),
                    '-frames:v', '1', str(preview)], check=True)
    digest = hashlib.sha256()
    with target.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    result = dict(split=split, video=str(target), sha256=digest.hexdigest(),
                  source_manifest=str(manifest_path), checkpoint_sha256=manifest['checkpoint_sha256'],
                  iteration=manifest['iteration'], panels=len(videos), columns=columns, rows=rows,
                  width=width, height=height, fps=fps, frames=longest, duration_s=longest/fps,
                  all_source_frames_retained=True, synchronized_start=True,
                  shorter_clips='final state held and explicitly labelled after source recording ends',
                  source_videos=[dict(id=v['id'], frames=v['encoded_frames'], video=v['video']) for v in videos],
                  output_probe=info)
    print(f'PASS {target}: {longest} decoded frames, {longest/fps:.2f}s', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Evaluation root containing videos and videos_train')
    parser.add_argument('--out', type=Path, required=True, help='New output directory')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    status = dict(status='running', created_at_utc=datetime.now(timezone.utc).isoformat(), grids=[])
    try:
        for split, directory, columns, width, height in (
                ('train', 'videos_train', 4, 1920, 1080), ('test', 'videos', 2, 1280, 720)):
            status['current_split'] = split
            (args.out / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
            result = compose(args.root / directory / 'video_manifest.json', args.out,
                             split, columns, width, height)
            status['grids'].append(result)
        status['status'] = 'completed'
    except BaseException as exc:
        status.update(status='failed', error=repr(exc))
        raise
    finally:
        (args.out / 'status.json').write_text(json.dumps(status, indent=2) + '\n')


if __name__ == '__main__':
    main()
