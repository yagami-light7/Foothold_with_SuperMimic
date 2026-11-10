"""Convert complete evaluation MP4s to full-frame GIFs for SwanLab's Video API."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def frame_count(path):
    result = subprocess.run([
        'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_read_frames,r_frame_rate,duration', '-of', 'json', str(path),
    ], check=True, capture_output=True, text=True)
    stream, = json.loads(result.stdout)['streams']
    return int(stream['nb_read_frames']), stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--colors', type=int, default=32)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest['status'] != 'completed':
        raise ValueError('Render the complete MP4s first')
    manifest['status'] = 'preparing_swanlab_gifs'
    output = args.manifest.with_name('swanlab_video_manifest.json')
    if output.exists():
        shutil.copyfile(output, output.with_name('swanlab_video_manifest.previous.json'))
    try:
        for video in manifest['videos']:
            source = Path(video['video'])
            target = source.with_suffix('.swanlab.gif')
            palette = source.with_suffix('.swanlab.palette.png')
            temporary = source.with_suffix('.swanlab.partial.gif')
            if target.exists():
                raise FileExistsError(target)
            print(f"GIF {video['id']}: converting all {video['source_frames']} frames", flush=True)
            # A two-pass palette avoids buffering an entire full-length movie.
            subprocess.run([
                'ffmpeg', '-n', '-loglevel', 'error', '-i', str(source),
                '-vf', f'scale={args.width}:-1:flags=lanczos,palettegen=max_colors={args.colors}:stats_mode=diff',
                '-frames:v', '1', str(palette),
            ], check=True)
            subprocess.run([
                'ffmpeg', '-n', '-loglevel', 'error', '-i', str(source), '-i', str(palette),
                '-lavfi', f'[0:v]scale={args.width}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=none',
                '-vsync', '0', '-loop', '0', str(temporary),
            ], check=True)
            frames, stream = frame_count(temporary)
            if frames != video['source_frames']:
                raise RuntimeError(f'GIF frame loss: {frames} != {video["source_frames"]}')
            # GIF frame delays have 0.01 s precision; 50 FPS is represented exactly.
            if abs(float(stream['duration']) - video['duration_s']) > .011:
                raise RuntimeError(f'GIF duration changed: {stream}')
            if temporary.stat().st_size > 10 * 1024**2:
                raise ValueError('SwanLab media must be <=10 MiB; reduce width/colors without dropping frames')
            temporary.replace(target)
            palette.unlink()
            video.update(swanlab_video=str(target), swanlab_frames=frames,
                         swanlab_video_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                         swanlab_video_probe=stream,
                         swanlab_note=f'{args.width}px GIF for SDK compatibility; every MP4 frame retained at 50 FPS')
            output.write_text(json.dumps(manifest, indent=2) + '\n')
        manifest['status'] = 'completed'
    except BaseException as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        output.write_text(json.dumps(manifest, indent=2) + '\n')
    print('PASS: SwanLab GIFs retain every original video frame and the full duration', flush=True)


if __name__ == '__main__':
    main()
