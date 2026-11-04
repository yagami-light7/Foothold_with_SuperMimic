"""Audit recorded S54 failures and root startup; no simulator or training is run."""

# Support both direct execution and python -m from the repository layout.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_root = str(_ToolPath(__file__).resolve().parents[2])
if _tool_root not in _tool_sys.path:
    _tool_sys.path.insert(0, _tool_root)

import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from supermimic.reference_quality import root_motion_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', type=Path, default=Path('logs/s54_motion'))
    parser.add_argument('--record', default='traj_10000.npz')
    parser.add_argument('--motion_dir', type=Path, help='Reference directory if evaluation overrode args.json')
    parser.add_argument('--out', type=Path, default=Path('results/s54/training_failure_10k'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with np.load(args.run_dir / args.record, allow_pickle=False) as archive:
        data = dict(archive)
    config = json.loads((args.run_dir / 'args.json').read_text())
    report = dict(run=str(args.run_dir), record=args.record, clips={},
                  scope='Recorded policy states; head angle is relative to its parent, not a render correction')
    for c, name in enumerate(data['names']):
        start, length = int(data['clip_start'][c]), int(data['clip_len'][c])
        dead = np.flatnonzero(~data['alive'][:, c])
        end = int(dead[0]) if len(dead) else min(length - 1, len(data['q']) - 1)
        frames = data['f'][:end + 1, c].astype(int)
        rotations = Rotation.from_quat(data['root_quat'][:end + 1, c][:, [1, 2, 3, 0]]).as_matrix()
        height = abs(data['root_p'][:end + 1, c, 2] - data['ref_root_p'][frames, 2])
        tilt = abs(rotations[:, 2, 2] - data['ref_root_R'][frames, 2, 2])
        failure = bool(height[-1] > .25 or tilt[-1] > .8)
        source = (args.motion_dir or Path(config['motion_dir'])) / f'{name}.csv'
        source_values = np.loadtxt(source, delimiter=',')
        heads = np.rad2deg(data['q'][:end + 1, c, -2:])
        clip = dict(recorded_stop_s=end * float(data['control_dt']),
                    full_clip_duration_s=(length - 1) * float(data['control_dt']),
                    termination_reason='tilt' if tilt[-1] > .8 else ('height' if failure else 'completed'),
                    final_height_error_m=float(height[-1]), final_tilt_error=float(tilt[-1]),
                    reference_root=root_motion_report(source_values),
                    head_median_deg=np.median(heads, axis=0).tolist(), head_min_deg=heads.min(0).tolist(),
                    head_max_deg=heads.max(0).tolist(), reference_heads_max_abs_deg=float(abs(np.rad2deg(source_values[:, -2:])).max()),
                    initial_root_position_error_m=float(abs(data['root_p'][0, c] - data['ref_root_p'][start]).max()),
                    initial_joint_error_rad=float(abs(data['q'][0, c] - data['ref_q'][start]).max()))
        report['clips'][str(name)] = clip
        samples = []
        for t in range(end + 1):
            samples.append(dict(time_s=t * .02, frame=int(frames[t] - start),
                                root_z=float(data['root_p'][t, c, 2]), reference_z=float(data['ref_root_p'][frames[t], 2]),
                                height_error=float(height[t]), tilt_error=float(tilt[t]),
                                head_yaw_deg=float(heads[t, 0]), head_pitch_deg=float(heads[t, 1])))
        with (args.out / f'{name}_recorded_errors.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(samples[0])); writer.writeheader(); writer.writerows(samples)
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        time = np.arange(len(height)) * .02
        axes[0].plot(time, height, label='Root height error (m)')
        axes[0].plot(time, tilt, label='Root up-axis error')
        axes[0].axhline(.25, ls='--', color='tab:blue', alpha=.5)
        axes[0].axhline(.8, ls='--', color='tab:orange', alpha=.5)
        axes[0].set(title=f'{name}: {clip["termination_reason"]} at {end*.02:.2f}s', xlabel='Time (s)')
        axes[1].plot(time, heads[:, 0], label='Head yaw')
        axes[1].plot(time, heads[:, 1], label='Head pitch')
        axes[1].axhline(0, ls='--', color='black', label='Reference head = 0')
        axes[1].set(title='Actual head joints (degrees)', xlabel='Time (s)', ylabel='Degrees')
        for ax in axes:
            ax.legend(fontsize=8); ax.grid(alpha=.25)
        fig.tight_layout(); fig.savefig(args.out / f'{name}_diagnosis.png', dpi=160); plt.close(fig)
    probe = Path('logs/s54_startup_aiming_diagnostic/ckpt_010000_eval.json')
    if probe.exists():
        report['same_checkpoint_corrected_10s_probe'] = json.loads(probe.read_text())
    (args.out / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
