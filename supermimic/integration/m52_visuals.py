"""从同种子独立复评归档输出一支四视角原速视频及姿态诊断 PDF。"""

from argparse import ArgumentParser
import json
import os
from pathlib import Path

if __name__ == '__main__':
    os.environ['SUPERMIMIC_ROBOT'] = 'roban'
    os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import torch

from ..kinematics import RobotKinematics
from ..motion import CONTROL_DT
from ..roban import JOINT_NAMES
from .m410_visuals import render_video
from .m50_reference import MANIFEST, REFERENCE_ROOT, load_clip, read_manifest
from .m51_metrics import read_guide_heading, trace_signals
from .m52_audit import frozen_task
from .m52_orientation import counterfactual, posture_signals, trace_states
from .m52_task import OUTPUT, SOURCE


def plots(output, traces, rows, labels, mdp, guide, info):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    layout = ((('rootRoll', 'Root roll (deg)'), ('rootPitch', 'Root pitch (deg)'),
               ('root_tilt', 'Root tilt error (deg)'), ('torso_tilt', 'Chest tilt error (deg)'),
               ('root_speed', 'Root angular speed (rad/s)'), ('root_xy', 'Root XY error (mm)'),
               ('foot_yaw', 'Foot yaw error (deg)'), ('foot_omega', 'Loaded normal angular speed (rad/s)')),
              (('hand_world_m', 'Hand world error (mm)'), ('hand_torso_local_m', 'Hand chest-local error (mm)'),
               ('elbow_world_m', 'Elbow world error (mm)'), ('elbow_torso_local_m', 'Elbow chest-local error (mm)'),
               ('upper_arm_world_deg', 'Upper arm direction error (deg)'), ('forearm_world_deg', 'Forearm direction error (deg)'),
               ('hand_orientation', 'Hand orientation error (deg)'), ('angular_velocity', 'Root angular velocity error (rad/s)')))
    values, reference = {}, None
    for name, trace in traces.items():
        actual, ref = trace_states(trace, mdp.motion)
        signal = posture_signals(actual, ref, mdp.kin)[0]
        foot = trace_signals(trace, mdp, guide)
        loaded = trace['contact'] & foot['planned']
        signal.update(rootRoll=signal['root_rpy'][:, 0], rootPitch=signal['root_rpy'][:, 1],
                      root_speed=actual[3].norm(dim=-1), root_xy=1000*signal['root_xy'],
                      foot_yaw=foot['error'][..., 0], foot_omega=np.where(loaded, foot['omega'], np.nan))
        values[name] = {key: np.asarray(value) for key, value in signal.items()}
        if reference is None or len(ref[0]) > len(reference['rootRoll']):
            reference = dict(rootRoll=signal['reference_rpy'][:, 0], rootPitch=signal['reference_rpy'][:, 1],
                             root_speed=ref[3].norm(dim=-1))
    with PdfPages(output) as pdf:
        for page in layout:
            fig, axes = plt.subplots(4, 2, figsize=(12, 8), layout='constrained')
            for ax, (key, title) in zip(axes.flat, page, strict=True):
                for name, row in values.items():
                    y = row[key]
                    if key.endswith('_m'):
                        y = 1000*y
                    t = np.arange(len(y))*CONTROL_DT
                    if y.ndim == 2:
                        left, = ax.plot(t, y[:, 0], label=labels[name])
                        ax.plot(t, y[:, 1], '--', color=left.get_color(), alpha=.8)
                    else:
                        ax.plot(t, y, label=labels[name])
                if key in reference:
                    y = reference[key]
                    ax.plot(np.arange(len(y))*CONTROL_DT, y, color='black', linewidth=1.4, label='Reference')
                ax.set(title=title, xlabel='s')
                ax.grid(alpha=.25)
            axes[0, 0].legend(ncol=3, fontsize=7)
            fig.suptitle('Paired trajectory diagnostics · L solid, R dashed when split')
            pdf.savefig(fig)
            plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout='constrained')
        diagnostics = {}
        for name, trace in traces.items():
            diagnostics[name] = rows[name].get('counterfactual') or counterfactual(
                *trace_states(trace, mdp.motion), mdp.kin)
        for ax, key, label in zip(axes.flat[:3], ('hand_world_m_rms', 'elbow_world_m_rms', 'hand_torso_local_m_rms'),
                                  ('Hand world', 'Elbow world', 'Hand chest-local'), strict=True):
            for name, cases in diagnostics.items():
                ax.plot(list(cases), [1000*np.mean(data[key]) for data in cases.values()],
                        'o-', label=labels[name])
            ax.set(title=f'{label} counterfactual FK', ylabel='RMSE (mm)')
            ax.grid(alpha=.25)
        axes[0, 0].legend(fontsize=8)
        ax = axes[1, 1]
        for index, (name, label) in enumerate(labels.items()):
            nominal = info['cases'][name]['nominal']
            y = [row['complete']/row['attempts'] for row in nominal.values()]
            ax.bar(np.arange(len(y))-.3+.2*index, y, width=.18, label=label)
        ax.set_xticks(np.arange(len(nominal)), list(nominal))
        ax.set(title='Complete footstep rate', ylim=(0., 1.05))
        pdf.savefig(fig)
        plt.close(fig)


def render(info, arrays, motion, rows, traces, labels, seed, target):
    def panels(names, _seed, _info, _arrays, reference):
        result = [dict(state=reference['state'], feet=reference['feet'], title='Original Reference', row=None)]
        for name in names[1:]:
            trace, row = traces[name], rows[name]
            posture = row['posture']['all']
            hand = [1000*x for x in posture['hand_world_m_rms']]
            caption = f'倾斜 RMSE {posture["tilt_rms_deg"]:.2f}° · 手世界 RMSE {hand[0]:.1f} / {hand[1]:.1f} mm'
            result.append(dict(state=trace['state'][:, :13+len(JOINT_NAMES)], feet=trace['foot'],
                               title=labels[name], row=dict(row, whole_body=row['body']), caption=caption))
        return result

    names = ('reference', *labels)
    sections = [(f'姿态跟踪 · {name}', seed, names, azimuth) for name, azimuth in
                (('正面', 180), ('侧面', 90), ('斜前方', 145), ('俯视', 90))]
    render_video(info, arrays, motion, target, sections=sections, panels_from=panels, show_hand_errors=True,
                 banner=f'Roban {motion.clip_name}', panel_clock=True, elevations=(-8., -8., -24., -89.))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    parser.add_argument('--reference-root', type=Path, default=REFERENCE_ROOT)
    parser.add_argument('--clip', choices=('F1', 'F2', 'F3'), default='F2')
    parser.add_argument('--seed', type=int, choices=(1001, 1002, 1003), default=1001)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with np.load(args.out/'cuda_evaluation.npz', allow_pickle=False) as package:
        info = json.loads(package['metadata'].item())
        if not info.get('complete'):
            parser.error('请先完成独立 CUDA 复评')
        winners = info['winners']
        labels = {'m51': 'M5.1-C1500'} | {winners[group]: f'M5.2-{winners[group].upper()}' for group in 'abc'}
        rows = {name: info['cases'][name]['seen'][args.clip][str(args.seed)] for name in labels}
        keys = ('state', 'frame', 'foot', 'foot_R', 'foot_velocity', 'force', 'contact', 'foot_angular_velocity')
        traces = {name: {key: package[row['prefix']+key] for key in keys} for name, row in rows.items()}
    kin = RobotKinematics('cpu')
    item = next(entry for entry in read_manifest(MANIFEST)['train'] if entry['clip_id'] == args.clip)
    motion = load_clip(item, kin, 'cpu', args.reference_root, 'train')
    guide = read_guide_heading(args.reference_root/item['file_path'])
    mdp = frozen_task(kin, motion, torch.load(SOURCE, map_location='cpu', mmap=True, weights_only=False))
    plots(args.out/f'{args.clip}_posture.pdf', traces, rows, labels, mdp, guide, info)
    render(info, {}, motion, rows, traces, labels, args.seed, args.out/f'{args.clip}_posture.mp4')


if __name__ == '__main__':
    main()
