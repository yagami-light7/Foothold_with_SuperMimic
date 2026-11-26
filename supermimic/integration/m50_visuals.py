"""将 M5.0 实测轨迹渲染为原速视频并汇总训练曲线。"""

from argparse import ArgumentParser
import csv
import json
import os
from pathlib import Path

if __name__ == '__main__':
    os.environ['SUPERMIMIC_ROBOT'] = 'roban'
    os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import torch

from ..kinematics import RobotKinematics
from .m410_visuals import render_video
from .m48_evaluation import load_archive
from .m50_evaluation import trial_ok
from .m50_reference import MANIFEST, REFERENCE_ROOT, load_clip, read_manifest
from .m50_task import OUTPUT, TRAIN_CLIPS


def demo_sections(info):
    name = info['selected']
    seen = info['seen'][name]['trials']
    unseen = info['heldout'][name]['trials']
    six = tuple((clip, model) for model in ('ref', name) for clip in TRAIN_CLIPS)
    sections = [('同一策略 · 三种步态 · 正面', 1001, six, 180),
                ('同一策略 · 三种步态 · 侧面', 1001, six, 90)]
    compare = tuple((clip, model) for clip in ('F1', 'F3') for model in ('ref', 'B1250', name))
    sections.append(('迁移前后 · F1 和 F3', 1001, compare, 132))
    for clips in (('H1', 'H2'), ('H3', 'H4')):
        panels = tuple((clip, model) for clip in clips for model in ('ref', name))
        sections.append((f'未见参数 · {clips[0]} 和 {clips[1]}', 1001, panels, 132))
    outcomes = {trial_ok(trials['1001']) for trials in unseen.values()}
    for wanted, label in ((True, '完整成功'), (False, '未完整匹配')):
        if wanted in outcomes:
            continue
        case = next(((clip, int(seed)) for clip, trials in unseen.items() for seed, row in trials.items()
                     if trial_ok(row) == wanted), None)
        if case is not None:
            clip, seed = case
            sections.append((f'未见参数带初态偏差 · {label}', seed, ((clip, 'ref'), (clip, name)), 132))
    if not all(trial_ok(rows['1001']) for rows in seen.values()):
        # 正常初态固定为 1001，保留失败轨迹及其实际终止时刻。
        print('正常种子 1001 的六宫格包含未完整匹配片段', flush=True)
    return sections


def video(args, manifest, info, arrays):
    kin = RobotKinematics('cpu')
    pose_dim = 13 + len(kin.joint_names)
    clips, ref_panels = {}, {}
    for split in ('train', 'heldout'):
        for entry in manifest[split]:
            name = entry['clip_id']
            motion = clips[name] = load_clip(entry, kin, 'cpu', args.reference_root, split)
            f = motion.at_time(motion.control_time)
            state = torch.cat([f[key] for key in ('root_p', 'root_quat', 'root_v', 'root_w', 'q')], dim=-1)
            touch, side = motion.touchdown_time.numpy(), motion.touchdown_side.numpy()
            targets = motion.at_time(touch)['foot_p'].numpy()[np.arange(len(side)), side]
            ref_panels[name] = dict(state=state.numpy(), feet=f['foot_p'].numpy(), touch=touch,
                                    sides=side, targets=targets, row=None, title=f'{name} · 参考',
                                    caption=f'{entry["step_length"]:.3f} m · {entry["step_frequency"]:.4f} Hz')

    def panels_from(names, seed, _info, _data, _ref):
        panels = []
        for clip, candidate in names:
            panel = ref_panels[clip].copy()
            if candidate != 'ref':
                split = 'heldout' if clip.startswith('H') else 'seen'
                row = info[split][candidate]['trials'][clip][str(seed)]
                key = row['prefix']
                title = 'B1250' if candidate == 'B1250' else f'M5.0 #{candidate}'
                panel.update(title=f'{clip} · {title}', row=dict(row, whole_body=row['body']),
                             state=arrays[key+'state'][:, :pose_dim], feet=arrays[key+'foot'])
            panels.append(panel)
        return panels

    render_video(info, arrays, clips['F1'], args.out/'multi_reference.mp4',
                 sections=demo_sections(info), panels_from=panels_from, banner='Roban M5.0',
                 show_hand_errors=True, panel_clock=True)


def plot_curves(out, info):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    config = json.loads((out/'config/train.json').read_text())
    adaptation = config['m50_adaptation_iters']
    with (out/'logs/train.csv').open() as csv_file:
        rows = list(csv.DictReader(csv_file))
    training = [row for row in rows if row['stage'] in ('adaptation', 'joint')]
    x = [int(row['iteration']) + (adaptation if row['stage'] == 'joint' else 0) for row in training]
    colors = dict(zip(TRAIN_CLIPS, ('#2476b3', '#168171', '#c55b32')))
    evaluations = json.loads((out/'evaluation/results.json').read_text())['milestones']
    milestones = sorted(evaluations.values(), key=lambda row: row['iteration'])
    x_eval = [adaptation + row['iteration'] for row in milestones]
    path = out/'training_curves.pdf'

    def values(field):
        series = np.asarray([float(row[field]) if row.get(field) else np.nan for row in training])
        if field.endswith('/episode_return'):
            count_field = field.replace('/episode_return', '/episodes')
            # 没有结束的 Episode 时均值为空，避免误绘为零奖励。
            has_episode = [float(row[count_field]) > 0 for row in training]
            series[~np.asarray(has_episode)] = np.nan
        return series

    def training_line(ax, field, label, scale=1., color=None, style='-'):
        ax.plot(x, values(field)*scale, style, label=label, color=color)

    def finish_page(pdf, fig, axes, title):
        for ax in axes.ravel():
            for boundary in (adaptation, adaptation+1000, adaptation+3000):
                ax.axvline(boundary, color='#82909a', alpha=.50, linestyle=':')
            ax.grid(alpha=.22)
            ax.set_xlabel('Updates including adaptation')
            ax.legend(fontsize=8, frameon=False)
        fig.suptitle(title)
        pdf.savefig(fig)
        plt.close(fig)

    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(3, 2, figsize=(12, 10), layout='constrained')
        a = axes.ravel()
        for clip in TRAIN_CLIPS:
            color = colors[clip]
            training_line(a[0], f'{clip}/episode_return', clip, color=color)
            training_line(a[1], f'{clip}/foot_rmse_m', clip, 1000., color)
            training_line(a[2], f'{clip}/root_xy_rmse_m', clip, 1000., color)
            for side, style in (('left', '-'), ('right', '--')):
                training_line(a[3], f'{clip}/hand_{side}_rmse_m', f'{clip} {side[0].upper()}', 1000., color, style)
            a[4].plot(x_eval, [row['nominal'][clip]['observed']['body']['arm_rmse_rad'] for row in milestones],
                      'o-', label=clip, color=color)
            a[5].plot(x_eval, [row['nominal'][clip]['complete']/row['nominal'][clip]['attempts'] for row in milestones],
                      'o-', label=clip, color=color)
        for ax, title in zip(a, ('Real episode return', 'Foot position RMSE (mm)', 'Root XY RMSE (mm)',
                                 'Torso-local hand RMSE (mm)', 'Arm RMSE (rad), frozen CUDA',
                                 'Complete gait rate, frozen CUDA')):
            ax.set_title(title)
        finish_page(pdf, fig, axes, 'M5.0 per-reference tracking')

        fig, axes = plt.subplots(3, 2, figsize=(12, 10), layout='constrained')
        a = axes.ravel()
        for clip in TRAIN_CLIPS:
            training_line(a[0], f'{clip}/replay_fraction', clip, color=colors[clip])
            for horizon, style in ((1, ':'), (4, '--'), (8, '-')):
                training_line(a[1], f'{clip}/wm_foot_rmse_{horizon}', f'{clip} {horizon}', 1000.,
                              colors[clip], style)
            training_line(a[5], f'{clip}/real_imagined_gap', clip, color=colors[clip])
        for horizon in (1, 4, 8):
            training_line(a[2], f'wm_error_{horizon}', f'{horizon} steps')
        training_line(a[3], 'v_loss', 'Critic loss')
        training_line(a[3], 'actor_bootstrap_gradnorm', 'Bootstrap gradient')
        a[3].set_yscale('symlog', linthresh=.001)
        for field, label in (('reset_probability', 'Perturbed reset fraction'), ('reset_scale', 'Range scale')):
            training_line(a[4], field, label)
        for ax, title in zip(a, ('Replay transition share', 'WM foot RMSE (mm)', 'Normalized WM prediction error',
                                 'Critic loss and Actor bootstrap gradient', 'Initial-state perturbation curriculum',
                                 'Real minus imagined reward')):
            ax.set_title(title)
        finish_page(pdf, fig, axes, 'M5.0 training stability and curriculum')

        name = info['selected']
        groups = ('nominal', 'perturb', 'holdout_reset')
        seen = info['seen'][name]['summary']
        unseen = info['heldout'][name]['summary'] if name in info.get('heldout', {}) else {}
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
        for ax, metric, ylabel in zip(axes, ('complete', 'foot_position_rmse_m'), ('Complete gait rate', 'Foot RMSE (mm)')):
            all_clips = [*TRAIN_CLIPS, *unseen.get('nominal', {})]
            j = np.arange(len(all_clips))
            for i, group in enumerate(groups):
                entries = {**seen[group], **unseen.get(group, {})}
                heights = []
                for clip in all_clips:
                    item = entries.get(clip)
                    if item is None:
                        heights.append(np.nan)
                    elif metric == 'complete':
                        heights.append(item['complete']/item['attempts'])
                    else:
                        height = item['completed'][metric]
                        heights.append(np.nan if height is None else height*1000)
                ax.bar(j+(i-1)*.24, heights, width=.23, label=group)
            ax.set(xticks=j, xticklabels=all_clips, ylabel=ylabel)
            ax.grid(axis='y', alpha=.22)
            ax.legend(frameon=False)
        fig.suptitle(f'M5.0 selected update {name} | Foot RMSE on complete trials')
        pdf.savefig(fig)
        plt.close(fig)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, default=MANIFEST)
    parser.add_argument('--reference-root', type=Path, default=REFERENCE_ROOT)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    parser.add_argument('--curves-only', action='store_true', help='仅更新已复评结果的曲线 PDF')
    args = parser.parse_args()
    torch.set_num_threads(2)
    archive = args.out/'cuda_evaluation.npz'
    if args.curves_only:
        with np.load(archive, allow_pickle=False) as data:
            plot_curves(args.out, json.loads(data['metadata'].item()))
        return
    info, arrays = load_archive(archive)
    plot_curves(args.out, info)
    video(args, read_manifest(args.library), info, arrays)


if __name__ == '__main__':
    main()
