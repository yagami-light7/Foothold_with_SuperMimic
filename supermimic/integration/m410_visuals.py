"""由最终归档生成一支同步回放和一份诊断曲线。"""

from argparse import ArgumentParser
import csv
from functools import cache
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace

if __name__ == '__main__':
    os.environ['SUPERMIMIC_ROBOT'] = 'roban'
    os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import torch

from ..kinematics import RobotKinematics
from ..mdp import unflatten_state
from ..motion import CONTROL_DT
from ..roban import JOINT_NAMES
from .m410_evaluation import GROUPS, ROOT, archive_trace, load_archive
from .m410_quality import local_upper_points
from .m410_reward import HUBER_TERMS, ROOT_TERMS, root_errors
from .m48_evaluation import UPPER
from .m4_reference import F2MotionReference


COLORS = ('#2563a6', '#c95f30', '#158067')
DISPLAY = {'m43_best': 'M4.3 Best', 'm48c_best': 'M4.8-C Best', 'm49a_9000': 'M4.9-A9000',
           'm49c_8500': 'M4.9-C8500', 'm49c_9500': 'M4.9-C9500'}
ARMS = [i for i in UPPER if JOINT_NAMES[i].startswith('zarm_')]
WIDTH, HEIGHT = 1920, 1080
FONT_DIR = Path('/usr/share/fonts/opentype/noto')


@cache
def best_cases():
    return {group: f'{group}_{json.loads((ROOT / name / "evaluation/results.json").read_text())["best_iteration"]}'
            for group, name in GROUPS.items()}


def label(name):
    if name in DISPLAY:
        return DISPLAY[name]
    group, iteration = name.split('_')
    return f'M4.10-{group.upper()}{iteration}'


def outcome(row):
    return row['survived_full_reference'] and row['consecutive_planned_touchdowns'] == len(row['footsteps'])


def demo_sections(info):
    best = best_cases()
    ensemble = ('reference', best['a'], best['b'], best['c'])
    nominal = ('reference', 'm48c_best', 'm49a_9000', 'm49c_9500', best['c'], 'c_10000')
    perturb = info['seeds']['perturb']
    candidates = (best['c'], best['a'], best['b'], 'm48c_best', 'm43_best')
    success = next((seed for name in candidates
                    for seed in perturb if outcome(info['cases'][name]['full'][str(seed)])), None)
    rows = info['cases'][best['c']]['full']
    failed = [seed for seed in perturb if not outcome(rows[str(seed)])]
    # 优先展示一次真实跌倒，再选未连续落脚的试次。
    failure = next((seed for seed in failed if not rows[str(seed)]['survived_full_reference']),
                   failed[0] if failed else None)
    sections = [('参考与历代权重 · 侧面', 1001, nominal, 90),
                ('A / B / C 消融 · 正面', 1001, ensemble, 180)]
    compare = ('reference', 'm48c_best', 'a_10000', best['c'])
    if success is not None:
        sections.append(('初态扰动 · 成功案例对照', success, compare, 132))
    if failure is not None:
        sections.append(('初态扰动 · 失败案例对照', failure, compare, 132))
    return sections


@cache
def video_font(size, bold=False):
    from PIL import ImageFont

    family = 'NotoSansCJK-Bold.ttc' if bold else 'NotoSansCJK-Regular.ttc'
    return ImageFont.truetype(str(FONT_DIR / family), size, index=2)


def video_panels(names, seed, info, data, ref):
    panels = []
    for name in names:
        if name == 'reference':
            panels.append(dict(title='原始 F2 Reference', state=ref['state'], feet=ref['feet'], row=None))
        else:
            row = info['cases'][name]['full'][str(seed)]
            trace = archive_trace(data, row)
            title = label(name)
            if name == best_cases()['c']:
                title += ' · 步态与全身最佳'
            elif name == 'c_10000':
                title += ' · 末轮'
            panels.append(dict(title=title, state=trace['state'][:, :13+len(JOINT_NAMES)],
                               feet=trace['foot'], row=row))
    return panels


def render_video(info, data, motion, output, *, sections=None, panels_from=video_panels,
                 show_hand_errors=False, hold_end_frames=0, banner='Roban F2', panel_clock=False,
                 elevations=None):
    import mujoco
    from PIL import Image, ImageDraw

    from .contact_diagnostics import RobotGeometry

    f2 = motion.at_time(motion.control_time)
    ref_state = torch.cat((f2['root_p'], f2['root_quat'], f2['root_v'], f2['root_w'], f2['q']), -1).numpy()
    ref = dict(state=ref_state, feet=f2['foot_p'].numpy())
    times = motion.control_time.numpy()
    touch = motion.touchdown_time.numpy()
    sides = motion.touchdown_side.numpy()
    targets = motion.at_time(touch)['foot_p'].numpy()[np.arange(len(sides)), sides]
    colors = ((.22, .83, .96), (1., .63, .25))
    floor = (.18, .225, .26, 1.)
    identity = np.eye(3).ravel()
    robot = RobotGeometry(SimpleNamespace(sizes=()))
    model = robot.model
    model.vis.headlight.ambient[:] = .36
    model.vis.headlight.diffuse[:] = .56
    model.vis.headlight.specular[:] = .12
    camera = mujoco.MjvCamera()
    camera.elevation, camera.distance = -18, 2.20
    options = mujoco.MjvOption()
    options.geomgroup[0], options.geomgroup[1] = 0, 1

    def add(scene, kind, size, xyz, rgba):
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], kind, np.asarray(size, dtype=np.float64),
                            np.asarray(xyz, dtype=np.float64), identity, np.asarray(rgba, dtype=np.float32))
        scene.ngeom += 1

    def render_pose(renderer, panel, frame):
        state = panel['state'][frame]
        robot.set_robot(state[:3], state[3:7], state[13:])
        renderer.update_scene(robot.data, camera=camera, scene_option=options)
        scene = renderer.scene
        box, cyl = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER
        add(scene, box, (4., 4., .006), (0., 0., -.006), floor)
        for value in np.arange(-2.5, 3., .25):
            shade = (.28, .34, .40, 1.)
            add(scene, box, (4., .001, .0003), (0., value, .0003), shade)
            add(scene, box, (.001, 4., .0003), (value, 0., .0003), shade)
        for xyz, side in zip(panel.get('targets', targets), panel.get('sides', sides)):
            add(scene, cyl, (.035, .001, .001), (*xyz[:2], .0014), (*colors[side], .9))
            add(scene, cyl, (.028, .001, .001), (*xyz[:2], .0026), floor)
        picture = Image.fromarray(renderer.render())
        markers = ImageDraw.Draw(picture)
        # 将实测足底中心投影到画面，落脚时仍可看清标记。
        mean = lambda name: np.mean([getattr(view, name) for view in scene.camera], axis=0)
        eye, forward, up = (mean(name) for name in ('pos', 'forward', 'up'))
        near, bottom, top, center = (mean(name) for name in
                                     ('frustum_near', 'frustum_bottom', 'frustum_top', 'frustum_center'))
        scale = picture.height / (top - bottom)
        right = np.cross(forward, up)
        for side, point in enumerate(panel['feet'][frame]):
            delta = point - eye
            depth = delta @ forward
            if depth <= near:
                continue
            x = picture.width / 2 + (near * (delta @ right) / depth - center) * scale
            y = (top - near * (delta @ up) / depth) * scale
            markers.ellipse((x-5., y-5., x+5., y+5.), fill=tuple(round(255*c) for c in colors[side]),
                            outline='#16212a', width=2)
        return picture

    def frame_image(renderer, panels, heading, seed, frame, width, columns, section_times):
        canvas = Image.new('RGB', (WIDTH, HEIGHT), '#0c1420')
        draw = ImageDraw.Draw(canvas)
        draw.text((22, 16), heading, font=video_font(33, True), fill='#f4f8fa')
        draw.text((700, 28), f'{banner} · seed {seed} · 1×', font=video_font(19), fill='#b5c7d4')
        draw.text((1898, 14), f'{section_times[frame]:.2f} / {section_times[-1]:.2f} s',
                  font=video_font(32, True), fill='#f4f8fa', anchor='rt')
        for index, panel in enumerate(panels):
            x = 22 + index % columns * (width + 14)
            y = 80 + index // columns * 470
            row = panel['row']
            i = min(frame, len(panel['state']) - 1)
            canvas.paste(render_pose(renderer, panel, i), (x, y + 43))
            for y1, y2 in ((y, y+42), (y+414, y+457)):
                draw.rectangle((x, y1, x+width-1, y2), fill='#152235')
            draw.text((x+12, y+7), panel['title'], font=video_font(22, True), fill='#f4f8fa')
            if row is None:
                caption = '青 / 橙对应左右足。圆环为计划落脚，圆点为实测足底。'
                plan_touch = panel.get('touch', touch)
                progress = f'计划 {np.sum(plan_touch <= section_times[frame]+1.e-8)}/{len(plan_touch)}'
                if panel_clock:
                    progress += f' · {section_times[i]:.2f} s'
            else:
                body = row['whole_body']
                foot = row['foot_position_rmse_m']*1000
                root = row['root_xy_rmse_m']*1000
                caption = f'全程 RMSE   足 {foot:.1f} / Root XY {root:.1f} mm   臂 {body["arm_rmse_rad"]:.2f} rad'
                caption = panel.get('caption', caption)
                landed = sum(ev['actual_touchdown_s'] is not None
                             and ev['actual_touchdown_s'] <= section_times[i]+1.e-8
                             for ev in row['footsteps'])
                status = '运行中'
                if frame >= row['steps']:
                    reason = {'reference_end': '参考结束', 'root_height': '根高终止', 'root_tilt': '倾斜终止',
                              'root_height+root_tilt': '根高及倾斜终止'}
                    status = f'{reason.get(row["termination"], row["termination"])} · {section_times[i]:.2f} s'
                progress = f'落脚匹配 {landed}/{len(row["footsteps"])}   |   {status}'
            if show_hand_errors:
                draw.text((x+width-12, y+8), progress, font=video_font(18), fill='#a9c7d7', anchor='rt')
                draw.text((x+12, y+415), caption, font=video_font(17), fill='#e3edf4')
                if row is not None:
                    left, right = [value * 1000 for value in body['hand_local_rmse_m']]
                    draw.text((x+12, y+436), f'躯干局部手部 RMSE   左 {left:.1f} mm   右 {right:.1f} mm',
                              font=video_font(17), fill='#a9c7d7')
                elif 'caption' in panel:
                    draw.text((x+12, y+436), panel['caption'], font=video_font(17), fill='#a9c7d7')
            else:
                draw.text((x+12, y+416), caption, font=video_font(15), fill='#e3edf4')
                draw.text((x+12, y+437), progress, font=video_font(15), fill='#a9c7d7')
        bar = (25, 1895)
        xp = lambda t: bar[0] + (bar[1] - bar[0]) * t / section_times[-1]
        draw.line((bar[0], 1029, bar[1], 1029), width=4, fill='#2a3c51')
        draw.line((bar[0], 1029, xp(section_times[frame]), 1029), width=4, fill='#aad7f0')
        for t in (() if panel_clock else touch):
            u = xp(t)
            draw.ellipse((u-4, 1025, u+4, 1033), fill='#f4f8fa')
            draw.text((u, 1040), f'{t:.2f} s', font=video_font(15), fill='#b5c7d4', anchor='mt')
        return canvas

    sections = demo_sections(info) if sections is None else sections
    with TemporaryDirectory(dir=output.parent) as scratch:
        tmp = Path(scratch) / output.name
        command = ['ffmpeg', '-nostdin', '-v', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24',
                   '-video_size', f'{WIDTH}x{HEIGHT}', '-framerate', '25', '-i', '-', '-an',
                   '-c:v', 'libx264', '-preset', 'fast', '-crf', '17', '-threads', '4',
                   '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(tmp)]
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            for section, (heading, seed, names, azimuth) in enumerate(sections):
                panels = panels_from(names, seed, info, data, ref)
                section_times = (np.arange(max(len(panel['state']) for panel in panels)) * CONTROL_DT
                                 if panel_clock else times)
                columns = 3 if len(names) > 4 else 2
                width = (WIDTH - 44 - (columns-1)*14) // columns
                xy = np.concatenate([panel['state'][:, :2] for panel in panels])
                camera.lookat[:] = (*((xy.min(0) + xy.max(0))/2), .53)
                camera.azimuth = azimuth
                if elevations is not None:
                    camera.elevation = elevations[section]
                model.vis.global_.offwidth, model.vis.global_.offheight = width, 371
                with mujoco.Renderer(model, height=371, width=width) as renderer:
                    for frame in range(0, len(section_times), 2):
                        pixels = frame_image(renderer, panels, heading, seed, frame, width, columns,
                                             section_times).tobytes()
                        encoder.stdin.write(pixels)
                    for _ in range(hold_end_frames):
                        encoder.stdin.write(pixels)
                print(heading, seed, '已渲染', flush=True)
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError('FFmpeg 编码失败')
        tmp.replace(output)


def read_train(root):
    runs = {}
    for group, name in GROUPS.items():
        with (root / name / 'logs/train.csv').open() as f:
            rows = list(csv.DictReader(f))
        runs[group] = {key: np.asarray([float(row[key]) if row[key] else np.nan for row in rows])
                       for key in rows[0]}
    return runs


def plot_curves(info, arrays, motion, output):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import font_manager

    family = FONT_DIR / 'NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(family)
    matplotlib.rcParams.update({'font.family': font_manager.FontProperties(fname=family).get_name(),
                               'pdf.fonttype': 3, 'axes.unicode_minus': False, 'axes.spines.top': False,
                               'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .20})
    train = read_train(ROOT)
    best = best_cases()
    nominal = {group: archive_trace(arrays, info['cases'][name]['full']['1001'])
               for group, name in best.items()}
    f2 = motion.at_time(motion.control_time)
    qp = f2['q'].numpy()
    time = motion.control_time.numpy()
    landings = motion.touchdown_time.tolist()
    q0 = np.asarray(info['physics']['action_center'])
    alpha = np.asarray(info['physics']['action_scale'])
    ylim = arrays['joint_limits']
    term_names = dict(zip((*ROOT_TERMS, *HUBER_TERMS),
                          ('Root XY', 'Root Z', '航向', '肩 Huber', '肘 Huber', '腰 Huber')))
    group_colors = dict(zip('abc', COLORS))

    def tidy(fig, title, xlabel=''):
        fig.suptitle(title, fontsize=16, x=.04, ha='left')
        for ax in fig.axes:
            if xlabel:
                ax.set_xlabel(xlabel)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc='best', fontsize=7, ncol=2 if len(labels) > 4 else 1, frameon=False)
        pdf.savefig(fig)
        plt.close(fig)

    def plot_train(ax, group, field, *, label=None, style='-', smooth=False):
        row = train[group]
        x, y = row['iteration'], row[field]
        color = group_colors[group]
        if smooth:
            ax.plot(x, y, color=color, linewidth=.7, alpha=.35)
            x = x[1:-1]
            y = np.convolve(y, np.ones(3)/3, mode='valid')
        ax.plot(x, y, style, color=color, label=label, linewidth=1.15)

    with TemporaryDirectory(dir=output.parent) as scratch:
        tmp = Path(scratch) / output.name
        with PdfPages(tmp) as pdf:
            fig, axes = plt.subplots(3, 2, figsize=(12, 10), layout='constrained')
            for group in 'abc':
                name = group.upper()
                plot_train(axes[0, 0], group, 'ep_ret_norm', label=name, smooth=True)
                plot_train(axes[0, 1], group, 'baseline_reward', label=name, smooth=True)
                plot_train(axes[1, 0], group, 'total_reward', label=name+' 真实')
                plot_train(axes[1, 0], group, 'imagined_reward', label=name+' 想象', style='--')
                s = json.loads((ROOT/GROUPS[group]/'evaluation/results.json').read_text())['milestones']
                axes[1, 1].plot([int(k) for k in s], [r['aggregate']['complete_tracking_rate']*3 for r in s.values()],
                                '.-', color=group_colors[group], label=name)
            for ax, terms, groups in ((axes[2, 0], ROOT_TERMS, 'ac'), (axes[2, 1], HUBER_TERMS, 'bc')):
                for group in groups:
                    for term, style in zip(terms, ('-', '--', ':')):
                        plot_train(ax, group, term+'_contribution', label=group.upper()+' '+term_names[term], style=style)
            for ax, title in zip(axes.flat, ('Episode Return（各组自身分母）', '真实每步共同基线奖励',
                                            '每步真实与想象奖励', '周期名义完整四步（/3）',
                                            'Root 每步分项贡献', 'Huber 每步分项贡献')):
                ax.set_title(title)
            tidy(fig, '训练与奖励 · 粗线含三点滑动平均，CSV 保留原样本', '训练轮次')

            fig, axes = plt.subplots(3, 2, figsize=(12, 10), layout='constrained')
            for group in 'abc':
                plot_train(axes[0, 0], group, 'wm_foot_rmse_8', label=group.upper())
                plot_train(axes[0, 1], group, 'v_loss', label=group.upper())
                plot_train(axes[1, 0], group, 'clamp_leg_l5_joint', label=group.upper()+' 左')
                plot_train(axes[1, 0], group, 'clamp_leg_r5_joint', label=group.upper()+' 右', style='--')
                plot_train(axes[1, 1], group, 'pd_target_clamp_ratio', label=group.upper())
            c = train['c']
            axes[0, 1].set_yscale('log')
            for i, term in enumerate(('foot', *term_names)):
                color = plt.get_cmap('tab10')(i)
                axes[2, 0].plot(c['iteration'], c[f'term_{term}_actor_gradnorm'], color=color,
                                label=term_names.get(term, 'Foot'), linewidth=1.)
                if term in term_names:
                    axes[2, 1].plot(c['iteration'], c[f'grad_cosine_foot_{term}'], color=color,
                                    label=term_names[term], linewidth=1.)
                    axes[2, 1].plot(c['iteration'], c[f'grad_cosine_bootstrap_{term}'], color=color,
                                    linestyle='--', linewidth=.7, alpha=.72)
            for ax, title in zip(axes.flat, ('WM 八步足误差 m', 'Critic Loss', '左右踝裁剪比例',
                                            '总体目标裁剪比例', 'C 组梯度范数',
                                            'C 组梯度余弦 · 实线 Foot / 虚线 Bootstrap')):
                ax.set_title(title)
            tidy(fig, '模型更新与控制', '训练轮次')

            fig, axes = plt.subplots(3, 3, figsize=(13, 9), layout='constrained')
            selected = ('zarm_l2_joint', 'zarm_r2_joint', 'zarm_l4_joint', 'zarm_r4_joint', 'waist_yaw_joint', 'leg_l4_joint')
            for group, trace in nominal.items():
                f = trace['frame']
                state = unflatten_state(torch.as_tensor(trace['state']))
                err = root_errors(state[0], state[1], f2['root_p'][f], f2['root_R'][f])
                root_values = [v.numpy()*s for v, s in zip(err, (1000., 1000., 180./np.pi))]
                joints = [state[4][:, JOINT_NAMES.index(k)].numpy() for k in selected]
                for ax, y in zip(axes.flat, (*root_values, *joints)):
                    ax.plot(time[f], y, color=group_colors[group], label=group.upper(), linewidth=1.)
            for ax, joint in zip(axes.flat[3:], selected):
                ax.plot(time, qp[:, JOINT_NAMES.index(joint)], '--', color='#43474b', linewidth=1., label='F2')
            for ax, title in zip(axes.flat, ('Root XY 误差 mm', 'Root Z 误差 mm', '航向误差 °', '左肩 2', '右肩 2',
                                            '左肘', '右肘', '腰偏航', '左膝')):
                ax.set_title(title)
                for t in landings:
                    ax.axvline(t, color='.75', linewidth=.65)
            for ax in axes.flat[3:]:
                ax.set_ylabel('关节角 rad')
            tidy(fig, '名义 Best · Root 和关节跟踪 · seed 1001', 'F2 时间 s')

            fig = plt.figure(figsize=(13, 9), layout='constrained')
            hand_axes = [fig.add_subplot(2, 3, 1, projection='3d'), fig.add_subplot(2, 3, 2, projection='3d')]
            speed_axes = [fig.add_subplot(2, 3, i) for i in (3, 4, 5, 6)]
            ref_state = tuple(f2[key] for key in ('root_p', 'root_R', 'root_v', 'root_w', 'q', 'qd'))
            signals = [('F2', ref_state, np.arange(len(time)), '#43474b')]
            for group, trace in nominal.items():
                signals.append((group.upper(), unflatten_state(torch.as_tensor(trace['state'])), trace['frame'],
                                group_colors[group]))
            for name, state, frames, color in signals:
                hands = local_upper_points(state, motion.kin)[:, 4:].numpy()
                for i, ax in enumerate(hand_axes):
                    ax.plot(*hands[:, i].T, color=color, label=name, linewidth=1.)
                qd = state[5][:, UPPER].numpy()
                vals = [qd, np.diff(qd, axis=0)/CONTROL_DT, np.diff(qd, n=2, axis=0)/CONTROL_DT**2]
                for i, (ax, y) in enumerate(zip(speed_axes, vals)):
                    ax.plot(time[frames[i:]], np.sqrt(np.square(y).mean(-1)), color=color, linewidth=1., label=name)
                if name != 'F2':
                    t = nominal[name.lower()]['target'][:, UPPER]
                    dv = np.sqrt(np.square(np.diff(t, n=2, axis=0)).mean(-1))
                    speed_axes[-1].plot(time[frames[2:-1]], dv, color=color, label=name, linewidth=1.)
            for i, ax in enumerate(hand_axes):
                ax.set_title(('左手', '右手')[i]+' · 躯干局部 3D（m）')
                ax.set(xlabel='前后', ylabel='横向', zlabel='高')
            for ax, title in zip(speed_axes, ('上身角速度 RMS rad/s', '上身角加速度 RMS rad/s²',
                                              '上身 Jerk RMS rad/s³', '实际 PD 目标二阶差分 RMS rad')):
                ax.set_title(title)
                ax.set_xlabel('F2 时间 s')
                ax.axvline(.20, color='.5', linewidth=.8)
            tidy(fig, '上身运动与启动 · seed 1001')

            fig, axes = plt.subplots(3, 2, figsize=(12, 10), layout='constrained')
            for side, joint in enumerate(('leg_l5_joint', 'leg_r5_joint')):
                j = JOINT_NAMES.index(joint)
                axes[0, side].axhline((ylim[0, j]-q0[j])/alpha[j], color='.35', linestyle=':', label='下限阈值')
                axes[1, side].plot(time, qp[:, j], '--', color='#43474b', linewidth=1., label='F2')
                for group, trace in nominal.items():
                    t = time[trace['frame']]
                    color = group_colors[group]
                    axes[0, side].plot(t[:-1], trace['raw_action'][:, j], color=color, label=group.upper())
                    axes[1, side].plot(t[:-1], trace['target'][:, j], color=color, linestyle='--',
                                       label=group.upper()+' PD')
                    axes[1, side].plot(t, trace['state'][:, 13+j], color=color, linewidth=1., label=group.upper()+' 实测')
                    axes[2, side].plot(t, trace['actuator_torque'][:, j], color=color, label=group.upper())
                leg = ('左踝', '右踝')[side]
                for ax, title in zip(axes[:, side], ('Raw Action', 'PD 目标及关节角 rad', '实测力矩 N·m')):
                    ax.set_title(leg+' · '+title)
            tidy(fig, '双踝控制与关节限位 · seed 1001', 'F2 时间 s')
        tmp.replace(output)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--m4_reference', type=Path, required=True)
    parser.add_argument('--archive', type=Path, default=ROOT / 'cuda_evaluation.npz')
    parser.add_argument('--kind', choices=('all', 'video', 'curves'), default='all')
    args = parser.parse_args()
    torch.set_num_threads(2)
    info, data = load_archive(args.archive)
    if 'comparison' not in info:
        raise ValueError('请先完成 CUDA 复评与汇总')
    motion = F2MotionReference(args.m4_reference, RobotKinematics('cpu'))
    if args.kind in ('all', 'video'):
        render_video(info, data, motion, ROOT / 'whole_body_comparison.mp4')
    if args.kind in ('all', 'curves'):
        plot_curves(info, data, motion, ROOT / 'diagnostic_curves.pdf')


if __name__ == '__main__':
    main()

