"""F2 上身参考和冻结回放的测量与可视化。"""

import argparse
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

if __name__ == '__main__':
    os.environ['SUPERMIMIC_ROBOT'] = 'roban'
    os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import torch

from ..mdp import unflatten_state
from ..motion import CONTROL_DT
from ..roban import JOINT_NAMES, actuator_parameters
from ..rotations import mat_to_quat


UPPER = [i for i, name in enumerate(JOINT_NAMES) if name.startswith(('waist_', 'zarm_'))]
ARMS = UPPER[1:]
HAND_NAMES = ('l_handball', 'r_handball')


def rmse(x):
    return float(np.sqrt(np.mean(np.square(x))))


@torch.no_grad()
def upper_series(state, kin):
    positions, rotations, _, _ = kin.forward(*state, all_bodies=True)
    hands = positions[:, [kin.body_names.index(name) for name in HAND_NAMES]]
    fore_aft = ((hands - state[0][:, None]) * state[1][:, None, :, 0]).sum(-1)
    R = rotations[:, 0]
    tilt = torch.stack((torch.atan2(R[:, 2, 1], R[:, 2, 2]), torch.asin(-R[:, 2, 0].clamp(-1., 1.))), -1)
    return tuple(value.detach().cpu().numpy() for value in (state[4][:, UPPER], state[5][:, UPPER], hands, fore_aft, tilt))


def high_frequency(x):
    # 对速度去均值并加 Hann 窗，以 8 Hz 为界统计功率比例。
    if len(x) < 4:
        return 0.
    spectrum = np.abs(np.fft.rfft((x-x.mean(0))*np.hanning(len(x))[:, None], axis=0))**2
    freq = np.fft.rfftfreq(len(x), CONTROL_DT)
    return float(spectrum[freq >= 8].sum() / max(float(spectrum.sum()), 1e-20))


def upper_metrics(trace, motion):
    state = torch.as_tensor(trace['state'], device=motion.q.device)
    q, qd, hand, phase, tilt = upper_series(unflatten_state(state), motion.kin)
    f = torch.as_tensor(trace['frame'], device=motion.q.device)
    ref = motion.at_time(motion.control_time[f])
    r, rd, rh, _, rt = upper_series(tuple(ref[key] for key in ('root_p', 'root_R', 'root_v', 'root_w', 'q', 'qd')), motion.kin)
    error, speed_error = q-r, qd-rd
    leg_swing = ~ref['contact'].detach().cpu().numpy()
    coupling = [float(phase[leg_swing[:, side], side].mean()-phase[~leg_swing[:, side], side].mean())
                if leg_swing[:, side].any() and (~leg_swing[:, side]).any() else 0. for side in range(2)]
    return dict(arm_rmse_rad=rmse(error[:, 1:]), waist_rmse_rad=rmse(error[:, 0]),
                arm_speed_rmse_rad_s=rmse(speed_error[:, 1:]), waist_speed_rmse_rad_s=rmse(speed_error[:, 0]),
                wrist_rmse_m=np.sqrt(np.square(hand-rh).sum(-1).mean(0)).tolist(),
                arm_speed_rms=rmse(qd[:, 1:]), arm_acceleration_rms=rmse(np.diff(qd[:, 1:], axis=0)/CONTROL_DT)
                if len(qd) > 1 else 0., arm_high_frequency=high_frequency(qd[:, 1:]),
                root_roll_pitch_rmse_deg=np.degrees(np.sqrt(np.square(np.arctan2(np.sin(tilt-rt), np.cos(tilt-rt))).mean(0))).tolist(),
                arm_fore_aft_mm=(np.ptp(phase, axis=0)*1000.).tolist(),
                hand_fore_aft_swing_minus_stance_m=coupling)


@torch.no_grad()
def scored_upper_metrics(trace, mdp):
    motion = mdp.motion
    metrics = upper_metrics(trace, motion)
    # 与真实 Reward 使用同一帧、动作与步后状态。
    frame = torch.as_tensor(trace['frame'][1:], device=motion.q.device)
    state = unflatten_state(torch.as_tensor(trace['state'][1:], device=motion.q.device))
    action, previous = (torch.as_tensor(trace[k], device=motion.q.device) for k in ('action','previous_action'))
    anchor = motion.anchor(torch.zeros_like(frame))
    history = {}
    if mdp.action_history_enabled:
        history = dict(a_prev2=torch.as_tensor(trace['previous_action2'], device=motion.q.device),
                       pd_target=torch.as_tensor(trace['target'], device=motion.q.device))
    _, terms = mdp.reward(state, frame, action, previous, True, command=anchor, ep_step=frame, **history)
    metrics['body_prior_contribution'] = float(terms['body_prior_contribution'].mean())
    metrics['body_prior_unweighted'] = float(terms['body_prior'].mean())
    return metrics


def reference_review(reference, archive, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import mujoco

    from ..kinematics import RobotKinematics
    from .contact_diagnostics import RobotGeometry
    from .m4_reference import F2MotionReference

    kin = RobotKinematics('cpu')
    motion = F2MotionReference(reference, kin)
    ref = motion.at_time(motion.control_time)
    time = motion.control_time.numpy()
    q, qd, hand, phase, _ = upper_series(tuple(ref[key] for key in ('root_p', 'root_R', 'root_v', 'root_w', 'q', 'qd')), kin)
    lo, hi = kin.q_lo[UPPER].numpy(), kin.q_hi[UPPER].numpy()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(archive, allow_pickle=False) as record:
        trials = [{key: record[f'm43_{label}_s{seed}_{key}'] for key in ('state', 'frame')}
                  for label in ('best', 'final') for seed in record['seeds']]
    data = np.stack([t['state'] for t in trials]).reshape(2, 3, len(time), -1)
    accel_time = time[1:]-CONTROL_DT/2
    fig, axes = plt.subplots(len(UPPER), 3, figsize=(11, 14), sharex='col', layout='constrained')
    for j, name, row in zip(range(len(UPPER)), (JOINT_NAMES[i] for i in UPPER), axes):
        for col, grid in enumerate(row):
            for k, traces in enumerate(data):
                series = traces[:, :, (34 if col else 13)+UPPER[j]]
                if col == 2:
                    series = np.diff(series, axis=1)/CONTROL_DT
                grid.plot(time if col < 2 else accel_time, series.mean(0), color=('tab:orange', 'tab:green')[k],
                          lw=.8, label=('M4.3 4500', 'M4.3 5000')[k])
            expected = (q[:, j], qd[:, j], np.diff(qd[:, j])/CONTROL_DT)[col]
            grid.plot(time if col < 2 else accel_time, expected, color='tab:blue', lw=1.2, label='F2')
            for t in motion.touchdown_time.tolist():
                grid.axvline(t, color='.7', lw=.6)
            grid.grid(alpha=.15)
        row[0].set_ylabel(name.replace('_joint', ''))
        v = qd[:, j]
        print(name, 'q', float(q[:, j].min()), float(q[:, j].max()), 'v/a', float(abs(v).max()),
              float(abs(np.diff(v)).max()/CONTROL_DT), 'margin', float(np.minimum(q[:, j]-lo[j], hi[j]-q[:, j]).min()),
              'URDF speed', actuator_parameters()[name]['velocity'], 'hf', high_frequency(v[:, None]))
    for col, label in enumerate(('Position (rad)', 'Velocity (rad/s)', 'Acceleration (rad/s²)')):
        axes[0, col].set_title(label)
        axes[-1, col].set_xlabel('Time (s)')
    axes[0, 0].legend(fontsize=7, ncol=3)
    fig.savefig(output/'arm_waist.png', dpi=145)
    plt.close(fig)

    # 使用相同机身相对相机检查四次落脚、3.32 秒及起始姿态。
    geom = RobotGeometry(SimpleNamespace(sizes=()))
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 140, -10, .95
    option = mujoco.MjvOption()
    option.geomgroup[0], option.geomgroup[1] = 0, 1
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), layout='constrained')
    with mujoco.Renderer(geom.model, width=420, height=310) as renderer:
        for sec, ax in zip((0, 1.6, 2.6, 3.32, 3.6, 4.6), axes.flat):
            i = round(sec/CONTROL_DT)
            p = ref['root_p'][i].numpy()
            geom.set_robot(p, mat_to_quat(ref['root_R'][i]).numpy(), ref['q'][i].numpy())
            camera.lookat[:] = p + [0., 0., .12]
            renderer.update_scene(geom.data, camera=camera, scene_option=option)
            ax.imshow(renderer.render())
            ax.set_title(f'F2 | {sec:.2f} s')
            ax.axis('off')
    fig.savefig(output/'f2_upper_body.png', dpi=145)
    plt.close(fig)
    print('手部相对躯干前后幅度 mm', np.ptp(phase, axis=0)*1000)
    print('左右手前后 Pearson', np.corrcoef(phase.T)[0, 1])
    for side in range(2):
        swing = ~ref['contact'][:, side].numpy()
        print('同侧脚摆动/支撑时手的前后位移 m', side, phase[swing, side].mean(), phase[~swing, side].mean())
    for label, subset in zip((4500, 5000), (trials[:3], trials[3:])):
        rows = [upper_metrics(t, motion) for t in subset]
        print(label, {key: np.mean([r[key] for r in rows], axis=0).tolist() for key in rows[0]})


def final_visuals(record, reference, output, run_dir, seed, *, render=True):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from ..kinematics import RobotKinematics
    from .m42v3_rollout import render_comparison
    from .m4_reference import F2MotionReference

    motion = F2MotionReference(reference, RobotKinematics('cpu'))
    ref = motion.at_time(motion.control_time)
    q, _, hand, phase, _ = upper_series(tuple(ref[key] for key in ('root_p','root_R','root_v','root_w','q','qd')), motion.kin)
    base, candidate = (f'm43_best_s{seed}_', f'm45_best_s{seed}_')
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True, layout='constrained')
    time = motion.control_time.numpy()
    for side in range(2):
        axes[0,side].plot(time, hand[:,side,0], label='F2 wrist X', color='.4')
        axes[2,side].plot(time, phase[:,side]*1000, label='F2', color='.4')
    for prefix, label in ((base,'M4.3 Best'),(candidate,'M4.5 Best')):
        t = record[prefix+'time']; f = record[prefix+'frame']
        x, _, p, ph, _ = upper_series(unflatten_state(torch.from_numpy(record[prefix+'state'])), motion.kin)
        for side in range(2):
            axes[0,side].plot(t,p[:,side,0],label=label)
            axes[1,side].plot(t,np.sqrt(np.square(x[:,1+4*side:5+4*side]-q[f,1+4*side:5+4*side]).mean(-1)),label=label)
            axes[2,side].plot(t,ph[:,side]*1000,label=label)
    for side in range(2):
        axes[0,side].set_title(('Left wrist','Right wrist')[side])
        for ax, label in zip(axes[:,side], ('World X (m)','Arm joint error (rad)','Body-relative fore-aft (mm)')):
            ax.set_ylabel(label)
            ax.grid(alpha=.2)
            for lift, land, foot in zip(motion.liftoff_time,motion.touchdown_time,motion.touchdown_side):
                if foot == side:
                    ax.axvspan(float(lift),float(land),alpha=.09,color='black')
            ax.axvline(3.32,color='.45',ls='--',lw=.8)
        axes[-1,side].set_xlabel('Time (s)')
    axes[0,0].legend(fontsize=8)
    fig.savefig(output/'wrist_and_arm_error.png',dpi=145)
    plt.close(fig)

    summary = json.loads((output/'results.json').read_text())
    fig, axes = plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for directory, label in ((Path('outputs/m4_3'),'M4.3'), (run_dir,'M4.5')):
        with (directory/'logs/train.csv').open() as stream:
            rows = {int(row['iteration']):row for row in csv.DictReader(stream)}
        for ax, key, title in zip(axes.flat, ('wm_loss','wm_foot_rmse_8','foot_rmse_m','torque_saturation_ratio'),
                                  ('WM loss','WM 8-step foot RMSE (m)','Training foot RMSE (m)','Actuator saturation')):
            ids = [i for i in sorted(rows) if rows[i].get(key)]
            ax.plot(ids,[float(rows[i][key]) for i in ids], label=label)
            ax.set_title(title)
            ax.set_xlabel('Iteration')
            ax.grid(alpha=.2)
    axes[0,0].legend()
    fig.savefig(output/'training_curves.png',dpi=145)
    plt.close(fig)
    if render:
        chapters = [(base, f'M4.3 Best | {summary["comparison"]["cases"]["m43_best"]["iteration"]}'),
                    (candidate, f'M4.5 Best | {summary["best_iteration"]}')]
        render_comparison(output/'rollouts.npz',reference,output/'upper_body_comparison.mp4',chapters=chapters,
                          upper_pairs=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m4_reference', required=True, type=Path)
    parser.add_argument('--archive', type=Path, default=Path('outputs/m4_4_unclipped/evaluation/rollouts.npz'))
    parser.add_argument('--out', type=Path, default=Path('outputs/m4_5/reference'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    reference_review(args.m4_reference, args.archive, args.out)
