"""M4.11 的配对 CUDA Newton 轨迹、留出扰动与最终对比。"""

from argparse import ArgumentParser, Namespace
import json
import os
from pathlib import Path

if __name__ == '__main__':
    os.environ['SUPERMIMIC_ROBOT'] = 'roban'
    os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import torch

from ..kinematics import RobotKinematics
from ..mdp import unflatten_state
from ..motion import CONTROL_DT
from ..roban import ELBOW_JOINT_NAMES, JOINT_NAMES
from .m410_evaluation import mean_metrics
from .m410_quality import local_upper_points, whole_body_metrics
from .m410_preflight import replay_case
from .m411_hand import torso_local_points
from .m43_evaluation import DEFAULT_SEEDS, INVALID_TERMINATIONS
from .m43_wm_audit import task_for
from .m44_evaluation import PERTURB_LIMITS, PERTURB_SEEDS, scene, trial
from .m45_upper import ARMS, UPPER, upper_series
from .m48_evaluation import FIELDS, load_archive, movement_metrics, power_ratio, save_archive


ROOT = Path('outputs/m4_11')
HISTORY = Path('outputs/m4_10/cuda_evaluation.npz')
HOLDOUT = (1201, 1202, 1203, 1204, 1205, 1206)
SEEDS = dict(nominal=DEFAULT_SEEDS, perturb=PERTURB_SEEDS, holdout=HOLDOUT)
CASES = {
    'A10000': ('outputs/m4_10/a_root_only/checkpoints/iter_10000.pt', 10000),
    'C9000': ('outputs/m4_10/c_root_huber/checkpoints/iter_9000.pt', 9000),
    'A250': ('outputs/m4_11/a_continue_baseline/checkpoints/iter_0250.pt', 250),
    'A1000': ('outputs/m4_11/a_continue_baseline/checkpoints/iter_1000.pt', 1000),
    'A2000': ('outputs/m4_11/a_continue_baseline/checkpoints/iter_2000.pt', 2000),
    'B1250': ('outputs/m4_11/b_hand_huber/checkpoints/iter_1250.pt', 1250),
    'B1750': ('outputs/m4_11/b_hand_huber/checkpoints/iter_1750.pt', 1750),
    'B2000': ('outputs/m4_11/b_hand_huber/checkpoints/iter_2000.pt', 2000),
}
HELD_OUT_CASES = ('B1250', 'B1750', 'B2000')
CAPTURE_FIELDS = (*FIELDS, 'foot_R', 'foot_velocity', 'clamp_mask', 'joint_limit_force', 'reward')
STEADY_START = round(.20 / CONTROL_DT)


def seeds_for(case):
    return DEFAULT_SEEDS + PERTURB_SEEDS + (HOLDOUT if case in HELD_OUT_CASES else ())


def capture(reference, output):
    args = Namespace(m4_reference=reference, device='cuda:0', visualizer=None)
    historical, historical_arrays = load_archive(HISTORY)
    old = historical['cases']['c_9000']['full']
    resets = {int(seed): historical_arrays[row['prefix'] + 'state'][0] for seed, row in old.items()}
    del historical_arrays
    info, arrays = load_archive(output)
    protocol = dict(physics=historical['physics'], control_dt=CONTROL_DT,
                    seeds={name: list(seeds) for name, seeds in SEEDS.items()},
                    perturb_limits=PERTURB_LIMITS, minimum_root_lift_m=.006,
                    joint_names=list(JOINT_NAMES), gpu=torch.cuda.get_device_name())
    if 'protocol' in info and info['protocol'] != protocol:
        raise ValueError('已有归档的独立复评协议或 GPU 不一致')
    info['protocol'] = protocol
    for entry in info['cases'].values():
        for seed, row in entry['full'].items():
            initial = arrays[row['prefix'] + 'state'][0]
            np.testing.assert_allclose(initial, resets.setdefault(int(seed), initial), atol=1.e-7, rtol=0)

    for name, (path, iteration) in CASES.items():
        entry = info['cases'].setdefault(name, dict(checkpoint=path, iteration=iteration, full={}))
        rows = entry['full']
        pending = [seed for seed in seeds_for(name) if str(seed) not in rows]
        while pending:
            with scene(args, Path(path), expected_iteration=iteration) as (env, actor, contacts, solver, _, physics):
                if json.loads(json.dumps(physics)) != protocol['physics']:
                    raise ValueError(f'{name} 的物理契约与 M4.10 归档不一致')
                if 'joint_limits' not in arrays:
                    kin = env.mdp.kin
                    arrays['joint_limits'] = torch.stack((kin.q_lo, kin.q_hi)).cpu().numpy()
                    arrays['effort'] = env.mdp.effort.cpu().numpy()
                while pending:
                    seed = pending.pop(0)
                    trace, row = trial(env, actor, contacts, solver, seed, perturb=seed not in DEFAULT_SEEDS)
                    initial = trace['state'][0]
                    np.testing.assert_allclose(initial, resets.setdefault(seed, initial), atol=1.e-7, rtol=0)
                    np.testing.assert_array_equal(trace['frame'], np.arange(len(trace['state'])))
                    prefix = f'{name}_s{seed}_'
                    arrays.update({prefix + key: trace[key] for key in CAPTURE_FIELDS if key in trace})
                    rows[str(seed)] = dict(row, prefix=prefix)
                    print(f'{name} seed={seed} | {row["steps"]} 步 | '
                          f'连续落脚={row["consecutive_planned_touchdowns"]}/4 | {row["termination"]}', flush=True)
                    if row['termination'] in ('fatal_dynamics', 'nonfinite'):
                        # 保存已观测故障，剩余种子使用新建场景。
                        break
                save_archive(output, info, arrays)
            torch.cuda.empty_cache()


def read_trace(arrays, row):
    prefix = row['prefix']
    return {key: arrays[prefix + key] for key in CAPTURE_FIELDS if prefix + key in arrays}


def completed(row):
    return row['survived_full_reference'] and row['consecutive_planned_touchdowns'] == len(row['footsteps'])


def wrist_positions(flat_state, kin):
    state = unflatten_state(torch.as_tensor(flat_state))
    world = upper_series(state, kin)[2]
    local = torso_local_points(*state[:2], torch.from_numpy(world)).numpy()
    return local, world


def wrist_jitter(points, *, dt=CONTROL_DT):
    from scipy.signal import butter, sosfiltfilt

    extent = round(.20 / dt)
    if len(points) < max(25, 2 * extent + 2):
        return None
    points = points.astype(np.float64)
    # Hann 单边谱按窗能量归一，三轴功率求和后再开方。
    samples = len(points)
    win = np.hanning(samples)
    spectrum = np.abs(np.fft.rfft((points - points.mean(0)) * win[:, None, None], axis=0))**2
    spectrum /= samples * np.square(win).sum()
    spectrum[1:(-1 if samples % 2 == 0 else None)] *= 2
    frequencies = np.fft.rfftfreq(samples, dt)
    band = spectrum[frequencies >= 8.].sum(axis=(0, 2))
    # 峰值另用零相位高通估计，舍去两端各 0.2 秒的滤波边界。
    filt = butter(4, 8., btype='highpass', fs=1. / dt, output='sos')
    high = sosfiltfilt(filt, points, axis=0)
    amplitude = np.linalg.norm(high[extent:-extent], axis=-1)
    actual_acceleration = np.linalg.norm(np.diff(points, n=2, axis=0) / dt**2, axis=-1)
    result = dict(hf_rms_mm=(1000. * band**.5).tolist(),
                  highpass_p99_mm=(1000. * np.quantile(amplitude, .99, axis=0)).tolist(),
                  highpass_max_mm=(1000. * amplitude.max(0)).tolist(),
                  acceleration_rms_m_s2=np.sqrt(np.square(actual_acceleration).mean(0)).tolist())
    if frequencies[-1] > 25.:
        upper = spectrum[frequencies > 25.].sum(axis=(0, 2))
        result['hf_8_to_25_rms_mm'] = (1000. * np.sqrt(np.maximum(band - upper, 0.))).tolist()
        result['hf_above_25_rms_mm'] = (1000. * np.sqrt(upper)).tolist()
    return result


def control_limits(trace, limits, end):
    n = len(JOINT_NAMES)
    q = trace['state'][1:end+1, 13:13+n]
    excess = np.maximum(np.maximum(limits[0] - q, q - limits[1]), 0.)
    clamp = trace['clamp_mask'][:end]
    elbow = [JOINT_NAMES.index(name) for name in ELBOW_JOINT_NAMES]
    q_steady = trace['state'][STEADY_START:end+1, 13:13+n]
    near = np.minimum(q_steady[:, elbow] - limits[0, elbow], limits[1, elbow] - q_steady[:, elbow]) <= .02
    ankles = [JOINT_NAMES.index(name) for name in ('leg_l5_joint', 'leg_r5_joint')]
    return dict(clamp_ratio=float(clamp.mean()),
                ankle_clamp_ratio=clamp[:, ankles].mean(0).tolist(),
                elbow_near_limit_ratio=near.mean(0).tolist() if len(near) else None,
                angle_over_limit_frame_ratio=float((excess > 1.e-5).any(-1).mean()),
                joint_limit_max_excess_rad=float(excess.max()),
                limit_force_peak_nm=float(np.abs(trace['joint_limit_force'][1:end+1]).max()))


def motion_metrics(trace, mdp, limits, *, end):
    frames = trace['frame'][1:end+1]
    foot_delta = trace['foot'][1:end+1] - mdp.motion.foot_p[frames].numpy()
    tracking = whole_body_metrics(trace, mdp, end=end)
    tracking['foot_position_rmse_m'] = float(np.sqrt(np.square(foot_delta).sum(-1).mean()))
    control = movement_metrics([(trace, end)], limits, STEADY_START)
    # 单条试次保留肩肘等细分频谱，方便定位高速关节通道。
    speeds = trace['state'][STEADY_START:end+1, 13+len(JOINT_NAMES):][:, ARMS]
    if len(speeds) > 4:
        control['arm_hf_power_by_joint'] = [power_ratio(speeds[:, j:j+1])[0] for j in range(len(ARMS))]
    return dict(tracking=tracking, smoothness=control or None, limits=control_limits(trace, limits, end),
                hand_jitter={key: wrist_jitter(trace[key][STEADY_START:end+1]) for key in ('hand_local', 'hand_world')})


def cohort_metrics(names, seeds, rows, traces, mdp, limits):
    by_case = {name: {seed: rows[name][seed] for seed in seeds} for name in names}
    shared = {seed: min(rows[name][seed]['steps'] for name in names) for seed in seeds
              if all(rows[name][seed]['termination'] not in INVALID_TERMINATIONS and rows[name][seed]['steps']
                     for name in names)}
    comparison = {}
    for name, trials in by_case.items():
        valid = {seed: row for seed, row in trials.items() if row['termination'] not in INVALID_TERMINATIONS}
        success = [seed for seed, row in valid.items() if completed(row)]
        gait = dict(
            expected=len(seeds), valid=list(valid), four=success,
            first_swing=[seed for seed, row in valid.items() if row['first_swing']],
            first_touchdown=[seed for seed, row in valid.items() if row['first_touchdown']],
            survived=[seed for seed, row in valid.items() if row['survived_full_reference']],
            reasons={str(seed): row['termination'] for seed, row in trials.items()},
            means={key: mean_metrics([row.get(key) for row in valid.values()]) for key in
                   ('episode_length_s', 'foot_position_rmse_m', 'root_xy_rmse_m', 'root_position_rmse_m',
                    'contact_agreement', 'torque_saturation_ratio', 'common_reward_per_step')},
        )
        observed = [row['metrics'] for row in valid.values() if 'metrics' in row]
        matching = [rows[name][seed]['metrics'] if end == rows[name][seed]['steps'] else
                    motion_metrics(traces[name][seed], mdp, limits, end=end) for seed, end in shared.items()]
        comparison[name] = dict(gait=gait, observed=mean_metrics(observed),
                                completed=mean_metrics([valid[seed]['metrics'] for seed in success]),
                                common=mean_metrics(matching))
    return dict(common_end_s={seed: end * CONTROL_DT for seed, end in shared.items()}, cases=comparison)


def require_full_cases(info):
    if set(info['cases']) != set(CASES) or any(
            set(info['cases'][name]['full']) != set(map(str, seeds_for(name))) for name in CASES):
        raise ValueError('八份权重的配对闭环及三个 B 候选的留出试次需要全部齐备')


def probe_wrist(reference, output):
    from .m47_startup_audit import record_substeps

    info, arrays = load_archive(output)
    require_full_cases(info)
    args = Namespace(m4_reference=reference, device='cuda:0', visualizer=None)
    names = ('C9000', 'B1750')
    common = [seed for seed in PERTURB_SEEDS
              if all(completed(info['cases'][name]['full'][str(seed)]) for name in names)]
    seeds = (DEFAULT_SEEDS[0], *common[:1])
    probe = info.setdefault('wrist_substep_probe', dict(seeds=list(seeds), cases={}))
    if probe['seeds'] != list(seeds):
        raise ValueError('已归档的手部子步探针种子不一致')
    kin = RobotKinematics('cpu')
    for name in names:
        rows = probe['cases'].setdefault(name, {})
        pending = [seed for seed in seeds if str(seed) not in rows]
        if not pending:
            continue
        ckpt, iteration = CASES[name]
        with scene(args, Path(ckpt), expected_iteration=iteration, capture_substeps=True) as (
                env, actor, contacts, solver, _, physics):
            if json.loads(json.dumps(physics)) != info['protocol']['physics']:
                raise ValueError('手部子步探针的物理契约不同')
            substeps = round(CONTROL_DT / env.physics_dt)
            probe['dt'] = env.physics_dt
            for seed in pending:
                readings = []
                with record_substeps(solver, readings):
                    trace, row = trial(env, actor, contacts, solver, seed, perturb=seed not in DEFAULT_SEEDS)
                prefix = f'{name}_wrist_substeps_s{seed}_'
                entry = dict(prefix=prefix, steps=row['steps'], termination=row['termination'],
                             consecutive=row['consecutive_planned_touchdowns'])
                if row['steps'] and row['termination'] not in INVALID_TERMINATIONS:
                    angles = torch.stack(readings)[:, 0].cpu()
                    if len(angles) != row['steps'] * substeps:
                        raise RuntimeError('手部子步样本的数量与控制时钟不符')
                    q = env.motion.q[0].cpu().expand(len(angles), -1).clone()
                    q[:, UPPER] = angles
                    p = q.new_zeros((len(q), 3))
                    rotation = torch.eye(3).expand(len(q), -1, -1)
                    # 躯干坐标下的双手只由同一浮动根下的上身关节决定。
                    local = local_upper_points((p, rotation, p, p, q, torch.zeros_like(q)), kin)[:, 4:].numpy()
                    control_local, _ = wrist_positions(trace['state'][1:], kin)
                    np.testing.assert_allclose(local[substeps-1::substeps], control_local, atol=2.e-6, rtol=0)
                    start = round(.20 / env.physics_dt) - 1
                    entry['hand_local'] = wrist_jitter(local[start:], dt=env.physics_dt)
                    arrays[prefix + 'local'] = local
                rows[str(seed)] = entry
                print(f'{name} 400 Hz seed={seed} | {row["steps"]} 步 | {row["termination"]}', flush=True)
            save_archive(output, info, arrays)
        torch.cuda.empty_cache()


def finalize(reference, output):
    from ..mdp import flatten_state

    info, arrays = load_archive(output)
    require_full_cases(info)
    source = torch.load(CASES['C9000'][0], map_location='cpu', weights_only=False, mmap=True)
    task = task_for(source, RobotKinematics('cpu'), reference, 'cpu')
    _, motion, _, mdp = task
    if source['data_contract'] != motion.validation:
        raise ValueError('复算参考与 C9000 数据契约不一致')
    reference_values = motion.at_time(motion.control_time)
    state = tuple(reference_values[key] for key in ('root_p', 'root_R', 'root_v', 'root_w', 'q', 'qd'))
    ref_local, ref_world = wrist_positions(flatten_state(state), mdp.kin)
    limits = arrays['joint_limits']
    rows, traces = {}, {}
    for name, case in info['cases'].items():
        rows[name], traces[name] = {}, {}
        for key, row in case['full'].items():
            seed = int(key)
            rows[name][seed] = row
            if row['termination'] in INVALID_TERMINATIONS or not row['steps']:
                continue
            trace = read_trace(arrays, row)
            local, world = wrist_positions(trace['state'], mdp.kin)
            trace.update(hand_local=local, hand_world=world, hand_fore_aft=local[..., 0])
            traces[name][seed] = trace
            row['metrics'] = motion_metrics(trace, mdp, limits, end=row['steps'])
            replay = replay_case(task, arrays, row, seed not in DEFAULT_SEEDS)
            row['common_reward_per_step'] = float(replay['rewards']['combined'].mean())

    cohorts = {label: cohort_metrics(HELD_OUT_CASES if label == 'holdout' else CASES,
                                     seeds, rows, traces, mdp, limits) for label, seeds in SEEDS.items()}
    # 高频诊断对同一扰动种子使用双方成功轨迹，另保留终止前共同窗口。
    pair = cohort_metrics(('C9000', 'B1750'), PERTURB_SEEDS, rows, traces, mdp, limits)
    pair_complete = [seed for seed in PERTURB_SEEDS
                     if all(completed(rows[name][seed]) for name in ('C9000', 'B1750'))]
    info['summary'] = dict(cohorts=cohorts, c9000_b1750_pair=pair, pair_complete=pair_complete,
                           pair_completed_metrics={name: mean_metrics([rows[name][seed]['metrics']
                                                                       for seed in pair_complete])
                                                   for name in ('C9000', 'B1750')},
                           reference_hands=dict(
                               amplitude_mm=(1000. * np.diff(np.quantile(ref_local[1:, :, 0], (.05, .95), axis=0),
                                                                  axis=0)[0]).tolist(),
                               local=wrist_jitter(ref_local[STEADY_START:]),
                               world=wrist_jitter(ref_world[STEADY_START:])))
    save_archive(output, info, arrays)
    for label, cohort in cohorts.items():
        print(label, {name: (case['gait']['four'], case['gait']['survived'])
                      for name, case in cohort['cases'].items()}, flush=True)


def render_comparison(reference, archive, output):
    from .m410_visuals import render_video
    from .m4_reference import F2MotionReference

    info, arrays = load_archive(archive)
    if 'summary' not in info:
        raise ValueError('请先复算最终归档指标')
    motion = F2MotionReference(reference, RobotKinematics('cpu'), 'cpu')
    names = ('reference', 'C9000', 'B1250', 'B1750')
    labels = dict(C9000='M4.10-C9000 · 共同起点', B1250='M4.11-B1250 · 步态最佳',
                  B1750='M4.11-B1750 · 全身最佳')
    sections = [('M4.11 双手跟踪 · 正面', DEFAULT_SEEDS[0], names, 180),
                ('M4.11 双手跟踪 · 侧面', DEFAULT_SEEDS[0], names, 90)]

    def panels_from(names, seed, info, arrays, ref):
        panels = [dict(title='原始 F2 Reference', state=ref['state'], feet=ref['feet'], row=None)]
        for name in names[1:]:
            row = info['cases'][name]['full'][str(seed)]
            trace = read_trace(arrays, row)
            metric_row = dict(row, whole_body=row['metrics']['tracking'])
            panels.append(dict(title=labels[name], state=trace['state'][:, :13+len(JOINT_NAMES)],
                               feet=trace['foot'], row=metric_row))
        return panels

    render_video(info, arrays, motion, output, sections=sections, panels_from=panels_from,
                 show_hand_errors=True, hold_end_frames=13)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('capture', 'wrist-probe', 'summary', 'video'))
    parser.add_argument('--m4_reference', type=Path, required=True)
    parser.add_argument('--archive', type=Path, default=ROOT / 'cuda_evaluation.npz')
    parser.add_argument('--video', type=Path, default=ROOT / 'final_comparison.mp4')
    cli = parser.parse_args()
    torch.set_num_threads(2)
    if cli.mode == 'video':
        render_comparison(cli.m4_reference, cli.archive, cli.video)
    else:
        command = {'capture': capture, 'wrist-probe': probe_wrist, 'summary': finalize}[cli.mode]
        command(cli.m4_reference, cli.archive)


if __name__ == '__main__':
    main()
