"""用同一 CUDA Newton 协议复评 M4.8 的 Best、Last 与 M4.3 基线。"""

from argparse import ArgumentParser, Namespace
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

if __name__ == '__main__':
    os.environ['SUPERMIMIC_ROBOT'] = 'roban'

import numpy as np
import torch

from .m43_evaluation import INVALID_TERMINATIONS, m48_trial_metrics
from .m44_evaluation import DEFAULT_SEEDS, PERTURB_SEEDS, scene, trial
from .m45_upper import upper_series
from .m47_control import upper_joint_indices
from .m47_startup_audit import record_substeps
from ..mdp import unflatten_state
from ..motion import CONTROL_DT
from ..roban import JOINT_NAMES


GROUPS = {'a': 'a_startup_only', 'b': 'b_smoothness_only', 'c': 'c_startup_smoothness'}
CHECKPOINTS = {'m43_best': ('outputs/m4_3/checkpoints/best.pt', 4500)}
CHECKPOINTS.update({f'{group}_{tag}': (f'outputs/m4_8/{directory}/checkpoints/{tag}.pt', iteration)
                    for group, directory in GROUPS.items()
                    for tag, iteration in (('best', 4500), ('last', 5000))})
SEEDS = tuple(DEFAULT_SEEDS) + tuple(PERTURB_SEEDS)
UPPER = upper_joint_indices(JOINT_NAMES)
ELBOW = JOINT_NAMES.index('zarm_r4_joint')
PROBE_STEPS = 12
FIELDS = ('state', 'frame', 'raw_action', 'effective_action', 'target',
          'foot', 'force', 'contact', 'planned_contact', 'actuator_torque')


def load_archive(path):
    if not path.exists():
        return {'cases': {}}, {}
    with np.load(path, allow_pickle=False) as package:
        return json.loads(package['metadata'].item()), {key: package[key] for key in package.files if key != 'metadata'}


def save_archive(path, info, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as scratch:
        tmp = Path(scratch) / path.name
        np.savez_compressed(tmp, metadata=np.asarray(json.dumps(info, ensure_ascii=False, allow_nan=False)), **arrays)
        tmp.replace(path)


def evaluate(args, path, names, phase):
    info, arrays = load_archive(path)
    info['seeds'] = dict(nominal=DEFAULT_SEEDS, perturb=PERTURB_SEEDS)
    info['control_dt'] = CONTROL_DT
    info['upper_joints'] = [JOINT_NAMES[j] for j in UPPER]
    info['probe_channels'] = ('q', 'qd', 'qacc', 'torque', 'joint_limit_force')
    starts = {}
    for case in info['cases'].values():
        for seed, row in case.get('full', {}).items():
            starts.setdefault(int(seed), arrays[row['prefix']+'state'][0])
    probe = phase == 'probe'
    for name in names:
        source, expected_iteration = CHECKPOINTS[name]
        entry = info['cases'].setdefault(name, dict(checkpoint=source, iteration=expected_iteration))
        rows = entry.setdefault(phase, {})
        if all(str(seed) in rows for seed in SEEDS):
            continue
        started = perf_counter()
        with scene(args, Path(source), expected_iteration=expected_iteration, capture_substeps=probe) as (
                env, actor, contacts, solver, _, physics):
            physics = json.loads(json.dumps(physics))
            if physics != info.setdefault('physics', physics):
                raise ValueError('检查点的 Newton 物理配置不同')
            mdp = env.mdp
            info['physics_dt'] = env.physics_dt
            if 'joint_limits' not in arrays:
                arrays['joint_limits'] = torch.stack((mdp.kin.q_lo, mdp.kin.q_hi)).cpu().numpy()
                arrays['effort'] = mdp.effort.cpu().numpy()
            entry.update(startup_steps=mdp.startup_steps, smoothness_weight=mdp.cfg.upper_action_smoothness_weight,
                         body_prior_weight=mdp.cfg.body_prior_weight)
            if probe:
                env.max_episode_steps = PROBE_STEPS
            for seed in SEEDS:
                if str(seed) in rows:
                    continue
                perturb = seed in PERTURB_SEEDS
                if probe:
                    readings = []
                    with record_substeps(solver, readings):
                        trace, row = trial(env, actor, contacts, solver, seed, perturb=perturb)
                else:
                    trace, row = trial(env, actor, contacts, solver, seed, perturb=perturb)
                initial = trace['state'][0]
                np.testing.assert_allclose(initial, starts.setdefault(seed, initial), rtol=0, atol=1e-7)
                prefix = f'{name}_{phase}_s{seed}_'
                if not len(trace['action']) and row['termination'] not in INVALID_TERMINATIONS:
                    raise RuntimeError(f'试次没有控制转移：{name} {seed}')
                if 'effective_action' not in trace:
                    target = torch.as_tensor(trace['target'], device=env.device)
                    trace['effective_action'] = mdp.effective_action(target).cpu().numpy()
                if probe:
                    sample = torch.stack(readings).cpu().numpy() if readings else np.empty((0, 5, len(UPPER)))
                    substeps = round(CONTROL_DT / env.physics_dt)
                    if row['termination'] not in INVALID_TERMINATIONS and len(sample) != row['steps'] * substeps:
                        raise RuntimeError(f'物理子步数量不匹配：{name} {seed}')
                    arrays.update({prefix+k: trace[k] for k in ('state', 'frame', 'target')})
                    arrays[prefix+'substeps'] = sample
                    rows[str(seed)] = dict(prefix=prefix, steps=row['steps'], termination=row['termination'])
                else:
                    m48_trial_metrics(trace, row, mdp)
                    state = unflatten_state(torch.as_tensor(trace['state'], device=env.device))
                    _, _, _, hands, _ = upper_series(state, env.motion.kin)
                    arrays.update({prefix+k: trace[k] for k in FIELDS})
                    arrays[prefix+'hand_fore_aft'] = hands
                    row['prefix'] = prefix
                    rows[str(seed)] = row
                print(name, phase, seed, row['steps'], row['termination'], row['matched_touchdowns'], flush=True)
                if row['termination'] in ('fatal_dynamics', 'nonfinite'):
                    save_archive(path, info, arrays)
                    raise FloatingPointError(f'冻结评估停止：{name} {seed} {row["termination"]}')
        entry[phase+'_wall_s'] = perf_counter() - started
        save_archive(path, info, arrays)


def means(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def power_ratio(speed):
    # 去均值后作 Hann 加窗，按窗能量归一，并对单边谱的非端点进行补偿。
    n = len(speed)
    win = np.hanning(n)
    spectrum = np.abs(np.fft.rfft((speed - speed.mean(axis=0)) * win[:, None], axis=0))**2
    spectrum /= n * np.square(win).sum()
    spectrum[1:(-1 if n % 2 == 0 else None)] *= 2
    total = spectrum.sum()
    high = spectrum[np.fft.rfftfreq(n, CONTROL_DT) >= 8.].sum()
    return float(high / speed.shape[1]), float(high / total) if total else None


def movement_metrics(windows, limits, start):
    n = len(JOINT_NAMES)
    stops = [end for _, end in windows]
    if not stops or min(stops) < start + 4:
        return {}
    q = [item['state'][start:end+1, 13:13+n][:, UPPER].astype(np.float64) for item, end in windows]
    qd = [item['state'][start:end+1, 13+n:][:, UPPER].astype(np.float64) for item, end in windows]
    speed = np.concatenate(qd)
    acc = np.concatenate([np.diff(x, axis=0)/CONTROL_DT for x in qd])
    jerk = np.concatenate([np.diff(x, n=2, axis=0)/CONTROL_DT**2 for x in qd])
    high = [power_ratio(x[:, 1:]) for x in qd]
    rms = lambda x: float(np.sqrt(np.square(x).mean()))
    result = dict(velocity_rms=rms(speed), acceleration_rms=rms(acc), jerk_rms=rms(jerk),
                  acceleration_p95=float(np.quantile(np.abs(acc), .95)),
                  acceleration_p99=float(np.quantile(np.abs(acc), .99)),
                  acceleration_max=float(np.abs(acc).max()),
                  arm_hf_power=float(np.mean([x[0] for x in high])),
                  arm_hf_fraction=means([dict(ratio=x[1]) for x in high], 'ratio'))
    for key in ('raw_action', 'effective_action', 'target'):
        x = np.concatenate([np.diff(item[key][start:end, UPPER].astype(np.float64), n=2, axis=0)
                            for item, end in windows])
        result[key+'_diff2_rms'] = rms(x)

    pooled = np.concatenate(q)
    lo, hi = limits[:, UPPER]
    result['p5_p95_joint_angles'] = np.quantile(pooled, (.05, .95), axis=0).tolist()
    result['joint_near_or_beyond_limit_ratio'] = (np.minimum(pooled-lo, hi-pooled) <= .02).mean(axis=0).tolist()
    hands = [item['hand_fore_aft'][start:end+1] for item, end in windows]
    result['hand_fore_aft_span_m'] = np.mean([np.diff(np.quantile(x, (.05, .95), axis=0), axis=0)[0]
                                              for x in hands], axis=0).tolist()

    def correlation(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-8 and b.std() > 1e-8 else None

    result['opposite_hand_correlation'] = means([dict(r=correlation(x[:, 0], x[:, 1])) for x in hands], 'r')
    masks = [~item['planned_contact'][start:end+1] for item, end in windows]
    result['same_side_planned_swing_correlation'] = [means([dict(r=correlation(x[:, side], m[:, side]))
                                                           for x,m in zip(hands, masks)], 'r') for side in (0, 1)]
    return result


def probe_signals(arrays, row, dt):
    p = row['prefix']
    raw = arrays[p+'substeps'].astype(np.float64)
    i = UPPER.index(ELBOW)
    qd0 = float(arrays[p+'state'][0, 13+len(JOINT_NAMES)+ELBOW])
    acc = np.diff(np.r_[qd0, raw[:, 1, i]]) / dt
    return raw, i, acc


def probe_metrics(arrays, row, dt):
    raw, i, acc = probe_signals(arrays, row, dt)
    p = row['prefix']
    first_period = round(CONTROL_DT / dt)
    startup = round(.20 / dt)
    # 早终止或故障轨迹仅保留原读数，峰值对照使用完整的共同观测窗。
    if len(acc) < round(.24 / dt) or row['termination'] in INVALID_TERMINATIONS:
        return {}
    k = int(np.abs(acc[:startup]).argmax())
    effort = float(arrays['effort'][ELBOW])
    return dict(first_error=float(arrays[p+'target'][0, ELBOW] - arrays[p+'state'][0, 13+ELBOW]),
                first_velocity=float(raw[first_period-1, 1, i]),
                first_avg_acc=float(acc[:first_period].mean()),
                first_torque_peak=float(np.abs(raw[:first_period, 3, i]).max()),
                first_torque_saturation=int((np.abs(raw[:first_period, 3, i]) >= .99 * effort).sum()),
                peak=float(abs(acc[k])), peak_time_s=(k+.5)*dt,
                direct_peak=float(np.abs(raw[:startup, 2, i]).max()),
                peak_at_release=float(np.abs(acc[startup:]).max()),
                startup_torque_peak=float(np.abs(raw[:startup, 3, i]).max()),
                startup_torque_saturation=float((np.abs(raw[:startup, 3, i]) >= .99 * effort).mean()),
                joint_stop_force_peak=float(np.abs(raw[:startup, 4, i]).max()))


def comparison(info, data):
    cases = info['cases']
    result = {}
    for split, seeds in info['seeds'].items():
        sections = {name: {int(k): v for k,v in entry['full'].items()
                           if int(k) in seeds and v['termination'] not in INVALID_TERMINATIONS}
                    for name,entry in cases.items()}
        shared = {seed: min(rows[seed]['steps'] for rows in sections.values())
                  for seed in seeds if all(seed in rows for rows in sections.values())}
        hosts = dict(common_end_s={seed: end*CONTROL_DT for seed,end in shared.items()}, cases={})
        for name, rows in sections.items():
            run = cases[name]
            row_data = {seed: {k: data[row['prefix']+k] for k in (*FIELDS, 'hand_fore_aft')}
                        for seed,row in rows.items()}
            complete = [seed for seed,row in rows.items()
                        if row['survived_full_reference'] and row['consecutive_planned_touchdowns'] == len(row['footsteps'])]
            contacts = [event for row in rows.values() for event in row['footsteps']]
            fields = ('first_swing', 'first_touchdown', 'survived_full_reference')
            gait = {key: [seed for seed,row in rows.items() if row[key]] for key in fields}
            gait.update(full=complete, valid_seeds=list(rows),
                        first_support_s=means([row['footsteps'][0] for row in rows.values()], 'continuous_support_s'),
                        mean_support_s=means(contacts, 'continuous_support_s'))
            for key in ('matched_touchdowns', 'consecutive_planned_touchdowns', 'foot_position_rmse_m',
                        'root_xy_rmse_m', 'contact_agreement', 'episode_length_s', 'torque_saturation_ratio',
                        'clamp_ratio', 'left_ankle_clamp_ratio'):
                gait[key] = means(rows.values(), key)
            gait['joint_limit_max_rad'] = max((row['joint_limit_excess_rad'] for row in rows.values()
                                               if 'joint_limit_excess_rad' in row), default=None)
            probes = {int(seed): probe_metrics(data, row, info['physics_dt']) for seed,row in run['probe'].items()
                      if int(seed) in seeds}
            hosts['cases'][name] = dict(gait=gait, probes=probes,
                peak_startup_rad_s2=max((p['peak'] for p in probes.values() if p), default=None),
                whole=movement_metrics([(row_data[s], row['steps']) for s,row in rows.items()], data['joint_limits'], 0),
                steady=movement_metrics([(row_data[s], end) for s,end in shared.items()], data['joint_limits'],
                                        round(.20 / CONTROL_DT)))
        result[split] = hosts
    return result


def startup_figure(info, data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dt = info['physics_dt']
    names = ['m43_best', 'a_best', 'b_best', 'c_best']
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.1), sharex=True, sharey='row', layout='constrained')
    fig.suptitle('Roban right elbow · M4.8', fontsize=13)
    for column, seed in enumerate(('1001', '1101')):
        axes[0, column].set_title('Nominal · 1001' if column == 0 else 'Perturbed · 1101')
        for name,label,color in zip(names, ('M4.3 Best', 'M4.8 A Best', 'M4.8 B Best', 'M4.8 C Best'),
                                    ('#30343b', '#2563a6', '#b85226', '#138066')):
            row = info['cases'][name]['probe'][seed]
            raw, i, acc = probe_signals(data, row, dt)
            time = (np.arange(len(acc))+.5)*dt*1000
            axes[0, column].plot(time, acc, color=color, label=label, linewidth=1.25)
            axes[1, column].plot(time, raw[:, 3, i], color=color, linewidth=1.25)
        axes[1, column].set_xlabel('Time since Reset (ms)')
        for axis in axes[:, column]:
            axis.axvline(200, color='0.6', linestyle=':', linewidth=1)
            axis.grid(alpha=.2)
    axes[0, 0].set_ylabel('Elbow acceleration (rad/s²)')
    axes[1, 0].set_ylabel('Actuator torque (N·m)')
    axes[0, 1].legend(fontsize=8, ncols=2)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--m4_reference', type=Path)
    parser.add_argument('--out', type=Path, default=Path('outputs/m4_8/cuda_evaluation.npz'))
    parser.add_argument('--cases', nargs='+', choices=CHECKPOINTS, default=list(CHECKPOINTS))
    parser.add_argument('--phase', choices=('all', 'full', 'probe'), default='all')
    parser.add_argument('--summarize', action='store_true', help='从归档复算配对指标')
    parser.add_argument('--plot', type=Path, help='保存一张 400 Hz 启动对照图')
    cli = parser.parse_args()
    torch.set_num_threads(2)
    if cli.summarize:
        info, arrays = load_archive(cli.out)
        if not info['cases'] or any(set(map(str, SEEDS)) != set(run.get(key, {}))
                                    for run in info['cases'].values() for key in ('full', 'probe')):
            parser.error('完整闭环和物理子步归档齐备后才能复算')
        info['comparison'] = comparison(info, arrays)
        save_archive(cli.out, info, arrays)
        if cli.plot:
            startup_figure(info, arrays, cli.plot)
        return
    if cli.m4_reference is None:
        parser.error('CUDA 评估需要 --m4_reference')
    args = Namespace(m4_reference=cli.m4_reference, device='cuda:0', visualizer=None)
    for phase in (('full', 'probe') if cli.phase == 'all' else (cli.phase,)):
        evaluate(args, cli.out, cli.cases, phase)


if __name__ == '__main__':
    main()
