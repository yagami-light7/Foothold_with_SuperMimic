"""M4.3 冻结 Newton 评估、步态优先权重选择和回放交付。"""

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

if __name__ == '__main__':
    os.environ['SUPERMIMIC_ROBOT'] = 'roban'
    os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import torch

from .m4_evaluation import (CONTACT_FORCE_N, MIN_SWING_LIFT_M, MIN_SWING_S,
                            TOUCHDOWN_WINDOW_S, TOUCHDOWN_XY_ERROR_M, _checkpoint_identity, _mean)
from .m42v3_rollout import EffectiveFriction, collect_trial, render_comparison, summarize
from .m47_control import upper_joint_indices
from ..motion import CONTROL_DT
from ..roban import ELBOW_JOINT_NAMES, FOOT_BODY_NAMES, JOINT_NAMES


INVALID_TERMINATIONS = ('epa_horizon', 'fatal_dynamics', 'nonfinite')
DEFAULT_SEEDS = (1001, 1002, 1003)


def _runs(mask):
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    return zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))


def swing_metrics(trace, event, side, landing_target, planned_height=None):
    """保持 M4.2 的成功定义，并对未匹配事件给出逐阶段的力和足位证据。"""
    t = trace['time']
    lift, land = event['planned_lift_s'], event['planned_touchdown_s']
    measured = trace['contact_measured']
    stance = trace['contact']
    scheduled = measured & (t >= lift - 1e-8) & (t < land - 1e-8)
    unloaded = ~stance[:, side]
    sole = trace['foot'][:, side]
    supported = stance[:, 1 - side]
    landing = measured & (np.abs(t - land) <= TOUCHDOWN_WINDOW_S + 1e-8)
    arrival_xy = np.linalg.norm(sole[:, :2] - landing_target[:2], axis=1)
    off_ground = list(_runs(scheduled & unloaded))
    off = max((b - a for a, b in off_ground), default=0) * CONTROL_DT
    single = max((b - a for a, b in _runs(scheduled & unloaded & supported)), default=0) * CONTROL_DT
    window = (t >= lift - TOUCHDOWN_WINDOW_S) & (t <= land + TOUCHDOWN_WINDOW_S)
    qualified_lifts = [(a, b) for a, b in _runs(window & unloaded)
                       if a > 0 and stance[a - 1, side] and (b - a) * CONTROL_DT + 1e-8 >= MIN_SWING_S]
    lifted = [(a, b) for a, b in qualified_lifts if (sole[a:b, 2] - sole[a - 1, 2]).max() >= MIN_SWING_LIFT_M]
    arrivals = [b for _, b in lifted if b < len(t) and stance[b, side]
                and abs(t[b] - land) <= TOUCHDOWN_WINDOW_S + 1e-8]
    positioned = [b for b in arrivals if arrival_xy[b] <= TOUCHDOWN_XY_ERROR_M]
    if event['actual_touchdown_s'] is not None:
        failure = None
    elif not scheduled.any():
        failure = 'before_swing'
    elif not qualified_lifts:
        failure = 'unloading_duration'
    elif not lifted:
        failure = 'lift_height'
    elif not arrivals:
        failure = 'landing_time'
    elif not positioned:
        failure = 'landing_position'
    else:
        # 与原检测器一致，触地后的连续承重仍需覆盖完整验证窗。
        failure = 'stable_support'
    achieved_lift = max((float(sole[a:b, 2].max() - sole[a - 1, 2]) for a, b in off_ground if a > 0),
                        default=0.)
    return dict(off_ground_s=off, single_support_s=single, lift_m=max(achieved_lift, 0.),
                max_height_m=float(sole[scheduled, 2].max()) if scheduled.any() else None,
                planned_max_height_m=planned_height,
                swing_position_rmse_m=float(np.sqrt(np.mean(trace['foot_target_error'][scheduled, side]**2)))
                if scheduled.any() else None,
                loaded_fraction=float(stance[scheduled, side].mean()) if scheduled.any() else None,
                double_off_fraction=float((~stance[scheduled].any(axis=-1)).mean()) if scheduled.any() else None,
                nearest_landing_xy_m=float(arrival_xy[landing].min()) if landing.any() else None,
                failure=failure)


def summarize_gait(trace, motion, mdp):
    row = summarize(trace, motion)
    sides = motion.touchdown_side.cpu().numpy()
    targets = motion.at_time(motion.touchdown_time)['foot_p'].cpu().numpy()
    full_time = np.arange(motion.num_frames) * CONTROL_DT
    full_feet = motion.at_time(full_time)['foot_p'].cpu().numpy()
    for k, (event, side) in enumerate(zip(row['footsteps'], sides)):
        plan_window = ((full_time >= event['planned_lift_s'] - 1e-8)
                       & (full_time < event['planned_touchdown_s'] - 1e-8))
        planned_height = float(full_feet[plan_window, side, 2].max())
        event.update(swing_metrics(trace, event, side, targets[k, side], planned_height))
    first = row['footsteps'][0]
    for key in ('off_ground_s', 'single_support_s', 'lift_m', 'max_height_m', 'planned_max_height_m',
                'swing_position_rmse_m', 'loaded_fraction', 'double_off_fraction', 'nearest_landing_xy_m'):
        row['first_' + key] = first[key]
    row['fault_detail'] = str(trace['fault_detail']) if 'fault_detail' in trace else None
    for key in ('clamp_ratio', 'joint_speed_max_rad_s', 'joint_speed_p99_rad_s', 'joint_limit_frames',
                'torque_saturation_ratio', 'action_rate_rms', 'loaded_foot_speed_m_s'):
        row.setdefault(key, None)
    if row['steps']:
        raw = trace['raw_action']
        abs_p95, abs_p99 = np.quantile(np.abs(raw), [.95, .99])
        q = trace['state'][1:, 13:13 + len(JOINT_NAMES)]
        qd = np.abs(trace['state'][1:, 13 + len(JOINT_NAMES):])
        q_lo, q_hi = (x.cpu().numpy() for x in (mdp.kin.q_lo, mdp.kin.q_hi))
        loaded = trace['contact'][1:]
        row.update(raw_action=dict(mean=float(raw.mean()), std=float(raw.std()), abs_p95=float(abs_p95),
                                   abs_p99=float(abs_p99), abs_max=float(np.abs(raw).max())),
                   action_rate_rms=float(np.sqrt(np.mean((trace['action'] - trace['previous_action'])**2))),
                   per_joint_clamp=dict(zip(JOINT_NAMES, trace['clamp_mask'].mean(axis=0).tolist())),
                   joint_speed_max_rad_s=float(qd.max()), joint_speed_p99_rad_s=float(np.quantile(qd, .99)),
                   joint_limit_frames=int((np.maximum(q_lo - q, q - q_hi) > 1e-5).any(axis=-1).sum()),
                   loaded_foot_speed_m_s=float(np.linalg.norm(trace['foot_velocity'][1:][loaded, :2], axis=-1).mean())
                   if loaded.any() else None)
        if 'actuator_torque' in trace:
            torque = np.abs(trace['actuator_torque'][1:])
            row['torque_saturation_ratio'] = float((torque >= mdp.effort.cpu().numpy() * .99).mean())
    return row


def aggregate(trials, expected_count):
    # 评估故障属于缺失观测，禁止用删去异常种子的平均值选择最佳权重。
    valid = [row for row in trials if row['termination'] not in INVALID_TERMINATIONS]
    if not valid:
        return dict(valid_for_best=False, valid_resets=0, expected_resets=expected_count)
    reduced = {k: _mean(valid, k) for k in (
        'episode_length_s', 'matched_touchdowns', 'consecutive_planned_touchdowns',
        'root_xy_rmse_m', 'root_orientation_rmse_deg', 'foot_position_rmse_m', 'contact_agreement',
        'normalized_return', 'full_clip_normalized_return', 'clamp_ratio', 'first_off_ground_s', 'first_lift_m',
        'first_single_support_s', 'first_max_height_m', 'first_planned_max_height_m',
        'first_swing_position_rmse_m', 'first_loaded_fraction', 'first_double_off_fraction', 'first_nearest_landing_xy_m',
        'joint_speed_p99_rad_s', 'joint_limit_frames', 'torque_saturation_ratio', 'action_rate_rms',
        'loaded_foot_speed_m_s')}
    reduced.update(valid_for_best=len(valid) == expected_count, valid_resets=len(valid), expected_resets=expected_count,
                   joint_speed_max_rad_s=max(row['joint_speed_max_rad_s'] or 0 for row in valid),
                   survival_rate=_mean(valid, 'survived_full_reference'),
                   standing_rate=_mean(valid, 'stable_initial_standing'),
                   first_swing_rate=_mean(valid, 'first_swing'), first_touchdown_rate=_mean(valid, 'first_touchdown'),
                   complete_tracking_rate=float(np.mean([
                       row['survived_full_reference'] and row['consecutive_planned_touchdowns'] == len(row['footsteps'])
                       for row in valid])))
    with_actions = [row for row in valid if 'raw_action' in row]
    if with_actions:
        reduced['mean_seed_raw_action'] = {key: float(np.mean([row['raw_action'][key] for row in with_actions]))
                                           for key in with_actions[0]['raw_action']}
        reduced['mean_seed_joint_clamp'] = {name: float(np.mean([row['per_joint_clamp'][name] for row in with_actions]))
                                            for name in JOINT_NAMES}
    return reduced


def gait_rank(result, *, m48=False, m49=False):
    score = result['aggregate']
    errors = tuple(-score[key] if score[key] is not None else -np.inf for key in
                   ('foot_position_rmse_m', 'root_xy_rmse_m', 'root_orientation_rmse_deg'))
    if m49:
        elbows = score.get('elbows')
        posture = -float(np.mean([part['near_limit_ratio'] for part in elbows.values()])) if elbows else -np.inf
        accel = score.get('upper_smoothness', {}).get('after_startup', {}).get('acceleration_rms_rad_s2')
        return (score['survival_rate'], score['complete_tracking_rate'], score['consecutive_planned_touchdowns'],
                score['first_touchdown_rate'], *errors, posture, -accel if accel is not None else -np.inf,
                result['iteration'])
    if m48:
        return (score['complete_tracking_rate'], score['first_touchdown_rate'],
                score['consecutive_planned_touchdowns'], score['survival_rate'],
                score['first_swing_rate'], *errors, result['iteration'])
    return (score['complete_tracking_rate'], score['consecutive_planned_touchdowns'],
            score['first_touchdown_rate'], score['first_swing_rate'], *errors, score['survival_rate'],
            result['iteration'])


def upper_body_best(milestones):
    # 完成率、连续落脚与首次事件必须一致，足部和根误差仅允许毫米级差异。
    pool = [r for r in milestones.values() if r['aggregate']['valid_for_best']]
    if not pool:
        return None
    primary = max(gait_rank(r)[:4] for r in pool)
    pool = [r for r in pool if gait_rank(r)[:4] == primary]
    for key, tolerance in (('foot_position_rmse_m', .002), ('root_xy_rmse_m', .005)):
        floor = min(r['aggregate'][key] for r in pool)
        pool = [r for r in pool if r['aggregate'][key] <= floor+tolerance]

    def rank(result):
        upper = result['aggregate']['upper_body']
        return (-upper['arm_rmse_rad'], -upper['waist_rmse_rad'], -upper['arm_acceleration_rms'],
                upper['body_prior_contribution'], -result['aggregate']['torque_saturation_ratio'], *gait_rank(result))

    return max(pool, key=rank)


def _prefix(iteration, seed, baseline):
    return f'baseline_s{seed}_' if baseline else f'i{iteration}_s{seed}_'


def m48_trial_metrics(trace, row, mdp):
    # 复用同一落脚检测器识别的触地点，统计实测连续承重。
    for event, side in zip(row['footsteps'], mdp.motion.touchdown_side.cpu().tolist()):
        actual = event['actual_touchdown_s']
        if actual is None:
            event['continuous_support_s'] = None
        else:
            start = round(actual / CONTROL_DT)
            loaded = trace['contact'][start:, side]
            off = np.flatnonzero(~loaded)
            event['continuous_support_s'] = (int(off[0]) if len(off) else len(loaded)) * CONTROL_DT

    if not row['steps']:
        return
    n = len(JOINT_NAMES)
    q = trace['state'][:, 13:13+n]
    qd = trace['state'][:, 13+n:].astype(np.float64)
    upper = upper_joint_indices(JOINT_NAMES)
    elbow = JOINT_NAMES.index('zarm_r4_joint')
    target = trace['target']
    raw = trace['raw_action']
    effective = trace['effective_action']

    def rms(values):
        return float(np.sqrt(np.square(values).mean())) if values.size else None

    def smoothness(start):
        speed = qd[start:, upper]
        return dict(velocity_rms_rad_s=rms(speed),
                    acceleration_rms_rad_s2=rms(np.diff(speed, axis=0) / CONTROL_DT),
                    jerk_rms_rad_s3=rms(np.diff(speed, n=2, axis=0) / CONTROL_DT**2),
                    raw_action_diff2_rms=rms(np.diff(raw[start:, upper], n=2, axis=0)),
                    effective_action_diff2_rms=rms(np.diff(effective[start:, upper], n=2, axis=0)),
                    pd_target_diff2_rms_rad=rms(np.diff(target[start:, upper], n=2, axis=0)))

    # 统一以 0.20 s 切分，便于和关闭过渡的 B 组对照。
    row['upper_smoothness'] = dict(whole=smoothness(0), after_startup=smoothness(round(.20 / CONTROL_DT)),
                                  first_elbow_target_error_rad=float(target[0, elbow] - q[0, elbow]),
                                  first_elbow_velocity_rad_s=float(qd[1, elbow]),
                                  first_elbow_acceleration_rad_s2=float((qd[1, elbow] - qd[0, elbow]) / CONTROL_DT))
    lo, hi = mdp.kin.q_lo.cpu().numpy(), mdp.kin.q_hi.cpu().numpy()
    row['joint_limit_excess_rad'] = float(np.maximum(lo - q[1:], q[1:] - hi).clip(min=0).max())
    row['left_ankle_clamp_ratio'] = row['per_joint_clamp']['leg_l5_joint']


def elbow_posture(trace, mdp):
    # 与 M4.8 报告同口径，使用启动 0.20 s 后实际观测到的关节角。
    joints = [JOINT_NAMES.index(name) for name in ELBOW_JOINT_NAMES]
    q = trace['state'][round(.20/CONTROL_DT):, 13:13+len(JOINT_NAMES)][:, joints]
    if not len(q):
        return {}
    lo, hi = (x.cpu().numpy()[joints] for x in (mdp.kin.q_lo, mdp.kin.q_hi))
    near = (np.minimum(q-lo, hi-q) <= .02).mean(0)
    return {name: dict(min_rad=float(a), max_rad=float(b), near_limit_ratio=float(n))
            for name,a,b,n in zip(ELBOW_JOINT_NAMES, q.min(0), q.max(0), near)}


def collect_checkpoint(checkpoint, args, *, baseline=False):
    from isaaclab.app import launch_simulation

    from .m4_dynamics import SolverReadback
    from .m4_task import M4TrackingSim, build_m4_task, make_m4_sim
    from ..kinematics import RobotKinematics
    from ..networks import Policy
    from ..roban import assert_checkpoint
    from ..transfer import assert_policy_checkpoint

    ck = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert_checkpoint(ck)
    config = SimpleNamespace(**ck['args'])
    device = args.device
    if torch.device(device).type != torch.device(config.device).type:
        raise ValueError('冻结评估须使用同类 Newton 求解器设备')
    motion, goals, mdp, _ = build_m4_task(config, RobotKinematics(device), [str(args.m4_reference)], device)
    mdp.assert_checkpoint_task(ck)
    goals.spec.assert_checkpoint(ck)
    physics = mdp.physics_contract(config)
    if ck['physical_contract'] != physics or ck['data_contract'] != motion.validation:
        raise ValueError(f'参考或物理契约与检查点不一致：{checkpoint}')
    assert_policy_checkpoint(ck, config.hist, config.obs_adapter, not config.freeze_decoder,
                             obs_dim=mdp.obs_dim, observation_mode=mdp.observation_mode)
    pi = Policy(mdp.obs_dim, goals.spec.dim, hist=config.hist, obs_adapter=config.obs_adapter).to(device)
    pi.load_state_dict(ck['pi'])
    pi.eval()
    iteration = int(ck['iteration'])
    m49_periodic = getattr(args, 'm49_periodic', False)
    control_periodic = getattr(args, 'm48_periodic', False) or m49_periodic
    # 轨迹跟随训练检查点的奖励定义，最终表格的任务指标始终共用同一判定。
    metadata = dict(seeds=args.seeds, dt=CONTROL_DT, joint_names=JOINT_NAMES, foot_names=FOOT_BODY_NAMES,
                    planned_liftoff_time=motion.liftoff_time.cpu().numpy(),
                    planned_touchdown_time=motion.touchdown_time.cpu().numpy(),
                    planned_touchdown_side=motion.touchdown_side.cpu().numpy())
    recorded, trials, upper_rows = dict(metadata), [], []
    cfg = make_m4_sim(device, config)
    with launch_simulation(cfg, args):
        env = M4TrackingSim(cfg, 1, motion, mdp, gamma=config.gamma, eval_mode=True, reset_noise=False, goals=goals)
        contacts = env.ground_contacts()
        cuda = torch.device(device).type == 'cuda'
        solver = SolverReadback(env) if cuda else None
        friction = EffectiveFriction(env) if cuda else None
        for seed in args.seeds:
            trace = collect_trial(env, pi, contacts, seed, friction, solver=solver, capture_faults=True)
            row = summarize_gait(trace, motion, mdp)
            if control_periodic:
                m48_trial_metrics(trace, row, mdp)
            if m49_periodic:
                row['elbows'] = elbow_posture(trace, mdp)
            if (getattr(args, 'upper_body', False) or getattr(args, 'upper_metrics_only', False)) and len(trace['action']):
                from .m45_upper import upper_metrics

                upper = upper_metrics(trace, motion)
                upper['body_prior_contribution'] = float(trace['reward_terms'][:, list(trace['reward_term_names']).index('body_prior')].mean())
                row['upper_body'] = upper
                upper_rows.append(upper)
            trials.append(row)
            prefix = _prefix(iteration, seed, baseline)
            if not control_periodic:
                recorded.update({prefix + key: value for key, value in trace.items()})
            print(f'独立评估 iter={iteration} seed={seed} steps={row["steps"]} '
                  f'first_swing={int(row["first_swing"])} first_touchdown={int(row["first_touchdown"])} '
                  f'first_failure={row["footsteps"][0]["failure"]} reason={row["termination"]}', flush=True)
            # EPA 可由原有世界重置隔离。严重故障不再复用同一个求解器做其它试次。
            if row['termination'] in ('fatal_dynamics', 'nonfinite'):
                break
    if baseline:
        recorded['baseline_iteration'] = iteration
    reduced = aggregate(trials, len(args.seeds))
    if upper_rows:
        reduced['upper_body'] = {key: np.mean([x[key] for x in upper_rows], axis=0).tolist() for key in upper_rows[0]}
    if control_periodic:
        valid = [row for row in trials if row['termination'] not in INVALID_TERMINATIONS]
        executed = [row for row in valid if 'upper_smoothness' in row]
        motion_rows = [row['upper_smoothness'] for row in executed]
        reduced.update(epa_resets=sum(row['termination'] == 'epa_horizon' for row in trials),
                       fatal_errors=sum(row['termination'] in ('fatal_dynamics', 'nonfinite') for row in trials),
                       joint_limit_excess_rad=max((row.get('joint_limit_excess_rad', 0.) for row in valid), default=0.),
                       left_ankle_clamp_ratio=_mean(executed, 'left_ankle_clamp_ratio'),
                       touchdown_continuous_support_s=_mean([event for row in valid for event in row['footsteps']],
                                                           'continuous_support_s'))
        if motion_rows:
            metrics = {key: _mean(motion_rows, key) for key in motion_rows[0] if not isinstance(motion_rows[0][key], dict)}
            for phase in ('whole', 'after_startup'):
                phase_rows = [row[phase] for row in motion_rows]
                metrics[phase] = {key: _mean(phase_rows, key) for key in phase_rows[0]}
            reduced['upper_smoothness'] = metrics
        if m49_periodic:
            poses = [row['elbows'] for row in valid if row['elbows']]
            if poses:
                reduced['elbows'] = {name: dict(min_rad=min(p[name]['min_rad'] for p in poses),
                                               max_rad=max(p[name]['max_rad'] for p in poses),
                                               near_limit_ratio=_mean([p[name] for p in poses], 'near_limit_ratio'))
                                     for name in ELBOW_JOINT_NAMES}
    return dict(iteration=iteration, checkpoint=str(checkpoint), aggregate=reduced,
                per_reset=trials), recorded, physics


def _save_archive(directory, summary, traces):
    with TemporaryDirectory(dir=directory) as scratch:
        tmp = Path(scratch)
        (tmp / 'results.json').write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
        if traces is not None:
            np.savez_compressed(tmp / 'rollouts.npz', **traces)
            (tmp / 'rollouts.npz').replace(directory / 'rollouts.npz')
        (tmp / 'results.json').replace(directory / 'results.json')


def _alias(source, destination, identity):
    with TemporaryDirectory(dir=destination.parent) as scratch:
        alias = Path(scratch) / destination.name
        alias.hardlink_to(source)
        if _checkpoint_identity(alias) != identity:
            raise RuntimeError(f'评估期间源检查点被替换：{source}')
        alias.replace(destination)


def main():
    from isaaclab.app import add_launcher_args

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m4_reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('outputs/m4_3'))
    parser.add_argument('--checkpoint', type=Path, help='指定此轮冻结权重，默认 last.pt')
    parser.add_argument('--baseline', type=Path, help='指定 M4.2 第 5000 轮权重')
    parser.add_argument('--baseline-only', action='store_true', help='只存公共比较基线')
    parser.add_argument('--final', action='store_true', help='合并结果并生成带阶段标记的对照视频')
    upper = parser.add_mutually_exclusive_group()
    upper.add_argument('--upper-body', action='store_true', help='步态等价时使用上身跟踪决定最佳检查点')
    upper.add_argument('--upper-metrics-only', action='store_true', help='记录上身指标，沿用 M4.3 排名且仅保存汇总')
    upper.add_argument('--m48-periodic', action='store_true', help='记录 M4.8 步态与控制平滑性，仅保存汇总')
    upper.add_argument('--m49-periodic', action='store_true', help='记录 M4.9 步态、平滑性及双肘姿态，仅保存汇总')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(DEFAULT_SEEDS))
    add_launcher_args(parser)
    args = parser.parse_args()
    if len(set(args.seeds)) < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error('需要至少两个不同种子')
    if args.baseline_only and args.baseline is None:
        parser.error('--baseline-only 需要 --baseline')
    if args.baseline_only and args.final:
        parser.error('--baseline-only 与 --final 不能一起使用')
    control_periodic = args.m48_periodic or args.m49_periodic
    summary_only = args.upper_metrics_only or control_periodic
    if summary_only and (args.baseline or args.final):
        parser.error('仅汇总模式用于训练周期评估')

    output = args.out / 'evaluation'
    output.mkdir(parents=True, exist_ok=True)
    summary_path, trace_path = output / 'results.json', output / 'rollouts.npz'
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        if summary['seeds'] != args.seeds:
            raise ValueError('已有实验轨迹使用不同的独立评估种子')
        if summary_only:
            traces = None
        else:
            with np.load(trace_path) as record:
                traces = {key: record[key] for key in record.files}
    else:
        summary = dict(seeds=args.seeds, contact_threshold_n=CONTACT_FORCE_N, milestones={})
        traces = None if summary_only else {}

    def verify_contract(physics):
        # 只比对已经解析的物理配置，不扫描或校验文件哈希。
        physical = json.loads(json.dumps(physics))
        if 'physics' in summary and physical != summary['physics']:
            raise ValueError('基线与候选的物理契约不同')
        summary['physics'] = physical

    if args.baseline is not None and 'baseline' not in summary:
        base, payload, physics = collect_checkpoint(args.baseline, args, baseline=True)
        verify_contract(physics)
        summary['baseline'] = base
        traces.update(payload)
        _save_archive(output, summary, traces)

    if args.baseline_only:
        return

    checkpoint = args.checkpoint or (args.out / 'checkpoints' / 'last.pt')
    identity = _checkpoint_identity(checkpoint)
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    iteration = int(ck['iteration'])
    del ck
    key = str(iteration)
    if key not in summary['milestones']:
        result, payload, physics = collect_checkpoint(checkpoint, args)
        verify_contract(physics)
        if _checkpoint_identity(checkpoint) != identity:
            raise RuntimeError(f'评估期间源检查点被替换：{checkpoint}')
        milestone = args.out / 'checkpoints' / f'iter_{iteration}.pt'
        if milestone.is_file() and _checkpoint_identity(milestone) == identity:
            result['checkpoint'] = str(milestone)
        elif control_periodic:
            raise ValueError(f'周期评估缺少与 Last 对应的里程碑：{milestone}')
        summary['milestones'][key] = result
        if traces is not None:
            traces.update(payload)
            traces['iterations'] = np.asarray(sorted(map(int, summary['milestones'])))
        best = summary.get('best_iteration')
        if args.upper_body:
            chosen = upper_body_best(summary['milestones'])
            if chosen is not None and chosen['iteration'] != best:
                source = Path(chosen['checkpoint'])
                _alias(source, args.out / 'checkpoints' / 'best.pt', _checkpoint_identity(source))
                summary['best_iteration'] = chosen['iteration']
        else:
            rank_options = dict(m48=args.m48_periodic, m49=args.m49_periodic)
            best_score = gait_rank(summary['milestones'][str(best)], **rank_options) if best is not None else ()
            if result['aggregate']['valid_for_best'] and gait_rank(result, **rank_options) > best_score:
                if not control_periodic:
                    _alias(checkpoint, args.out / 'checkpoints' / 'best.pt', identity)
                summary['best_iteration'] = iteration
        valid_success = next((row for row in result['per_reset'] if row['first_touchdown']), None)
        if valid_success is not None and 'first_success' not in summary:
            destination = Path(result['checkpoint'])
            if not args.upper_body and not summary_only:
                destination = args.out / 'checkpoints' / 'first_success.pt'
                _alias(checkpoint, destination, identity)
            summary['first_success'] = dict(iteration=iteration, seed=valid_success['seed'], checkpoint=str(destination))
        _save_archive(output, summary, traces)
        print(f'评估汇总 iter={iteration} {json.dumps(result["aggregate"], ensure_ascii=False)}', flush=True)

    if control_periodic:
        # 汇总先原子写入。中断后用里程碑恢复 Best，避免别名与已确认的排序分离。
        best = summary.get('best_iteration')
        if best is not None:
            source = Path(summary['milestones'][str(best)]['checkpoint'])
            destination = args.out / 'checkpoints' / 'best.pt'
            source_identity = _checkpoint_identity(source)
            if not destination.is_file() or _checkpoint_identity(destination) != source_identity:
                _alias(source, destination, source_identity)
        if summary['milestones'][key]['aggregate']['fatal_errors']:
            raise FloatingPointError(f'第 {iteration} 轮周期评估出现严重物理异常')

    success = summary.get('first_success')
    success_video = output / 'first_touchdown.mp4'
    if success is not None and not args.upper_body and not summary_only and not success_video.is_file():
        prefix = _prefix(success['iteration'], success['seed'], False)
        try:
            render_comparison(trace_path, args.m4_reference, success_video,
                              chapters=[(prefix, f'First touchdown | M4.3 iter {success["iteration"]} | seed {success["seed"]}')])
        except (RuntimeError, OSError, ImportError) as exc:
            # 训练环境若缺少 EGL，已保存的权重和逐步轨迹供最终评估补渲染。
            if args.final:
                raise
            print(f'首次落脚视频等待最终渲染：{exc}', flush=True)

    if args.final:
        if 'baseline' not in summary:
            parser.error('最终对照需要已有基线或 --baseline')
        seed = args.seeds[0]
        best = summary.get('best_iteration')
        chapters = [(_prefix(0, seed, True), f'M4.2 v2 | iteration {summary["baseline"]["iteration"]} | seed {seed}'),
                    (_prefix(iteration, seed, False), f'M4.3 {"Last + Best" if best == iteration else "Last"} | iteration {iteration} | seed {seed}')]
        if best is not None and best != iteration:
            chapters.append((_prefix(best, seed, False), f'M4.3 Best | iteration {best} | seed {seed}'))
        render_comparison(trace_path, args.m4_reference, output / 'reference_vs_newton.mp4', chapters=chapters)


if __name__ == '__main__':
    main()
