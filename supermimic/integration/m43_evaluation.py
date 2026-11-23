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
from ..motion import CONTROL_DT
from ..roban import FOOT_BODY_NAMES, JOINT_NAMES


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


def gait_rank(result):
    score = result['aggregate']
    errors = tuple(-score[key] if score[key] is not None else -np.inf for key in
                   ('foot_position_rmse_m', 'root_xy_rmse_m', 'root_orientation_rmse_deg'))
    return (score['complete_tracking_rate'], score['consecutive_planned_touchdowns'],
            score['first_touchdown_rate'], score['first_swing_rate'], *errors, score['survival_rate'],
            result['iteration'])


def _prefix(iteration, seed, baseline):
    return f'baseline_s{seed}_' if baseline else f'i{iteration}_s{seed}_'


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
    # 轨迹跟随训练检查点的奖励定义，最终表格的任务指标始终共用同一判定。
    metadata = dict(seeds=args.seeds, dt=CONTROL_DT, joint_names=JOINT_NAMES, foot_names=FOOT_BODY_NAMES,
                    planned_liftoff_time=motion.liftoff_time.cpu().numpy(),
                    planned_touchdown_time=motion.touchdown_time.cpu().numpy(),
                    planned_touchdown_side=motion.touchdown_side.cpu().numpy())
    recorded, trials = dict(metadata), []
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
            trials.append(row)
            prefix = _prefix(iteration, seed, baseline)
            recorded.update({prefix + key: value for key, value in trace.items()})
            print(f'独立评估 iter={iteration} seed={seed} steps={row["steps"]} '
                  f'first_swing={int(row["first_swing"])} first_touchdown={int(row["first_touchdown"])} '
                  f'first_failure={row["footsteps"][0]["failure"]} reason={row["termination"]}', flush=True)
            # EPA 可由原有世界重置隔离。严重故障不再复用同一个求解器做其它试次。
            if row['termination'] in ('fatal_dynamics', 'nonfinite'):
                break
    if baseline:
        recorded['baseline_iteration'] = iteration
    return dict(iteration=iteration, checkpoint=str(checkpoint), aggregate=aggregate(trials, len(args.seeds)),
                per_reset=trials), recorded, physics


def _save_archive(directory, summary, traces):
    with TemporaryDirectory(dir=directory) as scratch:
        tmp = Path(scratch)
        np.savez_compressed(tmp / 'rollouts.npz', **traces)
        (tmp / 'results.json').write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
        for name in ('rollouts.npz', 'results.json'):
            (tmp / name).replace(directory / name)


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
    parser.add_argument('--seeds', type=int, nargs='+', default=list(DEFAULT_SEEDS))
    add_launcher_args(parser)
    args = parser.parse_args()
    if len(set(args.seeds)) < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error('需要至少两个不同种子')
    if args.baseline_only and args.baseline is None:
        parser.error('--baseline-only 需要 --baseline')
    if args.baseline_only and args.final:
        parser.error('--baseline-only 与 --final 不能一起使用')

    output = args.out / 'evaluation'
    output.mkdir(parents=True, exist_ok=True)
    summary_path, trace_path = output / 'results.json', output / 'rollouts.npz'
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        if summary['seeds'] != args.seeds:
            raise ValueError('已有实验轨迹使用不同的独立评估种子')
        with np.load(trace_path) as record:
            traces = {key: record[key] for key in record.files}
    else:
        summary, traces = dict(seeds=args.seeds, contact_threshold_n=CONTACT_FORCE_N, milestones={}), {}

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
        summary['milestones'][key] = result
        traces.update(payload)
        traces['iterations'] = np.asarray(sorted(map(int, summary['milestones'])))
        best = summary.get('best_iteration')
        best_score = gait_rank(summary['milestones'][str(best)]) if best is not None else ()
        if result['aggregate']['valid_for_best'] and gait_rank(result) > best_score:
            _alias(checkpoint, args.out / 'checkpoints' / 'best.pt', identity)
            summary['best_iteration'] = iteration
        valid_success = next((row for row in result['per_reset'] if row['first_touchdown']), None)
        if valid_success is not None and 'first_success' not in summary:
            destination = args.out / 'checkpoints' / 'first_success.pt'
            _alias(checkpoint, destination, identity)
            summary['first_success'] = dict(iteration=iteration, seed=valid_success['seed'], checkpoint=str(destination))
        _save_archive(output, summary, traces)
        print(f'评估汇总 iter={iteration} {json.dumps(result["aggregate"], ensure_ascii=False)}', flush=True)

    success = summary.get('first_success')
    success_video = output / 'first_touchdown.mp4'
    if success is not None and not success_video.is_file():
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
