"""比较两种 Roban 位置目标映射，评估真实关节状态、执行力矩和 F2 步态。"""

import argparse
from contextlib import contextmanager
from copy import copy
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

if __name__ == '__main__':
    os.environ['SUPERMIMIC_ROBOT'] = 'roban'
    os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import torch

from .m42v3_rollout import collect_trial, render_comparison
from .m43_evaluation import DEFAULT_SEEDS, _runs, _save_archive, aggregate, summarize_gait
from ..motion import CONTROL_DT
from ..roban import JOINT_NAMES


ANKLE = 'leg_l5_joint'
PERTURB_SEEDS = (1101, 1102, 1103, 1104, 1105, 1106)
PERTURB_LIMITS = {'root_xy_m': .02, 'root_lift_m': .012, 'root_yaw_deg': 3., 'root_roll_pitch_deg': 1.,
                  'joint_position_rad': .01, 'root_v_m_s': .02, 'root_w_rad_s': .05, 'joint_speed_rad_s': .10}


def perturbed_start(env, seed):
    from ..rotations import mat_to_quat, so3_exp

    # 记录扰动参数，以同一随机生成器对所有权重施加逐值相同的扰动。F2 时钟与计划保持原值。
    rng = np.random.default_rng(seed)
    p, R, v, w, q, qd = env.state()
    vals = PERTURB_LIMITS
    dp = q.new_tensor([*rng.uniform(-vals['root_xy_m'], vals['root_xy_m'], 2), rng.uniform(.006, vals['root_lift_m'])])
    da = q.new_tensor([*rng.uniform(-vals['root_roll_pitch_deg'], vals['root_roll_pitch_deg'], 2),
                       rng.uniform(-vals['root_yaw_deg'], vals['root_yaw_deg'])]).deg2rad()
    dv = q.new_tensor(rng.uniform(-vals['root_v_m_s'], vals['root_v_m_s'], 3))
    dw = q.new_tensor(rng.uniform(-vals['root_w_rad_s'], vals['root_w_rad_s'], 3))
    dq = q.new_tensor(rng.uniform(-vals['joint_position_rad'], vals['joint_position_rad'], q.shape))
    dqd = q.new_tensor(rng.uniform(-vals['joint_speed_rad_s'], vals['joint_speed_rad_s'], qd.shape))
    q1 = q+dq
    if not bool(((q1 >= env.mdp.kin.q_lo) & (q1 <= env.mdp.kin.q_hi)).all()):
        raise ValueError('扰动在初始化时超过机械限位')
    rotation = mat_to_quat(so3_exp(da[None]) @ R)[:, [1, 2, 3, 0]]
    pose = torch.cat((p + env.scene.env_origins + dp, rotation), dim=-1)
    env.robot.write_root_link_pose_to_sim_index(root_pose=pose)
    env.robot.write_root_link_velocity_to_sim_index(root_velocity=torch.cat((v+dv, w+dw), dim=-1))
    env.robot.write_joint_position_to_sim_index(position=q1[:, env.sim_to_canonical])
    env.robot.write_joint_velocity_to_sim_index(velocity=(qd+dqd)[:, env.sim_to_canonical])
    env.scene.write_data_to_sim()
    env.refresh()
    env.refresh_startup_pose()


def _quantiles(samples):
    return np.quantile(samples, [.05, .50, .95, .99, 1.]).tolist() if samples.size else None


def _corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 and min(a.std(), b.std()) > 1e-8 else None


def baseline_checkpoints(directory):
    # 历史 M4.3 没有单独保留 4500 文件时，使用权重轮次经过验证的 best 别名。
    root = directory / 'checkpoints'
    iter_4500 = root / 'iter_4500.pt'
    return (iter_4500 if iter_4500.is_file() else root / 'best.pt'), root / 'iter_5000.pt'


def control_trace(trace, mdp, motion, *, expected_target=None):
    # 位置超限与实际裁剪分别存储，避免把无裁剪策略的超限误报为零。
    q0, alpha = (x.cpu().numpy() for x in (mdp.default_q, mdp.alpha))
    raw_target = q0 + alpha * trace['raw_action']
    bounds = np.stack([mdp.kin.q_lo.cpu().numpy(), mdp.kin.q_hi.cpu().numpy()])
    lower, upper = raw_target < bounds[0], raw_target > bounds[1]
    qref = motion.q.cpu().numpy()[trace['frame']]
    trace.update(raw_target=raw_target, raw_target_lower=lower, raw_target_upper=upper,
                 raw_target_out_of_range=lower | upper, planned_joint_position=qref)
    if expected_target is None and mdp.startup_steps:
        device = mdp.default_q.device
        reset_q = torch.as_tensor(trace['state'][0, 13:13+len(JOINT_NAMES)], device=device).clamp(mdp.kin.q_lo, mdp.kin.q_hi)
        expected_target = mdp.executed_target(torch.as_tensor(trace['raw_action'], device=device), reset_q,
                                              torch.arange(len(raw_target), device=device)).cpu().numpy()
    if expected_target is None:
        expected_target = raw_target if mdp.unclipped_pd_target else np.clip(raw_target, *bounds)
    np.testing.assert_allclose(trace['target'], expected_target, atol=1e-6, rtol=1e-6)


def controller_stats(traces, mdp):
    valid = [t for t in traces if len(t['action'])]
    if not valid:
        return {}
    n = len(JOINT_NAMES)
    cat = lambda key: np.concatenate([t[key] for t in valid])
    post = lambda key: np.concatenate([t[key][1:] for t in valid])
    pre = lambda key: np.concatenate([t[key][:-1] for t in valid])
    lo, hi, effort, kp, kd = (value.detach().cpu().numpy() for value in
                             (mdp.kin.q_lo, mdp.kin.q_hi, mdp.effort, mdp.kp, mdp.kd))
    a, raw, target = (cat(key) for key in ('raw_action', 'raw_target', 'target'))
    q, speed = post('state')[:, 13:13+n], post('state')[:, 13+n:]
    q_before, speed_before = pre('state')[:, 13:13+n], pre('state')[:, 13+n:]
    torque, limit_force = post('actuator_torque'), post('joint_limit_force')
    lower, upper = cat('raw_target_lower'), cat('raw_target_upper')
    outside = lower | upper
    excess = np.maximum(0., np.maximum(lo-raw, raw-hi))
    actual_excess = np.maximum(0., np.maximum(lo-q, q-hi))
    raw_demand = kp*(raw-q_before)-kd*speed_before
    sat = np.abs(torque) >= .99*effort
    plan, contact, clock = post('planned_contact'), post('contact'), post('time')
    error = post('foot_target_error')
    loaded_velocity_xy = post('foot_velocity')[contact, :2]
    joint_stop = limit_force > 1e-6
    # 在有止挡约束力时，按离哪个边界更近判定施力是否继续指向止挡。
    nearest_lower = q-lo < hi-q
    pushes_stop = joint_stop & np.where(nearest_lower, torque < 0., torque > 0.)
    planned_phase = {'double_support': plan.all(axis=-1)}
    events = zip(mdp.motion.liftoff_time.cpu().tolist(), mdp.motion.touchdown_time.cpu().tolist(),
                 mdp.motion.touchdown_side.cpu().tolist())
    for i, (lift, touch, side) in enumerate(events, 1):
        planned_phase[f'swing_{i}_{("left", "right")[side]}'] = (clock >= lift-1e-8) & (clock < touch-1e-8)
    joints = {}
    for j, name in enumerate(JOINT_NAMES):
        side = 0 if name.startswith('leg_l') else (1 if name.startswith('leg_r') else None)
        f_err = error[:, side] if side is not None else error.mean(axis=-1)
        phases = dict(planned_phase)
        if side is not None:
            phases.update(planned_support=plan[:, side], planned_swing=~plan[:, side],
                          loaded=contact[:, side], unloaded=~contact[:, side])
        excess_where = np.where(lower[:,j], lo[j], hi[j])
        close = np.abs(q[:,j] - excess_where) <= .02
        use = outside[:,j]
        run = max((b-a for t in valid for a,b in _runs(t['raw_target_out_of_range'][:,j])), default=0)
        joints[name] = dict(action=_quantiles(a[:,j]), raw_target_rad=_quantiles(raw[:,j]),
                            applied_target_rad=_quantiles(target[:,j]), actual_angle_rad=_quantiles(q[:,j]),
                            reference_angle_rad=_quantiles(post('planned_joint_position')[:,j]),
                            actual_speed_abs_rad_s=_quantiles(np.abs(speed[:,j])), torque_nm=_quantiles(torque[:,j]),
                            torque_abs_max_nm=float(np.abs(torque[:,j]).max()),
                            torque_saturation=float(sat[:,j].mean()), lower=float(lower[:,j].mean()),
                            upper=float(upper[:,j].mean()), longest_outside_s=run*CONTROL_DT,
                            mechanical_stop_active=float(joint_stop[:,j].mean()),
                            pushing_mechanical_stop=float(pushes_stop[:,j].mean()),
                            stop_reaction_when_active_nm=_quantiles(limit_force[:,j][joint_stop[:,j]]),
                            actuator_load_while_pushing_stop_nm=_quantiles(np.abs(torque[:,j][pushes_stop[:,j]])),
                            outside_rad=_quantiles(excess[:,j][use]), actual_violation_max_rad=float(actual_excess[:,j].max()),
                            actual_violation_ratio=float((actual_excess[:,j]>1e-5).mean()),
                            near_bound_given_outside=float(close[use].mean()) if use.any() else None,
                            raw_pd_exceeds_effort_given_outside=float((np.abs(raw_demand[:,j][use]) >= effort[j]).mean())
                            if use.any() else None,
                            pearson_excess_abs_torque=_corr(excess[:,j], np.abs(torque[:,j])),
                            pearson_excess_torque_saturation=_corr(excess[:,j], sat[:,j].astype(float)),
                            pearson_excess_foot_error=_corr(excess[:,j], f_err),
                            phases={key: dict(n=int(mask.sum()), ratio=float(outside[mask,j].mean()))
                                    for key, mask in phases.items() if mask.any()})
    ankle = joints[ANKLE]
    foot_normal = post('force')[:, :, 2]
    return dict(raw_target_out_of_range=float(outside.mean()), left_ankle_outside=ankle['lower']+ankle['upper'],
                left_ankle_torque_saturation=ankle['torque_saturation'],
                left_ankle_stop_active=ankle['mechanical_stop_active'],
                left_ankle_pushing_stop=ankle['pushing_mechanical_stop'],
                actuator_saturation=float(sat.mean()), actual_joint_limit_violation=float((actual_excess>1e-5).mean()),
                actual_joint_limit_violation_max_rad=float(actual_excess.max()), speed_max_rad_s=float(np.abs(speed).max()),
                speed_p99_rad_s=float(np.quantile(np.abs(speed),.99)),
                loaded_slip_speed_m_s=float(np.linalg.norm(loaded_velocity_xy, axis=-1).mean())
                if len(loaded_velocity_xy) else None,
                loaded_normal_force_n={name: _quantiles(foot_normal[:, i][contact[:, i]])
                                       for i, name in enumerate(('left', 'right'))},
                per_joint=joints)


def evaluator(traces, rows, expected, mdp):
    report = aggregate(rows, expected)
    report.update(controller_stats(traces, mdp))
    report['completed'] = sum(row['survived_full_reference'] for row in rows)
    report['first_swing_successes'] = sum(row['first_swing'] for row in rows)
    report['first_touchdown_successes'] = sum(row['first_touchdown'] for row in rows)
    report['full_tracking_successes'] = sum(row['survived_full_reference']
                                            and row['consecutive_planned_touchdowns'] == len(row['footsteps']) for row in rows)
    report['epa'] = sum(row['termination']=='epa_horizon' for row in rows)
    report['fatal'] = sum(row['termination']=='fatal_dynamics' for row in rows)
    return dict(aggregate=report, trials=rows)


@contextmanager
def scene(args, checkpoint, *, expected_iteration=None, num_envs=1, capture_substeps=False):
    from isaaclab.app import launch_simulation

    from .m4_dynamics import SolverReadback
    from .m4_task import M4TrackingSim, build_m4_task, make_m4_sim
    from ..kinematics import RobotKinematics
    from ..networks import Policy
    from ..roban import assert_checkpoint
    from ..transfer import assert_policy_checkpoint

    ck = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert_checkpoint(ck)
    iteration = int(ck['iteration'])
    if expected_iteration is not None and iteration != expected_iteration:
        raise ValueError(f'检查点 {checkpoint} 为第 {iteration} 轮，期望第 {expected_iteration} 轮')
    cfg = SimpleNamespace(**ck['args'])
    md = args.device
    motion, goals, mdp, _ = build_m4_task(cfg, RobotKinematics(md), [str(args.m4_reference)], md)
    mdp.assert_checkpoint_task(ck)
    goals.spec.assert_checkpoint(ck)
    physics = mdp.physics_contract(cfg)
    if ck['physical_contract'] != physics or ck['data_contract'] != motion.validation:
        raise ValueError(f'参考或物理契约与检查点不一致：{checkpoint}')
    assert_policy_checkpoint(ck, cfg.hist, cfg.obs_adapter, not cfg.freeze_decoder,
                             obs_dim=mdp.obs_dim, observation_mode=mdp.observation_mode)
    pi = Policy(mdp.obs_dim,goals.spec.dim,hist=cfg.hist,obs_adapter=cfg.obs_adapter).to(md)
    pi.load_state_dict(ck['pi'])
    pi.eval()
    sim = make_m4_sim(md, cfg)
    if capture_substeps:
        # Python 只在非图模式下逐个观察实际运行的物理子步。
        sim.physics.use_cuda_graph = False
    with launch_simulation(sim,args):
        env = M4TrackingSim(sim,num_envs,motion,mdp,gamma=cfg.gamma,eval_mode=True,reset_noise=False,goals=goals)
        yield env, pi, env.ground_contacts(), SolverReadback(env), iteration, physics


def trial(env, pi, contacts, solver, seed, *, fixed_actions=None, perturb=False):
    hook = (lambda sim: perturbed_start(sim,seed)) if perturb else None
    trace = collect_trial(env,pi,contacts,seed,solver=solver,capture_faults=True,
                          fixed_actions=fixed_actions,reset_hook=hook,capture_joint_limits=True)
    control_trace(trace, env.mdp, env.motion)
    return trace, summarize_gait(trace, env.motion, env.mdp)


def frozen_safety(args, output):
    archive = {}
    for expected, candidate in zip((4500, 5000), baseline_checkpoints(args.baseline)):
        with scene(args, candidate, expected_iteration=expected) as (env,pi,contacts,solver,i,_):
            mdp = env.mdp
            fields = {'q0_rad':mdp.default_q,'alpha_rad_per_action':mdp.alpha,
                      'lower_rad':mdp.kin.q_lo,'upper_rad':mdp.kin.q_hi,'effort_nm':mdp.effort,
                      'kp':mdp.kp,'kd':mdp.kd}
            stats = {'hard': ([],[]), 'unclipped': ([],[])}
            for seed in args.seeds:
                mdp.unclipped_pd_target = False
                hard, r0 = trial(env,pi,contacts,solver,seed)
                stats['hard'][0].append(hard)
                stats['hard'][1].append(r0)
                if r0['termination']=='fatal_dynamics':
                    break
                mdp.unclipped_pd_target = True
                unclipped, r1 = trial(env,pi,contacts,solver,seed,fixed_actions=hard['raw_action'])
                n = len(unclipped['action'])
                np.testing.assert_array_equal(hard['state'][0],unclipped['state'][0])
                np.testing.assert_array_equal(hard['action'][:n],unclipped['action'])
                stats['unclipped'][0].append(unclipped)
                stats['unclipped'][1].append(r1)
                for name,row in (('hard',r0),('unclipped',r1)):
                    print(f'同动作 {i} {seed} {name} 步数={row["steps"]} 足误差={row["foot_position_rmse_m"]} '
                          f'终止={row["termination"]}',flush=True)
                if r1['termination']=='fatal_dynamics':
                    break
            rows = {name:evaluator(*data,len(args.seeds),mdp) for name,data in stats.items()}
            archive[str(i)] = dict(runtime={k:dict(zip(JOINT_NAMES,v.cpu().tolist())) for k,v in fields.items()},
                                   **rows)
            print('冻结对比',i, json.dumps({name: {k: row['aggregate'].get(k) for k in (
                'completed','fatal','epa','actuator_saturation','left_ankle_outside',
                'left_ankle_torque_saturation','actual_joint_limit_violation_max_rad','speed_max_rad_s')}
                for name,row in rows.items()},ensure_ascii=False),flush=True)
    (output/'safety.json').write_text(json.dumps(dict(seeds=args.seeds, values=archive),
                                               ensure_ascii=False,allow_nan=False,indent=2)+'\n')


def final_comparison(args, output):
    with (output/'results.json').open() as stream:
        report = json.load(stream)
    with np.load(output/'rollouts.npz',allow_pickle=False) as stored:
        saved = {key:stored[key] for key in stored.files}
    if report['seeds'] != args.seeds:
        raise ValueError('统一评估须沿用原独立评估种子')
    m43_best, m43_final = baseline_checkpoints(args.baseline)
    checkpoints = args.out / 'checkpoints'
    phase = 'm45' if args.upper_body else 'm44'
    candidates = {'m43_best': (m43_best, 4500),
                  'm43_final': (m43_final, 5000),
                  phase+'_best': (checkpoints/'best.pt', report['best_iteration']),
                  phase+'_final': (checkpoints/'iter_5000.pt', 5000)}
    missing = [str(path) for path, _ in candidates.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('统一评估缺少检查点：' + '、'.join(missing))
    report['comparison'] = {'nominal_seeds':args.seeds, 'perturb_seeds':args.perturb_seeds,
                            'perturb_limits':dict(PERTURB_LIMITS, root_lift_min_m=.006), 'cases':{}}
    cases = report['comparison']['cases']
    # 先发布全部名义轨迹，梯度审计可直接使用同一档案，随后在同一套引擎上复测初态扰动。
    for group, seeds in (('nominal', args.seeds), ('perturb', args.perturb_seeds)):
        starts = {}
        for label, (path, expected) in candidates.items():
            with scene(args, path, expected_iteration=expected) as (env, pi, contacts, solver, iteration, physics):
                if json.loads(json.dumps(physics)) != report['physics']:
                    raise ValueError(f'统一评估的物理配置不一致：{path}')
                unclipped = env.mdp.unclipped_pd_target
                if unclipped != label.startswith('m44_'):
                    raise ValueError(f'统一评估的目标映射与候选标签不一致：{path}')
                if args.upper_body:
                    from .m45_upper import scored_upper_metrics

                    expected_weight = .1 if label.startswith('m45_') else 0.
                    if env.mdp.cfg.body_prior_weight != expected_weight:
                        raise ValueError(f'统一评估的上身奖励权重与标签不一致：{path}')
                    # 冻结策略按各自训练契约运行，另用 0.10 统一离线打分。
                    scoring = copy(env.mdp)
                    scoring.cfg = replace(scoring.cfg, body_prior_weight=.1)
                result = cases.setdefault(label, dict(iteration=iteration, checkpoint=str(path), unclipped=unclipped))
                traces, rows = [], []
                for seed in seeds:
                    trace,row = trial(env,pi,contacts,solver,seed,perturb=group=='perturb')
                    if args.upper_body and len(trace['action']):
                        row['upper_body'] = scored_upper_metrics(trace, scoring)
                    previous = starts.setdefault(seed, trace['state'][0])
                    np.testing.assert_allclose(previous,trace['state'][0],rtol=0,atol=1e-7)
                    traces.append(trace)
                    rows.append(row)
                    prefix = label + ('_perturb' if group=='perturb' else '') + f'_s{seed}_'
                    saved.update({prefix+key:value for key,value in trace.items()})
                    print(f'统一评估 {label} {group} {seed} 步数={row["steps"]} '
                          f'落脚={row["matched_touchdowns"]} 终止={row["termination"]}',flush=True)
                    if row['termination']=='fatal_dynamics':
                        break
                result[group] = evaluator(traces,rows,len(seeds),env.mdp)
                if args.upper_body:
                    metrics = [r['upper_body'] for r in rows if 'upper_body' in r]
                    if metrics:
                        result[group]['aggregate']['upper_body'] = {
                            key: np.mean([m[key] for m in metrics],axis=0).tolist() for key in metrics[0]}
            # 每个候选独立落盘，长时间复测中断后保留已完成的证据。
            _save_archive(output, report, saved)
        print(f'统一评估 {group} 轨迹已写入 {output}', flush=True)
    if not args.no_video:
        seed=args.seeds[0]
        if args.upper_body:
            from .m45_upper import final_visuals

            final_visuals(saved,args.m4_reference,output,args.out,seed)
        else:
            chapters=[(f'{label}_s{seed}_',f'{label.upper()} | Iteration {o["iteration"]} | Seed {seed}')
                      for label,o in report['comparison']['cases'].items() if f'{label}_s{seed}_state' in saved]
            render_comparison(output/'rollouts.npz',args.m4_reference,output/'reference_vs_newton.mp4',chapters=chapters)


def main():
    from isaaclab.app import add_launcher_args

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m4_reference',type=Path,required=True)
    parser.add_argument('--mode',choices=('safety','final'),required=True)
    parser.add_argument('--out',type=Path,default=Path('outputs/m4_4_unclipped'))
    parser.add_argument('--baseline',type=Path,default=Path('outputs/m4_3'))
    parser.add_argument('--seeds',nargs='+',type=int,default=list(DEFAULT_SEEDS))
    parser.add_argument('--perturb-seeds',nargs='+',type=int,default=list(PERTURB_SEEDS))
    parser.add_argument('--no-video',action='store_true')
    parser.add_argument('--upper-body',action='store_true',help='评估保留目标裁剪的 M4.5 上身消融')
    add_launcher_args(parser)
    args=parser.parse_args()
    if torch.device(args.device).type!='cuda':
        parser.error('实测力矩和统一评估需要 CUDA Newton')
    if len(args.seeds)!=len(set(args.seeds)) or len(args.perturb_seeds)!=len(set(args.perturb_seeds)):
        parser.error('同组评估种子须各不相同')
    output=args.out/'evaluation'
    output.mkdir(parents=True,exist_ok=True)
    (frozen_safety if args.mode=='safety' else final_comparison)(args,output)


if __name__ == '__main__':
    main()
