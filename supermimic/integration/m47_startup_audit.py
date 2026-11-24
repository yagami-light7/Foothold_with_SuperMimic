"""冻结 F2 Actor 的四种起步执行方式与实际物理子步诊断。"""
from argparse import ArgumentParser, Namespace
from contextlib import contextmanager
import json
import os
from pathlib import Path
from time import perf_counter

if __name__ == '__main__':
    os.environ['SUPERMIMIC_ROBOT'] = 'roban'

import numpy as np
import torch

from .m42v3_rollout import collect_trial
from .m43_evaluation import _save_archive, summarize_gait
from .m44_evaluation import DEFAULT_SEEDS, PERTURB_SEEDS, control_trace, evaluator, perturbed_start, scene
from .m4_evaluation import _swing_and_touchdown
from .m47_control import blend_startup_target, upper_joint_indices
from ..motion import CONTROL_DT
from ..roban import JOINT_NAMES


CHECKPOINTS = {
    'm43': ('outputs/m4_3/checkpoints/best.pt', 4500),
    'w002': ('outputs/m4_6/w002/checkpoints/iter_5000.pt', 5000),
    'w005': ('outputs/m4_6/w005/checkpoints/iter_4500.pt', 4500),
}
STARTUP_TICKS = {'A': 0, 'B': 0, 'C': 5, 'D': 10}
UPPER = upper_joint_indices(JOINT_NAMES)
ELBOW = JOINT_NAMES.index('zarm_r4_joint')
FIELDS = ('state','frame','action','previous_action','target','foot','force','contact','foot_target_error',
          'actuator_torque','joint_limit_force','planned_contact','termination')


@contextmanager
def execution(env, mode, seed, perturb):
    mdp = env.mdp
    mdp_target = mdp._reward_pd_target
    base_target = env.pd_target
    ticks = STARTUP_TICKS[mode]
    upper = torch.zeros_like(mdp.default_q, dtype=torch.bool)
    upper[UPPER] = True
    reset_q = None

    def initial(sim):
        nonlocal reset_q
        if perturb:
            perturbed_start(sim, seed)
        reset_q = sim.state()[4].clamp(mdp.kin.q_lo, mdp.kin.q_hi)
        if mode == 'B':
            # 原缓冲存 Actor 空间值，反算同一硬限位目标。观测历史仍重复真实首帧。
            sim.a_prev[:] = (reset_q - mdp.default_q) / mdp.alpha

    def mapped(a, xi=None):
        return blend_startup_target(base_target(a, xi), reset_q, env.ep_step, upper, ticks)

    def reward_target(a, frame):
        return blend_startup_target(mdp_target(a, frame), reset_q, (frame - 1).clamp_min(0), upper, ticks)

    if ticks:
        env.pd_target = mapped
        mdp._reward_pd_target = reward_target
    try:
        yield initial
    finally:
        if ticks:
            env.pd_target = base_target
            mdp._reward_pd_target = mdp_target


@contextmanager
def record_substeps(solver, rows):
    step = solver.solver.step

    def record(*args, **kwargs):
        step(*args, **kwargs)
        rows.append(torch.stack((solver.qpos[0, solver.model.jnt_qposadr[solver.joints]][UPPER],
                                 solver.qvel[0, solver.dofs][UPPER], solver.acceleration[0, solver.dofs][UPPER],
                                 solver.torque[0, solver.dofs][UPPER], solver.joint_limit_forces()[0, UPPER])))

    solver.solver.step = record
    try:
        yield
    finally:
        solver.solver.step = step


def diagnostics(trace, env, mode):
    mdp = env.mdp
    n = len(JOINT_NAMES)
    target = mdp.standing_pd_target(torch.as_tensor(trace['action'],device=env.device))
    duration = STARTUP_TICKS[mode]
    if duration:
        upper = torch.zeros_like(mdp.default_q, dtype=torch.bool)
        upper[UPPER] = True
        reset = torch.as_tensor(trace['state'][0,13:13+n],device=env.device).clamp(mdp.kin.q_lo,mdp.kin.q_hi)
        target = blend_startup_target(target, reset, torch.arange(len(target),device=env.device), upper, duration)
    control_trace(trace, mdp, env.motion, expected_target=target.cpu().numpy())
    speed = trace['state'][:, 13+n:]
    acc = np.diff(speed,axis=0)[:,UPPER] / CONTROL_DT
    act2 = np.diff(trace['action'][:,UPPER],n=2,axis=0)
    rms = lambda x: float(np.sqrt(np.square(x).mean())) if x.size else None
    signal = dict(arm_hand_acceleration_rms=rms(acc), arm_hand_jerk_rms=rms(np.diff(acc,axis=0)/CONTROL_DT),
                  action_diff2_rms=rms(act2), arm_hand_acceleration_p99=float(np.quantile(np.abs(acc),.99)) if len(acc) else None,
                  arm_hand_acceleration_max=float(np.abs(acc).max()) if len(acc) else None)
    if not len(acc):
        return signal
    # 从真实起始状态及已下发目标计算启动误差和第一控制步变化。
    error = trace['target'][0] - trace['state'][0, 13:13+n]
    k = UPPER.index(ELBOW)
    signal.update(first_elbow_error_rad=float(error[ELBOW]), first_elbow_v_rad_s=float(speed[1,ELBOW]),
                  first_elbow_acc_rad_s2=float(acc[0,k]), first_upper_error_rms=rms(error[UPPER]),
                  blend_end_acc_max_rad_s2=float(np.abs(acc[max(0,duration-1):duration+2]).max())
                  if duration and len(acc) >= duration+1 else None)
    return signal


def one_trial(env, actor, contacts, solver, mode, seed, perturb, *, substeps=None):
    with execution(env, mode, seed, perturb) as reset:
        if substeps is None:
            trace = collect_trial(env,actor,contacts,seed,solver=solver,capture_faults=True,
                                  capture_joint_limits=True,reset_hook=reset)
        else:
            env.max_episode_steps = 10
            with record_substeps(solver, substeps):
                trace = collect_trial(env,actor,contacts,seed,solver=solver,capture_faults=True,
                                      capture_joint_limits=True,reset_hook=reset)
    metrics = diagnostics(trace,env,mode)
    return trace, summarize_gait(trace,env.motion,env.mdp), metrics


def phase_metrics(output, reference):
    from ..kinematics import RobotKinematics
    from .m43_wm_audit import write_json
    from .m4_reference import F2MotionReference

    motion = F2MotionReference(reference,RobotKinematics('cpu'),'cpu')
    lifts, lands, sides = (value.numpy() for value in (motion.liftoff_time,motion.touchdown_time,motion.touchdown_side))
    feet = motion.at_time(lands)['foot_p'].numpy()
    report = json.loads((output/'results.json').read_text())
    with np.load(output/'probe'/'rollouts.npz',allow_pickle=False) as probe:
        q0,scale,lo,hi = (probe['m43_A_nominal_r1_s1001_'+key] for key in ('q0','alpha','lo','hi'))
    dt = CONTROL_DT
    with np.load(output/'rollouts.npz',allow_pickle=False) as archive:
        for name,run in report['runs'].items():
            regions = {k: {} for k in ('whole','startup','later','touchdown','perturb_recovery')}
            for row in run['rows']:
                prefix = f'{name}_s{row["seed"]}_'
                read = lambda key: archive[prefix+key]
                t, foot, contact = read('frame')*dt, read('foot'), read('contact')
                spans = []
                for i,(side,lift,land) in enumerate(zip(sides,lifts,lands)):
                    _, landing = _swing_and_touchdown(t,foot[:,side],contact[:,side],feet[i,side],lift,land,dt)
                    length = None
                    if landing is not None:
                        k = int(round(landing/dt))
                        off = np.flatnonzero(~contact[k:,side])
                        length = float((off[0] if len(off) else len(t)-k)*dt)
                    spans.append(dict(plan_s=float(land),actual_s=landing,loaded_s=length))
                row['landings'] = spans
                if row['termination'] in ('epa_horizon','fatal_dynamics','nonfinite'):
                    continue
                raw = read('action')[:,UPPER].astype(np.float64)
                target = read('target')[:,UPPER].astype(np.float64)
                hard = np.clip(q0 + scale*raw,lo,hi)
                qd = read('state')[:,13+len(JOINT_NAMES):][:,UPPER].astype(np.float64)
                qdd = np.diff(qd,axis=0)/dt
                # 速度差分用相邻状态中点，动作差分用奖励对应的最后一条指令时刻。
                series = dict(velocity=(qd,t),acceleration=(qdd,(t[1:]+t[:-1])*.5),
                              jerk=(np.diff(qdd,axis=0)/dt,t[1:-1]))
                for key, values in dict(raw=raw,effective=(target-q0)/scale,hard_target=hard,actual_target=target).items():
                    series[key+'_diff2'] = (np.diff(values,n=2,axis=0),t[2:-1])
                for key,(values,time) in series.items():
                    masks = dict(whole=np.ones_like(time,dtype=bool),startup=time<.2,later=time>=.2,
                                 touchdown=(np.abs(time[:,None]-lands)<=.100001).any(-1))
                    if '_perturb_' in name:
                        masks['perturb_recovery'] = time<lifts[0]
                    for phase,mask in masks.items():
                        if mask.any():
                            regions[phase].setdefault(key,[]).append(values[mask].ravel())
            run['execution_metrics'] = {}
            for phase, values in regions.items():
                if not values:
                    continue
                arrays = {key:np.concatenate(samples) for key,samples in values.items()}
                summary = {key+'_rms':float(np.sqrt(np.square(v).mean())) for key,v in arrays.items()}
                if 'acceleration' in arrays:
                    acc = np.abs(arrays['acceleration'])
                    summary.update(acceleration_p95=float(np.quantile(acc,.95)),
                                   acceleration_p99=float(np.quantile(acc,.99)),acceleration_max=float(acc.max()))
                run['execution_metrics'][phase] = summary
    write_json(output/'results.json',report)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--m4_reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('outputs/m4_7/startup_diagnosis'))
    parser.add_argument('--cases', nargs='+', choices=tuple(CHECKPOINTS), default=['m43'])
    parser.add_argument('--modes', nargs='+', choices=tuple(STARTUP_TICKS), default=list(STARTUP_TICKS))
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--probe', action='store_true', help='只测起步 0.20 s 并逐物理子步读回')
    parser.add_argument('--phase_metrics',action='store_true',help='从已完成的真实轨迹计算分相位指标')
    cli = parser.parse_args()
    torch.set_num_threads(2)
    if cli.phase_metrics:
        phase_metrics(cli.out,cli.m4_reference)
        return
    args = Namespace(m4_reference=cli.m4_reference,device='cuda:0',visualizer=None)
    output = cli.out
    output.mkdir(parents=True,exist_ok=True)
    if cli.probe:
        # 子步分开归档，独立运行的图模式闭环轨迹不受探针影响。
        output = output / 'probe'
        output.mkdir(exist_ok=True)
    metadata = output/'results.json'
    if metadata.exists():
        report = json.loads(metadata.read_text())
        with np.load(output/'rollouts.npz', allow_pickle=False) as previous:
            archive = {k:previous[k] for k in previous.files}
    else:
        report, archive = dict(upper_names=[JOINT_NAMES[i] for i in UPPER],runs={}), {}
    starts = {}
    for key,data in report['runs'].items():
        for seed in data['seeds']:
            if (field := f'{key}_s{seed}_state') in archive:
                initial = archive[field][0]
                np.testing.assert_allclose(initial, starts.setdefault(seed,initial), atol=1e-7,rtol=0)
    for name in cli.cases:
        checkpoint, i = CHECKPOINTS[name]
        with scene(args,Path(checkpoint),expected_iteration=i,capture_substeps=cli.probe) as (env,actor,contacts,solver,_,physics):
            current = json.loads(json.dumps(physics))
            if current != report.setdefault('physics',current):
                raise ValueError('冻结对照使用了不同物理配置')
            report['physics_dt'] = env.physics_dt
            for mode in cli.modes:
                for split, seeds in (('nominal',DEFAULT_SEEDS),('perturb',PERTURB_SEEDS)):
                    key = f'{name}_{mode}_{split}_r{cli.repeat}'
                    if key in report['runs']:
                        continue
                    started = perf_counter()
                    if cli.probe:
                        seeds = seeds[:1]
                    traces,rows,metrics = [],[],[]
                    for seed in seeds:
                        sub = [] if cli.probe else None
                        trace,row,measure = one_trial(env,actor,contacts,solver,mode,seed,split=='perturb',substeps=sub)
                        initial = trace['state'][0]
                        np.testing.assert_allclose(initial,starts.setdefault(seed,initial),atol=1e-7,rtol=0)
                        traces.append(trace)
                        rows.append(row)
                        metrics.append(dict(seed=seed,termination=row['termination'],matched=row['matched_touchdowns'],
                                            consecutive=row['consecutive_planned_touchdowns'],
                                            failures=[event['failure'] for event in row['footsteps']], **measure))
                        prefix = f'{key}_s{seed}_'
                        archive.update({prefix+k:trace[k] for k in FIELDS})
                        if sub:
                            values = torch.stack(sub).cpu().numpy()
                            if len(values) != int(len(trace['action'])*round(CONTROL_DT/env.physics_dt)):
                                raise RuntimeError('非图模式的物理子步采集长度不匹配')
                            # 列为 q、qd、solver qacc、驱动力矩、机械止挡约束力。
                            archive[prefix+'substeps'] = values
                            for k,val in dict(q0=env.mdp.default_q,alpha=env.mdp.alpha,qref=env.motion.q[0],
                                              lo=env.mdp.kin.q_lo,hi=env.mdp.kin.q_hi,effort=env.mdp.effort,
                                              kp=env.mdp.kp,kd=env.mdp.kd).items():
                                archive[prefix+k] = val[UPPER].cpu().numpy()
                            for k in ('raw_action','observation','goal','reward_terms','reward_term_names','raw_target'):
                                archive[prefix+k] = trace[k]
                        print(key, seed, row['steps'], row['termination'], row['matched_touchdowns'],
                              f'右肘首步加速度={measure.get("first_elbow_acc_rad_s2",0.):.1f}',flush=True)
                        if row['termination']=='fatal_dynamics':
                            break
                    agg = evaluator(traces,rows,len(seeds),env.mdp)['aggregate']
                    keep = ('valid_resets','completed','full_tracking_successes','first_swing_successes','first_touchdown_successes',
                            'matched_touchdowns','foot_position_rmse_m','root_xy_rmse_m','contact_agreement','actuator_saturation','epa','fatal')
                    report['runs'][key] = dict(checkpoint=checkpoint,iteration=i,seeds=seeds,rows=metrics,
                                               wall_s=perf_counter()-started, aggregate={k:agg.get(k) for k in keep})
                    _save_archive(output,report,archive)
if __name__ == '__main__':
    main()

