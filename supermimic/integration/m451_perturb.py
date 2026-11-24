"""F2 冻结策略的配对初态扰动、强度扫描和失衡时序。"""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

if __name__ == '__main__':
    os.environ['SUPERMIMIC_ROBOT'] = 'roban'

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from ..roban import JOINT_NAMES
from ..rotations import mat_to_quat, so3_exp
from .m42v3_rollout import collect_trial
from .m43_evaluation import summarize_gait
from .m44_evaluation import PERTURB_LIMITS, baseline_checkpoints, control_trace, scene
from .m45_upper import upper_metrics


BASE_SEEDS = tuple(range(1101, 1107))
EXTRA_SEEDS = tuple(range(1107, 1113))
FACTORS = ('xy', 'height', 'roll_pitch', 'yaw', 'joint_position', 'root_velocity', 'joint_velocity', 'combined')
SCALES = (0., .5, 1., 1.5, 2.)
OUTPUT = Path('outputs/m4_5_1')
COMPARISON = ('m43_best', 'm45_final')
ANKLES = [JOINT_NAMES.index(f'leg_{side}5_joint') for side in ('l', 'r')]
ARMS = [i for i, n in enumerate(JOINT_NAMES) if n.startswith('zarm_')]


def perturbation(seed, factor, scale):
    """保留 M4.5 的整组抽样和抽样顺序，再屏蔽无关轴。"""
    factor = factor.removesuffix('_scan')
    rng = np.random.default_rng(seed)
    v = PERTURB_LIMITS
    dp = np.r_[rng.uniform(-v['root_xy_m'], v['root_xy_m'], 2), rng.uniform(.006, v['root_lift_m'])]
    da = np.r_[rng.uniform(-v['root_roll_pitch_deg'], v['root_roll_pitch_deg'], 2),
               rng.uniform(-v['root_yaw_deg'], v['root_yaw_deg'])]
    dv = rng.uniform(-v['root_v_m_s'], v['root_v_m_s'], 3)
    dw = rng.uniform(-v['root_w_rad_s'], v['root_w_rad_s'], 3)
    dq = rng.uniform(-v['joint_position_rad'], v['joint_position_rad'], len(JOINT_NAMES))
    dqd = rng.uniform(-v['joint_speed_rad_s'], v['joint_speed_rad_s'], len(JOINT_NAMES))
    if factor != 'combined':
        dp[:2] *= factor == 'xy'
        dp[2] *= factor == 'height'
        da[:2] *= factor == 'roll_pitch'
        da[2] *= factor == 'yaw'
        dv *= factor == 'root_velocity'
        dw *= factor == 'root_velocity'
        dq *= factor == 'joint_position'
        dqd *= factor == 'joint_velocity'
    return tuple(scale*x for x in (dp, da, dv, dw, dq, dqd))


def apply_perturbation(env, seeds, factor, scale):
    # 仿真器完成正规 Reset 后只写受扰初态，保持原参考时钟和策略观测构造。
    if scale == 0.:
        return
    p, R, v, w, q, qd = env.state()
    dp, da, dv, dw, dq, dqd = (q.new_tensor(np.stack(col))
                                for col in zip(*(perturbation(s, factor, scale) for s in seeds)))
    q1 = q+dq
    if not bool(((q1 >= env.mdp.kin.q_lo) & (q1 <= env.mdp.kin.q_hi)).all()):
        raise ValueError('扰动产生的初态超出机械关节限位')
    rotation = mat_to_quat(so3_exp(da.deg2rad()) @ R)[:, [1, 2, 3, 0]]
    env.robot.write_root_link_pose_to_sim_index(root_pose=torch.cat((p + env.scene.env_origins + dp, rotation), -1))
    env.robot.write_root_link_velocity_to_sim_index(root_velocity=torch.cat((v+dv, w+dw), -1))
    env.robot.write_joint_position_to_sim_index(position=q1[:, env.sim_to_canonical])
    env.robot.write_joint_velocity_to_sim_index(velocity=(qd+dqd)[:, env.sim_to_canonical])
    env.scene.write_data_to_sim()
    env.refresh()


def track_com(env):
    data = env.robot.data
    mass = data.body_mass.torch.reshape(env.N, -1, 1)
    weights = mass/mass.sum(1, keepdim=True)
    positions = data.body_com_pos_w.torch - env.scene.env_origins[:, None]
    velocity = data.body_com_vel_w.torch[..., :3]
    return (weights*positions).sum(1), (weights*velocity).sum(1)


class DiagnosticContacts:
    """原接触读取旁路记录质心和 solver，保留到下一动作前的暖启动。"""

    def __init__(self, env, contacts, solver):
        self.env, self.contacts, self.solver = env, contacts, solver
        self.rows = {}

    def record(self):
        p, v = track_com(self.env)
        for key, value in (('com_position', p), ('com_velocity', v), ('solver_nefc', self.solver.nefc),
                           ('solver_niter', self.solver.niter), ('solver_overflow', self.solver.overflow),
                           ('constraint_force', self.solver.constraint_force[:, self.solver.dofs])):
            self.rows.setdefault(key, []).append(value[0].cpu().numpy().copy())

    def read(self):
        values = self.contacts.read()
        self.record()
        return values


def collect_single(env, pi, contacts, solver, seed, factor, scale):
    observed = DiagnosticContacts(env, contacts, solver)

    def reset_hook(sim):
        apply_perturbation(sim, [seed], factor, scale)
        observed.record()

    trace = collect_trial(env, pi, observed, seed, solver=solver, capture_faults=True,
                          reset_hook=reset_hook, capture_joint_limits=True)
    trace.update({key: np.asarray(v)[:len(trace['state'])] for key, v in observed.rows.items()})
    return trace


def iter_traces(env, pi, contacts, solver, seeds, factor, scale, *, batched):
    if batched:
        from .m451_repro import collect_parallel

        return collect_parallel(env, pi, contacts, solver, seeds,
                                reset_hook=lambda sim: apply_perturbation(sim, seeds, factor, scale))
    return [collect_single(env, pi, contacts, solver, s, factor, scale) for s in seeds]


def first_run(mask, time, frames=3):
    starts = np.flatnonzero(np.convolve(mask, np.ones(frames, dtype=int), 'valid') == frames) if len(mask) >= frames else []
    return float(time[starts[0]]) if len(starts) else None


def early_metrics(trace, mdp):
    t, s = trace['time'], trace['state']
    R = Rotation.from_quat(s[:, 3:7][:, [1, 2, 3, 0]])
    tilt = R.as_euler('xyz', degrees=True)[:, :2]
    planned = trace['planned_contact'].astype(bool)
    actual = trace['contact'].astype(bool)
    measured = trace['contact_measured'].astype(bool)
    # 初始接触尚在形成，失载判读从 0.2 s 开始。三帧连续越阈值排除孤立脉冲。
    loss = (planned & ~actual).any(-1) & measured & (t >= .2-1e-8)
    drop = trace['planned_root_p'][:, 2]-s[:, 2] > .035
    foot = (trace['foot_target_error'] > .08).any(-1) & (t >= .2-1e-8)
    lift, land = float(mdp.motion.liftoff_time[0]), float(mdp.motion.touchdown_time[0])
    phases = {}
    for phase, lo, hi in (('initial_support',0.,lift), ('first_swing',lift,land),
                          ('first_touchdown',land-.1,land+.2)):
        f = (t >= lo-1e-8) & (t < hi-1e-8)
        if not f.any():
            continue
        fa = f[:-1]
        fm = f & measured
        qd = s[f, 34:]
        com = trace.get('com_position')
        cv = trace.get('com_velocity')
        phases[phase] = dict(
            frames=int(f.sum()), root_z_min_m=float(s[f,2].min()),
            tilt_abs_max_deg=np.abs(tilt[f]).max(0).tolist(),
            root_linear_max_m_s=float(np.linalg.norm(s[f,7:10], axis=-1).max()),
            root_angular_max_rad_s=float(np.linalg.norm(s[f,10:13], axis=-1).max()),
            supporting_right_normal_min_n=float(trace['force'][fm,1,2].min()) if fm.any() else None,
            supporting_right_loaded_fraction=float(actual[fm,1].mean()) if fm.any() else None,
            left_clearance_max_m=float(trace['foot'][f,0,2].max()),
            left_xy_error_max_m=float(np.linalg.norm(trace['foot'][f,0,:2]-trace['planned_foot_p'][f,0,:2], axis=-1).max()),
            com_z_min_m=float(com[f,2].min()) if com is not None else None,
            com_speed_max_m_s=float(np.linalg.norm(cv[f], axis=-1).max()) if cv is not None else None,
            arm_speed_rms_rad_s=float(np.sqrt(np.mean(qd[:, ARMS]**2))),
            waist_speed_rms_rad_s=float(np.sqrt(np.mean(qd[:, 0]**2))),
            upper_raw_action_rms=float(np.sqrt(np.mean(trace['raw_action'][fa][:, [0,*ARMS]]**2))) if fa.any() else None,
            ankle_torque_peak_nm=np.abs(trace['actuator_torque'][f][:, ANKLES]).max(0).tolist(),
            upper_clamp_ratio=float(trace['clamp_mask'][fa][:, [0,*ARMS]].mean()) if fa.any() else None)
    return dict(root_drop_start_s=first_run(drop,t), expected_support_loss_start_s=first_run(loss,t),
                foot_error_start_s=first_run(foot,t), phases=phases)


def compare_early(candidate, baseline):
    t = candidate['time'][:len(baseline['time'])]
    s, r = candidate['state'][:len(t)], baseline['state'][:len(t)]
    q, R = (Rotation.from_quat(a[:, 3:7][:, [1,2,3,0]]) for a in (s,r))
    angle = np.rad2deg((R.inv()*q).magnitude())
    df = np.linalg.norm(candidate['foot'][:len(t)]-baseline['foot'][:len(t)],axis=-1).max(-1)
    return dict(
        root_position_delta_s=first_run(np.linalg.norm(s[:,:3]-r[:,:3],axis=-1)>.03,t),
        root_orientation_delta_s=first_run(angle>4.,t),
        foot_position_delta_s=first_run(df>.04,t),
        arm_angle_delta_s=first_run(np.sqrt(((s[:,13:34]-r[:,13:34])[:,ARMS]**2).mean(-1))>.20,t),
        support_force_delta_s=first_run((np.abs(candidate['force'][:len(t),:,2]-baseline['force'][:len(t),:,2]).max(-1)>50.)
                                       & (t>=.20-1e-8),t))


def prefix(label, factor, scale, seed):
    return f'{label}_{factor}_x{scale:g}_s{seed}_'.replace('.', 'p')


def case_key(factor, scale):
    return f'{factor}_x{scale:g}'


def summary(rows):
    total = len(rows)
    complete = sum(r['survived_full_reference'] and r['consecutive_planned_touchdowns']==4 for r in rows)
    vals = {key: [r[key] for r in rows if r.get(key) is not None] for key in
            ('matched_touchdowns','foot_position_rmse_m','root_xy_rmse_m','contact_agreement','clamp_ratio',
             'torque_saturation_ratio')}
    return dict(n=total, full=complete, first_touchdown=sum(r['first_touchdown'] for r in rows),
                first_swing=sum(r['first_swing'] for r in rows), survived=sum(r['survived_full_reference'] for r in rows),
                root_height=sum('root_height' in r['termination'] for r in rows),
                faults=sum(r['termination'] in ('epa_horizon','fatal_dynamics','nonfinite') for r in rows),
                **{k: float(np.mean(v)) if v else None for k,v in vals.items()})


def store_bundle(output, report, archive):
    with TemporaryDirectory(dir=output) as tmp:
        tmp = Path(tmp)
        p, q = tmp/'perturb_results.json', tmp/'perturb_rollouts.npz'
        p.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)+'\n')
        np.savez_compressed(q, **archive)
        q.replace(output/q.name)
        p.replace(output/p.name)


def run_suite(args, requested, models, report, archive, *, batched):
    starts = {}
    for label in models:
        path, expected = models[label]
        model = report['models'].setdefault(label, dict(path=str(path), iteration=expected, trials={}, duration_s=0.))
        pending = [(f,x,[s for s in seeds if prefix(label,f,x,s)+'state' not in archive])
                   for f,x,seeds in requested]
        pending = [(f,x,seeds) for f,x,seeds in pending if seeds]
        if not pending:
            continue
        begin = perf_counter()
        with scene(args, path, expected_iteration=expected, **({'num_envs': min(6, args.worlds)} if batched else {})) as (
                env, pi, contacts, solver, iteration, physics):
            loaded_physics = json.loads(json.dumps(physics))
            if env.mdp.unclipped_pd_target or loaded_physics != report.setdefault('physics', loaded_physics):
                raise ValueError('冻结对照没有使用同一 Hard Clamp 物理契约')
            for factor, scale, seeds in pending:
                key = case_key(factor,scale)
                entry = model['trials'].setdefault(key, dict(rows=[], path='parallel' if batched else 'single'))
                if batched and len(seeds) % env.N:
                    raise ValueError('并行扫描的样本数量必须能被环境数量整除')
                for offset in range(0, len(seeds), env.N if batched else len(seeds)):
                    subset = seeds[offset:offset+env.N if batched else len(seeds)]
                    traces = iter_traces(env, pi, contacts, solver, subset, factor, scale, batched=batched)
                    for seed,t in zip(subset,traces,strict=True):
                        init = starts.setdefault((factor,scale,seed),t['state'][0])
                        np.testing.assert_allclose(init,t['state'][0],rtol=0,atol=1e-7)
                        control_trace(t,env.mdp,env.motion)
                        row = summarize_gait(t,env.motion,env.mdp)
                        row['upper_body'] = upper_metrics(t,env.motion)
                        row['early'] = early_metrics(t,env.mdp)
                        stem = prefix(label,factor,scale,seed)
                        archive.update({stem+k:v for k,v in t.items()})
                        if label.startswith('m45'):
                            base = prefix('m43_best',factor,scale,seed)
                            if base+'state' in archive:
                                paired = {k:archive[base+k] for k in ('state','time','foot','force')}
                                np.testing.assert_allclose(paired['state'][0],t['state'][0],rtol=0,atol=1e-7)
                                row['first_divergence'] = compare_early(t,paired)
                        entry['rows'].append(row)
                        print(f'{label} {key} seed={seed} 接触={row["matched_touchdowns"]}/4, '
                              f'首次={int(row["first_touchdown"])}, 截止={row["episode_length_s"]:.2f}s '
                              f'{row["termination"]}',flush=True)
                entry['aggregate'] = summary(entry['rows'])
        model['duration_s'] += perf_counter()-begin
        store_bundle(args.out,report,archive)


def select_risks(report):
    a,b = (report['models'][n]['trials'] for n in COMPARISON)
    gaps = {}
    for name in FACTORS:
        x,y = (p[case_key(name,1.)]['aggregate'] for p in (a,b))
        gaps[name] = [x['first_touchdown']-y['first_touchdown'], x['full']-y['full'],
                      y['root_height']-x['root_height'],
                      y['foot_position_rmse_m']-x['foot_position_rmse_m']]
    # 组合是原问题。附加扫描首落脚损失最大的两类单因素，并保留全部初筛证据。
    ordered = sorted(FACTORS[:-1], key=lambda name: gaps[name], reverse=True)
    selected = [name for name in ordered if max(gaps[name][:3]) >= 2][:2]
    if not selected:
        selected = [ordered[0]]
    return ['combined',*selected], gaps


def main():
    from isaaclab.app import add_launcher_args

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m4_reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    parser.add_argument('--stage', choices=('combined','screen','scan'), required=True)
    parser.add_argument('--worlds', type=int, choices=(1,2,3,6), default=6)
    add_launcher_args(parser)
    args = parser.parse_args()
    if torch.device(args.device).type != 'cuda':
        parser.error('还原历史扰动需要 CUDA Newton')
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out/'perturb_results.json'
    if path.is_file():
        report = json.loads(path.read_text())
        with np.load(args.out/'perturb_rollouts.npz', allow_pickle=False) as data:
            archive = {k:data[k] for k in data.files}
    else:
        archive = {}
        report = dict(models={}, base_seeds=BASE_SEEDS, extra_seeds=EXTRA_SEEDS, scales=SCALES,
                      perturb_limits=dict(PERTURB_LIMITS,root_lift_min_m=.006),
                      versions={k: version(k) for k in ('isaaclab','newton','mujoco-warp','warp-lang')})
    m43_best,m43_final = baseline_checkpoints(Path('outputs/m4_3'))
    models = {'m43_best':(m43_best,4500), 'm43_final':(m43_final,5000),
              'm45_4500':(Path('outputs/m4_5/checkpoints/iter_4500.pt'),4500),
              'm45_final':(Path('outputs/m4_5/checkpoints/iter_5000.pt'),5000)}
    if args.stage == 'combined':
        run_suite(args,[('combined',1.,BASE_SEEDS)],models, report, archive, batched=False)
        return
    pairs = {k:models[k] for k in COMPARISON}
    if args.stage == 'screen':
        run_suite(args,[(f,1.,BASE_SEEDS) for f in FACTORS[:-1]],pairs,report,archive,batched=args.worlds>1)
        return
    selected,gaps = select_risks(report)
    report['risk_factors'],report['screen_risk_gaps'] = selected,gaps
    # 同一批新增种子跨尺度配对，组合的初筛结果保留原单环境协议。
    requested = [(f+'_scan',x,EXTRA_SEEDS) for f in selected for x in SCALES]
    run_suite(args,requested,pairs,report,archive,batched=args.worlds>1)


if __name__ == '__main__':
    main()
