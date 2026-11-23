"""与训练使用同一控制路径的 Newton/MJWarp 固定动作对照。"""
import argparse
import json
import os
from collections import deque
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

if __name__ == '__main__':
    os.environ['SUPERMIMIC_ROBOT'] = 'roban'

import numpy as np
import torch
import warp as wp

from ..mdp import flatten_state, unflatten_state
from ..motion import CONTROL_DT
from ..roban import JOINT_NAMES, model_contract
from ..rotations import mat_to_quat
from .m4_task import (JOINT_LIMIT_SLACK_RAD, ROOT_ANGULAR_SPEED_MAX_RAD_S,
                      ROOT_SPEED_MAX_M_S, ROOT_STEP_MAX_M)


class SolverReadback:
    """从同一个 MJWarp 实例读取实际施力及约束数据。"""

    def __init__(self, env):
        self.solver = env.physics._solver
        self.model = self.solver.mj_model
        data = self.solver.mjw_data
        names = [self.model.joint(i).name for i in range(self.model.njnt)]
        self.joints = []
        for name in JOINT_NAMES:
            indices = [i for i, value in enumerate(names) if value == name or value.endswith(('/' + name, '_' + name))]
            if len(indices) != 1:
                raise ValueError(f'MJWarp 关节索引不唯一：{name} {indices}')
            self.joints.append(indices[0])
        self.dofs = self.model.jnt_dofadr[self.joints].tolist()
        self.qpos = wp.to_torch(data.qpos)
        self.qvel = wp.to_torch(data.qvel)
        self.torque = wp.to_torch(data.qfrc_actuator)
        self.extra_force = wp.to_torch(data.qfrc_applied)
        self.constraint_force = wp.to_torch(data.qfrc_constraint)
        self.acceleration = wp.to_torch(data.qacc)
        self.warmstart = wp.to_torch(data.qacc_warmstart)
        self.nefc = wp.to_torch(data.nefc)
        self.niter = wp.to_torch(data.solver_niter)
        self.overflow = wp.to_torch(data.overflow)
        self.nacon = wp.to_torch(data.nacon)
        self.world = wp.to_torch(data.contact.worldid)
        self.geom = wp.to_torch(data.contact.geom)
        self.dist = wp.to_torch(data.contact.dist)
        self.pos = wp.to_torch(data.contact.pos)
        self.frame = wp.to_torch(data.contact.frame)
        self.address = wp.to_torch(data.contact.efc_address)[:, 0]
        self.ground = torch.as_tensor(self.model.geom_bodyid == 0, device=env.device)
        self.nenv = env.N

    def contacts(self):
        size = min(int(self.nacon[0]), self.world.numel())
        ids = self.world[:size].long()
        active = self.address[:size] >= 0
        ground = self.ground[self.geom[:size].long()].any(-1) & active
        counts = torch.zeros(self.nenv, 3, device=ids.device, dtype=torch.int32)
        counts.index_add_(0, ids, torch.stack((torch.ones_like(active), active, ground), -1).int())
        min_dist = torch.zeros(self.nenv, device=ids.device)
        min_dist.scatter_reduce_(0, ids, self.dist[:size], reduce='amin')
        return counts, min_dist

    def contact_details(self, env_ids):
        size = min(int(self.nacon[0]), self.world.numel())
        rows = torch.isin(self.world[:size], env_ids).nonzero(as_tuple=True)[0]
        return dict(contact_env_ids=self.world[rows], contact_geom_pairs=self.geom[rows],
                    contact_pos_m=self.pos[rows], contact_normal=self.frame[rows, 0],
                    contact_dist_m=self.dist[rows], contact_penetration_m=(-self.dist[rows]).clamp_min(0.),
                    contact_constraint_address=self.address[rows])

    def snapshot(self, contact_stats):
        counts, depth = contact_stats
        return dict(torque=self.torque[:, self.dofs], extra_force=self.extra_force[:, self.dofs],
                    constraint_force=self.constraint_force[:, self.dofs], qacc=self.acceleration[:, self.dofs],
                    nefc=self.nefc.clone(), solver_iterations=self.niter.clone(), overflow=self.overflow.clone(),
                    contacts=counts, contact_depth=depth)

    def contact_impulses(self, dt):
        from mujoco_warp import contact_force

        size = min(int(self.nacon[0]), self.world.numel())
        ids = torch.arange(size, device=self.world.device, dtype=torch.int32)
        peak = torch.zeros(self.nenv, device=self.world.device)
        pair = torch.full((self.nenv, 2), -1, device=self.world.device, dtype=torch.int32)
        if size:
            force = wp.zeros(size, dtype=wp.spatial_vector, device=str(self.world.device))
            contact_force(self.solver.mjw_model, self.solver.mjw_data, wp.from_torch(ids), False, force)
            # 接触局部坐标的第一分量为法向力，乘子步时间估计法向冲量。
            w = self.world[:size].long()
            impulse = wp.to_torch(force)[:, 0] * dt
            peak.scatter_reduce_(0, w, impulse, reduce='amax')
            first = torch.full((self.nenv,), size, device=ids.device, dtype=torch.int32)
            matches = (impulse > 0.) & (impulse == peak[w])
            first.scatter_reduce_(0, w, torch.where(matches, ids, size), reduce='amin')
            pair = torch.where(first[:, None] < size, self.geom[first.clamp_max(size - 1).long()], pair)
        return peak, pair

    @contextmanager
    def record_substeps(self, rows):
        step = self.solver.step

        def record(state_in, state_out, control, contacts, dt, *args, **kwargs):
            step(state_in, state_out, control, contacts, dt, *args, **kwargs)
            sample = self.snapshot(self.contacts())
            impulse, pair = self.contact_impulses(dt)
            sample.update(qpos=self.qpos, qvel=self.qvel, normal_impulse=impulse, impulse_geom_pair=pair)
            rows.append({key: value.clone() for key, value in sample.items()})

        self.solver.step = record
        try:
            yield
        finally:
            self.solver.step = step

    def audit(self, env):
        import mujoco

        source, rows = model_contract()
        j, d = self.joints, self.dofs
        m, data = self.model, self.solver.mjw_data
        gpu = self.solver.mjw_model
        ranges = gpu.jnt_range.numpy()[:, j]
        np.testing.assert_allclose(ranges, np.broadcast_to(m.jnt_range[j], ranges.shape), atol=1e-6)
        solref = gpu.jnt_solref.numpy()[:, j]
        np.testing.assert_allclose(solref, np.broadcast_to(m.jnt_solref[j], solref.shape), atol=1e-6)
        np.testing.assert_allclose(m.jnt_range[j], [x['range'] for x in rows], atol=1e-6)
        axis_delta = float(abs(m.jnt_axis[j] - [x['axis'] for x in rows]).max())
        assert m.jnt_limited[j].all() and (m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE).all()
        # 用相同起态计算固定胸部时的关节惯量块及自由关节等效惯量。
        from ..roban import FOOT_BODY_NAMES, foot_geometry
        source_data = mujoco.MjData(source)
        p, R = env.motion.ref_state(torch.zeros(1, dtype=torch.long, device=env.device))[:2]
        source_data.qpos[:3] = p[0].cpu().numpy()
        source_data.qpos[3:7] = mat_to_quat(R)[0].cpu().numpy()
        source_data.qpos[[row['qposadr'] for row in rows]] = env.motion.q[0].cpu().numpy()
        mujoco.mj_forward(source, source_data)
        mass = np.empty((source.nv, source.nv))
        mujoco.mj_fullM(source, source_data, mass)
        sd = [row['dofadr'] for row in rows]
        fixed = mass[np.ix_(sd, sd)]
        effective = 1 / np.diag(np.linalg.inv(fixed))
        foot_clearance = {}
        for body, geom in zip(FOOT_BODY_NAMES, foot_geometry()):
            b = source.body(body).id
            positions = source_data.xpos[b] + np.asarray(geom['corners']) @ source_data.xmat[b].reshape(3, 3).T
            foot_clearance[body] = float(positions[:, 2].min())
        selected = {}
        for name in ('leg_r1_joint', 'leg_r3_joint', 'zarm_r2_joint'):
            i = JOINT_NAMES.index(name)
            motors = np.flatnonzero((m.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT) &
                                     (m.actuator_trnid[:, 0] == j[i]))
            selected[name] = dict(range=m.jnt_range[j[i]].tolist(), axis=m.jnt_axis[j[i]].tolist(),
                                  source_axis=rows[i]['axis'],
                                  solref=m.jnt_solref[j[i]].tolist(), solimp=m.jnt_solimp[j[i]].tolist(),
                                  source_solref=source.jnt_solref[rows[i]['mj_joint_id']].tolist(),
                                  armature=float(m.dof_armature[d[i]]), passive_damping=float(m.dof_damping[d[i]]),
                                  friction_loss=float(m.dof_frictionloss[d[i]]),
                                  inertia_fixed_diagonal=float(fixed[i, i]), inertia_fixed_effective=float(effective[i]),
                                  drive_gain=m.actuator_gainprm[motors, 0].tolist(),
                                  drive_bias=m.actuator_biasprm[motors, :3].tolist(),
                                  force_limits=m.actuator_forcerange[motors].tolist(),
                                  force_limited=m.actuator_forcelimited[motors].tolist(),
                                  joint_force_limits=m.jnt_actfrcrange[j[i]].tolist(),
                                  joint_force_limited=bool(m.jnt_actfrclimited[j[i]]))
        return dict(joints=selected, max_axis_delta=axis_delta, num_actuators=m.nu, limit_count=int(m.jnt_limited[j].sum()),
                    robot_mass_kg=float(m.body_mass.sum()), original_mass_kg=float(source.body_mass.sum()),
                    reset_sole_corner_min_z=foot_clearance,
                    naconmax=int(data.naconmax), njmax=int(data.njmax), solver_limit=m.opt.iterations,
                    tolerance=float(m.opt.tolerance), integrator=int(m.opt.integrator),
                    geometry={m.geom(i).name: [float(x) for x in m.geom_friction[i]] for i in range(m.ngeom)
                              if m.geom_bodyid[i] == 0 or 'leg_l6_link' in m.body(m.geom_bodyid[i]).name})


def pack_step(before, after, raw, action, target, prev_target, mdp):
    error = target - before[4]
    proportional, damping = mdp.kp * error, -mdp.kd * before[5]
    return dict(before=flatten_state(before), after=flatten_state(after), raw=raw, action=action, target=target,
                target_delta=target - prev_target, position_error=error, proportional=proportional, damping=damping,
                torque_theory=proportional + damping, qd=after[5], acceleration_fd=(after[5] - before[5]) / CONTROL_DT)


class TrainingDynamics:
    """只在启用诊断时缓存八个控制步，优先保留碰撞和停机快照。"""

    def __init__(self, env, path, config):
        self.env, self.path, self.seed = env, path, config.seed
        self.config = json.dumps(vars(config), ensure_ascii=False)
        self.probe = SolverReadback(env)
        self.rows = deque(maxlen=8)
        self.prev_target = env.state()[4]
        self.saved_severity = 0
        self.standing_offset = env.mdp.action_mapping == 'standing_offset'
        self.unclipped_target = env.mdp.unclipped_pd_target
        self.sums = torch.zeros(7, dtype=torch.float64, device=env.device)
        self.maxima = torch.zeros(9, device=env.device)
        self.swing_aware = bool(getattr(config, 'roban_reward_config', {}).get('swing_aware'))
        if self.swing_aware:
            self.action_samples, self.speed_samples = [], []
            self.joint_clamps = torch.zeros(len(JOINT_NAMES), device=env.device)
            self.effective_action_sq = torch.zeros((), device=env.device)
        if self.unclipped_target:
            self.target_excess_samples = []
            self.raw_outside = torch.zeros(len(JOINT_NAMES), device=env.device)
            self.target_excess_max = torch.zeros_like(self.raw_outside)

    def set_action(self, logits, action, iteration, sim_steps):
        self.logits, self.action = logits, action
        self.iteration, self.sim_steps = iteration, sim_steps
        self.warmstart_before = self.probe.warmstart.clone()
        self.root_quat_before = self.env.robot.data.root_link_pose_w.torch[:, 3:7].clone()

    def record(self, before, after, target, excess, invalid, fatal, epa_recoverable):
        env, probe = self.env, self.probe
        previous = torch.where(env.ep_step[:, None] == 0, before[4], self.prev_target)
        self.prev_target = target
        row = pack_step(before, after, self.logits, self.action, target, previous, env.mdp)
        row.update(probe.snapshot(probe.contacts()), frame=env.f.clone(),
                   warmstart_before=self.warmstart_before, root_quat_before_xyzw=self.root_quat_before,
                   solver_qpos_after=probe.qpos.clone(), solver_qvel_after=probe.qvel.clone(),
                   joint_limit_excess=excess.clamp_min(0.), invalid=invalid, fatal=fatal,
                   epa_recoverable=epa_recoverable)
        self.rows.append(row)
        torque_ratio = row['torque'].abs() / env.mdp.effort
        saturated_action = (env.mdp.target_clip_mask(self.action) if self.standing_offset else self.action.abs() >= .95)
        if self.swing_aware:
            # 每控制步采少量轮换世界估计分位数，全批累计逐关节裁剪。
            stride = max(1, env.N // 8)
            sample = slice(len(self.action_samples) % stride, None, stride)
            self.action_samples.append(self.logits[sample].abs())
            self.speed_samples.append(after[5][sample].abs())
            self.joint_clamps += saturated_action.sum(0)
            self.effective_action_sq += ((target-env.mdp.default_q)/env.mdp.alpha).square().sum()
            if self.unclipped_target:
                kin = env.mdp.kin
                target_excess = torch.maximum(kin.q_lo - target, target - kin.q_hi).clamp_min(0.)
                self.raw_outside += (target_excess > 0.).sum(0)
                self.target_excess_samples.append(target_excess[sample].clone())
                self.target_excess_max = torch.maximum(self.target_excess_max, target_excess.amax(0))
        self.sums += torch.stack((self.logits.double().sum(), self.logits.double().square().sum(),
                                  saturated_action.sum(), (torque_ratio >= .98).sum(),
                                  invalid.sum(), fatal.sum(), self.logits.new_tensor(self.logits.numel())))
        self.maxima = torch.maximum(self.maxima, torch.stack((self.logits.abs().max(), row['target_delta'].abs().max(),
                                    excess.max().clamp_min(0.), after[5].abs().max(), after[2].norm(dim=-1).max(),
                                    after[3].norm(dim=-1).max(), row['contacts'][:, 0].max(), probe.nefc.max(),
                                    probe.niter.max())))
        if self.saved_severity == 3 or not invalid.any():
            return
        if fatal.any():
            severity, mask = 3, fatal
        elif epa_recoverable.any():
            severity, mask = 2, epa_recoverable
        else:
            severity, mask = 1, invalid
        if severity > self.saved_severity:
            ids = mask.nonzero(as_tuple=True)[0][:4]
            samples = {key: torch.stack([entry[key][ids] for entry in self.rows]).cpu().numpy() for key in row}
            # 接触表来自末子步，EPA 中途失败的几何体对可能没有生成接触记录。
            samples.update({key: value.cpu().numpy() for key, value in probe.contact_details(ids).items()})
            samples.update(env_ids=ids.cpu().numpy(), seed=self.seed, iteration=self.iteration,
                           sim_steps=self.sim_steps, joint_names=JOINT_NAMES, physics_dt=env.physics_dt,
                           geom_names=[probe.model.geom(i).name for i in range(probe.model.ngeom)],
                           geom_body_ids=probe.model.geom_bodyid, geom_types=probe.model.geom_type,
                           geom_sizes=probe.model.geom_size, contact_snapshot_substep=round(CONTROL_DT / env.physics_dt),
                           repro_config=self.config, torch_rng_state=torch.get_rng_state().numpy(),
                           torch_cuda_rng_state=torch.cuda.get_rng_state(env.device).numpy(),
                           versions=json.dumps({key: version(key) for key in ('isaaclab', 'newton', 'mujoco-warp', 'warp-lang')}),
                           njmax=int(probe.solver.mjw_data.njmax), naconmax=int(probe.solver.mjw_data.naconmax))
            tmp = self.path.with_suffix('.tmp')
            with tmp.open('wb') as handle:
                np.savez_compressed(handle, **samples)
            tmp.replace(self.path)
            self.saved_severity = severity
            print(f'M4 Newton 异常快照已写入 {self.path}', flush=True)

    def report(self):
        total, square, saturated_action, saturation, invalid, fatal, count = self.sums.tolist()
        raw_max, delta, excess, qd, v, w, contacts, nefc, niter = self.maxima.tolist()
        mean = total / count if count else 0.
        std = max(0., square / max(count, 1.) - mean**2)**.5
        self.sums.zero_()
        self.maxima.zero_()
        action_ratio = saturated_action / max(count, 1.)
        torque_ratio = saturation / max(count, 1.)
        label = 'target_clamp' if self.standing_offset else 'near_one'
        message = (f'raw {mean:.3g}±{std:.3g} max {raw_max:.3g} | {label} {action_ratio:.2%} '
                   f'| invalid {int(invalid)} fatal {int(fatal)} | Δtarget {delta:.3g}rad limit {excess:.3g}rad '
                   f'| qd {qd:.3g}rad/s v {v:.3g}m/s w {w:.3g}rad/s | sat {torque_ratio:.2%} '
                   f'| contacts {int(contacts)} nefc {int(nefc)} niter {int(niter)}')
        metrics = dict(actor_raw_mean=mean, actor_raw_std=std, actor_raw_abs_max=raw_max,
                       pd_target_clamp_ratio=action_ratio, torque_saturation_ratio=torque_ratio,
                       joint_limit_excess_rad=excess, invalid_transitions=int(invalid), physics_exceptions=int(fatal))
        if self.swing_aware and count:
            p95, p99 = torch.quantile(torch.cat(self.action_samples).flatten(), self.maxima.new_tensor([.95, .99])).tolist()
            qd99 = float(torch.quantile(torch.cat(self.speed_samples).flatten(), .99))
            metrics.update(actor_raw_abs_p95=p95, actor_raw_abs_p99=p99, joint_speed_max_rad_s=qd,
                           joint_speed_p99_rad_s=qd99, wm_effective_action_rms=float((self.effective_action_sq / count).sqrt()))
            rates = (self.joint_clamps / (count / len(JOINT_NAMES))).tolist()
            metrics.update({f'clamp_{name}': rate for name, rate in zip(JOINT_NAMES, rates)})
            if self.unclipped_target:
                # 分位数包含未越界的零值，峰值使用完整训练批次。
                quantiles = torch.quantile(torch.cat(self.target_excess_samples),
                                           self.maxima.new_tensor([.95, .99]), dim=0)
                ratios = self.raw_outside / (count / len(JOINT_NAMES))
                metrics['raw_target_out_of_range_ratio'] = float(ratios.mean())
                message += f' | raw_target_out_of_range {metrics["raw_target_out_of_range_ratio"]:.2%}'
                for name, (rate, p95, p99, peak) in zip(JOINT_NAMES, torch.cat((ratios[None], quantiles,
                                                   self.target_excess_max[None])).T.tolist()):
                    metrics.update({f'raw_target_out_of_range_{name}': rate,
                                    f'raw_target_excess_p95_{name}': p95,
                                    f'raw_target_excess_p99_{name}': p99,
                                    f'raw_target_excess_max_{name}': peak})
                self.raw_outside.zero_()
                self.target_excess_samples.clear()
                self.target_excess_max.zero_()
            self.action_samples.clear()
            self.speed_samples.clear()
            self.joint_clamps.zero_()
            self.effective_action_sq.zero_()
        return message, metrics


def fault_masks(before, after, probe, mdp):
    p, R, v, w, q, qd = after
    excess = torch.maximum(mdp.kin.q_lo - q, q - mdp.kin.q_hi).clamp_min(0.)
    v_norm, w_norm = v.norm(dim=-1), w.norm(dim=-1)
    finite = torch.cat((p, R.flatten(1), v, w, q, qd), -1).isfinite().all(-1)
    fatal = (~finite | ((p - before[0]).norm(dim=-1) > ROOT_STEP_MAX_M) |
             (v_norm > ROOT_SPEED_MAX_M_S) | (w_norm > ROOT_ANGULAR_SPEED_MAX_RAD_S))
    joint_fault = ((excess > JOINT_LIMIT_SLACK_RAD) | (qd.abs() > mdp.joint_speed_limit)).any(-1)
    from mujoco_warp import OverflowType
    overflow = (probe.overflow & ~int(OverflowType.ITERATIONS)) != 0
    return fatal, joint_fault, overflow, excess, v_norm, w_norm


def fix_base(scene):
    from pxr import Gf, UsdGeom, UsdPhysics

    spawn = scene.robot.spawn.func

    def spawn_with_fixed_base(*args, **kwargs):
        robot = spawn(*args, **kwargs)
        stage = robot.GetStage()
        path = robot.GetPath().AppendPath('Geometry/base_link')
        link = stage.GetPrimAtPath(path)
        link.GetAttribute('newton:jointsAddMobility').Set(False)
        joint = UsdPhysics.FixedJoint.Define(stage, robot.GetPath().AppendChild('PDTestFixture'))
        joint.CreateBody1Rel().SetTargets([path])
        point = UsdGeom.XformCache().GetLocalToWorldTransform(link).ExtractTranslation()
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*point))
        return robot

    # 与现有 Roban PD 工具相同，在 Newton 克隆前构造固定胸部的试验夹具。
    scene.robot.spawn.func = spawn_with_fixed_base


@torch.no_grad()
def run_pd_steps(env, probe, steps):
    """固定胸部、无接触下的阶跃和制动测量。"""
    cases = ('small_step', 'medium_step', 'near_limit', 'brake')
    names = ('leg_r1_joint', 'leg_r3_joint', 'zarm_r2_joint')
    j = torch.tensor([JOINT_NAMES.index(name) for name in names] * len(cases), device=env.device)
    ids = torch.arange(len(j), device=env.device)
    initial = env.state()
    q, qd = initial[4], torch.zeros_like(initial[5])
    lo, hi = env.mdp.kin.q_lo, env.mdp.kin.q_hi
    target = q.clone()
    g = len(names)
    target[ids[:g], j[:g]] += .05
    target[ids[g:2*g], j[g:2*g]] -= .35
    target[ids[2*g:3*g], j[2*g:3*g]] = lo[j[2*g:3*g]] + .02
    qd[ids[-g:], j[-g:]] = -45.
    target = target.clamp(lo, hi)
    env.robot.write_joint_velocity_to_sim_index(velocity=qd[:, env.sim_to_canonical])
    env.refresh()
    action = 2. * (target - lo) / (hi - lo) - 1.
    q_steps, speed_steps, torque_steps, limit_steps = [q[ids, j]], [qd[ids, j]], [], []
    peak_root_drift = 0.
    for _ in range(steps):
        state = env.advance_physics(env.pd_target(action))
        q_steps.append(state[4][ids, j])
        speed_steps.append(state[5][ids, j])
        torque_steps.append(probe.torque[:, probe.dofs][ids, j])
        limit_steps.append(torch.maximum(lo - state[4], state[4] - hi).clamp_min(0.)[ids, j])
        peak_root_drift = max(peak_root_drift, float((initial[0] - state[0]).norm(dim=-1).max()))
    if peak_root_drift > 1e-5 or int(probe.nacon[0]):
        raise AssertionError(f'固定基座存在根部位移 {peak_root_drift} 或接触')
    for i in range(len(j)):
        y = torch.stack(q_steps)[:, i]
        speed = torch.stack(speed_steps)[:, i]
        delta = target[ids, j][i] - y[0]
        elapsed = None
        overshoot = None
        if i < 3*g:
            progress = (y - y[0]) / delta
            hit = (progress >= .9).nonzero(as_tuple=True)[0]
            elapsed = int(hit[0]) * CONTROL_DT if len(hit) else None
            overshoot = max(0., (float(progress.max()) - 1.) * 100.)
        else:
            hit = (speed.abs() <= 4.5).nonzero(as_tuple=True)[0]
            elapsed = int(hit[0]) * CONTROL_DT if len(hit) else None
        print(json.dumps(dict(pd_case=cases[i//g], joint=names[i%g],
                              t90_or_brake_s=elapsed, overshoot_pct=overshoot,
                              max_qd_rad_s=float(speed[1:].abs().max()),
                              peak_torque_Nm=float(torch.stack(torque_steps)[:, i].abs().max()),
                              max_limit_excess_rad=float(torch.stack(limit_steps)[:, i].max()),
                              final_error_rad=float(abs(y[-1] - target[ids, j][i])),
                              peak_root_drift_m=peak_root_drift), ensure_ascii=False), flush=True)


def _standing_state(env):
    from ..roban import foot_geometry

    kin = env.mdp.kin
    q = env.mdp.default_q.expand(env.N, -1)
    p = q.new_zeros(env.N, 3)
    R = torch.eye(3, device=env.device, dtype=q.dtype).expand(env.N, -1, -1)
    v, w, qd = torch.zeros_like(p), torch.zeros_like(p), torch.zeros_like(q)
    body_p, body_R, _, _ = kin.forward(p, R, v, w, q, qd)
    feet = foot_geometry()
    ids = [kin.tracked_names.index(foot['body']) for foot in feet]
    local = q.new_tensor([foot['corners'] for foot in feet])
    corners = body_p[:, ids, None] + (body_R[:, ids, None] @ local[..., None]).squeeze(-1)
    # 以实际足底最低点为基准保留 1 mm 间隙，站姿诊断不读取参考姿态。
    p[:, 2] = .001 - corners[..., 2].amin(dim=(1, 2))
    return p, R, v, w, q, qd


@torch.no_grad()
def run_case(env, probe, case, args, policy, recorded):
    mdp = env.mdp
    dev = env.device
    env_ids = torch.arange(env.N, device=dev)
    env.reset(env_ids)
    if args.replay_fault or case == 'stand':
        # 故障回放从记录的真实状态起步，由正常写状态接口清理求解器缓存。
        p, R, v, w, q, qd = unflatten_state(args.start_state) if args.replay_fault else _standing_state(env)
        quat = mat_to_quat(R)
        root_quat = (args.start_root_quat if args.replay_fault and args.start_root_quat is not None
                     else quat[:, [1, 2, 3, 0]])
        env.robot.write_root_link_pose_to_sim_index(
            root_pose=torch.cat((p + env.scene.env_origins, root_quat), -1))
        env.robot.write_root_link_velocity_to_sim_index(root_velocity=torch.cat((v, w), -1))
        env.robot.write_joint_position_to_sim_index(position=q[:, env.sim_to_canonical])
        env.robot.write_joint_velocity_to_sim_index(velocity=qd[:, env.sim_to_canonical])
    env.scene.write_data_to_sim()
    env.refresh()
    if args.replay_fault and args.start_warmstart is not None:
        probe.warmstart.copy_(args.start_warmstart)
    probe.overflow.zero_()
    start = env.state()
    state = start
    lo, hi = mdp.kin.q_lo, mdp.kin.q_hi
    joint_range = mdp.action_mapping == 'joint_range'
    center = (lo + hi) * .5 if joint_range else mdp.default_q
    scale = (hi - lo) * .5 if joint_range else mdp.alpha
    hold = 2 * (start[4] - lo) / (hi - lo) - 1 if joint_range else (start[4] - center) / scale
    generator = torch.Generator(device=dev).manual_seed(args.seed)
    history = None
    previous_target = start[4]
    captured, trace = [], deque(maxlen=8)
    replay_trace = []
    first = first_fatal = None
    valid = torch.ones(env.N, dtype=torch.bool, device=dev)
    maxes = torch.zeros(11, device=dev)
    moments = torch.zeros(7, device=dev, dtype=torch.float64)
    event_counts = torch.zeros(4, device=dev, dtype=torch.long)
    first_fall = torch.full((env.N,), args.steps, device=dev, dtype=torch.long)
    physics_wall = 0.
    min_clearance = 0.
    peak_impulse = 0.
    peak_tilt = peak_drift = 0.
    standing_contacts = env.ground_contacts() if case == 'stand' and not args.no_contacts else None
    double_support = 0
    for tick in range(args.steps):
        if case == 'actor' and recorded is not None:
            raw, action = recorded[tick].unbind(0)
        elif case == 'actor':
            frame = env.f.fill_(min(tick, env.motion.num_frames - 1))
            env.ep_step.copy_(frame)
            obs = env.goals.obs(mdp, state, env.a_prev, env.xi, frame, env.command)
            history = (obs[:, None].expand(-1, policy.hist, -1) if history is None
                       else torch.cat((history[:, 1:], obs[:, None]), dim=1))
            goal = env.goals.goal(state, frame, env.ep_step, env.command)
            if joint_range:
                raw = policy.decode(history, policy.encode(goal))
            else:
                raw = policy(history, goal)
            action = raw + args.action_noise * torch.randn(raw.shape, generator=generator, device=dev)
            if joint_range:
                action = action.tanh()
        elif case == 'reference':
            ref_q = env.motion.q[min(tick, env.motion.num_frames - 1)]
            command = 2 * (ref_q - lo) / (hi - lo) - 1 if joint_range else (ref_q - center) / scale
            action = command.expand_as(hold)
            raw = action.clamp(-.999999, .999999).atanh() if joint_range else action
        else:
            random = 2. * torch.rand(hold.shape, generator=generator, device=dev) - 1.
            action = {'zero': torch.zeros_like(hold), 'stand': hold, 'hold': hold, 'small': .1 * random, 'full': random}[case]
            raw = action.clamp(-.999999, .999999).atanh() if joint_range else action
        if case == 'actor' and args.save_actions:
            captured.append(torch.stack((raw, action), 0).cpu())
        target = env.pd_target(action)
        expected = center + scale * action
        if not joint_range:
            if not mdp.unclipped_pd_target:
                expected = expected.clamp(lo, hi)
            torch.testing.assert_close(center + scale * mdp.wm_action(action, env.f), target)
        torch.testing.assert_close(target, expected)
        if case == 'reference':
            torch.testing.assert_close(target, ref_q.expand_as(target))
        torch.cuda.synchronize(dev)
        before_time = perf_counter()
        substeps = []
        if args.trace_substeps:
            with probe.record_substeps(substeps):
                after = env.advance_physics(target)
        else:
            after = env.advance_physics(target)
        torch.cuda.synchronize(dev)
        physics_wall += perf_counter() - before_time
        torch.testing.assert_close(env.robot.data.joint_pos_target.torch[:, env.canonical_to_sim], target)
        contact_stats = probe.contacts()
        telem = probe.snapshot(contact_stats)
        fatal, joint, overflow, excess, v_norm, w_norm = fault_masks(state, after, probe, mdp)
        bad = fatal | joint | overflow
        fallen = mdp.termination(after)
        first_fall = torch.where(valid & fallen, first_fall.clamp_max(tick + 1), first_fall)
        event_counts += torch.stack([(valid & value).sum() for value in (bad, fatal, joint, overflow)])
        if args.trace:
            row = pack_step(state, after, raw, action, target, previous_target, mdp) | telem
            row['frame'] = args.start_frame + tick
            if substeps:
                row.update({f'substep_{k}': torch.stack([x[k] for x in substeps], 1) for k in substeps[-1]})
            row = {k: v.clone() for k, v in row.items()}
            trace.append(row)
            if args.replay_fault:
                replay_trace.append(row)
        def first_event(mask, prefix):
            indices = (valid & mask).nonzero(as_tuple=True)[0]
            if not len(indices):
                return None
            idx = int(indices[0])
            j = int(excess[idx].argmax()) if excess[idx].max() > JOINT_LIMIT_SLACK_RAD else int(after[5][idx].abs().argmax())
            if args.trace:
                for key in trace[-1]:
                    args.traces[f'{prefix}_{key}'] = torch.stack([row[key][idx] for row in trace]).cpu().numpy()
                args.traces[f'{prefix}_env'] = idx
            return dict(tick=tick + 1, frame=int(args.start_frame[idx]) + tick + 1, env=idx,
                        joint=JOINT_NAMES[j], q=float(after[4][idx, j]),
                        q_before=float(state[4][idx, j]), qd=float(after[5][idx, j]), target=float(target[idx, j]),
                        penetration=float(excess[idx, j]), root_v=float(v_norm[idx]), root_w=float(w_norm[idx]),
                        contact_counts=telem['contacts'][idx].tolist(), nefc=int(probe.nefc[idx]),
                        iterations=int(probe.niter[idx]), overflow=int(probe.overflow[idx]),
                        after_fall=bool(first_fall[idx] < tick + 1))

        if first is None:
            first = first_event(bad, case)
        if first_fatal is None:
            first_fatal = first_event(fatal, f'{case}_fatal')

        if valid.any():
            torque_ratio = telem['torque'].abs() / mdp.effort
            numbers = [excess, after[5].abs(), v_norm, w_norm, telem['contacts'][:, 0], probe.nefc, probe.niter,
                       torque_ratio, telem['extra_force'].abs(), (target - previous_target).abs(), raw.abs()]
            maxes = torch.maximum(maxes, torch.stack([x[valid].max() for x in numbers]))
            moments += torch.stack((raw[valid].double().sum(), raw[valid].double().square().sum(),
                                    (action[valid].abs() >= .95).sum(), (torque_ratio[valid] >= .98).sum(),
                                    valid.sum() * len(JOINT_NAMES), (telem['contacts'][valid, 2] > 0).sum(),
                                    mdp.target_clip_mask(action[valid]).sum() if not joint_range else valid.new_zeros(())))
            if case == 'stand':
                peak_tilt = max(peak_tilt, float(torch.rad2deg(torch.acos(after[1][valid, 2, 2].clamp(-1., 1.))).max()))
                peak_drift = max(peak_drift, float((after[0][valid, :2] - start[0][valid, :2]).norm(dim=-1).max()))
                if standing_contacts is not None:
                    _, forces, _ = standing_contacts.read()
                    double_support += (forces[valid.cpu().numpy(), :, 2] >= 10.).all(axis=-1).sum()
            min_clearance = min(min_clearance, float(telem['contact_depth'][valid].min()))
            if substeps:
                peak_impulse = max(peak_impulse, max(float(s['normal_impulse'][valid].max()) for s in substeps))
        previous_target = target
        state = after
        env.a_prev.copy_(action)
        valid &= ~(bad | fallen) if args.stop_on_fall else ~bad
        # 保留相同动作时间轴，故障环境恢复起态且不再累计指标。
        reset_ids = (~valid).nonzero(as_tuple=True)[0]
        if len(reset_ids):
            env.reset(reset_ids)
            probe.overflow[reset_ids] = 0
            state = env.state()

    total, square, near_one, saturated, count, ground, clipped = moments.tolist()
    keys = ('max_limit_excess_rad', 'max_qd_rad_s', 'max_root_v_m_s', 'max_root_w_rad_s', 'max_contacts',
            'max_nefc', 'max_solver_iterations', 'peak_torque_ratio', 'peak_extra_joint_force_Nm',
            'max_target_delta_rad', 'raw_abs_max')
    mean = total / count
    if replay_trace:
        args.traces.update({f'replay_{key}': torch.stack([row[key] for row in replay_trace]).cpu().numpy()
                            for key in replay_trace[0]})
    finite_summary = {k: v if np.isfinite(v) else None for k, v in zip(keys, maxes.tolist())}
    standing = dict(root_height_m=float(start[0][0, 2]), peak_tilt_deg=peak_tilt, peak_xy_drift_m=peak_drift,
                    double_support_fraction=float(double_support) * len(JOINT_NAMES) / count) if case == 'stand' else {}
    print(json.dumps(dict(case=case, seed=args.seed, envs=env.N, steps=args.steps, **finite_summary, **standing,
                          raw_mean=mean, raw_std=max(0., square / count - mean**2)**.5,
                          action_near_one=near_one / count if joint_range else None,
                          target_clamp=clipped / count if not joint_range else None, saturation=saturated / count,
                          ground_step_ratio=ground * len(JOINT_NAMES) / count, min_contact_dist_m=min_clearance,
                          peak_normal_impulse_Ns=peak_impulse if args.trace_substeps else None,
                          first_fault=first, first_fatal=first_fatal, faults=event_counts.tolist(),
                          first_fall_steps=float(first_fall.float().mean()), physics_fps=env.N * args.steps / physics_wall),
                     ensure_ascii=False), flush=True)
    return torch.stack(captured) if captured else None


def main():
    from isaaclab.app import add_launcher_args, launch_simulation

    from ..env import TrackingSceneCfg
    from ..kinematics import RobotKinematics
    from ..networks import Policy
    from ..roban_runtime import M4_GUARDED_MAPPINGS, add_m4_action_mapping_arg, add_m4_solver_args
    from .m4_action_diagnostics import _load_actor
    from .m4_task import M4TrackingSim, build_m4_task, make_m4_sim

    profile = argparse.ArgumentParser(add_help=False)
    profile.add_argument('--roban_config', type=Path, help='读取训练配置中的 M4 物理参数')
    parser = argparse.ArgumentParser(description=__doc__, parents=[profile])
    parser.add_argument('--m4_reference', required=True)
    add_m4_action_mapping_arg(parser)
    add_m4_solver_args(parser)
    parser.set_defaults(m4_njmax=1024, m4_nconmax=256, m4_solver_iterations=150, m4_action_mapping='joint_range',
                        reference_fps=None, goal_frames=10, goal_dt=.1, roban_reward_config={},
                        body_prior_weight=0., wm_use_terrain=False, critic_use_body_reference=True)
    parser.add_argument('--num_envs', type=int, default=32)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cases', choices=('stand', 'zero', 'hold', 'small', 'full', 'reference', 'actor'), nargs='+',
                        default=['zero', 'hold', 'small', 'full', 'actor'])
    parser.add_argument('--actor_checkpoint', type=Path)
    parser.add_argument('--action_noise', type=float, default=.1)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--save_actions', type=Path)
    actions.add_argument('--replay_actions', type=Path)
    actions.add_argument('--replay_fault', type=Path, help='从训练快照的首个状态回放对应环境的记录动作')
    parser.add_argument('--trace', type=Path, help='保存首次异常前的控制步或完整故障回放轨迹')
    parser.add_argument('--trace_substeps', action='store_true', help='在异常快照中加入每个物理子步的约束和状态')
    parser.add_argument('--no_contacts', action='store_true', help='停用接触生成以隔离关节和积分')
    parser.add_argument('--stop_on_fall', action='store_true', help='按训练终止条件停止统计已跌倒的环境')
    parser.add_argument('--pd_steps', action='store_true', help='固定胸部并实测三关节阶跃与制动')
    parser.add_argument('--no_graph', action='store_true', help='禁用 CUDA Graph 以单步诊断')
    add_launcher_args(parser)
    selected, _ = profile.parse_known_args()
    if selected.roban_config:
        values = json.loads(selected.roban_config.read_text())
        keys = ('m4_substeps', 'm4_limit_time_constant', 'm4_pd_damping', 'm4_integrator', 'hist',
                'm4_njmax', 'm4_nconmax', 'm4_solver_iterations', 'goal_frames', 'goal_dt', 'action_noise',
                'm4_action_mapping', 'm4_action_scale_factor', 'm4_standing_pose', 'm4_unclipped_pd_target')
        parser.set_defaults(**{key: values[key] for key in keys if key in values})
    args = parser.parse_args()
    dev = args.device or 'cuda:0'
    if args.steps < 1 or args.num_envs < 1 or torch.device(dev).type != 'cuda':
        parser.error('需要正的步数和环境数，以及 CUDA 设备')
    if args.m4_action_mapping not in M4_GUARDED_MAPPINGS:
        parser.error('此物理对照需要 M4.1 或 M4.2 动作映射')
    if (args.m4_action_mapping == 'joint_range' and not args.pd_steps and 'actor' in args.cases
            and not (args.actor_checkpoint or args.replay_actions or args.replay_fault)):
        parser.error('actor 实验需要 Actor 检查点、动作文件或训练异常快照')
    if args.trace_substeps and not args.trace:
        parser.error('--trace_substeps 需要 --trace')
    recorded = None
    if args.replay_fault:
        with np.load(args.replay_fault) as data:
            if data['joint_names'].tolist() != JOINT_NAMES:
                parser.error('训练异常快照的关节顺序与当前 Roban 不一致')
            frames = data['frame']
            # 若缓存跨过环境重置，从所有记录环境共同连续的末段开始回放。
            discontinuities = np.flatnonzero((np.diff(frames, axis=0) != 1).any(axis=1))
            start = int(discontinuities[-1]) + 1 if len(discontinuities) else 0
            recorded = torch.as_tensor(np.stack((data['raw'][start:], data['action'][start:]), axis=1), device=dev)
            args.start_state = torch.as_tensor(data['before'][start], device=dev)
            args.start_frame = torch.as_tensor(frames[start], device=dev)
            args.start_root_quat = (torch.as_tensor(data['root_quat_before_xyzw'][start], device=dev)
                                    if 'root_quat_before_xyzw' in data else None)
            args.start_warmstart = (torch.as_tensor(data['warmstart_before'][start], device=dev)
                                    if 'warmstart_before' in data else None)
        args.steps, _, args.num_envs, _ = recorded.shape
        args.cases = ['actor']
    else:
        args.start_frame = torch.zeros(args.num_envs, device=dev, dtype=torch.long)
    if args.replay_actions:
        with np.load(args.replay_actions) as loaded:
            recorded = torch.as_tensor(loaded['actions'], device=dev)
        if recorded.shape != (args.steps, 2, args.num_envs, len(JOINT_NAMES)):
            parser.error('动作文件的步数、环境数或关节数不一致')
    torch.manual_seed(args.seed)
    motion, goals, mdp, _ = build_m4_task(args, RobotKinematics(dev), [args.m4_reference], dev)
    cfg = make_m4_sim(dev, args)
    if args.pd_steps:
        args.num_envs = 12
        args.no_contacts = True
        cfg.gravity = (0., 0., 0.)
    cfg.physics.solver_cfg.disable_contacts = args.no_contacts
    if args.no_graph or args.trace_substeps:
        cfg.physics.use_cuda_graph = False
    args.traces = {}
    with launch_simulation(cfg, args):
        scene = TrackingSceneCfg(num_envs=args.num_envs, env_spacing=0., replicate_physics=True)
        if args.pd_steps:
            fix_base(scene)
        env = M4TrackingSim(cfg, args.num_envs, motion, mdp, goals=goals, scene_cfg=scene)
        probe = SolverReadback(env)
        policy = _load_actor(env, args.actor_checkpoint) if args.actor_checkpoint else None
        if policy is None and 'actor' in args.cases and recorded is None and not args.pd_steps:
            policy = Policy(mdp.obs_dim, goals.spec.dim, hist=getattr(args, 'hist', 10)).to(dev).eval()
        print(json.dumps(dict(versions={key: version(key) for key in ('isaaclab', 'newton', 'mujoco-warp', 'warp-lang')},
                              control_dt=cfg.dt, physics_dt=env.physics_dt, contacts_enabled=not args.no_contacts,
                              **probe.audit(env)), ensure_ascii=False), flush=True)
        if args.pd_steps:
            run_pd_steps(env, probe, args.steps)
            return
        # 首次 CUDA Graph 构建计入初始化，避免把编译延迟算成物理步开销。
        env.advance_physics(env.state()[4])
        torch.cuda.synchronize(dev)
        for case in args.cases:
            captured = run_case(env, probe, case, args, policy, recorded)
            if captured is not None:
                np.savez_compressed(args.save_actions, actions=captured.numpy(), seed=args.seed)
    if args.trace and args.traces:
        np.savez_compressed(args.trace, **args.traces, joint_names=JOINT_NAMES, seed=args.seed,
                            physics_dt=env.physics_dt, geom_names=[probe.model.geom(i).name for i in range(probe.model.ngeom)])


if __name__ == '__main__':
    main()
