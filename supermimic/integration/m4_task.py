"""Roban F2 单条参考的任务配置、目标、奖励及 Newton 环境。"""
from dataclasses import asdict

if __name__ == '__main__':
    import os

    os.environ['SUPERMIMIC_ROBOT'] = 'roban'

import torch

from ..env import TrackingSim, make_sim
from ..goals import GoalSpec
from ..motion import CONTROL_DT
from ..roban import FOOT_BODY_NAMES, JOINT_NAMES, NUM_J, default_joint_pos_list
from ..roban_task import FootRewardConfig, RobanFootGoals, RobanFootMDP
from ..rotations import mat_to_6d, quat_to_mat, yaw_mat
from ..terrain import make_terrain
from .m4_reference import F2MotionReference


class F2GoalSpec(GoalSpec):
    @property
    def frame_dim(self):
        return super().frame_dim + 9

    def contract(self):
        value = super().contract()
        value.update(
            version=4,
            reference='m3.7.1/F2',
            foot_frame='source_ankle_origin_minus_0.04m_local_z',
            stance='source_foot_trajectory',
            root_goal='relative_position_m_and_6d_rotation_in_current_heading',
            observation_reference='proprio_only_sole_and_root_goals',
            lookahead='clamped_to_reference_end_with_actual_time_offset',
        )
        return value


class F2Goals(RobanFootGoals):
    def goal(self, S, f, ep_step=None, command=None):
        if command is None or ep_step is None:
            raise ValueError('F2 goals require the episode reference clock and anchor')
        torch._assert_async(torch.all(f == command.start_frame + ep_step), 'F2 reference frame/episode clock mismatch')

        # 参考和策略都用控制时钟，未来目标只在 5.2 秒的有效区间内取样。
        frames = self.motion.indices(f, self.spec.frames, self.spec.dt)
        ref = self.motion.sample(frames, command)
        root_p, root_R = S[:2]
        heading_inv = yaw_mat(root_R).transpose(-1, -2)[:, None]
        foot_p = (heading_inv[:, :, None] @ (ref['foot_p'] - root_p[:, None, None])[..., None]).squeeze(-1)
        foot_R = mat_to_6d(heading_inv[:, :, None] @ ref['foot_R'])
        feet = torch.cat((foot_p, foot_R, ref['contact'][..., None].to(root_p.dtype), ref['phase'][..., None]), -1)
        lookahead = ((frames - f[:, None]).to(root_p.dtype) * CONTROL_DT)[..., None]
        ref_root_p = (heading_inv @ (ref['root_p'] - root_p[:, None])[..., None]).squeeze(-1)
        ref_root_R = mat_to_6d(heading_inv @ ref['root_R'])
        return torch.cat((feet.flatten(-2), lookahead, ref_root_p, ref_root_R), -1).flatten(1)


class F2TrackingMDP(RobanFootMDP):
    def reward(self, S, f, a, a_prev, terms=False, *, command=None, ep_step=None):
        result = super().reward(S, f, a, a_prev, terms, command=command, ep_step=ep_step)
        if terms:
            _, values = result
            ref = self._reference(f, command)
            values['root_position_error'] = (S[0] - ref['root_p'][:, 0]).norm(dim=-1)
        return result

    def contract(self):
        value = super().contract()
        value.update(mode='roban_m4_f2', foot_frame='F2_source_sole; current_robot_heading',
                     reset='first_frame_without_domain_randomization', duration_s=self.motion.duration)
        return value


def build_m4_task(args, kin, files, device):
    if len(files) != 1:
        raise ValueError('M4.0 requires exactly one F2 reference')
    motion = F2MotionReference(files[0], kin, device, args.reference_fps)
    spec = F2GoalSpec('roban_foot', args.goal_frames, args.goal_dt)
    goals = F2Goals(motion, kin, spec)
    default_q = torch.tensor(default_joint_pos_list(), device=device)
    rewards = dict(args.roban_reward_config, body_prior_weight=args.body_prior_weight)
    cfg = FootRewardConfig(**rewards)
    terrain = make_terrain(kin, vars(args)) if args.wm_use_terrain else None
    mdp = F2TrackingMDP(kin, motion, default_q, goals, cfg,
                        critic_use_body_reference=args.critic_use_body_reference, terrain=terrain)
    args.reward_config = asdict(cfg)
    args.motion_obs_mode = mdp.observation_mode
    return motion, goals, mdp, terrain


def make_m4_sim(device, args):
    cfg = make_sim(device, args)
    if torch.device(device).type == 'cpu':
        # 仅显式选择 CPU 时切换求解器，用于本地诊断。
        cfg.physics.use_cuda_graph = False
        cfg.physics.solver_cfg.use_mujoco_cpu = True
    return cfg


class M4TrackingSim(TrackingSim):
    def __init__(self, sim_cfg, num_envs, motion, mdp, gamma=0.99, eval_mode=False,
                 max_episode_steps=None, reset_noise=False, goals=None):
        steps = motion.num_frames - 1
        if abs(steps * CONTROL_DT - motion.duration) > 1e-6 or abs(sim_cfg.dt - CONTROL_DT) > 1e-9:
            raise ValueError('F2 episode clock does not match the control or physics clock')
        if max_episode_steps is not None:
            if max_episode_steps < 1:
                raise ValueError('F2 episode length must be positive')
            steps = min(steps, max_episode_steps)
        self.physics_dt = sim_cfg.dt / sim_cfg.physics.num_substeps
        super().__init__(sim_cfg, num_envs, motion, mdp, gamma=gamma, eval_mode=eval_mode,
                         max_episode_steps=steps, reset_noise=reset_noise,
                         goals=goals, domain_randomization=False)
        self.max_episode_steps = steps

    def sample_frames(self, n):
        # 每个 episode 均验证 F2 从第一帧开始，完整末帧由 clip_end 处理。
        frames = torch.zeros(n, dtype=torch.long, device=self.device)
        return frames, self.motion.frame_bin[frames]

    def reset(self, env_ids, frames=None):
        super().reset(env_ids, frames)
        self.done_eval[env_ids] = False
        self.failed_eval[env_ids] = False

    def ground_contacts(self):
        from ..standing_diagnostics import NewtonGroundContacts

        return NewtonGroundContacts(self, foot_body_names=FOOT_BODY_NAMES)


def physics_smoke():
    """实测固定步数的 Newton 状态、PD 力矩和足部接触。"""
    import argparse
    import json

    import numpy as np
    import warp as wp
    from isaaclab.app import add_launcher_args, launch_simulation

    from ..control import bound_action
    from ..kinematics import RobotKinematics

    parser = argparse.ArgumentParser(description=physics_smoke.__doc__)
    parser.add_argument('--m4_reference', required=True)
    parser.add_argument('--num_envs', type=int, choices=range(1, 9), default=1)
    parser.add_argument('--steps', type=int, default=16, help='最多执行的控制步数，首次跌倒时停止')
    add_launcher_args(parser)
    parser.set_defaults(reference_fps=None, goal_frames=5, goal_dt=.1, roban_reward_config={},
                        body_prior_weight=0, wm_use_terrain=False, critic_use_body_reference=True)
    args = parser.parse_args()
    if not 1 <= args.steps <= 260:
        parser.error('--steps 必须介于 1 和 260')
    device, steps = args.device or 'cuda:0', args.steps
    if torch.device(device).type != 'cuda' or not torch.cuda.is_available():
        parser.error('MJWarp 物理检查需要可用的 CUDA 设备')
    motion, goals, mdp, _ = build_m4_task(args, RobotKinematics(device), [args.m4_reference], device)
    cfg = make_m4_sim(device, args)
    with launch_simulation(cfg, args):
        env = M4TrackingSim(cfg, args.num_envs, motion, mdp, eval_mode=True, goals=goals, max_episode_steps=steps)
        ids = torch.arange(env.N, device=device)
        initial = env.state()
        if not all(x.isfinite().all() for x in initial):
            raise AssertionError('重置后的状态出现 NaN/Inf')
        reset_error = max(float((x - y).abs().max()) for x, y in zip(initial, motion.ref_state(env.f)))
        names = env.robot.joint_names
        if [names[i] for i in env.canonical_to_sim.tolist()] != JOINT_NAMES or env.a_prev.shape != (env.N, NUM_J):
            raise AssertionError('Roban 的关节名称或动作维度不一致')
        state = env.robot.data
        for actual, expected in ((state.joint_stiffness, mdp.kp), (state.joint_damping, mdp.kd),
                                 (state.joint_effort_limits, mdp.effort)):
            torch.testing.assert_close(actual.torch[:, env.canonical_to_sim], expected.expand(env.N, -1))

        contacts, solver = env.ground_contacts(), env.physics._solver
        mj = solver.mj_model
        mj_names = [mj.joint(i).name for i in range(mj.njnt)]
        dofs = []
        for name in JOINT_NAMES:
            match = [i for i, value in enumerate(mj_names) if value == name or value.endswith(('/' + name, '_' + name))]
            if len(match) != 1:
                raise AssertionError(f'求解器关节 {name} 的映射不唯一')
            dofs.append(int(mj.jnt_dofadr[match[0]]))
        feet = [env.robot.body_names.index(name) for name in FOOT_BODY_NAMES]
        peak_force = peak_torque = peak_ratio = fk_error = 0.
        saturated = 0
        first_unload = None
        first_lift, first_touch = float(motion.liftoff_time[0]), float(motion.touchdown_time[0])
        first_side = int(motion.touchdown_side[0])
        for tick in range(steps):
            # 只下发参考关节位置对应的 PD 动作，基座由物理引擎推进。
            action = bound_action((motion.q[tick + 1] - env.default_q) / env.alpha).expand(env.N, -1)
            result = env.step(action)
            if not (result['s1'].isfinite().all() and result['r'].isfinite().all() and (env.f == tick + 1).all()):
                raise AssertionError('Newton 状态、奖励或参考时钟无效')
            torque = wp.to_torch(solver.mjw_data.qfrc_actuator)[:, dofs].abs()
            ratio = torque / mdp.effort
            _, force, _ = contacts.read()
            if not (np.isfinite(force).all() and torque.isfinite().all()):
                raise AssertionError('求解器足力或关节力矩出现 NaN/Inf')
            peak_force = max(peak_force, float(force[..., 2].max()))
            peak_torque = max(peak_torque, float(torque.max()))
            peak_ratio = max(peak_ratio, float(ratio.max()))
            saturated += int((ratio >= .98).sum())
            poses = state.body_link_pose_w.torch[:, feet]
            rotation = quat_to_mat(poses[..., [6, 3, 4, 5]])
            sole = poses[..., :3] - env.scene.env_origins[:, None] + (rotation @ motion.sole_offsets[..., None]).squeeze(-1)
            fk_error = max(fk_error, float((sole - motion.feet(env.state())[0]).norm(dim=-1).max()))
            elapsed = (tick + 1) * CONTROL_DT
            if result['term'].any():
                break
            if first_unload is None and first_lift <= elapsed < first_touch and force[0, first_side, 2] < 10.:
                first_unload = elapsed

        fallen = bool(result['term'].any())
        env.reset(ids)
        repeat_error = max(float((x - y).abs().max()) for x, y in zip(initial, env.state()))
        if peak_ratio > 1.001 or peak_force <= 1 or max(reset_error, repeat_error, fk_error) >= 1e-4:
            raise AssertionError('PD 限幅、足力、FK 或重置超出校验范围')
        if (not fallen and not result['timeout'].all()) or env.done_eval.any():
            raise AssertionError('短时钟结束或重置清理异常')
        ended_by = 'fall' if fallen else 'reference_end' if steps == motion.num_frames - 1 else 'step_budget'
        print(json.dumps(dict(status='PASS', envs=env.N, steps=tick + 1, elapsed_s=elapsed, ended_by=ended_by,
                              first_planned_swing_foot_unloaded_s=first_unload, actions=NUM_J, control_dt=cfg.dt,
                              physics_dt=env.physics_dt, reset_max_abs_error=reset_error,
                              repeat_reset_max_abs_error=repeat_error, max_sole_fk_error_m=fk_error,
                              max_foot_ground_force_z_N=peak_force, peak_abs_joint_torque_Nm=peak_torque,
                              peak_effort_ratio=peak_ratio,
                              last_substep_saturation_fraction=saturated / ((tick + 1) * env.N * NUM_J))),
              flush=True)


if __name__ == '__main__':
    physics_smoke()
