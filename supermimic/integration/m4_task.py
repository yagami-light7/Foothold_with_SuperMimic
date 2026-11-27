"""Roban F2 单条参考的任务配置、目标、奖励及 Newton 环境。"""
from dataclasses import asdict
import math

if __name__ == '__main__':
    import os

    os.environ['SUPERMIMIC_ROBOT'] = 'roban'

import torch

from ..control import bound_action
from ..env import TrackingSceneCfg, TrackingSim, make_sim
from ..goals import GoalSpec
from ..motion import CONTROL_DT
from ..roban import FOOT_BODY_NAMES, JOINT_NAMES, NUM_J, actuator_parameters, default_joint_pos_list
from ..roban_runtime import M4_ACTION_MAPPINGS, M4_GUARDED_MAPPINGS, add_m4_action_mapping_arg, add_m4_solver_args
from ..roban_task import FootRewardConfig, RobanFootGoals, RobanFootMDP
from ..rotations import mat_to_6d, quat_to_mat, yaw_mat
from ..terrain import make_terrain
from ..transfer import SIMULATION, physical_contract
from .m4_reference import F2MotionReference
from .m47_control import blend_startup_target, upper_joint_indices


# 为跌倒、接触冲击和软限位留出裕量，只拦截严重偏离物理尺度的样本。
ROOT_STEP_MAX_M = .5
ROOT_SPEED_MAX_M_S = 30.
ROOT_ANGULAR_SPEED_MAX_RAD_S = 150.
JOINT_LIMIT_SLACK_RAD = .75
JOINT_SPEED_LIMIT_MULTIPLIER = 16.


def m4_solver_overrides(args):
    defaults = dict(njmax=SIMULATION['njmax'], nconmax=SIMULATION['nconmax'], iterations=100)
    options = dict(njmax=getattr(args, 'm4_njmax', None),
                   nconmax=getattr(args, 'm4_nconmax', None),
                   iterations=getattr(args, 'm4_solver_iterations', None))
    for key, value in options.items():
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f'M4 求解器参数 {key} 必须为正整数')
    overrides = {key: value for key, value in options.items() if value is not None and value != defaults[key]}
    integrator = getattr(args, 'm4_integrator', None)
    if integrator is not None:
        if integrator not in ('implicitfast', 'implicit'):
            raise ValueError(f'M4 不支持积分器 {integrator}')
        if integrator != SIMULATION['integrator']:
            overrides['integrator'] = integrator
    return overrides


def m4_substeps(args):
    value = getattr(args, 'm4_substeps', None)
    if value is None:
        return SIMULATION['num_substeps']
    if type(value) is not int or value < 1:
        raise ValueError('M4 物理子步数必须为正整数')
    return value


def m4_limit_time_constant(args):
    value = getattr(args, 'm4_limit_time_constant', None)
    if value is not None and (type(value) not in (int, float) or not math.isfinite(value)
                              or value < 2 * CONTROL_DT / m4_substeps(args)):
        raise ValueError('MJWarp 关节限位时间常数必须不小于两个物理子步')
    return value


def m4_pd_damping(args):
    values = getattr(args, 'm4_pd_damping', None)
    if values is None:
        return {}
    if not isinstance(values, dict) or set(values) - set(JOINT_NAMES):
        raise ValueError('M4 PD 阻尼必须是以 Roban 关节名称为键的对象')
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0. for x in values.values()):
        raise ValueError('M4 PD 阻尼值必须是有限非负数')
    return values


def set_scene_joint_limit(scene, time_constant):
    from pxr import Sdf, Usd, UsdPhysics

    spawn = scene.robot.spawn.func

    def spawn_with_joint_limit(*args, **kwargs):
        robot = spawn(*args, **kwargs)
        found = set()
        for joint in Usd.PrimRange(robot):
            if joint.IsA(UsdPhysics.RevoluteJoint) and joint.GetName() in JOINT_NAMES:
                joint.CreateAttribute('mjc:solreflimit', Sdf.ValueTypeNames.DoubleArray).Set([time_constant, 1.])
                found.add(joint.GetName())
        if found != set(JOINT_NAMES):
            raise ValueError(f'Roban 关节限位参数缺少关节：{set(JOINT_NAMES) - found}')
        return robot

    # 在 Newton 解析并复制模板资产前写入会话 USD，源模型和关节范围保持一致。
    scene.robot.spawn.func = spawn_with_joint_limit


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
    def __init__(self, kin, motion, default_q, goals, cfg, *, action_mapping='neutral',
                 critic_use_body_reference=False, terrain=None, limit_time_constant=None,
                 pd_damping=None, action_scale_factor=.25, unclipped_pd_target=False, startup_steps=0):
        if action_mapping not in M4_ACTION_MAPPINGS:
            raise ValueError(f'未知的 M4 动作映射：{action_mapping}')
        if type(unclipped_pd_target) is not bool or (unclipped_pd_target and action_mapping != 'standing_offset'):
            raise ValueError('未裁剪位置目标必须是布尔开关，且要求 standing_offset 映射')
        if action_mapping in M4_GUARDED_MAPPINGS:
            if kin.q_lo.shape != (NUM_J,) or kin.q_hi.shape != (NUM_J,):
                raise ValueError('M4 动作映射需要 21 个 Roban 关节的上下限')
            valid = kin.q_lo.isfinite() & kin.q_hi.isfinite() & (kin.q_lo < kin.q_hi)
            if not valid.all():
                invalid = [JOINT_NAMES[j] for j in (~valid).nonzero(as_tuple=True)[0].tolist()]
                raise ValueError(f'Roban 关节缺少有效的有限位置上下限：{invalid}')
            if not critic_use_body_reference:
                raise ValueError('当前 M4 动作映射需要启用 critic_use_body_reference')
        if action_mapping == 'standing_offset':
            if (type(action_scale_factor) not in (int, float) or not math.isfinite(action_scale_factor)
                    or action_scale_factor <= 0.):
                raise ValueError('M4.2 动作的力矩比例系数必须是有限正数')
            if default_q.shape != kin.q_lo.shape or not ((default_q > kin.q_lo) & (default_q < kin.q_hi)).all():
                raise ValueError('M4.2 固定站姿的关节角必须严格位于真实限位内')
        super().__init__(kin, motion, default_q, goals, cfg,
                         critic_use_body_reference=critic_use_body_reference, terrain=terrain)
        self.action_mapping = action_mapping
        self.unclipped_pd_target = unclipped_pd_target
        if type(startup_steps) is not int or startup_steps not in (0, 5, 10):
            raise ValueError('启动过渡必须为 0、5 或 10 个控制步')
        self.startup_steps = startup_steps
        self.action_history_enabled = bool(startup_steps or cfg.upper_action_smoothness_weight)
        if self.action_history_enabled:
            if action_mapping != 'standing_offset' or unclipped_pd_target or not cfg.swing_aware:
                raise ValueError('M4.7 动作历史仅用于 F2 Hard Clamp 摆动奖励')
            self.startup_mask = torch.zeros_like(default_q, dtype=torch.bool)
            self.startup_mask[upper_joint_indices(JOINT_NAMES)] = True
        self.limit_time_constant = limit_time_constant
        self.pd_damping = pd_damping or {}
        for name, value in self.pd_damping.items():
            # 已有力矩代理使用与真实执行器一致的 Kd，保留原奖励项及权重。
            self.kd[JOINT_NAMES.index(name)] = value
        if action_mapping == 'standing_offset':
            self.alpha = action_scale_factor * self.effort / self.kp
            if not (self.alpha.isfinite() & (self.alpha > 0.)).all():
                raise ValueError('M4.2 的动作缩放超出张量数值范围')
        if action_mapping in M4_GUARDED_MAPPINGS:
            params = actuator_parameters()
            self.joint_speed_limit = kin.q_lo.new_tensor(
                [params[name]['velocity'] * JOINT_SPEED_LIMIT_MULTIPLIER for name in JOINT_NAMES])

    def residual_pd_target(self, a, f):
        # f 是动作下发时的当前参考帧。目标限位使用 Roban 模型的关节硬限位。
        target = self.motion.q[f] + self.alpha * bound_action(a)
        return target.clamp(self.kin.q_lo, self.kin.q_hi)

    def wm_action(self, a, f):
        if self.action_mapping == 'standing_offset':
            if self.unclipped_pd_target:
                return a
            target = self.standing_pd_target(a)
        elif self.action_mapping == 'reference_residual':
            target = self.residual_pd_target(a, f)
        else:
            return a
        # 用已执行的限位目标反算 WM 输入，保留 21 维动作尺度及限位处真实梯度。
        return self.effective_action(target)

    def standing_raw_target(self, a):
        return self.default_q + self.alpha * a

    def standing_pd_target(self, a):
        target = self.standing_raw_target(a)
        # 机械限位与驱动力矩限幅由 Newton 原有约束处理。
        return target if self.unclipped_pd_target else target.clamp(self.kin.q_lo, self.kin.q_hi)

    def executed_target(self, a, reset_q, ep_step):
        target = self.standing_pd_target(a)
        if self.startup_steps:
            return blend_startup_target(target, reset_q, ep_step, self.startup_mask, self.startup_steps)
        return target

    def effective_action(self, target):
        return (target - self.default_q) / self.alpha

    def target_out_of_range_mask(self, a):
        target = self.standing_raw_target(a)
        return (target < self.kin.q_lo) | (target > self.kin.q_hi)

    def target_clip_mask(self, a):
        if self.unclipped_pd_target:
            return torch.zeros_like(a, dtype=torch.bool)
        return self.target_out_of_range_mask(a)

    def joint_range_pd_target(self, a):
        # 等价于 q_mid + alpha * a，端点直接还原上下限。
        return torch.lerp(self.kin.q_lo, self.kin.q_hi, (a + 1.) * .5)

    def _reward_pd_target(self, a, f):
        if self.action_mapping == 'neutral':
            return super()._reward_pd_target(a, f)
        if self.action_mapping == 'joint_range':
            return self.joint_range_pd_target(a)
        if self.action_mapping == 'standing_offset':
            return self.standing_pd_target(a)
        # 奖励评估的是步后状态。已有力矩项使用该步实际下发的目标。
        return self.residual_pd_target(a, (f - 1).clamp_min(0))

    def reward(self, S, f, a, a_prev, terms=False, *, command=None, ep_step=None, a_prev2=None, pd_target=None):
        result = super().reward(S, f, a, a_prev, terms, command=command, ep_step=ep_step,
                                a_prev2=a_prev2, pd_target=pd_target)
        if terms:
            _, values = result
            ref = self._reference(f, command)
            values['root_position_error'] = (S[0] - ref['root_p'][:, 0]).norm(dim=-1)
        return result

    def contract(self):
        value = super().contract()
        value.update(mode='roban_m4_f2', foot_frame='F2_source_sole; current_robot_heading',
                     reset='first_frame_without_domain_randomization', duration_s=getattr(self.motion, 'duration', None))
        if self.action_mapping == 'reference_residual':
            value.update(version=2, action_mapping=self.action_mapping,
                         wm_action='neutral_equivalent_of_limit_clamped_pd_target')
        elif self.action_mapping == 'joint_range':
            value.update(version=3, action_mapping=self.action_mapping,
                         wm_action='tanh_normalized_joint_range')
        elif self.action_mapping == 'standing_offset':
            value.update(version=4, action_mapping=self.action_mapping,
                         wm_action='standing_equivalent_of_limit_clamped_pd_target',
                         replay_action='noisy_linear_output', action_rate_input='linear_action')
            if self.unclipped_pd_target:
                value.update(version=5, pd_target_clip=False, wm_action='raw_linear_pd_target')
        if self.action_history_enabled:
            value.update(version=6, startup_steps=self.startup_steps, startup_clock='episode_control_step',
                         wm_action='standing_equivalent_of_executed_pd_target',
                         smoothness_history='two_previous_raw_actions_within_episode')
        return value

    def physics_contract(self, args):
        value = physical_contract(obs_dim=self.obs_dim)
        value['solver'] = {**value['solver'], **m4_solver_overrides(args), 'num_substeps': m4_substeps(args)}
        if self.limit_time_constant is not None:
            value['solver']['joint_limit_solref'] = [self.limit_time_constant, 1.]
        value['actuators'] = {name: {**params, 'damping': self.pd_damping.get(name, params['damping'])}
                              for name, params in value['actuators'].items()}
        if self.action_mapping == 'joint_range':
            lo, hi = self.kin.q_lo, self.kin.q_hi
            value.update(action_mapping=self.action_mapping, action_limits=[lo.tolist(), hi.tolist()],
                         action_center=((lo + hi) * .5).tolist(), action_scale=((hi - lo) * .5).tolist())
        elif self.action_mapping == 'standing_offset':
            lo, hi = self.kin.q_lo, self.kin.q_hi
            value.update(action_mapping=self.action_mapping, action_limits=[lo.tolist(), hi.tolist()],
                         action_center=self.default_q.tolist(), action_scale=self.alpha.tolist())
        return value

    def assert_wm_statistics(self, checkpoint):
        wm = checkpoint['wm']
        if float(wm['out_norm.count']) == 0.:
            return
        # 由单步状态守卫推导输入 WM 的增量二阶矩上界。
        scalar_limits = [ROOT_STEP_MAX_M] * 3 + [math.pi] * 3
        scalar_limits += [2 * ROOT_SPEED_MAX_M_S] * 3 + [2 * ROOT_ANGULAR_SPEED_MAX_RAD_S] * 3
        limits = torch.cat((self.kin.q_lo.new_tensor(scalar_limits),
                            self.kin.q_hi - self.kin.q_lo + 2 * JOINT_LIMIT_SLACK_RAD,
                            2 * self.joint_speed_limit))
        mean = wm['out_norm.mean']
        var = wm['out_norm.var']
        rms = (mean.square() + var).sqrt()
        valid = (var >= 0) & (rms <= limits.to(rms.device) * 1.01)
        if not valid.all():
            idx = int((~valid).nonzero(as_tuple=True)[0][0])
            raise ValueError(f'M4 检查点 WM 增量统计越界：通道 {idx} 的 RMS={float(rms[idx]):.4g}，'
                             f'上界={float(limits[idx]):.4g}。请选择异常发生前的检查点。')

    def assert_checkpoint_task(self, checkpoint):
        saved = checkpoint.get('task_contract') or {}
        saved_mapping = saved.get('action_mapping', 'neutral')
        if saved_mapping != self.action_mapping:
            raise ValueError(f'M4 检查点动作映射为 {saved_mapping}，当前配置为 {self.action_mapping}')
        if self.action_mapping == 'standing_offset' and saved.get('pd_target_clip', True) != (not self.unclipped_pd_target):
            raise ValueError('检查点位置目标裁剪契约与当前配置不一致')
        if saved != self.contract():
            raise ValueError('M4 检查点任务契约与当前配置不一致')


def build_m4_task(args, kin, files, device):
    if len(files) != 1:
        raise ValueError('M4.0 requires exactly one F2 reference')
    motion = F2MotionReference(files[0], kin, device, args.reference_fps)
    spec = F2GoalSpec('roban_foot', args.goal_frames, args.goal_dt)
    return build_reference_tracking(args, kin, motion, spec)


def build_reference_tracking(args, kin, motion, spec, mdp_type=F2TrackingMDP):
    goals = F2Goals(motion, kin, spec)
    mapping = getattr(args, 'm4_action_mapping', 'neutral')
    default_q = default_joint_pos_list()
    if mapping == 'standing_offset':
        pose = getattr(args, 'm4_standing_pose', None)
        if not isinstance(pose, dict) or set(pose) != set(JOINT_NAMES):
            raise ValueError('M4.2 固定站姿必须按名称给出全部 21 个 Roban 关节角')
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in pose.values()):
            raise ValueError('M4.2 固定站姿必须使用有限数值')
        default_q = [pose[name] for name in JOINT_NAMES]
    default_q = kin.q_lo.new_tensor(default_q)
    rewards = dict(args.roban_reward_config, body_prior_weight=args.body_prior_weight,
                   upper_action_smoothness_weight=getattr(args, 'upper_action_smoothness_weight', 0.),
                   soft_body_prior_weight=getattr(args, 'soft_body_prior_weight', 0.),
                   soft_body_prior_tolerance=getattr(args, 'soft_body_prior_tolerance', None))
    cfg = getattr(mdp_type, 'reward_config_type', FootRewardConfig)(**rewards)
    terrain = make_terrain(kin, vars(args)) if args.wm_use_terrain else None
    mdp = mdp_type(kin, motion, default_q, goals, cfg, action_mapping=mapping,
                        critic_use_body_reference=args.critic_use_body_reference, terrain=terrain,
                        limit_time_constant=m4_limit_time_constant(args), pd_damping=m4_pd_damping(args),
                        action_scale_factor=getattr(args, 'm4_action_scale_factor', .25),
                        unclipped_pd_target=getattr(args, 'm4_unclipped_pd_target', False),
                        startup_steps=getattr(args, 'm47_startup_steps', 0))
    if getattr(args, 'm411_group', None) and mdp.hand_reference is None:
        mdp.prepare_hand_tracking()
    args.reward_config = cfg.omit_inactive_whole_body(asdict(cfg))
    if not mdp.action_history_enabled:
        args.reward_config.pop('upper_action_smoothness_weight')
    if not cfg.soft_body_prior_weight:
        del args.reward_config['soft_body_prior_weight'], args.reward_config['soft_body_prior_tolerance']
    args.motion_obs_mode = mdp.observation_mode
    return motion, goals, mdp, terrain


def make_m4_sim(device, args):
    cfg = make_sim(device, args)
    cfg.physics.num_substeps = m4_substeps(args)
    for key, value in m4_solver_overrides(args).items():
        setattr(cfg.physics.solver_cfg, key, value)
    if torch.device(device).type == 'cpu':
        # 仅显式选择 CPU 时切换求解器，用于本地诊断。
        cfg.physics.use_cuda_graph = False
        cfg.physics.solver_cfg.use_mujoco_cpu = True
    return cfg


class M4EpaEvaluationError(FloatingPointError):
    """EPA 使当前评估轨迹失效，评估器可从下一次 Reset 继续。"""


class M4TrackingSim(TrackingSim):
    def __init__(self, sim_cfg, num_envs, motion, mdp, gamma=0.99, eval_mode=False,
                 max_episode_steps=None, reset_noise=False, goals=None, scene_cfg=None):
        steps = int(motion.clip_len.max()) - 1
        if (getattr(motion, 'duration', None) is not None and abs(steps * CONTROL_DT - motion.duration) > 1e-6
                or abs(sim_cfg.dt - CONTROL_DT) > 1e-9):
            raise ValueError('参考帧、控制步与物理时钟不一致')
        if max_episode_steps is not None:
            if max_episode_steps < 1:
                raise ValueError('F2 episode length must be positive')
            steps = min(steps, max_episode_steps)
        self.physics_dt = sim_cfg.dt / sim_cfg.physics.num_substeps
        if scene_cfg is None and (mdp.limit_time_constant is not None or mdp.pd_damping):
            scene_cfg = TrackingSceneCfg(num_envs=num_envs, env_spacing=0., replicate_physics=True)
        if mdp.limit_time_constant is not None:
            set_scene_joint_limit(scene_cfg, mdp.limit_time_constant)
        for name, value in mdp.pd_damping.items():
            scene_cfg.robot.actuators[name].damping = value
        self.a_prev2 = mdp.default_q.new_zeros((num_envs, NUM_J)) if mdp.action_history_enabled else None
        self.startup_q = torch.empty_like(self.a_prev2) if self.a_prev2 is not None else None
        super().__init__(sim_cfg, num_envs, motion, mdp, gamma=gamma, eval_mode=eval_mode,
                         max_episode_steps=steps, reset_noise=reset_noise,
                         goals=goals, scene_cfg=scene_cfg, domain_randomization=False)
        self.max_episode_steps = steps
        self.dynamics = None
        self.joint_resets = 0
        self.epa_resets = 0
        self._reported_joint_position = False
        self._reported_joint_speed = False
        if mdp.action_mapping == 'standing_offset':
            self.default_q, self.alpha = mdp.default_q, mdp.alpha
        if mdp.action_mapping in M4_GUARDED_MAPPINGS:
            limits = self.robot.data.joint_pos_limits.torch[:, self.canonical_to_sim]
            expected = torch.stack((mdp.kin.q_lo, mdp.kin.q_hi), dim=-1).expand_as(limits)
            if not torch.allclose(limits, expected):
                raise ValueError('Newton 实际关节位置限位与 Roban MJCF 不一致')
            damping = self.robot.data.joint_damping.torch[:, self.canonical_to_sim]
            if not torch.allclose(damping, mdp.kd.expand_as(damping)):
                raise ValueError('Newton 实际执行器阻尼与 M4 物理配置不一致')
            # MJWarp 的溢出位按环境累积，控制步结束时可检查全部物理子步。
            data = getattr(self.physics._solver, 'mjw_data', None)
            self._solver_overflow = None
            if getattr(data, 'overflow', None) is not None:
                import warp as wp
                from mujoco_warp import OverflowType

                self._solver_overflow = wp.to_torch(data.overflow)
                self._solver_nefc = wp.to_torch(data.nefc)
                self._njmax = data.njmax
                self._overflow_type = OverflowType
                self._epa_overflow = int(OverflowType.EPA_HORIZON)
                self._fatal_overflow = int(OverflowType.NEFC | OverflowType.NJMAX_NNZ |
                                           OverflowType.BROADPHASE | OverflowType.NARROWPHASE |
                                           OverflowType.EPA_HORIZON)
                if mdp.action_mapping == 'standing_offset':
                    self._fatal_overflow = ~int(OverflowType.ITERATIONS | OverflowType.EPA_HORIZON)

    def sample_frames(self, n):
        # 每个 episode 均验证 F2 从第一帧开始，完整末帧由 clip_end 处理。
        frames = torch.zeros(n, dtype=torch.long, device=self.device)
        return frames, self.motion.frame_bin[frames]

    def reset(self, env_ids, frames=None):
        super().reset(env_ids, frames)
        if getattr(self, 'a_prev2', None) is not None and len(env_ids):
            self.a_prev2[env_ids] = 0.
            self.refresh_startup_pose(env_ids)
        overflow = getattr(self, '_solver_overflow', None)
        if overflow is not None and self.mdp.action_mapping == 'standing_offset':
            # 先完成 Newton 的局部重置和暖启动清理，再清除这些环境的 EPA 累积位。
            overflow[env_ids] &= ~self._epa_overflow
        self.done_eval[env_ids] = False
        self.failed_eval[env_ids] = False

    def refresh_startup_pose(self, env_ids=slice(None)):
        if self.startup_q is not None:
            q = self.state()[4][env_ids].clamp(self.mdp.kin.q_lo, self.mdp.kin.q_hi)
            self.startup_q[env_ids] = q

    def action_context(self, target, previous):
        if self.a_prev2 is None:
            return super().action_context(target, previous)
        prev2 = self.a_prev2.clone()
        self.a_prev2.copy_(previous)
        return (dict(a_prev2=prev2, pd_target=target),
                dict(a_prev2=prev2, a_exec=self.mdp.effective_action(target),
                     startup_q=self.startup_q.clone(), pd_target=target))

    def pd_target(self, a, xi=None):
        mapping = self.mdp.action_mapping
        if mapping in M4_GUARDED_MAPPINGS:
            if a.shape != self.a_prev.shape:
                raise ValueError(f'期望 {self.a_prev.shape} 的 Roban 动作，收到 {a.shape}')
            if mapping == 'standing_offset':
                torch._assert_async(a.isfinite().all(), 'M4.2 动作必须为有限值')
                if self.mdp.startup_steps:
                    return self.mdp.executed_target(a, self.startup_q, self.ep_step)
                return self.mdp.standing_pd_target(a)
            torch._assert_async((a.abs() <= 1.).all(), '关节范围动作必须有限且位于 [-1, 1]')
            return self.mdp.joint_range_pd_target(a)
        if mapping == 'reference_residual':
            return self.mdp.residual_pd_target(a, self.f)
        return super().pd_target(a, xi)

    def step(self, a):
        if self.mdp.action_mapping == 'reference_residual':
            a = bound_action(a)
        elif self.mdp.action_mapping == 'standing_offset' and self._solver_overflow is not None:
            # 迭代上限位会累积。清除旧记录，检查迭代到顶和 EPA 是否在同一控制步发生。
            self._solver_overflow.bitwise_and_(~int(self._overflow_type.ITERATIONS))
        return super().step(a)

    def validate_physics_step(self, before, after, target):
        if self.mdp.action_mapping not in M4_GUARDED_MAPPINGS:
            return
        p, R, v, w, q, qd = after
        dp = (p - before[0]).norm(dim=-1)
        v_norm, w_norm = v.norm(dim=-1), w.norm(dim=-1)
        finite = torch.cat((p, R.flatten(1), v, w, q, qd), -1).isfinite().all(-1)
        kin = self.mdp.kin
        position_excess = torch.maximum(kin.q_lo - q, q - kin.q_hi)
        speed_excess = qd.abs() - self.mdp.joint_speed_limit
        position_fault = (position_excess > JOINT_LIMIT_SLACK_RAD).any(-1)
        speed_fault = (speed_excess > 0.).any(-1)
        joint_fault = position_fault | speed_fault
        fatal = ~(finite & (dp <= ROOT_STEP_MAX_M) & (v_norm <= ROOT_SPEED_MAX_M_S) &
                  (w_norm <= ROOT_ANGULAR_SPEED_MAX_RAD_S))
        epa_recoverable = torch.zeros_like(fatal)
        overflow = self._solver_overflow
        if overflow is not None:
            fatal |= (overflow & self._fatal_overflow) != 0
            if self.mdp.action_mapping == 'standing_offset':
                epa = (overflow & self._epa_overflow) != 0
                # EPA 可能漏掉一个接触。累积位只有 EPA 且状态检查通过时，可局部丢弃。
                epa_recoverable = (overflow == self._epa_overflow) & ~(fatal | joint_fault)
                fatal |= epa & ~epa_recoverable
        invalid = fatal | joint_fault | epa_recoverable
        if self.dynamics is not None:
            self.dynamics.record(before, after, target, position_excess, invalid, fatal, epa_recoverable)
        if not invalid.any():
            return

        # 遇到不可信转移时结束当前评估轨迹。训练只重置可恢复的环境。
        has_fatal = bool(fatal.any())
        stop = self.eval_mode or has_fatal
        report = fatal if has_fatal else invalid
        if not stop:
            self.joint_resets += int(joint_fault.count_nonzero())
            report = epa_recoverable.clone() if self.epa_resets == 0 else torch.zeros_like(invalid)
            self.epa_resets += int(epa_recoverable.count_nonzero())
            if not self._reported_joint_position:
                report |= position_fault
                self._reported_joint_position = bool(position_fault.any())
            if not self._reported_joint_speed:
                report |= speed_fault
                self._reported_joint_speed = bool(speed_fault.any())
            if not report.any():
                return ~invalid

        details = []
        bad = report.nonzero(as_tuple=True)[0]
        for i in bad[:4].tolist():
            flags = ''
            if self._solver_overflow is not None:
                overflow = self._overflow_type(int(self._solver_overflow[i])).name or '0'
                flags = f' overflow={overflow} nefc(末子步)={int(self._solver_nefc[i])}/{self._njmax}'
            joint = ''
            if position_fault[i]:
                j = int(position_excess[i].argmax())
                joint = (f' joint={JOINT_NAMES[j]} q_before={float(before[4][i, j]):.4g}rad '
                         f'q={float(q[i, j]):.4g}rad q_target={float(target[i, j]):.4g}rad '
                         f'limit=[{float(kin.q_lo[j]):.4g},{float(kin.q_hi[j]):.4g}]rad '
                         f'excess={float(position_excess[i, j]):.4g}rad qd={float(qd[i, j]):.4g}rad/s')
            if speed_fault[i]:
                j = int(speed_excess[i].argmax())
                joint += (f' fast_joint={JOINT_NAMES[j]} qd_before={float(before[5][i, j]):.4g}rad/s '
                          f'qd={float(qd[i, j]):.4g}rad/s speed_limit={float(self.mdp.joint_speed_limit[j]):.4g}rad/s')
            details.append(f'env={i} F2帧={int(self.f[i]) + 1} Δp={float(dp[i]):.4g}m '
                           f'|v|={float(v_norm[i]):.4g}m/s |ω|={float(w_norm[i]):.4g}rad/s '
                           f'|qd|max={float(qd[i].abs().max()):.4g}rad/s '
                           f'状态有限={bool(finite[i])}{flags}{joint}')
        detail = ' | '.join(details)
        if stop:
            error = M4EpaEvaluationError if not has_fatal and not joint_fault.any() else FloatingPointError
            raise error(f'M4 Newton 异常，共 {len(bad)} 个环境，已拦截本批转移。{detail}')
        print(f'M4 Newton 异常，丢弃对应转移并重置环境。{detail}', flush=True)
        return ~invalid

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

    from ..kinematics import RobotKinematics

    parser = argparse.ArgumentParser(description=physics_smoke.__doc__)
    parser.add_argument('--m4_reference', required=True)
    parser.add_argument('--num_envs', type=int, choices=range(1, 9), default=1)
    parser.add_argument('--steps', type=int, default=16, help='最多执行的控制步数，首次跌倒时停止')
    add_m4_action_mapping_arg(parser)
    add_m4_solver_args(parser)
    parser.add_argument('--action_test', action='store_true', help='测量零动作及固定探针对应的目标、力矩和接触')
    parser.add_argument('--actor_checkpoint', help='在动作对照中额外评估一个冻结的中立映射 Actor')
    add_launcher_args(parser)
    parser.set_defaults(reference_fps=None, goal_frames=10, goal_dt=.1, roban_reward_config={},
                        body_prior_weight=0, wm_use_terrain=False, critic_use_body_reference=False)
    args = parser.parse_args()
    if not 1 <= args.steps <= 260:
        parser.error('--steps 必须介于 1 和 260')
    if args.actor_checkpoint and not args.action_test:
        parser.error('--actor_checkpoint 需要 --action_test')
    if args.m4_action_mapping in M4_GUARDED_MAPPINGS:
        args.critic_use_body_reference = True
    device, steps = args.device or 'cuda:0', args.steps
    if torch.device(device).type != 'cuda' or not torch.cuda.is_available():
        parser.error('MJWarp 物理检查需要可用的 CUDA 设备')
    motion, goals, mdp, _ = build_m4_task(args, RobotKinematics(device), [args.m4_reference], device)
    cfg = make_m4_sim(device, args)
    with launch_simulation(cfg, args):
        env = M4TrackingSim(cfg, args.num_envs, motion, mdp, eval_mode=True, goals=goals, max_episode_steps=steps)
        ids = torch.arange(env.N, device=device)
        initial = env.state()
        reset_error = max(float((x - y).abs().max()) for x, y in zip(initial, motion.ref_state(env.f)))
        if not all(x.isfinite().all() for x in initial) or reset_error >= 1e-4:
            raise AssertionError('重置后的状态与 F2 起始帧不一致')
        names = env.robot.joint_names
        if [names[i] for i in env.canonical_to_sim.tolist()] != JOINT_NAMES or env.a_prev.shape != (env.N, NUM_J):
            raise AssertionError('Roban 的关节名称或动作维度不一致')
        state = env.robot.data
        torch.testing.assert_close(env.alpha, mdp.alpha)
        limits = torch.stack((mdp.kin.q_lo, mdp.kin.q_hi), dim=-1)
        torch.testing.assert_close(state.joint_pos_limits.torch[:, env.canonical_to_sim], limits.expand(env.N, -1, -1))
        if mdp.action_mapping == 'joint_range':
            for a, expected in ((-1., limits[:, 0]), (0., limits.mean(-1)), (1., limits[:, 1])):
                target = env.pd_target(torch.full_like(env.a_prev, a))
                torch.testing.assert_close(target, expected.expand_as(target))
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
        torque_w = wp.to_torch(solver.mjw_data.qfrc_actuator)
        if args.action_test:
            from .m4_action_diagnostics import run_action_diagnostics

            run_action_diagnostics(env, contacts, torque_w, dofs, steps, args.actor_checkpoint)
            return
        feet = [env.robot.body_names.index(name) for name in FOOT_BODY_NAMES]
        peak_force = peak_torque = peak_ratio = fk_error = 0.
        saturated = 0
        first_unload = None
        first_lift, first_touch = float(motion.liftoff_time[0]), float(motion.touchdown_time[0])
        first_side = int(motion.touchdown_side[0])
        for tick in range(steps):
            # 中立映射沿用原烟测的下一帧目标，其余映射发送零动作。
            action = torch.zeros_like(env.a_prev)
            if mdp.action_mapping == 'neutral':
                action = bound_action((motion.q[tick + 1] - env.default_q) / env.alpha).expand(env.N, -1)
            result = env.step(action)
            if mdp.action_mapping == 'joint_range' and tick == 0:
                actual_target = state.joint_pos_target.torch[:, env.canonical_to_sim]
                torch.testing.assert_close(actual_target, limits.mean(-1).expand_as(actual_target))
            if not (result['s1'].isfinite().all() and result['r'].isfinite().all() and (env.f == tick + 1).all()):
                raise AssertionError('Newton 状态、奖励或参考时钟无效')
            torque = torque_w[:, dofs].abs()
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
        print(json.dumps(dict(status='PASS', action_mapping=mdp.action_mapping, envs=env.N, steps=tick + 1, elapsed_s=elapsed, ended_by=ended_by,
                              first_planned_swing_foot_unloaded_s=first_unload, actions=NUM_J, control_dt=cfg.dt,
                              physics_dt=env.physics_dt, reset_max_abs_error=reset_error,
                              repeat_reset_max_abs_error=repeat_error, max_sole_fk_error_m=fk_error,
                              max_foot_ground_force_z_N=peak_force, peak_abs_joint_torque_Nm=peak_torque,
                              peak_effort_ratio=peak_ratio,
                              last_substep_saturation_fraction=saturated / ((tick + 1) * env.N * NUM_J))),
              flush=True)


if __name__ == '__main__':
    physics_smoke()
