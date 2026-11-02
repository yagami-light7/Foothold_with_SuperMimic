"""Minimal DirectRLEnv for KUAVO S54 + Newton MJWarp."""

from __future__ import annotations

from collections.abc import Sequence
import torch

from isaaclab.envs import DirectRLEnv

from tasks.foot_placement.spec import (
    ACTION_DIM,
    ACTOR_OBS_DIM,
    get_joint_indices,
    JOINT_NAMES,
)

from tasks.foot_placement.observations import(
    build_actor_observation,
)

from tasks.foot_placement.foot_placement_env_cfg import(
    S54FootPlacementEnvCfg,
)

from tasks.foot_placement.goal_sampler import(
    FootPlacementGoalSampler,
)

from tasks.foot_placement.foot_frames import(
    get_sole_pose, get_sole_velocity,
)

from tasks.foot_placement.rewards import (
    FootRewardCfg,
    compute_foot_placement_reward,
    reward_reference_upper,
    episode_return_metrics,
)
from tasks.foot_placement.heading import heading_error
from configs.s54_curriculum import sync_observation_space
from configs.walk_training import PROFILE_DEFAULTS, group_counts, validate_profile
from .robustness import S54Robustness
from configs.s54_flat_reward import FlatWalkRewardCfg
from .first_step_metrics import FirstStepMetrics


class S54FootPlacementEnv(DirectRLEnv):
    """First-stage S54 RL environment.

    This version validates:
    - Joint position actions
    - 103D actor observations
    - Robot resets
    - Newton simulation stepping

    Foot-placement goals and rewards are placeholders.
    """

    cfg:S54FootPlacementEnvCfg  

    def __init__(
            self, 
            cfg: S54FootPlacementEnvCfg, 
            render_mode: str | None = None, 
            **kwargs
    ):
        # 基类初始化时会创建Gym观测空间，因此必须提前确定维度。
        sync_observation_space(cfg)
        validate_profile({name: getattr(cfg, name) for name in PROFILE_DEFAULTS})
        if cfg.command_groups and (not cfg.observe_gait_frequency or cfg.gait_frequency_obs_dim != 2):
            raise ValueError("混合命令要求105维左右节奏观测")
        independent = any(getattr(cfg, name) is not None for name in (
            "left_gait_frequency", "right_gait_frequency", "left_gait_frequency_range", "right_gait_frequency_range"))
        if independent and (not cfg.observe_gait_frequency or cfg.gait_frequency_obs_dim != 2):
            raise ValueError("左右独立节奏需要observe_gait_frequency=True且gait_frequency_obs_dim=2")

        super().__init__(cfg, render_mode, **kwargs)

        # The scene has already been constructed by DirectRLEnv.
        self.robot = self.scene["robot"]

        self.foot_contact = self.scene["foot_contact"]

        self.contact_left_id = (
            self.foot_contact.sensor_names.index("leg_l6_link")
        )

        self.contact_right_id = (
            self.foot_contact.sensor_names.index("leg_r6_link")
        )

        self.actions = torch.zeros(
            self.num_envs,
            ACTION_DIM,
            device=self.device
        )
        self.raw_action_clip_mask: torch.Tensor | None = None

        # Previous policy actions for action-rate penalty.
        self.prev_actions = torch.zeros_like(self.actions)

        # First-stage foot-placement reward configuration.
        self.reward_cfg = cfg.reward_cfg

        # Store reward terms for debugging.
        self.reward_terms = {}
        # 归一化回报只用于日志，不反馈给PPO，保留原实验的奖励尺度。
        self.reward_reference = torch.zeros(self.num_envs, device=self.device)
        self._episode_reward_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_reference_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_reward_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._swing_has_lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._reward_swing_has_lifted = self._swing_has_lifted.clone()
        self.first_step_metrics = FirstStepMetrics(self.num_envs, self.device)

        print("[S54] Contact sensor names:",
            self.foot_contact.sensor_names)

        print("[S54] Ground filter objects:",
            self.foot_contact.filter_object_names)

        # Map our canonical 27DoF order to the actual solver order.
        joint_indices = get_joint_indices(self.robot.joint_names)

        self.joint_ids = torch.tensor(
            joint_indices,
            dtype=torch.long,
            device=self.device,
        )

        if not (0.0 < cfg.action_smoothing <= 1.0):
            raise ValueError("action_smoothing must be in (0, 1]")
        scales = cfg.action_scale_by_joint
        if scales is not None and len(scales) != ACTION_DIM:
            raise ValueError("action_scale_by_joint must follow the canonical 27-joint order")
        self.action_scales = torch.tensor(
            scales if scales is not None else [cfg.action_scale] * ACTION_DIM,
            device=self.device,
        ).unsqueeze(0)
        if not torch.all(self.action_scales > 0):
            raise ValueError("All action scales must be positive")

        self.actions = torch.zeros(
            self.num_envs,
            ACTION_DIM,
            device=self.device,
        )

        # Full joint target buffer in simulation joint order.
        self.joint_targets = (
            self.robot.data.default_joint_pos.torch.clone()
        )

        # 16D foot-placement goal.
        self.left_foot_id = self.robot.body_names.index("leg_l6_link")
        self.right_foot_id = self.robot.body_names.index("leg_r6_link")

        self._reset_foot_poses = tuple(
            pose.clone() for pose in self._get_foot_poses()
        )
        # 缓存每个环境真正写入的reset根部朝向，供部分reset重建足印参考。
        self._reset_root_quat_w = self.robot.data.root_quat_w.torch.clone()

        self.goal_sampler = FootPlacementGoalSampler(
            num_envs=self.num_envs,
            device=self.device,
            gait_frequency=cfg.gait_frequency,
            step_length=cfg.step_length,
            feet_distance=cfg.feet_distance,
            retain_stance_targets=cfg.retain_stance_targets,
            planar_targets=cfg.planar_targets,
            flat_ground_height=cfg.flat_ground_height,
            fixed_heading_targets=cfg.fixed_heading_targets,
            step_length_range=cfg.step_length_range,
            gait_frequency_range=cfg.gait_frequency_range,
            left_gait_frequency=cfg.left_gait_frequency,
            right_gait_frequency=cfg.right_gait_frequency,
            left_gait_frequency_range=cfg.left_gait_frequency_range,
            right_gait_frequency_range=cfg.right_gait_frequency_range,
            command_groups=cfg.command_groups,
            command_resample_half_cycles=cfg.command_resample_half_cycles,
            command_max_length_change=cfg.command_max_length_change,
            command_max_frequency_change=cfg.command_max_frequency_change,
            command_warmup_half_cycles=cfg.command_warmup_half_cycles,
            initial_swing_side=cfg.initial_swing_side,
        )
        self.robustness = S54Robustness(self)
        self._command_slices = []
        start = 0
        if cfg.command_groups:
            for group, count in zip(cfg.command_groups, group_counts(cfg.command_groups, self.num_envs)):
                if count:
                    self._command_slices.append((group["name"], slice(start, start + count)))
                start += count


    # 获取足部位姿
    def _get_foot_poses(self):
        """Read both foot-link poses in world coordinates."""
        body_pos = self.robot.data.body_pos_w.torch
        body_quat = self.robot.data.body_quat_w.torch

        left_pos_w = body_pos[:, self.left_foot_id, :]
        left_quat_w = body_quat[:, self.left_foot_id, :]

        right_pos_w = body_pos[:, self.right_foot_id, :]
        right_quat_w = body_quat[:, self.right_foot_id, :]

        left_sole_pos, left_sole_quat = get_sole_pose(
            left_pos_w, left_quat_w
        )

        right_sole_pos, right_sole_quat = get_sole_pose(
            right_pos_w, right_quat_w
        )

        return(
            left_sole_pos,
            left_sole_quat,
            right_sole_pos,
            right_sole_quat,
        )


    def reset(self, seed=None, options=None):
        """Reset robots and synchronize initial foot targets."""

        # Base reset restores robot state and runs sim.forward().
        _, extras = super().reset(
            seed=seed,
            options=options,
        )

        # Refresh articulation data after forward kinematics.
        self.scene.update(dt=0.0)

        # Cache foot poses from the actual reset configuration.
        self._reset_foot_poses = tuple(
            pose.clone()
            for pose in self._get_foot_poses()
        )

        # Rebuild foot-placement goals using valid foot poses.
        self.goal_sampler.reset(
            *self._reset_foot_poses,
            reference_quat_w=self.robot.data.root_quat_w.torch,
        )
        self._reset_root_quat_w.copy_(self.robot.data.root_quat_w.torch)

        # A manual full reset must not retain old reset flags.
        self.reset_buf.zero_()

        # Recompute observations with synchronized goals.
        self.obs_buf = self._compute_observations()

        return self.obs_buf, extras

    # 更新phase信息
    def _reset_envs_from_buffer(self):
        """Advance action history and gait phase, then reset."""

        # Reward has already been computed at this point.

        # Save the action that produced the current transition.
        self.prev_actions.copy_(self.actions)

        # Advance the gait target exactly once per policy step.
        foot_poses = self._get_foot_poses()

        old_swing = self.goal_sampler.left_is_swing.clone()
        first = self.goal_sampler.completed_half_cycles == 0
        self.goal_sampler.step(
            dt=self.step_dt,
            left_pos_w=foot_poses[0],
            left_quat_w=foot_poses[1],
            right_pos_w=foot_poses[2],
            right_quat_w=foot_poses[3],
        )
        self._swing_has_lifted &= old_swing == self.goal_sampler.left_is_swing
        valid = ~(self.reset_terminated | self.reset_time_outs)
        finished = first & ((old_swing != self.goal_sampler.left_is_swing) | ~valid)
        # 指标在自动reset之前结算，因此首步摔倒仍计为失败。
        log = dict(self.extras.get("log", {}))
        log.update(self.first_step_metrics.finish(finished, valid, self.goal_sampler.initial_left_is_swing))
        self.extras["log"] = log

        # This calls _reset_idx() for terminated environments.
        return super()._reset_envs_from_buffer()

    
    # policy step 开始前
    def _pre_physics_step(
            self, 
            actions: torch.Tensor
    ) -> None:
        """Convert 27D policy actions into PD position targets."""
        self.robustness.before_step()
        
        clamped_actions = torch.clamp(
            actions,
            min=-1.0,
            max=1.0,
        ).clone()
        alpha = self.cfg.action_smoothing
        self.actions = (1.0 - alpha) * self.actions + alpha * clamped_actions

        default_pos = (
            self.robot.data.default_joint_pos.torch
        )

        # Start with the validated standing pose.
        self.joint_targets.copy_(default_pos)

        # Add action offsets in canonical joint order.
        self.joint_targets[:, self.joint_ids] = (
            default_pos[:, self.joint_ids]
            + self.action_scales * self.actions
        )

        # Respect S54 soft joint position limits.
        limits = (
            self.robot.data.soft_joint_pos_limits.torch
        )

        self.joint_targets.clamp_(
            min=limits[..., 0],
            max=limits[..., 1],
        )

    # 物理仿真推进前 向PD控制器发送目标
    def _apply_action(self) -> None:
        """Send position targets to the existing PD actuators."""

        self.robot.actuators.target_command.set_position_index(
            value=self.joint_targets,
        )


    # 物理步进及必要的 Reset 后 获取Policy输入
    def _get_observations(
            self,
    ) -> dict[str, torch.Tensor]:
        """Build the 103D policy observation."""

        data = self.robot.data

        gravity = data.projected_gravity_b.torch
        ang_vel = data.root_ang_vel_b.torch

        joint_pos = (
            data.joint_pos.torch[:, self.joint_ids]
            - data.default_joint_pos.torch[:, self.joint_ids]
        )

        joint_vel = (
            data.joint_vel.torch[:, self.joint_ids]
        )
        gravity, ang_vel, joint_pos, joint_vel = self.robustness.noisy_observation(
            gravity, ang_vel, joint_pos, joint_vel)

        foot_poses = self._get_foot_poses()

        reset_mask = self.reset_buf.unsqueeze(-1)

        foot_poses = tuple(
            torch.where(reset_mask, default_pose, current_pose)
            for default_pose, current_pose in zip(
                self._reset_foot_poses, foot_poses
            )
        )

        # Convert stored world-frame targets into the
        # currently selected stance-foot frame.
        goal_obs = self.goal_sampler.get_observation(
            left_pos_w=foot_poses[0],
            left_quat_w=foot_poses[1],
            right_pos_w=foot_poses[2],
            right_quat_w=foot_poses[3],
        )

        # 104维仅兼容旧单频格式；新任务一次提供左右两个半周期节奏命令。
        gait_frequency_obs = (
            self.goal_sampler.gait_frequencies.unsqueeze(-1)
            if self.cfg.observe_gait_frequency and self.cfg.gait_frequency_obs_dim == 1
            else None
        )
        gait_commands = (
            torch.stack((self.goal_sampler.left_gait_frequencies,
                         self.goal_sampler.right_gait_frequencies), dim=-1)
            if self.cfg.observe_gait_frequency and self.cfg.gait_frequency_obs_dim == 2 else None
        )

        obs = build_actor_observation(
            projected_gravity=gravity,
            base_ang_vel=ang_vel,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            last_action=self.actions,
            goal_obs=goal_obs,
            gait_frequency=gait_frequency_obs,
            gait_frequencies=gait_commands,
        )

        # 实际输出必须与基类创建的Gym观测空间一致。
        assert obs.shape == (
            self.num_envs,
            self.cfg.observation_space,
        )

        return {"policy": obs}


    def flat_reward_state(self) -> dict[str, torch.Tensor]:
        """奖励所需的特权仿真状态；不添加到actor观测，不改变105维接口。"""
        data = self.robot.data
        result = {}
        for side, body in (("left",self.left_foot_id),("right",self.right_foot_id)):
            quat = data.body_quat_w.torch[:,body]
            result[side+'_quat'] = quat
            result[side+'_velocity'] = get_sole_velocity(quat,data.body_link_lin_vel_w.torch[:,body],
                                                        data.body_link_ang_vel_w.torch[:,body])
            result[side+'_target_quat'] = getattr(self.goal_sampler,side+'_target_quat_w')
        result.update(root_height=data.root_pos_w.torch[:,2]-self.scene.env_origins[:,2],
                      first_swing=self.goal_sampler.completed_half_cycles == 0,
                      nominal_height=data.default_root_pose.torch[:,2],
                      root_vz=data.root_lin_vel_w.torch[:,2],
                      upper_joint_offsets=(data.joint_pos.torch-data.default_joint_pos.torch)[:,self.joint_ids[12:]])
        return result

    # 每个 Policy Step 计算奖励
    def _get_rewards(self) -> torch.Tensor:
        """Compute actual foot-placement rewards."""

        # 1. Read actual sole poses.
        (
            left_pos_w,
            _,
            right_pos_w,
            _,
        ) = self._get_foot_poses()

        # 2. Read filtered ground reaction forces.
        left_force_w, right_force_w = (
            self._get_ground_contact_forces()
        )

        # 3. GoalSampler stores targets in world coordinates.
        sampler = self.goal_sampler
        swing_fz = torch.where(sampler.left_is_swing, left_force_w[:, 2], right_force_w[:, 2])
        swing_z = torch.where(sampler.left_is_swing, left_pos_w[:, 2], right_pos_w[:, 2])
        target_z = torch.where(sampler.left_is_swing, sampler.left_target_pos_w[:, 2], sampler.right_target_pos_w[:, 2])
        self._swing_has_lifted |= ((swing_fz < self.reward_cfg.contact_threshold)
                                & (swing_z - target_z > 0.005))
        self._reward_swing_has_lifted = self._swing_has_lifted.clone()

        # 4. Compute the reward using the CURRENT gait phase.
        reward, terms = compute_foot_placement_reward(
            left_pos_w=left_pos_w,
            right_pos_w=right_pos_w,

            left_target_w=sampler.left_target_pos_w,
            right_target_w=sampler.right_target_pos_w,

            left_force_w=left_force_w,
            right_force_w=right_force_w,

            phase=sampler.phase,
            left_is_swing=sampler.left_is_swing,

            projected_gravity_b=(
                self.robot.data.projected_gravity_b.torch
            ),

            actions=self.actions,
            prev_actions=self.prev_actions,

            dt=self.step_dt,
            cfg=self.reward_cfg,
            swing_has_lifted=self._reward_swing_has_lifted,
            root_quat_w=self.robot.data.root_quat_w.torch,
            reference_heading_w=sampler.reference_heading_w,
            root_yaw_rate=self.robot.data.root_ang_vel_b.torch[:, 2],
            flat_state=self.flat_reward_state() if isinstance(self.reward_cfg,FlatWalkRewardCfg) else None,
        )

        # 5. Keep raw terms for diagnostics.
        self.reward_terms = terms
        self.reward_reference = reward_reference_upper(sampler.phase, sampler.left_is_swing,
                                                       self.step_dt, self.reward_cfg,
                                                       sampler.completed_half_cycles == 0)
        # 只统计真正的第一个半周期；低频和右脚起步同样适用。
        swing_pos = torch.where(sampler.left_is_swing[:, None], left_pos_w, right_pos_w)
        swing_target = torch.where(sampler.left_is_swing[:, None], sampler.left_target_pos_w, sampler.right_target_pos_w)
        local = torch.where(sampler.left_is_swing, 2 * sampler.phase, 2 * sampler.phase - 1)
        self.first_step_metrics.observe(
            sampler.completed_half_cycles == 0, local, swing_fz,
            torch.where(sampler.left_is_swing, right_force_w[:, 2], left_force_w[:, 2]),
            swing_z - target_z, (swing_pos[:, :2] - swing_target[:, :2]).norm(dim=-1),
            ~(self.reset_terminated | self.reset_time_outs),
            self.reward_cfg.contact_threshold, self.reward_cfg.swing_height)
        self._episode_reward_sum += reward
        self._episode_reference_sum += self.reward_reference
        self._episode_reward_steps += 1

        # 6. Export mean per-step reward terms.
        # RSL-RL retains these dicts until the PPO update: never mutate an old one.
        log = dict(self.extras.get("log", {}))
        log = {key: value for key, value in log.items() if not key.startswith("Startup/")}
        # 完成episode的指标只记录在结束帧，避免被后续帧重复采样。
        for name in ("Episode/normalized_return", "Episode/normalized_quality", "Episode/survival_fraction"):
            log.pop(name, None)

        for name, value in terms.items():
            log[f"Reward/{name}"] = (
                value.mean() * self.step_dt
            )

        log["Reward/total"] = reward.mean()
        log["Reward/normalized"] = (reward / self.reward_reference.clamp_min(1e-8)).clamp(0, 1).mean()
        log["Command/initial_left_swing_fraction"] = sampler.initial_left_is_swing.float().mean()
        log["Command/left_gait_frequency_mean_hz"] = sampler.left_gait_frequencies.mean()
        log["Command/right_gait_frequency_mean_hz"] = sampler.right_gait_frequencies.mean()
        log["Command/effective_gait_frequency_mean_hz"] = sampler.gait_frequencies.mean()
        log["Command/left_swing_duration_mean_s"] = (0.5 / sampler.left_gait_frequencies).mean()
        log["Command/right_swing_duration_mean_s"] = (0.5 / sampler.right_gait_frequencies).mean()
        log["Command/nominal_forward_speed_mean_m_s"] = (2 * sampler.gait_frequencies * sampler.step_lengths).mean()
        if sampler.step_length_range is not None or self.cfg.command_groups:
            # 导出真正使用的命令分布；不要把cfg.step_length=0.03误当成当前随机目标。
            log["Command/step_length_mean_m"] = sampler.step_lengths.mean()
            log["Command/step_length_min_m"] = sampler.step_lengths.min()
            log["Command/step_length_max_m"] = sampler.step_lengths.max()
        if self.cfg.curriculum_stage == "straight":
            # 这里记录原始物理误差，越小越好；与上面的正向奖励分数区分开。
            log["Tracking/heading_error_rad"] = heading_error(
                self.robot.data.root_quat_w.torch, sampler.reference_heading_w).abs().mean()
            log["Tracking/yaw_rate_abs"] = self.robot.data.root_ang_vel_b.torch[:, 2].abs().mean()
        log["Termination/terminated_fraction"] = self.reset_terminated.float().mean()
        log["Termination/truncated_fraction"] = self.reset_time_outs.float().mean()
        # 每个控制步记录真正处于各组边界内的环境数；PPO取64步平均后乘64即transition数。
        in_bounds = sampler.commands_in_group_bounds() if self._command_slices else None
        for name, selection in self._command_slices:
            assert in_bounds is not None
            log[f"Group/{name}/command_count"] = in_bounds[selection].float().sum()
            log[f"Group/{name}/initial_left_swing_fraction"] = sampler.initial_left_is_swing[selection].float().mean()
            log[f"Group/{name}/reward"] = reward[selection].mean()
            log[f"Group/{name}/normalized_reward"] = (reward[selection] / self.reward_reference[selection].clamp_min(1e-8)).clamp(0, 1).mean()
            log[f"Group/{name}/terminated_fraction"] = self.reset_terminated[selection].float().mean()
            log[f"Group/{name}/step_length_m"] = sampler.step_lengths[selection].mean()
            log[f"Group/{name}/left_hz"] = sampler.left_gait_frequencies[selection].mean()
            log[f"Group/{name}/right_hz"] = sampler.right_gait_frequencies[selection].mean()
        if self.cfg.command_groups:
            data = self.robot.data
            error = self.joint_targets - data.joint_pos.torch
            # 隐式PD的策略步末估算，不能当成500Hz物理子步的实测峰值力矩。
            estimated = data.joint_stiffness.torch * error - data.joint_damping.torch * data.joint_vel.torch
            limits = data.joint_effort_limits.torch.clamp_min(1e-8)
            over_limit = (estimated.abs() >= limits).float().mean(0)[self.joint_ids]
            errors = error.abs().mean(0)[self.joint_ids]
            speeds = data.joint_vel.torch.abs().mean(0)[self.joint_ids]
            raw_clip = getattr(self, "raw_action_clip_mask", None)
            clip = None if raw_clip is None else raw_clip.float().mean(0)
            for index, name in enumerate(JOINT_NAMES):
                log[f"Control/position_error_rad/{name}"] = errors[index]
                log[f"Control/velocity_abs_rad_s/{name}"] = speeds[index]
                log[f"Control/estimated_pd_limit_fraction/{name}"] = over_limit[index]
                if clip is not None:
                    log[f"Control/action_clip_fraction/{name}"] = clip[index]
            log["Robustness/mass_scale_mean"] = self.robustness.mass_scales.mean()
            log["Robustness/push_count"] = self.robustness.push_counts.sum().float()
        self.extras["log"] = log

        # 7. Basic safety checks.
        if reward.shape != (self.num_envs,):
            raise RuntimeError(
                f"Invalid reward shape: {reward.shape}"
            )

        if not torch.isfinite(reward).all():
            raise RuntimeError(
                "Foot-placement reward contains NaN or Inf"
            )

        return reward


    # 每个 Policy Step 判断跌倒或超时
    def _get_dones(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return termination and timeout flags."""

        root_pos = self.robot.data.root_pos_w.torch

        relative_height = (
            root_pos[:, 2]
            - self.scene.env_origins[:, 2]
        )

        upright_projection = (
            -self.robot.data.projected_gravity_b.torch[:, 2]
        )

        fell = (
            (relative_height < self.cfg.termination_height)
            | (
                upright_projection
                < self.cfg.min_upright_projection
            )
        )

        timeout = (
            self.episode_length_buf
            >= self.max_episode_length
        )

        return fell, timeout


    # 重置环境
    def _reset_idx(
        self, 
        env_ids: Sequence[int],
    ):
        """Reset selected environments to the S54 stand pose."""

        if env_ids is None:
            env_ids = torch.arange(
                self.num_envs,
                device=self.device,
                dtype=torch.long,
            )

        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        completed = ids[self._episode_reward_steps[ids] > 0]
        if completed.numel():
            log = dict(self.extras.get("log", {}))
            # 返回向量，RSL-RL会拼接本次rollout中所有完成episode再求均值。
            log.update(episode_return_metrics(
                self._episode_reward_sum[completed], self._episode_reference_sum[completed],
                self._episode_reward_steps[completed], self.max_episode_length))
            self.extras["log"] = log
        self._episode_reward_sum[ids] = 0
        self._episode_reference_sum[ids] = 0
        self._episode_reward_steps[ids] = 0
        self.first_step_metrics.reset(ids)
        super()._reset_idx(env_ids)

        # Clear previous actions.
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self._swing_has_lifted[env_ids] = False

        # Copy the default robot pose.
        root_pose = (
            self.robot.data.default_root_pose.torch[env_ids]
            .clone()
        )

        # Each environment has a different world origin.
        root_pose[:, :3] += (
            self.scene.env_origins[env_ids]
        )

        root_vel = (
            self.robot.data.default_root_vel.torch[env_ids]
            .clone()
        )

        joint_pos = (
            self.robot.data.default_joint_pos.torch[env_ids]
            .clone()
        )

        joint_vel = (
            self.robot.data.default_joint_vel.torch[env_ids]
            .clone()
        )
        if self.cfg.reset_joint_position_noise:
            joint_pos += torch.empty_like(joint_pos).uniform_(
                -self.cfg.reset_joint_position_noise, self.cfg.reset_joint_position_noise)
            limits = self.robot.data.soft_joint_pos_limits.torch[env_ids]
            joint_pos.clamp_(min=limits[..., 0], max=limits[..., 1])
        if self.cfg.reset_root_velocity_noise:
            root_vel[:, :2] += torch.empty_like(root_vel[:, :2]).uniform_(
                -self.cfg.reset_root_velocity_noise, self.cfg.reset_root_velocity_noise)

        # Write the robot state into Newton.
        self.robot.write_root_pose_to_sim_index(
            root_pose=root_pose,
            env_ids=env_ids,
        )

        self.robot.write_root_velocity_to_sim_index(
            root_velocity=root_vel,
            env_ids=env_ids,
        )

        self.robot.write_joint_state_to_sim_index(
            position=joint_pos,
            velocity=joint_vel,
            env_ids=env_ids,
        )

        # Restore the standing PD targets.
        self.joint_targets[env_ids] = joint_pos
        self.robustness.reset(ids)
        if self.cfg.reset_joint_position_noise:
            # 随机姿态改变足端位置；先做FK再建目标，不能使用旧站姿的脚底缓存。
            self.sim.forward()
            self.scene.update(dt=0.)
            for cache, actual in zip(self._reset_foot_poses, self._get_foot_poses()):
                cache[ids] = actual[ids]

        # Restore gait phase and targets for selected environments.
        # 只更新被重置环境的参考，其他环境不能因一次局部reset而突然改变目标方向。
        self._reset_root_quat_w[env_ids] = root_pose[:, 3:7]
        self.goal_sampler.reset(
            *self._reset_foot_poses,
            env_ids=env_ids,
            reference_quat_w=self._reset_root_quat_w,
        )


    # 获取左右脚地面接触力
    def _get_ground_contact_forces(self):
        """Return left/right ground contact forces in world frame.

        Returns:
            left_force:  [N, 3]
            right_force: [N, 3]
        """

        force_matrix = (
            self.foot_contact.data.force_matrix_w
        )

        if force_matrix is None:
            raise RuntimeError(
                "Ground-filtered contact force matrix is unavailable"
            )

        # Shape: [N, num_sensors, num_ground_shapes, 3]
        forces = force_matrix.torch

        if forces.ndim != 4 or forces.shape[2] == 0:
            raise RuntimeError(
                f"Unexpected contact force shape: {forces.shape}"
            )

        # Sum forces from all matched ground collision shapes.
        forces = forces.sum(dim=2)

        left_force = forces[:, self.contact_left_id, :]
        right_force = forces[:, self.contact_right_id, :]

        return left_force, right_force
