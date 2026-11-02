"""Phase-based foot-placement goal sampler for S54.

Stage 1:
- Fixed forward step length
- Alternating left/right swing foot
- World-frame target caching
- Stance-frame 16D observation

Quaternion convention: xyzw.
"""

import math

import torch
from .heading import quat_from_yaw, yaw_from_quat
from .gait import advance_phase, effective_frequency, frequency_range
from configs.walk_training import group_counts, validate_command_groups
from isaaclab.utils.math import(
    combine_frame_transforms,
    subtract_frame_transforms,
)

class FootPlacementGoalSampler:

    def __init__(
        self,
        num_envs:int, 
        device:str, 
        gait_frequency:float=1.0, # 左右脚交替频率 0.5s换一次摆动脚
        step_length:float = 0.12, # 摆动脚相对支撑脚向X轴移动0.12m
        feet_distance:float=0.19, # 摆动脚相对支撑脚向Y轴移动0.19m
        retain_stance_targets: bool = False,
        planar_targets: bool = False,
        flat_ground_height: float | None = None,
        fixed_heading_targets: bool = False,  # 直行阶段启用；默认关闭以兼容旧模型。
        step_length_range: tuple[float, float] | None = None,
        gait_frequency_range: tuple[float, float] | None = None,
        left_gait_frequency: float | None = None,
        right_gait_frequency: float | None = None,
        left_gait_frequency_range: tuple[float, float] | None = None,
        right_gait_frequency_range: tuple[float, float] | None = None,
        command_groups: list[dict] | None = None,
        command_resample_half_cycles: int = 0,
        command_max_length_change: float = 0.02,
        command_max_frequency_change: float = 0.10,
        command_warmup_half_cycles: int = 0,
        initial_swing_side: str = "left",
    ):
        self.num_envs = num_envs
        self.device = device
        if initial_swing_side not in ("left", "right", "random"):
            raise ValueError("initial_swing_side必须为left/right/random")
        self.initial_swing_side = initial_swing_side

        # 固定步频必须为有限正数，单位 Hz。
        if not (math.isfinite(gait_frequency) and gait_frequency > 0.0):
            raise ValueError("gait_frequency 必须为有限正数")

        self.gait_frequency = float(gait_frequency)

        # 随机范围允许上下限相等，便于测试固定频率。
        if gait_frequency_range is not None:
            low, high = gait_frequency_range
            if not (
                math.isfinite(low)
                and math.isfinite(high)
                and 0.0 < low <= high
            ):
                raise ValueError(
                    "gait_frequency_range 必须满足 0 < 下限 <= 上限，且均为有限数"
                )
            gait_frequency_range = (float(low), float(high))

        self.gait_frequency_range = gait_frequency_range
        self.left_gait_frequency = left_gait_frequency
        self.right_gait_frequency = right_gait_frequency
        self.left_gait_frequency_range = frequency_range(left_gait_frequency_range, "left_gait_frequency_range")
        self.right_gait_frequency_range = frequency_range(right_gait_frequency_range, "right_gait_frequency_range")
        for side in ("left", "right"):
            fixed = getattr(self, f"{side}_gait_frequency")
            bounds = getattr(self, f"{side}_gait_frequency_range")
            if fixed is not None:
                if not (math.isfinite(fixed) and fixed > 0):
                    raise ValueError(f"{side}_gait_frequency必须为有限正数")
                if bounds is not None:
                    raise ValueError(f"{side}侧固定频率与随机范围不能同时指定")

        # 构造时先填固定值或范围下限；真正的随机采样在 reset 中进行。
        initial_frequency = (
            self.gait_frequency
            if self.gait_frequency_range is None
            else self.gait_frequency_range[0]
        )

        # 每个环境独立保存一个步频命令，下一步用于推进各自的相位。
        self.gait_frequencies = torch.full(
            (num_envs,),
            initial_frequency,
            dtype=torch.float32,
            device=device,
        )
        self.left_gait_frequencies = self.gait_frequencies.clone()
        self.right_gait_frequencies = self.gait_frequencies.clone()
        maxima = []
        for side in ("left", "right"):
            fixed = getattr(self, f"{side}_gait_frequency")
            bounds = getattr(self, f"{side}_gait_frequency_range")
            values = getattr(self, f"{side}_gait_frequencies")
            if bounds is not None:
                values.fill_(bounds[0])
                maxima.append(bounds[1])
            elif fixed is not None:
                values.fill_(fixed)
                maxima.append(fixed)
            else:
                maxima.append(gait_frequency_range[1] if gait_frequency_range is not None else self.gait_frequency)
        self.max_gait_frequency = max(maxima)
        self.gait_frequencies.copy_(effective_frequency(self.left_gait_frequencies, self.right_gait_frequencies))

        self.step_length = step_length
        if step_length_range is not None:
            low, high = step_length_range
            if not (math.isfinite(low) and math.isfinite(high) and 0 <= low <= high):
                raise ValueError("step_length_range必须满足0 <= 下限 <= 上限，且均为有限数")
            step_length_range = (float(low), float(high))
        self.step_length_range = step_length_range

        # 每个环境独立保存本episode的水平步长；默认None时保持原固定步长行为。
        self.step_lengths = torch.full(
            (num_envs,), 
            float(step_length), 
            dtype=torch.float32,
            device=device,
        )
        # 按环境编号固定分组；部分reset只重抽该组范围，困难环境不会变成简单环境。
        self.command_groups = validate_command_groups(command_groups)
        self.command_group_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._group_length_bounds = torch.zeros(num_envs, 2, device=device)
        self._group_frequency_bounds = torch.zeros_like(self._group_length_bounds)
        self._group_coupled = torch.ones(num_envs, dtype=torch.bool, device=device)
        if self.command_groups:
            start = 0
            for index, (group, count) in enumerate(zip(self.command_groups, group_counts(self.command_groups, num_envs))):
                end = start + count
                self.command_group_ids[start:end] = index
                self._group_length_bounds[start:end] = torch.tensor(group["step_length_range"], device=device)
                self._group_frequency_bounds[start:end] = torch.tensor(group["frequency_range"], device=device)
                self._group_coupled[start:end] = group["coupled"]
                self.max_gait_frequency = max(self.max_gait_frequency, group["frequency_range"][1])
                start = end
        self.command_resample_half_cycles = command_resample_half_cycles
        self.command_max_length_change = command_max_length_change
        self.command_max_frequency_change = command_max_frequency_change
        self.command_warmup_half_cycles = command_warmup_half_cycles
        if command_resample_half_cycles and not self.command_groups:
            raise ValueError("周期内重采样需要command_groups，防止隐式改变旧任务")
        self._command_half_cycles = torch.zeros(num_envs, dtype=torch.long, device=device)
        # 始终计数，不能使用仅在命令重采样时递增的_command_half_cycles判断首步。
        self.completed_half_cycles = torch.zeros_like(self._command_half_cycles)
        self._warmup_remaining = torch.zeros_like(self._command_half_cycles)
        self._desired_commands = torch.zeros(num_envs, 3, device=device)
        self.feet_distance = feet_distance
        self.retain_stance_targets = retain_stance_targets
        self.planar_targets = planar_targets
        self.flat_ground_height = flat_ground_height
        self.fixed_heading_targets = fixed_heading_targets
        # 每个环境保存自己的 reset 参考方向；支撑脚切换时不能更新它。
        self.reference_heading_w = torch.zeros(num_envs, device=device)
        self.reference_quat_w = quat_from_yaw(self.reference_heading_w)

        # 每个环境的周期进度
        self.phase = torch.zeros(num_envs, device=device)

        # True: left swing, right stance
        # False: right swing, left stance
        self.left_is_swing = torch.ones(
            num_envs,
            dtype=torch.bool,
            device=device
        )
        # 记录本episode从哪只脚开始，不能用随时间切换的left_is_swing代替。
        self.initial_left_is_swing = self.left_is_swing.clone()

        self.left_target_pos_w = torch.zeros(
            num_envs,
            3,
            device = device
        )

        self.right_target_pos_w = torch.zeros(
            num_envs,
            3,
            device=device
        )

        self.left_target_quat_w = torch.zeros(
            num_envs,
            4,
            device=device
        )

        self.right_target_quat_w = torch.zeros(
            num_envs,
            4,
            device=device
        )

        self.left_target_quat_w[:, 3] = 1.0
        self.right_target_quat_w[:, 3] = 1.0

    # 计算目标位姿
    def _target_quat(self, quat):
        if not self.planar_targets:
            return quat
        return quat_from_yaw(yaw_from_quat(quat))

    def _target_frame_quat(self, stance_quat, env_ids):
        """选择足印偏移的旋转基准：固定参考朝向，或旧版的当前支撑脚朝向。"""
        if self.fixed_heading_targets:
            return self.reference_quat_w[env_ids]
        return self._target_quat(stance_quat)

    @torch.no_grad()
    def _assign_targets(
        self,
        env_ids,
        left_pos_w,
        left_quat_w,
        right_pos_w,
        right_quat_w,
    ):
        """Sample targets at the start of each swing phase."""

        left_ids = env_ids[self.left_is_swing[env_ids]]
        right_ids = env_ids[~self.left_is_swing[env_ids]]

        # Right foot stance, left foot swing.
        if left_ids.numel() > 0:
            offset = torch.zeros_like(right_pos_w[left_ids])
            offset[:, 0] = self.step_lengths[left_ids]
            offset[:, 1] = self.feet_distance

            # 坐标变换得到目标位姿
            pos, quat = combine_frame_transforms(
                right_pos_w[left_ids],
                # 目标仍从实际支撑脚位置出发；直行模式只固定偏移方向，保留前进能力。
                self._target_frame_quat(right_quat_w[left_ids], left_ids),
                offset,
            )
            if self.flat_ground_height is not None:
                pos[:, 2] = self.flat_ground_height

            # 保存左脚目标
            self.left_target_pos_w[left_ids] = pos
            self.left_target_quat_w[left_ids] = quat

            # Keep right stance target fixed in world frame.
            if not self.retain_stance_targets:
                self.right_target_pos_w[left_ids] = right_pos_w[left_ids]
                self.right_target_quat_w[left_ids] = right_quat_w[left_ids]

        # Left foot stance, right foot swing.
        if right_ids.numel() > 0:
            offset = torch.zeros_like(left_pos_w[right_ids])
            offset[:, 0] = self.step_lengths[right_ids]
            offset[:, 1] = -self.feet_distance

            # 坐标变换得到目标位姿
            pos, quat = combine_frame_transforms(
                left_pos_w[right_ids],
                self._target_frame_quat(left_quat_w[right_ids], right_ids),
                offset,
            )
            if self.flat_ground_height is not None:
                pos[:, 2] = self.flat_ground_height

            # 保存右脚目标
            self.right_target_pos_w[right_ids] = pos
            self.right_target_quat_w[right_ids] = quat

            # Keep left stance target fixed in world frame.
            if not self.retain_stance_targets:
                self.left_target_pos_w[right_ids] = left_pos_w[right_ids]
                self.left_target_quat_w[right_ids] = left_quat_w[right_ids]

    # 重置环境
    @torch.no_grad()
    def commands_in_group_bounds(self):
        """逐环境核对实际命令；用于rollout覆盖统计，不把声明配额当成实采样。"""
        if not self.command_groups:
            raise RuntimeError('命令覆盖统计需要command_groups')
        tolerance = 1e-6  # 浮点端点允许微小舍入误差。
        low_d, high_d = self._group_length_bounds.unbind(dim=1)
        low_f, high_f = self._group_frequency_bounds.unbind(dim=1)
        return ((self.step_lengths >= low_d - tolerance) & (self.step_lengths <= high_d + tolerance)
                & (self.left_gait_frequencies >= low_f - tolerance)
                & (self.left_gait_frequencies <= high_f + tolerance)
                & (self.right_gait_frequencies >= low_f - tolerance)
                & (self.right_gait_frequencies <= high_f + tolerance)
                & (~self._group_coupled
                   | ((self.left_gait_frequencies - self.right_gait_frequencies).abs() <= tolerance)))

    def reset(
        self,
        left_pos_w,
        left_quat_w,
        right_pos_w,
        right_quat_w,
        env_ids = None,
        reference_quat_w: torch.Tensor | None = None,
    ):
        """Reset the specified environments"""

        # 重置所有环境
        if env_ids is None:
            env_ids = torch.arange(
                self.num_envs,
                device=self.device,
                dtype=torch.long,
            )

        env_ids = torch.as_tensor(
            env_ids, 
            device=self.device, 
            dtype=torch.long,
        )

        if self.initial_swing_side == "random":
            initial_left = torch.rand(env_ids.numel(), device=self.device) < .5
        else:
            # 固定模式不消耗随机数，保留旧实验的随机命令序列。
            initial_left = torch.full((env_ids.numel(),), self.initial_swing_side == "left",
                                      dtype=torch.bool, device=self.device)
        self.initial_left_is_swing[env_ids] = initial_left
        self.left_is_swing[env_ids] = initial_left
        # phase<0.5表示左摆动，phase>=0.5表示右摆动；必须同步更新，不能只改布尔值。
        self.phase[env_ids] = torch.where(initial_left, 0., .5)

        # 只在reset重新采样，整个episode内不变；切换左右脚时保持同一个步长。
        # 禁止每个50Hz控制步重采样，否则足印会在摆动途中跳变。
        if self.step_length_range is None:
            self.step_lengths[env_ids] = self.step_length
        else:
            low, high = self.step_length_range
            if low == high:
                self.step_lengths[env_ids] = low
            else:
                self.step_lengths[env_ids] = torch.empty(
                    env_ids.numel(), 
                    dtype=self.step_lengths.dtype,
                    device=self.device,
                ).uniform_(low, high)

        # 只在 reset 时更新步频，同一 episode 内保持不变。
        # env_ids 保证局部 reset 不影响其他仍在运行的环境。
        if self.gait_frequency_range is None:
            self.gait_frequencies[env_ids] = self.gait_frequency
        else:
            low, high = self.gait_frequency_range

            if low == high:
                # 固定频率不消耗随机数，便于复现与旧行为对照。
                self.gait_frequencies[env_ids] = low
            else:
                self.gait_frequencies[env_ids] = torch.empty(
                    env_ids.numel(),
                    dtype=self.gait_frequencies.dtype,
                    device=self.device,
                ).uniform_(low, high)

        # 共同命令只抽样一次并分配到两侧；指定单侧范围时再独立覆盖该侧。
        for side in ("left", "right"):
            values = getattr(self, f"{side}_gait_frequencies")
            fixed = getattr(self, f"{side}_gait_frequency")
            bounds = getattr(self, f"{side}_gait_frequency_range")
            values[env_ids] = self.gait_frequencies[env_ids]
            if bounds is not None:
                low, high = bounds
                if low == high:
                    values[env_ids] = low
                else:
                    values[env_ids] = torch.empty(env_ids.numel(), dtype=values.dtype,
                                                  device=self.device).uniform_(low, high)
            elif fixed is not None:
                values[env_ids] = fixed
        if self.command_groups:
            commands = self._draw_group_commands(env_ids)
            self.step_lengths[env_ids] = commands[:, 0]
            self.left_gait_frequencies[env_ids] = commands[:, 1]
            self.right_gait_frequencies[env_ids] = commands[:, 2]
        self._command_half_cycles[env_ids] = 0
        self.completed_half_cycles[env_ids] = 0
        self._warmup_remaining[env_ids] = self.command_warmup_half_cycles
        self._desired_commands[env_ids] = torch.stack((self.step_lengths[env_ids],
            self.left_gait_frequencies[env_ids], self.right_gait_frequencies[env_ids]), dim=-1)
        if self.command_warmup_half_cycles:
            # 高级索引返回副本，需要显式写回。
            self.step_lengths[env_ids] = self.step_lengths[env_ids].clamp_max(.12)
            self.left_gait_frequencies[env_ids] = self.left_gait_frequencies[env_ids].clamp_max(.75)
            self.right_gait_frequencies[env_ids] = self.right_gait_frequencies[env_ids].clamp_max(.75)
        self.gait_frequencies[env_ids] = effective_frequency(
            self.left_gait_frequencies[env_ids], self.right_gait_frequencies[env_ids])

        # 参考方向只在 reset 时记录，并且只修改 env_ids 对应的环境。
        # 正常环境传入根部姿态；单独测试 sampler 时允许回退到右脚姿态。
        source = right_quat_w if reference_quat_w is None else reference_quat_w
        self.reference_heading_w[env_ids] = yaw_from_quat(source[env_ids])
        self.reference_quat_w[env_ids] = quat_from_yaw(self.reference_heading_w[env_ids])

        self.left_target_pos_w[env_ids] = left_pos_w[env_ids]
        self.left_target_quat_w[env_ids] = left_quat_w[env_ids]
        self.right_target_pos_w[env_ids] = right_pos_w[env_ids]
        self.right_target_quat_w[env_ids] = right_quat_w[env_ids]
        if self.fixed_heading_targets:
            # 目标位置和目标姿态共用同一参考方向，方向误差可通过现有16维goal进入policy。
            self.left_target_quat_w[env_ids] = self.reference_quat_w[env_ids]
            self.right_target_quat_w[env_ids] = self.reference_quat_w[env_ids]

        # 生成对应环境的初始落脚目标
        self._assign_targets(
            env_ids=env_ids,
            left_pos_w=left_pos_w,
            left_quat_w=left_quat_w,
            right_pos_w=right_pos_w,
            right_quat_w=right_quat_w,
        )

    def _draw_group_commands(self, env_ids):
        lengths = self._group_length_bounds[env_ids]
        frequencies = self._group_frequency_bounds[env_ids]
        random = torch.rand(env_ids.numel(), 3, device=self.device)
        length = lengths[:, 0] + random[:, 0] * (lengths[:, 1] - lengths[:, 0])
        left = frequencies[:, 0] + random[:, 1] * (frequencies[:, 1] - frequencies[:, 0])
        independent_right = frequencies[:, 0] + random[:, 2] * (frequencies[:, 1] - frequencies[:, 0])
        right = torch.where(self._group_coupled[env_ids], left, independent_right)
        return torch.stack((length, left, right), dim=-1)

    def _update_commands_at_switch(self, env_ids):
        """只在换脚时渐变命令，摆动途中不跳目标；频率变化不重置phase。"""
        self._command_half_cycles[env_ids] += 1
        warming = self._warmup_remaining[env_ids] > 0
        current = torch.stack((self.step_lengths[env_ids], self.left_gait_frequencies[env_ids],
                               self.right_gait_frequencies[env_ids]), dim=-1)
        if self.command_warmup_half_cycles:
            remaining = self._warmup_remaining[env_ids].clamp_min(1).unsqueeze(-1)
            warm = current + (self._desired_commands[env_ids] - current) / remaining
            current = torch.where(warming[:, None], warm, current)
            self._warmup_remaining[env_ids] -= warming.long()
        if self.command_resample_half_cycles:
            change = ~warming & (self._command_half_cycles[env_ids] % self.command_resample_half_cycles == 0)
            selected = env_ids[change]
            desired = self._draw_group_commands(selected)
            limit = torch.tensor([self.command_max_length_change, self.command_max_frequency_change,
                                  self.command_max_frequency_change], device=self.device)
            current[change] += (desired - current[change]).clamp(-limit, limit)
        self.step_lengths[env_ids] = current[:, 0]
        self.left_gait_frequencies[env_ids] = current[:, 1]
        self.right_gait_frequencies[env_ids] = current[:, 2]
        self.gait_frequencies[env_ids] = effective_frequency(current[:, 1], current[:, 2])

    # 推进相位
    @torch.no_grad
    def step(
        self,
        dt,
        left_pos_w,
        left_quat_w,
        right_pos_w,
        right_quat_w,
    ):
        """Advance phase and resample on support-leg switches."""

        # 固定模式检查固定值；随机模式检查频率上限。
        # 构造时已保证频率为正，因此上限合法就能覆盖所有环境。
        max_frequency = self.max_gait_frequency

        # 单个控制步推进必须小于半周期，保证不会漏掉换脚事件。
        if not (0.0 < dt * max_frequency < 0.5):
            raise ValueError(
                "每个控制步的相位增量必须满足 0 < dt * 最大步频 < 0.5"
            )

        old_left_is_swing = self.left_is_swing.clone()

        # phase 和 gait_frequencies 都是 [num_envs] 张量。
        # 每个环境使用自己的步频推进相位。
        self.phase = advance_phase(self.phase, dt, self.left_gait_frequencies,
                                   self.right_gait_frequencies)

        # 判断下一个step的摆动脚
        new_left_is_swing = self.phase < 0.5

        # 是否更换摆动脚标志位
        switched = new_left_is_swing != old_left_is_swing
        self.completed_half_cycles += switched.long()
        
        self.left_is_swing.copy_(new_left_is_swing)

        # 确定需要切换的环境
        env_ids = torch.where(switched)[0]

        if self.command_resample_half_cycles or self.command_warmup_half_cycles:
            self._update_commands_at_switch(env_ids)

        self._assign_targets(
            env_ids,
            left_pos_w,
            left_quat_w,
            right_pos_w,
            right_quat_w
        )

    # 构建16维观测向量
    @torch.no_grad
    def get_observation(
        self,
        left_pos_w,
        left_quat_w,
        right_pos_w,
        right_quat_w,
    ):
        """Return [N,16] goals in the current stance frame."""

        left_swing = self.left_is_swing.unsqueeze(-1)

        # 根据left_swing条件选择支撑脚
        stance_pos = torch.where(
            left_swing,
            right_pos_w,
            left_pos_w
        )

        stance_quat = torch.where(
                    left_swing,
                    right_quat_w,
                    left_quat_w
        )

        l_pos_s, l_quat_s = subtract_frame_transforms(
            stance_pos,
            stance_quat,
            self.left_target_pos_w,
            self.left_target_quat_w,
        )

        r_pos_s, r_quat_s = subtract_frame_transforms(
            stance_pos,
            stance_quat,
            self.right_target_pos_w,
            self.right_target_quat_w,
        )

        # xyzw hemisphere correction: w >= 0
        # 同一个旋转可以用 q 和 -q 两个四元数表示  这里确保符号为正
        l_quat_s = torch.where(
            l_quat_s[:, 3:4] < 0, -l_quat_s, l_quat_s
        )
        r_quat_s = torch.where(
            r_quat_s[:, 3:4] < 0, -r_quat_s, r_quat_s
        )

        angle = 2.0 * torch.pi * self.phase

        phase_obs = torch.stack(
            (torch.cos(angle), torch.sin(angle)),
            dim=-1,
        )

        goal_obs = torch.cat(
            (
                l_pos_s,
                l_quat_s,
                r_pos_s,
                r_quat_s,
                phase_obs,
            ),
            dim = -1
        )

        assert goal_obs.shape == (self.num_envs, 16)
        assert torch.isfinite(goal_obs).all()

        return goal_obs
