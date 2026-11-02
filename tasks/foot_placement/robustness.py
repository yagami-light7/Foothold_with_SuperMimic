"""小幅抗扰动；全部默认关闭，不改变旧环境的随机数序列和控制接口。"""

import torch
from typing import Any, cast


class S54Robustness:
    def __init__(self, env):
        self.env = env
        self.cfg = env.cfg
        self.steps = 0
        self.push_counts = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.next_push = torch.zeros_like(self.push_counts)
        self.mass_scales = torch.ones(env.num_envs, device=env.device)
        self.material_event = None
        if self.cfg.body_mass_scale_range is not None:
            # 每次构建模型只随机化一次，并同比缩放惯量；不会在reset中累乘质量。
            masses = env.robot.data.body_mass.torch.clone()
            inertias = env.robot.data.body_inertia.torch.clone()
            low, high = self.cfg.body_mass_scale_range
            self.mass_scales.uniform_(low, high)
            env.robot.set_masses_index(masses=masses * self.mass_scales[:, None])
            env.robot.set_inertias_index(inertias=inertias * self.mass_scales[:, None, None])
        if self.cfg.foot_friction_range is not None:
            # 使用Isaac Lab针对Newton的材质事件，通知MJWarp刷新模型参数。
            from isaaclab.envs.mdp.events import randomize_rigid_body_material
            from isaaclab.managers import EventTermCfg, SceneEntityCfg
            entity = SceneEntityCfg("robot", body_ids=[env.left_foot_id, env.right_foot_id])
            params: dict[str, Any] = dict(asset_cfg=entity, static_friction_range=tuple(self.cfg.foot_friction_range),
                          dynamic_friction_range=tuple(self.cfg.foot_friction_range),
                          restitution_range=(0., 0.), num_buckets=64, make_consistent=True)
            # 官方接口同时接受事件类和函数，SDK的func静态注解只覆盖了函数。
            event_cfg = EventTermCfg(func=cast(Any, randomize_rigid_body_material), mode="startup", params=params)
            self.material_event = randomize_rigid_body_material(event_cfg, env)
            self.material_event(env, None, **params)

    def reset(self, ids):
        if self.cfg.push_interval_s is not None:
            low, high = self.cfg.push_interval_s
            intervals = torch.empty(ids.numel(), device=self.env.device).uniform_(low, high)
            self.next_push[ids] = self.steps + (intervals / self.env.step_dt).ceil().long()

    def before_step(self):
        self.steps += 1
        if self.cfg.push_velocity_max <= 0:
            return
        ids = torch.where(self.next_push <= self.steps)[0]
        if ids.numel():
            velocity = self.env.robot.data.root_vel_w.torch[ids].clone()
            # 世界系xy速度增量表示有界外部冲量；不修改高度、朝向或关节状态。
            velocity[:, :2] += torch.empty(ids.numel(), 2, device=self.env.device).uniform_(
                -self.cfg.push_velocity_max, self.cfg.push_velocity_max)
            self.env.robot.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=ids)
            self.push_counts[ids] += 1
            self.reset(ids)

    def noisy_observation(self, gravity, angular_velocity, joint_position, joint_velocity):
        scale = self.cfg.observation_noise_scale
        if not scale:
            return gravity, angular_velocity, joint_position, joint_velocity
        # 标准差单位：重力投影无量纲、rad/s、rad、rad/s。命令、phase和动作历史不加噪声。
        return tuple(value + torch.randn_like(value) * std * scale for value, std in (
            (gravity, .01), (angular_velocity, .02), (joint_position, .002), (joint_velocity, .02)))
