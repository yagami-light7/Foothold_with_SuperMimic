"""记录原始动作裁剪率，保持RSL-RL原有step和timeout语义。"""

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class S54RslRlVecEnvWrapper(RslRlVecEnvWrapper):
    def step(self, actions):
        from tasks.foot_placement.foot_placement_env import S54FootPlacementEnv
        task = self.unwrapped
        if isinstance(task, S54FootPlacementEnv):
            limit = self.clip_actions if self.clip_actions is not None else 1.0
            # 必须在父wrapper裁剪之前记录；动作限幅不等于实际力矩饱和。
            task.raw_action_clip_mask = actions.abs() > limit
        return super().step(actions)
