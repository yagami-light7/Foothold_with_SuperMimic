"""CPU 回归测试：验证固定足印朝向、方向奖励及旧 checkpoint 的配置兼容。"""

import sys
from pathlib import Path

# 支持直接运行测试文件，无需额外设置PYTHONPATH。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import math
import unittest
from dataclasses import replace

import torch

from agents.rsl_rl_ppo_cfg import S54FootPlacementPPORunnerCfg
from configs.s54_curriculum import configure_stage, restore_task_config
from isaaclab.utils import to_dict
from tasks.foot_placement.foot_placement_env_cfg import S54FootPlacementEnvCfg
from tasks.foot_placement.goal_sampler import FootPlacementGoalSampler
from tasks.foot_placement.heading import heading_error, quat_from_yaw, yaw_from_quat
from tasks.foot_placement.rewards import FootRewardCfg, compute_foot_placement_reward


class HeadingControlTests(unittest.TestCase):
    def test_reference_survives_stance_turn_and_partial_reset(self):
        # 两个环境分别朝向 0°、90°，足印偏移应随参考方向正确旋转。
        left = torch.tensor([[0., .095, 0.]]).repeat(2, 1)
        right = torch.tensor([[0., -.095, 0.]]).repeat(2, 1)
        reset_q = quat_from_yaw(torch.tensor([0., math.pi / 2]))
        sampler = FootPlacementGoalSampler(2, "cpu", gait_frequency=.75, step_length=.03,
                                           retain_stance_targets=True, planar_targets=True,
                                           flat_ground_height=0., fixed_heading_targets=True)
        sampler.reset(left, reset_q, right, reset_q, reference_quat_w=reset_q)
        torch.testing.assert_close(sampler.left_target_pos_w[:, :2] - right[:, :2],
                                   torch.tensor([[.03, .19], [-.19, .03]]), atol=1e-6, rtol=1e-5)
        previous_target = sampler.left_target_pos_w.clone()
        turned_q = quat_from_yaw(torch.tensor([.6, math.pi / 2 + .6]))
        for _ in range(35):
            sampler.step(.02, left, turned_q, right, turned_q)
        torch.testing.assert_close(sampler.reference_heading_w, torch.tensor([0., math.pi / 2]))
        torch.testing.assert_close(sampler.left_target_pos_w, previous_target)
        torch.testing.assert_close(sampler.right_target_pos_w[0] - left[0], torch.tensor([.03, -.19, 0.]))
        # 沿用已有 16 维 goal，用目标四元数表达“参考朝向 - 当前支撑脚朝向”。
        obs = sampler.get_observation(left, turned_q, right, turned_q)
        torch.testing.assert_close(yaw_from_quat(obs[:, 10:14]), torch.full((2,), -.6), atol=1e-6, rtol=1e-5)
        changed_reset = quat_from_yaw(torch.tensor([.9, -.3]))
        # 只重置环境 1：环境 0 的参考朝向和相位应保持连续。
        sampler.reset(left, changed_reset, right, changed_reset, env_ids=[1], reference_quat_w=changed_reset)
        torch.testing.assert_close(sampler.reference_heading_w, torch.tensor([0., -.3]))
        self.assertGreater(sampler.phase[0].item(), .5)
        self.assertEqual(sampler.phase[1].item(), 0.)

    def test_angle_wrap_and_direction_reward(self):
        # 跨越 ±π 的两角只差 0.02 rad，不能直接相减得到接近 2π 的误差。
        error = heading_error(quat_from_yaw(torch.tensor([-math.pi + .01])), torch.tensor([math.pi - .01]))
        torch.testing.assert_close(error, torch.tensor([.02]), atol=1e-6, rtol=1e-5)
        pos = torch.zeros(1, 3)
        force = torch.tensor([[0., 0., 660.]])
        def evaluate(cfg, angle=0., rate=0.):
            return compute_foot_placement_reward(pos, pos, pos, pos, force, force,
                torch.tensor([.25]), torch.tensor([True]), torch.tensor([[0., 0., -1.]]),
                torch.zeros(1, 27), torch.zeros(1, 27), .02, cfg,
                root_quat_w=quat_from_yaw(torch.tensor([angle])), reference_heading_w=torch.zeros(1),
                root_yaw_rate=torch.tensor([rate]))
        base = FootRewardCfg()
        directed = replace(base, heading_w=1., yaw_rate_w=.5)
        old, terms = evaluate(base)
        aligned, aligned_terms = evaluate(directed)
        deviated, deviated_terms = evaluate(directed, angle=.35, rate=.3)
        self.assertNotIn("heading", terms)
        # 奖励只乘一次控制步长：0.02 s × (1.0 + 0.5) = 0.03。
        torch.testing.assert_close(aligned - old, torch.tensor([.03]))
        torch.testing.assert_close(deviated_terms["heading"], torch.tensor([math.exp(-1)]))
        torch.testing.assert_close(deviated_terms["yaw_rate"], torch.tensor([.5 * math.exp(-1)]))
        self.assertGreater(aligned.item(), deviated.item())
        self.assertEqual(aligned_terms["heading"].item(), 1.)

    def test_profiles_and_old_snapshot_restore(self):
        # 新阶段不改变动作尺度或网络维度，缺少新字段的旧配置按默认值恢复。
        straight, agent = S54FootPlacementEnvCfg(), S54FootPlacementPPORunnerCfg()
        configure_stage(straight, agent, "straight")
        step, step_agent = S54FootPlacementEnvCfg(), S54FootPlacementPPORunnerCfg()
        configure_stage(step, step_agent, "step")
        self.assertTrue(straight.fixed_heading_targets)
        self.assertEqual(straight.action_scale_by_joint, step.action_scale_by_joint)
        self.assertEqual(straight.action_space, 27)
        self.assertEqual(straight.observation_space, 103)
        assert isinstance(straight.reward_cfg,FootRewardCfg) and isinstance(step.reward_cfg,FootRewardCfg)
        self.assertEqual(straight.reward_cfg.clearance_w, step.reward_cfg.clearance_w)
        saved = to_dict(straight)
        restored = S54FootPlacementEnvCfg()
        restore_task_config(restored, saved)
        self.assertTrue(restored.fixed_heading_targets)
        self.assertEqual(restored.reward_cfg, straight.reward_cfg)
        old = to_dict(step)
        old.pop("fixed_heading_targets")
        for key in ("heading_w", "heading_sigma", "yaw_rate_w", "yaw_rate_sigma"):
            old["reward_cfg"].pop(key)
        restore_task_config(restored, old)
        self.assertFalse(restored.fixed_heading_targets)
        self.assertEqual(restored.reward_cfg.heading_w, 0.)
        configure_stage(restored, agent, "straight", "targets_only")
        self.assertTrue(restored.fixed_heading_targets)
        self.assertEqual(restored.reward_cfg.heading_w, 0.)
        configure_stage(restored, agent, "straight", "rewards_only")
        self.assertFalse(restored.fixed_heading_targets)
        self.assertEqual(restored.reward_cfg.heading_w, 1.)


if __name__ == "__main__":
    unittest.main()
