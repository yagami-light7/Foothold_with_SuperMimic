"""随机水平步长回归检查：缓存、部分reset、种子和旧配置兼容。"""

import sys
from pathlib import Path

# 支持直接运行测试文件，无需额外设置PYTHONPATH。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest

import torch

from configs.s54_curriculum import restore_task_config
from tasks.foot_placement.foot_placement_env_cfg import S54FootPlacementEnvCfg
from tasks.foot_placement.goal_sampler import FootPlacementGoalSampler
from tasks.foot_placement.heading import quat_from_yaw


class RandomStepTests(unittest.TestCase):
    def poses(self, count):
        left = torch.tensor([[0., .095, 0.]]).repeat(count, 1)
        right = torch.tensor([[0., -.095, 0.]]).repeat(count, 1)
        quat = quat_from_yaw(torch.zeros(count))
        return left, quat, right, quat

    def test_episode_sampling_and_foot_switch(self):
        torch.manual_seed(42)
        poses = self.poses(512)
        sampler = FootPlacementGoalSampler(512, "cpu", gait_frequency=.75,
                                           step_length_range=(.06, .12), fixed_heading_targets=True)
        sampler.reset(*poses)
        lengths = sampler.step_lengths.clone()
        self.assertTrue(((lengths >= .06) & (lengths <= .12)).all())
        self.assertGreater(lengths.std().item(), .015)
        torch.testing.assert_close(sampler.left_target_pos_w[:, 0], lengths)
        for _ in range(35):
            sampler.step(.02, *poses)
        self.assertFalse(sampler.left_is_swing.any())
        # 换成右脚摆动后仍使用同一个episode步长，不在摆动途中改目标分布。
        torch.testing.assert_close(sampler.step_lengths, lengths, rtol=0, atol=0)
        torch.testing.assert_close(sampler.right_target_pos_w[:, 0], lengths)

    def test_seed_reproducibility_and_partial_reset(self):
        poses = self.poses(16)
        sequences = []
        for _ in range(2):
            torch.manual_seed(17)
            sampler = FootPlacementGoalSampler(16, "cpu", step_length_range=(.06, .12))
            sampler.reset(*poses)
            first = sampler.step_lengths.clone()
            sampler.reset(*poses, env_ids=[1, 3])
            second = sampler.step_lengths.clone()
            mask = torch.ones(16, dtype=torch.bool)
            mask[[1, 3]] = False
            torch.testing.assert_close(first[mask], second[mask], rtol=0, atol=0)
            self.assertFalse(torch.equal(first[[1, 3]], second[[1, 3]]))
            sequences.append((first, second))
        for first, second in zip(*sequences):
            torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_fixed_goals_preserve_rng_and_old_configs(self):
        poses = self.poses(8)
        state = torch.random.get_rng_state()
        sampler = FootPlacementGoalSampler(8, "cpu", step_length=.03)
        sampler.reset(*poses)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        torch.testing.assert_close(sampler.left_target_pos_w[:, 0], torch.full((8,), .03))
        cfg = S54FootPlacementEnvCfg()
        restore_task_config(cfg, {"step_length": .09, "step_length_range": [.06, .12]})
        self.assertEqual(cfg.step_length_range, (.06, .12))
        restore_task_config(cfg, {"step_length": .03})
        self.assertIsNone(cfg.step_length_range)

    def test_bad_ranges_and_fixed_evaluation_point(self):
        for bounds in ((-.1, .12), (.12, .06), (float("nan"), .12), (.06, float("inf"))):
            with self.assertRaises(ValueError):
                FootPlacementGoalSampler(2, "cpu", step_length_range=bounds)
        sampler = FootPlacementGoalSampler(2, "cpu", step_length_range=(.12, .12))
        sampler.reset(*self.poses(2))
        torch.testing.assert_close(sampler.step_lengths, torch.full((2,), .12))


if __name__ == "__main__":
    unittest.main()
