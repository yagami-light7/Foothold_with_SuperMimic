"""左右节奏、观测迁移、归一化回报的CPU回归；不启动仿真或PPO训练。"""

import sys
from pathlib import Path

# 支持直接运行测试文件，无需额外设置PYTHONPATH。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import copy
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import Mock

import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.utils.logger import Logger

from agents.checkpoint_compat import migrate_checkpoint_dict
from configs.s54_curriculum import restore_task_config
from scripts.gait_cli import add_gait_arguments, apply_gait_overrides, frequency_corners
from tasks.foot_placement.foot_placement_env_cfg import S54FootPlacementEnvCfg
from tasks.foot_placement.gait import advance_phase, effective_frequency
from tasks.foot_placement.goal_sampler import FootPlacementGoalSampler
from tasks.foot_placement.observations import build_actor_observation
from tasks.foot_placement.rewards import episode_return_metrics, normalized_episode_return, reward_reference_upper
from tasks.foot_placement.rewards import compute_foot_placement_reward
from agents.rsl_rl_ppo_cfg import S54FootPlacementPPORunnerCfg
from configs.s54_curriculum import configure_stage


class GaitCommandTests(unittest.TestCase):
    def poses(self, count):
        left = torch.tensor([[0., .095, 0.]]).repeat(count, 1)
        right = torch.tensor([[0., -.095, 0.]]).repeat(count, 1)
        quat = torch.tensor([[0., 0., 0., 1.]]).repeat(count, 1)
        return left, quat, right, quat

    def test_equal_frequency_preserves_legacy_phase_and_rng(self):
        for bounds in (None, (.75, .75)):
            rng = torch.random.get_rng_state().clone()
            sampler = FootPlacementGoalSampler(4, 'cpu', gait_frequency=.75, gait_frequency_range=bounds)
            poses = self.poses(4)
            sampler.reset(*poses)
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            expected = torch.zeros(4)
            for _ in range(1000):
                expected = torch.remainder(expected + .02 * .75, 1.)
                sampler.step(.02, *poses)
                torch.testing.assert_close(sampler.phase, expected, rtol=0, atol=0)

    def test_asymmetric_boundaries_preserve_remaining_time(self):
        phase = torch.tensor([.49, .99], dtype=torch.float64)
        result = advance_phase(phase, .02, torch.tensor([.5, .5]), torch.tensor([1., 1.]))
        # 左边正好到0.5；右边先花0.01秒跨1.0，剩0.01秒按左频率0.5推进。
        torch.testing.assert_close(result, torch.tensor([.5, .005], dtype=torch.float64), rtol=0, atol=1e-12)
        result = advance_phase(torch.tensor([.49], dtype=torch.float64), .04,
                               torch.tensor([.5]), torch.tensor([1.]))
        torch.testing.assert_close(result, torch.tensor([.52], dtype=torch.float64), rtol=0, atol=1e-12)

    def test_asymmetric_cycle_matches_wall_clock(self):
        sampler = FootPlacementGoalSampler(1, 'cpu', left_gait_frequency=.5, right_gait_frequency=1.)
        poses = self.poses(1)
        sampler.reset(*poses)
        switches = []
        for i in range(160):
            old = sampler.left_is_swing.clone()
            sampler.step(.02, *poses)
            if not torch.equal(old, sampler.left_is_swing):
                switches.append((i + 1) * .02)
        # 浮点边界允许至多一个控制步的离散延迟，长期周期不应错误地使用平均频率。
        torch.testing.assert_close(torch.tensor(switches[:4]), torch.tensor([1., 1.5, 2.5, 3.]), rtol=0, atol=.02001)
        self.assertAlmostEqual(sampler.gait_frequencies.item(), 2/3, places=6)

    def test_independent_sampling_partial_reset_and_seed(self):
        sequences = []
        for _ in range(2):
            torch.manual_seed(42)
            sampler = FootPlacementGoalSampler(512, 'cpu', left_gait_frequency_range=(.65, .75),
                                                right_gait_frequency_range=(.8, .9))
            poses = self.poses(512)
            sampler.reset(*poses)
            left, right = sampler.left_gait_frequencies.clone(), sampler.right_gait_frequencies.clone()
            self.assertTrue(((left >= .65) & (left <= .75)).all())
            self.assertTrue(((right >= .8) & (right <= .9)).all())
            for _ in range(40):
                sampler.step(.02, *poses)
            torch.testing.assert_close(sampler.left_gait_frequencies, left, rtol=0, atol=0)
            sampler.reset(*poses, env_ids=[1, 3])
            mask = torch.ones(512, dtype=torch.bool)
            mask[[1, 3]] = False
            torch.testing.assert_close(sampler.left_gait_frequencies[mask], left[mask], rtol=0, atol=0)
            torch.testing.assert_close(sampler.right_gait_frequencies[mask], right[mask], rtol=0, atol=0)
            sequences.append((left, right, sampler.left_gait_frequencies.clone()))
        for a, b in zip(*sequences):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_shared_sampling_remains_equal(self):
        sampler = FootPlacementGoalSampler(64, 'cpu', gait_frequency_range=(.65, .85))
        sampler.reset(*self.poses(64))
        torch.testing.assert_close(sampler.left_gait_frequencies, sampler.right_gait_frequencies, rtol=0, atol=0)

    def test_bad_inputs(self):
        invalid_factories = (
            lambda: FootPlacementGoalSampler(2, 'cpu', left_gait_frequency=0.),
            lambda: FootPlacementGoalSampler(2, 'cpu', right_gait_frequency_range=(.8, .6)),
            lambda: FootPlacementGoalSampler(2, 'cpu', left_gait_frequency=.7, left_gait_frequency_range=(.6, .8)),
        )
        for factory in invalid_factories:
            with self.assertRaises(ValueError):
                factory()
        sampler = FootPlacementGoalSampler(2, 'cpu', left_gait_frequency_range=(.65, 30.))
        with self.assertRaises(ValueError):
            sampler.step(.02, *self.poses(2))

    def test_observation_three_layouts(self):
        tensors = dict(projected_gravity=torch.randn(4, 3), base_ang_vel=torch.randn(4, 3),
                       joint_pos=torch.randn(4, 27), joint_vel=torch.randn(4, 27),
                       last_action=torch.randn(4, 27), goal_obs=torch.randn(4, 16))
        old = build_actor_observation(**tensors)
        for size, kwargs in ((1, {'gait_frequency': torch.full((4, 1), .75)}),
                             (2, {'gait_frequencies': torch.tensor([[.65, .85]]).repeat(4, 1)})):
            obs = build_actor_observation(**tensors, **kwargs)
            self.assertEqual(obs.shape, (4, 103 + size))
            torch.testing.assert_close(obs[:, :87], old[:, :87], rtol=0, atol=0)
            torch.testing.assert_close(obs[:, -16:], old[:, -16:], rtol=0, atol=0)

    def test_restore_and_cli_sync_dimensions(self):
        cfg = S54FootPlacementEnvCfg()
        restore_task_config(cfg, {'observe_gait_frequency': True})
        self.assertEqual(cfg.observation_space, 104)
        restore_task_config(cfg, {'observe_gait_frequency': True, 'gait_frequency_obs_dim': 2})
        self.assertEqual(cfg.observation_space, 105)
        restore_task_config(cfg, {})
        self.assertEqual(cfg.observation_space, 103)
        parser = argparse.ArgumentParser()
        add_gait_arguments(parser)
        apply_gait_overrides(cfg, parser.parse_args(['--left-gait-frequency', '.65', '--right-gait-frequency', '.85']))
        self.assertEqual(cfg.observation_space, 105)
        self.assertEqual(cfg.left_gait_frequency, .65)
        self.assertEqual(frequency_corners(vars(cfg)), [(.65, .85)])


class CheckpointMigrationTests(unittest.TestCase):
    @staticmethod
    def models(dim):
        obs = TensorDict({'policy': torch.zeros(8, dim)}, batch_size=[8])
        groups = {'actor': ['policy'], 'critic': ['policy']}
        actor = MLPModel(obs, groups, 'actor', 27, hidden_dims=[256, 128], obs_normalization=True,
                         distribution_cfg={'class_name': 'rsl_rl.modules.distribution:GaussianDistribution', 'init_std': .15})
        critic = MLPModel(obs, groups, 'critic', 1, hidden_dims=[256, 128], obs_normalization=True)
        return actor, critic

    def fixture(self, dim):
        torch.manual_seed(42)
        actor, critic = self.models(dim)
        optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=1e-4)
        obs = TensorDict({'policy': torch.randn(8, dim)}, batch_size=[8])
        # 只在合成模型上生成Adam测试状态，不是机器人PPO更新。
        loss = actor(obs, stochastic_output=False).square().mean() + critic(obs).square().mean()
        loss = loss + actor.get_parameter('distribution.std_param').square().mean()
        loss.backward()
        optimizer.step()
        saved = {'actor_state_dict': actor.state_dict(), 'critic_state_dict': critic.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(), 'iter': 17,
                 'infos': {'env_cfg': {'gait_frequency': .75}}}
        return actor, critic, saved

    def test_103_to_105_models_and_optimizer(self):
        actor, critic, saved = self.fixture(103)
        original = copy.deepcopy(saved)
        converted, report = migrate_checkpoint_dict(saved, 105)
        self.assertEqual(report['optimizer_tensors_expanded'], 4)
        self.assertEqual(converted['iter'], 17)
        self.assertEqual(saved['actor_state_dict']['mlp.0.weight'].shape[1], 103)
        new_actor, new_critic = self.models(105)
        new_actor.load_state_dict(converted['actor_state_dict'])
        new_critic.load_state_dict(converted['critic_state_dict'])
        obs = torch.randn(8, 103)
        commands = torch.tensor([[.6, .9]]).repeat(8, 1)
        expanded = torch.cat((obs[:, :87], commands, obs[:, 87:]), dim=-1)
        old_td, new_td = TensorDict({'policy': obs}, [8]), TensorDict({'policy': expanded}, [8])
        torch.testing.assert_close(actor(old_td, stochastic_output=False), new_actor(new_td, stochastic_output=False), atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(critic(old_td), new_critic(new_td), atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(converted['actor_state_dict']['mlp.0.weight'][:, :87],
                                   original['actor_state_dict']['mlp.0.weight'][:, :87], rtol=0, atol=0)
        optimizer = torch.optim.Adam([*new_actor.parameters(), *new_critic.parameters()])
        optimizer.load_state_dict(converted['optimizer_state_dict'])
        # 验证迁移后的Adam状态能实际执行下一步合成优化，不出现矩阵形状错误。
        optimizer.zero_grad()
        (new_actor(new_td, stochastic_output=False).square().mean() + new_critic(new_td).square().mean()).backward()
        optimizer.step()

    def test_104_to_105_preserves_equal_command_function(self):
        actor, _, saved = self.fixture(104)
        converted, _ = migrate_checkpoint_dict(saved, 105)
        new_actor, _ = self.models(105)
        new_actor.load_state_dict(converted['actor_state_dict'])
        obs = torch.randn(8, 104)
        expanded = torch.cat((obs[:, :88], obs[:, 87:88], obs[:, 88:]), dim=-1)
        torch.testing.assert_close(actor(TensorDict({'policy': obs}, [8]), stochastic_output=False),
                                   new_actor(TensorDict({'policy': expanded}, [8]), stochastic_output=False),
                                   atol=2e-6, rtol=2e-5)

    def test_rejects_input_shrinking(self):
        _, _, saved = self.fixture(105)
        with self.assertRaises(ValueError):
            migrate_checkpoint_dict(saved, 103)


class NormalizedReturnTests(unittest.TestCase):
    def test_rsl_logger_weights_episodes_instead_of_ending_frames(self):
        # 同一rollout：一帧结束1个失败episode，另一帧结束99个成功episode。
        # 正确均值为0.892，原先逐帧求mean后再mean会错误得到0.5。
        logger = Logger(None, {'algorithm': {}, 'num_steps_per_env': 64}, {},
                        100, False, 1, 0, 'cpu')
        writer = Mock()
        logger.writer = writer
        for count, quality in ((1, .1), (99, .9)):
            metrics = episode_return_metrics(torch.full((count,), quality), torch.ones(count),
                                             torch.full((count,), 500), 500)
            logger.ep_extras.append(metrics)
        # 没有完成episode的帧不能重复采样上一次的分数。
        logger.ep_extras.append({})
        with redirect_stdout(StringIO()):
            logger.log(0, 0, 1, 1., .1, {}, 1e-4, torch.tensor([.15]), None)
        values = {call.args[0]: float(call.args[1]) for call in writer.add_scalar.call_args_list}
        self.assertAlmostEqual(values['Episode/normalized_return'], .892, places=6)
        self.assertAlmostEqual(values['Episode/normalized_quality'], .892, places=6)
        self.assertEqual(values['Episode/survival_fraction'], 1.)

    def test_bound_and_short_episode_penalty(self):
        score = normalized_episode_return(torch.tensor([80., 40., -1., 120.]),
                                           torch.tensor([100., 50., 100., 100.]),
                                           torch.tensor([500, 250, 500, 500]), 500)
        torch.testing.assert_close(score, torch.tensor([.8, .4, 0., 1.]))

    def test_reward_is_below_phase_reference(self):
        cfg, agent = S54FootPlacementEnvCfg(), S54FootPlacementPPORunnerCfg()
        configure_stage(cfg, agent, 'straight')
        n = 1000
        phase = torch.linspace(0, .9999, n)
        left_is_swing = phase < .5
        position, target = torch.randn(n, 3), torch.randn(n, 3)
        force = torch.randn(n, 3) * 300
        quat = torch.zeros(n, 4)
        quat[:, 3] = 1
        reward, _ = compute_foot_placement_reward(position, position, target, target, force, force,
            phase, left_is_swing, torch.randn(n, 3), torch.randn(n, 27), torch.randn(n, 27),
            .02, cfg.reward_cfg, swing_has_lifted=torch.ones(n, dtype=torch.bool),
            root_quat_w=quat, reference_heading_w=torch.zeros(n), root_yaw_rate=torch.randn(n))
        upper = reward_reference_upper(phase, left_is_swing, .02, cfg.reward_cfg)
        self.assertTrue((reward <= upper + 1e-6).all())
        self.assertTrue((upper > 0).all())


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
