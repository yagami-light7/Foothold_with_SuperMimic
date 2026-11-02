"""CPU regression tests for opt-in curriculum signals and retained footholds."""

import sys
from pathlib import Path

# 支持直接运行测试文件，无需额外设置PYTHONPATH。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from agents.rsl_rl_ppo_cfg import S54FootPlacementPPORunnerCfg
from configs.s54_curriculum import STAGES, configure_stage, restore_task_config
from isaaclab.utils import to_dict
from tasks.foot_placement.foot_placement_env_cfg import S54FootPlacementEnvCfg
from tasks.foot_placement.goal_sampler import FootPlacementGoalSampler
from tasks.foot_placement.rewards import FootRewardCfg, compute_foot_placement_reward


def evaluate(cfg, lift=0.0, swing_fz=330.0, action_delta=0.0, stance_fz=660.0, has_lifted=False, phase=0.25):
    left = torch.tensor([[0.0, 0.095, 0.0]])
    right = torch.tensor([[0.0, -0.095, 0.0]])
    target = left + torch.tensor([[0.12, 0.0, 0.0]])
    current = left + torch.tensor([[0.0, 0.0, lift]])
    return compute_foot_placement_reward(current, right, target, right,
        torch.tensor([[0.0, 0.0, swing_fz]]), torch.tensor([[0.0, 0.0, stance_fz]]),
        torch.tensor([phase]), torch.tensor([True]), torch.tensor([[0.0, 0.0, -1.0]]),
        torch.full((1,27), action_delta), torch.zeros(1,27), 0.02, cfg,
        swing_has_lifted=torch.tensor([has_lifted]))


def main():
    legacy = FootRewardCfg()
    reward, terms = evaluate(legacy)
    expected = 0.02 * (torch.exp(torch.tensor(-1.44)) + 2 + 2*torch.exp(torch.tensor(-1.44)) + 0.5 + 1)
    torch.testing.assert_close(reward[0], expected)
    cfg = S54FootPlacementEnvCfg()
    agent = S54FootPlacementPPORunnerCfg()
    configure_stage(cfg, agent, "lift")
    standing, standing_terms = evaluate(cfg.reward_cfg)
    lifted, lifted_terms = evaluate(cfg.reward_cfg, lift=cfg.reward_cfg.swing_height, swing_fz=0)
    assert standing_terms['clearance'].item() == 0
    assert lifted.item() > standing.item()
    assert lifted_terms['clearance'].item() > standing_terms['clearance'].item()
    _, unloading = evaluate(cfg.reward_cfg, swing_fz=150)
    assert unloading['unloading'].item() > standing_terms['unloading'].item()
    _, unsupported = evaluate(cfg.reward_cfg, lift=0.02, swing_fz=0, stance_fz=0)
    assert unsupported['clearance'].item() == unsupported['swing_air'].item() == 0
    _, never_lifted = evaluate(cfg.reward_cfg, phase=0.45)
    assert never_lifted['touchdown'].item() == 0
    _, smooth = evaluate(cfg.reward_cfg, action_delta=0.1)
    torch.testing.assert_close(smooth['action_rate'], torch.tensor([-0.0005]))
    for stage in STAGES:
        e, a = S54FootPlacementEnvCfg(), S54FootPlacementPPORunnerCfg()
        configure_stage(e, a, stage)
        saved = to_dict(e)
        restored = S54FootPlacementEnvCfg()
        restore_task_config(restored, saved)
        assert restored.reward_cfg == e.reward_cfg
        assert restored.action_scale_by_joint == e.action_scale_by_joint
        assert restored.step_length == e.step_length
        if stage != 'legacy':
            assert e.action_scale_by_joint is not None and len(e.action_scale_by_joint) == 27
            assert e.action_scale_by_joint[3] == 0.6
    sampler = FootPlacementGoalSampler(2, 'cpu', retain_stance_targets=True)
    left = torch.tensor([[0.0,0.095,0.0]]).repeat(2,1)
    right = torch.tensor([[0.0,-0.095,0.0]]).repeat(2,1)
    quat = torch.tensor([[0.0,0.0,0.0,1.0]]).repeat(2,1)
    sampler.reset(left,quat,right,quat)
    target = sampler.left_target_pos_w.clone()
    for _ in range(26):
        sampler.step(0.02,left,quat,right,quat)
    torch.testing.assert_close(sampler.left_target_pos_w,target)
    sampler.reset(left,quat,right,quat,env_ids=[1])
    assert sampler.phase[1] == 0 and sampler.phase[0] > 0.5
    pitch = torch.tensor([[0.0, torch.sin(torch.tensor(0.2)), 0.0, torch.cos(torch.tensor(0.2))]]).repeat(2,1)
    flat = FootPlacementGoalSampler(2, 'cpu', planar_targets=True, flat_ground_height=0.0)
    flat.reset(left,pitch,right,pitch)
    assert torch.all(flat.left_target_pos_w[:, 2] == 0)
    tilted = FootPlacementGoalSampler(2, 'cpu')
    tilted.reset(left,pitch,right,pitch)
    assert torch.all(tilted.left_target_pos_w[:, 2] < -0.04)
    print(f'Tilted stance target Z: legacy={tilted.left_target_pos_w[0,2].item():.5f}, planar={flat.left_target_pos_w[0,2].item():.5f}')
    print(f'Legacy reward: {reward.item():.6f}; curriculum standing/lifted: {standing.item():.6f}/{lifted.item():.6f}')
    print('Curriculum, dense reward, retained stance and snapshot restoration tests PASSED')


if __name__ == '__main__':
    main()
