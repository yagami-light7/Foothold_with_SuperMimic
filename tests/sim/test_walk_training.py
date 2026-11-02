"""Newton混合采样/扰动接口检查；零动作运行60秒，不进行PPO更新。"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--num_envs', type=int, default=16)
parser.add_argument('--steps', type=int, default=3010)
parser.add_argument('--output', type=Path, required=True)
add_launcher_args(parser)
parser.set_defaults(device='cuda:0')
args = parser.parse_args()

import torch
import warp as wp
from agents.rsl_rl_ppo_cfg import S54FootPlacementPPORunnerCfg
from agents.vec_env import S54RslRlVecEnvWrapper
from configs.s54_curriculum import configure_stage, sync_observation_space
from configs.walk_training import apply_task_profile, load_adaptive_plan
from tasks.foot_placement.foot_placement_env import S54FootPlacementEnv
from tasks.foot_placement.foot_placement_env_cfg import S54FootPlacementEnvCfg


def main():
    cfg, agent = S54FootPlacementEnvCfg(), S54FootPlacementPPORunnerCfg()
    configure_stage(cfg, agent, 'straight')
    plan = load_adaptive_plan(ROOT/'configs/walk_hard_region.json')
    apply_task_profile(cfg, plan['phases'][-1]['profile'])
    cfg.episode_length_s = 60.
    cfg.observe_gait_frequency = True
    cfg.gait_frequency_obs_dim = 2
    sync_observation_space(cfg)
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    with launch_simulation(cfg.sim, args):
        env = S54FootPlacementEnv(cfg)
        try:
            wrapper = S54RslRlVecEnvWrapper(env, clip_actions=1.)
            obs = wrapper.get_observations()
            assert obs['policy'].shape == (args.num_envs,105)
            scales = env.robustness.mass_scales
            masses = env.robot.data.body_mass.torch.clone()
            inertia = env.robot.data.body_inertia.torch.clone()
            # 检查实际模型质量/惯量，而不只是配置字符串。各环境只差同比例缩放。
            nominal = masses / scales[:,None]
            torch.testing.assert_close(nominal, nominal[0:1].expand_as(nominal))
            nominal_inertia = inertia / scales[:,None,None]
            torch.testing.assert_close(nominal_inertia, nominal_inertia[0:1].expand_as(nominal_inertia))
            assert .97 <= scales.min().item() < scales.max().item() <= 1.03
            assert masses.sum(-1).std() > 0
            # 仅测试读取Newton实际shape系数；生产写入使用官方材质事件。
            event = env.robustness.material_event
            assert event is not None
            material = event._impl
            friction = wp.to_torch(material._friction_binding)[:, material._shape_indices.to(env.device)].clone()
            assert .45 <= friction.min().item() < friction.max().item() <= .55
            assert friction.std() > 0
            noiseless = tuple(torch.zeros(args.num_envs,d,device=env.device) for d in (3,3,27,27))
            noisy = env.robustness.noisy_observation(*noiseless)
            assert all((a != b).any() for a,b in zip(noiseless,noisy))
            ids = [0]
            untouched_phase = env.goal_sampler.phase[1:].clone()
            env._reset_idx(ids)
            torch.testing.assert_close(env.goal_sampler.phase[1:],untouched_phase)
            torch.testing.assert_close(env.robot.data.body_mass.torch,masses)
            # reset关节噪声后，目标以新FK支撑脚为起点。
            left, _, right, _ = env._get_foot_poses()
            if env.goal_sampler.left_is_swing[0]:
                torch.testing.assert_close(env.goal_sampler.right_target_pos_w[0,:2],right[0,:2],atol=1e-5,rtol=0)
            else:
                torch.testing.assert_close(env.goal_sampler.left_target_pos_w[0,:2],left[0,:2],atol=1e-5,rtol=0)
            offsets = env.robot.data.joint_pos.torch - env.robot.data.default_joint_pos.torch
            assert offsets.abs().max() > 0
            assert offsets.abs().max() <= cfg.reset_joint_position_noise + 1e-6
            # 下面从所有环境新episode开始统计真实60秒timeout；不通过放宽终止条件制造存活。
            wrapper.reset()
            groups = env.goal_sampler.command_group_ids.clone()
            raw = torch.zeros(args.num_envs,27,device=env.device)
            total_terminated = total_truncated = 0
            for step in range(args.steps):
                raw.zero_()
                if step == 5:
                    raw[:, -1] = 1.1  # 仅一帧小幅上肢动作，验证裁剪前遥测。
                obs, rewards, _, _ = wrapper.step(raw)
                assert torch.isfinite(obs['policy']).all() and torch.isfinite(rewards).all()
                torch.testing.assert_close(env.goal_sampler.command_group_ids,groups)
                torch.testing.assert_close(obs['policy'][:,87:89],torch.stack((
                    env.goal_sampler.left_gait_frequencies,env.goal_sampler.right_gait_frequencies),-1))
                total_terminated += int(env.reset_terminated.sum().item())
                total_truncated += int(env.reset_time_outs.sum().item())
                if step == 5:
                    assert env.extras['log']['Control/action_clip_fraction/zarm_r7_joint'] == 1.
            assert env.robustness.push_counts.sum() > 0
            torch.testing.assert_close(env.robot.data.body_mass.torch,masses)
            if args.steps >= 3000:
                assert env.max_episode_length == 3000
                assert total_truncated > 0, '需要实际运行到60秒timeout'
            result = dict(status='passed',num_envs=args.num_envs,steps=args.steps,
                          observation_dim=105,action_dim=27,max_episode_steps=env.max_episode_length,
                          terminated=total_terminated,truncated=total_truncated,
                          push_count=int(env.robustness.push_counts.sum().item()),
                          mass_scale_min_max=[scales.min().item(),scales.max().item()],
                          total_mass_min_max_kg=[masses.sum(-1).min().item(),masses.sum(-1).max().item()],
                          friction_min_max=[friction.min().item(),friction.max().item()],
                          reset_noise_and_fk_checked=True,command_observation_checked=True,
                          mass_not_accumulated_on_reset=True,raw_action_clip_checked=True)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(result,indent=2),flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    main()
