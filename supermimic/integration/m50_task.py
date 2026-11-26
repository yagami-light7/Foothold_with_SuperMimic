"""M5.0 使用冻结的 M4.11 控制和奖励，在各环境独立采样参考。"""

from pathlib import Path

import torch

from ..roban import assert_checkpoint
from ..rotations import so3_exp
from .m4_task import F2GoalSpec, F2TrackingMDP, M4TrackingSim, build_reference_tracking
from .m44_evaluation import PERTURB_LIMITS
from .m50_reference import MultiReferenceProvider, REPO, SOURCE, read_manifest


TRAIN_CLIPS = ('F1', 'F2', 'F3')
STAGES = ((1000, (.20, .60, .20)), (3000, (.30, .40, .30)), (6000, (1/3, 1/3, 1/3)))
CONFIG = REPO / 'configs/roban/m5_0.json'
OUTPUT = REPO / 'outputs/m5_0'


class MultiReferenceGoalSpec(F2GoalSpec):
    def contract(self):
        return super().contract() | {'reference': 'immutable_offline_multiclip'}


class MultiReferenceMDP(F2TrackingMDP):
    def contract(self):
        value = super().contract()
        value.pop('duration_s')
        return value | dict(mode='roban_m50_multiclip', foot_frame='generator_source_sole; current_robot_heading',
                            reset='clip_first_frame_with_scheduled_pose_perturbation')


def build_m50_task(args, kin, device, *, entries=None):
    manifest = read_manifest(args.m50_library)
    motion = MultiReferenceProvider(manifest['train'] if entries is None else entries, kin, device,
                                    root=args.m50_reference_root, version=manifest['reference_version'])
    spec = MultiReferenceGoalSpec('roban_foot', args.goal_frames, args.goal_dt)
    return build_reference_tracking(args, kin, motion, spec, MultiReferenceMDP)


def curriculum_at(iteration):
    # 第 1001、3001 轮切片段概率，初态偏差在另外的轮次调整。
    stage = int(iteration > 1000) + int(iteration > 3000)
    if iteration < 1501:
        probability, scale = 0., 0.
    elif iteration < 3501:
        probability, scale = .25, .50
    else:
        probability, scale = .50, .50 + .50 * (iteration - 3500) / 2500
    return STAGES[stage][1], probability, min(scale, 1.)


class MultiReferenceSim(M4TrackingSim):
    def __init__(self, sim_cfg, num_envs, motion, mdp, *, seed, iteration=0, **kwargs):
        if tuple(motion.names) != TRAIN_CLIPS:
            raise ValueError('M5.0 正式课程的训练清单必须依次为 F1、F2、F3')
        self.clip_generator = torch.Generator(device=motion.q.device).manual_seed(50_003 + seed)
        self.pose_generator = torch.Generator(device=motion.q.device).manual_seed(50_021 + seed)
        self.reset_counts = torch.zeros(len(motion.names), device=motion.q.device, dtype=torch.long)
        self.perturbed_counts = torch.zeros_like(self.reset_counts)
        self.sample_probabilities = motion.q.new_empty(len(motion.names))
        self.set_iteration(iteration)
        super().__init__(sim_cfg, num_envs, motion, mdp, **kwargs)

    def set_iteration(self, iteration):
        weights, self.perturb_probability, self.perturb_scale = curriculum_at(iteration)
        weights = self.sample_probabilities.new_tensor(weights)
        self.sample_probabilities.copy_(weights)

    def sample_frames(self, n):
        ids = torch.multinomial(self.sample_probabilities, n, replacement=True, generator=self.clip_generator)
        return self.motion.clip_start[ids], ids

    def reset(self, env_ids, frames=None):
        if not len(env_ids):
            return
        if frames is not None:
            torch._assert_async((frames == self.motion.clip_start[self.motion.clip_id[frames]]).all(),
                                'M5.0 只允许在合法首帧复位')
        super().reset(env_ids, frames)

    def prepare_motion_reset(self, state, frames):
        p, R, v, w, q, qd = state
        clip = self.motion.clip_id[frames]
        ones = torch.ones_like(clip)
        self.reset_counts.scatter_add_(0, clip, ones)
        if not self.perturb_probability or self.eval_mode:
            return state
        random = torch.rand((len(clip), 1), generator=self.pose_generator, device=p.device)
        jitter = random < self.perturb_probability
        self.perturbed_counts.scatter_add_(0, clip, jitter[:, 0].long())
        params = PERTURB_LIMITS

        def symmetric(tensor, bounds):
            samples = torch.rand(tensor.shape, device=p.device, generator=self.pose_generator)
            return (2 * samples - 1.) * bounds * (self.perturb_scale * jitter)

        xy = symmetric(p[:, :2], params['root_xy_m'])
        lift = .006 + torch.rand(random.shape, device=p.device, generator=self.pose_generator) * (params['root_lift_m'] - .006)
        position = p + torch.cat((xy, lift * (self.perturb_scale * jitter)), dim=1)
        angles = p.new_tensor([params['root_roll_pitch_deg']] * 2 + [params['root_yaw_deg']]).deg2rad()
        actual_q = q + symmetric(q, params['joint_position_rad'])
        torch._assert_async(((actual_q >= self.mdp.kin.q_lo) & (actual_q <= self.mdp.kin.q_hi)).all(),
                            'M5.0 初态偏差超过机械限位')
        return (position, so3_exp(symmetric(p, angles)) @ R,
                v + symmetric(v, params['root_v_m_s']), w + symmetric(w, params['root_w_rad_s']),
                actual_q, qd + symmetric(qd, params['joint_speed_rad_s']))

    def curriculum_state(self):
        return dict(clip_rng=self.clip_generator.get_state(), pose_rng=self.pose_generator.get_state(),
                    reset_counts=self.reset_counts.clone(), perturbed_counts=self.perturbed_counts.clone())

    def restore_curriculum(self, data):
        self.clip_generator.set_state(data['clip_rng'].cpu())
        self.pose_generator.set_state(data['pose_rng'].cpu())
        self.reset_counts.copy_(data['reset_counts'])
        self.perturbed_counts.copy_(data['perturbed_counts'])


def initialize_b1250(models, mdp, goals, physics, policy, imagination, *, source=SOURCE):
    ck = torch.load(Path(source), map_location='cpu', weights_only=False, mmap=True)
    assert_checkpoint(ck)
    mdp.assert_wm_statistics(ck)
    if ck.get('iteration') != 1250 or ck['args'].get('m411_group') != 'b':
        raise ValueError('M5.0 Warm Start 需要 M4.11-B1250')
    expected = dict(physical_contract=physics, policy_contract=policy, imagination_contract=imagination)
    context = {'mode', 'duration_s', 'reset', 'foot_frame'}
    comparable = lambda task: {key: value for key, value in task.items() if key not in context}
    saved_goal = dict(ck['goal_contract'], reference='immutable_offline_multiclip')
    if (any(ck.get(key) != value for key, value in expected.items()) or saved_goal != goals.spec.contract()
            or comparable(ck['task_contract']) != comparable(mdp.contract())):
        raise ValueError('B1250 的网络、奖励、动作或物理契约与 M5.0 不一致')
    for name, model in models.items():
        model.load_state_dict(ck[name], strict=True)
    return dict(strategy='m50_b1250_all_models', source_path=str(Path(source).relative_to(REPO)),
                source_iteration=ck['iteration'], source_sim_steps=ck['sim_steps'],
                source_total_wall_seconds=ck['wall_time'])


class ClipTrainingStats:
    names = ('transitions', 'reward', 'foot_rmse_m', 'root_xy_rmse_m', 'hand_left_rmse_m',
             'hand_right_rmse_m', 'episodes', 'episode_length', 'survival', 'episode_return')
    more = ('resets', 'perturbed_resets', 'replay_fraction')

    def __init__(self, motion):
        self.motion = motion
        self.sums = motion.q.new_zeros((len(motion.names), len(self.names)), dtype=torch.float64)
        self.columns = tuple(f'{clip}/{field}' for clip in motion.names for field in self.names + self.more)

    def record(self, out):
        f = out['f']
        clip = self.motion.clip_id[f]
        valid = out.get('valid', torch.ones_like(f, dtype=torch.bool))
        done = out['done']
        terms = out['terms']
        delta = out['s1'][:, :2] - self.motion.root_p[f + 1, :2]
        episode_return = torch.zeros_like(out['r'])
        if 'ep_ret_norm' in out:
            episode_return[done] = out['ep_ret_norm']
        values = torch.stack((torch.ones_like(out['r']), out['r'], terms['foot_rmse'].square(),
                              delta.square().sum(-1), terms['hand_left_error_sq'], terms['hand_right_error_sq'],
                              done, (out['ep_step'] + 1) * done, out['timeout'] & ~out['term'], episode_return), dim=-1)
        self.sums.index_add_(0, clip, (values * valid[:, None]).to(self.sums.dtype))

    def read(self, buffer, env):
        values = self.sums.clone()
        count, done = values[:, 0].clone(), values[:, 6].clone()
        values[:, 1:6] /= count[:, None].clamp_min(1)
        values[:, 2:6].sqrt_()
        values[:, 7:] /= done[:, None].clamp_min(1)
        valid = buffer.data['ep_step'][:buffer.size] >= 0
        frames = buffer.data['f'][:buffer.size]
        counts = torch.bincount(self.motion.clip_id[frames[valid]], minlength=len(self.motion.names))
        fraction = counts / counts.sum().clamp_min(1)
        values = torch.cat((values, env.reset_counts[:, None], env.perturbed_counts[:, None], fraction[:, None]), dim=1)
        self.sums.zero_()
        return dict(zip(self.columns, values.flatten().tolist()))
