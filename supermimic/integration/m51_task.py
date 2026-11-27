"""M5.1 三组多参考训练共用设置与奖励契约。"""

from dataclasses import dataclass
from pathlib import Path

import torch

from ..roban import assert_checkpoint
from ..roban_task import FootRewardConfig
from .m50_reference import REPO
from .m50_task import ClipTrainingStats, MultiReferenceMDP, MultiReferenceSim
from .m51_foot import PARAMETERS, TERMS, WEIGHTS, foot_axes, stance_signals


OUTPUT = REPO / 'outputs/m5_1'
SOURCE = REPO / 'outputs/m5_0/checkpoints/iter_5500.pt'
GROUPS = {'a': 'a_baseline', 'b': 'b_stance_yaw', 'c': 'c_yaw_angular_slip'}
CONFIGS = {key: REPO / f'configs/roban/m5_1_{key}.json' for key in GROUPS}
SCALES = ('_tolerance', '_sigma')
REWARD_KEYS = (*TERMS, *PARAMETERS)
LOG_TERMS = (*TERMS, 'foot_position', 'swing_orientation')
GRAD_PAIRS = tuple((base, term) for term in TERMS
                   for base in ('foot_position', 'swing_orientation', 'root', 'joint_huber', 'hand_huber', 'bootstrap'))
GRAD_PAIRS += (('stance_yaw_huber', 'stance_angular_slip'),)
STEP_SUMS = ('left_yaw_sq', 'right_yaw_sq', 'stance_yaw_sq', 'stance_omega_sq', 'stance_proxy_miss',
             'stance_yaw_count', 'stance_count')


@dataclass(frozen=True)
class StanceFootRewardConfig(FootRewardConfig):
    stance_yaw_huber: float = 0.
    stance_yaw_huber_tolerance: float = PARAMETERS['stance_yaw_huber_tolerance']
    stance_yaw_huber_sigma: float = PARAMETERS['stance_yaw_huber_sigma']
    stance_angular_slip: float = 0.
    stance_angular_slip_tolerance: float = PARAMETERS['stance_angular_slip_tolerance']
    stance_angular_slip_sigma: float = PARAMETERS['stance_angular_slip_sigma']

    def omit_inactive_whole_body(self, values):
        values = super().omit_inactive_whole_body(values)
        for name in TERMS:
            if not getattr(self, name):
                for key in (name, *(name+suffix for suffix in SCALES)):
                    del values[key]
        return values


class StanceFootMDP(MultiReferenceMDP):
    reward_config_type = StanceFootRewardConfig
    stance_foot_terms = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.terrain is not None or not self.cfg.swing_aware:
            raise ValueError('M5.1 足部支撑奖励需要平面和原有高度接触代理')
        self.foot_axis = self.default_q.new_tensor(foot_axes()[0])
        self.reference_foot_w = self.motion.body_w[:, self.motion.foot_indices]

    def contract(self):
        return super().contract() | dict(mode='roban_m51_multiclip', reset='clip_first_frame_full_range_half')

    def foot_signals(self, rotation, angular, reference, frames, anchor, soft_contact):
        ref_w = self.reference_foot_w[frames.clamp_max(self.motion.num_frames - 1)]
        if anchor is not None:
            ref_w = (anchor.rotation[:, None] @ ref_w[..., None]).squeeze(-1)
        return stance_signals(rotation, angular, reference, ref_w, self.foot_axis, soft_contact)

    def stance_rewards(self, rotation, body_w, reference, frames, anchor, soft_contact, diagnostics):
        c = self.cfg
        if not diagnostics and not any(getattr(c, key) for key in TERMS):
            return {}, {}
        angular = body_w[:, self.motion.foot_indices]
        s = self.foot_signals(rotation, angular, reference, frames, anchor, soft_contact)
        raw, _ = s.raw_rewards(c)
        telemetry = {}
        if diagnostics:
            yaw_sq = s.yaw.square()
            stance = s.plan & s.yaw_valid
            telemetry.update(left_yaw_sq=yaw_sq[:, 0], right_yaw_sq=yaw_sq[:, 1],
                             stance_yaw_sq=(yaw_sq*stance).sum(-1), stance_yaw_count=stance.sum(-1),
                             stance_omega_sq=(angular[..., 2].square()*s.plan).sum(-1),
                             stance_proxy_miss=((s.contact < .5) & s.plan).sum(-1),
                             stance_count=s.plan.sum(-1))
        return raw, telemetry


class StanceFootSim(MultiReferenceSim):
    def set_iteration(self, iteration):
        self.perturb_probability, self.perturb_scale = .50, 1.
        self.sample_probabilities.fill_(1/3)


def expected_config(parent, group):
    result = dict(parent, m51_group=group, m51_critic_warmup=200, iterations=2000)
    del result['m50_adaptation_iters']
    result['roban_reward_config'] = dict(parent['roban_reward_config']) | PARAMETERS | dict(zip(TERMS, WEIGHTS[group]))
    return result


def initialize_m50(models, mdp, goals, physics, policy, imagination, *, source=SOURCE):
    ck = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    assert_checkpoint(ck)
    mdp.assert_wm_statistics(ck)
    fields = dict(physical_contract=physics, policy_contract=policy, imagination_contract=imagination,
                  goal_contract=goals.spec.contract(), data_contract=mdp.motion.validation)
    task = mdp.contract()
    task['config'] = {key: value for key, value in task['config'].items() if key not in REWARD_KEYS}
    task.update(mode='roban_m50_multiclip', reset='clip_first_frame_with_scheduled_pose_perturbation')
    if ck['iteration'] != 5500 or ck.get('task_contract') != task or any(ck.get(k) != v for k, v in fields.items()):
        raise ValueError('M5.1 只接受规则、物理、参考和网络契约匹配的 M5.0 第 5500 轮权重')
    for name, model in models.items():
        model.load_state_dict(ck[name], strict=True)
    return dict(strategy='m51_m50_5500_all_models', source_path=str(Path(source).relative_to(REPO)),
                source_iteration=ck['iteration'], source_sim_steps=ck['sim_steps'],
                source_total_wall_seconds=ck['wall_time'])


class StanceTrainingStats(ClipTrainingStats):
    metrics = ('foot_yaw_left_rms_deg', 'foot_yaw_right_rms_deg', 'stance_yaw_rms_deg',
               'stance_omega_planned_rms_rad_s', 'stance_proxy_miss_rate', *(key+'_contribution' for key in TERMS))

    def __init__(self, motion):
        super().__init__(motion)
        self.contact_columns = tuple(f'{clip}/{name}' for clip in motion.names for name in self.metrics)
        self.contact_sums = motion.q.new_zeros((len(motion.names), 1+len(STEP_SUMS)+len(TERMS)), dtype=torch.float64)

    def record(self, out):
        super().record(out)
        terms = out['terms']
        keys = (*STEP_SUMS, *(key+'_contribution' for key in TERMS))
        values = torch.stack((torch.ones_like(out['r']), *(terms[key] for key in keys)), -1)
        if 'valid' in out:
            values = values.masked_fill(~out['valid'][:, None], 0.)
        self.contact_sums.index_add_(0, self.motion.clip_id[out['f']], values.double())

    def read(self, buffer, env):
        row = super().read(buffer, env)
        count, left, right, yaw, speed, miss, yaw_count, planned, *costs = self.contact_sums.unbind(dim=1)
        rms = lambda total, size: (total/size.clamp_min(1)).sqrt()
        values = torch.stack((rms(left, count).rad2deg(), rms(right, count).rad2deg(),
                              rms(yaw, yaw_count).rad2deg(), rms(speed, planned), miss/planned.clamp_min(1),
                              *(c/count.clamp_min(1) for c in costs)), -1)
        self.contact_sums.zero_()
        row.update(zip(self.contact_columns, values.flatten().tolist()))
        return row
