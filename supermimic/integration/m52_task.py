"""从 M5.1-C1500 初始化的姿态奖励消融与分片段统计。"""

from dataclasses import dataclass

import torch

from ..roban import assert_checkpoint
from .m51_task import CONFIGS as M51_CONFIGS, OUTPUT as M51_OUTPUT, REPO, StanceFootMDP
from .m51_task import StanceFootRewardConfig, StanceFootSim, StanceTrainingStats
from .m52_orientation import tilt_error


OUTPUT = REPO/'outputs/m5_2'
SOURCE = M51_OUTPUT/'c_yaw_angular_slip/checkpoints/iter_1500.pt'
GROUPS = {'a': 'a_baseline', 'b': 'b_add_tilt', 'c': 'c_replace_upright'}
CONFIGS = {key: REPO/f'configs/roban/m5/m5_2_{key}.json' for key in GROUPS}
REWARD_SPEC = {'a': (.5, 0., 0), 'b': (.5, .5, 1), 'c': (0., .5, 0)}
REWARD_KEYS = ('root_tilt', 'root_tilt_sigma', 'root_tilt_penalty')
TRANSFER = 'm52_m51_c1500_all_models'
GRAD_PAIRS = tuple((left, 'root_tilt') for left in ('upright', 'foot_position', 'stance_yaw_huber',
                                                   'stance_angular_slip', 'bootstrap'))
ROOT_LOGS = ('upright_quality', 'root_tilt_quality', 'root_tilt_contribution')


@dataclass(frozen=True)
class RootTiltRewardConfig(StanceFootRewardConfig):
    root_tilt: float = 0.
    root_tilt_sigma: float = .03
    root_tilt_penalty: int = 0

    def __post_init__(self):
        super().__post_init__()
        if self.root_tilt_penalty not in (0, 1):
            raise ValueError('根部倾斜惩罚开关应为 0 或 1')

    @property
    def positive_sum(self):
        # B 组只增加非正项。C 组用同权重的正项补回 Upright 上界。
        return super().positive_sum + (0. if self.root_tilt_penalty else self.root_tilt)


def tilt_reward(rotation, target, quality, cfg):
    exponent = -tilt_error(rotation, target) / cfg.root_tilt_sigma
    value = exponent.exp()
    # expm1 保留小误差时的精度。质量门只作用在新增项。
    raw = exponent.expm1() if cfg.root_tilt_penalty else value
    return quality * raw, value


class RootTiltMDP(StanceFootMDP):
    reward_config_type = RootTiltRewardConfig
    root_tilt_terms = True

    def contract(self):
        return super().contract() | {'mode': 'roban_m52_multiclip'}

    def root_tilt_reward(self, rotation, reference):
        return tilt_reward(rotation, reference['root_R'][:, 0], reference['quality'][:, 0], self.cfg)


class RootTiltSim(StanceFootSim):
    stop_on_invalid_physics = True


def expected_config(parent, group):
    upright, weight, penalty = REWARD_SPEC[group]
    reward = dict(parent['roban_reward_config'], upright=upright, root_tilt=weight,
                  root_tilt_penalty=penalty, root_tilt_sigma=.03)
    return dict(parent, m52_group=group, roban_reward_config=reward)


def initialize_m51(models, mdp, goals, physics, policy, imagination):
    ck = torch.load(SOURCE, map_location='cpu', weights_only=False, mmap=True)
    assert_checkpoint(ck)
    mdp.assert_wm_statistics(ck)
    fields = dict(physical_contract=physics, policy_contract=policy, imagination_contract=imagination,
                  goal_contract=goals.spec.contract(), data_contract=mdp.motion.validation)
    task = mdp.contract()
    task['config'] = {key: value for key, value in task['config'].items() if key not in REWARD_KEYS}
    task['config']['upright'] = .5
    task['mode'] = 'roban_m51_multiclip'
    if (ck['iteration'] != 1500 or ck['args']['m51_group'] != 'c' or ck.get('task_contract') != task
            or any(ck.get(key) != value for key, value in fields.items())):
        raise ValueError('M5.2 需要任务、物理、参考和网络契约匹配的 M5.1-C1500')
    for name, model in models.items():
        model.load_state_dict(ck[name], strict=True)
    return dict(strategy=TRANSFER, source_path=str(SOURCE.relative_to(REPO)), source_iteration=ck['iteration'],
                source_sim_steps=ck['sim_steps'], source_total_wall_seconds=ck['wall_time'])


class TiltTrainingStats(StanceTrainingStats):
    def __init__(self, motion):
        super().__init__(motion)
        self.root_columns = tuple(f'{clip}/{name}' for clip in motion.names for name in ROOT_LOGS)
        self.root_sums = motion.q.new_zeros((len(motion.names), 1+len(ROOT_LOGS)), dtype=torch.float64)

    def record(self, out):
        super().record(out)
        terms = out['terms']
        values = torch.stack((torch.ones_like(out['r']), terms['upright'], terms['root_tilt_quality'],
                              terms['root_tilt_contribution']), -1)
        if 'valid' in out:
            values = values.masked_fill(~out['valid'][:, None], 0.)
        self.root_sums.index_add_(0, self.motion.clip_id[out['f']], values.double())

    def read(self, buffer, env):
        row = super().read(buffer, env)
        average = self.root_sums[:, 1:] / self.root_sums[:, :1].clamp_min(1.)
        row.update(zip(self.root_columns, average.flatten().tolist()))
        self.root_sums.zero_()
        return row
