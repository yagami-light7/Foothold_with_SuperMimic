"""SwanLab精简仪表盘；本地metrics.jsonl始终保留全部原始指标。"""

TRAIN_KEYS = {
    'Episode/normalized_return':'Train/return_0to1',
    'Reward/total':'Train/reward_per_step',
    'Train/mean_episode_length':'Train/episode_steps',
    'Termination/terminated_fraction':'Train/fall_per_control_step',
    'Loss/learning_rate':'PPO/learning_rate',
    'Loss/value':'PPO/value_loss',
    'Policy/mean_std':'PPO/action_std',
    'Command/initial_left_swing_fraction':'Train/left_start_fraction',
    **{f'Startup/{side}_{metric}':f'Startup/{side}_{metric}'
       for side in ('left', 'right') for metric in ('success', 'xy_error_m')},
}
REWARD_KEYS = ('tracking','swing_phase','stance_slip','stance_hop','body_motion','posture','action_rate','heading')
EVAL_KEYS = ('survival_min','left_success_min','right_success_min','late_xy_max_m',
             'both_air_max_fraction','stance_hop_max_fraction','stance_slip_max_m_s',
             'left_start_success_min','right_start_success_min',
             'first_left_success_min','first_right_success_min',
             'first_left_xy_max_m','first_right_xy_max_m',
             'corner_survival_fraction','corner_xy_error_m')


def compact_metrics(values):
    """固定核心曲线，首步专项增加8条；禁止每关节×每工况无限展开。"""
    result = {}
    for key,value in values.items():
        raw = key
        if '/' in key and key.split('/',1)[0] in ('straight','stand','lift','step','walk'):
            raw = key.split('/',1)[1]
        if raw in TRAIN_KEYS:
            result[TRAIN_KEYS[raw]]=value
        elif raw.startswith('Reward/') and raw.split('/',1)[1] in REWARD_KEYS:
            result[raw]=value
        elif raw.startswith('Eval/') and raw.split('/',1)[1] in EVAL_KEYS:
            result[raw]=value
        elif raw == 'Progress/remaining_wall_hours':
            result['Progress/remaining_hours']=value
        elif raw == 'Progress/command_phase':
            result['Progress/stage']=value
        elif raw == 'Curriculum/evaluation_passed':
            result['Progress/eval_passed']=value
        elif raw in ('Retention/candidate_accepted', 'Retention/best_updated'):
            result[raw] = value
        elif raw in ('Coverage/cell_fraction', 'Coverage/min_cell_transitions'):
            result[raw] = value
        elif raw == 'Hardware/used_mib':
            result['GPU/memory_MiB']=value
        elif raw == 'Hardware/util_percent':
            result['GPU/utilization_percent']=value
    return result
