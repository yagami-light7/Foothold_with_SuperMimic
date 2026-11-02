"""专项训练的可序列化配置；只依赖标准库，SwanLab监督器可直接使用。"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import replace, fields
from pathlib import Path
from configs.s54_flat_reward import FlatWalkRewardCfg


PROFILE_DEFAULTS = {
    # 旧checkpoint缺少此字段时保持左脚起步；新课程显式启用random。
    "initial_swing_side": "left",
    "command_groups": None,
    "command_resample_half_cycles": 0,
    "command_max_length_change": 0.02,
    "command_max_frequency_change": 0.10,
    "command_warmup_half_cycles": 0,
    "observation_noise_scale": 0.0,
    "reset_joint_position_noise": 0.0,
    "reset_root_velocity_noise": 0.0,
    "body_mass_scale_range": None,
    "foot_friction_range": None,
    "push_interval_s": None,
    "push_velocity_max": 0.0,
}


def finite_number(value, name, minimum=0.0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name}必须是大于等于{minimum}的有限数")
    return float(value)


def bounds(value, name, positive=False):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name}需要[下限,上限]")
    low, high = [finite_number(x, name) for x in value]
    if low > high or (positive and low == 0):
        raise ValueError(f"{name}范围不合法")
    return [low, high]


def validate_command_groups(groups):
    if groups is None:
        return None
    if not isinstance(groups, list) or not groups:
        raise ValueError("command_groups必须是非空列表或None")
    result, names = [], set()
    for group in groups:
        if not isinstance(group, dict) or set(group) - {"name", "fraction", "step_length_range", "frequency_range", "coupled"}:
            raise ValueError("command_groups包含未知字段")
        name = group.get("name")
        if not isinstance(name, str) or not name.isidentifier() or name in names:
            raise ValueError("命令组名称必须是唯一标识符")
        fraction = finite_number(group.get("fraction"), "fraction")
        if not 0 < fraction <= 1:
            raise ValueError("命令组fraction必须在(0,1]内")
        coupled = group.get("coupled", True)
        if not isinstance(coupled, bool):
            raise ValueError("coupled必须是bool")
        result.append({"name": name, "fraction": fraction,
                       "step_length_range": bounds(group.get("step_length_range"), name + ".step_length_range"),
                       "frequency_range": bounds(group.get("frequency_range"), name + ".frequency_range", True),
                       "coupled": coupled})
        names.add(name)
    if not math.isclose(sum(g["fraction"] for g in result), 1.0, abs_tol=1e-8):
        raise ValueError("所有命令组fraction之和必须为1")
    return result


def group_counts(groups, count):
    """最大余数法分配整数环境数；总数精确，reset不会改变分组。"""
    raw = [g["fraction"] * count for g in groups]
    counts = [math.floor(x) for x in raw]
    order = sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)
    for index in order[:count - sum(counts)]:
        counts[index] += 1
    return counts


def coverage_cells(phase, num_envs=None):
    """检查声明的二维网格无空洞；每格必须对应一个固定环境组。"""
    grid = phase.get('coverage_grid')
    if grid is None:
        return []
    if not isinstance(grid, dict) or set(grid) != {'step_edges', 'frequency_edges'}:
        raise ValueError('coverage_grid需要step_edges和frequency_edges')
    for name, positive in (('step_edges', False), ('frequency_edges', True)):
        edges = grid[name]
        if not isinstance(edges, list) or len(edges) < 2:
            raise ValueError(f'coverage_grid.{name}至少需要两个边界')
        for value in edges:
            finite_number(value, name, 1e-8 if positive else 0.)
        if any(a >= b for a, b in zip(edges, edges[1:])):
            raise ValueError(f'coverage_grid.{name}必须严格递增')
    profile = phase['profile']
    if profile.get('command_warmup_half_cycles', 0):
        raise ValueError('固定网格覆盖不能启用命令预热，否则实际命令会落到格外')
    groups = profile['command_groups']
    counts = group_counts(groups, num_envs) if num_envs is not None else [None] * len(groups)
    cells = []
    for low_d, high_d in zip(grid['step_edges'], grid['step_edges'][1:]):
        for low_f, high_f in zip(grid['frequency_edges'], grid['frequency_edges'][1:]):
            matches = [(g, n) for g, n in zip(groups, counts)
                       if g['coupled'] and all(math.isclose(a, b, abs_tol=1e-8) for a, b in zip(
                           g['step_length_range'] + g['frequency_range'], [low_d, high_d, low_f, high_f]))]
            if len(matches) != 1:
                raise ValueError(f'覆盖网格[{low_d},{high_d}]m×[{low_f},{high_f}]Hz需要唯一的同频命令组')
            group, count = matches[0]
            if count == 0:
                raise ValueError(f'环境数不足：覆盖区间{group["name"]}分到0个环境')
            cells.append(dict(name=group['name'], step_length_range=[low_d, high_d],
                              frequency_range=[low_f, high_f], environments=count))
    return cells


def validate_profile(profile):
    if not isinstance(profile, dict):
        raise ValueError("task profile必须是对象")
    allowed = set(PROFILE_DEFAULTS) | {"episode_length_s", "reward_cfg"}
    if set(profile) - allowed:
        raise ValueError(f"task profile包含未知字段：{set(profile) - allowed}")
    result = copy.deepcopy(profile)
    for name, value in profile.items():
        if name == "initial_swing_side":
            if value not in ("left", "right", "random"):
                raise ValueError("initial_swing_side必须为left/right/random")
        elif name == "command_groups":
            result[name] = validate_command_groups(value)
        elif name in ("body_mass_scale_range", "foot_friction_range", "push_interval_s"):
            result[name] = None if value is None else bounds(value, name, True)
        elif name == "reward_cfg":
            if isinstance(value, dict) and value.get('profile') == 'flat_walk_v2':
                allowed_reward = {f.name for f in fields(FlatWalkRewardCfg)}
                if set(value)-allowed_reward:
                    raise ValueError("flat_walk_v2包含未知奖励参数")
                cfg = FlatWalkRewardCfg(**value)
                for f in fields(cfg):
                    if f.name != 'profile':
                        finite_number(getattr(cfg,f.name),f.name)
                for key in ('swing_height','hop_height_scale','hop_velocity_scale','heading_sigma','yaw_rate_sigma',
                            'swing_xy_coarse_sharpness'):
                    if getattr(cfg,key) <= 0:
                        raise ValueError(f'{key}必须为正数')
                if not 0 <= cfg.swing_xy_coarse_weight <= 1:
                    raise ValueError('swing_xy_coarse_weight必须在[0,1]内')
                if not 0 < cfg.swing_window_half_width <= .25:
                    raise ValueError('摆动奖励半窗必须在(0,.25]')
                continue
            if not isinstance(value, dict) or set(value) - {"xy_coarse_weight", "xy_coarse_sharpness"}:
                raise ValueError("奖励profile只允许显式设置双尺度XY参数")
            for key, number in value.items():
                finite_number(number, key, 1e-8 if key.endswith("sharpness") else 0.)
            if not 0 <= value.get("xy_coarse_weight", 0) <= 1:
                raise ValueError("xy_coarse_weight必须在[0,1]内")
        else:
            finite_number(value, name, 0.02 if name == "episode_length_s" else 0.)
            if name in ("command_resample_half_cycles", "command_warmup_half_cycles"):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"{name}必须是非负整数")
    if result.get("push_velocity_max", 0) and result.get("push_interval_s") is None:
        raise ValueError("启用推扰需要push_interval_s")
    return result


def apply_task_profile(cfg, path):
    """在checkpoint恢复之后覆盖本次明确给出的参数，其余奖励和控制参数保留。"""
    if path is None:
        return
    data = json.loads(Path(path).read_text()) if not isinstance(path, dict) else path
    for name, value in validate_profile(data).items():
        if name == "reward_cfg":
            cfg.reward_cfg = FlatWalkRewardCfg(**value) if value.get('profile') == 'flat_walk_v2' else replace(cfg.reward_cfg, **value)
        else:
            setattr(cfg, name, value)


def clean_profile():
    """固定点评估关闭训练分组和扰动；奖励版本沿用checkpoint或阶段的显式配置。"""
    return {**copy.deepcopy(PROFILE_DEFAULTS), "episode_length_s": 10.0}


def load_adaptive_plan(path):
    data = json.loads(Path(path).read_text())
    if (data.get("schema_version") != 2 or data.get("length_unit") != "m"
            or data.get("frequency_unit") != "Hz"):
        raise ValueError("自适应课程要求schema_version=2，单位m/Hz")
    phases = data.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("自适应课程phases不能为空")
    names = set()
    for phase in phases:
        name = phase.get("name")
        if not isinstance(name, str) or not name.isidentifier() or name in names:
            raise ValueError("阶段名称必须是唯一标识符")
        names.add(name)
        for key in ("minimum_updates", "passes_required"):
            if isinstance(phase.get(key), bool) or not isinstance(phase.get(key), int) or phase[key] < (1 if key == "passes_required" else 0):
                raise ValueError(f"{key}数值非法")
        finite_number(phase.get("learning_rate"), "learning_rate", 1e-8)
        if phase["learning_rate"] >= .01:
            raise ValueError("learning_rate应小于0.01")
        phase["profile"] = validate_profile(phase["profile"])
        if not phase["profile"].get("command_groups"):
            raise ValueError("自适应阶段必须声明混合command_groups")
        coverage_cells(phase)
        for key in ("required_cases", "full_cases"):
            cases = phase.get(key, [])
            if not isinstance(cases, list) or (key == "required_cases" and not cases):
                raise ValueError("required_cases不能为空")
            for case in cases:
                if not isinstance(case, list) or len(case) != 3:
                    raise ValueError("评估点应为[步长,左频率,右频率]")
                for i, value in enumerate(case):
                    finite_number(value, "eval case", 0. if i == 0 else 1e-8)
    interval = data.get("full_eval_every_chunks", 3)
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
        raise ValueError("full_eval_every_chunks必须是正整数")
    data["full_eval_every_chunks"] = interval
    if 'evaluation' in data:
        settings = data['evaluation']
        flags = {'enabled', 'baseline', 'fast', 'monitor_max_corner', 'fixed_seed'}
        if not isinstance(settings, dict) or set(settings) - flags - {'reserve_seconds'}:
            raise ValueError('evaluation包含未知字段')
        for key in flags & set(settings):
            if not isinstance(settings[key], bool):
                raise ValueError(f'evaluation.{key}必须是bool')
        finite_number(settings.get('reserve_seconds', 0), 'reserve_seconds')
        if not settings.get('enabled', True):
            if settings.get('baseline') or settings.get('reserve_seconds', 0):
                raise ValueError('关闭evaluation时，baseline必须关闭且reserve_seconds必须为0')
            if len(phases) != 1:
                raise ValueError('关闭evaluation仅支持单阶段，避免未经评估自动晋级')
    for phase in phases:
        if not isinstance(phase.get('require_first_step', False), bool):
            raise ValueError('require_first_step必须是bool')
        if 'checkpoint_selection' in phase:
            selection = phase['checkpoint_selection']
            keys = {'minimum_survival', 'max_xy_regression_fraction', 'absolute_xy_tolerance_m',
                    'max_success_drop', 'minimum_score_improvement'}
            if not isinstance(selection, dict) or set(selection) != keys:
                raise ValueError('checkpoint_selection必须完整声明五个筛选参数')
            for key, value in selection.items():
                finite_number(value, key)
            if (not 0 < selection['minimum_survival'] <= 1 or selection['max_success_drop'] > 1
                    or selection['max_xy_regression_fraction'] > 1
                    or selection['minimum_score_improvement'] <= 0):
                raise ValueError('checkpoint_selection筛选范围非法')
            settings = data.get('evaluation', {})
            if settings.get('enabled', True) and (not settings.get('baseline') or not settings.get('fixed_seed')):
                raise ValueError('checkpoint筛选要求baseline和fixed_seed开启')
            if [.4, 2., 2.] not in phase['required_cases']:
                raise ValueError('能力保持筛选必须将40cm/2Hz列入required_cases')
    return data


def adaptive_gate(result):
    """用物理指标决定晋级；独立于训练奖励的宽/窄核和归一化分数。"""
    checks = {
        "survival_ge_95pct": result["survival_fraction"] >= .95,
        "both_feet_success_ge_80pct": min(result["half_cycle_success_left_right"]) >= .8,
        "alternating_8_ge_80pct": result["consecutive_8_success_fraction"] >= .8,
        "late_xy_le_3cm": result["late_xy_error_m"] <= .03,
        "upright_ge_95pct": result["mean_upright"] >= .95,
        "heading_le_10deg": result["max_abs_heading_error_rad"] <= math.radians(10),
        "forward_ge_half_nominal": result["mean_reference_forward_velocity"] >= .5 * result["command_nominal_forward_speed_m_s"],
    }
    if result.get("initial_swing_side") == "random":
        by_start = result.get("by_initial_swing", {})
        checks["both_start_sides_evaluated"] = set(by_start) == {"left", "right"}
        for side in ("left", "right"):
            scores = by_start.get(side, {}).get("half_cycle_success_left_right", [0., 0.])
            checks[f"{side}_start_both_feet_success_ge_80pct"] = min(scores) >= .8
    if result.get('reward_profile') == 'flat_walk_v2':
        # 平地步行的质量门槛，独立于奖励分数；不把腾空跳跃或拖脚算作成功。
        checks['both_air_fraction_le_2pct'] = result.get('both_air_fraction',float('inf')) <= .02
        checks['stance_hop_fraction_le_2pct'] = result.get('stance_hop_fraction',float('inf')) <= .02
        checks['contact_slip_speed_le_0p10m_s'] = result.get('contact_foot_slip_speed_m_s',float('inf')) <= .10
    return all(checks.values()), checks
