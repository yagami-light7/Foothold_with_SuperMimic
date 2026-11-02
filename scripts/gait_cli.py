"""训练、回放、评估共用的节奏参数；顶层只用标准库，兼容独立SwanLab进程。"""

import argparse
import math


GAIT_FIELDS = ("gait_frequency", "gait_frequency_range", "left_gait_frequency", "right_gait_frequency",
               "left_gait_frequency_range", "right_gait_frequency_range", "observe_gait_frequency")


def add_gait_arguments(parser):
    parser.add_argument("--initial-swing-side", "--initial_swing_side", choices=("left", "right", "random"),
                        default=None, help="每次reset的起始摆动脚；random为左右各50%，覆盖checkpoint配置")
    for name in GAIT_FIELDS[:-1]:
        options = ("--" + name.replace("_", "-"), "--" + name)
        if name.endswith("range"):
            parser.add_argument(*options, dest=name, type=float, nargs=2, metavar=("MIN_HZ", "MAX_HZ"),
                                help="本侧摆动时长为1/(2f)；范围内每个episode采样一次")
        else:
            parser.add_argument(*options, dest=name, type=float, help="等效频率Hz；本侧摆动时长为1/(2f)")
    parser.add_argument("--observe-gait-frequency", "--observe_gait_frequency",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="启用105维左右节奏观测；给定节奏参数时默认自动启用")


def validate_gait_arguments(args):
    for name in GAIT_FIELDS[:-1]:
        value = getattr(args, name, None)
        if value is None:
            continue
        values = value if name.endswith("range") else (value,)
        if any(not math.isfinite(x) or x <= 0 for x in values):
            raise ValueError(f"{name}必须为有限正数")
        if name.endswith("range") and value[0] > value[1]:
            raise ValueError(f"{name}下限不能大于上限")
    for side in ("left", "right"):
        if (getattr(args, f"{side}_gait_frequency", None) is not None
                and getattr(args, f"{side}_gait_frequency_range", None) is not None):
            raise ValueError(f"{side}侧固定频率和随机范围不能同时指定")


def apply_gait_overrides(env, args):
    from configs.s54_curriculum import sync_observation_space
    validate_gait_arguments(args)
    initial_side = getattr(args, "initial_swing_side", None)
    if initial_side is not None:
        if initial_side not in ("left", "right", "random"):
            raise ValueError("initial_swing_side必须为left/right/random")
        env.initial_swing_side = initial_side
    if (getattr(args, "gait_frequency", None) is not None
            or getattr(args, "gait_frequency_range", None) is not None):
        # 显式共同命令先清除旧单侧覆盖，随后再应用本次明确给出的单侧参数。
        env.left_gait_frequency = env.right_gait_frequency = None
        env.left_gait_frequency_range = env.right_gait_frequency_range = None
    changed = False
    for name in GAIT_FIELDS[:-1]:
        value = getattr(args, name, None)
        if value is None:
            continue
        changed = True
        setattr(env, name, tuple(value) if name.endswith("range") else value)
        # 显式固定命令必须清除保存配置中的同侧随机范围，反之亦然。
        if name.endswith("range"):
            if name != "gait_frequency_range":
                setattr(env, name[:-6], None)
        else:
            setattr(env, name + "_range", None)
    if changed or getattr(args, "observe_gait_frequency", None) is True:
        env.observe_gait_frequency = True
        env.gait_frequency_obs_dim = 2
    if changed:
        # 显式回放/固定点评估的命令优先于checkpoint里的混合分组。
        env.command_groups = None
        env.command_resample_half_cycles = 0
    if getattr(args, "observe_gait_frequency", None) is False:
        env.observe_gait_frequency = False
    return sync_observation_space(env)


def gait_cli_args(args):
    """监督器只转发显式参数；未指定的字段由checkpoint配置恢复。"""
    result = []
    initial_side = getattr(args, "initial_swing_side", None)
    if initial_side is not None:
        result += ["--initial-swing-side", initial_side]
    for name in GAIT_FIELDS[:-1]:
        value = getattr(args, name, None)
        if value is not None:
            result.append("--" + name.replace("_", "-"))
            result.extend(str(x) for x in (value if name.endswith("range") else (value,)))
    observe = getattr(args, "observe_gait_frequency", None)
    if observe is not None:
        result.append("--observe-gait-frequency" if observe else "--no-observe-gait-frequency")
    return result


def frequency_corners(config):
    """左右节奏边界组合，供监督器分档eval使用；不依赖Torch/Isaac Lab。"""
    from itertools import product
    common = config.get("gait_frequency_range") or (config.get("gait_frequency", .75),) * 2
    limits = []
    for side in ("left", "right"):
        fixed = config.get(f"{side}_gait_frequency")
        bounds = config.get(f"{side}_gait_frequency_range")
        limits.append(bounds or ((fixed, fixed) if fixed is not None else common))
    return sorted(set(product(*limits)))
