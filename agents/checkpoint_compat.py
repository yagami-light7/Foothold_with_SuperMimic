"""S54观测扩维迁移：保留网络、归一化和Adam历史，不修改原checkpoint。"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from tasks.foot_placement.spec import ACTOR_OBS_DIM, BODY_OBS_DIM


def checkpoint_input_dim(saved: dict) -> int:
    return int(saved["actor_state_dict"]["mlp.0.weight"].shape[1])


def checkpoint_task_config(path: Path) -> dict:
    """优先读取伴随配置，缺失时使用checkpoint内配置；网络维度为最终依据。"""
    path = Path(path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    companion = path.parent / "env_cfg.json"
    config = json.loads(companion.read_text()) if companion.is_file() else copy.deepcopy(
        (saved.get("infos") or {}).get("env_cfg", {}))
    dim = checkpoint_input_dim(saved)
    if dim not in (103, 104, 105):
        raise ValueError(f"不支持的S54输入维度：{dim}")
    config.update(observe_gait_frequency=dim != 103,
                  gait_frequency_obs_dim=dim - ACTOR_OBS_DIM if dim != 103 else 2,
                  observation_space=dim)
    return config


def _expand_columns(value: torch.Tensor, source_dim: int, target_dim: int,
                    kind: str = "weight", fill: float = 0.0) -> torch.Tensor:
    old_commands, new_commands = source_dim - ACTOR_OBS_DIM, target_dim - ACTOR_OBS_DIM
    result = value.new_full((*value.shape[:-1], target_dim), fill)
    result[..., :BODY_OBS_DIM] = value[..., :BODY_OBS_DIM]
    result[..., BODY_OBS_DIM + new_commands:] = value[..., BODY_OBS_DIM + old_commands:]
    if old_commands == 1:
        old = value[..., BODY_OBS_DIM:BODY_OBS_DIM + 1]
        if kind == "weight":
            # 相同左右输入时，两列之和等于原单列，保持旧函数。
            result[..., BODY_OBS_DIM:BODY_OBS_DIM + new_commands] = old / new_commands
        elif kind == "normalizer":
            result[..., BODY_OBS_DIM:BODY_OBS_DIM + new_commands] = old
        # 旧单命令拆成两个参数后，仅命令列的Adam历史重置；其他列完整保留。
    return result


def migrate_checkpoint_dict(saved: dict, target_dim: int) -> tuple[dict, dict]:
    source_dim = checkpoint_input_dim(saved)
    report = {"source_dim": source_dim, "target_dim": target_dim,
              "migrated": source_dim != target_dim, "optimizer_tensors_expanded": 0}
    if source_dim == target_dim:
        return saved, report
    if source_dim not in (103, 104) or target_dim not in (104, 105) or target_dim <= source_dim:
        raise ValueError(f"不支持{source_dim}→{target_dim}；请恢复checkpoint对应的观测开关")
    migrated = copy.deepcopy(saved)
    first_shapes = set()
    reference = float(((saved.get("infos") or {}).get("env_cfg") or {}).get("gait_frequency", .75))
    for model_name in ("actor_state_dict", "critic_state_dict"):
        model = migrated[model_name]
        weight = model["mlp.0.weight"]
        if weight.shape[1] != source_dim:
            raise ValueError("actor与critic输入维度不一致，拒绝隐式迁移")
        first_shapes.add(tuple(weight.shape))
        model["mlp.0.weight"] = _expand_columns(weight, source_dim, target_dim)
        for suffix, fill in (("_mean", reference), ("_var", .01), ("_std", .1)):
            key = "obs_normalizer." + suffix
            if key in model:
                model[key] = _expand_columns(model[key], source_dim, target_dim, "normalizer", fill)
    # Adam的参数编号和分组不变；首层矩阵对应的动量/方差补零到新形状。
    optimizer = migrated.get("optimizer_state_dict", {})
    for state in optimizer.get("state", {}).values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and tuple(value.shape) in first_shapes:
                state[key] = _expand_columns(value, source_dim, target_dim, "moment")
                report["optimizer_tensors_expanded"] += 1
    infos = migrated.get("infos") or {}
    config = infos.setdefault("env_cfg", {})
    config.update(observe_gait_frequency=True, gait_frequency_obs_dim=target_dim - ACTOR_OBS_DIM,
                  observation_space=target_dim)
    infos["observation_migration"] = report
    migrated["infos"] = infos
    return migrated, report


def load_s54_checkpoint(runner, path, load_cfg=None, map_location=None) -> dict:
    """训练/回放/eval共用加载入口；只在维度扩展时迁移，默认继承完整PPO状态。"""
    saved = torch.load(path, map_location=map_location or "cpu", weights_only=False)
    target_dim = int(runner.alg.actor.state_dict()["mlp.0.weight"].shape[1])
    saved, report = migrate_checkpoint_dict(saved, target_dim)
    load_iteration = runner.alg.load(saved, load_cfg, strict=True)
    if load_iteration:
        runner.current_learning_iteration = saved["iter"]
    if report["migrated"]:
        print("[S54 checkpoint] 观测迁移：", report, flush=True)
    return report
