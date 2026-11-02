#!/usr/bin/env bash
# 从仓库内最新预训练模型完整续训12小时，2048环境、SwanLab在线精简上传。
# 仅将首步XY倍率1.5提高到2.0；保留15格固定配额和10%角点，不运行任何eval。
# 默认每250次更新保存checkpoint，可用--save-interval覆盖；墙钟截止另存model_interrupted.pt。
# 用法：bash scripts/train_walk_retention_12h.sh [--checkpoint FILE] [--hours 12] [--save-interval 250] [--dry-run]
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# 恢复本次实验时，继承其配置、SwanLab身份和剩余预算。
for argument in "$@"; do
  if [[ "$argument" == "--resume-dir" ]]; then
    exec bash "$TASK_ROOT/scripts/train_walk_12h.sh" --chunk-iterations 50000 "$@"
  fi
done

# 随Git携带actor/critic/优化器/归一化及伴随配置，服务器无需复制整个logs目录。
INITIAL_CHECKPOINT="$TASK_ROOT/pretrained/s54_walk_12h_20261102/model.pt"
TRAIN_OUTPUT="$TASK_ROOT/logs/walk/$(TZ=Asia/Shanghai date +%Y-%m-%d_%H-%M-%S)_walk_retention_firstxy2_12h"
exec bash "$TASK_ROOT/scripts/train_walk_12h.sh" \
  --checkpoint "$INITIAL_CHECKPOINT" \
  --adaptive-plan "$TASK_ROOT/configs/walk_retention_12h.json" \
  --hours 12 --num-envs 2048 --chunk-iterations 50000 \
  --swanlab-detail compact \
  --output-dir "$TRAIN_OUTPUT" "$@"
