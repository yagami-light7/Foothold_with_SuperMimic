#!/usr/bin/env bash
# 四档步长/节奏续训：前三阶段各3000次PPO更新，第四阶段补训到总计12小时。
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
TRAIN_HOURS=12
HOURS_EXPLICIT=false
TRAIN_ENVS=2048
ENVS_EXPLICIT=false
TRAIN_CHUNK=50000
INITIAL_CHECKPOINT="$TASK_ROOT/logs/straight/2026-10-30_18-26-34_stride_cadence_lr105/training/00012_straight/model_interrupted.pt"
CHECKPOINT_EXPLICIT=false
TRAIN_OUTPUT=""
RESUME_DIRECTORY=""
SCHEDULE_FILE="$TASK_ROOT/configs/walk_4stage_12h.json"
SCHEDULE_EXPLICIT=false
ADAPTIVE_FILE=""
SCHEDULE_TAIL=hold-last
TAIL_EXPLICIT=false
DRY_RUN=false
SWANLAB_DETAIL=compact
EVAL_ENVS=64
EVAL_STEPS=1000
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hours) TRAIN_HOURS="${2:?缺少小时数}"; HOURS_EXPLICIT=true; shift 2 ;;
    --num-envs) TRAIN_ENVS="${2:?缺少环境数}"; ENVS_EXPLICIT=true; shift 2 ;;
    --chunk-iterations) TRAIN_CHUNK="${2:?缺少每段PPO更新次数}"; shift 2 ;;
    --checkpoint) INITIAL_CHECKPOINT="${2:?缺少checkpoint}"; CHECKPOINT_EXPLICIT=true; shift 2 ;;
    --output-dir) TRAIN_OUTPUT="${2:?缺少输出目录}"; shift 2 ;;
    --resume-dir) RESUME_DIRECTORY="${2:?缺少恢复目录}"; shift 2 ;;
    --schedule) SCHEDULE_FILE="${2:?缺少课程JSON}"; SCHEDULE_EXPLICIT=true; shift 2 ;;
    --adaptive-plan) ADAPTIVE_FILE="${2:?缺少自适应课程JSON}"; shift 2 ;;
    --tail) SCHEDULE_TAIL="${2:?缺少stop或hold-last}"; TAIL_EXPLICIT=true; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --swanlab-detail) SWANLAB_DETAIL="${2:?缺少compact或full}"; shift 2 ;;
    --eval-envs) EVAL_ENVS="${2:?缺少评估环境数}"; shift 2 ;;
    --eval-steps) EVAL_STEPS="${2:?缺少评估控制步数}"; shift 2 ;;
    --help|-h)
      echo "用法: bash scripts/train_walk_12h.sh [--checkpoint FILE] [--dry-run]"
      echo "      [--hours 12] [--num-envs 2048] [--output-dir DIR] [--resume-dir DIR]"
      echo "      [--chunk-iterations 50000]"
      echo "      [--schedule configs/walk_4stage_12h.json] [--tail hold-last|stop]"
      echo "      [--adaptive-plan configs/walk_hard_region.json]（按物理评估晋级，与schedule互斥）"
      echo "      [--swanlab-detail compact|full] 默认精简上传，完整诊断保存在本地"
      echo "      [--eval-envs 64] [--eval-steps 1000] 评估步数至少1000"
      echo "步长单位为米，频率单位Hz；默认前三阶段各3000次PPO更新，第四阶段训练到总12小时。"
      echo "恢复同一实验时继承剩余时间、课程进度和SwanLab身份；checkpoint与resume-dir互斥。"
      echo "checkpoint保存间隔取agents/rsl_rl_ppo_cfg.py的save_interval；每个训练段结束后评估。"
      echo "adaptive-plan设置evaluation.enabled=false时关闭基线、周期和末尾eval，使用全部剩余预算训练。"
      echo "默认训练段在阶段边界结束；缩短chunk-iterations可定期评估，课程仍按更新数切换。"
      exit 0 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done
if [[ -n "$ADAPTIVE_FILE" ]] && "$SCHEDULE_EXPLICIT"; then
  echo "adaptive-plan与schedule不能同时指定" >&2; exit 2
fi
if [[ -n "$ADAPTIVE_FILE" ]] && "$TAIL_EXPLICIT"; then
  echo "自适应课程按评估晋级并持续到总预算，不使用tail参数" >&2; exit 2
fi

arguments=(
  --start-stage straight
  --rollout-steps 64 --mini-batches 4 --learning-rate 0.0001
  --observe-gait-frequency --normalization-prior-rollouts 0
  --max-iterations 50000 --chunk-iterations "$TRAIN_CHUNK"
  --eval-envs "$EVAL_ENVS" --eval-steps "$EVAL_STEPS" --eval-frequency-corners --eval-every-chunks 1
  --print-every 10 --seed 42 --project S54-Locomotion --mode online --continue-after-pass)
arguments+=(--swanlab-detail "$SWANLAB_DETAIL")
if [[ -n "$RESUME_DIRECTORY" ]]; then
  ! "$CHECKPOINT_EXPLICIT" || { echo "resume-dir不能与checkpoint并用" >&2; exit 2; }
  [[ -z "$TRAIN_OUTPUT" ]] || { echo "resume-dir不能与output-dir并用" >&2; exit 2; }
  [[ -f "$RESUME_DIRECTORY/state.json" ]] || { echo "恢复目录缺少state.json" >&2; exit 2; }
  TRAIN_OUTPUT="$RESUME_DIRECTORY"
  arguments+=(--resume-dir "$RESUME_DIRECTORY")
  if "$ENVS_EXPLICIT"; then arguments+=(--num-envs "$TRAIN_ENVS"); fi
  # 默认使用实验快照，不因后来编辑JSON或脚本而改变正在恢复的实验。
  if "$SCHEDULE_EXPLICIT"; then arguments+=(--command-schedule "$SCHEDULE_FILE"); fi
  if [[ -n "$ADAPTIVE_FILE" ]]; then arguments+=(--adaptive-plan "$ADAPTIVE_FILE"); fi
  if "$TAIL_EXPLICIT"; then arguments+=(--schedule-tail "$SCHEDULE_TAIL"); fi
  if "$HOURS_EXPLICIT"; then arguments+=(--max-hours "$TRAIN_HOURS"); fi
else
  [[ -f "$INITIAL_CHECKPOINT" ]] || { echo "checkpoint不存在: $INITIAL_CHECKPOINT" >&2; exit 2; }
  TRAIN_OUTPUT="${TRAIN_OUTPUT:-$TASK_ROOT/logs/walk/$(TZ=Asia/Shanghai date +%Y-%m-%d_%H-%M-%S)_walk_4stage}"
  [[ ! -e "$TRAIN_OUTPUT" ]] || { echo "输出目录已存在，请用resume-dir或新目录" >&2; exit 2; }
  arguments+=(--num-envs "$TRAIN_ENVS" --resume-checkpoint "$INITIAL_CHECKPOINT" --output-dir "$TRAIN_OUTPUT"
              --max-hours "$TRAIN_HOURS")
  if [[ -n "$ADAPTIVE_FILE" ]]; then
    arguments+=(--adaptive-plan "$ADAPTIVE_FILE")
  else
    arguments+=(--command-schedule "$SCHEDULE_FILE" --schedule-tail "$SCHEDULE_TAIL")
  fi
fi
if "$DRY_RUN"; then
  # 监督器顶层只依赖标准库；预览不安装SDK、不创建实验、不访问GPU或上传。
  exec python3 scripts/train_4060.py "${arguments[@]}" --dry-run
fi
echo "输出目录: $TRAIN_OUTPUT"
echo "按课程切换目标分布，完整继承PPO；终端remaining显示总预算剩余时长。"
mkdir -p -- "$(dirname -- "$TRAIN_OUTPUT")"
exec > >(trap '' INT TERM; exec tee -a "${TRAIN_OUTPUT}.console.log") 2>&1
exec bash scripts/train_4060.sh "${arguments[@]}"
