#!/usr/bin/env bash
# Reuse the existing kit-less image; each command gets its own temporary container.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
mode="${1:-help}"
if (( $# )); then shift; fi
case "$mode" in
  prepare) script=tools/prepare_roban_asset.py ;;
  test) script=tools/test_roban_foot.py ;;
  smoke) script=tools/test_roban_smoke.py ;;
  train) script=supermimic/train.py ;;
  evaluate) script=tools/roban_rollout.py ;;
  pd) script=tools/validate_roban_pd.py ;;
  m7) script=tools/roban_m7.py ;;
  m7-test) script=tools/test_roban_m7.py ;;
  m7-batch) script=tools/run_roban_m7_batch.py ;;
  m4-train) script=supermimic/train.py ;;
  m4-smoke) script=supermimic/integration/m4_task.py ;;
  m4-dynamics) script=supermimic/integration/m4_dynamics.py ;;
  m4-evaluate) script=supermimic/integration/m4_evaluation.py ;;
  m4-v3-rollout) script=supermimic/integration/m42v3_rollout.py ;;
  m4-v3-ac) script=supermimic/integration/m42v3_ac.py ;;
  m4-v3-wm) script=supermimic/integration/m42v3_wm.py ;;
  m4-v43-eval) script=supermimic/integration/m43_evaluation.py ;;
  m4-v43-audit) script=supermimic/integration/m43_wm_audit.py ;;
  m4-v44-eval) script=supermimic/integration/m44_evaluation.py ;;
  m4-v44-audit) script=supermimic/integration/m44_wm_audit.py ;;
  *)
    printf '%s\n' 'Usage: bash tools/run_roban.sh prepare|test|smoke|train|evaluate|pd|m7|m7-test|m7-batch|m4-smoke|m4-dynamics|m4-train|m4-evaluate|m4-v3-rollout|m4-v3-ac|m4-v3-wm|m4-v43-eval|m4-v43-audit|m4-v44-eval|m4-v44-audit [arguments]' \
      'Example: bash tools/run_roban.sh train --roban_config configs/roban/E1.json --log_dir logs/roban_E1_smoke'
    exit 0 ;;
esac
dataset_mount=()
python_command=("$script")
case "$mode" in
  m7|m7-test|m7-batch)
    dataset="${PMT_DATASET_ROOT:-$(dirname "$repo")/PMT/TCRS/outputs/roban_pillar_dataset}"
    if [[ ! -f "$dataset/dataset_manifest.json" ]]; then
      printf '%s\n' "PMT dataset manifest not found: $dataset/dataset_manifest.json" >&2
      exit 1
    fi
    dataset_mount=(--mount "type=bind,source=$dataset,target=/pmt_dataset,readonly" --env PMT_DATASET_ROOT=/pmt_dataset)
    ;;
  m4-smoke|m4-dynamics|m4-train|m4-evaluate|m4-v3-rollout|m4-v3-ac|m4-v3-wm|m4-v43-eval|m4-v43-audit|m4-v44-eval|m4-v44-audit)
    # 仅挂载本次实验的参考文件，生成器与其余数据留在容器外。
    reference="${M4_REFERENCE:-$(dirname "$repo")/Humanoid_Online_Motion_Generator/outputs/m3_7_1/forward/F2/reference.npz}"
    if [[ ! -f "$reference" ]]; then
      printf '%s\n' "F2 文件不存在，请将 M4_REFERENCE 设为原始 reference.npz 的绝对路径：$reference" >&2
      exit 1
    fi
    reference="$(realpath -- "$reference")"
    dataset_mount=(--mount "type=bind,source=$reference,target=/m4_reference.npz,readonly")
    if [[ "$mode" == m4-train ]]; then
      python_command+=(--roban_config configs/roban/m4_0.json --log_dir outputs/m4_0/checkpoints)
    elif [[ "$mode" == m4-smoke ]]; then
      python_command=(-m supermimic.integration.m4_task)
    elif [[ "$mode" == m4-dynamics ]]; then
      python_command=(-m supermimic.integration.m4_dynamics)
    elif [[ "$mode" == m4-evaluate ]]; then
      python_command=(-m supermimic.integration.m4_evaluation)
    else
      # v3 诊断共用参考文件，分别运行采样、价值与世界模型审计。
      python_command=(-m "${script%.py}")
      python_command[1]="${python_command[1]//\//.}"
    fi
    if [[ "$mode" == m4-train || "$mode" == m4-evaluate || "$mode" == m4-v3-rollout || "$mode" == m4-v43-eval || "$mode" == m4-v44-eval ]]; then
      dataset_mount+=(--env "MUJOCO_GL=${MUJOCO_GL:-egl}" --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics)
    fi
    # Isaac Lab 3 kit-less 默认未指定 visualizer，即为 Headless。
    if [[ "$mode" == m4-v3-wm ]]; then
      python_command+=(--reference /m4_reference.npz)
    else
      python_command+=(--m4_reference /m4_reference.npz)
    fi
    ;;
esac
docker_flags=()
launch=(python "${python_command[@]}" "$@")
if [[ -n "${ROBAN_TMUX_SESSION:-}" ]]; then
  # 镜像内 tmux 独立于 SSH。容器等待训练窗结束，并保留实际退出码。
  [[ "$mode" == m4-train && "$ROBAN_TMUX_SESSION" =~ ^[A-Za-z0-9_-]+$ ]]
  [[ "${ROBAN_TMUX_RUN_DIR:-}" =~ ^outputs/(m4_3|m4_4_unclipped)(/[A-Za-z0-9_.-]+)?$ ]]
  commit="$(git -C "$repo" rev-parse HEAD)"
  docker_flags=(--detach --name "$ROBAN_TMUX_SESSION" --env "S54_SOURCE_GIT_SHA=$commit")
  launch=(bash -c '
    set -euo pipefail
    session=$1; run=$2; shift 2
    mkdir -p "$run/logs"
    rm -f -- "$run/train.exitcode"
    printf -v trainer "%q " "$@"
    printf -v log "%q" "$run/logs/console.log"
    printf -v marker "%q" "$run/train.exitcode"
    tmux new-session -d -s "$session" -e SHELL=/bin/bash \
      "$trainer >>$log 2>&1; rc=\$?; echo \$rc >$marker; tmux wait-for -S ${session}-end; exit \$rc"
    tmux wait-for "${session}-end"
    exit "$(cat "$run/train.exitcode")"
  ' bash "$ROBAN_TMUX_SESSION" "$ROBAN_TMUX_RUN_DIR" "${launch[@]}")
fi
docker run --rm --gpus all --network=none --shm-size=2g "${docker_flags[@]}" \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=$repo,target=/workspace/supermimic" \
  --mount "type=bind,source=$repo/library,target=/workspace/supermimic/library,readonly" \
  --workdir /workspace/supermimic \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env WARP_CACHE_PATH=/workspace/supermimic/.cache/roban_warp \
  --env MPLCONFIGDIR=/tmp/roban_mpl \
  "${dataset_mount[@]}" \
  "${DOCKER_IMAGE:-supermimic:latest}" "${launch[@]}"
