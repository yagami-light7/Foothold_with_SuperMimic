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
  m4-evaluate) script=supermimic/integration/m4_evaluation.py ;;
  *)
    printf '%s\n' 'Usage: bash tools/run_roban.sh prepare|test|smoke|train|evaluate|pd|m7|m7-test|m7-batch|m4-smoke|m4-train|m4-evaluate [arguments]' \
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
  m4-smoke|m4-train|m4-evaluate)
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
    else
      python_command=(-m supermimic.integration.m4_evaluation)
      dataset_mount+=(--env "MUJOCO_GL=${MUJOCO_GL:-egl}" --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics)
    fi
    # Isaac Lab 3 kit-less 默认未指定 visualizer，即为 Headless。
    python_command+=(--m4_reference /m4_reference.npz)
    ;;
esac
docker run --rm --gpus all --network=none --shm-size=2g \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=$repo,target=/workspace/supermimic" \
  --mount "type=bind,source=$repo/library,target=/workspace/supermimic/library,readonly" \
  --workdir /workspace/supermimic \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env WARP_CACHE_PATH=/workspace/supermimic/.cache/roban_warp \
  --env MPLCONFIGDIR=/tmp/roban_mpl \
  "${dataset_mount[@]}" \
  "${DOCKER_IMAGE:-supermimic:latest}" python "${python_command[@]}" "$@"
