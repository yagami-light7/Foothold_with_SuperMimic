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
  *)
    printf '%s\n' 'Usage: bash tools/run_roban.sh prepare|test|smoke|train|evaluate|pd [arguments]' \
      'Example: bash tools/run_roban.sh train --roban_config configs/roban/E1.json --log_dir logs/roban_E1_smoke'
    exit 0 ;;
esac
docker run --rm --gpus all --network=none --shm-size=2g \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=$repo,target=/workspace/supermimic" \
  --mount "type=bind,source=$repo/library,target=/workspace/supermimic/library,readonly" \
  --workdir /workspace/supermimic \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env WARP_CACHE_PATH=/workspace/supermimic/.cache/roban_warp \
  --env MPLCONFIGDIR=/tmp/roban_mpl \
  "${DOCKER_IMAGE:-supermimic:latest}" python "$script" "$@"
