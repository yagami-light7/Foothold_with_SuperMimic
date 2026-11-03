#!/bin/bash
# Reuse an existing image; this S54 checkout always gets its own container.
set -euo pipefail
GPU=${1:-0}
MEM=${2:-32g}
NAME="${CONTAINER_NAME:-supermimic_s54}"
IMAGE="${DOCKER_IMAGE:-supermimic:latest}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
if [[ "$NAME" == supermimic ]]; then
  echo 'Refusing original G1 container name supermimic' >&2; exit 1
fi
docker image inspect "$IMAGE" >/dev/null
# Existing names are never removed or restarted by this script.
docker run -d --name "$NAME" --gpus "\"device=${GPU}\"" --memory="$MEM" --memory-swap="$MEM" --shm-size=2g --user root \
  -e HOME=/root -e WARP_CACHE_PATH=/workspace/supermimic/.cache/warp -e MPLCONFIGDIR=/tmp/mpl \
  -e TORCHINDUCTOR_CACHE_DIR=/workspace/supermimic/.cache/inductor \
  -v "$REPO":/workspace/supermimic -w /workspace/supermimic "$IMAGE" sleep infinity
