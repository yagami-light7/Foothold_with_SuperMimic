#!/usr/bin/env bash
# Host launcher; every run has a fresh tmux session, diagnostics and frozen eval.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO"
MODE="${1:-train}"
RUN="${2:-s54_m34_$(date +%Y%m%d_%H%M%S)}"
CONFIG="${3:-configs/s54_m34_4060.json}"
CONTAINER="${CONTAINER_NAME:-supermimic_s54_m34}"
if [[ "$MODE" != train && "$MODE" != resume ]]; then
  echo 'Usage: bash tools/run_s54_m34.sh train RUN [CONFIG]'
  echo '       bash tools/run_s54_m34.sh resume NEW_RUN CONFIG logs/OLD/last.pt'
  exit 2
fi
[[ "$RUN" =~ ^s54_[A-Za-z0-9_.-]+$ ]] || exit 2
[[ -f "$CONFIG" && ! -e "logs/$RUN" ]] || { echo 'Missing config or run already exists'; exit 2; }
if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
  CONTAINER_NAME="$CONTAINER" bash docker/run_container.sh 0 16g
fi
MOUNT="$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/workspace/supermimic"}}{{.Source}}{{end}}{{end}}' "$CONTAINER")"
[[ "$MOUNT" == "$REPO" ]] || { echo "Wrong container mount: $MOUNT"; exit 1; }
if [[ "$(docker inspect --format '{{.State.Running}}' "$CONTAINER")" != true ]]; then
  docker start "$CONTAINER" >/dev/null
fi
BUSY="$(docker exec "$CONTAINER" nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
[[ ! "$BUSY" =~ [0-9] ]] || { echo "GPU busy: $BUSY"; exit 1; }
ARGS=(python -u tools/run_s54_m34.py --run "$RUN" --config "$CONFIG")
if [[ "$MODE" == resume ]]; then
  [[ -f "${4:-}" ]] || { echo 'Resume checkpoint required'; exit 2; }
  ARGS+=(--resume "$4")
fi
printf -v JOB '%q ' bash tools/run_s54_job.sh "logs/$RUN" "${ARGS[@]}"
SOURCE_SHA="$(git rev-parse HEAD)"
docker exec -w /workspace/supermimic "$CONTAINER" tmux new-session -d -s "$RUN" \
  -e "S54_SOURCE_GIT_SHA=$SOURCE_SHA" -c /workspace/supermimic "$JOB"
echo "Started: $RUN (PD -> smoke -> train with periodic frozen train/test eval -> videos/report)"
echo "Log: logs/$RUN/stdout.log"
echo "Attach: docker exec -it $CONTAINER tmux attach -t $RUN"
