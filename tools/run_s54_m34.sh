#!/usr/bin/env bash
# Host launcher; every run has a fresh tmux session, diagnostics and frozen eval.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO"

# SwanLab runs on the Docker HOST. Install/login once under the same user.
# Enable with SWANLAB_ENABLED=true or edit the defaults here.
SWANLAB_ENABLED="${SWANLAB_ENABLED:-false}"
SWANLAB_MODE="${SWANLAB_MODE:-online}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-SuperMimic}"
SWANLAB_WORKSPACE="${SWANLAB_WORKSPACE:-}"
SWANLAB_EXPERIMENT="${SWANLAB_EXPERIMENT:-}"
SWANLAB_POLL_SECONDS="${SWANLAB_POLL_SECONDS:-5}"
SWANLAB_STARTUP_TIMEOUT="${SWANLAB_STARTUP_TIMEOUT:-3600}"
SWANLAB_PYTHON="${SWANLAB_PYTHON:-$REPO/.venv-swanlab/bin/python}"

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
if [[ "$MODE" == resume ]]; then
  [[ -f "${4:-}" ]] || { echo 'Resume checkpoint required'; exit 2; }
fi
[[ "$SWANLAB_ENABLED" == true || "$SWANLAB_ENABLED" == false ]] || { echo 'SWANLAB_ENABLED must be true or false'; exit 2; }
[[ "$SWANLAB_MODE" == online || "$SWANLAB_MODE" == offline ]] || { echo 'SWANLAB_MODE must be online or offline'; exit 2; }
SYNC_ARGS=(--run_dir "logs/$RUN" --project "$SWANLAB_PROJECT" --mode "$SWANLAB_MODE"
  --experiment_name "${SWANLAB_EXPERIMENT:-$RUN}" --core_only --follow --completion_marker
  --poll_seconds "$SWANLAB_POLL_SECONDS" --startup_timeout "$SWANLAB_STARTUP_TIMEOUT"
  --container "$CONTAINER" --session "$RUN")
if [[ -n "$SWANLAB_WORKSPACE" ]]; then SYNC_ARGS+=(--workspace "$SWANLAB_WORKSPACE"); fi
if [[ "$SWANLAB_ENABLED" == true ]]; then
  [[ -x "$SWANLAB_PYTHON" ]] || { echo "Missing SwanLab Python: $SWANLAB_PYTHON; install the host SDK first"; exit 1; }
  "$SWANLAB_PYTHON" tools/sync_swanlab.py --check --mode "$SWANLAB_MODE"
fi
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
if ! docker exec -w /workspace/supermimic "$CONTAINER" python tools/check_s54_cuda.py; then
  echo 'CUDA preflight failed; no run directory, tmux job or uploader was created.' >&2
  exit 1
fi
ARGS=(python -u tools/run_s54_m34.py --run "$RUN" --config "$CONFIG")
if [[ "$MODE" == resume ]]; then
  ARGS+=(--resume "$4")
fi
printf -v JOB '%q ' bash tools/run_s54_job.sh "logs/$RUN" "${ARGS[@]}"
SOURCE_SHA="$(git rev-parse HEAD)"
# The host uploader must own the run directory; the container may create
# root-owned files, but the host can still create its own sync status/ID files.
mkdir -p logs
mkdir "logs/$RUN"
docker exec -w /workspace/supermimic "$CONTAINER" tmux new-session -d -s "$RUN" \
  -e "S54_SOURCE_GIT_SHA=$SOURCE_SHA" -c /workspace/supermimic "$JOB"
if [[ "$SWANLAB_ENABLED" == true ]]; then
  nohup setsid "$SWANLAB_PYTHON" -u tools/sync_swanlab.py "${SYNC_ARGS[@]}" \
    > "logs/$RUN/swanlab_sync.log" 2>&1 < /dev/null &
  echo "SwanLab: $SWANLAB_PROJECT / ${SWANLAB_EXPERIMENT:-$RUN} ($SWANLAB_MODE, core only)"
  echo "SwanLab log: logs/$RUN/swanlab_sync.log"
else
  echo 'SwanLab: disabled (set SWANLAB_ENABLED=true to enable)'
fi
if [[ "$MODE" == resume ]]; then
  echo "Started: $RUN (resume checkpoint -> periodic frozen train/test eval -> videos/report; PD/fresh smoke skipped)"
else
  echo "Started: $RUN (PD -> smoke -> train with periodic frozen train/test eval -> videos/report)"
fi
echo "Log: logs/$RUN/stdout.log"
echo "Attach: docker exec -it $CONTAINER tmux attach -t $RUN"
