#!/usr/bin/env bash
# Walking Stage1 only. The trainer and its uploader each live in tmux.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT"
MODE="${1:-start}"
STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
RUN="${2:-s54_stage1_walk_$STAMP}"
[[ "$RUN" =~ ^s54_[A-Za-z0-9_.-]+$ ]] || { echo 'Run name must start s54_'; exit 2; }
DIR="logs/$RUN"
S54_PYTHON="${S54_PYTHON:-$ROOT/../../IsaacLab_3_0/.venv/bin/python}"
SWANLAB_PYTHON="${SWANLAB_PYTHON:-$ROOT/.venv-swanlab/bin/python}"
PROFILE="${S54_STAGE1_PROFILE:-configs/motion_prior/stage1/stage1_walk_3090.json}"
MANIFEST="results/s54/motion_prior_stage0_verified/manifest.json"
RESUME="${3:-}"
case "$MODE" in
 attach) exec tmux attach -t "=$RUN" ;;
 status)
  for file in training_status.json launcher_exit.json train.exitcode runtime_stats.json swanlab_sync_status.json final_sync.exitcode; do
   [[ ! -f "$DIR/$file" ]] || { printf '%s\n' "$file"; cat "$DIR/$file"; printf '\n'; }
  done
  tail -n 5 "$DIR/stdout.log"; exit ;;
 stop)
  pid="$(cat "$DIR/trainer.pid")";cmd="$(ps -p "$pid" -o args=)"
  [[ "$cmd" == *supermimic/train.py* && "$cmd" == *"logs/$RUN"* ]] || { echo 'Stale trainer PID; refusing signal'; exit 1; }
  kill -TERM "$pid"; echo 'Safe save/stop requested; uploader finishes independently.'; exit ;;
 start|dry-run|resume|_worker) ;;
 *) echo "Usage: $0 {start|dry-run|status|attach|stop} [s54_RUN] | resume NEW_RUN logs/OLD/last.pt"; exit 2 ;;
esac
[[ -x "$S54_PYTHON" && -x "$SWANLAB_PYTHON" ]] || { echo 'Set S54_PYTHON and SWANLAB_PYTHON to existing Python executables'; exit 1; }
if [[ "$MODE" == resume || ( "$MODE" == _worker && -n "$RESUME" ) ]]; then
 [[ -f "$RESUME" ]] || { echo 'Resume checkpoint missing'; exit 1; }
 PROFILE="$(dirname "$RESUME")/train_profile.json"
 [[ -f "$PROFILE" ]] || { echo 'Resume requires the source train_profile.json'; exit 1; }
fi
ARGS=(--stage stage1 --stage1-train --manifest "$MANIFEST" --profile "$PROFILE" --run "$RUN")
ARGS+=(--experiment-name "Stage1_walk1_subject1_${RUN#s54_stage1_walk_}_seed54")
if [[ -n "$RESUME" ]]; then
 TARGET="$("$S54_PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["train"]["iterations"])' "$PROFILE")"
 ARGS+=(--resume "$RESUME" --resume_iterations "$TARGET")
fi
if [[ "$MODE" == _worker ]]; then
 exec > >(tee -a "logs/${RUN}_launcher.log") 2>&1
 exec 9>logs/.s54_curriculum_gpu.lock
 flock -n 9 || { echo 'Another S54 GPU job holds the lock'; exit 73; }
 [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || { echo 'GPU busy; no training started'; exit 73; }
 exec "$S54_PYTHON" -u tools/train/run_s54_motion_prior.py "${ARGS[@]}" --execute --stream-log --swanlab-mode online
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
# Validate the exact selected reference, including hashes, without allocating CUDA.
"$S54_PYTHON" - <<'PY'
from pathlib import Path
from supermimic.motion_prior_data import load_manifest
files,_=load_manifest('results/s54/motion_prior_stage0_verified/manifest.json','train')
if [Path(f).stem for f in files]!=['walk1_subject1']:
    raise SystemExit('Stage1 launcher requires exactly walk1_subject1 in the training manifest')
PY
"$S54_PYTHON" tools/train/run_s54_motion_prior.py "${ARGS[@]}" --dry-run
[[ "$MODE" != dry-run ]] || exit 0
[[ ! -e "$DIR" ]] || { echo 'Existing run protected; use resume with a NEW run name'; exit 1; }
command -v tmux >/dev/null
tmux has-session -t "=$RUN" 2>/dev/null && { echo 'Session exists'; exit 1; }
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || { echo 'GPU busy; retry when idle'; exit 1; }
FREE="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)"
(( FREE >= 8192 )) || { echo 'Need >=8192MiB free for the tested Stage1 profile'; exit 1; }
mkdir -p logs
printf -v JOB '%q ' env "S54_PYTHON=$S54_PYTHON" "SWANLAB_PYTHON=$SWANLAB_PYTHON" \
 "S54_STAGE1_PROFILE=$PROFILE" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
 bash tools/train/run_s54_stage1_walk.sh _worker "$RUN" "$RESUME"
tmux new-session -d -s "$RUN" -c "$ROOT" "$JOB"
tmux set-option -w -t "$RUN" remain-on-exit on
echo "Started $RUN"
echo "Live terminal: tmux attach -t $RUN"
echo "Log: tail -F $DIR/stdout.log"
echo "SwanLab: Foothold_with_SuperMimic; uploader session ${RUN}_upload"
