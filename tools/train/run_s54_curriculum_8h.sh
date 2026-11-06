#!/usr/bin/env bash
# Host launcher. All GPU work (including evaluation) stays in one tmux trainer.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$REPO"
STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
MODE="${1:-help}"; (( $#==0 )) || shift
RUN="s54_curriculum_$STAMP"
[[ "$MODE" != fresh ]] || RUN="s54_curriculum_fresh_$STAMP"
if (( $# )) && [[ "$1" != --* ]]; then RUN="$1"; shift; fi
[[ "$RUN" =~ ^s54_[A-Za-z0-9_.-]+$ ]] || { echo 'Invalid run name' >&2; exit 2; }
DIR="logs/$RUN"; SESSION="$RUN"
S54_PYTHON="${S54_PYTHON:-/home/rm/workspace/IsaacLab_3_0/.venv/bin/python}"
SWANLAB_PYTHON="${SWANLAB_PYTHON:-$REPO/.venv-swanlab/bin/python}"
case "$MODE" in
  status) cat "$DIR/training_status.json" "$DIR/train.exitcode" "$DIR/swanlab_sync_status.json" 2>/dev/null || true; tail -n 4 "$DIR/stdout.log"; exit ;;
  attach) exec tmux attach-session -t "=$SESSION" ;;
  stop)
    PID="$(cat "$DIR/trainer.pid")"
    CMD="$(ps -p "$PID" -o args=)"
    [[ "$CMD" == *supermimic/train.py* && "$CMD" == *"$DIR"* ]] || { echo 'Trainer PID is stale; refusing to signal' >&2; exit 1; }
    kill -TERM "$PID"; echo 'Safe stop requested; wait for train.exitcode'; exit ;;
  sync) exec "$SWANLAB_PYTHON" tools/ops/sync_swanlab.py --run_dir "$DIR" --project Foothold_with_SuperMimic --experiment_name "$(cat "$DIR/swanlab_experiment_name")" --mode "${SWANLAB_MODE:-online}" ;;
  start|fresh|resume|smoke|_worker|_sync) ;;
  *) echo 'Usage: tools/train/run_s54_curriculum_8h.sh {fresh|start|resume|smoke|status|attach|stop|sync} s54_RUN [extra train arguments]'; exit 2 ;;
esac
if [[ "$MODE" == _sync ]]; then
  exec 8>"$DIR/uploader.lock"; flock -n 8 || exit 1
  delay=5
  while true; do
    set +e
    "$SWANLAB_PYTHON" -u tools/ops/sync_swanlab.py --run_dir "$DIR" --project Foothold_with_SuperMimic --experiment_name "$(cat "$DIR/swanlab_experiment_name")" --mode "${SWANLAB_MODE:-online}" --follow --completion_marker
    status=$?
    set -e
    echo "$status" > "$DIR/upload.exitcode"
    if (( status==0 )) && [[ "${SWANLAB_MODE:-online}" == online ]]; then
      # Drain durable SDK caches too, including logs from a disconnected SDK session.
      for cache in "$DIR"/swanlab/run-*; do
        [[ -d "$cache" ]] || continue
        # Failed authentication creates a directory before any replayable
        # record file. Keep it for diagnosis; it contains no metrics to drain.
        records=("$cache"/run-*.swanlab)
        if [[ ! -f "${records[0]}" ]]; then
          echo "Skipping uninitialized SDK directory (no record file): $cache" >&2
          continue
        fi
        "$SWANLAB_PYTHON" -m swanlab sync "$cache" --project Foothold_with_SuperMimic --id "$(cat "$DIR/swanlab_run_id")" || status=$?
      done
      echo "$status" > "$DIR/final_sync.exitcode"
    fi
    (( status==0 )) && exit 0
    echo "SwanLab sync failed ($status); retry in $delay s; local metrics preserved" >&2
    sleep "$delay"; delay=$(( delay<300 ? delay*2 : 300 ))
  done
fi
if [[ "$MODE" == _worker ]]; then
  exec >> "$DIR/stdout.log" 2>&1
  finish_worker() {
    local status=$?
    trap - EXIT
    echo "$status" > "$DIR/train.exitcode.tmp"; mv "$DIR/train.exitcode.tmp" "$DIR/train.exitcode"
    exit "$status"
  }
  trap finish_worker EXIT
  exec 9>logs/.s54_curriculum_gpu.lock; flock -n 9 || { echo 'Another S54 curriculum job is active' >&2; exit 1; }
  # Check again inside the lock; never terminate other users' jobs.
  if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo 'GPU compute process exists; refusing shared-device contention' >&2; exit 1
  fi
  export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
  export WARP_CACHE_PATH="$REPO/.cache/warp" TORCHINDUCTOR_CACHE_DIR="$REPO/.cache/inductor"
  set +e
  "$S54_PYTHON" -u supermimic/train.py "$@" >> "$DIR/stdout.log" 2>&1 &
  child=$!; echo "$child" > "$DIR/trainer.pid"
  trap 'kill -TERM "$child"; wait "$child"' TERM INT
  wait "$child"; status=$?
  exit "$status"
fi
command -v tmux >/dev/null
[[ -x "$S54_PYTHON" ]] || { echo 'Set S54_PYTHON to Isaac Lab 3 Python' >&2; exit 1; }
if tmux has-session -t "=$SESSION" 2>/dev/null; then echo 'Session already exists' >&2; exit 1; fi
GPU_PROCS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
[[ -z "$GPU_PROCS" ]] || { echo 'GPU compute process exists; retry when idle' >&2; exit 1; }
FREE="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)"
MIN_FREE_MIB=16000
if [[ "$MODE" == fresh || -f "$DIR/train_profile.json" ]]; then MIN_FREE_MIB=21500; fi
(( FREE>=MIN_FREE_MIB )) || { echo "Need at least $MIN_FREE_MIB MiB free" >&2; exit 1; }
ARGS=(--goal_mode foot_trajectory --foot_task_config configs/s54/curriculum/s54_curriculum_task.json
 --curriculum_config configs/s54/curriculum/s54_curriculum_8h.json --foot_config configs/s54/foot_tracking/s54_foot_trajectory.json
 --goal_frames 10 --goal_dt .1 --num_envs 512 --buffer_steps 512 --warmup_steps 256
 --wm_window 8 --wm_batches 2 --wm_batch 256 --pi_batch 64 --horizon 8 --hist 2
 --iterations 1000000 --wm_only_iters 200 --lr .00005 --utd .4 --action_noise .03
 --critic_epochs 2 --critic_minibatches 2 --ckpt_every 1000 --log_every 20 --seed 54
 --amp bf16 --no_compile --viz none --log_dir "$DIR")
PROFILE_SOURCE=""
if [[ "$MODE" == fresh ]]; then
  PROFILE_SOURCE="configs/s54/curriculum/s54_curriculum_fresh_3090.json"
  PROFILE_TEXT="$(python3 tools/train/s54_training_profile.py "$PROFILE_SOURCE")"
  mapfile -t PROFILE_ARGS <<< "$PROFILE_TEXT"
  ARGS+=("${PROFILE_ARGS[@]}")
elif [[ "$MODE" == resume && -f "$DIR/train_profile.json" ]]; then
  PROFILE_TEXT="$(python3 tools/train/s54_training_profile.py "$DIR/train_profile.json" --for-resume --saved-args "$DIR/args.json")"
  mapfile -t PROFILE_ARGS <<< "$PROFILE_TEXT"
  ARGS+=("${PROFILE_ARGS[@]}")
fi
if [[ "$MODE" == resume ]]; then
  [[ -f "$DIR/latest.pt" ]] || { echo 'Missing latest.pt' >&2; exit 1; }
  ARGS+=(--resume "$DIR/latest.pt")
else
  [[ ! -e "$DIR" ]] || { echo 'Existing run protected; use resume' >&2; exit 1; }
  mkdir -p "$DIR"
  [[ -z "$PROFILE_SOURCE" ]] || cp "$PROFILE_SOURCE" "$DIR/train_profile.json"
  printf '%s_%s\n' "$STAMP" "$RUN" > "$DIR/swanlab_experiment_name"
fi
if [[ "$MODE" == smoke ]]; then
  ARGS+=(--num_envs 8 --warmup_steps 32 --buffer_steps 64 --wm_batch 32 --pi_batch 16
         --wm_only_iters 2 --wm_batches 1 --iterations 6 --log_every 1 --ckpt_every 3 --check_foot_gradients)
fi
[[ -f "$DIR/swanlab_experiment_name" ]] || printf '%s_%s\n' "$STAMP" "$RUN" > "$DIR/swanlab_experiment_name"
ARGS+=("$@")
printf -v JOB '%q ' bash tools/train/run_s54_curriculum_8h.sh _worker "$RUN" "${ARGS[@]}"
if [[ -f "$DIR/train.exitcode" ]]; then mv "$DIR/train.exitcode" "$DIR/train.previous.exitcode"; fi
date -u +%FT%TZ >> "$DIR/launch_times.txt"
printf '%s\n' "$JOB" >> "$DIR/launch_commands.txt"
tmux new-session -d -s "$SESSION" -c "$REPO" "$JOB"
# Independent tmux uploader survives trainer exit and SSH loss. Retry + durable local spool.
if [[ "${S54_NO_UPLOAD:-0}" != 1 ]] && ! tmux has-session -t "=${RUN}_upload" 2>/dev/null; then
  printf -v SYNC '%q ' bash tools/train/run_s54_curriculum_8h.sh _sync "$RUN"
  tmux new-session -d -s "${RUN}_upload" -c "$REPO" "$SYNC >> '$DIR/swanlab_sync.log' 2>&1"
fi
echo "Started $SESSION; attach: tmux attach -t $SESSION; logs: $DIR/stdout.log"
