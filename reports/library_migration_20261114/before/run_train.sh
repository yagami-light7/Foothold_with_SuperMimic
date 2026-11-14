#!/usr/bin/env bash
# SuperMimic / KUAVO S54 launcher (host side, RTX 4060 8 GiB).
# Edit the variables below; no environment-variable overrides are required.
set -euo pipefail

# ============================================================================
# 1. Experiment configuration (edit here)
# ============================================================================
CONTAINER_NAME="${CONTAINER_NAME:-supermimic_s54}"
MOTION_DIR="data/retarget_data/reference_startup_fixed"
# Use clip stems, not BVH paths. CLIPS=() selects all accepted top-level CSVs.
# Examples: CLIPS=("dance1_subject1" "walk1_subject1") or CLIPS=("dance*")
CLIPS=("dance1_subject1")

# SwanLab runs on the host and follows this experiment's CSV; no image rebuild.
# Log in once: .venv-swanlab/bin/swanlab login
SWANLAB_ENABLED=true
SWANLAB_MODE="online"
SWANLAB_PROJECT="SuperMimic"
SWANLAB_WORKSPACE=""       # Empty: your default workspace.
SWANLAB_EXPERIMENT=""      # Empty: use the training run name.
SWANLAB_POLL_SECONDS=5

# Simulator / replay buffer
NUM_ENVS=2048
BUFFER_STEPS=1024
WARMUP_STEPS=256
UTD=0.4

# World Model
WM_WINDOW=8
WM_BATCH=1024
WM_BATCHES=16

# Policy / BPTT
PI_BATCH=256
HORIZON=32
GOAL_FRAMES=10
GOAL_MODE="motion"             # motion | foot_trajectory
GOAL_DT=""                     # empty: motion=0.02s, foot_trajectory=0.1s
FOOT_CONFIG=""                 # optional AlternatingConfig JSON path
FOOT_TASK_CONFIG=""            # FootTrackingCfg JSON (e.g. standing balance)
CHECK_FOOT_GRADIENTS=false
HIST=10

# Training schedule
ITERATIONS=10000
WM_ONLY_ITERS=1000
LR=0.0002
LR_DECAY_ITERS=40000
CKPT_EVERY=1000
LOG_EVERY=100
SEED=0
AMP="bf16"
NO_COMPILE=true

# RTX 4060 safety limits (change only if your hardware permits)
MAX_ENVS=2048
MIN_FREE_MIB=6000

# Smoke test overrides: same batch sizes, 30 iterations total.
SMOKE_ITERATIONS=30
SMOKE_WM_ONLY_ITERS=10
SMOKE_CKPT_EVERY=30
SMOKE_LOG_EVERY=10

# ============================================================================
# 2. Paths and command-line interface
# ============================================================================
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$REPO"

usage() {
  cat <<'EOF'
Usage:
  bash run_train.sh [--dry-run] smoke  [s54_smoke] [options]
  bash run_train.sh [--dry-run] train  [s54_dance1_v2] [options]
  bash run_train.sh [--dry-run] resume NEW_RUN logs/s54_OLD/last.pt [options]

Options:
  --clips NAME [NAME ...]  Select clips (quoted globs supported).
  --all-clips              Select all accepted CSVs in the motion directory.
  --iterations N           Total iteration target, including already completed iterations.
  --motion-dir DIR         Override the reference directory.
  --goal-mode MODE         motion (default) or foot_trajectory.
  --goal-dt SECONDS        Future sample spacing (foot default: 0.1).
  --foot-config FILE       Optional alternating footstep sampler JSON.
  --foot-task-config FILE  Independent foot MDP/standing task JSON.
  --check-foot-gradients   Audit pure foot Actor gradients and replay timing.
  --swanlab-project NAME   Cloud project (default: SuperMimic).
  --swanlab-workspace NAME Cloud workspace (default: your account).
  --swanlab-name NAME      Cloud experiment name (default: run name).
  --swanlab-mode MODE      online (default) or offline.
  --no-swanlab             Disable cloud tracking for this run.

Hyperparameters, default clips and cloud settings are configured at the top of run_train.sh.
Example: bash run_train.sh train s54_dance_walk --clips dance1_subject1 walk1_subject1
Smoke runs in the foreground; train/resume run in a Docker tmux session.
An existing stopped S54 container is started automatically after its mount is checked.
SwanLab automatically follows the same run on the host, including after this shell exits.
--dry-run validates clip names and prints commands without Docker or cloud access.
One-time login: .venv-swanlab/bin/swanlab login
EOF
}

DRY_RUN=false
ITERATIONS_OVERRIDE=""
POSITIONAL=()
need_value() {
  if (( $# < 2 )) || [[ -z "$2" || "$2" == --* ]]; then
    echo "Missing value for $1" >&2
    exit 2
  fi
}
while (( $# )); do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    --clips)
      shift
      CLIPS=()
      while (( $# )) && [[ "$1" != --* ]]; do
        CLIPS+=("$1"); shift
      done
      if (( ${#CLIPS[@]} == 0 )); then
        echo '--clips requires one or more clip names' >&2; exit 2
      fi
      ;;
    --all-clips) CLIPS=(); shift ;;
    --iterations) need_value "$@"; ITERATIONS_OVERRIDE="$2"; shift 2 ;;
    --motion-dir|--motion_dir) need_value "$@"; MOTION_DIR="$2"; shift 2 ;;
    --goal-mode|--goal_mode) need_value "$@"; GOAL_MODE="$2"; shift 2 ;;
    --goal-dt|--goal_dt) need_value "$@"; GOAL_DT="$2"; shift 2 ;;
    --foot-config|--foot_config) need_value "$@"; FOOT_CONFIG="$2"; shift 2 ;;
    --foot-task-config|--foot_task_config) need_value "$@"; FOOT_TASK_CONFIG="$2"; shift 2 ;;
    --check-foot-gradients|--check_foot_gradients) CHECK_FOOT_GRADIENTS=true; shift ;;
    --swanlab-project) need_value "$@"; SWANLAB_PROJECT="$2"; shift 2 ;;
    --swanlab-workspace) need_value "$@"; SWANLAB_WORKSPACE="$2"; shift 2 ;;
    --swanlab-name) need_value "$@"; SWANLAB_EXPERIMENT="$2"; shift 2 ;;
    --swanlab-mode) need_value "$@"; SWANLAB_MODE="$2"; shift 2 ;;
    --no-swanlab) SWANLAB_ENABLED=false; shift ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done
set -- "${POSITIONAL[@]}"

MODE="${1:-help}"
case "$MODE" in
  -h|--help|help) usage; exit 0 ;;
  smoke|train|resume) shift ;;
  *) usage >&2; exit 2 ;;
esac

case "$MODE" in
  smoke) RUN="${1:-s54_smoke}" ;;
  train) RUN="${1:-s54_dance1_v2}" ;;
  resume) RUN="${1:-}" ;;
esac

if [[ ! "$RUN" =~ ^s54_[A-Za-z0-9_.-]+$ ]]; then
  echo "Invalid run name: must start with s54_" >&2
  exit 2
fi
if [[ "$SWANLAB_MODE" != online && "$SWANLAB_MODE" != offline ]]; then
  echo 'SwanLab mode must be online or offline' >&2
  exit 2
fi
if [[ "$GOAL_MODE" != motion && "$GOAL_MODE" != foot_trajectory ]]; then
  echo 'Expected --goal-mode motion or foot_trajectory' >&2; exit 2
fi

if [[ "$CONTAINER_NAME" == "supermimic" ]]; then
  echo "Refusing to launch in the original G1 container" >&2
  exit 1
fi

RESUME_ARGS=()
if [[ "$MODE" == "resume" ]]; then
  if (( $# != 2 )) || [[ "$2" != logs/s54_*/last.pt ]] || [[ ! -f "$2" ]]; then
    echo "Usage: ... resume NEW_RUN logs/s54_OLD/last.pt (checkpoint must exist)" >&2
    exit 2
  fi
  RESUME_ARGS=(--resume "$2")
elif (( $# > 1 )); then
  usage >&2
  exit 2
fi

# Never overwrite an existing experiment directory.
if [[ -e "logs/$RUN" ]]; then
  echo "Run already exists: logs/$RUN (choose another run name)" >&2
  exit 1
fi

# Smoke test uses the same main batch sizes, but a shorter schedule.
RUN_ITERATIONS=$ITERATIONS
RUN_WM_ONLY=$WM_ONLY_ITERS
RUN_CKPT_EVERY=$CKPT_EVERY
RUN_LOG_EVERY=$LOG_EVERY
if [[ "$MODE" == "smoke" ]]; then
  RUN_ITERATIONS=$SMOKE_ITERATIONS
  RUN_WM_ONLY=$SMOKE_WM_ONLY_ITERS
  RUN_CKPT_EVERY=$SMOKE_CKPT_EVERY
  RUN_LOG_EVERY=$SMOKE_LOG_EVERY
fi
if [[ -n "$ITERATIONS_OVERRIDE" ]]; then
  RUN_ITERATIONS="$ITERATIONS_OVERRIDE"
fi

# Simple argument sanity checks before consuming GPU resources.
for value in "$NUM_ENVS" "$BUFFER_STEPS" "$WARMUP_STEPS" "$WM_WINDOW" "$WM_BATCH" \
             "$WM_BATCHES" "$PI_BATCH" "$HORIZON" "$GOAL_FRAMES" "$HIST" \
             "$RUN_ITERATIONS" "$RUN_CKPT_EVERY" "$RUN_LOG_EVERY" "$LR_DECAY_ITERS"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Expected a positive integer, got: $value" >&2
    exit 2
  fi
done
if [[ ! "$RUN_WM_ONLY" =~ ^[0-9]+$ ]] || \
   (( NUM_ENVS > MAX_ENVS || RUN_WM_ONLY >= RUN_ITERATIONS )); then
  echo "Invalid num_envs or world-model warmup schedule" >&2
  exit 2
fi

# Keep argument construction in one place for all modes.
ARGS=(
  --motion_dir "$MOTION_DIR"
  --num_envs "$NUM_ENVS"
  --iterations "$RUN_ITERATIONS"
  --buffer_steps "$BUFFER_STEPS"
  --warmup_steps "$WARMUP_STEPS"
  --utd "$UTD"
  --wm_window "$WM_WINDOW"
  --wm_batches "$WM_BATCHES"
  --wm_batch "$WM_BATCH"
  --pi_batch "$PI_BATCH"
  --horizon "$HORIZON"
  --goal_frames "$GOAL_FRAMES"
  --goal_mode "$GOAL_MODE"
  --hist "$HIST"
  --wm_only_iters "$RUN_WM_ONLY"
  --lr "$LR"
  --lr_decay_iters "$LR_DECAY_ITERS"
  --ckpt_every "$RUN_CKPT_EVERY"
  --log_every "$RUN_LOG_EVERY"
  --seed "$SEED"
  --amp "$AMP"
  "${RESUME_ARGS[@]}"
)
if [[ -n "$GOAL_DT" ]]; then ARGS+=(--goal_dt "$GOAL_DT"); fi
if [[ -n "$FOOT_CONFIG" ]]; then ARGS+=(--foot_config "$FOOT_CONFIG"); fi
if [[ -n "$FOOT_TASK_CONFIG" ]]; then ARGS+=(--foot_task_config "$FOOT_TASK_CONFIG"); fi
if [[ "$CHECK_FOOT_GRADIENTS" == true ]]; then ARGS+=(--check_foot_gradients); fi
if [[ "$NO_COMPILE" == true ]]; then
  ARGS+=(--no_compile)
fi
if [[ "$GOAL_MODE" == motion ]]; then
  SELECT_ARGS=(--motion_dir "$MOTION_DIR")
  if (( ${#CLIPS[@]} )); then
    ARGS+=(--clips "${CLIPS[@]}")
    SELECT_ARGS+=(--clips "${CLIPS[@]}")
  fi
  PYTHONDONTWRITEBYTECODE=1 python3 -m supermimic.reference_selection "${SELECT_ARGS[@]}"
else
  echo 'Independent foot task: Motion directory and clip selection are unused'
fi
TRAIN_CMD=(python supermimic/train.py --log_dir "logs/$RUN" "${ARGS[@]}")
JOB_CMD=(bash tools/run_s54_job.sh "logs/$RUN" "${TRAIN_CMD[@]}")
SWANLAB_PYTHON="${SWANLAB_PYTHON:-$REPO/.venv-swanlab/bin/python}"
if [[ ! -x "$SWANLAB_PYTHON" ]]; then
  SWANLAB_PYTHON="$(command -v python3)"
fi
SYNC_ARGS=(--run_dir "logs/$RUN" --project "$SWANLAB_PROJECT"
           --experiment_name "${SWANLAB_EXPERIMENT:-$RUN}" --mode "$SWANLAB_MODE"
           --poll_seconds "$SWANLAB_POLL_SECONDS")
if [[ -n "$SWANLAB_WORKSPACE" ]]; then
  SYNC_ARGS+=(--workspace "$SWANLAB_WORKSPACE")
fi

printf 'Mode: %s | Run: %s | Container: %s\n' "$MODE" "$RUN" "$CONTAINER_NAME"
printf 'Command: '
printf '%q ' "${TRAIN_CMD[@]}"
printf '\n'
if [[ "$SWANLAB_ENABLED" == true ]]; then
  printf 'SwanLab (%s): ' "$SWANLAB_MODE"
  printf '%q ' "$SWANLAB_PYTHON" tools/sync_swanlab.py "${SYNC_ARGS[@]}" --follow --completion_marker
  printf '\n'
else
  echo 'SwanLab: disabled'
fi
if [[ "$DRY_RUN" == true ]]; then
  exit 0
fi

# ============================================================================
# 3. Check container and available GPU memory
# ============================================================================
MOUNT="$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/workspace/supermimic"}}{{.Source}}{{end}}{{end}}' "$CONTAINER_NAME")"
if [[ "$MOUNT" != "$REPO" ]]; then
  echo "Wrong container mount: $MOUNT (expected $REPO)" >&2
  exit 1
fi
CONTAINER_STATE="$(docker inspect --format '{{.State.Status}}' "$CONTAINER_NAME")"
case "$CONTAINER_STATE" in
  running) ;;
  exited|created)
    echo "Starting existing container: $CONTAINER_NAME"
    docker start "$CONTAINER_NAME" > /dev/null
    if [[ "$(docker inspect --format '{{.State.Running}}' "$CONTAINER_NAME")" != true ]]; then
      echo "Container did not remain running: $CONTAINER_NAME" >&2
      exit 1
    fi
    ;;
  *)
    echo "Container $CONTAINER_NAME is $CONTAINER_STATE; inspect its state before training" >&2
    exit 1
    ;;
esac

FREE_MIB="$(docker exec "$CONTAINER_NAME" nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:]')"
if [[ ! "$FREE_MIB" =~ ^[0-9]+$ ]] || (( FREE_MIB < MIN_FREE_MIB )); then
  echo "Insufficient GPU memory: ${FREE_MIB} MiB free, need >= ${MIN_FREE_MIB} MiB" >&2
  exit 1
fi

# Avoid running two GPU training jobs on the 8 GiB card at once.
COMPUTE="$(docker exec "$CONTAINER_NAME" nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
if [[ "$COMPUTE" =~ [0-9] ]]; then
  echo "GPU compute process already running: $COMPUTE" >&2
  exit 1
fi
if [[ "$SWANLAB_ENABLED" == true ]]; then
  # Authenticate before starting GPU work; no experiment is created by --check.
  "$SWANLAB_PYTHON" tools/sync_swanlab.py --check --mode "$SWANLAB_MODE"
fi

# ============================================================================
# 4. Start training
# ============================================================================
mkdir -p logs
mkdir "logs/$RUN"
SYNC_PID=""
start_sync() {
  if [[ "$SWANLAB_ENABLED" == true ]]; then
    # A separate process session survives cleanup of the launching terminal's
    # process group. nohup alone only protects against SIGHUP.
    nohup setsid "$SWANLAB_PYTHON" -u tools/sync_swanlab.py "${SYNC_ARGS[@]}" --follow --completion_marker "$@" \
      > "logs/$RUN/swanlab_sync.log" 2>&1 < /dev/null &
    SYNC_PID=$!
    printf '%s\n' "$SYNC_PID" > "logs/$RUN/swanlab_sync.pid"
    echo "SwanLab sync log: logs/$RUN/swanlab_sync.log"
  fi
}
if [[ "$MODE" == "smoke" ]]; then
  start_sync
  set +e
  docker exec -w /workspace/supermimic "$CONTAINER_NAME" "${JOB_CMD[@]}"
  TRAIN_STATUS=$?
  set -e
  # Docker itself can fail before the container-side runner starts.
  if [[ ! -f "logs/$RUN/train.exitcode" ]]; then
    printf '%s\n' "$TRAIN_STATUS" > "logs/$RUN/train.exitcode"
  fi
  SYNC_STATUS=0
  if [[ -n "$SYNC_PID" ]]; then
    wait "$SYNC_PID" || SYNC_STATUS=$?
  fi
  if (( TRAIN_STATUS )); then exit "$TRAIN_STATUS"; fi
  docker exec "$CONTAINER_NAME" python tools/check_s54_run.py \
    --run_dir "logs/$RUN" --iteration "$RUN_ITERATIONS"
  if (( SYNC_STATUS )); then
    echo "SwanLab sync failed; see logs/$RUN/swanlab_sync.log" >&2
    exit "$SYNC_STATUS"
  fi
else
  printf -v QUOTED_JOB '%q ' "${JOB_CMD[@]}"
  docker exec -w /workspace/supermimic "$CONTAINER_NAME" tmux new-session \
    -d -s "$RUN" -c /workspace/supermimic "$QUOTED_JOB"
  start_sync --container "$CONTAINER_NAME" --session "$RUN"
  echo "Started tmux in $CONTAINER_NAME: $RUN"
  echo "Training log: logs/$RUN/stdout.log"
  echo "View with: docker exec -it $CONTAINER_NAME tmux attach -t $RUN"
fi
