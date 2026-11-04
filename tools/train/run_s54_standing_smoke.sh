#!/usr/bin/env bash
# Bounded 8-env / 40-iteration diagnostic using the unchanged SuperMimic loop.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$REPO"
RUN="${1:-s54_standing_smoke}"
if (( $# > 1 )) || [[ ! "$RUN" =~ ^s54_[A-Za-z0-9_.-]+$ ]]; then
  echo 'Usage: bash tools/train/run_s54_standing_smoke.sh [NEW_s54_RUN]' >&2; exit 2
fi
S54_PYTHON="${S54_PYTHON:-/mnt/sage_data/workspace/IsaacLab_3_0/.venv/bin/python}"
if [[ -e "logs/$RUN" ]]; then
  echo 'Choose a fresh run name; existing diagnostics are preserved' >&2; exit 2
fi
mkdir -p "logs/$RUN"
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  WARP_CACHE_PATH="$REPO/.cache/warp" TORCHINDUCTOR_CACHE_DIR="$REPO/.cache/inductor" \
  "$S54_PYTHON" -u supermimic/train.py \
  --goal_mode foot_trajectory --foot_task_config configs/s54/standing/s54_standing_balance.json \
  --foot_config configs/s54/foot_tracking/s54_foot_trajectory.json --goal_frames 10 --goal_dt 0.1 \
  --num_envs 8 --iterations 40 --buffer_steps 64 --warmup_steps 32 \
  --wm_window 8 --wm_batches 2 --wm_batch 32 --pi_batch 16 --horizon 8 --hist 2 \
  --wm_only_iters 20 --lr 0.00005 --action_noise 0.03 --critic_epochs 2 --critic_minibatches 2 \
  --ckpt_every 40 --log_every 5 --seed 54 --amp bf16 --no_compile \
  --check_foot_gradients --log_dir "logs/$RUN" --viz none 2>&1 | tee "logs/$RUN/stdout.log"
