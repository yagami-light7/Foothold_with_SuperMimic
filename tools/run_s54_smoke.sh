#!/bin/bash
# Run INSIDE the independent S54 container. Stops on the first failed milestone.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f /workspace/supermimic/AGENTS.md && "$PWD" == /workspace/supermimic ]] || { echo 'Run inside the S54 container at /workspace/supermimic' >&2; exit 1; }
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
[[ "$FREE" =~ ^[0-9]+$ && "$FREE" -ge 5000 ]] || { echo 'Need at least 5000 MiB GPU free; wait for G1 jobs to finish' >&2; exit 1; }
[[ ! -e logs/s54_smoke ]] || { echo 'logs/s54_smoke exists; preserve it and choose a fresh run manually' >&2; exit 1; }
mkdir -p results/s54 logs/s54_smoke
python tools/inspect_s54_model.py | tee results/s54/inspect.log
python tools/prepare_s54_motion.py --expected_frames 3945 | tee results/s54/prepare_motion.log
python tools/prepare_s54_asset.py | tee results/s54/prepare_asset.log
OPENBLAS_NUM_THREADS=1 python tools/test_s54_cpu.py | tee results/s54/cpu_tests.log
python tools/mjcf_to_usd.py | tee results/s54/import.log
python tools/test_env_s54.py --num_envs 1 | tee results/s54/newton_1env.log
python tools/test_env_s54.py --num_envs 16 | tee results/s54/newton_16env.log
python supermimic/train.py --motion_dir library/motion_reference/kuavo_s54/retarget_data/reference --log_dir logs/s54_smoke \
  --num_envs 16 --iterations 30 --buffer_steps 64 --warmup_steps 32 \
  --wm_window 8 --wm_batches 1 --wm_batch 64 --pi_batch 32 --horizon 8 --wm_only_iters 10 \
  --ckpt_every 30 --log_every 10 --amp bf16 --no_compile 2>&1 | tee logs/s54_smoke/stdout.log
python tools/check_s54_run.py
python supermimic/evaluate.py --run_dir logs/s54_smoke --motion_dir library/motion_reference/kuavo_s54/retarget_data/reference \
  --ckpt logs/s54_smoke/ckpt_000030.pt --record logs/s54_smoke/traj_30.npz | tee logs/s54_smoke/evaluate.log
python tools/check_s54_run.py --require_eval
MUJOCO_GL=egl python tools/render.py logs/s54_smoke/traj_30.npz logs/s54_smoke/dance_30.mp4 \
  --clips dance1_subject1 --start 0 --seconds 5 --width 960 --height 540 --fps 30
