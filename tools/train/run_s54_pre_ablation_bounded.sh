#!/usr/bin/env bash
# Serial bounded checks only. No formal A-G training.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
MODE="${1:-start}"
STAMP="${2:-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
[[ "$STAMP" =~ ^[0-9]{8}_[0-9]{6}$ ]] || { echo 'Use YYYYMMDD_HHMMSS'; exit 2; }
SESSION="s54_preablation_$STAMP"
OUT="results/s54/pre_ablation_$STAMP"
S54_PYTHON="${S54_PYTHON:-$ROOT/../../IsaacLab_3_0/.venv/bin/python}"
case "$MODE" in
 attach) exec tmux attach -t "=$SESSION" ;;
 status) cat "$OUT/status.txt"; tail -n 6 "$OUT/worker.log"; exit ;;
 stop)
  child="$(cat "$OUT/active.pid")"
  cmd="$(ps -p "$child" -o args=)"
  [[ "$cmd" == *tools/*s54* || "$cmd" == *supermimic/train.py* ]] || { echo 'PID changed; refusing signal'; exit 1; }
  touch "$OUT/STOP_REQUESTED"
  # Signal launcher/evaluator; bounded launcher saves its child on interrupt.
  kill -INT "$child"; exit ;;
 start)
  [[ ! -e "$OUT" ]] || { echo 'Existing output protected'; exit 1; }
  [[ -x "$S54_PYTHON" ]] || { echo 'Set S54_PYTHON to Isaac Lab Python'; exit 1; }
  mkdir -p "$OUT"
  printf -v job '%q ' env "S54_PYTHON=$S54_PYTHON" bash tools/train/run_s54_pre_ablation_bounded.sh _worker "$STAMP"
  tmux new-session -d -s "$SESSION" -c "$ROOT" "$job"
  echo "Started $SESSION; log: $OUT/worker.log; attach: tmux attach -t $SESSION"
  exit ;;
 _worker) ;;
 *) echo "Usage: $0 {start|status|attach|stop} [YYYYMMDD_HHMMSS]"; exit 2 ;;
esac
exec >> "$OUT/worker.log" 2>&1
exec 9>logs/.s54_curriculum_gpu.lock
flock -n 9 || { echo 'Another S54 GPU check/train holds the lock'; exit 73; }
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || { echo 'GPU busy; retry after other jobs finish'; exit 73; }
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv > "$OUT/hardware.csv"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export WARP_CACHE_PATH="$ROOT/.cache/warp" MPLCONFIGDIR=/tmp/s54_matplotlib
cp configs/motion_prior/validation/pre_ablation_gates.json "$OUT/protocol.json"
trap 'printf "finished exit=%s\n" "$?" > "$OUT/status.txt"' EXIT
run_check() {
 local label="$1"; shift
 [[ ! -f "$OUT/STOP_REQUESTED" ]] || exit 130
 printf '%s\n' "$label" > "$OUT/status.txt"
 printf '%q ' "$@" >> "$OUT/commands.txt"; printf '\n' >> "$OUT/commands.txt"
 "$@" > "$OUT/$label.log" 2>&1 &
 local child=$!; printf '%s\n' "$child" > "$OUT/active.pid"
 local code=0; wait "$child" || code=$?
 printf '%s\n' "$code" > "$OUT/$label.exitcode"
 [[ "$code" == 0 ]] || { echo "$label failed with $code; outputs preserved"; exit "$code"; }
}
QA="$OUT/data_qa"
BOOT="s54_preablation_bootstrap_$STAMP"
WM="s54_preablation_wm_$STAMP"
TEACHER="s54_preablation_teacher_$STAMP"
run_check qa_dry "$S54_PYTHON" tools/data/prepare_s54_motion_prior.py --dry-run
run_check qa "$S54_PYTHON" tools/data/prepare_s54_motion_prior.py --out "$QA"
run_check bootstrap_dry "$S54_PYTHON" tools/train/run_s54_motion_prior.py --stage stage1 --manifest "$QA/manifest.json" --run "$BOOT" --dry-run
run_check bootstrap "$S54_PYTHON" tools/train/run_s54_motion_prior.py --stage stage1 --manifest "$QA/manifest.json" --run "$BOOT" --execute
run_check wm_dry "$S54_PYTHON" tools/train/run_s54_motion_prior.py --stage stage2 --group C --source "logs/$BOOT/last.pt" --profile configs/motion_prior/validation/calibration_bounded.json --run "$WM" --dry-run
run_check wm "$S54_PYTHON" tools/train/run_s54_motion_prior.py --stage stage2 --group C --source "logs/$BOOT/last.pt" --profile configs/motion_prior/validation/calibration_bounded.json --run "$WM" --execute
run_check standing timeout -k 15s 285s "$S54_PYTHON" tools/validation/evaluate_s54_motion_prior.py --checkpoint "logs/$WM/calibration_before.pt" --out "$OUT/standing" --durations 2 5 10 --seeds 54 55 56 --reset_noise --viz none
run_check wm_paired "$S54_PYTHON" tools/validation/compare_s54_wm_calibration.py --before "logs/$WM/calibration_before.pt" --after "logs/$WM/last.pt" --evaluation "$OUT/standing" --out "$OUT/wm_comparison.json"
run_check default_pd timeout -k 15s 285s "$S54_PYTHON" tools/validation/evaluate_s54_motion_prior.py --checkpoint "logs/$WM/calibration_before.pt" --controller default_pd --out "$OUT/default_pd" --durations 10 --seeds 54 55 56 --reset_noise --viz none
run_check teacher_dry "$S54_PYTHON" tools/train/run_s54_motion_prior.py --stage stage1 --manifest "$QA/manifest.json" --profile configs/motion_prior/stage1/stage1_bounded.json --run "$TEACHER" --dry-run
run_check teacher "$S54_PYTHON" tools/train/run_s54_motion_prior.py --stage stage1 --manifest "$QA/manifest.json" --profile configs/motion_prior/stage1/stage1_bounded.json --run "$TEACHER" --execute --swanlab-mode offline
run_check teacher_eval timeout -k 15s 285s "$S54_PYTHON" tools/validation/evaluate_s54_motion_prior.py --checkpoint "logs/$TEACHER/last.pt" --manifest "$QA/manifest.json" --split train --out "$OUT/teacher" --durations 5 10 --seeds 54 55 56 --start_seconds 0 30 60 --reset_noise --viz none
run_check readiness "$S54_PYTHON" tools/validation/summarize_s54_pre_ablation.py --root "$OUT" --manifest "$QA/manifest.json" --teacher-evaluation teacher --standing-evaluation standing
echo 'Bounded checks complete; summaries require gate review. No Teacher qualification or A-G launch is automatic.'
