#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$REPO"
S54_PYTHON="${S54_PYTHON:-$REPO/../../IsaacLab_3_0/.venv/bin/python}"
[[ -x "$S54_PYTHON" ]] || { echo 'Set S54_PYTHON to the Isaac Lab Python executable' >&2; exit 1; }
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
exec "$S54_PYTHON" -u tools/train/run_s54_abc_short.py "$@" --python "$S54_PYTHON"
