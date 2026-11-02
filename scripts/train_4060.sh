#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/s54-uv-cache}"
export UV_LINK_MODE=copy
LOGGER_PYTHON="$TASK_ROOT/.venv-swanlab/bin/python"
if [[ ! -x "$LOGGER_PYTHON" ]]; then
  uv venv "$TASK_ROOT/.venv-swanlab" --python /mnt/sage_data/workspace/IsaacLab_3_0/.venv/bin/python
fi
if ! "$LOGGER_PYTHON" -c 'from importlib.metadata import version; assert version("swanlab") == "0.10.3"' >/dev/null 2>&1; then
  uv pip install --python "$LOGGER_PYTHON" 'swanlab==0.10.3'
fi
exec "$LOGGER_PYTHON" -u scripts/train_4060.py "$@"
