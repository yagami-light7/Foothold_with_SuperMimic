#!/usr/bin/env bash
# Compatibility entrypoint. Settings live in tools/train/run_train.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "$ROOT/tools/train/run_train.sh" "$@"
