#!/usr/bin/env bash
# 使用共同的 C9000 入口串行运行 M4.11 两组。
set -euo pipefail
export M4_ABLATION_REVISION=11
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_m48_ablation.sh" "$@"
