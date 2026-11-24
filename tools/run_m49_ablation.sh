#!/usr/bin/env bash
# 复用已验证的容器、tmux、断点恢复和防休眠调度。
M4_ABLATION_REVISION=9 exec bash "$(dirname "${BASH_SOURCE[0]}")/run_m48_ablation.sh" "$@"
