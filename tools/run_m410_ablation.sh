#!/usr/bin/env bash
# 默认查看状态。确认正式实验后通过 start 启动三组串行训练。
if (( $# == 0 )); then
    set -- status
fi
M4_ABLATION_REVISION=10 exec bash "$(dirname "${BASH_SOURCE[0]}")/run_m48_ablation.sh" "$@"
