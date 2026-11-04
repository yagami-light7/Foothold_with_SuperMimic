#!/usr/bin/env bash
# Container-side runner: preserve the trainer/tee exit code for the host uploader.
set -euo pipefail
RUN_DIR="${1:?Expected logs/s54_RUN and a training command}"
shift
[[ "$RUN_DIR" =~ ^logs/s54_[A-Za-z0-9_.-]+$ ]] || exit 2
(( $# > 0 )) || exit 2
mkdir -p "$RUN_DIR"
finish() {
  local status=$?
  trap - EXIT
  printf '%s\n' "$status" > "$RUN_DIR/train.exitcode.tmp"
  mv "$RUN_DIR/train.exitcode.tmp" "$RUN_DIR/train.exitcode"
  exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"$@" 2>&1 | tee -a "$RUN_DIR/stdout.log"
