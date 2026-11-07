#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
repo="$PWD"
command="${1:-start}"
name="${2:-s54_stage2_curriculum_$(date +%Y%m%d_%H%M%S)}"
[[ "$name" =~ ^s54_stage2_curriculum_[A-Za-z0-9_]+$ ]] || { echo 'Invalid Stage2 session name' >&2; exit 2; }
run_dir="$repo/logs/$name"
start_uploader() {
    local upload_name="${name}_swanlab"
    if tmux has-session -t "=$upload_name" 2>/dev/null; then
      if [[ "$(tmux list-panes -t "${upload_name}:0" -F '#{pane_dead}')" == 1 ]]; then tmux kill-session -t "=$upload_name"; fi
    fi
    if ! tmux has-session -t "=$upload_name" 2>/dev/null; then
      tmux new-session -d -s "$upload_name" "bash -c 'cd \"$repo\"; set -o pipefail; .venv-swanlab/bin/python -u tools/ops/sync_s54_stage2.py --run-dir \"$run_dir\" 2>&1 | tee -a \"$run_dir/swanlab_sync.log\"'"
      tmux set-option -t "${upload_name}:0" remain-on-exit on
    fi
}
start_monitor() {
    local monitor_name="${name}_monitor"
    if tmux has-session -t "=$monitor_name" 2>/dev/null; then
      if [[ "$(tmux list-panes -t "${monitor_name}:0" -F '#{pane_dead}')" == 1 ]]; then tmux kill-session -t "=$monitor_name"; fi
    fi
    if ! tmux has-session -t "=$monitor_name" 2>/dev/null; then
      tmux new-session -d -s "$monitor_name" "cd \"$repo\" && python3 -u tools/ops/monitor_s54_stage2.py --run-dir \"$run_dir\" >> \"$run_dir/monitor.log\" 2>&1"
      tmux set-option -t "${monitor_name}:0" remain-on-exit on
    fi
}
case "$command" in
  start|resume)
    command -v tmux >/dev/null
    if tmux has-session -t "=$name" 2>/dev/null; then
      if [[ "$(tmux list-panes -t "${name}:0" -F '#{pane_dead}')" == "0" ]]; then echo "Already active: $name" >&2; exit 1; fi
      tmux kill-session -t "=$name"
    fi
    mkdir -p "$run_dir"
    if [[ "$command" == resume ]]; then
      [[ -f "$run_dir/latest.pt" && -f "$run_dir/profile.json" ]] || { echo 'No Stage2 checkpoint to resume' >&2; exit 1; }
      resume_arg='--resume'
    else
      [[ ! -e "$run_dir/latest.pt" ]] || { echo 'Existing run: use resume or a new timestamp' >&2; exit 1; }
      [[ -f "$run_dir/profile.json" ]] || cp configs/s54/stage2/curriculum_3090_8h.json "$run_dir/profile.json"
      resume_arg=''
    fi
    python3 - "$run_dir/status.json" "$command" <<'PY'
import json,sys,time
from pathlib import Path
p=Path(sys.argv[1]);old=json.loads(p.read_text()) if p.exists() else {}
old.update(status='RESUME_PENDING' if sys.argv[2]=='resume' else 'PREFLIGHT',requested_at=time.time());p.write_text(json.dumps(old,indent=2))
PY
    # Remain-on-exit keeps the last stdout and exit status visible inside tmux.
    tmux new-session -d -s "$name" "bash -c 'cd \"$repo\"; set -o pipefail; python3 -u tools/train/stage2_pipeline.py --run-dir \"$run_dir\" $resume_arg 2>&1 | tee -a \"$run_dir/stdout.log\"; code=\$?; echo \$code > \"$run_dir/launcher.exitcode\"; exit \$code'"
    tmux set-option -t "${name}:0" remain-on-exit on
    start_uploader
    start_monitor
    echo "Started $name; attach: tmux attach -t $name"
    echo "Log: $run_dir/stdout.log"
    ;;
  status)
    [[ ! -f "$run_dir/status.json" ]] || cat "$run_dir/status.json"
    [[ ! -f "$run_dir/swanlab/status.json" ]] || cat "$run_dir/swanlab/status.json"
    [[ ! -f "$run_dir/pipeline_error.json" ]] || cat "$run_dir/pipeline_error.json"
    tmux list-panes -t "${name}:0" -F 'tmux: #{session_name} dead=#{pane_dead} pid=#{pane_pid}' 2>/dev/null || true
    ;;
  attach) exec tmux attach -t "$name" ;;
  sync) start_uploader ;;
  stop)
    # Only the named session; uploader stays alive to flush local data.
    python3 - "$run_dir" <<'PY'
import os,signal,sys
from pathlib import Path
d=Path(sys.argv[1]);pid=int((d/'pipeline.pid').read_text())
cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
if b'stage2_pipeline.py' not in cmd or str(d).encode() not in cmd:raise RuntimeError('PID does not match this Stage2 supervisor')
os.kill(pid,signal.SIGTERM)
PY
    echo 'Safe stop requested; wait for status=STOPPED and latest.pt.'
    ;;
  *) echo "Usage: $0 {start|status|attach|stop|resume|sync} [s54_stage2_curriculum_TIMESTAMP]" >&2; exit 2 ;;
esac
