#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
action=${1:-start}
case "$action" in
  start)
    name=${2:-s54_stage15_$(date +%Y%m%d_%H%M%S)}
    config=${3:-configs/motion_prior/stage15/lafan1_skill_transfer_3090.json}
    [[ -f "$config" ]] || { echo "Missing config: $config"; exit 2; }
    [[ "$name" =~ ^s54_stage15_[a-zA-Z0-9_]+$ ]] || { echo 'Invalid session name'; exit 2; }
    dest="logs/$name"
    [[ ! -e "$dest" ]] || { echo "Directory exists: $dest"; exit 2; }
    mkdir -p "$dest"
    cat > "$dest/run.sh" <<EOF
#!/usr/bin/env bash
set -uo pipefail
cd $(printf '%q' "$PWD")
python3 -u tools/train/run_s54_stage15_pipeline.py --config $(printf '%q' "$config") --out $(printf '%q' "$dest") 2>&1 | tee $(printf '%q' "$dest/stdout.log")
code=\${PIPESTATUS[0]}
printf '%s\\n' "\$code" > $(printf '%q' "$dest/process.exitcode")
exit "\$code"
EOF
    tmux new-session -d -s "$name" "bash $dest/run.sh"
    tmux set-window-option -t "$name:0" remain-on-exit on
    cat > "$dest/upload.sh" <<EOF
#!/usr/bin/env bash
set -uo pipefail
cd $(printf '%q' "$PWD")
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy
.venv-swanlab/bin/python -u -B tools/ops/sync_s54_stage15.py --run-dir $(printf '%q' "$dest") 2>&1 | tee $(printf '%q' "$dest/swanlab_upload.log")
code=\${PIPESTATUS[0]}
printf '%s\\n' "\$code" > $(printf '%q' "$dest/swanlab_upload.exitcode")
exit "\$code"
EOF
    tmux new-session -d -s "${name}_upload" "bash $dest/upload.sh"
    tmux set-window-option -t "${name}_upload:0" remain-on-exit on
    echo "Started $name; tmux attach -t $name; tail -f $dest/stdout.log"
    ;;
  status) cat "logs/${2:?session required}/status.json" ;;
  attach) exec tmux attach -t "${2:?session required}" ;;
  stop) tmux send-keys -t "${2:?session required}" C-c ;;
  *) echo "Usage: $0 start [new_session] | status/attach/stop session";exit 2 ;;
esac
