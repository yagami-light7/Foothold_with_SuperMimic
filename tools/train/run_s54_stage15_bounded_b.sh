#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
baseline=${1:?Usage: run_s54_stage15_bounded_b.sh baseline_run_dir [new_session]}
name=${2:-s54_stage15_$(date +%Y%m%d_%H%M%S)_bounded_b}
[[ "$name" =~ ^s54_stage15_[a-zA-Z0-9_]+$ ]] || { echo 'Invalid session name';exit 2; }
[[ -f "$baseline/diagnostic_masked_oracle/summary.json" ]] || { echo 'Completed baseline diagnosis required';exit 2; }
dest="logs/$name"
[[ ! -e "$dest" ]] || { echo 'Choose a fresh session';exit 2; }
mkdir -p "$dest"
cat > "$dest/run.sh" <<EOF
#!/usr/bin/env bash
set -uo pipefail
cd $(printf '%q' "$PWD")
python3 -u tools/train/run_s54_stage15_bounded_b.py --baseline $(printf '%q' "$baseline") --out $(printf '%q' "$dest") 2>&1 | tee $(printf '%q' "$dest/stdout.log")
code=\${PIPESTATUS[0]}
printf '%s\\n' "\$code" > $(printf '%q' "$dest/process.exitcode")
exit "\$code"
EOF
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
cat > "$dest/finalize.sh" <<EOF
#!/usr/bin/env bash
set -uo pipefail
cd $(printf '%q' "$PWD")
python3 -u tools/validation/finalize_s54_stage15.py --run-dir $(printf '%q' "$dest") 2>&1 | tee $(printf '%q' "$dest/finalize.log")
code=\${PIPESTATUS[0]}
printf '%s\\n' "\$code" > $(printf '%q' "$dest/finalize.exitcode")
exit "\$code"
EOF
tmux new-session -d -s "$name" "bash $dest/run.sh"
tmux set-window-option -t "$name:0" remain-on-exit on
tmux new-session -d -s "${name}_upload" "bash $dest/upload.sh"
tmux set-window-option -t "${name}_upload:0" remain-on-exit on
tmux new-session -d -s "${name}_finalize" "bash $dest/finalize.sh"
tmux set-window-option -t "${name}_finalize:0" remain-on-exit on
echo "Started $name; tmux attach -t $name; tail -f $dest/stdout.log"
