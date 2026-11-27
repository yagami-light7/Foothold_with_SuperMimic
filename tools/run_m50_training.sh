#!/usr/bin/env bash
# 通过验收后运行 Warm Start，完成训练后复评。
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$repo"
out=outputs/m5_0
session=roban-m50-training
launcher=tools/run_m50_training.sh
trainer=(python -u supermimic/train.py --roban_config configs/roban/m5/m5_0.json
         --device cuda:0 --m50_library configs/roban/m5_0_motion_library.json
         --m50_reference_root /reference_data --log_dir "$out/checkpoints")

die() { printf '%s\n' "$*" >&2; exit 1; }

preflight() {
    python - <<'PY'
import json,numpy as np
from pathlib import Path
root=Path('outputs/m5_0')
r=json.loads((root/'preflight.json').read_text())
assert [x['envs'] for x in r['smokes']]==[8,32,256] and all(x['epa']==0 for x in r['smokes'])
assert {'F1','F2','F3','H1','H2','H3','H4'}==set(r['cpu']['references'])
assert r['cpu']['reward_delta_max']==r['gpu']['reward_delta_max']==0
assert set(r['benchmark'])=={'legacy_f2','packed_f2','mixed'}
with np.load(root/'zero_shot.npz',allow_pickle=False) as z:
    e=json.loads(z['metadata'].item())['nominal']
assert set(e)=={'F1','F2','F3'} and all(x['attempts']==3 for x in e.values()) and e['F2']['complete']==3
PY
}

worker() {
    local mode=$1 rc=0
    mkdir -p "$out/logs"
    exec > >(tee -a "$out/logs/console.log") 2>&1
    if [[ ! -s "$out/train.exitcode" || $(<"$out/train.exitcode") != 0 ]]; then
        if [[ "$mode" == resume && -s "$out/checkpoints/last.pt" ]]; then
            trainer+=(--resume "$out/checkpoints/last.pt")
        fi
        printf '%s\n' running > "$out/train.exitcode"
        printf 'M5.0 %s %s\n' "$mode" "$(date -Is)"
        "${trainer[@]}" || rc=$?
        printf '%s\n' "$rc" > "$out/train.exitcode"
        (( rc == 0 )) || return "$rc"
    fi
    python -u -m supermimic.integration.m50_evaluation final --reference-root /reference_data
    python -u -m supermimic.integration.m50_visuals --reference-root /reference_data
}

case "${1:-start}" in
    _check)
        [[ "${M50_IN_CONTAINER:-}" == 1 ]] || die '仅容器内部使用。'
        preflight
        "${trainer[@]}" --m50_check >/dev/null
        python -c 'import torch; print("CUDA",torch.cuda.get_device_name())'
        tmux -V
        exit ;;
    _worker)
        [[ "${M50_IN_CONTAINER:-}" == 1 ]] || die '仅容器内部使用。'
        worker "$2"
        exit ;;
    _supervise)
        [[ "${M50_IN_CONTAINER:-}" == 1 ]] || die '仅容器内部使用。'
        printf -v command '%q ' bash "$launcher" _worker "$2"
        tmux -f /dev/null -L "$session" new-session -d -s "$session" -e SHELL=/bin/bash "$command"
        trap 'tmux -L "$session" send-keys -t "$session" C-c 2>/dev/null || true' TERM INT
        while tmux -L "$session" has-session -t "$session" 2>/dev/null; do
            sleep 1 &
            wait "$!" || true
        done
        test -s "$out/cuda_evaluation.npz" && test -s "$out/multi_reference.mp4"
        exit ;;
esac

mode=${1:-start}
case "$mode" in
    status|attach|stop)
        active="$(docker inspect --format '{{if .State.Running}}{{ index .Config.Labels "org.supermimic.m50.repo" }}{{end}}' "$session" 2>/dev/null)" || active=
        if [[ "$mode" == status ]]; then
            printf 'M5.0 容器 %s\n' "$([[ "$active" == "$repo" ]] && printf running || printf inactive)"
            [[ ! -s "$out/train.exitcode" ]] || printf '训练退出标记 %s\n' "$(<"$out/train.exitcode")"
            [[ ! -f "$out/logs/console.log" ]] || tail -n 7 "$out/logs/console.log"
            exit 0
        fi
        [[ "$active" == "$repo" ]] || die '未找到本仓库运行中的 M5.0 容器。'
        if [[ "$mode" == attach ]]; then
            exec docker exec -it "$session" tmux -L "$session" attach-session -t "$session"
        fi
        exec docker stop --time 90 "$session" ;;
    start|resume|preflight|check|evaluate) ;;
    *) die "用法 bash $launcher [preflight|check|start|resume|status|attach|stop|evaluate]" ;;
esac

mkdir -p "$out" "$repo/.cache/roban_warp"
reference="$(realpath -- "${M50_REFERENCE_ROOT:-../Humanoid_Online_Motion_Generator/outputs}")"
[[ -f "$reference/m3_7_1/forward/F2/reference.npz" ]] || die '找不到原始直走参考。'
image=${DOCKER_IMAGE:-supermimic:latest}
runtime=(--rm --init --network none --shm-size=2g --gpus all --user "$(id -u):$(id -g)"
    --mount "type=bind,source=$repo,target=/workspace/supermimic,readonly"
    --mount "type=bind,source=$repo/$out,target=/workspace/supermimic/$out"
    --mount "type=bind,source=$repo/.cache/roban_warp,target=/workspace/supermimic/.cache/roban_warp"
    --mount "type=bind,source=$reference,target=/reference_data,readonly"
    --mount "type=bind,source=/usr/share/fonts/opentype/noto,target=/usr/share/fonts/opentype/noto,readonly"
    --workdir /workspace/supermimic --env PYTHONDONTWRITEBYTECODE=1 --env M50_IN_CONTAINER=1
    --env WARP_CACHE_PATH=/workspace/supermimic/.cache/roban_warp --env MPLCONFIGDIR=/tmp/roban_mpl
    --env MUJOCO_GL=egl --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
    --env "S54_SOURCE_GIT_SHA=$(git rev-parse HEAD)"
    --env "M50_SOURCE_GIT_STATUS=$(GIT_OPTIONAL_LOCKS=0 git status --porcelain)")

active="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)"
[[ -z "$active" ]] || die "GPU 正被计算任务使用
$active"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
inhibit=(systemd-inhibit --what=sleep:idle:handle-lid-switch --mode=block --who="$session" --why='Roban M5.0 CUDA')
if ! "${inhibit[@]}" true >/dev/null 2>&1; then
    inhibit=()
fi
if [[ "$mode" == preflight ]]; then
    exec "${inhibit[@]}" docker run "${runtime[@]}" "$image" bash -ec '
        if [[ ! -s outputs/m5_0/zero_shot.npz ]]; then
            python -u -m supermimic.integration.m50_evaluation zero-shot --reference-root /reference_data
        fi
        python -u tools/test_m50.py --root /reference_data --cuda --save
        python -u tools/test_m50.py --root /reference_data --benchmark'
fi
if [[ "$mode" == evaluate ]]; then
    exec "${inhibit[@]}" docker run "${runtime[@]}" "$image" bash -ec '
        python -u -m supermimic.integration.m50_evaluation final --reference-root /reference_data
        python -u -m supermimic.integration.m50_visuals --reference-root /reference_data'
fi
docker run "${runtime[@]}" "$image" bash "$launcher" _check
[[ "$mode" != check ]] || exit 0
if [[ "$mode" == start ]]; then
    [[ ! -d "$out/checkpoints" && ! -e "$out/train.exitcode" ]] || die '已有正式训练，使用 resume。'
else
    [[ -s "$out/checkpoints/last.pt" ]] || die '找不到 Last 检查点。'
fi
cid="$(docker create "${runtime[@]}" --name "$session" --stop-timeout 90 \
    --label "org.supermimic.m50.repo=$repo" "$image" bash "$launcher" _supervise "$mode")"
nohup setsid "${inhibit[@]}" docker start --attach "$cid" </dev/null >/dev/null 2>&1 &
for attempt in {1..30}; do
    if docker exec "$cid" tmux -L "$session" has-session -t "$session" 2>/dev/null; then
        printf 'M5.0 tmux 已启动 %s\n' "$session"
        exit 0
    fi
    sleep .25
done
die 'M5.0 tmux 未确认启动，请查 status。'
