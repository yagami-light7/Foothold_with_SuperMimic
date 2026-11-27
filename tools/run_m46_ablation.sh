#!/usr/bin/env bash
# 用容器内同一个 tmux 会话串行运行两组，宿主终端退出后仍可继续。
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$repo"
session=roban-m46-ablation
output=outputs/m4_6
groups=(w002 w005)

die() {
    printf '%s\n' "$*" >&2
    exit 1
}

group_done() {
    local dir="$output/$1"
    [[ -s "$dir/train.exitcode" && "$(<"$dir/train.exitcode")" == 0 &&
       -s "$dir/checkpoints/iter_5000.pt" && -s "$dir/checkpoints/best.pt" &&
       -s "$dir/evaluation/results.json" && -s "$dir/logs/console.log" ]] &&
        grep -Eq '"5000"[[:space:]]*:' "$dir/evaluation/results.json" &&
        grep -q '^M4 completed | iteration=5000 | ' "$dir/logs/console.log"
}

print_checkpoint() {
    local dir="$output/$1/checkpoints"
    if [[ -s "$dir/last.pt" ]]; then
        printf '%s\n' "$dir/last.pt"
    elif [[ -s "$dir/initial.pt" ]]; then
        printf '%s\n' "$dir/initial.pt"
    else
        printf '%s\n' '尚无有效检查点'
    fi
}

validate_output() {
    local group part dir
    for part in outputs "$output"; do
        [[ ! -L "$part" ]] || die "拒绝写入符号链接 $part"
    done
    for group in "${groups[@]}"; do
        dir="$output/$group"
        for part in "$dir" "$dir/"{checkpoints,config,logs,evaluation,train.exitcode,train.exitcode.tmp} \
            "$dir/logs/"{console.log,train.csv}; do
            [[ ! -L "$part" ]] || die "拒绝写入符号链接 $part"
        done
    done
}

train_command() {
    local group=$1
    trainer=(python -u supermimic/train.py --roban_config "configs/roban/m4/m4_6_${group}.json"
             --m4_reference /m4_reference.npz --device cuda:0 --m46_group "$group"
             --log_dir "$output/$group/checkpoints")
}

check_configs() {
    local group
    for group in "${groups[@]}"; do
        train_command "$group"
        "${trainer[@]}" --fresh_start --m46_check
    done
}

run_group() {
    local group=$1 mode=$2 dir log cp previous_lines rc action=--fresh_start
    local -a result
    dir="$output/$group"
    log="$dir/logs/console.log"
    cp="$dir/checkpoints/last.pt"
    mkdir -p "$dir/logs" "$dir/config"
    touch "$log"
    train_command "$group"
    if [[ "$mode" == resume && ! -e "$cp" && -s "$dir/checkpoints/initial.pt" ]] &&
        ! compgen -G "$dir/checkpoints/iter_*.pt" >/dev/null; then
        # 第一次周期保存前中断时，从同组未更新过的初始状态恢复。
        ln -- "$dir/checkpoints/initial.pt" "$cp"
    fi
    if [[ "$mode" == resume && -s "$cp" ]]; then
        trainer+=(--resume "$cp")
        action="--resume $cp"
    elif [[ -e "$dir/config/train.json" || -e "$dir/logs/train.csv" ]] ||
        compgen -G "$dir/checkpoints/*.pt" >/dev/null; then
        printf '[%s] %s 既有训练无法自动恢复，最近检查点=%s。\n' \
            "$(date -Is)" "$group" "$(print_checkpoint "$group")" | tee -a "$log" >&2
        return 1
    else
        trainer+=(--fresh_start)
    fi

    previous_lines=$(wc -l < "$log")
    printf '%s\n' running > "$dir/train.exitcode.tmp"
    mv -f -- "$dir/train.exitcode.tmp" "$dir/train.exitcode"
    printf '\n[%s] %s %s\n' "$(date -Is)" "$group" "$action" | tee -a "$log"
    if "${trainer[@]}" 2>&1 | tee -i -a "$log"; then
        result=("${PIPESTATUS[@]}")
    else
        result=("${PIPESTATUS[@]}")
    fi
    rc=${result[0]}
    printf '%s\n' "$rc" > "$dir/train.exitcode.tmp"
    mv -f -- "$dir/train.exitcode.tmp" "$dir/train.exitcode"
    if (( rc != 0 || result[1] != 0 || stop_requested != 0 )) || ! group_done "$group" ||
        ! awk -v first="$previous_lines" 'NR > first && /^M4 completed \| iteration=5000 \| / {done=1} END {exit !done}' "$log"; then
        printf '[%s] %s 未完成，训练退出码=%s，最近检查点=%s。原因见本日志末尾。\n' \
            "$(date -Is)" "$group" "$rc" "$(print_checkpoint "$group")" | tee -a "$log" >&2
        return 1
    fi
    printf '[%s] %s 完成，训练退出码=0。\n' "$(date -Is)" "$group" | tee -a "$log"
}

worker() {
    local group mode=$1
    stop_requested=0
    trap 'stop_requested=1' INT TERM
    validate_output
    for group in "${groups[@]}"; do
        (( stop_requested == 0 )) || return 130
        if group_done "$group"; then
            printf '%s 已完成，跳过。\n' "$group"
            continue
        fi
        run_group "$group" "$mode"
    done
}

supervise() {
    local command delay
    printf -v command '%q ' bash tools/run_m46_ablation.sh _worker "$1"
    tmux -f /dev/null -L "$session" new-session -d -s "$session" -e SHELL=/bin/bash "$command"
    # docker stop 与 tmux 中断都先通知训练入口保存，再结束调度。
    trap 'tmux -L "$session" send-keys -t "$session" C-c 2>/dev/null || true' INT TERM
    while tmux -L "$session" has-session -t "$session" 2>/dev/null; do
        sleep 1 &
        delay=$!
        wait "$delay" || true
    done
    group_done w002 && group_done w005
}

usage() {
    printf '%s\n' '用法：bash tools/run_m46_ablation.sh [start|resume|attach|status|check]' \
        'start 新训练；resume 仅继续未完成组；check 仅验证两组有效配置。'
}

mode=${1:-start}
case "$mode" in
    _check|_worker|_supervise)
        [[ ${M46_IN_CONTAINER:-} == 1 ]] || die '此入口仅在启动器创建的容器中使用。'
        case "$mode" in
            _check) check_configs ;;
            _worker) worker "$2" ;;
            _supervise) supervise "$2" ;;
        esac
        exit ;;
    start|resume|attach|status|check) [[ $# -le 1 ]] || die '不支持额外参数。' ;;
    -h|--help|help) usage; exit ;;
    *) usage >&2; exit 2 ;;
esac

if [[ "$mode" == attach || "$mode" == status ]]; then
    active_repo=$(docker container inspect --format '{{if .State.Running}}{{index .Config.Labels "org.supermimic.m46.repo"}}{{end}}' \
        "$session" 2>/dev/null) || active_repo=
    if [[ "$mode" == attach ]]; then
        [[ "$active_repo" == "$repo" ]] || die '当前仓库没有运行中的 M4.6 容器。'
        exec docker exec -it "$session" tmux -L "$session" attach-session -t "$session"
    fi
    if [[ "$active_repo" == "$repo" ]]; then
        printf '容器 %s 正在运行。\n' "$session"
    else
        printf '当前仓库没有运行中的 M4.6 容器。\n'
    fi
    for group in "${groups[@]}"; do
        status=未开始
        if group_done "$group"; then
            status=已完成
        elif [[ -r "$output/$group/train.exitcode" ]]; then
            rc=$(<"$output/$group/train.exitcode")
            status="未完成，训练退出码=$rc"
            if [[ "$rc" == running ]]; then
                status=运行中
                [[ "$active_repo" == "$repo" ]] || status=容器中断
            fi
        elif [[ -e "$output/$group/logs/console.log" ]]; then
            status=未完成
        fi
        printf '%s %s，最近检查点 %s\n' "$group" "$status" "$(print_checkpoint "$group")"
        if [[ -f "$output/$group/logs/console.log" ]]; then
            tail -n 3 -- "$output/$group/logs/console.log"
        fi
    done
    exit 0
fi

if [[ "$mode" != check ]]; then
    validate_output
    if docker container inspect "$session" >/dev/null 2>&1; then
        die "容器名称 $session 已在使用，请先查看 status。"
    fi
    if [[ "$mode" == start && ( -e "$output/w002" || -e "$output/w005" ) ]]; then
        die '已存在 M4.6 组目录。请运行 resume 续接，以保护已有文件。'
    fi
    if [[ "$mode" == resume && ! -d "$output/w002" && ! -d "$output/w005" ]]; then
        die '尚无 M4.6 训练可续接。新训练请使用 start。'
    fi
    if [[ "$mode" == resume ]] && group_done w002 && group_done w005; then
        printf '%s\n' '两组均已完整结束。'
        exit 0
    fi
fi

# 沿用 run_roban.sh 的镜像、挂载、缓存和只读参考文件。
reference="$(dirname "$repo")/Humanoid_Online_Motion_Generator/outputs/m3_7_1/forward/F2/reference.npz"
[[ -r "$reference" ]] || die "缺少原始 F2 文件 $reference"
reference=$(realpath -- "$reference")
image=${DOCKER_IMAGE:-supermimic:latest}
commit=$(git rev-parse HEAD)
git_status=$(GIT_OPTIONAL_LOCKS=0 git status --porcelain=v1 --untracked-files=normal)
runtime=(--rm --network=none --shm-size=2g --user "$(id -u):$(id -g)"
    --mount "type=bind,source=$repo,target=/workspace/supermimic"
    --mount "type=bind,source=$repo/library,target=/workspace/supermimic/library,readonly"
    --mount "type=bind,source=$reference,target=/m4_reference.npz,readonly"
    --workdir /workspace/supermimic --env M46_IN_CONTAINER=1 --env "S54_SOURCE_GIT_SHA=$commit"
    --env "M46_SOURCE_GIT_STATUS=$git_status" --env PYTHONDONTWRITEBYTECODE=1
    --env WARP_CACHE_PATH=/workspace/supermimic/.cache/roban_warp --env MPLCONFIGDIR=/tmp/roban_mpl
    --env "MUJOCO_GL=${MUJOCO_GL:-egl}" --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics)
docker run "${runtime[@]}" "$image" bash tools/run_m46_ablation.sh _check
[[ "$mode" != check ]] || exit 0
docker run "${runtime[@]}" --detach --init --stop-timeout 90 --gpus all --name "$session" \
    --label "org.supermimic.m46.repo=$repo" "$image" bash tools/run_m46_ablation.sh _supervise "$mode"
printf '已启动 %s，查看进度请运行 bash tools/run_m46_ablation.sh attach\n' "$session"
