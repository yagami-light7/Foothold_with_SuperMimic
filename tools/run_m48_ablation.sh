#!/usr/bin/env bash
# 容器内 tmux 串行运行实验组，宿主休眠锁随容器生命周期释放。
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$repo"
revision=${M4_ABLATION_REVISION:-8}
stride=500
best=best.pt
case "$revision" in
    8) budget=5000; groups=(a_startup_only b_smoothness_only c_startup_smoothness) ;;
    9) budget=10000; groups=(a_long_baseline b_original_prior c_soft_prior) ;;
    10) budget=10000; groups=(a_root_only b_huber_prior_only c_root_huber) ;;
    11) budget=2000; stride=250; best=best_foot.pt; groups=(a_continue_baseline b_hand_huber) ;;
    *) printf '%s\n' '无法识别消融版本。' >&2; exit 2 ;;
esac
phase="M4.$revision"
session="roban-m4$revision-ablation"
output="outputs/m4_$revision"
launcher="tools/run_m4${revision}_ablation.sh"
repo_label="org.supermimic.m4${revision}.repo"

die() {
    printf '%s\n' "$*" >&2
    exit 1
}

group_done() {
    local dir="$output/$1"
    if [[ "$revision" == 11 && ! -s "$dir/checkpoints/critic_warmup.pt" ]]; then
        return 1
    fi
    [[ -s "$dir/train.exitcode" && "$(<"$dir/train.exitcode")" == 0 &&
       -s "$dir/checkpoints/initial.pt" && -s "$dir/checkpoints/last.pt" &&
       -s "$dir/checkpoints/$best" && -s "$dir/checkpoints/iter_$budget.pt" &&
       -s "$dir/evaluation/results.json" ]]
}

resume_checkpoint() {
    local dir="$output/$1/checkpoints" checkpoint iteration
    if [[ -s "$dir/last.pt" ]]; then
        printf '%s\n' "$dir/last.pt"
        return
    fi
    for ((iteration=budget; iteration>=stride; iteration-=stride)); do
        if [[ "$revision" == 11 ]]; then
            printf -v checkpoint '%s/iter_%04d.pt' "$dir" "$iteration"
        else
            checkpoint="$dir/iter_$iteration.pt"
        fi
        if [[ -s "$checkpoint" ]]; then
            printf '%s\n' "$checkpoint"
            return
        fi
    done
    if [[ "$revision" == 11 && -s "$dir/critic_warmup.pt" ]]; then
        printf '%s\n' "$dir/critic_warmup.pt"
        return
    fi
    if [[ -s "$dir/initial.pt" ]]; then
        printf '%s\n' "$dir/initial.pt"
    fi
}

validate_output() {
    local group dir part
    for part in outputs "$output"; do
        [[ ! -L "$part" ]] || die "输出路径是符号链接：$part"
    done
    for group in "${groups[@]}"; do
        dir="$output/$group"
        for part in "$dir" "$dir/"{checkpoints,config,logs,evaluation,train.exitcode,train.exitcode.tmp,dynamics_fault.npz} \
            "$dir/config/"{train.json,train.tmp} "$dir/logs/"{console.log,train.csv} \
            "$dir/evaluation/results.json" "$dir/checkpoints/"{*.pt,*.tmp}; do
            [[ ! -L "$part" ]] || die "输出路径是符号链接：$part"
        done
    done
}

trainer_args() {
    local group=$1 key=${1%%_*}
    trainer=(python -u supermimic/train.py --roban_config "configs/roban/m4/m4_${revision}_$key.json" "--m4${revision}_group" "$key"
             --m4_reference /m4_reference.npz --device cuda:0 --log_dir "$output/$group/checkpoints")
}

check_configs() {
    local group
    for group in "${groups[@]}"; do
        trainer_args "$group"
        if [[ "$revision" != 11 ]]; then
            trainer+=(--fresh_start)
        fi
        "${trainer[@]}" "--m4${revision}_check" >/dev/null
        printf '%s 配置通过。\n' "$group"
    done
    python -c 'import torch; print("CUDA 检查通过", torch.cuda.get_device_name(0), torch.ones(1, device="cuda:0").item())'
    tmux -V
}

run_group() {
    local group=$1 mode=$2 dir log checkpoint rc action=fresh_start
    local -a result
    dir="$output/$group"
    log="$dir/logs/console.log"
    checkpoint=$(resume_checkpoint "$group")
    trainer_args "$group"
    if [[ "$mode" == resume && -n "$checkpoint" ]]; then
        trainer+=(--resume "$checkpoint")
        action="resume $checkpoint"
    elif [[ -e "$dir/config/train.json" || -e "$dir/logs/train.csv" ]] ||
        compgen -G "$dir/checkpoints/*.pt" >/dev/null; then
        printf '%s 已有训练文件，但没有可用的同组恢复点。\n' "$group" >&2
        return 1
    elif [[ "$revision" == 11 ]]; then
        action='从 M4.10-C9000 初始化'
    else
        trainer+=(--fresh_start)
    fi
    mkdir -p "$dir/logs"
    printf '%s\n' running > "$dir/train.exitcode.tmp"
    mv -f -- "$dir/train.exitcode.tmp" "$dir/train.exitcode"
    printf '\n[%s] %s %s | 宿主休眠锁=%s\n' "$(date -Is)" "$group" "$action" "${M4_ABLATION_INHIBIT:-${M48_INHIBIT:-不可用}}" | tee -a "$log"
    if "${trainer[@]}" 2>&1 | tee -i -a "$log"; then
        result=("${PIPESTATUS[@]}")
    else
        result=("${PIPESTATUS[@]}")
    fi
    rc=${result[0]}
    if (( stop_requested )); then
        rc=75
    elif (( result[1] )); then
        rc=${result[1]}
    fi
    printf '%s\n' "$rc" > "$dir/train.exitcode.tmp"
    mv -f -- "$dir/train.exitcode.tmp" "$dir/train.exitcode"
    if ! group_done "$group"; then
        printf '%s 未完成，退出码=%s。后续组已停止。\n' "$group" "$rc" | tee -a "$log" >&2
        return 1
    fi
    printf '%s 已完成 %s 轮。\n' "$group" "$budget" | tee -a "$log"
}

worker() {
    local group
    stop_requested=0
    trap 'stop_requested=1' INT TERM
    validate_output
    for group in "${groups[@]}"; do
        (( stop_requested == 0 )) || return 75
        if ! group_done "$group"; then
            run_group "$group" "$1"
        else
            printf '%s 已完成，跳过。\n' "$group"
        fi
    done
}

supervise() {
    local command delay
    printf -v command '%q ' bash "$launcher" _worker "$1"
    tmux -f /dev/null -L "$session" new-session -d -s "$session" -e SHELL=/bin/bash "$command"
    # 容器收到停止信号后，给训练入口时间保存安全检查点。
    trap 'tmux -L "$session" send-keys -t "$session" C-c 2>/dev/null || true' INT TERM
    while tmux -L "$session" has-session -t "$session" 2>/dev/null; do
        sleep 1 &
        delay=$!
        wait "$delay" || true
    done
    for group in "${groups[@]}"; do
        group_done "$group" || return 1
    done
}

mode=${1:-start}
[[ $# -le 1 || ( $# == 2 && "$mode" == _* ) ]] || die '不支持额外参数，请运行 --help。'
case "$mode" in
    _check|_worker|_supervise)
        [[ ${M4_ABLATION_IN_CONTAINER:-} == "$revision" || ( "$revision" == 8 && ${M48_IN_CONTAINER:-} == 1 ) ]] ||
            die "此入口仅在 $phase 容器中使用。"
        case "$mode" in
            _check) check_configs ;;
            _worker) worker "$2" ;;
            _supervise) supervise "$2" ;;
        esac
        exit ;;
    start|resume|attach|status|stop|check|smoke) ;;
    -h|--help|help)
        printf '%s\n' "用法：bash $launcher [start|resume|attach|status|stop|check|smoke]" \
            'start 从头训练；resume 从同组检查点续训；stop 保存后停止；check 和 smoke 使用独立检查入口。'
        exit ;;
    *) die '未知命令，请运行 --help。' ;;
esac

if [[ "$mode" == attach || "$mode" == status || "$mode" == stop ]]; then
    active_repo=$(docker container inspect --format "{{if .State.Running}}{{index .Config.Labels \"$repo_label\"}}{{end}}" \
        "$session" 2>/dev/null) || active_repo=
    if [[ "$mode" != status ]]; then
        [[ "$active_repo" == "$repo" ]] || die "当前仓库没有运行中的 $phase 容器。"
        if [[ "$mode" == attach ]]; then
            exec docker exec -it "$session" tmux -L "$session" attach-session -t "$session"
        fi
        exec docker stop --time 90 "$session"
    fi
    printf '运行中的 %s 容器：%s\n' "$phase" "$(if [[ "$active_repo" == "$repo" ]]; then printf '%s' "$session"; else printf '无'; fi)"
    for group in "${groups[@]}"; do
        dir="$output/$group"
        status=未开始
        if group_done "$group"; then
            status=已完成
        elif [[ -r "$dir/train.exitcode" ]]; then
            rc=$(<"$dir/train.exitcode")
            status="未完成，退出码=$rc"
            if [[ "$rc" == running ]]; then
                status=容器中断
                [[ "$active_repo" != "$repo" ]] || status=运行中
            fi
        fi
        checkpoint=$(resume_checkpoint "$group")
        printf '%s %s，恢复点 %s\n' "$group" "$status" "${checkpoint:-无}"
        if [[ -f "$dir/logs/console.log" ]]; then
            tail -n 3 -- "$dir/logs/console.log"
        fi
    done
    exit
fi

reference="$(dirname "$repo")/Humanoid_Online_Motion_Generator/outputs/m3_7_1/forward/F2/reference.npz"
[[ -r "$reference" ]] || die "缺少原始 F2 参考：$reference"
reference=$(realpath -- "$reference")
image=${DOCKER_IMAGE:-supermimic:latest}
runtime=(--rm --network=none --shm-size=2g --gpus all --user "$(id -u):$(id -g)"
    --mount "type=bind,source=$repo,target=/workspace/supermimic"
    --mount "type=bind,source=$repo/library,target=/workspace/supermimic/library,readonly"
    --mount "type=bind,source=$reference,target=/m4_reference.npz,readonly"
    --workdir /workspace/supermimic --env "M4_ABLATION_IN_CONTAINER=$revision" --env PYTHONDONTWRITEBYTECODE=1
    --env WARP_CACHE_PATH=/workspace/supermimic/.cache/roban_warp --env MPLCONFIGDIR=/tmp/roban_mpl
    --env "MUJOCO_GL=${MUJOCO_GL:-egl}" --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics)
if [[ "$revision" -ge 9 ]]; then
    git_commit=$(git rev-parse HEAD)
    git_worktree=$(GIT_OPTIONAL_LOCKS=0 git status --porcelain=v1 --untracked-files=normal)
    runtime+=(--env "S54_SOURCE_GIT_SHA=$git_commit" --env "M4${revision}_SOURCE_GIT_STATUS=$git_worktree")
fi
if [[ "$mode" == check ]]; then
    exec docker run "${runtime[@]}" "$image" bash "$launcher" _check
fi

inhibit=(systemd-inhibit --what=sleep:idle:handle-lid-switch --mode=block --who="$session" --why="Roban $phase CUDA")
inhibit_status=已启用
if ! command -v systemd-inhibit >/dev/null || ! "${inhibit[@]}" true >/dev/null 2>&1; then
    inhibit=()
    inhibit_status=不可用
fi
if [[ "$mode" == smoke ]]; then
    printf '宿主休眠锁=%s\n' "$inhibit_status"
    if [[ "$revision" == 11 ]]; then
        exec "${inhibit[@]}" docker run "${runtime[@]}" "$image" python -u tools/test_m411_hand.py --smoke
    fi
    if [[ "$revision" == 10 ]]; then
        exec "${inhibit[@]}" docker run "${runtime[@]}" "$image" python -u tools/test_m410_rewards.py --smoke
    fi
    exec "${inhibit[@]}" docker run "${runtime[@]}" "$image" python -u tools/test_m47_controls.py --smoke "--m4$revision"
fi

validate_output
if docker container inspect "$session" >/dev/null 2>&1; then
    die "容器 $session 已存在，请先查看 status。"
fi
existing=0
finished=0
for group in "${groups[@]}"; do
    if [[ -e "$output/$group" ]]; then
        existing=1
    fi
    if group_done "$group"; then
        finished=$((finished + 1))
    fi
done
if [[ "$mode" == start && "$existing" == 1 ]]; then
    die "已有 $phase 组目录，请使用 resume。"
fi
if [[ "$mode" == resume && "$existing" == 0 ]]; then
    die "没有可续训的 $phase 组目录，请使用 start。"
fi
if (( finished == ${#groups[@]} )); then
    printf '%s\n' '各组均已完成。'
    exit 0
fi
docker run "${runtime[@]}" "$image" bash "$launcher" _check
cid=$(docker create "${runtime[@]}" --init --stop-timeout 90 --name "$session" \
    --label "$repo_label=$repo" --env "M4_ABLATION_INHIBIT=$inhibit_status" \
    "$image" bash "$launcher" _supervise "$mode")
# 将休眠锁和附着的 Docker 客户端放在独立会话，SSH 关闭后两者仍随容器运行。
nohup setsid "${inhibit[@]}" docker start --attach "$cid" </dev/null >/dev/null 2>&1 &
launcher_pid=$!
for attempt in {1..30}; do
    if docker exec "$cid" tmux -L "$session" has-session -t "$session" 2>/dev/null; then
        printf '已启动 %s，宿主休眠锁=%s。使用 bash %s attach 监看。\n' "$session" "$inhibit_status" "$launcher"
        exit 0
    fi
    kill -0 "$launcher_pid" 2>/dev/null || break
    sleep .25
done
# 仅清理由本次命令创建且尚未启动的容器。
if [[ $(docker container inspect --format '{{.State.Status}}' "$cid" 2>/dev/null) == created ]]; then
    docker rm "$cid" >/dev/null
fi
die "未检测到 $phase tmux，检查 status 和各组 console.log。"
