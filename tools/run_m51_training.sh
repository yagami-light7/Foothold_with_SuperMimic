#!/usr/bin/env bash
# M5.1 正式运行入口。正式组只由显式命令启动。
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$repo"
out=outputs/m5_1
session=roban-m51-training
launcher=tools/run_m51_training.sh
die() { printf '%s\n' "$*" >&2; exit 1; }

group_path() {
    case "$1" in
        a) printf '%s/a_baseline\n' "$out" ;;
        b) printf '%s/b_stance_yaw\n' "$out" ;;
        c) printf '%s/c_yaw_angular_slip\n' "$out" ;;
        *) die '请选择 a、b 或 c' ;;
    esac
}

train() {
    local group=$1 directory
    directory="$(group_path "$group")"
    shift
    python -u supermimic/train.py --roban_config "configs/roban/m5/m5_1_${group}.json" --device cuda:0 \
        --m50_library configs/roban/m5_0_motion_library.json --m50_reference_root /reference_data \
        --log_dir "$directory/checkpoints" "$@"
}

check_preflight() {
    python - <<'PY'
import json
from pathlib import Path
path=Path('outputs/m5_1/preflight.json')
data=json.loads(path.read_text())
assert data['unit']['passed'] and set(data['newton'])=={'F1','F2','F3'}
assert set(data['offline']['baseline'])=={'F1','F2','F3'}
assert set(data['offline']['gradient'])=={'F1','F2','F3'}
assert [(s['group'],s['envs']) for s in data['smokes']]==[('a',8),('b',32),('c',256)]
assert all(s['epa']==0 and s['warmup_actor_wm_frozen'] for s in data['smokes'])
assert data['smokes'][-1]['resumed'] and data['smokes'][-1]['angular_grad']>0
print('M5.1 训练前检查通过')
for clip, groups in data['offline']['baseline'].items():
    row=groups['nominal']['contact']
    support=row['support']
    print(f'{clip} 支撑 Yaw {row["planned_stance"]["yaw_rms_deg"]:.2f}° | '
          f'角速度 {support["omega"]["rms"]:.3f} rad/s | 水平滑动 {1000*support["speed"]["rms"]:.1f} mm/s')
for smoke in data['smokes']:
    print(f'{smoke["group"].upper()} {smoke["envs"]} 环境 | {smoke["transitions"]} 个转移 | '
          f'峰值 {smoke["maximum_device_mib"]:.0f} MiB | EPA {smoke["epa"]}')
PY
}

run_group() (
    local group=$1 action=$2 directory rc=0
    directory="$(group_path "$group")"
    local marker="$directory/train.exitcode" checkpoint="$directory/checkpoints/last.pt"
    if [[ -s "$marker" && $(<"$marker") == 0 ]]; then
        [[ -s "$directory/checkpoints/iter_2000.pt" && -s "$directory/evaluation/results.json" ]] ||
            die "M5.1-$group 的完成标记缺少联合末轮或评估"
        [[ "$action" == all ]] || die "M5.1-$group 已完成"
        printf 'M5.1-%s 已完成\n' "$group"
        return
    fi
    local -a resume=()
    if [[ "$action" == resume ]]; then
        [[ -s "$checkpoint" ]] || die "M5.1-$group 缺少 Last 检查点"
        resume=(--resume "$checkpoint")
    else
        [[ ! -e "$marker" && ! -d "$directory/checkpoints" ]] ||
            die "M5.1-$group 已有运行记录，请检查日志或使用 resume $group"
    fi
    mkdir -p "$directory/logs" || die "无法创建 M5.1-$group 日志目录"
    exec > >(tee -a "$directory/logs/console.log") 2>&1
    printf 'running\n' > "$marker" || die "无法写入 M5.1-$group 状态"
    printf 'M5.1-%s %s %s\n' "$group" "$action" "$(date -Is)"
    train "$group" "${resume[@]}" || rc=$?
    if (( rc == 0 )); then
        [[ -s "$directory/checkpoints/iter_2000.pt" && -s "$directory/checkpoints/best_contact_stability.pt"
           && -s "$directory/evaluation/results.json" ]] || rc=1
    fi
    printf '%s\n' "$rc" > "$marker"
    return "$rc"
)

select_groups() {
    case "$1" in
        all) groups=(a b c) ;;
        a|b|c) groups=("$1") ;;
        *) die '组参数应为 all、a、b 或 c' ;;
    esac
}

case "${1:-}" in
    _dry|_preflight|_check|_worker|_supervise|_evaluate|_visuals)
        [[ "${M51_IN_CONTAINER:-}" == 1 ]] || die '仅容器内部使用'
        case "$1" in
            _dry)
                select_groups "$2"
                for group in "${groups[@]}"; do
                    train "$group" --m50_check | python -c '
import json,sys
line=next(s for s in sys.stdin if " effective config " in s)
cfg=json.loads(line.split(" effective config ",1)[1])
reward=cfg["roban_reward_config"]
keys=("num_envs","m51_critic_warmup","iterations","lr","actor_lr","wm_window","horizon","utd",
      "wm_batch","pi_batch","buffer_steps","seed","m4_substeps","ckpt_every","m4_eval_every")
print("M5.1-"+cfg["m51_group"].upper(), json.dumps({k:cfg[k] for k in keys} |
      dict(stance_yaw=reward["stance_yaw_huber"],angular_slip=reward["stance_angular_slip"],
           reference_sampling="1:1:1",perturb_probability=.5,perturb_scale=1),ensure_ascii=False))
'
                done ;;
            _preflight) exec python -u tools/test_m51.py --root /reference_data --stage "${2:-all}" ;;
            _check)
                check_preflight
                bash "$launcher" _dry "$2"
                python -c 'import torch; print("CUDA",torch.cuda.get_device_name())'
                tmux -V ;;
            _worker)
                action=$2
                select_groups "$3"
                for group in "${groups[@]}"; do
                    run_group "$group" "$action" || exit $?
                done ;;
            _supervise)
                printf -v command '%q ' bash "$launcher" _worker "$2" "$3"
                tmux -f /dev/null -L "$session" new-session -d -s "$session" -e SHELL=/bin/bash "$command"
                trap 'tmux -L "$session" send-keys -t "$session" C-c 2>/dev/null || true' TERM INT
                while tmux -L "$session" has-session -t "$session" 2>/dev/null; do
                    sleep 1 &
                    wait "$!" || true
                done
                select_groups "$3"
                for group in "${groups[@]}"; do
                    marker="$(group_path "$group")/train.exitcode"
                    [[ -s "$marker" && $(<"$marker") == 0 ]] || exit 1
                done ;;
            _evaluate) exec python -u -m supermimic.integration.m51_evaluation final --reference-root /reference_data ;;
            _visuals) exec python -u -m supermimic.integration.m51_visuals --reference-root /reference_data ;;
        esac
        exit ;;
esac

mode=${1:-}
case "$mode" in
    status|attach|stop)
        active="$(docker inspect --format '{{if .State.Running}}{{ index .Config.Labels "org.supermimic.m51.repo" }}{{end}}' "$session" 2>/dev/null)" || active=
        if [[ "$mode" == status ]]; then
            printf 'M5.1 tmux %s\n' "$([[ "$active" == "$repo" ]] && printf running || printf inactive)"
            for group in a b c; do
                directory="$(group_path "$group")"
                if [[ -s "$directory/train.exitcode" ]]; then
                    printf 'M5.1-%s %s\n' "$group" "$(<"$directory/train.exitcode")"
                    [[ ! -f "$directory/logs/console.log" ]] || tail -n 4 "$directory/logs/console.log"
                else
                    printf 'M5.1-%s 待启动\n' "$group"
                fi
            done
            exit
        fi
        [[ "$active" == "$repo" ]] || die '未找到本仓库正在运行的 M5.1 会话'
        if [[ "$mode" == attach ]]; then
            exec docker exec -it "$session" tmux -L "$session" attach-session -t "$session"
        fi
        exec docker stop --time 90 "$session" ;;
    --dry-run|check) target=${2:-all}; select_groups "$target" ;;
    all|a|b|c) target=$mode ;;
    resume) target=${2:-}; group_path "$target" >/dev/null ;;
    preflight|evaluate|visuals) target=all ;;
    *) die "用法 bash $launcher [--dry-run all|check|preflight|all|a|b|c|resume c|status|attach|stop|evaluate|visuals]" ;;
esac

mkdir -p "$out" "$repo/.cache/roban_warp"
reference="$(realpath -- "${M50_REFERENCE_ROOT:-../Humanoid_Online_Motion_Generator/outputs}")"
[[ -f "$reference/m3_7_1/forward/F2/reference.npz" && -s outputs/m5_0/checkpoints/iter_5500.pt ]] ||
    die '找不到原始直走参考或 M5.0 第 5500 轮权重'
image=${DOCKER_IMAGE:-supermimic:latest}
runtime=(--rm --init --network none --shm-size=2g --user "$(id -u):$(id -g)"
    --mount "type=bind,source=$repo,target=/workspace/supermimic,readonly"
    --mount "type=bind,source=$repo/$out,target=/workspace/supermimic/$out"
    --mount "type=bind,source=$repo/.cache/roban_warp,target=/workspace/supermimic/.cache/roban_warp"
    --mount "type=bind,source=$reference,target=/reference_data,readonly"
    --mount "type=bind,source=/usr/share/fonts/opentype/noto,target=/usr/share/fonts/opentype/noto,readonly"
    --workdir /workspace/supermimic --env PYTHONDONTWRITEBYTECODE=1 --env M51_IN_CONTAINER=1
    --env WARP_CACHE_PATH=/workspace/supermimic/.cache/roban_warp --env MPLCONFIGDIR=/tmp/roban_mpl
    --env MUJOCO_GL=egl --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
    --env "S54_SOURCE_GIT_SHA=$(git rev-parse HEAD)"
    --env "M51_SOURCE_GIT_STATUS=$(GIT_OPTIONAL_LOCKS=0 git status --porcelain)")
if [[ "$mode" == --dry-run ]]; then
    exec docker run "${runtime[@]}" "$image" bash "$launcher" _dry "$target"
fi
if [[ "$mode" == preflight && "${2:-all}" == offline ]]; then
    exec docker run "${runtime[@]}" "$image" bash "$launcher" _preflight offline
fi

active="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)"
[[ -z "$active" ]] || die "检测到正在占用 GPU 的计算任务
$active"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
runtime+=(--gpus all)
inhibit=(systemd-inhibit --what=sleep:idle:handle-lid-switch --mode=block --who="$session" --why='Roban M5.1 CUDA')
if ! "${inhibit[@]}" true >/dev/null 2>&1; then inhibit=(); fi

case "$mode" in
    preflight) exec "${inhibit[@]}" docker run "${runtime[@]}" "$image" bash "$launcher" _preflight "${2:-all}" ;;
    evaluate) exec "${inhibit[@]}" docker run "${runtime[@]}" "$image" bash "$launcher" _evaluate ;;
    visuals) exec "${inhibit[@]}" docker run "${runtime[@]}" "$image" bash "$launcher" _visuals ;;
esac

free="$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits)"
(( free >= 2500 )) || die 'CUDA:0 剩余显存低于 256 环境冒烟峰值所需余量'
docker run "${runtime[@]}" "$image" bash "$launcher" _check "$target"
[[ "$mode" != check ]] || exit 0
cid="$(docker create "${runtime[@]}" --name "$session" --stop-timeout 90 \
    --label "org.supermimic.m51.repo=$repo" "$image" bash "$launcher" _supervise "$mode" "$target")"
nohup setsid "${inhibit[@]}" docker start --attach "$cid" </dev/null >/dev/null 2>&1 &
for attempt in {1..30}; do
    if docker exec "$cid" tmux -L "$session" has-session -t "$session" 2>/dev/null; then
        printf 'M5.1 tmux 已启动 %s\n' "$session"
        exit 0
    fi
    sleep .25
done
die 'M5.1 tmux 未确认启动，请查看 status'
