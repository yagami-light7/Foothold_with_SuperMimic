#!/usr/bin/env bash
# 所有正式训练均由显式组命令启动，独立评估读取冻结的权重。
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$repo"
out=outputs/m5_2
session=roban-m52-training
launcher=tools/run_m52_training.sh
die() { printf '%s\n' "$*" >&2; exit 1; }

group_path() {
    case "$1" in
        a) printf '%s/a_baseline\n' "$out" ;;
        b) printf '%s/b_add_tilt\n' "$out" ;;
        c) printf '%s/c_replace_upright\n' "$out" ;;
        *) die '请选择 a、b 或 c' ;;
    esac
}

select_groups() {
    case "$1" in
        all) groups=(a b c) ;;
        a|b|c) groups=("$1") ;;
        *) die '组参数应为 all、a、b 或 c' ;;
    esac
}

train() {
    local group=$1 directory
    directory="$(group_path "$group")"
    shift
    python -u supermimic/train.py --roban_config "configs/roban/m5/m5_2_${group}.json" --device cuda:0 \
        --m50_library configs/roban/m5_0_motion_library.json --m50_reference_root /reference_data \
        --log_dir "$directory/checkpoints" "$@"
}

dry() {
    select_groups "$1"
    for group in "${groups[@]}"; do
        train "$group" --m50_check | python -c '
import json,sys
line=next(s for s in sys.stdin if " effective config " in s)
cfg=json.loads(line.split(" effective config ",1)[1])
keys=("num_envs","m51_critic_warmup","iterations","lr","actor_lr","wm_window","horizon",
      "utd","critic_epochs","critic_minibatches","wm_batch","pi_batch","buffer_steps","seed",
      "m4_substeps","ckpt_every","m4_eval_every")
reward=cfg["roban_reward_config"]
print("M5.2-"+cfg["m52_group"].upper(),json.dumps(
    {k:cfg[k] for k in keys} | {k:reward[k] for k in ("upright","root_tilt","root_tilt_sigma","root_tilt_penalty")} |
    dict(reference_sampling="1:1:1",perturb_probability=.5,perturb_scale=1),ensure_ascii=False))
'
    done
}

check_preflight() {
    python - <<'PY'
import json
from pathlib import Path
data=json.loads(Path('outputs/m5_2/preflight.json').read_text())
goal=data['orientation']['goal']
assert goal['obs_dim']==69 and goal['goals']==[10,32] and goal['critic_dim']==970
assert goal['foot_and_position_unchanged'] and goal['packed_boundary_clamps']
assert data['unit']['passed'] and data['cache']['passed']
assert data['cuda']['unit']['passed'] and data['cuda']['cache']['passed']
assert set(data['dynamics'])=={'F1','F2','F3'}
assert all(row['replay']['a_exact'] and row['replay']['tracks']==15 for row in data['dynamics'].values())
assert [(row['group'],row['envs']) for row in data['smokes']]==[('a',8),('b',32),('c',256)]
assert all(row['epa']==0 and row['warmup_actor_wm_frozen'] and row['max_reward_delta']<=1e-5
           for row in data['smokes'])
assert data['smokes'][-1]['resumed'] and data['smokes'][-1]['tilt_grad']>0
print('M5.2 训练前验收通过')
for smoke in data['smokes']:
    print(f'{smoke["group"].upper()} {smoke["envs"]} 环境，{smoke["transitions"]} 条转移，'
          f'峰值 {smoke["maximum_device_mib"]:.0f} MiB，Tilt Actor 梯度 {smoke["tilt_grad"]:.4g}')
PY
}

complete_group() {
    [[ -s "$1/checkpoints/iter_2000.pt" && -s "$1/checkpoints/best_contact_stability.pt"
       && -s "$1/checkpoints/best_multi_reference.pt" && -s "$1/evaluation/results.json" ]]
}

run_group() (
    local group=$1 mode=$2 directory
    directory="$(group_path "$group")"
    local marker="$directory/train.exitcode" last="$directory/checkpoints/last.pt" rc=0
    if [[ -s "$marker" && $(<"$marker") == 0 ]]; then
        complete_group "$directory" || die "M5.2-$group 完成记录缺少权重或评估"
        [[ "$mode" == all ]] || die "M5.2-$group 已完成"
        printf 'M5.2-%s 已完成\n' "$group"
        return
    fi
    local -a resume=()
    if [[ "$mode" == resume ]]; then
        [[ -s "$last" ]] || die "M5.2-$group 缺少本组 Last"
        resume=(--resume "$last")
    else
        [[ ! -e "$marker" && ! -d "$directory/checkpoints" ]] ||
            die "M5.2-$group 已有运行记录，请检查日志或使用 resume $group"
    fi
    mkdir -p "$directory/logs"
    exec > >(tee -a "$directory/logs/console.log") 2>&1
    printf 'running\n' > "$marker"
    printf 'M5.2-%s %s %s\n' "$group" "$mode" "$(date -Is)"
    train "$group" "${resume[@]}" || rc=$?
    if (( rc == 0 )); then
        complete_group "$directory" || rc=1
    fi
    printf '%s\n' "$rc" > "$marker"
    return "$rc"
)

if [[ "${M52_IN_CONTAINER:-}" == 1 ]]; then
    mode=$1
    shift
    case "$mode" in
        _dry) dry "$1" ;;
        _check)
            check_preflight
            dry "$1"
            python -c 'import torch; print("CUDA",torch.cuda.get_device_name())'
            tmux -V ;;
        _orientation) exec python -u -m supermimic.integration.m52_audit --reference-root /reference_data ;;
        _audit|_cuda) exec python -u tools/test_m52.py --root /reference_data --stage "${mode#_}" ;;
        _evaluate) exec python -u -m supermimic.integration.m52_evaluation final --reference-root /reference_data ;;
        _visuals) exec python -u -m supermimic.integration.m52_visuals --reference-root /reference_data "$@" ;;
        _online) exec python -u -m supermimic.integration.m52_online --reference-root /reference_data "$@" ;;
        _worker)
            select_groups "$2"
            for group in "${groups[@]}"; do
                run_group "$group" "$1" || exit $?
            done ;;
        _supervise)
            printf -v command '%q ' bash "$launcher" _worker "$1" "$2"
            tmux -f /dev/null -L "$session" new-session -d -s "$session" -e SHELL=/bin/bash "$command"
            trap 'tmux -L "$session" send-keys -t "$session" C-c 2>/dev/null || true' TERM INT
            while tmux -L "$session" has-session -t "$session" 2>/dev/null; do
                sleep 1 &
                wait "$!" || true
            done
            select_groups "$2"
            for group in "${groups[@]}"; do
                marker="$(group_path "$group")/train.exitcode"
                [[ -s "$marker" && $(<"$marker") == 0 ]] || exit 1
            done ;;
        *) die '未知的容器内部模式' ;;
    esac
    exit
fi

mode=${1:-}
case "$mode" in
    status|attach|stop)
        active="$(docker inspect --format '{{if .State.Running}}{{ index .Config.Labels "org.supermimic.m52.repo" }}{{end}}' \
            "$session" 2>/dev/null)" || active=
        if [[ "$mode" == status ]]; then
            printf 'M5.2 tmux %s\n' "$([[ "$active" == "$repo" ]] && printf running || printf inactive)"
            for group in a b c; do
                directory="$(group_path "$group")"
                if [[ -s "$directory/train.exitcode" ]]; then
                    printf 'M5.2-%s %s\n' "$group" "$(<"$directory/train.exitcode")"
                    [[ ! -f "$directory/logs/console.log" ]] || tail -n 3 "$directory/logs/console.log"
                else
                    printf 'M5.2-%s 待启动\n' "$group"
                fi
                if [[ -s "$directory/swanlab_sync_status.json" ]]; then
                    python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); print("SwanLab",s.get("state",s.get("status","unknown")),s.get("last_step",-1))' \
                        "$directory/swanlab_sync_status.json"
                else
                    printf 'SwanLab 待同步，训练指标保存在本地 CSV\n'
                fi
            done
            exit
        fi
        [[ "$active" == "$repo" ]] || die '当前没有本仓库运行中的 M5.2 tmux'
        if [[ "$mode" == attach ]]; then
            exec docker exec -it "$session" tmux -L "$session" attach-session -t "$session"
        fi
        exec docker stop --time 90 "$session" ;;
    --dry-run|check) target=${2:-all}; select_groups "$target" ;;
    all|a|b|c) target=$mode ;;
    resume) target=${2:-}; group_path "$target" >/dev/null ;;
    swanlab)
        directory="$(group_path "${2:-}")"
        exec "${SWANLAB_PYTHON:-.venv-swanlab/bin/python}" tools/sync_swanlab.py --roban_m52 \
            --run_dir "$directory" --follow --completion_marker ;;
    preflight) [[ "${2:-all}" =~ ^(all|orientation|audit|cuda)$ ]] || die 'preflight 支持 all、orientation、audit、cuda' ;;
    evaluate|visuals|online) ;;
    *) die "用法 bash $launcher [--dry-run all|check|preflight|all|a|b|c|resume c|status|attach|stop|evaluate|visuals|online|swanlab c]" ;;
esac

mkdir -p "$out" "$repo/.cache/roban_warp"
generator="$(realpath -e -- "${M52_GENERATOR_ROOT:-../Humanoid_Online_Motion_Generator}")"
reference="$(realpath -e -- "${M50_REFERENCE_ROOT:-$generator/outputs}")"
model="$(realpath -e -- ../PMT/library/model/roban_s22_0901)"
[[ -f "$reference/m3_7_1/forward/F2/reference.npz"
   && -s outputs/m5_1/c_yaw_angular_slip/checkpoints/iter_1500.pt ]] ||
    die '找不到原始直走参考或 M5.1-C1500 权重'
image=${DOCKER_IMAGE:-supermimic:latest}
runtime=(--rm --init --network none --shm-size=2g --user "$(id -u):$(id -g)"
    --mount "type=bind,source=$repo,target=/workspace/supermimic,readonly"
    --mount "type=bind,source=$repo/$out,target=/workspace/supermimic/$out"
    --mount "type=bind,source=$repo/.cache/roban_warp,target=/workspace/supermimic/.cache/roban_warp"
    --mount "type=bind,source=$reference,target=/reference_data,readonly"
    --mount "type=bind,source=$generator,target=/workspace/Humanoid_Online_Motion_Generator,readonly"
    --mount "type=bind,source=$model,target=/workspace/PMT/library/model/roban_s22_0901,readonly"
    --workdir /workspace/supermimic --env M52_IN_CONTAINER=1 --env PYTHONDONTWRITEBYTECODE=1
    --env PYTHONPATH=/workspace/Humanoid_Online_Motion_Generator:/workspace/supermimic
    --env WARP_CACHE_PATH=/workspace/supermimic/.cache/roban_warp --env MPLCONFIGDIR=/tmp/roban_mpl
    --env MUJOCO_GL=egl --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
    --env "S54_SOURCE_GIT_SHA=$(git rev-parse HEAD)"
    --env "M52_SOURCE_GIT_STATUS=$(GIT_OPTIONAL_LOCKS=0 git status --porcelain)")
[[ ! -d /usr/share/fonts/opentype/noto ]] ||
    runtime+=(--mount type=bind,source=/usr/share/fonts/opentype/noto,target=/usr/share/fonts/opentype/noto,readonly)
if [[ "$mode" == --dry-run ]]; then
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
    exec docker run "${runtime[@]}" "$image" bash "$launcher" _dry "$target"
fi

if [[ "$mode" == preflight ]]; then
    stage=${2:-all}
    if [[ "$stage" == all || "$stage" == orientation ]]; then
        original="$(realpath -e -- ../PMT/TCRS/outputs/roban_motion_dataset/train/flat/forward_012/npz/body_reference.npz)"
        docker run "${runtime[@]}" --mount "type=bind,source=$original,target=/workspace/PMT/TCRS/outputs/roban_motion_dataset/train/flat/forward_012/npz/body_reference.npz,readonly" \
            "$image" bash "$launcher" _orientation
    fi
    if [[ "$stage" == all || "$stage" == audit ]]; then
        docker run "${runtime[@]}" "$image" bash "$launcher" _audit
    fi
    [[ "$stage" == all || "$stage" == cuda ]] || exit 0
fi

check_gpu() {
    local active training
    training="$(docker ps --filter 'name=^/roban-m5[012]-training$' --format '{{.Names}}')"
    [[ -z "$training" ]] || die "检测到运行中的训练会话
$training"
    active="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)"
    [[ -z "$active" ]] || die "检测到运行中的 GPU 计算任务
$active"
}
check_gpu
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
runtime+=(--gpus all)
inhibit=(systemd-inhibit --what=sleep:idle:handle-lid-switch --mode=block --who="$session" --why='Roban M5.2 CUDA')
if ! "${inhibit[@]}" true >/dev/null 2>&1; then inhibit=(); fi

case "$mode" in
    preflight) exec "${inhibit[@]}" docker run "${runtime[@]}" "$image" bash "$launcher" _cuda ;;
    evaluate|visuals|online)
        shift
        exec "${inhibit[@]}" docker run "${runtime[@]}" "$image" bash "$launcher" "_$mode" "$@" ;;
esac

free="$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits)"
(( free >= 2500 )) || die 'CUDA:0 可用显存不足 2500 MiB'
docker run "${runtime[@]}" "$image" bash "$launcher" _check "$target"
[[ "$mode" != check ]] || exit 0
check_gpu
cid="$(docker create "${runtime[@]}" --name "$session" --stop-timeout 90 \
    --label "org.supermimic.m52.repo=$repo" "$image" bash "$launcher" _supervise "$mode" "$target")"
nohup setsid "${inhibit[@]}" docker start --attach "$cid" </dev/null >/dev/null 2>&1 &
for attempt in {1..30}; do
    if docker exec "$cid" tmux -L "$session" has-session -t "$session" 2>/dev/null; then
        printf 'M5.2 tmux 已启动 %s\n' "$session"
        exit 0
    fi
    sleep .25
done
die 'M5.2 tmux 未确认启动，请查看 status'
