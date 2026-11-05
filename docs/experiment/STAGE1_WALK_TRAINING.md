# Stage 1：walk1_subject1 训练

目标：用QA通过的 `walk1_subject1` 训练 Motion Encoder、Shared Decoder、World Model和Critic，为后续A–G提供同源WM/Decoder。当前从头训练；不自动启动Bridge或A–G。

## 启动与实时终端

从仓库根目录执行：

```bash
RUN="s54_stage1_walk_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
bash tools/train/run_s54_stage1_walk.sh start "$RUN" && tmux attach -t "$RUN"
```

tmux内每20轮打印一次，Python使用 `-u`；完整stdout/stderr保存到 `logs/$RUN/stdout.log`，启动器输出另存 `${RUN}_launcher.log`。`Ctrl-b d`离开tmux，SSH断开不影响训练和独立上传进程。训练结束后保留终端输出。

```bash
# 查看参数和状态
cat "logs/$RUN/args.json"
bash tools/train/run_s54_stage1_walk.sh status "$RUN"
tail -F "logs/$RUN/stdout.log"

# 重新登录后，填写原run名；用 tmux ls 查找
tmux attach -t "$RUN"

# 安全停止：等待checkpoint写完及上传器最终同步
bash tools/train/run_s54_stage1_walk.sh stop "$RUN"
```

## 默认参数

配置：[stage1_walk_3090.json](../../configs/motion_prior/stage1/stage1_walk_3090.json)。

| 参数 | 值 |
|---|---:|
| Reference | `walk1_subject1`，manifest train split仅此一段 |
| num_envs | 1024 |
| Replay buffer_steps / warmup_steps | 1024 / 256 |
| WM batch / 每轮更新数 / window | 1024 / 4 / 8 |
| WM-only | 前1000 iterations |
| Actor batch | 256 |
| hist / goal_frames / goal_dim | 10 / 10 / 730 |
| BPTT Horizon | 8，即0.16s imagined rollout |
| WM / Actor / Critic LR | 5e−5 / 1e−5 / 1e−5 |
| UTD / action_noise | .4 / .03 |
| Critic epochs / minibatches | 2 / 4 |
| seed / AMP | 54 / bf16 |
| 停止条件 | 20000总iterations或单次进程2小时，先到即停 |
| 周期保存 / 保留数 | 每500轮 / 最近6个；另外保留last和WM warmup里程碑 |

Horizon8与真实环境采样长度不同。UTD=.4使每轮平均采集2.5个vector steps；1000轮WM-only结束约累计2822144条真实transition，20000轮的本次fresh上限约51462144条。归一化和Replay随真实采样更新。前1000轮 `pi/v/actor_grad=0` 属预期，第1001轮开始Actor/Critic更新。

预留保存/退出时间：训练自身wall上限7170s，启动器在2h绝对deadline前15s发TERM，到deadline才强制结束。恢复run的2小时预算重新计时，checkpoint的wall和样本计数仍累计；总iteration target仍是20000。

## SwanLab

项目/实验组：`Foothold_with_SuperMimic`；名称 `Stage1_walk1_subject1_时间戳_seed54`。上传器在独立 `${RUN}_upload` tmux session运行，每5秒读取已落盘标量，稳定ID记录于 `swanlab_run_id`，缓存留在run的 `swanlab/`；错误记录于 `swanlab_sync.log`并重试，正常结束检查 `final_sync.exitcode=0`。仅使用主机已有登录/环境凭据。

```bash
tail -F "logs/$RUN/swanlab_sync.log"
cat "logs/$RUN/swanlab_sync_status.json"
cat "logs/$RUN/final_sync.exitcode"
```

## 恢复与评估

```bash
OLD=s54_stage1_walk_填写原时间戳
NEW="s54_stage1_walk_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
bash tools/train/run_s54_stage1_walk.sh resume "$NEW" "logs/$OLD/last.pt"
tmux attach -t "$NEW"
```

恢复使用原profile和合同、全模型/optimizer/norm/RNG/计数；新日志目录保留原run。Replay未序列化，需要重新采样并做20轮WM-only恢复。先停止训练、确认GPU空闲，再串行评估：

```bash
export S54_PYTHON=/home/rm/workspace/IsaacLab_3_0/.venv/bin/python
$S54_PYTHON tools/validation/evaluate_s54_motion_prior.py \
  --checkpoint "logs/$RUN/last.pt" \
  --manifest results/s54/motion_prior_stage0_verified/manifest.json --split train \
  --out "results/s54/${RUN}_eval" --durations 5 10 --seeds 54 55 56 \
  --start_seconds 0 30 60 --reset_noise --viz none
```

WM warmup结束会保存 `wm_warmup_complete.pt`；这不表示WM Gate或Teacher验收通过。完整clip包含较快片段，本配置是第一轮训练，不能保证2小时收敛。独立Walking holdout仍缺失，Dance不计入Walking泛化。

## Policy 与 Reference 叠加视频

统一入口使用同一个 `--seconds` 控制真实Newton Policy执行与视频渲染，默认叠加灰色Policy、青色Reference的**世界轨迹**：

```bash
/home/rm/workspace/IsaacLab_3_0/.venv/bin/python tools/visualization/render.py \
  --run_dir logs/s54_stage1_walk_20261105_014722 \
  --seconds 300 \
  --out "results/s54/walk1_policy_reference_$(date +%Y%m%d_%H%M%S).mp4"
```

当前Reference有效区间为 `[0,261.3]` 秒，请求300秒会自动将Newton与视频目标都截断为261.3秒。`--start 60 --seconds 300` 则从Reference第60秒初始化Policy，最多执行201.3秒。时间按0.02秒控制步向下取整；物理终止会提前结束，不用静止画面补满请求时长。完整Reference不会重复或延长。

可用 `--checkpoint FILE.pt` 指定权重；默认使用run的 `last.pt`。添加 `--align_reference` 可将青色Reference的XY/yaw对齐到Policy，专门比较姿态；`--global_ghost` 显式选择默认的世界轨迹模式。使用 `--dry-run` 可只检查Reference区间与两阶段命令，不启动GPU仿真。

该入口在一个GPU锁内串行录制和渲染，拒绝其他GPU计算任务竞争，默认整个流程预算600秒（`--process_budget_s`）。视频旁的 `视频名_recording/` 保存真实 `trajectory.npz`、`evaluation.json`、目标时长 `plan.json`、Newton/渲染日志及 `status.json`。现有输出不会覆盖。只改变评估超时和显示方式，不改变训练奖励、物理终止条件或有界Teacher Gate。

`--run_dir` 输入的是权重，所以先执行Policy生成轨迹，再渲染视频；checkpoint本身不包含逐帧运动。终端分为 `[Preview] Phase 1/2`（Newton）与 `Phase 2/2`（渲染），每约5秒打印步数或帧数、吞吐量与ETA；初始化长时间无输出时会打印进程存活提示。`status.json` 记录当前阶段。仿真秒数不等于实际等待秒数，资源服务器超时、逐步Policy推理和MP4编码都会增加等待时间。

完整 `.npz` 保存后，换视角或重新编码应使用下面的文件回放模式，从而省去重复rollout。

已有录制仍可用文件回放模式，无需重新仿真。它也直接支持 `evaluate_s54_motion_prior.py` 的有界Newton格式，通过对应训练manifest恢复同一份Reference，使用录制帧号和训练相同的50Hz重采样。

例如，对已经录制的新实验10秒轨迹执行：

```bash
PREVIEW=results/s54/stage1_walk_20261105_014722_preview
PATH="$PWD/.cache/bin:$PATH" MUJOCO_GL=egl \
  /home/rm/workspace/IsaacLab_3_0/.venv/bin/python tools/visualization/render.py \
  "$PREVIEW/newton/walk1_subject1_10s_seed54.npz" \
  "$PREVIEW/walk1_reference_$(date +%Y%m%d_%H%M%S).mp4" \
  --reference_manifest results/s54/motion_prior_stage0_verified/manifest.json \
  --seconds 10 --fps 25 --title Stage1_iter4772
```

文件回放模式的 `--seconds` 仅裁剪已有数据，默认限制到Policy实际录制长度。要增加真实执行时长，使用上面的 `--run_dir` 统一入口。显式加 `--reference_tail` 才会在文件回放中继续播放Reference并标注保持的Policy姿态。画面顶部显示对齐模式和Policy实际录制时长，正常Timeout不会标为物理跌倒。`render_s54_motion_prior.py` 仍用于多策略并排回放，只显示各录制的Policy。

2026-11-05已验证：4项CPU回归通过；统一入口在RTX3090执行30秒/1500步Newton rollout，正常Timeout、物理终止0，生成750帧/25FPS视频 `results/s54/stage1_walk_20261105_014722_unified_30s.mp4`。边界验证 `--start 260 --seconds 300` 自动截断为1.3秒，实际执行65步正常Timeout；视频帧均在有效Reference范围内。以上为预览功能验证，不替代多窗口、多种子Teacher验收。

## 已执行验证

目录整理前，按相同1024env、WM1024×4、Actor256、H8/hist10完成16轮短测试：303104条真实transition；WM/Encoder/Decoder/Critic均实际更新；子进程40.91s；Torch峰值1909MiB、reserved2094MiB、设备保存时总占用3220MiB。tmux实时打印及SwanLab在线最终同步成功。见 `results/s54/stage1_walk_launch_validation/smoke_result.json`。

这些是资源/启动验证，不是Teacher质量评估。整理后再按新路径运行smoke，证据写入 `results/s54/layout_validation_20261105/`；正式2小时训练尚未由本轮自动启动。
