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

## 已执行验证

目录整理前，按相同1024env、WM1024×4、Actor256、H8/hist10完成16轮短测试：303104条真实transition；WM/Encoder/Decoder/Critic均实际更新；子进程40.91s；Torch峰值1909MiB、reserved2094MiB、设备保存时总占用3220MiB。tmux实时打印及SwanLab在线最终同步成功。见 `results/s54/stage1_walk_launch_validation/smoke_result.json`。

这些是资源/启动验证，不是Teacher质量评估。整理后再按新路径运行smoke，证据写入 `results/s54/layout_validation_20261105/`；正式2小时训练尚未由本轮自动启动。
