# Flat 6063 → 四级楼梯：8 小时单动作微调实验

本目录准备了可手动启动的实验。用户执行的 v2 Smoke 在 warmup 被奖励硬门禁停止，正式微调未启动；此次修复只执行 CPU 检查，未自动重启 GPU。失败日志保留在 `logs/s54_stair_flat6063_smoke_v2`，新 Smoke 使用 `logs/s54_stair_flat6063_smoke_v3`。新的 GPU Smoke 未通过前，正式训练入口会拒绝启动。

## 数据与初始化

- 起点：`release/S54-SuperMimic-Flat-6000/checkpoints/ckpt_006063.pt`，实际 iteration **6063**，SHA256 `e49e968a1a6bd7c5fcef53290b805fafb011bea5533e9bc0c050e5130e0bbe88`；与原 Flat `latest.pt` 字节一致，只读使用。
- 四级 Reference：`data/s54_stair_dataset/stair_up_4.npz`，436 帧 / 50 Hz / 29 DoF / 30 bodies。该目录仅放这一份 NPZ，SHA256 `288d3abd2fd37d3d299a17272bcf4d99a6fac0fc28dcb3e10e1c830a633bbf65`。
- 契约仍放在 `results/s54_stair_t15_20261110/data/newton_import`，加载实际的 data 下 NPZ，并用原 manifest 的字节哈希、内容哈希、模型与顺序进行严格校验。不要删除这个契约目录。
- 楼梯：`assets/terrains/stairs_straight_4_v1`，Mesh SHA256 `1503c3e9481b92e0a6979631d3fd7748d21f7978954c48dc7cffb8b5cc6a43b2`，世界变换为 identity。Height Map / Footstep / 原始碰撞 Mesh 均未修改。
- 四级由用户明确指定为训练动作；上游 NPZ 中 `dataset_split=test` 不改写，不再作为独立楼梯泛化测试。只混合原 Flat 的 **14 条 train**，不读取 Flat test 的轨迹，不加载二级。
- Flat 配对地面与 Reference 统一固定平移 +20 m；楼梯保持源世界坐标。没有逐帧对齐或漂移后重设 Anchor。
- 保留 103D Observation、10 帧 History、730D Motion Goal、Motion Encoder + Shared Decoder + WM + Critic + BPTT。Goal 仍包含未来 Root Z，Observation 仍为原来的 XY / Heading 反馈；没有为了迁移改变网络接口。
- 从 Flat 复制完整 WM、Policy 和归一化统计，包括原始 count；按上次训练方式，仅用训练集真实采集数据在线更新 normalizer，不压低有效计数。独立 Eval 冻结统计。Critic、optimizers、replay 新建，因为楼梯采用固定世界 Body/Foot、Root XYZ 和 stance 足速罚项。
- 使用 MJWarp 求解器、原执行器与力矩限制、20 ms 控制 / 4 子步；Newton 接触生成保留原楼梯三角网格。源与本机 Newton/MJWarp 版本不同，属于显式迁移，不宣称严格 Flat resume。具体版本记录在 `preflight.json`。

## 预算与采样

| 参数 | Smoke | 整晚微调 |
|---|---:|---:|
| num_envs | 2048 | 2048 |
| iterations 上限 | 64 | 10000 |
| wall_seconds | 900 | **28800（8h）** |
| WM-only iterations | 32 | 1000 |
| buffer / warmup steps | 1024 / 256 | 1024 / 256 |
| WM batch × batches | 1024 × 16 | 1024 × 16 |
| Policy batch / BPTT horizon | 256 / 32 | 256 / 32 |
| WM / Actor / Critic LR | 2e-4 / 2e-4 / 2e-4 | 相同 |
| UTD / action_noise | 0.4 / 0.1 | 相同 |
| eval interval | gate 时独立评估 | **1000 iterations** |
| checkpoint interval | 32 | 250，保留最近 8 个 periodic |

正式训练的 39 项核心规模、优化与网络参数按 `release/S54-SuperMimic-Flat-6000/configs/args.json` 恢复，对照清单见 `flat_training_profile.json`。`actor_lr=null` / `critic_lr=null` 表示继承 `lr=2e-4`。不再使用之前的 64 envs / 降低 Actor LR / 缩小 WM batch 的候选配置，也不设置之前额外的 7100 MiB 提前停止阈值。完整参数在当前 Native Newton 后端的显存与吞吐，仍需新 GPU Smoke 实测。

训练以累计 8h 墙钟预算或 10000 iteration 中先到的条件结束；8h 是上限。微调任务从原 Flat 6063 权重初始化，新任务的 iteration 从 0 开始。初始化与周期评估计入预算，预留最后 300 秒保存和最终独立评估。到时保存 latest，正常标为 stopped；失败会保留错误信息，不自动宣称训练完成。Smoke 仅缩短轮数与 WM-only 阶段等诊断周期，环境数、replay 容量和 WM/Policy batch 与正式训练一致。

采样 A：前 1000 轮 Flat 80% / Stair 20%；B：1000–5999 轮 Flat 40% / Stair 60%；C：6000 轮起 Flat 50% / Stair 50%。Flat 内先等运动模式、再等 clip，clip 内均匀/失败 bin 混合采样，不按帧数分配组权重。按原配置，前 1000 轮为 WM-only，因此 Stage A 校准 WM，Actor/Critic 从 Stage B 开始更新。

训练 Reset 位姿/关节噪声、摩擦、COM、关节目标偏移随机化恢复上次 Flat 的设置。独立评估同时包含 seed 0 名义初态和 seeds 1/2 姿态扰动。

## 推荐的一条启动命令

在仓库根目录运行：

```bash
export ISAAC_PYTHON=/mnt/sage_data/workspace/Toolbox/IsaacLab_3_0/.venv/bin/python
"$ISAAC_PYTHON" tools/run_s54_stair.py night
```

这条命令会启动 tmux 控制任务，依次做 CPU 检查、64 轮 GPU Smoke、Smoke 稳定性及独立物理评估验收，通过后才从 **原 Flat Checkpoint** 初始化正式 8h 微调，并启动 SwanLab 核心数据上传。GPU Smoke 和验收需要额外准备时间；正式训练预算为 8h。SSH 断开不影响 tmux 任务；电脑挂起会暂停本机计算。

当前机器没有系统 tmux 时，启动器使用已存在的 `.cache/tmux/usr/bin/tmux`。所有命令、session、socket、退出码位置写入 `logs/t15_jobs/*.json`。

查看控制任务与正式训练：

```bash
tail -f logs/s54_stair_flat6063_night_v3.log
# 正式任务通过门禁并启动后：
tail -f logs/s54_stair_flat6063_8h/stdout.log
```

控制任务失败时检查 `logs/s54_stair_flat6063_night_v3.exitcode`、`logs/s54_stair_flat6063_smoke_v3/stdout.log`、`smoke_gate_v3.json`；控制日志会附带实际 trainer 错误尾部。不会越过门禁自动开始正式训练。不要恢复 v2 的 iteration 0 Checkpoint；v3 重新从原 Flat 初始化。不要在同一目录重复启动；需要重试新的 Smoke 时修改 `smoke.json` 的 log_dir，保留旧诊断。

也可以分步执行，等待 Smoke 的 `train.exitcode` 为 0 后再运行 gate：

```bash
"$ISAAC_PYTHON" tools/run_s54_stair.py check
"$ISAAC_PYTHON" tools/run_s54_stair.py smoke
# 等 Smoke 结束，再执行；gate 包括独立 GPU Rollout：
"$ISAAC_PYTHON" tools/run_s54_stair.py gate
"$ISAAC_PYTHON" tools/run_s54_stair.py train
```

## 门禁与旧异常

v2 保存的实际失败转移来自 `stair_up_4` 第 244 帧、episode step 41。Root 速度 1.334 m/s、角速度 1.427 rad/s、关节速度峰值 20.969 rad/s，全部通过当时的速度诊断界限，状态/奖励有限。原始 Reward = -5.340873，stance 足速罚项 = -5.120515，其余奖励与罚项合计 -0.220357；除以最大正奖励 0.13 后为 -41.083637。唯一触发项是“归一化奖励 <-20”。

这是奖励硬门禁把有限的 Tracking 失败当作 Newton 数值异常。此罚项是参考 stance 下的足部 link XY 速度平方，在实际足部偏离参考时可很大；负回报没有有限下界。修复不改变奖励公式、罚项权重或参考数据，不裁剪/丢弃这条学习反馈。精确 CPU 重算、原转移哈希与修复前代码见 `repairs/smoke_v2_reward_guard/diagnosis.json` 及 `before/`；这不证明新 GPU 训练已经通过。

更早 `logs/s54_stair_t15_smoke` 的 -1384 等异常仍保留，不能仅用本条 v2 样本解释其全部原因，也不能作为收敛结果。此次按用户要求恢复 Flat 训练参数、在线统计与 Reset 设置，保持完整 source count。

非有限状态/动作/奖励，或 Root 速度 >20 m/s、角速度 >50 rad/s、关节速度 >200 rad/s，仍立即保存 `failure_physics_transition.pt` 并终止，不写入 WM replay。有限的归一化奖励 <-20 只记录 `tracking_reward_diagnostics.jsonl` 和首个 `tracking_low_reward_sample.pt`，原样进入 replay；这表示跟踪表现差，不能用作动力学成功证据。FK、地形几何、接触验收和原速度界限未放宽。

Smoke 验收检查：完整 64 轮、有限模型/训练指标和非零 Actor 梯度、真实/replay 奖励同步、原 normalizer count 不被清零/压低、无物理异常停止。真实/想象回报、WM 8-step error、Critic loss 完整保留为训练质量诊断；不以任意有限 loss 或负回报阈值判定 Newton 失败。随后独立评估三类 Flat train 和四级 Stair，Flat 每个选定动作的多 seed 平均归一化回报仍须 ≥0.85。普通 Tracking 失败、数值异常与实验流水线完成分别报告；Smoke 通过不等于楼梯收敛或泛化。

## 评估、Checkpoint 与 SwanLab

每 1000 轮冻结一份 Checkpoint，用独立 Newton 进程评估 `stair_up_4` 和 Flat train 的 `straight_flat_18 / turn_flat_10 / spin_flat_02`，seeds 0/1/2。训练期间只保存数值与轨迹，不反复导出视频。评估不会更新参数、normalizer 或采样权重。Flat test 保留隔离，本实验没有独立楼梯泛化集。

Normalized Return = 实际累计 reward /（完整 Reference 步数 × 最大正奖励 0.13）。失败后的缺失步数仍留在分母；同时保存原始 Reward Sum，可能有负回报，不裁到 [0,1]。Root XYZ、Foot Pose、动作平滑、接触和 WM 1/4/8-step 误差见各 rollout summary/trace。stance slip 指标为参考 stance 下的足部 link XY 速度代理，不能等同于接触点实测滑移。

正式输出：`logs/s54_stair_flat6063_8h/`：

- `latest.pt` / `last.pt`：完整模型与 optimizer；`ckpt_*.pt`：最近 8 份周期存档。
- `evals/iter_XXXXXX`：周期独立评估及不可变 Checkpoint；`evals/final_XXXXXX`：结束时补做评估，若同轮已有完整周期结果则复用。
- `best.pt`：仅当 Flat 三类保留阈值通过时，按 Stair **训练动作**回报选取，指向已评估的原封不动 Checkpoint。没有符合条件的评估就没有 best，不能用 test 选择模型。
- `train_log.csv`、`gradient_terms.jsonl`、`training_status.json`、`runtime_stats.json`：本地诊断。
- SwanLab 项目 **SuperMimic-S54-Stair**，每个 run 单独记录 run ID；上传核心训练配置/指标与上述独立 Eval 标量，**不上传 GIF/视频**。本机已配置过登录；若凭据过期，先在 `.venv-swanlab` 中执行 SwanLab 登录再启动。日志上传失败保留本地数据与退出码。

手动评估并导出真实物理视频（每条轨迹保存实际全部 rollout 帧，跌倒后不伪造后续成功帧）：

```bash
"$ISAAC_PYTHON" tools/run_s54_stair.py eval \
  --checkpoint logs/s54_stair_flat6063_8h/latest.pt \
  --out results/s54_stair_flat6063_8h_video_eval --video
```

新输出目录内 `attempt_001/StairTrain/` 和三个 `FlatTrain_*/` 放各 seed MP4，根目录 summary.json 包含归一化回报与原始 Reward。手动导出目录需为新的目录；训练运行时优先使用 `ckpt_XXXXXX.pt` 或 `evals/.../checkpoint.pt` 避免 latest 被更新。

只看参考运动学轨迹：

```bash
"$ISAAC_PYTHON" tools/view_s54_stair_reference.py
```

## 中断恢复

```bash
"$ISAAC_PYTHON" tools/run_s54_stair.py resume
```

恢复的是新 Stair run 的 latest，不再从 Flat 初始化；保留网络/optimizer/进度，重填真实 replay 后先进行 20 轮 WM-only recovery（与上次设置一致；同时仍遵守初始 1000 轮 WM-only 阶段）。默认仍以本次实验累计 8h 为上限（不是恢复后再送 8h）。确实需要再跑 8h 时，将 `--wall-seconds` 设为 checkpoint 已用秒数 +28800，例如已用 7200 秒：

```bash
"$ISAAC_PYTHON" tools/run_s54_stair.py resume --wall-seconds 36000
```

本轮验证证据见 `preflight.json` 与 `verification.json`；GPU Smoke/训练结果只在实际执行后生成，不用 CPU FK 或接口单测替代真实动力学验收。
