# S54 训练脚本参数说明

核对日期：2026-11-02。本文依据仓库当前参数解析和执行代码编写；表中的“默认值”指直接调用 Python 入口时的默认值。Shell 启动脚本、checkpoint 和课程 JSON 可能覆盖它们。

## 1. 先区分三个入口

| 入口 | 用途 | 使用的 Python |
|---|---|---|
| [scripts/train_4060.py](../scripts/train_4060.py) | 长时间训练监督器：启动训练子进程、管理预算和课程、同步 SwanLab | `.venv-swanlab/bin/python`；查看帮助或 dry-run 可用系统 `python3` |
| [scripts/train.py](../scripts/train.py) | 实际创建 Newton 环境、运行 RSL-RL PPO 的训练进程 | Isaac Lab 环境的 Python |
| [scripts/train_walk_retention_12h.sh](../scripts/train_walk_retention_12h.sh) | 当前 12 小时直行续训的快捷入口，内部经过 `train_walk_12h.sh` 和 `train_4060.sh` 调用监督器 | 由脚本选择 |

`train_4060.py` 的名称来自最初使用的显卡，并不将训练限制在 RTX 4060；可在安装好相应依赖的 RTX 3090 上运行。

**本文主要说明 `train_4060.py` 的参数。** 它常用连字符，例如 `--num-envs`；`train.py` 常用下划线，例如 `--num_envs`。除明确提供的别名外，两种写法不能随意互换。

服务器建议直接调用监督器，并显式传入 `--isaac-python`。目前 Shell 启动器创建日志环境时仍使用本机 Isaac Lab 路径，且没有接收 `--isaac-python` 的选项。

## 2. 你问到的六个参数

### `--start-stage straight`：从哪个任务阶段开始

可选值为 `stand`、`lift`、`step`、`walk`、`straight`，默认 `stand`。

| 值 | 任务意图 | 新建阶段时的基础前向步长 |
|---|---|---:|
| `stand` | 站立 | 0 m |
| `lift` | 原地交替抬脚 | 0 m |
| `step` | 小步前进 | 0.03 m |
| `walk` | 扩大前进步长 | 0.06 m |
| `straight` | 固定参考方向的直行，加入方向约束 | 0.03 m |

这些是 [configure_stage](../configs/s54_curriculum.py) 的基础值。完整续训会恢复 checkpoint 的任务设置，课程 JSON 可以再次覆盖步长、奖励等；因此 `straight` **不表示当前实验只训练 3 cm**。当前 retention 课程实际覆盖 20–40 cm。

选择 `stand/lift/step/walk` 时，普通课程会从对应位置继续 `stand → lift → step → walk`；选择 `straight` 时，监督器只管理该阶段。`--adaptive-plan` 和 `--command-schedule` 要求 `straight`。

当前 `straight` 保持 reset 时的参考方向，还没有可控转弯或绕圈任务。

### `--rollout-steps 64`：每次 PPO 更新前，每个环境采集多少步

默认 `64`，取正整数，传给训练进程的 `--num_steps_per_env`。

一次 PPO iteration 的流程是：所有环境各运行 T 个控制步，收集 Observation、Action、Reward、Done，再计算 GAE 和 PPO 损失，优化网络，之后重新采样。

设并行环境数为 N、rollout 长度为 T，则每次 iteration 的新 transition 数为：

```text
B = N × T
2048 × 64 = 131072 条 transition
```

当前控制周期是 0.02 s，因此 `64` 步对应每个环境 **1.28 s 的仿真推进**；若期间跌倒或超时，该环境会 reset，64 步可以跨 episode。这不是 1.28 s 的实际耗时，也不要求 episode 在第 64 步结束。

增加 T 通常增加 rollout 存储和每次更新的采样量，也延长两次策略更新之间的间隔。它与仿真频率、步态频率是不同参数。

### `--mini-batches 4`：将一次 rollout 分成多少份进行优化

默认 `4`，取正整数，传给 `--num_mini_batches`。必须满足：

```text
(num_envs × rollout_steps) % mini_batches == 0
```

这里的 `4` 是份数。使用 2048 环境、64 步 rollout 时，每份包含 `131072 / 4 = 32768` 条 transition。

当前课程的 `num_learning_epochs=4`，同一批 rollout 数据会训练 4 遍，因此一次 PPO iteration 包含 `4 × 4 = 16` 次 mini-batch 梯度更新。日志里的一次 PPO iteration 仍只增加 1，新采样量仍是 131072。

增加 mini-batch 份数会减小每次梯度计算的样本数，通常降低这部分显存需求，但增加梯度更新次数，也会改变优化行为；rollout 缓冲本身不会因此缩小。

### `--chunk-iterations 50000`：一个训练子进程最多运行多少次 PPO iteration

监督器直接调用时默认 `100`；当前 12 小时入口显式设置为 `50000`。

chunk 是一段训练进程的长度。段结束后，监督器接收 checkpoint，再按课程配置决定评估、切换阶段或启动下一段。普通课程的阶段边界、剩余更新预算和时间截止都可能提前结束这一段。

| 参数 | 控制什么 | 当前常用值 |
|---|---|---:|
| `--rollout-steps` | 一次 PPO iteration 前，每个环境采集的控制步数 | 64 |
| `--chunk-iterations` | 一段训练进程最多执行的 PPO iteration 数 | 50000 |
| `save_interval` | 训练进程内部，按 iteration 保存 checkpoint 的间隔 | 250 |

因此 `chunk-iterations=50000` 时仍会中途保存模型。当前课程 `evaluation.enabled=false`，chunk 结束也不会触发独立 eval。大 chunk 可以减少环境重建和重复起步的开销；下一段会恢复模型和优化器，但会重新创建环境，仿真轨迹不连续。

### `--normalization-prior-rollouts 0`：迁移 Actor 时，保留多少归一化历史权重

默认 `0`，可取非负整数。**当前代码只在 `--warm-start` 到 `straight` 时应用此参数**；`--resume-checkpoint` 的完整续训不会进行这项调整。

Actor 的观测归一化保存历史均值、方差和样本计数。该参数只折算历史计数：

```text
0：保留原来的 count
K > 0：new_count = min(old_count, K × num_envs × rollout_steps)
```

例如 K=10、2048 环境、64 步时，历史计数最多保留为 1310720。历史计数变小，新观测对后续均值/方差的影响通常更大。

调整时保持已有均值、方差和网络权重不变，并检查刚加载时的确定性动作完全一致。它不是额外运行 K 次 rollout，也不是给输入乘一个比例；`0` 同样不表示关闭观测归一化。当前完整续训保留 `0` 即可。

### `--swanlab-detail compact`：向 SwanLab 上传哪些曲线

默认 `compact`，可选 `compact`、`full`。

| 值 | 上传行为 |
|---|---|
| `compact` | 只上传固定的核心曲线，并将部分名称简化，便于查看 |
| `full` | 上传监督器收到的全部指标，保留阶段前缀，曲线会明显增多 |

`compact` 的主要指标包括：

| 曲线 | 含义 |
|---|---|
| `Train/return_0to1` | 当前实现的 episode 归一化回报；是奖励汇总指标，不是物理行走成功率 |
| `Train/reward_per_step` | 平均每个控制步的训练奖励 |
| `Train/episode_steps` | 平均 episode 长度；需结合超时上限判断 |
| `Train/fall_per_control_step` | 控制步中 terminated 的比例，不是整个 episode 的跌倒概率 |
| `Startup/left_success`、`Startup/right_success` | 左右起步的首步成功统计 |
| `Startup/left_xy_error_m`、`Startup/right_xy_error_m` | 左右首步 XY 跟踪误差，单位 m |
| `PPO/learning_rate`、`PPO/value_loss`、`PPO/action_std` | 学习率、价值损失和动作分布标准差 |
| `Reward/*` | Tracking、摆动相位、支撑脚滑动/跳动、身体运动、姿态、动作变化和朝向分项 |
| `Coverage/cell_fraction`、`Coverage/min_cell_transitions` | 实际 rollout 对课程网格的覆盖 |
| `GPU/memory_MiB`、`GPU/utilization_percent` | 显存占用与 GPU 利用率 |

指标筛选定义见 [swanlab_metrics.py](../scripts/swanlab_metrics.py)。训练进程的完整原始指标仍保存在本地 `training/<chunk>/metrics.jsonl`；`compact` 不会删掉这些本地数据。关闭 eval 时不会凭空生成 eval 曲线。

## 3. 监督器完整参数表：`scripts/train_4060.py`

### 运行环境和显卡探测

| 参数 | 直接调用默认值 | 可选值 / 意义 |
|---|---|---|
| `--isaac-python PATH` | `/mnt/sage_data/workspace/IsaacLab_3_0/.venv/bin/python` | Isaac Lab Python 文件路径。服务器必须按实际安装位置指定 |
| `--num-envs N` | `auto` | `auto` 或正整数；显式指定数量会跳过自动探测 |
| `--probe-envs N ...` | `256 512 1024 2048 3072 3456 4096` | 自动探测候选环境数 |
| `--probe-iterations N` | `5` | 每个候选实际执行的短训更新次数 |
| `--probe-timeout SEC` | `300` | 每个候选短训的超时秒数 |
| `--target-vram FRACTION` | `0.85` | 探测时的显存占比预算，要求 `0 < 值 < 1` |
| `--reserve-mib MIB` | `768` | 探测时保留的显存，至少 256 MiB |
| `--benchmark-only` | 关闭 | 只运行实际 PPO 短训测速，不进入长训、不初始化 SwanLab；会生成探测记录 |

探测预算为 `min(总显存 × target_vram, 总显存 − reserve_mib)`。在合格候选中，优先选吞吐不低于最快候选 90% 的最大环境数。这些参数用于选环境数量，不会限制显卡功率；显式 `--num-envs 2048` 时也不会自动按预算缩减环境数。

### PPO 更新和训练预算

| 参数 | 直接调用默认值 | 可选值 / 意义 |
|---|---|---|
| `--rollout-steps N` | `64` | 每环境每次更新前采集的控制步数，正整数 |
| `--mini-batches N` | `4` | 每轮优化拆分份数，正整数，须整除 rollout 样本数 |
| `--max-hours H` | `12` | 正数，总墙钟预算，单位小时 |
| `--max-iterations N` | `20000` | 本次实验累计 PPO iteration 上限，包含各阶段，不把旧 checkpoint 的历史 iteration 计为本次新增更新 |
| `--chunk-iterations N` | `100` | 单个训练子进程的 iteration 上限，正整数 |
| `--save-interval N` | 未指定，PPO 配置默认 `250` | 每 N 次 PPO iteration 保存 checkpoint；正整数，别名 `--save_interval` |
| `--learning-rate LR` | 未指定 | 显式覆盖学习率，要求 `0 < LR < 0.01`；adaptive plan 使用阶段内的 `learning_rate` |
| `--normalization-prior-rollouts N` | `0` | Actor 迁移时的归一化历史计数折算，见第 2 节 |
| `--seed N` | `42` | 随机种子，不能单靠它保证 GPU 物理逐位可复现 |
| `--print-every N` | `10` | 每 N 次更新打印一次终端摘要；段首等位置也会打印 |

时间和更新数是两个上限，先达到哪个就按哪个停止；课程完成也可能提前结束。墙钟预算包含训练监督器运行期间的启动、探测、训练和评估耗时；截止信号之后，模型保存与 SwanLab 收尾可能继续占用少量时间。

使用命令课程 `--command-schedule` 且 `--schedule-tail stop` 时，代码会将更新上限设为计划中 `updates` 的总和。使用 `hold-last` 时，`--max-iterations` 必须大于该总和。

### 课程、任务和消融

| 参数 | 直接调用默认值 | 可选值 / 意义 |
|---|---|---|
| `--start-stage STAGE` | `stand` | `stand / lift / step / walk / straight` |
| `--step-length-range MIN MAX` | 未指定 | 每个 episode 的前向步长范围，单位 m；仅 `straight` 支持，要求有限且 `0 <= MIN <= MAX` |
| `--command-schedule JSON` | 未指定 | 按更新次数切换步长/频率的 schema 1 课程 |
| `--adaptive-plan JSON` | 未指定 | schema 2 混合命令课程；可按物理 eval 晋级，也可通过 JSON 关闭 eval，持续单阶段训练 |
| `--schedule-tail VALUE` | 未指定，处理后为 `stop` | `stop`：课程更新数完成即停；`hold-last`：末阶段继续，受时间/更新上限约束 |
| `--ablation NAME` | `none` | 见下表；完整续训中的奖励/控制仍可能被 checkpoint 或 profile 覆盖 |
| `--continue-after-pass` | 关闭 | 普通课程最后阶段连续通过后仍继续训练至预算；adaptive 入口另按其课程逻辑运行 |

| 消融值 | 基础配置中的作用 |
|---|---|
| `none` | 使用正常课程 |
| `legacy_reward` | 使用早期奖励配置 |
| `legacy_scale` | 取消逐关节 action scale，回到旧动作缩放 |
| `no_smoothing` | 动作平滑系数设为 1，取消平滑 |
| `targets_only` | 仅 `straight`：保留参考方向目标，关闭新增方向奖励 |
| `rewards_only` | 仅 `straight`：保留方向奖励，关闭固定参考方向目标 |

`--command-schedule` 和 `--adaptive-plan` 互斥。两者管理命令分布时，不能再同时传 `--step-length-range` 或六个频率值/范围参数；要改分布应编辑对应 JSON。`--adaptive-plan` 不允许 `--warm-start`，推荐以 `--resume-checkpoint` 完整续训。

### 左右起步和步频：训练、play、eval 共用参数

这些参数由 [gait_cli.py](../scripts/gait_cli.py) 定义，同时支持将连字符替换为下划线的别名。默认均为“未指定”，由阶段、checkpoint 或课程恢复。

| 参数 | 可选值 / 意义 |
|---|---|
| `--initial-swing-side SIDE` | `left / right / random`；random 在每次 reset 以各约 50% 概率选起步脚 |
| `--gait-frequency F` | 设置左右共同的固定频率 |
| `--gait-frequency-range MIN MAX` | 共同随机频率：reset 时采样一次并赋给左右两侧；显式单侧参数可再次覆盖 |
| `--left-gait-frequency F` | 单独指定左脚摆动半周期的频率参数 |
| `--right-gait-frequency F` | 单独指定右脚摆动半周期的频率参数 |
| `--left-gait-frequency-range MIN MAX` | 左侧每个 episode 采样范围 |
| `--right-gait-frequency-range MIN MAX` | 右侧每个 episode 采样范围 |
| `--observe-gait-frequency` | 启用左右两个频率输入；通常得到 105 维观测。显式设置频率也会自动启用 |
| `--no-observe-gait-frequency` | 显式关闭频率观测；当前迁移器不支持将 105 维权重直接缩回 103 维 |

频率均要求有限正数，范围下限不大于上限。同一侧固定值和随机范围不能同时给出。左右单侧范围会分别采样；共同 `--gait-frequency-range` 在没有单侧覆盖时让左右同频。JSON 分组则通过 `coupled=true` 让该组左右同频。

这里的 gait frequency 按半周期时长定义：

```text
左摆动时长 T_L = 1 / (2 × f_L)
右摆动时长 T_R = 1 / (2 × f_R)
完整左右周期频率 = 1 / (T_L + T_R)
                  = 2 × f_L × f_R / (f_L + f_R)
```

当两侧都是 `1 Hz` 时，每侧摆动约 0.5 s，左右完整周期约 1 s，每秒交替迈步约 2 次。`2 Hz` 则对应每秒交替迈步约 4 次。以上按仿真时间计算；可视化窗口的实际播放速度还取决于运行速度。

### Checkpoint、输出和恢复

| 参数 | 默认值 | 意义 |
|---|---|---|
| `--resume-checkpoint FILE` | 未指定 | 从同阶段 checkpoint 完整恢复 Actor、Critic、优化器、归一化和训练 iteration，建立一个新实验 |
| `--warm-start FILE` | 未指定 | 只迁移 Actor 及其归一化；Critic、优化器和 iteration 重新开始 |
| `--resume-dir DIR` | 未指定 | 恢复已有实验的 `state.json`、课程快照和 SwanLab 身份，继续其进度 |
| `--output-dir DIR` | `logs/long_runs/<时间戳>_s54_4060` | 新实验的输出目录；必须尚不存在 |
| `--dry-run` | 关闭 | 校验参数并打印计划；不启动仿真/训练、不创建实验目录、不上传。必须显式指定环境数 |
| `-h`、`--help` | — | 查看入口支持的参数并退出 |

`--warm-start` 与 `--resume-checkpoint` 互斥；`--resume-dir` 不能和两者或 `--output-dir` 同时使用。

对于命令课程和 adaptive plan，`--resume-dir` 会继承已用墙钟时间；未显式改 `--max-hours` 时使用原总预算，而不是每次再增加 12 h。它还恢复原 rollout、mini-batch、随机种子和日志身份等设置，不适合用来静默更换任务分布。更换课程、模型或实验应使用 `--resume-checkpoint` 和新目录。

checkpoint 的伴随 `env_cfg.json` 优先用于恢复任务，迁移服务器时应与 `.pt` 一起保留。`agent_cfg.json` 是核对 PPO 配置的快照；不要假设其中每个字段都会自动覆盖当前训练代码。

### 评估与 SwanLab

| 参数 | 默认值 | 可选值 / 意义 |
|---|---|---|
| `--eval-envs N` | `64` | 独立评估的环境数量，正整数 |
| `--eval-steps N` | `1000` | 独立评估控制步数，要求至少 1000；50 Hz 下为 20 s 仿真推进 |
| `--eval-every-chunks N` | `1` | 普通课程每 N 个 chunk 评估一次；最终更新边界也可能评估 |
| `--eval-frequency-corners` | 关闭 | 普通课程额外评估左右频率范围的角点组合 |
| `--project NAME` | `S54-Locomotion` | SwanLab 项目名 |
| `--workspace NAME` | 未指定 | SwanLab 工作空间；未指定时由 SDK 按账号解析 |
| `--mode MODE` | `online` | `online`：在线上传；`offline`：本地记录，不在线同步 |
| `--swanlab-detail DETAIL` | `compact` | `compact / full`，见第 2 节 |

adaptive 入口的评估节奏由 JSON 的 `evaluation`、`full_eval_every_chunks` 等控制，不要用普通课程的 eval 参数推断它的行为。当前 `walk_retention_12h.json` 明确关闭全部 eval；即使传入 `--eval-envs`，也不会开启评估。

## 4. 当前 12 小时配置实际用了什么

配置来源：[walk_retention_12h.json](../configs/walk_retention_12h.json)、[PPO 配置](../agents/rsl_rl_ppo_cfg.py) 和上一条服务器启动命令。

| 项目 | 当前显式设置 / 实际含义 |
|---|---|
| 起点 | `pretrained/s54_walk_12h_20261102/model.pt`，完整续训，105 维观测、27 维动作 |
| 任务 | `straight`，阶段名 `retention` |
| 环境 / rollout / mini-batch | 2048 / 64 / 4；131072 条新样本/iteration |
| Epoch / 学习率 | 4 / fixed `2e-5`，学习率取 JSON 阶段值 |
| 预算 | 最多 12 h，最多 50000 次新增 iteration；单 chunk 最多 50000 次 |
| Episode | 10 s = 500 控制步，跌倒可提前结束 |
| 保存 | `save_interval=250`；正常段末保存 `model_final.pt`，收到截止/中断信号时尝试保存 `model_interrupted.pt` |
| SwanLab | `online`、`compact` |
| Eval | `evaluation.enabled=false`，基线、周期和末尾均不执行 |
| 起步脚 | 每次 reset 随机选择左右脚 |
| 基础组 | 20% 环境：20–25 cm，左右独立 1–1.2 Hz |
| 网格组 | 70% 环境：25–40 cm × 1–2 Hz，共 15 格；每格固定配额，左右同频 |
| 网格重点 | 30–35 cm、1.4–1.8 Hz 两格合计占全部环境 40% |
| 角点组 | 10% 环境：40 cm、左右 2 Hz |
| Episode 内换命令 | 关闭；只在 reset 时重采样，不代表已经训练过连续变化命令 |

当前 `minimum_updates=250`、`passes_required=2` 是通用课程结构里的晋级条件。**关闭 eval 后，它们不要求额外执行更新、不触发评估，也不代表训练 250 次就结束。**

要修改保存频率，可在监督器或 Shell 命令后添加 `--save-interval 100`，表示按每 100 次 PPO iteration 的间隔保存 checkpoint。直接运行 `train.py` 时也支持 `--save_interval 100`。未指定时使用 `agents/rsl_rl_ppo_cfg.py` 的 `save_interval=250`。恢复同一实验时，未指定则继承原 CLI 保存间隔，显式指定则覆盖。`--print-every` 只控制终端摘要，`--chunk-iterations` 控制训练分段，二者都不改变内部周期保存间隔。

RSL-RL 按训练 iteration 索引的整数倍进行周期保存；加载旧模型后沿用其 iteration 索引，因此首个周期保存不一定刚好发生在续训后的第 N 次更新。正常结束与可处理的中断仍会分别保存 `model_final.pt` / `model_interrupted.pt`，不受周期保存间隔限制。正在运行的进程不会自动读取新 CLI 或文件修改，新的间隔在下次启动/恢复时生效。

### 配置覆盖顺序

任务侧：环境默认值 → `configure_stage` → 完整续训恢复的任务配置 → `--task-profile` → 显式步频/步长 CLI。adaptive plan 会生成 `--task-profile`，且入口禁止同时传入直接步长/频率覆盖，以免混淆。

优化器恢复可能带回旧学习率，因此训练代码在 load 之后再次应用显式学习率。adaptive plan 会把阶段的 `learning_rate` 传给子进程；该路径下它优先于监督器的 `--learning-rate`。

实际运行请核对 `launcher.json`、`profiles/`、训练目录中的 `env_cfg.json`、`agent_cfg.json` 和 `manifest.json`，不要只看参数类中的默认值。

## 5. 直接调用 `scripts/train.py` 时的参数

通常由监督器生成这些参数。它本身不会像监督器那样自动上传 SwanLab；长训建议使用前面的监督器入口。

| 参数 | 默认值 | 意义 |
|---|---|---|
| `--num_envs N` | `16` | 并行环境数 |
| `--max_iterations N` | `3` | 本次进程额外执行的 PPO iteration 数，不是加载后 iteration 的绝对目标值 |
| `--run_name NAME` | `smoke` | 自动生成日志目录时的运行名后缀 |
| `--stage NAME` | `legacy` | `legacy / stand / lift / step / walk / straight`；legacy 不应用课程配置 |
| `--ablation NAME` | `none` | 与监督器相同的六种消融值 |
| `--seed N` | `42` | 随机种子 |
| `--log_root DIR` | 未指定 | 自动日志目录的根目录，默认根为 `logs/rsl_rl` |
| `--output_dir DIR` | 未指定 | 显式新运行目录，优先于自动命名 |
| `--resume FILE` | 未指定 | 同阶段完整恢复；不同阶段应使用 warm start |
| `--warm_start FILE` | 未指定 | 只加载 Actor，与 `--resume` 互斥 |
| `--num_steps_per_env N` | 未指定 | 覆盖 rollout 长度；课程通常为 64，legacy 配置基础值为 32 |
| `--num_mini_batches N` | 未指定 | 覆盖 PPO mini-batch 份数，基础值为 4 |
| `--save_interval N` | 未指定，配置默认 `250` | 每 N 次 PPO iteration 保存 checkpoint；正整数，别名 `--save-interval` |
| `--learning_rate LR` | 未指定 | load 优化器后再次覆盖学习率 |
| `--normalization_prior_rollouts N` | `0` | 仅 straight 的 Actor 迁移使用，含义见第 2 节 |
| `--normalization_warmup_steps N` | `0` | 从零训练时额外收集观测来预热归一化；加载 checkpoint 时不运行 |
| `--step_length_range MIN MAX` | 未指定 | `straight` 步长范围，单位 m；别名 `--step-length-range` |
| `--task-profile JSON` | 未指定 | 恢复 checkpoint 任务后，显式应用混合命令、奖励和随机化配置 |
| `--metrics_jsonl` | 关闭 | 将完整训练指标写成 JSONL，供监督器读取；也保留 TensorBoard 事件 |
| `--validate_only` | 关闭 | 初始化并验证配置、模型与动作输出，执行 0 次 PPO 更新；会创建验证运行目录 |
| `-h`、`--help` | — | 查看完整帮助并退出 |

第 3 节的九类起步/步频开关同样适用于此入口。Isaac Lab 还注入启动器参数，常用的是 `--device cuda:0`、`--viz none`、`--viz newton_gl` 和 `--max_visible_envs N`；其余渲染、直播、XR 和录制参数以所安装版本的 `train.py --help` 为准。监督器当前固定给训练子进程传 `--viz none`，没有提供 `--device` 或 `--viz` 的同名透传选项。

## 6. Shell 入口支持的选项

`train_walk_retention_12h.sh` 向 `train_walk_12h.sh` 转发参数，后者接受：

| Shell 选项 | 默认值 / 作用 |
|---|---|
| `--checkpoint FILE` | retention 入口默认仓库内预训练模型；映射为监督器 `--resume-checkpoint` |
| `--hours H` | `12`；映射为 `--max-hours` |
| `--num-envs N` | `2048` |
| `--chunk-iterations N` | `50000` |
| `--save-interval N` | 未指定，配置默认 `250`；支持别名 `--save_interval` |
| `--output-dir DIR` | 自动生成 `logs/walk/` 下的时间戳目录 |
| `--resume-dir DIR` | 恢复同一实验；不与 checkpoint 或新输出目录并用 |
| `--schedule JSON` | 通用入口默认 `configs/walk_4stage_12h.json`；映射为 `--command-schedule` |
| `--adaptive-plan JSON` | retention 入口预设 `configs/walk_retention_12h.json`；与 schedule 互斥 |
| `--tail VALUE` | 通用命令课程默认 `hold-last`；对应 `--schedule-tail`，不与 adaptive plan 并用 |
| `--swanlab-detail VALUE` | `compact`；可选 `compact / full` |
| `--eval-envs N` | `64` |
| `--eval-steps N` | `1000` |
| `--dry-run` | 只预览，不安装日志 SDK、不启动训练 |
| `--help`、`-h` | 显示 Shell 参数帮助 |

Shell 入口还预设了 64 步 rollout、4 个 mini-batch 和 50000 次总更新上限。它不接受所有监督器选项，例如 `--rollout-steps`、`--isaac-python`；需要这些选项时直接调用 `train_4060.py`。

## 7. 可复制的使用示例

以下命令在项目根目录执行。`ISAAC_PYTHON` 是我们自行定义并传入 CLI 的 Shell 变量，脚本不会自动读取它。请将路径改为服务器实际安装位置；双引号中的 `~` 不会展开，使用 `$HOME`。

### 首次准备 SwanLab 环境

```bash
export ISAAC_PYTHON="$HOME/workspace/IsaacLab_3_0/.venv/bin/python"
"$ISAAC_PYTHON" --version

# 仅首次创建，已经存在可用日志环境时不用重复创建。
uv venv .venv-swanlab --python "$ISAAC_PYTHON"
uv pip install --python .venv-swanlab/bin/python 'swanlab==0.10.3'
.venv-swanlab/bin/swanlab login
```

### 按当前配置完整续训 12 小时

```bash
.venv-swanlab/bin/python -u scripts/train_4060.py \
  --isaac-python "$ISAAC_PYTHON" \
  --start-stage straight \
  --resume-checkpoint pretrained/s54_walk_12h_20261102/model.pt \
  --adaptive-plan configs/walk_retention_12h.json \
  --num-envs 2048 \
  --rollout-steps 64 \
  --mini-batches 4 \
  --max-hours 12 \
  --max-iterations 50000 \
  --chunk-iterations 50000 \
  --save-interval 250 \
  --normalization-prior-rollouts 0 \
  --project S54-Locomotion \
  --mode online \
  --swanlab-detail compact \
  --output-dir "logs/walk/$(TZ=Asia/Shanghai date +%Y-%m-%d_%H-%M-%S)_3090"
```

首次检查可在这条命令最后加 `--dry-run`：只输出计划，不会启动训练或上传。它验证的是参数和计划，不会验证 CUDA、物理引擎或模型运行效果。

若要试训半小时，将 `--max-hours 12` 改成 `--max-hours 0.5`，并使用新的输出目录。

### 中断后恢复同一实验

将下面的目录换为实际实验根目录，该目录应包含 `state.json`、`launcher.json`。

```bash
.venv-swanlab/bin/python -u scripts/train_4060.py \
  --isaac-python "$ISAAC_PYTHON" \
  --resume-dir logs/walk/你的实验目录 \
  --max-iterations 50000 \
  --chunk-iterations 50000 \
  --swanlab-detail compact
```

命令课程/adaptive 实验恢复时会继承原总时限和已用时间。如果原 12 小时预算已经用完，要新开一轮续训，应使用新的 `--resume-checkpoint` 和输出目录。

### 随时查看当前代码的参数帮助

```bash
python3 scripts/train_4060.py --help
"$ISAAC_PYTHON" scripts/train.py --help
bash scripts/train_walk_12h.sh --help
```

当前课程训练的是平地直行。评估实际效果仍需检查左右脚落点误差、首步表现、存活率、支撑脚滑动/跳动等；归一化回报接近 1 不等于所有命令都已走稳。
