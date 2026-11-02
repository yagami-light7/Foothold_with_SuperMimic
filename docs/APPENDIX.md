# APPENDIX：S54 足印跟踪实验配置

记录日期：2026-11-01；更新：2026-11-02。正文采用中文说明，章节沿用用户指定的 A–D 结构。

本文记录 **KUAVO S54 / Newton + MJWarp / RSL-RL PPO** 的项目实现。以 `walk_precision_1h` 的完整参数为基准，记录已完成的 `walk_retention_30m` 及已配置的 `walk_retention_12h` 续训覆盖，同时列出已有 `flat_walk_v2` 基线和可选抗扰动阶段。结构参考 [Mind Your Steps 附录 A–D](https://arxiv.org/pdf/2606.08253#page=13)；机器人参数及本项目公式以代码和实验快照为准。

### precision 基准实验标识与配置来源

| 项目 | 当前记录 |
|---|---|
| 主训练入口 | `scripts/history_train/train_walk_precision_1h.sh`（本地归档） |
| 训练计划 | [configs/walk_precision_1h.json](../configs/walk_precision_1h.json)，`precision` 阶段 |
| 奖励配置类型 | `FlatWalkRewardCfg`，`profile="flat_walk_v2"`，附加 precision 显式覆盖 |
| 起点 checkpoint | `logs/walk/2026-11-01_17-59-35_walk_hard_region_v2/training/00003_bridge/model_final.pt` |
| 起点 SHA-256 | `e1cd33cdd47c81b31c3db241b76d4bc63255abfeeb8e1c0d81b94d2879367274` |
| 恢复方式 | `full_resume`：actor、critic、优化器和观测归一化统计；105→105，无输入维度扩展 |
| 仿真 / 控制周期 | 0.002 s / 0.020 s，即 500 Hz / 50 Hz，decimation=10 |
| 机器人资产 / 碰撞 | `assets/biped_s54_colored.usda`，自由基座，自碰撞开启 |
| 接触求解配置 | MJWarp，`nconmax=256`、`njmax=512`；当前快照迭代 100、线搜索 50、tolerance=`1e-6` |
| 并行环境 / 硬件 | 2048 / RTX 4060 Laptop GPU，约 8 GB 显存 |
| 地形 / 环境间距 | 平地，目标足底高度 0 m；环境原点间距 3 m |
| 当前训练时限 | 总墙钟预算约 1 h，包含训练、进程启动、评估；保存及上传收尾可能额外耗时 |
| 当前可核查快照 | [env_cfg.json](../reports/walk_precision_1h_2026-11-01/ppo_final_smoke/env_cfg.json)、[agent_cfg.json](../reports/walk_precision_1h_2026-11-01/ppo_final_smoke/agent_cfg.json)、[manifest.json](../reports/walk_precision_1h_2026-11-01/ppo_final_smoke/manifest.json) |

上述快照来自最终配置的 **2 次 PPO 更新短测**，用于核对配置与恢复方式；完整训练的结果应查对应运行目录。参数配置、短测通过和策略已经掌握任务是不同的记录事项。

快照中的软件版本：Python 3.12.15、PyTorch 2.11.0+cu130、Warp 1.17.0、Newton 1.6.1、RSL-RL 5.5.1。Isaac Lab 工作目录名为 `IsaacLab_3_0`，该环境的 Python distribution 元数据为 `isaaclab==27.2.0`，因此同时保留目录名与元数据版本。MJWarp 后端配置不保证位级确定性。

实际参数的读取顺序是：配置类 → `configure_stage("straight")` → checkpoint 任务快照 → 本轮 task profile / CLI 显式覆盖。优化器恢复后再次应用显式学习率。复现实验时应保留运行目录的 `launcher.json`、`profiles/`、`env_cfg.json`、`agent_cfg.json`、`manifest.json` 与 checkpoint。

### 已完成：walk_retention_12h（2026-11-02）

从刚结束的半小时实验 `2026-11-02_00-32-11_walk_retention_30m/training/00000_retention/model_interrupted.pt` 完整续训，**仅将 `first_swing_xy_multiplier` 从1.5提高到2.0**。常规摆动倍率仍为2，因此首步总倍率为4，首步后的总倍率仍为2；D节的首步系数据此从 $6u^2$ 覆盖为 $8u^2$。其余奖励、105维观测、控制和下述15格采样沿用半小时配置。

预算12h，2048环境，LR=2e-5 fixed、episode10s；不运行基线、周期或末尾eval，SwanLab online/compact，每250更新保存。实际完成13,333次更新、1,747,582,976条transition，15格每次rollout均完整。训练结束后追加同条件确定性评估：30cm/1.5Hz、35cm/1.7Hz、40cm/2Hz的起步后末段XY分别从2.78/4.45/11.74cm降到1.40/3.26/9.64cm；三点均无跌倒。首步综合成功率仍为0且误差变大，首步倍率2.0没有达到原先起步改善的目标。各32环境、两个10s episode、单seed结果的范围及细节见[结果报告](../reports/walk_retention_12h_results_2026-11-02/REPORT.md)。

配置：[walk_retention_12h.json](../configs/walk_retention_12h.json)；入口：[train_walk_retention_12h.sh](../scripts/train_walk_retention_12h.sh)；[配置校验记录](../reports/walk_retention_12h_2026-11-02/REPORT.md)。

### 已完成短训配置：walk_retention_30m（2026-11-02）

本轮从 precision 的 **500更新候选** 继续；1000更新最终模型在角点补测发生退化，详情见 [结果评审](../reports/precision_review_2026-11-01/REPORT.md)。下列字段覆盖后文 precision 基准值，观测、奖励、控制接口及关闭的域随机化配置沿用候选。

| 项目 | retention 当前值 |
|---|---|
| 配置 / 入口 | [walk_retention_30m.json](../configs/walk_retention_30m.json) / `scripts/history_train/train_walk_retention_30m.sh`（本地归档） |
| 起点 | `logs/walk/2026-11-01_22-30-22_walk_precision_1h/training/00000_precision/model_final.pt`，完整恢复PPO，105维观测 |
| 总墙钟预算 / 环境数 | 0.5 h（含启动、训练和日志）/ 2048；截止后保存及SDK退出可能另需数十秒 |
| 学习率 / 训练 episode | fixed `2e-5` / 10 s = 500控制步 |
| 基础组 | 20%，410环境，20–25cm，左右独立1–1.2Hz |
| 中高难度网格 | 25–40cm × 1–2Hz；5cm × 0.2Hz划分15格，各格左右同频；总1433环境 |
| 网格重点 | 30–35cm、1.4–1.8Hz占全部环境约40%（819环境）；其余13格均有固定配额 |
| 角点组 | 10%，205环境，固定40cm、左右2Hz，另计于15格之外 |
| 采样保证 | 固定分组且reset不换组；每格至少41环境，每64步rollout至少2624条transition |
| 评估 | `evaluation.enabled=false`；关闭启动基线、周期和末尾eval，预留时间为0 |
| 更新段 / 保存 | 同一进程最多50000更新，受30分钟墙钟预算提前截断；仍每250更新保存，截止保存interrupted |
| 资格与模型选择 | 本轮不晋级、不自动选择最佳模型，`minimum_updates` / `passes_required`仅保留通用结构，不触发额外训练 |

网格覆盖是**命令样本覆盖**，不能解释成15格都走稳或都走满10秒。左右独立高频组合的三维网格不在本轮覆盖承诺中；基础组保留独立左右频率采样。每次reset在原格内均匀重采样，指令预热和周期内重采样关闭。频率沿用完整左右周期Hz的定义。

训练日志直接核对每个控制步实际步长、左右频率及同频约束；监督器汇总为每rollout每格transition数。缺失日志、格外命令或漏格会停止训练并留下记录。SwanLab仅增加 `Coverage/cell_fraction`（应为1）和 `Coverage/min_cell_transitions`（默认应≥2624）；逐格原始统计在 `coverage.jsonl`，累计值在 `coverage.json`。固定角点及基础组也检查实际采样数。

无eval时不会生成基线对比、自动推荐或“已通过行走资格”的结论。训练后按需另做固定seed物理评估，再与500更新起点比较角点存活、落点误差和左右首步；不能仅凭训练回报判断改进。本轮保留 D 节 precision 的全部奖励系数，后续再分别消融首步系数、摆动XY系数和宽核。配置及短测记录见 [retention 报告](../reports/walk_retention_30m_2026-11-02/REPORT.md)。

## A. Observations for Actor and Critic Networks

### A.1 输入向量与索引

当前 actor 和 critic 都读取 `obs_groups={"actor":["policy"], "critic":["policy"]}`，输入同为 **105 维**。两个网络分别拥有自己的参数和经验归一化模块。

令机身坐标系为 B、当前支撑足底坐标系为 S，名义关节位置为 $q_0$。索引列使用零基半开区间 `[start,end)`，对应 Python 切片 `start:end`。

| 输入 | 符号 / 含义 | 维度 | 索引 | Actor | Critic | precision 观测噪声 |
|---|---|---:|---|:---:|:---:|---|
| 重力投影 | $g^B$，单位重力方向在机身系的投影 | 3 | `[0,3)` | ✓ | ✓ | 关闭 |
| 机身角速度 | $\omega^B$，rad/s | 3 | `[3,6)` | ✓ | ✓ | 关闭 |
| 关节位置偏移 | $q-q_0$，rad | 27 | `[6,33)` | ✓ | ✓ | 关闭 |
| 关节速度 | $\dot q$，rad/s | 27 | `[33,60)` | ✓ | ✓ | 关闭 |
| 上次执行动作 | 上一控制步经过裁剪、平滑后的无量纲动作 | 27 | `[60,87)` | ✓ | ✓ | 不加噪声 |
| 左右节奏命令 | $(f_L,f_R)$，Hz | 2 | `[87,89)` | ✓ | ✓ | 不加噪声 |
| 左脚目标位置 | ${}^S p_L^*$，m | 3 | `[89,92)` | ✓ | ✓ | 不加噪声 |
| 左脚目标姿态 | ${}^S q_L^*$，四元数 XYZW | 4 | `[92,96)` | ✓ | ✓ | 不加噪声 |
| 右脚目标位置 | ${}^S p_R^*$，m | 3 | `[96,99)` | ✓ | ✓ | 不加噪声 |
| 右脚目标姿态 | ${}^S q_R^*$，四元数 XYZW | 4 | `[99,103)` | ✓ | ✓ | 不加噪声 |
| 步态相位 | $(\cos 2\pi\phi,\sin 2\pi\phi)$ | 2 | `[103,105)` | ✓ | ✓ | 不加噪声 |
| **合计** | 身体/动作历史 87 + 节奏 2 + Goal 16 | **105** | `[0,105)` | ✓ | ✓ | |

角度与速度未设置额外的逐分量固定缩放系数；拼接后的输入使用网络的经验均值/方差归一化。当前没有观测历史堆叠，history length=1；动作历史为一个控制步。两网络均为 `105 → 256 → 128` 的 ELU MLP；actor 输出 27 维高斯均值，critic 输出一个状态价值。

当前 critic 没有额外的基座线速度、接触力、随机化参数或地形特权输入。奖励所需的基座高度、足底速度、接触状态以及 `first_swing` 来自仿真状态，不会自动成为 critic 输入。若以后启用观测噪声，同一 `policy` 向量中的噪声也会进入 critic。论文采用非对称 actor–critic，critic 包含额外无噪声状态；本项目当前实现与此不同。[论文附录 A](https://arxiv.org/pdf/2606.08253#page=13)

### A.2 Goal 坐标系、足底定义与节奏

目标位姿缓存在世界系，每次观测转换到当前支撑足底系：

$$
{}^S p_i^*=R(q_S)^\top(p_i^*-p_S),\qquad
{}^S q_i^*=q_S^{-1}\otimes q_i^*,\quad i\in\{L,R\}.
$$

四元数顺序统一为 **XYZW**，相对四元数做半球修正使 $q_w\ge0$。平地模式的世界目标姿态由 episode reset 时的机身参考 yaw 生成；目标水平偏移沿该参考方向，步态过程中参考 yaw 保持不变。目标位置以实际支撑脚位置为起点，左右横向偏移分别为 +0.19 m / −0.19 m，竖直分量固定为地面高度。`retain_stance_targets=True`，落脚后保留该脚原目标，使支撑脚偏离原落点时仍有误差信号。

足底参考点是脚链接系中的 `(0.048838, 0, -0.0595)` m，即脚尖/脚跟中央标记的中点。奖励使用的足底点速度包含转动补偿：

$$
v_{sole}=v_{link}+\omega_{link}\times R(q_{link})r_{sole}.
$$

$\phi\in[0,0.5)$ 为左脚摆动，$\phi\in[0.5,1)$ 为右脚摆动。左右节奏分别定义为：

$$
T_L=\frac{1}{2f_L},\quad T_R=\frac{1}{2f_R},\quad
f_{cycle}=\frac{1}{T_L+T_R}=\frac{2f_Lf_R}{f_L+f_R}.
$$

等频 $f_L=f_R=f$ 时，f 是完整左右周期的频率，每秒交替迈步次数约为 2f。本项目步长 d 是目标相对支撑脚的前向偏移；`feet_distance=0.19` 是横向足距。日志中的 $2d f_{cycle}$ 是名义前进速度估算，当前观测没有独立的线速度/转向速度命令。

reset 时以 0.5 概率选择左右起步，初始相位分别为 0 或 0.5。`completed_half_cycles` 从 0 开始，每次换脚加 1；等于 0 时认定为首步。该标志只供奖励和统计使用。当前 precision 不使用命令 warmup 或 hold-still 模式。

旧格式兼容：103 维不含节奏，104 维含单个节奏，105 维含左右两个节奏。当前起点和目标都是 105 维，因此不触发网络扩展；跨维度恢复需另用项目的 checkpoint 迁移逻辑。

实现：[observations.py](../tasks/foot_placement/observations.py)、[spec.py](../tasks/foot_placement/spec.py)、[goal_sampler.py](../tasks/foot_placement/goal_sampler.py)、[gait.py](../tasks/foot_placement/gait.py)、[foot_frames.py](../tasks/foot_placement/foot_frames.py)、[PPO 配置](../agents/rsl_rl_ppo_cfg.py)。

## B. Domain Randomization

### B.1 本轮启用项与可选抗扰动配置

precision 本轮关闭物理参数、复位状态和观测噪声随机化，保留任务命令与起步脚随机采样。下表的 `robust` 列来自 `walk_hard_region.json` 的可选阶段；是否实际进入过该阶段，应查运行记录。

PPO 的高斯动作探索仍然开启，动作分布与恢复方式见 C 节。

记 $U[a,b]$ 为均匀分布，$\mathcal N(0,\sigma^2)$ 为高斯噪声。长度、角度、速度单位分别为 m、rad、m/s 或 rad/s。

| 随机项 | 采样方式与时机 | precision 当前值 | 可选 robust 阶段 |
|---|---|---|---|
| 身体质量及惯量 | 每个环境一个共同倍率，构建时采样一次，全部刚体质量/惯量同比缩放 | 关闭，倍率 1 | $U[0.97,1.03]$ |
| 足部接触材质 | 构建时设置静/动摩擦，64 档材质，保持动摩擦≤静摩擦 | 不额外随机化；地面静/动摩擦配置均 0.5，恢复系数 0 | 足部静/动摩擦范围 `[0.45,0.55]` |
| 复位关节位置 | 每关节加均匀偏移，再裁剪到软限位；每次 reset | 0 | $U[-0.004,0.004]$ rad |
| 复位机身水平速度 | 世界系 x/y 各加均匀偏移；每次 reset | 0 | 每轴 $U[-0.025,0.025]$ m/s |
| 重力投影观测 | 每次观测加高斯噪声 | $\sigma=0$ | $\sigma=0.005$ |
| 机身角速度观测 | 每次观测加高斯噪声 | $\sigma=0$ | $\sigma=0.010$ rad/s |
| 关节位置观测 | 每次观测加高斯噪声 | $\sigma=0$ | $\sigma=0.001$ rad |
| 关节速度观测 | 每次观测加高斯噪声 | $\sigma=0$ | $\sigma=0.010$ rad/s |
| 外部扰动 | 在设定间隔后给世界系水平速度增加偏移 | 关闭 | 间隔 $U[6,9]$ s；x/y 各 $U[-0.06,0.06]$ m/s |
| 起始摆动脚 | 每次 reset 重新采样 | 左/右各约 50% | 同左 |

观测噪声由 `observation_noise_scale=s` 统一缩放，基础标准差依次为：重力投影 0.01、角速度 0.02 rad/s、关节位置 0.002 rad、关节速度 0.02 rad/s。precision 的 s=0，robust 的 s=0.5。命令、Goal、相位和动作历史不加噪声。

外部扰动实现为**速度增量**，没有施加指定牛顿数的持续外力；`0.06 m/s` 是每个水平分量的幅度上限。质量随机化使用每环境共享倍率，不是逐链接独立采样。当前没有实现重力、质心偏移、几何尺寸、PD 增益、执行器延迟或电机强度的随机化。论文的域随机化覆盖范围更广，本项目未将其默认启用。[论文附录 B](https://arxiv.org/pdf/2606.08253#page=13)

### B.2 当前任务命令分布

分组按环境固定分配；每次 reset 在所属组的范围内均匀采样本 episode 的命令。episode 内命令保持不变，`command_resample_half_cycles=0`，`command_warmup_half_cycles=0`。

| 组名 | 占比 / 2048 环境数 | 前向步长 d | 节奏范围 | 左右频率关系 |
|---|---:|---|---|---|
| `retain` | 20% / 410 | `[0.20,0.25]` m | `[1.0,1.2]` Hz | 两侧独立采样 |
| `frontier` | 60% / 1229 | `[0.30,0.35]` m | `[1.4,1.8]` Hz | 共用同一次采样，$f_L=f_R$ |
| `stride` | 20% / 409 | `[0.35,0.40]` m | `[1.0,1.3]` Hz | 共用同一次采样，$f_L=f_R$ |

整数数量采用最大余数法，总数精确为 2048。可选 robust 阶段还支持每 4 个半周期尝试重新抽取命令，并将每次步长/频率变化限制在 0.02 m / 0.10 Hz；该功能在 precision 关闭。

实现：[robustness.py](../tasks/foot_placement/robustness.py)、[环境 reset 与观测](../tasks/foot_placement/foot_placement_env.py)、[walk_training.py](../configs/walk_training.py)、[walk_hard_region.json](../configs/walk_hard_region.json)。

## C. PPO Common Training Hyperparameters

### C.1 当前实际生效参数

以下值以 precision 入口及恢复后的快照为准。配置类中的 `learning_rate=3e-4`、`schedule="adaptive"`、rollout=32、entropy=0.005 属于基础默认值；正式课程和本次 profile 已覆盖这些值。

| 参数 | 当前值 | 定义 / 说明 |
|---|---|---|
| 算法 | PPO，RSL-RL 5.5.1 | 独立 actor / critic，同一观测组 |
| 优化器 | Adam | 从起点恢复优化器状态 |
| 学习率 / 调度 | `5e-5` / `fixed` | 恢复优化器后显式覆盖；上次 bridge 为 `1e-4` |
| 折扣因子 $\gamma$ | 0.995 | 按 50 Hz 控制步折扣 |
| GAE $\lambda$ | 0.95 | 优势估计 |
| PPO clipping $\epsilon$ | 0.2 | 概率比裁剪 |
| 熵系数 | 0.002 | 鼓励动作探索 |
| Value loss 系数 | 1.0 | 使用 clipped value loss |
| 最大梯度范数 | 1.0 | 梯度裁剪 |
| `desired_kl` | 0.01 | 保留的配置项；fixed 调度不据此调整学习率 |
| 并行环境 N | 2048 | CUDA 单卡 |
| 每环境 rollout H | 64 个控制步 | 每次采样跨度 1.28 s / 环境 |
| 每次 PPO 更新样本数 | 131072 | $N\times H$ |
| 每次更新 epochs | 4 | 样本重复使用 4 轮 |
| 每 epoch mini-batches | 4 | 每个 mini-batch 为 32768 条 transition |
| 每次 PPO 更新的梯度更新数 | 16 | 4 epochs × 4 mini-batches |
| 优势归一化 | 整个 rollout 批次归一化 | `normalize_advantage_per_mini_batch=False` |
| 观测归一化 | Actor / Critic 各自开启 | EmpiricalNormalization，续训保留历史统计 |
| PPO 输入奖励 | D 节奖励率之和 × 0.02 | 无额外 reward 标准化 wrapper；`normalized_return` 只记日志 |
| Actor / Critic 隐藏层 | `[256,128]` / `[256,128]` | ELU；无历史堆叠、无循环网络 |
| 动作分布 | 对角高斯，27 维均值及可学习 std | `std_type="scalar"` 表示 std 直接参数化，每动作维一个参数 |
| fresh straight 的初始 std | 0.15 / 维 | 基础类为 0.35；完整续训恢复已学习 std，不重置到 0.15 |
| 有效动作 std 范围 | `[1e-6,1e6]` | 当前高斯分布的 `std_range`；与执行动作的 `[-1,1]` 裁剪分别配置 |
| 原始动作裁剪 | `[-1,1]` | 接着进行动作平滑、关节角缩放和软限位裁剪 |
| 随机种子 | 42 | 固定评估种子；物理结果仍可能有浮点差异 |
| RND / 对称增广 / 混合精度 | 关闭 / 关闭 / 关闭 | 与当前快照一致 |
| 随机 episode 初始长度 | 关闭 | `init_at_random_ep_len=False` |
| checkpoint 保存间隔 | 每 250 次 PPO 更新 | 另在训练段结束保存 final；时间中断保存 interrupted |
| 总 PPO 更新数保护上限 | 50000 | 当前入口值；本轮优先由 1 h 墙钟预算停止 |

当前 MLP、采样量、epochs、学习率及奖励归一化路径均采用本项目配置，并未复制论文的 PPO 参数表。[论文附录 C](https://arxiv.org/pdf/2606.08253#page=14)

### C.2 控制接口及终止条件

27 维动作按以下顺序映射到关节，仿真内部关节顺序通过名称索引转换：

```text
leg_l1_joint … leg_l6_joint,
leg_r1_joint … leg_r6_joint,
waist_yaw_joint,
zarm_l1_joint … zarm_l7_joint,
zarm_r1_joint … zarm_r7_joint
```

令 $a_t$ 为高斯动作，$\bar a_t$ 为执行动作，则：

$$
\bar a_t=0.8\bar a_{t-1}+0.2\,\operatorname{clip}(a_t,-1,1),\qquad
q_t^{des}=\operatorname{clip}_{soft}(q_0+S\bar a_t).
$$

| 关节组 | Action scale / 维（rad） | Kp（N·m/rad） | Kd（N·m·s/rad） | 配置力矩上限（N·m） |
|---|---:|---:|---:|---:|
| `leg_[lr]1_joint` | 0.10 | 3000 | 27.3861 | 88.9 |
| `leg_[lr]2_joint` | 0.15 | 3000 | 27.3861 | 49.7 |
| `leg_[lr]3_joint` | 0.30 | 2400 | 27.3861 | 92.4 |
| `leg_[lr]4_joint` | 0.60 | 2400 | 32.8634 | 196.0 |
| `leg_[lr]5_joint` | 0.30 | 900 | 41.0792 | 39.9 |
| `leg_[lr]6_joint` | 0.10 | 1500 | 41.0792 | 39.9 |
| `waist_yaw_joint` | 0.05 | 1200 | 38.3406 | 71.4 |
| `zarm_[lr]1_joint` | 0.05 | 450 | 32.8634 | 46.2 |
| `zarm_[lr]2_joint` | 0.05 | 900 | 16.4317 | 52.5 |
| `zarm_[lr]3_joint` | 0.05 | 450 | 16.4317 | 39.9 |
| `zarm_[lr]4_joint` | 0.05 | 900 | 32.8634 | 52.5 |
| `zarm_[lr][567]_joint` | 0.05 | 450 | 16.4317 | 9.87 |

表中 Kp=原始配置×30，Kd=原始配置×√30，是当前仿真调试值；力矩限幅按项目配置保留。引擎使用隐式 PD，表值不等同于实机额定连续能力。实际关节力矩应读取仿真反馈；策略步末的 PD 估计值不能代替 500 Hz 峰值实测。

reset 名义基座位置为 `(0,0,0.969025)` m 加环境原点，四元数 `(0,0,0,1)`。双腿每侧关节 1–6 名义角为 `[0,0,-0.10,0.25,-0.15,0]` rad；腰为 0；左/右臂第二关节为 +0.075 / −0.075 rad，其余手臂关节为 0。初始关节速度及机身速度为 0，历史动作清零。

precision 训练 episode 最长 **4 s = 200 控制步**。`terminated` 在基座相对地面高度 <0.55 m 或 upright=`−g_z^B`<0.5 时触发；`truncated` 在到达 episode 上限时触发，分别记录，允许同一帧同时满足两个条件。`is_finite_horizon=False`，wrapper 将 timeout 标志传给 RSL-RL 进行超时 bootstrap；环境沿用 `compute_final_obs=False`。

实现：[s54_robot_cfg.py](../configs/s54_robot_cfg.py)、[s54_physics_cfg.py](../configs/s54_physics_cfg.py)、[动作与终止逻辑](../tasks/foot_placement/foot_placement_env.py)、[train.py](../scripts/train.py)、[checkpoint_compat.py](../agents/checkpoint_compat.py)。

### C.3 训练预算、评估与实验版本

本轮每个训练段最多 500 次 PPO 更新，约 6554 万条 transition。`minimum_updates=500` 是资格判定条件，不能覆盖一小时墙钟预算。监督器预留 180 s 给末尾评估，按实测速率缩短末段训练。默认每 10 次更新打印简洁进度；SwanLab 使用 compact 模式，完整指标保存在本地 `metrics.jsonl`。

训练前先评估起点；每个训练段结束后以相同 seed 对以下三个固定命令评估：25 cm / 1.2 Hz、30 cm / 1.5 Hz、35 cm / 1.7 Hz。每点 16 环境×1000 控制步，总仿真时间 20 s；每个 episode 为 10 s，包含 reset，不能作为连续 20 s 行走结果。输出包括左右首步、首步之后和整体跟踪、滑动/跳动、terminated/truncated；`pilot_comparison.json` 对比基线和最新一次已完整评估的模型。

| 配置版本 | 用途 / 核心差异 | 参数来源 |
|---|---|---|
| `walk_4stage_12h` | 历史四阶段范围：步长 `[.05,.25]→[.05,.30]→[.05,.40]→[.05,.40]` m，频率 `[.5,1.5]→[.5,1.5]→[.5,1.8]→[.5,2.0]` Hz；奖励以当次快照为准 | [walk_4stage_12h.json](../configs/walk_4stage_12h.json) |
| `walk_recovery` | 20–25 cm、1–1.2 Hz；随机左右起步，10 s episode，默认 v2，无 precision XY 覆盖，LR=5e−5 | [walk_recovery.json](../configs/walk_recovery.json) |
| `walk_hard_region` | bridge/frontier/hard/robust；前三阶段关闭扰动，最后阶段采用 B 节可选随机化；默认 v2，无 precision XY 覆盖 | [walk_hard_region.json](../configs/walk_hard_region.json) |
| **`walk_precision_1h`** | 本文主配置；集中中高难度、4 s episode、摆动及首步 XY 加权、混合距离核，LR=5e−5 | [walk_precision_1h.json](../configs/walk_precision_1h.json) |
| **`walk_retention_30m`** | 当前纯训练；从500更新候选出发，10 s episode、15格固定采样、10%角点、LR=2e−5，无自动eval，奖励沿用precision | [walk_retention_30m.json](../configs/walk_retention_30m.json) |

历史模型回放使用各自保存的任务配置；本文的 precision 奖励参数不反向应用到历史实验。运行及短测记录见 [precision 检查报告](../reports/walk_precision_1h_2026-11-01/REPORT.md)，其余实验以各自 `reports/` 和 `logs/` 快照为准。

## D. Additional Reward Terms

### D.1 总奖励、符号与接触门控

本节完整列出当前 8 个日志奖励项。Tracking、Gait 和 Regularization 是说明性分类，代码仍使用 8 个 term 名称。

$$
r_t=\Delta t\left(r_{tracking}+r_{swing\_phase}+r_{stance\_slip}+r_{stance\_hop}
+r_{body\_motion}+r_{posture}+r_{action\_rate}+r_{heading}\right),\qquad \Delta t=0.02\ \mathrm s.
$$

公式中的各项均为**奖励率**，每控制步求和后只乘一次 dt；日志 `Reward/<term>` 也已乘 dt。网络收到的是该原始步奖励，可能为负。

| 符号 | 定义 |
|---|---|
| s / st | 当前摆动脚 / 支撑脚 |
| $u$ | 本侧摆动进度：左摆为 $2\phi$，右摆为 $2\phi-1$，裁剪至 `[0,1]` |
| $e_s,e_{st}$ | 足底实际位置到本脚目标的 XY 欧氏距离，m |
| $C_i$ | 足端对地法向力 $F_{z,i}>20$ N 的接触指示 |
| $h_{st}$ | $\max(z_{st}-z_{st}^*,0)$ |
| $G$ | $C_{st}\,\mathbf1[h_{st}\le0.025]$，有效支撑门控 |
| $W$ | $\mathbf1[\lvert u-0.5\rvert\le0.2]$，摆动奖励窗口 |
| $I$ | $\mathbf1[0.1<u<0.9]$，惩罚缺失支撑的内部窗口 |
| $\Delta\psi_i$ | 各脚实际 yaw 相对目标 yaw 的 wrap 角误差，rad |
| $e_H$ | 机身 yaw 相对 reset 参考 yaw 的 wrap 角误差，rad |
| $H,H_0$ | 相对环境原点的基座高度 / 名义基座高度 0.969025 m |
| $q_U-q_{U,0}$ | 腰+双臂 15 维关节参考偏移，rad |

接触力由脚部 ContactSensor 的对地过滤力读取，对匹配地面碰撞形状求和。G 依据实际支撑状态计算；摆动脚逼近目标但缺乏有效支撑时，tracking 和 swing_phase 均不发放。

### D.2 Tracking：足印 XY 与足端 Yaw

precision 使用的摆动距离核及权重为：

$$
K(e)=0.75e^{-100e^2}+0.25e^{-25e^2},\qquad
m_t=\begin{cases}3,&\text{首个摆动半周期},\\2,&\text{后续半周期}.\end{cases}
$$

$$
r_{tracking}=G\left[
2\left(m_tu^2K(e_s)+e^{-100e_{st}^2}\right)
+0.25\left(e^{-25\Delta\psi_L^2}+e^{-25\Delta\psi_R^2}\right)
\right].
$$

`tracking_xy_w=4` 在代码内部乘 1/2，因此支撑脚 XY 系数是 2；摆动脚系数在首步/后续分别为 $6u^2$/$4u^2$。`tracking_yaw_w=0.5` 对两脚平均后各为 0.25。宽核提供远距离接近信号，窄核保留精细落点奖励；首步加权覆盖整个首个摆动半周期，$u^2$ 使临近落脚时权重更大。

不含 precision 覆盖的默认 v2 使用 $m_t=1$、$K(e)=e^{-100e^2}$。相关覆盖均来自当前 JSON：`swing_xy_multiplier=2`、`first_swing_xy_multiplier=1.5`、`swing_xy_coarse_weight=0.25`、`swing_xy_coarse_sharpness=25`。修改后同步更新归一化分母。

### D.3 Gait：指定相位抬脚、滑动和跳动

**摆动相位抬脚**：

$$
r_{swing\_phase}=4W(1-C_s)G\,
\operatorname{clip}\left(\frac{z_s-z_s^*}{0.03},0,1\right).
$$

该项在摆动窗口内、摆动脚离地且支撑有效时，按抬脚高度发奖，3 cm 达满分。没有整条 Z 正弦轨迹跟踪奖励，亦没有独立的 touchdown 奖励项。

**接地脚滑动**：

$$
r_{stance\_slip}=-4\sum_{i\in\{L,R\}}C_i\|v_{sole,i,xy}\|_2^2.
$$

尽管日志名包含 stance，所有实际接地的脚都会参与该项，速度使用 A.2 的足底点世界系速度。竖直速度另由 hop 项处理。

**支撑脚跳动与双脚腾空**：

$$
c_h=\min\left[\left(\frac{\max(h_{st}-0.025,0)}{0.02}\right)^2,4\right],\qquad
c_v=\min\left[\left(\frac{\max(|v_{st,z}|-0.05,0)}{0.50}\right)^2,4\right],
$$

$$
r_{stance\_hop}=-4\left[c_h+c_v+I(1-C_{st})+\mathbf1[C_L=0\land C_R=0]\right].
$$

2.5 cm 的支撑高度容差允许正常足掌滚动，0.05 m/s 为支撑脚竖直速度死区；高度/速度的尺度分别为 0.02 m / 0.50 m/s。高度和竖直速度成本在所有相位计算，缺失支撑指示项仅在 I 窗口内计算；双脚腾空项全相位计算。

### D.4 Regularization：身体、关节、动作与方向

$$
r_{body\_motion}=-10(H-H_0)^2-0.5v_{base,z}^2,
$$

$$
r_{posture}=0.5\,\operatorname{clip}(-g_z^B,0,1)
\exp\left(-4\|q_U-q_{U,0}\|_2^2\right),
$$

$$
r_{action\_rate}=-0.05\,\frac{1}{27}\|\bar a_t-\bar a_{t-1}\|_2^2,
$$

$$
r_{heading}=\exp\left[-\left(\frac{e_H}{0.35}\right)^2\right]
+0.5\exp\left[-\left(\frac{\omega_{base,z}^B}{0.30}\right)^2\right].
$$

身体项使用基座高度而非全身质心高度；动作项使用平滑后的无量纲执行动作，先对 27 维求均值。heading 项鼓励沿 reset 参考 yaw 前进并抑制机身转动，当前目标 yaw-rate=0。姿态奖励联合考虑直立与上身关节参考姿态。

### D.5 系数、理论上界与归一化记录

| 日志 term | 当前核心系数 | 该项奖励率的理论上界 | precision 相对默认 v2 的变化 |
|---|---|---|---|
| `tracking` | XY 总权重 4；Yaw 总权重 0.5 | $2(1+m_tu^2)+0.5$ | 仅摆动 XY 加权和宽/窄核混合；支撑/Yaw 系数保留 |
| `swing_phase` | 4，高度 0.03 m | $4W$ | 保留 |
| `stance_slip` | 4 | 0 | 保留 |
| `stance_hop` | 4 | 0 | 保留 |
| `body_motion` | 高度 10、竖直速度 0.5 | 0 | 保留 |
| `posture` | 外系数 0.5、关节指数系数 4 | 0.5 | 保留 |
| `action_rate` | 0.05，27 维均值 | 0 | 保留 |
| `heading` | heading 1、yaw-rate 0.5 | 1.5 | 保留 |

上界按各正向子项同时满分、各惩罚为 0 计算，是逐相位**参考上界**，不保证动力学可同时达到。不能把每个 term 的最大值分别跨相位取值后当成每帧实际可达分数。

precision 的逐控制步参考分母为：

$$
r_t^{ref}=0.02\left[4.5+2m_tu^2+4W\right].
$$

逐步日志为 $\operatorname{clip}(r_t/r_t^{ref},0,1)$。完成 episode 的归一化回报为：

$$
R_{norm}=\operatorname{clip}\left(\frac{\sum_t r_t}{\sum_t r_t^{ref}},0,1\right)
\operatorname{clip}\left(\frac{T_{episode}}{T_{max}},0,1\right).
$$

该记录同时体现奖励质量与 episode 存活长度。它不反馈给 PPO，也不是成功率；4 s 训练 episode 和 10 s 评估 episode 的存活比例分母各自计算。对比不同奖励版本时，应同时看物理 XY、首步成功率及存活率，不能直接用历史归一化分数判优。

### D.6 与论文平地设置的对应关系

论文平地设置关闭足端 Z 目标跟踪与摆动膝高度奖励，保留足印 XY/Yaw、相位抬脚和运动正则化；附录还列出上身姿态、身体高度、动作变化、接触脚滑动项。[论文附录 D、F](https://arxiv.org/pdf/2606.08253#page=14)

| 类别 | S54 当前处理 |
|---|---|
| 足印跟踪 | 使用本节摆/支分权、$u^2$、接触门控；precision 增加首步系数与混合距离核 |
| 平地高度处理 | 关闭足端 Z 目标跟踪/膝高度项；用相位窗口中的实际足底离地高度发奖 |
| 防滑与防跳 | 接地脚 XY 速度惩罚；额外增加支撑脚高度/竖直速度、缺失支撑和双脚腾空成本 |
| 上身 / 身体 | 上身参考姿态乘直立程度；使用基座参考高度及竖直速度惩罚 |
| 动作平滑 | 27 维动作差平方均值，系数 0.05，输入为经过滤波的动作 |
| 方向保持 | 本项目额外使用固定 heading 与零 yaw-rate 跟踪 |
| 其他运动正则化 | 当前 v2 只有本文 8 个 term，没有额外独立的力矩、能耗、关节加速度或终止奖励项 |

因此应将本项目标为“参考足印跟踪方法的 S54 平地适配实验”。本实现未完整复现论文及其 LocoMuJoCo 基础奖励集合，系数不能脱离平均/求和、门控、动作单位及 dt 缩放直接比较。

### D.7 首步物理指标与试训判据

首步成功由实际状态判断：摆动中段 $0.2<u<0.8$ 曾满足摆动脚接触力 <20 N 且抬脚高度 >0.018 m；中段持续有支撑脚接触；末段 $u\ge0.8$ 曾满足摆动脚接触力 >20 N 且 XY 误差 <0.03 m；首步结束时未提前结束 episode。左右起步分别统计。提前失败计入成功率分母，没有末段 XY 样本时报告缺失计数。

当前晋级/资格判据包括：各工况存活比例≥95%、左右半周期成功率各≥80%、8次连续成功比例≥80%、末段平均 XY≤3 cm、平均 upright≥0.95、最大 heading 误差≤10°、平均参考方向前进速度≥名义值一半、双脚腾空比例≤2%、支撑跳动比例≤2%、接地脚平均滑动速度≤0.10 m/s。随机左右起步的分组结果也需分别通过；新增首步左右各至少4个样本、成功率≥80%、末段平均 XY≤3 cm。阶段至少500次更新且连续两次评估通过，才记录对应 checkpoint 的资格。

一小时 pilot 主要检查物理趋势，严格门槛未通过时仍按总预算停止。完整公式与记录实现见 [flat_rewards.py](../tasks/foot_placement/flat_rewards.py)、[s54_flat_reward.py](../configs/s54_flat_reward.py)、[rewards.py](../tasks/foot_placement/rewards.py)、[first_step_metrics.py](../tasks/foot_placement/first_step_metrics.py)、[quick_walk_eval.py](../scripts/quick_walk_eval.py)、[adaptive_walk.py](../scripts/adaptive_walk.py)。
