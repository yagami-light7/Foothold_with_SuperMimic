# A/B/C Short Ablation 实验报告

报告日期：2026-11-06（Asia/Shanghai）。审计对象为 2026-11-05 完成的 `s54_abc_short_20261105_184038` 批次。**本轮仅读取已有文件、在 CPU 上检查 Checkpoint/NPZ/SwanLab 本地记录并生成本文；没有修改训练代码、配置、Checkpoint 或历史日志，没有启动训练、仿真、渲染或上传。**

复核结果：已重新读取三组最终与 WM-only Checkpoint，核对实际参数、Adam 更新计数及 Decoder/Encoder 权重；27 个已有 NPZ 的回报与 JSON 一致，三组相同 seed 的初始命令一致。离线 SwanLab 记录的 5100 个训练点及 183 个评估点与原始文件一致；484 个源码/配置快照哈希核验通过，42 个关键历史文件的内容、大小和修改时间未改变。本文 105 个本地证据链接均指向已存在文件。

## 1. 主要结论与范围

- **已验证事实：**三组均完成 2000 iterations、5,251,072 条真实 Newton 训练 transitions、32,000 次 WM optimizer step、1000 次 Actor optimizer step。Actor 更新期间仅 Foot Encoder 可训练；三组 Decoder 均被冻结。
- **已验证事实：**B 的 WM 预测误差比 A 低，但 A/B 的真实存活时间、回报和足部误差基本相同，未观察到控制能力收益。C 相比 B 更早发生高度终止：平均约 1.18 s，对比 B 的约 1.92 s；不能声称直接 Decoder Transfer 改善了行走。
- **结论边界：**本次是“冻结 Decoder、只适配 Encoder”的单训练种子实验。A/B 冻结了随机 Decoder，不能作为完整 Actor 可训练的从零基线；三组尚无稳定前进行走证据。D/F/G 正式性能消融的能力条件仍为 **BLOCKED**；有界接口/Bridge 诊断与正式长训需分开。

自动定位依据：扫描 `logs/s54_abc_short_*/batch_status.json`，选中唯一 `status=completed` 且 `completed_groups=[A,B,C]` 的批次。较早的 `s54_abc_short_20261105_182228` 为停止批次，其 A 仅训练到 640 iterations，B/C 未完成，**不混入统计**。完成证据为 [batch_status.json](../../logs/s54_abc_short_20261105_184038/batch_status.json)，不是根据目录名判断。

| 组 | 实际训练目录 | 最终 Checkpoint | 训练状态 / exit code |
| --- | --- | --- | --- |
| A | `logs/s54_abc_short_20261105_184038_A` | [last.pt](../../logs/s54_abc_short_20261105_184038_A/last.pt) / [ckpt_002000.pt](../../logs/s54_abc_short_20261105_184038_A/ckpt_002000.pt) | [completed](../../logs/s54_abc_short_20261105_184038_A/training_status.json) / `0` |
| B | `logs/s54_abc_short_20261105_184038_B` | [last.pt](../../logs/s54_abc_short_20261105_184038_B/last.pt) / [ckpt_002000.pt](../../logs/s54_abc_short_20261105_184038_B/ckpt_002000.pt) | [completed](../../logs/s54_abc_short_20261105_184038_B/training_status.json) / `0` |
| C | `logs/s54_abc_short_20261105_184038_C` | [last.pt](../../logs/s54_abc_short_20261105_184038_C/last.pt) / [ckpt_002000.pt](../../logs/s54_abc_short_20261105_184038_C/ckpt_002000.pt) | [completed](../../logs/s54_abc_short_20261105_184038_C/training_status.json) / `0` |

本文所有均值均说明分母。训练曲线的阶段表来自 CSV 记录均值；评估表来自单个 Checkpoint 的真实 Newton Rollout，不用训练 Reward 推断成功率。未发现本批次的独立 Standing 评估，故当前三组的 Standing 能力标为 **NOT EVALUATED**。

## 2. 训练配置与参数审计

实际参数来源：[A-args](../../logs/s54_abc_short_20261105_184038_A/args.json)；[B-args](../../logs/s54_abc_short_20261105_184038_B/args.json)；[C-args](../../logs/s54_abc_short_20261105_184038_C/args.json)。预算与硬件来源：[A-status](../../logs/s54_abc_short_20261105_184038_A/training_status.json)；[B-status](../../logs/s54_abc_short_20261105_184038_B/training_status.json)；[C-status](../../logs/s54_abc_short_20261105_184038_C/training_status.json)。[A-runtime](../../logs/s54_abc_short_20261105_184038_A/runtime_stats.json)；[B-runtime](../../logs/s54_abc_short_20261105_184038_B/runtime_stats.json)；[C-runtime](../../logs/s54_abc_short_20261105_184038_C/runtime_stats.json)。Checkpoint 内 `args` 与各自 `args.json` 的学习超参数一致；两项可视化启动器字段不同，见 §2.1。

| 参数 | A | B | C | 真实语义 / 差异说明 |
| --- | --- | --- | --- | --- |
| num_envs | 1024 | 1024 | 1024 | 并行 Newton 环境数 |
| iterations | 2000 | 2000 | 2000 | 每组总目标；不是 Actor 更新数 |
| horizon | 32 | 32 | 32 | Imagined BPTT：32×0.02=0.64 s |
| hist | 10 | 10 | 10 | 10 帧观测历史 |
| goal_frames | 10 | 10 | 10 | 10 个未来 Goal |
| goal_dt | 0.1 | 0.1 | 0.1 | 秒；未来 0.1…1.0 s |
| buffer_steps | 1024 | 1024 | 1024 | 每环境环形存储时间容量；总上限 1,048,576 条 transitions |
| warmup_steps | 128 | 128 | 128 | 128 个 vector steps，合计 131,072 条真实 transitions |
| wm_only_iters | 1000 | 1000 | 1000 | 前 1000 iterations 不更新 Actor/Critic optimizer |
| utd | 0.4 | 0.4 | 0.4 | 每轮累计采样 1/0.4=2.5 个 vector steps |
| wm_window | 8 | 8 | 8 | WM 监督展开 8 步=0.16 s |
| wm_batch | 1024 | 1024 | 1024 | 每次 WM 更新 1024 个 Replay 序列 |
| wm_batches | 16 | 16 | 16 | 每 iteration 16 次 WM optimizer step |
| pi_batch | 256 | 256 | 256 | 256 个 imagination 起点 |
| WM 初始 LR | 5e-05 | 5e-05 | 5e-05 | Adam，betas=(0.9,0.999) |
| Actor 初始 LR | 0.0001 | 0.0001 | 0.0001 | Adam，betas=(0.9,0.95)，仅 Encoder 获得参数梯度 |
| Critic 初始 LR | 0.0001 | 0.0001 | 0.0001 | Adam，betas=(0.9,0.999) |
| LR 衰减长度 | 20000 | 20000 | 20000 | 共享 cosine 比例，见下文 |
| gamma | 0.99 | 0.99 | 0.99 | 折扣 |
| lam | 0.95 | 0.95 | 0.95 | imagined λ-return |
| Action Noise | 0.03 | 0.03 | 0.03 | 真实采样动作添加 Gaussian std=0.03 后裁剪 |
| critic_epochs | 2 | 2 | 2 | 每次 policy update 2 轮 |
| critic_minibatches | 4 | 4 | 4 | 每轮 4 个 minibatch |
| critic_real_batch | 0 | 0 | 0 | 0：本批次无真实 MC-return Critic 辅助样本 |
| critic_real_weight | 0.25 | 0.25 | 0.25 | 本批次不生效；real window/recent steps 均为 512 |
| freeze_decoder | true | true | true | 三组均冻结 |
| train_decoder | false | false | false | 三组均不微调 Decoder |
| obs_adapter | mask_reference_orientation | mask_reference_orientation | mask_reference_orientation | 每帧归一化观测的 93:99 六维参考朝向置零 |
| 训练 seed | 54 | 54 | 54 | 每组仅训练一次、共用同一 seed；没有同组多种子重复 |
| deterministic | false | false | false | 未启用严格确定性执行 |
| AMP | bf16 | bf16 | bf16 | 仅 MLP matmul autocast，状态积分/FK/Reward 为 fp32 |
| no_compile | true | true | true | 未 torch.compile |
| log_every | 20 | 20 | 20 | 20 iterations 一行 |
| ckpt_every | 500 | 500 | 500 | 500 iterations |
| keep_checkpoints | 6 | 6 | 6 | 保留设置 |
| gradient_audit_every | 100 | 100 | 100 | 额外梯度审计间隔 |
| wall_seconds | null | null | null | null：训练按 2000 iterations 完成，没有 0.5 h 截止 |

没有独立 `rollout_steps` 参数：collector 使用累计器交替采样 2/3 个 vector steps，每组训练循环共 5000 个 vector steps，另有 128 个 warmup。真实采样量为 **1024×(128+5000)=5,251,072**。`horizon=32` 是 WM 中的可微想象长度，不能当作 PPO rollout 长度；不能将 Replay 重采样、WM 监督窗口或 imagination 计入新增真实 Newton transitions。实现：[supermimic/train.py](../../logs/s54_abc_short_20261105_184038_A/source_snapshot/supermimic/train.py)。

学习率实际按零起始 iteration `i` 使用 `s(i)=0.03+0.97×0.5×[1+cos(πi/20000)]`，三组件各自初始 LR 乘以 `s(i)`；达到 20000 后保持初始 LR 的 3%。本次最后一次更新 `i=1999`，Checkpoint 中 WM LR 为 **4.8814297337e-5**，Actor/Critic LR 为 **9.7628594675e-5**。2000 次迭代只完成衰减周期的一小部分。`critic_real_batch=0` 意味着 `.25` 的 real-loss 权重及相关窗口参数没有参与目标。

### 2.1 所有不同参数与快照差异

逐键比较全部实际 `args.json`，仅发现以下 **3 个键**存在组间差异：

| 键 | A | B | C | 解释 |
| --- | --- | --- | --- | --- |
| init_from_motion | None | `…/s54_abc_short_20261105_184038/teacher_v1.pt` | 同 B | **消融变量**：A 不迁移，B/C 同一源文件 |
| transfer_components | `wm_decoder`（默认） | `wm` | `wm_decoder` | **消融变量**：A 没有 init 源，此默认字符串不触发迁移 |
| log_dir | `…184038_A` | `…184038_B` | `…184038_C` | 输出路径差异，不是学习超参数 |

另逐键比较每组的 `args.json` 与 `last.pt/args`，三组均有相同的两项差异，不能称为文件内容完全一致：

| 字段 | 启动时 args.json | 最终 Checkpoint 内 args | A/B/C 是否同样变化 |
| --- | --- | --- | --- |
| kit_visualizer | null | false | 是 |
| livestream | -1 | 0 | 是 |

这些是仿真可视化/串流字段；未发现其他启动参数与 Checkpoint 实存参数的差异，也未发现三组在这两项上的差异。它们不改变本次对学习超参数或组间初始化变量的比较。

比较各 `source_snapshot/manifest.json`，三组 `supermimic/*.py` 的文件集合和逐文件 SHA256 全部一致；`physical_contract`、Foot Goal/Task、采样和 Reward 实际参数一致。**整个快照并不完全一致**：上传故障修复先后发生，以下文件的存在性或内容有差异：

- `configs/motion_prior/stage2/abc_short_3090.json`
- `tools/ops/swanlab_jobs.py`
- `tools/ops/sync_eval_swanlab.py`
- `tools/tests/test_abc_short.py`
- `tools/train/run_s54_abc_short.py`
- `tools/train/run_s54_curriculum_8h.sh`
- `tools/train/run_s54_motion_prior.py`

这些差异涉及批次恢复、上传代理/空缓存处理和测试；源配置中的上传策略也由等待 120 s 改为独立队列。实际三组使用相同批次 `profile.json` 的训练参数，核心学习实现一致。不能仅凭同一 Git SHA 声称整个工作树相同。证据：[A-snapshot](../../logs/s54_abc_short_20261105_184038_A/source_snapshot/manifest.json)；[B-snapshot](../../logs/s54_abc_short_20261105_184038_B/source_snapshot/manifest.json)；[C-snapshot](../../logs/s54_abc_short_20261105_184038_C/source_snapshot/manifest.json)。[A-launch](../../logs/s54_abc_short_20261105_184038_A/launch.json)；[B-launch](../../logs/s54_abc_short_20261105_184038_B/launch.json)；[C-launch](../../logs/s54_abc_short_20261105_184038_C/launch.json)。共同输入摘要：[shared_inputs.json](../../logs/s54_abc_short_20261105_184038/shared_inputs.json)。

### 2.2 耗时、真实样本和资源

| 指标 | A | B | C |
| --- | --- | --- | --- |
| 实际 iterations | 2000 | 2000 | 2000 |
| 真实训练 Newton transitions | 5,251,072 | 5,251,072 | 5,251,072 |
| training_status 单调计时 / s | 3530.119 | 3383.824 | 3390.487 |
| 训练耗时 / min | 58.84 | 56.40 | 56.51 |
| launcher process wall / s | 3534.270 | 3387.953 | 3394.216 |
| Checkpoint wall_time / s | 3528.895 | 3382.393 | 3389.315 |
| 真实训练 transitions / 完整训练秒 | 1487.5 | 1551.8 | 1548.8 |
| 独立评估 wall / s | 141.842 | 66.813 | 47.550 |
| 独立评估新增 Newton transitions | 863 | 863 | 531 |
| 实测 GPU | NVIDIA GeForce RTX 3090 | NVIDIA GeForce RTX 3090 | NVIDIA GeForce RTX 3090 |
| Torch peak allocated / MiB | 3024.3 | 3236.5 | 3270.7 |
| Torch peak reserved / MiB | 3054.0 | 3274.0 | 3308.0 |
| Checkpoint 时设备总占用快照 / MiB | 4190.1 | 4410.1 | 4444.1 |
| 全程 GPU utilization 均值 / 峰值 | N/A | N/A | N/A |

三组真实训练合计 **15,753,216** 条 transitions，评估另计 **2257** 条。Torch peak 不含 Warp；设备占用是保存时快照，不能称为整个仿真进程的显存峰值。GPU 型号来自当时 `runtime_stats.json`，确为 RTX 3090，不是 RTX 4060，也没有用本轮机器当前占用替代历史测量。训练计时和 launcher 计时起止不同，且包含的保存/初始化开销略有差异；上传故障等待不计为 Actor/WM 训练时间。

机器人/物理合同：29 DoF、state 71、obs 99、action 29、ξ 33、WM 133→70、控制 dt=0.02 s；Newton 1.6.1、mujoco-warp 3.12.0、warp-lang 1.17.0，4 个物理子步，`implicitfast`、pyramidal cone、`use_mujoco_contacts=True`。三组 MJCF 摘要、关节顺序、执行器/PD 参数和动作缩放均相同，详见各 `last.pt/physical_contract`。这些是 Checkpoint 实存版本；Isaac Lab 安装来源为项目的 `IsaacLab_3_0` 环境。

## 3. Foot Goal、Reward 与终止条件

来源：实际 `args.json` 中完整解析后的 `foot_config`、`foot_task_config`、`reward_config`；快照实现 [supermimic/foot_trajectory.py](../../logs/s54_abc_short_20261105_184038_A/source_snapshot/supermimic/foot_trajectory.py)、[supermimic/foot_mdp.py](../../logs/s54_abc_short_20261105_184038_A/source_snapshot/supermimic/foot_mdp.py)、[supermimic/env.py](../../logs/s54_abc_short_20261105_184038_A/source_snapshot/supermimic/env.py)。以下设置三组逐值相同。

| 项目 | A/B/C 实际值与含义 |
| --- | --- |
| 任务分布 | 100% `alternating` 保守短步；没有 Standing、Turning、Walking+Turn 混合，也未启用 Curriculum |
| 数据来源 | 程序生成 Quintic Foot 轨迹；`clips=[]`、`motion_dir=None`、`reference_manifest=None`。Stage 2 不读取 LAFAN1 Clip |
| Goal | 230D=10×23；每未来时刻含两脚 position3+rotation6+contact1+phase1，另有 time1；Stage 1 Motion Goal 为 730D |
| 轨迹坐标 / 足部原点 | 固定世界系命令，Goal 转至当前 root yaw 系；目标与 FK 跟踪 `leg_l6_link/leg_r6_link` 踝 link 原点，不是足底接触中心/COP |
| 每 episode 计划 | 32 个交替落脚事件，左脚先迈；初始双支撑 0.15 s；步宽继承初始脚间距 |
| step_length | 逐事件独立 U[0.02,0.06] m 后累加；是左右相邻目标的前向增量，同一脚通常移动两个增量之和 0.04–0.12 m |
| Swing / Double Support | 分别独立 U[0.65,0.90] s / U[0.15,0.25] s；相邻 liftoff 间隔为两者之和 |
| 隐含步频 | 未直接采样 frequency；1/(Swing+DS) 为 0.87–1.25 次落脚/s，完整左右周期频率约为其一半。本实验没有 0 Hz 除法路径 |
| Swing 高度 / 插值 | U[0.030,0.045] m；quintic 位置曲线与平滑抬脚，非跳变目标 |
| Yaw / 命令切换 | yaw_offset=[0,0]，直行固定 heading；每 reset 采样一次整段命令，episode 内不切换 |
| Episode 上限 | 500 个控制步=10 s；本任务无 Motion Clip 结束计数。32 步计划覆盖时间长于当前 episode 与未来 Goal 窗口 |
| Domain Randomization | false：固定摩擦 ξ[0]=1，关节偏置/躯干 COM 偏置为 0；没有本批次随机推力记录 |
| Reset 扰动半宽 | 关节 ±0.005 rad、root xy ±0.005 m、roll/pitch 旋转向量 ±0.01 rad、yaw ±0.05 rad、线/角/关节速度对应 ±0.01 |
| 观测噪声 | 训练使用均匀噪声：q ±0.01、qd ±0.5、角速度 ±0.2、重力方向 ±0.05、前一动作 0、参考朝向6D ±0.05（之后 adapter mask） |
| Action / 梯度裁剪 | 真实采样与 imagination 均 clamp 到 [-20,20]；WM/Critic grad norm≤1，Actor grad norm≤1 且 grad value≤0.1 |

Reset 扰动、观测噪声和 Action Noise 并未随 `domain_randomization=false` 关闭。相同训练 seed 保证初始化/采样制度可比较，不能保证不同策略在不同终止时刻后仍采到逐条相同的训练命令。独立评估的初始命令张量则已从 NPZ 逐项验证 A/B/C 一致。

### 3.1 Reward 系数

| 分项 | A 权重 | B 权重 | C 权重 | 误差 / 标度 |
| --- | --- | --- | --- | --- |
| Swing Position | 1.0 | 1.0 | 1.0 | position_std=0.08 m |
| Swing Velocity | 0.25 | 0.25 | 0.25 | velocity_std=0.5 m/s |
| Stance Position | 1.0 | 1.0 | 1.0 | position_std=0.08 m |
| Stance Velocity | 0.25 | 0.25 | 0.25 | 目标速度为 0；std=0.5 m/s |
| Foot Orientation | 0.25 | 0.25 | 0.25 | orientation_std=0.35；e=0.5×Frobenius(R−R*)² |
| Scheduled Touchdown | 0.5 | 0.5 | 0.5 | touchdown_std=0.05 m；事件首次跨越区间才奖励 |
| Base Upright | 1.0 | 1.0 | 1.0 | upright_std=0.2；误差为 1−Rzz |
| Base Height | 0.5 | 0.5 | 0.5 | base_height_std=0.08 m |
| Base XY Balance | 0.5 | 0.5 | 0.5 | base_xy_std=0.15 m；目标为两脚参考位置均值+固定 root offset |
| Action Rate | -0.1 | -0.1 | -0.1 | Σ(a−a_prev)² |
| Joint Limit | -2.0 | -2.0 | -2.0 | soft_limit_factor=0.9；越界距离之和 |
| Default Posture | -0.05 | -0.05 | -0.05 | Σ(q−q_default)²，不是 Motion Upper Prior |
| Joint Velocity | -0.001 | -0.001 | -0.001 | mean(qd²) |

正向质量函数为 `Q(e,σ)=1/(1+e/σ²)`，汇总正负项后统一乘 dt=0.02。Stance/Swing 使用**计划接触**，位置/速度来自可微 FK；Touchdown 使用计划时刻与目标，不要求传感器已经确认真实落地。真实接触另在评估中测量。最大归一化单步 Reward 上界为 **r_max=0.08**；Foot 原点 4 cm 的位置误差在 σ=8 cm 时仍有 Q=0.8，因此较高短时 Reward 不代表精确、持续行走。未启用动态 Upper Prior，且没有独立的物理终止附加惩罚项；imagination 另有 soft-survival 权重，不能与该 Reward 项混同。

### 3.2 终止条件

Physical Termination：状态含 NaN/Inf，或 `abs(root_z−nominal_z)>0.25 m`，或 `1−Rzz>0.5`（root 倾斜超过 60°）。500 步是 Timeout，不算跌倒；Foot 任务没有 Clip End。独立评估将 horizon 改为 100/250/500 步，保留原物理阈值；同一步满足物理终止和 Timeout 时，原因归为物理终止。Goal Tracking Failure 没有单独在线终止标志，需根据跟踪指标另行判定，不能把所有 Timeout 标为成功。

## 4. 消融变量、权重迁移与更新隔离

| 组 | WM 初始化 | Foot Encoder 初始化 | Decoder 初始化 | Decoder 是否冻结 | 实际目标 |
| --- | --- | --- | --- | --- | --- |
| A | 随机；无加载源 | 随机 | 随机 | **是** | Foot RL，只适配 Encoder |
| B | Stage 1 WM + WM input/output norm | 随机，与 A 相同 | 随机，与 A **逐 tensor 相同** | **是** | 同 A |
| C | 与 B 同一 Stage 1 WM + norm 来源 | 随机，与 A/B 相同 | Stage 1 Shared Decoder | **是** | 直接 Foot RL，无 Bridge / Prior |

迁移证据：[B-transfer_report](../../logs/s54_abc_short_20261105_184038_B/transfer_report.json)、[C-transfer_report](../../logs/s54_abc_short_20261105_184038_C/transfer_report.json)；A 的 `transfer_report.json` 为 **N/A（不迁移，Checkpoint transfer_report=None）**。B 严格加载 20 个 WM 参数/归一化 tensor，C 加载同样的 20 个 WM tensor 与 10 个 Decoder tensor。B/C 均跳过 Motion Encoder、Motion Goal/ObsNorm、Motion Critic 和全部源 Optimizer；新建 Foot Encoder/GoalNorm/ObsNorm/Critic/Optimizers/Replay。

实际共享 Teacher：`logs/s54_stage1_walk_20261105_014722/ckpt_004772.pt`，执行时读取只读副本 `logs/s54_abc_short_20261105_184038/teacher_v1.pt`。两份文件 SHA256 相同：

```text
1cf556f384cefa8635975ca4f8cb8c0e144a98292c5fb2fb8cff33e9b07c3fc4
```
源数据可查 [Stage 1 args](../../logs/s54_stage1_walk_20261105_014722/args.json) 与 [Stage 0 数据 manifest](../../results/s54/motion_prior_stage0_verified/manifest.json)。Teacher 为 iteration **4772**，预训练真实 transitions **12,478,464**，预训练 wall **7170.713 s**，数据合同指定 LAFAN1 `walk1_subject1`。这不是原先 12 iterations 的接口 Smoke。当前用户选定的是 v1 候选 Teacher，选择本身不构成独立 Walking holdout 或长期稳定性认证。

三组 Stage 2 样本预算相等，但端到端数据成本不同：A 无预训练，B/C 使用了共享 Teacher 的额外预训练数据；每条 B/C 流水线训练总数为 **17,729,536**。若整个对照共享预训练只计一次，三组 Stage 2 加 Teacher 为 **28,231,680** 条真实训练 transitions，评估另计，不能将预训练成本隐藏为“同总样本”。本批次 Bridge、paired latent supervision 和 Motion Reward imagination 均为 0。

| 核验项 | A | B | C | 证据 |
| --- | --- | --- | --- | --- |
| 初始 Foot Encoder SHA 前16位 | c4d467bda2144fe6 | c4d467bda2144fe6 | c4d467bda2144fe6 | wm_calibrated.pt 的 enc.*：WM-only 时无 Encoder optimizer 更新 |
| Decoder SHA 前16位（WM-only→final） | 1ecbb65b5d9fdc92 | 1ecbb65b5d9fdc92 | 3c0066b19e69850e | 三组均不变；C 与 Teacher bitwise 相等，A/B 与 Teacher 不同 |
| Actor/Critic optimizer 在 iteration1000 的 state 数 | 0 / 0 | 0 / 0 | 0 / 0 | wm_calibrated.pt |
| WM Adam step（1000→2000） | 16000→32000 | 16000→32000 | 16000→32000 | Checkpoint opt_wm.state.step |
| Actor Adam step（final） | 1000 | 1000 | 1000 | 仅 10 个 Encoder 参数 tensor 有 optimizer state |
| Critic Adam step（final） | 8000 | 8000 | 8000 | 1000×2 epochs×4 minibatches |
| Foot Encoder final 是否改变 | 是 | 是 | 是 | warm 与 final enc.* 哈希不同 |
| WM warm 后是否继续训练 | 是 | 是 | 是 | WM 哈希改变且 optimizer step 增加 |
| 最终权重有限 | 是 | 是 | 是 | CPU 对 WM/π/Critic 全部浮点 tensor 检查 |

以上由 CPU 重新读取 [A-final](../../logs/s54_abc_short_20261105_184038_A/last.pt)；[B-final](../../logs/s54_abc_short_20261105_184038_B/last.pt)；[C-final](../../logs/s54_abc_short_20261105_184038_C/last.pt)。[A-warm](../../logs/s54_abc_short_20261105_184038_A/wm_calibrated.pt)；[B-warm](../../logs/s54_abc_short_20261105_184038_B/wm_calibrated.pt)；[C-warm](../../logs/s54_abc_short_20261105_184038_C/wm_calibrated.pt)；并复核 [A-audit](../../logs/s54_abc_short_20261105_184038_A/training_audit.json)；[B-audit](../../logs/s54_abc_short_20261105_184038_B/training_audit.json)；[C-audit](../../logs/s54_abc_short_20261105_184038_C/training_audit.json)，结果如上表。本批次没有迭代 0 Checkpoint，故 WM 初始化瞬间的逐 tensor 副本 **N/A**；其加载源依据严格迁移映射、实存 `transfer_report` 和相同执行源码核实，不把 1000 次训练后的 WM 误当作原始源 WM。

**随机 Decoder 冻结的关键限制：**源码对新 Policy 的 Decoder 最后一层 weight 先乘 0.01、bias 清零；A/B 随后冻结整个 Decoder。Encoder 输出又经 tanh 限幅，不能通过任意放大 latent 抵消这个固定输出层。两组真实评估动作 RMS 仅约 2.9e-4；这检验的是“固定随机映射下的 Encoder-only 适配”，不是通常可训练完整 Actor 的从零能力。C 同时面对新的随机 Foot latent、重置后的 ObsNorm/GoalNorm 与参考朝向 mask，不能预期复现原 Motion Encoder 驱动的 Teacher 行为。实现证据：[supermimic/networks.py](../../logs/s54_abc_short_20261105_184038_A/source_snapshot/supermimic/networks.py)、[supermimic/transfer.py](../../logs/s54_abc_short_20261105_184038_A/source_snapshot/supermimic/transfer.py)。

WM-only 表示不执行 Actor/Critic optimizer；Actor 的观测/Goal running normalization 仍随采样更新。因此 C 在前 1000 iterations 的行为变化不能归因于 Actor 权重更新；不同阶段的 episode 分布与归一化变化仍会影响表现。

## 5. 训练结果与趋势

曲线来源：[A-CSV](../../logs/s54_abc_short_20261105_184038_A/train_log.csv)；[B-CSV](../../logs/s54_abc_short_20261105_184038_B/train_log.csv)；[C-CSV](../../logs/s54_abc_short_20261105_184038_C/train_log.csv)。每组 100 行、20→2000（步长20），无缺行、重复 iteration 或非有限数值。补充来源：[A-grad](../../logs/s54_abc_short_20261105_184038_A/gradient_terms.jsonl)；[B-grad](../../logs/s54_abc_short_20261105_184038_B/gradient_terms.jsonl)；[C-grad](../../logs/s54_abc_short_20261105_184038_C/gradient_terms.jsonl)。[A-extra](../../logs/s54_abc_short_20261105_184038_A/curriculum_metrics.jsonl)；[B-extra](../../logs/s54_abc_short_20261105_184038_B/curriculum_metrics.jsonl)；[C-extra](../../logs/s54_abc_short_20261105_184038_C/curriculum_metrics.jsonl)。[A-reward](../../logs/s54_abc_short_20261105_184038_A/reward_terms.jsonl)；[B-reward](../../logs/s54_abc_short_20261105_184038_B/reward_terms.jsonl)；[C-reward](../../logs/s54_abc_short_20261105_184038_C/reward_terms.jsonl)。

### 5.1 指标口径

- `sim_rew`：采集的真实 Newton 单步 Reward 除以 r_max 后，跨环境/采集步均值。
- `ep_ret_norm`：每个已结束 episode 的 `Σr/(实际执行步数×r_max)`；CSV 再对各采集步内已结束 episode 的均值取平均，因此不是按全部 episode 数重新加权的全局均值。`ep_len` 同样仅统计已结束 episode。
- `term_frac`：日志窗口内物理终止计数/所有 done 计数，而不是每个环境步的“跌倒概率”。训练三组所有窗口均为 1.0。
- `imag_rew`：Replay 起点的 32 步 imagination，按 soft survival 加权的 Reward 总和除以 `32×r_max`；与真实 collector 的状态分布及权重不同。前 1000 iterations 的 0 表示**未计算**，不是模型预测回报为零。
- `pi_loss`：负的 discounted imagined Reward 加 Critic bootstrap 目标，不是 normalized return；越负不能直接推出 Newton 更好。`v_loss` 是返回的最后一个 Critic minibatch MSE 再按日志窗口平均，不是全部 Critic batch 的完整总体误差。
- `pi_gradnorm`：Actor 裁剪前总梯度范数；`foot_actor_gradnorm`：排除动作成本、Posture、Base/Critic 和 survival 导数后的纯 Foot Reward 梯度。其非零证明链路存在，不能证明梯度方向对应真实动力学。
- `wm_loss/wm_err8`：归一化状态预测误差，不能直接当作米或度；CSV 每次记录取的是当轮最后一次 WM batch 的返回值再按窗口平均。物理单位误差另见配对评估。

### 5.2 初期 / 中期 / 末期（阶段窗口均值）

每个表内窗口含 10 条 CSV 记录；每条记录通常覆盖前 20 iterations。`20–200` 包含起始 warmup 的采样统计；`820–1000` 为 WM-only 末期；后面三个窗口分别覆盖 RL 早期、中期、末期。

| 组 | 窗口 | sim_rew | imag_rew | ep_ret_norm | ep_len / steps | 对应时长 / s | term_frac |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | WM 初期 20–200 | 0.686563 | N/A（未更新） | 0.682944 | 97.8366 | 1.957 | 1 |
| A | WM 末期 820–1000 | 0.681632 | N/A（未更新） | 0.682405 | 97.672 | 1.953 | 1 |
| A | RL 初期 1020–1200 | 0.682319 | 0.542008 | 0.682206 | 97.6012 | 1.952 | 1 |
| A | RL 中期 1420–1600 | 0.682852 | 0.544394 | 0.682521 | 97.7698 | 1.955 | 1 |
| A | RL 末期 1820–2000 | 0.682505 | 0.544135 | 0.68256 | 97.6205 | 1.952 | 1 |
| B | WM 初期 20–200 | 0.686563 | N/A（未更新） | 0.682944 | 97.8366 | 1.957 | 1 |
| B | WM 末期 820–1000 | 0.681653 | N/A（未更新） | 0.68241 | 97.6876 | 1.954 | 1 |
| B | RL 初期 1020–1200 | 0.682288 | 0.544689 | 0.682249 | 97.5947 | 1.952 | 1 |
| B | RL 中期 1420–1600 | 0.682862 | 0.547702 | 0.682646 | 97.7442 | 1.955 | 1 |
| B | RL 末期 1820–2000 | 0.682552 | 0.54602 | 0.682568 | 97.6338 | 1.953 | 1 |
| C | WM 初期 20–200 | 0.294947 | N/A（未更新） | 0.401658 | 133.027 | 2.661 | 1 |
| C | WM 末期 820–1000 | 0.427212 | N/A（未更新） | 0.435689 | 58.9274 | 1.179 | 1 |
| C | RL 初期 1020–1200 | 0.60887 | 0.320215 | 0.602809 | 60.4031 | 1.208 | 1 |
| C | RL 中期 1420–1600 | 0.610402 | 0.412455 | 0.61057 | 60.2753 | 1.206 | 1 |
| C | RL 末期 1820–2000 | 0.597405 | 0.39085 | 0.597199 | 57.5386 | 1.151 | 1 |

| 组 | 窗口 | wm_loss | wm_err8 | pi_loss | v_loss | pi_gradnorm | foot_actor_gradnorm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | WM 初期 20–200 | 0.0517973 | 0.0663527 | N/A | N/A | N/A | N/A |
| A | WM 末期 820–1000 | 0.00594681 | 0.00619697 | N/A | N/A | N/A | N/A |
| A | RL 初期 1020–1200 | 0.00536105 | 0.00561655 | -1.71859 | 0.00730632 | 0.000261459 | 1.36004e-05 |
| A | RL 中期 1420–1600 | 0.00452337 | 0.00470736 | -1.73963 | 0.000600366 | 0.000183571 | 1.68393e-05 |
| A | RL 末期 1820–2000 | 0.00389458 | 0.00394229 | -1.74207 | 0.000387291 | 0.000185667 | 1.56893e-05 |
| B | WM 初期 20–200 | 0.00427023 | 0.00766643 | N/A | N/A | N/A | N/A |
| B | WM 末期 820–1000 | 0.00099181 | 0.00174549 | N/A | N/A | N/A | N/A |
| B | RL 初期 1020–1200 | 0.000814211 | 0.00142355 | -1.74461 | 0.00761144 | 0.000159236 | 8.9763e-06 |
| B | RL 中期 1420–1600 | 0.000627484 | 0.00106823 | -1.76534 | 0.000880281 | 4.07934e-05 | 2.78363e-06 |
| B | RL 末期 1820–2000 | 0.000541186 | 0.00090152 | -1.76052 | 0.000562907 | 4.65619e-05 | 1.06634e-05 |
| C | WM 初期 20–200 | 0.0521552 | 0.0743739 | N/A | N/A | N/A | N/A |
| C | WM 末期 820–1000 | 0.0157838 | 0.0245475 | N/A | N/A | N/A | N/A |
| C | RL 初期 1020–1200 | 0.0123314 | 0.01918 | -0.844575 | 0.00300753 | 0.153213 | 0.0826437 |
| C | RL 中期 1420–1600 | 0.00525042 | 0.00869066 | -1.07091 | 0.000708711 | 0.175199 | 0.107178 |
| C | RL 末期 1820–2000 | 0.00439252 | 0.0070917 | -0.996339 | 0.000654739 | 0.236383 | 0.116223 |

趋势解释：A/B 的真实 Reward 和 episode 时长从 WM-only 到 RL 几乎不变；B 的预测误差显著更低。C 的 Reward 在 RL 初期从约 0.43 提升到约 0.61，但 episode 始终只有约 60 步；RL 末期 `ep_ret_norm` 回落到约 0.597、时长约 57.5 步，未出现持续存活改善。C 的 WM 误差下降与控制能力改善不能等同。

### 5.3 最后一行与更新/完成情况

| 最终 CSV 指标（iteration2000） | A | B | C |
| --- | --- | --- | --- |
| sim_rew | 0.687121 | 0.686139 | 0.594263 |
| imag_rew | 0.541914 | 0.543642 | 0.386673 |
| ep_ret_norm | 0.6835 | 0.683465 | 0.59473 |
| ep_len | 97.7711 | 97.8613 | 57.2963 |
| term_frac | 1 | 1 | 1 |
| wm_loss | 0.00382344 | 0.000550767 | 0.00437781 |
| wm_err8 | 0.00393449 | 0.000926709 | 0.00705755 |
| pi_loss | -1.71791 | -1.74031 | -0.98433 |
| v_loss | 0.000355315 | 0.000574351 | 0.000621535 |
| pi_gradnorm | 0.000227219 | 4.74363e-05 | 0.206194 |
| foot_actor_gradnorm | 1.31286e-05 | 1.22988e-05 | 0.114545 |

| 完成/更新统计 | A | B | C |
| --- | --- | --- | --- |
| Actor 更新次数（stage_stats 与 Adam 双重核实） | 1000 | 1000 | 1000 |
| WM optimizer step（Adam 计数） | 32000 | 32000 | 32000 |
| Critic optimizer step | 8000 | 8000 | 8000 |
| imagined Foot transitions（非真实仿真） | 8,192,000 | 8,192,000 | 8,192,000 |
| imagined Motion transitions / paired supervision | 0 / 0 | 0 / 0 | 0 / 0 |
| 已完成 episode 数（CSV completed_episodes 求和） | 53,262 | 53,264 | 81,793 |
| 日志窗口最大 v_loss | 0.0429973 | 0.0410261 | 0.0108514 |
| 日志窗口最大 pi_gradnorm | 0.000626409 | 0.000468393 | 0.303405 |
| 终态 / Checkpoint 有限 | completed / finite | completed / finite | completed / finite |

更新次数不只由 iterations×配置推算：已核对 Checkpoint 中 Adam 的 `state.step`。全部 CSV 记录有限且终态权重有限；本批次没有先前实验那样 Critic 数值爆炸的证据。最大值是**记录窗口均值的最大值**，不能当作未记录的单次更新峰值。训练的逐 episode height/tilt/nonfinite 原因清单为 N/A；终止分项由独立评估另列。

### 5.4 Reward 分项与梯度诊断

| 真实采样分项均值（RL末期1820–2000） | A | B | C |
| --- | --- | --- | --- |
| swing_tracking | 0.29483 | 0.294929 | 0.288264 |
| swing_velocity | 0.346832 | 0.346929 | 0.303251 |
| stance_stability | 0.510543 | 0.510505 | 0.547006 |
| stance_velocity | 0.592662 | 0.592546 | 0.570725 |
| foot_orientation | 0.896283 | 0.896199 | 0.80189 |
| base_upright | 0.716892 | 0.716998 | 0.620725 |
| base_height | 0.866605 | 0.866635 | 0.778863 |
| base_balance | 0.638448 | 0.638518 | 0.550466 |
| action_rate | 0.0519307 | 0.0519268 | 0.836241 |
| joint_limit | 0.000588802 | 0.000591374 | 0.00535968 |
| posture | 0.0197665 | 0.0197374 | 1.17667 |
| foot_objective | 0.0253594 | 0.0253601 | 0.0251624 |

这些是原始分项均值：正项多数是质量值，`action_rate/joint_limit/posture` 是**尚未乘负权重**的误差，`foot_objective` 已乘 dt。C 的 stance position 分项更高，但 Swing velocity、orientation、upright/height/balance 更低，动作变化与偏离默认姿态的代价明显更大。计划 touchdown 分项大多数采样步为 0，不能将其全步均值当作真实触地误差。

| 末次额外梯度审计（iteration2000；日志窗口口径） | A | B | C |
| --- | --- | --- | --- |
| actor_reward_gradnorm | 4.97015e-05 | 3.44921e-05 | 0.138834 |
| actor_bootstrap_gradnorm | 0.000409446 | 5.22209e-05 | 0.248408 |
| actor_grad_cosine | -0.681971 | -0.70589 | 0.576002 |
| pure_foot_gradnorm | 1.32914e-05 | 1.46627e-05 | 0.12187 |
| imagined_action_clip_fraction | 0 | 0 | 0.00100876 |

A/B 的 pure-foot 梯度很小但非零；C 明显更大。这与冻结随机 Decoder 的小输出层以及 C 的迁移动作尺度相符，是训练瓶颈的**初步迹象**，不是已验证的唯一失效原因。A/B 末次 reward 与 bootstrap 梯度 cosine 为负，表示这次记录中两个目标方向冲突；不能据此单独宣布 Critic 错误。同期纯 Foot goal/reward 回放时钟误差最大值分别为 A/B 1.78814e-7 / 3.72529e-8，C 5.24521e-6 / 4.09782e-8，均低于源码 1e-5 检查阈值。

## 6. 独立 Newton 评估

已有评估，**不需要根据 Reward 推断，也未在本轮补跑**。来源：[A-eval](../../logs/s54_abc_short_20261105_184038_A/evaluation/summary.json)；[B-eval](../../logs/s54_abc_short_20261105_184038_B/evaluation/summary.json)；[C-eval](../../logs/s54_abc_short_20261105_184038_C/evaluation/summary.json)。每组 1 环境×seed 54/55/56×时限2/5/10 s，`controller=policy`、reset_noise=true、无训练动作/观测噪声、固定域参数。27 个 NPZ 的动作、真实状态、接触与 Reward 记录可读且状态有限，按 NPZ Reward 重算的 normalized return 全部与 JSON 一致；不同组在同一 seed 下的初始 Foot 命令张量完全相同。

**Survival Rate 定义：**执行完整协议时限且未发生 Physical Termination 的 episode 比例；它不是完整 Walking Success Rate。2 s 的 seed55 在第100步同时触及时限/高度阈值，按物理终止计。各 horizon 重用了相同三个 seed/初始命令，因此 9 个结果不是 9 个独立随机初态；只有一个训练种子，不能给出跨训练重复的显著性结论。

| 组 | 时限 / s | Survival Rate | norm_return | 平均实际执行 / s | Foot Position RMSE / cm | Scheduled TD Error / cm | 终止 / 3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 2 | 33.3% (1/3) | 0.66074884 | 1.913 | 4.2983 | 5.8639 | 2 height / 1 timeout |
| A | 5 | 0.0% (0/3) | 0.26454155 | 1.92 | 4.3436 | 5.8639 | 3 height / 0 timeout |
| A | 10 | 0.0% (0/3) | 0.13227069 | 1.92 | 4.3436 | 5.8639 | 3 height / 0 timeout |
| B | 2 | 33.3% (1/3) | 0.6608541 | 1.913 | 4.2968 | 5.8617 | 2 height / 1 timeout |
| B | 5 | 0.0% (0/3) | 0.2645837 | 1.92 | 4.3419 | 5.8617 | 3 height / 0 timeout |
| B | 10 | 0.0% (0/3) | 0.13229155 | 1.92 | 4.342 | 5.8617 | 3 height / 0 timeout |
| C | 2 | 0.0% (0/3) | 0.36725391 | 1.18 | 3.7919 | 5.3218 | 3 height / 0 timeout |
| C | 5 | 0.0% (0/3) | 0.14691274 | 1.18 | 3.7905 | 5.3218 | 3 height / 0 timeout |
| C | 10 | 0.0% (0/3) | 0.073464332 | 1.18 | 3.788 | 5.3191 | 3 height / 0 timeout |

除 Survival/计数外，上表是三个 seed 的算术均值；Foot/Touchdown 是**各 episode RMSE 的均值**，不是把所有帧/触地事件池化后的 RMSE。误差只覆盖实际执行到物理终止的区间，尚未完成的后续目标不在误差分母中。C 的较低短时 Foot/Touchdown RMSE 尤其受更早终止、事件覆盖减少影响。

### 6.1 每个种子的物理终止

| 组 | seed | 2s：步数 / 原因 | 5s：步数 / 原因 | 10s：步数 / 原因 | 10s协议实际时长 / s |
| --- | --- | --- | --- | --- | --- |
| A | 54 | 100 / timeout | 101 / height | 101 / height | 2.02 |
| A | 55 | 100 / height | 100 / height | 100 / height | 2.00 |
| A | 56 | 87 / height | 87 / height | 87 / height | 1.74 |
| B | 54 | 100 / timeout | 101 / height | 101 / height | 2.02 |
| B | 55 | 100 / height | 100 / height | 100 / height | 2.00 |
| B | 56 | 87 / height | 87 / height | 87 / height | 1.74 |
| C | 54 | 59 / height | 59 / height | 59 / height | 1.18 |
| C | 55 | 60 / height | 60 / height | 60 / height | 1.20 |
| C | 56 | 58 / height | 58 / height | 58 / height | 1.16 |

没有这些 Walking episode 的 Clip End；没有将 Timeout 当作物理跌倒。跟踪失败不是本次 JSON 的独立 stop reason；从向后速度、足部误差和提前 height 终止可确认没有完成持续前进，但没有补造独立的 Goal Tracking Failure 次数或 Walking Success Rate。

### 6.2 姿态、接触、滑移、上半身与实际动作

| 10s协议指标（实际均提前结束） | A | B | C |
| --- | --- | --- | --- |
| Root Roll 最大绝对值 / deg | 0.450474 | 0.450474 | 5.96971 |
| Root Pitch 最大绝对值 / deg | 45.4101 | 45.3707 | 33.6682 |
| Root Height 最大偏差 / m | 0.268486 | 0.267943 | 0.281215 |
| 计划平均前进速度 / m/s | 0.0403124 | 0.0403124 | 0.0403124 |
| 实际平均前进轴速度 / m/s | -0.384231 | -0.383943 | -0.602908 |
| 实际平均 yaw rate / rad/s | 0.00320199 | 0.00322084 | 0.0594713 |
| Foot Orientation RMSE / deg | 11.6581 | 11.6442 | 14.3128 |
| 承载足 XY 速度（slip proxy）/ m/s | 0.0331272 | 0.0330781 | 0.0600692 |
| 计划/测得 contact state accuracy | 0.59892 | 0.59892 | 0.644001 |
| 实际承载接触边沿数（seed54/55/56） | 1, 2, 2 | 1, 2, 2 | 3, 2, 3 |
| 计划落脚事件数（实际执行区间；seed54/55/56） | 1, 2, 1 | 1, 2, 1 | 1, 1, 1 |
| Upper-Body Pose Error / rad | N/A：无配对全身参考 | N/A：无配对全身参考 | N/A：无配对全身参考 |
| Upper joint velocity RMS / rad/s | 0.0176568 | 0.017652 | 0.524883 |
| Upper joint acceleration RMS / rad/s² | 0.403285 | 0.403187 | 10.9295 |
| 真实动作 RMS（跨seed的均值；未乘关节scale） | 0.000287887 | 0.000294533 | 0.773475 |
| 真实动作最大绝对值（跨seed最大） | 0.000961304 | 0.000999451 | 4.03125 |

Root Roll/Pitch/Height 行取三个 seed 的**最大值**，其他连续指标取三个 episode 的均值。计划平均速度约 +0.0403 m/s，而实际速度均为负，C 向后速度更大。C 的较小 Pitch 最大值不能解释为更稳：其观察窗口较短，已经因高度越界停止。Upper velocity/acceleration 只描述运动幅度，不能替代自然摆臂、全身姿态跟踪或 LAFAN1 配对误差；A/B 上肢几乎不动也不等于自然。

真实接触来自 Newton 最后物理子步 solver contacts/world forces，承载定义 `Fz>1 N`；不是计划 contact mask，也不是整个控制周期累计冲量。滑移字段实际是承载时 **踝 link 原点**的 XY 速度均值，未在真实接触点重建切向滑移，故本文标为 slip proxy。Contact 边沿数可能含接触抖动，不能作为完成了多少个有效步态周期。Scheduled TD Error 也不是对传感器实际触地瞬间的单独误差。口径：[评估器快照](../../logs/s54_abc_short_20261105_184038_B/source_snapshot/tools/validation/evaluate_s54_motion_prior.py)、[supermimic/standing_diagnostics.py](../../logs/s54_abc_short_20261105_184038_A/source_snapshot/supermimic/standing_diagnostics.py)。

| 产物 | A | B | C |
| --- | --- | --- | --- |
| 独立 summary.json | [已存在](../../logs/s54_abc_short_20261105_184038_A/evaluation/summary.json) | [已存在](../../logs/s54_abc_short_20261105_184038_B/evaluation/summary.json) | [已存在](../../logs/s54_abc_short_20261105_184038_C/evaluation/summary.json) |
| 真实轨迹/接触 NPZ 数 | 9 | 9 | 9 |
| 诊断 PNG 数 | 9 | 9 | 9 |
| 代表轨迹（10s协议，seed54） | [NPZ](../../logs/s54_abc_short_20261105_184038_A/evaluation/alternating_10s_seed54.npz) | [NPZ](../../logs/s54_abc_short_20261105_184038_B/evaluation/alternating_10s_seed54.npz) | [NPZ](../../logs/s54_abc_short_20261105_184038_C/evaluation/alternating_10s_seed54.npz) |
| MP4/视频 | N/A：未发现 | N/A：未发现 | N/A：未发现 |

已检索各组目录及 `results/` 的本批次 ABC/ablation 命名视频，未找到对应 MP4/WebM/AVI；没有为了补报告启动新渲染。现有评估虽标记 `split=train`，这里没有 MotionLib/Clip 输入，不能将它描述成 LAFAN1 全 Clip 评估或独立 Walking holdout。

## 7. WM 物理配对误差与 Imagined/Simulation 差异

来源为各 `evaluation/summary.json` 的 `wm_paired.horizons`，协议是同一已记录 Newton 起点、相同动作/ξ/Foot 命令与 tick 的开环配对；没有新增仿真。以下列出 **10s协议实际失败轨迹**上、三个 seed 指标的均值：

| WM8 配对指标 | A | B | C |
| --- | --- | --- | --- |
| Base Height RMSE / mm | 1.03056 | 0.266484 | 0.200667 |
| Root Pitch RMSE / deg | 0.0645337 | 0.0330755 | 0.0605396 |
| Foot FK RMSE / mm | 0.942322 | 0.364981 | 2.03382 |
| Normalized paired reward bias（pred−Newton） | -0.000187868 | -1.72183e-05 | -0.000845107 |
| Normalized paired reward MAE（窗口均值口径） | 0.000546554 | 0.000147675 | 0.000954559 |
| 已记录的配对 horizon | 1 / 4 / 8 | 1 / 4 / 8 | 1 / 4 / 8 |
| 16 / 32 步配对误差 | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED |

Foot FK 误差是 WM 预测足位置相对 Newton 真值的误差，不是足位置相对目标的 Foot Tracking RMSE。`paired_reward_mae` 的实现是每个8步窗口平均 Reward 差的绝对值再取均值，存在窗口内正负抵消；它不是 `per_step_reward_mae`，两者不能混称。

WM8 在这些已访问短失败轨迹上的高度、Pitch、Foot FK 和 normalized reward MAE 数值均落在仓库现有 WM 阈值内（0.01 m / 1° / 0.015 m / 0.02）；这只覆盖观测状态上的数值项，**不认证接触边缘/动作扰动覆盖、不认证动作导数，也不认证 H32 BPTT**。B 的误差比 A 小提供 WM Transfer 改善本分布预测的证据；C 与 B 的轨迹/动作分布明显不同，不能仅比较其归一化训练 loss 宣称同一分布上的 WM 迁移失败。阈值来源：[pre_ablation_gates.json](../../configs/motion_prior/validation/pre_ablation_gates.json)。

RL末期的 `imag_rew` 均低于 `sim_rew`：A约0.544/0.683，B约0.546/0.683，C约0.391/0.597。但其采样分布、Replay 起点、soft survival 权重不同，**不能直接将二者相减当作模型 Reward Bias**。配对 WM8 bias 则很小且平均为负，没有这批8步样本上明显乐观高估的证据；尚不能排除32步滚动偏差、未覆盖动作或接触附近的错误梯度。长期真实表现差与短期预测准确可以同时成立：WM 可以准确预测一个会跌倒的策略。

此外，独立 `norm_return=Σ实际Reward/(计划100/250/500步×r_max)`；物理终止后剩余时限没有 Reward。训练 `ep_ret_norm` 则除以**已经执行的步数**，所以 A/B 训练均值约0.683与10s独立评估约0.132并不矛盾：平均只执行1.92/10的时限。C 也从训练约0.597降为10s评估约0.073，主要受1.18s提前终止影响。不能把训练均值当成10秒持续任务回报。

## 8. SwanLab 本地记录与数据完整性

本轮只离线读取 `.swanlab` 二进制记录（SDK DataStoreReader、CRC检查、protobuf ScalarRecord）及既有上传回执，没有新建实验或调用云端 API。每组 17 个映射训练指标×100个iteration，共1700个点，与实际 CSV 逐点比较，无缺失、无冲突；每组61个 Eval点与持久化 Eval bundle 一致。

| 组 | 实验名 / 链接 | 训练上传最后step | Eval回执 | 离线CSV核对 | 本地 Eval 点 |
| --- | --- | --- | --- | --- | --- |
| A | [A_20261105_184038_seed54_walking](https://swanlab.cn/@yagami_light/Foothold_with_SuperMimic/runs/d9595555796a417b/chart) | 2000 | [verified](../../logs/s54_abc_short_20261105_184038_A/evaluation/upload_status.json) | 1700/1700，一致 | 61 |
| B | [B_20261105_184038_seed54_walking](https://swanlab.cn/@yagami_light/Foothold_with_SuperMimic/runs/2630266baa764aeb/chart) | 2000 | [verified](../../logs/s54_abc_short_20261105_184038_B/evaluation/upload_status.json) | 1700/1700，一致 | 61 |
| C | [C_20261105_184038_seed54_walking](https://swanlab.cn/@yagami_light/Foothold_with_SuperMimic/runs/1ab1745e857846b2/chart) | 2000 | [verified](../../logs/s54_abc_short_20261105_184038_C/evaluation/upload_status.json) | 1700/1700，一致 | 61 |

三个 `evaluation/upload_status.json` 均记录当时云端读回验证成功、`original_training_curves_unchanged=true`；`cloud_eval_readback.json` 保存验证结果。本轮不将这些历史回执表述为重新在线核验。A 曾因旧 tmux 代理失效而上传超时，恢复后复用了完整 A 训练/评估并继续 B/C；没有重跑或伪造 A 结果。批次状态里的上传值是最后阶段切换时的快照，C 当时为 `checking_cloud`，应以随后各组独立的 verified 回执为准。

`foot_actor_gradnorm` 未作为 CSV 列直接映射到 SwanLab 的100个训练点；本地 CSV 完整保留，另外 `Optimization/pure_foot_gradnorm` 等只在额外梯度审计时存在。不能把 UI 未显示该列解释为零梯度。本报告对该指标使用 CSV/gradient_terms，而非缺失的 UI 曲线。

本地原始记录代表文件：

- A：[logs/s54_abc_short_20261105_184038_A/swanlab/run-20261105_200044-d9595555796a417b/run-d9595555796a417b.swanlab](../../logs/s54_abc_short_20261105_184038_A/swanlab/run-20261105_200044-d9595555796a417b/run-d9595555796a417b.swanlab)
- B：[logs/s54_abc_short_20261105_184038_B/swanlab/run-20261105_200055-2630266baa764aeb/run-2630266baa764aeb.swanlab](../../logs/s54_abc_short_20261105_184038_B/swanlab/run-20261105_200055-2630266baa764aeb/run-2630266baa764aeb.swanlab)
- C：[logs/s54_abc_short_20261105_184038_C/swanlab/run-20261105_205838-1ab1745e857846b2/run-1ab1745e857846b2.swanlab](../../logs/s54_abc_short_20261105_184038_C/swanlab/run-20261105_205838-1ab1745e857846b2/run-1ab1745e857846b2.swanlab)

Checkpoint、args、训练CSV/JSONL、原评估JSON及Teacher的内容摘要在本次审计前后复核。只新增本文，没有将统计结果回写任何原始文件；审计临时计算文件仅位于 `/tmp`。

## 9. 六个问题的客观回答

### 9.1 A vs B：能否支持 World Model Transfer 有效？

**已验证事实：**B 的 RL末期 wm_err8 为 0.00090152, A 为 0.00394229，约降低 77.1%；配对 WM8 高度/Pitch/FK/Reward 误差也更低。A/B 的 Encoder 初值和随机冻结 Decoder 一致，实际超参数相同。支持“本次固定随机 Decoder、保守短步分布上，预训练 WM+归一化迁移改善预测/拟合效率”。

**不能推出：**更好的真实行走或更高策略样本效率。A/B 5/10s Survival都为0，执行步数相同；10s norm_return差仅 0.00002086，且只有一个训练种子。WM归一化统计有不同历史，训练loss的尺度比较需由物理配对误差辅助；目前也没有多训练seed显著性证据。

### 9.2 B vs C：能否支持 Shared Decoder Transfer 有效？

**已验证事实：**C 确实加载并冻结了同一 Teacher Decoder，训练 Foot Encoder 的梯度/更新有效，动作尺度与 B 明显不同。但平均实际存活由1.92降到1.18s，10s norm_return由0.13229降到0.07346，2s Survival由1/3降到0/3，滑移 proxy 与上肢运动幅度上升。

**结论：**不能支持“直接 Decoder Transfer 改善本 Foot 任务”；在本设置、单种子下，主性能结果反而更差。C 的 Foot/TD误差和contact accuracy等局部指标较好，但评估前缀更短、落脚事件更少，不能解释为连续行走进步，也不能据此否定经过Bridge或微调后的Decoder迁移。

### 9.3 哪些改善、哪些退化？

| 对照 | 改善的已测指标 | 持平 / 退化 / 限制 |
| --- | --- | --- |
| A→B | WM预测误差与拟合速度；配对WM8物理误差 | 真实episode长度/Survival基本不变；训练Reward几乎相同；不能确认策略收益 |
| B→C | 短失败前缀Foot/TD RMSE较小；contact accuracy约0.599→0.644；Encoder梯度更大 | 存活/return更差；滑移约0.0331→0.0601m/s；foot orientation误差更大；上肢velocity约0.0177→0.525rad/s |
| C内早期→中期→末期 | WM拟合持续下降；RL初期Reward提升 | 存活没有随Reward增长；中期到末期ep_ret约0.611→0.597，ep_len约60.3→57.5 |

误差和contact的改善仅是已发生轨迹上的条件统计；Pitch峰值下降及更大上肢运动幅度均不能直接当作稳定性/自然性改善。

### 9.4 Imagined Reward 是否与真实表现不一致？

**已验证事实：**训练的 imagined/sim 均值数值不同，训练短episode Reward 与长时限评估return也不同。前者不是配对batch，后者分母不同，因此不能用曲线差直接诊断 WM 幻觉。WM8配对结果未发现大幅乐观RewardBias；H32真实配对误差及动作敏感度尚未验证。**尚未验证的假设：**长预测、接触动作导数、Critic bootstrap 或新latent/normalization错配限制了策略学习。它们需要独立诊断，当前不选定一个作为确定原因。

### 9.5 是否具备开展 D/F/G 的条件？

| Gate | 当前证据 | 判定 |
| --- | --- | --- |
| ABC工程完成 / 初始化隔离 | 三组均完整预算；D冻结、E更新、WM独立更新、args/物理合同一致 | PASS（本批次） |
| 短观测轨迹WM8数值 | 已测1/4/8步；物理误差小 | 局部数值项满足；覆盖/导数/H32未认证 |
| 连续Walking能力 | 5/10s三组均0/3Survival，实际前进速度为负 | FAIL |
| 同一Foot合同的Standing基线 | 本批次全部alternating；无独立Standing记录 | NOT EVALUATED |
| v1 Teacher独立Walking holdout/全身自然性 | 源manifest仅train walk1、heldout dance1，缺stand/turn/slow_walk/start_stop | 未满足；Dance不是Walking holdout |
| v1 Teacher的Stage1.5 Bridge效果 | 本批次没有Bridge；历史接口Smoke不是本v1配对实验 | NOT EVALUATED |
| F/G动态Upper Prior闭环效果 | ABC prior_config=None；上半身pose误差缺少合法匹配参考 | NOT EVALUATED |

**总体：BLOCKED（正式D/F/G性能消融）**。已有接口和本批次上传/Checkpoint不构成稳定能力验收。可把有界D/Bridge适配作为下一步诊断，但需要先预登记同一v1来源、只更新Foot Encoder的Motion Reward+BPTT目标、预算和真实评估；不能直接将F/G长训当成ABC失败的补救，也不能仅加大Prior权重掩盖跌倒。`evaluation/summary.json` 中的 `standing_gates.passed=false` 是Standing专用逻辑，对alternating必然不通过，不将这个字段冒充独立Walking Gate；上面的Walking失败结论来自真实height终止及速度指标。

### 9.6 是否需要 A-full？

**需要。**应新增随机初始化且 Encoder+Decoder 全部可训练的 **A-full**，保留当前 A 为“A-frozen-random”历史对照。否则无法回答正常从零Foot Actor能否学会任务，也无法判断C优劣是否只是固定随机Decoder限制了A/B。为了验证WM迁移对完整Actor的收益，还应有同样Decoder可训练的 **B-full**；不能只把A改为full、保留B frozen后继续声称A/B只差WM。A-full/B-full采用新的实验ID、相同Foot分布/reset/reward/物理阈值/真实样本预算，单独报告预训练成本，不能覆盖当前ABC。

## 10. 下一步建议与未验证假设

| 优先级 | 最小下一步（本轮未执行） | 要解决的问题 / 验收方式 |
| --- | --- | --- |
| 1 | 为当前Foot合同补独立Standing与zero/default-PD对照；检查初态、足接触、后倾与PD恢复 | 用同一2/5/10s协议和原物理阈值区分被动初态失败、策略失败；不要放宽height/tilt |
| 2 | 新ID进行A-full/B-full的有界动作/梯度Smoke，再以相同真实样本预算比较 | 先确认动作可离开近零区、Encoder/Decoder均更新且不会数值失稳；A/B间仅改变WM来源 |
| 3 | 利用保存轨迹设计WM 16/32步配对与有限动作扰动检查 | 检查H32接触/动作导数，而不是继续延长已接近收敛的WM-only来假定能解决控制 |
| 4 | 补v1Teacher独立Walking holdout和合法Upper-Body配对评估 | 原manifest仅walk1训练、dance heldout；添加Walking holdout后预登记自然性/跟踪门槛 |
| 5 | 单独有界D/Stage1.5 Bridge验证，再决定是否进入F/G | 固定v1WM/Decoder，Motion Reward+BPTT只训练Foot Encoder，明确源/监督/真实样本和wall cost；需真实闭环收益后才做Prior对照 |
| 6 | 扩充训练seed并保持固定评估协议 | 建议训练seed54/55/56作为独立重复；评估seed和命令固定，报告跨训练seed波动，保留历史结果 |

上述为建议，不是已执行结果。冻结随机Decoder的小输出尺度、新Foot latent与Teacher latent分布不匹配、归一化/adapter变化、长Horizon梯度质量以及Foot reward在失败前仍较高等，都是有证据支持进一步检查的候选机制；目前没有干预对照能将其排序为唯一根因。

## 11. 结果文件与校验摘要

| 组 | 关键统计证据 | 最终 last.pt SHA256 |
| --- | --- | --- |
| A | [CSV](../../logs/s54_abc_short_20261105_184038_A/train_log.csv) / [audit](../../logs/s54_abc_short_20261105_184038_A/training_audit.json) / [Newton summary](../../logs/s54_abc_short_20261105_184038_A/evaluation/summary.json) / [Eval readback](../../logs/s54_abc_short_20261105_184038_A/evaluation/cloud_eval_readback.json) | 3ba00822ce3cf0ecdde66b900ca70f9714ec692b16d64649b6e4d4a6d4a9136c |
| B | [CSV](../../logs/s54_abc_short_20261105_184038_B/train_log.csv) / [audit](../../logs/s54_abc_short_20261105_184038_B/training_audit.json) / [Newton summary](../../logs/s54_abc_short_20261105_184038_B/evaluation/summary.json) / [Eval readback](../../logs/s54_abc_short_20261105_184038_B/evaluation/cloud_eval_readback.json) | 5394e3cd1a07244f5b117c9c3e610ead7855529f7b9107584130cc5bad5692f2 |
| C | [CSV](../../logs/s54_abc_short_20261105_184038_C/train_log.csv) / [audit](../../logs/s54_abc_short_20261105_184038_C/training_audit.json) / [Newton summary](../../logs/s54_abc_short_20261105_184038_C/evaluation/summary.json) / [Eval readback](../../logs/s54_abc_short_20261105_184038_C/evaluation/cloud_eval_readback.json) | 701da60cdd9338978b78812a64198424db857b067711ae4a911252570cc681eb |

补充证据：[A-解析后参数](../../logs/s54_abc_short_20261105_184038_A/args.json)；[B-解析后参数](../../logs/s54_abc_short_20261105_184038_B/args.json)；[C-解析后参数](../../logs/s54_abc_short_20261105_184038_C/args.json)。[A-启动命令与profile](../../logs/s54_abc_short_20261105_184038_A/launch.json)；[B-启动命令与profile](../../logs/s54_abc_short_20261105_184038_B/launch.json)；[C-启动命令与profile](../../logs/s54_abc_short_20261105_184038_C/launch.json)。[A-运行配置](../../logs/s54_abc_short_20261105_184038_A/train_profile.json)；[B-运行配置](../../logs/s54_abc_short_20261105_184038_B/train_profile.json)；[C-运行配置](../../logs/s54_abc_short_20261105_184038_C/train_profile.json)。[A-实测资源](../../logs/s54_abc_short_20261105_184038_A/runtime_stats.json)；[B-实测资源](../../logs/s54_abc_short_20261105_184038_B/runtime_stats.json)；[C-实测资源](../../logs/s54_abc_short_20261105_184038_C/runtime_stats.json)。[A-梯度记录](../../logs/s54_abc_short_20261105_184038_A/gradient_terms.jsonl)；[B-梯度记录](../../logs/s54_abc_short_20261105_184038_B/gradient_terms.jsonl)；[C-梯度记录](../../logs/s54_abc_short_20261105_184038_C/gradient_terms.jsonl)。[A-Eval上传包](../../logs/s54_abc_short_20261105_184038_A/evaluation/eval_bundle.json)；[B-Eval上传包](../../logs/s54_abc_short_20261105_184038_B/evaluation/eval_bundle.json)；[C-Eval上传包](../../logs/s54_abc_short_20261105_184038_C/evaluation/eval_bundle.json)。

现有任务/物理/采样/Reward口径以运行快照和Checkpoint为准，而不是当前配置文件名。审计复核484个源码/配置快照文件的实存哈希与manifest，无不一致；42个原始实验文件的SHA256、大小和修改时间在审计前后保持一致。通过CPU加载Checkpoint，逐tensor比较Encoder/Decoder与Teacher/WM-only状态；检查Adam step、原始CSV/JSONL；直接读取27个已有NPZ验证Reward和命令一致性；离线读SwanLab二进制校验5100个训练点与183个Eval点。临时脚本与中间JSON只写入`/tmp`，最终交付仅本文。

