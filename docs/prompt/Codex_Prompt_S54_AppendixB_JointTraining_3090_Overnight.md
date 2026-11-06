# Codex 执行 Prompt：S54 Appendix B 风格 Motion–Foot 联合训练（RTX 3090，一晚，独立实验）

> 可直接整段复制给具有仓库和 RTX 3090 服务器权限的 Codex。**这是开发和运行指令，不是已完成的运行记录。**

## Role & goal

你在 `Foothold_with_SuperMimic` 仓库工作。请先阅读同目录的 `S54_AppendixB_JointMotionFoot_ExperimentPlan.md`，再读取当前本地工作树/AGENTS.md（若存在）、`MotionPrior_FootTrajectory.md`、`ABC_SHORT_ABLATION_REPORT.md`、`S54_STAGE1_CHECKPOINT_EVAL_REPORT.md`，核对原论文 [SuperMimic Appendix B + Appendix G.1 + Algorithm A.1](https://arxiv.org/pdf/2610.28480)。

**目标**：实现**真正同一训练进程内**的双接口联合训练，并在服务器实际运行一次总耗时受限于 **8 小时**的 RTX 3090 训练。

- `Motion`：LAFAN1 S54 Motion Encoder `E_m` + 共享 Decoder `D` 继续用 Motion Tracking Reward/BPTT 训练。
- `Foot`：230D Quintic Foot Goal → 新 Foot Encoder `E_f`，用 **Foot Tracking Reward/BPTT 只更新 `E_f`**；下游梯度可以穿过 D，但**不能修改 D 参数**。
- 两个任务分别拥有 Critic、Replay、Goal/Observation Norm、Reward/Termination、Episode Clock，**共享唯一的 Decoder 和唯一的 World Model**。
- World Model 用真实 Motion + Foot Newton Transitions 混合更新；在任一 Actor BPTT 中冻结 WM 参数而非状态梯度。
- 不使用 Teacher Action MSE、随机匹配上身 Prior 或默认配对 latent loss；`paired_latent_weight=0`。
- 初始 Teacher：`logs/s54_stage1_walk_20261105_014722/ckpt_004772.pt`，检查 SHA256 `1cf556f384cefa8635975ca4f8cb8c0e144a98292c5fb2fb8cff33e9b07c3fc4`。找不到/哈希不符则给出准确阻塞原因，**禁止悄悄换教师**。

**工作目标不是复训或覆盖旧 A/B/C**。该实验代号 `S54_JOINT_AB1`，它是论文 Velocity Interface 联合训练机制的 **Foot Goal 扩展**。旧 A/B/C 所有目录、checkpoint、训练数据与 SwanLab run 必须保持不变。

## A. 必须先完成的工程实现

1. **真实双接口运行器**：新增可审计入口（建议 `supermimic/joint_train.py` 和 `tools/train/run_s54_joint_ab1.py`），保持原 `supermimic/train.py` 默认行为兼容。不要仅启动两个单任务进程后复制权重。
2. **真正共享的网络对象**：`E_m(730→32)`、`E_f(230→32)`、`D((10×99+32)→29)`、`WM(133→70)`；只存在一套 `D` 真实参数；两个 Critic 独立（Motion ≈914D，Foot ≈365D）。`E_m,D,WM` 从 Stage1 严格读取兼容权重/norm，`E_f,V_f` 初始化为新参数；Motion Critic 是否兼容地载入须记录。
3. **严格的更新权限**：Motion BPTT 更新 `E_m,D`；Foot BPTT 更新 `E_f` 而不更新 `D`；两个 Critic 更新各自参数，WM 只在其监督阶段更新。使用不同优化器/参数组，确保 Foot 优化器不会通过 Adam momentum/weight decay 修改 D。Foot 前向不能使用会切断 `∂L_foot/∂z_f` 的 `torch.no_grad()`。每隔若干更新保存 `ΔD_motion`, `ΔD_foot`, `grad_z_foot`, `ΔE_f`, `ΔWM_actor` 审计证据。
4. **接口数据严格隔离**：Motion 使用 S54 LAFAN1 manifest + reference frame/clip-end；Foot 使用原 `AlternatingConfig` 的短步直行 Quintic 轨迹与 Foot Goal/Reward；使用每接口独立 Replay 和命令/Goal schema；保存 `start_frame/command_id/ep_step/ξ/action_executed`，避免跨 reset 错配。Foot 演算不能依赖全身参考。
5. **同一 WM 混合训练**：每轮总计预设数量 WM updates，分别从两个 Replay 按实际记录的比例抽样窗口；不允许把 Motion 的 730D Goal 或 Foot 的 230D Goal 输入 WM；保持 state 71、ξ 33、a 29、WM 133→70 不变。按接口记录 WM 误差及 1/4/8/16/32 步配对 Newton 物理误差。
6. **观测语义/Normalizer**：处理 Motion 与 Foot 最后 6D Root/Heading 的不同语义。当前 Foot 使用 mask `obs[...,93:99]`，这是有损工程适配。优先保持原合同并显式审计，必要时实现测试过的可见 adapter；绝不能“形状相同就静默复制 Norm”。保留 Stage1 旧 Motion 推理行为的回归验证。
7. **训练安全**：本地优先 SwanLab，异步/独立重传；源代码快照、配置与 hash、所有真实 Newton/想象样本预算、checkpoint+optimizer+Norm+RNG+任务/接口合同严格持久化；支持同任务 `resume`，独立 run_id，tmux SSH 断开不影响训练。

## B. 先写并实际运行 Gate 测试（不可只给计划）

**CPU 单元测试**：双接口共用同一个 D；Foot Loss 更新 `E_f`、`grad_z_f` 有限非零、`D` 不变、`E_m` 不变；Motion Loss 更新 `E_m` 和 D、`E_f` 不变；WM supervision 更新 WM，但 Actor update 不改变 WM；两 Critic 互不影响；配对 Goal 时钟/Reset/Replay 不泄漏；Teacher/Goal/Physics/ObsNorm 合同兼容及错误拒绝；同任务 Resume 恢复成功。

**Newton Smoke**：先 1/16 env 多轮、再 256+256 env 有界运行；验证 Motion/Foot 都真实产生 Transition，日志不混、Contact/Reset 时钟正确，显存/算力无异常，Checkpoint 可读取并恢复。只有 Smoke 通过才允许长训。如果 512+512 env 经实测稳定可用，固定选择该规模；否则 256+256 并固定保存 profile。禁止在正式 run 中随意变更配置而不新建实验 ID。

**优先解决致命工程错误**，不要为通过 Gate 而扩大终止阈值、篡改 Reward、把 Foot Task 改成 Motion Task，或者偷偷加入 Teacher action MSE。

## C. 一晚训练参数（提案，实测后固定）

| 参数 | 初始设定 |
|---|---|
| GPU | 单 RTX 3090，保存 nvidia-smi 与物理引擎版本 |
| 总训练 Wall cap | 12 小时；到达 cap 或设定最大迭代**安全保存退出**，严禁无界后台运行 |
| 最高 Iterations | 50000（仅上限；以实际吞吐为准） |
| Newton env | Motion 512 + Foot 512；失败回退各256，固定后再主训 |
| WM batches | 每 iteration 总16；每 batch 共1024 windows，按 50/50 取 Motion/Foot |
| WM window / imagination horizon | 8 / 32 |
| imagined start batch | Motion 128 + Foot 128（各一轮 Actor/Critic 更新） |
| UTD / buffer / warmup | 0.4 / 1024 steps每env每接口 / 128 vector steps |
| WM-only | 200 iterations（预训练 WM 适应 Foot 状态分布） |
| LR | WM 5e-5；Motion Encoder 2e-5；Shared Decoder 2e-5（仅 Motion）；Foot Encoder 1e-4；双 Critic 1e-4 |
| gamma / lam | 0.99 / 0.95 |
| Goal | Motion 730D 0.02s间隔；Foot 230D 0.1s间隔；hist 10 |
| Motion:Foot real data / imagined data | 各50%（按真实数据量核验） |
| `prior_config` / `paired_latent_weight` | none / 0 |
| train seed | 54 |
| Checkpoint | 每500 iter或约30min + 最终/信号保存；保留有证据的best及last |
| 实际 Newton Eval | Foot固定 seed 54/55/56、2/5/10s；Motion固定短参考探针，完整261.3s全片只稀疏运行 |

请先对本机 RTX3090 显存、吞吐和源码特定限制做 dry-run/smoke，再冻结最终 profile。**不得把这些建议值当成已经测得的稳定配置**。若 8h 内迭代不足，保存现有合法结果，不编造剩余轮次或声称收敛。

## D. 正式运行流程与自动化

1. 生成唯一 run_id（例如 `s54_joint_ab1_20261106_<time>_seed54`），新目录 `logs/<run_id>`、`results/s54/<run_id>`；在 `configs/motion_prior/joint_ab1/` 新建训练配置。保证不覆盖旧文件。
2. 实际执行 dry-run → tests → Newton smoke → `tmux` 主训练；在终端输出准确的启动命令、tmux session、logs 路径与 checkpoint 路径。SSH 断线后训练必须继续。
3. 训练前、中、后固定评估，独立记录 Motion 与 Foot 的 Reward、Survival、RMSE、Root XY 漂移、实际前进速度、Upper 动作、真实接触/滑移、WM配对误差；保持 `Reward/Simulator` 和 `Reward/Imagined` 各任务分开。训练中**不许以 Foot 短期高 Reward 代替10s真实存活**。
4. `Motion` 验收参考：初始 Stage1 **完整261.3秒评估** Walking NR=0.820197；同协议全片评估若长期低于约0.77（预设绝对下降0.05）要标记/保存 last-good。若采用固定10s短探针，必须先测 Teacher 在**相同短探针协议**下的基线，绝不可把0.820197的全片数值与短探针直接比较。`Foot` 的首要改善目标为2/5/10s存活及正向运动，而非仅足端误差降低。真实完整成功需 `10s Survival ≥2/3`，且向前速度不再显著为负（工程预设门槛）。
5. 保存 `last.pt`、所有优化器状态/归一化/Run RNG/接口计数；收到 SIGINT/SIGTERM / wall cap 可安全停止并保存。网络上传失败不应阻塞主循环；SwanLab本地落盘、同步异步完成。
6. 有非有限梯度、结构性 OOM、无法保存 Checkpoint、Motion Teacher 异常遗忘时，留存故障证据并安全停止。**不要为了跑满8小时自动调整 Reward、终止条件或变成另一个实验方法。**

## E. 必须提交的 Markdown 报告

请生成 `docs/experiment/S54_JOINT_AB1_RUN_REPORT.md`，清楚记录：

- 源 Teacher 和 checkpoint SHA、提交版本、所有源码快照/训练参数/物理/Goal contract；
- `train_status`、总wall time、RTX3090实测显存/吞吐、实际 Motion/Foot env 和想象batch；
- 实际总迭代、每接口 Newton transitions、WM更新次数、Motion/Foot Actor次数、Critic次数、想象步数，**含 Stage1 预训练成本单列**；
- 梯度隔离验证（Motion改变D、Foot不改变D但能更新 E_f）、独立 Critic/Replay/Norm 的测试与运行证据；
- 训练全过程分任务曲线、Newton 2/5/10s、Motion tracking/Naturalness、Foot真实速度/接触/滑移、WM 1/4/8/16/32配对误差；
- 新 run 下的Checkpoint/CSV/JSON/NPZ/视频/SwanLab本地与云端同步状态；
- `COMPLETED_BY_WALL_CAP` / `COMPLETED_BY_ITER_CAP` / `STOPPED_BY_GATE` / `CRASHED` 等准确状态；缺失指标标 `NOT EVALUATED`；
- 与原 A/B/C 区分的研究结论、可能的失效原因和下一步实验建议，注明无法直接以不同 Motion 监督预算作公平因果比较。

**不允许**修改历史 A/B/C、Stage1日志/Checkpoint、用户原始S54数据；不允许把只通过Smoke当作收敛。工程改动保持原CLI行为和现有严格合同检查。若未能实际开始主训，交付已完成的测试与明确阻塞原因，不声称“已开始通宵训练”。

## F. 最终回复格式

简要汇报：`实施完成状态 / 测试通过和失败 / 是否实际启动训练 / tmux session / 已固定配置 / 实际耗时与迭代 / 最新Checkpoint / 报告路径 / SwanLab本地保存及上传状态`。只能报告真实已执行事实。
