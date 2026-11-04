# Codex 工程任务：S54 SuperMimic 两阶段先验迁移与足端轨迹训练准备

> 以下内容是**给 Codex 的完整执行 Prompt**。请先阅读仓库及本文件配套的实验设计文档 `S54_SuperMimic_MotionPrior_FootTrajectory_Experiment.md`。  
> **目标：完成可审查、可测试、可重复的代码修改与实验准备。不要未经验证直接启动长时间训练。**  
> **v2 更正**：SuperMimic Appendix B 的 Distillation 使用配对参考的 **Motion Tracking Reward + WM/BPTT**，**不是** Teacher–Student Action MSE。Stage 1.5 为可选实验，不得成为 Stage 2 必要前置条件。详细对照见 <https://arxiv.org/abs/2610.28480>（Appendix B、Table A.4）。

---

你现在是该项目的 Robotics RL / Model-Based RL 工程师。请在当前仓库 `Foothold_with_SuperMimic` 内实现以下方案，机器人是**KUAVO S54 29-DoF**，不是 G1。当前栈：Isaac Lab 3.0、Newton/MJWarp、PyTorch SuperMimic（learned differentiable World Model + BPTT Actor + Critic）。优先使用现有模块，**不要推倒重写 SuperMimic，不要改变 WM/BPTT 的核心算法**。

## A. 必须先做代码审计与计划

1. 执行 `pwd`、`git status --short`、`git rev-parse HEAD`，只读取当前工作树；不擅自清理、覆盖、reset、rebase、切分支、删除已有 logs/ckpt，不改动不属于本任务的代码。报告工作区是否有未提交修改，并避让。
2. 阅读以下文件并核对**真实实现**，不要根据这个 Prompt 的维度和字段盲改：
   - `supermimic/s54.py`, `supermimic/networks.py`, `supermimic/mdp.py`, `supermimic/motion.py`
   - `supermimic/goals.py`, `supermimic/foot_trajectory.py`, `supermimic/foot_mdp.py`
   - `supermimic/buffer.py`, `supermimic/train.py`, `supermimic/env.py`, `supermimic/evaluate.py`, `supermimic/foot_evaluate.py`
   - `supermimic/reference_selection.py`, `run_train.sh`, `README_S54.md`
   - `STANDING_DIAGNOSIS.md`, `docs/S54_FOOT_TRACKING_TRAINING.md`, `docs/S54_FOOT_TRAJECTORY.md`
   - `configs/s54_standing_3090.json`, `configs/s54_foot_trajectory.json`。
3. 检查 S54 数据、Motion CSV + JSON sidecar、`--clips` 选择机制、30 FPS→50 Hz 重采样、Goal Contract 与 Task Contract。**不要把原 G1 的 LAFAN1 PKL 直接用于 S54**。如缺数据，输出清晰的缺失信息与生成方法，不虚构已存在 clips。
4. 先输出简要**实施计划**（模块 / 文件级别、依赖关系、测试、风险）。然后直接开始执行，不要只给建议，不要等待重复确认；遇到外部权限、GPU/数据缺失则完成仍可进行的静态与 CPU 工作，并在最终报告明确 blocker。

## B. 总体任务与硬性模型合同

实现 **Stage 1 `motion` Full-Body Tracking → Stage 2 `foot_trajectory` Quintic Tracking** 的直接迁移主路径；**Stage 1.5 LAFAN1 Paired Foot Transfer 作为可选、可跳过的过渡消融**，不要要求 Stage 2 只能从 Stage 1.5 启动。外加公平消融和可复现训练入口。

现有接口预期（必须从代码再次核对）：

```text
Robot/action: S54 29 DOF, action29 = implicit PD position target offset
Physical state: (root p3, root quaternion4, v3, w3, q29, qd29) = 71D
Obs: 99D/frame
Stage 1 Motion Goal: 73D/frame × 10 = 730D; dt=0.02s
Stage 2 Foot Goal: 23D/frame × 10 = 230D; dt=0.1s
World Model: invariant state71 + action29 + xi33 = 133D → state increment70D
Critic: Motion 914D → scalar; Foot 365D → scalar
```

关键不变项：

- 两阶段应使用相同 Robot / actuator / `control_dt` / joint ordering / action scaling / physical state definition。
- **不要把 Motion Goal / Foot Goal 塞入 WM**。真实 Newton transition 的 `(S,a,xi,S1)` 是 WM 监督样本；Foot 控制会产生新数据分布，因此 Stage 2 要持续更新 WM。
- Stage 1 和 Stage 2 统一使用相同 `hist`，默认建议 10（若资源必须使用 2，则两个阶段都统一使用 2 并在契约里写明）；**不同 hist 必须阻止无依据的 Decoder 迁移**。
- 现有 Obs 的最后 6D 相对朝向语义不同（Motion relative root vs Foot fixed command heading）；迁移时必须显式适配/验证，不能只看维度一样就复制 normalization。
- 保持已有 `--resume` 的**同任务恢复**语义，新增独立的跨任务 `--init_from_motion`（名称可等价），不得放宽现有 Goal/Task Contract 校验。
- **明确网络边界**：`Goal Encoder = 完整 MLP (goal→z32)`，`Action Decoder = 完整 MLP ([z32,obs_hist]→action29)`，World Model 为独立的动力学 MLP (state,action,ξ→next state)。Encoder/Decoder 不是一个 MLP 的首层/末层。当前 `Policy` 只有单 Encoder+Decoder；双 Encoder + Shared Decoder 是待新增接口。
- **原论文 vs 本任务改动**：SuperMimic §3 的 Motion Mimic 用原有 Encoder+Decoder 直接通过 WM/BPTT 学全身跟踪；Appendix B 允许多个目标接口共用 Decoder/WM，Distillation 用**参考运动跟踪奖励**，RL 用**各自任务奖励**。将其扩展为 Foot Goal 属于本实验；直接 Teacher Action MSE 是**可选独立消融**而非论文原方法。

## C. Task 1 —— Stage 0 数据选择、检查与可视化

新增/复用**非侵入**的数据准备工具：

1. 从现有 S54 Retargeted LAFAN1 CSV 中按名字/glob 筛选 walk/slow walk/turn/start-stop/stand；若数据类别不全，以实际筛选结果和质量指标为准，不要编造 clips。
2. 保留 train/held-out clip list/manifest，记录原始路径、sha256、帧率、时长、关节顺序、四元数约定、通过/淘汰原因、seed；禁止同一原始 clip 的窗口在 train/eval 之间泄漏。
3. 用 S54 FK 导出每帧左右脚位置、朝向、速度、Root 状态、upper-body 关节、contact phase 估计。接触采用高度+速度+持续时间+滞回规则，阈值可配置，并以曲线/可视化检查；计划 contact label 不等于 Newton 实际接触力。
4. 尽量复用已有 metadata 与数据 Loader；禁止覆盖源文件。提供 `--dry-run` 和 CPU 单测，遇到缺文件要明确失败。
5. 产出 `results/s54/...` 下的 clip 数据质量清单与代表性曲线/预览。不要依赖 GPU 才能跑基础数据检查。

验收：至少一个可用 S54 clip 完成加载/FK/采样维度/时间连续性检查；若没有可用数据，报告具体缺口，不伪造成功。

## D. Task 2 —— Stage 1 Motion Full-Body 训练准备

保留 `goal_mode=motion` 的既有 Reward、环境、网络、BPTT/WM/Critic 流程，创建独立 config/launcher；只在确有 bug 时修复原逻辑并补回归测试。Motion Reward 由 `supermimic/mdp.py` 的 14 tracked bodies 的位置/朝向/速度跟踪、Root 位置/朝向、Head、Action Rate、Joint Limit 构成。**不误称原实现包含所有关节的显式 Joint Tracking 奖励**。

要求：

- CLI 支持固定 clip manifest / 筛选、独立 seed、checkpoint 频率与 bounded smoke；run id 置于不冲突的 `logs/s54_*` 新目录。
- 保存 Stage 1 checkpoint 所有必须字段、归一化器、机器人 / Goal / 数据 manifest 合同；记录自然度、稳定性、WM 预测、独立真实 Newton eval；不要以训练 reward 代替真实 eval。
- 保证 Stage 1 结果可以作为 **Teacher Policy**，但在文档中称为 reference-conditioned teacher，而不是无需 reference 的通用“动作生成先验”。
- 不启动正式长训；只提供经过测试的 launch 命令，以及资源允许时的**受限 smoke**。

## E. Task 3 —— Stage 1.5 配对数据与 Foot Encoder Adaptation（可选，不阻塞 Stage 2）

**先实现 Stage 1 → Stage 2 直接迁移入口**（Stage1 WM + 兼容 Decoder + 新 Foot Encoder，直接 Foot Reward + WM/BPTT），确保在没有 Stage 1.5 配对数据时也能运行 bounded smoke。之后再增量实现可选 Bridge。

### E.1 可重用配对参考数据

1. 新增独立 `reference_foot`（或 `paired_foot`）数据源，从**同一段 S54 LAFAN1 全身参考**按 FK 抽取双脚位置/朝向/速度/计划接触/相位；保持 Foot Goal 的 23×K 字段合同及当前 Root Yaw 坐标系。不可把原始参考盲目灌进 `sample_alternating`，后者是假设 Quintic 的随机命令生成器。
2. 配对元数据须记录 `clip_id, frame_id, control time, paired motion goal, foot goal, phase`，保障 Newton Collector / Replay / Imagined Rollout 的时序和 reset/clip-end 正确；校验 Foot 的未来 0.1s 与 Motion 未来 0.02s 采样从**同一当前时刻**出发。
3. 同一物理状态下的 Motion Observer 与 Foot Observer 的最后 6D 相对朝向语义不同，应分别由正确参考方向构建；新 Foot Goal Norm 独立统计，不能根据维度相同盲目迁移 Obs Norm。

### E.2 先实现原论文 Appendix B 启发的 BPTT Encoder Adaptation

- 从 Stage 1 checkpoint 载入 `E_m`、`D`、`WM`；新增 `E_f: foot_goal230 → z32`。**必经路径**：`g_foot → E_f(MLP) → z32 → D(MLP) 与 obs_hist 合并 → action29 → WM → reward / Critic`，而不是 Foot Encoder 直接输出 Action。
- **配对 Motion-Reward Bridge**：从 paired Foot Goal 控制机器人的情况下，仍利用对应的完整 S54 LAFAN1 Motion Reference 计算原 Motion Tracking Reward；从相同的真实 Newton/Replay 状态出发，通过已学 WM 的 imagined rollout 和 BPTT 优化 `E_f`。完整 Motion Reference **不进入 Student Actor 的 Goal**，只作为训练奖励的监督信号。
- 训练新 Foot Encoder 时优先**冻结 Decoder 参数**，让回报梯度经 `WM → Decoder → Foot Encoder` 反传，且 Decoder 和 WM 的 Actor 更新参数不变；**禁止在 Decoder 或 WM 前向外层使用 `torch.no_grad()` / `detach()` 导致 E_f 无梯度**。WM 在每轮用真实 `(S,a,ξ,S1)` Replay 继续更新，Actor 更新时冻结参数。
- 论文是联合多接口训练，Shared Decoder 仍由 Motion Objective 更新；若本项目采用「先 Stage 1，后冻结 Decoder」的顺序简化，必须标为工程改编，不能宣称精确复现 Appendix B。可以为后续 joint training 增加独立开关，当前不强求大规模多任务训练器。
- 新增 Foot Critic，严格说明采用哪一种 reward / termination / reference 时间语义。配对 Bridge 的 Motion Tracking Reward 必须在真实和 imagined 侧计算一致。若当前 Foot MDP 不支持此语义，新增显式 paired MDP/Reward Adapter，不能偷换成原 Foot Reward 后仍称 Motion-Reward Distillation。

### E.3 直接 Foot-RL（更简洁、必须支持）

- 不做 Stage 1.5 也应支持：`g_foot → E_f → D → WM → r_foot + V_foot`，BPTT 只更新 `E_f`（或按显式开关渐进微调 D）。`E_f` 的初始权重为新建，不需要 Teacher Action。
- 对同段 paired 数据可**可选**加 `L_latent=||E_f(g_foot)-stopgrad(E_m(g_motion))||²`，但只作用于有对应运动参考的 sample；不将任意 Quintic Foot Goal 与不匹配 LAFAN1 强制对齐。

### E.4 Action MSE：仅作为可选独立对照

- 可实现 `L_action=||D(o_foot,E_f(g_foot))-stopgrad(D(o_motion,E_m(g_motion)))||²` 用于单步 warm start，明确其**不是原论文 Appendix B 的 Distillation**；它可以在没有 WM 的单步监督路径上训练，但**不能**替代闭环 Newton 验证，不得在不匹配参考的随机 Quintic Goal 上强制使用。
- 默认不启用该损失；仅当开启显式 CLI/config 开关时运行，单独保留指标、运行 ID 与消融结果。不可让它成为 Stage 1.5 的验收唯一标准。

### E.5 工程验收

- 完整比对 Stage 1 和新 Foot Encoder 的 I/O、参数迁移映射、观测语义、配对 Goal 合同；`--init_from_motion` **不依赖配对数据**。
- 在 CPU 测试中，检查训练 `E_f` 的梯度有限且非零、冻结 D/WM 参数的 `.grad` 不更新，同时反传路径不断；检查 `E_f` 与 `D` 的输出维度与单步 Teacher Action MSE（可选）。
- 可选 paired Bridge 在真实 Newton 中测试站立、短行走、真实接触和上身自然性；报告损失下降是否带来**闭环**改善。不可以仅凭 action/latent loss 下降判定成功。

## F. Task 4 —— Stage 2 Foot Reward + 动态 Motion Prior

保持现有 `FootTrackingMDP.reward` 的基本项和原始系数（可设置通过配置调整、默认不改）：Swing position/velocity、Stance stability、Touchdown、Foot Orientation、Base Upright/Height/XY Balance、Action Rate、Joint Limit、Posture-to-default、Joint Velocity。

**新增加可开关、独立日志的三类 Motion Prior**：

- Upper-Body Pose Prior（优先）：\(r_u=\exp[-\|q_{upper}-q_{upper}^{prior}(v,\phi)\|_W^2/\sigma_u^2]\)。以 S54 明确关节名选头、腰、上肢；Walking 使用与速度/相位匹配的 LAFAN1 模板，Standing 使用站立模板（若缺站立资料，不得假造）。
- Torso Orientation Prior：基于 SO(3) 相对目标旋转误差，允许合适的转弯/加速姿态，避免与 Base Upright 双重过强约束。
- Upper-Body Velocity Prior / Smoothness：基于相位匹配速度或轻量正则，避免和现有 `joint_velocity` 叠加导致僵硬。

**非常重要的目标冲突规则**：

- 在 `reference_foot` paired 样本上，可以使用同一段 LAFAN1 的全身/上身模仿辅助奖励；
- 在**随机 quintic** 样本上，禁止无依据地跟踪任意 LAFAN1 下肢 / Root reference。只有速度/相位明确匹配的轻量上身先验才可用；匹配失败时应退化为弱的静态姿态约束或关闭 prior，不阻断足端任务。
- 默认先保留 `q_default` 正则作为 baseline，与动态 prior 做独立可配置消融，**不应盲目叠加强约束**。

梯度与时序检查：

- 上述 prior 应在 `S` / FK / SO(3) 中可微，且通过 frozen WM 对 Actor 有非零有限梯度；提供 finite-difference / autograd 数值检查。
- 真实采样和 imagined rollout 使用同一 reward 函数、同一时间语义（动作后按 `ep_step+1` 计算），相同 command 快照；不可在 replay 时重新随机 `command`。
- 原 `check_foot_gradients` / `goal_sync` / `reward_sync` 必须继续通过。
- 对真实 Newton 接触力、足滑移等作为独立诊断；不要把离散接触布尔量直接塞进可微 Reward 造成不可导或错误梯度。
- 记录 reward term mean/std、权重后贡献、Actor per-term gradient norm，以观察 prior 对目标的干扰。

请实现 config-only 的 prior 权重开关、standing/walking 模式切换和**受真实 eval 指标门控**的课程：**直接 Foot-RL 的保守训练入口必须存在**；可选 paired-only Bridge → 保守 quintic 混合 → 更广步长/步频/转向采样。不要立即用大步长和高步频冲击尚未稳定的 Standing Policy。

## G. Task 5 —— World Model 迁移与 Critic

- WM 保持相同 `133→70`，输出局部坐标状态增量，更新 SO(3) 使用 Log/Exp；Stage 2 从 Stage 1 checkpoint **显式加载** WM 权重及 in/out normalization，重新初始化优化器（或说明重用的合法依据）。
- WM 用新的真实 Foot rollout 继续训练；可选混合旧 Stage 1 motion transitions 以缓解遗忘。任何 replay 数据混合都必须符合 S54 robot/actuator/physics contract 且窗口不跨 reset。
- 在 WM update 中继续使用真实 `(S,a,ξ,S1)`；Foot Goal 仅影响 Actor / Reward / Critic，不需要扩展 WM 网络输入。
- Critic 因 Goal/Reward/Feature 不同，Stage 2 重新初始化（当前 Motion 914D，Foot 365D）；不要尝试直接从 Stage 1 恢复旧 Critic / optimizer。
- 增加受限 calibration 与 held-out WM 诊断：同 Newton 起点、相同动作序列、相同 ξ 和 command，对比 1 / 4 / 8 步状态 / Root Pitch / Height / Feet FK，以及同窗口真实 vs imagined reward。报告状态预测误差之外的 Actor 梯度合理性局限。
- Stage 2 early actor 训练必须防止世界模型预测不准被 BPTT 利用；如果姿态/高度误差明显恶化，阻止进一步加大课程难度。

## H. Task 6 —— 跨任务 Checkpoint / CLI 与保护

新增例如：

```bash
# 独立示例；以 Codex 最终实现的真实参数为准
python supermimic/train.py --goal_mode foot_trajectory \
  --init_from_motion logs/s54_motion_<run>/last.pt \
  --log_dir logs/s54_foot_transfer_<run> \
  ...
```

加载合同与迁移清单：

- 机器人 id、S54 29-DoF/关节排列、PD/action scale、物理状态/`dt`、模型版本、obs_dim、hist、goal schema 均要求明确校验。
- `WM + compatible Decoder` 可以复制；新 Foot Encoder / Goal RunningNorm、Foot Critic / optimizer 新建；ObsNorm 是否复制需要语义检测，默认保守处理。
- 输出 JSON/text mapping report：loaded/skipped/new/reason/shape；**形状不匹配、错机器人、错 task contract 必须 fail fast**。
- 保留现有 `--resume` 及 `goal_contract.assert_checkpoint` 功能；不可用 `strict=False` 静默漏载关键模块。
- 不能复用 G1 checkpoint，不能覆盖旧 checkpoint / logs / 数据或破坏已有实验。

## I. Task 7 —— 消融、指标、部署与测试

按以下实验矩阵准备**可复现实验入口**，先 dry-run，不需实际把所有组正式跑完：

| Group | Actor init | WM init | Prior reward | Paired bridge |
|---|---|---|---|---|
| A | random | random | no | no |
| B | random | Stage1 | no | no |
| C | compatible Stage1 Decoder + new Foot Encoder | Stage1 | no | no: direct Foot-RL |
| D | as C, with paired Motion-Reward BPTT Encoder Adaptation | Stage1 | no | yes: paper-inspired |
| E (optional) | as C, with paired Action MSE warm start | Stage1 | no | yes: Action-MSE ablation |
| F | as D | Stage1 | yes | same D |
| G (optional) | as C | Stage1 | yes | no |

A↔B isolates WM transfer; B↔C isolates decoder initialization; C↔D isolates paired BPTT Bridge; D↔E compares Bridge objectives; D↔F isolates prior reward. All Stage2 groups need the **same Stage2 goal distribution, evaluation script, and independent seeds**. Report **both** Stage2-only real Newton steps and total real Newton steps including Stage1.5; report wall time and supervision steps separately to avoid unfair efficiency comparisons.

**评估与记录**：

- Standing：2/5/10 秒、多个**独立 reset 种子**，成功率、后倾 / 高度失败、Root pitch/roll、base height、足部实际受力与滑移。
- Walking：脚位置/朝向/速度 RMSE、触地点误差、按任务相位的接触准确度、前进速度、步频/步长命令适应。
- Turning：原地转向 / 行进转向，命令跟踪误差与失败率。
- Naturalness：上肢姿态（与合适的速度/相位参考对比）、关节速度/加速度、躯干转向、录制全身视频（同镜头对比），不能只看 Foot Reward。
- Learning/WM：wall time、实际 env steps、真实/想象 reward 差、1/4/8 步 WM 误差、Actor/WM/Critic loss、梯度范数，保存 CSV/JSON；SwanLab 可用时同步关键指标和 checkpoint 索引，失败时本地保留后可重传。
- 记录 `git_sha`、配置快照、数据 hash、版本、seed、训练命令、ckpt schema。Stage 2 部署模型只能接受 obs_hist + foot_goal，**不得隐性依赖全身 Motion Reference 或 Teacher**。

**测试必须覆盖**：

1. CPU：S54 FK / 重定向数据合同、配对 Foot Goal 时序、LAFAN1 holdout、standing/walking 先验缺失回退、Reward 与数值梯度。
2. CPU：Motion 730D vs Foot 230D、相同 hist、S54 29D Action；WM 133D input、70D output；Critic 输入合同；序列化/归一化器迁移。
3. CPU：同任务 `--resume` 成功、跨任务错误 resume 拒绝、`--init_from_motion` 成功与错型号/错 shape/错 hist 拒绝。
4. CPU / small GPU：Goal/Reward Newton-collect vs Replay vs Imagined 同步（`ep_step` off-by-one、touchdown 边界、reset、clip end）。
5. Newton 物理仿真：1 env / 16 env、短 bounded smoke；Stage1 和直接 Stage2 的 smoke 必须具备，**Stage1.5 仅在已有配对数据/已训练的 Stage1 checkpoint 且资源允许时执行**；站立真实接触和倾倒原因诊断；不允许把 smoke 通过等同于策略已收敛。
6. 回归：原 `motion` 训练、原 `foot_trajectory` 无 prior 训练/评估入口不应被破坏；现有合同、reward、动作尺度默认值保持不变。

## J. 执行顺序、硬件与安全门槛

按顺序：**静态审计 → Stage0 数据检查 → Stage1 配置 → 直接 Stage1→Stage2 参数迁移 / Foot-RL → CPU 测试 → 可选 Stage1.5 Paired BPTT Bridge → reward prior → Newton bounded smoke → launch 配置 / 消融脚本 → 报告**。

- 先检查 `nvidia-smi`, CUDA、Docker/uv、Newton 和可用显存，不猜测 3090 已完成某实验；`num_envs` / `pi_batch` / `wm_batch` / `horizon` 以**实际显存与吞吐**小规模校准。
- 确认现有站立失败：参考 `STANDING_DIAGNOSIS.md` 的约 1.92s 后倾和 1/4/8 步 WM 偏差；不要声称目前已经具备稳定站立，更不能直接开 8 小时正式训练。
- 默认只运行有限 CPU 测试和 GPU smoke（通过显式上限 / timeout / 小并行），不得启动无上限的训练、上传或删除。
- 生成 tmux / 可中断恢复运行脚本；如果要结合 SwanLab，采用异步/缓冲日志的现有能力或安全容错方式，并保留本地记录，不需要连接个人账号或询问 token。
- 每个实验 run 目录唯一、`logs/s54_*`、不覆盖；若没有 GPU/数据，仍产出可用代码和无法执行的明确检查结果。
- 只有**真实 Newton eval** 证实站立与姿态门槛通过后，才在文档中列出正式训练步骤；正式长训由我单独决定何时启动。

## K. Codex 必须交付的内容与最终回复格式

请交付：

1. **实际代码修改**（路径/模块、改动原因、兼容性）；必要时将这份 Prompt 与实验记录归档到 `docs/`，不要无意覆盖同名文件。
2. **配置与入口**：Stage1、**可直接独立启动的 Stage2**、可选 Stage1.5，以及 A/B/C/D/F（可选 E/G）消融的启动 / dry-run / smoke / eval 命令，明确工作目录、数据需求、logs 路径、checkpoint 和归一化迁移规则。
3. **自动化测试**：新增测试路径，实际执行结果、通过/失败、失败原因，区分 CPU、Newton smoke、未运行的长训。
4. **实验材料**：选定 clip manifest、S54 retarget 检查报告、配对 contact 数据统计、WM 1/4/8 步对照脚本、Reward 分项记录、可视化 / 结果目录。
5. **迁移报告**：哪个权重从 Stage1 复制、哪个重置、哪些不同维/obs语义有处理、Teacher 何时停用。
6. **风险 / 待办清单**：已知 standing 后倾、未匹配先验风险、WM contact generalization、数据缺失、GPU/依赖限制，避免夸称实验成功。

最终汇报请按以下小节：

```text
Summary
Changed files
Contract / I-O comparison
Stage 0 data QA
Stage 1 / Stage 1.5 / Stage 2 implementations
Checkpoint migration table
Reward differences and weighting switches
Tests run (command, pass/fail, actual evidence)
Smoke evaluation / standing gates
Ablation run commands
Unrun items and blockers
Next bounded experiment
```

请优先做**工程上最小、可测试、不会破坏基线**的实现；如果较高级的动态运动先验 / paired BPTT bridge 尚未有足够数据或 Stage1 checkpoint 支撑，可以先交付**直接 Foot-RL 路径**的可测试实现和明确开关，再报告未完成项；不得以未经执行的训练假装已经成功。Action MSE 默认关闭，且绝不称为论文原始 Distillation。
