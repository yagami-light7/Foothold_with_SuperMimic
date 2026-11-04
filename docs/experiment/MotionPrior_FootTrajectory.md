# 实验记录：LAFAN1 全身运动先验 → 足端插补轨迹控制（S54 + SuperMimic）

> 状态：**工程实现及有界 smoke 已完成；Standing Gate 未通过，尚未获得收敛策略**。实际执行记录、命令与限制见第 12 节。  
> 编制日期：2026-11-04  
> **修订说明（v2）**：校正 SuperMimic Appendix B 的下游接口训练定义。Stage 1.5 是可选的配对数据桥接实验，默认优先采用 **BPTT 训练 Foot Encoder**；Action MSE 仅作为独立消融，不是原论文的蒸馏损失。保留原始 v1 文件以便对照。  
> 项目：`Foothold_with_SuperMimic`（以检查时 GitHub `main` 的 S54 分支实现为依据）  
> 机器人：**乐聚 KUAVO S54，29 DoF**；仿真与训练：Isaac Lab 3.0 + Newton/MJWarp + PyTorch SuperMimic（可微世界模型、BPTT、Critic）  
> 目标硬件：RTX 3090；本轮实际使用 **RTX 4060 Laptop 8 GB**，GPU 验证在沙盒外执行。3090 未实测。

## 0. 实验摘要与研究问题

此前使用**五次多项式生成足端轨迹**作为训练目标，SuperMimic 策略在 Standing / 初期 Foot Tracking 中表现出上身姿态不自然、向后倾倒等问题。尤其应区分：

- **欠约束问题**：只规定双脚轨迹，躯干、肩、肘、腰等自由度的自然协调没有足够监督；
- **基本平衡尚未学习**：2026-11-03 的 `STANDING_DIAGNOSIS.md` 显示，早期 Smoke Checkpoint 在 96/100 控制步（约 1.92 s）后倾并触发高度终止；同一站姿的固定默认 PD 也不能长期平衡；
- **World Model 偏差**：诊断中存在多步预测的高度和 Pitch 偏差，不能只依据 imagined reward 认定策略有效。

因此，本实验不是“只把 LAFAN1 动作拼接到 Foot Reward 中”，而是检验：**先通过全身参考运动训练自然动作能力，再将其迁移给足端轨迹条件控制器，是否能在保持足端跟踪与平衡的同时改善上身自然性和训练效率？**

预期分阶段执行：

1. **Stage 0 数据与物理准备**：筛选 S54 重定向的 LAFAN1 Walk/Stand/Turn 片段，检查参考可执行性、数据契约和站立基线。
2. **Stage 1 Full-body Imitation**：保留现有 `motion` 模式，通过 LAFAN1 全身参考训练 Motion Policy 和 World Model。
3. **Stage 1.5 Paired Transfer（可选桥接与消融）**：从 *同一段* LAFAN1 生成双脚参考，优先通过 **Motion Tracking Reward + WM/BPTT** 更新 Foot Encoder；可附加配对 latent alignment。直接 Action MSE 仅作为可选对照。
4. **Stage 2 Foot-trajectory Control（可直接从 Stage 1 启动）**：随机采样五次项足端轨迹，训练 Foot Policy；保留仅在语义匹配时才使用的上身动作先验与平衡约束。
5. **对比和消融**：区分 WM 迁移、Actor 迁移、上身奖励、过渡课程的独立贡献。

> 术语界定：Stage 1 训练的是 **reference-conditioned motion-tracking policy**，不是已经学会独立生成自然动作的通用先验网络。Stage 2 能否在不输入全身参考时保留这种能力，必须实测。

## 1. 当前仓库已实现的接口（基线事实）

第 1–9 节保留实施前的设计依据；涉及“拟新增/待实现”的文字表示当时的状态，最终实现与差异以第 12 节为准。Encoder 和 Decoder **各自都是完整的 MLP**，不是一个大 MLP 的首层与末层。

### 1.1 机器人状态、观测、动作

- `NUM_J = 29`（S54，**不是 G1**；G1 的 LAFAN1 数据不能未经 S54 retarget 就用于本实验）。
- 物理状态：`S=(p_root, R_root, v_root, ω_root, q, qdot)`，序列化时 **71D** = `3+4+3+3+29+29`；旋转内部采用 SO(3)。
- 机器人本体观测：**99D/帧** = 29 关节位置偏差 + 29 关节速度 + 3 角速度 + 3 projected gravity + 29 previous action + 6 相对目标 Root/Heading 朝向。
- 历史长度默认 `hist=10` → 990D 展平输入。注意已有 `configs/s54_standing_3090.json` 用 `hist=2`，**如果 Stage 1 / Stage 2 的历史长度不同，Decoder 的输入尺寸改变，不能无条件复用权重**。设计时必须固定两阶段相同 `hist`，或显式实现适配与拒绝迁移。
- Policy 输出 **29D Action**，环境将其转换为 `q_target = q_default + ξ_joint_offset + action_scale ⊙ action` 的隐式 PD 关节位置目标偏移；**不是力矩输出**。

### 1.2 Stage 1：现有 Motion Goal

`goal_mode=motion`，每个未来参考帧 73D：

| 字段 | 维度 | 说明 |
|---|---:|---|
| Root 相对位置 | 3 | 基于参考 heading 的相对平移 |
| Root 朝向 6D | 6 | 参考朝向 |
| Root 线速度 | 3 | 局部表达 |
| Root 角速度 | 3 | 局部表达 |
| 关节位置 | 29 | S54 关节顺序 |
| 关节速度 | 29 | 在 Goal 中有数值缩放 |
| **合计** | **73** | 默认 K=10 帧 → **730D** |

采样步长 `0.02s`，默认未来窗口 `0.02…0.20s`。`MotionLib` 从重定向 CSV + JSON sidecar 提取真实 S54 参考，源数据 30 FPS，内部重采样至 50 Hz。

### 1.3 Stage 2：现有 Foot Goal

`goal_mode=foot_trajectory`，每个未来时刻 23D：

| 字段 | 维度 | 说明 |
|---|---:|---|
| 左脚相对位置、朝向6D、计划接触mask、摆动相位 | 3+6+1+1=11 | 目标表示在**当前 Root Yaw 局部坐标系** |
| 右脚同上 | 11 | 同上 |
| 未来时间偏移 | 1 | 轨迹时钟 |
| **合计** | **23** | 默认 K=10、间隔0.1s → **230D**，覆盖未来1s |

`FootstepCommand` 由轨迹参数和触地时钟确定，`GoalSource` 负责查询。**计划 contact mask 不等于真实受力接触**。当前 `foot_trajectory.py` 直接生成交替步态五次项；Stage 1.5 要另加一个能表达原始 LAFAN1 足端轨迹的 GoalSource，不能假装任意 LAFAN1 足轨都天然符合当前 AlternatingConfig 的五次项形式。

### 1.4 当前 Policy / World Model / Critic

```text
Stage 1:  goal_motion(730) ── Motion Encoder ── z(32) ┐
                                                    ├─ Decoder(obs_hist, z) ── action(29)
           obs_hist(10 × 99) ────────────────────────┘
Stage 2:  goal_foot(230) ─── Foot Encoder ─── z(32) ──┘ [拟新增独立编码器]

World Model (两个阶段共享接口):
  input  = invariant_state(71) + action(29) + normalized ξ(33) = 133D
  output = Δroot_position(3) + Δrotation_SO3(3) + Δv(3) + Δω(3)
           + Δq(29) + Δqdot(29) = 70D
```

世界模型学习 `fω(S_t,a_t,ξ)→S_{t+1}`，不需要把 LAFAN1 Goal 730D 或 Foot Goal 230D 加入动力学输入。只要机器人、执行器、地形环境模型和仿真 `dt` 未变，结构可复用；**但 Stage 2 的接触和动作分布改变时，仍需用真实 Foot Rollout 持续校准 WM**。

当前 Critic **任务依赖**：Motion Critic 输入 914D，Foot Critic 输入 365D，均输出标量价值。**不能跨任务原样加载 Critic**。

重要差异：两模式的第 99D 观测中的相对 Root 朝向也存在语义差异——Motion 相对于参考 Root，Foot 相对于命令 heading。即使同为 99D，复制 Obs Normalizer / Decoder 也不是自动无风险；Stage 1.5 必须有配对验证。

### 1.5 从整体 Policy 到 Encoder–Decoder 的对应关系

原始直观理解 `π(o,g) → a → WM(s,a,ξ) → s'` 完全正确。当前代码更明确地把 Policy 分成 **Goal Encoder（完整 MLP）** 和 **Action Decoder（另一个完整 MLP）**：

```text
Motion:  g_motion[730] ── E_m（MLP） ── z_m[32] ┐
                                           ├── D（MLP）── a[29]
          obs_history[hist×99] ─────────────┘
Foot:    g_foot[230]   ── E_f（MLP） ── z_f[32] ┐
                                           ├── 同一个 D（拟迁移）── a[29]
          obs_history[hist×99] ─────────────┘
                       a[29], s[71], ξ[33] ── World Model（独立 MLP）── Δs[70]
```

- `z = E(g)`：Goal Encoder 接受**目标**，而非把整段 Obs History 也全部塞入 Motion Encoder。Obs History 在 Decoder 与 `z` 合并；论文的速度接口是接收部分观测历史特征的特殊情况。
- `a = D(o_hist,z)`：Decoder 是完整 MLP，不是最后一层 Linear。它输出 S54 29D PD 目标偏移。
- `s_next = fω(s,a,ξ)`：WM 学共享动力学，与 Policy Encoder、Decoder 是独立网络。WM 在每轮数据采集后继续更新，但在每次 Actor-BPTT 中冻结**参数**。
- 训练新 `E_f` 时若冻结 `D` 权重，仍必须**保留 `D` 前向计算图**以将梯度经 `D` 传回 `E_f`；不能对整个 Decoder 前向使用 `torch.no_grad()` 或 `detach()`，否则梯度被切断。

**文献界定**：SuperMimic 主体使用 Motion Encoder + Decoder + WM/BPTT；论文 Appendix B 扩展为每个接口独立 Encoder、Critic，跨接口共享 Decoder 和 WM。下游「distillation」指**从参考运动读出命令、用 Motion Tracking Reward 沿 WM/BPTT 更新下游 Encoder**，并非 Action MSE。论文还研究了下游任务 RL 及配对 Latent Alignment。见 <https://arxiv.org/abs/2610.28480>（Appendix B、Table A.4、Algorithm A.1）。

## 2. 假设与可证伪标准

- **H1：Motion Prior 可迁移**：在相同 Foot Goal、WM 和训练预算下，迁移 Actor 相比从零初始化可降低上身异常 / 提升真实仿真稳定性，而不明显牺牲 Foothold Tracking。
- **H2：WM Transfer 可迁移**：预训练 WM 相比从零初始化，能提高初期多步状态预测精度和足端任务的样本效率；如 Foot 数据下误差更大，则该假设不成立。
- **H3：动态上身先验优于默认姿态惩罚**：速度/相位匹配的 Upper-Body Prior 比固定 `q_default` 正则更自然；没有对应姿态时不能强制使用不匹配 LAFAN1 clip。
- **H4：Paired Bridge 的增益（待验证，非必要前提）**：先利用 LAFAN1 成对足端目标进行 BPTT Bridge，是否优于 Stage 1 → Stage 2 的直接 Foot-RL 迁移；若无改进，可跳过 Stage 1.5。

判据以 **真实 Newton Rollout** 的落脚 RMSE、站立时长/跌倒率、上身姿态质量为准，训练 Reward 只作为辅助；各奖励定义不同，**不能直接比较两阶段总 Reward 数值**。

## 3. Stage 0：准备及安全门槛

### 3.1 数据

1. 从**S54 版本** retargeted LAFAN1 中筛选 Walking、Slow Walking、Start/Stop、Turning、Standing（若原数据缺失则如实记录，不臆造 clip）。优先使用仓库 `select_reference_files()` 的 glob 筛选；保存 manifest 包含具体文件、时长、路径、校验和。
2. 每段检查源/目标 FPS、S54 29 关节排列、CSV/JSON sidecar 一致、xyzw/wxyz、重采样（SLERP/数值连续）、限位、Root 高度、脚穿地/脚滑移、初始落地及样本重复。
3. 将 **训练 clips 与评估 clips** 按原 clip 分割，禁止同一原始片段或相邻窗口泄漏到 holdout。
4. 保存每段参考的 Root / 29关节曲线、双脚 FK 位置/速度、估计接触事件和可视化；接触标签使用高度/速度/持续时间与滞回，不把计划接触当真实受力测量。
5. 如原始参考无法物理执行，先修复或剔除，记录理由；不要靠放宽 Reward 终止阈值掩盖数据质量问题。

### 3.2 基线诊断

仓库 2026-11-03 Standing Smoke 在约 1.92s 后倾终止；默认 neutral pose PD 也无法持久维持直立。先独立复现相关评估并保留已有结果，确认机器人物理、初态、接触力、姿态及控制 Action 尺度。

**门槛：没有通过稳定站立和 WM 配对预测检查之前，不开启无监督的长时正式 Foot 训练。** 可以进行有边界的训练、数据检查和单步 / 少量环境 Smoke，但不可将其作为已收敛能力。

## 4. Stage 1：LAFAN1 全身 Motion Tracking 预训练

### 4.1 目标与 I/O

```text
输入：obs_hist [B, hist, 99] + goal_motion [B, 730]
Motion Encoder：730 → 2048 → 1024 → 512 → 256 → z32
Decoder：(99×hist + 32) → 2048 → 1024 → 512 → 256 → action29
环境：29D PD target offset → S54 Newton
世界模型：[invariant_state71, action29, ξ33] → state_delta70
Critic：Motion 914D → V_motion [B]
```

**固定两阶段使用相同 `hist` 与同一执行器 / S54 模型契约**，避免伪装成可迁移而发生维度不一致。

### 4.2 保留当前 Motion Reward（先不大改）

源代码 `supermimic/mdp.py` 当前计算：

\[
\begin{aligned}
r_{motion} = \Delta t\big(&\exp(-e_{body,pos}/\sigma_p^2)
+\exp(-e_{body,ori}/\sigma_R^2)\\
&+\exp(-e_{body,linvel}/\sigma_v^2)
+\exp(-e_{body,angvel}/\sigma_\omega^2)\\
&+0.5\exp(-e_{root,pos}/\sigma_{rp}^2)
+0.5\exp(-e_{root,ori}/\sigma_{rR}^2)\\
&+w_{action}\|a_t-a_{t-1}\|^2
+w_{limit}c_{limit}
+w_{head}c_{head}\big).
\end{aligned}
\]

- 主体是 **14 个 tracked bodies** 的位置/姿态/速度以及 Root 参考跟踪；不要写成“代码已经有所有关节的独立位置奖励”。
- 另外已有 Action Rate、Joint Limit、Head Position/Velocity 惩罚。
- Reference position 会按当前机器人 Root XY/Yaw 重锚定（见 `mdp.py`），需保持真实/imagined 两侧同定义。
- Stage 1 先使用仓库已验证的标准配置；若要调整奖励，另做配置版本与消融。

### 4.3 Stage 1 验收

- **真实仿真**下站立 / 行走 / 转向参考片段按需可执行；必须检查 Head、肩、腰、骨盆、足端视觉自然性。
- 记录每个 clip 的 body pos/orientation tracking、Root 高度和 Pitch、累计成功步数 / timeout、关节越限和真实接触。
- 保存 `WM`（含 in/out normalization）、`Actor`（含 observation/goal normalization）、`Critic`、优化器状态、机器人 / Goal / Task Contract、筛选 clip manifest、随机种子、配置快照。
- 记录 World Model 的 1 / 4 / 8 步同动作配对预测误差和真实/想象 Reward 差异；不得仅以较低 WM Loss 宣布通过。

## 5. Stage 1.5：Paired Foot Transfer（可选；以论文 Appendix B 为依据）

**定位：Stage 1.5 不是 Stage 2 的必经之路。** 最简单可执行路线为 `Stage 1（Motion） → Stage 2（Foot RL）`：加载 WM + 兼容的 Shared Decoder，新增 Foot Encoder，直接通过 Foot Reward + WM/BPTT 训练。Stage 1.5 仅用于检验配对参考是否改善迁移稳定性与样本效率。

### 5.1 数据和网络准备（所有迁移方案共用）

1. 从**同一段** LAFAN1/S54 全身参考通过 FK 抽取左右脚的连续位置、朝向、速度和接触/相位；构建 Foot Goal 原有 **23×K** 的目标合同，并保留原始 Motion Reference（仅用于训练中的监督、评估或 Teacher）。这不是用随机 Quintic 轨迹替代原参考。
2. 配对 `g_motion(t)` 与 `g_foot(t)` 必须使用同一 clip、同一控制时刻和匹配的坐标系；各自仍遵循 Motion 0.02s、Foot 0.1s 的未来采样间隔。Foot 的计划接触标注需要 QA，不能假装是 Newton 实际接触力。
3. 载入 Stage 1：`E_m`（可以冻结用于参考）、`D`（加载为共享 Decoder）、`WM`（加载且在后续真实 rollout 继续更新）；新建 `E_f` 与独立 Goal Norm / Foot Critic。处理 `obs_motion` 与 `obs_foot` 最后 6D 的参考 heading 差异，不得直接复用语义不匹配的 Observation Norm。
4. `E_f(g_foot) → z_f(32)`，然后 `D(o_hist,z_f) → a(29)`。Foot Encoder **不会绕开 Decoder 直接输出 29D Action**。

### 5.2 训练路线 1：论文式参考运动蒸馏（推荐的配对 Bridge）

论文 Appendix B 的「Distillation」不是简单模仿 Teacher Action；它从同一段运动提取下游目标，并对下游接口**继续使用 Motion Tracking Reward**，经 WM/BPTT 更新该接口 Encoder。针对 Foot 的扩展是：

```text
Paired g_foot(t) → E_f → z_f → D(冻结参数) → a_t
                               ↓
                        WM（Actor 更新时参数冻结）
                               ↓
                 predicted state / full-body reference
                               ↓
                  Motion Tracking Reward + Critic
                               ↓ BPTT
                         更新 E_f 参数
```

即优化 `max_{θ_f} E[J_motion(π_foot)]`，Motion Reference 是**训练奖励的监督目标**，不会作为 Foot Actor 部署输入。这里的配对参考与 Foot Goal 基本相容，但不同运动模式、接触标签噪声或参考时窗不匹配仍可能造成冲突。

论文中的多接口训练还会继续用 Motion 任务训练共享 Decoder；若当前工程为了简单采用顺序训练并始终冻结 Decoder，这是**受论文启发的工程简化**，不能表述成严格复现论文的跨接口联合训练。开发时明确提供冻结 Decoder 和可选联合 Motion 数据更新 Decoder 两种模式，优先验证简单模式。

### 5.3 训练路线 2：直接 Foot Reward 的 RL Encoder Adaptation（Stage 1.5 可跳过）

不使用完整参考做 Actor Loss，而直接通过新任务奖励优化 `E_f`：

\[
\max_{\theta_f}\mathbb E\left[\sum_{k=0}^{H-1}\gamma^k r_{foot}(\hat s_{t+k},g_{t+k}^{foot})+\gamma^H V_{foot}(\hat s_{t+H},g_{t+H}^{foot})\right].
\]

梯度：`Foot Reward → predicted states → frozen-parameter WM → frozen-parameter D → E_f`。训练期间 `WM` 仍用真实足端任务 Replay 更新。此方法**不需要 Teacher Action，也不必须使用 LAFAN1 Paired Data**；但必须保证任务奖励足以约束自然姿态并监控 WM 偏差。

可选在配对参考样本上增加潜变量对齐：

\[
L_{latent}=\|E_f(g_{foot})-\operatorname{sg}(E_m(g_{motion}))\|_2^2.
\]

该项仅用于同段、同时间且语义可比较的样本，系数通过独立消融决定；**不可**对不匹配的随机 Quintic Goal 强制匹配某段任意 LAFAN1 Latent。

### 5.4 训练路线 3：Action MSE（仅可选实验，不应冒充原论文蒸馏）

如需要单步监督 warm start，可在**相同底层机器人状态**和同段 Motion/Foot 目标下比较：

\[
L_{action}=\|D(o_{foot},E_f(g_{foot}))-\operatorname{sg}(D(o_{motion},E_m(g_{motion})))\|_2^2.
\]

它可以不用 WM 完成单步监督更新；但 **Teacher Action 不是唯一正确答案**，配对的两种 Obs Relative Heading 并不完全相同，且单步动作接近**不证明**闭环足端跟踪或稳定平衡。此方法列为额外对照而非主实现或必须通过的 Gate。

### 5.5 配对 Bridge 的验证与退出规则

- 必须在真实 Newton 中进行闭环验证，记录足端误差、实际接触、Base Pitch/Height、上肢运动及跌倒原因；配对训练 Loss 下降不等于迁移成功。
- 依次检查：冻结 D 时 E_f 梯度非零且有限；D 参数保持不变；WM 的 Actor-BPTT 权重冻结但梯度可经其输入传递；Stage 1 Motion 旧行为未被新接口破坏。
- Stage 1.5 **实验失败或收益不明显时**，允许直接走 Stage 1 → Stage 2，不阻塞后续主要研究。只有 Stage 2 正式训练前的站立/WM 验证门槛不能跳过。

## 6. Stage 2：五次项 Foot Trajectory Tracking

### 6.1 I/O 与迁移

```text
输入：obs_hist [B, hist,99] + foot_goal [B,230]
输出：action [B,29] → 同一 S54 implicit PD
WM：133 → 70，结构不变；继续在 Foot 的 Newton Replay 上训练
Critic：Foot 输入 365D → scalar V_foot；重新初始化
```

- 新任务参数 `--init_from_motion <path>` **独立于**现有 `--resume`；后者要求任务合同一致，仅用于同任务续训。
- Stage 2 可以复制 compatible 的 WM 权重与 in/out normalizers；WM optimizer 建议重新初始化，并对新 Foot Replay 做 bounded warmup。
- Policy：复制兼容的 Decoder，单独初始化 Foot Encoder / `goal_norm`；只有验证 `hist`、Joint/Action contract 和 Observation 语义一致后才能迁移 `obs_norm`。
- Critic：Foot 奖励与目标分布变化且输入为 365D，**重新初始化**并训练。
- Stage 1 Motion Replay 和 Stage 2 Foot Replay 可用于 WM 混合训练（如有保留 / 合法的转移数据）；禁止在 WM 单独输入 230D 目标。WM 由真实 \((S,a,\xi,S')\) 训练，而非对 reference 轨迹直接拟合。

### 6.2 当前 Foot Reward（现有代码）

`foot_mdp.py` 当前使用平滑质量函数：

\[
Q(e,\sigma)=\frac{1}{1+e/\sigma^2},\quad e\ge0.
\]

| 奖励 / 惩罚 | 当前默认权重 | 目的 |
|---|---:|---|
| Swing Position / Velocity | 1.00 / 0.25 | 摆动期双脚位置、速度追踪 |
| Stance Position / Velocity | 1.00 / 0.25 | 支撑期脚位稳定、低速 |
| Touchdown Accuracy | 0.50 | 计划时刻落脚点精度；当前按**计划事件**计算 |
| Foot Orientation | 0.25 | 脚方向跟踪 |
| Base Upright | 1.00 | Root 直立 |
| Base Height / XY Balance | 0.50 / 0.50 | 站高与 Root 相对支撑区域位置 |
| Action Rate | −0.10 | 抑制控制跳变 |
| Joint Limit | −2.00 | 软限位惩罚 |
| Posture to `default_q` | −0.05 | **静态中立姿态**，不等于自然 Walking Prior |
| Joint Velocity | −0.001 | 高速关节正则 |

当前 `dt=0.02s`，最终 reward 乘以 `dt`。这套值只是已有实现基线，**不是推荐直接增大 prior 权重的证据**。

### 6.3 拟增加的自然运动先验奖励（尚未实现）

建议先保持已有 Foot + Balance 结构，只在独立配置中增加：

\[
\boxed{
 r_{Stage2}=r_{foot}+r_{balance}+r_{regularization}
 +\lambda_u r_{upper}\,+\lambda_T r_{torso}\,+\lambda_{uv}r_{upper,vel}
}
\]

1. **动态 Upper-Body Posture Prior**（优先级最高）：
   \[
   r_{upper}=\exp\!\left[-\frac{\|q_{upper}-q^{prior}_{upper}(v,\varphi)\|_W^2}{\sigma_u^2}\right].
   \]
   `q_upper` 使用 S54 的显式关节名索引（头、腰、肩肘等）；Walking 应使用速度 / 相位匹配的 LAFAN1 模板，Standing 应使用站立模板。**不强制模仿与新足端目标不相容的双腿参考**。
2. **Torso Orientation Prior**：
   \[
   r_{torso}=\exp[-\|\mathrm{Log}(R_{torso}^{prior\,T}R_{torso})\|^2/\sigma_T^2].
   \]
   允许转弯、加减速所需的适度躯干运动，不能把世界直立约束写成“永远零倾角”。
3. **Upper-Body Velocity Prior / Smoothness**：
   \[
   r_{upper,vel}=\exp[-\|\dot q_{upper}-\dot q^{prior}_{upper}\|^2/\sigma_{uv}^2].
   \]
   与已有 `joint_velocity`、`action_rate` 共同监控，避免重复强惩罚。

**适用条件与失配处理：**

- 若样本来自同段 LAFAN1 paired foot reference，可以合理使用对应上肢姿态 / Root 参考作为辅助奖励；
- 若样本来自**随机五次项**，原则上不能强制跟踪随便挑选的 LAFAN1 全身参考；可用速度/相位匹配模板、弱上肢正则、或 teacher/student 正则等，若没有可信匹配则自动减弱 / 禁用该项，至少保留基础平衡奖励；
- 相位/速度对齐应有明确来源、插值、缺失值策略和单测；不要用 reward hacking 的方式使躯干僵硬、足端不跟踪。

### 6.4 推荐课程与参数策略

| 阶段 | 采样 | 先验 | 训练重点 |
|---|---|---|---|
| Stage 1.5（可选） | 100% LAFAN1 对齐 Foot | 论文式 Motion Tracking Reward + WM/BPTT（或可选 Action MSE 消融） | 检验低维 Goal 的配对迁移是否有效 |
| Stage 2A | LAFAN1 Foot + 保守幅度的 Quintic Foot 混合 | 中等，匹配样本更强 | 学会偏离参考并保持站立 |
| Stage 2B | 增大 Quintic 比例及步长 / 步频 / 转向范围 | 弱至中，防止扭曲 | 泛化与落脚精度 |

**不要预设固定 1.0/0.3/0.1 的系数就投入长训**。先记录各分量的 `mean`, `std`, Actor 梯度范数及实际占比，再小范围调权。课程升级由真实 eval 指标触发，不以迭代数单独触发。尤其 Standing 稳定性未通过时不可直接上大步、快速转向。

### 6.5 SuperMimic 特别要求：可微性与时间一致性

- Foot reward / Upper prior 应在预测 \(S_{t+1}\) 上经 FK、关节 / 姿态计算，**保证 WM → Reward → Actor 的非零且有限梯度**。
- 离散实际接触事件不可随意直接替换 `FootTrackingMDP` 的可微目标。真实接触力 / 足滑移宜作为独立评估与辅助指标，若引入可微代理须明确验证。
- 真实 rollout、Replay 重建、Imagination 必须使用**同一命令快照、相同的 `ep_step` / `ep_step+1` 规则**；不能在 World Model 预测未来时重新随机采样轨迹。
- 当 WM 进入新动作分布，需在 1/4/8 步、落地前后状态比较同动作的 Newton 真值和 WM 预测；不允许只比较不同批次 Reward 均值。
- 先验证 Stand 2/5/10 秒（多个初态种子、无噪声+扰动版），再验证前进、原地转向、行进转向。

## 7. 消融实验（公平对照）

建议至少以下实验，**Stage 2 使用同一训练目标分布、同一评估集、相同总仿真步数、相同实测资源预算**；隔离变量、分别保存种子：

| ID | Stage 2 Actor 初始化 | Stage 2 WM 初始化 | Upper Prior | Stage 1.5 Bridge | 检验问题 |
|---|---|---|---|---|---|
| A | 随机 | 随机 | 无 | 无 | 原始 Foot 基线 |
| B | 随机 | Stage1 WM | 无 | 无 | 仅 WM 迁移 |
| C | Stage1 Decoder + 新 Foot Encoder | Stage1 WM | 无 | 无；**直接 Foot-RL** | Actor Decoder 初始化的额外作用 |
| D | 同 C，先做配对 BPTT Encoder Adaptation | Stage1 WM | 无 | **Motion Reward + BPTT** | 配对桥接价值 |
| E（可选） | 同 C，先做配对 Action-MSE warm start | Stage1 WM | 无 | **Action MSE** | 不同过渡优化目标的价值 |
| F | 同 D | Stage1 WM | 有 | 同 D | 动态自然姿态奖励增益 |
| G（可选） | 同 C | Stage1 WM | 有 | 无 | 在不做桥接时 prior 是否有效 |

- A↔B：仅 WM 初始化改变；B↔C：仅 Decoder 参数初始化改变（其余相同）；C↔D：增加论文式 Paired BPTT Bridge；D↔E：替换 Bridge 优化目标；D↔F：仅增加 prior reward；C↔G：单独检验 prior。
- Stage 1.5 额外真实仿真步数应单独报告；既报告 **Stage 2 环境步数**，也报告**含桥接的总环境步数**，避免把额外训练量误当作样本效率提升。无 WM 监督蒸馏仅需分开统计其训练步数与计算时间。
- 主指标：2/5/10 s 站立成功率、跌倒率 / 终止原因、Root 高度/roll/pitch/yaw、脚轨迹位置/速度 RMSE、touchdown 误差、真实接触/足滑移、速度/转向跟踪、上肢姿态与速度、动作平滑。
- 自然性：既有 LAFAN1 holdout（可计算相位/速度匹配误差）+ 统一脚本录制全身视频，避免只依据静态 `q_default` 评分。
- 训练效率：达到同样客观性能门槛所需 Newton 样本量、Wall Time、WM 多步误差、Imagined/Simulator 配对奖励差；区分 GPU 时间和环境步数。
- 所有指标记录均值、分位数 / 方差及多个独立 seed；不得用 4 个相同环境初态当独立种子。

## 8. Checkpoint 迁移、复现及运行安全

### 8.1 模型保存与加载合同

- Stage 1 checkpoint：`wm`, `pi`, `critic`, `opt_*`, normalization stats, `model_contract`, `goal_contract`, train args, clip manifest/hash, seeds。
- Stage 2 `--init_from_motion`：检查 robot id、29 DoF、关节排列、action_scale、`control_dt`、WM input/output、观测历史 `hist`、坐标/朝向语义。仅加载 **WM + 可兼容 Policy Decoder**；新建 Foot Encoder/goal_norm/Critic/优化器。**直接 Stage 1 → Stage 2 是必须支持的主路径；Stage 1.5 可选。**
- `--resume` 仅用于相同 Stage / 相同 Goal 与 Task Contract 的恢复；**不要通过绕过合同检查“强行 resume”**。
- 若历史长度 / 观测语义不同：拒绝迁移或实现经过测试的显式 adapter；不可 `strict=False` 后静默忽略关键权重。
- 保存迁移映射报告：每个模块已复制/重新初始化的名称、shape、norm 是否复用和 reason。
- Stage 2 保留可选 Frozen Teacher 用于配对蒸馏，最终部署 Actor 不接收全身参考 / Teacher。

### 8.2 训练与实验运营

- 将实验运行目录设为 `logs/s54_*` **唯一名称**；不能覆盖旧 4060/3090/G1 logs、checkpoint 或现有诊断。
- 先 dry-run / CPU 单测 / Newton 1/16 env / 2–5min bounded smoke，再逐步增大并行数；`3090` 的内存 / 数据吞吐应**实测**再设置 `num_envs` 和 `rollout`。
- SSH 可能断开时使用 tmux 或安全的启动脚本；保存 stdout/stderr、配置快照、Git commit id、数据 manifest 和运行状态。
- 若已有 SwanLab 集成，记录 eval 关键指标、WM loss、Actor/Critic loss、奖励分量、梯度、checkpoint 索引；网络失败应本地持久记录并允许之后重传，不要因远端记录失败中断实验。
- Smoke 通过不代表已经完成正式训练；8 小时是以往目标，不是本实验已验证的训练耗时承诺。

## 9. 代码影响范围与验收清单

建议的待改区域：

- `supermimic/goals.py`：`reference_foot` 或 `paired` GoalSource、统一 goal contract / replay time。
- `supermimic/motion.py` 与独立预处理脚本：S54 LAFAN1 双脚 FK 提取、相位/contact 信息、manifest。
- `supermimic/networks.py`：可选 Dual Encoder / Foot Student，兼容原 `Policy`。
- `supermimic/train.py`：`--init_from_motion` 独立迁移；paired batch、reward prior、学习率和课程开关；旧模式不回归。
- `supermimic/foot_mdp.py`：动态姿态 / 躯干 / 上肢速度 reward，可微并分项记录。
- `supermimic/buffer.py`：必要时保存 paired reference id/phase 与 immutable command snapshots，防止重放时错配。
- `supermimic/foot_evaluate.py`：stand/walk/turn、holdout clip、真实接触与视频统一评估。
- `configs/`、`tools/`、`docs/`：stage 配置、测试、独立日志、README 与迁移记录。

**最低验收**：

- [x] Stage 1 / Stage 2 的输入维度、训练目标及 Goal Contract 有单测；不混用 G1 与 S54。
- [x] 保持 WM `133→70`、Action `29`、S54 的 PD 定义不变。
- [x] 可选 Stage 1.5 的 `motion ↔ foot` 配对字段与时间同步可复现，Clip holdout 无泄漏；不做 Bridge 仍可直接进入 Stage 2 的 bounded smoke。
- [x] Cross-task `--init_from_motion` 有独立命令，迁移成功/不兼容失败均有测试。
- [x] 可微 Upper Prior 在 Standing / Walking、匹配 / 不匹配目标上有正确行为和梯度。
- [x] 真实 Newton / Replay / Imagined 的 Goal 与 Reward 重建一致，跨 reset 不污染。
- [x] 复现 standing 失败诊断，保留原 baseline，不把 smoke 写成成功。
- [x] 各消融可在相同采样 / eval 合同下独立启动，具有 checkpoint、可中断恢复和日志。
- [ ] Stage 2 真实 Newton 的稳定性、自然性与足端精度满足预先记录的评价门槛后才进入长训。

## 10. 消融状态（2026-11-04）

| 日期 / Git SHA | run_id | 机器人 / 数据版本 | 实验组 | seed | env steps | WM 8步误差 | Stand 10s 成功率 | Foot RMSE | Upper-Body 指标 | 备注 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 2026-11-04 / fd1722e + 工作树修改 | s54_mp_stage2_A_smoke_20261104 | S54 / manifest verified | A | 54 | 92 | 见 §12 | 未测 | 0.01565 m (2s nominal) | 无匹配 standing 参考 | 第97步 height；仅接口 smoke |
| 同上 | s54_mp_stage2_B_verified_20261104 | 同上 | B | 54 | 92 + 上游92 | 见 inventory | 未测 | 未测 | 未测 | WM-only 迁移 smoke |
| 同上 | s54_mp_stage2_C_smoke_20261104 | 同上 | C | 54 | 92 + 上游92 | 见 §12 | 0/3 | 0.01535 m (2s nominal) | 无匹配 standing 参考 | Standing Gate FAIL |
| 同上 | — | 同上 | D | — | — | — | — | — | — | 入口 dry-run；Bridge 与 F 闭环接口另测 |
| 同上 | s54_mp_stage2_F_verified_20261104 | 同上 | F | 54 | 92 + 上游184 | 见 inventory | 未测 | 未测 | standing prior 自动关闭 | 缺真实站立模板；仅接口 smoke |
| 同上 | s54_mp_stage2_G_walking_smoke_20261104 | 同上 | G | 54 | 92 + 上游92 | 见 §12 | 不适用 | 0.04225 m (2s) | pose RMSE 0.3704 rad | 2s timeout 但 Pitch43°，非成功行走 |

## 11. 现有代码与文档出处

- `supermimic/mdp.py`：Motion Reward、99D Observation、71D Invariant Features。
- `supermimic/goals.py`：`motion` / `foot_trajectory` Goal，730D / 230D。
- `supermimic/networks.py`：Policy Encoder/Decoder，WM 输入与状态增量。
- `supermimic/train.py`：Replay、WM Update、BPTT、Critic、`--resume` Goal Contract 检查。
- SuperMimic 原论文：<https://arxiv.org/abs/2610.28480>，§3 主任务、Appendix B 多 Goal 接口 / Distillation 与 RL、Table A.4 网络结构；**不要把可选 Action MSE 当作论文原方法**。
- `supermimic/foot_mdp.py`：Foot Reward、Foot Critic 和终止。
- `supermimic/foot_trajectory.py`：五次项轨迹，计划 Touchdown 与命令时钟。
- `supermimic/s54.py`、`README_S54.md`：S54 关节/资产/运动文件合同。
- `docs/S54_FOOT_TRACKING_TRAINING.md`、`STANDING_DIAGNOSIS.md`：2026-11-03 的诊断、WM 配对误差、Smoke 成功与未通过的 Standing Gate。

> 仓库：<https://github.com/yagami-light7/Foothold_with_SuperMimic>。下方记录本轮实际代码修改及执行证据；未提交、未推送、未改动源运动数据或旧实验。

## 12. 实施与实验记录（2026-11-04）

### Summary

完成 Stage 0、Stage 1 有界入口、**不依赖 Bridge 的直接 Motion→Foot 迁移**、可选配对 Motion-Reward BPTT Bridge、速度/相位匹配上身先验、严格加载合同、评估/消融/部署导出。原始 Motion 默认路径与 Foot 无 prior 路径保留。没有修改 WM 状态定义、损失、状态积分或核心 BPTT 算法。

**工程接口通过，稳定 Standing 未通过。不得据此开始正式长训。** 原 Standing checkpoint 再现第 96 步 height 失败；新 A/C 的相同无噪声初态均第 97 步 height 失败。C 的 5/10 秒评估 0/3 成功。短窗口训练 reward≈0.97 不能作为收敛证据。

工作目录 `/mnt/sage_data/workspace/supermimic_s54_main`；基准 Git SHA `fd1722e127d8925e80aeb3202e3a451f81a211b8`，本轮修改尚未提交。开始时 tracked 文件无修改，用户的 `docs/prompt/` 与 `docs/experiment/` 为 untracked；保留原设计并在本文追加执行记录。无 AGENTS.md。Prompt 提到的 README/诊断实际位于 `docs/`，以真实路径为准。

核对用户提供的本地论文 `.../Zotero/storage/VLXYNR87/Liao et al. - 2026 - SuperMimic Scaling Humanoid Whole-Body Control with One Differentiable World Model.pdf`，第 14 页 Appendix B；同时核对 [公开 PDF](https://arxiv.org/pdf/2610.28480)。Bridge 使用 Motion Tracking Reward 经 WM/BPTT 更新 Foot Encoder；Action MSE 未实现且不是默认目标。论文联合训练还让共享 Decoder 接收 Motion 更新；这里采用顺序训练、冻结 D，属于工程改编。没有实现完整联合多接口训练器。

### Changed files

| 文件 | 改动及兼容性 |
|---|---|
| `supermimic/transfer.py`（新增） | 物理/Policy 合同、逐项严格校验、Motion/Bridge 初始化及权重迁移映射；初始化与 resume 分开 |
| `supermimic/motion_prior_data.py`（新增） | 带 hash 的 train/heldout manifest；S54 FK 数据、滞回接触/相位估计 |
| `supermimic/reference_foot.py`（新增） | 原参考 FK GoalSource、不可变配对命令、显式 PairedMotionMDP；不拟合随机五次项 |
| `supermimic/upper_prior.py`（新增） | 合格步态周期 bank、三个可关先验、无匹配/无站立模板回退 |
| `supermimic/motion_prior_runtime.py`（新增） | 严格重建三种任务；独立 Foot Actor 不加载 MotionLib |
| `supermimic/networks.py` | 显式 `encode/decode`、DualEncoderPolicy、观测语义适配；旧 Policy 默认 identity，原权重键保留 |
| `supermimic/goals.py` | 统一 23D 编码、`reference_foot` 合同；Motion/Quintic 原合同不改 |
| `supermimic/mdp.py`、`foot_mdp.py` | Motion Task Contract、加载校验；原奖励/终止公式及系数不变 |
| `supermimic/env.py`、`buffer.py` | 重置传入参考帧；共享仿真参数常量；按命令 schema 恢复配对 Replay，拒绝混合类型 |
| `supermimic/train.py` | 独立迁移开关、冻结审计、可选 paired latent、有限时限/信号保存、分项奖励/梯度、合同与分阶段成本记录 |
| `supermimic/evaluate.py`、`foot_evaluate.py` | 重建 adapter/paired/prior，支持 manifest holdout；原入口继续可用 |
| `supermimic/standing_diagnostics.py` | 同窗口 Motion/paired/Foot WM 1/4/8 步诊断，新增 Pitch 和足部 FK 误差 |
| `supermimic/curriculum.py`、`curriculum_runtime.py` | 可选真实姿态及配对 WM 双门控；未配置时保留旧行为 |
| `tools/prepare_s54_motion_prior.py` | CPU/dry-run 数据 QA、配对数据、contact 图与训练模板 bank |
| `tools/run_s54_motion_prior.py`、`ablate_s54_motion_prior.py` | 有界 launcher、独立 resume、新 run、18 条公平消融 dry-run 命令；新 run 保存源码快照 |
| `tools/evaluate_s54_motion_prior.py`、`render_s54_motion_prior.py` | 独立 Newton 评估、真实受力、配对 WM、全身同镜头录像 |
| `tools/export_s54_foot_actor.py` | 仅 obs_hist + foot_goal 的 TorchScript Actor，嵌入 norm/adapter |
| `tools/test_motion_prior_transfer.py`、`test_motion_prior_newton.py` | CPU 接受测试及真实课程门控集成测试 |
| `configs/motion_prior/*.json`、本文、`docs/README_S54.md` | Stage/profile/prior/curriculum 配置、记录与导航 |

未更改 `s54.py`、关节定义、源 MJCF/CSV、原 `foot_trajectory.py`、`run_train.sh`、SwanLab 实现或已有 logs/checkpoint。未上传、清理或覆盖已有实验。

### Contract / I-O comparison

| 合同 | Stage 1 | 可选 Stage 1.5 | Stage 2 |
|---|---|---|---|
| Goal | Motion 730D，10×73，0.02s | reference_foot 230D，10×23，0.1s | Quintic Foot 230D，10×23，0.1s |
| Goal contract | v1 motion | v2 reference_foot | v1 foot_trajectory |
| Actor 路径 | E_m→z32→D | E_f→z32→共享 D | E_f→z32→D |
| Obs / hist / action | 99 / 10 / 29 | 同左 | 同左 |
| 物理状态 / ξ / WM | 71 / 33 / 133→70 | 同左 | 同左 |
| Critic 总输入 | 184+730=914 | 135+230=365 | 135+230=365 |
| 重置、奖励、终止 | 原 Motion | 原 Motion（clip end 保留） | 独立 Foot MDP，无 Motion |
| 观测最后6D | Motion 相对根朝向 | 固定命令 heading，经 adapter mask | 同左 |

每个 Foot 时刻为 `[左p3,R6,c,phase；右p3,R6,c,phase；Δt1]`。命令保存世界坐标，局部编码 `Hᵀ(p_target−p_root), HᵀR_target`；通过预测根姿态保留梯度。命令时间 `ep_step×0.02`，动作后奖励使用 `ep_step+1`。配对模式另外保存 `start_frame,clip_id`，并断言 `start_frame+ep_step==f`；查询越过 clip end 时保持末帧，采样窗口仍不跨 clip/reset。

迁移时 Foot ObsNorm 新建，归一化后将每帧 `93:99` 置零。保留 projected gravity `61:64`，不删除倾斜传感器；不修改源 Motion Policy 的 observer 或 forward。该保守适配会舍弃命令相对朝向输入，朝向信息仍在 Foot Goal 中；需后续闭环评估其代价。

### Stage 0 data QA

最终数据目录：[motion_prior_stage0_verified](../../results/s54/motion_prior_stage0_verified/manifest.json)。训练 `walk1_subject1`、heldout `dance1_subject1`；按原始 clip 身份及 CSV/sidecar hash 防泄漏。**没有独立 heldout walking clip**，不能把 dance holdout 结果解释为 walking 同分布泛化。

| 数据 | 源帧数 / 30FPS | 控制帧 / 50Hz | 时长 | 估计接触比例 L/R | 估计落地 L/R |
|---|---:|---:|---:|---:|---:|
| walk1_subject1 (train) | 7840 | 13066 | 261.30s | .6062 / .5997 | 173 / 172 |
| dance1_subject1 (heldout) | 3945 | 6574 | 131.46s | .4273 / .4133 | 128 / 120 |

两段通过现有 S54 Loader/证书/限位/FK/重采样时间网格检查；源 xyzw、内部 wxyz，完整29关节。最小几何足底间隙约 0.003m，时间网格误差 <4e-14s。最大单控制步关节变化分别约 .350/.376rad，**几何 QA PASS 不代表参考动态可执行或足滑移合格**。头部两关节为既有 sidecar 标注的补齐值，不能据此宣称有独立头部运动监督。

估计接触采用 enter/exit 高度 `.025/.04m`、速度 `.35/.6m/s`、最短持续 `.06s` 的滞回/去抖；输出原始曲线，不能替代 Newton 受力。`*_paired.npz` 保存 root/feet/upper/phase，`paired_goal_samples.npz` 保存16个同 clip/tick 的 Motion/Foot Goal。

训练 bank 仅用 train walking 的完整左离地→下一左离地周期，右离地固定为相位 .5；筛掉明显转弯/侧移，最终 **40 个直行周期**。缺 stand、turn、slow_walk、start_stop 类别；没有合成站立模板。较早 `motion_prior_stage0/` 为探索产物，最终配置只引用 `..._verified/`。manifest 的原始源身份 hash 来自已有可信 sidecar；不宣称本轮重新读取了所有历史 raw 源文件。

### Stage 1 / Stage 1.5 / Stage 2 implementations

Stage 1 继续使用 14 tracked bodies/root/head 的原 Motion 奖励（不存在所有关节逐一显式 tracking 项）。manifest 固定为 train；Motion Goal 不进入 WM。

Stage 2 主路径 `--init_from_motion` 直接加载 WM+norm 与完整 D，建立 E_f、新 Norm/Critic/optimizer/Replay，继续用真实 Foot transitions 拟合 WM。**不加载 Teacher、不要求 paired CSV/Bridge**。`--transfer_components wm` 为 B，`wm_decoder` 为 C/G。`--train_decoder` 可明确微调 D；本轮所有 Stage2 消融统一冻结 D，包括 random A/B，属于“只适配 Encoder”的消融制度；不能把它等同于原始从零全 Actor 可训练基线。

Bridge 使用 `PairedMotionMDP`：Foot-only Actor 接收同段 FK 足部 Goal；Motion 参考只用于原奖励、重置/终止和特权 Critic。冻结 E_m/D；Actor 更新时 WM 参数冻结，但 D/WM 前向保留计算图。WM 每轮单独拟合真实 replay；`--paired_latent_weight` 默认0且只允许 reference_foot，无法用于任意随机 Quintic。没有 Teacher action 调用或 Action MSE。Stage2 从 Bridge 初始化后丢弃 E_m 及其 norm，并重置 Quintic GoalNorm。

本轮 Stage1 仅训练12 iteration，不是合格的 reference-conditioned teacher。Bridge smoke 仅证明接口可执行、Motion Reward 梯度能到达 E_f，尚未证明闭环改善。

### Checkpoint migration table

| 模块 | Motion→Foot C/G | Motion→Bridge | Bridge→Foot D/F |
|---|---|---|---|
| WM weights + input/output norm | 复制 | 复制 | 复制；继续真实数据校准 |
| 完整 D MLP | 复制，默认冻结 | 复制，冻结 | 复制，默认冻结 |
| E_m / Motion norm | 不载入 | 独立载入并冻结 | 丢弃 |
| E_f | 新建 | 新建 | 复制 |
| Foot ObsNorm | 新建 + 显式 mask | 新建 + 显式 mask | 复制同语义适配器 |
| Foot GoalNorm | 新建 | 新建 | 新建（原 FK→新 Quintic 分布） |
| Critic / 全部 optimizer / Replay | 新建 | 新建 | 新建 |

`--resume` 保持同任务严格校验，不能代替迁移。新增 robot/actuator/scale/solver/backend/dt/state、Goal、Task、hist、adapter/freeze、data hash 合同；键/shape/非有限值校验在修改模块前完成。错误 S54/G1、hist、shape、physics、task 均拒绝；不会 `strict=False` 静默漏载。历史 Motion 同任务读取仍支持，**缺新物理合同的历史 checkpoint 不自动获准跨任务迁移**。

每个 run 有 `transfer_report.json`，含 loaded/new/skipped 和逐权重形状/原因。实际 C 的 D 与 Stage1 最大权重差为0，Actor optimizer 仅产生 E_f 状态；source Critic/optimizer 未载入。最终报告修正了 Bridge→Foot 映射中误列已丢弃 Motion 权重的问题，并由单测覆盖；较早 smoke 的旧映射记录保留，不覆写历史。

最终代码又在 CPU 上严格重建了实际 Motion、Bridge、C、F 四个 checkpoint，并生成准确迁移映射：[final_contract_reconstruction.json](../../results/s54/motion_prior_validation/final_contract_reconstruction.json)。最终源代码快照及 hash receipt 位于 `results/s54/motion_prior_final_code/`。

checkpoint schema v2 保存合同、args、数据hash、seed/RNG、阶段实际步数/上游步数/监督步数、上游 wall time。较新的 B/F run 额外保存 `source_snapshot/` 及 manifest SHA，已验证文件 hash。更早 smoke 有基准 SHA、参数与数据合同，但没有完整 dirty 源码快照；以后应从最终代码及新 snapshot 重跑正式对照，不能凭同一基准 SHA 声称所有开发中 run 使用完全相同代码。Replay 未序列化；resume 重新采集并经过 WM-only 恢复期，非逐位延续旧 Replay。

Actor-only 导出：[motion_prior_foot_actor.ts](../../results/s54/motion_prior_foot_actor.ts)，[输入合同](../../results/s54/motion_prior_foot_actor.json)。CPU 可变 batch 前向比较 PASS；输入 `[B,10,99] + [B,230]`、输出 `[B,29]`；嵌入 norm/adapter/action bound，不依赖 WM、Teacher、CSV 或 prior bank。导出成功不表示适合上真机。

### Reward differences and weighting switches

原 Foot 的 Swing position/velocity、Touchdown、Stance position/velocity、Foot orientation、Base upright/height/XY balance、Action Rate、Joint Limit、Posture-default、Joint Velocity 均保留默认系数。`prior_off.json` 或不指定 prior 与 baseline 完全一致；无站立模板时，Standing 的奖励及最大单步奖励也完全一致。

`prior_matched.json` 的 `w_upper_pose=.03, w_torso=.02, w_upper_velocity=.01`，标准差 `.5rad/.35rad/1.5rad/s`。仅匹配 forward speed±.12m/s、yaw≤.15rad/s、lateral≤.05m/s 的同相位模板；上身为 S54 腰/双臂/头17关节，**不跟踪任意下肢或 Root 平移 reference**。躯干 SO(3) 目标随命令脚 yaw 转向；较大转弯或匹配失败时三个 prior 置零，保留原 q_default 正则。

真实与 imagined 共用同一 `mdp.reward`。真实接触布尔/受力只做独立诊断，不进入可微奖励。`reward_terms.jsonl` 记录 mean/std 和三个 prior 的权重后实际贡献；`gradient_terms.jsonl` 记录各 prior、base height/upright、foot 的 Actor 梯度。

实际 G/learned-WM 第12轮 per-term Actor norm：pose **1.91e-7**、torso **4.15e-8**、upper velocity **5.69e-8**；pure foot **6.06e-6**，总 Actor **3.13e-4**。全部有限非零，Actor 更新期间 WM/D 参数版本及 `.grad is None` 审计通过。另有 finite-difference/state gradient 和三步 frozen-WM BPTT CPU 测试。小网络 synthetic fixture 只用于证明链路，不是物理预测精度证据。

梯度很小且早期 bootstrap 梯度更大；WM 动作导数、Critic bootstrap 的正确性仍不能由梯度非零证明。不据此修改核心 BPTT。

### Tests run (command, pass/fail, actual evidence)

在 repo 根目录使用现有 Isaac Lab venv；无需安装或改其他工作树：

```bash
export S54_PYTHON=/mnt/sage_data/workspace/IsaacLab_3_0/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/s54_matplotlib
export WARP_CACHE_PATH="$PWD/.cache/warp"

$S54_PYTHON tools/prepare_s54_motion_prior.py --dry-run
# 复跑时选择新的输出目录；已有 verified 目录受保护
$S54_PYTHON tools/prepare_s54_motion_prior.py --out results/s54/motion_prior_stage0_repeat
$S54_PYTHON tools/test_motion_prior_transfer.py
$S54_PYTHON tools/test_foot_trajectory.py --out results/s54/motion_prior_trajectory_repeat.json
$S54_PYTHON tools/test_foot_tracking.py --out results/s54/motion_prior_foot_mdp_repeat.json
PYTHONPATH=tools $S54_PYTHON -m unittest tools.test_curriculum tools.test_curriculum_evaluation tools.test_curriculum_replay tools.test_standing_diagnostics -v
PYTHONPATH=tools $S54_PYTHON -m unittest tools.test_s54_launcher tools.test_s54_control_guards tools.test_curriculum_logging tools.test_s54_audit_fixes -v
$S54_PYTHON tools/test_env_s54.py --num_envs 16 --steps 60 --out results/s54/motion_prior_newton_16env_repeat.json
```

| 检查 | 实际结果 / 证据 |
|---|---|
| 新 CPU 迁移/配对/梯度/prior/合同/课程测试 | **21 PASS**，`results/s54/motion_prior_validation/motion_prior_cpu_final_contract.log` |
| 原 trajectory/Goal/Replay/net | **13 PASS**，validation/trajectory.json；其中旧兼容集成保留414D混合Critic检查，新独立Foot365D由新测试及FootMDP验证 |
| 原独立 Foot MDP | **7 PASS**，validation/foot_mdp.json |
| 真实课程门控集成 | **PASS**，motion_prior_curriculum_gate_integration.json；98步物理失败、WM8 Pitch3.286°，课程正确保持stage0；模型参数不变、评估协议恢复 |
| 原 curriculum/evaluation/replay/standing diagnostics | **30 测试执行 PASS**；其中重复发现9个已导入 CurriculumTests，不冒充30个独立用例；保存 validation/curriculum_replay_diagnostics.log |
| launcher/control/logging/audit | **35 PASS / 1 FAIL**，validation/launcher_control_logging.log；见下述已有失败 |
| Newton 1 env | Motion/paired/Foot 独立评估均完成，保留真实状态和实际接触 |
| Newton 16 env ×60步 | **PASS**，0 failed；两脚均真实触地，峰值832 contacts；FK最大位置误差5.60e-7m、姿态3.33e-7rad；见 motion_prior_newton_16env.json |
| Goal / Reward sync | smoke 训练 Goal 最大约4e-9；真实/replay reward 在浮点容差内；独立 G paired reward sync 1.23e-7 |
| 同任务实际 resume | C 12→24轮 **PASS**，新 run，184步本阶段+92上游；恢复 WM-only 时 Actor 不更新 |
| 错误跨任务 resume、错型号/shape/hist/physical/task | CPU **PASS**（正确拒绝） |
| Actor-only 导出 / 同镜头视频 | **PASS**，TS 可变 batch 与 eager 对齐；MP4 50帧、25FPS |
| git diff --check | **PASS** |

已有失败是 `tools.test_s54_launcher.SelectionAndSyncTests.test_failed_training_still_uploads_final_rows_then_finishes_crashed`：测试要求子进程 exit7 时抛 RuntimeError，当前 `sync_swanlab.py` 已改为记录失败并结束同步，不抛异常。两个文件工作树内容与 HEAD 完全相同，均非本轮修改；没有借本任务重写云日志行为。此前直接运行部分 unittest 曾因旧测试的模块搜索路径报 import error，使用 `PYTHONPATH=tools` 后通过。无在线 SwanLab 登录或上传，本地 CSV/JSON 可供现有同步工具后传。

### Smoke evaluation / standing gates

所有训练 smoke 均2环境、hist10、H8、12轮（先6轮 WM-only）、seed54、训练进程wall预算240s，写入独立新目录。wall按迭代边界软停止；启动器另有90s超时宽限与最多45s终止保存期，不能把240s称为包含保存的绝对硬时限；本轮实际每次均约15–20s结束。每轮 collector 受 UTD 调度，总共 **92条真实 transition**，不是“12次 Actor 更新”；Actor 实际只在后6轮更新。一个环境连续窗口不足1s，因此 `ep_len=0` 表示该日志窗口没有完成 episode，不表示无限存活。

| Run（均 `logs/s54_mp_*_20261104`） | 当前阶段 / 上游物理步 | checkpoint wall(s) | 最后 WM loss | Actor GN |
|---|---:|---:|---:|---:|
| stage1_smoke | 92 / 0 | 15.86 | 4.2997 | .290 |
| stage2_A_smoke | 92 / 0 | 19.09 | .5344 | 4.40e-5 |
| stage2_B_verified | 92 / 92 | 15.07 | .0266 | 2.96e-4 |
| stage2_C_smoke | 92 / 92 | 14.30 | .0271 | 2.94e-4 |
| bridge_verified | 92 / 92 | 14.24 | 2.8632 | 2.24e-4 |
| stage2_F_verified | 92 / 184 | 17.67 | .0223 | 3.14e-4 |
| stage2_G_walking_smoke | 92 / 92 | 17.04 | .0271 | 3.13e-4 |
| stage2_C_resume_smoke | 184 / 92 | 26.71（累计） | .0191 | 0（恢复校准） |

详细 loss/reward/梯度、上游 wall/supervision_steps、资源见 [run inventory](../../results/s54/motion_prior_run_inventory.json)；[训练曲线](../../results/s54/motion_prior_training_curves.png)。这些 smoke 没有单独 Action-MSE 监督步；可选 latent 默认0。两阶段数据分布、reward 不同，WM normalized loss 及训练 imag/sim 的非配对 batch 均值不用于推断迁移增益或模型利用。

计数口径：checkpoint 的 `supervision_steps` 只计可选 latent 样本，本轮为0；**Motion Reward 本身仍有参考监督**。本轮 Stage1 和 verified Bridge 各6次 Actor 更新，各使用 `6×8×4=192` 条 imagined Motion-Reward transitions，inventory 单独记录，不能称 Bridge 无监督。其他12轮Foot run各192条 imagined Foot transitions；恢复run为0。它们是重复利用 Replay 起点的 BPTT 计算量，不是额外真实物理样本。上述独立评估、16env preflight 等列出的额外验证成本共2708条真实 transition，详见 validation/evaluation_costs.json；与训练样本量分开统计。

全局预算另见 validation/training_costs.json：表中主run实际新增736条真实训练transition（resume计数184包含此前C的92，去重）；旧探索Bridge/F另消耗184条，共列出920条训练transition。按单组比较时仍计入其所需Stage1/Bridge上游成本，共享来源在全局预算只计一次。探索成本没有隐藏成“零代价迁移”。

真实设备 **RTX 4060 Laptop GPU，约7806MiB**；F 实测 Torch峰值701MiB、reserved712MiB、设备checkpoint快照1003MiB（含Warp）。不是3090测量。旧探索 Bridge/F run 保留但不作为最终对照：Bridge 曾额外构建 Policy 消耗 RNG，现已修正并测试相同seed E_f一致；早期 F 在无 stand 模板时仍放宽理论奖励上界，现已修正为 baseline 并重新跑 verified F。原奖励值/物理未被改动。

独立真实评估均无动作噪声；`reset_noise` 只影响初态，种子54/55/56独立重置。同种子不同duration共用一组试验，不能当9个独立种子。

| 对照 | 2s | 5s | 10s | 判断 |
|---|---|---|---|---|
| 旧 Standing final，nominal | 第96步 height | 原报告已有 | 原报告已有 | 成功复现原失效 |
| A random，nominal | 第97步 height | 未测 | 未测 | 非稳定站立 |
| 默认站姿零Action PD，nominal | 第97步 height | 未测 | 未测 | 实际对照复现；无主动平衡 |
| C transfer，nominal | 第97步 height | — | — | 与A无稳定性改善证据 |
| C transfer，独立reset 54/55/56 | 100 timeout /100 height /87 height | 101/100/87 全height | 101/100/87 全height | 5/10s成功率0/3 |

A/C nominal 初始 `state71` 逐元素差0；足部RMSE .01565/.01535m，最大Pitch45.56/45.17°，根高度下降 .271/.265m。实际两脚多数仍有受力、loaded slip .0317/.0324m/s；**脚位置误差小不能证明基座平衡**。与旧诊断一样是后倾后高度越界，不是单凭高 reward 判定成功。

Standing Gate 预先设置：2/5/10s、至少3个独立扰动reset，全部正常 timeout；pitch/roll<10°、高度偏差<.05m、足位置RMSE<.035m、朝向RMSE<10°、实际承载足滑<.02m/s、足承载样本覆盖≥80%；WM8高度<.01m、pitch<1°。C **FAIL**；一条2s timeout 的最大Pitch43°也不满足姿态门槛。实际接触是 Newton 最后物理子步 solver contact/force，可确认承载与方向，不等于整控制周期累计接触冲量。

匹配窗口 WM 对照来自同一 Newton 起点、同动作、同 ξ、同命令/tick；不重采样目标。以下为 C nominal 8步：高度RMSE **.03594m**、Pitch **4.112°**、足FK **.00967m**、线速度 **.1555m/s**、角速度 **.1058rad/s**、关节q **.01304rad**、qd **.1412rad/s**。高度bias +.01453m；paired simulation/imagined reward **.849855/.867675**、bias **+.017820**。1/4/8步、分时段及逐关节完整结果在相应 `summary.json`。C扰动reset最差高度 .03970m、Pitch4.211°；后期状态更难预测，足以阻止加难，尚不能证明 Actor 主动利用了模型。

其他真实检查：Stage1 dance holdout2s timeout，但足RMSE .02848m、upper pose RMSE .6593rad，WM8高度 .05219m/Pitch7.939°，不是成熟 Teacher。Bridge train-walk2s timeout，但WM8高度 .03947m/Pitch7.946°、paired reward bias −.09084，未证明有效蒸馏。G walking2s timeout、足RMSE .04225m、速度误差RMSE .1244m/s、contact accuracy .595、Pitch43.32°；匹配上身pose RMSE .3704rad、速度误差 .4270rad/s，不能称为自然行走。

最终 G 评估补齐命令适应指标并在新目录重跑：计划平均前进速度 **+.04156m/s**，实际 **−.35243m/s**（受后倾跌落主导）；计划平均落脚频率 **1.0336Hz**、前进落脚间隔 **.04106m**；窗口内1次计划触地，触地时位置误差 **.04065m**。受力记录出现1个承载变化沿，仅作诊断，不能等同完成一步。见 `results/s54/motion_prior_G_walking_verified_eval/summary.json`；turning 命令适应仍未运行。

证据目录：`results/s54/motion_prior_stage2_C_eval/`、`..._A_nominal_eval/`、`..._C_nominal_eval/`、`motion_prior_stage1_holdout_eval/`、`motion_prior_bridge_verified_eval/`、`motion_prior_G_walking_verified_eval/`、`motion_prior_default_pd_eval/`。状态、q/qd、Action、PD Target、reference/command、时间、真实接触与force均保存在 NPZ，图为 PNG。[同镜头全身视频](../../results/s54/motion_prior_same_camera.mp4)回放真实记录状态；仅渲染使用 MuJoCo，不重做物理仿真；末帧保持在画面中明确标注。

### Ablation run commands

以下默认只打印，不训练；实际训练必须显式 `--execute`。所有 run 必须是新名字，已有目录会拒绝覆盖。脚本无云上传，支持 SIGINT/SIGTERM 保存完整 checkpoint；可在 tmux 中运行同一有限命令，不必依赖 Docker。

```bash
# Stage0、Stage1/直接Stage2/可选Bridge
$S54_PYTHON tools/run_s54_motion_prior.py --stage stage1 --run s54_mp_stage1_new --dry-run
S1=logs/s54_mp_stage1_smoke_20261104/last.pt
BR=logs/s54_mp_bridge_verified_20261104/last.pt
$S54_PYTHON tools/run_s54_motion_prior.py --stage stage2 --group C --source "$S1" --run s54_mp_direct_new --dry-run
$S54_PYTHON tools/run_s54_motion_prior.py --stage bridge --source "$S1" --run s54_mp_bridge_new --dry-run
$S54_PYTHON tools/run_s54_motion_prior.py --stage stage2 --group D --source "$BR" --run s54_mp_bridge_to_foot_new --dry-run
$S54_PYTHON tools/run_s54_motion_prior.py --stage stage2 --group G --task walking --source "$S1" --run s54_mp_prior_new --dry-run
$S54_PYTHON tools/run_s54_motion_prior.py --stage stage2 --group C --task curriculum --source "$S1" --run s54_mp_gated_new --dry-run

# ABCDFG × seeds54/55/56，完全相同的Stage2分布/预算/eval制度，生成而不运行
$S54_PYTHON tools/ablate_s54_motion_prior.py --stage1 "$S1" --bridge "$BR" --out results/s54/motion_prior_ablation_repeat.json
# 已生成：results/s54/motion_prior_ablation_verified_commands.json（18条DRY_RUN_ONLY）

# 真正执行一个受限smoke：同一命令将 --dry-run 换成 --execute，至多12轮/240s
# 同任务恢复到总24轮，新目录；重新填充Replay，恢复期只更新WM
$S54_PYTHON tools/run_s54_motion_prior.py --stage stage2 --resume logs/s54_mp_stage2_C_smoke_20261104/last.pt --resume_iterations 24 --run s54_mp_resume_new --dry-run

# 独立Newton评估（无优化器）
$S54_PYTHON tools/evaluate_s54_motion_prior.py --checkpoint logs/s54_mp_stage2_C_smoke_20261104/last.pt --out results/s54/motion_prior_eval_repeat --durations 2 5 10 --seeds 54 55 56 --reset_noise
$S54_PYTHON tools/evaluate_s54_motion_prior.py --checkpoint "$S1" --manifest results/s54/motion_prior_stage0_verified/manifest.json --split heldout --out results/s54/motion_prior_holdout_repeat --durations 2 --seeds 54
$S54_PYTHON tools/test_motion_prior_newton.py --checkpoint logs/s54_mp_stage2_C_smoke_20261104/last.pt --out results/s54/motion_prior_gate_repeat.json

# 仅Actor导出；输出文件须为新文件
$S54_PYTHON tools/export_s54_foot_actor.py --checkpoint logs/s54_mp_stage2_C_smoke_20261104/last.pt --out results/s54/motion_prior_actor_repeat.ts
```

默认12轮 profile 是接口验证预算；ABCDFG 公平对照仍需要合格 Stage1、完整相同评估与多训练seed。`--task standing/walking/curriculum` 区分任务，不能用本轮G walking与C standing直接估计先验增益。课程保守步长2–6cm、较长摆动/双支撑；双门控失败不升级。300s launcher 只作有界验证，原课程完整升级还要求 full eval 的时间配额，不保证在5分钟内发生任何升级。

### Unrun items and blockers

- **NOT RUN**：正式长训、RTX3090资源标定、多训练种子完整消融、独立 heldout walking、turning/启动停止闭环、joint Motion/Foot 同时更新共享 D。E Action-MSE 为可选未实现；latent非零训练未跑（默认0）。
- **FAIL（能力）**：Standing 稳定性/姿态/WM门槛，Stage1 Teacher 质量；当前不能判断迁移或prior改善效果。可微链路通过不等于模型接触/动作导数准确。
- **数据缺口**：没有可靠 standing/turn/slowwalk/startstop 模板；Standing prior 正确关闭，F standing 实际不能评估动态prior效果。需要更多合格S54片段，不能用同walk的相邻窗口充当holdout。
- **已有回归失败**：一项 SwanLab exit7 异常预期与现有实现不一致，记录但未修改非本任务文件。
- **工程限制**：Replay不存checkpoint；恢复会额外花费真实采样。探索run未全部保存dirty源码快照，新的run已补齐。网络/云同步未测试，实验文件保存在本地。

### Next bounded experiment

下一步先做**只更新 WM 的 Standing calibration**，检查后倾/接触状态的分布覆盖和同窗口1/4/8步误差；不先提高 prior 权重或放宽站立终止。已提供 `calibration_bounded.json`：16env、最多80轮、WM-only80、hist10、H8、300s上限，Actor/Critic不更新。**仅dry-run验证，未执行此calibration**；目标硬件仍需实测，不声称3090吞吐。

```bash
$S54_PYTHON tools/run_s54_motion_prior.py --stage stage2 --group C --source "$S1" --profile configs/motion_prior/calibration_bounded.json --run s54_mp_calibration_new --dry-run
# 需要进行该次有界实验时显式 --execute；结束后仍用上述2/5/10s与paired WM评估。
```

应覆盖>1.2s后倾、接触边缘和有限动作扰动；若误差仍高，先检查足/踝接触、执行器及动作导数，勿以normalized WM loss通过代替物理误差通过。另需在有限 Motion pilot 中得到真正可跟踪的 Stage1，再对相同Foot分布比较C/D/F；本轮 smoke 权重可用于接口校验，不能当成熟Teacher。只有真实站立与姿态门槛通过后再讨论正式训练配置，本记录不列出长训步骤。
