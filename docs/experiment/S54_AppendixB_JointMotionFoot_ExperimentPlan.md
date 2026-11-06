# S54 × SuperMimic：Appendix B 风格 Motion–Foot 联合训练实验方案

> **实验代号**：`S54_JOINT_AB1`　｜　**状态**：设计与预注册；**尚未实施、未获得任何联合训练结果**  
> **目标平台**：RTX 3090 24 GiB，Isaac Lab 3.0 + Newton/MJWarp，S54 29 DoF  
> **训练时间预算**：单次独立实验**最多 8 小时墙钟时间**（以实际资源/守护进程为准），保存可恢复检查点。  
> **原论文**：[SuperMimic, arXiv:2610.28480](https://arxiv.org/pdf/2610.28480)，尤其 Appendix B（PDF 第 13–14 页）、Appendix G.1（PDF 第 18–19 页）、Algorithm A.1。  
> **性质说明**：论文示例的下游接口是**速度命令**；此方案把它扩展到 **S54 的 230D Foot-trajectory Goal**，是**受论文启发的研究扩展**，不是论文已经证明有效的足端实现。本文中的学习率、时长、验收门槛都是**工程预设值**，并非论文报告的最优参数。

## 0. 核心问题与既有证据

研究假设：维持一条 LAFAN1 的 Motion Tracking 支路持续训练共享动作 Decoder，可防止 Decoder 失去已学会的自然全身运动分布，并让 Foot Encoder 逐步学会调用这套动作映射，最终提高足端轨迹控制的稳定性、自然性及样本效率。

当前已完成、应保持原样的历史实验：

- Stage 1 Teacher：`logs/s54_stage1_walk_20261105_014722/ckpt_004772.pt`；与本实验的原始来源保持内容哈希一致（用户报告 SHA256：`1cf556f384cefa8635975ca4f8cb8c0e144a98292c5fb2fb8cff33e9b07c3fc4`）。完整训练 Walk Clip 的 `norm_return=0.820197`，完整 261.3s 无物理终止。其世界坐标 Root XY RMSE 仍达 2.33m：不可称为全局路径跟踪已解决。
- 既有 A/B/C 每组 2000 iterations、1000 次 Actor 更新、5,251,072 条真实转移。**三组都冻结 Decoder**；A/B 冻结随机初始化的小输出层 Decoder，控制动作 RMS 约 `2.9e-4`，无法作为完整从零 Actor 基线；C 冻结 Teacher Decoder，但 Foot Encoder 随机开始，平均仅存活约 1.18s。A/B 平均约 1.92s，三组 10s Survival 均为 `0/3`。
- 现有训练模块已支持单接口 `motion`、`foot_trajectory`、`reference_foot`，但**不等于已经有真正双接口联合训练器**。不得通过把两个进程分别训练或每轮相互复制 Decoder 参数，冒充论文式共享参数联合更新。

**本实验与原 A/B/C 物理隔离**：不覆盖、不复用输出目录，不接续旧 A/B/C 的 optimizer、Replay、Foot Policy；仅从**只读 Stage 1 Teacher**初始化联合模型。旧 A/B/C 的结果只作为历史失败模式和有限对照，不直接用不同训练/监督预算声称严格优劣。

## 1. 原论文 Appendix B：准确对应关系

论文的每个目标接口 `i` 都有独立的 Goal Encoder `E_i` 与 Critic `V_i`，共享同一个 Decoder `D` 与 World Model `f`。每个接口定义自己的 Goal、Reward、Termination，更新：

- **Motion Tracking batch**：更新 Motion Encoder `E_m` 和 **Shared Decoder `D`**；更新 Motion Critic `V_m`。
- **Downstream batch**：下游奖励的梯度经过 `D` 传到下游 Encoder `E_f`，**但不更新 `D` 的参数**；更新 Foot Critic `V_f`。
- **共享 WM**：真实 Motion 和 Foot 的转移都训练同一个 `f_ω`；在 Actor BPTT 阶段，WM 权重固定，但须保留从未来状态回到 Action 的可微梯度。
- **Appendix G.1 的具体实现**：Motion + Velocity 的模拟环境与 imagined batch 各占一半，联合训练 100,000 iterations。论文该实验的两个接口在各自实验中从零开始联合训练；**本方案改为从已合格的 Stage 1 Teacher warm-start，再联合训练**，也是一项工程差异。这里还把 Velocity 改为 Foot；**不是**直接照搬 G.1 的速度命令范围、100k iterations 或 H100 计算时间。
- **下游目标的两种方法**：Distillation 是用参考动作的 **Motion Tracking Reward 经 WM/BPTT** 训练下游 Encoder，**不是 Teacher Action MSE**；RL 是使用下游自己的任务 Reward。**本夜主实验选择 Foot RL**。合法成对 reference 的 Latent Alignment 可另设实验，默认 `0`，避免混淆。

> 特别注意：“Downstream 冻结 Decoder 权重”不是把 Decoder 整个前向包在 `torch.no_grad()` 中。它仍是一个对输入 `z_f` 可微的函数。此实验还需保证在 Motion 更新时 Decoder 的权重真正变化，并只由 Motion Loss 驱动。

## 2. 数学定义、变量及维度

### 2.1 接口、状态、目标及控制

令目标接口集合 `I={m,f}`，其中 `m` 表示 LAFAN1 Motion Tracking、`f` 表示 Quintic Foot Tracking，接口采样概率 `p(m)=p(f)=0.5`。

机器人状态：

$$
S_t=(\mathbf p_t,\mathbf R_t,\mathbf v_t,\boldsymbol\omega_t,\mathbf q_t,\dot{\mathbf q}_t).
$$

- `t`：离散控制时间索引，`Δt=0.02 s`；`S_t`：机器人真实或预测状态。
- `p_t∈ℝ³`：Root 平移；`R_t∈SO(3)`：Root 旋转；`v_t∈ℝ³`：Root 线速度；`ω_t∈ℝ³`：Root 角速度。
- `q_t, qdot_t∈ℝ²⁹`：S54 29 个关节位置和速度；状态序列化 71D（旋转用四元数 4D）。
- `ξ∈ℝ³³`：摩擦、执行器偏置、质心等仿真上下文；Action `a_t∈ℝ²⁹` 为 PD 目标偏移控制量，不是力矩。
- `h_t^i∈ℝ^{10×99}`：接口 `i` 的 10 帧本体观测历史；Foot 与 Motion 最后 6D 的参考航向语义**不同**，必须使用可审计的观测 adapter 和接口级 norm，不可静默混用。
- `g_t^m∈ℝ^{730}`：10×73D LAFAN1 全身未来运动参考（步距 0.02s）；`g_t^f∈ℝ^{230}`：10×23D 足端未来目标（步距 0.1s）。Foot Goal 不携带上身 Motion Reference。

每个接口的策略定义：

$$
\mathbf z_t^i=\tanh\big(E_{\theta_i}(\operatorname{Norm}_{g,i}(\mathbf g_t^i))\big)\in[-1,1]^{32},
\qquad
\mathbf a_t^i=\operatorname{clip}\left(D_{\phi}(\operatorname{Adapt}_i(\mathbf h_t^i),\mathbf z_t^i),a_{\min},a_{\max}\right).
$$

- `E_{θ_i}`：第 `i` 条目标支路的 Encoder；`θ_m/θ_f`：两套**独立** Encoder 的可训练参数。
- `Norm_{g,i}`：各 Goal 空间独立统计归一化；`Adapt_i`：本体观测的接口语义适配（含合法的 Obs Norm），必须不改变原始物理含义。
- `D_φ`：**唯一的一份** Shared Decoder MLP，参数为 `φ`；输入 10×99 的观测历史与 32D Latent，输出 29D Action。不要创建两份失去共享语义的 Decoder。
- `a_min,a_max`：原代码的动作裁剪阈值；必须从现有控制合同读取，不能随意修改物理动作范围。

实际施加的关节 PD 目标：

$$
\mathbf q^{\mathrm{PD}}_{t}=\mathbf q_{\mathrm{default}}+\boldsymbol\xi_{\mathrm{joint}}+\boldsymbol\alpha\odot\mathbf a_t^i.
$$

- `q_default∈ℝ²⁹`：默认关节姿态；`ξ_joint∈ℝ²⁹`：关节偏置；`α∈ℝ²⁹`：逐关节动作缩放；`⊙`：逐元素乘法。

### 2.2 一个共享 World Model

$$
\hat S_{t+1}=f_\omega(\hat S_t,\mathbf a_t^i,\boldsymbol\xi),\qquad i\in\{m,f\}.
$$

- `f_ω`：共享可微动力学模型，`ω` 为参数；输入为 71D 状态不变量特征 + 29D 动作 + 33D 上下文，共 133D；网络输出 70D 状态增量，随后由现有 SO(3) 积分得到 `Ŝ_{t+1}`。
- 上标 `^`：WM 想象的预测状态；无上标 `S`：Newton 真值；**Goal 不进入 WM 动力学输入**。
- `B_m`、`B_f`：两接口独立 Replay；保存真实 `(S_t,a_t^{executed},ξ,S_{t+1})` 及相应时钟、Goal/命令/参考身份；噪声执行后的 Action 必须与存入 Replay 的动作一致。

沿用项目现有 WM 损失，使用 `k=8` 的多步自回归预测：

$$
\mathcal L_{WM}(\omega)=\frac{1}{|\mathcal B|kD_s}
\sum_{(S,a,\xi)\in\mathcal B}
\sum_{j=1}^{k}m_j
\sum_{d=1}^{D_s}\left[
\frac{e_d(\hat S_{t+j},S_{t+j})}{\max(\sigma_d,\epsilon_d)}
\right]^2,
\quad D_s=70.
$$

- `𝔅`：从 `B_m` 和 `B_f` **按预定混合比例**采样的 WM 窗口批次，不能跨环境 reset、不能跨接口、不能越过 Clip/命令时钟失效位置。
- `k`：WM 监督窗口长度，建议仍为 8；`j`：窗口内未来步索引。
- `m_j∈{0,1}`：有效预测步掩码；`D_s=70`：状态误差维度。
- `e_d`：第 `d` 个局部状态差（含 Root 局部平移、SO(3) 对数旋转差、局部速度差、关节 q/qd 差）；`σ_d`：WM Output RunningNorm 标准差；`ε_d`：对应数值下限。
- `ω` **只由 WM Loss 更新**；Actor Update 必须冻结 `ω`，但 `∂f_ω/∂a` 仍须可微。

> 这是对本仓库 `wm_loss` 的表达，不要与论文 Eq.(1) 的抽象多头 loss 完全混为一谈：论文的 reward/observation/termination head 可以是解析的、因而不另设学习损失。本项目当前主要训练 70D 动力学增量，Reward/Observation/Termination 按现有任务实现计算。

### 2.3 两个任务的 BPTT Return

对每个接口 `i∈{m,f}`，从对应 Replay 的合法起始状态 `Ŝ_0` 执行 `H=32` 步想象：

$$
\hat S_{t+1}=f_{\omega}(\hat S_t,\pi_i(\hat h_t^i,g_t^i),\xi),
\qquad \hat c_0=1,\quad
\hat c_{t+1}=\hat c_t(1-\hat d_{t+1}^i).
$$

- `π_i`：接口策略 `D_φ∘E_{θ_i}`；`ĥ_t^i`：由预测状态构造的接口观测历史。
- `ĉ_t∈[0,1]`：可微的累计存活权重；`d̂_{t+1}^i∈[0,1]`：该任务在预测状态上的 soft termination，来自现有任务公式；真实物理终止仍是硬判据。
- `r̂_{t+1}^i`：第 `i` 条接口在想象的下一状态计算的单步 Reward；Motion 与 Foot 的**奖励不同**，不能用其原始绝对数值直接排名。

每条接口的优化目标：

$$
\hat J_i(\theta_i,\phi)=
\mathbb E_{\hat S_0\sim\mathcal B_i}
\left[
\sum_{t=0}^{H-1}\gamma^t\hat c_t\hat r_{t+1}^i
+\gamma^H\hat c_HV_{\psi_i}(\hat S_H,g_H^i,\xi)
\right].
$$

- `γ=0.99`：折扣因子；`H=32`：BPTT 想象长度；`V_{ψ_i}`：第 `i` 个接口的 Critic，其参数 `ψ_i` 彼此独立；`g_H^i`：第 H 步的接口 Goal。
- `L_m=-Ĵ_m`、`L_f=-Ĵ_f`：Motion 与 Foot Actor 的最小化损失。
- Motion 使用原始 LAFAN1 Motion Tracking Reward；Foot 使用原始五次项 Foot Tracking + 平衡奖励。初始实验 `prior_config=None`，`paired_latent_weight=0`，不叠加新的 Upper Prior、不使用 Teacher Action MSE。

训练时的**参数更新规则**（实验最关键的可证伪约束）：

$$
\begin{aligned}
\theta_m&\leftarrow\theta_m+\eta_m\nabla_{\theta_m}\hat J_m,\\
\phi&\leftarrow\phi+\eta_D\nabla_{\phi}\hat J_m,\\
\theta_f&\leftarrow\theta_f+\eta_f\nabla_{\theta_f}\hat J_f,\\
\color{gray}{\nabla_{\phi}\hat J_f}&\color{gray}{\text{ 不用于更新 }\phi},\\
\omega&\leftarrow\omega-\eta_{WM}\nabla_\omega\mathcal L_{WM}.
\end{aligned}
$$

- `η_m`：Motion Encoder 学习率；`η_D`：Shared Decoder **仅 Motion 分支**学习率；`η_f`：Foot Encoder 学习率；`η_WM`：WM 学习率。
- Foot 的梯度虽不更新 `φ`，但必须满足链式法则：

$$
\nabla_{\theta_f}\mathcal L_f
=\frac{\partial\mathcal L_f}{\partial\mathbf a}
\underbrace{\frac{\partial D_\phi}{\partial\mathbf z_f}}_{\text{必须保留}}
\frac{\partial E_{\theta_f}}{\partial\theta_f}.
$$

实现上最稳妥的是 **Foot 与 Motion Actor 使用分离的 Optimizer 参数组**：`opt_foot={θ_f}`、`opt_motion={θ_m,φ}`；对 Foot 支路不执行 `opt_motion.step()`。可使用已冻结权重的 `D` 正常可微前向，或 `torch.func.functional_call` 对 `D` 权重执行仅参数 detach 的函数式前向。避免共享 Adam 动量、weight decay 或 grad accumulation 将 Foot 梯度偷偷更新 Decoder；每次更新都检查 `.grad`、参数版本、前后哈希。

### 2.4 两个独立 Critic

$$
\mathcal L_{V_i}(\psi_i)=
\mathbb E\left[(V_{\psi_i}(\hat S_t,g_t^i,\xi)-\operatorname{sg}(\hat G_t^{\lambda,i}))^2\right],
$$

$$
\hat G_t^{\lambda,i}=
\hat r_{t+1}^i+\gamma(1-\hat d_{t+1}^i)
\left[(1-\lambda)V_{\psi_i}(\hat S_{t+1},g_{t+1}^i,\xi)+\lambda\hat G_{t+1}^{\lambda,i}\right].
$$

- `ψ_m,ψ_f`：两套 Critic 参数；`Ĝ_t^{λ,i}`：第 `i` 条想象轨迹的递推回报目标；`λ=0.95`：GAE/λ-return 混合系数；`sg`：停止梯度；初始化终点 `Ĝ_H^{λ,i}=V_{ψ_i}(Ŝ_H,g_H^i,ξ)`。
- Motion Critic 输入现有任务合同约 914D；Foot Critic 输入约 365D；**不跨任务复制 Critic**，不能共用 Norm 或真实 Reward Scale。

### 2.5 联合目标的精确含义

论文中的形式为 `max E_{i~p(I)} Ĵ_i`。本实验的联合采样概率取 0.5/0.5；**训练期间并不直接把两个任务的不同 Reward 数值相加后共同更新所有参数**：Motion Return 决定 Decoder 和 Motion Encoder 的更新；Foot Return 只决定 Foot Encoder 的更新。WM 则接收两任务的真实转移。

对可选合法配对样本才可单独定义：

$$
\mathcal L_{align}=\lambda_z\left\|E_{\theta_f}(g_f^{\rm paired})-\operatorname{sg}(E_{\theta_m}(g_m^{\rm paired}))\right\|^2.
$$

其中 `g_f^paired,g_m^paired` 必须出自同一 S54 clip 与时间戳，`λ_z` 为配对对齐系数。**本夜固定 `λ_z=0`**；论文 G.1 的速度接口使用 0.1，但不能未经检查就搬到任意 Quintic Foot Goal 上。

## 3. 训练结构与防止遗忘

```text
                   LAFAN1 Walking Full-body Goal (730D)
                                  │
                             Motion Encoder θ_m ───┐
                                                   │ z_m(32)
                                                   ▼
    Observation Hist (10×99) ────────────────── Shared Decoder φ ──► Action(29)
                                                   ▲
                                                   │ z_f(32)
     Quintic Foot Goal (230D) ──► Foot Encoder θ_f ┘

    Motion Newton Rollout ──► Replay B_m ─┐
                                          ├──► 共享 WM f_ω ◄── WM optimizer
    Foot Newton Rollout ────► Replay B_f ─┘
                     
    Motion BPTT + Motion Critic V_m ─► 更新 θ_m 和 φ
    Foot   BPTT + Foot   Critic V_f ─► **只**更新 θ_f
    WM Actor rollout: ω 参数冻结，但 ∂f_ω/∂Action 可微
```

关键工程问题（必须由 Codex 解决并在报告中给证据）：

1. **共同 Decoder 及接口观测语义**：不能让 Motion Encoder 使用一种 ObsNorm、Foot Decoder 使用另一种未经校准的 ObsNorm 而不测试；当前 Foot 的 `mask_reference_orientation` 会置零每帧 `93:99`，而 Motion 仍使用参考朝向。选择“保留各接口明确 adapter + norm 并记录分布差异”或实现有测试的公共表示；第一晚以最小兼容改动优先，禁止隐形改写 Stage1 Teacher。
2. **不同 reset 和 episode 合同**：Motion 从 LAFAN1 bin 启动且 Clip End 可截断；Foot 从独立 Quintic 命令/默认 S54 状态启动且有 500 步上限。收集器可以在同一个 Newton Scene 中分任务环境，也可以有两个独立 Scene，但必须验证真实 env 数、独立状态、reset、rollout 时钟和 Replay ID；不要假装一条任务的 episode 时钟属于另一条任务。
3. **Replay 分离与共享 WM**：Motion 与 Foot 的 Goals/Commands 类型不同；建议 `ReplayBuffer_m` 与 `ReplayBuffer_f` 分开，各自采样有效 k-step 窗口供 WM 混合，不能把两种命令结构强行塞入同一 Foot 缓冲。WM 更新使用独立状态/动作/ξ，不依赖 Goal 维度。
4. **共享 Decoder 动态性**：若在 Motion 更新后 `φ` 变化，下一次 Foot Encoder 更新应使用新的 D，不能缓存过期图或相互覆盖两份 Decoder state_dict。训练时定期审计 Motion 更新前后 `Δφ>0`、Foot 更新前后 `Δφ=0`、Foot Encoder 更新 `Δθ_f>0`，并验证 Foot loss 关于 `z_f` 的梯度非零有限。
5. **防止 Motion 技能遗忘**：在固定 Walking Clip/片段上定期评估 Motion NR、Body Pose RMSE、Root XY drift；一旦 Teacher 相对 Tracking 退化明显，先保存/标记退化 Checkpoint，不能只看 Foot reward 提升而忽略 Motion 能力丢失。Motion 与 Foot Reward 刻度独立记录，绝对不混到一个平均 `episode_return`。
6. **非匹配先验**：没有独立 Stand/Turn 模板，不能让任意 Quintic Command 对应随机 LAFAN1 的上肢姿态。第一夜不要启用动态 Upper Prior，也不要假造 Walking holdout。

## 4. 单张 3090 的首夜配置：预注册建议

以下为**目标配置及资源受限回退规则**，不是已经通过性能测试的数值。首夜最多 8h，`max_iterations` 是**安全上限**而非保证训练达到的次数。

| 参数 | 建议默认值 | 说明与允许调整 |
|---|---:|---|
| 训练方式 | `joint_motion_foot` | 新训练入口，**独立于 A/B/C 与原 Stage1** |
| 初始化 | Stage 1 `ckpt_004772.pt` | 只读加载 E_m、D、WM 和兼容 norm；Foot E、Foot Critic 全新；Motion Critic 优先严格兼容地载入，否则新建并声明 |
| 单次预算 | 8h wall, `max_iterations=15000` | 到任一界限即**安全保存并退出**，不能承诺 15k 必达；正式启动时记录实际硬件吞吐 |
| 真实并行数 | 初测 **512 Motion + 512 Foot** | 总 1024 env，先测显存、Warp 和接触；不通过则整次固定为 256+256，**不要在同一实验 run 中悄悄改变** |
| Motion:Foot env 比例 | 1:1 | 两条任务的真实 transition、数据量分别计数 |
| imagined Batch | **128 Motion + 128 Foot** | 总 256 起始状态/iteration；保证两支都有真正 Actor 更新；若内存不足按总 batch 降级且快照记录 |
| `horizon` | 32（0.64s） | WM-H32 配对评估至少做一次；若 H32 发生明显数值不稳定或梯度问题，**停并开启新实验版本**，不能在原 run 中静默改 16 |
| `wm_window` | 8 | WM 监督时长 0.16s |
| `wm_batches`, `wm_batch` | 16 × 1024（每次来源 50/50） | 一次 iteration 总共 16 个 WM 更新，**非每个任务各 16 次**；有独立窗口掩码和 Loss 分项 |
| `warmup_steps` | 128 vector steps/环境（若适用） | 先验证初始化 Replay 有效；这是数据预热，不代表 Actor 已学习 |
| `wm_only_iters` | 200（建议） | 使用预训练 WM，仅短期新分布校准；对照原 Foot 的 1000；如果预热未过数值检查，不得直接跳过或无限加长 |
| `buffer_steps` | 1024/环境/任务 | 与总显存预算配合，按接口分别创建；OOM 在 smoke 阶段修正固定配置 |
| `utd` | 0.4 | 需要在新联合训练循环中审计具体真实 transition / WM 更新 / Actor 更新的对应关系 |
| `lr_WM` | 5e-5 | 先保持与已验证流程接近 |
| `lr_MotionEncoder` | 2e-5 | 保护已有 Teacher 行为；不是论文给出的 LR |
| `lr_SharedDecoder` | 2e-5 | **只由 Motion Loss 更新** |
| `lr_FootEncoder` | 1e-4 | 允许新的接口更快适应，需梯度审计 |
| `lr_Critic_m/f` | 1e-4 / 1e-4 | 分开的 Critic 和 Adam 状态；若载入 Motion Critic，需明确 optimizer 新建还是恢复 |
| `gamma`, `lambda` | 0.99 / 0.95 | 与当前 SuperMimic 风格一致 |
| Motion Reward | 原 Stage1 | 不在此 run 临时改权重 |
| Foot Reward | 原 Foot | 不在此 run 临时改权重、动作限位、终止阈值 |
| `paired_latent_weight` / Upper Prior | 0 / Off | 第一夜检验纯 Joint Motion + Foot RL，避免混淆额外监督 |
| Seed | 54（首夜） | 单种子实验，不能据此宣称统计显著 |
| AMP | `bf16`（若原 WM/Actor 可兼容） | 依实际 RTX3090 测试；归一化、积分、FK 和 Reward 留 FP32 |
| logging | 每 20 iter + 独立实时诊断 | SwanLab 本地优先；网络同步与主训练解耦 |
| checkpoint | 每 500 iter 或约 30 min，最终/错误/信号均保存 | 限制新增目录容量，保留明确 milestone、best-on-fixed-eval 和 last |
| Evaluation | Foot 每约 30–60 min；Motion 短探针每约 60 min；完整 Motion 至多启动/中段/末段几次 | Eval 耗时计入 8h 预算；不得每几十 iter 全片重评 261 秒 |

**重要：**512+512 与 128+128 并非论文实验的4096 env、4096 imagined batch；它们是为 RTX 3090 单卡预拟定的规模。原 ABC 的纯 Foot 1024 env、batch256 曾在3090上运行，但联合 Scene、双 Encoder/Critic 与混合 WM 会增加成本；必须以完整显存/吞吐 Smoke 证据最终决定。

## 5. 一晚训练的运行流程

### Phase 0：开训前自检（必须实际运行）

- 阅读仓库当前 `AGENTS.md`（若存在）、最新 `source_snapshot`、`ABC_SHORT_ABLATION_REPORT.md`、`MotionPrior_FootTrajectory.md`、Stage1 Teacher `args.json`/contract，核验 S54/物理/训练数据/关节顺序/WM/Goal合同。
- 校验 Stage1 Teacher SHA256、源文件不修改；确定 source checkpoint 的 ObsNorm/GoalNorm/Actor/WM/Critic 的真实映射；生成 `transfer_mapping.json`。
- 实现并运行 CPU 单测：同一 Shared Decoder identity； Motion 梯度仅更新 `E_m,D`；Foot 梯度仅更新 `E_f`，但 `∂L_f/∂z_f≠0`；WM 梯度只在监督环节更新；Critic 互不串梯度；不同 Goal/Command/Clock/Replay 不串线；接口 norm/adapter 正确。
- Newton 1/16 env 双支路各至少数十步；随后 256+256 env 小时长 smoke，实际测显存/吞吐、保存和同任务 Resume；只通过 smoke 才用目标 512+512。
- 旧代码禁止 `reference_foot + train_decoder` 并专门假设单接口冻结 D；需要新的 `joint_motion_foot` 代码路径，**不得简单绕过其严格合同检查**。

### Phase 1：联合训练主实验

1. 初始化共享 `WM,E_m,D` 自 Stage1、独立 `E_f,V_f`；Motion Critic 仅在严格同任务兼容时可载入（记录是否载入）。
2. Motion env 用 LAFAN1，Foot env 用原 Quintic 保守直行短步；两接口数据分别进 `B_m,B_f`。
3. 每轮按 50/50 抽取两个接口的 WM 数据，进行总计16个 WM更新；冻结 WM 权重进行两条接口的想象。
4. **Foot step**：Foot Reward/Foot Critic → 更新 `E_f`，Decoder 不变。
5. **Motion step**：Motion Reward/Motion Critic → 更新 `E_m,D`，Foot Encoder 不变。严格隔离优化器/梯度及正常更新顺序；每轮两种 actor 预算需与日志一致。
6. 保存每任务的 Newton transitions、WM samples、想象样本、Actor/Critic 更新数以及运行耗时；每隔预定间隔做固定协议真实 Newton 评估。

### Phase 2：自动安全收尾

- 采用 `tmux`；训练进程必须和 SSH 会话隔离。命令必须能生成独立 run_id 和唯一目录，并具有 `--dry-run`、`smoke`、`train`、`resume` 命令；不要将用户的旧日志作为新日志目录。
- 8h 为**终止条件**，而不是第二天继续在后台无限训练。用单进程或外部 watchdog 发信号触发已验证的完整 checkpoint 保存，保留最后安全退出记录；进程崩溃、非有限数、OOM 不得假报 completed。
- 本地优先保存 `train_log.csv`、`interface_metrics.jsonl`、`wm_paired.json`、`eval/*`、`args.json`、`source_snapshot`、`manifest/hash`、`stdout/stderr`、`training_status.json`、`last.pt`、`best_foot.pt`（仅固定真实评估有合法改善时）。独立 SwanLab 同步队列可失败重试，不阻塞训练。
- Resume 时要读取真实 Checkpoint 的 `WM,E_m,E_f,D,V_m,V_f`、各优化器、Norm、时钟、累计样本与 interface schedule，检查同一 Contract；若 Replay 不保存，明确重采集并进行 bounded WM 恢复。**新 run 的续训属于本联合实验，不允许 `--resume` 加载 A/B/C 的 Foot checkpoint。**

## 6. 验收口径、进展门槛与比较规则

### 6.1 必须常驻监控

**Motion 支路：**`norm_return_10s`、完整 Walking `norm_return`（稀疏）、14 body position/orientation RMSE、Root XY drift、Motion survival/ClipEnd、`E_m/D` 参数变化。基线为 Teacher 的 NR=0.820197；下表的保持门槛为工程预设值。

**Foot 支路：**2/5/10s Survival、`n_failed/stop_reasons`、Foot Position RMSE、Scheduled Touchdown Error、Root Height/Pitch/Roll、真实前进轴速度、Contact/Slip Proxy、上肢角速度/加速度与视频。必要时区分“跌倒前前缀的 RMSE 低”与“完整时长目标跟踪成功”；真实向后速度时不能报告成功前进行走。

**WM：**两接口分项 `wm_loss/wm_err8`，同一已记录 Newton 轨迹上的 1/4/8/16/32 步 Height、Pitch、Foot FK 物理 RMSE 与 Reward MAE；尤其检查 H32 与接触边缘，不用 imagined-sim 两条不同分布曲线相减代替配对误差。

**梯度：**`norm(E_f grad)`、Foot Loss 对 `z_f` 的 grad、Motion D 更新量、Foot D 更新量必须零、Actor Reward vs Bootstrap grad norm/cosine、Action RMS/clip fraction、NaN/Inf。

### 6.2 预注册工程判据（非论文结论）

| Gate | 初步判据 | 失败时行动 |
|---|---|---|
| Correctness | 全部合同/状态/梯度/Replay 单测 PASS；Motion step 确实更新 D；Foot step 不改 D，`E_f` 梯度有限非零 | **禁止正式主训**，先修工程问题 |
| Physics | Newton 双接口 16 env smoke 无无效状态/时间串线/不合理重置 | 禁止主训；修正后重跑 smoke |
| Stability | 不出现 NaN/Inf、持续梯度爆炸、不可恢复 OOM、写损坏 checkpoint | 安全停机并保留诊断，**不可自动改 reward/termination 以继续** |
| Motion retention | 同一固定参考/起点/时限协议下，NR 不应持续低于该协议的 Teacher 基线超过 `0.05` 绝对值。**只有在完整261.3秒评估协议中**，基线0.820197才对应约0.77；短10s探针须先测各自Teacher基线，不能混用 | 标记 drift；保存 last-good，继续与否记录为预注册停机策略，不假报 Foot 成功 |
| Foot progress | 与初始 C-frozen 及零动作对照，优先要求 2s 3/3、5s ≥2/3、最终 10s ≥2/3，并且实际前进轴速度不再明显为负 | 未达也可如实完成 8h 实验；不能称已收敛 |
| Foot quality | 在存活达标的样本上足端 RMSE 与自然性改善，且真实接触/滑移无严重退化 | 停留在候选策略，不进入 D/F/G 正式性能宣称 |
| Generalization | 当前没有独立 Walking Holdout，Dance 不代替 Walking | 标记 `NOT EVALUATED`，不要声称跨 Walking Clip 泛化 |

**成功完成 8h 工作流 ≠ 策略已成功学会行走。** 任何 Gate 状态必须基于真实数据；缺失就是 `NOT EVALUATED`。

### 6.3 与原 A/B/C 的公平性边界

- 当前联合实验多了**持续 Motion 训练的额外真实采样、Actor 更新和监督**；它不能与仅 2k iterations 的 C 组直接以相同 total wall time 或同样 foot samples 声称算法更优秀。
- 同时报告 `(1) Foot Newton transitions`、`(2) Motion Newton transitions`、`(3) 总真实 transitions`、`(4) 预训练 Stage1 transitions`、`(5) BPTT imagined steps`、`(6) wall-clock`。
- 实际评价先问“这种联合训练能不能让 Frozen Foot Decoder 的接口在 2/5/10s 站稳并迈步”；然后才设计相同算力预算的对照，例如“独立 Foot frozen D vs Joint Motion/Foot，保持 Foot 样本数一致”。

## 7. 文件隔离与实际交付（由服务器上的 Codex 落地）

唯一实验前缀推荐 `s54_joint_ab1_<UTC-or-local-timestamp>_seed54`；所有新文件限定在**新的**目录下：

```text
configs/motion_prior/joint_ab1/                 # 新实验的 immutable profile
supermimic/joint_*.py                           # 如需新代码，独立入口/运行时；不重写历史训练语义
tools/train/run_s54_joint_ab1.py                # dry-run/smoke/train/resume
logs/s54_joint_ab1_<run-id>/
   source_snapshot/ manifest.json args.json transfer_mapping.json
   train_log.csv interface_metrics.jsonl wm_pair_metrics.jsonl
   evaluation/ checkpoints/ stdout.log stderr.log training_status.json
   last.pt
results/s54/joint_ab1_<run-id>/                 # 新的诊断与图表
docs/experiment/S54_JOINT_AB1_RUN_REPORT.md    # 服务器实测完成后填写
```

**不得修改、清理、覆盖**已有 `logs/s54_abc_short_*`、`logs/s54_stage1_walk_*` 或既有 `results/s54/motion_prior_*`；Teacher 原文件只读。对原代码必须做必要修改时保持原 CLI 的无新 flag 行为不变，并执行回归测试。

## 8. 必须交付的运行报告框架

运行完成后，Codex 应输出 `docs/experiment/S54_JOINT_AB1_RUN_REPORT.md`：

1. 运行状态 `COMPLETED_BY_WALL_CAP` / `COMPLETED_BY_ITER_CAP` / `STOPPED_BY_GATE` / `CRASHED`，实际训练总时长、GPU/显存、源码SHA、Teacher哈希、任务/Goal合同；
2. 真实参数快照 vs 预注册提案，实际两接口 env、batch、WM更新、Actor/Critic更新、LR、seed；如果 smoke 选择小配置必须记录并固定；
3. 证据：同一 D 实例，Motion Update 改 D、Foot Update 不改 D，Foot梯度通过 D，WM分别接两任务数据，Critic独立；
4. 按接口分开的 Reward/Survival、Loss/梯度/WM 1/8/32 预测误差和真实接触；对 Foot 做固定种子/命令 Newton 2/5/10s 检验；
5. 与源 Stage1 Teacher 同协议比较 Motion NR 和对齐误差，评估自然性与遗忘；
6. 可复现命令、Checkpoint 列表、恢复程序与 SwanLab 同步状态、本地文件有效性；
7. 明确主要发现、限制和下一轮建议；不得声称已跑过原论文速度接口实验，也不得隐去 Stand/Turn 数据缺口。

## 9. 论文与历史资料

- Liao et al. (2026), **SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model**，Appendix B, C, G.1，https://arxiv.org/pdf/2610.28480 。
- 用户已完成的 `ABC_SHORT_ABLATION_REPORT.md`（2026-11-06，冻结随机 Decoder 的限制、Foot 实测失败、训练预算/梯度证据）。
- 用户已完成的 `S54_STAGE1_CHECKPOINT_EVAL_REPORT.md`（2026-11-05，Teacher 4772 的真实 Newton 评估、World Model 预测与 Root XY 漂移）。
- 用户已完成的 `MotionPrior_FootTrajectory.md`（此前 Stage1/Stage1.5/Stage2 合同与桥接设计）。

