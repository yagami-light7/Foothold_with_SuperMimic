# 任务：SuperMimic 架构重构——Foot-conditioned Actor + Asymmetric Critic + Terrain-conditioned World Model

## 一、项目背景

本项目使用 Isaac Lab 3.0 + Newton/MJWarp 作为仿真环境，并基于 SuperMimic 的 World Model + BPTT 框架训练人形机器人 Motion Tracking Policy。

当前已完成：

- 原 S54 机器人的 SuperMimic 复现及平地 Motion Tracking 实验。
- 机器人已由 S54 更换为同公司的 **Roban** 型号。
- 已准备 Roban 对应的 Motion Reference。
- 已有完整的 Newton 仿真、World Model、Actor、Critic、训练及 Evaluation 框架。
- 后续计划使用 TCRS 根据地形和 Footstep Goals 批量生成 Foot Trajectory 与 Whole-body Motion Reference。

现在希望将原来的 Full-body Motion Tracking Policy 改造成 **Foot Trajectory-conditioned Locomotion Policy**。

请先完整阅读当前仓库，理解现有架构，再进行最小必要修改。

**本次任务重点是代码架构、Reward、World Model、数据接口和测试，不要求立即进行正式训练。**

---

## 二、目标网络架构

目标是实现 Asymmetric Actor-Critic。

### 2.1 Actor Policy

Actor 在部署时只能访问：

1. Proprioception：关节位置、速度、Base 角速度、投影重力、上一时刻动作等实际可获取的信息。
2. Observation History：一定长度的本体历史观测。
3. Future Foot Trajectory：未来左右脚的位置、朝向目标。
4. Contact / Phase：左右脚预期接触状态、摆动相位。

数学表达：

\[
a*t=\pi*\theta(o*{t-k:t}^{prop},g*{t:t+H}^{foot},c\_{t:t+H})
\]

实现要求：

- 新建或改造 Foot Goal Encoder。
- Actor 不允许接收全身 Motion Reference。
- Actor 不允许使用仿真器真实地形高度图、全身特权状态等部署不可获取的信息。
- Foot Trajectory 使用机器人 Heading-aligned Local Frame。
- 足部位置、方向、Contact/Phase 使用明确的数据协议。
- Future Goal 的采样时间必须与 Reference FPS、Control DT 一致。
- 支持通过配置调整 Future Goal Horizon 和采样帧数。
- 优先复用原 SuperMimic Actor/Decoder，不进行不必要的网络重写。
- 如果存在 Shared Decoder 或共享特征路径，确保 Critic 的特权信息不会泄漏到 Actor。

特别注意：机器人目前为 Roban，所有关节、刚体和足部索引必须从当前模型及 Motion Reference 中获取，不允许硬编码原 S54 的 29 DoF 或 Body Index。

### 2.2 Asymmetric Critic

Critic 训练阶段可以使用：

- 完整机器人状态。
- Future Foot Trajectory。
- Whole-body Motion Reference。
- Root / Joint / Body Reference。
- Terrain Privileged Features。
- 必要的仿真特权参数。

数学表达：

\[
V\_\psi(s_t,g_t^{foot},g_t^{body},h_t,\xi)
\]

关键要求：

Critic 的输入包含 Whole-body Motion Reference，但 Critic 估计的仍然是当前 Reward Function 定义的未来累计回报。

不要错误地把 Whole-body Reference 直接当作 Critic 的监督标签。

保持与 SuperMimic 原有 Value Target、Bootstrap、Discount、Termination 和 BPTT 更新逻辑一致。

要求支持以下两种配置：

- `critic_use_body_reference=false`
- `critic_use_body_reference=true`

用于后续 Asymmetric Critic Ablation。

本阶段不增加独立的 Body Value Head，先使用单 Critic。

### 2.3 Reward Function

我们的主要目标是 **Foot Trajectory Tracking**，不是恢复原来的全身 Motion Mimic。

将 Reward 分为：

**Primary Task Rewards**

- Swing Foot Position Tracking
- Swing Foot Orientation Tracking
- Stance Foot Position Stability
- Contact Schedule Tracking
- Foot Touchdown Accuracy

**Stability / Safety Rewards**

- Base Upright
- Survival / Fall Termination
- Stance Foot Slip Penalty
- Joint Limit Penalty
- Torque / Action Rate Penalty
- 不合理身体碰撞惩罚

**Auxiliary Whole-body Motion Prior**

- 低权重上半身姿态跟踪。
- 低权重腰部姿态跟踪。
- 可选 Root Orientation / Height Prior。
- 不强制全身关节严格 Mimic。
- 特别避免较大的下肢 Joint Tracking Reward 与 Foot Tracking 发生冲突。

目标形式：

\[
r*t=r*{foot}+r*{contact}+r*{stability}
+\lambda*{prior}r*{body_prior}-p\_{unsafe}
\]

实现要求：

1. Reward 必须模块化，所有权重可配置。
2. `body_prior_weight=0` 时，完全禁用全身参考跟踪奖励。
3. 支持使用低权重 Body Prior 的模式。
4. 各 Reward Term 必须检查归一化尺度，避免 Body Prior 实际主导训练。
5. Whole-body Prior 使用最终 Roban Motion Reference。
6. 对 IK 质量不可靠的参考帧，应支持 Mask 或质量门控。
7. 保留 SuperMimic 的真实 Dynamics / Imagined Rollout 奖励计算一致性。
8. 检查 Reward 是否能够正确参与 BPTT 梯度传播。
9. 如果参考数据中缺少接触标签，不允许未经验证地将足踝高度阈值作为唯一的 Contact Truth。

不要仅修改 Reward 日志名称，必须确认 Actor 实际优化目标也已更新。

---

## 三、World Model 修改

保留现有 SuperMimic Differentiable World Model 和 BPTT 框架。

目标形式：

\[
\hat s*{t+1}=f*\omega(s_t,a_t,h_t^{local},\xi)
\]

其中 Terrain Feature 是训练阶段的特权动力学信息，不作为 Actor 部署输入。

### 3.1 地形输入接口

新增通用的 Terrain Feature Provider。

支持：

- Flat Terrain：零相对高度、有效地面 Mask。
- Pillar Terrain：局部柱顶高度、支撑区域、边缘信息。
- Stair Terrain：未来接入参数化楼梯 Height Map / Mesh。
- 后续支持更多参数化地形。

统一输出建议：

```python
TerrainFeatures(
    relative_heights,
    valid_mask,
    edge_features,
)
```

第一版建议支持基于支撑脚、摆动脚及目标落脚点的 Local Height Patch，例如每个位置采样 5×5 个点，实际尺寸由配置决定。

高度使用相对于当前支撑面的高度差。

对柱子场景，需要正确处理无支撑区域；不能把空中区域当作高度为零的实体地面。

### 3.2 WM 输入与梯度

要求：

- 通过配置 `wm_use_terrain` 控制是否使用地形特征。
- 保持原有 State / Action Prediction Target 不变，除非检查发现必须修改。
- 不把未来 Whole-body Reference 当成真实 Dynamics 输入。
- 地形特征应对应真实当前状态及其时间戳。
- 在 Imagined Rollout 中，地形查询必须能够跟随预测状态更新。
- 若当前 Terrain Sampling 不可微，需要明确处理对预测位置的梯度，并提供可测试的实现或说明限制。
- 保持状态归一化、输入归一化、Loss Mask 和终止处理正确。
- 保留已有 WM Loss、Prediction Metrics 和 BPTT 更新逻辑。

后续可能分别训练 Pillar WM 与 Stair WM，但本次要求统一输入协议，避免未来重新修改 Actor 和训练框架。

对于当前 Flat Motion 实验，允许先使用 Flat Terrain Provider 验证完整接口。

不要在本次任务中实现完整动态 Moving Pillars 物理环境。

---

## 四、Roban Motion Reference 接口

当前机器人已更换为 Roban，必须首先审查：

- Roban Articulation / URDF / USD / MJCF。
- Joint Names、DoF 数量及顺序。
- Body Names、左右脚 Link。
- Foot Sole Frame、Foot Contact Point、Toe/Heel Offset。
- Motion NPZ Schema。
- Root Position / Quaternion Order。
- Joint Velocity / Root Velocity Convention。
- FPS、Motion Length、Contact 信息。
- PD Actuator 配置及控制频率。

建立统一的：

```python
MotionReferenceProvider
```

能够从同一份 Roban Reference 生成：

1. Actor Foot Trajectory Goals。
2. Critic Whole-body Reference。
3. Reward Reference。
4. Evaluation Ground Truth。

这些信息必须共享相同的时间索引、坐标变换和 Episode Anchor。

避免 Actor Goal 与 Critic Reference 出现一帧偏差或坐标系不一致。

如果旧 S54 的关节或刚体映射还存在，清理不再适用的硬编码，但不要破坏历史实验。

---

## 五、训练配置与消融实验

本阶段重点验证两个问题：

1. 在 Actor 仅接收 Foot Trajectory 的情况下，加入低权重 Whole-body Motion Prior Reward，是否能够改善机器人运动稳定性和全身姿态？
2. 当 Reward 已包含 Whole-body Motion Prior 时，Critic 额外接收 Whole-body Motion Reference，是否能够进一步改善价值估计和策略训练？

所有实验均保持 Actor 输入一致：

\[
a*t=\pi*\theta(o_t^{prop},g_t^{foot})
\]

优先准备以下三组配置。

### E0：Foot-only Baseline

```yaml
actor_goal_type: foot_trajectory
critic_use_body_reference: false
body_prior_weight: 0.0
wm_use_terrain: false
```

- Actor 仅接收本体观测和未来足部轨迹。
- Critic 不接收 Whole-body Motion Reference。
- Reward 仅包含 Foot Tracking、Contact、Stability 和必要的控制惩罚。
- 作为基础对照组，验证纯足部目标能否实现稳定行走。

### E3：Foot Tracking + Body Prior Reward

```yaml
actor_goal_type: foot_trajectory
critic_use_body_reference: false
body_prior_weight: 0.03
wm_use_terrain: false
```

- Actor 输入与 E0 完全相同。
- Critic 输入与 E0 完全相同。
- Reward 额外加入低权重 Whole-body Motion Prior。
- Whole-body Reference 仅作为训练阶段的 Reward 监督，不进入 Actor 或 Critic 输入。

**E0 vs E3：单独验证 Body Prior Reward 对足部跟踪精度、身体稳定性和运动自然性的影响。**

注意：`body_prior_weight=0.03` 仅为初始候选值，需要结合 Reward Term 实际尺度调整，不代表全身奖励占总奖励的 3%。

### E2：Foot Tracking + Body Prior + Asymmetric Critic

```yaml
actor_goal_type: foot_trajectory
critic_use_body_reference: true
body_prior_weight: 0.03
wm_use_terrain: false
```

- Actor 输入与 E3 完全相同。
- Reward 定义及权重与 E3 完全一致。
- Critic 额外接收 Whole-body Motion Reference。
- Critic 仍然预测当前总 Reward 对应的长期累计回报，不引入独立 Body Value Head。

**E3 vs E2：单独验证全身 Reference 作为 Critic 特权输入是否改善训练效率、长期价值估计和真实 Newton Tracking 性能。**

### 实验优先级

第一阶段仅进行 E0 和 E3，优先确定 Body Prior Reward 是否有效。

第二阶段在 E3 基础上增加 E2，评估 Asymmetric Critic 的额外贡献。

暂时不进行 `critic_use_body_reference=true` 且 `body_prior_weight=0` 的 E1 实验，因为该配置并不提供明确的全身 Mimic 优化目标，优先级较低。

### World Model Terrain Interface Smoke Test

另外准备：

```yaml
wm_use_terrain: true
terrain_provider: flat
```

用于验证 Terrain-conditioned World Model 接口的输入维度、Forward、Prediction Loss 及 BPTT 梯度传播。

本阶段不需要训练 Moving Pillars / Stair World Model。

### 实验要求

1. 上述字段必须真正接入现有 Trainer、Reward、Critic 和 World Model，不能创建无法生效的孤立 YAML。
2. E0/E3 除 Body Prior Reward 外，其余配置保持一致；E3/E2 除 Critic 输入外，其余配置保持一致。
3. 所有实验使用相同的 Motion Dataset、环境配置、训练预算、Evaluation Protocol 和随机种子集合。
4. 分别记录 Foot Tracking、Stability、Body Prior 原始值及加权值、Critic Loss、WM Loss 和 Actor Gradient。
5. Evaluation 统一比较 Foot RMSE、Contact Accuracy、Survival、Root Stability、Joint Smoothness 以及全身姿态质量。
6. 不直接以不同 Reward 配置下的训练 Return 大小判断策略优劣。
7. 必须保证 E3 中 Whole-body Reference 仅进入 Reward 计算，不能通过共享 Encoder 或其他路径泄漏给 Actor / Critic。
8. E2 中 Whole-body Reference 只能作为 Critic 特权输入及 Reward 监督，不得进入 Actor 推理路径。
9. 保留原有 SuperMimic World Model + BPTT 优化机制，不额外引入辅助 Value Head。

**本次仅要求准备配置、实现对应开关、完成单元测试及短时 Smoke Test，不启动长时间正式训练。**

---

## 六、代码实现要求

请先梳理现有模块依赖：

1. Environment / Observation Builder。
2. Motion Reference Loader。
3. Actor / Goal Encoder / Decoder。
4. Critic / Value Target。
5. Reward Function。
6. World Model。
7. BPTT Trainer。
8. Newton Rollout / Evaluation。
9. Checkpoint Save / Load。

优先扩展已有模块，不要复制一套独立的 Trainer。

特别检查：

- Actor 的 Forward 输入是否真正去掉 Whole-body Reference。
- 是否通过共享 Latent 间接访问 Critic Privileged Observation。
- Critic 的梯度和 Value Loss 是否正确。
- BPTT 是否实际使用新的 Reward。
- WM 是否正确接收可选 Terrain Features。
- 使用不同 Actor Goal Schema 后，Checkpoint 加载是否正确处理 Shape Mismatch。
- 已有 Flat Checkpoint 是否能够部分复用兼容权重。
- 优化器状态能否正确恢复。
- Evaluation 是否使用真实 Newton Dynamics，而非直接设置 Root / Joint State。

不要强行加载旧 Goal Encoder 的不兼容权重，也不要因为修改网络结构而静默丢弃大量权重。

---

## 七、测试与验收

本次任务重点是 **架构正确性**。

### 必须完成的单元测试

1. Actor Observation Shape / Dtype / Device 检查。
2. Actor 对 Whole-body Reference 的信息隔离测试：仅改变全身 Reference，Actor 输出应保持不变。
3. Critic Privileged Observation 接入测试。
4. Reward Term 数值及尺度测试。
5. `body_prior_weight=0` 的行为验证。
6. World Model Terrain Feature 开关测试。
7. Terrain Feature Batch 维度与环境数量一致性。
8. WM Prediction Loss 与 BPTT Gradient 有限性测试。
9. Motion Reference 时间索引和坐标变换一致性测试。
10. Roban Joint / Body Mapping 与 FK 对齐测试。
11. Checkpoint Save / Load 测试。
12. Flat Newton Environment 短时 Rollout。

### 训练 Smoke Test

完成一次短时训练验证：

- 能正确创建 Roban Newton Environment。
- Actor、Critic、WM 正常 Forward。
- Critic Loss 和 WM Loss 正常计算。
- Actor 可以收到 BPTT 梯度并更新。
- 无 NaN、Inf 和明显的 Shape Mismatch。
- Checkpoint 能正常保存、恢复。
- Evaluation 不依赖全身 Reference 作为 Actor 输入。

短时 Smoke Test 通过仅代表代码链路正确，不代表 Policy 已学会行走。

本次不启动长时间正式训练。

---

## 八、交付文档

生成：

`docs/ROBAN_FOOT_CONDITIONED_ARCHITECTURE.md`

必须包括：

1. 修改前后网络结构对比。
2. Actor / Critic / WM 输入输出维度。
3. Reward 数学定义及实际配置。
4. Body Prior 的作用及梯度路径。
5. World Model Terrain Feature 数据协议。
6. Roban Motion Reference 数据映射。
7. 各配置的实际生效位置。
8. 修改文件及兼容性说明。
9. Unit Test 和 Smoke Test 结果。
10. 后续 Moving Pillars + TCRS 数据接入方式。

最终给出模块结构图、测试结果、存在的问题及可直接执行的训练命令。

---

## 九、重要约束

- 当前任务只针对 **Roban**，不再假设机器人是 S54。
- 不修改 TCRS 仓库。
- 不重写 SuperMimic 的核心 World Model + BPTT 算法。
- 保留旧模型与训练结果。
- 不直接修改现有实验日志及 Checkpoint。
- 不启动长时间训练。
- 不以训练回报较高代替真实 Newton Evaluation。
- 不允许为通过测试而关闭关键物理约束。
- 不允许把 Full-body Reference 隐式加入 Actor Observation。
- 如现有代码与本任务要求冲突，先定位具体实现，选择最小改动方案。

**执行方式：先审查仓库并列出修改计划，然后直接完成代码修改、测试和文档，不要只输出设计方案。**

本阶段完成标准：Roban Foot-conditioned Actor、Asymmetric Critic、模块化 Reward 与 Terrain-conditioned WM 接口全部具备可运行实现，且短时 Newton/训练 Smoke Test 通过。
