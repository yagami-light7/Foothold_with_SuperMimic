# S54 M3.4：XY Error、Root Position Reward 与 Actor 输入审查

审查对象：`logs/s54_m34_run02/ckpt_005500.pt`，2026-11-09。只读加载 checkpoint 与已保存的 Newton rollout；未修改模型、奖励、数据划分或归一化统计，未重启训练。

核心结论：当前 Actor 没有直接获得相对参考轨迹的 XY 位置误差；Motion Goal 以当前参考帧为原点。只平移机器人 1 m 时，Observation、Goal 和动作完全相同。Root Position 是正向奖励理论上限的 10%，且大误差区间的指数核与梯度快速衰减。

## 1. XY Error 的定义与现有指标

对同一参考时刻，令位置差为 Δp = p_policy − p_reference，位置取 `base_link` 原点，单位 m。`env.state()` 已减去各环境的 `env_origins`。评估未做逐帧 XY 平移、yaw 对齐或轨迹最优配准。

- 单帧 XY 距离：`sqrt(Δx² + Δy²)`。
- XY RMSE：`sqrt(mean_t(Δx² + Δy²))`，没有除以坐标维数 2。
- 三维 Root RMSE：`sqrt(mean_t(Δx² + Δy² + Δz²))`。
- 原 `results.json/root_position_rmse_m` 和 SwanLab `RootPositionRMSE_m` 使用三维 Root RMSE。
- 单轨迹 MP4 上的 `root error` 使用当前帧的三维距离；它既不是 XY-only 指标，也不是时间 RMSE。
- 重算与原评估一致：只用每次物理 step 后的状态 S1，对齐参考 f+1；排除初始化帧，保留末次物理状态。已验证 18 条轨迹的 frame 连续、三维 RMSE 与原结果在 1e-6 容差内一致。

代码：`supermimic/env.py:144`、`tools/m34_rollout.py:130`、`tools/m34_rollout.py:180`、`tools/render_s54_m34_eval.py:103`。

| Test reference | XY RMSE (m) | XYZ RMSE (m) | Z RMSE (m) | 最后 XY 偏差 (m) |
|---|---:|---:|---:|---:|
| forward_unseen | 0.473971 | 0.474107 | 0.011368 | 0.678451 |
| backward_unseen | 4.239873 | 4.239890 | 0.012055 | 7.033194 |
| turn_unseen | 0.949076 | 0.949346 | 0.022640 | 1.637242 |
| spin_unseen | 0.222450 | 0.222973 | 0.015254 | 0.108861 |

14 条 train 轨迹的等轨迹权重 XY RMSE 均值：0.149365 m。4 条 test：1.471343 m。完整逐轨迹结果保存在 `logs/s54_m34_audit_005500/xy_reward_observation_audit.json`。

## 2. Root Position Reward 的构成与占比

`r_root_pos = 0.02 × 0.5 × exp(−||p_reference − p_policy||² / 0.3²)`。位置项是全局 XYZ 误差，不是 XY-only。指数分母没有额外的 2。`0.5` 是 mdp.py 中的固定系数，`root_pos_std=0.3` 来自 checkpoint 的 reward_config。

| 正向奖励项 | 系数 | 单步理论上限（乘 dt=0.02 后） | 正向理论上限占比 |
|---|---:|---:|---:|
| Body Position | 1.0 | 0.02 | 20% |
| Body Orientation | 1.0 | 0.02 | 20% |
| Body Linear Velocity | 1.0 | 0.02 | 20% |
| Body Angular Velocity | 1.0 | 0.02 | 20% |
| Root Position | 0.5 | 0.01 | 10% |
| Root Orientation | 0.5 | 0.01 | 10% |

正向总上限为 `(4×1 + 2×0.5)×0.02 = 0.1`。总奖励另加负向项：action-rate −0.1、joint-limit −2.0、head-position −0.5、head-velocity −0.005，最后统一乘 dt。负向项使实际总回报的分母改变，所以实际 Root Position 占比不是固定 10%。

Body Position 和 Body Orientation 会将参考锚定到机器人的当前 XY 与 yaw；其中 Body Position 仍保留参考高度。它们不能直接测量绝对 XY 轨迹偏差。速度项仍比较世界系速度，因此仍能约束运动方向/速度，但不直接表达累计位置偏差。Root Orientation 单独约束参考姿态。

距离为 0.3 / 0.5 / 1.0 m 时，Root Position 的指数核分别为 0.367879 / 0.062177 / 0.000014945。以该核表示的连续误差惩罚在大误差区间接近饱和；提高系数不会为 Actor 增加缺失的 XY 误差观测。

代码：`supermimic/mdp.py:130`、`supermimic/mdp.py:175`。终止条件只检查高度与倾斜，没有 XY 漂移阈值，因此跑满参考长度不能证明全局轨迹准确。

下表从保存的真实 rollout 重算 Root Position 分量；归一化分母与原评估一致，为整条参考步数 × 0.1。

| Test reference | 总归一化回报 | Root Position 分量 | 占实际总回报 |
|---|---:|---:|---:|
| forward_unseen | 0.876612 | 0.038999 | 4.449% |
| backward_unseen | 0.830301 | 0.011254 | 1.355% |
| turn_unseen | 0.813374 | 0.018943 | 2.329% |
| spin_unseen | 0.898781 | 0.063045 | 7.014% |

## 3. Policy Observation

checkpoint 确认：`goal_mode=motion`、`obs_adapter=identity`、`obs_dim=99`、`hist=10`；最后 6 个参考姿态通道未屏蔽。

| 通道（Python 切片） | 维数 | 内容 |
|---|---:|---|
| 0:29 | 29 | q − 模型默认关节角 − 随机化关节 offset；不是 q − q_reference |
| 29:58 | 29 | 实际关节速度 qd |
| 58:61 | 3 | 实际 Root 局部角速度 Rᵀω |
| 61:64 | 3 | 机体系重力方向 −Rᵀe_z |
| 64:93 | 29 | 上一步实际 action |
| 93:99 | 6 | 当前实际 Root 到当前参考 Root 的相对旋转，6D 表示 |

不直接包含：Root 世界坐标、Root 相对参考的 XY/XYZ 位置误差、Root 线速度或线速度误差、足底接触、足位置误差。这里的“未直接包含”不排除网络通过关节历史间接学习部分运动线索。

History 为当前帧及之前 9 帧，共 99×10=990 维，采样间隔 0.02 s，时间跨度 −0.18..0 s。正常训练 reset 时用当前观测填满 history；独立评估也采用这一初始化。训练采集和 BPTT 对观察加噪，当前 seed-0 nominal 评估不加噪。

RunningNorm 的 obs/goal 样本计数均为 28,684,288，评估不更新。`OBS_SCALE` 虽被定义并保存到 mdp.obs_scale，但当前 Motion Actor 的调用路径没有乘它；实际使用 RunningNorm。不要将注释中的固定速度缩放当作实际 Observation 预处理。

代码：`supermimic/mdp.py:117`、`supermimic/networks.py:137`、`supermimic/train.py:609`、`tools/m34_rollout.py:115`。

## 4. Motion Goal

未来 10 个参考帧 f+1..f+10，对应 0.02..0.20 s；每帧 73 维，合计 730 维。

| 每个未来帧的内容 | 维数 | 坐标/尺度 |
|---|---:|---|
| 未来参考 Root 位移 | 3 | H_ref(f)ᵀ × [p_ref(f+k) − p_ref(f)] |
| 未来参考 Root 旋转 | 6 | H_ref(f)ᵀ × R_ref(f+k)，6D |
| 未来参考 Root 线速度 | 3 | 未来参考自身完整 Root 姿态局部系 R_ref(f+k)ᵀv_ref(f+k) |
| 未来参考 Root 角速度 | 3 | 未来参考自身完整 Root 姿态局部系 R_ref(f+k)ᵀω_ref(f+k) |
| 未来参考关节角 | 29 | q_ref(f+k) |
| 未来参考关节速度 | 29 | 0.1 × qd_ref(f+k) |

最关键的一行：`p_ref(f+k) − p_ref(f)`。原点是当前参考 Root，不是当前实际机器人 Root；`GoalSource.goal()` 的 Motion 分支直接调用 `motion.goal(f,K)`，没有使用输入状态 S。故 Motion Goal 不含当前累计 XY 偏差。坐标契约的 `reference_heading` 概括的是位移/姿态部分，速度通道实际使用未来参考完整 Root 的局部坐标。

Encoder：730 → 2048 → 1024 → 512 → 256 → 32，最后 tanh。共享 Decoder 输入为 32+990=1022，输出 29 维 action；未启用 Foot Encoder 或低维 Command Policy。

时间检查：18 条参考的未来索引均在本 clip 内；末尾 clamp 到最后参考帧。真实物理 step 后在 S1 上计算 f+1 的奖励；BPTT 同样推进 f1=f+1，并限制起点满足 f+horizon≤clip_end。上述检查未发现跨 clip 的未来索引或控制步长不一致。

代码：`supermimic/goals.py:135`、`supermimic/motion.py:132`、`supermimic/m34.py:171`、`supermimic/env.py:264`、`supermimic/train.py:349`、`supermimic/train.py:407`。

## 5. 冻结 checkpoint 的输入敏感性检查

CPU float32，固定 forward_unseen 的参考帧和其余状态/历史通道，不使用新物理 rollout。

| 对机器人状态施加的改变 | Obs 最大变化 | Goal 最大变化 | Action 最大变化 |
|---|---:|---:|---:|
| 仅世界 X 位置 +1 m | 0.000000000 | 0.000000000 | 0.000000000 |
| 仅世界线速度 vx +1 m/s | 0.000000000 | 0.000000000 | 0.000000000 |
| 仅 Root yaw +20° | 0.342012078 | 0.000000000 | 0.958951116 |

Critic 会收到 Root 相对参考位置误差（表达在实际 Root 的 yaw 坐标系），1 m 平移时 Critic feature 的最大变化约 0.999939；Actor 输入仍完全相同。Critic 的额外信息和 BPTT 的位置奖励梯度并不等于 Actor 获得了位置误差反馈。

结论范围：已经证明当前输入映射缺少绝对 XY 误差的直接闭环反馈，这与轨迹漂移一致；该检查没有证明它是全部失败的唯一原因。Backward 仍属于跨运动模式外推，不能通过调参测试集来修正这一项。

当前 mdp.py / goals.py / motion.py / networks.py / env.py / train.py 的 SHA-256 均与 run02 保存的训练源码一致。此次没有修改算法或重新训练。
