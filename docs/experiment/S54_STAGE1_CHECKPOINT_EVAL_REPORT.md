# S54 Stage 1：三个 Checkpoint 独立评估统计

日期：2026-11-05（Asia/Shanghai）。训练 run：`logs/s54_stage1_walk_20261105_014722`。

本报告复用用户正在执行的三个整段 eval，待它们全部结束后才追加一次短 Newton 配对探针。没有覆盖历史结果，没有启动训练，没有更新模型或优化器。新增探针、离线计算的临时工作文件放在 `/tmp/s54_checkpoint_audit_20261105/`；本轮最终交付仅本 Markdown 文件。

实际比较的是 `ckpt_002500.pt`、`ckpt_003500.pt`、`ckpt_004772.pt`。最初所列 `ckpt_004722.pt` 不存在，未创建或冒用该编号。

## 结论

- 整段 Walking 的 Normalized Return：**0.661004 → 0.767451 → 0.820197**。三个 checkpoint 均完成261.3秒，物理终止均为0。
- 按训练奖励方式对齐的 Body Position RMSE 改善 **44.4%**；但全局平移跟踪不单调，Root XY RMSE 为 **2.773 / 5.265 / 2.330 m**。因此，“持续不倒”和“高 Return”不能直接等同于世界轨迹跟踪合格。
- 相同短探针协议下，六个正向奖励分项均随 checkpoint 改善；这是**独立评估的三个时间点趋势**，不是历史训练逐 iteration 的六分项曲线。后者没有记录，无法补造。
- 同一共同 Newton 数据集上，32步 Foot FK 配对 RMSE 从 **132.0 mm** 降至 **31.0 mm**，改善 **76.5%**。4772 的32步误差仍明显大于8步误差。
- 历史训练日志可还原物理终止数，但 Clip 结束与500步时限被合并，实际训练起始帧直方图未保存；独立 eval 和本次探针的结束原因、起始帧则可以从真实录制逐项统计。

## 1. 数据与公平比较口径

仅有同一份已通过QA的 **`walk1_subject1`**，长度261.3秒、50 Hz控制、13,066个状态帧。此处“相同 Walking Clips”实际为**同一个 Walking clip**；当前没有独立 Walking holdout，不能据此声称跨clip泛化，Dance也未充当Walking测试集。

| 协议 | 整段独立 eval（主 Return / Tracking RMSE） | 追加短探针（六分项 / WM 配对） |
| --- | --- | --- |
| 模型 | 2500、3500、4772，分别独立执行 | 同三个冻结 checkpoint |
| Reference | walk1_subject1，全片 | 同clip的0、60、120秒起点 |
| 时长 | 每个261.3秒，13,065步 | 每个起点10秒，500步 |
| 起始帧 | 每个checkpoint为0 | 每个checkpoint为0 / 3000 / 6000 |
| seed | 54 | 54 |
| 初始扰动 / 观测噪声 / 动作噪声 | 全部关闭 | 全部关闭 |
| 域随机化 / 摩擦 | 关闭 / 1.0 | 关闭 / 1.0 |
| 执行方式 | 用户原有流程：3个1环境eval串行 | 原eval全部结束后，1个Newton进程、9个隔离环境；每个Policy batch=3 |
| 精度 | CUDA bf16 MLP，FP32物理状态与几何计算 | 相同MLP精度；WM积分、FK、reward保持FP32 |
| 真实样本量 | 3 × 13,065 = 39,195条已存在transition | 新增9 × 500 = 4,500条transition |
| 硬件 | 实际RTX3090 | 实际RTX3090 |

全部checkpoint的机器人、物理、数据和MDP任务契约一致。探针未构建优化器，Policy参数及归一化状态哈希前后一致。WM只做冻结推理，不用预测轨迹替代真实Newton数据。

当前训练实际设置是：`num_envs=1024`、`wm_batches=16`、`wm_batch=1024`、`wm_window=8`、`pi_batch=256`、`horizon=32`、`hist=10`；WM LR=`5e-5`，Actor/Critic LR=`2e-4`，前1000 iterations为WM-only。**WM监督窗口8步与Actor BPTT的32步是不同参数。**

本报告主表严格使用用户本批次 `results/s54/checkpoint_eval_20261105_163654/` 的结果。旧的4772权重全片预览曾得到0.820641，本批为0.820197；未混用旧录制的RMSE或选择较好的Return。重复运行的小差异尚未做确定性原因定位；这里是单次固定seed评估，不提供多种子置信区间。

## 2. 独立 Normalized Return 与 Tracking RMSE

每步真实奖励为 `r_t = 0.02 × (六个正向分项 + 原有负向项)`，最大正向和为5，故 `max_step_reward=0.1`。

`Normalized Return = Σ r_t / (13,065 × 0.1)`，分母为预定全片长度；提前物理终止也不缩短分母。表中使用原始奖励，包括动作变化、关节限位和头部惩罚，没有删除惩罚来提高分数。

| Checkpoint | Normalized Return | 实际步数 | 实际秒数 | 物理终止 |
| --- | --- | --- | --- | --- |
| 2500 | 0.661004 | 13065 | 261.3 | 0 |
| 3500 | 0.767451 | 13065 | 261.3 | 0 |
| 4772 | 0.820197 | 13065 | 261.3 | 0 |

以下RMSE使用**全部13,065个真实post-action状态**，不是只选择成功片段。位置RMSE为三维误差向量长度的均方根；Joint q为29关节分量的均方根；姿态使用SO(3)测地角。Foot FK指`leg_l6_link` / `leg_r6_link`的连杆坐标原点，不等同于实测足底接触点。

### 2.1 与训练奖励一致的姿态对齐口径

Reference身体位置和方向按Policy当前root XY与yaw重新锚定，保留Reference高度。这一口径衡量相对全身姿态，主动去掉全局XY/yaw偏移；Joint q不做此对齐。

| Checkpoint | Body Position (m) | Body Orientation (°) | Foot FK (m) | Joint q (rad) |
| --- | --- | --- | --- | --- |
| 2500 | 0.0977 | 25.437 | 0.1570 | 0.3025 |
| 3500 | 0.0675 | 15.068 | 0.1113 | 0.2063 |
| 4772 | 0.0544 | 12.054 | 0.0879 | 0.1798 |

### 2.2 不做位置或航向对齐的世界坐标口径

| Checkpoint | Body Position (m) | Foot FK (m) | Root XY (m) | Root Height (m) | Root Orientation (°) |
| --- | --- | --- | --- | --- | --- |
| 2500 | 2.7794 | 2.7885 | 2.7732 | 0.0252 | 7.627 |
| 3500 | 5.2639 | 5.2617 | 5.2647 | 0.0214 | 4.464 |
| 4772 | 2.3304 | 2.3229 | 2.3305 | 0.0183 | 3.087 |

3500的相对姿态比2500更准，但Root XY偏移更大；4772改善了相对姿态与全局误差，仍有米级平移漂移。原奖励的Body Tracking会重锚定，Root Position仅占最大正奖励的10%，因此较高Return可以与明显世界坐标漂移并存。这是当前结果中最需要注意的能力缺口，不能用对齐后的Foot FK RMSE代替全局足部轨迹误差。

## 3. Episode Length、终止原因与起始帧

### 3.1 三个整段独立 eval：实际计数

采用互斥优先级 **Physical Termination > Clip End > Time Limit Only**。末尾恰好同时达到Clip边界和评估时间上限时，归入Clip End；原始`timeout=1`不会计为跌倒，也不重复计数。

| Checkpoint | Episode Length（步 / 秒） | 物理终止 | Clip End | Time Limit Only | 起始帧 / 秒 |
| --- | --- | --- | --- | --- | --- |
| 2500 | 13065 / 261.3 | 0 | 1 | 0 | 0 / 0.00 |
| 3500 | 13065 / 261.3 | 0 | 1 | 0 | 0 / 0.00 |
| 4772 | 13065 / 261.3 | 0 | 1 | 0 | 0 / 0.00 |

每个checkpoint只有1个整段episode；起始帧经验分布均为 **frame 0：100%**，末帧均为13065。三次原始`timeout`标记均为1，均经轨迹末帧验证确实到达Clip末尾。此表不代表随机起点或扰动下的成功率。

### 3.2 历史训练曲线：能还原的次数

训练默认上限是500步／10秒，随机reference起点可能使episode更早到达Clip末尾。日志中的`term_frac = 物理终止数 / 已结束episode数`。利用`completed_episodes`以及CSV的六位有效数字格式，逐行寻找唯一一致的整数物理次数；全部238行均唯一还原。

| 日志点 | Episode Length（步 / 秒） | 已结束数 | 物理终止 | 非物理结束 | Clip / 500步时限拆分 |
| --- | --- | --- | --- | --- | --- |
| 2500 | 427.543 / 8.551 | 138 | 1 | 137 | 未记录 |
| 3500 | 254.761 / 5.095 | 217 | 0 | 217 | 未记录 |
| 4760 | 258.075 / 5.162 | 175 | 0 | 175 | 未记录 |

每行对应该日志点前20 iterations的采样窗口，不是累计次数，也不是该checkpoint的独立eval。累计日志覆盖到4760，含最初replay预热：

| 范围 | 已结束数 | 物理终止 | 非物理结束 | 物理终止比例 |
| --- | --- | --- | --- | --- |
| 全部已记录窗口 | 109424 | 82409 | 27015 | 75.312% |
| 日志点4000–4760 | 8109 | 11 | 8098 | 0.136% |

早期WM-only和尚未学会的策略贡献了大多数历史物理终止；不能用整个训练累计比例代替最终Policy表现。最后checkpoint是4772，但末尾12 iterations／30,720条transition没有形成新的完整CSV窗口，计数未冒充覆盖到4772。

**历史缺失项：**训练代码把`Clip End`与500步时限做OR后统一记为`timeout`，没有持久化两个计数或逐episode事件。因而27,015次累计非物理结束无法拆分，不能把它们全部叫作Clip结束。

**Episode Length还有统计口径问题：**当前先对每个vector step里结束的episode求均值，再对有结束事件的vector steps求均值，即 `mean_j(mean_e L[j,e])`，而非 `sum(L)/sum(episode_count)`。各步完成episode数量不同时会产生加权差异；仍在运行的episode也不计入。由于没有保存各步长度总和，旧训练的严格episode加权平均无法还原。下降本身不能判定为更容易跌倒。

### 3.3 起始帧：规则与实测分布必须分开

| 数据集 | 可确认的实际起点分布 |
| --- | --- |
| 本批整段独立eval | 每个checkpoint：0秒起点100% |
| 本次短探针 | 每个checkpoint：0 / 60 / 120秒各1次、各33.33%；合计9个episode，各起点3次 |
| 历史训练 | 未保存起始帧直方图，无法还原实测分布 |

历史训练采样规则是：Reference切为262个桶，每桶50帧（1秒），选桶概率 `p_i = 0.5/262 + 0.4×softmax(-25 G_i/G_max) + 0.1×EMA_i/ΣEMA`，再在桶内随机选合法起点。较低回报的桶获得较高自适应权重。初始为均匀选桶，**不等于全体帧严格均匀**：最后一个桶仅15个合法起始帧。

合法起始帧为0–13064，即0–261.28秒；最后一帧13065保留作后继终点。例如从260秒开始，即使不跌倒，也会在1.3秒后到达Clip结束。checkpoint只保存模型、优化器、RNG等，没有保存`bin_G`、`bin_w_avg`、reset直方图或Replay内容；所以最终动态权重和历史实际起始帧分布均未知，50/40/10不能冒充实测比例。

## 4. Motion Reward 六个正向分项的趋势

**历史训练的逐iteration六分项曲线不可还原**：本run的`reward_term_logging=false`，没有逐分项历史数据，Replay也未保存。以下使用相同协议的独立Newton短探针，对2500→3500→4772做三点趋势；不将它标成训练期间原始曲线。

六项保持原公式和系数：Body Position / Orientation / Linear Velocity / Angular Velocity各最大1；Root Position / Orientation各最大0.5。表中值是**含原权重、未乘dt的真实逐步均值**，越大越接近相应参考；Root两项没有重新缩放到1。速度取Newton真实速度，不由位姿差分近似。

| Checkpoint | Body Pos ≤1 | Body Ori ≤1 | Body LinVel ≤1 | Body AngVel ≤1 | Root Pos ≤0.5 | Root Ori ≤0.5 |
| --- | --- | --- | --- | --- | --- | --- |
| 2500 | 0.89095 | 0.49391 | 0.74366 | 0.79077 | 0.17330 | 0.46936 |
| 3500 | 0.93737 | 0.69452 | 0.89146 | 0.87184 | 0.24783 | 0.48395 |
| 4772 | 0.95047 | 0.76330 | 0.92639 | 0.91152 | 0.31172 | 0.49058 |

六项在该短探针上均改善，主要提升来自Body Orientation与两项身体速度跟踪。相同探针的净Normalized Return和负向项如下；负向项同样是未乘dt、含原系数的均值：

| Checkpoint | 10秒探针平均NR | Action Rate | Joint Limit | Head |
| --- | --- | --- | --- | --- |
| 2500 | 0.699445 | -0.043240 | -0.012062 | -0.009428 |
| 3500 | 0.814395 | -0.046459 | -0.000080 | -0.008453 |
| 4772 | 0.860289 | -0.045933 | -0.000622 | -0.005986 |

九个探针episode均执行500步，均为正常Time Limit Only，没有Physical Termination或Clip End；本表每个checkpoint等权平均三个起点。原正负项重建的每步奖励与MDP实际奖励最大差为 `1.118e-07`，校验通过，没有只提高或删除某项奖励。

为检验短探针是否掩盖长时漂移，下表给出整段eval中能够由完整位姿**精确**重建的四项；整段文件没有保存真实速度和动作，因此没有伪造整段的两项速度奖励：

| Checkpoint | 全片Body Pos | 全片Body Ori | 全片Root Pos（权重0.5） | 全片Root Ori（权重0.5） |
| --- | --- | --- | --- | --- |
| 2500 | 0.90357 | 0.49980 | 0.01820 | 0.45400 |
| 3500 | 0.95195 | 0.72990 | 0.01539 | 0.48174 |
| 4772 | 0.96823 | 0.82394 | 0.02664 | 0.49112 |

4772的全片Root Position仅为最大值的 **5.33%**，显著低于短探针。短窗口频繁从Reference状态重置，不能替代整段全局位移评估。当前body重锚定、Root Position较小权重，以及Policy未直接观测root XY误差，是需要进一步验证的任务契约线索；本报告不把它们单独宣称为漂移的已证实因果。

## 5. World Model：8 / 16 / 32步真实仿真配对误差

### 5.1 配对协议

每条WM轨迹都从**真实Newton状态**开始，逐步重放该真实episode保存的同一动作序列、`xi`、Reference帧和时间。比较第8 / 16 / 32步的预测终点与真实Newton终点，分别对应0.16 / 0.32 / 0.64秒。预测期间不再调用Actor，没有Critic、bootstrap、soft survival或奖励折扣。

每个10秒episode以stride=10个控制步选择起点，固定取足够容纳32步的窗口：每episode47个窗口。所有horizon复用完全相同的起点，不跨episode/reset边界，保留真实终止端点。MLP使用CUDA bf16，状态积分、FK和奖励为FP32。

- **主比较：共同数据池。**9个真实episode合并，每个WM都在完全相同的423个真实起点、动作序列上评估，隔离Policy访问状态不同造成的混淆。
- **补充：各自原生Policy数据。**每个WM只用对应Policy的3个episode，共141个窗口。此口径反映当前Policy分布内误差，但不同checkpoint的数据分布不同。

Foot FK RMSE为两个足部连杆原点的三维误差向量长度的均方根。Joint q RMSE为29关节分量均方根。Reward Bias为 `mean[(Σr_WM − Σr_Newton)/(h×0.1)]×100`，单位为Normalized Reward的百分点（pp），正值表示WM高估；Reward MAE是该累计归一化误差的绝对值均值，不是WM训练loss。

### 5.2 共同真实轨迹上的主比较（每模型、每horizon 423窗口）

| Checkpoint | 步数 | Height RMSE (mm) | Pitch RMSE (°) | Foot FK RMSE (mm) | Joint q RMSE (rad) | Reward Bias (pp) | Reward MAE (pp) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2500 | 8 | 3.054 | 0.818 | 26.389 | 0.02202 | -0.855 | 1.042 |
| 2500 | 16 | 7.505 | 2.857 | 64.097 | 0.03307 | -1.427 | 1.594 |
| 2500 | 32 | 17.334 | 9.144 | 131.959 | 0.04889 | -3.292 | 3.426 |
| 3500 | 8 | 1.424 | 0.299 | 10.773 | 0.00789 | 0.008 | 0.228 |
| 3500 | 16 | 2.235 | 0.794 | 18.923 | 0.01001 | -0.033 | 0.249 |
| 3500 | 32 | 5.688 | 2.654 | 40.830 | 0.01714 | -0.239 | 0.410 |
| 4772 | 8 | 1.274 | 0.299 | 8.678 | 0.00651 | 0.059 | 0.198 |
| 4772 | 16 | 1.935 | 0.697 | 14.522 | 0.00870 | 0.033 | 0.217 |
| 4772 | 32 | 4.309 | 1.878 | 30.995 | 0.01386 | -0.113 | 0.330 |

### 5.3 各自Policy真实轨迹上的补充比较（每模型、每horizon 141窗口）

| Checkpoint | 步数 | Height RMSE (mm) | Pitch RMSE (°) | Foot FK RMSE (mm) | Joint q RMSE (rad) | Reward Bias (pp) | Reward MAE (pp) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2500 | 8 | 1.400 | 0.276 | 10.814 | 0.00692 | 0.071 | 0.266 |
| 2500 | 16 | 2.709 | 0.848 | 19.086 | 0.00905 | 0.078 | 0.259 |
| 2500 | 32 | 5.256 | 2.559 | 36.648 | 0.01358 | -0.035 | 0.367 |
| 3500 | 8 | 0.796 | 0.209 | 8.174 | 0.00508 | 0.046 | 0.170 |
| 3500 | 16 | 1.255 | 0.605 | 13.183 | 0.00673 | 0.038 | 0.176 |
| 3500 | 32 | 2.919 | 1.745 | 24.556 | 0.00983 | -0.027 | 0.189 |
| 4772 | 8 | 0.646 | 0.182 | 5.505 | 0.00375 | 0.087 | 0.148 |
| 4772 | 16 | 1.081 | 0.423 | 9.463 | 0.00469 | 0.097 | 0.150 |
| 4772 | 32 | 1.779 | 1.161 | 18.306 | 0.00711 | 0.095 | 0.161 |

所有配对中的真实状态/动作/时间重建奖励与保存奖励的最大差为 **1.192e-07**，小于1e-5；所有表内指标有限，无NaN/Inf，也未丢弃坏窗口。共同池上2500 WM对后续Policy状态的32步误差明显大于其原生分布内误差，因此只看on-policy误差会低估跨分布问题。4772总体更好，但32步误差相对8步仍会积累；`wm_window=8`的训练误差不能代表32步精度。未为16/32步临时设定通过阈值。

这里的32步只是0.64秒开放环预测，不等于261.3秒闭环轨迹保证；更小的短期WM误差不能抵消整段世界坐标漂移。本报告未测独立Walking holdout、真实足底接触/滑移验收或多种子扰动成功率，因此4772仍应视为候选Teacher，不能仅凭存活和Return标为完整能力验收通过。

## 6. 数据来源与复核

### 6.1 用户原有整段eval（全部已正常结束）

- 2500：[evaluation.json](results/s54/checkpoint_eval_20261105_163654/ckpt_002500_recording/evaluation.json)、[trajectory.npz](results/s54/checkpoint_eval_20261105_163654/ckpt_002500_recording/trajectory.npz)、[status.json](results/s54/checkpoint_eval_20261105_163654/ckpt_002500_recording/status.json)。

- 3500：[evaluation.json](results/s54/checkpoint_eval_20261105_163654/ckpt_003500_recording/evaluation.json)、[trajectory.npz](results/s54/checkpoint_eval_20261105_163654/ckpt_003500_recording/trajectory.npz)、[status.json](results/s54/checkpoint_eval_20261105_163654/ckpt_003500_recording/status.json)。

- 4772：[evaluation.json](results/s54/checkpoint_eval_20261105_163654/ckpt_004772_recording/evaluation.json)、[trajectory.npz](results/s54/checkpoint_eval_20261105_163654/ckpt_004772_recording/trajectory.npz)、[status.json](results/s54/checkpoint_eval_20261105_163654/ckpt_004772_recording/status.json)。


历史训练次数来源：[train_log.csv](logs/s54_stage1_walk_20261105_014722/train_log.csv)，及此前已经存在的[逐行整数还原统计](results/s54/stage1_episode_statistics_20261105.json)。统计逻辑对照该run的 `source_snapshot/supermimic/{train,env,mdp,motion}.py`；历史训练文件和结果未改写。

### 6.2 完整性标识

| Checkpoint | 文件SHA256 | 整段轨迹SHA256 |
| --- | --- | --- |
| 2500 | `40e04f9068525c70511f6679ae55e5cd784c6c8832449940854791e31d50cb98` | `0e842f61faea3fff4e2c11522534b6501ab7e11b57ad9f4c6c917aed0d9ad4eb` |
| 3500 | `682293597a9876d692dbb835eda62344b7fbd1ec154f45d399f7db2c24f93b7e` | `e21d81c7ff028add755ec885c60659585e57c761cbd5f58d5123b3e3617ae916` |
| 4772 | `1cf556f384cefa8635975ca4f8cb8c0e144a98292c5fb2fb8cff33e9b07c3fc4` | `ed2a66cff0b8b810aca94140983540e24dd94e4d2488f54bea3dad7fc1b94718` |

Reference CSV SHA256：`ca84b2a06dde8e4e7251e10ed3f14cd10317b0f49cd5c4ea0818c7acbb26f26f`。

Manifest SHA256：`78271cca5426b803ddac8b9dde60f038e4812528b8ec09dc7b9d659dabfe6e7e`。

新增配对探针NPZ SHA256：`e8c3772f9134f96a8410c5170158fd3c69dba8b809f621684eb2b9d102dba0a7`。


临时探针与计算脚本位于 `/tmp/s54_checkpoint_audit_20261105/audit.py`，原始短探针为同目录 `paired_newton_probe.npz`，奖励分项和完整WM数值为 `probe_reward_heads.json`、`wm_2500.json`、`wm_3500.json`、`wm_4772.json`。这些是本机临时复核工作文件，不是新增仓库交付，系统清理后不保证保留；本报告已内嵌所请求的全部统计表和协议。原有整段eval路径及checkpoint仍保留。

复核代码路径：`MDP.reward`的原有六个正向项、`standing_diagnostics.paired_predictions(horizons=(8,16,32), stride=10)`、`valid_windows`。全片RMSE从原始pose做FK；零速度占位仅用于不依赖速度的几何计算，未将其用于速度奖励或WM误差。CPU上的旧10秒缓存配对只用于实现自检，未混入上述CUDA bf16主表。

本次只读分析通过：checkpoint契约一致、4772与last.pt的Policy/WM状态一致、奖励重建、配对时间/动作重建、窗口不越界和指标有限性检查。没有修改物理终止条件、训练奖励、模型权重或旧实验统计。
