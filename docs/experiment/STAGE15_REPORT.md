# Stage 1.5 — LAFAN1 Foot Encoder Skill Transfer

实验目录：[logs/s54_stage15_20261106_172329_bounded_b](/home/rm/workspace/ljy/Foothold_with_SuperMimic/logs/s54_stage15_20261106_172329_bounded_b)。执行状态：**COMPLETED**；技能验收：**BLOCKED_QUALITY**。所有训练、评估和上传均使用独立 tmux；没有改写 Stage 1、ABC、Joint 的历史文件。

## Teacher、数据与接口

- Teacher：`logs/s54_joint_ab1_20261106_025912_seed54/last.pt`，iteration **14,484**；SHA256 `ef5671d4e223ab004c1f69837fffca6f461ca1c2c661c111b5afc2f82e6e5544`。严格校验机器人、物理、Motion Goal、源数据及模型形状契约。
- 保留并冻结 Teacher Motion Encoder、Shared Decoder 及其 Motion Norm；WM 保留初始化。Foot Encoder 重新初始化；不继承原 Quintic Encoder 的饱和权重或 GoalNorm。Phase A 仅优化 E_f，WM/Critic 不更新。
- 唯一参考：`data/retarget_data/reference_startup_fixed/walk1_subject1.csv`，50Hz 控制重采样。FootGoal 是原始 FK 的 10 个未来时刻：0.02/0.04/…/0.20s，与 MotionGoal 的 f+1…f+10 严格相同。没有 Quintic、随机足轨迹或轨迹拟合。
- 230D = 10 × [左足(position3 + rotation6 + contact1 + swing phase1) + 右足同11D + delta_time1]。位置/旋转位于实际 root yaw 坐标系；contact 是原提取器的几何/速度滞回估计，并非真实接触力。
- Teacher MotionPolicy 接收完整 MotionGoal；FootPolicy 只接收 FootGoal 和10帧本体观测。Decoder 的93:99虚拟朝向通道由相同230D FootGoal与本体重力构造；其余93通道仍为原本体观测。ObsNorm复制Teacher并冻结。完整 reference 不进入 Foot Encoder/Decoder；仅用于初始化、蒸馏标签、训练奖励、特权 Critic及独立评分。
- 新建独立 Stage15 契约，未来时间跨度与旧 Foot .1s×10 不同，不将旧 checkpoint 强行恢复为新任务。

## 归一化与可复现拆分

- 拟合帧数 **11755**，时间块验证帧 **1300**；训练集含原状态及两组固定seed的root XY/yaw扰动，共 **35265** 组。
- 修复版蒸馏使用walk1训练clip全部有效区间，并按1s块留出10%验证数据。0/60/120s窗口大部分帧已用于离线蒸馏与训练集Norm拟合；这次独立性指新的真实Newton闭环rollout，**不是未见时间窗或未见clip泛化**。没有在Eval阶段改Norm。
- GoalNorm固定训练集均值/方差，标准差下限：位置0.05m、Rotation6D每通道0.15、contact/phase0.25、时间0.10s。ObsNorm逐张量复制Teacher并固定。Eval不重新归一化、不裁剪输入。
- GoalNorm SHA256：`740097d2995f6beb2f4d2938d4bc2ffd470bc70b7f8d07b0409b70840b616743`；训练/验证最大绝对归一化输入 **12.009/3.778**。精确帧索引、seed、增强和Norm均已保存：[logs/s54_stage15_20261106_172329_bounded_b/phase_a/normalization.json](/home/rm/workspace/ljy/Foothold_with_SuperMimic/logs/s54_stage15_20261106_172329_bounded_b/phase_a/normalization.json)。

## Smoke 与 Phase A 实测

- 5项单元测试覆盖时间对齐、无关节reference输入泄露、时间拆分、旋转标准差下限、mask不变性与冻结Decoder的输入梯度。
- 初版独立Newton Smoke曾发现paired WM诊断缺少episode_lengths/xi字段；修复后原版及因果heading版均在新目录重跑通过。短于32步时明确不算32步WM误差。早期失败日志保留在 `logs/s54_stage15_20261106_165359/smoke/newton.log`。
- Phase A：**4000**次更新、batch=512、Adam lr=0.0003、latent MSE；共 **2,048,000** 次离线样本呈现（不是新增Newton transitions）。耗时 **51.67s**，**77.42 updates/s**。
- 最优时间块验证checkpoint为 iteration **625**：MSE **0.148262**，cosine **0.8464**，abs(z)>.95比例 **6.96%**。Teacher验证集对应比例 **13.45%**；饱和下降本身不证明技能成功。
- 最后train/val MSE **0.008419/0.160848**。训练误差继续下降而验证误差未同步下降，存在时间泛化差距；使用验证最优而非最后权重做Newton验收。
- 所有Phase A冻结/Norm/梯度权限断言均通过：[logs/s54_stage15_20261106_172329_bounded_b/phase_a/phase_a_audit.json](/home/rm/workspace/ljy/Foothold_with_SuperMimic/logs/s54_stage15_20261106_172329_bounded_b/phase_a/phase_a_audit.json)。
- Phase A最佳latent checkpoint：[logs/s54_stage15_20261106_172329_bounded_b/phase_a/phase_a_best.pt](/home/rm/workspace/ljy/Foothold_with_SuperMimic/logs/s54_stage15_20261106_172329_bounded_b/phase_a/phase_a_best.pt)；末次：[logs/s54_stage15_20261106_172329_bounded_b/phase_a/phase_a_last.pt](/home/rm/workspace/ljy/Foothold_with_SuperMimic/logs/s54_stage15_20261106_172329_bounded_b/phase_a/phase_a_last.pt)。最佳latent不等于合格Walking controller。

Phase B接口Smoke：RTX3090，16env，H=32/hist=10/WMwindow=8，672真实transitions、2次WM更新、2次E_f Actor更新、16次Critic minibatch更新；峰值Torch 1311.4MiB。首次latent grad=0.016215，E_f改变量=0.017846，D/actor期间WM改变量=0。所有权限与归一化断言通过。此模型只是工程Smoke，**不是Phase B技能实验**。

## 独立 Newton 结果

固定seed54、无观测/动作噪声和DR；每个起点三种控制器使用相同完整物理状态（不是中立站姿）。原Motion与Foot物理终止条件取并集，外加nonfinite。正常10s Timeout计完整存活；物理终止之后不补奖励。

NR=实际累计原MotionReward / (请求500步×原每步理论上限0.1)，并非前缀平均奖励。所有RMSE、接触及动作统计只覆盖实际执行前缀；不同前缀不能直接当同等时长质量比较。

| Controller | 起点/s | 执行/s | Stop | Motion NR | 足世界RMSE/m | Root roll/pitch最大/deg | 上身RMSE/rad | 接触准确率 | Action RMS / clip |
| --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | --- |
| Teacher Motion | 0 | 10.00 | timeout | 0.8706 | 0.0821 | 5.04/5.07 | 0.4757 | 0.909 | 2.533 / 0.000 |
| Teacher Motion | 60 | 10.00 | timeout | 0.8793 | 0.2112 | 6.92/9.45 | 0.1942 | 0.831 | 2.552 / 0.000 |
| Teacher Motion | 120 | 10.00 | timeout | 0.9310 | 0.0791 | 5.93/4.33 | 0.2012 | 0.864 | 3.037 / 0.000 |
| Phase A | 0 | 10.00 | timeout | 0.7589 | 0.1994 | 5.94/6.62 | 0.4618 | 0.869 | 2.418 / 0.000 |
| Phase A | 60 | 10.00 | timeout | 0.7189 | 0.6363 | 5.38/6.34 | 0.2269 | 0.847 | 2.335 / 0.000 |
| Phase A | 120 | 10.00 | timeout | 0.7909 | 0.2434 | 4.33/6.44 | 0.3051 | 0.832 | 2.727 / 0.000 |
| Phase B | 0 | 10.00 | timeout | 0.7010 | 0.3843 | 5.30/12.72 | 0.4820 | 0.861 | 2.469 / 0.000 |
| Phase B | 60 | 10.00 | timeout | 0.6417 | 1.4951 | 8.62/12.55 | 0.2594 | 0.785 | 2.467 / 0.000 |
| Phase B | 120 | 10.00 | timeout | 0.7547 | 0.3152 | 5.18/9.76 | 0.2898 | 0.834 | 2.814 / 0.000 |
| Teacher-z + Foot Obs diagnostic | 0 | 10.00 | timeout | 0.8470 | 0.0954 | 5.19/5.09 | 0.4766 | 0.888 | 2.613 / 0.000 |
| Teacher-z + Foot Obs diagnostic | 60 | 10.00 | timeout | 0.7620 | 0.6579 | 8.01/6.15 | 0.2009 | 0.806 | 2.448 / 0.000 |
| Teacher-z + Foot Obs diagnostic | 120 | 10.00 | timeout | 0.9080 | 0.0936 | 4.85/6.31 | 0.2053 | 0.825 | 3.087 / 0.000 |

接触：真实Newton末子步Fz>1N vs reference估计；sliding为承载足link XY速度代理，不是接触点滑移真值。Touchdown误差和角速度/加速度等逐窗口统计均保留在JSON，接触准确率不等价“自然摆臂”。

| 分支 | 10s存活 | 运动质量Gate |
| --- | --- | --- |
| Teacher | 3/3 | 对照，不宣称完美足轨迹精度 |
| Phase A | 3/3 | 0/3 |
| Phase B | 3/3 | 0/3 |

预注册门槛：三个窗口均完整10s；相对同窗Teacher NR下降≤0.10、足RMSE增加≤0.05m、上身RMSE增加≤0.15rad。所有窗口满足才进入正式B；该工程门槛不等于独立数据泛化认证。

## WM 配对误差

| 分支 | 起点/s | H | Height RMSE/m | Pitch RMSE/deg | Foot FK RMSE/m | Reward Bias/normalized |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| eval_a/motion | 0 | 8 | 0.00046 | 0.057 | 0.00485 | 0.0005 |
| eval_a/motion | 0 | 16 | 0.00074 | 0.354 | 0.00723 | 0.0006 |
| eval_a/motion | 0 | 32 | 0.00195 | 1.091 | 0.00914 | -0.0000 |
| eval_a/motion | 60 | 8 | 0.00036 | 0.093 | 0.00359 | 0.0003 |
| eval_a/motion | 60 | 16 | 0.00070 | 0.243 | 0.00551 | 0.0004 |
| eval_a/motion | 60 | 32 | 0.00140 | 0.681 | 0.00965 | 0.0002 |
| eval_a/motion | 120 | 8 | 0.00041 | 0.087 | 0.00342 | 0.0007 |
| eval_a/motion | 120 | 16 | 0.00097 | 0.358 | 0.00720 | 0.0005 |
| eval_a/motion | 120 | 32 | 0.00149 | 0.951 | 0.00833 | 0.0001 |
| eval_a/foot | 0 | 8 | 0.00097 | 0.182 | 0.00592 | -0.0002 |
| eval_a/foot | 0 | 16 | 0.00111 | 0.616 | 0.00798 | 0.0001 |
| eval_a/foot | 0 | 32 | 0.00163 | 0.826 | 0.01196 | -0.0002 |
| eval_a/foot | 60 | 8 | 0.00054 | 0.149 | 0.00934 | 0.0003 |
| eval_a/foot | 60 | 16 | 0.00113 | 0.741 | 0.01445 | -0.0001 |
| eval_a/foot | 60 | 32 | 0.00165 | 1.367 | 0.01966 | -0.0010 |
| eval_a/foot | 120 | 8 | 0.00034 | 0.098 | 0.00378 | 0.0003 |
| eval_a/foot | 120 | 16 | 0.00075 | 0.399 | 0.00791 | -0.0002 |
| eval_a/foot | 120 | 32 | 0.00107 | 0.696 | 0.01069 | -0.0006 |
| eval_b/foot | 0 | 8 | 0.00069 | 0.171 | 0.00538 | 0.0002 |
| eval_b/foot | 0 | 16 | 0.00150 | 0.443 | 0.01361 | 0.0002 |
| eval_b/foot | 0 | 32 | 0.00398 | 1.789 | 0.03711 | -0.0011 |
| eval_b/foot | 60 | 8 | 0.00041 | 0.198 | 0.00602 | 0.0009 |
| eval_b/foot | 60 | 16 | 0.00147 | 0.452 | 0.00933 | 0.0007 |
| eval_b/foot | 60 | 32 | 0.00128 | 1.033 | 0.01843 | 0.0003 |
| eval_b/foot | 120 | 8 | 0.00062 | 0.099 | 0.00463 | 0.0005 |
| eval_b/foot | 120 | 16 | 0.00055 | 0.434 | 0.00639 | 0.0001 |
| eval_b/foot | 120 | 32 | 0.00168 | 1.121 | 0.00969 | -0.0003 |
| diagnostic_masked_oracle/foot | 0 | 8 | 0.00047 | 0.079 | 0.00374 | 0.0014 |
| diagnostic_masked_oracle/foot | 0 | 16 | 0.00062 | 0.270 | 0.00677 | 0.0013 |
| diagnostic_masked_oracle/foot | 0 | 32 | 0.00160 | 1.081 | 0.01193 | 0.0005 |
| diagnostic_masked_oracle/foot | 60 | 8 | 0.00078 | 0.130 | 0.00656 | 0.0004 |
| diagnostic_masked_oracle/foot | 60 | 16 | 0.00084 | 0.495 | 0.00934 | 0.0011 |
| diagnostic_masked_oracle/foot | 60 | 32 | 0.00163 | 1.282 | 0.01686 | 0.0005 |
| diagnostic_masked_oracle/foot | 120 | 8 | 0.00050 | 0.099 | 0.00382 | 0.0013 |
| diagnostic_masked_oracle/foot | 120 | 16 | 0.00062 | 0.303 | 0.00669 | 0.0004 |
| diagnostic_masked_oracle/foot | 120 | 32 | 0.00188 | 0.925 | 0.01222 | -0.0004 |

每次预测从同一真实Newton状态开始，复用相同真实执行动作；所有H使用同一组完整32步窗口，不含Critic/bootstrap，不跨物理终止。1/4步等完整细节见评估JSON。WM误差小不能替代Foot Policy的闭环验收。

## Phase B 设置与实际执行状态

- 实现沿用原Motion reward+BPTT+imagined lambda-return Critic；独立365D Critic随机初始化（Teacher Motion Critic是914D，禁止强行迁移）。Actor优化器仅持有E_f；D/E_m冻结。WM监督更新与Actor分开，Actor上下文内冻结WM参数且保留状态/动作输入梯度。无额外Prior、latent loss或Foot Reward。
- 配置候选：256env，rollout_steps=5，buffer=256步/环境，WMbatch=512×4，Actorbatch=128，H32（想象反传长度，不是Newton rollout长度）；前100iterations仅WM。
- B候选预算：最多400iterations / 600s。lr WM/Actor/Critic=5e-05/1e-05/0.0001；gamma/lam=0.99/0.95，Critic2epochs×4minibatches，原Motion reward系数与动作clamp±20不变。
- checkpoint原子写入，含Actor/Teacher/WM/独立Critic/各Optimizer/Norm/RNG/计数。Replay不完整序列化：恢复时从新reference物理状态填充warmup_steps，再20次WM-only恢复；不声称Newton接触求解器bitwise连续恢复。
- 256env配置已在本次有界B实测，资源与吞吐见下文；更长训练、多seed稳定性与独立clip泛化尚未验证。

## 结论、诊断与下一步

**A_QUALITY_GATE_FAILED**：Phase A未达到原质量Gate。完成接口与分布诊断后，执行有界B修复（最多400次/600s），没有将A失败改为成功，没有修改最终质量Gate或物理终止。
为定位观测接口，额外将**真实Teacher Motion latent**送入同一个冻结D，使用相同Foot ObsNorm及当前观测适配，完成三个10s窗口诊断；完整存活 **3/3**。该分支调用Motion Encoder，**只用于诊断，绝不计Foot成功**。
该结果可区分mask/Decoder兼容性与E_f拟合问题，但未构成单因素完备因果消融；逐窗Gate与误差见JSON。
下一步优先验收Teacher-z + 因果朝向接口，尤其60s窗口。先检查基于Foot位置/旋转/contact/phase的朝向估计能否达到原root朝向接口精度，再考虑扩大E_f训练；不能将完整Motion root reference重新输入Foot Policy。

### 有界 Phase B 实测

实际Newton transitions=520,192；WM updates=1600；Actor updates=300；Critic minibatch updates=2400；wall=472.39s；峰值Torch=1686.4MiB。Teacher/Decoder冻结、WM仅监督更新、两套Norm固定、E_f实际更新的全部断言：{'teacher_and_decoder_unchanged': True, 'goal_norm_unchanged': True, 'obs_norm_matches_teacher': True, 'foot_encoder_updated': True, 'wm_supervised_updated': True}。
Best B为本次唯一独立Newton评估候选；排序不等于通过质量Gate。Checkpoint：[logs/s54_stage15_20261106_172329_bounded_b/phase_b/phase_b_best.pt](/home/rm/workspace/ljy/Foothold_with_SuperMimic/logs/s54_stage15_20261106_172329_bounded_b/phase_b/phase_b_best.pt)。
Actor阶段采样Collection FPS中位数=7884.1；迭代速度中位数=0.91it/s。FPS只计真实Newton transitions；H32想象样本不计入。
nvidia-smi训练期总显存峰值=2695MiB（含桌面/上下文；启动前桌面约499MiB），GPU利用率峰值=42%。
最终结论：**BLOCKED_QUALITY**。Foot已能存活并产生Walking，但仍未达到接近Teacher的轨迹/全身质量要求；不启动Stage2或正式长训。
独立Walking clip泛化：**NOT EVALUATED**。当前仅有合格walk1_subject1；dance holdout不当作Walking泛化结果。单seed、3个时间窗不能代表全部Walking能力。

## 产物、SwanLab与命令

- 配置快照：[logs/s54_stage15_20261106_172329_bounded_b/profile.json](/home/rm/workspace/ljy/Foothold_with_SuperMimic/logs/s54_stage15_20261106_172329_bounded_b/profile.json)；执行命令：[logs/s54_stage15_20261106_172329_bounded_b/commands.jsonl](/home/rm/workspace/ljy/Foothold_with_SuperMimic/logs/s54_stage15_20261106_172329_bounded_b/commands.jsonl)；完整stdout：[logs/s54_stage15_20261106_172329_bounded_b/stdout.log](/home/rm/workspace/ljy/Foothold_with_SuperMimic/logs/s54_stage15_20261106_172329_bounded_b/stdout.log)。
- eval_a完整指标与trace路径：[logs/s54_stage15_20261106_172329_bounded_b/eval_a/summary.json](/home/rm/workspace/ljy/Foothold_with_SuperMimic/logs/s54_stage15_20261106_172329_bounded_b/eval_a/summary.json)。
- eval_b完整指标与trace路径：[logs/s54_stage15_20261106_172329_bounded_b/eval_b/summary.json](/home/rm/workspace/ljy/Foothold_with_SuperMimic/logs/s54_stage15_20261106_172329_bounded_b/eval_b/summary.json)。
- diagnostic_masked_oracle完整指标与trace路径：[logs/s54_stage15_20261106_172329_bounded_b/diagnostic_masked_oracle/summary.json](/home/rm/workspace/ljy/Foothold_with_SuperMimic/logs/s54_stage15_20261106_172329_bounded_b/diagnostic_masked_oracle/summary.json)。
- SwanLab：**VERIFIED**；[https://swanlab.cn/@yagami_light/Foothold_with_SuperMimic_Core/runs/83941f5667cc468f/chart](https://swanlab.cn/@yagami_light/Foothold_with_SuperMimic_Core/runs/83941f5667cc468f/chart)。仅核心参数/指标；全量逐窗口/逐horizon明细保存在本地。
- 对比视频：[logs/s54_stage15_20261106_172329_bounded_b/videos/teacher_phase_a_phase_b_all_windows.mp4](/home/rm/workspace/ljy/Foothold_with_SuperMimic/logs/s54_stage15_20261106_172329_bounded_b/videos/teacher_phase_a_phase_b_all_windows.mp4)（青色reference仅可视化，物理终止后该侧冻结并标红）。
- 对比视频：[logs/s54_stage15_20261106_172329_bounded_b/videos/teacher_phase_a_phase_b_offset0.mp4](/home/rm/workspace/ljy/Foothold_with_SuperMimic/logs/s54_stage15_20261106_172329_bounded_b/videos/teacher_phase_a_phase_b_offset0.mp4)（青色reference仅可视化，物理终止后该侧冻结并标红）。
- 对比视频：[logs/s54_stage15_20261106_172329_bounded_b/videos/teacher_phase_a_phase_b_offset120.mp4](/home/rm/workspace/ljy/Foothold_with_SuperMimic/logs/s54_stage15_20261106_172329_bounded_b/videos/teacher_phase_a_phase_b_offset120.mp4)（青色reference仅可视化，物理终止后该侧冻结并标红）。
- 对比视频：[logs/s54_stage15_20261106_172329_bounded_b/videos/teacher_phase_a_phase_b_offset60.mp4](/home/rm/workspace/ljy/Foothold_with_SuperMimic/logs/s54_stage15_20261106_172329_bounded_b/videos/teacher_phase_a_phase_b_offset60.mp4)（青色reference仅可视化，物理终止后该侧冻结并标红）。

## 核心比较与证据边界

| 分支 | 10s存活 | 平均Motion NR | 平均足RMSE/m | 平均上身RMSE/rad |
| --- | --- | ---: | ---: | ---: |
| Teacher | 3/3 | 0.8936 | 0.1241 | 0.2904 |
| Phase A | 3/3 | 0.7562 | 0.3597 | 0.3312 |
| Phase B | 3/3 | 0.6991 | 0.7315 | 0.3437 |

**已验证**：A和B确实能连续走10s；其三窗实际根部行程分别约3.95–5.86m、3.04–5.83m，因此不是只在原地站住。但B相对A的平均Motion NR下降、足轨迹误差增大；不能把Critic loss下降或想象reward平稳称为控制质量改善。
WM在观察到的同动作32步配对窗口上没有出现巨大正Reward Bias；这不能证明其Actor梯度处处准确，也不足以将B退化单独归因于WM或Critic。

| 分支 | 起点/s | 闭环latent MSE | Cosine | Rotation6D最大abs归一化 | 朝向目标估计误差RMSE/deg | 动作步间Delta RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| eval_a | 0 | 0.2160 | 0.7750 | 2.842 | 2.72 | 0.1476 |
| eval_a | 60 | 0.1453 | 0.8361 | 2.869 | 14.15 | 0.1297 |
| eval_a | 120 | 0.1502 | 0.8348 | 2.495 | 5.17 | 0.1312 |
| eval_b | 0 | 0.4079 | 0.6108 | 2.843 | 2.72 | 0.1445 |
| eval_b | 60 | 0.3537 | 0.6207 | 2.884 | 14.15 | 0.1291 |
| eval_b | 120 | 0.2785 | 0.7177 | 2.493 | 5.17 | 0.1208 |

Latent对齐是记录的bf16 Foot latent与同帧CPU fp32 Teacher标签的后验评分；未进入Actor输入或更新。朝向误差使用完整reference仅做独立评分，动作Delta为相邻控制步command差的RMS，单位为原action单位。
**已验证**：Rotation6D归一化离群值已明显收敛至约2.5–2.9；当前B退化不是原99.61旋转异常的简单重复。B的闭环latent MSE比A增大，足位置归一化值在60s窗口又因位置漂移增至约12。
**已验证的剩余接口问题**：因果heading的目标yaw在60s窗口相对Teacher root yaw有约14.15deg RMSE（其他两窗约2.72/5.17deg）；该差值独立于A/B Encoder权重。连真实Teacher-z oracle在此窗也有0.658m足RMSE，说明单纯提高latent拟合不能保证恢复Teacher的朝向反馈。
**尚未验证的机制**：B的bootstrap梯度、短时模型误差、回报权重与Decoder可控性各贡献多少；离线latent在1s时间块留出上的泛化差距有多少来自足目标歧义。当前没有单因素因果结论。
最小下一步实验：保留固定E_m/D，先做有界Teacher-z oracle与因果朝向估计的逐窗对照；目标是先通过全部三窗接口Gate。随后才对奖励梯度/Bootstrap梯度做同Replay、同seed的小步数诊断，避免直接延长B。
详细后验数据：[logs/s54_stage15_20261106_172329_bounded_b/latent_alignment_posthoc.json](/home/rm/workspace/ljy/Foothold_with_SuperMimic/logs/s54_stage15_20261106_172329_bounded_b/latent_alignment_posthoc.json)。

本次B目录通过只读符号链接复用heading修复版的A、Smoke和对照；没有重新计算4000次A更新，也没有覆盖来源。来源记录：[logs/s54_stage15_20261106_172329_bounded_b/baseline_source.json](/home/rm/workspace/ljy/Foothold_with_SuperMimic/logs/s54_stage15_20261106_172329_bounded_b/baseline_source.json)。
A的best按时间块latent验证MSE选择；B仅评估末次一个候选，best_B是该候选的明确命名，**不是全训练轨迹或历史所有checkpoint的全局最佳**。中间B checkpoint未独立评估，不能推断其性能。
三种Policy在0/60/120s的初始71D物理状态最大差均为0，Teacher SHA前后相同：[logs/s54_stage15_20261106_172329_bounded_b/completion_audit.json](/home/rm/workspace/ljy/Foothold_with_SuperMimic/logs/s54_stage15_20261106_172329_bounded_b/completion_audit.json)。不声称Newton接触求解器内部warm state逐位一致。

## 恢复 Smoke

另在独立Smoke目录实际测试恢复，没有改写正式A/B checkpoint：A从iteration3继续到6，Adam step3→6；B从iteration2继续到24，先补Replay/20次WM预热，再2次Actor更新。B Actor/WM/Critic Adam step分别2→4、2→24、16→32，计数与权重持续，Teacher/D及Norm冻结断言通过。
恢复测试不是新增技能评估；新恢复阶段产生的真实transitions单独在[logs/s54_stage15_20261106_172329_bounded_b/resume_verification.json](/home/rm/workspace/ljy/Foothold_with_SuperMimic/logs/s54_stage15_20261106_172329_bounded_b/resume_verification.json)中记录。没有恢复Newton接触内部warm state。
模型/计数/RNG/优化器持久化与重新采样机制已有实测；更长训练中断、电源故障与网络长期中断恢复并未全部复现。

```bash
tmux attach -t s54_stage15_20261106_172329_bounded_b_review  # 本次训练已结束，此session仅查看日志
tail -f logs/s54_stage15_20261106_172329_bounded_b/stdout.log
bash tools/train/run_s54_stage15.sh status s54_stage15_20261106_172329_bounded_b
# 在全新时间戳目录复现实验（GPU空闲时运行）
bash tools/train/run_s54_stage15.sh start "s54_stage15_$(date +%Y%m%d_%H%M%S)_heading" configs/motion_prior/stage15/lafan1_skill_transfer_heading_3090.json
# 完成诊断后，有界B最多400 iterations /600秒（创建新目录）
bash tools/train/run_s54_stage15_bounded_b.sh logs/s54_stage15_20261106_heading_repair
# 仅重渲染已有轨迹，不运行policy
/home/rm/workspace/IsaacLab_3_0/.venv/bin/python tools/visualization/render_s54_stage15.py --run-dir logs/s54_stage15_20261106_172329_bounded_b --offset 0 --out results/s54/stage15_offset0_new.mp4
# CPU缓存重试上传，使用已有稳定实验ID
.venv-swanlab/bin/python tools/ops/sync_s54_stage15.py --run-dir logs/s54_stage15_20261106_172329_bounded_b
```


## 有界修复与原方案对照

原始mask版保留于 `logs/s54_stage15_20261106_165827`：3/3存活10s，但Motion NR为0.6431/0.4543/0.7023、足RMSE为0.6717/1.3209/0.5005m，质量Gate0/3；未启动其正式B。
原mask版的Teacher-z oracle在0/60/120s分别NR约0.855/0.839/0.847，足RMSE约0.126/0.300/0.558m。Oracle调用Motion Encoder，只用于确定mask会退化heading反馈；不算Foot成功。
修复版使用未来足旋转的contact加权圆均值估计期望yaw，并以身体重力确定实际tilt；固定参考tilt与foot-yaw偏置只在训练集标定。每一步heading输入由**相同FootGoal**生成，历史按时刻保存；不读逐帧完整reference root姿态、不调用Motion Encoder。
训练集校准常量：{"training_frame_count": 11755, "source": "walk1 training frames only; 1s validation blocks excluded", "fit": "circular mean contact-weighted future foot yaw; mean reference roll/pitch", "seed": 54}；bias/roll/pitch=[-0.06616310775279999, 0.006717108655720949, 0.059499938040971756] rad。
同时扩大离线root扰动覆盖为XY/yaw标准差(0.10m,0.20rad)、(0.25m,0.50rad)，并覆盖指定训练clip。Teacher latent标签仍对应原同一帧；只训练E_f。
这是有界功能修复，heading适配、数据覆盖和增强同时改变，**不能将改善单独归因于某一个因素**。原时间留出结果和全部历史权重不变。
原方案报告：[logs/s54_stage15_20261106_165827/STAGE15_REPORT.md](/home/rm/workspace/ljy/Foothold_with_SuperMimic/logs/s54_stage15_20261106_165827/STAGE15_REPORT.md)。
