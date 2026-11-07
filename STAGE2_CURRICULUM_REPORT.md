# Stage 2 S54 Foot Trajectory Curriculum

更新：2026-11-07T00:05:23+08:00。独立目录：`logs/s54_stage2_curriculum_20261106_191829`。

状态：**STOPPED**；当前 iteration=8601；课程=C0_balance_short；累计进程 wall time=16246.7s / 28800s。

本报告不把 Smoke Test、5s 存活或 imagined reward 当作 10s 轨迹跟踪成功。正式训练达到时间上限或看门狗条件后安全保存；尚未达到的阶段没有能力结论。

## 初始化与接口

- E_f：`logs/s54_stage15_20261106_heading_repair/phase_a/phase_a_best.pt`，Phase A best iteration 625；**未使用 Phase B**。
- Shared Decoder / WM / ObsNorm 来自对应 Teacher；D 与 Policy 两个 Norm 完全冻结；独立 Foot Critic 与三个 Adam 从零建立。WM 继续真实 Newton Replay 学习，Actor BPTT 期间冻结 WM 参数但保留输入梯度。
- Policy 仅输入 generated Foot Goal230 + 10帧本体观测。没有 MotionLib、完整 Motion Reference 或 Motion Encoder 参与前向。Teacher checkpoint 中 Motion Encoder 权重仅用于审计 Decoder 来源。
- 每帧23D：左足 p3 / Rotation6 / Contact / Phase + 右足同样11D + Δt1。共10帧，0.02–0.20s。位置与姿态在当前 root-yaw 坐标系，足部使用与 Phase A 相同的 ankle link 原点。Rotation6 为旋转矩阵前两列的 row-major 3×2 展开。
- Heading 使用 Phase A 相同的 contact-weighted foot-yaw 推算，保留校准 bias=-0.0661631078rad、roll=0.0067171087rad、pitch=0.0594999380rad；未增加独立 Heading 优化。
- GoalNorm/ObsNorm 按 Phase A 原样复制，无在线重估、截断或评估专用调整。
- 脚步频率统一为**每秒交替落脚次数**，完整左右周期频率为其一半。0Hz=双脚站立，不做除法；ℓf 为理想前进速度。Yaw 命令为 rad/s。

## 课程与预算

最多8h（包括正式进程初始化、评估、保存；停止前预留90s），50000 iterations 上限。以下7.4h是建议分配，余0.6h留给评估/保存；**时间到达不强制晋级**，失败可以停留或回退。

| 阶段 | 建议h | Standing / Short / Medium / Long / Turn / Curve 基础权重 | 步长上限m | 频率Hz | Yaw绝对值上限rad/s | 摆高m |
|---|---:|---|---:|---|---:|---:|
| C0 | 1.4 | 25% / 75% / 0% / 0% / 0% / 0% | 0.055 | [0.85, 1.05] | 0 | 0.018 |
| C1 | 1.8 | 15% / 35% / 40% / 10% / 0% / 0% | 0.2 | [0.8, 1.5] | 0 | 0.025 |
| C2 | 1.2 | 15% / 20% / 10% / 5% / 50% / 0% | 0.22 | [0.8, 1.5] | 0.25 | 0.025 |
| C3 | 1.5 | 15% / 15% / 10% / 5% / 20% / 35% | 0.25 | [0.8, 1.6] | 0.3 | 0.03 |
| C4 | 1.5 | 15% / 20% / 20% / 10% / 15% / 20% | 0.3 | [0.7, 1.8] | 0.35 | 0.035 |

短/中/长分桶约为 .02–.08 / .08–.16 / .16–阶段上限 m；C0 实际从 .02–.033m 开始，按成功逐渐到 .055m。C0 小步中10%为零前进、非零频率的交替重心转移。困难任务与高频分桶受成功率权重调节，长步频率≤1.15Hz；左右转向等概率。动态调整后的**实际比例不等于表中基础权重**，完整采样/拒绝统计保存在本地 JSONL。
每脚摆动 .72/f、双支撑 .28/f；场景命令的 minimum swing=.32s、DS=.12s。使用现有 C2 Quintic + 足部 SLERP，目标在整个 Episode 中保持不可变，C4 可在完整左右步对和双支撑边界切换；WM Replay 同时保存 command ID 和步数，不跨 reset 接续。
联合作用域检查：速度≤.45m/s、same-foot stride≤.65m、每步 |yaw_delta|≤.25rad、swing速度≤1.8m/s、加速度≤12m/s²。`.4m/2Hz` 等组合被拒绝；这是轨迹包络筛选，**不等于已证明动力学可执行**。大步/高速/转向只有通过真实 Newton 门槛后才能声称掌握。

晋级门槛：当前阶段全部独立代表任务 success≥85%、survival≥90%；Foot RMSE≤.06m（standing≤.04m）、落脚误差≤.07m、root |roll/pitch|≤18°、height偏差≤.10m。Walking 另需实际前进≥理想值60%、速度误差限制、≥2次真实加载接触落脚和摆脚抬高，turn另需正确方向与yaw跟踪。
每阶段至少900s，连续通过至少2次评估，且10s Full确认通过才晋级。连续失败且站立下降可回退。Fast=5s、4env、固定seed8541，每900s串行复用同一Newton scene；Full=10s，晋级确认或结尾有足够余时才运行。正式周期评估预算约5%，初次基线单独记录。Physical termination 优先于 timeout；Goal tracking failure 独立统计。

## 3090配置与测试

| 参数 | 值 |
|---|---:|
| `num_envs` | 256 |
| `hist` | 10 |
| `horizon` | 32 |
| `wm_window` | 8 |
| `goal_frames` | 10 |
| `goal_dt` | 0.02 |
| `rollout_steps` | 5 |
| `buffer_steps` | 512 |
| `warmup_steps` | 40 |
| `wm_only_iters` | 500 |
| `resume_wm_iters` | 100 |
| `wm_batch` | 512 |
| `wm_batches` | 4 |
| `pi_batch` | 128 |
| `lr_wm` | 5e-05 |
| `lr_actor` | 1e-05 |
| `lr_critic` | 0.0001 |
| `gamma` | 0.99 |
| `lam` | 0.95 |
| `action_noise` | 0.02 |
| `critic_epochs` | 2 |
| `critic_minibatches` | 4 |

H32为0.64s可微 imagined BPTT；rollout_steps=5 是每环境每轮真实采样0.10s，两者不同。WM:4×512个8步窗口/轮；Actor128 starts/轮，Critic2epochs×4minibatches。Adam学习率固定，Actor前300次更新线性升至1e-5；前500迭代只训练WM。BF16 MLP，几何/状态积分保持FP32。DR关闭，原物理模型/PD/关节顺序不变。

| Smoke | GPU | Newton transitions | Actor/WM updates | peak Torch MiB | 技术断言 |
|---|---|---:|---|---:|---|
| smoke16 | NVIDIA GeForce RTX 3090 | 1920 | 6/32 | 1625.8 | {'decoder_unchanged': True, 'policy_normalizers_unchanged': True, 'foot_encoder_updated': True, 'wm_updated': True, 'foot_actor_nonzero_gradient': True, 'actor_did_not_update_wm': True} |
| smoke64 | NVIDIA GeForce RTX 3090 | 7680 | 6/32 | 1651.2 | {'decoder_unchanged': True, 'policy_normalizers_unchanged': True, 'foot_encoder_updated': True, 'wm_updated': True, 'foot_actor_nonzero_gradient': True, 'actor_did_not_update_wm': True} |

Checkpoint恢复 Smoke：`s54_stage2_curriculum_20261106_191829/smoke64/resume_smoke_audit.json`：PASS，实际再次更新WM/E_f，D与Norm不变。

smoke16 实测平均采样FPS=378；Actor阶段迭代=0.64it/s（短Smoke，包含首次审计/周期保存抖动，不是8h平均）。证据：`logs/s54_stage2_curriculum_20261106_191829/smoke16/train_metrics.jsonl`。

smoke64 实测平均采样FPS=1519；Actor阶段迭代=0.61it/s（短Smoke，包含首次审计/周期保存抖动，不是8h平均）。证据：`logs/s54_stage2_curriculum_20261106_191829/smoke64/train_metrics.jsonl`。

归一化审计：5×64条计划；Rotation6D最大z=1.349；全Goal最大z=2.734；|z|>6比例为0。Latent saturation约15.5%–18.4%。根沿支撑中心的输入审计，非真实闭环成绩。`logs/s54_stage2_curriculum_20261106_191829/normalization_audit.json`。

Phase A SHA256：`8a7230988dc93e6795ab79a59c0d0b36aaf82712094a8c5d9945f9a8216736b5`；Teacher SHA256：`ef5671d4e223ab004c1f69837fffca6f461ca1c2c661c111b5afc2f82e6e5544`。迁移证明：`transfer_report.json`，完整契约：`normalization_audit.json`。

原 Foot Reward 保持不变：swing/stance position各1，velocity各.25，foot orientation .25，touchdown .5，root upright1、height .5、balanceXY .5；action-rate −.1、joint-limit −2、posture −.05、joint-velocity −.001，乘control_dt=.02。inverse-quadratic tracking质量函数，标准差与原 FootTrackingCfg 相同。
物理终止保持 root-height 偏差>0.25m、1−Rzz>0.5或非有限状态；成功门槛比物理终止严格，正常500步timeout不当作跌倒。

正式256env最新记录：iteration=8600；采样FPS=5094；iteration/s=0.56；Torch peak=1745.1MiB；WM loss=0.00051、err8=0.00075；Actor updates=8100、WM updates=34400。`train_metrics.jsonl`中的newton仅计训练采样，newton_eval单列评估中所有实际模拟环境步。

正式进程首次开始：2026-11-06T19:34:08+08:00。不中断时8h上限约为 2026-11-07T03:34:08+08:00，看门狗可提前结束。目录时间戳对应准备开始，不是8h预算起点。

GPU只读监控（每30s抽样）：总显存24,576MiB，已观察最大占用=5452MiB，采样平均利用率=44.5%。这是当前阶段观测，不是完整8h统计；`gpu_telemetry.jsonl`。未为了占满显存而盲目增加batch。

## 实际独立Newton结果

| 来源 | 阶段 | 任务 | 时长s | Survival | Success | Foot RMSE m | 物理终止 | Root roll/pitch max° |
|---|---|---|---:|---:|---:|---:|---:|---|
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/baseline_0/summary.json` | C0 | standing | 5.0 | 1.00 | 0.00 | 0.5756 | 0.00 | 7.5/7.6 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/baseline_0/summary.json` | C0 | weight_shift | 5.0 | 1.00 | 0.00 | 0.9479 | 0.00 | 8.0/7.7 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/baseline_0/summary.json` | C0 | short_walk | 5.0 | 1.00 | 0.00 | 1.0842 | 0.00 | 8.0/9.0 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_1513_fast/summary.json` | C0 | standing | 5.0 | 1.00 | 0.00 | 0.1622 | 0.00 | 4.5/7.6 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_1513_fast/summary.json` | C0 | weight_shift | 5.0 | 1.00 | 0.00 | 0.1738 | 0.00 | 5.4/7.7 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_1513_fast/summary.json` | C0 | short_walk | 5.0 | 1.00 | 0.00 | 0.2209 | 0.00 | 5.4/8.6 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_2051_fast/summary.json` | C0 | standing | 5.0 | 1.00 | 0.00 | 0.1296 | 0.00 | 3.4/7.9 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_2051_fast/summary.json` | C0 | weight_shift | 5.0 | 1.00 | 0.00 | 0.1197 | 0.00 | 4.6/8.0 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_2051_fast/summary.json` | C0 | short_walk | 5.0 | 1.00 | 0.00 | 0.1401 | 0.00 | 4.6/8.6 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_2570_fast/summary.json` | C0 | standing | 5.0 | 1.00 | 0.00 | 0.0806 | 0.00 | 6.7/7.4 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_2570_fast/summary.json` | C0 | weight_shift | 5.0 | 1.00 | 0.00 | 0.0875 | 0.00 | 6.7/10.4 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_2570_fast/summary.json` | C0 | short_walk | 5.0 | 1.00 | 0.00 | 0.1031 | 0.00 | 6.7/9.0 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_3023_fast/summary.json` | C0 | standing | 5.0 | 1.00 | 0.00 | 0.0784 | 0.00 | 5.0/9.0 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_3023_fast/summary.json` | C0 | weight_shift | 5.0 | 1.00 | 0.00 | 0.0778 | 0.00 | 5.0/11.3 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_3023_fast/summary.json` | C0 | short_walk | 5.0 | 1.00 | 0.00 | 0.0962 | 0.00 | 4.9/10.3 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_3438_fast/summary.json` | C0 | standing | 5.0 | 1.00 | 0.00 | 0.0531 | 0.00 | 4.3/7.5 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_3438_fast/summary.json` | C0 | weight_shift | 5.0 | 1.00 | 0.00 | 0.0613 | 0.00 | 4.3/8.7 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_3438_fast/summary.json` | C0 | short_walk | 5.0 | 1.00 | 0.00 | 0.0859 | 0.00 | 4.2/9.6 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_3889_fast/summary.json` | C0 | standing | 5.0 | 1.00 | 0.00 | 0.0767 | 0.00 | 4.5/6.6 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_3889_fast/summary.json` | C0 | weight_shift | 5.0 | 1.00 | 0.00 | 0.0568 | 0.00 | 4.4/7.2 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_3889_fast/summary.json` | C0 | short_walk | 5.0 | 1.00 | 0.00 | 0.0847 | 0.00 | 4.4/9.2 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_4370_fast/summary.json` | C0 | standing | 5.0 | 1.00 | 0.00 | 0.0578 | 0.00 | 4.6/5.9 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_4370_fast/summary.json` | C0 | weight_shift | 5.0 | 1.00 | 0.00 | 0.0490 | 0.00 | 4.0/6.4 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_4370_fast/summary.json` | C0 | short_walk | 5.0 | 1.00 | 0.00 | 0.0732 | 0.00 | 4.0/8.4 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_4851_fast/summary.json` | C0 | standing | 5.0 | 1.00 | 0.00 | 0.0436 | 0.00 | 3.7/5.9 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_4851_fast/summary.json` | C0 | weight_shift | 5.0 | 1.00 | 0.00 | 0.0397 | 0.00 | 3.6/6.0 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_4851_fast/summary.json` | C0 | short_walk | 5.0 | 1.00 | 0.00 | 0.0635 | 0.00 | 3.5/6.4 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_5339_fast/summary.json` | C0 | standing | 5.0 | 1.00 | 0.00 | 0.0405 | 0.00 | 4.0/5.4 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_5339_fast/summary.json` | C0 | weight_shift | 5.0 | 1.00 | 0.00 | 0.0395 | 0.00 | 3.8/5.3 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_5339_fast/summary.json` | C0 | short_walk | 5.0 | 1.00 | 0.00 | 0.0463 | 0.00 | 3.8/5.7 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_5826_fast/summary.json` | C0 | standing | 5.0 | 1.00 | 0.00 | 0.0389 | 0.00 | 3.7/5.1 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_5826_fast/summary.json` | C0 | weight_shift | 5.0 | 1.00 | 0.00 | 0.0362 | 0.00 | 3.5/5.2 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_5826_fast/summary.json` | C0 | short_walk | 5.0 | 1.00 | 0.00 | 0.0535 | 0.00 | 3.5/5.6 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_6314_fast/summary.json` | C0 | standing | 5.0 | 1.00 | 0.00 | 0.0377 | 0.00 | 3.6/4.6 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_6314_fast/summary.json` | C0 | weight_shift | 5.0 | 1.00 | 0.00 | 0.0359 | 0.00 | 3.5/4.8 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_6314_fast/summary.json` | C0 | short_walk | 5.0 | 1.00 | 0.00 | 0.0486 | 0.00 | 3.5/5.0 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_6799_fast/summary.json` | C0 | standing | 5.0 | 1.00 | 0.00 | 0.0352 | 0.00 | 3.9/4.1 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_6799_fast/summary.json` | C0 | weight_shift | 5.0 | 1.00 | 0.00 | 0.0365 | 0.00 | 3.7/4.9 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_6799_fast/summary.json` | C0 | short_walk | 5.0 | 1.00 | 0.00 | 0.0422 | 0.00 | 3.6/5.0 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_7288_fast/summary.json` | C0 | standing | 5.0 | 1.00 | 0.00 | 0.0364 | 0.00 | 4.1/4.3 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_7288_fast/summary.json` | C0 | weight_shift | 5.0 | 1.00 | 0.00 | 0.0325 | 0.00 | 3.9/4.3 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_7288_fast/summary.json` | C0 | short_walk | 5.0 | 1.00 | 0.00 | 0.0443 | 0.00 | 3.8/4.3 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_7777_fast/summary.json` | C0 | standing | 5.0 | 1.00 | 0.00 | 0.0348 | 0.00 | 3.9/3.9 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_7777_fast/summary.json` | C0 | weight_shift | 5.0 | 1.00 | 0.00 | 0.0354 | 0.00 | 3.8/3.8 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_7777_fast/summary.json` | C0 | short_walk | 5.0 | 1.00 | 0.00 | 0.0409 | 0.00 | 3.7/4.2 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_8264_fast/summary.json` | C0 | standing | 5.0 | 1.00 | 0.00 | 0.0338 | 0.00 | 3.8/4.0 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_8264_fast/summary.json` | C0 | weight_shift | 5.0 | 1.00 | 0.00 | 0.0346 | 0.00 | 3.8/3.4 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_8264_fast/summary.json` | C0 | short_walk | 5.0 | 1.00 | 0.00 | 0.0406 | 0.00 | 3.7/3.7 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_941_fast/summary.json` | C0 | standing | 5.0 | 1.00 | 0.00 | 0.3566 | 0.00 | 7.9/6.9 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_941_fast/summary.json` | C0 | weight_shift | 5.0 | 1.00 | 0.00 | 0.3711 | 0.00 | 8.0/8.1 |
| `logs/s54_stage2_curriculum_20261106_191829/evaluation/it_941_fast/summary.json` | C0 | short_walk | 5.0 | 1.00 | 0.00 | 0.4679 | 0.00 | 7.9/8.3 |
| `logs/s54_stage2_curriculum_20261106_191829/smoke64/evaluation/post_update_12/summary.json` | C0 | standing | 5.0 | 1.00 | 0.00 | 0.5546 | 0.00 | 7.5/7.8 |
| `logs/s54_stage2_curriculum_20261106_191829/smoke64/evaluation/post_update_12/summary.json` | C0 | weight_shift | 5.0 | 1.00 | 0.00 | 0.9341 | 0.00 | 8.1/8.4 |
| `logs/s54_stage2_curriculum_20261106_191829/smoke64/evaluation/post_update_12/summary.json` | C0 | short_walk | 5.0 | 1.00 | 0.00 | 1.0609 | 0.00 | 8.0/7.9 |
| `logs/s54_stage2_curriculum_20261106_191829/smoke64/evaluation/post_update_8/summary.json` | C0 | standing | 5.0 | 1.00 | 0.00 | 0.5540 | 0.00 | 7.9/7.9 |
| `logs/s54_stage2_curriculum_20261106_191829/smoke64/evaluation/post_update_8/summary.json` | C0 | weight_shift | 5.0 | 1.00 | 0.00 | 0.9458 | 0.00 | 8.1/8.5 |
| `logs/s54_stage2_curriculum_20261106_191829/smoke64/evaluation/post_update_8/summary.json` | C0 | short_walk | 5.0 | 1.00 | 0.00 | 1.0663 | 0.00 | 7.9/8.5 |

尚未评估的高阶段能力、10s持续跟踪和全身自然性均为 **NOT EVALUATED**。Phase A原本仅学过LAFAN Walking，不能据此假定已会站立、转弯或大步。已知源Teacher方向近似与新任务Reward分布差异仍是风险。

## Checkpoint、监控与持久化
- `latest.pt`/`last.pt`、每500迭代且最长900s一个周期checkpoint、阶段切换checkpoint。最近4个周期文件，阶段与best单独保留；原子fsync+rename，别名硬链接。完整模型/三个Adam/Norm/课程分布/RNG/全局计数/累计wall保存。
- `best_standing.pt`、`best_walking.pt`、`best_overall.pt`按独立Newton success→survival→低RMSE→return排序，记录5s provisional或10s qualified。存在best文件不意味着已经合格。
- Replay不序列化，恢复后 grounded reset、warmup40真实步、100轮WM-only再恢复Actor；保留训练进度/课程/采样RNG/optimizers。保留环境快照供审计，但不宣称精确恢复Newton接触求解器状态。
- NaN/Inf、缺失/非有限Actor latent梯度立即诊断退出；连续3次独立评估连Standing/Walking 5s存活都失败且运行≥30min时安全停止。CPU看门狗持久记录最佳真实表现，RMSE再改善≥5%或成功/存活率改善才重置进展时钟；1h无任务成功且无实际改善则停止（`monitor_state.json` / `external_watchdog.json`）。训练内另保留相对初始基线的退化检查，连续2次RMSE>基线2倍则停止。不因单次WM loss上升中断。
- SwanLab独立CPU tmux上传，23条核心指标，训练25迭代一次，网络断开本地缓存，最终sync+API读回核验；训练不等待网络。完整命令直方图保存在本地，不扩张云端曲线数量。
- SwanLab：https://swanlab.cn/@yagami_light/Foothold_with_SuperMimic_Core/runs/3fdcb1bf0786485a/chart；上传状态：VERIFIED；已上传数据实际API读回证据：`swanlab/live_cloud_check.json`（若存在）。
- Best metadata：`{"standing": {"score": [0.0, 1.0, -0.03383617103099823, 0.7746261954307556], "evaluation": "/home/rm/workspace/ljy/Foothold_with_SuperMimic/logs/s54_stage2_curriculum_20261106_191829/evaluation/it_8264_fast/summary.json", "seconds": 5.0, "iteration": 8264, "qualified": false}, "walking": {"score": [0.0, 1.0, -0.04063893109560013, 0.7633160352706909], "evaluation": "/home/rm/workspace/ljy/Foothold_with_SuperMimic/logs/s54_stage2_curriculum_20261106_191829/evaluation/it_8264_fast/summary.json", "seconds": 5.0, "iteration": 8264, "qualified": false}, "overall": {"score": [0.0, 1.0, -0.03637110193570455, 0.7711325486501058], "evaluation": "/home/rm/workspace/ljy/Foothold_with_SuperMimic/logs/s54_stage2_curriculum_20261106_191829/evaluation/it_8264_fast/summary.json", "seconds": 5.0, "iteration": 8264, "qualified": false}}`。
- 可用候选路径：`logs/s54_stage2_curriculum_20261106_191829/best_standing.pt`, `logs/s54_stage2_curriculum_20261106_191829/best_walking.pt`, `logs/s54_stage2_curriculum_20261106_191829/best_overall.pt`。

## 运行与恢复命令
```bash
cd /home/rm/workspace/ljy/Foothold_with_SuperMimic
bash tools/train/run_s54_stage2_curriculum.sh status s54_stage2_curriculum_20261106_191829
tmux attach -t s54_stage2_curriculum_20261106_191829
tail -f logs/s54_stage2_curriculum_20261106_191829/stdout.log
bash tools/train/run_s54_stage2_curriculum.sh stop s54_stage2_curriculum_20261106_191829
bash tools/train/run_s54_stage2_curriculum.sh resume s54_stage2_curriculum_20261106_191829
```

SSH断线不影响tmux；Ctrl-b d离开会话，Ctrl-c会请求安全停止。resume使用原run ID和累计8h预算，不重置8h。

视频：`videos/`中为保存的真实Newton轨迹与cyan程序足目标，不调用额外policy rollout，不放置LAFAN全身ghost；物理终止后冻结并标注。正式worker结束后串行渲染所选best记录，期间不启动第二个GPU仿真。

## 当前结论
**运行状态与上表为准。** 技术Smoke通过仅说明迁移/梯度/恢复管线可用。真实目标误差大的阶段保留失败记录，不放宽物理终止或成功门槛，不将存活解释为学会足轨迹跟踪。
