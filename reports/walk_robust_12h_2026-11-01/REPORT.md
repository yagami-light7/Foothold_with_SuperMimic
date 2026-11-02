# 大步幅、高步频 12 小时专项训练准备

从recovery继续训练的当前入口与配置见 [hard_region v2报告](../hard_region_v2_2026-11-01/REPORT.md)。本文保留改版前的实验记录。

2026-11-01。已完成代码修改和短测试，尚未启动正式长训练。默认从当前 `logs/walk` 中最新实验的 `2026-10-30_22-45-31_walk_4stage/training/00003_walk_4/model_interrupted.pt` 完整续训；SHA256 记录在 `initial_checkpoint.json`。短测试权重留在本报告目录，不用作正式训练的默认起点。

原模型在 40 cm、左右 2 Hz 的固定角点会跌倒；此前 20 秒测试的无跌倒比例为 0。原均匀独立采样命令只有约 3.17% 落入“步长≥30 cm、左右频率均≥1.5 Hz”。因此本次优先修改采样和晋级方式。

## 训练配置

`configs/walk_hard_region.json` 现在是 schema=2 的自适应课程。步长单位是米；频率为每侧 `T_swing=1/(2f)` 的 Hz，左右等频时 2 Hz 对应每秒四次左右换脚。40 cm / 2 Hz 的名义前向速度是 1.6 m/s，实际速度另外测量。

| 阶段 | 环境分组（reset 后保持分组） | 最少 PPO 更新 | 每段初始学习率 |
| --- | --- | --- | --- |
| bridge | 20% 已掌握任务；35% 30–40 cm / 1–1.4 Hz；40% 20–30 cm / 1.5–2 Hz；5% 固定最大角点 | 1500 | 1e-4 |
| frontier | 20% 已掌握任务；70% 30–35 cm / 1.5–1.8 Hz；10% 固定最大角点 | 1500 | 1e-4 |
| hard | 20% 已掌握任务；60% 30–40 cm / 1.5–2 Hz；20% 固定最大角点 | 3000 | 1e-4 |
| robust | hard 分组，加小幅随机化、命令渐变和推扰 | 1000 | 5e-5 |

已掌握任务为 15–25 cm、左右各 0.8–1.2 Hz 独立采样；其余组左右等频。在 2048 环境下，bridge 实际分配为 410 / 717 / 819 / 102；hard、robust 为 410 / 1229 / 409。困难环境即使频繁跌倒，也不会通过 reset 换成简单环境。

晋级同时要求：达到最少更新数、连续两次评估通过、期间至少一次完整网格通过。失败清零连续通过和完整网格标记，不按更新数强行换档。总预算用尽时可以仍停留在前面的阶段，12 小时并不保证达到最大角点。最后阶段通过后继续训练至预算结束。

默认 2048 环境，rollout=64，minibatches=4，每次更新 131072 条 transition；每段最多 1000 次更新后评估。每阶段首次评估以及每 3 段做完整网格，其余评估关键固定点。所有阶段都监控最大角点，bridge/frontier 暂不以最大角点作为晋级门槛。最后阶段额外检查直接起步和四次换脚渐进加速的两组真正 60 秒 episode，终止条件保持原值。

每个必测点均须满足：整个窗口无跌倒比例≥95%、左右半周期成功率各≥80%、至少连续八次成功换脚的环境比例≥80%、摆动末期 XY 误差≤3 cm、平均直立投影≥0.95、最大朝向误差≤10°、参考方向实际速度≥本次平均名义速度的 50%。左右不等频的保留任务也进入完整评估。

robust 使用：质量及惯量同比缩放 0.97–1.03、足端摩擦系数 0.45–0.55；reset 关节位置±0.004 rad、根部水平速度±0.025 m/s；观测高斯噪声标准差分别为重力投影 0.005、角速度 0.01 rad/s、关节位置 0.001 rad、关节速度 0.01 rad/s；每 6–9 s 增加 xy 速度扰动，每轴±0.06 m/s。命令每四次换脚可重采样，每次变化不超过 2 cm、0.1 Hz，只在换脚边界生效。

105D 观测、27D 动作、关节映射、action scale、PD、动作平滑、原奖励权重和终止条件均沿用；actor、critic、optimizer 和 normalization 完整恢复。PPO 在 `configure_stage` 中使用固定学习率（`schedule="fixed"`），实际学习率看 `Loss/learning_rate`。新增双尺度 XY 奖励开关用于后续消融，本次所有阶段 `xy_coarse_weight=0`，没有启用。

后续单脚跳诊断与奖励v2修订见 [新报告](../flat_reward_v2_2026-11-01/REPORT.md)。本文描述修订前的实验，保留用于消融对照。

## 已运行的检查

| 检查 | 实际输出 / 结论 |
| --- | --- |
| CPU 回归 | `unit.log`：55 tests，OK；覆盖固定分组、局部 reset、渐变命令、晋级失败保持、完整网格门槛、预算和云端身份恢复、恢复时拒绝更换课程、60 秒评估入口 |
| 原独立测试 | `*_regression.log`：reward、sampler、103D observation contract、课程恢复全部 PASSED |
| Newton 物理接口 | `physics_contract.json`：16 环境、3010 步、每 episode 3000 步；terminated=0、truncated=16、119 次推扰。实际质量缩放 0.97060–1.02537、模型总质量 65.50–69.19 kg、足端实际摩擦 0.45043–0.54992；惯量同比缩放、reset 不累乘质量、FK 和裁剪遥测断言通过。这是零动作站立检查。 |
| 新 profile 的旧策略回放 | `robust_smoke.json`：32 环境、520 步；terminated=15、truncated=23；无跌倒比例 71.875%，左右成功率 10.91% / 20.60%，末期 XY 误差 12.29 cm；奖励重算误差=0、phase 误差=0。说明接口可运行，也说明旧策略尚未掌握这组扰动困难任务。 |
| PPO 完整续训 | `ppo_smoke`：128 环境、2 次更新，实际学习率 1e-4；`ppo_robust_smoke`：接着 1 次更新，实际学习率 5e-5；均有限值、正常保存。两次加载都是 105→105，未迁移网络列、未扩展优化器张量。 |
| 监督器与 SwanLab | `supervisor_smoke.log`：离线 SDK、2 次更新、固定点评估跑通；126 秒测试预算到达后结束，状态记录约 129.3 秒，未晋级，checkpoint 保留。这里只完成部分网格，不是课程通过。正式脚本配置为 online。 |
| 静态检查 | `pyright_new_modules.json`：新增 6 个 Python 模块，0 errors。全库 `pyright.json` 仍有 SDK 动态 configclass 构造签名的既有 reportCallIssue；本次没有修改 Isaac Lab。 |

短 PPO 测试发现部分腿部 roll/踝关节动作裁剪率约 95–100%。裁剪率指原始 policy 动作超过 [-1,1]，不等同于实际电机力矩饱和。新增 `Control/action_clip_fraction/*`、关节误差/速度及 `estimated_pd_limit_fraction/*` 供后续检查；后者仅是策略步末 PD 估算，不是 500 Hz 子步的实测峰值。本次没有据此改变已学的动作映射。

## 启动与恢复

一键启动默认的 12 小时、2048 环境实验，日志写入 `logs/walk/时间_walk_hard_region`，自动上传 SwanLab：

```bash
cd /mnt/sage_data/workspace/KUAVO_S54_Newton
bash scripts/train_walk_hard_region.sh
```

仅预览，不创建实验、不上传、不训练：

```bash
bash scripts/train_walk_hard_region.sh --dry-run
```

指定起点，或者恢复同一个实验：

```bash
bash scripts/train_walk_hard_region.sh --checkpoint /绝对路径/model_final.pt
bash scripts/train_walk_hard_region.sh --resume-dir logs/walk/本次实验目录
```

同实验恢复沿用剩余预算、课程进度、配置快照和 SwanLab 身份。12 小时包括监督器内的初始化、训练、评估；退出保存有少量收尾时间。checkpoint 定期保存仍由 `agents/rsl_rl_ppo_cfg.py` 的 `save_interval=250` 控制；每段结束和中断也保存。

看 `Group/*/normalized_reward` 与 `terminated_fraction` 区分各组；最终以 `Eval/adaptive/*` 的物理指标和 `Curriculum/*` 判断。`state.json` 的 `adaptive_qualified` 保留通过门槛的模型路径；最新 checkpoint 继续更新后不自动继承“已通过”标签。优先回放已通过模型，未评估的新权重不能当作最优模型。

## 修改文件

入口与调度：`scripts/train_walk_hard_region.sh`、`scripts/train_walk_12h.sh`、`scripts/train_4060.py`、新增 `scripts/adaptive_walk.py`。

配置与环境：`configs/walk_hard_region.json`、新增 `configs/walk_training.py`、`configs/s54_curriculum.py`、`tasks/foot_placement/foot_placement_env_cfg.py`、`foot_placement_env.py`、`goal_sampler.py`、新增 `robustness.py`。`rewards.py` 仅增加默认关闭的 XY 宽核选项。

训练/评估/回放适配：`scripts/train.py`、`scripts/gait_cli.py`、`scripts/preflight_signals.py`、`scripts/play.py`、新增 `agents/vec_env.py`。

测试：新增 `tests/unit/test_adaptive_walk.py`、`tests/sim/test_walk_training.py`，更新 `tests/unit/test_command_curriculum.py`。最终代码快照在 `snapshot/`，校验值在 `source_sha256.json`，课程快照在 `plan_snapshot.json`。未启动正式训练，未宣称策略已学会大步高频稳健行走。
