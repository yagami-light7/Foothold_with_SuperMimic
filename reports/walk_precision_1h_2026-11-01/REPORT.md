# 中高难度落点精度与首步专项：1小时试训

已准备可执行配置并完成短测，**没有启动1小时训练或上传测试数据**。从最新hard_region_v2继续，保留原checkpoint；本次目标是观察落点和首步的改善趋势，通过之后再决定长训。

## 启动

```bash
cd /mnt/sage_data/workspace/KUAVO_S54_Newton
bash scripts/train_walk_precision_1h.sh
```

默认2048环境、SwanLab在线精简上传、新目录 `logs/walk/<时间>_walk_precision_1h/`。已有SwanLab登录沿用现有入口。预览不启动训练、不创建实验、不上传：

```bash
bash scripts/train_walk_precision_1h.sh --dry-run
```

可用 `--checkpoint /绝对路径/model_final.pt` 选择其他起点；`--resume-dir logs/walk/<本次目录>` 继续同一实验的剩余预算及SwanLab身份。默认起点：

`logs/walk/2026-11-01_17-59-35_walk_hard_region_v2/training/00003_bridge/model_final.pt`

## 配置与改动理由

| 项目 | 上次hard_region_v2 bridge | 本轮precision |
|---|---|---|
| 主训练分布 | 大步与高频分开训练，联合最大角点5% | 60%：30–35cm、1.4–1.8Hz |
| 保留任务 | 20%：20–25cm、1–1.2Hz | 同比例同范围，保留左右独立频率 |
| 其余环境 | 35%大步中频、40%小步高频等 | 20%：35–40cm、1–1.3Hz |
| 起步 | 随机左/右 | 随机左/右，不缩短首步目标、不延迟相位 |
| episode | 10s | 4s，增加约2.5倍起步机会；提前摔倒仍算失败 |
| 摆动脚XY系数 | 2u² | 后续4u²，首步6u² |
| 支撑脚XY系数 | 2 | 2 |
| 摆动脚距离核 | exp(−100e²) | 75% exp(−100e²) + 25% exp(−25e²) |
| PPO学习率 | 1e−4 | 固定5e−5，加载优化器后显式覆盖 |
| 评估 | 每250更新，多工况完整诊断 | 每500更新，3个固定工况的快速物理评估 |

步长单位m；表中为cm。等时左右交替时，Hz指完整左右周期，一侧摆动时长0.5/f；1.7Hz是每秒约3.4次交替落脚。frontier/stride组左右等频，retain组保留独立左右频率。

新宽核只作用于摆动脚XY。在30cm误差处，窄核约0.000123，混合核约0.02644，能提供更明显的接近奖励；仍以窄核为主。PPO通过采样回报优化，不能把这一核值解释成直接对足端位置求导。所有奖励变更是待验证假设，本轮包含组合改动，不能据此单独归因哪项有效；后续可设两个multiplier=1、coarse_weight=0做消融。

网络仍为105维观测、27维动作，完整恢复actor/critic/优化器/归一化统计；新增首步标志只供奖励与统计使用。Action scale沿用双腿 `[.10,.15,.30,.60,.30,.10]`、上身`.05`，动作平滑alpha`.2`、PD与力矩限幅沿用起点。未改变Newton/MJWarp控制接口。rollout64、mini-batch4、epochs4、gamma`.995`、lambda`.95`、entropy`.002`保持原配置；学习到的动作std保留。每次PPO更新131072条transition，500更新约6554万条。

## 奖励仍为8项

令u为当前摆动半周期进度，G表示支撑脚接触且足底参考点高度不超过2.5cm，W表示u∈[.3,.7]，K(e)为上面的混合距离核。奖励率：

1. **tracking**：`G × [2 exp(−100 e_st²) + a u² K(e_sw) + .25(exp(−25 yaw_L²)+exp(−25 yaw_R²))]`，a后续为4、首步为6。支撑脚离地时跟踪奖励为0。
2. **swing_phase**：`4 W × 摆动脚离地 × G × clamp(抬脚高度/.03,0,1)`。
3. **stance_slip**：`−4 Σ接地脚 ||足底水平速度||²`，保留转动偏移补偿。
4. **stance_hop**：`−4 × (超容差高度成本 + 超死区竖直速度成本 + 摆动内部缺失支撑 + 双脚腾空)`；高度/速度二次成本分别截断4，参数沿用原v2。
5. **body_motion**：`−10(height−参考高度)² − .5 vz²`。
6. **posture**：`.5 × upright × exp(−4 Σ上身关节偏移²)`。
7. **action_rate**：`−.05 mean27((action−previous_action)²)`，使用平滑后的动作。
8. **heading**：`exp(−(heading_error/.35)²) + .5 exp(−(yaw_rate/.30)²)`。

八项求和后只乘一次控制dt=.02。相位参考上界同步更新为 `.02 × [4.5 + a u² + 4W]`；惩罚上界为0，混合核上界仍为1。归一化只用于日志，不改变传给PPO的回报。本轮分母/奖励与历史不同，历史总分不能直接横比。

## 当前模型基线：确实需要首步专项

同源checkpoint、seed42、确定性policy、随机左右起步、16环境×1000控制步。每episode10s，即总20s含两段10s；不是连续20s/60s稳定性证明。Newton配置不保证位级确定性。

| 固定命令 | 无跌倒 | 全部/首步之后的末期XY | 首步左/右末期XY | 首步左/右成功率 |
|---|---:|---:|---:|---:|
| 25cm / 1.2Hz | 100% | 1.40 / 1.11cm | 2.24 / 10.83cm | 100% / 0% |
| 30cm / 1.5Hz | 100% | 2.72 / 2.28cm | 19.16 / 15.49cm | 0% / 0% |
| 35cm / 1.7Hz | 100% | 7.59 / 7.11cm | 29.76 / 19.28cm | 0% / 0% |

各点terminated=0、truncated=32；首步左/右样本为10/22。随机起步不会保证小样本严格各半，指标分侧统计。全部半周期成功率左/右分别为93.9%/91.4%、83.8%/86.4%、0%/11.0%。说明小任务整体分数会掩盖右脚首步问题，30cm的主要缺口之一是起步，35cm则连后续步态也需优化。

原始输出见 `baseline_*_final.json`，配置、hash、参数与摘要见 `experiment_summary.json`，评估命令与实际进程耗时见 `evaluation_commands.json`。

## 已执行检查

- 单元测试81项通过，含旧功能、部分reset、左右不等频的首步计数、宽核、奖励上界、防跳门控、预算与基线比较流程：`unit.log`。
- 12个改动Python文件的Pyright检查0错误：`pyright.json`；shell语法和一键脚本dry-run通过：`dry_run.json`。
- 完整契约回归：16环境×100步，reward/phase对齐误差均0，105维obs通过原有逐帧断言：`full_contract.json`。
- 同条件快速评估与完整诊断对照：成功半周期计数完全一致，主要物理指标差异小于1e−6；100步测得循环3.14s→2.61s，快评不生成完整trace：`fast_vs_full.json`。保留默认完整诊断路径。
- 最终配置2048环境×2次PPO更新通过，共262144条transition；实际采样+学习7.98s。manifest确认full_resume、105→105、无维度迁移、实际学习率5e−5：`ppo_final_smoke/`。随机探索出现1次提前终止、0次timeout；仅128步尚未达到4秒/200步timeout。此时episode回报主要来自早结束样本，不能视为全体稳态均值。
- 此短测checkpoint在30cm/1.5Hz、16环境×500步推理未跌倒，terminated=0、truncated=16：`smoke_policy_30cm_1p5hz.json`。仅验证可推理，不把2次更新解释成已改善步态。

## 一小时后怎样判断

监督器先记录起点基线，再训练。每500次更新或最后一个缩短的训练段结束时，使用相同seed与25cm/1.2Hz、30cm/1.5Hz、35cm/1.7Hz对照。评估16环境×1000步、episode10s；训练episode4s。默认不重复扫描40cm/2Hz角点。训练、评估、进程启动均计入总墙钟1小时；末尾预留180s给评估，按实测速率缩小末段训练，必要时中断并保存model_interrupted.pt。停止保存/上传收尾可能再多几十秒，不保证恰好满60分钟。minimum_updates=500只参与资格判定，不会越过时间预算继续训练。

自动写 `pilot_comparison.json`：含基线与最新已完成评估checkpoint路径、三个工况的XY变化、左右首步/整体成功率、首步后XY及滑动/跳动。若预算内最后评估被中断，该文件仍对应上一个已完整评估模型，不能冒充最新checkpoint结果。

SwanLab沿用compact，仅增加4条训练首步和4条评估首步曲线，固定约38条；不按关节或每个工况展开。优先看：

- `Eval/late_xy_max_m`下降、`Eval/left_success_min`/`right_success_min`上升。
- `Eval/first_left_xy_max_m`/`first_right_xy_max_m`下降、对应`first_*_success_min`上升。训练过程也有`Startup/left_success`、`right_success`、`left_xy_error_m`、`right_xy_error_m`。
- `Eval/survival_min`保持≥95%，双脚腾空和支撑跳动比例不升高，25cm保留任务无明显退化。

首步成功要求中段实际离地、末段3cm内接触、中段持续有支撑且不提前结束。首步提前摔倒计入失败分母；没有末段XY样本时只报告成功失败与缺失计数，不把误差填0。完整JSON保留样本数。

建议把**中高难度XY下降至少10%、首步XY下降至少15%、存活率不退化**作为继续投入长训的初步趋势，不是训练已成功的证明。严格资格门槛仍要求各点存活≥95%、两脚半周期成功≥80%、XY≤3cm、滑动/跳动约束，并新增左右首步各≥4样本、成功≥80%、末期XY≤3cm；至少500更新且连续两次通过。未通过就在本阶段训练到预算结束，不自动启动更大范围或更长训练。趋势不足时先检查首步时长与实际控制响应，不直接延长10小时。

## 文件清单

- 新增 `configs/walk_precision_1h.json`、`scripts/train_walk_precision_1h.sh`：本轮配置和一键入口。旧hard/recovery计划未改写。
- `configs/s54_flat_reward.py`、`tasks/foot_placement/flat_rewards.py`、`rewards.py`：可选摆动XY权重、首步权重、宽核和匹配上界；默认值保持旧奖励。
- `goal_sampler.py`、`foot_placement_env.py`、新增 `first_step_metrics.py`：独立半周期计数、首步加权、分侧事件统计；不改变obs。
- `configs/walk_training.py`、`scripts/adaptive_walk.py`：参数校验、可选快评/基线、首步门槛、末尾预算、自动对照；旧课程默认行为保留。
- 新增 `scripts/quick_walk_eval.py`，修改 `preflight_signals.py`：快评与完整诊断共用启动入口，保留全诊断，适配首步奖励重算。
- `scripts/swanlab_metrics.py`、`train_walk_12h.sh`：精简上传新首步指标，允许指定评估环境/步数。
- 新增 `tests/unit/test_walk_precision.py` 与本报告、原始短测结果。
