# S54 平地奖励 v2、随机起步与回溯检查

本次完成代码修改、CPU 回归、短回放和 2 次 PPO 更新；没有启动正式训练，也没有证据宣称新奖励已经让策略学会不跳的稳健行走。

## 1. 结论与回溯模型

建议回到 **10 月 30 日上一轮实验**，而不是失败实验内部的前一个保存点：

```text
logs/walk/2026-10-30_22-45-31_walk_4stage/training/00003_walk_4/model_interrupted.pt
SHA256: 2c43060c21791cd4e94188c91bf9a87508ab78ea327f59db8e0887d96fc28df2
```

失败轮最后状态为 `budget_exhausted`，累计 12304 次更新，仍在 bridge，未通过晋级。它在摆动目标误差之外学到了明显的左右不对称行为。原奖励没有直接惩罚接地脚速度和机身竖直运动，跟踪项也未统一绑定有效支撑，提供了利用跳动/拖移缩小 XY 误差的空间；这是实现与回放支持的漏洞分析，尚非单因素因果消融。

以下均为确定性策略，seed=42，干净物理，**16 环境 × 1000 控制步（每环境 20 秒）**，无梯度更新。成功率按完成半周期分别统计，不是每帧 touchdown ratio。

| 模型 | 固定命令 | 起步 | 20 秒无跌倒环境比例 | 左/右半周期成功率 | 摆动末期 XY 误差 |
|---|---|---|---:|---:|---:|
| 推荐旧模型 | 25 cm / 1.2 Hz | 左 | 100% | 91.7% / 90.9% | 1.11 cm |
| 推荐旧模型 | 25 cm / 1.2 Hz | 右 | 100% | 81.8% / 75.0% | 2.44 cm |
| 失败轮最新 | 25 cm / 1.2 Hz | 左 | 100% | 0% / 90.9% | 13.08 cm |
| 失败轮最新 | 25 cm / 1.2 Hz | 右 | 0% | 0% / 0% | 25.54 cm |

15 cm / 0.8 Hz 随机起步时，两模型无跌倒比例分别仅 6.25% / 12.5%，右脚起步组的两脚成功率均为 0。因此不宜马上在低频至最大频率全区间继续长训。原始结果和六条精确命令见 [comparison_summary.json](../random_start_rollback_2026-11-01/comparison_summary.json)、[comparison_commands.json](../random_start_rollback_2026-11-01/comparison_commands.json)。

## 2. 生效范围与共同定义

新配置 `FlatWalkRewardCfg(profile="flat_walk_v2")` 在 `configs/s54_flat_reward.py`；公式在 `tasks/foot_placement/flat_rewards.py`。`walk_recovery.json` 与 `walk_hard_region.json` 显式启用它。旧 checkpoint 未声明 profile 时恢复旧 `FootRewardCfg`，不会悄悄改变历史实验的奖励；单独使用其他旧训练入口也不会自动切换到 v2。

以下公式先计算**奖励率**，环境最后统一乘 **控制 dt=0.02 s**；物理 dt=0.002 s。SwanLab `Reward/*` 已是乘 dt 后的单控制步贡献。正权重的惩罚项在公式中带负号。

- 完整相位 `φ∈[0,1)`；左摆动 `φ<0.5`，右摆动 `φ≥0.5`；半周期局部相位 `u=2φ` 或 `2φ−1`。
- 左/右接触 `C_i=1(Fz_i>20 N)`，使用环境提供的地面接触力。
- 指定支撑脚高度 `h_st=max(z_st−z_st,target,0)`；有效支撑门控 `G=C_st × 1(h_st≤0.025 m)`。
- 抬脚窗口 `W=1(|u−0.5|≤0.2)`，对应完整相位左 `[0.15,0.35]`、右 `[0.65,0.85]`。不是整个摆动半周期。
- `e_sw/e_st` 为足底参考点相对缓存目标的水平误差，单位 m；yaw 误差使用 wrap 到 `[-π,π]` 的弧度。
- 足底点世界速度由 `v_link + ω_link × R(q) p_sole` 得到，包含脚踝转动造成的参考点运动，不直接用刚体质心速度。
- 支撑高度容差曾用 1 cm；实测正常足掌滚动可抬高参考点约 2.38 cm，故最终设为 **2.5 cm**。这不是将摆动抬脚目标升高到 2.5 cm。

## 3. 当前全部 8 类奖励

表中最大值是**单项奖励率上界 / 单步贡献上界**；各单项上界不一定能同时达到。

| 日志项 | 当前公式（奖励率） | 参数 | 物理意义 | 最大值：率 / 单步 |
|---|---|---|---|---:|
| `tracking` | `G × {2[u² exp(−100‖e_sw‖²)+exp(−100‖e_st‖²)] + 0.25[exp(−25ψ_L²)+exp(−25ψ_R²)]}` | XY 总权重 4；足 yaw 总权重 0.5 | 支撑脚守住目标，摆动末期更重视落脚；两脚朝向跟踪 | `2(1+u²)+0.5`，全局上界 4.5 / 0.09 |
| `swing_phase` | `4 W (1−C_sw) G clamp((z_sw−z_sw,target)/0.03,0,1)` | 权重 4；充分抬脚高度 3 cm；半窗 0.1 完整相位 | 在指定相位抬正确的脚，且另一脚有效支撑 | 4 / 0.08 |
| `stance_slip` | `−4 Σ_i C_i ‖v_sole,i,xy‖²` | 权重 4 | 惩罚任一接地脚水平滑动；不惩罚腾空摆动脚正常前移 | 0 / 0 |
| `stance_hop` | `−4 (H + V + M + A)`，定义见下 | 权重 4 | 惩罚指定支撑脚抬高、竖直跳动、支撑缺失和双脚腾空 | 0 / 0；下界 −40 / −0.8 |
| `body_motion` | `−10(h_root−h_ref)² −0.5 v_root,z²` | 高度 10；竖直速度 0.5 | 抑制蹲伏和机身上下弹跳；S54 `h_ref=0.969025 m` | 0 / 0 |
| `posture` | `0.5 clamp(−g_projected,z,0,1) exp(−4 Σ_upper (q−q_default)²)` | 权重 0.5；指数系数 4；上半身 15 关节 | 维持直立及腰、双臂参考姿态；不强迫腿保持站姿 | 0.5 / 0.01 |
| `action_rate` | `−0.05 mean_27[(a_filtered,t−a_filtered,t−1)²]` | 权重 0.05；27 维取均值 | 抑制实际施加的归一化动作突变；不是力矩惩罚 | 0 / 0 |
| `heading` | `exp[−(e_heading/0.35)²] +0.5 exp[−(ω_root,z/0.30)²]` | 方向 1；偏航角速度 0.5；尺度 rad、rad/s | 保持 reset 时参考方向，减少无指令转弯 | 1.5 / 0.03 |

`stance_hop` 的四个无量纲代价：

```text
H = min([max(h_st − 0.025, 0) / 0.02]², 4)
V = min([max(|v_st,z| − 0.05, 0) / 0.50]², 4)
M = 1(0.1 < u < 0.9) × (1 − C_st)
A = (1 − C_L) × (1 − C_R)
```

高度和速度项分别截断到 4，限制碰撞异常值。换脚前后各 10% 半周期允许短暂接触转移，但双脚腾空项一直生效。即使把脚移到正确 XY，只要支撑无效，`tracking` 和 `swing_phase` 都为 0。

全部正项的逐相位参考上界为：

```text
R_ref(u) = 0.02 × [2(1+u²) + 0.5 + 4W + 0.5 + 1 + 0.5]
```

窗口内最高参考上界在 `u≈0.7`，约 **0.1896/控制步**；0.21 是把各项全局最大值相加得到的宽松上界，不能当作每帧应达到的分数。未接触、无法同时满足等约束会使真实可达值更低。

归一化回报保持 `[0,1]`：`clip(Σr/ΣR_ref,0,1) × clip(episode_steps/max_episode_steps,0,1)`。它表示奖励质量乘存活时长比例；**不是行走成功概率**。v2 与旧版的原始 reward、归一化分数不能直接比较，物理评估指标才是共同尺度。

高度项用 root/base 高度近似身体高度，未计算整机质心。当前没有速度指令跟踪奖励，也没有能量、力矩、关节加速度、额外终止奖励或膝盖高度项；任务仍是足印命令驱动的直线平地行走。

### 修改前后的奖励差异

失败轮实际保存的参数：摆动 XY=4、支撑 XY=0.5、相对 Z clearance=4、swing_air=3、stance_contact=0.5、touchdown=2、upright=0.5、unloading=1、动作变化=0.05、heading=1、yaw_rate=0.5，宽 XY 核权重为 0。

v2 合并足印位置与方向为 `tracking`，不沿用原 swing/stance 权重比例；删除独立 clearance 正弦跟踪、swing_air、stance_contact、touchdown、unloading；用指定窗口的抬脚项替代。新增接地脚滑动、支撑脚跳动和机身高度/竖直速度惩罚；upright 与上半身姿态合并。动作变化及机身方向相关系数保持原值。旧算法分支保留，以便对照旧 checkpoint。

## 4. 与论文平地设置的对照

已核对 [Mind Your Steps 附录 D（PDF 第 14 页）](https://arxiv.org/pdf/2606.08253#page=14) 和 [附录 F（PDF 第 15 页）](https://arxiv.org/pdf/2606.08253#page=15)。下表是论文平地设置与本次 S54 适配的区别，不能据此声称复现原算法。

| 类别 | 论文平地设置 | S54 v2 |
|---|---|---|
| 足 XY / yaw | 权重 5 / 5；指数系数均 100 | 总权重 4 / 0.5；指数系数 100 / 25；左右取平均，摆动 XY 乘 `u²`，要求有效支撑 |
| 支撑期跟踪 | 保持摆动结束时的跟踪得分 | 对缓存的支撑目标持续重算，能够惩罚后续位移；未实现论文的得分冻结 |
| 足 Z / 膝盖 | 平地明确关闭 `ω₂=0、ω_k=0` | 无 Z 目标轨迹跟踪、无膝高项 |
| 指定相位抬脚 | 权重 6，相位半窗 0.1 | 权重 4，同半窗；额外加入支撑门控和 3 cm 高度进度 |
| 上半身姿态 | 权重 4、指数系数 4 | 权重 0.5、指数系数 4，叠加直立门控 |
| 身体高度 | 高度平方误差系数 10 | 系数 10，使用 S54 root 参考高度，另含竖直速度项 |
| 动作变化 | 系数 3，各维平方求和 | 系数 0.05，27 维滤波动作平方取均值；数值不可直接类比 |
| 足滑动 | 系数 4，接触脚三维速度平方和 | 系数 4，XY 速度平方和；竖直支撑运动单列 |
| 防跳与方向 | 框架另有常规运动正则 | 支撑脚高度/速度、支撑缺失、双脚腾空显式列项；保留 S54 heading/yaw-rate |

**关闭平地 Z 落点跟踪不等于取消相位抬脚。** 本次保留相位窗口，但不追踪整条摆动脚 Z 曲线。系数和动作尺度按本机控制接口选定，没有照搬 Booster T1 参数。

另核对 [LocoMuJoCo 官方 foot_placement.py](https://github.com/MontenegroAlessandro/loco-mujoco/blob/master/loco_mujoco/core/reward/foot_placement.py)，核对版本 blob SHA 为 `40f33a44ac722328bfdd7f79e3a883f5860217c5`。其代码包含接触速度、相位抬脚、no-fly、root 竖直速度及其他正则；本次只采用其中相关设计思想，没有复制它全部惩罚或声称参数一致。

## 5. 随机左右起步与 checkpoint

新训练每次 reset 对每环境独立 50% 选左/右先摆：左 `phase=0`、右 `phase=0.5`，同时同步摆动标签及目标。不是只在第一次创建环境时随机，也不是随机落在半周期中间。局部 reset 不影响其他环境。

GoalSampler 直接 API 和旧 checkpoint 缺省仍为左脚；新环境默认随机，两份新课程显式随机。`--initial-swing-side left|right|random` 可覆盖回放与诊断。评估同时报告“左起步组”和“右起步组”的两脚成功率；两组都必须满足门槛，避免总体均值掩盖右起步失败。

观测仍 **105D**、动作仍 **27D**，网络没有扩维；actor、critic、optimizer、normalizer 可完整恢复。2 次 PPO 短测记录 `source_dim=105, target_dim=105, migrated=false`。此次没有修改 PD、关节映射、action scale、动作滤波和终止条件。奖励改变后旧 critic 仍需适应新回报，因此恢复试训学习率降到 **5e-5**。旧模型的表现不能当作新奖励训练结果。

## 6. SwanLab 与评估门槛

默认 `--swanlab-detail compact`，只在 SDK 上传边界筛选，**本地 metrics.jsonl / TensorBoard 保留全部诊断**；旧云端 run 不删除。固定最多 30 种标量名称：

- Train/PPO 8 条：归一化回报、每步 reward、episode 步数、每控制步提前终止比例、学习率、value loss、动作 std、左起步比例。
- Reward 8 条：上表 8 类，均为乘 dt 后贡献。
- Eval 9 条：必测工况最差存活率、左右脚成功率、末期 XY 误差、双脚腾空占比、支撑跳动占比、接触滑速，以及左/右起步组各自的最差单脚成功率。
- Progress 3 条：剩余小时、当前阶段、是否通过评估。
- GPU 2 条：显存 MiB、利用率%。

短测每次本地有 149 项，compact 保留其中 14 项；episode 完成和 eval/GPU 采样后才会出现其余曲线。非必测最大角点仍记录在本地，未混入 compact 的“必测最差值”。需要全部上传可传 `--swanlab-detail full`。`fall_per_control_step` 是控制步上的 terminated 比例，不是 episode 失败概率；truncated 超时单独留在本地。

晋级仍要求存活率≥95%、两脚半周期成功率各≥80%、连续 8 次交替成功环境比例≥80%、末期 XY≤3 cm、直立≥0.95、最大朝向误差≤10°、前向速度≥名义速度一半。新增左右起步组分别两脚≥80%，以及双脚腾空≤2%、支撑跳动代理占比≤2%、接地足底水平平均滑速≤0.10 m/s。这些是本项目评估阈值，不是论文参数。支撑跳动代理统计半周期中段的缺失支撑或超高，不等同于所有跳动的可靠识别器。

## 7. 已运行检查及实际输出

原始数据在 `reports/random_start_rollback_2026-11-01/`。

| 检查 | 实际结果 | 可得结论 |
|---|---|---|
| CPU 单测 | `Ran 73 tests ... OK`，见 `unit.log` | 随机起步、局部 reset、非等时相位、奖励反例、足底速度、版本恢复和精简日志回归通过 |
| 合成中摆动相位对照 | 静态站立 0.091054；正确抬脚 0.180000；双脚腾空跳动 −0.504800 / 控制步 | 同一参考场景中正确摆动优于站立；腾空不能继续获得跟踪/抬脚奖励。见 `synthetic_reward_contrast.json`；这是解析状态测试，非训练结果 |
| v2 最终固定点回放 | 32 环境×1000 步，25 cm/1.2 Hz，随机起步：存活100%，terminated=0，truncated=64，左右成功85.87%/80.89%，双脚腾空0、支撑跳动代理0，滑速0.03253 m/s | 接口与物理指标正常；右起步右脚仅75%，不应通过晋级。见 `flat_reward_final.json` |
| 恢复范围探测 | 32 环境×1000 步，20–25 cm/1–1.2 Hz：存活100%，terminated=0，truncated=64；左右成功94.93%/83.87%；左起步100%/92.10%，右起步88.52%/74.45%；双脚腾空0、支撑跳动0.106%，滑速0.02780 m/s，末期XY1.50 cm | 此范围可作为修复起点；仍未达右起步成功门槛。见 `recovery_range_probe.json` |
| 最终 PPO 短更新 | 128 环境，rollout64，2次更新，LR5e-5，完整加载105D，指标有限、保存成功 | 训练接口可用；不是验证学会新步态。见 `flat_ppo_final_smoke/manifest.json` 与同名 `.log` |
| 原集成 smoke | `S54 Goal Integration Test PASSED`；`S54 Reward Integration Test PASSED` | 旧奖励/旧固定起步测试及局部 reset 不回归 |
| 新奖励 checkpoint 重载 | 80 步回放，恢复 v2 配置及105D，时序对齐误差0 | 序列化及恢复分支可用，见 `flat_checkpoint_reload.json`（使用早期短测模型） |
| 时序 | 最终两次长回放 reward 与 phase 对齐误差均0 | 本步奖励与相位、reset 对齐 |
| Shell/入口 | 3份 active shell 语法检查通过，1h恢复/12h大课程 dry-run通过，实际命令均只有一项learning_rate；logger不导入Torch | 入口和配置可复查；未连云端启动正式训练 |
| Pyright | 66 文件、56 个 `reportCallIssue`；与此前56个诊断的文件及消息完全相同 | 本次无新增诊断；全项目仍有SDK动态configclass构造参数的静态诊断，不能称为全项目零报错 |

1000 控制步为20秒；episode上限10秒，所以无跌倒测试可出现每环境两次 truncated，64次超时是正常结果。Newton 非严格确定性求解会带来复测差异；本次只有 seed42 的短测，不是多seed稳健性证明。

复查 CPU 回归：

```bash
cd /mnt/sage_data/workspace/KUAVO_S54_Newton
/mnt/sage_data/workspace/IsaacLab_3_0/.venv/bin/python -m unittest discover -s tests/unit
```

复查恢复范围（不更新模型）：

```bash
uv run --project /mnt/sage_data/workspace/IsaacLab_3_0 --no-sync \
  isaaclab -p scripts/preflight_signals.py --stage straight --mode policy \
  --checkpoint logs/walk/2026-10-30_22-45-31_walk_4stage/training/00003_walk_4/model_interrupted.pt \
  --task-profile reports/random_start_rollback_2026-11-01/recovery_profile.json \
  --num_envs 32 --steps 1000 --seed 42 --viz none \
  --output /tmp/s54_recovery_recheck.json
```

短更新完整参数保存在 `flat_ppo_final_smoke/manifest.json`；六次回溯比较的精确 argv 保存在 `comparison_commands.json`。早期 `flat_reward_metrics.*` / `flat_ppo_smoke/` 使用1cm容差，仅保留作容差消融；**正式配置以最终2.5cm及带 final 的结果为准**。

## 8. 推荐下一轮与一键命令

先运行 **1小时恢复验证**：2048 环境，20–25 cm，左右等时1–1.2 Hz，随机起步，干净物理；rollout64、4 minibatches、4 epochs，固定LR5e-5，gamma0.995、GAE0.95、entropy0.002，10秒episode。每500次PPO更新做评估，完整网格按现有调度，连续两次通过且完整网格通过才认为恢复合格。该小时预算包含训练、评估和启动；收尾保存可能有小幅超时。

```bash
cd /mnt/sage_data/workspace/KUAVO_S54_Newton
bash scripts/train_walk_recovery.sh
```

默认在线 SwanLab 项目 `S54-Locomotion`，compact；输出 `logs/walk/<时间>_walk_recovery/`，沿用本机已有 SwanLab 登录。仅预览可加 `--dry-run`。换起始权重可加 `--checkpoint /absolute/path/model.pt`。

这是**新实验**：不要用失败轮的 `--resume-dir`，那会按保存快照继续旧课程/旧奖励。恢复本次新实验中断才用其自己的 `--resume-dir`。保存频率仍由 `agents/rsl_rl_ppo_cfg.py` 的 `save_interval` 控制。

通过后再讨论扩到30–35cm及1.5Hz，并分别测试低频右起步、40cm2Hz最大角点；不建议立即启动覆盖全部难区间的12小时训练。左右等时时这里的Hz是完整左右周期频率，**单步频率是其2倍**，单侧摆动时长 `0.5/f`。

## 9. 本次文件修改

| 用途 | 文件 |
|---|---|
| 新奖励参数/公式 | 新增 `configs/s54_flat_reward.py`、`tasks/foot_placement/flat_rewards.py`；修改 `rewards.py`、`foot_placement_env.py`、`foot_placement_env_cfg.py`、`foot_frames.py`、`visualization.py` |
| 随机起步/旧版本恢复 | `goal_sampler.py`、`configs/s54_curriculum.py`、`configs/walk_training.py`、`scripts/gait_cli.py` |
| 课程与一键入口 | 新增 `configs/walk_recovery.json`、`scripts/train_walk_recovery.sh`；更新 `configs/walk_hard_region.json`、`scripts/train_walk_12h.sh` |
| 评估/云端精简 | 新增 `scripts/swanlab_metrics.py`；修改 `scripts/preflight_signals.py`、`scripts/adaptive_walk.py`、`scripts/train_4060.py` |
| 回归 | 新增 `test_initial_swing.py`、`test_flat_rewards.py`、`test_swanlab_metrics.py`；更新 `test_adaptive_walk.py`、`test_heading_control.py` 及三个旧sim检查 |

`agents/rsl_rl_ppo_cfg.py`、机器人控制参数、Newton/MJWarp配置本次没有修改。代码和课程快照见本目录 `snapshot/`、校验见 `source_sha256.json`；旧 checkpoint 和失败轮日志都保留。另修正旧报告中“adaptive KL”的表述：实际训练一直使用 `schedule="fixed"`。
