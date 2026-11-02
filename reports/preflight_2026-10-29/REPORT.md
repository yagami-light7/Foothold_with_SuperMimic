# S54 PPO 训练前检查 — 2026-10-29

## 结论

完成了代码、日志、动作权限、时间对齐、奖励及短程 PPO 检查。**当前没有经过验证的行走 Policy，不建议直接开启数千轮训练。** 已提供可运行的 `stand → lift → step → walk` 配置，应先通过每阶段的评估门槛。当前仅支持固定前向足印课程，速度指令、停止和转向尚需后续命令课程。

本轮共进行 **14 次 PPO 更新、57,344 个优化样本**，另有归一化预热及不更新网络的诊断 rollout。所有短程 checkpoint 都是测试产物，不能当作训练完成的步态。

## 1. 环境、动作和 PD

- 27D canonical action 到 solver joint 的映射为 `[0, …, 26]`；目标、软限位及实际响应均已检查。
- 500 Hz 物理、decimation=10、50 Hz policy，103D policy observation、27D action 正确。
- 诊断逐步重算 reward，并检查 action history、phase、reset 后观测。有效 transition 的 reward 重算最大误差为 **0**；原 Goal/Reward smoke 及局部 autoreset 测试通过。
- 扩展 Action Authority Test 共 **123 个案例**：包括全部 27 DoF 正负 0.05 rad、髋/膝/踝多幅度阶跃和协调屈腿。0.3 s 稳定后保持阶跃 0.5 s，123 案例均未跌倒；这不能外推为持续行走稳定。
- 两个肘关节正向目标被软限位正常裁剪。承重时踝 pitch 跟踪误差较大且会饱和，不能据此认定电机或映射失效。
- 已读取 MJWarp `qfrc_actuator` 并逐步核验 **joint actfrcrange 对 P+D 合力矩的限制**，没有越过限幅。`authority_all/results.json` 区分 PD 估算 effort 和 solver 实际关节力矩。

| 协调动作：髋/膝/踝增量 rad | 实际膝位移 rad | 左足最大抬升 | 跌倒 |
|---|---:|---:|---:|
| −0.05 / +0.10 / −0.05 | 0.0996 | 2.36 mm | 0 |
| −0.10 / +0.20 / −0.10 | 0.2036 | 7.62 mm | 0 |
| −0.20 / +0.40 / −0.20 | 0.3982 | 25.15 mm | 0 |
| −0.30 / +0.60 / −0.30 | 0.5961 | 49.84 mm | 0 |

**确认的问题是动作范围与原 60 mm clearance 目标不匹配。** 上表不是对所有动作的理论可达性证明，但说明统一 ±0.1 rad 对当前屈腿模式明显不足。单纯放大权限也不够：周期屈腿缺少重心转移，10 s rollout 仍会跌倒。

PD 数值沿用实测配置，本轮将配置中的硬编码乘数改为使用 `KP_SCALE/KD_SCALE`，以便正确做消融：

| 左右腿对应关节 | Kp | Kd | 力矩限幅 Nm |
|---|---:|---:|---:|
| hip yaw / hip roll | 3000 / 3000 | 27.386 / 27.386 | 88.9 / 49.7 |
| hip pitch / knee | 2400 / 2400 | 27.386 / 32.863 | 92.4 / 196.0 |
| ankle pitch / roll | 900 / 1500 | 41.079 / 41.079 | 39.9 / 39.9 |

## 2. 原奖励与探索问题

以下均为 **32 env × 500 steps，seed=42** 的实际 rollout；跌倒次数允许同一环境重置后重复计数。

| 原配置行为 | Reward/step | terminated | truncated | Swing-air 帧占比 | 末期 XY 误差 |
|---|---:|---:|---:|---:|---:|
| 零动作 | 0.083811 | 0 | 32 | 0 | 0.1200 m |
| baseline_v1 确定性 | 0.083016 | 0 | 32 | 0 | 0.1201 m |
| baseline_v1 随机采样 | 0.067906 | 132 | 0 | 0.1835 | 0.1119 m |
| 零均值 σ=0.35 探索 | 0.071902 | 159 | 0 | 0.0660 | 0.1436 m |

零动作的 stance、upright、stance-contact 合计约 **0.06998/step，约占总奖励 83.5%**；站立还获得约 0.00721 clearance。每次换脚又用实际位置重设 stance target，使漏迈步后仍重新获得高 stance reward。二值离地、晚期接触和 12 cm 目标难以抵消这种长期安全站立收益。

原 action-rate 使用 27 维求和；独立 σ=0.35 动作相邻变化的期望惩罚约 `0.02×0.05×27×2×0.35²=0.00662/step`。大噪声带来的承重破坏和提前终止进一步降低主动动作的回报。

另外确认了日志 aliasing：环境复用同一个 `extras['log']` 字典，而 RSL-RL 持有字典引用。本轮修复前 500 步只有 **1** 个 log 对象，修复后为 **500** 个。历史 `Reward/*` 曲线不能当作可靠的 rollout 均值；`Train/mean_episode_length` 及本次独立重算的统计不依赖该错误。

## 3. 已实现的修复与候选参数

`legacy` 保留原控制和奖励语义，用于旧 checkpoint 对照。修复后 legacy 零动作 reward 为 **0.0838112**，与修改前一致。

| 项目 | legacy | 新课程 |
|---|---|---|
| Action scale，canonical 顺序 | 全部 0.1 rad | 每腿 `[0.1, 0.15, 0.3, 0.6, 0.3, 0.1]`；腰和手臂 0.05 |
| 动作滤波 | α=1 | α=0.2：`a_applied←0.8 a_previous+0.2 clip(a)` |
| swing height | 0.06 m | stand=0；lift/step=0.02；walk=0.03 m |
| 步长 | 0.12 m | stand/lift=0；step=0.03；walk=0.06 m |
| 步态周期频率 | 1 Hz | 0.75 Hz |
| stance target | 换脚时改为实际位置 | 保留上个 swing 的目标，漏迈步不会被重新记为成功 |
| clearance | 站立也有正奖励 | 相对站立高度的对比奖励，原地不抬脚时基准为零 |
| 离地信号 | 二值 | 增加连续卸载项；抬脚/离地奖励要求另一脚支撑 |
| Touchdown | 晚期接触且误差<5 cm | 还必须在本半周期发生过离地，防止小步长课程静站刷奖励 |
| action-rate | 27 维求和 | 27 维平均，作用于滤波后的实际动作 |

新课程跟踪权重：swing XY=4（lift=1）、stance XY=0.5、clearance=4、air=3、unloading=1、stance contact=0.5、touchdown=2、upright=0.5、action-rate=0.05。奖励仍乘 **dt=0.02**。跨奖励版本的绝对 Reward 不可直接比较，应对照同一配置的零动作和策略表现，并使用物理成功率。

平地目标使用 yaw 和固定地面高度：测试中支撑脚 pitch=0.4 rad 会让旧目标落到地面以下 **−0.04673 m**，新目标为 **0 m**。Actor 的 103D 接口保持一致。

合成奖励测试：中摆动时，lift 课程的站立 reward 约 **0.035**，正确抬升 2 cm 且另一脚支撑时约 **0.195**。这是反事实奖励检查，不是动态稳定性证明；见 `reward_landscape.png/csv`。真实周期屈腿 rollout 虽提高离地率，仍未通过左右脚成功半周期门槛。

受控消融（candidate_v1，同 seed、相同 σ=0.15，32×500）：α=0.2 时跌倒 **51** 次，关闭滤波 **192** 次；改回统一 0.1 scale 后 **0** 次，但离地率也为 **0**。说明“少跌倒”和“有足够抬脚权限”需要同时满足。原始配置和对应源码保存在 `candidate_v1/`。

## 4. PPO 检查及推荐起点

- baseline_v1 的真实自适应 LR 为约 `1e-5～3.375e-5`，最后 `1.5e-5`；不能把失败简单归因于初始 `3e-4` 太大。训练回合平均约 86–151 步，最新 106.4 步，和本次随机回放中的实际跌倒相符。历史 run 未保存完整配置，样本数按用户记录为 153,600。
- 候选：**64 env × 64 rollout steps=4096 samples/update**，4 epochs、4 mini-batches、固定 LR=`1e-4`、initial std=`0.15`、entropy=`0.002`。网络 `[256,128]` ELU，actor/critic 均归一化；γ=0.995、λ=0.95、clip=0.2、value coefficient=1、grad norm=1、episode=10 s、终止高度 0.55 m、直立投影阈值 0.5。
- GAE 的衰减长度约 `1/(1−γλ)=18.3` 步，约 0.37 s；这也是采用连续卸载和高度奖励、而不单靠晚期 Touchdown 的原因。
- RSL-RL 5.5.1 已做 advantage normalization；当前没有额外 reward normalization。本轮没有改变 reward 的 dt 积分约定。
- 3 次更新对照：`3e-4` 的 surrogate loss 出现 0.2800；`1e-4` 为 −0.0154/−0.0144/−0.0313，二者分别累计 99/96 次 terminated。样本太少，不能据此宣称显著优越，只将 `1e-4` 作为保守起点。
- 64 步归一化预热的短程实验有 126 次 terminated，没有明确收益，所以**默认关闭**；可用 CLI 单独消融。
- 64 env 短训练点测 GPU 已用约 0.40 GiB，训练产物记录实际数值；这不是显存峰值或长训吞吐保证。推荐先用 64 env/headless，获得稳定曲线后再扩。
- 新模型 3 次更新后的评估：成功半周期左右均 **0**、terminated=160、truncated=0。即使向前速度读数为正也不构成行走证据。

## 5. 课程门槛和执行

先训练 `stand`。零动作 stand 对照存活率为 **100%**，平均 root height=**0.96458 m**，最低有效高度=**0.96398 m**，平均 upright=**0.99974**；这不是训练策略的成绩。

| 阶段 | 晋级门槛：至少 64 env、1000 steps，并检查多个 seed |
|---|---|
| stand | 无跌倒环境≥95%，最低有效 root height≥0.85 m，平均 upright≥0.95 |
| lift | stand 门槛；左右成功半周期各≥80%，至少连续 8 个交替半周期，不能仅看 air 帧占比 |
| step | 同上；late XY error≤0.03 m，平均前进速度≥0.025 m/s，横向漂移和转向不过大 |
| walk | 无跌倒环境≥95%，左右成功半周期各≥90%，late XY error≤0.03 m，前进速度≥0.06 m/s；再扩训练预算和指令范围 |

评估器的成功半周期要求：中段离地高度>`max(0.01,0.6×swing_height)`、Fz<20 N；晚段接触且 XY error<3 cm；中摆动持续有另一脚支撑。`consecutive_8_success_fraction` 统计整段无跌倒且至少连续成功 8 个半周期的环境比例。**当前所有试训 Policy 都未达到晋级门槛。** 横向/航向和滑移仍需结合 trace/可视化复核，不能由均值替代。

完整可执行命令见 [RUNBOOK.md](RUNBOOK.md)。推荐每阶段先做 100 iterations 的有界试训、评估后决定预算，不自动晋级。跨阶段使用 `--warm_start`（仅 actor，重置 critic/optimizer）；同阶段用 `--resume`。训练保存 env/agent 配置、源码 SHA256、版本、seed 与 checkpoint；回放恢复控制参数，避免新旧 action scale 混用。

后续速度、停止和转向需要可观察的命令/足印采样与相应奖励。当前 critic 也还是 103D；论文的 privileged base linear velocity 可作为后续独立消融，不能直接混入旧模型。

## 验证、文件及参考

- CPU：`test_observation.py`、`test_goal_sampler.py`、`test_rewards.py`、`test_robot_cfg.py`、`test_preflight.py` 通过。
- GPU：Goal/Reward smoke、123 案例 authority、各阶段零动作/随机诊断、短 PPO、stage transfer、resume、独立 checkpoint 配置恢复通过。这里的“通过”表示流程和断言通过，不表示学会步态。
- 静态诊断：36 个 Python 文件，0 errors / 0 warnings，见 `typecheck.json`。
- 主要修改：`tasks/foot_placement/{foot_placement_env.py,foot_placement_env_cfg.py,rewards.py,goal_sampler.py}`、`configs/{s54_robot_cfg.py,s54_physics_cfg.py}`、`scripts/{train.py,play.py,smoke_reward_env.py}`。
- 新增：`configs/s54_curriculum.py`、`scripts/{action_authority_test.py,preflight_signals.py,test_preflight.py,reward_landscape.py}`。
- 核对原始数据：`rollout_summary.json`、`pilot_summary.json`、`authority_all/results.json`、各 `.log/.jsonl`。修改前快照 `before_source.tar.gz`，修改后快照与逐文件 SHA256、补丁一并保留。

参考：[Mind Your Steps](https://arxiv.org/pdf/2606.08253) 的分解式足印追踪、保留站立阶段目标和分阶段训练思路；[参考奖励实现](https://github.com/MontenegroAlessandro/loco-mujoco/blob/master/loco_mujoco/core/reward/foot_placement.py)、[参考目标实现](https://github.com/MontenegroAlessandro/loco-mujoco/blob/master/loco_mujoco/core/observations/goals_foot_placement.py)。本实现是 S54 的初期课程，不是论文完整复现；没有移植 Booster T1 的关节增益、几何或动作尺度。读取的参考源码 URL/日期/内容哈希见 `references.json`。
