# S54：6–12cm随机水平步长续训

2026-10-29，Asia/Shanghai。按用户要求只修改代码、验证并提供启动命令，未启动训练、未创建新的SwanLab实验。GPU验证只包含初始化/加载和无梯度回放，PPO更新次数为0。

## 一键启动

```bash
cd /mnt/sage_data/workspace/KUAVO_S54_Newton
bash scripts/history_train/train_random_steps_12h.sh
```

仅查看完整命令，不创建目录、不运行仿真或训练：

```bash
bash scripts/history_train/train_random_steps_12h.sh --dry-run
```

默认12小时墙钟预算，包含初始化、训练、定期独立评估和日志采样；每100更新保存模型并评估。最后阶段连续通过两次也继续训练到预算结束，同时记录通过的模型。50000更新是额外上限，在当前4060吞吐下主要由12小时预算决定停止。

终端会打印运行目录和SwanLab链接。新实验保存到`logs/straight/YYYY-MM-DD_HH-MM-SS_random_step_6to12cm`，终端日志在同名目录旁的`.console.log`，详细训练和评估日志在运行目录内。SwanLab项目为`S54-Locomotion`，online模式，沿用本地登录。已确认SDK版本0.10.3、本地登录文件存在；本次没有执行云端init。

中断后恢复同一个新实验：

```bash
bash scripts/history_train/train_random_steps_12h.sh --resume-dir logs/straight/实际随机步长运行目录
```

恢复时沿用该实验的步长分布、seed和SwanLab ID。不要把旧固定3cm实验目录传给`--resume-dir`；脚本会拒绝用同一个实验偷偷改变目标分布。初次启动已通过`--resume-checkpoint`从旧模型建立新实验。

## 本次实际配置

| 项目 | 设置 |
|---|---|
| 起点 | `logs/straight/2026-10-29_heading_v1_keepnorm/training/00002_straight/model_final.pt` |
| 恢复方式 | 完整恢复actor、critic、优化器、迭代计数及归一化统计；不做actor-only迁移 |
| 水平步长 | 每个环境每episode独立采样`Uniform(0.06,0.12)m`，换脚不重采样 |
| 目标方向 | reset根部参考方向固定，支撑切换不更新参考 |
| 脚间距 / 抬脚目标 | 0.19m / 0.03m，保留起点配置 |
| 环境 / rollout | 2048 / 64，131072 samples/update |
| PPO | 4 epochs、4 minibatches、LR=1e-4 fixed、entropy=0.002、gamma=0.995、GAE λ=0.95 |
| 网络 / 接口 | 原256/128 ELU，103D observation、27D action |
| 探索与归一化 | 恢复checkpoint中已学到的std和统计；不重置为init_std，不折算历史count |
| heading奖励 | 权重1.0，sigma=0.35rad；`exp(-(wrap(yaw-yaw_ref)/sigma)^2)` |
| yaw-rate奖励 | 权重0.5，sigma=0.30rad/s；直行目标角速度为0 |
| 原足端奖励、PD、力矩、action scale、平滑 | 与起点逐项相同 |
| 时长 | 12小时预算；不因最后阶段提前通过而结束 |

单位说明：范围6–12cm是摆动脚目标相对支撑脚的前向偏移，不是抬脚高度。0.75Hz步态的理想名义速度范围约0.09–0.18m/s。同一只脚相邻两次理想落点的间距约为该偏移的两倍。

## 采样与兼容

`FootPlacementGoalSampler`增加可选`step_length_range`及每环境`step_lengths`缓冲区。仅reset传入的env_ids重新采样；足印在每半周期开始缓存到世界坐标。`None`仍使用旧固定步长且不额外消耗RNG。旧checkpoint缺少此字段时按None恢复，旧回放保留原任务。

新范围通过已有goal位置进入103D观测，未增加网络输入维度。play自动恢复新范围并打印；目标足印可视化仍沿用上一轮实现。

## 训练曲线和独立评估

训练新增`straight/Command/step_length_mean_m`、`min_m`和`max_m`，便于确认6–12cm确实生效。继续保留heading/yaw-rate、Reward各分项及terminated/truncated曲线。

每100更新先评估随机分布，再分别评估固定6、9、12cm，均为64环境×1000控制步。SwanLab分别使用`Eval/straight/*`和`Eval/straight/step_060mm/*`、`step_090mm/*`、`step_120mm/*`。固定分档可以避免随机平均值掩盖12cm失败。

通过条件保留存活≥95%、upright≥0.95、最低有效高度≥0.85m、平均绝对侧向速度≤0.1m/s、平均绝对yaw角速度≤0.15rad/s、左右半周期成功率分别≥80%、连续8个成功半周期比例≥80%、末期XY误差≤3cm、最大heading偏差≤10°。前向速度还需达到当前名义速度的至少50%。随机分布和三个固定档必须全部通过，才记录该候选通过；不降低门槛换取通过。

这轮训练检验的是指定范围的水平步长跟随，尚未加入停止、速度指令、转向指令或地形随机化。

## 已执行验证与起点能力

- 14项CPU测试：采样范围、episode保持、换脚、部分reset、固定seed复现、旧配置、方向奖励与调度逻辑均通过；见`cpu_tests.log`。
- 原preflight和reward测试通过；见`legacy_preflight.log`、`legacy_rewards.log`。
- 完整加载64环境：恢复到iteration=300，实际LR=1e-4，采样范围实测0.06085–0.11928m，PPO更新0次。见`load_validation/manifest.json`。
- 实际env/agent JSON对照：PPO配置及所有奖励/控制字段与起点一致，唯一新增任务字段为步长范围；见`configuration_comparison.json`。
- 32环境×1000步无梯度回放：0次terminated、64次truncated、reward重算最大误差0；phase/历史动作/reset/episode内步长缓存断言通过。
- 该旧模型在新目标下左/右半周期成功率仅0.223%/0.893%，末期XY误差8.81cm，参考前向速度0.0414m/s，平均绝对yaw角速度0.3277rad/s。说明6–12cm任务尚未掌握；这些是训练前基线，不是新训练结果。原始记录见`rollout_6to12cm_before_training.json`。
- 静态检查43文件，0错误、0警告；shell语法及dry-run通过。

本次没有运行任何PPO更新，无法预先保证12小时后的步态效果。用户启动后应根据分档评估、左右成功率和累计朝向误差判断，而不是仅依据总reward。

## 修改文件

- `tasks/foot_placement/goal_sampler.py`：episode级随机步长与缓存。
- `foot_placement_env_cfg.py`、`foot_placement_env.py`、`configs/s54_curriculum.py`：字段、恢复和命令日志。
- `scripts/train.py`：显式范围覆盖、完整恢复后的学习率处理、0更新校验模式及checkpoint来源记录。
- `scripts/train_4060.py`：完整checkpoint续训入口、随机/固定分档评估、继续至预算、恢复分布检查。
- `scripts/play.py`、`scripts/preflight_signals.py`：恢复并显示实际范围，评估实际任务及命令采样统计。
- 新增`train_random_steps_12h.sh`、`test_random_steps.py`；补充`test_train_4060.py`。

未修改`rewards.py`中的奖励公式/权重，也未改变机器人PD或action scale。新增逻辑均尽量使用中文注释。修改前后源码、SHA256和patch保存在本目录，checkpoint起点SHA256为`d0bf93ed9f7914ed643c6a887a2ba639655b83194b72e9d8ccca07b33fd92447`。

设计参考：[Isaac Lab命令API](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html)将命令采样范围与重采样时机分开配置。本项目选择episode级采样，属于本任务的实现选择，并非照搬官方速度命令或宣称已验证泛化。
