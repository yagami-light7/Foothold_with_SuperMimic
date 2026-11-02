# 固定参考方向与方向奖励：实现与短试训记录

## 完成的代码

- 新增`heading.py`：xyzw四元数读取yaw、yaw四元数生成、角度误差wrap。
- `goal_sampler.py`：reset捕获根部参考heading；新阶段用该固定方向旋转足印偏移及生成目标姿态；支撑切换不更新参考。
- `rewards.py`：新增可选heading/yaw-rate奖励，默认权重0以保留旧任务行为。
- `foot_placement_env.py`：连接全/部分reset、根部状态和新奖励，记录Tracking指标。
- `s54_curriculum.py`：新增`straight`，沿用step的动作、控制和足端奖励参数；默认heading权重1.0、sigma0.35rad，yaw-rate权重0.5、sigma0.30rad/s。
- train/play/监督器接入新阶段，保存和恢复新配置；同阶段resume恢复保存的task配置。
- evaluator增加固定参考方向速度、heading误差与侧向位移；新门槛要求最大有效heading误差≤10°。

旧模型103维输入、27维输出保留。PD、力矩限制、action scale及平滑alpha未修改。step/straight的水平偏移均为3cm；straight抬脚目标实际为3cm，旧step为2cm（此前说明遗漏此差异）。新增的12个/修改的代码文件列表见`changed_files.json`，完整patch及前后源码快照均已保存。

## 实际验证

- 新CPU方向测试3项通过：参考旋转、部分reset、周期角度边界、奖励公式、旧配置恢复等。
- 原预检和reward测试通过；监督器测试4项通过。
- 旧Goal集成测试通过；64环境3次PPO迁移、1000步新旧任务评估均正常完成。
- 所有1000步评估reward重算误差为0，phase/动作历史/reset断言通过。
- Pyright：41文件、0 errors、0 warnings。
- 首次归一化统计折算前后，实际actor动作最大变化为0。
- 新增核心逻辑与测试已补中文注释；对补注释前后的 AST 比较确认计算逻辑未变化。
- 通过SwanLab API读回了heading奖励、Tracking heading误差和Eval最大heading误差，上传已验证。

## 实验结果：指标改善不等于步态成功

起点都是上一轮`00052_step/model_final.pt`，seed42。评估为64环境、每环境1000控制步；每episode10秒，正常无跌倒时共有128次timeout。

| 实验 | 存活比例 | terminated | 最大heading偏差 | 平均绝对yaw角速度 | 左/右半周期成功率 | 末期XY误差 |
|---|---:|---:|---:|---:|---:|---:|
| 原step，未重训 | 1 | 0 | 31.57° | 0.1905rad/s | 0.8571 / 1 | 10.18mm |
| straight targets_only，未重训（抬脚目标亦为3cm） | 1 | 0 | 13.42° | 0.2133rad/s | 0 / 0 | 33.56mm |
| 方向奖励，64环境3次更新smoke | 1 | 0 | 11.25° | 0.2157rad/s | 0 / 1 | 23.19mm |
| 2048环境100更新，统计折算为8rollout | 0 | 384 | 12.11° | 0.2478rad/s | 0 / 0 | 120.51mm |
| 2048环境100更新，保留原统计 | 0 | 256 | 8.09° | 0.2429rad/s | 0 / 0 | 51.99mm |
| 2048环境200更新，保留原统计 | 1 | 0 | 11.97° | 0.2279rad/s | 0.1429 / 0.8571 | 36.67mm |
| 2048环境300更新，保留原统计 | 1 | 0 | 15.55° | 0.3394rad/s | 0 / 0.8650 | 43.52mm |

straight目标配置对照的累计方向偏差减小，但它同时包含2→3cm抬脚高度变化，不能把改善完全归因于固定参考；目标分布变化也影响了旧步态。两个100更新实验均发生步态回退，不能宣称已改善整体行走。保留统计的对照终止次数较少，但不足以把问题只归因于统计折算；当前奖励跟踪、动作饱和及迁移优化仍需后续数据判断。

默认normalization_prior_rollouts已设为0。首次统计折算的实验仅作为失败消融保留；它证明初始函数不变不代表后续统计快速适配一定安全。两组均未重置action scale或PD，避免把控制权限变化混入对照。

## 本次试训已结束

保留统计的实验已完成累计300次更新、39,321,600 samples；2048环境、rollout64、LR1e-4 fixed，每100更新自动评估。监督器状态为budget_exhausted，进程已退出，没有继续长训。

- 运行目录：`logs/straight/2026-10-29_heading_v1_keepnorm`
- 日志：`logs/straight/2026-10-29_heading_v1_keepnorm/console.log`
- SwanLab：https://swanlab.cn/@yagami_light/S54-Locomotion/runs/l664wvkh
- 首次100更新评估：运行目录中的`evaluations/00000_straight.json`。
- 200更新恢复无跌倒，但参考前向速度仅-0.000664m/s。300更新前向速度达到0.0430m/s，左右落脚成功率为0/0.865、连续8半周期成功率为0，方向及落脚门槛仍未通过。terminated=0、truncated=128只表示未跌倒，不能证明走好。

查看最终进度与日志：

```bash
tail -n 30 logs/straight/2026-10-29_heading_v1_keepnorm/console.log
cat logs/straight/2026-10-29_heading_v1_keepnorm/state.json
```

`straight`的所有门槛必须同时通过，不能只看heading变小或总reward升高；落脚、存活和前向推进仍是必需项。当前仍未验证可控速度、停止或按指令转向。

逐步代码讲解、可复验命令及全部参数见`WALKTHROUGH.md`。原始实验对照数据见`comparison.json`。本次没有修改或覆盖上一轮训练checkpoint。

下一轮应先做小规模的参考方向渐进过渡/迁移强度消融，逐项观察双脚对称落脚和连续半周期成功率；不建议直接把本次300更新模型作为正式长训起点。具体过渡参数尚未验证，本次未修改或偷偷启用。

2026-10-29补充：日志已经按stand/step/walk/straight归类，完整映射见`logs/index.json`；历史原始log/metrics保持原文。最新模型可用`logs/straight/latest/model_final.pt`回放，并用`--show-goals --policy-only`观察实际目标足印。
