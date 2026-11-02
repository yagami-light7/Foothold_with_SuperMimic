# 第一步直行优化：固定足印参考 + 方向奖励

这次从 `00052_step/model_final.pt`（上一轮累计5300次更新）迁移。保留27维动作、103维观测、3cm水平偏移、0.75Hz步态、逐关节action scale、alpha=0.2和平地/PD/力矩限制。新逻辑通过独立的 `straight` 阶段启用，旧stand/lift/step/walk的行为保留。

## 1. 先理解要解决的数据流

旧逻辑：当前支撑脚位置和朝向 → 新足印位置与姿态 → 16维goal → policy。支撑脚转动时，足印方向会一起转。

新逻辑：reset记录根部参考yaw → 保持参考yaw不变 → 当前支撑脚位置 + 参考方向旋转的足印偏移 → 16维goal → policy。

世界目标公式：

```text
p_target = p_stance + Rz(reference_yaw) @ [step_length, ±feet_distance, 0]
q_target = yaw_quaternion(reference_yaw)
```

仍用实际支撑脚的位置作为目标起点，缓存的支撑脚世界目标继续保持不动。修改的是旋转参考，不是把足印全部固定在最初位置。

## 2. 写角度辅助函数

文件：`tasks/foot_placement/heading.py`。

- `yaw_from_quat()`：使用Newton/Isaac Lab当前的xyzw约定读取yaw。
- `quat_from_yaw()`：生成只含yaw的参考四元数。
- `heading_error()`：`atan2(sin(yaw-reference), cos(yaw-reference))`，把误差包回[-pi,pi]。例如179°与-179°相差2°，不能直接得到358°误差。

这些函数没有仿真依赖，可在CPU上单测。

## 3. 把参考方向接到Goal Sampler与Reset

文件：`tasks/foot_placement/goal_sampler.py`。

新增 `fixed_heading_targets=False`，默认关闭以兼容旧模型；新增每环境的 `reference_heading_w` / `reference_quat_w` 缓冲区。

`reset()`仅更新传入env_ids的参考。环境传入本次根部reset四元数；直接使用sampler而不传reference时，才回退到右脚朝向。

`_target_frame_quat()`决定生成足印用哪个方向：

```python
if self.fixed_heading_targets:
    return self.reference_quat_w[env_ids]
return self._target_quat(stance_quat)
```

文件：`tasks/foot_placement/foot_placement_env.py`。全reset和部分reset都传入根部参考；切换支撑脚不更新参考。

目标姿态四元数仍通过已有16维goal进入观测，表达目标相对于实际支撑脚的方向差。因此本次网络输入维度保持103，旧actor权重可按原形状加载。输入分布发生变化，所以迁移后仍需微调和检查归一化。

## 4. 加两个方向奖励

文件：`tasks/foot_placement/rewards.py`。新增的权重默认都是0，旧奖励数值不变。

```text
e_heading = wrap(root_yaw - reference_yaw)
r_heading_rate = 1.0 * exp(-(e_heading / 0.35 rad)^2)
r_yaw_rate    = 0.5 * exp(-(root_angular_velocity_body_z / 0.30 rad/s)^2)

reward = old_reward + 0.02 * (r_heading_rate + r_yaw_rate)
```

- heading约束累计朝向，yaw项抑制来回扭动；只约束其中一个容易漏掉另一类问题。
- sigma是容忍尺度，误差等于sigma时分项为最大值的`exp(-1)`，不是硬门槛。
- 每步最大heading贡献0.02、yaw贡献0.01，合计0.03。
- 所有旧足端奖励保持原权重。这轮不添加动作幅值惩罚，不改上肢控制，以便先识别参考方向和方向奖励的效果。
- 这轮yaw目标固定0，不是可控转向任务。未来有转向命令时，应跟踪`yaw_rate_cmd`，而非一律要求yaw角速度为0。

## 5. 配置独立阶段与兼容旧回放

文件：`configs/s54_curriculum.py`、`foot_placement_env_cfg.py`。

`straight`沿用step的动作尺度与足端奖励权重，并启用固定朝向与上述两个奖励。需要更正：本轮straight实际抬脚目标为3cm，旧step为2cm；之前“全部足端参数不变”的说明不准确。checkpoint中保存的实际数值是评估依据。保存配置包含新字段；恢复旧JSON时，新开关默认False、新奖励权重默认0。

两个单因素消融：

```text
--stage straight --ablation targets_only   # 只固定目标参考，方向奖励权重0
--stage straight --ablation rewards_only   # 目标仍随支撑脚转，但增加方向奖励
```

`scripts/train.py` / `play.py`加入straight选项；`scripts/train_4060.py --start-stage straight`启动只含straight的独立课程，不需要重训站立和抬脚。旧默认课程顺序不变。新run中Progress/stage=0代表该课程的第一项straight，应结合curriculum配置及`straight/*`指标前缀理解。

## 6. 迁移actor，处理旧归一化统计

使用`--warm_start`：迁移actor（包括原均值、方差与动作分布），critic、optimizer和更新计数重新初始化。奖励变化后，不直接resume旧step优化器。

旧actor normalizer count约694,681,600。首次试验显式用了`--normalization_prior_rollouts 8`：保留均值/方差，只把历史有效样本数折算为最多8个新rollout，即2048×64×8=1,048,576，让新目标分布更快更新统计。

代码实际比较折算前后同一输入的确定性动作，要求逐元素完全一致，并记录`max_initial_action_change`，首轮实测为0。然而100次更新后的步态出现明显回退，说明“初始动作不变”不能保证后续快速适配不破坏旧控制。

因此最终默认设置为`--normalization_prior_rollouts 0`，保留原始历史计数，并启动第二组只改变该参数的受控试验。正数折算保留为显式消融功能，不作为默认推荐。相同straight阶段的resume保留完整统计，不会每100次再次折算；同时恢复checkpoint旁的env_cfg，避免后来编辑profile默认值悄悄改变恢复的任务。

## 7. 先测试，再开始训练

CPU测试：

```bash
PYTHONPATH="$PWD" /mnt/sage_data/workspace/IsaacLab_3_0/.venv/bin/python tests/unit/test_heading_control.py
PYTHONPATH="$PWD" /mnt/sage_data/workspace/IsaacLab_3_0/.venv/bin/python tests/unit/test_preflight.py
PYTHONPATH="$PWD" /mnt/sage_data/workspace/IsaacLab_3_0/.venv/bin/python tests/unit/test_train_4060.py
PYTHONPATH="$PWD" /mnt/sage_data/workspace/IsaacLab_3_0/.venv/bin/python tests/unit/test_rewards.py
```

检查覆盖：旋转参考90°、支撑脚旋转但参考保持、部分reset不影响其他环境、179°/-179°的角度边界、奖励精确公式、旧配置恢复、新门槛不接受无有效heading样本。

GPU检查已经执行：旧Goal集成、64环境3次PPO actor迁移、1000步奖励/phase/reset重算，以及旧step和straight targets_only的对照（后者同时采用3cm抬脚目标）。结果保存在本目录JSON；训练前old-step与fixed-target的最大heading偏差分别约31.6°和13.4°。后者落脚成功率出现回退，证明需要微调，不能只换目标后就宣称策略完成。

## 8. 本次启动的2048环境短试训

第一组明确使用8个rollout统计先验，100次更新后未通过，保存到`logs/straight/2026-10-29_heading_v1_pilot`。下面是第二组保留统计的实际启动命令；输出目录不允许重复创建，先查看状态和日志，再决定是否继续。

```bash
bash scripts/train_4060.sh \
  --num-envs 2048 --start-stage straight \
  --warm-start logs/step/2026-10-29_step_5300/model_final.pt \
  --max-iterations 100 --chunk-iterations 100 --max-hours 0.5 \
  --print-every 10 --normalization-prior-rollouts 0 \
  --output-dir logs/straight/2026-10-29_heading_v1_keepnorm
```

查看日志：

```bash
tail -f logs/straight/2026-10-29_heading_v1_keepnorm/console.log
cat logs/straight/2026-10-29_heading_v1_keepnorm/state.json
```

本次实际使用的续训命令如下，累计300次更新现已完成，无需重复执行：

```bash
bash scripts/train_4060.sh \
  --resume-dir logs/straight/2026-10-29_heading_v1_keepnorm \
  --max-iterations 300 --chunk-iterations 100 --max-hours 0.5 --print-every 10
```

SwanLab在新实验中保存`straight/Reward/heading`、`straight/Reward/yaw_rate`、`straight/Tracking/heading_error_rad`、`straight/Tracking/yaw_rate_abs`及独立Eval/straight指标。增加奖励后总reward尺度改变，不能直接与旧step的0.147–0.150按大小判断步态进步。

## 9. 判断结果

继续保留存活≥95%、左右半周期成功率各≥80%、连续至少8个成功半周期环境比例≥80%、落点误差≤3cm、upright≥0.95、最低高度≥0.85m、绝对侧向速度均值≤0.1m/s、绝对yaw角速度均值≤0.15rad/s。

straight还要求：有效heading样本存在、沿reset参考方向的前向速度≥0.02m/s、最大有效heading偏差≤10°。这里的速度沿固定参考方向投影，避免机器人边转边沿自己的body x前进也被当作直行。

每次独立评估仍是64环境×1000步、两段10秒episode。通过两次短评估后，还应做更长的无中途reset轨迹和初始扰动评估，再决定是否开始更长课程。尚未加入速度命令、停止或按指令转向。

## 文件与实验追溯

本目录保留修改前源码tar和SHA256、修改后快照及diff、优化前对照JSON、smoke checkpoint和数值测试结果。训练目录保存实际env/agent配置、源码SHA、manifest、TensorBoard、metrics.jsonl和checkpoint；SwanLab链接见该目录state.json。


## 本次300更新结果与回放

2048环境试训已经结束：39,321,600 samples，0次terminated、128次truncated，存活比例1.0；参考前向速度0.0430m/s，但左右半周期成功率为0/0.865，最大heading偏差15.55°，连续8半周期成功率0。该模型尚未通过直行标准，不推荐直接扩大长训。

下面命令可观察这次试验的实际动作，不能把运行成功等同于步态成功：

```bash
PYTHONPATH="$PWD" uv run \
  --project /mnt/sage_data/workspace/IsaacLab_3_0 --no-sync \
  isaaclab -p scripts/play.py \
  --checkpoint logs/straight/2026-10-29_heading_v1_keepnorm/training/00002_straight/model_final.pt \
  --num_envs 16 --steps 1000 --viz newton_gl
```

play自动恢复保存的straight环境配置；脚本前半环境执行策略，后半环境执行零动作基线。不要额外指定其他stage覆盖保存参数。关键实现与测试现已补中文注释；补注释前后可执行AST一致，Pyright仍为0错误、0警告。


## 2026-10-29日志整理及可视化补充

旧日志路径已归类，完整索引见`../../logs/README.md`，可恢复备份见`../../.log_archive/2026-10-29_before_cleanup/logs`。回放增加`--show-goals`（默认开启）和`--policy-only`。原sampler默认偏移是[0.12,±0.19]m，但step/straight课程覆盖为[0.03,±0.19]m；19cm是脚间距。回放打印checkpoint中的实际水平偏移与抬脚高度，两者不会混用。
