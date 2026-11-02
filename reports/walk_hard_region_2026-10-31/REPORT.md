# 大步幅、高步频专项续训准备

日期：2026-10-31（Asia/Shanghai）。本次没有启动正式长训练。

## 诊断结果

模型：`logs/walk/2026-10-30_22-45-31_walk_4stage/training/00003_walk_4/model_interrupted.pt`。

使用原任务奖励、控制、105D observation 和确定性策略；每组 32 个环境、1000 个控制步（20 s）、seed=42，固定步长和左右等频率。从默认静止姿态 reset，无加速预热。无跌倒比例表示整个窗口未发生 terminated 的环境比例；正常 timeout 不算跌倒。固定初态的并行环境不视作独立随机试验。

| 目标步长 | 左右频率 | 20s无跌倒比例 | terminated / truncated | 左/右半周期成功率 | 摆动末期XY误差 |
| --- | --- | --- | --- | --- | --- |
| 25 cm | 1.2 Hz | 100% | 0 / 64 | 91.7% / 90.9% | 1.11 cm |
| 40 cm | 1.2 Hz | 100% | 0 / 64 | 16.9% / 65.1% | 3.97 cm |
| 25 cm | 2.0 Hz | 100% | 0 / 64 | 80.0% / 84.2% | 2.50 cm |
| 40 cm | 2.0 Hz | 0% | 288 / 0 | 0% / 0% | 34.25 cm |

原始输出在 `baseline_summary.json` 及每个 case 的 `.json/.jsonl/.log`。奖励重算误差和相位推进误差四组均为 0；结果没有显示这两项的时间对齐错误。

结论：当前模型能处理部分大步慢走和短步快走，尚未掌握大步快走组合；大步慢走也有明显左右落脚差异。不能用宽分布平均分或“配置中训练过该范围”代替固定角点验证。40 cm / 2 Hz 的名义参考速度是 1.6 m/s，实际速度不等于这个指令值。

## 为什么优先调整采样分布

原末阶段：步长独立均匀采样 `[0.05,0.40] m`，左右频率分别独立均匀采样 `[0.5,2.0] Hz`，每 episode 固定。

落入“步长≥0.30 m、左右频率均≥1.5 Hz”的命令采样概率仅为：

`(0.10/0.35) * (0.50/1.50)^2 = 3.17%`。

更接近角点的“步长≥0.35 m、左右频率均≥1.8 Hz”仅约 0.254%。这些是 reset 时的理论命令概率；困难命令提前终止，还会进一步改变其在状态样本中的占比。

上一轮 state.json 记录 walk_1/2/3 的评估均未整体通过，仍按更新数推进。walk_4 在时间预算到达时保存了 interrupted checkpoint，没有完成阶段后的固定角点评估。因此本轮直接使用单一困难分布，不再自动扩大难度。

## 首轮专项配置

| 项目 | 原末阶段 | 本轮 |
| --- | --- | --- |
| 步长范围 | 5–40 cm | 30–40 cm |
| 左右频率范围 | 各0.5–2.0 Hz | 各1.5–2.0 Hz，仍独立采样 |
| 初始checkpoint | 上一阶段 | 上述最新walk_4模型，完整resume |
| 环境数 | 2048 | 2048 |
| 学习率 | 1e-4 | 1e-4 |
| rollout / minibatches | 64 / 4 | 64 / 4 |
| 首轮时间预算 | 12h课程 | 约3h，包含初始化和评估 |
| 独立评估 | 原课程阶段结束时 | 每1000次PPO更新 |

PPO网络、优化器历史、观测归一化、已学习的动作分布、奖励、action scale、PD、平滑系数和终止条件均沿用。显式学习率在完整恢复之后应用。训练区间中的指令仍随机，评估额外固定步长及左右频率角点，包含 40 cm / 2 Hz / 2 Hz。

恢复后的动作分布平均标准差约3.03，并非配置里的初始0.15。直接检查checkpoint可见：左右髋pitch约0.081/0.106、膝约0.031/0.073；较大的值主要来自手臂、腰和部分yaw/roll关节。不能把总体均值当成所有关节的探索尺度。本轮沿用这些已训练参数，未额外重置噪声。

单阶段配置的 `updates=3000` 是参考更新目标；`hold-last` 继续保持相同分布到时间预算。预算优先，不保证3小时一定完成3000次更新，也不以最低更新目标作为“训练通过”的标准。

## 修改文件及启动

- `configs/walk_hard_region.json`：单一困难区间。
- `scripts/train_walk_hard_region.sh`：默认3h、SwanLab online、最新checkpoint、输出到 `logs/walk/*_walk_hard_region`；支持 `--checkpoint`、`--hours`、`--output-dir`、`--resume-dir`、`--dry-run`。
- `scripts/train_walk_12h.sh`：增加 `--chunk-iterations` 选项，默认50000保持原行为；帮助文字改为读取实际 `save_interval`。当前配置保存间隔为250次更新，没有改动。
- `scripts/train_4060.py`：将末阶段和预算提示改为适用于任意阶段数量的文字，训练与升级逻辑未调整。
- `tests/unit/test_command_curriculum.py`：覆盖新入口的范围、完整resume、无副作用预览和断点预算/云端身份恢复。

预览（不训练、不上传、不创建实验）：

```bash
bash scripts/train_walk_hard_region.sh --dry-run
```

正式启动（由用户执行）：

```bash
cd /mnt/sage_data/workspace/KUAVO_S54_Newton
bash scripts/train_walk_hard_region.sh
```

自选checkpoint或预算：

```bash
bash scripts/train_walk_hard_region.sh --checkpoint /absolute/path/model.pt --hours 6
```

恢复同一实验（继承剩余总预算和SwanLab身份）：

```bash
bash scripts/train_walk_hard_region.sh --resume-dir logs/walk/实际时间戳_walk_hard_region
```

## 验收与复现

保留本报告中的四组固定点作为前后对照，至少换seed复核，另保留原宽分布评估观察遗忘。正式训练的 `launcher.json/env_cfg.json/agent_cfg.json/manifest.json` 和 SwanLab 保存范围、优化参数、初始checkpoint、源码哈希和曲线。

优先查看固定最大角点的 `terminated/truncated`、20秒无跌倒比例、左右半周期成功率、末期XY误差和连续8次交替成功率。现有严格gate要求无跌倒比例≥95%、左右成功率各≥80%、连续8次交替成功比例≥80%、末期XY误差≤3 cm，并检查朝向、姿态和前向速度；全部通过才算通过，训练时长耗尽不是通过。

41项CPU回归、shell语法检查和启动预览均通过；Pyright检查为0 error / 0 warning。GPU训练Smoke以128个环境完成2次PPO更新，指标全为有限数，并保存 `model_final.pt`。恢复方式为full_resume，105→105无需迁移，学习率1e-4。实际输出及恢复参数见 `ppo_smoke.log`、`ppo_smoke/manifest.json`、`ppo_smoke/metrics.jsonl`；该短测试仅验证恢复和执行，不证明已改善步态。启动时远程地面资产服务器超时后使用了已有本地缓存，最终运行正常退出。

本轮不宣称40 cm / 2 Hz一定可达。若专项训练仍无改善，需要进一步测量分关节动作/力矩饱和、真实足端可达范围和静止启动过程，再决定控制或奖励的下一项消融。
