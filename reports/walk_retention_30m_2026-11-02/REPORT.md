# 半小时能力保持续训：纯训练与分区采样

本轮已完成配置和短测，**未启动正式半小时训练**。按最新要求，启动基线、周期和末尾eval全部关闭。目标是保持已有角点能力，改善中高难度落点；采样覆盖与学习成功分别判断。

```bash
cd /mnt/sage_data/workspace/KUAVO_S54_Newton
bash scripts/train_walk_retention_30m.sh
```

配置：[walk_retention_30m.json](../../configs/walk_retention_30m.json)。入口：[train_walk_retention_30m.sh](../../scripts/train_walk_retention_30m.sh)。可用 `--dry-run` 预览、`--checkpoint FILE` 覆盖起点、`--resume-dir DIR` 延续同一实验剩余预算和SwanLab身份；checkpoint与resume-dir互斥。

## 起点与参数

起点为 `logs/walk/2026-11-01_22-30-22_walk_precision_1h/training/00000_precision/model_final.pt`，即500更新候选，SHA-256：`0dc90b5e4f10c239ea17c4f494d3c847f196b24443ed6d79132fcffcdd107d47`。不用角点已退化的1000更新最终模型。此前同seed角点补测：500更新候选存活100%、XY约12.15cm；1000更新最终模型存活0%、61次terminated。见 [评审](../precision_review_2026-11-01/REPORT.md)。

| 参数 | 本轮值 |
|---|---|
| 总墙钟预算 | 0.5h，含启动、训练和日志；无eval预留时间 |
| 环境 / rollout / mini-batches / epochs | 2048 / 64 / 4 / 4，每更新131072条transition |
| 学习率 / 调度 | `2e-5` / fixed，恢复优化器后再次显式覆盖 |
| episode | 10s = 500控制步，恢复原有站稳后连续步行的训练时长 |
| checkpoint恢复 | actor、critic、优化器、观测归一化完整继承，105→105，无维度迁移 |
| 起步脚 | 每次reset随机左/右 |
| 奖励 | 沿用500更新候选：flat_walk_v2；摆动XY倍率2、首步额外倍率1.5、宽核占比0.25/锐度25，其他v2项沿用 |
| 控制 | Action Scale、PD、力矩限制、动作平滑alpha=0.2不改 |
| 域随机化 | 物理、复位噪声、观测噪声继续关闭，PPO随机动作探索保留 |
| 自动评估 | `evaluation.enabled=false`；启动、周期和结束均不调用eval |
| 训练段 | 最多50000更新的单进程，墙钟截止提前停止；避免每250更新重启仿真 |
| 保存 | 每250更新；结束/时间中断另存final/interrupted，退出保存可能增加数十秒 |

`minimum_updates=250`、`passes_required=2` 和评估工况保留通用计划结构；本轮禁用eval，因此它们不触发晋级、额外更新或自动模型筛选，不影响30分钟预算。`chunk-iterations=50000`不是保存间隔，也不是必须完成50000更新。

## 中高难度15格覆盖

步长边界 `[0.25,0.30,0.35,0.40]` m，频率边界 `[1.0,1.2,1.4,1.6,1.8,2.0]` Hz，划成3×5格。每格内左右同频，每次reset在本格内均匀重采样；组ID不会因跌倒/reset改变。命令预热和周期内重采样关闭。

下表为2048环境的实际整数配额；每格每次rollout的transition数为环境数×64。

| 步长 / 频率 | 1.0–1.2Hz | 1.2–1.4Hz | 1.4–1.6Hz | 1.6–1.8Hz | 1.8–2.0Hz |
|---|---:|---:|---:|---:|---:|
| 25–30cm | 41 | 41 | 41 | 41 | 41 |
| 30–35cm | 51 | 51 | 410 | 409 | 102 |
| 35–40cm | 41 | 41 | 41 | 41 | 41 |

另有基础组410环境（20–25cm、左右独立1–1.2Hz）及角点组205环境（固定40cm、左右2Hz）。合计2048环境。重点30–35cm、1.4–1.8Hz占约40%，但其余13格均有固定配额。15格至少2624条/rollout，角点另有13120条/rollout。

保证的是定义网格内的**命令样本覆盖**，不保证连续域每个实数都被取到、不保证每条轨迹走满10秒，也不表示各格都已走稳。左右独立高频的三维组合不属于本轮15格覆盖承诺；独立左右频率训练保留在基础组。

检查机制：

- 加载计划校验每格都有唯一匹配命令组，拒绝缺格、重复格和预热越界。
- 按环境数分配后，若有格分到0环境，启动GPU前拒绝。
- 每控制步检查实际步长、左右频率及同频约束，记录格内环境数；RSL-RL汇总后乘64得到本rollout真实transition数。
- 监督器逐更新核对各格、基础组及固定角点达到配额；缺日志、漏格或越界停止训练并留证据。

SwanLab在线compact仅新增两条曲线：`Coverage/cell_fraction`应为1，`Coverage/min_cell_transitions`默认应≥2624。每个区间详细统计在实验根目录 `coverage.jsonl`，累计及最近一次统计在 `coverage.json`，原始指标在训练目录的 `metrics.jsonl`。不会按每格展开全部云端曲线。

## 验证输出

- **94项单元测试通过**，覆盖旧功能、无eval执行路径、网格完整性、局部reset、真实命令越界及缺日志停止；见 `unit.log`。
- 改动的9个Python文件Pyright **0错误/0警告**；shell语法和dry-run通过，见 `pyright.json`、`dry_run.json`。
- 在RTX4060上运行监督器真实短测：2048环境、**2次PPO更新，262144条transition**，全程SwanLab offline，无在线上传。PPO采样+学习约9.08秒，监督器总计约24.16秒。见 `supervisor_grid_smoke.log` 和 `supervisor_grid_smoke/`。
- 两次rollout均15/15格覆盖、最少2624条/格；固定角点累计26240条，全部17组累计262144条；`incomplete_updates=0`。见短测目录的 `coverage.json` / `coverage.jsonl`。
- 短测无 `evaluations/`、无 `best_checkpoint.json`、`evaluation_enabled=false`；配置实际LR=2e-5、episode10秒、full_resume 105→105。
- 短测随机探索产生9次terminated、0次truncated，未跑满500控制步。仅证明管线和覆盖统计工作，不能据2次更新判断稳定性或精度改善。本轮**未额外运行eval**。

该报告目录的旧 `ppo_smoke/`、`smoke_corner.json`、`selection_replay.json` 是最新“无eval+15格”要求之前的预检证据，其四组分布与旧评估策略不代表最终配置。本轮最终配置以 `experiment_summary.json`、`snapshot/` 及 `supervisor_grid_smoke/` 为准。

## 结果读取与改动文件

新实验保存在 `logs/walk/<时间>_walk_retention_30m/`，终端每10次更新打印摘要并显示剩余墙钟时长。最新模型路径记录在 `state.json` 的 `checkpoint`；时间截止通常保存 `model_interrupted.pt`，不代表保存失败。本轮无自动模型推荐，须训练后按需单独评估，比较角点存活、落点XY和左右首步，不能仅凭reward升高认定改善。

新建JSON、shell、`scripts/rollout_coverage.py`、`tests/unit/test_walk_retention.py`；修改 `configs/walk_training.py`、`scripts/adaptive_walk.py`、`scripts/train_4060.py`、`scripts/swanlab_metrics.py`、`scripts/train_walk_12h.sh`、`tasks/foot_placement/goal_sampler.py`、`tasks/foot_placement/foot_placement_env.py`、`docs/APPENDIX.md`。新增的 `scripts/retention_selection.py` 是可选评估筛选工具，当前计划不启用。

奖励、Action Scale和PD没有修改。附录同步在 [APPENDIX.md](../../docs/APPENDIX.md)；源文件哈希与训练起点见 `experiment_summary.json`，本轮改动快照在 `snapshot/`。旧JSON、checkpoint和历史日志保留。
