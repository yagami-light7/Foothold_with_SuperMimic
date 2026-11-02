# recovery → hard_region v2 续训准备

已更新入口并完成短测，**未启动正式长训**。用户随后将默认预算从12小时改为3小时试训。本轮继续扩大步幅与步频；[首步过渡问题](../startup_reward_2026-11-01/REPORT.md)仍需后续单独验收，本次未调整起步准备阶段或首步目标。

## 起点和启动

默认使用用户刚回放的最新recovery模型，完整恢复actor、critic、optimizer和normalizer：

```text
logs/walk/2026-11-01_16-14-59_walk_recovery/training/00002_recovery/model_interrupted.pt
SHA256: 2ea3f29f0792158a2e18010408a68d65e852ceb1aa0f8e6fa2b0cee7295648ac
```

默认2048环境、3小时监督器预算、SwanLab在线compact上传，输出 `logs/walk/<时间>_walk_hard_region_v2/`。

```bash
cd /mnt/sage_data/workspace/KUAVO_S54_Newton
bash scripts/train_walk_hard_region.sh
```

只预览加 `--dry-run`；指定权重加 `--checkpoint /absolute/path/model.pt`；改预算加 `--hours 6`。恢复本轮中断才用其自己的 `--resume-dir`，按快照继续剩余预算和云端身份。

recovery的1000更新模型 `00001_recovery/model_final.pt` 有课程通过记录，最新1103更新模型未完整重新认证。默认选择用户实际回放并短测过的最新模型；需要已有通过记录的版本可显式指定00001。最新权重与“通过全部评估”标签分开。

## 四阶段课程

步长指相对支撑脚的前向偏移，单位m；等频时Hz是完整左右周期频率，单步频率为其两倍，最大角点名义速度约1.6m/s。

| 阶段 | 分配比例与命令 | 至少更新数 | 固定LR |
|---|---|---:|---:|
| bridge | 20%保留；35% 30–40cm/1–1.4Hz；40% 20–30cm/1.5–2Hz；5% 40cm/2Hz | 500 | 1e-4 |
| frontier | 20%保留；70% 30–35cm/1.5–1.8Hz；10%最大角点 | 500 | 1e-4 |
| hard | 20%保留；60% 30–40cm/1.5–2Hz；20%最大角点 | 1000 | 1e-4 |
| robust | hard分组，加原有小扰动和命令重采样 | 500 | 5e-5 |

保留组由旧hard_region的15–25cm、0.8–1.2Hz改为recovery覆盖的 **20–25cm、1–1.2Hz**。左右仍在此小范围独立采样（coupled=false），大步/高频组保持左右等时（coupled=true）。完整评估中的0.8/1.2及1.2/0.8Hz同步改成1.0/1.2及1.2/1.0Hz。这改变了本轮的训练/验收范围，旧快照保留作对照。

困难组、分配比例、阶段顺序、LR和质量门槛沿用之前hard_region。3小时试训的最低更新数由1500/1500/3000/1000改成500/500/1000/500；连续通过次数仍为2。bridge/frontier监控最大角点，hard/robust要求它通过；最后阶段含直接与渐变起步的60秒episode检查。

3小时试训每250次PPO更新评估；每阶段首次及每3段完整扫点。最低更新数、连续两次通过、完整网格通过同时满足才晋级，失败继续当前阶段直到预算结束。**3小时不保证完成全部阶段**，预算包括监督器内启动、训练和评估，退出保存有少量收尾时间。

`minimum_updates`是本阶段累计PPO更新数，不是控制步数或总实验更新数。每更新2048×64=131072条transition。原计划最少7000次更新，按短测稳定更新耗时3.11秒估算约6.05小时，尚不含评估且两次通过要求还可能增加实际等待；新计划最少2500次约2.16小时，给评估和进程启动留出空间。每250次评估约相隔13分钟纯训练时间，实际随吞吐和评估耗时变化。评估仍可能不过而停在bridge；这轮首先验证困难任务能否改善，不强制按时间晋级。

robust沿用质量/惯量0.97–1.03、足端摩擦0.45–0.55、小幅reset/观测噪声和推力扰动，每四次换脚有界重采样。

## 奖励与PPO

全部阶段使用 `flat_walk_v2`、随机左右起步，参数见 [奖励报告](../flat_reward_v2_2026-11-01/REPORT.md)。本次未改变奖励、PD、action scale、动作平滑、接触阈值、终止条件和网络结构。

105D观测、27D动作；rollout64、4 minibatches、4 epochs，每更新131072条transition。gamma0.995、GAE0.95、entropy0.002、fixed LR schedule。recovery末尾LR5e-5，bridge沿用原hard_region的1e-4，robust降至5e-5。归一化历史完整保留；SwanLab默认compact，本地保留全指标。

首步计入原物理统计；当前门槛按全部半周期/起步分组汇总，**尚无独立的“每episode第一步必须通过”门槛**。随机起步分组门槛不能替代首步专项验收。

## 已执行验证

| 检查 | 实际结果 |
|---|---|
| Shell和默认dry-run | 初始12h预览通过；后改3h并复查，见 `launch_preview_3h.json`；4阶段、2048环境、compact/online，均从最新recovery恢复 |
| 显式checkpoint和时长覆盖 | 指定00001模型、0.5h的dry-run正确覆盖 |
| CPU回归 | 73 tests，OK；包含分组、reset、相位、奖励、晋级失败保持和同实验恢复 |
| 2048环境PPO短测 | bridge实际分组，2次更新通过，175项最终指标均有限、保存成功 |
| 恢复验证 | 加载iteration25299；105→105、migrated=false、optimizer展开数0、实际LR1e-4 |
| 显存 | 短测结束设备已用约4.35GB、总约8.18GB；设备采样值，非峰值 |

短测最后更新的分组归一化奖励：retain0.948、stride0.727、cadence0.766、corner0.311；corner每控制步terminated比例约0.996%。这是探索训练遥测，不是确定性成功率，困难角点仍需学习。本次不宣称已掌握40cm/2Hz。

日志与复现文件：`unit.log`、`launch_preview.json`、`override_preview.json`、`ppo_bridge_smoke.log`、`ppo_bridge_smoke/manifest.json`、`experiment_summary.json`。短测权重留在reports，不作为正式默认起点。

## 文件修改

- `scripts/train_walk_hard_region.sh`：默认recovery最新权重，输出标记v2，显式compact。
- `configs/walk_hard_region.json`：调整四阶段保留组及对应非等时评估范围。
- 本报告和旧报告的指向链接。

修改前见 `before_*`，初始12h准备版本见 `snapshot/`、`plan_snapshot.json`，校验见 `source_sha256.json`；3h入口及其校验另存 `train_walk_hard_region_3h.sh`、`source_sha256_3h.json`。旧模型、训练日志和实验快照未改写。

随后补充3h阶段调度修订：旧计划/入口另存 `plan_before_3h_schedule.json`、`entry_before_3h_schedule.sh`；当前版本为 `plan_3h_schedule.json`、`entry_3h_schedule.sh`，预览 `launch_preview_3h_schedule.json`，校验 `source_sha256_3h_schedule.json`。原2次PPO短测验证的命令/奖励/控制均相同；本次只再跑CPU晋级/恢复回归和入口预览，没有启动训练。已创建的实验用自己的launcher课程快照恢复，编辑此JSON只影响新实验。

3h调度检查：默认预算3h、每段250次更新、最低更新500/500/1000/500、连续通过2次；与修改前JSON比较，仅minimum_updates四值改变。全部评估均通过的调度模拟最早在累计500/1000/2000/2500次更新达到各阶段门槛，模拟不代表实际学会。更新 `tests/unit/test_command_curriculum.py` 的旧12h/1000段长预期后，73项回归通过（`unit_3h_schedule.log`）；首次旧预期失败日志另存供追溯。
