# S54 四阶段步长与步频课程：12小时续训

按用户选择执行：前三阶段各3000次PPO更新，第四阶段最低目标3000次，并继续到整个实验的12小时预算。12小时为硬上限；若机器变慢，时间先到则保存当前模型，不能保证完成最低12000次。控制、奖励和105维观测沿用源checkpoint，本次没有执行正式训练。

原消息中的长度按米实现，例如`0.05–0.25 m = 5–25 cm`；频率为当前代码的完整左右周期频率，左右各自在范围内独立采样。

| 阶段 | 新实验累计更新范围 | 水平步长 | 左右各自频率范围 | 双上界名义速度 |
|---|---|---|---|---|
| walk_1 | 0–3000 | 5–25 cm | 0.5–1.5 Hz | 0.75 m/s |
| walk_2 | 3000–6000 | 5–30 cm | 0.5–1.5 Hz | 0.90 m/s |
| walk_3 | 6000–9000 | 5–40 cm | 0.5–1.8 Hz | 1.44 m/s |
| walk_4 | 9000后，持续到时间预算 | 5–40 cm | 0.5–2.0 Hz | 1.60 m/s |

名义速度为`2 * f_effective * step_length`，不是实际速度承诺。每阶段3000次PPO更新包含每环境192000个控制步、总计393216000个环境样本。旧实验最近100次更新的采集加学习时间中位数为2.9255秒，12000次约9.75小时，尚未计入初始化、上传和阶段评估，因而第四阶段有机会超过3000次。新实验累计更新从0计数，RSL绝对迭代仍包含源模型的历史。

一键运行：

```bash
cd /mnt/sage_data/workspace/KUAVO_S54_Newton
bash scripts/train_walk_12h.sh
```

默认从本次已核对的最新模型开始：`logs/straight/2026-10-30_18-26-34_stride_cadence_lr105/training/00012_straight/model_interrupted.pt`。源文件SHA256为`e86e8dcaca220caf1d244bbf4257e7ae80d0e6954b736be83d666b0d2fa74048`。

指定其他checkpoint：

```bash
bash scripts/train_walk_12h.sh --checkpoint /绝对路径/model_final.pt
```

只预览，不创建实验、安装SDK、使用GPU或上传：

```bash
bash scripts/train_walk_12h.sh --dry-run
```

继续已经中断的同一课程实验：

```bash
bash scripts/train_walk_12h.sh --resume-dir logs/walk/<本次实验目录>
```

`--checkpoint`新建实验，`--resume-dir`恢复同一实验，二者互斥。恢复时从保存的课程快照继续，保留已用时间和SwanLab身份，不会重置为额外12小时；离线停机间隔不计入已用运行时间。可用`--hours`显式调整总时间上限。更改课程JSON需要新实验，不能借同名恢复悄悄更换目标分布。

训练设置为2048环境、64步rollout、4 epochs、4 minibatches、学习率1e-4固定、gamma=0.995、lambda=0.95、entropy=0.002。完整恢复actor、critic、Adam和归一化统计。输出为`logs/walk/<北京时间>_walk_4stage/`，SwanLab项目`S54-Locomotion`，单一云端实验，训练指标仍以`straight/`开头以保留已有直行任务语义。

前三阶段结束时才保存最终模型并重建下一阶段环境；阶段内每25次更新保存一次checkpoint，不再每250次重启。前三阶段结束分别运行一批评估，包含随机范围、三个固定步长点及四个左右频率角点。评估结果只记录，不阻止按更新数进入下一阶段，也不会将不同难度的结果拼成连续通过。第四阶段默认持续到时间预算，结束时保存`model_interrupted.pt`；最终独立eval需要另行执行，避免超出总预算。

文件组织通常为：

```text
logs/walk/<时间>_walk_4stage/
  launcher.json          # 完整课程内容、初始模型、预算与训练参数
  state.json             # 当前阶段、累计更新、已用时间、当前checkpoint、SwanLab ID
  training/
    00000_walk_1/
    00001_walk_2/
    00002_walk_3/
    00003_walk_4/
  evaluations/
```

如果恢复时需要新进程，编号会继续递增，应以`state.json`中的checkpoint为准。`state.json`的`command_phase_index`从0开始，`command_phase_updates`为本阶段更新次数，`wall_budget_seconds_used`为累计预算消耗。

新增SwanLab指标`Progress/command_phase`取1–4，`Progress/phase_updates`记录本阶段已训练次数，`Progress/phase_target_updates`默认3000；第四阶段前者超过3000是预期行为。`Progress/remaining_wall_hours`和终端`remaining=...h`表示总预算剩余时间。RSL自己的ETA按更新上限推算，不能用它判断12小时任务何时结束。

本次修改5个文件：

- `configs/walk_4stage_12h.json`：用户指定的四档范围和更新目标。
- `scripts/command_curriculum.py`：课程校验、边界选择、分布覆盖和恢复一致性。
- `scripts/train_4060.py`：复用原监督器，实现课程调度、阶段日志、全程时间预算和无副作用预览。
- `scripts/train_walk_12h.sh`：一键启动，支持自选checkpoint、恢复目录和dry-run。
- `tests/unit/test_command_curriculum.py`：8项CPU调度回归。

未修改机器人PD、action scale、平滑系数、奖励权重、Goal Sampler或Observation。最大阶段初始化校验成功，105维输入完整恢复，无维度迁移，机器人PPO更新次数为0。8项课程回归和32项原回归通过，Shell语法检查通过，Pyright结果保存在`pyright.json`。

现有模型在25cm/1.5Hz双上界曾全部跌倒；40cm/2Hz组合尚未被证明可学会。课程按固定更新次数推进是一项实验安排，3000次不是成功保证。应结合`Termination/terminated_fraction`、左右半周期成功率、落脚误差、实际前进速度和动作饱和判断训练效果。

复现CPU测试：

```bash
PYTHONPATH="$PWD" python3 -m unittest tests.unit.test_command_curriculum
PYTHONPATH="$PWD" /mnt/sage_data/workspace/IsaacLab_3_0/.venv/bin/python -m unittest \
  tests.unit.test_train_4060 tests.unit.test_gait_commands tests.unit.test_random_steps tests.unit.test_heading_control
```

`launch_dry_run.json`记录四个进程的实际命令；末阶段的41000是剩余更新上限，不是训练目标，通常会由12小时总预算提前停止。`validate_phase4.command.json`、`validate_phase4.log`和`validate_phase4/manifest.json`记录GPU最大档初始化验证。原代码、变更补丁和SHA256保存在本目录。
