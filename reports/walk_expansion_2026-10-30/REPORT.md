# S54 更大步幅与步频续训准备

默认启动命令：

```bash
cd /mnt/sage_data/workspace/KUAVO_S54_Newton
bash scripts/history_train/train_walk_3h.sh
```

自选起始模型：

```bash
bash scripts/history_train/train_walk_3h.sh \
  --checkpoint logs/straight/2026-10-30_18-26-34_stride_cadence_lr105/training/00012_straight/model_interrupted.pt
```

`--checkpoint FILE` 从同任务模型完整恢复 actor、critic、Adam、观测归一化和迭代数，建立新的 walk 实验。`--resume-dir DIR` 继续同一实验目录，沿用该目录的模型、目标分布和 SwanLab 身份，不能与 `--checkpoint` 同时指定。可使用绝对路径；相对路径按项目根目录解释。路径不存在会在仿真和上传之前报错。`--dry-run` 可只查看命令。

本次默认模型为 `logs/straight/2026-10-30_18-26-34_stride_cadence_lr105/training/00012_straight/model_interrupted.pt`，旧实验在 3213 次更新后正常达到时间预算。checkpoint 保存迭代为 10002，恢复后从 10003 继续；105维输入和27维动作保持兼容。源文件 SHA256 为 `e86e8dcaca220caf1d244bbf4257e7ae80d0e6954b736be83d666b0d2fa74048`，准备期间未修改源模型。

| 参数 | 旧实验 | 新实验默认值 |
|---|---|---|
| 随机水平步长 | 0.05–0.20 m | 0.05–0.22 m |
| 左侧频率 | 0.50–1.25 Hz | 0.50–1.35 Hz |
| 右侧频率 | 0.50–1.20 Hz | 0.50–1.35 Hz |
| 环境 / rollout | 2048 / 64 | 2048 / 64 |
| 学习率 / PPO epochs / minibatches | 1e-4 / 4 / 4 | 沿用 |
| gamma / lambda / entropy | 0.995 / 0.95 / 0.002 | 沿用 |
| 时间预算 / 上传 | 3 h / SwanLab在线 | 沿用 |
| 训练进程 | 每250次更新重建 | 一段覆盖50000次上限，通常由3小时预算结束 |
| 模型保存 | 每25次更新 | 沿用；定期保存不重建环境 |

左右频率独立、每个 episode 内固定；一侧摆动时长为 `1/(2f)`，等效完整左右周期频率为 `2*fL*fR/(fL+fR)`。参考速度约为 `2*f_effective*step_length`；新上界名义速度为 0.594 m/s，实际速度需要测量。

输出位于 `logs/walk/<时间>_stride_5to22cm_lr050to135/`，默认 SwanLab 项目为 `S54-Locomotion`。课程 stage 保持 `straight`，因此曲线前缀仍为 `straight/`，保留已有的固定方向参考和方向奖励。动作 scale、PD、平滑系数、奖励权重、3 cm 抬脚目标均沿用源模型。训练期间默认不插入独立 eval；时间到达保存 `model_interrupted.pt`。模型保存和云端上传收尾可能稍晚结束，RSL 按更新次数计算的 ETA 不代表3小时预算。

本次只执行配置校验和确定性回放，没有执行机器人 PPO 更新或启动正式续训。所有回放均使用上述最新模型，64环境、1000控制步、seed=42：

| 回放条件 | 提前终止 / 超时次数 | 20秒无跌倒环境比例 | 归一化episode回报 | 前进速度 m/s | 左/右半周期成功率 |
|---|---:|---:|---:|---:|---:|
| 旧随机范围 | 0 / 128 | 100% | 0.9667 | 0.2023 | 97.0% / 92.3% |
| 候选5–25cm、左右0.50–1.50Hz | 2 / 126 | 96.9% | 0.9457 | 0.2423 | 82.3% / 81.7% |
| 固定25cm、左右1.50Hz | 320 / 0 | 0% | 0.2286 | 0.5418 | 0% / 0% |
| 固定22cm、左右1.35Hz | 0 / 128 | 100% | 0.8592 | 0.4586 | 0% / 84.6% |
| 最终默认5–22cm、左右0.50–1.35Hz | 0 / 128 | 100% | 0.9623 | 0.2262 | 91.3% / 88.5% |

以上结果支持先用22cm、1.35Hz过渡。固定上界虽不跌倒，左脚成功率仍为0、摆动末期XY误差约4.66cm，尚未学好该边界；随机范围的较好均值不能替代边界检查，也不能保证任意非对称组合都通过。25cm和1.50Hz叠加时全部跌倒，不作为默认。用户可显式挑战更大的范围：

```bash
bash scripts/history_train/train_walk_3h.sh --checkpoint /绝对路径/model.pt \
  --step-range 0.05 0.25 --left-range 0.50 1.50 --right-range 0.50 1.50
```

日志修复：环境现在向 RSL-RL 传递完成 episode 的分数向量，由其拼接后求均值。之前先求结束帧均值，使1个失败和99个成功的两帧获得相同权重；回归检查中正确结果为0.892，旧计算会得到0.5。这个修改只影响日志，不改变 PPO 接收的奖励。首次启动仍可能出现少数提前结束的 episode 造成的低分，不能把它们删除。新旧训练的该曲线需留意统计口径变化；本报告中的 eval 一直按 episode 独立计算。

修改文件：

- `scripts/history_train/train_walk_3h.sh`：新增 walk 续训入口、推荐范围、自选 checkpoint 和参数示例。
- `scripts/history_train/train_stride_cadence_3h.sh`：允许指定分段和评估间隔，显示实际起始模型，检查 checkpoint 与 resume-dir 冲突。
- `tasks/foot_placement/foot_placement_env.py`、`rewards.py`：按 episode 数量正确汇总回报日志。
- `tests/unit/test_gait_commands.py`：使用实际 RSL-RL Logger 验证不等数量 episode 的加权结果。
- `tests/unit/test_train_4060.py`：验证自选 checkpoint 覆盖默认、含空格路径、缺失模型、互斥参数及 dry-run。

验证：CPU unittest共32项通过；旧奖励、103维观测、课程配置 smoke 均通过；Pyright 48个文件0错误0警告；GPU `validate_only` 完整恢复105维actor/critic/优化器成功、PPO更新0；全部5组回放的奖励和相位重算误差为0。命令、摘要、修改前源码、SHA256和原始JSON/日志保存在本目录。

复现 CPU 检查：

```bash
PYTHONPATH="$PWD" /mnt/sage_data/workspace/IsaacLab_3_0/.venv/bin/python -m unittest \
  tests.unit.test_gait_commands tests.unit.test_train_4060 tests.unit.test_random_steps tests.unit.test_heading_control
bash scripts/history_train/train_walk_3h.sh --dry-run
```

复现最终推荐范围回放：

```bash
/mnt/sage_data/workspace/IsaacLab_3_0/.venv/bin/python scripts/preflight_signals.py \
  --stage straight --mode policy --num_envs 64 --steps 1000 --viz none \
  --checkpoint logs/straight/2026-10-30_18-26-34_stride_cadence_lr105/training/00012_straight/model_interrupted.pt \
  --step_length_range 0.05 0.22 \
  --left-gait-frequency-range 0.50 1.35 --right-gait-frequency-range 0.50 1.35 \
  --output /tmp/s54_walk_recommended_eval.json
```
