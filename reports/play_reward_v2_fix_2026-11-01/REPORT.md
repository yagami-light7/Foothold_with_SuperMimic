# play.py 新奖励回放兼容修复

原因：`flat_walk_v2` 已删除 `swing_air`、`touchdown` 奖励，但回放脚本仍按旧奖励字典读取两项，首次控制步后触发 KeyError。这是奖励改版时漏同步的回放统计接口。

修改文件：`scripts/play.py`。改为按实际足底位置、接触力及本步开始时保存的相位、摆动脚和目标计算：

- Swing-air：局部相位在 (0.2, 0.8)，摆动脚法向力低于接触阈值，支撑脚法向力高于接触阈值，摆动脚高于目标平面 1 cm。
- Touchdown：局部相位≥0.8，摆动脚接触，XY目标误差<3 cm。
- 两项排除已自动 reset 的帧，均以全部控制帧数为分母，与 `preflight_signals.py` 的同名帧占比一致。它们不是半周期成功率，也不再取决于奖励权重和字典键。

本次只修改回放统计，模型文件和训练奖励未改动。

## 已执行检查

| 检查 | 实际结果 |
|---|---|
| 用户 recovery checkpoint，16环境、1000步、右起步、25cm/1.2Hz、policy-only、viz none | 正常退出；Falls/env=0，Timeouts/env=2，末期XY=0.0170m；Reward/step=0.12661，Swing-air ratio=0.584，Touchdown ratio=0.162 |
| 10月30日 checkpoint，16环境、520步、左起步、25cm/1.2Hz，8策略+8零动作对照、viz none | 正常退出；策略与对照均Falls/env=0、Timeouts/env=1；策略末期XY=0.0120m；零动作末期XY=0.2500m、两接触统计均0 |
| Python编译、play.py单文件Pyright | 通过；Pyright 0错误、0警告 |

GPU回放均只执行推理；未启动训练。日志分别是本目录 `recovery_right_headless.log`、`legacy_with_baseline.log`、`pyright.json`。1000控制步=20秒仿真，每10秒正常超时一次。

## 用户回放命令

```bash
cd /mnt/sage_data/workspace/KUAVO_S54_Newton
uv run --project /mnt/sage_data/workspace/IsaacLab_3_0 --no-sync \
  isaaclab -p scripts/play.py \
  --checkpoint logs/walk/2026-11-01_16-14-59_walk_recovery/training/00002_recovery/model_interrupted.pt \
  --num_envs 16 --steps 2000 --policy-only \
  --initial-swing-side right --step-length-range 0.25 0.25 \
  --gait-frequency 1.2 --show-goals --viz newton_gl
```

本次修复验证使用 `--viz none` 无窗口回放；可视化后端保持原接口。
