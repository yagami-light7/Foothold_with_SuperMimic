# S54 SuperMimic 灰阶配色

按用户反馈，将首版银白/黑色面罩方案替换为相邻 SuperMimic S54 工程的显示配色。

## 参考与修改

参考工程：`/mnt/sage_data/workspace/supermimic_s54`。
其 `data/s54_usd/biped_s54.usd` 与本项目原始 USD 的 SHA256 相同；`data/s54_description/xml/s54_sim.xml` 中大部分视觉网格也仍为白色。实际显示配色来自 `tools/render.py:apply_render_style`，README_S54.md 的“评估与渲染”也说明了该设置。

| 参考材质 | RGB | 本项目分区 |
| --- | --- | --- |
| POLICY_RGBA | `(0.78, 0.78, 0.78)` | 头部、骨盆、大腿、小腿及手臂部分外壳 |
| POLICY_JOINT_RGBA | `(0.30, 0.30, 0.30)` | 胸部 waist_yaw、髋、踝、肩部及腕部关节 |
| POLICY_DARK_RGBA | `(0.10, 0.10, 0.10)` | 左右手脚末端 |

按网格名称复用参考的部位分配规则。USD 独有的 `waist_camera` 使用参考函数的默认灰白色。材质设为非金属、较高粗糙度，接近参考中的低高光风格；两种渲染器的灯光和着色不同，最终亮度并非逐像素一致。

- `scripts/recolor_s54_usd.py`：三档灰阶配色，中文注释；删除首版面罩拆分逻辑。
- `assets/biped_s54_colored.usda`：从原始 USD 重新生成，仅材质及绑定，约 11.6 KB；完整保留头部和所有网格实例。
- `configs/s54_robot_cfg.py`：仍默认加载该外观层，注释更新为 SuperMimic 风格。

[新配色正面](after_front.png) · [新配色侧面](after.png)

## 验证

- 31 个视觉网格的属性、564962 个三角面、世界变换和实例代理状态与原始 USD 一致。
- 212 个非视觉/非材质 prim 的属性、关系、API 及实例状态一致。
- 对参考 `apply_render_style` 函数做直接对照，全部 31 个网格的显示 RGB 与分配规则一致。新材质覆盖 12 个灰白、15 个灰色和 4 个深灰网格。见 `asset_validation.json`、`asset_validation.log`。
- Newton 原模型与新模型均为 31 个 body、31 个 joint、81 个 shape；质量、惯量、质心、关节连接、初始状态及限位数组完全一致。见 `newton_validation.json`。
- Newton GL 实际离屏渲染通过；Pyright 静态检查 54 个文件为 0 error / 0 warning。见 `preview.log`、`pyright.json`。

没有修改奖励、动作缩放、PD、动力学、观测维度或 checkpoint。直接重新运行 play 即可应用，无需重新训练。

## 固定训练上限回放

最新四阶段 checkpoint 的训练上限为前向目标偏移 0.40 m、左右等效频率均 2.0 Hz。`--gait-frequency 2.0` 会清除恢复配置中的左右随机频率范围；`--policy-only` 使全部环境运行策略。

```bash
cd /mnt/sage_data/workspace/KUAVO_S54_Newton
uv run --project /mnt/sage_data/workspace/IsaacLab_3_0 --no-sync \
  isaaclab -p scripts/play.py \
  --checkpoint logs/walk/2026-10-30_22-45-31_walk_4stage/training/00003_walk_4/model_interrupted.pt \
  --num_envs 32 --steps 3000 \
  --policy-only --show-goals \
  --step-length-range 0.40 0.40 --gait-frequency 2.0 \
  --viz newton_gl
```

2 Hz 对应左右各 0.25 s 摆动、完整左右周期 0.5 s。这些是目标指令，实际步长与速度需要由跟踪结果衡量。

相同覆盖参数已用 2 个环境、50 个控制步做兼容性短回放：目标范围 `[0.400, 0.400] m`、左右频率 `2.000/2.000 Hz`、零动作环境数为 0，回放正常退出。该 1 s 窗口内无跌倒，摆动末期 XY 误差约 0.2332 m、touchdown ratio 为 0；说明参数已生效，但不能据此宣称策略已能准确完成最大步长/频率组合。见 `play_max_smoke.log`。
