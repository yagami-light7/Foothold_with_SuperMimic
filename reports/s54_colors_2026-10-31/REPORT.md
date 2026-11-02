# S54 官方风格配色

> 此报告保留首版方案记录。现用方案已按用户反馈替换为 [SuperMimic 灰阶配色](../s54_supermimic_colors_2026-10-31/REPORT.md)。

主训练环境和 `scripts/play.py` 已默认使用银白外壳、黑色面罩、深色关节、黑色手掌和脚部。

参考资料：

- [乐聚官方 KUAVO 5 产品图](https://www.letools.lejurobot.com/assets/kuavo5.png)，用于外观配色。
- [乐聚官方全身产品图](https://kuavo.lejurobot.com/leju_website/images/5e1133f6ed346b7b406dc0ad131eec63.png)，用于核对关节、手部和脚部颜色。
- [官方 S54 CAD/URDF](https://github.com/LejuRobotics/kuavo-ros-opensource/tree/main/src/kuavo_assets/models/biped_s54)，用于核对模型。URDF/MJCF 中多数网格是统一白色，不能完整表达实机外壳配色。

官方未公布精确 RGB 色号；本次按官方 KUAVO 5 系列产品图，在现有 S54 网格上近似复现。没有新增外壳、Logo 或改变模型轮廓。Newton GL 通过 Isaac Lab 的 OmniPBR 颜色兼容层读取基础色，实际金属光泽依赖渲染器。

## 配色和文件

| 材质 | sRGB 近似色 | 部位 |
| --- | --- | --- |
| ShellSilver | `#CCD1D6` | 胸腹、外壳、大腿、小腿和部分手臂 |
| JointGraphite | `#1F2429` | 髋部、肩部和肘部等关节 |
| MetalSteel | `#858C94` | 踝部金属结构 |
| RubberBlack | `#0B0D0F` | 颈部、腕部、手掌和脚部 |
| FaceBlack | `#06080A` | 前额和眼部面罩 |

- `assets/biped_s54_colored.usda`：新外观图层，以相对路径引用原始 USD。
- `configs/s54_robot_cfg.py`：默认资产路径改为上述外观图层。PD、力矩限制和初始姿态未调整。
- `scripts/recolor_s54_usd.py`：带中文注释的生成脚本；色值集中在 `PALETTE`，关节部位分配在 `link_material()`。

头部 CAD 原本把面罩和外壳合并成一个 STL。本次按局部坐标把 4750 个已有三角面划为黑色面罩，保留原顶点位置和三角面集合；不增加碰撞形状。其余部位采用祖先材质绑定，保留网格实例共享。

[修改前正面](before_front.png) · [修改后正面](after_front.png) · [修改后侧面](after.png)

## 验证

- 原始 `assets/biped_s54.usd` 保持不变，SHA256：`2a912a12f69b3508998385f67f937bc6a3c94dd5794a0ae1d71a4e9b57ec5607`。
- 197 个非视觉/物理 prim 的属性、关系及 API 配置一致；564962 个视觉三角面的顶点坐标、朝向和世界变换一致。见 `asset_validation.json`。
- Newton 导入后均为 31 个 body、31 个 joint；质量、惯量、质心、关节类型/连接/初始状态/限位数组完全一致。总 shape 数从 81 变为 82，新增项仅用于绘制面罩。见 `newton_validation.json`。
- Newton GL 离屏渲染已检查，见本目录 PNG 和 `preview.log`。
- 40 项 CPU unit tests 通过，机器人配置检查通过；Pyright 对 54 个源文件检查为 0 error / 0 warning。见 `unit.log`、`robot_cfg.log`、`pyright.json`。
- 使用现有 `00003_walk_4/model_interrupted.pt`，2 个环境、50 个控制步回放成功，105D observation / 27D action 正常，无跌倒或重置。见 `play_smoke.log`。该短回放用于兼容性检查，不作为新的步态效果评估。

无需迁移网络或重新训练；checkpoint 不保存视觉材质。以后修改颜色后可重新生成外观层：

```bash
cd /mnt/sage_data/workspace/KUAVO_S54_Newton
/mnt/sage_data/workspace/IsaacLab_3_0/.venv/bin/python scripts/recolor_s54_usd.py
```

回放查看：

```bash
cd /mnt/sage_data/workspace/KUAVO_S54_Newton
uv run --project /mnt/sage_data/workspace/IsaacLab_3_0 --no-sync \
  isaaclab -p scripts/play.py \
  --checkpoint logs/walk/2026-10-30_22-45-31_walk_4stage/training/00003_walk_4/model_interrupted.pt \
  --num_envs 32 --steps 2000 --policy-only --show-goals --viz newton_gl
```

保留原始 USD 和外观层两个文件；单独复制外观层会缺少其引用的基础资产。
