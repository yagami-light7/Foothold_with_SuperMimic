# 配置目录

| 目录 | 内容 |
|---|---|
| `motion_prior/stage1/` | LAFAN1 Motion训练：接口smoke、有界验证、3090 Walking训练 |
| `motion_prior/stage2/` | Foot任务、步态目标与课程分布 |
| `motion_prior/prior/` | 上身先验开关及权重 |
| `motion_prior/validation/` | WM校准与消融前能力门槛 |
| `s54/curriculum/` | 既有8小时Foot课程、fresh与smoke配置 |
| `s54/standing/` | 既有Standing任务与训练profile |
| `s54/foot_tracking/` | 既有足轨迹采样参数 |
| `s54/retarget/` | Grounding、自碰撞与上身重定向 |

当前Stage 1配置是 [motion_prior/stage1/stage1_walk_3090.json](motion_prior/stage1/stage1_walk_3090.json)。`train`字段进入训练CLI，`wall_seconds`由启动器处理；实际解析参数保存在每个run的 `args.json`，profile副本为 `train_profile.json`。

目录整理只调整配置路径，原任务/奖励/物理数值保持原样。旧路径映射见 [tools/layout_migration.json](../tools/layout_migration.json)；旧checkpoint加载采用兼容解析，不修改历史文件。
