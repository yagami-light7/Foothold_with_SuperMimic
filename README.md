# SuperMimic Workspace

后续机器人模型与动作统一存放在 **`library/`**。当前资产整理对象是 **Roban `roban_s22_0901`**：21 个活动关节、固定头部、球手。资产入口见 [library/README.md](library/README.md)，机器可读路径见 [registry.json](library/registry.json)。

Roban 已接入 Foot-conditioned Actor、非对称 Critic、模块化足部奖励和可选地形 World Model，复用原 SuperMimic BPTT Trainer。架构、数据协议和测试结果见 [Roban 文档](docs/ROBAN_FOOT_CONDITIONED_ARCHITECTURE.md)。`bash tools/run_roban.sh smoke --prefix roban_check_new` 运行有界的端到端验收；不启动正式长训练。`run_train.sh` 和 `s54_*` 工具仍是历史 S54 入口。

原 S54 模型和动作已归入 `library/model/kuavo_s54`、`library/motion_reference/kuavo_s54`，现有工具与配置已使用新路径。历史 Checkpoint、日志及数据契约保留原样；迁移清单位于 `library/manifests/`。

资产检查（CPU，无需启动 Newton）：

```bash
/mnt/sage_data/workspace/Toolbox/IsaacLab_3_0/.venv/bin/python tools/audit_library.py
```

原算法保留 Motion Encoder、Shared Decoder、可微 World Model、Critic 与 BPTT。历史 S54 迁移说明见 [docs/S54_PORTING_REPORT.md](docs/S54_PORTING_REPORT.md)。上游算法：[zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed)。
