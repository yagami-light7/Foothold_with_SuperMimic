# 2026-11-14 library 整理与路径迁移

用户明确选择本阶段只整理资源与迁移路径。Roban 的 Newton / Observation / Action / Training 适配为后续工作，本阶段未启动 GPU 训练。

资源已从 `/mnt/sage_data/workspace/Foothold/third_party/library` 复制，外部源文件保留。Roban 导入 16,520 个文件，含完整模型 47 个文件、16,471 条 CSV、两份生成日志，合计约 11.777 GB。Kimodo 原始 BVH 继续保留在外部来源目录。

Roban 的实际模型为 `roban_s22_0901`，21 个活动关节、固定头部、球手，浮动根为胸部 `base_link`。MuJoCo 编译得到 nq=28、nv=27、nu=21，37 个非 world body，其中 26 个有正质量；当前 XML 总质量为 40.69991336 kg。模型原 README 的总质量数值比当前文件旧，后续以模型文件及生成的契约为准。本阶段没有修改模型参数。

| 旧路径 | 新路径 |
|---|---|
| `data/s54_description` | `library/model/kuavo_s54` |
| `data/s54_newton_usd` | `library/model/kuavo_s54/usd` |
| `data/retarget_data` | `library/motion_reference/kuavo_s54/retarget_data` |
| `data/s54_flat_dataset` | `library/motion_reference/kuavo_s54/datasets/s54_flat_dataset` |
| `data/s54_stair_dataset` | `library/motion_reference/kuavo_s54/datasets/s54_stair_dataset` |
| `library/motion_reference/{bvh,csv,pkl}` | `library/motion_reference/kuavo_s54/{bvh,csv,pkl}` |

原布局的 673 个文件已迁移或按相同字节去重，每项 SHA256 都已检查。顶层 `data/` 已移除。33 个现有代码/配置文件的实际路径引用已更新，修改前副本及路径 diff 位于本目录。实验日志、Checkpoint、发布目录、原始 NPZ、metadata/manifest、历史实验快照及资产契约保持原样。

验证结果：

- 每个导入文件在复制时比较源/目标 SHA256，拒绝覆盖内容不同的同名目标。
- Roban MJCF 编译通过，URDF/MJCF 活动关节集合一致，mesh 依赖齐全。
- 全部 16,471 条 CSV 表头通过；Root quaternion 列为 XYZW，21 个关节与 MJCF 顺序相同，其余为诊断列。
- Kimodo、LAFAN1 的 CSV 未提供帧率声明，registry 标为 null；使用它们前应单独确认时间契约。
- 迁移后 S54 的 M3.4 Loader 只加载 14 条 train 动作、13,714 帧，32 个样本的全部 30 bodies FK/速度回归及原 WM/Policy/Critic 梯度检查通过。结果保存到本目录，原测试报告未被覆盖。
- 默认 S54 启动器 dry-run、shell 语法、30 个修改/新增 Python 文件的语法，以及导入器同内容幂等与冲突拒绝检查通过。

大体积资产与生成清单被 `.gitignore` 排除。跨机器部署时需另外传输 `library/`。历史配置里嵌入的绝对路径保留为历史记录；重用历史配置时按迁移映射传入新路径。

入口：`library/README.md`、`library/registry.json`。详细清单：`library/manifests/{roban_import,data_migration,roban_model_contract,library_audit}.json`。本目录的 `verification.json` 保存此次验证摘要。
