# 模型与动作库

后续所有模型、重定向动作及动作数据包统一使用本目录。路径均相对仓库根目录；`registry.json` 内路径相对 `library/`。当前资产对象为 **Roban `roban_s22_0901`**。

```text
library/
├── registry.json
├── model/
│   ├── roban_s22_0901/       # 原始 README、URDF、MJCF、scene、meshes、package.xml
│   └── kuavo_s54/           # S54 历史模型；含原 29 DoF simulation XML 和 USD
├── motion_reference/
│   ├── roban/
│   │   ├── kimodo_fallgetup/
│   │   ├── lafan1/
│   │   ├── snapmogen_roban_50hz/
│   │   └── style100_roban_50hz/
│   └── kuavo_s54/
│       ├── bvh/             # 原 S54 源动作
│       ├── csv/             # 原 S54 重定向动作
│       ├── pkl/
│       ├── retarget_data/   # 历史 reference_startup_fixed 等
│       └── datasets/
│           ├── s54_flat_dataset/dataset_export/
│           └── s54_stair_dataset/stair_up_4.npz
└── manifests/               # 导入 SHA256、旧/新路径映射、模型契约和检查结果
```

Roban 的权威模型为：

- `model/roban_s22_0901/xml/biped_s17_verified_handball_fixedhead.xml`
- `model/roban_s22_0901/urdf/biped_s17_verified_handball_fixedhead.urdf`

模型根 `base_link` 是躯干，腰部与骨盆在其下方；21 个活动关节的 MJCF/CSV 顺序为腰部、左腿 6、右腿 6、左臂 4、右臂 4。头部固定。MJCF 使用 WXYZ 四元数；CSV 表头的 root quaternion 为 **XYZW**。CSV 中 `Frame`、Root pose、21 个关节以外的列是重定向诊断量，后续 Loader 应按列名读取。

MuJoCo 编译后的模型含 37 个非 world body，其中包括相机等固定坐标系；不要把它们全部当作有质量的物理刚体，或套用 S54 的 30-body 顺序。精确名单、质量与源文件哈希见 `manifests/roban_model_contract.json`。

| Roban 动作集 | CSV 数量 | 当前帧率信息 |
|---|---:|---|
| kimodo_fallgetup | 8348 | CSV 无时间戳/帧率声明，使用前确认 |
| lafan1 | 89 | CSV 无时间戳/帧率声明，使用前确认 |
| snapmogen_roban_50hz | 7235 | 源目录声明 50 Hz，CSV 无独立时间戳 |
| style100_roban_50hz | 799 | 源目录/生成日志声明 50 Hz，CSV 无独立时间戳 |

共导入 **16,471 条重定向 CSV**，并保留源目录中的两份生成日志。模型和这些动作从 `/mnt/sage_data/workspace/Foothold/third_party/library` 复制，逐文件校验 SHA256；外部源目录保持原样。Kimodo 原始 BVH 留在该外部目录，当前动作库使用已重定向的 CSV。

本阶段检查资产完整性、模型可编译性和 CSV 表头。轨迹的速度定义、Root 坐标语义、接触与动力学可行性，需要在后续 Roban Loader/Newton 适配阶段验证。当前 `registry.json` 的 `active_robot` 用于资产组织；现有 S54 训练代码仍使用独立的 S54 契约。

在当前机器检查：

```bash
/mnt/sage_data/workspace/Toolbox/IsaacLab_3_0/.venv/bin/python tools/audit_library.py
# 可选：重新读取全部导入文件并复算 SHA256
/mnt/sage_data/workspace/Toolbox/IsaacLab_3_0/.venv/bin/python tools/audit_library.py --verify-hashes
```

在另一工作区导入相同资源（标准 Python 即可）：

```bash
python3 tools/prepare_library.py --source /path/to/third_party/library
```

导入器拒绝覆盖内容不同的同名文件；旧 `data/` 仅在所有文件迁移并通过哈希检查后移除。字节相同的 S54 模型副本合并到已有 `model/kuavo_s54`。迁移后的原始 NPZ、metadata/manifest 和 USD 契约不改写，保留原始哈希及历史路径记录。历史日志或 Checkpoint 中保存的旧路径不会自动更新，手动使用它们时按 `manifests/data_migration.json` 的映射替换。

大体积模型、动作和生成的清单由 `.gitignore` 排除；复制代码仓库到其他机器时，需另行传输整个 `library/`。新文件只按机器人与数据集归档，不再创建顶层 `data/`。
