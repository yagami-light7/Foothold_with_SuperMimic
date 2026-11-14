# S54 Newton GL 楼梯渲染诊断与修复

日期：2026-11-10。结果目录：[results/newton_terrain_render_20261110](results/newton_terrain_render_20261110)。

## 结论

**圆滑边缘、侧壁大面积明暗渐变的主因是 Newton 的普通 USD Mesh 导入路径没有读取已写入的平面法线，随后 Newton GL 对共享顶点重算法线，形成平滑着色。不是楼梯实际被细分或倒角。** 原生 Cuboid 的显示路径则主动拆分硬边顶点并提供正确法线，因此没有同样的问题。

已在加载器中修复：保持原始碰撞 Mesh，另外生成只有显示作用的平面着色 Mesh；其三角形角点独立，GL 即使重算法线也只能得到平面法线。源资产仍是 v1，运行时显示表示标记为 `flat_visual_v2`。随后单独将材质改为中性深灰哑光；未改变 GL 光照。

**物理回归通过“行为未改变”的检查，但不能宣称默认接触管线对所有踏面都正确。** 两种地形最高平台的原有落体测试通过；新增加的低踏面测试暴露了原有 MuJoCo 凸包接触问题。修复前后这些失败也完全一致。详见后面的独立物理结果。

![相同相机与 GL 光照，左：修复前；中：只修法线、保持原材质；右：最终深灰哑光](results/newton_terrain_render_20261110/stairs_before_after.png)

图中为便于观察的局部放大，各次原始完整截图均为 1280×800，另行保留，没有修改原始截图内容。本次交付采用截图，没有将静态截图伪装成物理视频。

## 运行环境与保留范围

- RTX 4060 Laptop GPU，8188 MiB，驱动 580.178.04。
- 原生 Python：`/mnt/sage_data/workspace/Toolbox/IsaacLab_3_0/.venv/bin/python`。
- 实际安装的 distribution 版本：isaaclab 27.2.0、isaaclab-newton 9.1.4、isaaclab-visualizers 2.1.0、Newton 1.6.1、Warp 1.17.0、MuJoCo/MuJoCo Warp 3.12.0、pyglet 2.1.15。这里使用安装包的实际版本号，区别于工作目录中的 IsaacLab_3_0 名称。
- 全部物理测试均为 Newton + MJWarp，重力开启，dt=0.005 s，600 步，无 Root/刚体位置覆盖或运动学播放。
- 未安装额外依赖，未修改 Isaac Lab/Newton 的安装源码，未改动 TCRS 生成器、Mesh NPZ、Height Map 或 Footstep Reference。
- 用户原加载器和 smoke 脚本保存在 [original_code](results/newton_terrain_render_20261110/original_code)。源代码版本和关键安装文件 SHA256 见 [environment.json](results/newton_terrain_render_20261110/environment.json)。

## 原始几何审查

审查了当前两套楼梯资产，也阅读了生成源：`../PMT/TCRS/s54/terrain_assets/straight_stairs.py` 和 `generate_stairs_asset.py`。原始 NPZ 是三角网格顶点与面索引，没有输入的平滑顶点法线。

| 检查项 | 直楼梯 stairs_straight_4_v1 | 金字塔 stairs_pyramid_4_v1 |
|---|---:|---:|
| 原始顶点 / 三角形 | 72 / 68 | 136 / 164 |
| 实际被面引用的顶点 | 56 | 120 |
| 闭合 Box / 开放边框平面组件 | 5 / 4 | 13 / 4 |
| 退化三角形 | 0 | 0 |
| 非有限顶点 / 越界索引 | 0 / 0 | 0 / 0 |
| 面绕序一致 | 是 | 是 |
| 闭合组件有向体积 | 全部为正 | 全部为正 |
| 完全重复的几何三角形 | 0 | 0 |
| 有正面积重叠的共面三角形对 | 28 | 128 |
| Box 之间正体积穿透 | 0 | 0 |
| 共享非平行面法线的顶点数 | 40 | 104 |
| 焊接坐标后边界边 / 非流形边 | 14 / 5 | 16 / 14 |

所有共面重叠对都是反向接合面，主要是台阶底面与基座上面、相邻 Box 的接合侧面。不存在“完全重复三角形”不代表没有局部共面重叠；这里使用 Shapely 在共面投影中计算交叠面积，并用 trimesh 检查组件、边连接、法线和体积。

整体 `is_watertight=False` 由开放边框等结构决定，不能据此判定资产错误。另有 16 个未被面引用的边框顶点，部分位于 z=-0.5；它们不构成实际三角面，原始资产也未因本次修复删除它们。Bounds 比较使用被引用三角面的表面范围。

原始审查与外表面诊断结果见两个 `*_geometry.json`；补充闭合组件有向体积见 [geometry_normals_recheck.json](results/newton_terrain_render_20261110/geometry_normals_recheck.json)。

## USD → Newton → GL 的证据链

### 实际 USD 属性

不是只看 Python 配置：各实验导出了 `scene.usda`，也在 `diagnosis.json` 保存了实际 USD 属性。直楼梯原始 Mesh 的属性为：

```text
points=72, triangles=68
subdivisionScheme="none"
normals count=68, interpolation="uniform"
displayColor count=72, interpolation="vertex", value=(0.4, 0.4, 0.4)
bound shader=UsdPreviewSurface
diffuseColor=(0.28, 0.33, 0.40), roughness=0.95, metallic=0
CollisionAPI=True
```

普通导入方块同样有 12 个 `uniform` 面法线，USD 设置成功。所有源输入本来就是三角形；导入后面索引逐项一致，没有额外改变 triangulation。USD float32 转换对原始顶点造成的最大误差为直楼梯 2.3842e-8 m、金字塔 4.7684e-8 m；修复前后碰撞网格的这些 float32 值完全相同。

Isaac Lab 安装源码的调用链：

1. `isaaclab/terrains/utils.py:59`：`create_prim_from_mesh()` 构造内存 `MeshFileCfg`，碰撞开启。
2. `isaaclab/sim/spawners/from_files/from_files.py:566`：`_spawn_mesh_data()` 复制 points/face indices，初始 subdivision 为 `bilinear`；加载器随后确实改成了 `none`。
3. `displayColor` 在 spawn 时写入。旧加载器在生成 USD **之后**才执行 `mesh.visual.vertex_colors = ...`，不会回写 USD；这解释了之前改顶点颜色没有预期效果。
4. 绑定的 PreviewSurface 材质实际被 Newton 读取，没有证据显示 `displayColor` 覆盖该材质。

另外检查了 `from_files_cfg.py:409` 的 `MeshFileCfg` 和 `from_files.py:182` 的 `spawn_from_mesh()`：本场景传入 `TrimeshObjectCfg`，走 `_spawn_mesh_data()` 的内存分支；不会调用面向磁盘 Mesh 文件的 MeshConverter。配置字段本身不携带 trimesh 法线，法线是加载器随后直接写入 USD 的。

### 导入时法线丢失

安装源码根目录：`/mnt/sage_data/workspace/Toolbox/IsaacLab_3_0/.venv/lib/python3.12/site-packages/newton/`。

- `_src/utils/import_usd.py:796`：`_get_mesh_cached(..., load_normals=False)` 默认不读法线。
- 同文件 `:1075`：普通、无纹理 `_get_mesh_with_visual_material()` 使用上述默认值，再复制 physics mesh；没有请求 `load_normals=True`。
- 带材质子集的其他分支有读取法线的代码，因此这里的结论限定于本场景的普通 Mesh 路径，不能扩大为“Newton 从不支持 USD 法线”。

实际 Newton Model 导出结果：原始方块、直楼梯、金字塔的 `imported_authored_normals=False`。points 和 indices 则仍与输入一致。

### GL 重新平滑法线

- `_src/viewer/viewer.py:1700`：Mesh 只有在 `_normals` 非空时才把法线交给 GL。
- `_src/viewer/gl/opengl.py:303`、`:380`：未给法线时调用 `compute_vertex_normals()`；多个互相垂直的面共享同一顶点，于是法线被平均。
- `_src/viewer/viewer.py:1756`：原生 BOX 使用 `duplicate_vertices=True` 并提供面方向法线。

诊断脚本直接读取 GL 的 `RenderVertex` 缓冲区，逐三角形比较角点法线和几何面法线，保存为 `gl_mesh_*.npz`。不是凭截图推测法线。

## A/B/C/D 实验

所有实验使用相同相机 eye=(6,-7,4)、target=(0,0,0.15)，分辨率 1280×800，同样的 GL 光照。除最终深灰材质实验外，颜色均为原来的 `(0.28,0.33,0.40)`，roughness=0.95，metallic=0。

“偏离角点数”采用角点法线与面法线点积 < 1−1e-5；单位均为显示网格，不将隐藏碰撞网格的法线计入修复效果。

| 实验 | GL 顶点 / 三角形 | 偏离角点 / 总角点 | 最大夹角 | 观察 |
|---|---:|---:|---:|---|
| A 原生 Cuboid | 24 / 12 | 0 / 36 | 0° | 平面均匀、直角清晰 |
| B trimesh → USD 单方块 | 8 / 12 | 36 / 36 | 84.40° | 单方块也出现圆滑明暗，排除 Box 交叠是必要原因 |
| B 仅拆分三角形角点 | 36 / 12 | 0 / 36 | 0° | 保持材质和光照即消除异常 |
| C 原始直楼梯 | 72 / 68 | 180 / 204 | 89.52° | 圆滑边缘、深色侧壁渐变 |
| D 无内部冗余面的直楼梯 | 72 / 132 | 306 / 396 | 89.13° | 仅移除内部面仍会平滑 |
| D 再拆分显示角点 | 396 / 132 | 0 / 396 | 0° | 正常 |
| C 原始金字塔 | 136 / 164 | 468 / 492 | 88.72° | 明显平滑异常 |
| D 金字塔外表面 | 228 / 444 | 1014 / 1332 | 88.21° | 移除内部面仍异常 |
| D 金字塔外表面拆分角点 | 1332 / 444 | 0 / 1332 | 0° | 正常 |
| C 最终直楼梯显示网格 | 204 / 68 | 0 / 204 | 0° | 保留原三角形，平面着色正常 |
| C 最终金字塔显示网格 | 492 / 164 | 0 / 492 | <0.000001° | 保留原三角形，平面着色正常 |

![A/B/C/D 全帧对照](results/newton_terrain_render_20261110/ab_straight.png)

**判断：** 内部接合面客观存在，可能导致接缝处的细小深度竞争，但它们解释不了单方块上的大面积平滑异常。D 对照也说明删除内部面不能代替处理硬边法线。因此生产修复保留了这些面，没有重做碰撞拓扑。C-flat 与 D-flat 的直楼梯截图只有 161 个像素不同（含接缝/光栅化差异，占全帧约 0.016%），不把这些细微差异误报为完全没有接缝或所有视角均无 Z-fighting。

D 仅是诊断外表面：依据原来的轴对齐表面坐标划分面片，去除 Box 内部面，保留开放边框，未替换生产碰撞资产。对全部 52,500 / 90,000 个 Height Map XY 采样点，覆盖相同、最高表面差异 ≤5.56e-17 m；D 顶点和面中心到原表面的最大距离分别为 4.48e-13 / 2.96e-14 m。面数增加来自面片重新划分，不是加厚、倒角或更改踏面高度。

## 光照与颜色

独立对照把 USD `DomeLight.intensity` 从 800 改到 0，PNG 文件 SHA256 完全相同：`d43e36bcd86f286158fa8d6f7fb2f5e89a3c3cd884a08ec463007d7f328703f8`。

当前 GL 使用自己的定向光和半球环境光，而不是该 USD DomeLight。实际记录：light_color=(1,1,1)、ambient_sky=(0.8,0.8,0.85)、ambient_ground=(0.3,0.3,0.35)。`newton_visualizer.py:2313` 从 GL Visualizer 配置设置光色；GL `shaders.py:397` 的直接光项乘 3、环境光项另有系数。因此继续只改 USD DomeLight 无法解决这里的亮度问题。

USD PreviewSurface 线性 RGB `(0.28,0.33,0.40)` 在 Model 中变为显示 RGB `(0.565728,0.609712,0.665185)`。转换来自 `_src/usd/utils.py:2835` 的 `_color_to_display_space()`；GL shader 再做其显示/光照计算。没有将正常的线性/显示颜色转换误判为重复 gamma 错误。

先完成原材质法线对照后，才把最终默认材质改为线性 RGB `(0.08,0.08,0.08)`、roughness=0.95、metallic=0。GL 光照未调节。可用 `spawn_stairs(..., diffuse_color=(0.28,0.33,0.40))` 复现中间的“只修法线”效果。

## 实施内容与几何回归

修改：[scripts/terrain/load_stairs_asset.py](scripts/terrain/load_stairs_asset.py)。

```text
/World/Stairs/mesh         原始 points / indices；CollisionAPI=True；visibility=invisible
/World/Stairs/flat_visual  同一组三角形角点，只改变显示顶点索引；CollisionAPI=False
                          subdivision=none；uniform normals；flat_visual_v2
```

`flat_visual` 每个三角形拥有自己的三个顶点；没有位移、增减三角形、改变法线方向或碰撞材料。保持显示 Mesh 和碰撞 Mesh 为兄弟 prim，避免隐藏父级时把显示 Mesh 一起隐藏。真实 Model 中前者 `visible=True, collision=False`，后者 `visible=False, collision=True`。

离屏场景要显式 `NewtonCfg(load_visual_shapes=True)`。当前版本的自动判断可能在离屏 viewer 初始化前发生，否则只用于显示的几何可能被省略；此时碰撞仍在，但画面为空。诊断脚本已显式设置，原 smoke 脚本也已设置。

验证器逐项检查，而非仅比较 Bounds：

- 两套资产共 10 个源文件的 SHA256 全部与修改前一致。
- 实际 USD 碰撞 points 等于原数组转换成 float32 的结果，indices 与原始完全一致。
- 显示 Mesh 的每一个三角形角点都与碰撞 Mesh 的对应三角形完全相同。
- 两套 Height Map 的全部采样点都验证，覆盖范围相同；显示 float32 表面对原始 float64 的最大高度差 1.1921e-8 m。修复前后碰撞 float32 数据差为 **0**。

数据见 [verification.json](results/newton_terrain_render_20261110/verification.json)。原 NPZ 和 manifest 的 schema、版本、hash 没有变更；仅运行时显示表示升级。

新增独立工具：

- [terrain_render_diagnostics.py](scripts/terrain/terrain_render_diagnostics.py)：CPU 几何审查、精确垂直三角面查询、诊断外表面生成与表面比较。
- [diagnose_newton_terrain_render.py](scripts/terrain/diagnose_newton_terrain_render.py)：真实 USD → Newton → GL A/B，完整截图、USD、Model 属性、GL 法线与真实落体轨迹导出。
- [verify_terrain_render_results.py](scripts/terrain/verify_terrain_render_results.py)：对保存的证据进行可重复断言，明确分开“物理未改变”和“接触高度正确”。

修改 [t12_newton_stairs_smoke.py](scripts/terrain/t12_newton_stairs_smoke.py)：修正原先 `--drop-x` 改变后仍查询 x=0 Height Map 的检查错误，新增 `--result-json`，启用只用于显示的几何导入。未改求解器、接触选项或误差门限。

## 独立真实物理回归与发现的已有问题

每个初始位置分别运行旧加载器与修复加载器：1 kg、10 cm 方块，释放中心高度=精确表面+0.5 m，重力 -9.81 m/s²，600×0.005=3 s。没有手工设置动态状态；每步仅 write_data、physics step、scene update。

| 地形 / drop x | 期望中心 z (m) | 原 / 修复最终中心 z (m) | 高度误差 (m) | 25 mm 门限 | 前后 600 步最大位置差 |
|---|---:|---:|---:|---|---:|
| 直楼梯 -2.0，地面 | 0.050000 | 0.222940 / 0.222940 | 0.172940 | 两者失败 | 0 |
| 直楼梯 -1.3，第 1 级 | 0.150000 | 0.310398 / 0.310398 | 0.160398 | 两者失败 | 0 |
| 直楼梯 -0.9，第 2 级 | 0.250000 | 0.360055 / 0.360055 | 0.110055 | 两者失败 | 0 |
| 直楼梯 -0.5，第 3 级 | 0.350000 | 0.409846 / 0.409846 | 0.059846 | 两者失败 | 0 |
| 直楼梯 0，最高平台 | 0.450000 | 0.446234 / 0.446234 | 0.003766 | 两者通过 | 0 |
| 金字塔 0，最高平台 | 0.450000 | 0.446238 / 0.446238 | 0.003762 | 两者通过 | 0 |

碰撞顶点和索引的 hash 在各对照中一致；全部位置轨迹逐点相同。原有 T1.2 smoke 脚本另行对两种地形各运行 600 步，最高平台均通过，JSON/日志保存在 `smoke_stairs_*`。

**低踏面的失败是已有接触表示问题，不能用最高平台测试掩盖。** 当前 `newton_mjwarp` preset 使用 `NewtonCfg()`，默认 `MJWarpSolverCfg.use_mujoco_contacts=True`。Newton 的 `solver_mujoco.py:6413` 将整块 MESH 导出为一个 MuJoCo mesh；`:6421` 的源码注释明确说明 MuJoCo mesh 经由 convex-hull 路径编译。楼梯凹角因此可能变成凸包斜面接触。

进一步做了独立诊断：保持 **同一 MJWarp 动力学求解器、原加载器、原 Mesh、同一释放位置 x=-1.3**，仅把接触生成切换为 `MJWarpSolverCfg(use_mujoco_contacts=False)`。最终中心 z=0.14994544，误差 **0.00005456 m**，通过。日志、`solver.xml`、位置轨迹保存在 [diagnostic_newton_contacts_low_step](results/newton_terrain_render_20261110/diagnostic_newton_contacts_low_step)。这支持了默认凸包接触路径是低踏面偏差来源的判断，而不是本次显示修复改变了接触。

本次**没有**把该选项设为生产默认，也没有用分解碰撞 Mesh 来修复渲染。后续开展机器人逐级踩踏前，应单独评估 Newton 接触管线（保持 MJWarp 求解器）、多 Box 碰撞或高度场方案的接触正确性、性能与稳定性；本次单点诊断不能等同于完整机器人或全地形接触验收。没有对摩擦系数、机器人足底接触或训练效果作出通过声明。

## 截图与失败记录

- 原始直楼梯：[完整截图](results/newton_terrain_render_20261110/C_straight_original/newton_gl.png)。
- 修复后直楼梯：[完整截图](results/newton_terrain_render_20261110/C_straight_final_matte/newton_gl.png)。
- 原始金字塔：[完整截图](results/newton_terrain_render_20261110/C_pyramid_original/newton_gl.png)。
- 修复后金字塔：[完整截图](results/newton_terrain_render_20261110/C_pyramid_final_matte/newton_gl.png)。
- 只修法线、不改材质：`C_straight_flat_v2_visuals/`、`C_pyramid_flat_v2_visuals/`。
- 每个有效实验保留 `scene.usda`、`diagnosis.json`、`gl_mesh_*.npz`；落体实验另有 `drop_trajectory.npz`，过程日志为同名 `.log`。

初期诊断工具的失败没有删除：没有自由关节的静态 MJWarp 场景、DISPLAY 存在时混用 Xlib/EGL 的 GL 上下文、离屏自动导入省略显示 Mesh 导致的空画面，均见 [attempt_notes.json](results/newton_terrain_render_20261110/attempt_notes.json)。空画面即使早期脚本写了 PASS，也明确排除出验收样本；当前脚本遇到“没有可见目标形状”会报错。低踏面物理失败保留 `FAILED` 状态，没有被验证器改写成接触通过。

当前 GL 的材质与光照仍是其自身实现，不承诺和 RTX/Kit 外观一致。本次已有单方块因果对照与 GL 缓冲证据，不需要安装 RTX 依赖或切换物理求解器才能解决硬边显示。若未来升级 Newton，可先验证普通导入路径是否开始保留 uniform/faceVarying 法线，再考虑移除本地显示网格兼容方案。

## 运行命令

从仓库根目录执行。新实验目录必须不存在，避免覆盖本次证据。

### 查看修复后的 Newton GL 场景并运行原 smoke

```bash
/mnt/sage_data/workspace/Toolbox/IsaacLab_3_0/.venv/bin/python \
  scripts/terrain/t12_newton_stairs_smoke.py \
  --asset-dir assets/terrains/stairs_pyramid_4_v1 \
  --physics newton_mjwarp --viz newton_gl --max-steps 600
```

将资产名换成 `stairs_straight_4_v1` 可看直楼梯。此命令只验证当前 x=0 的最高平台，不代表其他踏面的接触已通过。

### 离屏导出完整截图及 USD / GL 数据

```bash
ISAAC_PY=/mnt/sage_data/workspace/Toolbox/IsaacLab_3_0/.venv/bin/python
OUT="results/terrain_render_check_$(date +%Y%m%d_%H%M%S)"

"$ISAAC_PY" scripts/terrain/diagnose_newton_terrain_render.py \
  --case C --asset-dir assets/terrains/stairs_straight_4_v1 \
  --out "$OUT/straight"

"$ISAAC_PY" scripts/terrain/diagnose_newton_terrain_render.py \
  --case C --asset-dir assets/terrains/stairs_pyramid_4_v1 \
  --out "$OUT/pyramid"
```

输出分别为 `$OUT/straight/newton_gl.png`、`$OUT/pyramid/newton_gl.png`。脚本在导入 Isaac Lab 前仅在自身进程移除 DISPLAY，使 pyglet 一开始就选择 EGL；不改用户终端或 VS Code 的环境。

### 独立 A/B/C/D 复现

```bash
ISAAC_PY=/mnt/sage_data/workspace/Toolbox/IsaacLab_3_0/.venv/bin/python
OUT="results/terrain_ab_repeat_$(date +%Y%m%d_%H%M%S)"
OLD_LOADER=results/newton_terrain_render_20261110/original_code/load_stairs_asset.py

"$ISAAC_PY" scripts/terrain/diagnose_newton_terrain_render.py --case A --out "$OUT/A"
"$ISAAC_PY" scripts/terrain/diagnose_newton_terrain_render.py --case B --out "$OUT/B"
"$ISAAC_PY" scripts/terrain/diagnose_newton_terrain_render.py --case B --unweld --out "$OUT/B_flat"
"$ISAAC_PY" scripts/terrain/diagnose_newton_terrain_render.py --case C --loader "$OLD_LOADER" --out "$OUT/C_old"
"$ISAAC_PY" scripts/terrain/diagnose_newton_terrain_render.py --case D --out "$OUT/D"
"$ISAAC_PY" scripts/terrain/diagnose_newton_terrain_render.py --case D --unweld --out "$OUT/D_flat"
"$ISAAC_PY" scripts/terrain/diagnose_newton_terrain_render.py --case C \
  --diffuse-color 0.28 0.33 0.40 --out "$OUT/C_flat_same_material"
```

上述默认直楼梯。对 C/D 增加 `--asset-dir assets/terrains/stairs_pyramid_4_v1` 即为金字塔。`--unweld` 只用于一次性诊断场景，不能据此去改生产碰撞 NPZ。

### 独立物理与已有接触问题复现

```bash
ISAAC_PY=/mnt/sage_data/workspace/Toolbox/IsaacLab_3_0/.venv/bin/python
OUT="results/terrain_contact_check_$(date +%Y%m%d_%H%M%S)"

# 原接触配置：最高平台应通过，保存全部 600 步位置。
"$ISAAC_PY" scripts/terrain/diagnose_newton_terrain_render.py \
  --case C --steps 600 --drop-x 0 --out "$OUT/platform"

# 低踏面：当前默认 MuJoCo 接触配置会失败，失败资料仍保存。
"$ISAAC_PY" scripts/terrain/diagnose_newton_terrain_render.py \
  --case C --steps 600 --drop-x -1.3 --out "$OUT/low_default"

# 仅作诊断：同一 MJWarp 求解器，改用 Newton 接触生成，不改生产默认。
"$ISAAC_PY" scripts/terrain/diagnose_newton_terrain_render.py \
  --case C --steps 600 --drop-x -1.3 --contact-pipeline newton \
  --out "$OUT/low_newton_contacts"
```

若在 `set -e` 的 shell 脚本中复现预期失败的低踏面测试，应单独执行该命令或显式记录退出码，再运行下一项，不能把非零退出码当作渲染故障。

### 重新检查本次已保存证据

```bash
/mnt/sage_data/workspace/Toolbox/IsaacLab_3_0/.venv/bin/python \
  scripts/terrain/verify_terrain_render_results.py \
  --results results/newton_terrain_render_20261110
```

预期：`render_fix_and_unchanged_physics="PASS"`，同时保留 `default_contact_all_treads_correct=false`。两者是不同结论。
