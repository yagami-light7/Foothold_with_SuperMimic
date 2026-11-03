# S54 Reference 自碰撞修复与验收（2026-10-27）

已实现非相邻 link 避碰、时序优化、连续区间验收和训练加载门禁。
完整 dance1 已发布到 `data/retarget_data/reference_safe/`，并完成现有 Docker
容器内的 30-iteration Isaac Lab 3.0 / Newton / MJWarp / PyTorch WM+BPTT Smoke Test。
未修改原始 Leju-GMR、原始 supermimic_vibed、机器人资产、G1 数据或 Checkpoint。

## 实现

- 同时检查可视网格凸包与物理碰撞体，共 2551 个几何对。碰撞判断独立于仿真
  接触掩码，不会因为训练关闭自碰撞而漏检 Reference。
- 依据模型拓扑，将没有碰撞体的中间关节归入最近的碰撞-bearing link，仅排除
  同一机械组件及其直接连接组件。肩、肘、腕等关节装配处的设计重叠不属于
  非相邻 link 自碰撞；排除列表写入每段证书，没有针对失败动作添加特例。
- 先逐帧修正，再联合约束姿态、人体上肢方向、关节限位和所有关节的 URDF
  速度。仅接近身体的姿态需要改变，保持 FPS、时长和 root XY/旋转。
- 150 Hz 优化网格之外，用 link 运动的保守距离变化界自适应细分整个线性
  关节插值区间。验收要求所有非相邻几何对始终具有至少 1 mm 间隙。
  如果发现网格遗漏的时刻，将其精确位置加入优化约束，再重新证明。
- 修正后重新计算足底平滑抬升，独立检查实际 MotionLib 的 50 Hz 数据，
  再核对 CSV/raw CSV/PKL、模型和文件哈希、人体动作误差及有限值。
- 每段先写入 `.staging/`。只有所有检查通过，才发布到顶层训练 CSV 目录；
  失败候选放到 `rejected/`。训练加载器默认拒绝缺证书、失败证书或证书与
  CSV 哈希不匹配的数据，`--skip_existing` 也只复用合格结果。

训练环境的自身碰撞开关与原有 MDP、World Model、Policy、BPTT 算法保持不变。
本次保证的是固定 S54 几何模型下导出 Reference 的非相邻 link 分离，不能把
这项验收解释成训练策略每步都能避碰、已经动力学平衡或已经消除滑步。

## 实际结果

| 项目 | 结果 | 证据 |
|---|---|---|
| 完整 dance1 | PASS | 3945 个 30 Hz 源帧，6574 个实际 50 Hz 训练帧 |
| 连续区间自碰撞 | PASS | 2551 个几何对，71049 个自适应验收姿态，整个插值区间证明间隙至少 1 mm |
| 实际 50 Hz link 间隙 | PASS | 最小 4.852 mm |
| 实际 50 Hz 足底间隙 | PASS | 最小 3.010 mm |
| 关节范围、速度与有限值 | PASS | 包含下肢限速修正；头部保持模型默认值 |
| 全片上肢动作误差 | PASS | 上臂 P95：左 7.38° / 右 6.94°；前臂：5.25° / 5.38°；手腕位置：4.14 / 4.04 cm |
| 前 10 秒张臂检查 | PASS | 相比人体没有新增持续张臂；骨骼方向 P95 < 0.003° |
| 步行、跑步、跳跃粗测 | PASS | walk1_subject1、run2_subject1、jumps1_subject1，各 301 个源帧 / 501 个训练帧 |
| 打斗粗测 | PASS | fight1_subject2，601 个源帧 / 1001 个训练帧 |
| 瞄准粗测 | FAIL / 已隔离 | aiming2_subject3，601 帧；原有上肢阶段的前 10 秒方向误差检查未过，没有进入训练目录 |
| 距离 Jacobian、运动界回归 | PASS | 原生距离有限差分最大误差约 9.15e-10；包含跨帧穿越测试 |
| 隐藏子帧回归 | PASS | 真实案例的 150 Hz 网格间隙 8.45 mm，但另一时刻仅 0.129 mm；连续检查能发现并将精确时刻加入优化 |
| 失败候选隔离 | PASS | 故意关闭优化，实际碰撞候选被拒收，顶层可训练 CSV 数量为 0，输入文件保留 |
| CPU FK / MDP / WM / Policy / Critic / BPTT | PASS | 原生 FK 对照；全部损失与梯度有限且有效 |
| Docker / CUDA 30-iteration Smoke | PASS | RTX 4060 Laptop GPU；Newton/MJWarp；第 30 次完整 S54 Checkpoint 验证通过 |
| 修复前后渲染、桌面查看 | PASS | 31.43 秒处的人体 / 旧 Reference / 修正版三面板视频，960×480，20 FPS |

修复前，按本次完整几何规则，dance1 有 830 个源帧不满足 1 mm 分离间隙，
最深几何相交约 181.7 mm。修复后通过连续区间验收。并非所有关节都只改了
几度：最大关节变化 1.259 rad，下肢最大变化 0.572 rad。足底最终额外抬升
最大约 141 mm，大多数帧无需明显抬升。罕见快速动作会偏离原始姿态以满足
几何和速度限制，因此同时保留动作误差检查，未放宽碰撞验收。

全量 77 段尚未全部重新生成；以后每段都运行相同门禁。不能解算出合格结果的
clip 会明确报错并隔离，不会用未通过检查的 Reference 凑齐数据集。

## 生成、查看和训练

从项目根目录运行：

```bash
# dance1 已生成，复用已验收结果并打开；空格暂停，拖动旋转，滚轮缩放
MUJOCO_GL=glfw OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /data/conda_envs/leju_gmr/bin/python tools/generate_s54_reference.py \
  --clips dance1_subject1 --skip_existing --view

# 从原始 BVH 生成其余 76 段；失败项隔离，最终返回非零退出码
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /data/conda_envs/leju_gmr/bin/python tools/generate_s54_reference.py \
  --all --exclude dance1_subject1 --skip_existing --continue_on_error

# 或只生成指定 clips；不加 --max_frames 才是完整导出
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /data/conda_envs/leju_gmr/bin/python tools/generate_s54_reference.py \
  --clips walk1_subject1 jumps1_subject1 --skip_existing

# 已有旧数据可直接修复到独立目录，复用 GMR/上肢结果
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /data/conda_envs/leju_gmr/bin/python tools/repair_s54_collisions.py \
  --input_dir data/retarget_data/reference_grounded --all --skip_existing --continue_on_error

# 后续训练；此命令未执行正式训练。先确认需要的 clips 都已生成并验收。
MOTION_DIR=data/retarget_data/reference_safe ITERATIONS=10000 \
  bash tools/train_s54_4060.sh train s54_safe_4060
```

新数据目录：`data/retarget_data/reference_safe/`。同名结果拒绝覆盖；从头重算
请用新的 `--out_dir data/retarget_data/reference_safe_v2`。
直接运行外部 Leju-GMR 脚本不包含本项目的后处理与验收，训练不会接受缺证书的导出。

## 复核与产物

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /data/conda_envs/leju_gmr/bin/python \
  tools/test_s54_self_collision.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /data/conda_envs/leju_gmr/bin/python \
  tools/test_s54_reference_gate.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /data/conda_envs/leju_gmr/bin/python \
  tools/validate_s54_self_collision.py data/retarget_data/reference_safe/dance1_subject1.csv
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /data/conda_envs/leju_gmr/bin/python \
  tools/test_s54_cpu.py --out results/s54/collision_safe_cpu_tests.json
```

- 汇总：`results/s54/self_collision_acceptance_summary.json`。
- 原生验收：`reference_safe/diagnostics/dance1_subject1/self_collision_validation.json`。
- 动作质量与修正记录：同目录的 `self_collision_metrics.json`、`self_collision_adjustment.npz`。
- CPU：`results/s54/collision_safe_cpu_tests.json`。
- GPU Smoke：`logs/s54_collision_smoke_20261027_v1/`，`last.pt` 与 `ckpt_000030.pt`；
  `results/s54/collision_gpu_smoke.log`。
- 视频：`results/s54/dance1_self_collision_comparison.mp4`，开始时间 31.43 s，三秒对比。
- 批量粗测与拒收证据：`data/retarget_data/probes/safe/diagnostics/` 与 `rejected/`。

## 修改文件

- `configs/s54_self_collision.json`：碰撞间隙、优化和动作质量参数。
- `tools/s54_self_collision.py`：几何对规则、距离 Jacobian、碰撞修正、速度约束、
  稀疏时序 QP、自适应约束与连续区间证明。
- `tools/s54_reference_publish.py`：候选验收、隔离边界、证书校验与原子发布。
- `tools/repair_s54_collisions.py`：修复已有数据到独立目录，支持断点继续。
- `tools/validate_s54_self_collision.py`：序列化数据、实际 50 Hz 数据和穷举原生距离复核。
- `tools/test_s54_self_collision.py`、`tools/test_s54_reference_gate.py`、
  `tools/fixtures/s54_subframe_collision.json`：真实碰撞、梯度、隐藏子帧及隔离回归。
- `tools/generate_s54_reference.py`：默认完整修复流程、私有候选、失败隔离和新目录。
- `supermimic/motion.py`：训练默认强制校验证书及其 CSV 哈希绑定。
- `supermimic/train.py`、`supermimic/evaluate.py`、`tools/train_s54_4060.sh`：默认使用合格目录。
- `tools/s54_grounding.py`、`tools/validate_s54_grounding.py`：同步修正记录与足底复核。
- `tools/validate_s54_upper_body.py`、`tools/compare_s54_reference.py`：识别碰撞修正后的轨迹。
- `tools/view_s54_reference.py`、`tools/preview_s54_reference.py`：默认查看合格 Reference。
- `tools/test_s54_cpu.py`：支持合格多 clip 数据和输出路径，修正测试的片尾索引处理。
- `README_S54.md`、`data/retarget_data/README.md`、`reports/s54_foot_grounding.md`、本报告。
