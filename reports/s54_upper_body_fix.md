# S54 上肢 Reference 修正结果

> 数据清理说明（2026-10-27）：本文为历史实验记录，对应旧 Reference 和中间版本已删除。当前仅保留 `data/retarget_data/raw/` 的原始 BVH；重新生成请使用 `README_S54.md` 的命令，默认输出到 `data/retarget_data/reference/`。

2026-10-27。目标是消除与人体动作不符的持续双臂外展，保留较小误差；不追求每帧精确复刻。

## 结果

已生成完整修正版：`retarget_data/lafan1_s54_upper_fixed/dance1_subject1.csv`，3945 帧、30 FPS、36 列。
配套交互播放文件：`retarget_data/lafan1_s54_upper_fixed/reference/dance1_subject1_30fps.pkl`。
三栏视频：`results/s54/upper_body_comparison.mp4`，人体 BVH / 原 S54 / 修正 S54，15 秒、30 FPS、1440×600。

| 实际检查 | 状态 | 结果 |
|---|---|---|
| dance1 完整生成 | PASS | 3945×36，有限数值，头部按模型默认补齐 |
| 消除初期持续外展 | PASS | 前 10 秒左/右外展比例从 94.35%/91.03% → 0%/29.90%，与人体 0%/29.90% 一致 |
| 初期肩关节贴限位 | PASS | 前 10 秒第三肩关节从两侧 97.67% → 两侧 0% |
| 保留 root / 腿 / 腰 / 头 | PASS | 与原数据数值完全相同；133 个采样帧足底距离变化 0 m |
| 手臂角度和速度范围 | PASS | 使用原 MJCF 角度界限，14 个手臂关节 URDF 速度超限数 0 |
| 实际 MotionLib 加载 | PASS | 30→50 Hz，6574 个控制帧，PKL/CSV 一致 |
| 独立 MuJoCo ↔ PyTorch FK | PASS | 128 个混合合法/参考姿态、30 bodies；最大位置误差 8.70e-7 m，旋转误差 2.49e-7 rad |
| 新轨迹优化 Jacobian | PASS | 24 个随机姿态对中心差分，最大误差 1.50e-9 |
| 其他动作随机粗测 | PASS | seed=54，3 个 clips，各前 300 帧；结果如下 |
| 新 Reference 的 Newton 动力学 / RL 训练 | NOT RUN | 本次只修正 Reference；没有启动训练 |

完整舞蹈对人体的方向/位置误差：

| 指标（95 分位） | 原左臂 | 修正左臂 | 原右臂 | 修正右臂 |
|---|---:|---:|---:|---:|
| 上臂方向误差 | 96.08° | 6.36° | 94.64° | 4.39° |
| 前臂方向误差 | 94.28° | 5.21° | 89.76° | 3.51° |
| 腕中心位置误差 | 70.15 cm | 2.70 cm | 69.30 cm | 2.21 cm |

位置指标以机器人肩部为锚点，采用 S54 真实骨段长度构造人体方向目标；方向比较在同一 Root 坐标系。
它不是原始人体手部世界坐标与机器人的绝对距离，也不意味着两种骨架尺寸相同。
腕中心使用 `zarm_6_link`，手掌朝向使用 `zarm_7_link`。
全片第三肩关节贴限位比例从 95.18%/96.05% 降到 4.39%/3.12%。

仍有少量快动作误差：上臂误差超过 20° 的帧占左 0.76%、右 0.35%；
最大误差左 47.84°（50.40 s）、右 27.26°（44.57 s）。没有继续为这些少量帧精调。

## 修正方法和范围

旧 GMR 同时约束 `zarm_1_link`、上臂、前臂与手的朝向。`zarm_1_link` 只包含第一级肩轴，
其朝向不能独立匹配人体锁骨的三维转动；仅提高肩部位置权重也不能控制上臂方向。
原旋转 offset 大致对齐了骨轴，不能把全部问题简单归为一个 90° offset 写错。

新增独立上肢步骤：从原始 BVH 的 Arm→ForeArm、ForeArm→Hand 提取方向，用 S54 的真实
肩/肘/腕几何进行 7 DoF IK；保留弱手掌朝向约束。S54 肘部有非零骨段 X 偏移，几何伸直角
约 −8.916°，因此选择人体屈肘分支，避免求解器穿过伸直位置后卡在另一分支。
全片离线速度投影及局部窗口约束优化处理速度与分支连续性。没有删帧或改变时间轴。

修复位于本项目的工具中；没有修改 Leju-GMR 的源码、原 IK JSON、原模型、原 PKL/CSV，
没有覆盖旧训练数据或 checkpoint。直接运行旧 `bvh_to_pkl.py` 仍会使用旧上肢任务；请使用下面的新入口。

## 随机粗测

从本地 LAFAN1（排除已使用的 dance1_subject1 / walk1_subject1）以 seed=54 抽取以下 3 个 clips，
各从 BVH 头部实际生成 300 帧基线，再执行同样的上肢修正。它们是前约 10 秒粗测，不是全片验收。

| Clip | 帧数 | 左上臂误差 P95 | 右上臂误差 P95 | 结果 |
|---|---:|---:|---:|---|
| fallAndGetUp3_subject1 | 300 | 0.143° | 0.870° | PASS |
| pushAndStumble1_subject3 | 300 | 0.793° | 0.001° | PASS |
| walk3_subject3 | 300 | 0.670° | 0.001° | PASS |

三段均没有与人体不符的持续外展，手臂速度和角度检查通过；三段合并目录通过真实 MotionLib 加载。
详细结果为 `reports/s54_upper_body_samples.json`，粗测数据在 `.cache/s54_random_clip_probe/`。
另外验证 `walk1_subject1` 前 64 帧：PASS，左/右上臂 P95 为 0.004°/12.55°。
该片段人体本身两臂展开，所以不将其判为“人工持续外展”。

当前通用批处理采用用户要求的粗测标准：上臂/前臂 P95 <20°，按臂长适配的腕中心 P95 <10 cm，
外展比例相对人体增加 <20 个百分点，第三肩关节贴边 <30%；角度、速度和有限数值检查仍为硬条件。
完整 dance1 实际结果也通过更严格的 10°/5 cm 检查。

## 可运行命令

以下均在 `/mnt/sage_data/workspace/supermimic_s54` 执行，复用现有 Leju-GMR Python，CPU，无 GUI、无训练。

重新从 BVH 生成 dance1（包括基线 PKL 与上肢修正）：

```bash
OPENBLAS_NUM_THREADS=1 /data/conda_envs/leju_gmr/bin/python tools/generate_s54_reference.py \
  --clips dance1_subject1 --out_dir retarget_data/lafan1_s54_dance_regenerated
```

生成指定其他 clips：

```bash
OPENBLAS_NUM_THREADS=1 /data/conda_envs/leju_gmr/bin/python tools/generate_s54_reference.py \
  --clips walk1_subject1 walk2_subject1 --out_dir retarget_data/lafan1_s54_other
```

生成本地所有其他 clips（本地 dry-run 确认选中 76 段；本次没有执行全量生成）：

```bash
OPENBLAS_NUM_THREADS=1 /data/conda_envs/leju_gmr/bin/python tools/generate_s54_reference.py \
  --all --exclude dance1_subject1 --out_dir retarget_data/lafan1_s54_other \
  --skip_existing --continue_on_error
```

`--clips 'walk*'` 支持按名称筛选；`--dry_run` 只列计划。`--max_frames 300` 仅用于前缀粗测，
生成完整训练数据时省略。已存在同名输出时默认拒绝覆盖；`--skip_existing` 留下原输出。
失败候选移动到 `rejected/<clip>/`，不混入训练 CSV；每次任务的 manifest 在 `diagnostics/generation_*.json`。
CSV 根目录只放完整 36 列动作，原始 34 列在 `raw/`，逐动作诊断在 `diagnostics/<clip>/`，
修正版 PKL 和人体缓存分别为 `reference/<clip>_30fps.pkl`、`reference/<clip>_human.npz`。

已有 PKL 时可使用 `--reuse_baseline_dir /mnt/sage_data/workspace/Leju-GMR/output/kuavo_s54/pkl`；
缺少的片段自动调用 GMR API 生成。基线使用源脚本相同的膝关节初始弯曲比例 0.1、30 FPS、米制和 `offset_to_ground=False`。

查看当前完整修正版：

```bash
cd /mnt/sage_data/workspace/Leju-GMR
/data/conda_envs/leju_gmr/bin/python scripts/vis_robot_motion.py --robot kuavo_s54 \
  --robot_motion_path /mnt/sage_data/workspace/supermimic_s54/retarget_data/lafan1_s54_upper_fixed/reference/dance1_subject1_30fps.pkl
```

独立验证与重渲染：

```bash
cd /mnt/sage_data/workspace/supermimic_s54
OPENBLAS_NUM_THREADS=1 /data/conda_envs/leju_gmr/bin/python tools/validate_s54_upper_body.py
MUJOCO_GL=osmesa OPENBLAS_NUM_THREADS=1 /data/conda_envs/leju_gmr/bin/python \
  tools/compare_s54_reference.py --seconds 15 --out results/s54/upper_body_comparison.mp4
```

## 修改文件与限制

- 新增 `configs/s54_upper_body_retarget.json`、`tools/retarget_s54_upper_body.py`、`tools/s54_arm_trajectory.py`。
- 新增 `tools/generate_s54_reference.py`（无窗口单段/批量入口）、`tools/validate_s54_upper_body.py`、`tools/compare_s54_reference.py`。
- `tools/prepare_s54_motion.py` 增加独立 raw 目录参数；`tools/preview_s54_reference.py` 增加 motion_dir/clip 参数。
- `README_S54.md` 增加修正版生成、验证与播放说明。

完整舞蹈保留原下肢运动，因此原有穿地仍存在，另有 483 个下肢关节采样区间超过 URDF 速度；
本次没有把这些问题计为“已修复”。未执行新 Newton smoke、Evaluation 或训练；不能据此声称新策略性能提高。
本次可确认的是：Reference 的持续错误外展已消除，完整舞蹈和随机前缀粗测具备可重复的生成与验证入口。
