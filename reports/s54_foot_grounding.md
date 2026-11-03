# S54 足底穿地修复验证（2026-10-27）

本报告记录足底阶段的历史结果。后续已完成非相邻 link 自碰撞修复、连续区间
验收及端到端训练 Smoke Test，新数据目录为 `reference_safe/`；当前结论与命令
见 `reports/s54_self_collision_fix.md`。下文的自碰撞 FAIL 是修复前的诊断结果。

已完成本项目内的生成流程修复。完整 `dance1_subject1` 输出位于
`data/retarget_data/reference_grounded/`，桌面查看器播放真实训练 CSV 的 50 Hz 重采样结果。
未修改 Leju-GMR 源码、IK JSON、机器人资产、G1 数据或 Checkpoint。

## 原因与处理

1. S54 的 `leg_*5_link` 和 `leg_*6_link` 原点重合，旧 GMR 配置却分别将人体
   Toe 与 Foot 的不同位置约束到这两个原点。人体脚部骨骼原点也不等于机器人足底。
2. 本项目在内存中移除冲突 Toe 位置任务，用人体脚/脚尖高度及速度估计接触，
   在接触期间调整脚掌姿态、依据实际几何设置踝部高度，并添加脚与地面的 IK 约束。
3. 使用实际碰撞胶囊、盒子和可视网格计算足底支持高度；在 150 Hz 网格上构造
   穿地约束，将每个源帧区间内的最大所需抬升保守地分配到两个端点，求解平滑
   root Z 抬升。150 Hz 网格包含全部 30 Hz 和 50 Hz 采样点。
4. 上肢修正、脚部 IK、最终平滑抬升分阶段记录；最终 CSV、27 关节 raw CSV、
   PKL、元数据哈希同步更新。最终平滑阶段保持关节、root XY/旋转和时间轴不变。

## 实测结果

| 检查 | 结果 | 实际数据 |
|---|---|---|
| 完整 dance1 30 Hz 导出 | PASS | 3945 帧，最小足底间隙 3.010 mm |
| 实际 MotionLib 50 Hz 加载 | PASS | 6574 帧，最小足底间隙 3.0094 mm |
| 稠密 150 Hz 几何约束 | PASS | 19721 个采样点 |
| 步行 / 跳跃 / 跑步粗测 | PASS | walk1_subject1、jumps1_subject1、run2_subject1，各前 301 帧；每段实际训练加载 501 帧，无足底穿地 |
| 独立几何回归 | PASS | 24 个随机关节及根姿态，包含明确穿地样本；原生距离与完整网格检查误差 < 1e-12 m |
| 上肢检查与关节范围 | PASS | 原有上肢质量、速度、非持续张臂检查全部通过 |
| 可微 FK 与 MuJoCo 对照 | PASS | 128 个样本，30 个机器人 body，最大位置误差 8.85e-7 m |
| CSV / raw CSV / PKL 一致性 | PASS | 数值一致，训练 CSV 哈希有效 |
| 桌面播放及自动关闭 | PASS | GLFW 主线程窗口；鼠标旋转/缩放调用适配本机 MuJoCo 3.17 |
| 视频导出 | PASS | results/s54/dance1_grounded.mp4，640×480，30 FPS，前 10 秒 |
| 手臂与躯干自碰撞（追加诊断） | FAIL / 待修复 | 按碰撞代理几何，左臂 218 帧、右臂 200 帧相交超过 1 mm；不属于本次足底保证范围 |

完整 dance1 最终 root Z 额外抬升：中位数 6.25 mm，P95 26.66 mm，最大
106.63 mm。最大值出现在少量快速姿态，需要保留此信息，不能宣称所有帧都只
调整了几毫米。原来的 dance1 左/右脚碰撞间隙中位数约 -53.5/-48.4 mm。
最终平滑求解与几何检查约 8 秒（不含前面的 GMR 和上肢 IK）。

这里的 PASS 是足部几何与数据契约检查，不代表已验证动力学平衡、消除滑步、
倒地动作的全身接触或所有下肢速度限制。完整 dance1 仍有 373 个下肢关节区间
超过 URDF 速度限制；上肢新增超速为 0。没有启动正式训练。

用户检查时进一步指出手臂穿躯干。已追加原生 MuJoCo 显式几何距离检查，排除
肩部连接处，仅检查上臂、前臂、手相对骨盆/躯干。左臂 194 帧、右臂 183 帧的
碰撞代理相交超过 5 mm；最严重的前臂/躯干相交分别出现在约 31.43 s / 35.70 s。
证据见 `results/s54/grounded_self_collision_diagnosis.json`。这些距离来自简化碰撞
几何，不是可视网格精确交叠深度。现有上肢 IK 没有自碰撞项，训练配置
`supermimic/robot.py` 明确设为 `enabled_self_collisions=False`。因此足底 PASS
不能推广为整段 Reference 的物理可实现性 PASS；此自碰撞问题尚未修复。

## 生成和查看

在项目根目录运行，读取 `data/retarget_data/raw/` 下的原始 BVH：

```bash
# dance1 已生成时直接复用并打开；不存在则先生成再打开
MUJOCO_GL=glfw OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /data/conda_envs/leju_gmr/bin/python tools/generate_s54_reference.py \
  --clips dance1_subject1 --skip_existing --view

# 生成其余 76 段；失败段会隔离，不进入顶层训练 CSV 列表
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /data/conda_envs/leju_gmr/bin/python tools/generate_s54_reference.py \
  --all --exclude dance1_subject1 --skip_existing --continue_on_error

# 或指定 clips
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /data/conda_envs/leju_gmr/bin/python tools/generate_s54_reference.py \
  --clips walk1_subject1 jumps1_subject1 --skip_existing
```

新结果默认写入 `data/retarget_data/reference_grounded/`；训练时显式指定
`--motion_dir data/retarget_data/reference_grounded`。同名已生成数据不会被覆盖。
需要从头重算时使用新的 `--out_dir data/retarget_data/reference_grounded_v2`。
之前已经启动的旧批处理没有被终止，它不会自动应用本次新增的足底处理阶段。

独立复核：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /data/conda_envs/leju_gmr/bin/python \
  tools/test_s54_grounding.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /data/conda_envs/leju_gmr/bin/python \
  tools/validate_s54_grounding.py data/retarget_data/reference_grounded/dance1_subject1.csv
```

## 修改文件

- `configs/s54_grounding.json`：足底间隙、接触估计及平滑参数。
- `tools/s54_grounding.py`：实际足底几何、接触目标、IK 约束、平滑导出修正。
- `tools/generate_s54_reference.py`：接入足底处理、独立检查、新目录及 `--view`。
- `tools/validate_s54_grounding.py`：原生碰撞距离、完整网格、真实 MotionLib 与数据一致性。
- `tools/test_s54_grounding.py`：随机关节姿态、穿地检测、平移与重采样回归。
- `tools/view_s54_reference.py`：50 Hz Reference 桌面播放，空格暂停、鼠标旋转/缩放。
- `tools/validate_s54_upper_body.py`：允许已记录的 root Z 修正，继续独立验证上肢。
- `tools/compare_s54_reference.py`：对比时识别已记录的 root Z 修正。
- `data/retarget_data/README.md`：更新命令与目录说明。
- `README_S54.md`：将新生成及检查示例指向修正版目录。
- 本报告。

主要证据：`reference_grounded/diagnostics/dance1_subject1/` 下的
`grounding_metrics.json`、`grounding_validation.json`、`upper_body_metrics.json`，
以及 `results/s54/grounded_upper_body_validation.json`。
