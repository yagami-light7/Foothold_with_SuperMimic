# S54 Foot Trajectory Goal

本文保留 Goal 接口阶段的历史交付记录。后续独立 Foot MDP 与 Standing 最小训练已实现，
当前状态见 [S54_FOOT_TRACKING_TRAINING.md](S54_FOOT_TRACKING_TRAINING.md)。
下文的 Motion 依赖与 414 维 Critic 描述属于历史接口预览；当前 Foot 训练不再加载 Motion，Critic 输入为 365 维。

本次交付为连续足部命令和 SuperMimic 条件接口适配。原始 Motion 模式仍是默认值。
足部模式的 Policy 不再读取人体 Motion Goal 或 Motion 根朝向；重置、Reward、终止、
Critic 的特权误差特征暂时继续使用 Motion。World Model 的输入、预测目标和网络、
BPTT/GAE/优化器算法均未调整。没有启动训练，也没有生成训练 checkpoint。

## 代码和数据流

| 文件 | 修改 |
| --- | --- |
| `supermimic/foot_trajectory.py` | 显式步序列、纯 PyTorch 随机采样与无状态轨迹查询 |
| `supermimic/goals.py` | 两种 Goal 来源、维度/版本契约、局部编码、固定 heading 观测 |
| `supermimic/foot_geometry.py` | 从原 grounding 工具提取并复用足部几何计算 |
| `supermimic/s54.py` | 统一左右足 body 名称 |
| `supermimic/mdp.py` | 观测允许显式传入参考朝向；默认仍使用 Motion |
| `supermimic/env.py` | episode 命令、独立执行时钟、reset 前快照、管理器 API 兼容 |
| `supermimic/buffer.py` | 可选命令 ID、去重快照、引用回收、执行步数采样 |
| `supermimic/train.py` | 采集和 BPTT 共用 Goal 来源，按模式构建网络和 checkpoint 契约 |
| `supermimic/evaluate.py` | 从配置恢复 Goal 模式、固定采样种子、保存命令与时钟 |
| `supermimic/networks.py` | 更新 Goal Encoder 说明；网络结构不变 |
| `tools/train/run_train.sh` | Goal 模式、时间间隔、采样配置 CLI 透传 |
| `configs/s54/foot_tracking/s54_foot_trajectory.json` | 平地交替步态的默认测试采样范围 |
| `tools/tests/test_foot_trajectory.py` | 数值、Replay、网络及无优化器更新的 BPTT 验证 |
| `tools/visualization/view_s54_foot_trajectory.py` | Newton 批量 smoke 和 GL 参考轨迹/落脚点录制 |
| `tools/data/s54_grounding.py` | 重导出复用的 `FootGeometry`，保留原导入接口 |
| `tools/tests/test_s54_launcher.py` | 足部模式启动参数的 dry-run 回归 |
| `tools/validation/test_env_s54.py` | 通过环境兼容入口访问 Newton 管理器 |

```text
reset 的 S54 FK → 平地接触位姿 → 每个 episode 采样一次 FootstepCommand
                                        ↓
S + ep_step × 0.02s → query(command, future_times) → 当前根 yaw 局部编码 → Policy
                                        ↓
step 返回 reset 前命令快照 → Replay 按 ID 去重 → imagination 恢复同一命令
```

### 足步命令及查询

`FootstepCommand` 的第一维为批量 `B`，脚侧顺序固定为左、右，`M` 为填充后的步数。

| 字段 | Shape / 语义 |
| --- | --- |
| `initial_pos`, `initial_quat` | `[B,2,3]`、`[B,2,4]`；双脚初始踝部位姿 |
| `foot` | `[B,M] int64`；0 左，1 右 |
| `target_pos`, `target_quat` | `[B,M,3]`、`[B,M,4]`；落脚踝部位姿 |
| `liftoff`, `touchdown`, `height` | `[B,M]`；命令起点后的秒数、米 |
| `lengths` | `[B] int64`；有效事件前缀长度，允许零步静止命令 |
| `heading` | `[B,4]`；命令初始化时固定的 upright 根朝向 |

所有内部四元数为 **wxyz**，单位为 m、s、rad。位姿位于各环境固定惯性坐标系，
即仿真世界坐标减去固定 `scene.env_origins`；显示时只加回一次 origin。

调用 `command.validate()` 后使用 `query(command, times[B,Q])`，返回位置、四元数、
线速度、加速度、接触 bool 和 Swing 相位。位姿输出形状分别是 `[B,Q,2,3/4]`，
接触和相位是 `[B,Q,2]`。查询不推进状态，不访问机器人，不使用随机数。

水平位置采用 `10u³−15u⁴+6u⁵`，高度经较高端点上方 `height` 的顶点分两段五次插补。
朝向采用最短路径 SLERP 和相同的五次时间缩放。起落点线速度、加速度及角速度为零，
高度顶点保持 C2 连续。接触区间采用 `[liftoff,touchdown)` 为 Swing，落地时立即进入 Stance；
Double Support 由双脚接触同时为真表示。相位在 Stance 时为零。
时间早于起点或超过序列终点时，分别保持初始或最终双脚目标。

显式事件可以不是左右交替，只要求按离地时间排列且同一脚的 Swing 不重叠。
当前随机采样器只生成平地交替步态；默认 32 步，左脚先摆动，初始双支撑 0.1s。
每次落脚相对于初始 heading 的累积前向进度增加 0.08–0.20m，横向位置来自初始双脚间距；
摆动 0.35–0.55s，步间双支撑 0.08–0.15s，高度 0.04–0.08m，落脚 yaw 扰动 ±0.05rad。
这些是接口测试默认值，没有声明动力学可行性或完成步态调参。

S54 的 `leg_l6_link` / `leg_r6_link` 原点是踝部。采样初始化保留 reset 后 FK 的足部 XY，
将足部朝向设为根 yaw，并依据原有 `FootGeometry.local_low` 的 mesh/collision 几何
设置踝高，使最低几何点距离平地 3mm。不会假设踝部 z=0，也不移动机器人实际状态。
外部显式命令的输入位姿按原值查询，不自动投影到地面。

### Goal 编码契约

每个未来时刻按如下顺序编码：

| 范围 | 字段 |
| --- | --- |
| `0:3`, `3:9`, `9`, `10` | 左脚局部位置、6D 朝向、接触 0/1、Swing 相位 |
| `11:14`, `14:20`, `20`, `21` | 右脚对应字段 |
| `22` | 相对于当前时刻的未来时间偏移，单位秒 |

足部模式 `[B,K,23] → [B,23K]`，默认 `K=10, dt=0.1s`，即未来 0.1…1.0s，**230 维**。
位置为 `Hᵀ(p_foot_world−p_root_world)`，朝向为 `HᵀR_foot_world`，`H=yaw(R_root)`。
6D 朝向沿用现有 `mat_to_6d`：取旋转矩阵前两列，再按行展平。
目标世界位姿从不随根部重新锚定；imagined rollout 的局部转换保留对预测根状态的梯度。

观测最后 6 维改为机器人根姿态相对固定命令 heading 的旋转，观测总维度仍为 **99**。
Policy 输出仍为 **29**；Critic 总输入为 **414**，其中原有 184 维状态/特权特征加 230 维 Goal。
Motion 模式仍是每帧 73 维，默认 730 维 Goal 和 914 维 Critic 输入，未来间隔保持 0.02s。

Checkpoint 新增 `goal_contract`，包含模式、K、间隔、维度、坐标与观测语义、版本。
恢复前验证该契约，足部模式还比较采样参数；不进行 Motion/Foot 权重隐式迁移。
缺少契约的旧 checkpoint 仅按历史 Motion 模式处理。评估自动读取运行配置，并恢复种子，
记录 NPZ 另外包含 `command_*` 参数及 `command_step`，足以离线重建目标。

### Replay 与时间

足部模式每条 transition 新增 `command_id`，沿用原有 `ep_step` 作为命令执行步数。
命令参数按 ID 存为 detached 快照，多个时间行共享一份；环形覆盖后引用归零即释放。
`step()` 在自动 reset 前捕获完整命令，避免 reset 覆盖最后一条旧 episode transition。
imagination 在进入编译函数前解析命令，之后只推进整数执行步数并调用同一查询函数。
历史观测限制在同一 episode 内，使用该 episode 的固定命令 heading。
Motion 帧 `f` 只继续服务保留的 Reward、终止、重置和特权特征。

Replay 与此前一样不写入 checkpoint；resume 后重新填充。当前 Python 引用表和快照复制
优先保证正确性，大规模环境的采集吞吐和命令表峰值显存仍需后续测量。

## 可复现命令

已将原 worktree 的 S54 description、USD 及 dance/walk 合格 CSV/JSON 复制到当前忽略目录 `data/`，
没有改写原 worktree、G1 数据或既有容器。以下路径为本机现有依赖环境，不需要重建镜像。

```bash
cd /mnt/sage_data/workspace/supermimic_s54_main
S54_PYTHON=/mnt/sage_data/workspace/IsaacLab_3_0/.venv/bin/python

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 "$S54_PYTHON" tools/tests/test_foot_trajectory.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/tests/test_s54_launcher.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 "$S54_PYTHON" tools/tests/test_s54_cpu.py \
  --samples 32 --skip_core --motion_dir data/retarget_data/reference_startup_fixed \
  --out results/s54/foot_motion_regression.json

# 原 Motion 环境回归，同样不启动训练。
PYTHONDONTWRITEBYTECODE=1 WARP_CACHE_PATH="$PWD/.cache/warp" "$S54_PYTHON" \
  tools/validation/test_env_s54.py --num_envs 1 --steps 60 --viz none \
  --motion_dir data/retarget_data/reference_startup_fixed \
  --out results/s54/foot_motion_newton_regression.json

# 在宿主机可访问 GPU 的终端运行；不启动训练。
PYTHONDONTWRITEBYTECODE=1 WARP_CACHE_PATH="$PWD/.cache/warp" "$S54_PYTHON" \
  tools/visualization/view_s54_foot_trajectory.py --headless --viz none --num_envs 4 --steps 100 \
  --out results/s54/foot_trajectory_newton_4env.json

# Newton GL 离屏录制。去掉 --headless 可打开桌面预览。
PYTHONDONTWRITEBYTECODE=1 WARP_CACHE_PATH="$PWD/.cache/warp" "$S54_PYTHON" \
  tools/visualization/view_s54_foot_trajectory.py --headless --viz newton_gl --steps 100 \
  --video results/s54/foot_trajectory_newton.mp4 \
  --out results/s54/foot_trajectory_newton_visual.json

# 只检查未来训练参数，不创建训练运行。
bash tools/train/run_train.sh --dry-run train s54_foot_interface --clips dance1_subject1 \
  --goal-mode foot_trajectory --goal-dt 0.1 \
  --foot-config configs/s54/foot_tracking/s54_foot_trajectory.json --no-swanlab
```

可视化：蓝色为左脚、橙色为右脚；小点显示未来两秒路径，方框显示落脚踝部位姿，
大球是当前参考点，白色小球显示足部前向轴。脚本复用 Motion PD 目标推动真实仿真，
另将实际状态与足部 Goal 输入新建 Policy 检查前向结果；它不是足部跟踪策略评估。
每次 reset 后都会检查旧命令经 Replay 重建的当前及下一执行步 Goal。

## 验证记录

2026-11-03，CPU：torch 2.11.0+cu130 / MuJoCo 3.12.0；GPU：RTX 4060 Laptop。

| 检查 | 状态 / 结果 |
| --- | --- |
| 新增轨迹/Goal/Replay/S54 网络与无优化器更新的 BPTT | PASS，13 项；230/99/29/414 维 |
| 启动器、动作选择及日志回归 | PASS，20 项，包含新增足部参数 dry-run |
| 原 Motion 数据、32 个姿态的独立 MuJoCo FK、地面接触 | PASS；最大位置误差 7.28e-7m，姿态误差 2.63e-7rad，双脚均接触地面 |
| Newton 单环境 100 步、Policy 前向与 reset/Replay | PASS；最大 Goal 重建差 3.66e-7，触发 3 次 reset |
| Newton 4 环境 100 步 | PASS；最大 Goal 重建差 4.33e-7，触发 4 次 reset |
| Newton GL 双脚轨迹与落脚点录制 | PASS；960×720、100 帧、2 秒；320 个路径点、6 个落脚位姿、双脚参考点及朝向提示 |
| 原 Motion 模式 Newton 60 步 | PASS；0 failed，双脚均有真实地面接触，最大步进 FK 位置误差 1.69e-7m |

机器实测的 Isaac Lab 3 将 `SimulationContext.backend_manager` 改名为 `physics_manager`。
环境仅增加两种名称的兼容访问入口；摩擦、COM 随机化与物理参数计算不变。
离屏 GL 入口在导入 Isaac Lab/Newton 之前选择 Pyglet headless 和 EGL，避免桌面 DISPLAY
存在时混用 Xlib/Headless Canvas。验证期间没有重建镜像，也没有启动既有的 S54/G1 容器。

结果写入 `results/s54/foot_trajectory_cpu.json`、`foot_motion_regression.json`、
`foot_trajectory_newton_*.json/.npz` 及 GL 预览视频/PNG，均是忽略的生成产物。

## 后续训练前需要完成

1. 定义与足部位置、朝向、接触时序一致的可微 Reward，以及适合平地行走的终止条件。
2. 设计独立于 Motion 的平衡站立重置，解决当前 Motion reset 与平地足部命令可能不一致的问题。
3. 重新定义 Critic 特权特征，移除 Motion 姿态/全身误差；按新任务评估命令可行性和采样范围。
4. 检查真实接触和足部滑移，测量批量吞吐、Replay 显存和编译性能，再开展 WM/Policy 训练。

目前的 Reward 仍追踪全身 Motion，不能将足部模式接口 smoke 或原 Motion return 解释为
足部轨迹跟踪效果。World Model 本身无需因 Goal 维度变化而增加输入；策略/Critic 的足部权重需要重新初始化。
