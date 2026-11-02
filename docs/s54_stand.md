# S54 站立测试：参数来源与运行说明

本项目新增的 `tests/sim/test_s54_stand.py` 是 **固定参考姿态的关节 PD 基线测试**。
它用于检查模型、关节控制、限幅和地面接触；没有运行开源 RL 策略，也没有移植 MPC/WBC。
因此，源码中的参考姿态和 PD 参数不意味着机器人只靠这些常数就能持续站稳。

## 1. 先分清几类参数

以下路径均相对于 `/mnt/sage_data/workspace/kuavo-ros-opensource`，取自本机源码。

| 内容 | 文件与字段 | 本测试的用法 |
|---|---|---|
| 27 个模型关节顺序 | `src/humanoid-control/humanoid_controllers/config/kuavo_v54/mpc/task.info`，`jointNamesReal`，第 64 行起 | 左腿 6、右腿 6、腰 1、左臂 7、右臂 7；运行时按名称匹配 |
| 可解释为关节力矩 PD 的增益 | 同一配置目录下 `rl/skw_rl_param.info`，`jointKp` / `jointKd`，第 268 / 300 行起 | 默认 Kp / Kd 基线 |
| 完整的名义关节姿态 | 同上，`defaultJointState`，第 172 行起 | 默认目标位置，单位 rad |
| 常规/峰值配置力矩限幅 | `src/kuavo_assets/config/kuavo_v54/kuavo.json`，`joint_torque_limits` / `joint_peak_torque_limits`，第 96 / 102 行起 | 使用常规限幅；保存峰值供比较 |
| 速度限幅 | 同上，`joint_velocity_limits` | 传给求解器，并记录实际速度与限值之比；实际执行取决于后端 |
| 启动手臂姿态 | 同上，`init_arm_pos`，第 258 行 | 度转弧度；通过 `--arm-pose init` 选择 |
| MPC 站立手臂姿态 | `command/reference.info`，`standJointState`，第 294 行起 | 只有 14 个手臂关节；通过 `--arm-pose mpc` 选择 |
| 模型关节范围/轴向 | `src/kuavo_assets/models/biped_s54/urdf/biped_s54.urdf` | 提取用于查阅；最终以本项目 USD 实际导入范围为准 |

`AmpWalkController.cpp` 第 235–236 行读取 `jointKp/jointKd`，第 1989–2006 行将它们用于
位置误差到力矩的计算。`src/mujoco/src/mujoco_node.cc` 第 730 行也直接给出了：

```text
tau = tau_ff + Kp * (q_des - q) + Kd * (dq_des - dq)
```

本测试设 `dq_des=0`、`tau_ff=0`，并限制 `tau` 的绝对值。
旋转关节的 Kp 单位为 N·m/rad，Kd 为 N·m·s/rad。
默认使用 `ImplicitActuatorCfg`，由求解器执行驱动；`--actuator explicit` 使用 `IdealPDActuatorCfg`
在每个物理步计算并裁剪力矩。两者的离散数值行为不同。

**不能直接把 `kuavo.json` 的 `joint_kp/joint_kd` 当成这套仿真 PD。**
该文件只有 15 项这类参数；同一 RL 配置中的 `motor_pdo_kp/motor_pdo_kd` 注释把它们标为
EC_MASTER 电机参数。另有 Ruiwo 电机增益，硬件配置总关节数为 29，包含两个头部关节。
当前 USD 的两个头部关节固定，只有 27 个可动关节。尤其不能将 `kd=4880` 等整数直接赋给所有仿真关节。

`command/reference.info` 的 `joint_kp_/joint_kd_` 则属于另一套控制路径，配合控制模式和前馈力矩；
其手臂 Kp 为零，不能孤立地复制为全身保持姿态控制器。
厂商的 [kuavo.json 配置说明](https://github.com/LejuRobotics/kuavo-ros-opensource/blob/main/docs/kuavo_json%E6%96%87%E6%A1%A3%E8%AF%B4%E6%98%8E.md)
也区分了模型配置与电机信息；具体 S54 数值以本机版本为准。

## 2. 本次提取的参数

左右腿相同，`leg_l1..6_joint` / `leg_r1..6_joint` 依次如下。
限幅表示对称范围 `[-数值, +数值]`，并非只允许正力矩。

| 关节 | q0 (rad) | q0 (°) | Kp | Kd | 常规限幅 (N·m) | 峰值配置 (N·m) |
|---|---:|---:|---:|---:|---:|---:|
| 1 髋 roll（X） | 0 | 0 | 100 | 5 | 88.9 | 127 |
| 2 髋 yaw（Z） | 0 | 0 | 100 | 5 | 49.7 | 71 |
| 3 髋 pitch（Y） | -0.10 | -5.73 | 80 | 5 | 92.4 | 132 |
| 4 膝 pitch（Y） | +0.25 | +14.32 | 80 | 6 | 196 | 280 |
| 5 踝 pitch（Y） | -0.15 | -8.59 | 30 | 7.5 | 39.9 | 57 |
| 6 踝 roll（X） | 0 | 0 | 50 | 7.5 | 39.9 | 57 |
| 腰 yaw | 0 | 0 | 40 | 7 | 71.4 | 102 |

这是一组微屈膝姿态，髋、膝、踝 pitch 相加为零，直立基座下双脚近似水平。
腿部常规速度配置为 `[10.4, 8.7, 12.7, 15, 17.8, 17.8]` rad/s。

每条手臂按 `zarm_l1..7_joint` / `zarm_r1..7_joint`：

```text
Kp       = [15,   30,   15,   30,   15,   15,   15]
Kd       = [ 6,    3,    3,    6,    3,    3,    3]
常规限幅 = [46.2, 52.5, 39.9, 52.5, 9.87, 9.87, 9.87] N·m
峰值配置 = [66,   75,   57,   75,   14.1, 14.1, 14.1] N·m
RL 左臂  = [0, +0.075, 0, 0, 0, 0, 0] rad
RL 右臂  = [0, -0.075, 0, 0, 0, 0, 0] rad
init 双臂= [20, 0, 0, -30, 0, 0, 0] deg
MPC 双臂 = [0.15, 0, 0, -0.4, 0, 0, 0] rad
```

**踝关节限幅存在模型差异。** S54 硬件包含并联踝机构，而 USD 使用 pitch/roll 等效旋转关节。
`kuavo.json` 的踝限幅为常规 39.9、峰值 57 N·m；当前源码 URDF 的等效踝 pitch/roll effort
却为 91.6 / 68.4 N·m，当前 USD 的踝 roll 默认驱动 maxForce 又为 30 N·m。
本测试明确选择 JSON 常规值覆盖驱动，作为模型实验假设；这不是并联电机可行力矩域的精确映射，
也不能视为实机每个等效关节的额定连续能力。后续精确建模需要并联机构的雅可比映射和电机限幅。

## 3. 初始高度与模型版本

不要把所有出现的高度字段视为同一个量：JSON 中 `com_z=0.90`，
MPC `standBaseState` 中 z=0.78，RL `defaultBaseState` 中 z=0.97、pitch=0.02 rad。
它们来自不同控制路径，不能混成一组初始化条件。

脚本采用**基座直立**、RL 名义关节角，再根据实际 USD 的关节坐标系做正运动学，
计算脚部碰撞体包围盒最低点。这是测试的初始化选择，没有照搬 RL 的 0.02 rad 基座倾角。
当前模型结果：

- 总质量约 **67.4826 kg**，重力约 **662.00 N**。
- 足底最低点相对基座约 **-0.966025 m**。
- 默认基座高度 **0.969025 m**，包含 **3 mm** 初始离地间隙。
- 对称静态支撑下每脚约 **331 N**，允许左右负载有差异，重点观察两脚之和与 mg。

USD 的实例化碰撞几何也参与计算；胶囊用包围盒计算，因此高度是保守几何估计。
导入后还会再次检查目标角是否在模型关节范围内。
脚本在运行场景中解除实例化后设置摩擦材料，不改写资产文件；地板由静态长方体生成，顶面为 z=0。
自碰撞关闭，适合先隔离站立接触问题，不能用于验证复杂动作的自碰撞可行性。

## 4. 运行顺序

在项目根目录执行。Python 使用 IDE 当前配置的 Isaac Lab 虚拟环境：

```bash
cd /mnt/sage_data/workspace/KUAVO_S54_Newton
S54_PY=/mnt/sage_data/workspace/IsaacLab_3_0/.venv/bin/python

# 更新/重新提取参数；仅依赖 Python 标准库
python3 scripts/extract_s54_params.py

# 只检查配置、模型和初始足底高度，不运行物理仿真
"$S54_PY" tests/sim/test_s54_stand.py --inspect-only

# 第一阶段：固定基座，验证关节姿态和 PD；不能据此判断站立成功
"$S54_PY" tests/sim/test_s54_stand.py --fixed-base --num-steps 1000 \
  --device cuda:0 --viz newton_gl --log logs/s54_fixed.csv

# 第二阶段：源码参数基线（本次测试会跌倒，记录并比较结果）
"$S54_PY" tests/sim/test_s54_stand.py --num-steps 5000 \
  --device cuda:0 --viz newton_gl --log logs/s54_source_baseline.csv

# 第三阶段：已在 CPU 上通过 10 秒站立的仿真调试增益，保持常规力矩限幅
"$S54_PY" tests/sim/test_s54_stand.py --num-steps 5000 \
  --kp-scale 30 --kd-scale 5.47722558 \
  --device cpu --viz none --log logs/s54_stand.csv

# 有可用 GPU 时，将上一命令的设备和显示参数换成：
# --device cuda:0 --viz newton_gl

# 无界面或无 GPU 的接口检查，可用 MuJoCo Warp 的 CPU 路径（较慢）
"$S54_PY" tests/sim/test_s54_stand.py --device cpu --viz none --num-steps 100
```

本地 Isaac Lab 的四元数约定是 **XYZW**；脚本使用 `(0,0,0,1)` 表示直立。
API 使用本机版本的 `.torch`、`target_command.set_position_index()` 和 clone plan。
不要直接套用其他 Isaac Lab 版本的 WXYZ 示例。

当前集成的 `MJWarpSolverCfg(use_mujoco_cpu=True)` 会在接触初始化的
`get_max_contact_count()` 抛出 `NotImplementedError`。因此 CPU 命令仍使用 **MuJoCo Warp**，
没有切换到该原生 MuJoCo CPU 选项。

## 5. 如何判断结果、继续写控制器

主循环只设置一次固定的 `q_des`，随后每步执行：

```python
robot.write_data_to_sim()
sim.step()
robot.update(dt)
contact_sensor.update(dt)
```

初始化时一次性写入 root/joint 状态；循环中不能反复写入位置状态，否则会绕过物理响应。
所有 PD 和限幅按关节名配置，不依赖 USD 导入后的排列顺序。

CSV 记录基座高度、倾角、水平漂移、最大关节误差、双脚法向力、总支撑力/mg、
最大驱动力矩比例和最大速度比例。隐式驱动的 `applied_torque` 是执行器侧估计值，
不能当成完整关节反力或实测电机力矩。

当前通过条件是调试阈值，定义在脚本末尾：运行至少 5 秒，最后 1 秒的每个物理步均满足
倾角 <10°、漂移 <0.10 m、最大关节误差 <0.35 rad、每脚支撑力 >5 N，且平均总支撑力在
0.8–1.2 mg。跌倒（高度低于初始值的 70% 或倾角超过 35°）立即停止。
`FAIL` 返回码为 2；短时运行输出 `INCONCLUSIVE`；固定基座输出 `FIXED_BASE_DIAGNOSTIC`。
这些检查不等同于抗扰动、长期稳定或实机验证。

若出现问题，按现象定位：

1. **脚穿地、起步冲击大**：检查碰撞形状、初始间隙和关节方向；先解决几何问题。
2. **腿持续压弯、关节误差增加**：检查膝/髋驱动力矩、限幅以及 Kp。RL 名义角只是策略的参考中心，
   原控制器还会叠加策略输出；保持该中心不等于运行该策略。
3. **角度跟踪尚可但整机前后倒**：需要质心/姿态反馈、重力前馈或 WBC。仅提高 Kp 不保证平衡。
4. **抖动明显**：检查 dt、Kd、接触求解器和饱和；不要盲目同时提高 Kp/Kd。

实验性调参可用 `--kp-scale 2 --kd-scale 1.414`，但这些倍率是自行选择的调试值，
不是从源码提取的新参数，也不保证稳定。可通过 `--arm-pose init` 比较手臂姿态对质心的影响。
对于更可靠的站立控制，下一步应在本测试通过模型/接触检查后加入状态反馈与力矩前馈，
或移植开源 MPC/WBC/RL 控制链。

提取结果保存在 `configs/s54_stand.json`，包含源文件相对路径和 SHA-256，
便于区分本次参数与后续源码更新。重新提取会覆盖此 JSON；自定义调参应先复制配置，再用 `--config` 指定。

## 6. 本次实际验证结果（2026-10-18）

在当前 Isaac Lab 虚拟环境和当前 `assets/biped_s54.usd` 上，使用 CPU、MuJoCo Warp、
dt=0.002 s、重力 -9.81 m/s²、摩擦系数 1.0、自碰撞关闭。测试环境未检测到可用 CUDA，
GPU 与 GUI 效果未验证。

| 条件 | 结果 | 日志 |
|---|---|---|
| 原始 RL 增益，固定名义姿态，浮动基座 | 1.290 s 触发跌倒检测 | `logs/s54_stand_baseline_cpu.csv` |
| Kp×10、Kd×√10，浮动基座 | 1.588 s 触发跌倒检测 | `logs/s54_stand_scaled_cpu.csv` |
| **Kp×30、Kd×√30，浮动基座、隐式驱动** | **10 s 站立通过** | `logs/s54_stand_scaled30_cpu.csv` |
| 原始增益、固定基座、init 手臂姿态、显式驱动 | 2 s 接口诊断完成，最大关节误差约 0.097 rad | `logs/s54_fixed_cpu.csv` |

10 秒通过的运行结束时，基座高度约 0.9646 m，倾角约 1.27°，水平漂移约 0.0189 m，
最大关节误差约 0.0223 rad，双脚法向力约 330.24 / 331.75 N，总支撑力约 1.000 mg，
执行器侧估算力矩最大约为限幅的 50%。

**30 和 √30 是本次仿真实验选出的倍率，不是厂商站立 PD 参数。**
例如测试中的腿部 Kp 变为 `[3000,3000,2400,2400,900,1500]`；这组高刚度参数只验证了当前
理想驱动模型的静态场景。它没有补齐电机带宽、延迟、并联踝耦合、抗扰动和实机约束。
配置文件仍保留源码原值，必须显式传入倍率才会采用这组实验值。
