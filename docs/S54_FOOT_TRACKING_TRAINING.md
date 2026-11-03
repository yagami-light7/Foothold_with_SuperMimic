# S54 Foot Tracking MDP 与 Standing 训练前检查

后续失败归因见 [Standing 诊断报告](../STANDING_DIAGNOSIS.md)：已复现后倾与高度终止，补齐实际接触、配对 WM 误差及 2/5/10 秒多种子评估；当前不具备直接长时间正式训练的条件。

2026-11-03，实际修改位置为当前 worktree `supermimic_s54_main`。
独立 Foot MDP、时间同步、FK 及 WM→Actor 梯度检查通过，Standing 最小训练已实际运行。
**目前不具备直接启动长时间正式训练的条件：最终 checkpoint 的无噪声评估 4/4 在第 96/100 步终止。**
这证明训练链路可运行，不代表 Standing Balance 已收敛。没有启动长时间训练。

## 审查结论与修改文件

原 Goal 接口阶段仍用全身 Motion 奖励、参考高度/倾角终止、Motion bin reset、Motion 误差 Critic，
并按 clip 长度筛选 BPTT 起点。旧梯度测试也包含直接动作代价，不能独立证明足部奖励的反传。

| 环节 | 当前 Foot 任务行为 | 保留的 Motion 模式 |
| --- | --- | --- |
| Goal / Observation | 固定世界足步命令、当前根 yaw 局部表示、固定命令 heading；99 维观测、230 维 Goal | 原 Motion Goal、参考根朝向 |
| Reward | S54 可微 FK、命令接触/相位、计划落脚事件与机器人稳定项 | 原全身 Motion tracking reward |
| Reset | S54 模型合法默认关节角、几何落地高度、微小状态扰动 | 原 Motion bin 采样和参考状态 |
| Hard / Soft Termination | 根部相对模型站高的偏差、相对世界竖直的倾角；硬终止另外检查 finite | 原参考高度/倾角误差 |
| Critic | 当前足部位置/朝向/速度误差、base 误差、接触/相位与 xi，无人体参考 | 原关节/全身/根部参考误差 |
| Replay | 复用不可变命令 ID/参数快照和整数 `ep_step` | 原 `f`、历史、WM window 逻辑 |
| BPTT 起点 | 按 episode 执行步数和 horizon 筛选，无 clip 边界查询 | 原 clip 长度筛选 |
| Train / Evaluate / Launcher | 不选择或加载 Motion CSV，可在不存在的 Motion 路径下工作 | 原目录与 clip 选择 |

Foot 模式的 `MotionLib` 实例为 `None`；保留的 `CONTROL_DT` 导入只是共用 0.02s 控制常量，
没有读取人体数据。Replay 的 `f` 字段作为兼容字段保留，Foot MDP 不用它查询任何参考。
历史的 `view_s54_foot_trajectory.py` 仍是带 Motion PD 的接口预览工具；本次独立任务验证使用
`test_foot_tracking_newton.py`，训练与评估均通过新的独立 MDP。

| 文件 | 本轮改动 |
| --- | --- |
| `supermimic/foot_mdp.py`（新增） | FootTrackingCfg、独立奖励/终止/Critic、模型站姿、task 契约 |
| `supermimic/mdp.py` | 提取机器人公共初始化，给旧接口增加可忽略的命令/时间关键字；保留原奖励数学定义 |
| `supermimic/goals.py`、`foot_trajectory.py` | Standing 复用零步序列；统一时间转换、time-major 命令展开 |
| `supermimic/env.py` | 独立 reset/timeout，按当前命令与下一执行步计算奖励；原 Motion 分支继续保留 |
| `supermimic/train.py` | 按任务构造 MDP/Critic、传递命令和整数时钟、纯足部梯度及同步诊断、任务 checkpoint 契约 |
| `supermimic/foot_evaluate.py`（新增）、`evaluate.py` | 独立无噪声 Foot 评估、foot NPZ、恢复任务配置，原 Motion 指标保持原口径 |
| `configs/s54_standing_balance.json`（新增） | 固定双支撑、100 步 episode、小幅 reset 扰动、不做域随机化 |
| `run_train.sh` | Foot MDP 配置透传；Foot 模式跳过 Motion 选择 |
| `tools/run_s54_standing_smoke.sh`（新增） | 强制限制 8 环境、40 次迭代的本机测试入口，保存 stdout |
| `tools/test_foot_tracking.py`（新增） | 奖励响应、落脚边界、可微 FK、编译图、checkpoint 与学得 WM 梯度检查 |
| `tools/test_foot_tracking_newton.py`（新增） | 无 Motion 的实际 Newton 时钟/Replay/FK 检查 |
| `tools/test_s54_launcher.py` | 无 Motion 目录的 Standing 参数 dry-run 回归 |
| 本文、`README_S54.md`、`S54_FOOT_TRAJECTORY.md` | 当前状态、复现命令与历史接口阶段的区分 |

World Model 的状态、输入、输出、网络和监督损失未更改。BPTT return、survival、Critic bootstrap、GAE、
梯度裁剪及优化器规则未更改；新传递的命令/时间是 MDP 上下文。纯足部梯度的额外求导仅作诊断，不参与优化器更新。

## Foot MDP 定义

位置和线速度来自 `S54Kinematics.forward(S)`，脚侧按模型中的 `leg_l6_link`、`leg_r6_link` 解析。
位置、线速度采用平滑有界分数 `1 / (1 + squared_error / std²)`，避免大误差时指数项过早饱和。

| 分量 | 定义 | 默认权重 |
| --- | --- | --- |
| Swing Tracking | Swing 脚世界位置/线速度相对连续参考的误差，按双脚平均 | 1.0 / 0.25 |
| Touchdown Accuracy | 首次跨过计划 touchdown 的控制区间，比较实际踝部位置和目标落脚位置 | 0.5 |
| Stance Stability | Stance 脚相对固定世界接触目标的位置误差，以及世界脚速平方 | 1.0 / 0.25 |
| Foot Orientation | 实际与目标旋转矩阵的平滑弦距离 | 0.25 |
| Base Upright | 相对世界竖直的 `1 − Rzz` | 1.0 |
| Base Height / Balance | 相对模型站高的误差、相对双脚参考中点加模型偏置的水平误差 | 0.5 / 0.5 |
| Action Smoothness | 相邻动作差平方和 | −0.1 |
| Joint Limit | 越出软限位的角度距离和 | −2.0 |
| Posture / Joint Velocity | 相对 S54 模型默认角的偏差、关节速度平方均值 | −0.05 / −0.001 |

最终奖励统一乘 `CONTROL_DT=0.02s`。Standing 无 Swing/Touchdown，正奖励上界为 **0.07/step**；
默认 Alternating 上界为 **0.08/step**。训练日志的 `sim_rew`、`imag_rew` 都用相应任务上界归一化，
不是与旧 Motion return 的跨任务性能比较。

Touchdown 是**计划落脚事件**，Stance 是 FK 位置/速度稳定性代理；目前没有将实际接触力/滑移传感器
加入 WM 或可微奖励。它们仍需作为后续动力学验证指标。奖励不使用真实物理接触 bool 替换 imagined rollout 的定义。

S54 合法默认膝、肘关节位于硬限位端点。原对称软区间会对该默认站姿产生约 0.515rad 的总违例；
Foot 模式将合法默认角纳入软区间，原 Motion 模式的软限位不变。
站高由足部几何和模型 FK 推导，当前为约 **0.9747m**，最低足部几何初始离地 3mm，没有 G1 站高或关节角。
命令在 nominal reset 状态上构建，随后才扰动真实状态，因此不会把 reset 扰动重新定义为接触目标。

硬终止为相对 nominal 站高偏差超过 0.25m、`1−Rzz > 0.5`，或非有限状态；
可微终止用对应物理阈值的 sigmoid 近似。episode timeout 单独处理，BPTT 起点保证 H 步不跨实际 reset，
timeout 的 continuing-task bootstrap 沿用原算法。

Critic 状态部分是 **135 维**：71 维 invariant state + 24 维足部误差 + 3 维 base 误差 +
4 维接触/相位 + 33 维 xi；加 230 维 Goal 后输入为 **365 维**。
新增 `task_contract` 保存奖励/重置/终止配置及 Critic 维度；拒绝恢复旧的 Goal-only Foot checkpoint。
原 Motion 默认 730 维 Goal、914 维 Critic 输入及旧 checkpoint 兼容路径保持可用。

## 时间同步和梯度证据

`ep_step` 始终是命令起点后的整数控制步。当前状态 `S_t` 查询未来 Goal；
执行 `a_t` 后，真实环境与 imagined rollout 都用 **同一命令、`ep_step+1`** 计算 `S_{t+1}` 的奖励。
time-major 展平的 `(H,B)` 样本同时展开命令，Critic 的 `(H+1,B)` 上下文也包含初始时刻。
自动 reset 前捕获旧命令快照，Replay 持有其参数，reset 后仍可重新计算最后一条 transition。

新增检查修复了一个实际浮点边界问题：float32 的 `40×0.02` 可稍小于 float32 的 `0.8`。
现在先用 float64 根据整数步及未来偏移构造时间，再一次舍入到轨迹 dtype；
当前查询、未来 Goal 和落脚事件共用该转换。落脚区间两端分别由相邻整数步计算，避免重复事件。
单测检查 0.8s 的 touchdown 只触发一次，并与未来 Goal 的 contact 一致。

- 新 MDP CPU 检查 **7 项 PASS**：奖励响应、无 Motion 依赖、关节 FK 梯度、有限差分、time-major、落脚边界、编译图和契约。
- 原轨迹/网络/Replay 回归 **13 项 PASS**；启动器回归 **21 项 PASS**。
- Newton **4 环境×120步 PASS**：13 次计划落脚、4 次 reset，覆盖左右 Swing 和 Double Support。
  Goal 最大重建差 **1.19e-7**，Reward 最大重建差 **1.49e-8**，足部 FK 与仿真最大误差 **7.30e-7m**。
- 原 Motion Newton **1环境×60步 PASS**：0 failed，双脚真实接触，最大步进 FK 位置误差 **3.63e-7m**。
- 加载最终训练得到的 WM 和 Actor，在 CPU FP32 冻结 WM 复核 8 步 BPTT：
  **纯足部奖励 Actor 梯度范数 0.0589845**。不包含直接动作代价、Posture/Base 项、Critic 或 survival 的导数，且不执行优化器更新。

## Standing Smoke 实验与结论

三轮均严格限制 **8 环境、40 iterations、1056 条环境 transition**，完整使用原训练循环与全尺寸网络。
共同配置：buffer 64、warmup 32、WM window 8、WM batch 32×2、PI batch 16、horizon 8、history 2、BF16、eager。
没有运行长期训练。当前脚本采用第三轮设置。

下表为最后 5 次迭代的日志均值；Episode Length 按整轮已完成 episode 数加权，避免把无新 episode 的窗口记为长度 0。
Actor 梯度为原代码报告的**裁剪前**范数，优化器仍按原规则裁剪。

| 实验 | LR / WM-only / 动作噪声 | WM loss / 8步误差 | Imagined / Simulation reward | Actor / 纯足部梯度 | Episode Length / 物理终止 |
| --- | --- | --- | --- | --- | --- |
| `s54_standing_smoke_20261103` | 2e-4 / 10 / 0.1 | 5.0003 / 9.3808 | 0.4434 / 0.2388 | 222.096 / 0.6383 | 97.47 / 7 of 8 |
| `s54_standing_smoke_conservative_20261103` | 5e-5 / 20 / 0.1 | 1.7267 / 2.7288 | 0.7830 / 0.9687 | 0.7255 / 0.01259 | 97.83 / 6 of 8 |
| `s54_standing_smoke_final_20261103` | 5e-5 / 20 / 0.03 | **1.0214 / 1.7259** | **0.7606 / 0.9853** | **2.7808 / 0.02829** | **98.33 / 3 of 8** |

高仿真 reward 的末窗口位于 episode reset 之后，不能据此声称保持平衡能力提高。
首轮 stdout 的 `foot_grad` 曾误打印同步误差列，已修正；上表首轮数据取正确的 CSV `foot_actor_gradnorm`。
最终轮 Goal/Reward 同步误差分别不超过 1.19e-7 / 1.49e-8。

实际 CSV 与评估状态生成的[诊断图](../results/s54/standing_smoke_diagnostics.png)展示了三轮 WM loss、
最终轮奖励/梯度，以及评估根高度越过终止阈值的过程。

最终 checkpoint 的独立评估使用 **4 个环境、相同 nominal 初态、无动作/观测/reset 噪声**：

| 指标 | 结果 |
| --- | --- |
| Normalized return（按完整 100 步归一化） | 0.801660 |
| 平均实际 episode length | 96 步，即 1.92s |
| 提前终止 | **4/4** |
| Foot position RMSE | 0.01634m |
| 终止根高度 | 约 0.7048m，低于约 0.7247m 的下限 |
| 参考双脚目标随时间的最大变化 | **0**，保持固定 |

结论：**MDP、数据和梯度链路具备继续受控实验的条件；目前不具备直接开启长时间正式训练的条件。**
需要先提高独立/held-out 的 WM 多步预测质量，区分模型误差、Critic bootstrap 和 Actor 对模型的利用，
验证更长 episode、扰动恢复及实际接触/滑移。Moving Foot Tracking 还需真实 Swing 数据、可达性/平衡检验和课程设计，
不能把本次 stationary smoke 当作已经学会行走。

## 复现与产物

以下 GPU 命令在宿主机可访问 GPU 的终端执行；测试期间没有改动原 worktree、G1 或既有容器。

```bash
cd /mnt/sage_data/workspace/supermimic_s54_main
S54_PYTHON=/mnt/sage_data/workspace/IsaacLab_3_0/.venv/bin/python

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 "$S54_PYTHON" tools/test_foot_tracking.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 "$S54_PYTHON" tools/test_foot_trajectory.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_s54_launcher.py

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  WARP_CACHE_PATH="$PWD/.cache/warp" "$S54_PYTHON" tools/test_foot_tracking_newton.py \
  --num_envs 4 --steps 120 --command_mode alternating --viz none

# 有意限制为 8 env / 40 iterations，必须用新的运行名。
bash tools/run_s54_standing_smoke.sh s54_standing_check_new

# 已有最终 checkpoint 的纯足部梯度审计（CPU，不进行训练）。
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 "$S54_PYTHON" tools/test_foot_tracking.py \
  --checkpoint logs/s54_standing_smoke_final_20261103/last.pt \
  --out results/s54/standing_final_gradient.json

# 独立任务评估；不存在的 Motion 路径用于验证不会加载人体参考。
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  WARP_CACHE_PATH="$PWD/.cache/warp" "$S54_PYTHON" supermimic/evaluate.py \
  --run_dir logs/s54_standing_smoke_final_20261103 \
  --ckpt logs/s54_standing_smoke_final_20261103/ckpt_000040.pt \
  --motion_dir /nonexistent/unused_motion_reference --num_envs 4 --viz none \
  --record results/s54/standing_final_eval.npz

# 只检查通用启动器参数，不启动其默认大配置。
bash run_train.sh --dry-run train s54_standing_preview --goal-mode foot_trajectory \
  --foot-task-config configs/s54_standing_balance.json --check-foot-gradients --no-swanlab
```

本轮产物：

- `logs/s54_standing_smoke{,_conservative,_final}_20261103/`：每轮 args、stdout、CSV、`last.pt`、编号 checkpoint 和显存记录。
- `logs/s54_standing_smoke_final_20261103/ckpt_000040_eval.json`：实际评估失败结果。
- `results/s54/standing_training_summary.json`：三轮指标、梯度、Newton 同步和独立评估的汇总。
- `results/s54/standing_final_gradient.json`、`foot_tracking_newton_preflight.json`、`foot_mdp_motion_newton_regression.json`：验证证据。
- `results/s54/standing_final_eval.npz`：实际状态、固定足部命令及执行步数，可离线重建目标；此任务 NPZ 不使用旧 Motion 对比渲染器。

CSV 新增 `foot_actor_gradnorm`、`goal_sync_max_error`、`reward_sync_max_error`、`completed_episodes`。
`ep_len=0` 表示该日志窗口没有新完成的 episode；汇总应按 `completed_episodes` 加权。
完整 `last.pt` 包含 Critic/优化器，编号 checkpoint 供评估；恢复必须保持 Goal 和 Task 契约一致。
