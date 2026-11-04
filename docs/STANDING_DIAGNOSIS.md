# S54 Standing Balance 失败诊断

日期：2026-11-03。目标 checkpoint：`logs/s54_standing_smoke_final_20261103/ckpt_000040.pt`。

**结论：主要失稳机理是向后倾倒，随后根高度越界。当前策略没有学会主动平衡；默认模型站姿的固定 PD 也不能长期维持该初态。允许继续分阶段、有上限的实验，目前不具备直接开启长时间正式训练的证据。**

本次没有训练更新，没有修改 Reward 权重、终止阈值、机器人驱动参数或 SuperMimic 的 WM/BPTT 算法。GPU 验证在脱离沙盒的 RTX 4060 Laptop GPU 上完成；3090 配置是待验证方案，未在 3090 上运行。既有日志和 checkpoint 保留。

## 1. 复现及失败机理

通过原 `supermimic/evaluate.py` 入口，以相同 4 环境、nominal 初态、无 reset/观测/动作噪声复现：

| 指标 | 原结果 | 本次复现 |
|---|---:|---:|
| 实际 episode length | 96 步 | 4/4 均 96 步，1.92 s |
| 物理终止 | 4/4 | 4/4，均 `height` |
| 正常 timeout | 0 | 0 |
| Return / 完整 100 步奖励上界 | 0.801660 | 0.801660 |
| 实际存活期间平均 Reward / 单步上界 | 0.835063 | 0.835062 |
| 足部位置 RMSE | 0.01634 m | 0.01634 m |

新旧记录根位置最大差约 `1.60e-5 m`，关节位置最大差约 `2.29e-4 rad`；终止步数一致。
4 个 nominal 环境是相同初态的复现，不作为 4 个独立扰动试验。

单环境的同初态对照包含零动作默认 PD：目标严格为 S54 模型 `default_q`，不使用 IK、重力补偿或 Motion Reference。
另以 4 环境复核默认 PD，4/4 同样在第 97 步以高度越界终止。

| 时刻/控制器 | Root Height (m) | Root Pitch (deg) | 左/右膝位置 (rad) | Reward / 单步上界 |
|---|---:|---:|---:|---:|
| 初态 | 0.97470 | 0 | 0 / 0 | — |
| 策略，第 50 步 | 0.96809 | −6.52 | 0.034 / 0.035 | 0.9334 |
| 策略，第 75 步 | 0.94117 | −20.77 | 0.082 / 0.093 | 0.6899 |
| 策略，第 90 步 | 0.83619 | −34.62 | 0.010 / 0.015 | 0.4141 |
| 策略，第 96 步 | 0.70479 | −45.21 | −0.0015 / −0.0016 | 0.2931 |
| 默认 PD，第 97 步 | 0.70365 | −45.58 | −0.0035 / −0.0035 | 0.2889 |

模型膝关节已从 MJCF 核实为 `leg_l4_joint` / `leg_r4_joint`。负的微小终端角是求解器约束容差下的越限，不是大幅弯膝塌陷。

- 策略全程最大关节偏移 `0.1183 rad`、最大关节速度 `0.5396 rad/s`；终止时 Roll 仅约 `0.15°`，Pitch 单向增大。
- 最大动作绝对值 `0.0620`，相邻步最大动作变化 `0.00269`，动作变化 RMS `0.000931`；最大 PD 目标偏移 `0.02170 rad`。曲线没有高频大幅振荡特征。
- 默认 PD 零动作也沿相同方向失效，且只多存活 1 步。可确认当前策略未形成有效纠偏；不能凭这一帧差异认定某个关节输出是唯一原因。
- 双脚参考世界坐标漂移为 **0**。足部误差在前半段很小，而身体已逐步后倾；后半段踝部上抬、脚姿态改变，符合足部滚转支撑并倾倒的过程。
- 直接机理是身体后倾及支撑能力下降；没有证据把主要原因归于关节持续塌陷、接触被禁用或动作振荡。固定 PD 站姿失败也说明“合法模型 neutral pose”并不保证在当前动力学下稳定平衡。暂不凭单次轨迹改驱动或初始关节姿态。

证据：[策略完整诊断图](results/s54/standing_diagnosis/nominal_policy.png)、[默认 PD 完整诊断图](results/s54/standing_diagnosis/nominal_default_pd.png)、[左右脚 XYZ 图](results/s54/standing_diagnosis/nominal_policy_feet.png)。

## 2. 实际接触，而非计划接触

已获取实际接触。接口由本机固定版本源码核实：
`SolverMuJoCo.update_contacts(contacts, state)` 从 MJWarp 的 `efc.force` 导出接触，使用扩展属性 `contacts.force`；此版本的 `rigid_contact_force` 并未用于该导出，不能读取它来判断力为零。

诊断按 `shape_body` 和 `/World/envs/env_N/Robot/...` 标签，只汇总脚与世界地面的接触。世界坐标力的正负号与 Newton `SensorContact` 一致，取空间力数组前三项，shape0 正号、shape1 反号。

| 策略时刻 | 左/右地面接触点数 | 左/右竖直地面力 (N) |
|---|---:|---:|
| 第 25 步 | 24 / 24 | 327.6 / 333.3 |
| 第 75 步 | 8 / 8 | 307.7 / 315.6 |
| 第 90 步 | 4 / 4 | 198.3 / 180.2 |
| 第 96 步 | 4 / 4 | 107.0 / 90.2 |

S54 仿真模型总质量约 `67.43 kg`，早期总支撑力约 `661 N`，与重力数量级一致。末段接触点数及支撑力下降，但双脚仍接触地面，未观察到非脚部触地。

限制：记录是 **每个控制步最后一个物理子步** 的求解器接触和力，不是全 4 子步的平均力，也不是实机力传感器。刚释放后的第 1 步可有几何接触但力为零；“存在接触点”和“承载 >1 N”分开记录。reset 的第 0 帧接触未知，用 `count=-1`、力 `NaN` 标记。未据此声称已排除所有瞬时滑移、接触模型或实机参数误差。

## 3. WM 与 Newton 配对检查

协议：从录制的真实 Newton 状态启动 WM，逐步回放该段 **相同的实际动作序列**、`xi`、命令参数和整数执行步。真实路径与 WM 都按执行后的 `ep_step+1` 查询奖励。
三个 horizon 使用相同起点集合；允许包含物理终止的端点，禁止跨 episode/reset。策略 nominal 轨迹共 89 个重叠窗口，默认 PD 为 90 个。这些不是 89/90 次独立试验。

Reward 对比使用两侧相同的有限步原始奖励累加，再除以步数及单步上界。不包含 Critic、bootstrap、soft survival 或 discount；这能隔离动力学预测误差，不等同于完整 BPTT Actor loss。

**策略 nominal，BF16 网络、FP32 状态/FK：** 姿态为 SO(3) 角误差 RMSE；速度为逐轴分量 RMSE；关节为逐 DOF RMSE。

| 预测步数 | Base Height RMSE (m) | 姿态 RMSE (deg) | 线速度 (m/s) | 角速度 (rad/s) | 关节 q (rad) | 关节 qd (rad/s) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.00149 | 0.159 | 0.01735 | 0.00909 | 0.000554 | 0.01591 |
| 4 | 0.00686 | 0.771 | 0.03255 | 0.02981 | 0.001857 | 0.03477 |
| 8 | 0.01881 | 1.753 | 0.07072 | 0.08002 | 0.003730 | 0.05678 |

| 预测步数 | 相同窗口 Simulation Reward | 相同窗口 Imagined Reward | 平均偏差 (WM−Newton) | 窗口平均奖励差的 MAE |
|---|---:|---:|---:|---:|
| 1 | 0.873107 | 0.873899 | +0.000792 | 0.001699 |
| 4 | 0.863089 | 0.864355 | +0.001266 | 0.003878 |
| 8 | 0.848539 | 0.847927 | −0.000613 | 0.007601 |

8 步的逐步奖励 MAE 为 `0.00824`，高度绝对误差 p95 为 `0.02876 m`。
默认 PD 的 8 步高度 RMSE `0.01944 m`、姿态 `1.942°`，配对 Simulation/Imagined Reward 为 `0.848597 / 0.840035`，平均偏差 `−0.008563`。

存在有方向、随阶段变化的预测偏差：

- 策略 8 步平均高度偏差 `+5.74 mm`；起点在 0–0.5 s 时平均 `+13.04 mm`，中段 `+3.13 mm`，1.2 s 后 `+2.59 mm`。最后阶段高度 RMSE 增至 `29.07 mm`，不能只看平均偏差。
- 8 步局部旋转向量的 Pitch 分量平均偏差约 `+1.21°`，相对于实际负 Pitch，模型倾向低估后倾。线速度 X、角速度 Y 平均偏差约 `+0.0474 m/s`、`+0.0495 rad/s`。
- 3 个扰动种子的策略轨迹也做了配对检查：8 步高度 RMSE `0.0186–0.0202 m`，姿态 `1.54–1.81°`，奖励平均偏差 `−0.00268–+0.00137`，MAE `0.00737–0.00801`。
- 配对真实奖励重建最大差 nominal 为 `1.30e-8`；完整分关节 q/qd RMSE、逐起点预测误差及阶段指标均保存在 JSON/CSV。

**判断：** WM 多步误差和低估后倾的偏差仍需校准，但这些已测状态/动作上没有明显整体奖励高估。不能用训练 CSV 中不同批次的 imagined/simulation 均值归因“模型利用”。也不能由此排除未测试动作、长 horizon、Critic bootstrap 或模型梯度误差；状态预测接近不保证动作导数准确。

证据：[逐窗口 WM 图](results/s54/standing_diagnosis/wm_policy.png)、[配对指标和所有种子](results/s54/standing_diagnosis/summary.json)、[逐窗口 CSV](results/s54/standing_diagnosis/wm_policy.csv)。

## 4. Reward、终止及任务依赖审查

足部模式的 Goal、Observation heading、Reward、Termination、Critic 和 Replay 命令恢复均不需 Motion CSV。`FootTrackingMDP.motion=None`；脚位姿和速度由 S54 可微 FK 得到。真实与 imagined reward 共用 `FootTrackingMDP.reward()`，旧 Motion 路径保持单独配置。

| 项目 | 本次观察/判断 |
|---|---|
| Stance | 第 75 步位置质量仍 0.9972，而 Pitch 已约 −20.8°。仅看脚位置不足以判断 Standing 成功。计划 contact mask 不等于真实承载，实际接触作为独立诊断保留。 |
| Base Upright | 第 50/75/96 步质量约 0.8608 / 0.3809 / 0.1192，后倾能被检测。 |
| Base Height | 同三时刻约 0.9932 / 0.8506 / 0.0808；终止高度下限 `0.9747−0.25=0.7247 m`，实际 0.7048 m 已触发。 |
| Base Balance | 终止时质量约 0.0407；身体相对支撑中心偏移同样受惩罚。 |
| Posture | 是关节相对模型 neutral 的平方代价，不能检测浮动基座后倾。终止时 q 接近 neutral，代价仅约 0.00876；必须结合 Base 项解释。 |
| Action/Joints | 动作变化很小，未看到高频振荡；终止时有小幅关节越限，已有代价。 |
| 总 Reward | 从约 0.996 降至终止时 0.293；高度下降超过 8 cm 的 12 步平均仅 0.405。episode 平均 0.835 被早期接近直立的时段拉高。 |
| 终止 | nonfinite、绝对高度差 >0.25 m 或 `1−Rzz>0.5`（倾斜 >60°）；本次较早由 height 终止。物理终止与 timeout 同步发生时，归类为物理失败。 |

**未发现足以支持改权重的明确排序/实现错误，因此 Reward、驱动和终止逻辑全部保持原值；前后 Reward 配置相同，无调权对照实验。** 高 Stance 分数和较小整体脚 RMSE 不能作为 Standing 收敛证据。40 次迭代中只有后 20 次更新 Actor，当前实验更符合尚未学会纠偏，而不是已经学成的策略偶然掉高。

原梯度检查已证实纯足部奖励经 learned WM 到 Actor 的非零梯度（CPU FP32 范数 `0.0589845`，无 Critic/Base/直接动作代价）；本次未改变该链路。这证明导数路径存在，不证明模型导数准确或已经收敛。

## 5. 2/5/10 秒与多种子结果

3 个种子为 54/55/56，每个 1 环境，启用原配置中的小幅 reset 扰动，无观测和动作噪声，无 domain randomization。同 seed 的两种控制器初态及命令最大差均为 **0**。
延长评估时限只覆盖 `env.max_episode_steps`，不修改 checkpoint 的 MDP 契约或 Reward。

| 时限 | 策略：物理失败 / timeout | 默认 PD：物理失败 / timeout | 策略各 seed 实际步数 | 默认 PD 各 seed 实际步数 |
|---|---:|---:|---|---|
| 2 s / 100 步 | 3 / 0 | 2 / 1 | 100 / 99 / 87 | 100 / 100 / 87 |
| 5 s / 250 步 | 3 / 0 | 3 / 0 | 100 / 99 / 87 | 101 / 100 / 87 |
| 10 s / 500 步 | 3 / 0 | 3 / 0 | 100 / 99 / 87 | 101 / 100 / 87 |

策略 seed54 恰在第 100 步同时满足 height 和 timeout，明确记物理失败；默认 PD seed54 在 100 步 timeout，延长后第 101 步跌倒。**2 秒 timeout 不能视为稳定收敛。** 这些不同 duration 使用同一批种子，不是 9 个独立种子。

证据：[逐 episode CSV](results/s54/standing_diagnosis/evaluation.csv)、[训练曲线和评估存活时长](results/s54/standing_diagnosis/training_curves.png)。训练奖励曲线明确标注为不同批次均值，仅作描述。

## 6. 改动文件与数据

| 文件 | 本轮改动 |
|---|---|
| `supermimic/env.py` | 提取现有 PD 目标公式为 `pd_target()`，公开 transition `timeout`；公式和步进顺序不变。 |
| `supermimic/foot_evaluate.py` | 评估时限、种子、reset 扰动、默认 PD 对照、终止原因及完整状态/控制/奖励/接触记录；退出后恢复环境协议配置。 |
| `supermimic/evaluate.py` | 暴露 Foot 专属评估参数、多 seed、独立结果输出路径；Motion 原入口兼容。 |
| `supermimic/standing_diagnostics.py` | 实际 Newton ground contact/force 汇总、终止归因、配对窗口、1/4/8 步原单位误差与奖励比较。 |
| `tools/validation/diagnose_s54_standing.py` | 有界多时限/多种子仿真诊断，输出 JSON/NPZ/CSV/图；支持复用录制数据的 `--analyze_only`。 |
| `tools/visualization/plot_s54_standing_curves.py` | 复用现有训练 CSV 画曲线，无完成 episode 的窗口不画为长度 0。 |
| `tools/train/run_s54_standing_controlled.py` | 3090 方案默认 dry-run，分阶段、单 seed、固定迭代和 wall-time 上限；保留 stdout、CSV、checkpoint。 |
| `configs/s54/standing/s54_standing_3090.json`、`configs/s54/standing/s54_standing_3090_task.json` | 待验证受控配置，训练 episode timeout 改为 500 步（10 s）；Reward 参数不改。 |
| `tools/tests/test_standing_diagnostics.py` | 接触归属/力符号、同 tick 超时与物理失败优先级、窗口边界、oracle 配对误差、协议还原和训练上限检查。 |
| 本文、README、`docs/S54_FOOT_TRACKING_TRAINING.md` | 诊断结论与入口链接。 |

结果目录：`results/s54/standing_diagnosis/`。原始 NPZ：状态字段 `[T+1,N,...]`，动作、PD target、Reward/各项、term/timeout 为 `[T,N,...]`；每条动作作用于对应 `state[t]→state[t+1]`。保存 `state(71)`、根 pose/线角速度、29 关节 q/qd、前一动作、xi、完整不可变命令、执行 tick、足部 pose、实际接触及 force、首个 episode 长度。死亡环境后续帧不能纳入统计；配对窗口按各自首个终止长度截取。

## 7. 测试与限制

| 检查 | 状态 | 结果 |
|---|---|---|
| 原 4 环境 checkpoint 复现 | PASS（复现成功，策略表现 FAIL） | 4/4 第 96 步 height 失败 |
| 4 环境默认 PD，最多 500 步 | PASS（诊断执行成功，平衡表现 FAIL） | 4/4 第 97 步 height 失败 |
| 两控制器 × 三 seed × 2/5/10 秒 | PASS（协议执行） | 18 episodes，包含正常 timeout 和同 tick 物理失败 |
| 实际接触/力 | PASS | 真正 solver ground contact；非计划 bool；初始未知帧明确标记 |
| WM nominal + 三扰动 seed 配对 | PASS（检查执行） | 1/4/8 步及分关节误差、阶段偏差；并非“模型准确性达标” |
| 新诊断单测 | PASS | 7 项 |
| Foot MDP 单测 | PASS | 7 项 |
| 原轨迹/Goal/Replay/网络回归 | PASS | 13 项；该历史测试的 414 维 Critic 案例属于旧 Goal-only 接口，新独立 Foot Critic 仍为 365 维。 |
| 启动器回归 | PASS | 21 项 |
| 原 Motion Newton，1 环境 ×60步 | PASS | 0 failed，双脚真实接触，FK 最大位置误差 `2.46e-7 m` |
| 受控训练配置 | PASS（dry-run） | 校准 200、pilot 总计 600，单次最多 30 min；未启动训练 |
| 3090 显存/吞吐、较长训练收敛 | NOT RUN | 本机为 4060 Laptop，不能代替 3090 实测 |
| 超过 8 步 WM、实机稳定性、模型动作导数精度 | NOT RUN | 本轮结论不外推这些条件 |

Motion Newton 第一次调用默认 `data/retarget_data/reference` 因目录为空在构造仿真前退出；随后显式使用仓库已有 `reference_startup_fixed` 运行通过，无需创建或修改 Reference。

## 8. 下一阶段受控训练建议

先校准 WM 并重新做 paired audit，再进入有界 Actor pilot。当前 40-iteration checkpoint 不作为正式训练已收敛的起点。

配置：128 environments、buffer 512、warmup 256、WM batch 256×2、PI batch 64、horizon/window 8、history 2、LR `5e-5`、action noise `0.03`、BF16/eager，无 domain randomization。每 100 迭代保存 checkpoint，每 10 迭代写训练 CSV。

- **校准阶段**：200 次迭代，`wm_only_iters=200`，Actor 无更新；单次最多 30 分钟。先检查新 held-out 真实轨迹的 1/4/8 步原单位误差、早期沉降与后倾的有方向偏差、配对奖励误差，不能仅看 normalized WM loss。
- **pilot 阶段**：用户显式运行第二条命令恢复校准 `last.pt`，总目标 600 次迭代，额外单次最多 30 分钟；不会自动接续、循环或扩大 horizon。恢复会按现有机制重新填充 Replay。
- 新 task 的 10 s timeout 与原 smoke 的 2 s task contract 不同，必须 fresh 校准，不能直接 resume 旧 smoke。不要为加载旧 checkpoint 放宽契约。
- 初始 8 步仅覆盖 0.16 s，短期梯度未必能学出长期平衡。只有通过多步/导数质量检查后才考虑更长 horizon 或额外探索；本轮不改核心 BPTT。
- 每个保存点做 2/5/10 s、至少三个扰动 seed 的评估，检查根高度/倾角曲线和实际接触；每次运行一个训练 seed，之后分别做 54/55/56。**全部 seed 的 10 s timeout 且高度、倾角稳定**只是进入下一规模实验的最低证据，仍需独立种子和扰动恢复验证，不以平均高 Reward 替代。
- 若 WM 校准后仍不能准确预测或奖励偏差扩大，先停止扩大训练；若 WM 已改善但 Actor 仍接近默认 PD，下一次应审查动作导数、Critic 的长时估值及 horizon，而非无证据提高奖励权重。

本轮没有启动上述训练阶段。

## 9. 复现命令

在仓库根目录、可访问 GPU 的终端执行：

```bash
export S54_PYTHON=/mnt/sage_data/workspace/IsaacLab_3_0/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export WARP_CACHE_PATH="$PWD/.cache/warp"
export MPLCONFIGDIR=/tmp/s54_matplotlib
CKPT=logs/s54_standing_smoke_final_20261103/ckpt_000040.pt

# 本轮主要诊断（只仿真和离线推理，不训练）
"$S54_PYTHON" tools/validation/diagnose_s54_standing.py --checkpoint "$CKPT" \
  --out results/s54/standing_diagnosis --seeds 54 55 56 --durations 2 5 10 --viz none

# 使用已有 NPZ 重画和重算 WM，不重新仿真
"$S54_PYTHON" tools/validation/diagnose_s54_standing.py --checkpoint "$CKPT" \
  --out results/s54/standing_diagnosis --analyze_only --device cuda:0 --viz none

# 原入口、4 环境、原 2 秒无噪声复现；使用独立 JSON 避免覆盖历史结果
"$S54_PYTHON" supermimic/evaluate.py --run_dir logs/s54_standing_smoke_final_20261103 \
  --ckpt "$CKPT" --num_envs 4 --duration_s 2 --diagnostics \
  --record results/s54/standing_diagnosis/reproduce_4env_2s.npz \
  --out_json results/s54/standing_diagnosis/reproduce_4env_2s.json --viz none

# 同一入口对未来 checkpoint 的多 seed 10 秒评估；2/5 秒仅改 duration
"$S54_PYTHON" supermimic/evaluate.py --run_dir logs/s54_standing_smoke_final_20261103 \
  --ckpt "$CKPT" --num_envs 1 --duration_s 10 --seeds 54 55 56 --reset_noise \
  --out_json results/s54/standing_diagnosis/multiseed_10s.json --viz none

"$S54_PYTHON" tools/tests/test_standing_diagnostics.py
"$S54_PYTHON" tools/tests/test_foot_tracking.py --out results/s54/standing_diagnosis/mdp_regression.json
"$S54_PYTHON" tools/tests/test_foot_trajectory.py
"$S54_PYTHON" tools/tests/test_s54_launcher.py
"$S54_PYTHON" tools/validation/test_env_s54.py --num_envs 1 --steps 60 \
  --motion_dir data/retarget_data/reference_startup_fixed \
  --out results/s54/standing_diagnosis/motion_newton_regression.json --viz none

# 默认仅打印受控配置，不启动训练（本轮仅运行了这种 dry-run）
"$S54_PYTHON" tools/train/run_s54_standing_controlled.py --seed 54

# 下一阶段：运行校准。两个阶段分开执行；本轮均 NOT RUN。
"$S54_PYTHON" tools/train/run_s54_standing_controlled.py --seed 54 --execute
# 检查校准 checkpoint 的 paired diagnostics 后，再显式运行 pilot：
"$S54_PYTHON" tools/train/run_s54_standing_controlled.py --phase pilot --seed 54 \
  --resume logs/s54_standing_3090_calibration_seed54/last.pt --execute

# 训练曲线绘制；不再运行仿真或训练
"$S54_PYTHON" tools/visualization/plot_s54_standing_curves.py \
  --runs logs/s54_standing_smoke_20261103 logs/s54_standing_smoke_conservative_20261103 logs/s54_standing_smoke_final_20261103 \
  --evaluations results/s54/standing_diagnosis/evaluation.csv \
  --out results/s54/standing_diagnosis/training_curves.png
```
