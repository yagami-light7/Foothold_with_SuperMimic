# S54：10000 次训练后的启动翻倒与头部偏转诊断

日期：2026-10-28。复用现有 `supermimic_s54` Docker 容器，保持 Isaac Lab 3.0 Kit-less、Newton/MJWarp、PyTorch World Model/BPTT。
完成隔离验证后，已按用户新指令从原始 10000 次 Checkpoint 启动续训，目标 15000 次（追加 5000 次）。新日志目录为 `logs/s54_motion_15k_fixed`。原实验、原始 10000 次 Checkpoint、Leju-GMR 和 G1 数据均未修改。

## 已确认的原因

### aiming：首帧 IK 未收敛，导出了求解器启动瞬态

旧 `aiming1_subject1` 开头两个 30 Hz 根部位置相差约 2.27 m，产生 **68.034 m/s** 的初始化速度和 **31.679 rad/s** 的初始化角速度。
MotionLib 对轨迹做差分，环境按 Reference 初始状态 reset，导致机器人开场被赋予异常速度。
记录中的初始根部和关节与 Reference 完全相同，未发现关节映射或渲染置换。
实际在 **0.12 s** 触发倾斜终止：倾斜误差 1.1305，阈值 0.8；高度误差只有 0.0717 m，阈值 0.25 m。
截图中的 `t=1.10s [terminated]` 是 Reference 时间继续推进、Policy 保留真实终止状态，并非存活了 1.10 秒。

初次旧数据审计共 39 段，22 段未通过新的根部启动检查。终端中断后重新清点：旧目录已有 43 段，其中仍有 22 段根部检查失败，详见 [最新导出审计](s54_retarget_inventory_20261028.md)。初次名单与数值见
[`reference_root_audit.json`](../results/s54/training_failure_10k/reference_root_audit.json)。
这些是运动连续性检查，不能替代动态可执行性评估。

### 头部：可控关节没有对应的姿态跟踪奖励

S54 有 29 路可控关节，最后两路是 `zhead_1_joint` 和 `zhead_2_joint`，Reference 为中立角 0。
14 个跟踪 body 不含头部，原来的 body/root 奖励不会直接惩罚头部偏转。
动作变化惩罚只限制动作变化；软限位惩罚只约束接近关节边界的行为，都不能让固定歪头回到中立角。
数值测试确认：在软限位内，原奖励对头部角度的梯度为零。

旧 dance 完整轨迹中，头部偏航中位数 **−50.51°**、俯仰 **27.16°**，Reference 两路均为 0°。
它是实际关节姿态，非渲染颜色或坐标问题。详见
[`summary.json`](../results/s54/training_failure_10k/summary.json) 与
[`dance 诊断图`](../results/s54/training_failure_10k/dance1_subject1_diagnosis.png)。

## 实际代码修复

1. 在导出第 0 帧之前，反复求解同一人体首帧，使根部、腿和腰先收敛；最多 300 次，不改时间轴。
   连续两次根部位移变化小于 1e−5 m、姿态变化小于 1e−4 rad、腿腰关节变化小于 1e−4 rad 才可继续。
   上肢随后由独立 IK 重建，故上肢零空间漂移不作为下肢收敛条件。未收敛则拒绝导出。
2. 导出与训练加载均检查根部速度、角速度和启动脉冲，拒绝旧的启动瞬移；已有自碰撞证书不能绕过此检查。
   非有限值、非单位四元数和不足两帧也会被拒绝。
3. 处理 Mink 数值舍入导致的最大 1e−6 rad 以内边界越界，记录实际修正量；更大的越界仍拒绝。
   本次完整 aiming 的旧失败候选仅超过边界约 1.58e−7 rad。未放宽机器人物理限位。
4. 增加头部奖励：`Δr = −dt × (0.5 × Σ(q_head−q_ref)² + 0.005 × Σ(qd_head−qd_ref)²)`。
   dt=0.02 s。测试确认角度梯度指向 Reference，并抑制额外头部转速；中立头姿无额外惩罚。
5. 新训练保存完整奖励配置。历史实验评估保留原奖励定义，避免修改指标口径。
   Evaluation 默认读取运行目录中保存的 Reference 和 clips，也支持显式覆盖。
6. 生成、训练和预览默认目录统一为 `data/retarget_data/reference_startup_fixed`。
   新 aiming 完整重新生成；dance CSV 原样复用并独立通过新根部检查。旧实验数据保持原样。

## 实际验证结果

| 阶段 | 状态 | 结果与范围 |
| --- | --- | --- |
| 原始故障复现与定位 | PASS | aiming 0.12 s 倾斜终止；dance 头部奖励遗漏由梯度测试复现 |
| aiming 完整 Reference 导出 | PASS | 7184 个 30 Hz 源帧、11972 个 50 Hz 控制帧；239.42 s |
| aiming 首帧连续性 | PASS | IK 首帧收敛 80 次；初始根部速度降至 0.000926 m/s；整段最大间隔速度约 2.485 m/s |
| 足底/上肢/关节速度/连续自碰撞 | PASS | 完整 aiming 全部证书通过；50 Hz 最小足底间隙约 3.010 mm，自碰撞间隙约 3.110 mm |
| 其他 clip 启动粗测 | PASS | dance1、aiming2_subject2、obstacles5_subject2 各 31 帧基线；分别 121/61/24 次收敛。非完整 clip 认证 |
| 奖励与启动回归 | PASS | 15 项测试，包含旧头部奖励盲区与新梯度方向 |
| 启动器回归 | PASS | 19 项测试，包含 clips 选择和隔离、续训命令、同一云端实验 ID 和头部指标兼容 |
| 小配置 30 次 GPU Smoke：数值 | PASS | 16 环境，更新到 10030，保存正常且参数有限；不能代替完整性能检查 |
| 小配置 30 次 GPU Smoke：完整评估 | FAIL | aiming 21.34 s、dance 103.10 s 高度终止；已拒绝作为正式起点 |
| 原配置 30 次 GPU Smoke | PASS | 2048 环境、完整两段数据、原 batch/horizon；新奖励下两段完整存活；指标与权重均有限 |
| 原始 10000 权重 + 修复后的完整两段 Reference | PASS | 无失败；aiming 回报 0.8337，dance 0.7402；平均 0.786925，按帧加权 0.800537 |
| 头部奖励与改善方向 | PASS | 原配置 30 次后，dance 双关节 RMS 误差约 40.6°→24.0°，偏航中位数 −50.6°→−22.9°；俯仰约 27.2°→25.1°，仍需继续学习回正 |
| 正式续训 | RUNNING | 原始 10000 次权重 → 15000；独立 tmux 与日志；原 SwanLab 实验 1ohsjt5p |

原权重完整评估使用历史奖励，结果可与原实验比较。未修改 Policy 参数，只替换有问题的 Reference。
两段均完整存活：aiming 239.42 s、dance 131.46 s。原实验 aiming 约 −0.0007，原两段平均 0.369755。
完整新结果见 [`ckpt_010000_eval.json`](../logs/s54_startup_full_diagnostic/ckpt_010000_eval.json)，
实际轨迹见 [`traj_10000_fixed_refs.npz`](../logs/s54_startup_full_diagnostic/traj_10000_fixed_refs.npz)。

30 次 Smoke 为 16 个环境、WM batch 64×1、Policy batch 32、horizon 8、BF16；
使用完整 dance 与 aiming 前 301 个源帧（10 s）作训练验证，不代表正式训练超参数或收敛结果。
这个小配置通过数值检查但完整性能失败，正式续训不使用它的权重。数值结果见 [`检查日志`](../results/s54/training_failure_10k/startup_head_smoke_check.log) 与
[`GPU 日志`](../results/s54/training_failure_10k/startup_head_gpu_smoke.log)。

原配置 30 次 Smoke 独立保存在 `logs/s54_startup_head_config_smoke`，两段均无失败。
新奖励平均回报 0.753346，扣除新增头部惩罚后为 0.786526（aiming 0.8337、dance 0.7393），与原权重修复数据后的身体跟踪相近。
新增头部惩罚改变了奖励数值，不能直接把 0.753346 与旧奖励 0.786925 当作性能下降比较。
结果见 [完整评估 JSON](../logs/s54_startup_head_config_smoke/ckpt_010030_eval.json)。
正式续训仍从原始 10000 次 `last.pt` 开始，隔离测试更新不计入用户的 5000 次目标。
训练 CSV 与 SwanLab 新增 `Head/RMSErrorDeg` 和 `Head/NormalizedPenalty`；Evaluation 导出同一口径下扣除头部成本的对照回报及头部 RMS。

直接清零旧 Actor 的两路头部输出曾单独测试，dance 回报降至 **0.02588** 并失败，已拒绝采用。
即使其他网络参数不变，头部回传观测也会改变身体策略行为，不能用强制清零替代学习。
该测试产物隔离在 `results/s54/training_failure_10k/rejected_head_reset/`。

## 修改文件

| 文件 | 用途 |
| --- | --- |
| `supermimic/reference_quality.py` | 根部连续性和启动脉冲检查 |
| `supermimic/motion.py` | 训练/评估加载时强制执行检查 |
| `supermimic/mdp.py` | 头部角度与角速度奖励 |
| `supermimic/train.py` | 保存奖励配置、使用新数据默认路径 |
| `supermimic/evaluate.py` | 按保存的 Reference、clips 和奖励配置评估 |
| `tools/generate_s54_reference.py` | IK 首帧收敛、数值边界处理、新输出目录 |
| `tools/prepare_s54_motion.py` | 发布前检查并记录根部质量 |
| `tools/s54_reference_publish.py` | 已有数据也须通过新检查 |
| `tools/diagnose_s54_training.py` | 真实轨迹终止与头部诊断工具 |
| `tools/audit_s54_retarget.py` | 中断后的导出清点与完整 CSV 30/50 Hz 速度审计 |
| `tools/sync_swanlab.py`、`tools/test_s54_launcher.py` | 显式续接云端 ID，头部指标兼容与回归 |
| `tools/test_s54_startup_head.py` | 15 项必要回归 |
| `tools/view_s54_reference.py`、`tools/preview_s54_reference.py`、`tools/compare_s54_reference.py` | 默认查看新数据目录 |
| `run_train.sh` | 默认训练新数据目录；其他正式训练超参数保留 |
| `README_S54.md`、`docs/S54_LAUNCHER.md`、`data/retarget_data/README.md` | 数据变更、用法和验证范围说明 |
| `reports/s54_startup_head_fix.md` | 本报告 |

原始 `logs/s54_motion/last.pt` SHA-256：
`d0009075f28258504bb6992007bd5e340a873035d6f67f7a90535742bbf4784e`。
原编号 10000 权重与完整评估副本 SHA-256 均为
`e3d2e32bbf0aa6b3827357f107d28d7d8c492fdf54974ca5b8489c32228336e7`。
新目录 dance CSV 与旧文件 SHA-256 均为
`e0a8746693aecce793cd231c0494a5b3b6d1d89c35c46df5a29f8f3327c9ec7f`。

## 验证与后续命令

```bash
# 回归测试。
OPENBLAS_NUM_THREADS=1 /data/conda_envs/leju_gmr/bin/python tools/test_s54_startup_head.py
python3 tools/test_s54_launcher.py

# 查看新目录中现有的完整 dance1。
MUJOCO_GL=glfw OPENBLAS_NUM_THREADS=1 /data/conda_envs/leju_gmr/bin/python \
  tools/view_s54_reference.py --clip dance1_subject1

# 为其他动作生成首帧收敛、足底和自碰撞检查齐全的完整 Reference。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /data/conda_envs/leju_gmr/bin/python \
  tools/generate_s54_reference.py --all --skip_existing --continue_on_error

# 已启动的正式续训命令（勿重复执行）。
SWANLAB_RUN_ID=1ohsjt5p SWANLAB_RESUME=must bash run_train.sh \
  resume s54_motion_15k_fixed logs/s54_motion/last.pt \
  --iterations 15000 --clips dance1_subject1 aiming1_subject1 \
  --swanlab-project SuperMimic --swanlab-name s54_motion
```

`--iterations 15000` 表示从 10000 追加 5000 次。该命令已按用户授权启动，无需重复执行。
正式微调应继续使用原始 `last.pt`，保留 Critic 和三个优化器；30 次测试权重与被拒绝的头部重置权重均不作为正式起点。
新的头部惩罚会改变奖励口径，云端同一实验的续接点应注明奖励和数据修复，避免把前后训练 reward 直接当作同一指标比较。

本次 aiming 的 10 秒真实仿真对比视频：[MP4](../results/s54/training_failure_10k/aiming_10000_fixed_reference.mp4)，640×360、25 FPS、250 帧。

## 正式运行观察与日志说明

运行在 Docker 的 `s54_motion_15k_fixed` tmux 会话中，原训练参数逐项与旧 `args.json` 核对一致。
SwanLab 已显式以 `id=1ohsjt5p, resume=must` 续接旧实验，并在线记录 10100 之后的数据。
同步进程使用独立进程会话；`nohup` 单独使用可能随启动进程组的清理而退出，启动器已改为 `nohup setsid`。
运行参数与实际状态见 [运行核对 JSON](../results/s54/retarget_inventory/running_verification.json)。

本次新加的终端 `head` 字段最初用了错误的列表下标，显示的是归一化惩罚；
**CSV 的 `head_rms_deg`、`head_penalty_norm` 与云端两项指标始终正确**，不影响奖励、梯度或训练。
打印已改为使用明确变量；当前已启动的 Python 进程保留旧格式，故本轮查看头部误差请以 CSV／云端为准。
无需为这个显示问题打断已运行的训练。
