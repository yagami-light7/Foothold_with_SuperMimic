# S54 Motion Tracking：奖励与 Root 反馈 v2

配置入口是 `configs/s54_m34_4060.json` 的 `motion_reward_config` 和
`motion_obs_mode`。本次保留 `s54_m34_run02/pipeline_config.json` 的全部训练超参数，
只增加这两个配置项。正式训练尚未启动。

| 正奖励 | 权重 | 指数核尺度 |
|---|---:|---:|
| Body Position | 1.0 | 0.30 m |
| Body Orientation | 1.0 | 0.40 rad |
| Body Linear Velocity | 1.0 | 1.0 m/s，沿用原值 |
| Body Angular Velocity | 1.0 | 3.14 rad/s，沿用原值 |
| Root XY Position | 1.0 | 0.20 m |
| Root Orientation | 0.75 | 0.30 rad |
| Foot Position Tracking | 0.50 | 0.08 m |
| Foot Orientation Tracking | 0.25 | 0.20 rad |

位置核采用 `exp(-||error||² / std²)`，姿态核采用 SO(3) 旋转夹角平方。
Root Position 只使用 XY；Root Orientation 仍使用完整三维姿态。
Body 项沿用原来的参考对齐方式；Foot 项使用左右脚 `leg_l6_link`、
`leg_r6_link` 的原始世界坐标参考位姿，不跟随机体平移或航向重对齐。
左右脚分别计算指数核再取平均，支撑期和摆动期都计算。

动作变化、关节限位和头部惩罚沿用原值。所有项乘以控制步长 0.02 s。
正权重和为 6.5，归一化上界为 `max_step_reward = 0.13`。
独立评估的归一化回报为
`sum(actual_reward) / (full_reference_steps * max_step_reward)`；
跌倒或诊断截断仍使用完整参考长度作分母。新旧奖励定义不同，不能直接用归一化回报判断进步。

`root_xy_heading_v2` 在原有 99 维观测末尾增加四维，索引从 0 开始：

| 通道 | 内容 |
|---|---|
| 99–100 | `(H_actual.T @ (p_reference - p_actual))[:2]`，单位 m |
| 101–102 | `sin(yaw_reference-yaw_actual), cos(yaw_reference-yaw_actual)` |

`H_actual` 是实际机体的纯 yaw 旋转矩阵。反馈使用当前参考帧，航向 sin/cos
避免 ±π 处跳变。新增通道不加传感器噪声，保留已有的运行统计归一化。
原有 93–98 通道的 6D 相对姿态继续保留。

训练收集、Replay 重建、WM 想象展开及独立 Newton 评估使用同一观测接口。
Motion Goal 仍为 10×73=730 维，Motion Encoder latent 仍为 32 维；
10 帧历史下 decoder 输入变为 `10*103+32=1062`，输出仍为 29 维。
WM、Critic、真实 Newton 转移标签和 BPTT 更新方式沿用现有实现。

必须新建实验训练 103 维策略。旧 99 维 Actor 不接受直接 resume/init_weights，
原 checkpoint 按其保存的奖励和观测配置继续评估。`RewardCfg()` 和常量 `OBS_DIM=99`
保留旧接口；新 Motion 训练通过 `tracking_v2()` / JSON 配置和 `mdp.obs_dim=103` 选用新接口。

在当前 Docker 环境的仓库根目录启动：

```bash
SWANLAB_ENABLED=true \
SWANLAB_PROJECT=SuperMimic-S54 \
bash tools/run_s54_m34.sh train s54_m34_root_feedback_v2 configs/s54_m34_4060.json
```

启动器使用 tmux，先执行数据/PD 检查和 30 次 smoke，再正式训练。
正式参数沿用上次：2048 envs、10000 iterations、WM-only 1000 次、
前 3000 次提高 Straight 比例、每 250 次保存、每 1000 次独立冻结评估，
seeds `[0,1,2]`，15000 秒墙钟预算。独立评估包含 train 和 dataset_export/test；
seed 0 为标称重置，seeds 1/2 为扰动重置，分别汇报。
测试结果不进入优化、归一化统计、采样调节或 checkpoint 选择。

SwanLab 使用宿主机现有 `.venv-swanlab` 和已缓存登录或 `SWANLAB_API_KEY`。
上传核心训练参数（含奖励配置和观测模式）、训练标量以及每次完整评估的核心标量；
train/test、标称/扰动和各动作的曲线分开。后台进程跟随新结果，保存上传进度，
不上传 GIF、视频或其他媒体。兼容修复避免普通 `SWANLAB_PROJECT` 字符串被
新版 SDK 当作结构化 Settings JSON 解析，项目名称由启动器显式传给 SDK。

验证：CPU 回归覆盖奖励数值、反馈方向/坐标系/航向连续性、103 维历史、
WM→奖励→Actor 的 BPTT 梯度、旧接口和延迟评估上传。
另用独立随机未训练策略，在训练参考 `spin_flat_02` 上执行了 8 步真实 Newton
Physics Rollout，结果在 `logs/s54_root_feedback_v2_smoke/newton_rollout/results.json`。
该运行包括参考姿态和 16 个随机姿态 FK/速度门禁；诊断截断不计作存活成功，
也不构成训练收敛或泛化表现证据。

此外已通过与启动器相同的 30-iteration smoke（16 envs，前 10 次 WM-only），
使用真实 Newton Replay 拟合 WM 并进行 BPTT/Critic 更新。
日志和可恢复 checkpoint 位于 `logs/s54_root_feedback_v2_training_smoke/`；
所有 smoke 日志指标及模型权重有限，Actor 梯度非零。
宿主机已安装的 SwanLab 0.10.4 也通过了离线标量记录检查，
同一 iteration 的训练和随后完成的评估使用不同指标名记录，不包含媒体。
这些 smoke 和离线验证不会上传到正式实验。

续训使用新目录，并指定完整的 `last.pt`，例如：

```bash
SWANLAB_ENABLED=true \
SWANLAB_PROJECT=SuperMimic-S54 \
bash tools/run_s54_m34.sh resume \
  s54_m34_root_feedback_v2_resume02 \
  logs/s54_m34_root_feedback_v2/pipeline_config.json \
  logs/s54_m34_root_feedback_v2/last.pt
```

续训会跳过重复 PD 基线及从零初始化的 smoke，保留数据契约检查、
checkpoint 严格验证、Replay 重填和 20 次 WM 恢复预热；周期 train/test
评估和 SwanLab 数据上传继续运行。训练目标 iterations 是累计总次数。
续训目录没有新 PD 结果，最终可视化会跳过 PD 视频，不读取不存在的 `pd/results.json`。
原始 PD 诊断和失败日志保留在之前的实验目录。

启动器现先调用 `tools/check_s54_cuda.py` 执行实际 GPU 张量运算并同步，
失败时不会创建实验目录、tmux 任务或后台 uploader；直接运行 Python pipeline
也会在 CPU/PD/训练前检测 CUDA。`nvidia-smi` 正常不代表此检测一定通过。
挂起后若检测失败，可先重启容器并重试；仍失败时，重启宿主机后再检测。
重启后的检查命令为：

```bash
docker start supermimic_s54_m34
docker exec -w /workspace/supermimic supermimic_s54_m34 python tools/check_s54_cuda.py
```

2026-11-09 的 `resume01` 失败在 PD 前的 CUDA 初始化，日志包含 Warp error 999；
原训练可加载的最新 checkpoint 为 iteration 1750。宿主机重启、启动容器后，
实际 CUDA 运算检查已 PASS（RTX 4060 Laptop GPU / Torch 2.11.0+cu130）。
18 项 CPU 编排测试覆盖续训跳过 PD/smoke、CUDA 失败提前退出、
正常新训练保留 PD/smoke、续训后的最终渲染及冻结 train/test 评估。
