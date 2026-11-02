# S54 / RTX 4060 长训入口

在项目根目录运行。SwanLab 已有本机登录记录；默认上传到个人空间的私有 `S54-Locomotion` 项目。SDK 固定为 0.10.3，装在 `.venv-swanlab`，不会改动 Isaac Lab 的 Torch / CUDA / Newton 依赖。初测使用的旧版 0.7.11 查询协议与现服务端不兼容，因此最终脚本升级到新版接口。

## 开始训练

```bash
cd /mnt/sage_data/workspace/KUAVO_S54_Newton
bash scripts/train_4060.sh --num-envs auto --max-hours 12 --max-iterations 20000
```

第一次依次运行短 PPO 基准，读取实际显存、功耗和吞吐；显存预算为 GPU 总量的 85%，同时至少保留 768 MiB。根据前两档实测增长量预估下一档，预计超预算就跳过。候选中选择吞吐不低于最快档 90% 的最大并行数。训练阶段默认 headless。

本机独立短测：2048 环境峰值 4146 MiB / 67.7 W / 42672 samples/s；3072 环境 6054 MiB / 71.21 W / 45869 samples/s；3456 环境 6838 MiB / 72.11 W / 46735 samples/s。4096 环境在 CUDA graph launch 时 OOM，已结束并释放资源。完整数据保存在旁边的 `benchmark*/benchmark.json`。这些值含其他进程的 GPU 显存，会随温度和系统负载变化。

已知本机环境可跳过调优：

```bash
bash scripts/train_4060.sh --num-envs 3456 --max-hours 12 --max-iterations 20000
```

后台运行：

```bash
mkdir -p logs
nohup bash scripts/train_4060.sh --num-envs auto --max-hours 12 \
  --max-iterations 20000 > logs/train_4060.log 2>&1 &
echo "supervisor PID: $!"
tail -f logs/train_4060.log
```

前台按 Ctrl+C；后台对上述 supervisor PID 执行 `kill -TERM PID`。脚本通知当前训练保存 `model_interrupted.pt`，刷新曲线并退出；不要用 `kill -9` 正常停止。

## 恢复与课程

启动后会打印实际 `logs/long_runs/...` 目录和 SwanLab 链接。填入该目录恢复，继续同一个云端实验：

```bash
bash scripts/train_4060.sh --resume-dir logs/long_runs/实际运行目录 \
  --max-hours 12 --max-iterations 40000
```

`max-hours` 是本次启动的预算，包含调优和评估；`max-iterations` 是整条课程累计更新数的上限，恢复时如需延长须提高该值。每个 chunk 默认 100 次更新，保留 optimizer 和 observation normalization；跨阶段只迁移 actor，重置 critic 和 optimizer。仿真状态在 chunk 之间重置。

课程固定为 `stand → lift → step → walk`，每个 chunk 结束使用确定性 policy 独立评估 64 环境 × 1000 步（20 仿真秒）。连续两个不同训练 checkpoint 通过才晋级：

| 检查 | 门槛 |
|---|---|
| 所有阶段 | 无跌倒环境比例 ≥95%，平均 upright ≥0.95，最低有效 root height ≥0.85m |
| 所有阶段 | 平均绝对侧向速度 ≤0.10m/s，yaw rate ≤0.15rad/s |
| stand | 平均前向漂移绝对值 ≤0.05m/s |
| lift / step / walk | 左右脚成功半周期比例均 ≥80%，连续成功 8 个交替半周期且无跌倒的环境比例 ≥80%，摆动末期 XY 误差 ≤3cm |
| step / walk | 平均前向速度分别 ≥0.02 / 0.04m/s |

成功半周期必须包含有效抬脚、晚期触地到达目标及摆动期间支撑脚接触。单纯站立、双脚同时腾空、跌倒时的向前速度都不能通过行走门槛。全部门槛通过，或时间 / 更新预算耗尽，就保存并停止；未通过的阶段持续训练，绝不按 reward 自动宣称学会行走。

评估当前从相同默认姿态重置，尚无随机推力或地形测试；通过门槛仍需 `play.py` 可视化复核。现有任务只训练固定前向足印，尚未实现速度命令、停止和转向。

## 记录与参数

- 终端默认每 10 次更新打印每步 / 每回合 reward、episode 步数、每步跌倒 / timeout 比例、动作 std 和吞吐；`--print-every 1` 可每次更新打印。启动时会打印详细 `.log` 路径，可用 `tail -f` 查看原始 PPO 输出。此修改在下次启动 / 恢复时生效，当前进程继续写原有日志。
- 云端：各阶段 `Reward/*`、`Train/*`、`Loss/*`、动作 std、terminated / truncated 比例、吞吐，以及 `Hardware/*` 的功耗 / 显存 / 利用率 / 温度和 `Eval/*` 的独立评估指标。
- 本地：`state.json` 保存阶段、累计更新、训练样本数、最新及通过门槛的 checkpoint 指针；每段保存 TensorBoard、`metrics.jsonl`、完整 task / PPO 配置、版本、源码 SHA256、训练 / GPU 日志和评估 `.gate.json`。
- 当前课程 LR=1e-4 fixed、std 初始化 0.15、rollout=64、4 epochs / 4 mini-batches、action smoothing=0.2；3456 环境对应 221184 samples/update。
- 本次没有修改奖励、PD、动作 scale 或机器人。继续使用预检中的逐关节 scale：腿 yaw/roll/pitch/knee/ankle-pitch/ankle-roll = `[0.1, 0.15, 0.3, 0.6, 0.3, 0.1]` rad，腰 / 手臂 0.05 rad，target 为默认姿态加缩放动作并受软限位约束。
- 若换机器需要登录，运行 `.venv-swanlab/bin/swanlab login`，在终端输入 API Key；也支持 `SWANLAB_API_KEY`。不要把密钥写进脚本。`--mode offline` 明确只记录本地，之后可使用 `swanlab sync` 上传。

## 复验命令

```bash
# 只做实际 PPO 资源基准，不启动长训 / 上传。
bash scripts/train_4060.sh --benchmark-only --num-envs auto \
  --output-dir reports/4060_recheck

# 测试监督器、门槛、TensorBoard/JSON 指标流。
PYTHONPATH="$PWD" /mnt/sage_data/workspace/IsaacLab_3_0/.venv/bin/python tests/unit/test_train_4060.py
```

SwanLab 官方参考：[初始化](https://docs.swanlab.cn/en/api/py-init.html)、[认证与环境变量](https://docs.swanlab.cn/en/api/environment-variable.html)、[离线日志上传](https://docs.swanlab.cn/en/api/cli-swanlab-sync.html)。
