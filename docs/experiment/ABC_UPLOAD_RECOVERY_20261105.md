# ABC 上传故障与恢复记录（2026-11-05）

批次：`s54_abc_short_20261105_184038`。A 组训练和独立 Newton 评估均已完成；本次异常发生在上传等待环节，不是训练数值溢出。B/C 当时尚未启动。

## 原因与修复

- 旧 tmux 服务继承 `127.0.0.1:17897` 代理，端口不可用；当前终端为 `7897`。现有 SwanLab 登录验证通过，没有更换 Token。
- 原批次管理器等待上传 120 秒后直接抛异常，阻塞 B/C。现在先持久化 Eval 上传包，由独立 CPU/tmux 队列串行补传训练缓存、评估数据；失败保留数据并退避重试，不占用 GPU 训练调度。
- 新 tmux 会话显式继承调用终端的代理变量，包含已清除的变量，避免继续使用旧服务环境。本次 A 补传采用已验证可用的直连。
- 认证失败留下的 SDK 空目录没有 `run-*.swanlab` 记录文件。保留这些目录供诊断，同步时跳过；有记录文件的缓存仍需成功同步。
- 新增批次 `resume`：检查已完成组的参数、学习代码、真实样本数、参数更新和评估协议，复用完整结果。未完成的训练不会被静默重跑或作为等样本结果接受。

## 实际结果

A 保存的最终状态：2000 iterations、5,251,072 个真实训练 transitions、1000 次 Actor 更新；Decoder 与 WM-only 阶段相同，Foot Encoder 实际发生更新。原始 `last.pt`、`wm_calibrated.pt`、`train_log.csv`、`evaluation/summary.json` 的 SHA256 在恢复前后完全一致。

| A 独立评估窗口 | 物理终止 / 3 | Timeout / 3 | 平均实际执行 | Normalized Return 均值 |
|---|---:|---:|---:|---:|
| 2 s | 2 | 1 | 1.913 s | 0.66075 |
| 5 s | 3 | 0 | 1.920 s | 0.26454 |
| 10 s | 3 | 0 | 1.920 s | 0.13227 |

高度越界是真实失败；2 秒 Timeout 也不能认定为稳定行走。WM8 的约 1 mm 高度误差只反映对这些短轨迹的预测精度，不能替代 Policy 能力验收。

A 的训练数据及 61 个 Eval 标量已补传到原实验 ID `d9595555796a417b`，云端读回验证通过，原训练曲线未变：
[A_20261105_184038_seed54_walking](https://swanlab.cn/@yagami_light/Foothold_with_SuperMimic/runs/d9595555796a417b/chart)。
证据：A 目录的 `evaluation/upload_status.json`、`cloud_eval_readback.json`、`cloud_training_before.json`。此前失败状态保存在批次 `recovery/`，原日志继续保留。

已实际执行恢复命令，A 训练/评估均跳过；B 于约 20:00 恢复启动，之后将串行执行 C。每组仍为 2000 iterations，前 1000 次 WM-only，无训练时间截止。未修改算法、训练超参数、Reward 或终止条件。

## 命令与验证

```bash
# 已执行；当前批次正在运行，无需重复执行。
bash tools/train/run_s54_abc_short.sh resume --batch s54_abc_short_20261105_184038

# 查看实时输出 / 状态。
tmux attach -t s54_abc_short_20261105_184038
bash tools/train/run_s54_abc_short.sh status --batch s54_abc_short_20261105_184038
tail -F logs/s54_abc_short_20261105_184038/batch.log

# 安全停止当前组并禁止启动下一组。
bash tools/train/run_s54_abc_short.sh stop --batch s54_abc_short_20261105_184038
```

tmux 中 `Ctrl-b` 后按 `d` 脱离，SSH 断开不影响训练。批次状态中的 `completed_groups` 指本地训练与评估完成，`uploads` 单独显示云端验证状态。恢复命令只复用已完整训练的组；安全停止后的部分训练不属于此恢复模式。

验证命令：`/home/rm/workspace/IsaacLab_3_0/.venv/bin/python -m unittest tools.tests.test_abc_short tools.tests.test_pre_ablation`；24 个测试通过，包括云端不可用不阻塞组间推进、已完成组不重复执行、空 SDK 目录处理和上传顺序约束。Shell 语法检查通过。实际 A 云端补传及 B 真实 Newton 训练启动均已验证。
