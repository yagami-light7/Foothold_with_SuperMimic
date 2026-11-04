# S54 Critic 发散诊断与恢复（2026-11-04）

> 当前执行方案已改为：从零开始、WM-only 1000 iterations、4096环境大并行。直接运行 `bash tools/run_s54_curriculum_8h.sh fresh`，完整配置及实测见 [S54_FRESH_3090.md](S54_FRESH_3090.md)。下文的迁移/恢复方案保留为历史记录。

原实验：`logs/s54_curriculum_20261104_014110`。已在 iteration **1852** 用 SIGTERM 安全保存退出，exit code 0；不是完成训练或站立成功。停止时累计约 **1071 秒**，仍在 C0。原实验与 SwanLab 历史保留。不要直接用原 `latest.pt` 当作健康策略恢复。

## 证据

| Iteration | Critic MSE | Actor loss | 实际仿真归一化奖励 |
|---:|---:|---:|---:|
| 1400 | .0819 | -3.126 | .569 |
| 1580 | 607.623 | -43.954 | -1.338 |
| 1640 | 44,311 | -420.212 | -3.299 |
| 1840 | 428,028 | -2,076.929 | -9.141 |

第一次独立 Newton Fast Evaluation：Standing success=0%、physical termination=100%、foot RMSE=.677m，不能视为正常波动。浮点数尚未 NaN，并不意味着训练稳定。

Actor objective 为短期 imagined return 加 discounted terminal Critic bootstrap。`pi_loss` 越负不等于实际策略越好；此处 negative imagined reward 与巨大的 positive objective 同时出现，说明 bootstrap 正在主导。梯度裁剪只能限制更新幅度，不能纠正偏离的价值目标。

CPU checkpoint 探针（非真实 Newton 验证）发现：

- 每步正奖励上界为 .08，gamma=.99，合法 discounted return 保守上界为 **8**。
- Iter1000：nominal-grounded probes 的初始 V 均值4.66，8步后4.82；动作幅度约1.1。
- Iter1580：初始 V 均值25.97，7步后106.22；第7步动作幅度47.83，约44%分量超出真实动作范围。第8步 state 的 V 均值119.43。
- 真实采样原本执行 `clamp(action+noise,-20,20)`，而 `_imag_step` 原本把未限制的 Actor 输出直接交给 WM。独立 foot evaluation 原本也缺少该动作限制。

证据：`results/s54/value_growth_probe.json`。探针不是同分布 Replay/真实 Newton 配对验证，因此不能证明所有发散只来自这个问题；但动作契约不一致、超过理论上界的 Critic 高估已直接复现。

## 修复范围

1. 新增 `supermimic/control.py`，采样、BPTT imagination、两种 foot evaluation 共用 `bound_action`，仍是原来的 ±20。没有修改实际 PD 接口、关节顺序或物理模型。
2. 仅 foot task 对 Actor terminal bootstrap 与 Critic GAE 的 bootstrap value 使用可证明的上界 `.08/(1-.99)=8`；raw Critic 仍原样拟合到目标，MSE 和原始估值仍记录。没有把 reward、终止条件、成功率或报告中的 loss 截断来制造改善。Motion 的 value objective 保留原样。
3. 新增 raw value mean/max、target mean/max、bootstrap contribution、reward contribution、动作饱和比例诊断；随 JSONL 上传 SwanLab。
4. 可独立设置 `--actor_lr`、`--critic_lr`，默认仍继承 `--lr`；可设置 resume 后的 `--recovery_wm_iters`。
5. checkpoint 记录 imagination contract，网络维度与物理任务契约不变；旧 checkpoint 会明确提示该训练约束升级。

上界约束切断已被实证的不合理正 bootstrap 激励，但不能保证长期收敛。动作 clamp 在饱和区梯度为0，因此严重发散的 Actor 不适合直接继续训练。

## 显式恢复方案

已将 iter1000、1575、1580 的完整 checkpoint 用独立 hard link 保留到原运行 `diagnostics/`，不受最近3份清理影响。`best_standing.pt` 的0%成功候选不代表健康策略。

`tools/prepare_s54_recovery.py` 生成：

`logs/s54_curriculum_20261104_014110/diagnostics/recovery_from1000.pt`

采用 iter1000 的 WM/Actor 及其 normalization，重置 Critic 及所有 optimizer；保留停止时 iteration1852、已用wall time、curriculum、RNG和全局采样进度。iter1000只是更早的未发散候选，**同样没有通过长期站立**。这是一份明确标记的恢复迁移，不是精确恢复旧优化器的断点。

短测运行：`logs/s54_curriculum_value_guard_smoke_20261104`。从恢复文件开始，总 iteration1924，即72次更新；前20次WM-only，随后52次Actor更新。num_envs512、WM batch256×4、Actor batch64、Horizon8、WM LR5e-5、Actor/Critic LR1e-5。仍串行使用单个 GPU simulator，SwanLab 使用 Foothold_with_SuperMimic 独立时间戳实验。此短测不是正式长训练，也不是站立收敛证明。

继续训练时应使用**经检查后的短测 latest.pt**，而非原发散 latest，并明确设置新的学习率/WM更新频率。累计8h计时从checkpoint继承，不清零：

```bash
RUN="s54_curriculum_recovery_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
bash tools/run_s54_curriculum_8h.sh start "$RUN" \
  --resume logs/s54_curriculum_value_guard_smoke_20261104/latest.pt \
  --actor_lr 0.00001 --critic_lr 0.00001 \
  --wm_batches 4 --recovery_wm_iters 100
```

本次不会自动执行上面的长训练命令。

## 验证

新增 control guards 测试直接执行 train.py 的 `_imag_step`，验证 WM 接收到与真实采样一致的受限动作；验证上界数值、梯度、负奖励无法被无穷正 bootstrap 掩盖。日志、sampler、Replay、evaluation等共29项执行通过（含复用测试类重复项）；原 foot MDP 7项通过；编译及 diff whitespace 检查通过。

真实 Newton 短测结果及剩余风险将在同文件末尾补充；退出状态以短测目录的 `train.exitcode` 为准。

### 已完成的短测结果

- 正常到达 iteration1924，exit code 0；WM/Actor/Critic optimizer step 分别为288/52/208。WM/Actor参数（排除normalization计数）L2变化1.6583/.9354，确认实际更新。
- 记录到的 Actor 更新日志：WM loss .0129–.0192，Critic原始MSE .0674–.0919，Actor loss -.4596至-.8104，Actor原始梯度范数1.90–4.30。末段raw V max约.747、target max约1.121、动作超界比例0。没有隐藏raw value或裁剪日志loss。
- 这只有52次Actor更新，不能证明后续几百/几千次迭代的稳定性；同时进行了回退/重置和学习率调整，无法将短期改善全部归功于某一项修复。
- 末次真实Newton Standing Full评估仍为成功率0%、foot RMSE=.1106m，仍不得晋级C1。
- 初始化期间Isaac远程地面资源服务器曾超时，最终按Isaac现有逻辑使用已缓存的同一资源。未改地面物理或资源文件。启动/短测/保存共耗约275s，计入继承的累计预算；当前累计约1347s，剩余约7小时37分。
- 新完整checkpoint：`logs/s54_curriculum_value_guard_smoke_20261104/latest.pt`。短测与原长训练均已退出，不存在偷偷继续的长训练。
- SwanLab记录及最终同步完成（exit0）：[修复后短测](https://swanlab.cn/@yagami_light/Foothold_with_SuperMimic/runs/5619eacce3e7489b)。证据 `results/s54/value_guard_smoke.json` 和短测目录的 `eval_final.json`。

如果执行上面的继续命令，应保留这份模型/优化器/课程/计时状态并继续观察raw V、TD target、bootstrap贡献和真实存活时长；不能把loss暂时变小当作站立学会。建议先以有限迭代范围复查，再决定是否使用全部剩余预算。
