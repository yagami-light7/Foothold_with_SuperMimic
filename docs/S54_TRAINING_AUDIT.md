# S54 训练稳定性审查与重启记录（2026-11-04）

本次按照“先监测，确认问题后才停训”的要求执行。旧实验 `s54_curriculum_fresh_20261104_024404` 已在 iteration **2359** 安全停止，exit code **0**，完整 checkpoint 和原日志保留。修复及短测通过后，已启动新实验 **`s54_curriculum_fresh_auditfix_20261104_032902`**。这不是断点恢复；WM、Actor、Critic、优化器、normalization、Replay 和课程均从零开始。

## 结论与证据

- 旧实验先连续观察 **300 秒**（iteration1820–2200）：无 NaN/Inf，Critic MSE 约 .07–.13。用户指出的 iteration1100–1200 的上升随后回落，不能单凭这段 Loss 判定数值爆炸。记录：`results/s54/audit_monitor_20261104.jsonl`。
- 确认 **Replay 重置边界缺陷**：评估结束会重置训练环境，但旧 WM 窗口只检查 `done`。`ep_step=[10,11,0,1]`、没有终止行时，旧 mask 全为1，WM 会把跨 Episode 的重置当作动力学。现在同时要求整数 tick 连续，mask 为 `[1,1,0,0]`，不伪造物理终止。
- 确认 **评估效率和报告问题**：只评估8个环境，却逐帧在 CPU 处理4096个环境的接触对。改为在 GPU 筛选目标环境的真实地面接触，归属和力方向保留。旧首次评估耗45.47秒仍8/8未完成；修复后的同规模评估约7.2秒得到完整物理终止结果。不完整评估现在标记 unknown（`success_rate=null`），不参与晋级或 best 选择。
- 确认 **Critic 高估**：停止策略的8个相同 Newton 起点，raw V=7.8125–8.1875，实际跌倒前折扣回报3.4987–3.7533，平均高估 **4.3490**。这组策略仍约1.8秒物理终止，Standing success=0%，不能称为已学会站立。证据：`results/s54/audit_critic_newton_return.json`。

## Rollout、网络、梯度和动作审查

| 项目 | 核对结果 |
|---|---|
| 真实采样 | UTD=.4，每 iteration 交替2/3个向量控制步；4096环境平均10240 transitions/iteration |
| imagined rollout | Horizon=8，即0.16秒；与真实采样长度和 WM window 分开，不为占显存而拉长 |
| 观测/目标/动作 | 99 / 230 / 29；目标10帧，每帧23维；hist=2 |
| WM | 三层2048，SiLU+LayerNorm；8,822,854参数 |
| Actor | Goal encoder 与 decoder 各2048/1024/512/256，latent32；6,470,461参数 |
| Critic | 输入365，隐藏256/128，标量输出；126,721参数 |
| RunningNorm | 状态随checkpoint保存；未发现维度错位或梯度意外穿过统计更新 |
| 梯度路径 | 冻结 WM/Critic 参数仍保留对 Actor 输入的梯度；CPU float64 单个关键输出bias分量的有限差分相对误差约1e-8，无冻结参数梯度泄漏 |
| 动作裁剪 | 真实采样、imagined step、foot评估共用原±20边界；被审查checkpoint未饱和；无新增PD接口或关节顺序变动 |

梯度探针显示，旧策略的 bootstrap 梯度范数约21–24，直接奖励梯度约.22–.25，纯足部奖励梯度约.027–.030。它说明优化主要受价值项影响，**本身不是梯度计算错误的证明**；与真实回报高估一起构成校准 Critic 的依据。有限差分验证的是 learned WM 计算图，不能验证 WM 对 Newton 的梯度准确性。证据：`results/s54/gradient_path_audit.json`。

## 最小修改

1. Replay WM 窗口检查 `done` 与 `ep_step` 连续性。
2. 评估在GPU筛选接触；区分完整、未完成、物理终止、Timeout和目标失败；依据实测耗时积累下一次评估额度，仍受5%预算限制。
3. Critic 保留原 imagined GAE，增加可关闭的真实 Replay 回报校准：`critic_real_batch=256`、`window=512`、`recent_steps=512`、`weight=.25`。只接受近期连续片段中确实到达**物理终止**的折扣回报；Timeout、显式reset和Replay末尾的未完成片段不赋零尾值。无有效样本时跳过。
4. Critic loss 为 `imagined_MSE + .25 * real_terminal_MC_MSE`。CSV `v_loss` 保持原 imagined MSE；JSONL/SwanLab另报 `Optimization/critic_real_mc_loss` 和 `critic_real_samples`。raw V、target、Loss都不隐藏或伪造。
5. 每20次iteration记录奖励/价值/纯足部梯度、价值估计和动作饱和率。真实评估另记初始V与实际终止回报偏差。

没有更改物理模型、关节顺序、PD、奖励权重、物理终止阈值或成功标准；网络结构、BPTT、imagined GAE、Horizon=8保留。校准项改变了 Critic 的训练目标，已明确配置；它是近期行为策略的回报，存在离策略偏差，不能视为无偏的当前策略价值。没有把WM Loss一次上升设置为停训条件。

## 验证与资源

- CPU相关回归58项执行通过（含复用测试类重复执行），原 Standing diagnostics另7项通过；新增测试覆盖跨reset窗口、ring wrap、真实终止回报、Timeout/未完成片段删失、接触筛选。Python编译、shell语法、diff空白检查通过。
- 实际 GPU Smoke：`logs/s54_auditfix_real_mc_smoke_20261104`，**4096环境**、Replay3072、WM batch4096、Actor batch1024；仅短测把WM-only改4、warmup改128、总20iterations，正式配置保持WM-only1000、warmup256。
- Smoke exit0，45.55秒；从WM校准到结束实际Actor更新16次，Critic优化128步；三网权重变化均非零。真实回报样本平均每次148–172个，MC原始MSE约4.55–5.03，无NaN/Inf，动作饱和0。校准初期V接近0，真实MC Loss大于imagined MSE属量纲/目标不同，不能只比较两个Loss绝对数。
- Smoke设备占用约15948MiB（15.6GiB），Torch峰值14004MiB；Actor后段约1.40it/s，约14.3k真实transitions/s。大Replay本身约11.2GiB，不代表计算全部饱和。后续command_store增长和课程复杂度可能增加占用，仍保留21500MiB安全阈值。
- Smoke完整结果：`results/s54/auditfix_real_mc_smoke.json`；SwanLab最终同步exit0，实验ID `17a12cd663db4f2b`。

### 正式实验前15分钟实测

新实验已实际完成1000次WM预训练：`wm_calibrated.pt`中WM optimizer step=2000，Actor/Critic optimizer均为空；记录 `results/s54/auditfix_production_wm_gate.json`。Actor从iteration1001开始更新。

启用Actor后仍有短期波动：Critic imagined MSE日志峰值约6.48，WM loss一度约.49，真实奖励和Episode时长下降；随后到iteration1560，Critic MSE约.063、WM loss约.212、真实归一化奖励约.674。已观察至iteration1600，尚无NaN/Inf；这些Loss下降不等于策略已稳定。设备占用约16.6GiB，Actor段约1.1–1.3it/s。记录：`results/s54/auditfix_production_first_eval.json`。

首次正式Fast Evaluation（`eval_001560.json`）耗时5.70秒，8个固定起点均完成协议判定：**Standing success=0%，physical termination=100%，Timeout=0%，incomplete=0%**；82–86控制步后物理终止（1.64–1.72秒），foot RMSE=.0368m，继续C0。raw初始V=3.656–4.031，真实终止回报3.186–3.362，平均高估 **.5688**。该值小于此前旧策略测试的4.349，但两个策略迭代不同，不是严格配对A/B因果证明；当前站立表现仍未合格，也没有证据证明优于旧策略。

保留当前训练继续观察，独立监督器仍运行至启动后30分钟（约03:59）。不能把修复和校准描述为已经解决全部收敛问题。

## 当前正式实验与操作命令

正式配置：`configs/s54_curriculum_fresh_3090.json`；实际生效参数：`logs/s54_curriculum_fresh_auditfix_20261104_032902/args.json`。WM-only1000；4096环境；WM4096×2；Actor1024；Horizon8；Replay3072；WM/Actor/Critic LR=5e-5/1e-5/1e-5。未载入旧checkpoint。SwanLab项目 `Foothold_with_SuperMimic`，独立北京时间戳实验：[当前实验](https://swanlab.cn/@yagami_light/Foothold_with_SuperMimic/runs/f05d8c13db6b4235)。关键运行源码的SHA256保存在该run的 `source_sha256.json`。

```bash
RUN=s54_curriculum_fresh_auditfix_20261104_032902
# 查看训练、课程与退出状态
bash tools/run_s54_curriculum_8h.sh status "$RUN"
tail -n 20 -F "logs/$RUN/stdout.log"
cat "logs/$RUN/args.json"
cat "logs/$RUN/supervision_status.json"
tail -n 5 "logs/$RUN/swanlab_sync.log"
# 重新连接；Ctrl-b，然后d离开。训练stdout已持久化到文件，pane本身可能为空。
tmux attach -t "$RUN"
# 新窗口实时显示日志
tmux new-window -t "$RUN" -n logs "tail -n 20 -F '$PWD/logs/$RUN/stdout.log'"
# 安全停止；等train.exitcode出现后才可恢复
bash tools/run_s54_curriculum_8h.sh stop "$RUN"
bash tools/run_s54_curriculum_8h.sh resume "$RUN"
```

上面stop/resume为管理命令，不需要现在执行。已有正式实验正在运行时，不要再执行fresh重复启动；另开全新实验的通用命令是 `bash tools/run_s54_curriculum_8h.sh fresh`。

训练、上传、监督器分别运行在 `$RUN`、`${RUN}_upload`、`${RUN}_audit` tmux中，退出SSH不受影响。监督器从启动时连续运行30分钟，发现NaN/Inf，或连续3条Critic MSE>1e6，或连续3条MSE>1000且真实奖励<0且已完成Episode<.5秒，则安全停训。它不因单次WM/Critic峰值停止，不擅自改脚本或循环重启。观察结果写入 `supervision_status.json`，不是声称有人一直在线盯屏。

单次预算28800秒，从约北京时间03:29开始计，预计最晚约11:29退出，初始化、预热、评估与保存均计入；保存预留45秒。SIGTERM安全退出。15分钟checkpoint/评估调度保留，评估共用当前GPU串行执行。周期最近3份、阶段checkpoint、latest及真实评估best均保留；Replay恢复时重填，课程/优化器/RNG/累计时间继续。

课程C0–C5和各阶段样本分布保持 [S54_CURRICULUM_8H.md](S54_CURRICULUM_8H.md) 的定义；当前C0=100% Standing。频率为左右交替落脚总次数/秒，0Hz只表示Standing。未达站立成功标准不得晋级，不能承诺8小时内一定完成所有阶段。

## 尚未解决的风险

真实回报校准的长期效果需观察；只从真实物理终止片段取MC样本，会偏向失败轨迹，可能使价值校准偏悲观，不能仅看MC Loss下降就判成功。现有策略尚未通过长期站立，8小时内收敛不保证。H8短视与learned WM误差仍可能影响策略；原Critic未显式输入Actor历史或a_prev，动作平滑代价却依赖a_prev，因此仍是历史依赖价值的近似，未证明这是本次发散原因；本轮保留接口与网络结构。4096场景即使只评估8个环境仍要推进全部物理环境，已修复接触CPU瓶颈但不是完全独立的小场景评估。高容量Replay保留旧分布，WM1000只延长校准，无法消除Actor分布迁移；后续判断依赖真实Newton成功率、存活时长、MC价值偏差和跟踪误差。
