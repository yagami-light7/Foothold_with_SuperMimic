# S54 Foot Tracking：8 小时课程训练部署报告

> 当前执行方案已改为：从零开始、WM-only 1000 iterations、4096环境大并行。直接运行 `bash tools/train/run_s54_curriculum_8h.sh fresh`，完整配置及实测见 [S54_FRESH_3090.md](S54_FRESH_3090.md)。下文的迁移/恢复方案保留为历史记录。

初次交付为训练基础设施、短测及可执行计划。后续按用户授权进行稳定性审查并重新启动正式实验，当前运行与修复证据见 [S54_TRAINING_AUDIT.md](S54_TRAINING_AUDIT.md)，**尚未证明策略学会长期站立或行走**。2026-11-04，RTX 3090 24 GB。SwanLab 项目组统一为 **`Foothold_with_SuperMimic`**，新实验及默认本地目录按北京时间 `YYYYMMDD_HHMMSS` 分类。

## 1. 审查结论与最小修改

| 模块 | 原实现结论 | 本次处理 |
|---|---|---|
| `foot_trajectory.py` | `FootstepCommand` 可表达任意左右落脚位置、wxyz 旋转、起落时间；quintic + slerp；支持 lengths=0。`sample_alternating` 只沿固定 heading 平移，yaw_offset 不改变路径方向 | 保留 query，增加可选只读时间段标签；新 sampler 累积 SE(2) 中心位姿生成持续转向/曲线 |
| `goals.py` | 每未来帧 23 维：每脚 p3、R6、contact、phase，另有 delta_t；10 帧为 230 维。足部目标在当前 root yaw 坐标，obs orientation 相对固定命令 heading | 保留所有维度及观测接口，注入课程命令源 |
| `foot_mdp.py` | 独立于 Motion；已有 swing/stance 位置与速度、foot orientation、离散 touchdown、upright、height、balance 及动作/限位代价；不是接触传感器奖励 | 保留奖励权重和物理终止条件；仅课程模式的 base offset 随两脚圆均值 heading 旋转。任务契约升级 v2；没有重构 BPTT/AC |
| `buffer.py` | (time, env) 环形 GPU buffer，immutable command_id 引用计数、命令快照及整数 ep_step；原WM mask仅检查done，后续审查发现会跨显式评估reset | 命令/时间标签保留；增加ep_step连续性mask及真实物理终止回报采样，详见审查报告 |
| `networks.py` | Actor/WM/Critic 的 RunningNorm 是 state_dict buffer；Actor 依赖 hist 和 Goal 维度 | 保留网络；全量恢复严格检查契约；跨 Standing 任务采用显式迁移 |
| `train.py` | 原有 `last.pt` 含模型与优化器，历史 ckpt 是简版；原子 replace，但无 RNG/课程/时间预算/真实 eval best；恢复会清空 replay | 补齐完整 checkpoint、RNG、课程、串行 eval、预算和 signal 安全退出 |
| controlled standing launcher | calibration 限 200、pilot 总 600，单次 <=30 min；不是 8h 启动器 | 原样保留，新增独立 tmux 启动器 |
| SwanLab / Docker | 原 launcher 在 Docker 内 tmux，主机独立跟随 CSV；旧组为 SuperMimic | 本机无 Docker；直接使用已有 Isaac Lab venv。独立 tmux uploader 跟随 CSV+JSONL，并执行最终 SDK cache sync |

宿主实际 Python：`/home/rm/workspace/IsaacLab_3_0/.venv/bin/python`。旧 `/mnt/sage_data/...` 路径不存在。沙箱不暴露 CUDA/tmux，GPU 测试已获授权在宿主执行。测试前 GPU 无 compute 任务，只有桌面约 172 MiB；未停止任何其他用户进程，未改物理模型、关节顺序、PD 接口或终止阈值。

旧 pilot 末段 `wm_loss≈0.07–0.16`，但 episode 经常约 79–95 控制步就物理终止，不能证明 10 秒站立。WM loss 的短时上升不是停训条件。

### Checkpoint 复用

默认可从零开始；推荐命令显式提供 calibration iter200 的 `--init_weights`。只迁移严格匹配机器人、Goal 表示和 hist 的 **WM + Actor**；保留 WM normalization，重置 Actor goal normalization；Critic、全部优化器、课程、Replay、计时和训练 iteration 重新初始化。没有把旧 Standing optimizer/进度冒充课程续训。pilot600 尚不稳定，未选作默认迁移源。

真正课程 `--resume` 则严格检查机器人、Goal、foot sampling、任务及 curriculum config，恢复所有模型/优化器/normalization、课程/权重/统计、Python/NumPy/Torch/CUDA RNG、全局 iteration、sim_steps、累计 wall time 和 fractional collect accumulator。环境物理状态、历史和 Replay 不恢复：重置 episode、重填 Replay，再进行 20 iterations WM-only 恢复预热；因此不是逐 bit 的仿真续跑。

## 2. 统一命令与可执行包络

- **step_frequency = 每秒左右交替落脚总次数**；完整左右周期频率为 f/2，同脚 stride≈2L。中心目标速度≈L×f。全项目新增 frequency 均遵循该定义。
- f=0 只接受 L=0、yaw_rate=0，生成固定双脚 Standing，不计算 1/f。L=0 且 f>0 是踏步/转向，不是 Standing。
- 转向使用 `yaw_rate`，rad/s，正值向左；每次落脚的 heading 增量为 yaw_rate/f，脚旋转连续 slerp，不使用重置到 [-π,π] 的路径状态。
- f>0 时 period=1/f，swing=0.72/f，双支撑=0.28/f。2 Hz 对应 0.36 s / 0.14 s；要求 swing>=0.32 s、DS>=0.12 s。轨迹 C2 平移，抬脚高度 0.04 m，无双脚同时 swing。
- 配置化过滤：Lf<=0.45 m/s；脚间距离<=0.65 m；每次 yaw 增量<=0.25 rad；保守同脚位移 d≈2L+0.24|yaw_rate/f|；quintic 峰值速度 1.875d/T<=1.8 m/s，水平/竖直峰值加速度 max(5.774d,23.095h)/T²<=12 m/s²。
- 切换候选额外检查下一左右两步的**实际已计划端点间位移**，避免慢长步突然切换成高频短步导致过快摆动；不通过则保持当前命令并记录 `transition_envelope` 拒绝，不能偷偷截断落脚目标。

CPU joint-limited FK fitting 对独立网格取 13 个时刻、150 次优化，允许 pelvis 高度 [-0.08,+0.02] m 调整；包络内 25 个候选拟合误差最大约 2.0 cm。**这是必要的几何近似检查，不含自碰撞、摩擦锥、力矩、支撑平衡或连续 IK 可行性证明，不能当成动态可达域。** 原始结果：`results/s54/curriculum_reach.json`，重跑 `tools/validation/audit_s54_curriculum_reach.py`。

明确拒绝的测试组合包括：

| 命令 | 原因 |
|---|---|
| L=.4,f=1 | swing peak speed |
| L=.4,f=1.5/2 | 中心速度超包络 |
| L=.3,f=1.5/2 | 速度/摆动包络 |
| L=.15,f=2 | 摆动加速度 |
| f=1,yaw_rate=±.35 | 每步转角 .35>.25 rad |

这些是当前保守部署包络拒绝，不是对机器人所有可能步态的不可能性证明。L=.4,f=.5 几何拟合通过，但动态执行尚未通过。默认训练 f∈[.25,2] 加 f=0；(0,.25) 极慢步频在 10 s episode 内不足以可靠验证多次落脚，**本轮不声称覆盖**，协议可表达，但需要更长评估。所有受测命令均保留原值和拒绝原因，不均匀铺满 [.0,.4]×[.0,2]。

## 3. 课程与预算

总预算 28,800 s，包含启动、warmup、训练、eval、checkpoint；恢复后累计而非重新获得 8h。阶段时间是优先级分配目标，**不是强制晋级时间**。未掌握站立可以用尽全部预算留在 C0；后续阶段相应缩减或取消。末尾预留 45 s 保存，完整评估不能挤占退出保存时间。

| 阶段 | 总额目标（含开销） | 任务及范围 | 站立/短/中/长/原地转/曲线 基础权重 |
|---|---:|---|---|
| C0 | .8h | 固定双脚 | 1/0/0/0/0/0 |
| C1 | 1.6h | L=.02–.08，f=.5–.8 | .20/.80/0/0/0/0 |
| C2 | 2h | L 桶 .02–.08/.08–.20/.20–.40，逐步扩大桶内上界；f .25–2，经联合过滤 | .10/.50/.32/.08/0/0 |
| C3 | 1.2h | 左右原地转向，上限 .25 rad/s | .10/.18/.10/.02/.60/0 |
| C4 | 1.6h | 前进+转向，上限 .40 rad/s | .10/.22/.15/.03/.20/.30 |
| C5 | .8h | 全任务巩固，上限 .50 rad/s，仍受每步转角等包络约束 | .10/.20/.15/.05/.20/.30 |

最多约 .4h（5%）用于评估；约 .1h 给启动、checkpoint、退出等，其余约 7.5h 优化训练，开销在各阶段总额内扣除。慢启动实测有时接近 50s，已计入预算。中后期若评估过贵，保留训练和可信保存，不缩短成功协议冒充完成。

晋级：当前阶段每个非零任务权重的独立代表命令均达标，连续两次完整评估通过，其中晋级必须得到 10s Full 确认；阶段至少 600s。Standing success>=95%，其他任务>=85%，每个任务取最差代表命令成功率，foot RMSE<=.045m。8 个评估环境时离散阈值分别等效要求 8/8 与至少 7/8。

失败降低桶内扩展 progress；三次完整失败且 Standing<80% 回退一级；C0 不回退到别的任务。三次无进展输出 stalled 日志。可延长当前阶段，不能为了赶表修改终止或成功率。难度进度影响 C2 步长扩展及转向幅值，各环境可独立采样不同难度。后期一直保留 Standing 10% 及已学行走任务；Replay 继续保存近期混合样本，不是永久历史库。

### 采样分布

先按上述任务权重采样，再按步长类别采样桶内值；频率桶 [.25,.8]/[.8,1.3]/[1.3,2] 基础概率 .4/.4/.2。C1 限定低频；C2 短步有 5% 概率 L=0 踏步。中/长/转向/曲线权重乘以 `.25+.75×最近独立成功率`，高低频桶也依据对应 task_frequency bucket 成功率调节。初始未知成功率取 .5。最后经联合过滤，因此实际比例以日志为准，不宣称基础权重就是执行比例。

左右 yaw 符号等概率；yaw 幅度由课程上限的 .4+.6progress 倍逐步扩展。曲线默认先用短步，转向速率与 f 的联合包络控制曲率；没有一开始让所有大步/高频/高速转向组合同时出现。

每约 4s 有 .35 概率切换命令，实际切换等待完整左右 pair 后的双支撑；Standing 可停留一段后启动。整个 episode 的切换在 reset 时预采样为 immutable plan，目标不跟着实际脚的位置漂移；可在同 episode 变频、前进、站立、左右转向。标签包含生效时间与原始 L/f/yaw/task。统计分别记录 scheduled histogram、每个实际 rollout tick 执行的 histogram/比例、拒绝原因、任务及频率桶成功率。

## 4. 参数语义和 3090 实测

| 参数 | 实际语义 / 最终值 |
|---|---|
| num_envs | 一次 Newton 向量 step 的并行机器人数量；512 |
| rollout_steps | 本项目没有 PPO 式参数。每 iteration `step_acc += 1/utd`，utd=.4 时交替采样 2、3 个向量控制步 |
| buffer_steps | 每环境环形时间容量 512，即 262,144 transitions，约 10.24s/环境；不是优化 batch |
| warmup_steps | 更新前真实 Newton 向量采样 256 步（恢复也重填） |
| wm_window | Replay 上的 8 个连续真实状态转移，用于递归 WM 拟合，带 episode mask |
| horizon | **imagined BPTT** 8 步=0.16s，不是环境 rollout、也不是 episode 时长 |
| WM / Actor | wm_batch=256，wm_batches=2/iteration，pi_batch=64；horizon×pi_batch=512 个 imagined transitions |
| Episode / history | 500 控制步=10s，hist=2，Goal 10×.1s lookahead |
| 其余 | bf16 MLP、eager/no_compile、lr=5e-5、noise=.03、critic epochs=2/minibatches=2；初始 WM-only 200 iterations |

24 iterations/组的串行实测，剔除最初两段日志后取迭代速度中位数；早期 C0 包含跌倒，**是性能测试，不是成功率验证**。GPU 利用率和总显存每 2s 取样，非严格硬件峰值。训练吞吐不包含 cold startup 和最终磁盘写入；Collection FPS 含采样/Replay/normalization，但不含训练更新。

| Envs | WM batch | Actor batch | iterations/s | transitions/s（2.5×N×it/s） | collection FPS | 显存取样峰值 MiB | GPU 利用率均值 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 256 | 64 | 1.889 | 605 | 2,207 | 1,390 | 11.1% |
| 256 | 256 | 64 | 2.052 | 1,314 | 3,745 | 1,488 | 11.5% |
| 512 | 256 | 64 | 2.132 | 2,729 | 7,574 | 1,676 | 9.2% |
| 512 | 512 | 128 | 1.820 | 2,329 | 8,242 | 1,864 | 14.1% |

选择 512/256/64；PyTorch 峰值 allocated≈916MiB，设备快照含 Warp≈1.64GiB。短测并未填满 Replay，满 buffer tensor 底座约 250 MiB，命令/优化器/仿真另计，预计数 GiB 内，但 C5 频繁重置及命令存储峰值尚未实测。不要把 24GiB 当作增大 batch 的理由。低利用率说明 eager FK/BPTT/逐步调度及 Python 命令管理有明显开销；本次未重构核心算法，未试更长 Horizon。

基准：`tools/benchmarks/benchmark_s54_curriculum.py`，结果 `results/s54/curriculum_benchmark.json`。启动器拒绝已有 compute 进程、空闲显存不足 16GiB，且使用进程继承的全局 flock；不杀其他任务。环境无法保证其他用户在本任务启动后不加入负载，训练时有 GPU 监控日志。

## 5. Checkpoint / 评估 / SwanLab

完整原子写入：temp→flush/fsync→replace→directory fsync。保存模型、优化器、norm、curriculum state/分布/RNG/统计、随机状态、全局进度和预算。`last.pt` 与原子 hard-link `latest.pt` 指向完整状态；周期 900s 或 1000 iterations；保留最近 3 个完整历史。阶段转移 `stage_Cn_end.pt` 独立保存（每阶段槽覆盖），best_standing/best_walking/best_overall 根据实际 Newton success 和 foot RMSE 选择；best_overall 要先通过站立门槛，再优先已验证任务覆盖。没有相应测量就没有对应 best，文件存在也不代表任务已经解决。

SIGTERM/SIGINT 请求安全停止；NaN/Inf loss、非有限梯度报错，finally 保存可恢复状态并由 worker 记录非零退出。一次 WM loss 增大不触发停训。周期文件在进程 kill -9/断电前已完成原子写入的可用；无法承诺在强制 kill 后再执行 finally。

Fast 每约 900s，固定 seed8541、最多取前8环境、5s，当前已学任务少量固定命令；Full 晋级确认10s，固定 seed9541，并在预算允许的末段加入独立网格/未见组合。复用同一训练物理场景，训练与评估串行，不创建第二个 CUDA simulator；只对被评估环境执行 Actor，其他 slots 不计入指标。评估结束重置训练环境，冻结 normalization，样本不入 Replay。

评估总配额随 elapsed 按5%累积；Fast 最多90s，候选阶段 Full 先预留45/90/150/210/270/270s配额再执行，单次 Full 最多300s且保留退出时间。这样不会让频繁 Fast 用尽后续 Full 的配额。超额命令标记 budget_skipped；部分/中途截止协议不得晋级。剩余不足2min不启动；末段 Full 可能因实际耗时被跳过，必须如实保留未测命令。

成功条件必须同时满足：完整5/10s未物理终止、RMSE阈值、yaw error（零转向 .08rad/s，非零 .04+.25|yaw_rate|）、speed error<max(.025,.35Lf)；运动任务另要求 >=2 scheduled touchdown、平均落脚误差<.07m、实际 Newton 接触的至少一次同脚周期/stride 观测、频率和步长误差达标。没有实际接触观测时不能给运动任务成功。实际步频由同脚落脚间隔 `2/Δt` 估计，不拿计划相位冒充真实步频；零测量必须同时查看 measurement count，不能把数值0当作正确。

报告单列 physical_termination_rate、timeout_rate、goal_tracking_failure_rate、incomplete_rate。Timeout 只代表完整时长存活；不自动成功。接触是最后 solver substep 的 actual contacts，非整控制周期平均；滑移为实际 stance 足端水平速度。没有修改物理终止条件。

SwanLab 项目 **Foothold_with_SuperMimic**；时间戳实验名和 run ID 写入本地 manifest，resume沿用。CSV：WM/Actor/Critic loss、梯度、reward、episode、termination、it/s；JSONL：curriculum/progress、任务与难度、实际命令桶、采样FPS、GPU显存/利用率、实际评估误差/接触/滑移。直方图用标量桶计数，未高频上传帧状态/视频。日志每20 iterations刷新，上传每5s读取，略延迟最新行以合并评估统计。

独立 tmux uploader 使用本机现有登录，无凭据入库；断网保留 CSV/JSONL/SDK cache，失败以5–300s退避重试。训练结束后先 drain标量，再强制目标 project+ID 执行 SDK cache sync，记录 upload.exitcode/final_sync.exitcode 和状态文件；网络不可达时继续重试，不延长 GPU训练预算。最终网络同步完成时间不能在断网时保证，但不能静默丢数据。原来的 Standing 日志不改写。

## 6. 实测验证与当前策略状态

- CPU：新增22项 unittest 执行通过（包括复用测试类的重复项）；原 foot trajectory 13项、foot tracking 7项通过；编译检查、shell语法检查通过。
- sampler测试：0 Hz不除零、左右转、连续累计旋转>2π、事件C2/双支撑、联合拒绝、分桶覆盖、按成功率调权、时间标签与Replay快照独立、精确sampler续序、fixture晋级/回退、checkpoint恢复、timeout/物理失败/跟踪失败区分。
- 真实 Newton：8env 6 iterations后恢复到30，退出码0。WM/Actor参数（排除norm计数）L2变化 .4742/.3618；WM/Actor/Critic优化器step=30/8/32。Goal最大重算误差<4e-7、Reward<3e-8。完整模型/norm/优化器/RNG/课程已加载验证。
- 阶段切换只在**隔离的成功率fixture**及其保存/恢复测试中通过。真实policy没有通过C0，**没有伪造真实晋级**。
- 独立真实Newton多任务快速评估：9组命令、每组最多8env/5s，约55.1s。Standing、短/中/长步、左右原地转、左右曲线，成功率均0；Standing physical termination=100%，foot RMSE≈.0171m。策略早于完整时长跌倒，故不成功，即使脚位置短期误差小。该测试是calibration权重迁移后极短优化的基础设施测试，不是pilot600最终性能评估。
- 90s小预算测试：冷启动+warmup约52s，预算退出预留45s使其在iteration0正常保存退出，验证预算含启动；没有把该次测试称为发生Actor更新。
- 最终512env/6iter时间戳Smoke `logs/s54_curriculum_20261104_013243/` 已退出码0，Actor与WM实际更新；独立上传及最终同步退出码均0。该次冷启动接近90s，已包含在180s短测预算内。正式8h尚未启动。最终测试WM/Actor权重L2变化.3006/.2393，优化器step=12/4/16；设备显存快照1695MiB。
- SwanLab已完成云端记录：[时间戳Smoke实验](https://swanlab.cn/@yagami_light/Foothold_with_SuperMimic/runs/5eabb903c19246ba)。此前v1离线验证也已回传新组，作为基础设施历史记录；后续新实验默认按时间戳命名。

证据：`results/s54/curriculum_smoke_summary.json`、`curriculum_smoke_evaluation.json`、`curriculum_reach.json`、`curriculum_benchmark.json` 及各 `logs/s54_curriculum_*`。结果目录受现有.gitignore管理；本报告保留关键数值，重跑工具已入源码。

## 7. 启动、查看、重连与恢复

正式训练命令仅提供，**本次没有执行**。新目录/实验自动带北京时间时间戳：

```bash
# 一次性使用本机交互登录，凭据不放命令参数或仓库
.venv-swanlab/bin/swanlab login

# 建议的正式训练（自动打印本次 RUN 名；不是旧Standing全量resume）
bash tools/train/run_s54_curriculum_8h.sh start \
  --init_weights logs/s54_standing_3090_calibration_seed54/last.pt

# 也可显式提供时间戳run名，便于后续管理
RUN="s54_curriculum_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
bash tools/train/run_s54_curriculum_8h.sh start "$RUN" \
  --init_weights logs/s54_standing_3090_calibration_seed54/last.pt

bash tools/train/run_s54_curriculum_8h.sh status "$RUN"
tail -f "logs/$RUN/stdout.log"
tmux list-sessions
bash tools/train/run_s54_curriculum_8h.sh attach "$RUN"
# Ctrl-b 然后 d 离开；SSH退出不影响训练或上传
# 重连SSH后：tmux attach -t "$RUN"

# 安全停止：验证PID仍属于这个run，再发SIGTERM；等待退出码和latest
bash tools/train/run_s54_curriculum_8h.sh stop "$RUN"
cat "logs/$RUN/train.exitcode"

# 同一目录、同一SwanLab实验ID、相同配置恢复；累计计算8小时预算
bash tools/train/run_s54_curriculum_8h.sh resume "$RUN"

# 查看独立上传日志及最终结果
 tail -f "logs/$RUN/swanlab_sync.log"
 cat "logs/$RUN/swanlab_sync_status.json" "logs/$RUN/final_sync.exitcode"
# 上传进程异常退出，可单独在tmux重启，不启动GPU
 tmux new-session -d -s "${RUN}_upload" \
  "bash tools/train/run_s54_curriculum_8h.sh _sync '$RUN' >> 'logs/$RUN/swanlab_sync.log' 2>&1"
```

独立评估必须先确认 GPU compute 空闲，只运行一个进程：

```bash
nvidia-smi
/home/rm/workspace/IsaacLab_3_0/.venv/bin/python tools/validation/evaluate_s54_curriculum.py \
  --checkpoint "logs/$RUN/latest.pt" --num_envs 8 --full \
  --out results/s54/curriculum_full.json --viz none
```

开发短测、CPU回归：

```bash
bash tools/train/run_s54_curriculum_8h.sh smoke \
  --curriculum_config configs/s54/curriculum/s54_curriculum_smoke.json \
  --init_weights logs/s54_standing_3090_calibration_seed54/last.pt
/home/rm/workspace/IsaacLab_3_0/.venv/bin/python -m unittest \
  tools.tests.test_curriculum tools.tests.test_curriculum_replay tools.tests.test_curriculum_evaluation
```

启动器本身不依赖Docker；要换机器只设置 S54_PYTHON/SWANLAB_PYTHON，验证IsaacLab/Newton和机器人资产契约后再短测。resume自定义训练参数时必须重新提供相同覆盖参数（特别是smoke的num_envs、buffer、batch和curriculum_config），不能用正式配置盲目恢复smoke。

## 8. 尚未解决的风险

站立尚未通过长期评估，是最高风险；8h可能全部留在C0，不能保证C3/C4可学习。几何fit不等于力矩/接触/自碰撞可行，L=.4和极端转向只有包络筛查。持续>2π转向轨迹在单元测试验证，但真实policy尚未执行成功。回放只覆盖近期，混合采样降低遗忘风险，但不保证零遗忘。512环境的C5重置/非站立分布下吞吐和满Replay峰值未长期测量。变频命令是episode内提前可见的计划切换，尚未提供任意外部异步插入在线命令的接口。BPTT0.16s对长期平衡与落脚信用分配可能不足，但未经验证不延长。Full预算不足时会明确跳过而非降低成功门槛。SwanLab云端最终同步依赖网络可用性；测试离线缓存和独立上传不等同于断电容灾。

## 9. 2026-11-04 状态日志故障修复

正式启动 `s54_curriculum_20261104_014110` 在 iteration 1 因 `status_temp` 未定义退出。根因是最后补充运行状态输出时缩进错误：变量在 `log_every` 条件内定义，写文件与 telemetry emit 却在条件外执行。之前 `log_every=1` 的短测不能覆盖非日志迭代。已将状态写入、原子 replace、emit 一起放回日志条件内，同时修复了同路径的 `gpu_stats` 潜在未定义访问。

新增 `tools/tests/test_curriculum_logging.py` 直接从 train.py AST 提取并执行实际分支，覆盖首个非日志 iteration 1、日志 iteration 20、随后非日志 iteration 21，以及 nvidia-smi 查询失败。3 项测试通过。故障运行的完整 latest.pt 已验证可读取，保存于 iteration 1、C0、累计约 29.7s；不需要重新迁移权重或重新建 SwanLab 实验。

恢复原正式运行：

```bash
bash tools/train/run_s54_curriculum_8h.sh resume s54_curriculum_20261104_014110
tail -f logs/s54_curriculum_20261104_014110/stdout.log
```

独立真实 Newton 回归使用 `logs/s54_curriculum_logging_fix_smoke_20261104`，从原 checkpoint 恢复，8 环境、`log_every=20`、总 iteration 21。该真实 Newton 回归已从 iteration 1 运行到 21，正常输出 iteration 20 日志并以退出码 0 保存完整 checkpoint；第 21 次非日志迭代也正常完成。新增日志回归与 Replay 回归共 14 项测试通过。证据：`results/s54/curriculum_logging_fix.json`。测试不改写原运行目录，也不自动恢复正式长训练。

## 10. Critic 发散后续

本实验已因持续策略/价值发散在 iteration1852安全停止；动作一致性、foot bootstrap理论上界与恢复方案详见 [S54_VALUE_DIVERGENCE.md](S54_VALUE_DIVERGENCE.md)。原先“继续原latest”的命令不适用于此次失稳后的权重。新的学习率、WM更新频率和短测证据均以该诊断报告为准。
