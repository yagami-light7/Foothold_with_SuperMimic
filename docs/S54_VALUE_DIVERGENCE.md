# S54 M3.4 训练性能瓶颈诊断

诊断日期：2026-11-08，Asia/Shanghai。只读宿主采样从 16:36 开始，最后日志回查为 16:55；没有停止、暂停、重启或附加调试器到正式训练，没有改动其参数、源码快照、Checkpoint、日志或 SwanLab。新增脚本默认只打印计划，GPU 实验尚未执行。

## 1. 结论与证据边界

**最可能的方向是 eager WM 的主机提交开销和 GPU→CPU 标量同步，使 3090 的计算资源未能充分供给。尚未通过阶段计时或两机同版本 A/B 验证其贡献，不能把它宣布为已定位的单一根因。** 实测没有显示 CPU 核数不足、持续 GPU 降频、Swap 或持续磁盘 I/O 是主要限制。当前低利用率同样不足以排除 GPU 算子效率问题。

本次已确认两类不同现象：

1. **稳定的 WM-only 慢速**：正式 CSV 的 iteration 200–1000 为 0.922596–0.949288 it/s，等长窗口综合吞吐 0.936982 it/s，即平均 1067.256 ms/iteration。这不是 iteration 1000 评估导致的假象。
2. **长时间主进程等待**：iteration 1000 后串行执行四组冻结模型评估，Checkpoint 到评估 summary 的文件时间相差 551.112 秒。该阶段主进程等待子进程符合代码设计；评估结束后 iteration 已继续增长至至少 1700。没有证明死锁或异常 CPU/GPU 等待。

无法在不改变当前进程的条件下补出真实 Simulation/WM/Policy 毫秒分解。本报告明确保留这些缺项；独立工具用于 GPU 空闲后的测量，不能用当前 Policy 阶段的 CPU 样本替代此前 WM-only 阶段的 profile。

## 2. 两机已知性能

| 指标 | RTX 3090 服务器 | RTX 4060 Laptop |
|---|---|---|
| WM-only it/s | 用户约 0.94；CSV 实测综合 0.936982 | 用户约 1.23，未读取原始日志 |
| WM-only 每 iteration | 实测约 1067.3 ms | 按用户速度换算约 813.0 ms |
| GPU 利用率 | 用户 42%–48%；本次 Policy 无 I/O 窗口 19%–47%，均值 36.15% | 用户 68%–96% |
| 显存 | 用户约 4.67 GB；16:50 进程 4942 MiB；整卡采样 5742 MiB | 用户约 4.90 GB，统计口径未核实 |
| 功耗 | 用户约 236 W；无 I/O 窗口均值 212.406 W | 用户约 68 W |
| SM 时钟 | 用户 1905 MHz；无 I/O 窗口 1920 MHz | 用户 2490 MHz |
| CPU | Intel family 6/model 85/stepping 7，18C/36T | 用户 Ryzen 9 7940H，8C/16T |
| 代码/环境一致性 | 已读取本机版本和实际运行参数 | **Commit、未提交修改、依赖版本、Graph、线程配置均未核实** |

按上述速度，4060 吞吐约高 31.3%，3090 的 iteration 墙钟约多 254 ms。SM 频率比也接近 1.31，但跨架构不能用频率比证明因果。不能据此认定 3090 算力或 BF16 不如 4060。

## 3. 正式进程与运行环境

### 3.1 实际训练状态

- 宿主 PID **1101008**，父进程 1096360；cwd 为本仓库，实际解释器为指定 IsaacLab venv，解析到 CPython 3.12.16。
- 启动命令在 [只读快照 before.json](../results/s54/m34_performance_20261108/observe/before.json) 的 `argv` 中完整保留。关键参数与用户所列一致：2048 envs，buffer 1024，UTD 0.4，WM window 8、batch 1024、batches 16，Policy batch 256、horizon 32，WM-only 1000，straight 5000，seed 0，BF16，`--no_compile`。
- 其余实际参数包括 iterations 20000、warmup 256、hist 10、goal frames 10、goal dt 0.02、lr 0.0002、log every 100、ckpt every 250、wall seconds 18000、eval every 1000、eval seeds 0/1/2。
- 16:24:49 左右到达 iteration 1000；16:34:00 评估 summary 完成；之后 CSV 持续增长，最终回查至少 1700，Policy loss 与 Actor gradient 非零。其吞吐约 0.587–0.602 it/s，不能与 WM-only 的 1.23 it/s 直接比较。最后回查副本见 [training_windows.csv](../results/s54/m34_performance_20261108/training_windows.csv)。
- 20 秒观察内 CSV 最后行均为 1300，符合每 100 次更新才写日志、约 167–170 秒才出现下一行的周期；随后出现 1400 是增长证据。没有侵入式读取内存中的实时 iteration。
- 宿主进程列表与 NVML 查询只发现一个活动训练计算进程。存在 SwanLab uploader、Xorg、桌面、浏览器和远程桌面。显示服务占用部分显存；未量化其 GPU 时间，不能声称完全没有桌面干扰。

### 3.2 版本与硬件

| 项目 | 本机取证结果 |
|---|---|
| 项目 HEAD / launch git_sha | `6d9e66c2f20790cfbee4d84e3962c9bf4a6f7f2c` |
| 工作区状态 | 开始诊断时已有 8 个修改文件，包含配置和启动器；已保留 |
| 正式源码对应性 | train/env/buffer/networks/mdp/m34/motion/goals/rotations/kinematics 共 10 个文件与 run/source 快照逐字节一致 |
| IsaacLab checkout | `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4` |
| Python | 3.12.16 |
| PyTorch | `2.11.0+cu130`；torch/version.py 记录 CUDA 13.0 |
| NVIDIA 驱动 | 580.178.04；nvidia-smi 显示 CUDA 13.0（驱动支持版本） |
| Newton / Warp | 1.6.1 / 1.17.0 |
| MuJoCo / MuJoCo-Warp | 3.12.0 / 3.12.0 |
| IsaacLab / isaaclab-newton 发行包 | 28.0.0 / 9.1.4；目录名 IsaacLab_3_0 不代表包版本号 |
| GPU | RTX 3090，24576 MiB，PCIe 0000:65:00.0 |
| PCIe | 当前 Gen3 ×16，Host Max Gen3、Device Max Gen4；replay counter 0；单次 Tx 9570 KB/s、Rx 160205 KB/s |
| GPU 温度/限额 | 初始快照 67°C / 350 W；21 点采样 66–72°C，事件原因掩码全部 0 |
| GPU P-state | P2；未出现活动 Thermal/Power Brake/SW Power Cap 原因；历史累计 throttle counter 不能当作本次降频证据 |
| CPU governor / driver | `schedutil` / `intel_cpufreq`；`no_turbo=0`，允许 Turbo |
| CPU 实际频率抽样 | 全核 sysfs 1198.679–4500 MHz；主线程运行态快照所在核约 2100–3000 MHz，均值约 2682 MHz |
| CPU 亲和性 | 允许 CPU 0–35，单 NUMA 节点；未修改亲和性 |
| 线程环境 | `OMP_NUM_THREADS=4`、`OPENBLAS_NUM_THREADS=1`；MKL_NUM_THREADS、SM_PROFILE、CUDA_VISIBLE_DEVICES 在补查时未设置 |

频率来自 sysfs 时点读数，包含休眠/迁移影响，不等同于 APERF/MPERF 的忙时有效频率；尚未测到 4060 的对应忙时频率，因此只能把单线程性能差异列为候选。

版本取自 venv 的发行包元数据、torch/version.py 和 checkout；没有向正式进程注入查询，也没有启动第二个 CUDA context。初始沙箱隐藏宿主 PID 且无法连接驱动；随后通过获准的宿主只读查询取证，未把沙箱错误当作训练退出或驱动损坏。

### 3.3 内存和 I/O

- 系统约 31 GiB RAM，约 21 GiB available。Swap 总量 2 GiB、已用 3 MiB；采样 `si/so=0`，正式 PID `VmSwap=0`。
- 初始 PID RSS 3911708 KiB，线程 51；虚拟地址空间约 114 GiB 不等于实际内存消耗。pidstat RSS 均值约 3915754 KiB，无 major faults。
- iteration 1000 的原有 runtime_stats：Torch 峰值 allocated 2759.23 MiB、reserved 2848 MiB，整卡当时 used 4537.25 MiB。Torch 峰值不包含 Warp；当前 NVML 进程显存和整卡显存是另一个时点，不能混用。
- 16:40:38–58 窗口跨过 ckpt_001250（mtime 16:40:57.804），末端有约 88/290 MB/s 系统写盘突发，GPU 利用率降到 12%–13%。两者时间相关，未精确归因到 CUDA copy/序列化/fsync。
- 16:44:34–53 窗口 PID rchar/wchar/read_bytes/write_bytes 增量全部 0，无评估子进程。因此稳态低 GPU 利用率不能全部归于磁盘 I/O。
- vmstat 大部分样本系统 idle 95%、iowait 0；最后一次 checkpoint 附近 iowait 2%。pressure 的 CPU/memory avg10/60 为 0；I/O 有短暂低值，未见持续系统内存压力。

## 4. 非侵入式采样结果

原始文件在 [诊断目录](../results/s54/m34_performance_20261108)，数值汇总见 [summary.json](../results/s54/m34_performance_20261108/summary.json)。

| 观察窗口 | 主线程 CPU（单核=100%） | Autograd CPU | GPU | 解释边界 |
|---|---:|---:|---|---|
| 16:40:38–58，pidstat 20 个间隔 | 51.57% | 44.58% | 21 点均值 37.43%，12%–49% | Policy 阶段，包含 ckpt_1250 |
| 16:44:34–53，/proc 首末约 19.002 秒 | 53.36% | 46.68% | 20 点均值 36.15%，19%–47% | Policy 阶段，无子进程、无 PID I/O 增量 |

第二窗口进一步发现：

- 主线程 voluntary/nonvoluntary context switches 分别增加 431/13；Autograd 增加 432/42。前者约 22.7 次/s voluntary，不能仅凭此认定严重上下文切换。
- 主线程迁移 **0 次**，Autograd **3 次**；CPU runqueue 等待分别 **0.196 ms / 0.759 ms**，合计不足观察窗口的 0.006%。排队争抢 CPU 的证据很弱。
- 主线程 20 个 wchan 样本：11 个 `0`、9 个 `futex_wait_queue`；Autograd 为 9 个 `0`、11 个 futex。两者同时 futex 的时点有 6 个，但这是 1 Hz 快照计数，**不能换算成 30% 时间停顿**。
- CUDA event handler 多为 poll，其他线程大量 futex/sleep；这符合线程池闲置或等待任务/设备的多种情况。wchan 不能区分 Python/Autograd 协调、CUDA completion 和其他锁，需要独立 trace。
- 两个主要线程 CPU 时间总和约一个核，GPU 同时未满载，支持主机执行与 GPU 交错的假设；没有证明主线程持续 CPU 饱和或 GPU 死锁。

没有使用 perf/ptrace/py-spy attach。现有证据已足以排除若干方向，而附加采样不能替代真正的 CUDA 时间线。

## 5. 实际执行路径和同步候选

### 5.1 CUDA Graph、编译与 BF16

- [train.py](../supermimic/train.py) 的 `compile_fn` 在 `--no_compile` 下直接返回原函数；WM rollout/loss、imag_step、reward、critic features 都走 eager。代码没有对 WM/Actor optimizer 使用 `torch.cuda.CUDAGraph`。
- [env.py](../supermimic/env.py) 创建 NewtonCfg 时未关闭 `use_cuda_graph`。安装的 NewtonManagerCfg 默认 True；正式 stdout 明确记录 `CUDA graph took ... 0.518369 s`。
- 安装的 NewtonManager.step 在 cfg.use_cuda_graph、`_graph` 非空和 CUDA device 条件下执行 `wp.capture_launch`。Graph 覆盖 solver/可能包括 graph-safe actuator 的路径，**不是整个 collect、WM 或 Policy**。已有捕获证据和 replay 静态路径；未获得正式进程时间线来测量 replay 覆盖率，也未直接读取其 `_graph` 指针。
- [networks.py](../supermimic/networks.py) 将 MLP 放在 CUDA BF16 autocast 中，随后转回 FP32；旋转、积分、状态误差、FK/reward 等仍主要为 FP32。实际参数 amp=bf16 与代码一致；具体 matmul kernel/吞吐需要 trace，不能把 BF16 等同于整条流水线 BF16。
- 原有 `SM_PROFILE` 默认关闭，本次确认未设置。该旧开关会多次 synchronize，且外层只在 Policy 阶段特定 iteration 打印，无法提供干净的 WM-only 分解。**不要在正式进程启用它来比较性能。**

### 5.2 每次 iteration 做了什么

1. UTD accumulator 交替收集 2/3 个向量化 simulation steps，均值 **2.5**，不是每次 16 个仿真 step。
2. collect：策略推理、观测/目标/归一化、Newton step、reward/termination/reset、GPU replay 写入与 WM norm 更新。
3. **16 次独立 WM 更新**，每次 sample 1024 个 8-step windows；每 iteration 共有 128 个顺序的 WM step 调用，每次包含三层 2048 宽 MLP 和状态积分，并执行 backward、clip、Adam。该网络并非一个很小的 MLP，不能在没有计时前忽略 GEMM 成本。
4. 从循环 it=1000 起，增加一次 horizon=32 的 Actor BPTT；之后 Critic 默认 5 epochs ×4 minibatches，即 20 次 Critic optimizer 更新。
5. 每 100 次 logging；每 250 次 checkpoint；每 1000 次独立冻结评估，训练父进程同步等待。

### 5.3 静态确认的 CPU/GPU 同步边界

| 路径 | 已确认的实现 | 频次/风险 |
|---|---|---|
| WM loss | Python `if not torch.isfinite(loss)` | 每 iteration 16 次设备标量→主机分支 |
| WM gradient clip | `clip_grad_norm_(..., error_if_nonfinite=True)` | 安装的 torch/nn/utils/clip_grad.py 对 GPU total_norm 的 isnan/isinf 做 Python bool，另 16 次 |
| RunningNorm.update | `if self.count >= self.max_count`，count 为设备 scalar | collect 每 step 四处调用，均值约 10 次/iteration |
| collect state gate | `if not torch.isfinite(out['s1']).all()` | 均值约 2.5 次/iteration |
| env reset | CUDA `done.nonzero()` 得到动态长度索引 | 每 simulation step；还含 reset/forward 成本 |
| Policy / Critic | 动态有效 start、finite loss、clip、finite grad 等 | Policy 阶段额外增加；不能归入 WM-only 对比 |
| logging | 对积累的每个 GPU scalar 逐个 `float(v)` | 约每 100 次集中出现，可能有批量 host dispatch 开销 |
| checkpoint | 对模型每个 tensor 做 finite 判定，保存 last 和编号 checkpoint | 每 250 次及边界重复保存，多个 host 判定/序列化与写盘 |

WM-only 热路径仅上述显式 finite/count 分支就约 **44.5 次/iteration**（32+10+2.5），还未计 nonzero 和引擎内部同步。这是源码推导的执行次数，**不是测得的 CUDA API 次数或毫秒数**；已有 GPU 工作可能早已完成，不能按次数直接推算收益。

[buffer.py](../supermimic/buffer.py) 的 Motion replay 存于 GPU，sample_windows 为设备索引，未发现每 batch 大量 CPU→GPU 拷贝。Foot command 的 `.cpu().tolist()` 分支在本次 motion goals 下不启用，不能误判为 M3.4 的数据传输瓶颈。

## 6. 当前能给出的耗时分解

| 组成 | WM-only 当前证据 | 仍需测量 |
|---|---|---|
| 每 iteration 总墙钟 | 200–1000 综合 1067.256 ms；来自已有 CSV | 排除 checkpoint 的独立纯稳态 baseline |
| Newton/MJWarp | 已确认仿真 Graph 捕获，每 iteration 平均 2.5 simulation steps | `newton_step` CPU wall、CUDA span、kernel 时间 |
| Rollout/replay/normalization | 上述同步边界及 GPU replay 路径 | collect/env_step/replay_add/wm_sample 各自时间 |
| WM forward/backward/optimizer | 每 iteration 16×8 rollout；32 次 finite 分支 | WM forward/loss/backward/clip/Adam 分解 |
| Actor/Critic | iteration <=1000 不更新；之后约 0.59–0.60 it/s | policy_forward、Actor backward/optimizer、Critic 时间 |
| 同步/传输 | 静态多处 device scalar→host control flow | profiler 中 scalar 提取、同步 API、memcpy、kernel gap 的时间 |
| Checkpoint | 约 137 MiB/文件；ckpt_1250 时磁盘突发与 GPU 下降相关 | 边界完整墙钟、序列化/拷贝/磁盘比例 |
| Evaluation | iteration 1000 约 551 秒（文件时间上界式估算） | 子进程 setup/simulation 分解；非 WM 热循环成本 |

`it_per_s` 的第一窗口包括 warmup collect；后续窗口可含 checkpoint、logging 后处理。日志记录 t_log 后才逐个转 scalar；相应开销会进入下一窗口。评估主体通过调整 t_log 排除，但边界保存、同步和 empty_cache 不全部位于该排除区间。因而它适合确认长期慢速，不足以给阶段精确归因。

## 7. 根因分级

**已经证实：**

- 正式进程正常进入 Policy 阶段，iteration 持续增长；采样时没有第二个独立计算训练进程。
- WM-only 稳态约 0.937 it/s，评估不能解释全部差距。
- 仿真 Graph 已捕获，而 PyTorch WM/Actor eager；两种开关互相独立。
- 主机热路径存在频繁 device scalar 分支，Motion replay 已在 GPU。
- 当前样本没有持续降频、Swap 压力或 CPU 调度排队；无 I/O 的 Policy 窗口 GPU 仍未满载。
- iteration 1000 的大段等待对应设计内评估；checkpoint 会造成可见短时干扰。

**尚待验证，按优先级：**

1. eager WM 的大量 host launch/自动求导提交与同步，占据了显著 iteration 时间；Xeon 的有效单线程执行能力使其比 7940H 更受影响。需要同版本 WM-only trace 和阶段 A/B。
2. WM BF16 GEMM、LayerNorm、旋转/积分等 kernel 在两代 GPU 上的效率差异；可能与主机限制共存。需比较每类 kernel 的总时间、shape 和 launch gap。
3. 两机代码/依赖、Graph、线程或实际工作量不同。4060 尚未取证，在排除此项前不能严格称为同配置硬件对比。
4. CPU governor、忙时频率、桌面负载有次要影响。当前未证明线程迁移或核数是主要根因，也没有 PCIe Gen3 链路吞吐饱和证据。

## 8. 优化顺序与验证/回退

没有已验证收益百分比；下面按可能覆盖的热路径、风险和成本排序，**不对正式训练应用修改**。

| 优先级 | 建议 | 预期收益范围与成本/风险 | A/B 与回退 |
|---|---|---|---|
| 1 | 用已有 torch.compile 路径编译 WM rollout/loss，先保留全部 finite checks | 覆盖每次 iteration 的 128 个 WM step 和逐元素运算；潜在收益最高但未定量。成本中；编译启动慢、graph breaks、数值微差需检查 | baseline eager 对 `--compile`；比较预热后 wall/events 和 loss/梯度/权重有限性。回退为默认 `--no_compile` |
| 2 | 将 RunningNorm 的冻结判断改为设备端掩码/等价控制，保持同阈值和更新公式 | 减少 collect 每 iteration 约 10 次标量控制流；收益取决于 collect 占比。成本低中；需验证达到 max_count 前后、resume 时的等价性 | 在独立 checkout 做唯一代码变量的 A/B，先看 collect 与 scalar-sync 时间，再看总 it/s。回退该单独补丁 |
| 3 | 对单进程测试 4/8/16 CPU 线程，并视数据测试单进程 taskset | 低成本、低风险，无需系统配置变化；当前只有约一核活跃，不预期简单增加线程就有大收益 | baseline 4 对 threads8/threads16；两机均测，结合主线程/Autograd、忙时频率。恢复原启动环境即可回退 |
| 4 | GPU 上聚合 logging scalar，再少量拷回；研究减少重复 checkpoint 保存 | 降低边界停顿，对 WM 稳态收益可能小；成本低中，日志舍入和保存语义需要保持 | 对跨边界的完整墙钟独立测量，保持同日志频率、评估协议和 checkpoint 完整性；回退日志/保存补丁 |
| 5 | 仿真 Graph on/off 归因 | Graph 已开启，不应当成尚未启用的免费优化；关闭可能显著变慢 | `graph-off` 只改变独立实例 NewtonCfg.use_cuda_graph。保持 dt/substeps/solver/actuation 不变，对比状态误差、loss、newton_step 和整轮墙钟；默认 on 回退 |
| 6 | 改 env 数或 WM batch 做容量与利用率消融 | 改变每 iteration 实际样本工作量，it/s 不直接可比；可能 OOM。非首选修复手段 | 单项变更，记录 transitions/s、WM windows/s、峰值显存及 wall；恢复原 JSON 对应值 |

保留 loss/gradient 非有限检查的异常语义，不建议简单删除或延迟到多次 optimizer step 后再检查。不要改变 UTD、WM batches/window、BPTT horizon、数据 split 或训练目标以制造加速；GPU 更高利用率也不等同于更高有效吞吐。

## 9. 独立工具与实验协议

新增 [profile_s54_m34.py](../tools/profile_s54_m34.py)、[benchmark_s54_m34.sh](../tools/benchmark_s54_m34.sh) 与 [CPU 安全测试](../tools/test_s54_m34_performance.py)。**此次只执行 observe、plan、CPU 测试，未执行任何 GPU benchmark。**

工具约束：

- observe 只用 stdlib、/proc、sysfs 和 nvidia-smi；新建证据目录，不 attach、不导入 torch、不写正式 run。
- benchmark 默认打印计划，不创建 run；只有显式 `--execute` 才进入实验。
- 实验与正式 host launcher 使用同一 `logs/.s54_curriculum_gpu.lock`，非阻塞获取；再检查所有 GPU compute PID 和训练/pipeline/evaluation 进程。有占用、驱动查询失败或 PID 访问错误就拒绝，不排队、不等待正式训练结束后自动开始。
- 必须使用新的 `logs/s54_perf_*` 名称，已有目录拒绝。执行原 train.py 的内存 AST 插桩，现有文件不修改；上游训练结构不匹配则拒绝运行。
- 默认 10 次 iteration 预热、20 次测量，原 simulation warmup_steps=256 保留。最多 100 次预热、100 次测量，wall_seconds=600（沿用训练的 iteration 边界软停止语义，不是强制超时；单个卡住的调用或编译不能靠它中断）。一次命令只跑一组，不自动扫参或启动后续长训练。
- 所有测量窗口关闭 evaluation/logging/checkpoint；训练结束后仍会在**新 benchmark run**保存自己的 checkpoint，不计入测量。每组有额外磁盘/启动成本。
- Policy 必须提供相同的冻结 checkpoint，原模型/optimizer/RNG 恢复后 replay 均重新填充，再进行独立预热。checkpoint 只读；SHA256 写入 benchmark_plan。两机应复制同一不可变文件并核对哈希，不用训练中的可变 last.pt 作比较源。
- WM 可 fresh seed=0，或选定 WM 阶段 checkpoint；后者要求测量结束不超出 checkpoint 原始 WM-only 边界。Policy 用 phase=policy，不把 WM-only/recovery 混入测量。
- `--timing wall` 不记录 stage events；`events` 才开启分层 CUDA Events；`--trace` 额外使用 torch.profiler，最多 5 个测量 iterations，无 shapes/stacks/memory profiling。

### 9.1 当前正式训练期间允许执行

```bash
# 无 CUDA 计算，只打印配置；默认无需 --execute。
bash tools/benchmark_s54_m34.sh --name s54_perf_plan_check --case baseline --phase wm

# CPU 检查。
python3 tools/test_s54_m34_performance.py
python3 -m py_compile tools/profile_s54_m34.py tools/test_s54_m34_performance.py
bash -n tools/benchmark_s54_m34.sh

# 可选只读观察，目录必须不存在；在能看到宿主 PID/GPU 的命名空间执行。
ISAACLAB_PYTHON=/home/rm/workspace/IsaacLab_3_0/.venv/bin/python
"$ISAACLAB_PYTHON" tools/profile_s54_m34.py observe --pid 1101008 --seconds 20 \
  --run logs/s54_m34_3090_run01 --out results/s54/m34_performance_next
```

### 9.2 正式训练释放 GPU 后，手动选择一项执行

两台机器先统一源码与数据哈希、包版本、seed、阶段、fixed checkpoint、warmup 和采样长度。环境数与 batch 消融有意改变单项工作量，应另外标记，不能算严格相同超参数的提速。

```bash
# 各命令均为独立选择；本次没有执行。设置本机解释器后，一次只运行一项。
export ISAACLAB_PYTHON=/home/rm/workspace/IsaacLab_3_0/.venv/bin/python

# 先确认无插桩与 Events 插桩吞吐差异，再做短 trace。
bash tools/benchmark_s54_m34.sh --name s54_perf_wm_wall_r1 --phase wm \
  --case baseline --timing wall --warmup 10 --steps 20 --execute
bash tools/benchmark_s54_m34.sh --name s54_perf_wm_events_r1 --phase wm \
  --case baseline --timing events --warmup 10 --steps 20 --execute
bash tools/benchmark_s54_m34.sh --name s54_perf_wm_trace_r1 --phase wm \
  --case baseline --trace --warmup 10 --steps 3 --execute

# 使用已有编译路径；首次编译/重编译若未稳定，增加相同预热并重新测两组。
bash tools/benchmark_s54_m34.sh --name s54_perf_wm_compile_r1 --phase wm \
  --case baseline --compile --timing wall --warmup 10 --steps 20 --execute

# 其余单变量选项：graph-off、threads8、threads16、env1024、env4096、wm512、wm2048。
bash tools/benchmark_s54_m34.sh --name s54_perf_wm_graph_off_r1 --phase wm \
  --case graph-off --timing events --warmup 10 --steps 20 --execute

# Policy 必须用固定阶段的同一 checkpoint；迭代 1000 冻结文件为本机候选。
bash tools/benchmark_s54_m34.sh --name s54_perf_policy_baseline_r1 --phase policy \
  --checkpoint logs/s54_m34_3090_run01/evals/iter_001000/checkpoint.pt \
  --case baseline --timing events --warmup 10 --steps 20 --execute
```

env4096/wm2048 只应在先确认本机容量后单独执行，4060 不保证能容纳。脚本遇 OOM/失败不会继续扫参；未完成测量时不会输出成功性能结论。线程与 batch/env 消融应先比较 WM-only，再独立比较 Policy。

### 9.3 输出和判断方法

- `benchmark_plan.json`：实际参数、Python/包版本、Git/dirty 状态、所有 supermimic Python 文件哈希、可选 checkpoint 哈希。
- `performance.json`：it/s、包含最终 GPU 尾部的窗口总墙钟/每 iteration 墙钟、process CPU%、real transitions/s、WM windows/s、Torch 峰值、1 Hz NVML 利用率和整卡显存采样峰值、实际 torch 线程数和 Graph 对象是否存在。
- Events 模式提供每 iteration 的 collect/env_step/newton_step/replay_add、WM sample/forward/loss/finite-check/backward/clip/optimizer、Policy forward/Actor backward+optimizer/Critic+diagnostics 的 CPU wall 和 CUDA stream span。父子 scope 重叠，**不可相加**。
- CUDA Events 记录的是当前 stream 的经过时间，包含主机 launch gap，**不是纯 GPU kernel busy time**。Newton 可能使用其他 stream，应在 trace 核对依赖；如缺少依赖不能把 event span 当作其完整设备时间。
- 计时只在测量窗口首尾同步各一次；不在每阶段无条件 synchronize。per-iteration `enqueue_wall_ms` 不含最终尾部，与总窗口 wall 字段分开保存。
- `gpu_samples.json` 为测量期 1 Hz 原始数据；采样峰值可能漏掉瞬时峰值，并包含显示服务等整卡占用。Torch allocator 峰值不含 Warp。
- `trace.json` 只用于定位 kernel gap、cudaGraphLaunch、scalar extraction/synchronization、memcpy 和 kernel 类别；profiling 结果不与无 profiler 的原始 it/s 混比。

每组至少手动重复 3 次，交换 A/B 顺序，比较中位数及离散范围。先量化 events/trace 对 baseline 的扰动，吞吐收益以 wall 模式为准；每阶段归因以 events/trace 为准。编译若仍有重编译、温度/时钟未稳定或 checkpoint 边界进入窗口，则本组无效。收益应明显超出重复噪声，同时检查有限 loss、梯度、机器人/数据 contract 与状态误差；BF16/编译不要求跨 GPU bitwise 一致。

## 10. 本次交付验证和下一步

已通过 5 个 CPU 测试：真实 train.py 插桩前后去除纯观测节点后的 AST 完全相同；不匹配源码拒绝；忙 GPU/驱动不可访问/活动 pipeline 拒绝；在 `python -S` 无 site-packages 下只打印计划且不创建 run；resume 正确处理绝对 iteration、累计 wall budget，并排除测量期的 logging/checkpoint 取模边界。另已通过 Python 编译、bash 语法和实际 baseline plan 检查。

**CUDA Events、torch.profiler、Newton Graph 切换和 GPU benchmark 的运行效果尚未在设备上测试**，因为正式训练优先。首次空闲实验先跑 baseline wall/events，确认输出与时间线有效；这也是进一步根因定量的必要步骤。

下一步优先级：补齐 4060 运行元数据 → 同阶段两机 baseline/短 trace → 验证 compile、RunningNorm 控制流和线程配置。当前无需对正式训练做任何变更，也不需要为调查重启它。
