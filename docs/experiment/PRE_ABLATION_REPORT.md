# S54 正式消融前有界验证报告

> 2026-11-05 目录整理：本文历史命令保留执行时路径；当前训练入口与参数见 [STAGE1_WALK_TRAINING.md](STAGE1_WALK_TRAINING.md)，完整新目录索引见 [tools/README.md](../../tools/README.md)。历史日志、结果和checkpoint未移动。

日期：2026-11-05（Asia/Shanghai）。结论：**BLOCKED**，当前不具备开展正式 Short Ablation 的能力条件。

## 1. 主线与本轮范围

实验主线是 **LAFAN1 Walking Stage 1 全身 Motion Tracking → 合格的 Walking WM 与 Shared Decoder → A–G 的 Foot 任务消融**。可选 Stage 1.5 位于两者之间。正式 B/C/D/F/G 必须使用同一个合格 Walking Stage 1 来源。Standing 校准得到的 WM 不替代这个来源，历史 Standing Actor 不替代 Walking Decoder。

本轮按 §12 的诊断优先级完成 WM-only、Standing 对照、Walking Teacher pilot 和工程审计；没有启动正式 A–G 对比或长训。最后一个 D 路径的 12 轮 run 只验证初始化、梯度及上传接口，不用于比较效果。下一步主线应继续 Walking Stage 1。

实际硬件是 **RTX 3090，nvidia-smi 总显存 24576 MiB**，初始约488 MiB、0%利用率、没有计算任务。本文均为本机结果，不将原 §12 的 RTX 4060 Laptop 结果当作3090测量。

§12 引用的 `logs/s54_mp_*_20261104` 及 verified 数据目录原本不在本机，默认 `/mnt/sage_data/...` Python 也不存在。现已修复 Python 发现路径，并用本地 CSV 重新做 QA、重建12轮接口来源。该来源不是成熟 Teacher。`logs/walk/.../model_48000.pt` 是另一种 actor/critic checkpoint 格式，缺 SuperMimic WM 和 S54/Goal 合同，没有强行复用。

所有新产物在 [pre_ablation_20261105](../../results/s54/pre_ablation_20261105/)。历史日志、checkpoint 和旧结果未覆盖。

## 2. Gate 证据

| Gate | 实际证据 | 结果 |
|---|---|---|
| 本机数据几何 QA | Walking/Dance 两段 Loader、checksum、关节限位、FK、50Hz时间网格通过 | PASS；不代表动力学可执行 |
| WM-only 更新隔离 | 80轮；Actor含全部归一化、Critic逐 tensor 不变；其 optimizer state 为空；WM改变 | PASS |
| 新方案 WM8 | 最差seed高度1.150cm、Pitch2.108°，要求<1cm、<1° | FAIL |
| 接触边缘覆盖认证 | 有实际承载的足倾斜状态；训练的接触数≤2代理计数为0，未记录逐接触点压力分布 | 未完成，不能报PASS |
| 新 hist10 Motion→Foot Standing | 2/5/10s×54/55/56；5/10s全部height终止；2s有一次timeout但Pitch约45° | FAIL |
| 历史 Standing 阳性对照 | `best_standing` 9/9 timeout，真实姿态/足轨迹/接触/滑移及其自身WM门槛均满足 | 有效历史参考；不兼容新Shared Decoder |
| Walking Stage 1 Teacher | 200轮、10048条真实transition；18个固定窗口全部物理终止 | FAIL，不标记qualified |
| 独立Walking holdout | 只有train walk1、heldout dance1 | 缺失；Dance不算Walking泛化 |
| Standing / Turning参考 | 数据中缺这些独立类别 | 缺失，不声称Teacher掌握这些能力 |
| 梯度/迁移/恢复/日志 | 58项测试通过；实际Bridge、Resume、最终接口smoke通过 | 工程接口PASS，不代表能力PASS |

机器可读证据：[final_readiness.json](../../results/s54/pre_ablation_20261105/final_readiness.json)、[runtime_isolation_audit_final.json](../../results/s54/pre_ablation_20261105/runtime_isolation_audit_final.json)。预先声明的能力门槛在 [protocol.json](../../results/s54/pre_ablation_20261105/protocol.json)，可复用配置为 [pre_ablation_gates.json](../../configs/motion_prior/pre_ablation_gates.json)。没有放宽物理终止或成功率标准。

## 3. WM 校准：改善明显，尚未通过

采用原 `calibration_bounded.json`：16env、最多80轮、WM batch64×2、window8、hist10、H8、噪声0.03。保留原 WM/BPTT 算法、奖励及终止系数。新增 `--wm_calibration` 冻结 Actor 的归一化统计，保存 `calibration_before.pt`；否则“Actor参数没更新”仍可能因归一化变化而改变控制行为。

实跑80轮、6400条真实transition，保存结束约57.23s。Actor/Critic无更新。覆盖2448条1.2s后的状态，Pitch最低−46.22°，最大绝对动作0.1378；存在真实足底承载。`contact_edge_samples=0` 只说明原接触数量代理未捕获边缘，不能据此宣称没有边缘，也不能宣称覆盖已认证。

比较使用三个seed最长记录：每个 WM rollout 从**同一保存的Newton状态**出发，复用同动作、ξ、命令和tick，各horizon用同一组窗口，不跨reset。后模型直接复算原记录，不额外重跑Newton。Actor及归一化逐项相同，所以不能把WM改善解释为Standing策略变好。

| Horizon | Height RMSE cm 前→后 | Pitch RMSE ° 前→后 | Foot FK RMSE cm 前→后 | 归一化Reward MAE 前→后 |
|---|---:|---:|---:|---:|
| 1 | 0.611→0.094 | 1.770→0.240 | 0.989→0.089 | .01044→.00297 |
| 4 | 2.509→0.468 | 6.992→1.004 | 3.871→0.396 | .03218→.00721 |
| 8 | 5.198→1.150 | 13.859→2.108 | 7.340→0.823 | .08087→.01309 |

上表各项取最差seed。归一化有符号Reward Bias的三seed均值，1/4/8步分别为：`.00473/−.01729/−.07114 → −.00051/−.00226/−.00589`。MAE按现有Gate定义，是窗口累计reward差除以 `h×max_step_reward` 后取绝对均值；逐步MAE也另存于原JSON。

WM8四项门槛为 `.01m / 1° / .015m / .02`；目前只有后两项通过。这个诊断来源是重建的12轮Walking接口权重经Motion→Foot初始化，不能与远端原C已训练12轮的历史数值作严格前后对照。

证据：[wm_comparison.json](../../results/s54/pre_ablation_20261105/wm_comparison.json)、[配对误差图](../../results/s54/pre_ablation_20261105/wm_calibration.png)、[覆盖计数](../../logs/s54_preablation_wm_20261105/calibration_coverage.json)。

## 4. Standing：区分冷迁移失败与历史成功

同一实际Newton、相同seed与reset扰动；正常timeout不算跌倒，同时发生height终止与timeout时优先记物理失败。2/5/10s复用seed，因此是3个独立扰动seed，不是9个独立seed。

| 控制器/来源 | 10秒请求下 seed54/55/56 实际持续秒数 | 结果 |
|---|---|---|
| 新hist10冷迁移Actor | 2.02 / 2.00 / 1.74 | 全部height，Pitch约45° |
| 默认零Action PD | 2.02 / 2.00 / 1.74 | 与冷迁移几乎一致，无主动平衡 |
| 历史600轮Standing pilot | 1.46 / 1.46 / 1.38 | 全部height |
| 两踝pitch目标−.04rad诊断 | 2.34 / 2.38 / 2.72 | 延迟后倾，仍全部失败 |
| 两踝pitch目标+.04rad诊断 | 1.54 / 1.54 / 1.44 | 加速后倾，全部失败 |
| 历史35480轮best_standing | 10 / 10 / 10 | timeout，满足本轮姿态/足轨迹/真实接触门槛 |

踝偏置仅是独立评估控制器的有限PD目标偏移；没有修改源物理、关节顺序、PD系数、默认站姿或训练奖励。映射29关节检查通过；腿PD stiffness200，踝pitch damping约6.325、effort91.6Nm，来源代码明确这些PD增益是仿真假设。静态COM投影位于几何足包络中，但可见mesh包络不能代替Newton实际接触支撑多边形。

真实记录显示两脚大多仍承载，后期出现承载足倾斜、滑移、基座后倾及掉高。PD目标对实际运动有明确正负方向影响，不能将失败直接归因于“执行器没执行”。也没有证据支持直接改机器人模型。新冷迁移Actor最大动作仅约.002，几乎等同零动作；没有获得纠偏能力。

历史best的9个窗口最大Pitch3.578°、高度偏差最大7.39mm、足位置RMSE最大2.33cm、承载足滑移最大.0146m/s；其WM8最差高度.641mm、Pitch.264°。这是可信的**历史站立阳性对照**。其hist2、identity adapter及历史合同与新hist10方案不同，不能作为匹配A/B/C组或直接迁移Decoder。原evaluator的 `standing_gates.passed=false` 是因为该checkpoint任务标签为curriculum；本轮逐记录核实命令长度均为0，单独生成 [legacy_standing_acceptance.json](../../results/s54/pre_ablation_20261105/legacy_standing_acceptance.json)，未改写原始summary。

Standing reward有base height/upright等梯度，但短H8仅覆盖0.16s，而跌落约1.7–2s；初始几何接近不等于长期可稳定。当前证据支持“覆盖/预测偏差与缺少学会的纠偏并存”，没有证明唯一根因，也没有通过提高Prior掩盖问题。

证据：[真实接触与PD跟踪分析](../../results/s54/pre_ablation_20261105/standing_physics_traces.json)、[静态物理检查](../../results/s54/pre_ablation_20261105/static_physics_audit.json)。

## 5. Walking Stage 1 Teacher：尚未获得可用于消融的WM/Decoder

QA重建了train `walk1_subject1`（261.30s、13066控制帧）及heldout `dance1_subject1`。完整29关节；头部两关节仍是既有补齐值，不声称有独立头部运动监督。几何QA不认证接触动力学。缺独立Walking holdout、standing、turning、slow_walk/start_stop类别。

本轮另从头运行 [stage1_bounded.json](../../configs/motion_prior/stage1_bounded.json)：16env、200轮、128步warmup、前100轮WM-only、后100次Actor更新；WM64×2、Actor32、hist10、H8、lr5e−5、原Motion reward。真实样本上限10048，实际10048；imagined Motion-Reward transitions 25600，不能把它们当新增真实样本。训练子进程79.47s，未到300s预算。

验收预先要求：5/10s、seed54/55/56、clip偏移0/30/60s；真实timeout且body position<.08m、body orientation<15°、root tracking orientation<10°、高度偏差<.05m、pitch/roll<10°、joint/arm pose<.35rad、arm velocity<1.5rad/s、无关节越限、实测contact accuracy≥.8、承载滑移<.05m/s，并检查摆臂幅度。偏移窗口仍来自同一个训练clip，不能叫holdout。

实际18个窗口均height物理终止，持续 **.62–1.32s**，累计760条评估transition：

- 全身世界坐标body position RMSE `.250–.451m`，orientation `47.3–84.6°`；世界指标包含root XY/yaw漂移，不能当作纯局部关节误差。
- 关节pose RMSE `.388–.843rad`；手臂pose `.356–1.146rad`、velocity `1.393–3.967rad/s`。
- 实测contact accuracy `.328–.844`，承载足滑移 `.057–.283m/s`。
- 真实全身回放可见明显躯干倾倒，不能认定为自然摆臂或合格全身跟踪。

没有生成任何qualified Teacher标记。[验收明细](../../results/s54/pre_ablation_20261105/final_readiness.json)、[Teacher真实rollout](../../results/s54/pre_ablation_20261105/teacher_train_windows/summary.json)、[同视角全身GIF](../../results/s54/pre_ablation_20261105/full_body_comparison.gif)。GIF仅渲染记录状态，记录提前结束后明确标注末帧保持，不补造仿真成功时长。

## 6. A–G 公平性与恢复审计

- A随机WM/D；B仅加载Stage1 WM；C加载同一WM及D。相同seed下E_f保持同初始化，所有组同hist10与mask adapter。
- 当前制度**所有组冻结D，包括随机A/B**，比较的是只训练Foot Encoder的制度。它不能标为“从零训练全部Actor”的传统基线；若研究该问题，需另外定义全Actor可训练基线。
- C–G的D冻结但前向保留计算图；真实Bridge及最终接口更新中E_f改变、D不变，WM在Actor更新期间不变。最终纯Foot reward Actor grad约`.00521`，有限非零；不代表WM动作导数已经准确。
- 修复Bridge默认包含Critic bootstrap的问题：有界Bridge显式 `--no_critic --paired_latent_weight 0`，仅Motion Reward经WM/BPTT更新E_f；D/E_m/Critic不更新。WM仍在独立WM步骤拟合真实数据，没有修改核心算法。
- 修复D/F继承Bridge校准后WM的混杂：必须 `--wm-source` 指向与B/C/G相同的Stage1 checkpoint，初始化时覆盖Bridge WM及其normalization。保留Bridge E_f/同语义ObsNorm，Quintic GoalNorm重建；Critic/optimizer新建。实际接口路径及逐权重CPU检查通过。
- Bridge还携带Foot ObsNorm统计，故C↔D比较的是完整Bridge适配流程，不仅仅是几层encoder权重的差异。
- 生成Standing与Walking两个分开的18条dry-run矩阵；同组Stage2目标分布/预算/seed/评估相同，**未执行矩阵**。Standing缺先验模板时Prior关闭，D/F Standing不能用于证明动态prior增益。
- E Action-MSE仍为可选未实现组，不伪造E实验。
- checkpoint保存Actor/WM/Critic、优化器、norm、合同、RNG、阶段计数及上游成本；原子写入；每个run保留源码快照和hash。Replay不保存，resume重采样并WM-only恢复。
- 实际12→24恢复：真实累计184条，新增92条；恢复期Actor参数、Critic不变，WM继续更新。日志、train.exitcode、稳定SwanLab ID及离线缓存均保留。

正式矩阵仅为审查预览：[Standing](../../results/s54/pre_ablation_20261105/standing_matrix_dry_run.json)、[Walking](../../results/s54/pre_ablation_20261105/walking_matrix_dry_run.json)。**其中来源仍是未合格pilot，不能直接替代合格Teacher启动正式消融。**

## 7. 成本、日志与SwanLab

| Run | 新增真实transition | checkpoint累计wall s | Actor更新 |
|---|---:|---:|---:|
| 12轮bootstrap | 92 | 20.90 | 6 |
| WM calibration | 6400 | 56.04 | 0 |
| Walking Stage1 200轮 | 10048 | 75.13 | 100 |
| Bridge接口12轮 | 92 | 18.19 | 6 |
| 12→24恢复接口 | 92 | 36.28（含原run约20.90） | 0次新增 |
| 最终共用WM/冻结D接口12轮 | 92 | 20.71 | 6 |

去重新增训练 **16816** 条，独立Newton评估 **5704** 条。上游成本按依赖计入单组报告，全局共享来源只计一次。checkpoint wall在最终保存前取样，保存结束/launcher退出时间另见runtime/launcher JSON，不能混用。Bridge上游真实10048条，最终接口上游10140条，单独计账。

WM校准Torch峰值565MiB、设备保存时总占用1535MiB；Teacher Torch峰值644MiB、设备1568MiB。Teacher含初始化/退出的端到端吞吐约126条真实transition/s；WM按保存结束wall约112条/s。这里不是纯仿真FPS；这两次没有全程平均GPU利用率测量，也没有做大env吞吐搜索。16env是本轮明确上限，无显存填满目标。原SuperMimic无PPO rollout_steps参数，UTD=.4决定每轮平均采样2.5个vector steps；H8是imagined BPTT 0.16s，不是环境rollout长度。

继续使用SwanLab项目/实验组 **Foothold_with_SuperMimic**。正式实验名固定为 `A_时间戳_seed54_standing` 至 `G_...`，Walking单列；Stage1与Bridge单独命名、计费。默认时间戳分类、稳定run ID、本地CSV/JSON/SDK缓存、独立tmux上传进程及最终同步；凭据仅沿用主机登录或环境变量，不写仓库。

本轮Stage1验证数据已实际上传：[Stage1 Walking pilot](https://swanlab.cn/@yagami_light/Foothold_with_SuperMimic/runs/a6f023c0ae544a94)，`final_sync.exitcode=0`。第一次请求offline却实际online，原因是既有tmux server环境未继承；已改为在上传job中显式传递 `SWANLAB_MODE`。之后resume和最终接口验证均为**offline/finished**、upload exit0，证明本地缓存链路修复。没有把本轮验证称为正式A–G实验。

## 8. 实际命令与复现

Python：`/home/rm/workspace/IsaacLab_3_0/.venv/bin/python`。本轮完整后续命令在 [actual_commands.txt](../../results/s54/pre_ablation_20261105/actual_commands.txt)，初始bootstrap/calibration的精确命令、参数与源码快照分别在各run的 `launch.json / args.json / source_snapshot/`。核心命令如下，已执行，原run名受保护，重跑应换新名字：

```bash
export S54_PYTHON=/home/rm/workspace/IsaacLab_3_0/.venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1

$S54_PYTHON tools/prepare_s54_motion_prior.py --dry-run
$S54_PYTHON tools/prepare_s54_motion_prior.py --out results/s54/motion_prior_stage0_verified
$S54_PYTHON tools/run_s54_motion_prior.py --stage stage1 --run s54_preablation_bootstrap_20261105 --execute
$S54_PYTHON tools/run_s54_motion_prior.py --stage stage2 --group C \
  --source logs/s54_preablation_bootstrap_20261105/last.pt \
  --profile configs/motion_prior/calibration_bounded.json --run s54_preablation_wm_20261105 --dry-run
# 同一有限命令将 --dry-run 改 --execute 后实际执行了80轮。
$S54_PYTHON tools/evaluate_s54_motion_prior.py \
  --checkpoint logs/s54_preablation_wm_20261105/calibration_before.pt \
  --out results/s54/pre_ablation_20261105/standing_before \
  --durations 2 5 10 --seeds 54 55 56 --reset_noise --viz none
$S54_PYTHON tools/compare_s54_wm_calibration.py \
  --before logs/s54_preablation_wm_20261105/calibration_before.pt \
  --after logs/s54_preablation_wm_20261105/last.pt \
  --evaluation results/s54/pre_ablation_20261105/standing_before \
  --out results/s54/pre_ablation_20261105/wm_comparison.json
$S54_PYTHON tools/run_s54_motion_prior.py --stage stage1 \
  --profile configs/motion_prior/stage1_bounded.json \
  --run s54_preablation_teacher_20261105 --execute --swanlab-mode offline
$S54_PYTHON tools/evaluate_s54_motion_prior.py \
  --checkpoint logs/s54_preablation_teacher_20261105/last.pt \
  --manifest results/s54/motion_prior_stage0_verified/manifest.json --split train \
  --out results/s54/pre_ablation_20261105/teacher_train_windows \
  --durations 5 10 --seeds 54 55 56 --start_seconds 0 30 60 --reset_noise --viz none
```

今后需重做整套有界检查时，用下列统一入口；它自动创建tmux，串行GPU检查/训练，创建新时间戳目录，**不会启动正式A–G**：

```bash
STAMP=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)
bash tools/run_s54_pre_ablation_bounded.sh start "$STAMP"
bash tools/run_s54_pre_ablation_bounded.sh status "$STAMP"
tmux attach -t "s54_preablation_$STAMP"
# Ctrl-b d 断开tmux；SSH断开不终止任务。重新登录后用同一个STAMP attach。
bash tools/run_s54_pre_ablation_bounded.sh stop "$STAMP"
```

每个训练进程限300s（smoke profile240s），训练自身预留30s，外部watchdog在deadline前15s发TERM，到deadline强制终止；保存与退出计入预算。进程中断后保留完整checkpoint；同任务resume必须新run名。例如已实际执行过12→24恢复，重跑用新名称：

```bash
$S54_PYTHON tools/run_s54_motion_prior.py --stage stage1 \
  --resume logs/s54_preablation_bootstrap_20261105/last.pt --resume_iterations 24 \
  --run "s54_preablation_resume_$STAMP" --execute --swanlab-mode offline
```

测试实际命令：

```bash
PYTHONPATH=tools $S54_PYTHON -m unittest \
  tools.test_pre_ablation tools.test_standing_diagnostics tools.test_motion_prior_transfer \
  tools.test_s54_launcher tools.test_curriculum_logging -v
```

58 PASS。修正了旧SwanLab测试与实现的契约不一致：训练exit7仍必须排空最后指标、finish(crashed)、持久化 `finished_training_failed` 与原失败原因；不要求上传器抛异常。没有把失败训练标成成功。

## 9. 失败记录与最小下一步

已保留的工程失败：第一次评估用了本版本不支持的 `--headless`，参数解析即失败，改 `--viz none` 后完成；第一次配对比较因前一步无summary而失败，随后成功重跑。默认MP4渲染因无ffmpeg失败，新增Pillow GIF输出后完成50帧/25FPS全身比较。历史失败日志未覆盖。

**下一步首先解决Walking Stage 1，而不是开展A–G：**

1. 从原Walking数据挑一段连续、低速、几何合格的约5秒窗口，保留原时间/原clip身份；先做reference-PD与learned-Motion真实Newton可执行性对照。候选如10–14.98s，平均水平速度约.161m/s、p90约.354m/s；只是候选，尚未通过动力学验收。完整clip速度中位数约.604m/s、p95约1.780m/s，直接混合全范围会增加早期难度。详见 [候选窗口](../../results/s54/pre_ablation_20261105/walking_window_candidates.json)。
2. 对该窗口做明确限样本/限时间的Stage1过拟合pilot，保持Motion reward、动作接口及物理不变，分别验收Walking WM1/4/8误差和真实5/10秒跟踪；不能只延长WM-only后宣告Teacher成熟。当前脚本还未提供固定训练frame-window开关，应先做这一最小采样改动再运行，不能直接重跑相同配置期待通过。
3. 窗口能闭环跟踪后逐步扩大Walking分布，形成真正可复用的同源WM+Decoder。补独立Walking holdout；Standing/Turning数据缺口单列，不把Dance当Walking泛化。
4. Standing诊断分支补逐接触点位置/受力、heel/toe压力分布后再做WM校准；保留本轮负/正踝偏置与历史best阳性对照。它为Foot阶段提供物理检查，不替代Walking Teacher。
5. 合格Teacher、对应WM Gate及数据/评估门槛通过后，才创建正式A–G实验，沿用同一SwanLab组、按A–G+时间戳命名与同步。

本轮已建立可信的失败证据和历史站立阳性对照，尚未建立合格的Walking Stage1 Teacher；因此最终判断仍为 **BLOCKED**。
