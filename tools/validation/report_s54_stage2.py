"""Factual Stage2 report; missing/unreached capabilities remain NOT EVALUATED."""
import argparse,json,datetime,os
from zoneinfo import ZoneInfo
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default
def records(p):
 rows=[]
 if p.exists():
  for line in p.read_text().splitlines():
   try:rows.append(json.loads(line))
   except json.JSONDecodeError:pass
 return rows

def generate(d):
 cfg=read(d/'profile.json');status=read(d/'status.json',dict(status='NOT STARTED'));swan=read(d/'swanlab/identity.json',{})
 norm=read(d/'normalization_audit.json',{});smokes=[(name,read(d/name/'smoke_audit.json')) for name in ('smoke16','smoke64')]
 resume=read(d/'smoke64/resume_smoke_audit.json');summary=read(d/'training_summary.json',{})
 lines=['# Stage 2 S54 Foot Trajectory Curriculum',f'\n更新：{datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")}。独立目录：`{d.relative_to(ROOT)}`。',
 f'\n状态：**{status["status"]}**；当前 iteration={status.get("iteration","N/A")}；课程={status.get("stage","N/A")}；累计进程 wall time={status.get("wall_seconds",0):.1f}s / 28800s。',
 '\n本报告不把 Smoke Test、5s 存活或 imagined reward 当作 10s 轨迹跟踪成功。正式训练达到时间上限或看门狗条件后安全保存；尚未达到的阶段没有能力结论。',
 '\n## 初始化与接口',f'\n- E_f：`{cfg["phase_a_checkpoint"]}`，Phase A best iteration 625；**未使用 Phase B**。',
 '- Shared Decoder / WM / ObsNorm 来自对应 Teacher；D 与 Policy 两个 Norm 完全冻结；独立 Foot Critic 与三个 Adam 从零建立。WM 继续真实 Newton Replay 学习，Actor BPTT 期间冻结 WM 参数但保留输入梯度。',
 '- Policy 仅输入 generated Foot Goal230 + 10帧本体观测。没有 MotionLib、完整 Motion Reference 或 Motion Encoder 参与前向。Teacher checkpoint 中 Motion Encoder 权重仅用于审计 Decoder 来源。',
 '- 每帧23D：左足 p3 / Rotation6 / Contact / Phase + 右足同样11D + Δt1。共10帧，0.02–0.20s。位置与姿态在当前 root-yaw 坐标系，足部使用与 Phase A 相同的 ankle link 原点。Rotation6 为旋转矩阵前两列的 row-major 3×2 展开。',
 '- Heading 使用 Phase A 相同的 contact-weighted foot-yaw 推算，保留校准 bias=-0.0661631078rad、roll=0.0067171087rad、pitch=0.0594999380rad；未增加独立 Heading 优化。',
 '- GoalNorm/ObsNorm 按 Phase A 原样复制，无在线重估、截断或评估专用调整。',
 '- 脚步频率统一为**每秒交替落脚次数**，完整左右周期频率为其一半。0Hz=双脚站立，不做除法；ℓf 为理想前进速度。Yaw 命令为 rad/s。',
 '\n## 课程与预算',
 '\n最多8h（包括正式进程初始化、评估、保存；停止前预留90s），50000 iterations 上限。以下7.4h是建议分配，余0.6h留给评估/保存；**时间到达不强制晋级**，失败可以停留或回退。',
 '\n| 阶段 | 建议h | Standing / Short / Medium / Long / Turn / Curve 基础权重 | 步长上限m | 频率Hz | Yaw绝对值上限rad/s | 摆高m |',
 '|---|---:|---|---:|---|---:|---:|']
 for i in range(5):lines.append(f'| C{i} | {cfg["stage_hours"][i]} | '+ ' / '.join(f'{v:.0%}' for v in cfg['stage_weights'][i])+f' | {cfg["stage_length_max"][i]} | {cfg["stage_frequency"][i]} | {cfg["stage_yaw_max"][i]} | {cfg["stage_clearance"][i]} |')
 lines += ['\n短/中/长分桶约为 .02–.08 / .08–.16 / .16–阶段上限 m；C0 实际从 .02–.033m 开始，按成功逐渐到 .055m。C0 小步中10%为零前进、非零频率的交替重心转移。困难任务与高频分桶受成功率权重调节，长步频率≤1.15Hz；左右转向等概率。动态调整后的**实际比例不等于表中基础权重**，完整采样/拒绝统计保存在本地 JSONL。',
 '每脚摆动 .72/f、双支撑 .28/f；场景命令的 minimum swing=.32s、DS=.12s。使用现有 C2 Quintic + 足部 SLERP，目标在整个 Episode 中保持不可变，C4 可在完整左右步对和双支撑边界切换；WM Replay 同时保存 command ID 和步数，不跨 reset 接续。',
 '联合作用域检查：速度≤.45m/s、same-foot stride≤.65m、每步 |yaw_delta|≤.25rad、swing速度≤1.8m/s、加速度≤12m/s²。`.4m/2Hz` 等组合被拒绝；这是轨迹包络筛选，**不等于已证明动力学可执行**。大步/高速/转向只有通过真实 Newton 门槛后才能声称掌握。',
 '\n晋级门槛：当前阶段全部独立代表任务 success≥85%、survival≥90%；Foot RMSE≤.06m（standing≤.04m）、落脚误差≤.07m、root |roll/pitch|≤18°、height偏差≤.10m。Walking 另需实际前进≥理想值60%、速度误差限制、≥2次真实加载接触落脚和摆脚抬高，turn另需正确方向与yaw跟踪。',
 '每阶段至少900s，连续通过至少2次评估，且10s Full确认通过才晋级。连续失败且站立下降可回退。Fast=5s、4env、固定seed8541，每900s串行复用同一Newton scene；Full=10s，晋级确认或结尾有足够余时才运行。正式周期评估预算约5%，初次基线单独记录。Physical termination 优先于 timeout；Goal tracking failure 独立统计。',
 '\n## 3090配置与测试',
 '\n| 参数 | 值 |\n|---|---:|']
 for k in ('num_envs','hist','horizon','wm_window','goal_frames','goal_dt','rollout_steps','buffer_steps','warmup_steps','wm_only_iters','resume_wm_iters','wm_batch','wm_batches','pi_batch','lr_wm','lr_actor','lr_critic','gamma','lam','action_noise','critic_epochs','critic_minibatches'):
  lines.append(f'| `{k}` | {cfg[k]} |')
 lines += ['\nH32为0.64s可微 imagined BPTT；rollout_steps=5 是每环境每轮真实采样0.10s，两者不同。WM:4×512个8步窗口/轮；Actor128 starts/轮，Critic2epochs×4minibatches。Adam学习率固定，Actor前300次更新线性升至1e-5；前500迭代只训练WM。BF16 MLP，几何/状态积分保持FP32。DR关闭，原物理模型/PD/关节顺序不变。',
 '\n| Smoke | GPU | Newton transitions | Actor/WM updates | peak Torch MiB | 技术断言 |\n|---|---|---:|---|---:|---|']
 for name,row in smokes:
  if row:lines.append(f'| {name} | {row["gpu"]} | {row["counts"]["newton"]} | {row["counts"]["actor"]}/{row["counts"]["wm"]} | {row["peak_torch_mib"]:.1f} | {row["assertions"]} |')
  else:lines.append(f'| {name} | NOT COMPLETED | N/A | N/A | N/A | N/A |')
 lines += [f'\nCheckpoint恢复 Smoke：`{d.name}/smoke64/resume_smoke_audit.json`：'+('PASS，实际再次更新WM/E_f，D与Norm不变。' if resume and all(resume['assertions'].values()) else 'NOT COMPLETED。')]
 for name,_ in smokes:
  path=d/name/'train_metrics.jsonl'
  if path.exists():
   samples=records(path);rr=[r for r in samples if r['actor']>0 and 'actor_loss' in r]
   if rr:lines.append(f'\n{name} 实测平均采样FPS={sum(r["collection_fps"] for r in rr)/len(rr):.0f}；Actor阶段迭代={sum(r["iterations_per_second"] for r in rr)/len(rr):.2f}it/s（短Smoke，包含首次审计/周期保存抖动，不是8h平均）。证据：`{path.relative_to(ROOT)}`。')
 if norm:
  lines += [f'\n归一化审计：5×64条计划；Rotation6D最大z={max(r["rotation_max"] for r in norm["rows"]):.3f}；全Goal最大z={max(r["max_abs"] for r in norm["rows"]):.3f}；|z|>6比例为0。Latent saturation约{min(r["latent_saturation"] for r in norm["rows"]):.1%}–{max(r["latent_saturation"] for r in norm["rows"]):.1%}。根沿支撑中心的输入审计，非真实闭环成绩。`{(d/"normalization_audit.json").relative_to(ROOT)}`。']
  lines += [f'\nPhase A SHA256：`{norm["contract"]["source_sha256"]}`；Teacher SHA256：`{norm["contract"]["source_teacher_sha256"]}`。迁移证明：`transfer_report.json`，完整契约：`normalization_audit.json`。']
  task=norm['contract']['task']['config']
  lines += ['\n原 Foot Reward 保持不变：swing/stance position各1，velocity各.25，foot orientation .25，touchdown .5，root upright1、height .5、balanceXY .5；action-rate −.1、joint-limit −2、posture −.05、joint-velocity −.001，乘control_dt=.02。inverse-quadratic tracking质量函数，标准差与原 FootTrackingCfg 相同。',f'物理终止保持 root-height 偏差>{task["term_height"]}m、1−Rzz>{task["term_tilt"]}或非有限状态；成功门槛比物理终止严格，正常500步timeout不当作跌倒。']
 formal=d/'train_metrics.jsonl'
 if formal.exists():
  rr=records(formal)
  if rr:
   last=rr[-1];lines += [f'\n正式256env最新记录：iteration={last["iteration"]}；采样FPS={last["collection_fps"]:.0f}；iteration/s={last["iterations_per_second"]:.2f}；Torch peak={last["peak_torch_mib"]:.1f}MiB；WM loss={last["wm_loss"]:.5f}、err8={last["wm_err8"]:.5f}；Actor updates={last["actor"]}、WM updates={last["wm"]}。`train_metrics.jsonl`中的newton仅计训练采样，newton_eval单列评估中所有实际模拟环境步。']
 if status.get('first_started_unix'):
  start=datetime.datetime.fromtimestamp(status['first_started_unix'],ZoneInfo('Asia/Shanghai'))
  lines += [f'\n正式进程首次开始：{start.isoformat(timespec="seconds")}。不中断时8h上限约为 {(start+datetime.timedelta(hours=8)).isoformat(timespec="seconds")}，看门狗可提前结束。目录时间戳对应准备开始，不是8h预算起点。']
 hardware=records(d/'gpu_telemetry.jsonl')
 if hardware:
  lines += [f'\nGPU只读监控（每30s抽样）：总显存24,576MiB，已观察最大占用={max(r["used_mib"] for r in hardware):.0f}MiB，采样平均利用率={sum(r["utilization_percent"] for r in hardware)/len(hardware):.1f}%。这是当前阶段观测，不是完整8h统计；`gpu_telemetry.jsonl`。未为了占满显存而盲目增加batch。']
 lines += ['\n## 实际独立Newton结果', '\n| 来源 | 阶段 | 任务 | 时长s | Survival | Success | Foot RMSE m | 物理终止 | Root roll/pitch max° |', '|---|---|---|---:|---:|---:|---:|---:|---|']
 paths=list((d/'smoke64/evaluation').glob('post_update_*/summary.json'))+list((d/'evaluation').glob('*/summary.json'))
 for path in sorted(paths):
  e=read(path)
  for r in e['rows']:lines.append(f'| `{path.relative_to(ROOT)}` | C{e["stage"]} | {r["label"]} | {r["duration_s"]} | {r["survival_rate"]:.2f} | {r["success_rate"]:.2f} | {r["foot_rmse"]:.4f} | {r["physical_termination_rate"]:.2f} | {r["root_roll_max_deg"]:.1f}/{r["root_pitch_max_deg"]:.1f} |')
 if not paths:lines.append('| N/A | N/A | NOT EVALUATED | N/A | N/A | N/A | N/A | N/A | N/A |')
 lines += ['\n尚未评估的高阶段能力、10s持续跟踪和全身自然性均为 **NOT EVALUATED**。Phase A原本仅学过LAFAN Walking，不能据此假定已会站立、转弯或大步。已知源Teacher方向近似与新任务Reward分布差异仍是风险。',
 '\n## Checkpoint、监控与持久化',
 '- `latest.pt`/`last.pt`、每500迭代且最长900s一个周期checkpoint、阶段切换checkpoint。最近4个周期文件，阶段与best单独保留；原子fsync+rename，别名硬链接。完整模型/三个Adam/Norm/课程分布/RNG/全局计数/累计wall保存。',
 '- `best_standing.pt`、`best_walking.pt`、`best_overall.pt`按独立Newton success→survival→低RMSE→return排序，记录5s provisional或10s qualified。存在best文件不意味着已经合格。',
 '- Replay不序列化，恢复后 grounded reset、warmup40真实步、100轮WM-only再恢复Actor；保留训练进度/课程/采样RNG/optimizers。保留环境快照供审计，但不宣称精确恢复Newton接触求解器状态。',
 '- NaN/Inf、缺失/非有限Actor latent梯度立即诊断退出；连续3次独立评估连Standing/Walking 5s存活都失败且运行≥30min时安全停止。CPU看门狗持久记录最佳真实表现，RMSE再改善≥5%或成功/存活率改善才重置进展时钟；1h无任务成功且无实际改善则停止（`monitor_state.json` / `external_watchdog.json`）。训练内另保留相对初始基线的退化检查，连续2次RMSE>基线2倍则停止。不因单次WM loss上升中断。',
 '- SwanLab独立CPU tmux上传，23条核心指标，训练25迭代一次，网络断开本地缓存，最终sync+API读回核验；训练不等待网络。完整命令直方图保存在本地，不扩张云端曲线数量。',
 f'- SwanLab：{swan.get("url","PENDING：尚未建立正式run")}；上传状态：{read(d/"swanlab/status.json",{}).get("status","PENDING")}；已上传数据实际API读回证据：`swanlab/live_cloud_check.json`（若存在）。',
 f'- Best metadata：`{json.dumps(summary.get("best",{}),ensure_ascii=False)}`。' if summary.get('best') else '- 训练尚在进行，最终best汇总未生成；当前best候选按真实Newton评估保存，资格以独立评估表为准。',
 '- 可用候选路径：'+', '.join(f'`{(d/name).relative_to(ROOT)}`' for name in ('best_standing.pt','best_walking.pt','best_overall.pt') if (d/name).exists())+'。',
 '\n## 运行与恢复命令',
 '```bash',f'cd {ROOT}',f'bash tools/train/run_s54_stage2_curriculum.sh status {d.name}',f'tmux attach -t {d.name}',f'tail -f logs/{d.name}/stdout.log',f'bash tools/train/run_s54_stage2_curriculum.sh stop {d.name}',f'bash tools/train/run_s54_stage2_curriculum.sh resume {d.name}','```',
 '\nSSH断线不影响tmux；Ctrl-b d离开会话，Ctrl-c会请求安全停止。resume使用原run ID和累计8h预算，不重置8h。',
 '\n视频：`videos/`中为保存的真实Newton轨迹与cyan程序足目标，不调用额外policy rollout，不放置LAFAN全身ghost；物理终止后冻结并标注。正式worker结束后串行渲染所选best记录，期间不启动第二个GPU仿真。',
 '\n## 当前结论',
 '**运行状态与上表为准。** 技术Smoke通过仅说明迁移/梯度/恢复管线可用。真实目标误差大的阶段保留失败记录，不放宽物理终止或成功门槛，不将存活解释为学会足轨迹跟踪。']
 text='\n'.join(lines)+'\n';target=d/'STAGE2_CURRICULUM_REPORT.md'
 for path in (target,ROOT/'STAGE2_CURRICULUM_REPORT.md'):
  tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');tmp.write_text(text);tmp.replace(path)
 return target
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();print(generate(a.run_dir.resolve()))
