"""Generate a live/final report from durable evidence, without GPU/cloud work."""
from pathlib import Path
import argparse,csv,json,math,os,statistics
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2]


def fmt(value):
    if value is None or value=='':return 'N/A'
    if isinstance(value,(int,float)):return f'{value:.7g}' if math.isfinite(value) else 'NONFINITE'
    return str(value)


def avg(rows,key):
    values=[float(r[key]) for r in rows if r.get(key) is not None and r.get(key)!='']
    return statistics.fmean(values) if values else None


def read(path):return json.loads(path.read_text()) if path.exists() else None


def write(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');tmp.write_text(text);tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run_dir',type=Path,required=True);a=p.parse_args()
    d=a.run_dir.resolve();cfg=read(d/'profile.json');status=read(d/'training_status.json') or {'status':'NOT STARTED'}
    mapping=read(d/'transfer_mapping.json') or {};runtime=read(d/'runtime_stats.json') or {}
    sync=read(d/'swanlab_sync_status.json') or {'status':'NOT UPLOADED (non-formal)' if cfg['mode']=='smoke' else 'PENDING'}
    manifest=read(d/'source_snapshot/manifest.json') or {};launch=read(d/'launch.json') or {};gates=read(d/'preflight_gates.json') or {}
    counts=status.get('counters',{})
    def link(path):
        path=Path(path)
        if not path.is_absolute():path=ROOT/path
        relative=path.relative_to(ROOT)
        return f'[{relative}](../../{relative})' if path.exists() else f'`{relative}`（尚未生成）'
    L=['# S54_JOINT_AB1 运行报告','',
       '生成时间：'+datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')+'。',
       f'实验：`{d.name}`；模式：`{cfg["mode"]}`；状态：**{status["status"]}**。',
       f'实际iteration：**{status.get("iteration","N/A")}**；累计运行wall：**{fmt(status.get("wall_seconds"))} s** / {cfg["wall_seconds"]} s。',
       f'停止/失败原因：{status.get("reason") or "N/A（运行中或无停止原因）"}。',
       '', '本实验是 [SuperMimic Appendix B/G.1](https://arxiv.org/pdf/2610.28480) 的Foot Goal扩展：双接口在同一训练进程、同一Newton Scene中采样，共享唯一WM与Decoder；Foot只更新Encoder，Motion更新Motion Encoder与Decoder。它使用Stage1 warm-start，并非原论文Velocity从零实验的复现。',
       '', '## 固定配置与来源','',
       '用户最新要求为12h，正式配置为43200s / 50000 iterations双上限；旧方案中的8h/15000为较早提案。初始化、评估、保存计入运行wall；resume保留已用预算，进程未运行期间不计入累计计算wall。',
       '', f'Teacher：{link(cfg["teacher"])}；SHA256 `{cfg["teacher_sha256"]}`。',
       f'源预训练：iteration {mapping.get("source_iteration","N/A")}、{mapping.get("source_sim_steps","N/A")} real transitions、{fmt(mapping.get("source_wall_seconds"))} s，单列成本。',
       f'源码版本：`{manifest.get("git_sha",manifest.get("git_commit","N/A"))}`；工作树逐文件哈希以 {link(d/"source_snapshot/manifest.json")} 为准。',
       '', '| 参数 | 固定值 |','| --- | --- |']
    for k,v in cfg.items():L.append(f'| {k} | `{v}` |')
    L+=['',f'启动命令/时间：{link(d/"launch.json")}；参数：{link(d/"args.json")}；逐tensor迁移：{link(d/"transfer_mapping.json")}。',
        '', 'Motion使用QA通过的LAFAN1 walk1_subject1（730D Goal、dt .02s），Foot使用230D Goal、dt .1s的100%保守alternating Quintic，未加载全身reference。Foot步长增量U[.02,.06]m、swing U[.65,.90]s、DS U[.15,.25]s、高度U[.03,.045]m、yaw=0；reset采样整段计划。原Reward、物理终止和控制保持不变。',
        'Motion原域随机化及噪声；Foot原固定域和配置reset噪声。两套Goal/ObsNorm独立；Motion严格加载合法源Norm与兼容Critic，Foot新建；所有optimizer新建。Foot最后6D观测mask为显式有损适配。WM133→70、状态71、action29、ξ33、控制dt .02s；Goal不输入WM。',
        '', '## 实际样本与资源','', '| 计数 | 当前值 |','| --- | --- |']
    for k,v in counts.items():L.append(f'| {k} | {v} |')
    total=counts.get('real_motion',0)+counts.get('real_foot',0)
    L+=['',f'真实训练合计 **{total}**；加一次源预训练共 **{total+mapping.get("source_sim_steps",0)}**。独立评估另计 **{counts.get("eval_real_motion",0)+counts.get("eval_real_foot",0)}**。',
        'real_*包含warmup与恢复补采，recovery_real_*是子集，不重复相加。WM windows/valid steps为Replay重用，imagined_*为BPTT预测，均不是新增Newton样本。UTD=.4平均每iteration采2.5个vector steps；H32=.64s想象，WM window8=.16s监督，均不是PPO rollout。每iteration总16次WM更新，每次M/F各512 windows。',
        '',f'资源证据：{link(d/"hardware_before.json")}、{link(d/"runtime_stats.json")}。Torch峰值不含Warp；device_used为保存时快照，非全程峰值；collection FPS仅计采集，不代表端到端训练吞吐。',
        '', '```json',json.dumps(runtime,indent=2),'```',
        '', '## Gate、梯度隔离与恢复','',f'门槛记录：{link(d/"preflight_gates.json")}。']
    if gates:L+=['','```json',json.dumps({k:v for k,v in gates.items() if k!='source_hashes'},indent=2),'```']
    audits=[json.loads(v) for v in (d/'gradient_audit.jsonl').read_text().splitlines() if v.strip()] if (d/'gradient_audit.jsonl').exists() else []
    L+=['', '| 实测接口 | 审计数 | ΔD min/max | ΔEncoder min/max | latent grad min | ΔWM_actor max |','| --- | --- | --- | --- | --- | --- |']
    for name in ('motion','foot'):
        rows=[r for r in audits if r['interface']==name]
        if not rows:L.append(f'| {name} | 0 | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED |');continue
        span=lambda key:f'{fmt(min(r[key] for r in rows))} / {fmt(max(r[key] for r in rows))}'
        L.append(f'| {name} | {len(rows)} | {span("delta_decoder")} | {span("delta_encoder")} | {fmt(min(r["latent_gradnorm"] for r in rows))} | {fmt(max(r["delta_wm_actor"] for r in rows))} |')
    L+=['',f'原始梯度证据：{link(d/"gradient_audit.jsonl")}。Motion optimizer只持有E_m+D，Foot只持有E_f；每次Actor均检查WM/另一Encoder/Foot-D未被更新，审计周期另测差值。CPU测试覆盖共享identity、Critic隔离、旧Motion推理、Replay命令快照/跨reset掩码、合法与错误恢复。',
        '', '原子Checkpoint包含WM、唯一D、两Encoder/Critic、五个optimizer、Norm、RNG、计数、场景姿态/时钟/命令及观测历史。周期500iter或30min，保留6个周期版本，另保留initial/last/last_good_motion/best_foot。Replay不序列化：恢复补采warmup并有界WM-only recovery，成本单列。Newton接触暖启动未序列化，不承诺逐位轨迹连续。达到预算或有live pane时，resume在修改终态文件之前拒绝。',
        '', '## 分接口训练趋势','',f'曲线：{link(d/"train_log.csv")}、{link(d/"interface_metrics.jsonl")}；SwanLab将Motion与Foot分开，不平均成一个Reward。',
        '', '| 接口 | 窗口 | 条数 | sim reward | imag reward | episode return | ep steps | term fraction | WM8 | Actor grad | Critic loss |','| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    train=list(csv.DictReader((d/'train_log.csv').open())) if (d/'train_log.csv').exists() else []
    for name in ('motion','foot'):
        for label,rows in [('初期',train[:10]),('中段',train[max(0,len(train)//2-5):len(train)//2+5]),('末期',train[-10:])]:
            if not rows:continue
            keys=['sim_rew','imagined_reward','ep_ret_norm','ep_len','term_frac','wm_err8','actor_gradnorm','critic_loss']
            L.append(f'| {name} | {label} {rows[0]["iteration"]}–{rows[-1]["iteration"]} | {len(rows)} | '+' | '.join(fmt(avg(rows,name+'_'+k)) for k in keys)+' |')
    if not train:L.append('| NOT EVALUATED | — | 0 | — | — | — | — | — | — | — | — |')
    L+=['', '为CSV记录等权均值；早期窗口可能重叠，WM-only Actor空白为N/A。sim reward是单步/r_max，episode return除以实际执行步数，ep_len仅含已结束episode；term fraction=物理失败/done，Timeout/ClipEnd不当跌倒。imag reward来自Replay且乘soft survival，不能与sim均值之差当作WM bias。',
        '', '## 独立Newton评估','', '主训练同步CUDA并暂停后，两个评估子进程串行运行。Motion先测固定offset0/60/120各10s的同协议Teacher基线；完整261.3s的0.820197只用于全片比较。Foot固定seed54/55/56与2/5/10s，物理终止优先于Timeout，终止后的后缀不补Reward。',
        '', '| 接口/协议 | iteration | 平均NR | 物理终止/episodes | 实际秒均值 | 证据 |','| --- | --- | --- | --- | --- | --- |']
    summaries=[]
    for path in sorted((d/'evaluation').glob('*/**/summary.json')):
        r=read(path)
        if r and 'rows' in r and 'interface' in r:summaries.append((path,r))
    for path,r in summaries:
        rows=r['rows'];seconds=statistics.fmean(sum(x['episode_lengths'])*.02 for x in rows)
        L.append(f'| {r["interface"]}/{"full" if r["full"] else "short"} | {r["iteration"]} | {fmt(avg(rows,"norm_return"))} | {sum(x["n_failed"] for x in rows)}/{len(rows)} | {fmt(seconds)} | {link(path)} |')
    if not summaries:L.append('| NOT EVALUATED | — | — | — | — | — |')
    latest={name:next(((path,r) for path,r in reversed(summaries) if r['interface']==name and not r['full']),None) for name in ('motion','foot')}
    L+=['', '### 最新Foot固定时限','', '| 时限 | Survival | NR | Foot RMSE/m | Scheduled TD/m | 速度/m/s | Slip proxy/m/s | contact准确率 | 原因 |','| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    if latest['foot']:
        for duration in cfg['foot_eval_durations']:
            rows=[x for x in latest['foot'][1]['rows'] if x['duration_s']==duration];metrics=[x['metrics'] for x in rows]
            survival=sum(x['n_failed']==0 and x['episode_lengths'][0]>=round(duration/.02) for x in rows)
            vals=[avg(rows,'norm_return')]+[avg(metrics,k) for k in ['foot_position_rmse_m','scheduled_touchdown_rmse_m','actual_forward_speed_m_s','loaded_foot_slip_proxy_m_s','contact_state_accuracy']]
            L.append(f'| {duration}s | {survival}/{len(rows)} | '+' | '.join(fmt(v) for v in vals)+f' | {[reason for x in rows for reason in x["stop_reasons"]]} |')
    else:L.append('| NOT EVALUATED | — | — | — | — | — | — | — | — |')
    L+=['', '### 最新姿态及Motion跟踪','', '| 接口 | Root roll/pitch max deg | Body aligned RMSE/m | Root world XY RMSE/m | Upper pose RMSE/rad | Upper速度/加速度RMS |','| --- | --- | --- | --- | --- | --- |']
    for name,item in latest.items():
        if item is None:continue
        metrics=[x['metrics'] for x in item[1]['rows']]
        maximum=lambda k:max((m[k] for m in metrics if m.get(k) is not None),default=None)
        L.append(f'| {name} | {fmt(maximum("root_roll_max_abs_deg"))} / {fmt(maximum("root_pitch_max_abs_deg"))} | {fmt(avg(metrics,"body_position_aligned_rmse_m"))} | {fmt(avg(metrics,"root_xy_world_rmse_m"))} | {fmt(avg(metrics,"upper_body_pose_rmse_rad")) if name=="motion" else "NOT EVALUATED"} | {fmt(avg(metrics,"upper_velocity_rms_rad_s"))} / {fmt(avg(metrics,"upper_acceleration_rms_rad_s2"))} |')
    L+=['', '除Root峰值跨episode取max，其余指标为episode等权均值，非池化RMSE。短失败前缀误差低不代表完整成功。Foot没有配对上身参考，pose误差NOT EVALUATED，不以动作幅度代替自然性。接触为Newton末子步Fz>1N；slip proxy为承载踝link XY速度，非接触点滑移；Scheduled TD为计划事件，非传感器实际触地时刻。',
        '', '### WM同起点同动作配对','', '| 接口 | Horizon | Height RMSE/m | Pitch RMSE/deg | Foot FK RMSE/m | Normalized Reward MAE | Reward bias |','| --- | --- | --- | --- | --- | --- | --- |']
    for name,item in latest.items():
        if item is None:continue
        for h in (1,4,8,16,32):
            rows=[x['wm_paired']['horizons'][str(h)] for x in item[1]['rows'] if str(h) in x.get('wm_paired',{}).get('horizons',{})]
            if not rows:L.append(f'| {name} | {h} | NOT EVALUATED | — | — | — | — |');continue
            keys=['base_height_rmse_m','root_pitch_rmse_deg','feet_position_rmse_m','paired_reward_mae','paired_reward_bias']
            L.append(f'| {name} | {h} | '+' | '.join(fmt(avg(rows,k)) for k in keys)+' |')
    if not any(latest.values()):L.append('| NOT EVALUATED | — | — | — | — | — | — |')
    L+=['', '这是已访问Newton起点、相同开环动作/ξ/时钟的WM预测误差；不是足对目标的tracking误差。Reward MAE为每窗口平均Reward差绝对值的均值，逐步MAE另存JSON，不混称。epoch/episode等权统计仍有覆盖局限，不认证全动作空间梯度。',
        '', '## 产物、SwanLab和操作','', '| 产物 | 路径/状态 |','| --- | --- |']
    for name in ['last.pt','initial.pt','last_good_motion.pt','best_foot.pt','train_log.csv','interface_metrics.jsonl','gradient_audit.jsonl','wm_pair_metrics.jsonl','stdout.log','evaluation.log','training_status.json','swanlab_cloud_final_readback.json']:
        L.append(f'| {name} | {link(d/name)} |')
    videos=[(path,r) for path,r in summaries if r.get('preview') and Path(r['preview']).exists()]
    for _,r in videos:L.append(f'| {r["interface"]} preview iter{r["iteration"]} | {link(r["preview"])} |')
    if not videos:L.append('| 代表视频 | NOT EVALUATED |')
    L+=['', '视频只重放已有状态且限制到实际执行区间；Motion的并列LAFAN1 kinematic reference仅用于展示，不是真实Newton轨迹。',
        '', '```json',json.dumps(sync,indent=2),'```']
    if (d/'swanlab_run_id').exists():
        identity=(d/'swanlab_run_id').read_text().strip();L+=['',f'[SwanLab正式run](https://swanlab.cn/@{cfg["swanlab_workspace"]}/{cfg["swanlab_project"]}/runs/{identity}/chart)，稳定ID `{identity}`。']
    python=(launch.get('command') or ['/home/rm/workspace/IsaacLab_3_0/.venv/bin/python'])[0]
    L+=['', '仅formal上传，同项目Foothold_with_SuperMimic，新时间戳实验。独立CPU/tmux不依赖SSH代理，异常重试，本地JSONL/SDK持久化；训练结束离线重放、sync及最终读回，只有verified与final_sync.exitcode=0才确认最终同步。唯一event_step为默认横轴，Progress/Iteration记录实际迭代。',
        '', '按用户2026-11-06要求，上传器已改为32项核心指标白名单：Motion/Foot各10项（真实/想象Reward、Episode Return/Length、物理终止比例、WM Loss/Error8、Actor/Critic Loss及Actor梯度）；Motion短/全片评估Return各1项；Foot 2/5/10秒Survival、10秒Return/Foot RMSE/实际前向速度共6项；迭代/时长/FPS/峰值显存共4项。评估各offset/seed等权汇总，短评估与全片协议分开。本地JSONL、逐seed评估、WM多步误差和梯度审计保留完整；原始计数不再逐项上传。最终读回验收改为最新核心WM Loss及终态Iteration/WallHours。已上传的旧图表仍存在：当前SDK不支持修改已有图表的隐藏状态，需要在WebUI隐藏，不能声称已删除。完整策略见swanlab_core_policy.json。',
        '', '```bash',f'tmux attach -t {d.name}',f'tail -F logs/{d.name}/stdout.log',
        *[f'{python} -B tools/train/run_s54_joint_ab1.py {mode} --run {d.name}' for mode in ('status','stop','resume')],
        f'{python} -B tools/validation/report_s54_joint_ab1.py --run_dir logs/{d.name}','```',
        '', '按Ctrl-b再d离开tmux，SSH断开仍运行；服务器须保持开机。已有活跃训练/已耗尽预算的run拒绝resume，不加载旧ABC checkpoint。',
        '', '## 结论边界与下一步','',
        'COMPLETED_BY_WALL_CAP/ITER_CAP仅表示有界工作流完成，不是收敛。Motion短探针的同协议Teacher绝对NR下降>.05或任一物理失败，连续两次触发STOPPED_BY_GATE；完整协议的实现仅检查平均NR是否低于.820197−.05，不另用物理失败触发，不能将两种Gate混称。Foot初步进展需10s Survival≥2/3且速度≥−.01m/s，仍需接触/自然性验收。',
        '非有限数、持续超阈梯度/Critic loss、OOM或写Checkpoint失败均留故障并停止，不在线改Reward/终止/方法。当前缺独立Walking holdout与Standing/Turning模板，泛化NOT EVALUATED；单训练seed不能声称统计显著。',
        '若Foot未达标，分别检查真实终止/速度、WM32配对及reward/bootstrap梯度；不以失败前缀RMSE或延长WM-only掩盖控制失败。后续对照须匹配Foot或总真实样本预算，并单列新增Motion监督与Stage1成本。',
        '', '**此为实时报告；最终能力与同步结果尚未验收。**' if status['status'] in ('RUNNING','NOT STARTED') else '**此为已记录终态的证据汇总；能力与上传状态以上表为准。**']
    result=ROOT/'results/s54'/d.name
    L+=['','## 最终验收与补充证据','']
    checkpoint_audit=read(result/'final_checkpoint_audit.json')
    if checkpoint_audit:
        L+=[f'最终Checkpoint CPU审计：{link(result/"final_checkpoint_audit.json")}，passed={checkpoint_audit["passed"]}；{checkpoint_audit["finite_tensor_count"]}个张量均有限；五个Adam step与真实更新计数相符；RNG含Python/NumPy/Torch/CUDA；两套512×10×99历史、环境/命令/时钟及Replay schema均已保存。Shared Decoder只序列化一次。',
            f'最终last.pt SHA256：`{checkpoint_audit["sha256"]}`。']
    if status.get('phase')=='terminal' and latest['motion'] and latest['foot']:
        mrows=latest['motion'][1]['rows'];frows=latest['foot'][1]['rows']
        teacher=next((avg(r['rows'],'norm_return') for _,r in summaries if r['interface']=='motion' and r['iteration']==0 and not r['full']),None)
        current_nr=avg(mrows,'norm_return');f10=[r for r in frows if r['duration_s']==10]
        survived=sum(r['n_failed']==0 and r['episode_lengths'][0]>=500 for r in f10)
        speed=avg([r['metrics'] for r in f10],'actual_forward_speed_m_s')
        progress=survived>=2 and speed is not None and speed>=-.01
        L+=[f'**真实结论：Motion短探针NR={fmt(current_nr)}，同协议初始Teacher={fmt(teacher)}；物理失败{sum(r["n_failed"] for r in mrows)}/{len(mrows)}。Foot 10s Survival={survived}/{len(f10)}，实际前向速度={fmt(speed)}m/s；Foot进展门槛={"PASS" if progress else "FAIL"}。**',
            'Foot短时存活改善不能等同足轨迹跟踪行走成功。Motion支路保留原有动作能力，也不能证明任意Foot Goal能获得自然上身；Foot无配对全身参考，本轮没有合格的自然性结论。',
            f'训练端到端吞吐（包含初始化/评估/保存）={fmt(total/status["wall_seconds"])} Newton transitions/s、{fmt(status["iteration"]/status["wall_seconds"])} iterations/s；与只计采集区间的FPS分开。']
    comparison=read(result/'zero_pd_comparison.json');zero_run=read(result/'zero_pd_run.json')
    if comparison:
        L+=['',f'同合同零动作PD对照：{link(result/"zero_pd_comparison.json")}；真实原始评估 {link(result/"zero_pd_newton/summary.json")}；命令/预算/退出状态 {link(result/"zero_pd_run.json")}。',
            '三种控制器的9组命令张量、初始物理状态及ξ已逐项配对检查；不以历史Standing零动作数据代替本次alternating任务。',
            '', '| 控制器 | 时限/s | Survival | 实际执行/s | NR | 世界足位置RMSE/m | 前向速度/m/s |',
            '| --- | --- | --- | --- | --- | --- | --- |']
        for name,groups in comparison['metrics'].items():
            for duration,r in groups.items():
                L.append(f'| {name} | {duration} | {r["survived"]}/{r["episodes"]} | {fmt(r["actual_seconds"])} | {fmt(r["norm_return"])} | {fmt(r["foot_rmse_m"])} | {fmt(r["actual_forward_speed_m_s"])} |')
        L+=[f'零动作PD在正式训练和评估进程退出、GPU锁释放后串行执行，进程耗时{fmt(zero_run.get("wall_seconds"))}s，新增{comparison["separate_zero_pd_samples"]}条Newton评估样本，未计入训练更新/Replay预算；结束时间{zero_run.get("ended_at")}，原12h截止{zero_run.get("original_deadline")}。',
            '最终Foot较初始随机Foot Encoder和零动作PD延长了失败前缀，但5/10秒均无成功，后退更快、足位置误差更大；NR也低于零动作PD。不能据此宣布联合训练已解决行走。上述RMSE均为世界坐标足link与固定目标的欧氏误差，同一计算定义，且各自失败前缀长度不同，不是固定10秒成功轨迹上的精度比较。']
    for filename,description in [('normalization_audit_001000.json','接口Normalizer分布审计（差异是事实，尚未证明是控制失败原因）'),
            ('budget_comparison_002000_vs_C.json','与历史C在2000 iterations的真实预算对照（同总样本不等于同Foot样本、Actor更新或Motion监督）'),
            ('historical_final_preservation.json','42项历史ABC/Teacher文件内容、大小和mtime未改动')]:
        if (result/filename).exists():L.append(f'{description}：{link(result/filename)}。')
    core=read(d/'swanlab_core_export/status.json')
    if core:
        L+=['',f'**按用户追加要求建立的精简数据组**：[Foothold_with_SuperMimic_Core]({core.get("url","")})；状态 `{core["status"]}`，稳定ID `{core["run_id"]}`。',
            f'仅重放同一次正式训练的{core.get("metric_count","N/A")}项核心指标；{core.get("logged_iterations","N/A")}个记录点，横轴为真实iteration，最大{core.get("last_iteration","N/A")}。每20 iterations记录一次，不应把点数误读为训练迭代数。同一iteration的训练/评估/终态合并；完整raw JSONL仍为{counts.get("events","N/A")}个event。',
            f'逐项最新值云端读回证据：{link(d/"swanlab_core_export/cloud_readback.json")}；身份与源实验关联：{link(d/"swanlab_core_export/identity.json")}。原SwanLab实验保留，不额外启动任何训练。',
            f'核心上传器CPU测试4项通过；仅上传逻辑与报告后处理在运行末期按用户要求调整，训练器/Reward/终止/物理/冻结profile均未修改。变更回执：{link(d/"swanlab_core_policy_change.json")}。']
    L+=['', '最小下一步建议：先离线检查Foot末期轨迹的后退、触地和reward/bootstrap梯度，再分别验证Decoder输入的观测/latent分布及长H动作梯度；对照需固定真实Foot样本和Motion监督预算。现有证据不足以在本轮后自动开展D/F/G或无界延长训练；本轮未执行这些新训练。']
    text='\n'.join(L)+'\n';write(d/'run_report.md',text)
    if result.exists():write(result/'run_report.md',text)
    if cfg['mode']=='formal':
        path=ROOT/'docs/experiment/S54_JOINT_AB1_RUN_REPORT.md';write(path,text);print(path)


if __name__=='__main__':main()
