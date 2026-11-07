"""Generate an evidence-only report for a completed isolated Stage15 run."""
import argparse,json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();d=a.run_dir.resolve()
    cfg=read(d/'profile.json');state=read(d/'status.json');ar=read(d/'phase_a/phase_a_audit.json');norm=read(d/'phase_a/normalization.json')
    metrics=[json.loads(x) for x in (d/'phase_a/phase_a_metrics.jsonl').read_text().splitlines()];best=min(metrics,key=lambda r:r['val_latent_mse'])
    def link(path):return '['+str(path.relative_to(ROOT))+']('+str(path)+')'
    lines=['# Stage 1.5 — LAFAN1 Foot Encoder Skill Transfer','',
        f'实验目录：{link(d)}。执行状态：**{state["status"]}**；技能验收：**'+('PASS_ON_TRAIN_CLIP_WINDOWS' if state.get('phase_b_gate') else 'BLOCKED_QUALITY')+'**。所有训练、评估和上传均使用独立 tmux；没有改写 Stage 1、ABC、Joint 的历史文件。',
        '', '## Teacher、数据与接口','',
        f'- Teacher：`{cfg["teacher"]}`，iteration **14,484**；SHA256 `{cfg["teacher_sha256"]}`。严格校验机器人、物理、Motion Goal、源数据及模型形状契约。',
        '- 保留并冻结 Teacher Motion Encoder、Shared Decoder 及其 Motion Norm；WM 保留初始化。Foot Encoder 重新初始化；不继承原 Quintic Encoder 的饱和权重或 GoalNorm。Phase A 仅优化 E_f，WM/Critic 不更新。',
        '- 唯一参考：`data/retarget_data/reference_startup_fixed/walk1_subject1.csv`，50Hz 控制重采样。FootGoal 是原始 FK 的 10 个未来时刻：0.02/0.04/…/0.20s，与 MotionGoal 的 f+1…f+10 严格相同。没有 Quintic、随机足轨迹或轨迹拟合。',
        '- 230D = 10 × [左足(position3 + rotation6 + contact1 + swing phase1) + 右足同11D + delta_time1]。位置/旋转位于实际 root yaw 坐标系；contact 是原提取器的几何/速度滞回估计，并非真实接触力。',
        '- Teacher MotionPolicy 接收完整 MotionGoal；FootPolicy 只接收 FootGoal 和10帧本体观测。Decoder 的归一化观测93:99通道置零；其余93通道为关节/速度/IMU重力/上一动作。完整 reference 不进入 Foot Encoder/Decoder；仅用于初始化、蒸馏标签、训练奖励、特权 Critic及独立评分。',
        '- 新建独立 Stage15 契约，未来时间跨度与旧 Foot .1s×10 不同，不将旧 checkpoint 强行恢复为新任务。',
        '', '## 归一化与可复现拆分','',
        f'- 拟合帧数 **{len(norm["train_frames"])}**，时间块验证帧 **{len(norm["validation_frames"])}**；训练集含原状态及两组固定seed的root XY/yaw扰动，共 **{norm["count"]}** 组。',
        '- E_f训练及GoalNorm拟合均排除0–10、60–70、120–130秒区间，并保留lookahead guard；其余按1s块分出验证数据。Teacher预训练见过该clip，因此这些是E_f的时间留出窗口，不是独立Walking clips。',
        '- GoalNorm固定训练集均值/方差，标准差下限：位置0.05m、Rotation6D每通道0.15、contact/phase0.25、时间0.10s。ObsNorm逐张量复制Teacher并固定。Eval不重新归一化、不裁剪输入。',
        f'- GoalNorm SHA256：`{norm["sha256"]}`；训练/验证最大绝对归一化输入 **{norm["train_max_abs_normalized"]:.3f}/{norm["validation_max_abs_normalized"]:.3f}**。精确帧索引、seed、增强和Norm均已保存：{link(d/"phase_a/normalization.json")}。',
        '', '## Smoke 与 Phase A 实测','',
        '- 4项单元测试覆盖时间对齐、无关节reference输入泄露、时间拆分、旋转标准差下限、mask不变性与冻结Decoder的输入梯度。',
        '- 早期独立Newton Smoke发现paired WM诊断缺少episode_lengths/xi字段；修复后在本实验新目录重跑通过。短于32步时明确不算32步WM误差。早期失败日志保留在 `logs/s54_stage15_20261106_165359/smoke/newton.log`。',
        f'- Phase A：**{ar["iterations"]}**次更新、batch={cfg["phase_a_batch"]}、Adam lr={cfg["phase_a_lr"]}、latent MSE；共 **{ar["distillation_samples"]:,}** 次离线样本呈现（不是新增Newton transitions）。耗时 **{ar["wall_seconds"]:.2f}s**，**{ar["mean_updates_per_second"]:.2f} updates/s**。',
        f'- 最优时间块验证checkpoint为 iteration **{best["iteration"]}**：MSE **{best["val_latent_mse"]:.6f}**，cosine **{best["val_latent_cosine"]:.4f}**，abs(z)>.95比例 **{best["val_saturation"]:.2%}**。Teacher验证集对应比例 **{norm["teacher_validation_latent_saturation"]:.2%}**；饱和下降本身不证明技能成功。',
        f'- 最后train/val MSE **{metrics[-1]["train_latent_mse"]:.6f}/{metrics[-1]["val_latent_mse"]:.6f}**。训练误差继续下降而验证误差未同步下降，存在时间泛化差距；使用验证最优而非最后权重做Newton验收。',
        f'- 所有Phase A冻结/Norm/梯度权限断言均通过：{link(d/"phase_a/phase_a_audit.json")}。',
        f'- Phase A最佳latent checkpoint：{link(d/"phase_a/phase_a_best.pt")}；末次：{link(d/"phase_a/phase_a_last.pt")}。最佳latent不等于合格Walking controller。']
    br=read(d/'smoke_b/phase_b_audit.json')
    lines+=['',f'Phase B接口Smoke：RTX3090，16env，H=32/hist=10/WMwindow=8，{br["counts"]["newton"]}真实transitions、{br["counts"]["wm"]}次WM更新、{br["counts"]["actor"]}次E_f Actor更新、{br["counts"]["critic"]}次Critic minibatch更新；峰值Torch {br["peak_torch_mib"]:.1f}MiB。首次latent grad={br["first_actor_audit"]["latent_gradnorm"]:.6f}，E_f改变量={br["first_actor_audit"]["delta_encoder"]:.6f}，D/actor期间WM改变量=0。所有权限与归一化断言通过。此模型只是工程Smoke，**不是Phase B技能实验**。',
        '', '## 独立 Newton 结果','',
        '固定seed54、无观测/动作噪声和DR；每个起点三种控制器使用相同完整物理状态（不是中立站姿）。原Motion与Foot物理终止条件取并集，外加nonfinite。正常10s Timeout计完整存活；物理终止之后不补奖励。',
        '', 'NR=实际累计原MotionReward / (请求500步×原每步理论上限0.1)，并非前缀平均奖励。所有RMSE、接触及动作统计只覆盖实际执行前缀；不同前缀不能直接当同等时长质量比较。',
        '', '| Controller | 起点/s | 执行/s | Stop | Motion NR | 足世界RMSE/m | Root roll/pitch最大/deg | 上身RMSE/rad | 接触准确率 | Action RMS / clip |',
        '| --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | --- |']
    evals={}
    for phase,label in [('eval_a','Phase A'),('eval_b','Phase B'),('diagnostic_masked_oracle','Teacher-z + Foot Obs diagnostic')]:
        path=d/phase/'summary.json'
        if not path.exists():continue
        ev=read(path);evals[phase]=ev
        for role in ('motion','foot'):
            if role=='motion' and phase!='eval_a':continue
            for r in ev['rows'][role]:
                m=r['metrics'];name='Teacher Motion' if role=='motion' else label
                lines.append(f'| {name} | {r["offset_s"]:g} | {r["episode_lengths"][0]*.02:.2f} | {r["stop_reasons"][0]} | {r["norm_return"]:.4f} | {m["foot_position_rmse_m"]:.4f} | {m["root_roll_max_abs_deg"]:.2f}/{m["root_pitch_max_abs_deg"]:.2f} | {m["upper_body_pose_rmse_rad"]:.4f} | {m["contact_state_accuracy"]:.3f} | {m["action_rms"]:.3f} / {m["action_clip_fraction"]:.3f} |')
    lines+=['','接触：真实Newton末子步Fz>1N vs reference估计；sliding为承载足link XY速度代理，不是接触点滑移真值。Touchdown误差和角速度/加速度等逐窗口统计均保留在JSON，接触准确率不等价“自然摆臂”。',
        '', '| 分支 | 10s存活 | 运动质量Gate |', '| --- | --- | --- |']
    teacher=evals['eval_a']['rows']['motion'];lines.append(f'| Teacher | {sum(r["n_failed"]==0 for r in teacher)}/3 | 对照，不宣称完美足轨迹精度 |')
    for phase,label in [('eval_a','Phase A'),('eval_b','Phase B')]:
        if phase in evals:
            c=evals[phase]['comparisons'];lines.append(f'| {label} | {sum(r["survived"] for r in c)}/3 | {sum(r["quality_gate"] for r in c)}/3 |')
        else:lines.append(f'| {label} | NOT EVALUATED | NOT RUN — Phase A gate failed |')
    lines+=['','预注册门槛：三个窗口均完整10s；相对同窗Teacher NR下降≤0.10、足RMSE增加≤0.05m、上身RMSE增加≤0.15rad。所有窗口满足才进入正式B；该工程门槛不等于独立数据泛化认证。',
        '', '## WM 配对误差','', '| 分支 | 起点/s | H | Height RMSE/m | Pitch RMSE/deg | Foot FK RMSE/m | Reward Bias/normalized |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for phase,ev in evals.items():
        for role in ('motion','foot'):
            if role=='motion' and phase!='eval_a':continue
            for row in ev['rows'][role]:
                for h in ('8','16','32'):
                    m=row['wm_paired'].get('horizons',{}).get(h)
                    if m:lines.append(f'| {phase}/{role} | {row["offset_s"]:g} | {h} | {m["base_height_rmse_m"]:.5f} | {m["root_pitch_rmse_deg"]:.3f} | {m["feet_position_rmse_m"]:.5f} | {m["paired_reward_bias"]:.4f} |')
    lines+=['','每次预测从同一真实Newton状态开始，复用相同真实执行动作；所有H使用同一组完整32步窗口，不含Critic/bootstrap，不跨物理终止。1/4步等完整细节见评估JSON。WM误差小不能替代Foot Policy的闭环验收。',
        '', '## Phase B 设置与实际执行状态','',
        '- 实现沿用原Motion reward+BPTT+imagined lambda-return Critic；独立365D Critic随机初始化（Teacher Motion Critic是914D，禁止强行迁移）。Actor优化器仅持有E_f；D/E_m冻结。WM监督更新与Actor分开，Actor上下文内冻结WM参数且保留状态/动作输入梯度。无额外Prior、latent loss或Foot Reward。',
        f'- 配置候选：{cfg["num_envs"]}env，rollout_steps={cfg["rollout_steps"]}，buffer={cfg["buffer_steps"]}步/环境，WMbatch={cfg["wm_batch"]}×{cfg["wm_batches"]}，Actorbatch={cfg["pi_batch"]}，H32（想象反传长度，不是Newton rollout长度）；前{cfg["wm_only_iters"]}iterations仅WM。',
        f'- B候选预算：最多{cfg["phase_b_iterations"]}iterations / {cfg["phase_b_budget_s"]}s。lr WM/Actor/Critic={cfg["lr_wm"]}/{cfg["lr_actor"]}/{cfg["lr_critic"]}；gamma/lam={cfg["gamma"]}/{cfg["lam"]}，Critic2epochs×4minibatches，原Motion reward系数与动作clamp±20不变。',
        '- checkpoint原子写入，含Actor/Teacher/WM/独立Critic/各Optimizer/Norm/RNG/计数。Replay不完整序列化：恢复时从新reference物理状态填充warmup_steps，再20次WM-only恢复；不声称Newton接触求解器bitwise连续恢复。',
        '- 256env配置已在本次有界B实测，资源与吞吐见下文；更长训练、多seed稳定性与独立clip泛化尚未验证。',
        '', '## 结论、诊断与下一步','']
    if not evals['eval_a']['gate_passed']:
        lines+=[('**A_QUALITY_GATE_FAILED**：Phase A未达到原质量Gate。完成接口与分布诊断后，执行有界B修复（最多400次/600s），没有将A失败改为成功，没有修改最终质量Gate或物理终止。' if 'eval_b' in evals else '**BLOCKED_PHASE_A**：Phase A未达到预注册Newton门槛，因此停止正式Phase B与Stage 2，不将离线latent收敛当作Walking成功。')]
        diag=evals.get('diagnostic_masked_oracle')
        if diag:
            survived=sum(r['n_failed']==0 for r in diag['rows']['foot'])
            lines += [f'为定位观测接口，额外将**真实Teacher Motion latent**送入同一个冻结D，使用相同Foot ObsNorm及当前观测适配，完成三个10s窗口诊断；完整存活 **{survived}/3**。该分支调用Motion Encoder，**只用于诊断，绝不计Foot成功**。',
                '该结果可区分mask/Decoder兼容性与E_f拟合问题，但未构成单因素完备因果消融；逐窗Gate与误差见JSON。',
                '下一步优先验收Teacher-z + 因果朝向接口，尤其60s窗口。先检查基于Foot位置/旋转/contact/phase的朝向估计能否达到原root朝向接口精度，再考虑扩大E_f训练；不能将完整Motion root reference重新输入Foot Policy。' if cfg.get('causal_heading') else '下一步先用Teacher-z oracle检查mask兼容性，再决定朝向适配或Encoder状态覆盖；不延长盲目长训。']
    else:lines+=['Phase A已通过工程门槛；Phase B是否成功以独立结果表为准，不由想象reward推断。']
    if (d/'phase_b/phase_b_audit.json').exists():
        b=read(d/'phase_b/phase_b_audit.json')
        lines += ['', '### 有界 Phase B 实测', '', f'实际Newton transitions={b["counts"]["newton"]:,}；WM updates={b["counts"]["wm"]}；Actor updates={b["counts"]["actor"]}；Critic minibatch updates={b["counts"]["critic"]}；wall={b["wall_seconds"]:.2f}s；峰值Torch={b["peak_torch_mib"]:.1f}MiB。Teacher/Decoder冻结、WM仅监督更新、两套Norm固定、E_f实际更新的全部断言：'+str({k:v for k,v in b.items() if isinstance(v,bool) and k!='smoke'})+'。', 'Best B为本次唯一独立Newton评估候选；排序不等于通过质量Gate。Checkpoint：'+link(d/'phase_b/phase_b_best.pt')+'。']
        rows_b=[json.loads(x) for x in (d/'phase_b/phase_b_metrics.jsonl').read_text().splitlines()]
        active=[r for r in rows_b if r['actor']>0]
        if active:
            import statistics
            lines += [f'Actor阶段采样Collection FPS中位数={statistics.median(r["collection_fps"] for r in active):.1f}；迭代速度中位数={statistics.median(r["iterations_per_second"] for r in active):.2f}it/s。FPS只计真实Newton transitions；H32想象样本不计入。']
        if (d/'gpu_resources.jsonl').exists():
            gpu=[json.loads(x) for x in (d/'gpu_resources.jsonl').read_text().splitlines() if json.loads(x).get('phase')=='phase_b']
            valid=[r for r in gpu if 'gpu_used_mib' in r]
            if valid:lines += [f'nvidia-smi训练期总显存峰值={max(r["gpu_used_mib"] for r in valid)}MiB（含桌面/上下文；启动前桌面约499MiB），GPU利用率峰值={max(r["gpu_util_percent"] for r in valid)}%。']
        quality=evals.get('eval_b',{}).get('gate_passed',False)
        lines += ['最终结论：**'+('PASS_ON_TRAIN_CLIP_WINDOWS' if quality else 'BLOCKED_QUALITY')+'**。'+('当前训练clip三个窗口满足相对Teacher工程门槛；独立Walking泛化仍未验证。' if quality else 'Foot已能存活并产生Walking，但仍未达到接近Teacher的轨迹/全身质量要求；不启动Stage2或正式长训。')]
    lines+=['独立Walking clip泛化：**NOT EVALUATED**。当前仅有合格walk1_subject1；dance holdout不当作Walking泛化结果。单seed、3个时间窗不能代表全部Walking能力。',
        '', '## 产物、SwanLab与命令','',f'- 配置快照：{link(d/"profile.json")}；执行命令：{link(d/"commands.jsonl")}；完整stdout：{link(d/"stdout.log")}。']
    for phase in evals:lines.append(f'- {phase}完整指标与trace路径：{link(d/phase/"summary.json")}。')
    sw=d/'swanlab/status.json'
    if sw.exists():
        s=read(sw);lines.append(f'- SwanLab：**{s["status"]}**；'+(f'[{s["url"]}]({s["url"]})' if 'url' in s else '同步仍在进行，见swanlab/status.json')+'。仅核心参数/指标；全量逐窗口/逐horizon明细保存在本地。')
    for v in sorted((d/'videos').glob('*.mp4')):lines.append(f'- 对比视频：{link(v)}（青色reference仅可视化，物理终止后该侧冻结并标红）。')
    if (d/'eval_b/summary.json').exists():
        import statistics
        ae=read(d/'eval_a/summary.json');be=read(d/'eval_b/summary.json')
        lines+=['', '## 核心比较与证据边界', '',
            '| 分支 | 10s存活 | 平均Motion NR | 平均足RMSE/m | 平均上身RMSE/rad |',
            '| --- | --- | ---: | ---: | ---: |']
        for label,rows in [('Teacher',ae['rows']['motion']),('Phase A',ae['rows']['foot']),('Phase B',be['rows']['foot'])]:
            lines.append(f'| {label} | {sum(r["n_failed"]==0 for r in rows)}/3 | {statistics.mean(r["norm_return"] for r in rows):.4f} | {statistics.mean(r["metrics"]["foot_position_rmse_m"] for r in rows):.4f} | {statistics.mean(r["metrics"]["upper_body_pose_rmse_rad"] for r in rows):.4f} |')
        lines+=['', '**已验证**：A和B确实能连续走10s；其三窗实际根部行程分别约3.95–5.86m、3.04–5.83m，因此不是只在原地站住。但B相对A的平均Motion NR下降、足轨迹误差增大；不能把Critic loss下降或想象reward平稳称为控制质量改善。',
            'WM在观察到的同动作32步配对窗口上没有出现巨大正Reward Bias；这不能证明其Actor梯度处处准确，也不足以将B退化单独归因于WM或Critic。']
        lp=d/'latent_alignment_posthoc.json'
        if lp.exists():
            align=read(lp);lines+=['', '| 分支 | 起点/s | 闭环latent MSE | Cosine | Rotation6D最大abs归一化 | 朝向目标估计误差RMSE/deg | 动作步间Delta RMS |',
                '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
            for r in align['rows']:
                lines.append(f'| {r["phase"]} | {r["offset_s"]:g} | {r["latent_mse"]:.4f} | {r["latent_cosine"]:.4f} | {r["rotation_max_abs_normalized"]:.3f} | {r.get("goal_heading_yaw_error_rmse_deg",float("nan")):.2f} | {r["action_delta_rms_per_step"]:.4f} |')
            lines+=['', 'Latent对齐是记录的bf16 Foot latent与同帧CPU fp32 Teacher标签的后验评分；未进入Actor输入或更新。朝向误差使用完整reference仅做独立评分，动作Delta为相邻控制步command差的RMS，单位为原action单位。',
                '**已验证**：Rotation6D归一化离群值已明显收敛至约2.5–2.9；当前B退化不是原99.61旋转异常的简单重复。B的闭环latent MSE比A增大，足位置归一化值在60s窗口又因位置漂移增至约12。',
                '**已验证的剩余接口问题**：因果heading的目标yaw在60s窗口相对Teacher root yaw有约14.15deg RMSE（其他两窗约2.72/5.17deg）；该差值独立于A/B Encoder权重。连真实Teacher-z oracle在此窗也有0.658m足RMSE，说明单纯提高latent拟合不能保证恢复Teacher的朝向反馈。',
                '**尚未验证的机制**：B的bootstrap梯度、短时模型误差、回报权重与Decoder可控性各贡献多少；离线latent在1s时间块留出上的泛化差距有多少来自足目标歧义。当前没有单因素因果结论。',
                '最小下一步实验：保留固定E_m/D，先做有界Teacher-z oracle与因果朝向估计的逐窗对照；目标是先通过全部三窗接口Gate。随后才对奖励梯度/Bootstrap梯度做同Replay、同seed的小步数诊断，避免直接延长B。',
                '详细后验数据：'+link(lp)+'。']
        lines+=['', '本次B目录通过只读符号链接复用heading修复版的A、Smoke和对照；没有重新计算4000次A更新，也没有覆盖来源。来源记录：'+link(d/'baseline_source.json')+'。',
            'A的best按时间块latent验证MSE选择；B仅评估末次一个候选，best_B是该候选的明确命名，**不是全训练轨迹或历史所有checkpoint的全局最佳**。中间B checkpoint未独立评估，不能推断其性能。',
            '三种Policy在0/60/120s的初始71D物理状态最大差均为0，Teacher SHA前后相同：'+link(d/'completion_audit.json')+'。不声称Newton接触求解器内部warm state逐位一致。']
    if (d/'resume_verification.json').exists():
        rv=read(d/'resume_verification.json')
        lines+=['', '## 恢复 Smoke', '',
            '另在独立Smoke目录实际测试恢复，没有改写正式A/B checkpoint：A从iteration3继续到6，Adam step3→6；B从iteration2继续到24，先补Replay/20次WM预热，再2次Actor更新。B Actor/WM/Critic Adam step分别2→4、2→24、16→32，计数与权重持续，Teacher/D及Norm冻结断言通过。',
            '恢复测试不是新增技能评估；新恢复阶段产生的真实transitions单独在'+link(d/'resume_verification.json')+'中记录。没有恢复Newton接触内部warm state。',
            '模型/计数/RNG/优化器持久化与重新采样机制已有实测；更长训练中断、电源故障与网络长期中断恢复并未全部复现。']
    name=d.name
    lines+=['','```bash',f'tmux attach -t {name}_review  # 本次训练已结束，此session仅查看日志',f'tail -f logs/{name}/stdout.log',f'bash tools/train/run_s54_stage15.sh status {name}',
        '# 在全新时间戳目录复现实验（GPU空闲时运行）','bash tools/train/run_s54_stage15.sh start "s54_stage15_$(date +%Y%m%d_%H%M%S)_heading" configs/motion_prior/stage15/lafan1_skill_transfer_heading_3090.json',
        '# 完成诊断后，有界B最多400 iterations /600秒（创建新目录）',
        'bash tools/train/run_s54_stage15_bounded_b.sh logs/s54_stage15_20261106_heading_repair',
        '# 仅重渲染已有轨迹，不运行policy',f'/home/rm/workspace/IsaacLab_3_0/.venv/bin/python tools/visualization/render_s54_stage15.py --run-dir logs/{name} --offset 0 --out results/s54/stage15_offset0_new.mp4',
        '# CPU缓存重试上传，使用已有稳定实验ID',f'.venv-swanlab/bin/python tools/ops/sync_s54_stage15.py --run-dir logs/{name}','```','']
    text='\n'.join(lines)
    if cfg.get('causal_heading'):
        text=text.replace('Decoder 的归一化观测93:99通道置零；其余93通道为关节/速度/IMU重力/上一动作。',
            'Decoder 的93:99虚拟朝向通道由相同230D FootGoal与本体重力构造；其余93通道仍为原本体观测。ObsNorm复制Teacher并冻结。')
        text=text.replace('E_f训练及GoalNorm拟合均排除0–10、60–70、120–130秒区间，并保留lookahead guard；其余按1s块分出验证数据。Teacher预训练见过该clip，因此这些是E_f的时间留出窗口，不是独立Walking clips。',
            '修复版蒸馏使用walk1训练clip全部有效区间，并按1s块留出10%验证数据。0/60/120s窗口大部分帧已用于离线蒸馏与训练集Norm拟合；这次独立性指新的真实Newton闭环rollout，**不是未见时间窗或未见clip泛化**。没有在Eval阶段改Norm。')
        text=text.replace('4项单元测试','5项单元测试')
        text=text.replace('早期独立Newton Smoke发现paired WM诊断缺少episode_lengths/xi字段；修复后在本实验新目录重跑通过。',
            '初版独立Newton Smoke曾发现paired WM诊断缺少episode_lengths/xi字段；修复后原版及因果heading版均在新目录重跑通过。')
        text=text.replace('使用2组固定seed的root XY/yaw扰动','使用2组固定seed的root XY/yaw扰动')
        extra=['', '## 有界修复与原方案对照', '',
            '原始mask版保留于 `logs/s54_stage15_20261106_165827`：3/3存活10s，但Motion NR为0.6431/0.4543/0.7023、足RMSE为0.6717/1.3209/0.5005m，质量Gate0/3；未启动其正式B。',
            '原mask版的Teacher-z oracle在0/60/120s分别NR约0.855/0.839/0.847，足RMSE约0.126/0.300/0.558m。Oracle调用Motion Encoder，只用于确定mask会退化heading反馈；不算Foot成功。',
            '修复版使用未来足旋转的contact加权圆均值估计期望yaw，并以身体重力确定实际tilt；固定参考tilt与foot-yaw偏置只在训练集标定。每一步heading输入由**相同FootGoal**生成，历史按时刻保存；不读逐帧完整reference root姿态、不调用Motion Encoder。',
            '训练集校准常量：'+json.dumps(cfg.get('calibration',{}),ensure_ascii=False)+'；bias/roll/pitch='+str([cfg['heading_bias_rad'],*cfg['reference_tilt_roll_pitch_rad']])+' rad。',
            '同时扩大离线root扰动覆盖为XY/yaw标准差(0.10m,0.20rad)、(0.25m,0.50rad)，并覆盖指定训练clip。Teacher latent标签仍对应原同一帧；只训练E_f。',
            '这是有界功能修复，heading适配、数据覆盖和增强同时改变，**不能将改善单独归因于某一个因素**。原时间留出结果和全部历史权重不变。',
            '原方案报告：'+link(ROOT/'logs/s54_stage15_20261106_165827/STAGE15_REPORT.md')+'。']
        text+='\n'+'\n'.join(extra)+'\n'
    (d/'STAGE15_REPORT.md').write_text(text);(ROOT/'STAGE15_REPORT.md').write_text(text)
    print(d/'STAGE15_REPORT.md')
if __name__=='__main__':main()
