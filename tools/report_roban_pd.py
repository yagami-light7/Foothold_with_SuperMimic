"""Build a Chinese report and static figures from independent Roban PD traces."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def aggregate(rows):
    result=dict(count=len(rows),screen_pass=sum(r['screen_pass'] for r in rows))
    for key in ('t90_s','overshoot_pct','settling_5pct_s','final_mean_error_rad'):
        values=[r[key] for r in rows if r[key] is not None]
        result[key]=dict(median=float(np.median(values)),max=float(max(values)),unresolved=len(rows)-len(values))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixed',type=Path,required=True)
    p.add_argument('--standing',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('Choose a new report directory')
    a.out.mkdir(parents=True)
    fixed=json.loads((a.fixed/'summary.json').read_text())
    standing=json.loads((a.standing/'summary.json').read_text())
    traces=np.load(a.fixed/'trajectories.npz')
    holds=np.load(a.standing/'trajectories.npz')
    profiles=sorted(set(r['profile'] for r in fixed['results']))
    summary=dict(conclusion='PD 起动及时，但腰髋肩欠阻尼；不宜直接视为已调好的动态跟踪参数',
                 production_gains_changed=False,profiles={})
    for profile in profiles:
        rows=[r for r in fixed['results'] if r['profile']==profile and r['kind']=='step']
        free=[r for r in rows if r['contact_free']]
        summary['profiles'][profile]=dict(all_step_trials=aggregate(rows),contact_free_step_trials=aggregate(free),
            contact_affected_trials=len(rows)-len(free))
    baseline=[r for r in fixed['results'] if r['profile']=='baseline' and r['kind']=='step' and r['contact_free']]
    # Export per-trial rows as well as the per-joint compact report.
    with (a.out/'step_trials.csv').open('w',newline='') as f:
        rows=[r for r in fixed['results'] if r['kind']=='step']
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (a.out/'sine_trials.csv').open('w',newline='') as f:
        rows=[r for r in fixed['results'] if r['kind']=='sine']
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Roban PD 实测验证','',
        '结论：当前参数起动快，但腰、髋和肩的阻尼不足。能完成短时站立保持，不能据此认定已适合高质量动态跟踪。原有训练 PD、动作缩放和检查点均未修改。','',
        '## 测试设置','',
        '- 使用实际训练后端 Isaac Lab / Newton / MJWarp，implicitfast 积分器；50 Hz 位置指令、4 个物理子步，测量间隔 20 ms。因此 20 ms 内的响应无法继续分辨。',
        '- 保留原始质量、惯量、armature、关节限位、力矩限制、自碰撞；本测试直接发送关节位置目标，目标速度保持为零，与当前训练执行器一致。未叠加逆动力学或重力补偿。',
        '- 固定躯干、关闭重力以隔离关节动力学；21 关节 × 3 个姿态 × 正负两个方向 × 0.05/0.15 rad 两个阶跃幅度，共 252 例/参数组，每例 2 秒。超过限位附近的请求会缩小到距硬限位 0.01 rad，逐例实际幅度在 CSV 中。',
        '- 姿态来自模型零位，以及 Neutral_FW 训练片段第 0、50 帧的关节角。正弦测试每关节每姿态分别 1、2 Hz，幅度最多 0.1 rad，持续 8 秒；用第 2–8 秒拟合幅值及相位。',
        '- 正弦相位以计划连续轨迹 sin(2πft) 为基准，包含 50 Hz 指令的零阶保持影响；等效相位延迟不是通信延迟或阶跃起动延迟。',
        '- 站立保持恢复重力和自由基座，双脚放置于地面上方 1 mm，零关节目标保持 5 秒；初始俯仰为 0、±1°、±2°。只在重置时写入状态，不使用任何平衡控制或策略。',
        '- 读取求解器 `mjw_data.qfrc_actuator` 作为实际关节驱动力矩；不是 `applied_torque` 中的 PD 估算。每个控制周期只保存最后一个物理子步，因此未观察到饱和不能排除其他子步的短暂饱和。',
        '- 被动阻尼在运行模型中仍为 0.2 N·m·s/rad，与驱动 Kd 分开存在。原公式的 `2 sqrt(Kp × armature)` 没有计入连杆惯量。','',
        '## 原参数的响应','',
        '下表采用 0.05 rad 小阶跃，汇总各姿态与方向的最差值；剔除了观察到自碰撞约束的试验。稳定时间指此后始终处于目标 ±max(5%阶跃幅度, 0.0025 rad) 内。首次达到 90% 只是一次穿越，不代表已经稳定。','',
        '| 关节 | 最大 t90 (ms) | 最大超调 (%) | 最大稳定时间 (ms) | 未在 2s 内稳定 | 2 Hz 最大增益 |',
        '|---|---:|---:|---:|---:|---:|']
    joint_rows=[]
    for name in fixed['joint_names']:
        rows=[r for r in baseline if r['joint']==name and r['requested_amplitude_rad']==.05]
        sine=[r for r in fixed['results'] if r['profile']=='baseline' and r['kind']=='sine' and r['joint']==name and r['frequency_hz']==2 and r['active_contact_samples']==0]
        agg=aggregate(rows)
        out=dict(joint=name,t90_max_ms=1000*agg['t90_s']['max'],overshoot_max_pct=agg['overshoot_pct']['max'],
                 settling_max_ms=1000*agg['settling_5pct_s']['max'],unsettled=agg['settling_5pct_s']['unresolved'],gain_2Hz_max=max(r['gain'] for r in sine))
        joint_rows.append(out)
        lines.append(f"| {name} | {out['t90_max_ms']:.0f} | {out['overshoot_max_pct']:.1f} | {out['settling_max_ms']:.0f} | {out['unsettled']} | {out['gain_2Hz_max']:.2f} |")
    with (a.out/'joint_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(joint_rows[0]));writer.writeheader();writer.writerows(joint_rows)
    max_overshoot=max(baseline,key=lambda r:r['overshoot_pct'])
    max_gain=max((r for r in fixed['results'] if r['profile']=='baseline' and r['kind']=='sine' and r['active_contact_samples']==0),key=lambda r:r['gain'])
    summary.update(joints=joint_rows,worst_contact_free_overshoot=max_overshoot,worst_contact_free_sine_gain=max_gain)
    lines+=['',f"最差超调出现在 `{max_overshoot['joint']}`，达到 {max_overshoot['overshoot_pct']:.1f}%。最大正弦增益为 {max_gain['gain']:.2f}（{max_gain['frequency_hz']:.0f} Hz，`{max_gain['joint']}`）：目标运动幅度被显著放大。",'',
        '工程筛查阈值设为 t90≤150 ms、超调≤20%、稳定时间≤400 ms、末 0.2 秒平均绝对误差≤0.005 rad；这些是本次筛查的选择，不是 Roban 硬件规格。以下通过率只表示满足这组阈值。','',
        '| 参数组 | 无碰撞试验通过数 | 小/大阶跃超调中位数/最大值 | 稳定时间中位数/最大值 | 2秒内未稳定 |',
        '|---|---:|---:|---:|---:|']
    for profile in profiles:
        agg=summary['profiles'][profile]['contact_free_step_trials']
        lines.append(f"| {profile} | {agg['screen_pass']}/{agg['count']} | {agg['overshoot_pct']['median']:.1f}% / {agg['overshoot_pct']['max']:.1f}% | {agg['settling_5pct_s']['median']*1000:.0f} / {agg['settling_5pct_s']['max']*1000:.0f} ms | {agg['settling_5pct_s']['unresolved']} |")
    affected=[r for r in fixed['results'] if r['profile']=='baseline' and r['kind']=='step' and not r['contact_free']]
    lines+=['',f'原参数有 {len(affected)} 个阶跃试验观察到碰撞约束。这些试验保留原始数据，但不纳入纯 PD 通过率；尤其内收肩关节的较大阶跃可能撞到躯干，不能将其稳态误差直接归咎于增益。','',
        '| 参数组 | 频率 | 无碰撞样本增益范围 | 最大相位滞后 | 最大等效相位延迟 |',
        '|---|---:|---:|---:|---:|']
    sine_summary=[]
    for profile in profiles:
        for freq in (1.,2.):
            rows=[r for r in fixed['results'] if r['kind']=='sine' and r['profile']==profile and r['frequency_hz']==freq and r['active_contact_samples']==0]
            v=dict(profile=profile,frequency_hz=freq,count=len(rows),gain_min=min(r['gain'] for r in rows),gain_max=max(r['gain'] for r in rows),
                   max_phase_lag_deg=max(r['phase_lag_deg'] for r in rows),max_equivalent_delay_s=max(r['equivalent_delay_s'] for r in rows))
            sine_summary.append(v)
            lines.append(f"| {profile} | {freq:.0f} Hz | {v['gain_min']:.2f}–{v['gain_max']:.2f} | {v['max_phase_lag_deg']:.1f}° | {v['max_equivalent_delay_s']*1000:.0f} ms |")
    summary['sine_summary']=sine_summary
    lines+=['',
        '## 承重站立保持','',
        '| 参数组 | 初始俯仰 (°) | 5秒未倒 | 关节 RMSE (rad) | 最大躯干倾角 (°) | 双脚均测到支撑力 |',
        '|---|---:|---|---:|---:|---|']
    for r in standing['standing']:
        lines.append(f"| {r['profile']} | {r['initial_pitch_deg']:+.0f} | {'是' if r['fall_time_s'] is None else '否'} | {r['joint_rmse_before_fall_rad']:.4f} | {r['max_tilt_before_fall_deg']:.1f} | {'是' if r['feet_with_measured_ground_force']==2 else '否'} |")
    base_holds=[r for r in standing['standing'] if r['profile']=='baseline']
    summary['standing']=standing['standing']
    lines+=['',f"原参数 {sum(r['fall_time_s'] is None for r in base_holds)}/{len(base_holds)} 个试验均保持 5 秒未倒；最大倾角 {max(r['max_tilt_before_fall_deg'] for r in base_holds):.1f}°。站得住与精确保持姿态是两项要求。这只是零位、平地、短时间和小初始扰动，尚未验证行走冲击、单脚承重或外力推扰。跌倒判据为根高度低于 0.35 m 或躯干 Rzz<0.2。",'',
        '## 阻尼对照与处理建议','',
        '对照组保持 Kp 和全部物理参数不变，仅在测试进程内使用 `Kd = 2 × 0.7 × sqrt(Kp × median(Mjj))`。Mjj 是固定躯干、其余关节锁住时的质量矩阵对角项，取三个测试姿态中位数；包含连杆惯量与 armature。它用于定位欠阻尼，不是对耦合系统临界阻尼的精确解。','',
        '| 关节 | Kp | 原 Kd | 对照 Kd | Mjj 中位数 (kg·m²) |',
        '|---|---:|---:|---:|---:|']
    inertia=np.median(fixed['fixed_base_mass_diagonal_kg_m2'],axis=0)
    for i,name in enumerate(fixed['joint_names']):
        lines.append(f"| {name} | {fixed['kp'][i]:.0f} | {fixed['baseline_kd'][i]:.3f} | {fixed['inertia_probe_kd'][i]:.3f} | {inertia[i]:.6f} |")
    lines+=['',
        '对照组显著抑制了腰髋肩超调，但部分髋关节在 2 Hz 的相位滞后增加，不能只追求减小超调。建议下一步分别为腰髋、膝、踝、肩肘确定所需带宽，联合扫描 Kp/Kd，并用带重力的动作跟踪、单脚支撑与接触冲击验证。当前不直接将对照组写入训练默认值。',
        '修改 Kp 时还必须同步考虑 `action_scale=0.25×effort/Kp` 及检查点物理契约；不能把改变动作范围的影响混进纯增益对比。硬件通讯延迟、驱动器内部环和真实减速器摩擦不在本次仿真测量范围内。','',
        '## 可复现资料','',
        '- `fixed/summary.json`、`fixed/trajectories.npz`：固定躯干逐例结果和逐帧角度/速度/真实力矩/碰撞计数。',
        '- `standing/summary.json`、`standing/trajectories.npz`：自由基座逐帧状态及求解器足地接触力。',
        '- `step_trials.csv`、`sine_trials.csv`、`joint_summary.csv`：可直接分析的数据。',
        '- `pd_response.png` / `.pdf`、`standing_response.png` / `.pdf`：由保存轨迹生成的静态图。',
        '- `aggregate.json`：统计汇总。`verification.json`：源文件哈希及测试状态。','',
        '复现命令（从仓库根目录执行，输出目录必须不存在）：','',
        '```bash',
        'bash tools/run_roban.sh pd --mode fixed --include_damping_probe --out reports/roban_pd_repeat/fixed',
        'bash tools/run_roban.sh pd --mode standing --include_damping_probe --out reports/roban_pd_repeat/standing',
        'python tools/report_roban_pd.py --fixed reports/roban_pd_repeat/fixed --standing reports/roban_pd_repeat/standing --out reports/roban_pd_repeat/report',
        '```','']
    (a.out/'REPORT.md').write_text('\n'.join(lines))
    (a.out/'aggregate.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.25})
    fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
    chosen=['waist_yaw_joint','leg_l2_joint','leg_l4_joint','zarm_l1_joint','leg_l5_joint']
    for ax,name in zip(axes.flat,chosen):
        j=fixed['joint_names'].index(name)
        row=next(i for i,r in enumerate(fixed['rows']) if r['posture']==0 and r['joint']==j and r['direction']==1)
        for profile in profiles:
            key=f'{profile}_step_0.05'
            ax.plot(traces[key+'__time'],traces[key+'__q'][:,row,j],label=profile.replace('_damping_probe',' probe'))
        ax.axhline(.05,ls='--',c='k',lw=1,label='target')
        ax.set(title=name,xlabel='Time (s)',ylabel='Joint angle (rad)',xlim=(0,2))
    ax=axes.flat[5];j=fixed['joint_names'].index('leg_l2_joint')
    row=next(i for i,r in enumerate(fixed['rows']) if r['posture']==0 and r['joint']==j and r['direction']==1)
    for profile in profiles:
        key=f'{profile}_sine'
        ax.plot(traces[key+'__time'],traces[key+'__q'][:,row,j],label=profile.replace('_damping_probe',' probe'))
    ax.plot(traces['baseline_sine__time'],traces['baseline_sine__target'][:,row,j],'--k',lw=1,label='target')
    ax.set(title='Hip roll: 2 Hz sine',xlabel='Time (s)',ylabel='Joint angle (rad)',xlim=(6,8))
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Roban PD: fast first crossing, slow settling in proximal joints')
    for ext in ('png','pdf'):fig.savefig(a.out/f'pd_response.{ext}',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for profile in profiles:
        key=f'{profile}_hold';time=holds[key+'__time'];quat=holds[key+'__root'][:,0,3:]
        tilt=np.rad2deg(np.arccos(np.clip(1-2*(quat[:,0]**2+quat[:,1]**2),-1,1)))
        axes[0].plot(time,tilt,label=profile.replace('_damping_probe',' probe'))
        axes[1].plot(time,holds[key+'__foot_ground_force'][:,0,:,2].sum(-1),label=profile.replace('_damping_probe',' probe'))
    axes[0].set(title='Unperturbed free-root hold',xlabel='Time (s)',ylabel='Root tilt (deg)')
    axes[1].set(title='Measured Newton ground reaction',xlabel='Time (s)',ylabel='Total foot vertical force (N)')
    axes[0].legend(fontsize=8)
    for ext in ('png','pdf'):fig.savefig(a.out/f'standing_response.{ext}',dpi=170)
    plt.close(fig)
    print(json.dumps(summary['profiles'],indent=2))


if __name__=='__main__':main()
