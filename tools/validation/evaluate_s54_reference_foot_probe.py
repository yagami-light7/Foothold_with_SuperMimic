"""Bounded Newton probe of learned Foot Encoder on original Walking FK goals.

No optimizer, WM imagination, action teacher, normalization update, or checkpoint
rewrite. The Motion Encoder is a matched-state control. Both use the current
shared Decoder, original PD/physics, and the union of original physical guards.
"""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime
import argparse
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def encoding_contract(saved, reference):
    """Explicit source substitution for inference, never a resume conversion."""
    allowed={'mode','version'}
    if saved.get('mode')!='foot_trajectory' or reference.get('mode')!='reference_foot':
        raise ValueError('Expected Foot-trained weights and an explicitly labelled reference source')
    if {k:v for k,v in saved.items() if k not in allowed}!={k:v for k,v in reference.items() if k not in allowed}:
        raise ValueError('Reference Foot Goal encoding differs from trained input semantics')
    return dict(saved=saved,probe=reference,changed_fields=sorted(allowed),
                interpretation='Inference-only change of trajectory source; not compatible training resume')


def reference_window(offset,seconds,clip_start,clip_end,lookahead_s=1.,dt=.02):
    if any(not math.isfinite(x) for x in (offset,seconds)) or offset<0 or seconds<=0:
        raise ValueError('Invalid reference interval')
    ticks=round(seconds/dt);start=clip_start+round(offset/dt)
    if not math.isclose(ticks*dt,seconds,abs_tol=1e-8) or not math.isclose((start-clip_start)*dt,offset,abs_tol=1e-8):
        raise ValueError('Reference start/duration must be control-tick aligned')
    if start+ticks+math.ceil(lookahead_s/dt)>clip_end:
        raise ValueError('Window plus complete future Foot Goal exceeds valid reference')
    return start,ticks


def combine_videos(out,videos,timeout):
    from tools.visualization.render_s54_joint_comparison import resolve_ffmpeg
    out=Path(out);manifest=out/'video_sequence.txt';combined=out/'reference_foot_retrieval_all_windows.mp4'
    if manifest.exists() or combined.exists():raise FileExistsError('Combined-video artifact already exists')
    manifest.write_text(''.join("file '"+Path(video).name+"'\n" for video in videos))
    subprocess.run([resolve_ffmpeg(),'-n','-loglevel','error','-f','concat','-safe','0','-i',str(manifest),
                    '-c','copy','-movflags','+faststart',str(combined)],check=True,timeout=timeout)
    return str(combined)


def run(args):
    import numpy as np
    import torch
    from supermimic.joint_runtime import project,digest,json_write,profile_hash
    from supermimic.motion_prior_runtime import reconstruct
    from supermimic.reference_foot import ReferenceFootGoalSource
    from supermimic.goals import GoalSpec
    from supermimic.motion_prior_data import ContactConfig,load_manifest
    from supermimic.mdp import flatten_state,unflatten_state
    from supermimic.s54 import FOOT_BODY_NAMES,JOINT_NAMES
    from supermimic.rotations import so3_log,yaw_mat,mat_to_quat,quat_to_mat
    from supermimic.foot_trajectory import command_time
    from supermimic.control import bound_action
    from supermimic import networks
    started=time.monotonic();torch.set_num_threads(4)
    checkpoint=Path(args.checkpoint).resolve();source_sha=digest(checkpoint)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    cfg=ck['profile'];joint=ck['joint_contract']
    if joint.get('kind')!='joint_motion_foot' or joint['profile_sha256']!=profile_hash(cfg):
        raise ValueError('Invalid joint checkpoint/profile contract')
    requested=GoalSpec('reference_foot',cfg['goal_frames'],cfg['foot_goal_dt'])
    transport=encoding_contract(joint['goals']['foot'],requested.contract())
    manifest=ROOT/cfg['reference_manifest'];files,data=load_manifest(manifest,'train')
    if data!=joint['data'] or [Path(f).stem for f in files]!=['walk1_subject1']:
        raise ValueError('Reference must be the checkpoint verified Walking training source')
    protocol=dict(kind='reference_foot_skill_retrieval_probe',checkpoint=str(checkpoint),
        checkpoint_sha256=source_sha,iteration=ck['iteration'],reference_manifest=str(manifest),
        data_contract=data,offsets_s=args.offsets,duration_s=args.seconds,seed=args.seed,
        encoding_transport=transport,foot_actor='Saved 230D Foot Encoder + current shared Decoder; saved Foot norms and obs mask',
        actor_inputs='Foot branch: proprioception 10x99 + 10x23 foot samples; no whole-body goal',
        reference_source='Original S54 MotionLib FK; no Quintic fit or action distillation',
        reset='Identical noise-free full reference states for both controllers (diagnostic, not neutral Foot reset)',
        physical_guards='Stop both controllers at union of original Motion and original Foot terminations + nonfinite',
        reward='Original Motion reward only for independent scoring, never fed to or updating Actor',
        contact_labels=asdict(ContactConfig()),contact_measurement='Actual Newton last-substep contact forces, Fz>1N',
        no_updates=True,no_action_or_observation_noise=True,domain_randomization=False,
        max_newton_transitions=2*len(args.offsets)*round(args.seconds/.02),
        process_budget_seconds=args.process_budget_s,
        gates=dict(survive_full_requested_duration=True,world_foot_rmse_m=.10,upper_pose_rmse_rad=.40,
                   max_motion_normalized_return_drop_vs_control=.10,
                   note='Predeclared engineering diagnostic thresholds, not a statistical naturalness certification'),
        registered_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
    if args.dry_run:
        print(json.dumps(protocol,indent=2));return
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists() or (out/'evaluation').exists():
        raise FileExistsError('Probe output already contains an experiment; choose a fresh directory')
    json_write(out/'protocol.json',protocol)
    def alarm(*_):raise TimeoutError('Bounded reference-Foot evaluation process budget reached')
    previous=signal.signal(signal.SIGALRM,alarm);signal.alarm(math.ceil(args.process_budget_s))
    status='FAILED'
    try:
        from tools.visualization.policy_preview import gpu_slot
        with gpu_slot():
            dev='cuda:0';torch.backends.cuda.matmul.allow_tf32=True
            networks.AMP_DTYPE=torch.bfloat16 if cfg['amp']=='bf16' else None
            _,motion,mg,mdp,mp=reconstruct(project(ck,'motion'),dev)
            _,_,fg,foot_mdp,fp=reconstruct(project(ck,'foot'),dev)
            if not all(torch.equal(v,fp.dec.state_dict()[k]) for k,v in mp.dec.state_dict().items()):
                raise ValueError('Decoder inference projections differ')
            fp.dec=mp.dec
            assert fp.dec is mp.dec
            policies={'motion':mp,'foot':fp}
            for policy in policies.values():policy.eval().requires_grad_(False)
            before={name:{k:v.detach().cpu().clone() for k,v in policy.state_dict().items()} for name,policy in policies.items()}
            paired=ReferenceFootGoalSource(motion,mdp.kin,requested,fg.foot_config)
            clip=motion.names.index('walk1_subject1')
            windows=[reference_window(offset,args.seconds,int(motion.clip_start[clip]),int(motion.clip_end[clip]),
                                      lookahead_s=requested.frames*requested.dt) for offset in args.offsets]
            torch.cuda.synchronize()
            np.savez_compressed(out/'walk1_subject1_reference_feet.npz',control_dt=.02,
                position=paired.position.cpu().numpy(),quaternion=paired.quaternion.cpu().numpy(),
                velocity=paired.velocity.cpu().numpy(),contact=paired.contact.cpu().numpy(),
                swing_phase=paired.phase.cpu().numpy(),reference_frame=np.arange(motion.num_frames),
                contact_config=json.dumps(asdict(paired.contact_cfg)))
            ev=out/'evaluation'/f'iteration_{ck["iteration"]:06d}_reference_foot'
            for name in policies:(ev/name).mkdir(parents=True)
            reports={name:[] for name in policies};initial={};latent={}

            @torch.no_grad()
            def episode(env,controller,offset,start,H):
                goals=mg if controller=='motion' else paired;pi=policies[controller]
                env.goals=goals;env.max_episode_steps=H;env.reset_noise=False
                torch.manual_seed(args.seed);env.done_eval.zero_();env.failed_eval.zero_()
                env.reset(torch.tensor([0],device=dev),torch.tensor([start],device=dev))
                S=env.state();initial[(controller,offset)]=flatten_state(S).cpu().numpy().copy()
                from supermimic.standing_diagnostics import NewtonGroundContacts
                contacts=NewtonGroundContacts(env)
                command=paired.new_command(S,frames=env.f)
                path=ev/controller/f'{controller}_{args.seconds:g}s_seed{args.seed}_offset{offset:g}.npz'
                trace={k:[] for k in ('state','command_step','reference_frame','action','a_prev','reward',
                    'term','timeout','pd_target','contact_count','contact_force_w','reference_foot_p',
                    'reference_foot_quat','reference_contact','reference_phase','goal','latent')}
                counts=np.full((1,2),-1);force=np.full((1,2,3),np.nan)
                def capture():
                    S=env.state();ref=paired.query(command,command_time(env.ep_step,.02,torch.float64)[:,None])
                    values=dict(state=flatten_state(S),command_step=env.ep_step,reference_frame=env.f,
                        reference_foot_p=ref.position[:,0],reference_foot_quat=ref.quaternion[:,0],
                        reference_contact=ref.contact[:,0],reference_phase=ref.swing_phase[:,0])
                    for k,v in values.items():trace[k].append(v.detach().cpu().numpy().copy())
                    trace['contact_count'].append(counts.copy());trace['contact_force_w'].append(force.copy())
                capture();history=None;reason='running';native={};total=0.;step=0;max_input_z=0.
                while step<H:
                    S=env.state();obs=goals.obs(mdp,S,env.a_prev,env.xi,env.f,env.command if controller=='foot' else None)
                    history=obs[:,None].repeat(1,pi.hist,1) if history is None else torch.cat((history[:,1:],obs[:,None]),1)
                    goal=goals.goal(S,env.f,env.ep_step,env.command if controller=='foot' else None)
                    if goal.shape!=(1,730 if controller=='motion' else 230) or not torch.isfinite(goal).all():
                        raise ValueError('Invalid Actor Goal shape/value')
                    z=pi.encode(goal);action=bound_action(pi.decode(history,z))
                    if not torch.isfinite(action).all():raise FloatingPointError('Nonfinite Actor action')
                    max_input_z=max(max_input_z,float(pi.goal_norm(goal).abs().max()))
                    if step==0:latent[(controller,offset)]=z.cpu().numpy().copy()
                    for k,v in dict(goal=goal,latent=z,action=action,a_prev=env.a_prev,pd_target=env.pd_target(action)).items():
                        trace[k].append(v.detach().cpu().numpy().copy())
                    transition=env.step(action);step+=1;S1=env.state();total+=float(transition['r'][0])
                    finite=all(bool(torch.isfinite(v).all()) for v in S1)
                    mterm=bool(transition['term'][0]);fterm=bool(foot_mdp.termination(S1)[0])
                    physical=not finite or mterm or fterm
                    timeout=bool(transition['timeout'][0]);counts,force,_=contacts.read()
                    trace['reward'].append(transition['r'].cpu().numpy().copy())
                    trace['term'].append(np.array([physical]));trace['timeout'].append(np.array([timeout]))
                    capture()
                    if step%100==0:
                        print(f'[Newton] {controller} offset={offset:g}s: {step}/{H} control steps',flush=True)
                    if physical or timeout:
                        flags=[]
                        if not finite:flags.append('nonfinite')
                        if mterm:
                            if bool((S1[0][0,2]-motion.root_p[env.f][0,2]).abs()>mdp.cfg.term_height):flags.append('motion_height')
                            if bool((motion.root_R[env.f][0,2,2]-S1[1][0,2,2]).abs()>mdp.cfg.term_tilt):flags.append('motion_tilt')
                        if fterm:
                            if bool((S1[0][0,2]-foot_mdp.nominal_height).abs()>foot_mdp.cfg.term_height):flags.append('foot_height')
                            if bool(1-S1[1][0,2,2]>foot_mdp.cfg.term_tilt):flags.append('foot_tilt')
                        native=dict(motion_physical=mterm,foot_physical=fterm,nonfinite=not finite)
                        reason='+'.join(flags) if physical else 'timeout';break
                arrays={k:np.stack(v) for k,v in trace.items()}
                np.savez_compressed(path,**arrays,xi=env.xi.cpu().numpy(),episode_lengths=np.array([step]),
                    control_dt=.02,task_contract=json.dumps(mdp.contract()),
                    **{'command_'+k:v.cpu().numpy() for k,v in command.tensors().items()})
                metrics=measure(arrays)
                row=dict(interface=controller,seed=args.seed,offset_s=offset,duration_s=args.seconds,
                    episode_lengths=[step],stop_reasons=[reason],n_failed=int(reason!='timeout'),n_timeout=int(reason=='timeout'),
                    norm_return=total/(H*mdp.max_step_reward),prefix_normalized_return=total/(step*mdp.max_step_reward),
                    metrics=metrics,native_termination=native,max_abs_normalized_goal=max_input_z,
                    trace=str(path),record=str(path),actor_goal_dim=goal.shape[-1],controller=controller)
                print(json.dumps({k:row[k] for k in ('controller','offset_s','norm_return','episode_lengths','stop_reasons')}),flush=True)
                return row

            @torch.no_grad()
            def measure(trace):
                S=unflatten_state(torch.as_tensor(trace['state'][:,0],device=dev))
                frames=torch.as_tensor(trace['reference_frame'][:,0],device=dev)
                p,R,v,w=mdp.kin.forward(*S);feet=paired.foot_indices
                upper=[i for i,n in enumerate(JOINT_NAMES) if not n.startswith('leg_')]
                refp=motion.body_p[frames][:,feet];refR=motion.body_R[frames][:,feet]
                refc=paired.contact[frames]
                loaded=torch.as_tensor(trace['contact_force_w'][1:,0,:,2]>1,device=dev)
                err=(p[:,feet]-refp).norm(dim=-1)
                metrics=dict(foot_position_rmse_m=float(err[1:].square().mean().sqrt()),
                    foot_orientation_rmse_deg=float(so3_log(refR[1:].transpose(-1,-2)@R[1:,feet]).norm(dim=-1).square().mean().sqrt()*180/np.pi),
                    foot_velocity_rmse_m_s=float((v[1:,feet]-motion.body_v[frames][1:,feet]).square().sum(-1).mean().sqrt()),
                    root_xy_world_rmse_m=float((S[0][1:,:2]-motion.root_p[frames][1:,:2]).square().sum(-1).mean().sqrt()),
                    root_height_tracking_rmse_m=float((S[0][1:,2]-motion.root_p[frames][1:,2]).square().mean().sqrt()),
                    root_pitch_max_abs_deg=float(torch.asin((-S[1][:,2,0]).clamp(-1,1)).abs().max()*180/np.pi),
                    root_roll_max_abs_deg=float(torch.atan2(S[1][:,2,1],S[1][:,2,2]).abs().max()*180/np.pi),
                    upper_body_pose_rmse_rad=float((S[4][1:,upper]-motion.q[frames][1:,upper]).square().mean().sqrt()),
                    joint_pose_rmse_rad=float((S[4][1:]-motion.q[frames][1:]).square().mean().sqrt()),
                    upper_velocity_rms_rad_s=float(S[5][1:,upper].square().mean().sqrt()),
                    upper_acceleration_rms_rad_s2=float(((S[5][1:,upper]-S[5][:-1,upper])/.02).square().mean().sqrt()),
                    contact_state_accuracy=float((loaded==refc[1:]).float().mean()),
                    loaded_foot_slip_proxy_m_s=float((v[1:,feet,:2].norm(dim=-1)*loaded).sum()/loaded.sum().clamp_min(1)),
                    action_rms=float(np.sqrt(np.mean(trace['action']**2))),
                    action_clip_fraction=float(np.mean(np.abs(trace['action'])>=20)),
                    latent_saturation_fraction=float(np.mean(np.abs(trace['latent'])>.95)))
                edges=(~refc[:-1])&refc[1:]
                metrics['reference_touchdown_events']=int(edges.sum())
                metrics['reference_touchdown_rmse_m']=float(err[1:][edges].square().mean().sqrt()) if bool(edges.any()) else None
                return metrics

            from isaaclab.app import launch_simulation
            from supermimic.env import TrackingSim,make_sim
            sim=make_sim(dev,args)
            with launch_simulation(sim,args):
                env=TrackingSim(sim,1,motion,mdp,eval_mode=True,reset_noise=False,goals=mg)
                for offset,(start,H) in zip(args.offsets,windows):
                    for name in policies:
                        reports[name].append(episode(env,name,offset,start,H))
                        json_write(out/'partial.json',dict(status='INCOMPLETE_UNTIL_SUMMARY',reports=reports))
                    gap=float(np.max(np.abs(initial['motion',offset]-initial['foot',offset])))
                    if gap>1e-6:raise RuntimeError('Controller initial physical states differ')
            for name,policy in policies.items():
                if not all(torch.equal(v.cpu(),before[name][k]) for k,v in policy.state_dict().items()):
                    raise RuntimeError('Evaluation changed Actor or normalization')
                json_write(ev/name/'summary.json',dict(interface=name,iteration=ck['iteration'],checkpoint=str(checkpoint),
                    full=False,rows=reports[name],goal_source='reference_foot' if name=='foot' else 'motion',
                    protocol='Matched reference-reset diagnostic with union of original physical guards',normalization_unchanged=True))
            results=[]
            for m,f,offset in zip(reports['motion'],reports['foot'],args.offsets):
                survived=f['n_failed']==0 and f['episode_lengths'][0]==round(args.seconds/.02)
                track=survived and f['metrics']['foot_position_rmse_m']<=.10
                full=track and f['metrics']['upper_body_pose_rmse_rad']<=.40 and f['norm_return']>=m['norm_return']-.10
                a,b=latent['motion',offset].flatten(),latent['foot',offset].flatten()
                results.append(dict(offset_s=offset,foot_survived=survived,foot_tracking_gate=track,
                    retrieval_with_pose_gate=full,initial_state_max_abs_gap=float(np.max(np.abs(initial['motion',offset]-initial['foot',offset]))),
                    initial_latent_cosine=float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))))
            summary=dict(status='EVALUATED',kind=protocol['kind'],iteration=ck['iteration'],checkpoint=str(checkpoint),checkpoint_sha256=source_sha,
                protocol=str(out/'protocol.json'),evaluation_dir=str(ev),rows=reports,paired_results=results,
                shared_decoder_identity=fp.dec is mp.dec,all_weights_and_norms_unchanged=True,
                actual_newton_transitions=sum(r['episode_lengths'][0] for rows in reports.values() for r in rows),
                foot_tracking_passes=sum(r['foot_tracking_gate'] for r in results),
                retrieval_pose_passes=sum(r['retrieval_with_pose_gate'] for r in results),videos=[])
        if digest(checkpoint)!=source_sha:raise RuntimeError('Source checkpoint changed during evaluation')
        summary['wall_seconds']=time.monotonic()-started
        summary['source_checkpoint_unchanged']=True
        json_write(out/'summary.json',summary)
        report(args,out,summary)
        status='COMPLETED';print(out/'REPORT.md',flush=True)
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)
        json_write(out/'status.json',dict(status=status,wall_seconds=time.monotonic()-started,checkpoint_sha256=source_sha))


def report(args,out,summary):
    lines=['# Walking Reference → Foot Encoder 有界验证','',
        f'Checkpoint：`{summary["checkpoint"]}`；iteration **{summary["iteration"]}**；SHA256 `{summary["checkpoint_sha256"]}`。',
        '', '两分支从完全相同的reference全身状态开始，固定seed、无扰动。Foot仅接收自身历史观测和从walk1_subject1原始FK提取的230D足部Goal；当前Foot Encoder/共享Decoder/Norm/mask均保留。全身reference仅用于初始化、独立评分和可视化，未输入Foot Actor。没有训练或Normalizer更新。',
        '', '本轮是有利初态下的技能调用探针，不能等同原Quintic任务或从中立站姿启动；三个窗口都来自训练Walking clip，不是holdout。两分支均执行原Motion与原Foot物理终止条件的并集，没有放宽条件。',
        '', '| 分支 | 起点/s | 实际执行/s | 终止原因 | Motion NR | 足位置RMSE/m | 上身姿态RMSE/rad | 接触准确率 | Slip proxy/m/s |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for name,rows in summary['rows'].items():
        for row in rows:
            m=row['metrics']
            lines.append(f'| {name} | {row["offset_s"]:g} | {row["episode_lengths"][0]*.02:.2f} | {row["stop_reasons"][0]} | {row["norm_return"]:.4f} | {m["foot_position_rmse_m"]:.4f} | {m["upper_body_pose_rmse_rad"]:.4f} | {m["contact_state_accuracy"]:.3f} | {m["loaded_foot_slip_proxy_m_s"]:.3f} |')
    lines+=['','NR使用同一原Motion奖励、除以完整请求时长的理论最大回报；失败后的后缀不补奖励。RMSE仅覆盖实际执行前缀，各分支前缀长度不同。接触参考是几何/速度滞回估计，实际接触来自Newton末子步Fz>1N；slip为承载足link速度代理。',
        '', f'Foot完整存活：{sum(r["foot_survived"] for r in summary["paired_results"])}/{len(summary["paired_results"])}；完整时长且世界足RMSE≤0.10m：**{summary["foot_tracking_passes"]}/{len(summary["paired_results"])}**；再加上身RMSE≤0.40rad且Motion NR下降≤0.10：**{summary["retrieval_pose_passes"]}/{len(summary["paired_results"])}**。这些是运行前写入protocol.json的工程门槛，不是自然性统计认证。',
        f'实际Newton样本：{summary["actual_newton_transitions"]}；耗时：{summary["wall_seconds"]:.1f}s。所有模型权重和归一化逐张量未改变，Decoder为同一个对象，源Checkpoint SHA在执行前后相同。',
        '', '## 证据与视频','', '- [预注册协议](protocol.json)','- [完整结果](summary.json)',
        '- [提取的原始足部reference](walk1_subject1_reference_feet.npz)',
        '', '视频左侧Motion Encoder、右侧Foot Encoder，两侧青色机器人均为原Walking参考；右侧附足部目标路径。物理终止后该侧冻结且明确标注，不把后续静止画面计入存活或回报。']
    if summary.get('combined_video'):lines.append(f'- **[三个窗口合并视频]({Path(summary["combined_video"]).name})**')
    for path in summary['videos']:lines.append(f'- [offset {Path(path).stem.split("offset")[-1]} 秒视频]({Path(path).name})')
    lines+=['','## 输入分布诊断（仅观察，不是因果结论）','',
        '| 参考起点/s | Foot Goal 最大绝对归一化值 | Foot latent中abs(z)>0.95的比例 |',
        '| --- | --- | --- |']
    for row in summary['rows']['foot']:
        lines.append(f'| {row["offset_s"]:g} | {row["max_abs_normalized_goal"]:.2f} | {100*row["metrics"]["latent_saturation_fraction"]:.1f}% |')
    if (out/'goal_distribution_audit.json').exists():
        lines+=['', '[分通道复核](goal_distribution_audit.json)：最大离群来自足部旋转6D通道。部分训练通道的均值为0、标准差落在eps=0.01下限；原Walking的Pitch/Roll输入会被大幅放大。本轮保持冻结Normalizer，没有为了通过测试重估或裁剪输入。这是输入分布差异和latent饱和的直接证据，尚未证明它是唯一失败原因。']
    worst=max(summary['rows']['motion'],key=lambda row:row['metrics']['foot_position_rmse_m'])
    lines+=['', f'Motion对照也不是完美足轨迹跟踪上界：{worst["offset_s"]:g}秒窗口世界足RMSE={worst["metrics"]["foot_position_rmse_m"]:.3f}m。原Motion奖励允许一定全局足误差，不能把此对照称为完美足目标控制器。']
    if all(row['n_failed'] and row['episode_lengths'][0]*.02<1 for row in summary['rows']['foot']):
        lines+=['Foot三个窗口均在不到1秒时物理终止，负向结论不依赖0.10m精度阈值是否过严。']
    lines+=['本实验保留了各分支自己的ObsNorm/GoalNorm及Foot观测mask，因此验证的是当前完整Foot推理接口，而非已将Encoder权重、归一化和观测适配的因果影响分别隔离。']
    lines+=['', '结论：'+('在本轮部分窗口观察到足目标调用及姿态保持的正向证据，仍需更长及独立数据验证。' if summary['retrieval_pose_passes'] else
        '本轮未通过足目标技能调用与姿态保持门槛，不能认为当前Foot Encoder已能可靠调用既有Walking技能。'),
        '可能机制（尚未验证因果）：LAFAN足目标与Quintic训练分布不同、Foot Encoder的latent语义未与Motion Encoder对齐、两接口ObsNorm及最后6D观测适配不同。不能用单次失败断言Decoder本身失去Walking能力；Motion分支为本次匹配对照。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    from isaaclab.app import add_launcher_args
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--offsets',type=float,nargs='+',default=[0.,60.,120.])
    p.add_argument('--seconds',type=float,default=10.)
    p.add_argument('--seed',type=int,default=54);p.add_argument('--dry-run',action='store_true')
    p.add_argument('--no-video',action='store_true');p.add_argument('--process-budget-s',type=float,default=600.)
    add_launcher_args(p);args=p.parse_args()
    if (not 1<=len(args.offsets)<=3 or len(set(args.offsets))!=len(args.offsets)
            or not all(math.isfinite(x) and x>=0 for x in args.offsets)
            or not 0<args.seconds<=10 or not 30<=args.process_budget_s<=900):
        p.error('Use 1–3 distinct offsets, <=10 seconds per episode and a 30–900 second process budget')
    if args.no_video or args.dry_run:
        run(args);return
    # CPU-only orchestrator: wait for the Newton child to EXIT and release CUDA
    # before starting any renderer, which independently checks the GPU lock.
    started=time.monotonic();out=Path(args.out).resolve()
    subprocess.run([sys.executable,'-u','-B',str(Path(__file__).resolve()),*sys.argv[1:],'--no-video'],
                   check=True,timeout=args.process_budget_s+15)
    summary=json.loads((out/'summary.json').read_text())
    try:
        for offset in args.offsets:
            remaining=args.process_budget_s-(time.monotonic()-started)
            if remaining<5:raise TimeoutError('No process budget left for reference comparison videos')
            video=out/f'motion_vs_reference_foot_offset{offset:g}.mp4'
            subprocess.run([sys.executable,'-B',str(ROOT/'tools/visualization/render_s54_joint_comparison.py'),
                '--run_dir',str(Path(args.checkpoint).resolve().parent),'--eval-dir',summary['evaluation_dir'],
                '--motion-offset',str(offset),'--foot-duration',str(args.seconds),'--seed',str(args.seed),
                '--seconds',str(args.seconds),'--out',str(video)],check=True,timeout=remaining)
            summary['videos'].append(str(video))
        summary['combined_video']=combine_videos(out,summary['videos'],
            timeout=max(1,args.process_budget_s-(time.monotonic()-started)))
        summary['evaluation_wall_seconds']=summary['wall_seconds']
        summary['wall_seconds']=time.monotonic()-started
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        report(args,out,summary)
        (out/'status.json').write_text(json.dumps(dict(status='COMPLETED',phase='evaluation_and_video_complete',
            wall_seconds=summary['wall_seconds']),indent=2)+'\n')
        print(out/'REPORT.md',flush=True)
    except BaseException:
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        (out/'status.json').write_text(json.dumps(dict(status='EVALUATED_VIDEO_FAILED',
            wall_seconds=time.monotonic()-started),indent=2)+'\n')
        raise


if __name__=='__main__':main()
