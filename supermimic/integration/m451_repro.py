"""M4.5.1 冻结 Newton 复现与第四次落脚逐帧取证。"""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

if __name__ == '__main__':
    os.environ['SUPERMIMIC_ROBOT'] = 'roban'

import numpy as np
import torch

from ..mdp import flatten_state
from ..motion import CONTROL_DT
from ..roban import FOOT_BODY_NAMES, JOINT_NAMES, model_contract
from ..rotations import quat_to_mat
from .m43_evaluation import summarize_gait
from .m44_evaluation import baseline_checkpoints, control_trace, scene, trial
from .m4_evaluation import (CONTACT_FORCE_N, MIN_STANCE_S, MIN_SWING_LIFT_M, MIN_SWING_S,
                            TOUCHDOWN_WINDOW_S, TOUCHDOWN_XY_ERROR_M, read_only_pose)
from .m4_task import M4EpaEvaluationError


SEEDS = (1001, 1002, 1003)


@torch.no_grad()
def collect_parallel(env, pi, contacts, solver, seeds, reset_hook=None):
    """原 Actor 与守卫按完整批量执行，逐世界保留终止前的转移。"""
    if len(seeds) != env.N:
        raise ValueError('批量环境数与种子数量必须一致')
    torch.manual_seed(int(seeds[0]))
    ids = torch.arange(env.N, device=env.device)
    env.reset(ids)
    if reset_hook is not None:
        reset_hook(env)
    motion, mdp = env.motion, env.mdp
    data = env.robot.data
    foot_ids = [env.robot.body_names.index(name) for name in FOOT_BODY_NAMES]
    mass = data.body_mass.torch.reshape(env.N, -1, 1)
    rows = defaultdict(list)

    def record(**values):
        for key, value in values.items():
            rows[key].append(value.detach().cpu().numpy().copy() if torch.is_tensor(value) else value.copy())

    def capture(state, count, force, other, measured):
        pose = data.body_link_pose_w.torch[:, foot_ids]
        rotation = quat_to_mat(pose[..., [6, 3, 4, 5]])
        velocity = data.body_link_vel_w.torch[:, foot_ids]
        offset = (rotation @ motion.sole_offsets[..., None]).squeeze(-1)
        pos = pose[..., :3]-env.scene.env_origins[:, None]+offset
        com = (mass*data.body_com_pose_w.torch[..., :3]).sum(1)/mass.sum(1)-env.scene.env_origins
        com_vel = (mass*data.body_com_vel_w.torch[..., :3]).sum(1)/mass.sum(1)
        record(state=flatten_state(state), frame=env.f, xi=env.xi, foot=pos, foot_R=rotation,
               foot_angular_velocity=velocity[..., 3:],
               foot_velocity=velocity[..., :3]+torch.cross(velocity[..., 3:],offset,dim=-1),
               contact=force[...,2]>=CONTACT_FORCE_N, contact_measured=np.full(env.N,measured),
               foot_contact_count=count, nonfoot_ground_count=other, force=force.astype(np.float32),
               solver_warmstart=solver.warmstart, solver_qpos=solver.qpos, solver_qvel=solver.qvel,
               solver_nefc=solver.nefc, solver_niter=solver.niter, solver_overflow=solver.overflow,
               root_quat_xyzw=data.root_link_pose_w.torch[...,3:7], actuator_torque=solver.torque[:,solver.dofs],
               joint_limit_force=solver.joint_limit_forces(), com_position=com, com_velocity=com_vel)

    state = env.state()
    capture(state,np.zeros((env.N,2),dtype=np.int64),np.zeros((env.N,2,3)),np.zeros(env.N,dtype=np.int64),False)
    term_names, history = (), None
    steps = np.full(env.N, -1, dtype=int)
    reasons = ['reference_end']*env.N
    fault = {}
    with read_only_pose(env.robot):
        for tick in range(motion.num_frames-1):
            obs = env.goals.obs(mdp,state,env.a_prev,env.xi,env.f,env.command)
            goal = env.goals.goal(state,env.f,env.ep_step,env.command)
            history = (obs[:,None].expand(-1,pi.hist,-1) if history is None else
                       torch.cat((history[:,1:],obs[:,None]),dim=1))
            action,raw = pi(history,goal,return_logits=True)
            target,prev = env.pd_target(action),env.a_prev.clone()
            try:
                out = env.step(action)
                state = env.state()
                count,force,other = contacts.read()
                if not np.isfinite(force).all() or not bool(out['r'].isfinite().all()):
                    raise FloatingPointError('Newton 批量接触或奖励含非有限值')
            except FloatingPointError as exc:
                # 与原入口相同，故障步不作为有效数据，保留已有各世界轨迹。
                kind = 'epa_horizon' if isinstance(exc,M4EpaEvaluationError) else 'fatal_dynamics'
                for i in np.flatnonzero(steps<0):
                    steps[i],reasons[i] = tick,kind
                    fault[i] = np.asarray(str(exc))
                break
            term_names = tuple(k for k in out['terms'] if k.endswith('_contribution'))
            terms = torch.stack([out['terms'][k] for k in term_names],dim=1)
            torch.testing.assert_close(terms.sum(1),out['r'])
            record(observation=obs,goal=goal,action=action,raw_action=raw,previous_action=prev,target=target,
                   clamp_mask=mdp.target_clip_mask(action),reward=out['r'],reward_terms=terms,
                   term=out['term'],timeout=out['timeout'])
            capture(state,count,force,other,True)
            done = out['done'].cpu().numpy() & (steps<0)
            low = (state[0][:,2]<mdp.cfg.term_height).cpu().numpy()
            tilt = (1-state[1][:,2,2]>mdp.cfg.term_tilt).cpu().numpy()
            term = out['term'].cpu().numpy()
            for i in np.flatnonzero(done):
                steps[i] = tick+1
                if term[i]:
                    reasons[i] = '+'.join(k for k,flag in (('root_height',low[i]),('root_tilt',tilt[i])) if flag) or 'physical_termination'
            if (steps>=0).all():
                break

    if (steps<0).any():
        steps[steps<0] = len(rows['action'])
    state_keys = {k for k,v in rows.items() if len(v)==len(rows['state'])}
    arrays = {k:np.asarray(v) for k,v in rows.items()}
    traces = []
    for i,(seed,length) in enumerate(zip(seeds,steps)):
        trace = {k:v[:length+int(k in state_keys),i] for k,v in arrays.items()}
        cmd = env.command.index(ids[i:i+1])
        trace.update({f'command_{k}':v[0].cpu().numpy() for k,v in cmd.tensors().items()})
        trace['time'] = trace['frame'].astype(np.float64)*CONTROL_DT
        trace['termination'],trace['seed'] = np.asarray(reasons[i]),np.asarray(seed)
        trace['max_step_reward'] = np.asarray(mdp.max_step_reward)
        if i in fault:
            trace['fault_detail'] = fault[i]
        f = torch.as_tensor(trace['frame'],device=env.device)
        ref = motion.sample(f[None],cmd)
        for k in ('root_p','root_R','foot_p','foot_R','foot_v','contact','contact_valid','phase'):
            trace['planned_'+k] = ref[k][0].cpu().numpy()
        trace['foot_target_error'] = np.linalg.norm(trace['foot']-trace['planned_foot_p'],axis=-1)
        trace['reward_term_names'] = np.asarray([k.removesuffix('_contribution') for k in term_names])
        if 'action' not in trace:
            for k,width in (('action',mdp.alpha.numel()),('raw_action',mdp.alpha.numel()),('previous_action',mdp.alpha.numel()),
                            ('target',mdp.alpha.numel()),('clamp_mask',mdp.alpha.numel()),('observation',mdp.obs_dim),('goal',env.goals.spec.dim)):
                trace[k] = np.empty((0,width), dtype=bool if k=='clamp_mask' else np.float32)
            for k,dtype in (('reward',np.float32),('term',bool),('timeout',bool)):
                trace[k] = np.empty(0,dtype=dtype)
            trace['reward_terms'] = np.empty((0,0),dtype=np.float32)
        control_trace(trace,mdp,motion)
        traces.append(trace)
    return traces


def split_archive(record, prefix):
    return {k[len(prefix):]:record[k] for k in record if k.startswith(prefix)}


def checkpoints():
    b = baseline_checkpoints(Path('outputs/m4_3'))
    c = Path('outputs/m4_5/checkpoints')
    return {'m45_5000':c/'iter_5000.pt','m45_4500':c/'iter_4500.pt','m43_4500':b[0],'m43_5000':b[1]}


def run_periodic(args, path, iteration):
    # 原入口用硬链接发布 Best，临时目录必须与检查点在同一文件系统。
    with TemporaryDirectory(prefix='.periodic-',dir=args.out) as tmp:
        (Path(tmp)/'checkpoints').mkdir()
        subprocess.run([sys.executable,'-m','supermimic.integration.m43_evaluation',
                        '--out',tmp,'--device',args.device,'--m4_reference',str(args.m4_reference),
                        '--checkpoint',str(path),'--upper-body','--seeds',*map(str,SEEDS)],check=True)
        output = Path(tmp)/'evaluation'
        r = json.loads((output/'results.json').read_text())['milestones'][str(iteration)]['per_reset']
        with np.load(output/'rollouts.npz',allow_pickle=False) as z:
            return [(split_archive(z,f'i{iteration}_s{s}_'), row) for s,row in zip(SEEDS,r)]


def run_scene(args, path, iteration, seeds, *, parallel=False):
    with scene(args,path,expected_iteration=iteration,num_envs=len(seeds) if parallel else 1) as (env,pi,contacts,solver,_,_):
        if parallel:
            traces = collect_parallel(env,pi,contacts,solver,seeds)
            return [(trace,summarize_gait(trace,env.motion,env.mdp)) for trace in traces]
        return [trial(env,pi,contacts,solver,seed) for seed in seeds]


def collect(args):
    archive,runs = {'seeds':np.asarray(SEEDS)},{}
    args.out.mkdir(parents=True,exist_ok=True)
    if args.resume and (args.out/'reproduction.json').is_file():
        runs = json.loads((args.out/'reproduction.json').read_text())
        with np.load(args.out/'repro_rollouts.npz',allow_pickle=False) as z:
            archive.update({k:z[k] for k in z.files})
    for label,path in checkpoints().items():
        iteration = int(label.rsplit('_',1)[1])
        methods = ('periodic','independent','fresh_single','parallel') if label.startswith('m45') else ('independent',)
        for rep in range(1,args.repeats+1 if label.startswith('m45') else 2):
            for method in methods:
                key = f'{label}_{method}_r{rep}'
                if key in runs:
                    continue
                started = perf_counter()
                if method=='periodic':
                    result = run_periodic(args,path,iteration)
                elif method=='fresh_single':
                    result = [run_scene(args,path,iteration,[seed])[0] for seed in SEEDS]
                else:
                    result = run_scene(args,path,iteration,SEEDS,parallel=method=='parallel')
                r = []
                for trace,row in result:
                    seed = int(row['seed'])
                    archive.update({f'{key}_s{seed}_{k}':v for k,v in trace.items()})
                    r.append(row)
                runs[key] = dict(wall_s=perf_counter()-started,trials=r)
                print(key,'/'.join(str(row['matched_touchdowns']) for row in r),flush=True)
        # 按权重存档，保留中断前的证据。
        with TemporaryDirectory(dir=args.out) as tmp:
            path_tmp = Path(tmp)
            np.savez_compressed(path_tmp/'repro_rollouts.npz',**archive)
            (path_tmp/'reproduction.json').write_text(json.dumps(runs,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
            for name in ('repro_rollouts.npz','reproduction.json'):
                (path_tmp/name).replace(args.out/name)


def checkpoint_info():
    from importlib.metadata import version

    result = {}
    for name,path in checkpoints().items():
        ck = torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream,'sha256').hexdigest()
        norm = {k:dict(shape=list(v.shape),min=float(v.min()),max=float(v.max()))
                for k,v in ck['pi'].items() if '_norm.' in k and v.numel()}
        result[name] = dict(path=str(path),iteration=ck['iteration'],sha256=digest,normalization=norm,
                            saved_environment=ck['args'].get('environment_versions'),
                            runtime={k:version(k) for k in ('newton','mujoco-warp','warp-lang')})
    last = Path('outputs/m4_5/checkpoints/last.pt')
    best = last.with_name('best.pt')
    result['m45_aliases_same_file_as_5000'] = all(x.samefile(checkpoints()['m45_5000']) for x in (best,last))
    return result


def touchdown_candidates(trace):
    from .m43_evaluation import _runs

    time,foot,contact = (trace[k] for k in ('time','foot','contact'))
    foot,contact = foot[:,1],contact[:,1]
    plan = trace['planned_foot_p'][round(4.6/CONTROL_DT),1,:2] if time[-1]>=4.6 else None
    if plan is None:
        return []
    mask = (time>=3.8-TOUCHDOWN_WINDOW_S)&(time<=4.6+TOUCHDOWN_WINDOW_S)&~contact
    events = []
    for start,stop in _runs(mask):
        if not start or stop>=len(time) or not contact[start-1] or (stop-start)*CONTROL_DT+1e-8<MIN_SWING_S:
            continue
        if float((foot[start:stop,2]-foot[start-1,2]).max())<MIN_SWING_LIFT_M or abs(time[stop]-4.6)>TOUCHDOWN_WINDOW_S:
            continue
        xy = float(np.linalg.norm(foot[stop,:2]-plan))
        if xy>TOUCHDOWN_XY_ERROR_M:
            continue
        n = int(next((k for k,x in enumerate(contact[stop:]) if not x),len(contact)-stop))
        force = trace['force']
        events.append(dict(t=float(time[stop]),xy_error_m=xy,stance_frames=n,stance_s=n*CONTROL_DT,
                           overlap_ok=n*CONTROL_DT+1e-8>=MIN_STANCE_S,
                           loss_s=float(time[stop+n]) if stop+n<len(time) else None,
                           root_z_m=float(trace['state'][stop,2]),
                           touchdown_right_v_xyz=trace['foot_velocity'][stop,1].tolist(),
                           touchdown_l_r_force_z=force[stop,:,2].tolist(),
                           next_right_force_z=force[stop:min(stop+5,len(time)),1,2].tolist()))
    return events


def compare_traces(base, other):
    def maximum(field):
        if field not in base or field not in other:
            return np.empty((0,))
        n = min(len(base[field]),len(other[field]))
        return np.max(abs(base[field][:n]-other[field][:n]).reshape(n,-1),axis=1) if n else np.empty((0,))

    def first(x,threshold):
        crossings = np.flatnonzero(x>threshold)
        return int(crossings[0]) if len(crossings) else None

    fields = ('state','solver_qpos','solver_qvel','solver_warmstart','force','raw_action','target','observation','goal')
    series = {k:maximum(k) for k in fields}
    result = {k:dict(t0=float(v[0]),max=float(v.max()),first_gt_1e6=first(v,1e-6)) for k,v in series.items() if v.size}
    q1,q2 = base['state'],other['state']
    n = min(len(q1),len(q2))
    pos = np.max(abs(q1[:n,:3]-q2[:n,:3]),axis=1)
    joint = np.max(abs(q1[:n,13:34]-q2[:n,13:34]),axis=1)
    state_split = first(np.maximum(pos,joint),.001)
    force_split = first(series['force'],1.)
    contact_split = first(np.any(base['contact'][:n]!=other['contact'][:n],axis=1),.5)
    result.update(first_1mm_or_1mrad=state_split,first_force_gt_1n=force_split,first_contact=contact_split,
                  first_window=[])
    observed = [i for i in (state_split,force_split,contact_split) if i is not None]
    if observed:
        first_step = min(observed)
        for i in range(max(0,first_step-2),min(n,first_step+3)):
            row = dict(step=i,time_s=i*CONTROL_DT,root_m=float(pos[i]),joint_rad=float(joint[i]),
                       force_z_a=base['force'][i,:,2].tolist(),force_z_b=other['force'][i,:,2].tolist())
            row.update({k:float(series[k][i]) for k in ('solver_warmstart','solver_qpos','solver_qvel') if len(series[k])>i})
            if i and len(series['raw_action'])>=i:
                row['applied_a_delta'] = float(series['raw_action'][i-1])
                row['applied_target_delta'] = float(series['target'][i-1])
            for field in ('solver_nefc','solver_niter','solver_overflow'):
                row[field] = [int(t[field][i]) if field in t else None for t in (base,other)]
            result['first_window'].append(row)
    return result


def fourth_contact_csv(path, traces):
    import mujoco
    from scipy.spatial.transform import Rotation

    model,joints = model_contract()
    data = mujoco.MjData(model)
    lower,upper = np.asarray([j['range'] for j in joints]).T
    qidx = [j['qposadr'] for j in joints]
    vidx = [j['dofadr'] for j in joints]
    ankles = [JOINT_NAMES.index(k) for k in ('leg_l5_joint','leg_l6_joint','leg_r5_joint','leg_r6_joint')]

    def vector(row,key,values,axes='xyz'):
        row.update({f'{key}_{axis}':float(v) for axis,v in zip(axes,values)})

    with path.open('w',newline='') as stream:
        writer = None
        for key,trace in traces.items():
            time = trace['time']
            for i in np.flatnonzero((time>=3.78)&(time<=5.02)):
                state = trace['state'][i]
                rot = Rotation.from_quat(state[[4,5,6,3]])
                roll,pitch,_ = rot.as_euler('xyz',degrees=True)
                row = dict(run=key,frame=int(i),time_s=float(time[i]),root_z_m=float(state[2]),
                           root_roll_deg=float(roll),root_pitch_deg=float(pitch))
                if 'com_position' in trace:
                    com,vel = trace['com_position'][i],trace['com_velocity'][i]
                else:
                    # 历史数据用统一 MJCF 刚体质量及真实姿态、速度重建质心。
                    data.qpos[:7],data.qpos[qidx] = state[:7],state[13:34]
                    data.qvel[:3],data.qvel[3:6],data.qvel[vidx] = state[7:10],rot.inv().apply(state[10:13]),state[34:55]
                    mujoco.mj_forward(model,data)
                    mujoco.mj_subtreeVel(model,data)
                    com,vel = data.subtree_com[1],data.subtree_linvel[1]
                vector(row,'com',com); vector(row,'com_v',vel)
                for j,side in enumerate(('left','right')):
                    for field,caption in (('foot','actual'),('planned_foot_p','plan'),('foot_velocity','velocity'),('force','force')):
                        vector(row,f'{side}_{caption}',trace[field][i,j])
                    row[f'{side}_bearing_10N'] = int(trace['contact'][i,j])
                    row[f'{side}_xy_tracking_error'] = float(np.linalg.norm(trace['foot'][i,j,:2]-trace['planned_foot_p'][i,j,:2]))
                if len(time)>round(4.6/CONTROL_DT):
                    row['right_xy_landing_error'] = float(np.linalg.norm(trace['foot'][i,1,:2]-trace['planned_foot_p'][round(4.6/CONTROL_DT),1,:2]))
                else:
                    row['right_xy_landing_error'] = ''
                row['right_lost_contact'] = int(trace['contact'][i-1,1] and not trace['contact'][i,1])
                q = state[13:34]
                row['joints_outside_mechanical_limits'] = '|'.join(np.asarray(JOINT_NAMES)[(q<lower-1e-5)|(q>upper+1e-5)])
                row['target_clamped_joints'] = '|'.join(np.asarray(JOINT_NAMES)[trace['clamp_mask'][i-1]])
                for j in ankles:
                    name = JOINT_NAMES[j].removesuffix('_joint')
                    row[f'{name}_torque_Nm'] = float(trace['actuator_torque'][i,j])
                    row[f'{name}_stop_force_Nm'] = float(trace['joint_limit_force'][i,j]) if 'joint_limit_force' in trace else ''
                for j,name in enumerate(JOINT_NAMES):
                    row[f'raw_{name}'] = float(trace['raw_action'][i-1,j])
                    row[f'target_{name}'] = float(trace['target'][i-1,j])
                for field in ('solver_nefc','solver_niter','solver_overflow'):
                    row[field] = int(trace[field][i]) if field in trace else ''
                if writer is None:
                    writer = csv.DictWriter(stream,fieldnames=list(row))
                    writer.writeheader()
                writer.writerow({k:f'{v:.8g}' if isinstance(v,float) else v for k,v in row.items()})


def analyze(args):
    output = args.out/'reproduction.json'
    result = json.loads(output.read_text())
    runs = result.get('runs',result)
    traces,periodic,divergences,landing = {},{},{},{}
    historic_dir = Path('outputs/m4_5/evaluation')
    with np.load(historic_dir/'rollouts.npz',allow_pickle=False) as raw, np.load(args.out/'repro_rollouts.npz',allow_pickle=False) as archive:
        for label in ('m45_4500','m45_5000'):
            for seed in SEEDS:
                iteration = label.rsplit('_',1)[1]
                key = f'{label}_historical_periodic_s{seed}'
                traces[key] = periodic[label,seed] = split_archive(raw,f'i{iteration}_s{seed}_')
                if iteration=='5000':
                    traces[f'{label}_historical_independent_s{seed}'] = split_archive(raw,f'm45_best_s{seed}_')
        for label,group in runs.items():
            for row in group['trials']:
                seed = row['seed']
                key = f'{label}_s{seed}'
                traces[key] = split_archive(archive,key+'_')
                a = '_'.join(label.split('_')[:2])
                if (a,seed) in periodic:
                    divergences[key+'_versus_historical_periodic'] = compare_traces(periodic[a,seed],traces[key])
                if label.endswith('_r2'):
                    peer = f'{label[:-1]}1_s{seed}'
                    if peer in traces:
                        divergences[key+'_versus_repeat1'] = compare_traces(traces[peer],traces[key])
    for key,trace in traces.items():
        landing[key] = touchdown_candidates(trace)
        if 'historical_independent' in key:
            peer = key.replace('historical_independent','historical_periodic')
            divergences[key+'_versus_historical_periodic'] = compare_traces(traces[peer],trace)
    fourth_contact_csv(args.out/'fourth_contact.csv',traces)
    final = dict(runs=runs,checkpoint_identity=result.get('checkpoint_identity') or checkpoint_info(),
                 divergences=divergences,fourth_touchdown_candidates=landing)
    output.write_text(json.dumps(final,ensure_ascii=False,allow_nan=False,indent=2)+'\n')


def main():
    from isaaclab.app import add_launcher_args

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m4_reference',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=Path('outputs/m4_5_1'))
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--resume',action='store_true',help='追加未完成的回放批次')
    parser.add_argument('--analyze',action='store_true',help='汇总已采集的轨迹与第四次落脚')
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('重复次数至少为 1')
    torch.set_num_threads(4)
    (analyze if args.analyze else collect)(args)


if __name__ == '__main__':
    main()
