"""冻结 F2 共同状态上的补偿、BPTT 和 Newton 有限差分诊断。"""

import argparse
from copy import copy
import json
import os
from pathlib import Path
from time import perf_counter

if __name__ == '__main__':
    os.environ['SUPERMIMIC_ROBOT'] = 'roban'

import numpy as np
import torch

from ..kinematics import RobotKinematics
from ..mdp import normalize_xi, unflatten_state
from ..motion import CONTROL_DT
from ..networks import Critic
from ..rotations import so3_log
from .m42v3_ac import cosine, gradient, objective
from .m42v3_wm import select_windows
from .m43_wm_audit import SLOPE_FLOOR, make_policy, make_wm, restore, slope_metrics, task_for, write_json
from .m45_audit import reward_heads
from .m45_upper import ARMS, UPPER
from .m4_task import build_m4_task


EPSILON = .15
SUPPORT_FORCE = 10.
UPPER_WEIGHT = .10
SEED = 4510
TERMS = ('foot', 'clearance', 'touchdown', 'body', 'total')


def rms(x):
    return float(x.square().mean().sqrt())


def common_reward(mdp, state, frame, action, previous):
    reward, heads = reward_heads(mdp, state, frame, action, previous)
    raw = (heads['foot_objective'], heads['swing_clearance_contribution'],
           heads['touchdown_contribution'], heads['body_prior_contribution'], reward)
    return {key: value/mdp.max_step_reward for key, value in zip(TERMS, raw)}


def manifest(archive):
    # 只采用历史统一评估真实完成四步的正常试次，失败组固定为触发根高度终止的组合扰动。
    comparison = json.loads(archive.with_name('results.json').read_text())['comparison']['cases']
    result = {}
    for label, source, subset in (('baseline_success', 'm43_best', 'nominal'),
                                  ('prior_success', 'm45_final', 'nominal'),
                                  ('prior_fall', 'm45_final', 'perturb')):
        rows = comparison[source][subset]['trials']
        failures = subset == 'perturb'
        seeds = [r['seed'] for r in rows if ('root_height' in r['termination'] if failures
                 else r['matched_touchdowns'] == 4 and r['termination'] == 'reference_end')]
        result[label] = [(seed, f"{source}{'_perturb' if failures else ''}_s{seed}_") for seed in seeds]
    return result


def data_view(archive, trials):
    keys = ('frame', 'state', 'observation', 'goal', 'action', 'xi', 'force', 'raw_action',
            'clamp_mask', 'reward', 'term', 'timeout', 'command_start_frame', 'command_clip_id',
            'command_rotation', 'command_translation', 'foot', 'previous_action')
    return {f'i0_s{seed}_{key}': archive[prefix + key] for seed, prefix in trials for key in keys}


def fixture(archive, trials, device, *, perturbed=False):
    columns = {}
    for seed, key in trials:
        def read(name):
            return torch.as_tensor(archive[key+name], device=device)

        n = len(read('action'))
        f = read('frame')[:n]
        force = read('force')[:n, :, 2]
        a, prev = read('action'), read('previous_action')
        upper_amp = (a[:, UPPER] - prev[:, UPPER]).square().mean(-1).sqrt()
        supported = (force >= SUPPORT_FORCE).sum(-1)
        support_change = (supported != torch.roll(supported, 1)) & (f > 0)
        masks = {'initial_double_support': (f < 40) & (supported == 2),
                 'first_swing': (f >= 40) & (f < 80), 'first_touchdown': (f-80).abs() <= 7,
                 'contact_switch': support_change, 'upper_motion': upper_amp >= torch.quantile(upper_amp, .85),
                 'perturb_recovery': (f < 40) & perturbed, 'near_fall': (f >= n-12) & perturbed}
        idx = [torch.linspace(0, n-1, 5, device=device).round().long()]
        for mask in masks.values():
            hits = mask.nonzero(as_tuple=True)[0]
            if len(hits):
                idx.append(hits[torch.linspace(0, len(hits)-1, min(len(hits), 3), device=device).round().long()])
        i = torch.cat(idx).unique()
        prior = (i[:, None] + torch.arange(-9, 1, device=device)).clamp_min(0)
        items = {k: read(k)[i] for k in ('state', 'xi', 'frame', 'action', 'goal', 'target')}
        items.update(history=read('observation')[prior], previous=prev[i], force=force[i],
                     next_foot=read('foot')[i+1], warmstart=read('solver_warmstart')[i],
                     root_quat=read('root_quat_xyzw')[i], end=torch.full_like(i, n),
                     qpos=read('solver_qpos')[i], qvel=read('solver_qvel')[i], seed=torch.full_like(i, seed))
        items.update({name: m[i] for name, m in masks.items()})
        for name, value in items.items():
            columns.setdefault(name, []).append(value)
    return {name: torch.cat(x) for name, x in columns.items()}


def groups(samples):
    names = ('initial_double_support', 'first_swing', 'first_touchdown', 'contact_switch',
             'upper_motion', 'perturb_recovery', 'near_fall')
    return {'all': torch.ones_like(samples['frame'],dtype=torch.bool)} | {k: samples[k].bool() for k in names}


@torch.no_grad()
def predict(model, task, batch, masks):
    _, motion, _, mdp = task
    state = unflatten_state(batch['start'])
    result = {}
    for k in range(8):
        f, a, prev = (batch[name][:, k] for name in ('frame', 'action', 'prior'))
        state = model.step(state, mdp.wm_action(a, f), normalize_xi(batch['xi'][:, k]))
        if k not in (0, 3, 7):
            continue
        real = unflatten_state(batch['next_state'][:, k])
        dq = state[4]-real[4]
        real_d = batch['term'][:, k].float()
        pred_d = mdp.soft_termination(state, f+1)
        estimated = common_reward(mdp,state,f+1,a,prev)
        expected = common_reward(mdp,real,f+1,a,prev)
        errors = {'foot_m': (motion.feet(state)[0]-batch['next_foot'][:,k]).norm(dim=-1),
                  'root_height_m': (state[0]-real[0])[:, 2],
                  'root_orientation_deg': so3_log(state[1] @ real[1].transpose(-1,-2)).norm(dim=-1).rad2deg(),
                  'arm_rad': dq[:, ARMS], 'waist_rad': dq[:, 0],
                  'reward_normalized': estimated['total']-expected['total'],
                  'body_reward_normalized': estimated['body']-expected['body'],
                  'terminal_rmse': pred_d-real_d}
        result[str(k+1)] = {name: dict(n=int(mask.sum()), terminal_true=int(real_d[mask].sum()),
                                       terminal_predicted=int((pred_d[mask] > .5).sum()),
                                       **{key: rms(value[mask]) for key,value in errors.items()})
                            for name, mask in masks.items() if mask.any()}
    return result


def bptt(bundle, task, samples):
    ck, old, pi, model, critic = bundle
    cfg, motion, goals, mdp = task
    f = samples['frame']
    remaining = f+cfg.horizon <= motion.num_frames-1
    source = tuple(samples[k] for k in ('state','previous','frame','xi','history','goal'))
    params = tuple(pi.parameters())
    result = {}
    # 保留原 Critic 权重、归一化和正式上界配置。0.10 只用于共同分项诊断。
    value_upper = ck['imagination_contract']['bootstrap_upper_bound']
    for name, mask in groups(samples).items():
        mask = mask & remaining
        if not mask.any():
            continue
        packet = objective(tuple(x[mask] for x in source), pi, critic, model, motion, mdp, goals,
                           cfg, value_upper, SEED)
        values = {'foot': packet['C'], 'body': packet['direct']['body_prior'], 'bootstrap': packet['bootstrap']}
        norms, outputs = {}, {}
        for key, value in values.items():
            g = gradient(value, params)
            d = torch.autograd.grad(value, packet['actions'], retain_graph=True, allow_unused=True)
            z = torch.stack([torch.zeros_like(a) if b is None else b.detach() for a,b in zip(packet['actions'],d)])
            if not (g.isfinite().all() and z.isfinite().all()):
                raise FloatingPointError(f'{name} {key} 出现非有限梯度')
            norms[key] = g
            outputs[key] = {'arm_rms': rms(z[..., ARMS]), 'waist_rms': rms(z[..., 0]),
                            'leg_rms': rms(z[..., 1:13])}
        result[name] = dict(n=int(mask.sum()), norm={key: float(g.norm()) for key,g in norms.items()},
                            body_foot=cosine(norms['body'],norms['foot']),
                            body_bootstrap=cosine(norms['body'],norms['bootstrap']),
                            foot_bootstrap=cosine(norms['foot'],norms['bootstrap']), output_action=outputs)
    return result


def paired_slopes(model, mdp, samples, direction, truth, valid):
    f, a, prev = (samples[k] for k in ('frame','action','previous'))
    action = a.detach().requires_grad_(True)
    pred = model.step(unflatten_state(samples['state']),mdp.wm_action(action,f),normalize_xi(samples['xi']))
    estimates = common_reward(mdp,pred,f+1,action,prev)
    jac = {key: torch.autograd.grad(value.sum(),action,retain_graph=True)[0].detach() for key,value in estimates.items()}
    local_dot = torch.nn.functional.cosine_similarity(jac['body'],jac['foot'],dim=-1)
    masks = groups(samples) | {'wm_body_foot_conflict': (local_dot < -.2) &
                               (jac['body'].norm(dim=-1) >= SLOPE_FLOOR) & (jac['foot'].norm(dim=-1) >= SLOPE_FLOOR)}
    report = {}
    for key in TERMS:
        candidate = (jac[key]*direction).sum(-1)
        report[key] = {}
        for name, mask in masks.items():
            ok = valid & mask[None] & torch.isfinite(candidate)
            if not mask.any():
                continue
            score = slope_metrics(candidate,truth[key],ok,f[None],torch.ones_like(ok))
            score['slope_rmse_all'] = rms((candidate-truth[key])[ok]) if ok.any() else None
            report[key][name] = score
    return report


def restore_evidence(env, probe, samples, repeat, restored):
    # 上传到 Isaac 的状态会在第一物理子步同步到求解器。步前使用场景回读核对实际提交值。
    reference = unflatten_state(samples['state'].repeat(repeat,1))
    labels = ('root_p','root_R','root_v','root_w','joint_q','joint_qd')
    exact = {label: float((a-b).abs().max()) for label,a,b in zip(labels,restored,reference)}
    quat = env.robot.data.root_link_pose_w.torch[:,3:7]
    expected = samples['root_quat'].repeat(repeat,1)
    target = env.pd_target(samples['action'].repeat(repeat,1))
    return {'state_readback_max_abs':exact,
            'root_quaternion_max_abs':float(torch.minimum((quat-expected).abs().amax(1),(quat+expected).abs().amax(1)).max()),
            'saved_pd_target_max_abs':float((target-samples['target'].repeat(repeat,1)).abs().max()),
            'warmstart_exact':torch.equal(probe.warmstart, samples['warmstart'].repeat(repeat,1))}


@torch.no_grad()
def physics_fixture(env, probe, samples):
    from .m4_dynamics import fault_masks

    mdp = env.mdp
    n = len(samples['state'])
    f, a, prev = (samples[k] for k in ('frame','action','previous'))
    rng = torch.Generator(device=a.device).manual_seed(SEED)
    direction = torch.randn(3,*a.shape,generator=rng,device=a.device)
    direction[1,:,UPPER] = 0.
    direction[2,:,1:13] = 0.
    direction = torch.nn.functional.normalize(direction,dim=-1)
    count, repeat = len(direction), 2*len(direction)+1
    before = restore(env,probe,samples,repeat)
    evidence = restore_evidence(env,probe,samples,repeat,before)
    command = torch.cat((a+EPSILON*direction,a-EPSILON*direction,a[None])).flatten(0,1)
    after = env.advance_physics(env.pd_target(command))
    bad = torch.stack(fault_masks(before,after,probe,mdp)[:3]).any(0).view(repeat,n)
    ok = ~(bad[:count] | bad[count:2*count] | bad[-1])
    values = common_reward(mdp,after,(f+1).repeat(repeat),command,prev.repeat(repeat,1))
    pair = {key: (v.view(repeat,n)[:count]-v.view(repeat,n)[count:2*count])/(2*EPSILON)
            for key,v in values.items()}
    actual = env.motion.feet(tuple(v[-n:] for v in after))[0]
    evidence.update(invalid_worlds=int(bad.sum()), paired_starts=n, directions=('mixed','legs','upper'))
    good = ~bad[-1]
    evidence['nominal_replay_foot_rmse_m'] = rms((actual-samples['next_foot']).norm(dim=-1)[good]) if good.any() else None
    return direction, pair, ok, evidence


@torch.no_grad()
def short_rollouts(env, probe, samples, actors, horizon):
    from .m4_dynamics import fault_masks

    n = len(samples['state'])
    repeats = len(actors)+2
    assert repeats*n == env.N
    state = restore(env,probe,samples,repeats)
    history = samples['history'].repeat(repeats,1,1)
    previous = samples['previous'].repeat(repeats,1)
    xi = samples['xi'].repeat(repeats,1)
    start = samples['frame'].repeat(repeats)
    anchor = env.motion.anchor(torch.zeros_like(start))
    contacts = env.ground_contacts()
    healthy = torch.ones_like(start,dtype=torch.bool)
    invalid = torch.zeros_like(healthy)
    fallen = torch.zeros_like(healthy)
    series = {k: [] for k in ('valid','root_z','tilt','foot_m','swing_foot_m','support_n','support_loaded',
                              'arm_speed','waist_speed','foot_reward')}
    action_differences = {}
    first = actors['m43_4500']
    second = actors['m45_5000']
    for tick in range(horizon):
        frame = (start+tick).clamp_max(env.motion.num_frames-2)
        goal = env.goals.goal(state,frame,frame,anchor)
        command = torch.empty_like(previous)
        for i, policy in enumerate(actors.values()):
            sl = slice(i*n,(i+1)*n)
            command[sl] = policy(history[sl],goal[sl])
        for k, (legs,upper) in enumerate(((first,second),(second,first)),len(actors)):
            sl = slice(k*n,(k+1)*n)
            command[sl] = legs(history[sl],goal[sl])
            command[sl,UPPER] = upper(history[sl],goal[sl])[:,UPPER]
        if tick == 0:
            a = command[:n]
            action_differences = {key: {'arm_rms': rms((u-a)[:,ARMS]), 'waist_rms': rms((u-a)[:,0]),
                                        'legs_rms': rms((u-a)[:,1:13])}
                                  for key,u in zip(actors,command.split(n))}
        after = env.advance_physics(env.pd_target(command))
        bad = torch.stack(fault_masks(state,after,probe,env.mdp)[:3]).any(0)
        valid = healthy & ~bad & (start+tick < env.motion.num_frames-1)
        f1 = frame+1
        _, force, _ = contacts.read()
        force = torch.as_tensor(force[:, :, 2],device=frame.device)
        planned = env.motion.contact[f1]
        foot = (env.motion.feet(after)[0]-env.motion.foot_p[f1]).norm(dim=-1)
        scores = common_reward(env.mdp,after,f1,command,previous)
        tilt = torch.acos(after[1][:,2,2].clamp(-1.,1.)).rad2deg()
        values = dict(valid=valid,root_z=after[0][:,2],tilt=tilt,foot_m=foot.square().mean(-1),
                      swing_foot_m=(foot.square()*(~planned)).sum(-1)/(~planned).sum(-1).clamp_min(1),
                      support_n=(force*planned).sum(-1),support_loaded=((force >= SUPPORT_FORCE) & planned).any(-1).float(),
                      arm_speed=after[5][:,ARMS].square().mean(-1),waist_speed=after[5][:,0].square(),
                      foot_reward=scores['foot'])
        for k in series:
            series[k].append(values[k])
        term = env.mdp.termination(after,f1,command=anchor,ep_step=f1)
        invalid |= healthy & bad
        fallen |= valid & term
        healthy = valid & ~term & (f1 < env.motion.num_frames-1)
        if not healthy.any():
            break
        # 只隔离已结束的世界，其他世界沿原物理时钟推进。
        if (~healthy).any():
            env.reset((~healthy).nonzero(as_tuple=True)[0])
            after = env.state()
        observation = env.goals.obs(env.mdp,after,command,xi,f1,anchor)
        history = torch.cat((history[:,1:],observation[:,None]),dim=1)
        previous, state = command, after
    labels = (*actors,'m43_legs_m45_upper','m45_legs_m43_upper')
    stack = {k: torch.stack(v).reshape(len(v),repeats,n) for k,v in series.items()}
    result = {}
    for name, mask in groups(samples).items():
        if not mask.any():
            continue
        item = result[name] = {}
        # 误差使用相同可见转移窗口，提前结束与异常另外计数。
        common = stack['valid'].all(1) & mask[None]
        for j, label in enumerate(labels):
            selected = slice(j*n,(j+1)*n)
            row = dict(starts=int(mask.sum()), common_steps=int(common.sum()),
                       early_termination=int((fallen[selected] & mask).sum()),
                       invalid=int((invalid[selected] & mask).sum()))
            if common.any():
                for key, value in stack.items():
                    if key == 'valid':
                        continue
                    reduced = value[:,j][common].mean()
                    row[key] = float(reduced.sqrt() if key in ('foot_m','swing_foot_m','arm_speed','waist_speed') else reduced)
            item[label] = row
    return {'first_action_delta_vs_m43_4500':action_differences,'groups':result,'max_horizon':horizon}


def diagnostic_subset(samples, limit=36):
    if len(samples['frame']) <= limit:
        return samples
    rng = torch.Generator(device=samples['frame'].device).manual_seed(SEED)
    kept = []
    for name, mask in groups(samples).items():
        if name == 'all' or not mask.any():
            continue
        ids = mask.nonzero(as_tuple=True)[0]
        kept.append(ids[torch.randperm(len(ids),generator=rng,device=ids.device)[:5]])
    chosen = torch.cat(kept).unique()
    extra = torch.ones_like(samples['frame'],dtype=torch.bool)
    extra[chosen] = False
    rem = extra.nonzero(as_tuple=True)[0]
    ids = torch.cat((chosen,rem[torch.randperm(len(rem),generator=rng,device=rem.device)[:limit-len(chosen)]]))
    return {name: value[ids] for name,value in samples.items()}


def compensation(archive, matched, model, joints):
    import mujoco

    # 用当前 Newton 编译后的质量和惯量。雅可比给出每个刚体质心速度，关于整机质心求动量。
    data = mujoco.MjData(model)
    linear, angular = np.zeros((3,model.nv)), np.zeros((3,model.nv))
    bodies = model.jnt_bodyid[np.asarray(joints)[UPPER]]
    report = {}
    for name in ('m43_best','m45_final'):
        rows = []
        for seed, _ in matched:
            key = f'{name}_perturb_s{seed}_'

            def read(field):
                return archive[key+field]

            q = read('solver_qpos')[:81]
            v = read('solver_qvel')[:81]
            if q.shape[-1] != model.nq or v.shape[-1] != model.nv:
                return {'available':False,'reason':'存档求解器坐标与本次编译模型维数不一致'}
            rows.append((read('state')[:81],read('raw_action')[:80],read('force')[:81],read('actuator_torque')[:81],q,v))
        phases = {'initial':slice(1,40),'first_swing':slice(40,81)}
        result = report[name] = {}
        for phase, sl in phases.items():
            states = np.concatenate([row[0][sl] for row in rows])
            forces = np.concatenate([row[2][sl] for row in rows])
            torques = np.concatenate([row[3][sl] for row in rows])
            actions = np.concatenate([row[1][max(0,sl.start-1):sl.stop-1] for row in rows])
            angles = torch.as_tensor(states)
            p, R, _, w, _, qd = unflatten_state(angles)
            tilt = torch.acos(R[:,2,2].clamp(-1.,1.)).rad2deg()
            mom = []
            for _,_,_,_,q,v in rows:
                for qi,vi in zip(q[sl],v[sl]):
                    data.qpos[:], data.qvel[:] = qi,vi
                    mujoco.mj_kinematics(model,data)
                    mujoco.mj_comPos(model,data)
                    center = data.subtree_com[0]
                    total = np.zeros(3)
                    for b in bodies:
                        mujoco.mj_jacBodyCom(model,data,linear,angular,int(b))
                        rot = data.ximat[b].reshape(3,3)
                        omega = angular@vi
                        own = rot@(model.body_inertia[b]*(rot.T@omega))
                        total += own + np.cross(data.xipos[b]-center,model.body_mass[b]*(linear@vi))
                    mom.append(total)
            result[phase] = dict(n=len(p), arm_speed_rms=rms(qd[:,ARMS]), waist_speed_rms=rms(qd[:,0]),
                                 arm_action_rms=rms(torch.as_tensor(actions[:,ARMS])),
                                 waist_action_rms=rms(torch.as_tensor(actions[:,0])),
                                 upper_momentum_horizontal_rms=float(np.sqrt(np.mean(np.square(np.asarray(mom)[:,:2]).sum(1)))),
                                 root_z=float(p[:,2].mean()), root_tilt_rms_deg=rms(tilt),
                                 root_angular_xy_rms=rms(w[:,:2]),
                                 ankle_torque_rms=np.sqrt(np.square(torques[:,[5,6,11,12]]).mean(0)).tolist(),
                                 normal_force_mean=forces[...,2].mean(0).tolist(),
                                 left_zero_force_ratio=float((forces[:,0,2]<SUPPORT_FORCE).mean()),
                                 right_zero_force_ratio=float((forces[:,1,2]<SUPPORT_FORCE).mean()))
    return report


def physics(task, models, actors, fixtures, archive, trials):
    from isaaclab.app import launch_simulation

    from .m4_dynamics import SolverReadback
    from .m4_task import M4TrackingSim, make_m4_sim

    cfg, motion, goals, mdp = task
    output, observed = {}, None
    for name, samples in fixtures.items():
        samples = diagnostic_subset(samples)
        n = len(samples['state'])
        row = output[name] = {}
        sim = make_m4_sim(cfg.device,cfg)
        with launch_simulation(sim,cfg):
            env = M4TrackingSim(sim,7*n,motion,mdp,eval_mode=True,goals=goals,gamma=cfg.gamma)
            probe = SolverReadback(env)
            direction, truth, valid, evidence = physics_fixture(env,probe,samples)
            row['evidence'] = evidence
            if observed is None:
                observed = compensation(archive,trials['prior_fall'],probe.model,probe.joints)
        row['finite_difference'] = {key:paired_slopes(wm,mdp,samples,direction,truth,valid) for key,wm in models.items()}
        sim = make_m4_sim(cfg.device,cfg)
        with launch_simulation(sim,cfg):
            env = M4TrackingSim(sim,(len(actors)+2)*n,motion,mdp,eval_mode=True,goals=goals,gamma=cfg.gamma)
            row['actor_rollouts'] = short_rollouts(env,SolverReadback(env),samples,actors,16)
        print(f'完成 Newton 审计 {name} ({n} 个匹配起态)',flush=True)
    return output, observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m4_reference',required=True,type=Path)
    parser.add_argument('--archive',type=Path,default=Path('outputs/m4_5/evaluation/rollouts.npz'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--device',default='cuda:0')
    args = parser.parse_args()
    start = perf_counter()
    torch.set_num_threads(4)
    kin = RobotKinematics(args.device)
    sources = {'m43_4500':'outputs/m4_3/checkpoints/best.pt','m43_5000':'outputs/m4_3/checkpoints/iter_5000.pt',
               'm45_4500':'outputs/m4_5/checkpoints/iter_4500.pt','m45_5000':'outputs/m4_5/checkpoints/iter_5000.pt'}
    bundles, tasks = {}, {}
    for label, path in sources.items():
        ck = torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        assert int(ck['iteration']) == int(label[-4:]) and ck['args']['m4_action_mapping']=='standing_offset'
        assert not ck['args'].get('m4_unclipped_pd_target',False)
        original = task_for(ck,kin,args.m4_reference,args.device)
        oc, om, og, od = original
        pi = make_policy(ck,od,og,args.device)
        critic = Critic(970).to(args.device).eval().requires_grad_(False)
        critic.load_state_dict(ck['critic'])
        bundles[label] = (ck,original,pi,make_wm(ck,args.device),critic)
        cfg = copy(oc)
        cfg.body_prior_weight = UPPER_WEIGHT
        motion, goals, mdp, _ = build_m4_task(cfg,kin,[str(args.m4_reference)],args.device)
        tasks[label] = (cfg,motion,goals,mdp)
    common = tasks['m43_4500']
    trials = manifest(args.archive)
    result = dict(sources=sources,trials=trials,common_weight=UPPER_WEIGHT,group_seed=SEED,
                  original_bootstrap={name: bundle[0]['imagination_contract']['bootstrap_upper_bound']
                                      for name,bundle in bundles.items()},epsilon=EPSILON,slope_floor=SLOPE_FLOOR,
                  prediction={},bptt={})
    with np.load(args.archive,allow_pickle=False) as archive:
        fixtures = {name:fixture(archive,t,args.device,perturbed=name=='prior_fall') for name,t in trials.items()}
        for label, group in trials.items():
            view = data_view(archive,group)
            batch = select_windows(view,0,[s for s,_ in group],10,args.device)
            masks = {'all':torch.ones_like(batch['frame'][:,0],dtype=torch.bool)}
            if label=='prior_fall':
                indices = batch['frame'][:,-1]
                combined = torch.cat([torch.arange(7,len(archive[p+'action']),device=args.device)
                                      >= len(archive[p+'action'])-12 for _,p in group])
                masks['near_fall'] = combined
                masks['first_touchdown'] = (indices-79).abs() <= 7
            result['prediction'][label] = {key:predict(bundle[3],common,batch,masks) for key,bundle in bundles.items()}
            result['bptt'][label] = {key:bptt(bundle,tasks[key],fixtures[label]) for key,bundle in bundles.items()}
            print(f'完成预测与 BPTT {label} ({len(batch["start"])} 个窗口)',flush=True)
        if torch.device(args.device).type=='cuda':
            result['physics'],result['compensation'] = physics(common,{n:b[3] for n,b in bundles.items()},
                                                              {n:b[2] for n,b in bundles.items()},fixtures,archive,trials)
    result['elapsed_s'] = perf_counter()-start
    write_json(args.output,result)


if __name__ == '__main__':
    main()
