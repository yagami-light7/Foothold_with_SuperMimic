"""用真实 F2 回放起点标定 Soft Body Prior，并复核双肘裁剪梯度。"""

from argparse import ArgumentParser, Namespace
from dataclasses import replace
import json
import os
from pathlib import Path

if __name__ == '__main__':
    os.environ['SUPERMIMIC_ROBOT'] = 'roban'

import numpy as np
import torch

from .m42v3_ac import cosine, gradient, objective
from .m43_wm_audit import make_policy, make_wm, task_for, write_json
from .m47_smoothness_audit import PHASES, phases
from ..kinematics import RobotKinematics
from ..mdp import XI_DIM, normalize_xi, unflatten_state
from ..motion import CONTROL_DT
from ..networks import Critic
from ..roban import ELBOW_JOINT_NAMES as ELBOWS, JOINT_NAMES
from ..roban_task import UPPER_POSE_JOINTS


WEIGHTS = (.01, .02, .05, .10)
SELECTED_WEIGHT = .02
ELBOW_IDS = [JOINT_NAMES.index(name) for name in ELBOWS]
UPPER = [JOINT_NAMES.index(name) for name in UPPER_POSE_JOINTS]
LOWER = [j for j in range(len(JOINT_NAMES)) if j not in UPPER]
TOLERANCE = {name: .25 if name in ELBOWS else .20 if name.startswith('waist_') else .35
             for name in UPPER_POSE_JOINTS}
ARCHIVE48 = Path('outputs/m4_8/cuda_evaluation.npz')
ARCHIVE46 = Path('outputs/m4_6/cuda_retest.npz')
SOURCES = (
    ('m43', Path('outputs/m4_3/checkpoints/best.pt'), 4500, ARCHIVE48, 'm43_best'),
    ('w002', Path('outputs/m4_6/w002/checkpoints/iter_5000.pt'), 5000, ARCHIVE46, 'w002_final'),
    ('w005', Path('outputs/m4_6/w005/checkpoints/iter_4500.pt'), 4500, ARCHIVE46, 'w005_best'),
    ('b48', Path('outputs/m4_8/b_smoothness_only/checkpoints/best.pt'), 4500, ARCHIVE48, 'b_best'),
    ('c48', Path('outputs/m4_8/c_startup_smoothness/checkpoints/best.pt'), 4500, ARCHIVE48, 'c_best'),
)


def soft_scores(mdp, q, reference, tolerance):
    arms, waist, excess = mdp.posture_prior_terms(q, reference['q'][:, 0], tolerance)
    return (arms + waist) * reference['quality'][:, 0], excess


@torch.no_grad()
def episode(data, prefix, raw_key, task, perturb):
    args, motion, goals, mdp = task
    read = lambda key: torch.from_numpy(data[prefix + key])
    f = read('frame').long()
    state = read('state')
    raw = read(raw_key)
    previous = torch.cat((torch.zeros_like(raw[:1]), raw[:-1]))
    previous2 = torch.cat((torch.zeros_like(raw[:1]), previous[:-1]))
    after = unflatten_state(state[1:])
    f0, f1 = f[:-1], f[1:]
    command = motion.anchor(torch.zeros_like(f0))
    _, terms = mdp.reward(after, f1, raw, previous, True, command=command, ep_step=f1,
                          a_prev2=previous2, pd_target=read('target'))
    xi = state.new_zeros(len(raw), XI_DIM)
    xi[:, 0] = 1.
    before = unflatten_state(state[:-1])
    observation = goals.obs(mdp, before, previous, xi, f0, command)
    histories = observation[(f0[:, None] + torch.arange(1-args.hist, 1)).clamp_min(0)]
    goal = goals.goal(before, f0, f0, command)
    reset_q = unflatten_state(state[:1])[4].expand_as(raw)
    masks = phases(f0, read('force')[:-1], motion, perturb)
    batch = (state[:-1], previous, f0, xi, histories, goal, previous2, reset_q)
    measured = (f1, after[4], terms['body_prior'], terms['foot_objective'], terms['upper_action_smoothness_contribution'])
    return batch, masks, measured


def reference_audit(mdp):
    motion = mdp.motion
    lo, hi = mdp.kin.q_lo, mdp.kin.q_hi
    m = motion.q
    bounds = torch.stack((m.amin(0), m.amax(0)), -1)
    margin = torch.minimum(m-lo, hi-m).amin(0)
    elbows = torch.stack((hi[ELBOW_IDS[0]]-m[:,ELBOW_IDS[0]], m[:,ELBOW_IDS[1]]-lo[ELBOW_IDS[1]]), -1)
    gate = mdp._reference(torch.arange(motion.num_frames), None)['quality'][:, 0]
    return dict(joints={name: dict(range_rad=bounds[j].tolist(), min_margin_rad=float(margin[j]),
                                   tolerance_rad=TOLERANCE[name])
                        for j,name in zip(UPPER, UPPER_POSE_JOINTS)},
                quality_min=float(gate.min()), quality_max=float(gate.max()),
                elbow_tolerance_grid={str(v): dict(
                    near_stop_inside_fraction=((elbows-.02) <= v).float().mean(0).tolist(),
                    min_residual_rad=(elbows-.02-v).clamp_min(0).amin(0).tolist())
                                      for v in (.25, .30, .35, .40)})


def real_metrics(mdp, rows, tolerance):
    f, q, body, foot, smooth = (torch.cat(column) for column in zip(*[row[2] for row in rows]))
    ref = mdp._reference(f, None)
    q = q.requires_grad_()
    soft, excess = soft_scores(mdp, q, ref, tolerance)
    g, = torch.autograd.grad(soft.sum(), q)
    desired = (ref['q'][:, 0] - q.detach()).sign()
    mean_foot = float(foot.abs().mean())
    mean_smooth = float(smooth.abs().mean())
    metrics = dict(n=len(f), original_mean=float(body.mean()), soft_mean=float(soft.detach().mean()),
                   original_weighted_mean=float(body.mean() * .02 * CONTROL_DT),
                   foot_mean=mean_foot, smooth_abs_mean=mean_smooth,
                   violation={JOINT_NAMES[j]: dict(mean=float(excess[:,j].detach().mean()),
                       p95=float(torch.quantile(excess[:,j].detach(), .95)),
                       outside_fraction=float((excess[:,j] > 0).float().mean())) for j in UPPER})
    metrics['weights'] = {}
    for w in WEIGHTS:
        weighted = float(soft.detach().mean()) * CONTROL_DT * w
        metrics['weights'][str(w)] = dict(mean=weighted, foot_ratio=weighted/mean_foot, smooth_ratio=weighted/mean_smooth)

    near = (q[:,ELBOW_IDS[0]] >= mdp.kin.q_hi[ELBOW_IDS[0]]-.02,
            q[:,ELBOW_IDS[1]] <= mdp.kin.q_lo[ELBOW_IDS[1]]+.02)
    lone = torch.zeros_like(q)
    for j, mask in zip(ELBOW_IDS, near):
        mask = mask & (f > 10)
        if not mask.any():
            continue
        # 仅替换一个肘为参考，量化该肘独立贡献的奖励损失。
        lone[:] = q.detach()
        lone[:,j] = ref['q'][:,0,j]
        aligned, _ = soft_scores(mdp, lone, ref, tolerance)
        arm, _, _ = mdp.posture_prior_terms(q.detach()[mask], ref['q'][mask, 0], tolerance)
        slope = (desired[:,j]*g[:,j])[mask]
        # 用同容许带、同关节组系数的二次惩罚估计指数核的衰减量。
        quad_slope = (2*mdp.cfg.prior_upper/(len(mdp.upper_ids)*mdp.cfg.prior_joint_std**2)
                      * ref['quality'][mask,0] * excess[mask,j].detach())
        metrics.setdefault('near_stop', {})[JOINT_NAMES[j]] = dict(
            n=int(mask.sum()), exact_full_fraction=float((excess[mask,j] == 0.).float().mean()),
            min_residual=float(excess[mask,j].detach().min()),
            elbow_only_reward_loss_mean=float((aligned-soft.detach())[mask].mean()),
            arm_kernel_p05=float(torch.quantile(arm/.7, .05)),
            q_gradient_corrective_p05=float(torch.quantile(slope, .05)),
            q_gradient_corrective_mean=float(slope.mean()),
            quadratic_corrective_mean=float(quad_slope.mean()))
    return metrics


def channel_grad(objective_value, nodes):
    gradients = torch.autograd.grad(objective_value, nodes, retain_graph=True, allow_unused=True)
    return torch.stack([torch.zeros_like(node) if g is None else g.detach() for node,g in zip(nodes, gradients)])


def bptt_metrics(task, ck, batch, models, tolerance):
    args, motion, goals, mdp = task
    pi, wm, critic = models
    packet = objective(batch[:6], pi, critic, wm, motion, mdp, goals, args,
                       ck['imagination_contract']['bootstrap_upper_bound'], 4901, action_context=batch[6:])
    f = packet['reward_frames']
    ref = mdp._reference(f, None)
    soft, _ = soft_scores(mdp, packet['joint_position'], ref, tolerance)
    weighted = packet['fixed_weighted']
    soft_loss = -(soft.view(weighted.shape)*weighted*CONTROL_DT).sum(0).mean()
    objectives = dict(foot=packet['C'], smooth=packet['direct']['upper_action_smoothness'],
                      body=packet['direct']['body_prior'], soft_unit=soft_loss, bootstrap=packet['bootstrap'])
    params = tuple(pi.parameters())
    gg = {name: gradient(value, params) for name,value in objectives.items()}
    dg = {name: channel_grad(value, packet['actions']) for name,value in objectives.items()}
    effective = channel_grad(soft_loss, packet['effective'])
    if not all(g.isfinite().all() for g in (*gg.values(), *dg.values(), effective)):
        raise FloatingPointError('姿态梯度非有限')

    def rms_channels(g):
        return dict(upper=float(g[...,UPPER].square().mean().sqrt()),
                    lower=float(g[...,LOWER].square().mean().sqrt()),
                    elbows={name: float(g[...,j].square().mean().sqrt()) for name,j in zip(ELBOWS,ELBOW_IDS)})

    norms = {name: float(g.norm()) for name,g in gg.items()}
    clamp = packet['clip']
    saturated = {}
    desired = (ref['q'][:len(batch[0]), 0] - unflatten_state(batch[0])[4]).sign()
    for name,j in zip(ELBOWS,ELBOW_IDS):
        mask = clamp[0,:,j]
        raw, eff = dg['soft_unit'][0,:,j][mask], effective[0,:,j][mask]
        if len(raw):
            corrective = -raw * desired[mask,j]
            saturated[name] = dict(n=len(raw), raw_abs_mean=float(raw.abs().mean()), effective_abs_mean=float(eff.abs().mean()),
                                   raw_corrective_fraction=float((corrective > 1e-10).float().mean()))
    return dict(n=len(batch[0]), norm=norms,
                cosine={f'{a}_{b}': cosine(gg[a],gg[b]) for a,b in
                        (('foot','body'), ('foot','soft_unit'), ('smooth','body'), ('smooth','soft_unit'),
                         ('bootstrap','body'), ('bootstrap','soft_unit'))},
                raw_action={name: rms_channels(g) for name,g in dg.items()},
                soft_unit_effective_action=rms_channels(effective), first_clipped_elbows=saturated,
                soft_foot_ratio={str(w): w*norms['soft_unit']/max(norms['foot'], 1e-12) for w in WEIGHTS})


def offline(reference, output):
    result = dict(tolerance_rad=TOLERANCE, weights=WEIGHTS, selected_weight=SELECTED_WEIGHT, runs={})
    kin = RobotKinematics('cpu')
    for name,path,iteration,archive,label in SOURCES:
        ck = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        assert int(ck['iteration']) == iteration
        task = task_for(ck, kin, reference, 'cpu')
        args, motion, goals, mdp = task
        mdp.cfg = replace(mdp.cfg, body_prior_weight=.02, upper_action_smoothness_weight=.02)
        tolerance = mdp.pose_tolerance(TOLERANCE)
        if 'reference' not in result:
            result['reference'] = reference_audit(mdp)
        pi, wm = make_policy(ck, mdp, goals, 'cpu'), make_wm(ck, 'cpu')
        critic = Critic(mdp.critic_dim + goals.spec.dim).eval().requires_grad_(False)
        critic.load_state_dict(ck['critic'])
        is48 = archive == ARCHIVE48
        with np.load(archive, allow_pickle=False) as data:
            rows = [episode(data, f'{label}_full_s{seed}_' if is48 else f'{split}_{label}_s{seed}_',
                            'raw_action' if is48 else 'action', task, split=='perturb')
                    for split,seeds in (('nominal', (1001,1002,1003)), ('perturb', (1101,1102,1103,1104,1105,1106)))
                    for seed in seeds]
        run = result['runs'][name] = dict(checkpoint=str(path), real=real_metrics(mdp, rows, tolerance), bptt={})
        for phase in ('all', *PHASES):
            entries = []
            for batch, mask, _ in rows:
                f = batch[2]
                ok = (f >= 2) & (f+args.horizon <= motion.num_frames-1)
                if phase != 'all':
                    ok &= mask[phase]
                entries.append(tuple(x[ok] for x in batch))
            flat = tuple(torch.cat(x) for x in zip(*entries))
            picks = torch.linspace(0, len(flat[0])-1, min(20,len(flat[0]))).round().long()
            run['bptt'][phase] = bptt_metrics(task, ck, tuple(x[picks] for x in flat), (pi,wm,critic), tolerance)
        print(name, '离线标定', len(rows), '条轨迹，实测 Soft Prior', round(run['real']['soft_mean'], 4),
              '整体分项梯度', run['bptt']['all']['norm'], flush=True)
        write_json(output, result)
    for w in WEIGHTS:
        values = [phase['soft_foot_ratio'][str(w)] for run in result['runs'].values() for phase in run['bptt'].values()]
        result.setdefault('weight_ratios', {})[str(w)] = np.quantile(values, (0,.5,.9,1)).tolist()
    write_json(output, result)


def matched_samples(args, path):
    from .m42v3_rollout import collect_trial
    from .m44_evaluation import perturbed_start, scene

    labels, rows = [], []
    with scene(args, path) as (env, actor, contacts, probe, _, _):
        mdp = env.mdp
        lo, hi, q0, alpha, ref = (x.cpu().numpy() for x in
                                   (mdp.kin.q_lo, mdp.kin.q_hi, mdp.default_q, mdp.alpha, env.motion.q))
        for seed in (1001, 1101):
            hook = (lambda sim: perturbed_start(sim, seed)) if seed == 1101 else None
            # 两条轨迹补取暖启动，覆盖启动结束和右肘在片段末尾离开裁剪区。
            trace = collect_trial(env, actor, contacts, seed, solver=probe, reset_hook=hook)
            frame = trace['frame'][:-1]
            raw = trace['action']
            q = trace['state'][:-1,13:13+len(JOINT_NAMES)]
            for name,j,sign in zip(ELBOWS, ELBOW_IDS, (-1., 1.)):
                boundary = hi[j] if sign < 0. else lo[j]
                target_raw = q0[j] + alpha[j]*raw[:,j]
                desired = np.sign(ref[frame+1,j] - q[:,j])
                valid = (frame >= mdp.startup_steps) & (np.abs(ref[frame+1,j]-q[:,j]) > TOLERANCE[name]+.06)
                # 以朝内方向记号计算：过界时 (target - boundary) 与 sign 异号。
                saturated = ((desired == sign) & (np.abs(q[:,j]-boundary) <= .02)
                             & ((target_raw-boundary)*sign < -.30*alpha[j]))
                free = (target_raw > lo[j]+.20*alpha[j]) & (target_raw < hi[j]-.20*alpha[j])
                choices = [('saturated', saturated & (frame < 115)), ('free', free & (frame < 115))]
                if name == ELBOWS[1]:
                    choices.append(('free_after_stop', free & (desired == sign) & (frame >= 115)))
                for kind, mask in choices:
                    ids = np.flatnonzero(valid & mask)
                    if not len(ids):
                        continue
                    i = ids[len(ids)//2]
                    labels.append(dict(seed=seed, frame=int(frame[i]), joint=name, kind=kind, direction=float(desired[i])))
                    rows.append(dict(state=trace['state'][i], next_state=trace['state'][i+1], action=raw[i],
                                     target=trace['target'][i], frame=frame[i], xi=trace['xi'][i],
                                     warmstart=trace['solver_warmstart'][i], root_quat=trace['root_quat_xyzw'][i]))
    if not all(any(row['joint'] == j and row['kind'] == kind for row in labels)
               for j in ELBOWS for kind in ('saturated', 'free')):
        raise RuntimeError('小样本未同时覆盖双肘的裁剪和未裁剪状态')
    samples = {key: torch.as_tensor(np.stack([row[key] for row in rows]), device=args.device) for key in rows[0]}
    return labels, samples


def newton(reference, output):
    from .m4_dynamics import fault_masks
    from .m43_wm_audit import restore
    from .m44_evaluation import scene

    result = json.loads(output.read_text())
    args = Namespace(m4_reference=reference, device='cuda:0', visualizer=None)
    path = SOURCES[-1][1]
    labels, samples = matched_samples(args, path)
    n = len(labels)
    device = args.device
    row_id = torch.arange(n, device=device)
    index = torch.tensor([JOINT_NAMES.index(row['joint']) for row in labels], device=device)
    direction = torch.tensor([row['direction'] for row in labels], device=device)
    saturates = torch.tensor([row['kind'] == 'saturated' for row in labels], device=device)
    epsilon, inward = .15, .35
    with scene(args, path, num_envs=4*n) as (env, _, _, probe, _, _):
        mdp = env.mdp
        tolerance = mdp.pose_tolerance(TOLERANCE)
        f = samples['frame']
        ref = mdp._reference(f+1, None)
        raw = samples['action'].detach().requires_grad_()
        state = unflatten_state(samples['state'])
        q = state[4].detach().requires_grad_()
        q_value, _ = soft_scores(mdp, q, ref, tolerance)
        q_grad, = torch.autograd.grad(q_value.sum(), q)
        ck = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        wm = make_wm(ck, device)
        effective = mdp.wm_action(raw, f)
        predicted = wm.step(state, effective, normalize_xi(samples['xi']))
        value, _ = soft_scores(mdp, predicted[4], ref, tolerance)
        raw_grad, effective_grad = torch.autograd.grad(value.sum(), (raw, effective))

        with torch.no_grad():
            a = raw.detach()
            actions = a.expand(4,-1,-1).clone()
            actions[1,row_id,index] += direction*epsilon
            actions[2,row_id,index] -= direction*epsilon
            stop_q = torch.where(direction < 0., mdp.kin.q_hi[index], mdp.kin.q_lo[index])
            stop_a = (stop_q-mdp.default_q[index])/mdp.alpha[index]
            actions[3,row_id,index] = torch.where(saturates, stop_a + direction*inward, actions[1,row_id,index])
            before = restore(env, probe, samples, 4)
            target = env.pd_target(actions.flatten(0,1))
            torch.testing.assert_close(target[:n], samples['target'], rtol=0, atol=1e-6)
            after = env.advance_physics(target)
            faults = torch.stack(fault_masks(before, after, probe, mdp)[:3]).any(0)
            if faults.any():
                raise FloatingPointError('双肘匹配状态分支发生物理故障')
            reward, _ = soft_scores(mdp, after[4], mdp._reference((f+1).repeat(4), None), tolerance)
            reward = reward.view(4,n)
            next_q = after[4].view(4,n,-1)[:,row_id,index]
            replay_q = unflatten_state(samples['next_state'])[4]
            delta_q = (replay_q - after[4][:n]).abs().amax(dim=-1)
            target_j = target.view(4,n,-1)[:,row_id,index]

        for i,label in enumerate(labels):
            j = index[i]
            label.update(raw_action=float(a[i,j]), bound_raw=float(stop_a[i]) if bool(saturates[i]) else None,
                         q_before_rad=float(state[4][i,j]), q_reference_rad=float(ref['q'][i,0,j]),
                         q_baseline_next_rad=float(next_q[0,i]),
                         q_gradient_toward=float(q_grad[i,j]*direction[i]),
                         wm_raw_gradient_toward=float(raw_grad[i,j]*direction[i]),
                         wm_effective_gradient_toward=float(effective_grad[i,j]*direction[i]),
                         newton_raw_fd_toward=float((reward[1,i]-reward[2,i])/(2*epsilon)),
                         local_pd_target_difference_rad=float(target_j[1,i]-target_j[2,i]),
                         inward_pd_target_delta_rad=float(target_j[3,i]-target_j[0,i]),
                         inward_observed_q_delta_rad=float(next_q[3,i]-next_q[0,i]),
                         inward_soft_reward_gain=float(reward[3,i]-reward[0,i]),
                         restored_baseline_next_q_max_delta_rad=float(delta_q[i]))
    result['matched_newton'] = dict(raw_epsilon=epsilon, crossed_raw_margin=inward,
                                    unweighted=True, fault_count=0, rows=labels)
    write_json(output, result)
    print('Newton 双肘配对分支', len(labels), '起点，', 4*len(labels), '条单步分支，物理故障为零', flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--m4_reference', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path('outputs/m4_9/prior_calibration/results.json'))
    parser.add_argument('--newton', action='store_true', help='对少量 C 组真实状态复核硬裁剪的配对物理分支')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.newton:
        newton(args.m4_reference, args.output)
    else:
        offline(args.m4_reference, args.output)


if __name__ == '__main__':
    main()
