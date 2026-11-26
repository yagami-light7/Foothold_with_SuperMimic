"""用已归档 Newton 轨迹核对 M4.10 奖励及 BPTT 信号。"""

from argparse import ArgumentParser
from dataclasses import replace
import json
import os
from pathlib import Path

if __name__ == '__main__':
    os.environ['SUPERMIMIC_ROBOT'] = 'roban'

import numpy as np
import torch

from ..kinematics import RobotKinematics
from ..mdp import normalize_xi, unflatten_state
from ..motion import CONTROL_DT
from ..networks import Critic
from ..roban import ELBOW_JOINT_NAMES, JOINT_NAMES, model_contract
from .m410_quality import whole_body_metrics
from .m410_reward import GROUP_REWARDS, HUBER_TERMS, ROOT_TERMS, SHOULDER_JOINTS, WHOLE_BODY_TERMS
from .m42v3_ac import cosine, objective
from .m43_wm_audit import make_policy, make_wm, task_for, write_json
from .m49_calibration import episode


ARCHIVE = Path('outputs/m4_9/cuda_evaluation.npz')
CASES = ('m43_best', 'm48c_best', 'a_9000', 'b_9000', 'c_8500', 'c_9500')
ANKLE_CASES = ('m48c_best', 'a_9000', 'c_9500')
ANKLES = ('leg_l5_joint', 'leg_r5_joint')
IDS = [JOINT_NAMES.index(joint) for joint in ANKLES]
CHANNELS = SHOULDER_JOINTS + ELBOW_JOINT_NAMES + ANKLES
CHANNEL_IDS = [JOINT_NAMES.index(joint) for joint in CHANNELS]


def rms(tensor):
    return tensor.square().mean().sqrt().item() if tensor.numel() else None


def reward_columns(terms):
    values = {k: terms[f'{k}_contribution'] for k in WHOLE_BODY_TERMS}
    root = sum(values[k] for k in ROOT_TERMS)
    huber = sum(values[k] for k in HUBER_TERMS)
    baseline = terms['baseline_reward']
    return dict(**values, root=root, huber=huber, baseline=baseline, foot=terms['foot_objective'],
                root_only=baseline + root, huber_only=baseline + huber, combined=baseline + root + huber)


@torch.no_grad()
def replay_case(task, archive, row, perturb, *, verify_target=True):
    _, _, _, mdp = task
    prefix = row['prefix']
    data = {key: archive[prefix + key] for key in
            ('state', 'frame', 'foot', 'contact', 'force', 'target', 'raw_action', 'actuator_torque', 'effective_action')}
    batch, masks, _ = episode({prefix + key: value for key, value in data.items()}, prefix, 'raw_action', task, perturb)
    raw, target = (torch.from_numpy(data[key]) for key in ('raw_action', 'target'))
    f = batch[2]
    torque = torch.from_numpy(data['actuator_torque'][1:])
    if verify_target:
        torch.testing.assert_close(mdp.executed_target(raw, batch[7], f), target, atol=1.e-6, rtol=0)
    torch.testing.assert_close(mdp.effective_action(target), torch.from_numpy(data['effective_action']), atol=1.e-6, rtol=0)
    after = unflatten_state(torch.from_numpy(data['state'][1:]))
    context = dict(command=mdp.motion.anchor(torch.zeros_like(f)), ep_step=f+1, a_prev2=batch[6], pd_target=target)
    score, terms = mdp.reward(after, f + 1, raw, batch[1], True, **context)
    rewards = reward_columns(terms)
    torch.testing.assert_close(score, rewards['combined'], atol=1.e-7, rtol=0)
    assert all(torch.isfinite(value).all() for value in rewards.values())
    masks['startup'] = f < 10
    masks['single_support'] = torch.from_numpy(data['contact'][:-1]).sum(-1) == 1
    return dict(state=data['state'], raw=raw, target=target, batch=batch, after=after, torque=torque,
                masks=masks, rewards=rewards, data=data, row=row, contact=torch.from_numpy(data['contact'][:-1]),
                planned_contact=torch.from_numpy(archive[prefix+'planned_contact'][:-1]))


def reduce_real(entries, mdp):
    components = {k: torch.cat([row['rewards'][k] for row in entries]) for k in entries[0]['rewards']}
    measures = [whole_body_metrics(row['data'], mdp) for row in entries]
    observed = {name: np.mean([m[name] for m in measures], axis=0).tolist() for name in
                ('root_xy_rmse_m', 'root_z_rmse_m', 'root_heading_rmse_deg', 'arm_rmse_rad', 'shoulder_rmse_rad',
                 'elbow_rmse_rad', 'waist_rmse_rad', 'hand_local_rmse_m', 'hand_lateral_abs_p95_m',
                 'upper_arm_direction_rmse_deg', 'foot_orientation_rmse_deg')}
    return dict(trials=len(entries), steps=len(components['baseline']),
                full_four=sum(row['row']['survived_full_reference'] and row['row']['consecutive_planned_touchdowns'] == 4
                              for row in entries),
                mean_per_step={key: value.mean().item() for key, value in components.items()},
                min_per_step={key: value.min().item() for key, value in components.items()},
                negative_step_ratio={key: (value < 0.).float().mean().item() for key, value in components.items()},
                physical=observed)


def select(entries, horizon, phase, count):
    batches = []
    for row in entries:
        items = row['batch']
        enough = items[2] + horizon <= row['data']['frame'][-1]
        visible = enough if phase == 'all' else enough & row['masks'][phase]
        batches.append(tuple(item[visible] for item in items))
    merged = tuple(torch.cat(items) for items in zip(*batches))
    ids = torch.linspace(0, len(merged[0])-1, min(count, len(merged[0]))).round().long()
    return tuple(values[ids] for values in merged)


def bptt(task, batch, models):
    args, motion, goals, mdp = task
    pi, wm, critic = models
    packet = objective(batch[:6], pi, critic, wm, motion, mdp, goals, args, None, 41001, action_context=batch[6:])
    direct = packet['direct']
    losses = dict(foot=packet['C'], **{key: direct[key] for key in WHOLE_BODY_TERMS},
                  root=sum(direct[key] for key in ROOT_TERMS), huber=sum(direct[key] for key in HUBER_TERMS),
                  stability=direct['upright'] + direct['survival'],
                  smoothness=direct['upper_action_smoothness'], bootstrap=packet['bootstrap'], total=packet['A'])
    params = tuple(pi.parameters())
    raw, effective = tuple(packet['actions']), tuple(packet['effective'])
    n_params, horizon = len(params), len(raw)
    vectors, channels = {}, {}
    for label, loss in losses.items():
        nodes = (*params, *raw, *effective)
        grads = torch.autograd.grad(loss, nodes, retain_graph=True, allow_unused=True)
        grads = [torch.zeros_like(node) if grad is None else grad.detach() for node, grad in zip(nodes, grads)]
        if not all(torch.isfinite(g).all() for g in grads):
            raise FloatingPointError(f'BPTT 梯度非有限：{label}')
        vectors[label] = torch.cat([g.flatten() for g in grads[:n_params]])
        raw_grad = torch.stack(grads[n_params:n_params+horizon])
        effective_grad = torch.stack(grads[n_params+horizon:])
        channels[label] = dict(raw=raw_grad[..., CHANNEL_IDS].square().mean((0, 1)).sqrt().tolist(),
                               effective=effective_grad[..., CHANNEL_IDS].square().mean((0, 1)).sqrt().tolist())
    pairs = [(name, target) for name in (*WHOLE_BODY_TERMS, 'root', 'huber') for target in ('foot', 'bootstrap')]
    pairs.append(('root', 'huber'))
    return dict(starts=len(batch[0]), horizon=horizon, norm={key: g.norm().item() for key, g in vectors.items()},
                cosine={f'{name}_vs_{target}': cosine(vectors[name], vectors[target]) for name, target in pairs},
                channels_rms=channels, left_ankle_clip_ratio=packet['clip'][..., IDS[0]].float().mean().item())


def ankle_readback(mdp, entries):
    raw = torch.cat([row['raw'] for row in entries])
    target = torch.cat([row['target'] for row in entries])
    q, qd = (torch.cat([row['after'][i] for row in entries]) for i in (4, 5))
    frame = torch.cat([row['batch'][2]+1 for row in entries])
    q_ref = mdp.motion.q[frame]
    qd_ref = mdp.motion.at_time(frame.to(torch.float64)*CONTROL_DT)['qd']
    torque = torch.cat([row['torque'] for row in entries])
    contact = torch.cat([row['contact'] for row in entries])
    plan = torch.cat([row['planned_contact'] for row in entries])
    before = mdp.standing_raw_target(raw)
    lo, hi = mdp.kin.q_lo, mdp.kin.q_hi
    local = {row['name']: row for row in model_contract()[1]}
    result = {}
    for side, (name, j) in enumerate(zip(ANKLES, IDS)):
        low, high = before[:, j] < lo[j], before[:, j] > hi[j]
        sat = (torque[:, j].abs() >= .99 * mdp.effort[j])
        physical = (q[:, j] < lo[j]) | (q[:, j] > hi[j])
        quantile = lambda values: torch.quantile(values, raw.new_tensor([.05,.50,.95])).tolist()
        phases = dict(measured_support=contact[:, side], measured_off=~contact[:, side],
                      planned_support=plan[:, side], planned_swing=~plan[:, side])
        result[name] = dict(index=j, axis=local[name]['axis'], q0=mdp.default_q[j].item(), alpha=mdp.alpha[j].item(),
                            kp=mdp.kp[j].item(), kd=mdp.kd[j].item(), effort=mdp.effort[j].item(),
                            physical_limits=[lo[j].item(), hi[j].item()],
                            raw_limits=((torch.stack((lo[j], hi[j])) - mdp.default_q[j])/mdp.alpha[j]).tolist(),
                            q_reference_p05_p50_p95=quantile(q_ref[:, j]), q_actual_p05_p50_p95=quantile(q[:, j]),
                            raw_p05_p50_p95=quantile(raw[:, j]), preclamp_p05_p50_p95=quantile(before[:, j]),
                            target_p05_p50_p95=quantile(target[:, j]), torque_p05_p50_p95=quantile(torque[:, j]),
                            q_error_rms=rms(q[:, j]-q_ref[:, j]), qd_error_rms=rms(qd[:, j]-qd_ref[:, j]),
                            torque_abs_max=torque[:, j].abs().max().item(), torque_saturation=sat.float().mean().item(),
                            actual_limit_violation=physical.float().mean().item(),
                            actual_limit_peak_rad=torch.maximum(lo[j]-q[:, j], q[:, j]-hi[j]).clamp_min(0.).max().item(),
                            clamp_low=low.float().mean().item(), clamp_high=high.float().mean().item(),
                            low_excess_rms=rms((lo[j]-before[:, j])[low]),
                            conditional={key: dict(steps=int(mask.sum()), low_clamp=low[mask].float().mean().item())
                                         for key, mask in phases.items() if mask.any()})
    return result


def ankle_gradients(task, entries, wm):
    _, motion, _, mdp = task
    fields = ('raw', 'target')
    raw, applied = (torch.cat([row[k] for row in entries]) for k in fields)
    batch = tuple(torch.cat(v) for v in zip(*(row['batch'] for row in entries)))
    successor = torch.cat([torch.from_numpy(row['state'][1:]) for row in entries])
    selected = torch.linspace(0, len(raw)-1, min(len(raw), 144)).round().long()
    flat, previous, frame, xi, _, _, prev2, reset = (value[selected] for value in batch)
    raw = raw[selected].requires_grad_()
    target = mdp.executed_target(raw, reset, frame)
    effective = mdp.effective_action(target)
    f1 = frame+1
    context = dict(command=motion.anchor(torch.zeros_like(frame)), ep_step=f1, a_prev2=prev2)
    actual = unflatten_state(successor[selected])
    q = actual[4].detach().requires_grad_()
    _, measured = mdp.reward((*actual[:4], q, actual[5]), f1, raw.detach(), previous, True,
                             pd_target=applied[selected], **context)
    gq, = torch.autograd.grad(measured['foot_objective'].sum(), q)
    prediction = wm.step(unflatten_state(flat), effective, normalize_xi(xi))
    _, terms = mdp.reward(prediction, f1, raw, previous, True, pd_target=target, **context)
    cols = reward_columns(terms)
    values = dict(foot=cols['foot'], root=cols['root'], huber=cols['huber'],
                  stability=terms['upright_contribution']+terms['survival_contribution'])
    mask = mdp.target_clip_mask(raw).detach()
    result = {}
    for name, value in values.items():
        ga, ge = torch.autograd.grad(value.sum(), (raw, effective), retain_graph=True)
        if not all(torch.isfinite(x).all() for x in (ga, ge, gq)):
            raise FloatingPointError('左踝局部动作梯度非有限')
        result[name] = {joint: {label: dict(n=int(pick.sum()), raw_rms=rms(ga[:, j][pick]),
                                            effective_rms=rms(ge[:, j][pick]), foot_q_rms=rms(gq[:, j][pick]))
                                for label, pick in (('clipped', mask[:, j]), ('free', ~mask[:, j]))}
                        for joint, j in zip(ANKLES, IDS)}
    return result


def run(reference, output):
    torch.set_num_threads(4)
    kin = RobotKinematics('cpu')
    result = dict(archive=str(ARCHIVE), channels=CHANNELS, runs={})
    with np.load(ARCHIVE, allow_pickle=False) as archive:
        info = json.loads(archive['metadata'].item())
        for name in CASES:
            case = info['cases'][name]
            ck = torch.load(case['checkpoint'], map_location='cpu', mmap=True, weights_only=False)
            assert ck['iteration'] == case['iteration']
            task = task_for(ck, kin, reference, 'cpu')
            args, motion, goals, mdp = task
            assert ck['data_contract'] == motion.validation and args.horizon == 16
            mdp.cfg = replace(mdp.cfg, body_prior_weight=0., soft_body_prior_weight=0., soft_body_prior_tolerance=None,
                              upper_action_smoothness_weight=.02, **GROUP_REWARDS['c'])
            pi, wm = make_policy(ck, mdp, goals, 'cpu'), make_wm(ck, 'cpu')
            critic = Critic(mdp.critic_dim+goals.spec.dim).eval().requires_grad_(False)
            critic.load_state_dict(ck['critic'])
            groups = {split: [replay_case(task, archive, case['full'][str(seed)], split=='perturb')
                              for seed in seeds] for split, seeds in info['seeds'].items()}
            entries = groups['nominal'] + groups['perturb']
            summary = dict(checkpoint=case['checkpoint'],
                           real={split: reduce_real(rows, mdp) for split, rows in groups.items()}, bptt={})
            for phase in ('all', 'startup', 'single_support', 'touchdown'):
                batch = select(entries, args.horizon, phase, 24 if phase == 'all' else 16)
                if len(batch[0]):
                    summary['bptt'][phase] = bptt(task, batch, (pi, wm, critic))
            if name in ANKLE_CASES:
                summary['ankle'] = {split: ankle_readback(mdp, rows) for split, rows in groups.items()}
                summary['ankle_gradients'] = ankle_gradients(task, entries, wm)
            result['runs'][name] = summary
            print(name, '真实回放', len(entries), '条，BPTT 起点',
                  {phase: entry['starts'] for phase, entry in summary['bptt'].items()}, flush=True)
    write_json(output, result)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--m4_reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('outputs/m4_10/preflight_metrics.json'))
    cli = parser.parse_args()
    run(cli.m4_reference, cli.out)
