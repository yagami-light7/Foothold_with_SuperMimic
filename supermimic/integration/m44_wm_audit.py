"""统一足部奖励下的 M4.4 世界模型、左踝和 BPTT 审计。"""

import argparse
from copy import copy
import json
import os
from pathlib import Path

if __name__ == '__main__':
    os.environ['SUPERMIMIC_ROBOT'] = 'roban'

import numpy as np
import torch

from ..kinematics import RobotKinematics
from ..mdp import normalize_xi, unflatten_state
from ..roban import JOINT_NAMES
from .m42v3_wm import evaluate_windows, select_windows
from .m43_wm_audit import (SLOPE_FLOOR, actor_diagnostics, collect_samples, comparison_groups, make_wm,
                           restore, scores, slope_metrics, task_for, write_json)
from .m4_task import build_m4_task


ANKLE = JOINT_NAMES.index('leg_l5_joint')
EPS = (.075, .15, .30)
CENTER = .15


def view_trials(data, prefix, iteration):
    # 原诊断工具使用轮次前缀。用内存视图区分轮次相同的两种动作模式。
    names = data.files
    view = {'seeds': data['seeds']}
    for seed in view['seeds']:
        source = f'{prefix}_s{seed}_'
        target = f'i{iteration}_s{seed}_'
        for key in names:
            if key.startswith(source):
                view[target+key[len(source):]] = data[key]
        if target+'state' not in view:
            raise ValueError(f'缺少真实回放 {prefix} seed={seed}')
    return view


def evaluation_output(audit):
    keys = ('n', 'foot_m', 'root_m', 'tilt_deg', 'state_mse', 'reward_bias', 'return_bias',
            'predicted_swing_lift_m', 'real_swing_lift_m')
    return {str(h): {key: row[key] for key in keys if key in row}
            for h, groups in audit['horizons'].items() for row in (groups['all'],)}


def probe_directions(action, mdp, ascent, interior):
    q_raw = mdp.standing_raw_target(action)
    margin = mdp.alpha*max(EPS)
    allowed = ((q_raw > mdp.kin.q_lo+margin) & (q_raw < mdp.kin.q_hi-margin)
               if interior else torch.ones_like(action, dtype=torch.bool))
    generator = torch.Generator(device=action.device).manual_seed(4203)
    random = torch.randn((2, *action.shape), generator=generator, device=action.device)
    ankle = torch.zeros_like(action)
    ankle[:, ANKLE] = 1.
    directions = torch.cat((random, ascent[None], ankle[None]))
    return torch.nn.functional.normalize(directions * allowed, dim=-1)


def outcomes(task, state, frame, action, previous):
    mdp = task[3]
    values = scores(mdp, state, frame, action, previous)
    p = task[1].feet(state)[0]
    return {**values, 'ankle_angle': state[4][:, ANKLE], 'ankle_speed': state[5][:, ANKLE],
            'left_foot_height': p[:, 0, 2]}


@torch.enable_grad()
def model_jacobian(model, task, samples):
    frame, previous = samples['frame'], samples['previous']
    a = samples['action'].detach().requires_grad_(True)
    nxt = model.step(unflatten_state(samples['state']), task[3].wm_action(a, frame), normalize_xi(samples['xi']))
    return {name: torch.autograd.grad(value.sum(), a, retain_graph=True)[0].detach()
            for name, value in outcomes(task, nxt, frame, a, previous).items()}


def directional_derivatives(jacobian, direction):
    return {name: (gradient * direction).sum(-1) for name, gradient in jacobian.items()}


@torch.no_grad()
def physics_derivatives(env, probe, task, samples, direction):
    from .m4_dynamics import fault_masks

    n = len(samples['state'])
    d = len(direction)
    repeat = 2*d+1
    frame, prev = samples['frame'].repeat(repeat), samples['previous'].repeat(repeat, 1)
    action = samples['action']
    result = {}
    for eps in EPS:
        before = restore(env, probe, samples, repeat)
        a = torch.cat((action + eps*direction, action - eps*direction, action[None])).flatten(0, 1)
        nxt = env.advance_physics(env.pd_target(a))
        bad = torch.stack(fault_masks(before, nxt, probe, task[3])[:3]).any(0).view(repeat, n)
        valid = ~(bad[:d] | bad[d:2*d] | bad[-1]) & (direction.square().sum(-1) > 0)
        torque = probe.torque[:, probe.dofs[ANKLE]].view(repeat, n)
        values = outcomes(task, nxt, frame, a, prev) | {'actuator_torque': torque}
        differences = {key: (value.view(repeat, n)[:d] - value.view(repeat, n)[d:2*d])/(2*eps)
                       for key, value in values.items()}
        result[eps] = (differences, valid)
        if eps == CENTER:
            effort = float(task[3].effort[ANKLE])
            # 末子步约束反力矩区分真正受载的止挡与仅接近机械限位的姿态。
            stop_torque = probe.joint_limit_forces()[:, ANKLE].view(repeat, n)[-1]
            nominal_foot = task[1].feet(tuple(x[-n:] for x in nxt))[0]
            info = dict(bad=bad[-1], torque=torque[-1].clone(), limit=stop_torque > 1e-6,
                        stop_torque=stop_torque, ankle_q_after=nxt[4][-n:, ANKLE].clone(),
                        saturated_both=(torque[:d].abs() >= .99*effort) & (torque[d:2*d].abs() >= .99*effort)
                        & (torque[:d]*torque[d:2*d] > 0.),
                        foot_error=(nominal_foot-samples['next_foot']).square().sum(-1).mean(-1))
    return result, info


def strata(samples, mdp, info):
    q = unflatten_state(samples['state'])[4][:, ANKLE]
    torque = info['torque']
    raw = mdp.standing_raw_target(samples['action'])[:, ANKLE]
    lo, hi, alpha = mdp.kin.q_lo[ANKLE], mdp.kin.q_hi[ANKLE], mdp.alpha[ANKLE]
    clipped = mdp.target_clip_mask(samples['action'])[:, ANKLE]
    saturated = torque.abs() >= .99*mdp.effort[ANKLE]
    upper_stop = info['ankle_q_after'] >= (lo + hi)*.5
    return comparison_groups(samples, mdp.motion) | dict(
        torque_saturated=saturated, torque_spare=~saturated, joint_stop_active=info['limit'],
        ankle_both_perturb_saturated=info['saturated_both'][-1],
        ankle_near_stop=torch.minimum(q-lo, hi-q) <= .015, ankle_target_clipped=clipped,
        ankle_target_out=(raw < lo) | (raw > hi), ankle_interior=(raw >= lo) & (raw <= hi),
        ankle_beyond_perturb=(raw+alpha*CENTER < lo) | (raw-alpha*CENTER > hi),
        ankle_crosses_clamp=((raw-lo).abs() <= alpha*CENTER) | ((raw-hi).abs() <= alpha*CENTER),
        pushes_stop=info['limit'] & ((upper_stop & (torque > 0)) | (~upper_stop & (torque < 0))))


def error_metrics(pred, truth, valid, frame, mask):
    finite = pred.isfinite() & truth.isfinite()
    nonfinite = int((valid & mask & ~finite).sum())
    valid = valid & finite
    stats = slope_metrics(pred, truth, valid, frame, mask)
    if nonfinite:
        stats['nonfinite'] = nonfinite
    rows = valid & mask
    if rows.any():
        # 同时计入零斜率，区分符号一致率和实际可控性。
        stats.update(slope_rmse_all=float((pred[rows] - truth[rows]).square().mean().sqrt()),
                     actual_abs_slope_mean=float(truth[rows].abs().mean()),
                     wm_abs_slope_mean=float(pred[rows].abs().mean()))
    if stats['wm_positive']:
        stats['harmful_fraction'] = stats['wm_positive_real_negative'] / stats['wm_positive']
    return stats


def models_report(gradients, differences, samples, groups):
    true, valid = differences[CENTER]
    frame = samples['frame']
    report = {}
    for label, values in gradients.items():
        group_eval = {}
        for term, pred in values.items():
            selected = {name: error_metrics(pred, true[term], valid, frame[None], mask)
                        for name, mask in groups.items() if mask.any()}
            ankle = {name: error_metrics(pred[-1], true[term][-1], valid[-1], frame, mask)
                     for name, mask in groups.items() if (mask & valid[-1]).any()}
            ascent = error_metrics(pred[-2], true[term][-2], valid[-2], frame, groups['all'])
            sensitivity = {str(eps): error_metrics(pred, series[term], use, frame[None], groups['all'])
                           for eps, (series, use) in differences.items()}
            group_eval[term] = dict(directions=selected, ankle_direction=ankle, wm_ascent_direction=ascent,
                                    epsilons=sensitivity)
        report[label] = group_eval
    return report


def physical_report(differences, info, groups):
    true, valid = differences[CENTER]

    def response(mask):
        selected = valid[-1] & mask
        if not selected.any():
            return {'n': 0}
        return dict(n=int(selected.sum()),
                    **{key: float(value[-1, selected].abs().mean()) for key, value in true.items()},
                    zero_torque=int((true['actuator_torque'][-1, selected].abs() <= 1e-3).sum()),
                    zero_angle=int((true['ankle_angle'][-1, selected].abs() <= 1e-6).sum()),
                    zero_foot=int((true['foot'][-1, selected].abs() <= SLOPE_FLOOR).sum()))

    return dict(starts=len(groups['all']), invalid_nominal=int(info['bad'].sum()),
                nominal_trajectory_delta_m=float(info['foot_error'][~info['bad']].mean().sqrt())
                if (~info['bad']).any() else None,
                ankle_torque_abs_max_nm=float(info['torque'][~info['bad']].abs().max())
                if (~info['bad']).any() else None,
                ankle_limit_force_max_nm=float(info['stop_torque'][~info['bad']].max())
                if (~info['bad']).any() else None,
                nominal_strata={name: int((mask & ~info['bad']).sum()) for name, mask in groups.items()},
                ankle_response={name: response(mask) for name, mask in groups.items()})


def run_local(fixtures, device):
    from isaaclab.app import launch_simulation

    from .m4_dynamics import SolverReadback
    from .m4_task import M4TrackingSim, make_m4_sim

    report = {}
    for name, task, samples, direction, gradients in fixtures:
        cfg, motion, goals, mdp = task
        n_envs = (2*len(direction)+1)*len(samples['state'])
        sim = make_m4_sim(device, cfg)
        with launch_simulation(sim, cfg):
            env = M4TrackingSim(sim, n_envs, motion, mdp, eval_mode=True, goals=goals, gamma=cfg.gamma)
            series, info = physics_derivatives(env, SolverReadback(env), task, samples, direction)
        groups = strata(samples, mdp, info)
        report[name] = dict(physical=physical_report(series, info, groups),
                            models=models_report(gradients, series, samples, groups),
                            saved_warmstart='warmstart' in samples, saved_quaternion='root_quat' in samples)
        print('有限差分', name, json.dumps({k: report[name]['physical'][k] for k in
                                       ('starts', 'invalid_nominal', 'nominal_trajectory_delta_m')}, ensure_ascii=False),
              flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m4_reference', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=Path('outputs/m4_3'))
    parser.add_argument('--candidate', type=Path, default=Path('outputs/m4_4_unclipped'))
    parser.add_argument('--rollouts', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--local-newton', action='store_true')
    parser.add_argument('--baseline-only', action='store_true', help='正式训练前使用原 M4.3 轨迹诊断两种映射的左踝响应')
    parser.add_argument('--starts-per-seed', type=int, default=14)
    args = parser.parse_args()
    device = torch.device(args.device)
    if (args.local_newton and device.type != 'cuda') or args.starts_per_seed < 1:
        parser.error('有限差分需要 CUDA，起态采样数必须大于零')
    torch.set_num_threads(min(6, torch.get_num_threads()))
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
    kin = RobotKinematics(device)
    archive = args.rollouts or ((args.baseline if args.baseline_only else args.candidate)/'evaluation/rollouts.npz')
    paths = (('best', 'best.pt'), ('final', 'iter_5000.pt'))
    checkpoints = {f'm43_{key}': args.baseline/'checkpoints'/name for key, name in paths}
    if not args.baseline_only:
        checkpoints.update({f'm44_{key}': args.candidate/'checkpoints'/name for key, name in paths})
    study = dict(epsilons=EPS, center_epsilon=CENTER, slope_floor=SLOPE_FLOOR,
                 data_scope='独立无噪声回放，公共目标使用 M4.3 最终权重的已执行位置目标',
                 directions=('固定种子随机一', '固定种子随机二', '足部奖励的 WM 上升', '纯左踝正向'),
                 shared_ascent_model='m43_best', checkpoints={})
    with np.load(archive, allow_pickle=False) as data:
        study['seeds'] = data['seeds'].tolist()
        models, tasks, own, fixtures = {}, {}, {}, []
        for label, path in checkpoints.items():
            checkpoint = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            it = int(checkpoint['iteration'])
            task = task_for(checkpoint, kin, args.m4_reference, device)
            cfg, motion, goals, mdp = task
            prefix = f'i{it}' if args.baseline_only else label
            view = view_trials(data, prefix, it)
            model = make_wm(checkpoint, device)
            batch = select_windows(view, it, view['seeds'], cfg.hist, device)
            report = study['checkpoints'][label] = dict(iteration=it)
            with torch.no_grad():
                report['own_prediction'] = evaluation_output(evaluate_windows(model, mdp, motion, goals, batch, cfg.gamma))
            report['bptt'] = actor_diagnostics(checkpoint, task, model, view, it, device, False)[0]
            models[label], tasks[label], own[label] = model, task, batch
            if args.local_newton:
                sample = collect_samples(view, f'i{it}', motion, mdp, args.starts_per_seed, device, joint_index=ANKLE)
                torch.testing.assert_close(mdp.standing_pd_target(sample['action']), sample['target'])
                jacobian = model_jacobian(model, task, sample)
                # 随机、WM 上升与纯左踝方向均允许穿过本映射的裁剪折点。
                direction = probe_directions(sample['action'], mdp, jacobian['foot'], interior=False)
                fixtures.append((label, task, sample, direction, {label: directional_derivatives(jacobian, direction)}))

        baseline = tasks['m43_final']
        cfg, motion, goals, mdp = baseline
        reference = own['m43_final']
        # 公共数据用实际执行过的 PD 目标反编码，比较相同物理命令对应的预测。
        common = reference | {'action': mdp.wm_action(reference['action'], reference['frame']),
                              'prior': mdp.wm_action(reference['prior'], reference['frame'])}
        for label, (tc, tm, tg, td) in tasks.items():
            with torch.no_grad():
                pred = evaluate_windows(models[label], td, tm, tg, common, tc.gamma, rescore=True)
            study['checkpoints'][label]['shared_prediction'] = evaluation_output(pred)
        if args.local_newton:
            it = study['checkpoints']['m43_final']['iteration']
            prefix = f'i{it}' if args.baseline_only else 'm43_final'
            public = collect_samples(data, prefix, motion, mdp, args.starts_per_seed, device, joint_index=ANKLE)
            common_samples = public | {'action': (public['target']-mdp.default_q)/mdp.alpha,
                                       'previous': mdp.wm_action(public['previous'], public['frame'])}
            canonical = {label: model_jacobian(models[label], task, common_samples)
                         for label, task in tasks.items()}
            direction = probe_directions(common_samples['action'], mdp, canonical['m43_best']['foot'], interior=True)
            gradients = {label: directional_derivatives(jac, direction) for label, jac in canonical.items()}
            fixtures.append(('shared_applied_target', baseline, common_samples, direction, gradients))
            raw_jacs = {label: model_jacobian(models[label], task, public) for label, task in tasks.items()}
            raw_direction = probe_directions(public['action'], mdp, raw_jacs['m43_best']['foot'], interior=False)
            alternate = copy(cfg)
            alternate.m4_unclipped_pd_target = True
            nm, ng, nd, _ = build_m4_task(alternate, kin, [str(args.m4_reference)], device)
            linear_task = (alternate, nm, ng, nd)
            originals = {label: directional_derivatives(jac, raw_direction) for label, jac in raw_jacs.items()
                         if not tasks[label][3].unclipped_pd_target}
            unclipped = {label: directional_derivatives(jac, raw_direction) for label, jac in raw_jacs.items()
                         if tasks[label][3].unclipped_pd_target}
            fixtures += [('shared_raw_clipped', baseline, public, raw_direction, originals),
                         ('shared_raw_unclipped', linear_task, public, raw_direction, unclipped)]
            study['finite_difference'] = run_local(fixtures, device)
    write_json(args.output, study)
    for label, row in study['checkpoints'].items():
        print(label, json.dumps({name: row[name]['8']['foot_m'] for name in ('own_prediction', 'shared_prediction')},
                               ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
