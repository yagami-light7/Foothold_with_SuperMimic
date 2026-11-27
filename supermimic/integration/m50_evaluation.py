"""按参考片段评估已冻结的 Roban 策略。"""

from argparse import ArgumentParser, Namespace
from contextlib import contextmanager
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

if __name__ == '__main__':
    os.environ['SUPERMIMIC_ROBOT'] = 'roban'
    os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import torch

from ..kinematics import RobotKinematics
from ..mdp import unflatten_state
from ..motion import CONTROL_DT
from ..networks import Policy
from ..roban import assert_checkpoint
from ..transfer import assert_policy_checkpoint
from .m410_evaluation import mean_metrics
from .m410_quality import local_upper_points, whole_body_metrics
from .m43_evaluation import DEFAULT_SEEDS, INVALID_TERMINATIONS, m48_trial_metrics
from .m44_evaluation import PERTURB_SEEDS, trial
from .m411_evaluation import control_limits
from .m48_evaluation import FIELDS, load_archive, movement_metrics, save_archive
from .m4_dynamics import SolverReadback
from .m4_task import M4TrackingSim, build_reference_tracking, make_m4_sim
from .m50_reference import MANIFEST, REFERENCE_ROOT, SOURCE, load_clip, read_manifest
from .m50_task import MultiReferenceGoalSpec, MultiReferenceMDP, OUTPUT


HOLDOUT_SEEDS = (1201, 1202, 1203, 1204, 1205, 1206)
STAGE_BOUNDARIES = (1000, 3000, 6000)
TRACE_FIELDS = (*FIELDS, 'foot_R', 'foot_velocity', 'clamp_mask', 'joint_limit_force', 'reward', 'hand_fore_aft')


@contextmanager
def scene_for(args, source, item, split):
    from isaaclab.app import launch_simulation

    checkpoint = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    assert_checkpoint(checkpoint)
    config = Namespace(**checkpoint['args'])
    device = 'cuda:0'
    kin = RobotKinematics(device)
    motion = load_clip(item, kin, device, args.reference_root, split)
    spec = MultiReferenceGoalSpec('roban_foot', config.goal_frames, config.goal_dt)
    mdp_type = MultiReferenceMDP
    if getattr(config, 'm51_group', None):
        from .m51_task import StanceFootMDP

        mdp_type = StanceFootMDP
    _, goals, mdp, _ = build_reference_tracking(config, kin, motion, spec, mdp_type)
    saved_goal = dict(checkpoint['goal_contract'], reference=spec.contract()['reference'])
    saved = checkpoint['task_contract']
    context = {'mode', 'duration_s', 'reset', 'foot_frame'}
    task = lambda value: {key: x for key, x in value.items() if key not in context}
    if (saved_goal != spec.contract() or task(saved) != task(mdp.contract())
            or mdp.physics_contract(config) != checkpoint['physical_contract']):
        raise ValueError('冻结策略与独立评估任务、奖励或物理契约不符')
    if split == 'train' and 'clips' in checkpoint['data_contract']:
        expected = next(c for c in checkpoint['data_contract']['clips'] if c['parameters']['clip_id'] == item['clip_id'])
        # 训练允许 TF32，独立复评仍沿用 M4.11 的精度设置。载入阶段已分别检查 FK 容差。
        identity = lambda data: {key: value for key, value in data.items() if key != 'fk'}
        if identity(motion.validation) != identity(expected):
            raise ValueError(f'{item["clip_id"]} 评估使用的参考不同于训练时')
    assert_policy_checkpoint(checkpoint, config.hist, config.obs_adapter, not config.freeze_decoder,
                             obs_dim=mdp.obs_dim, observation_mode=mdp.observation_mode)
    pi = Policy(mdp.obs_dim, spec.dim, hist=config.hist, obs_adapter=config.obs_adapter).to(device).eval()
    pi.load_state_dict(checkpoint['pi'])
    sim_cfg = make_m4_sim(device, config)
    launcher = Namespace(visualizer=None, device=device)
    with launch_simulation(sim_cfg, launcher):
        env = M4TrackingSim(sim_cfg, 1, motion, mdp, gamma=config.gamma, eval_mode=True, goals=goals)
        yield env, pi, env.ground_contacts(), SolverReadback(env)


def trial_metrics(trace, row, mdp):
    if not row['steps']:
        return
    m48_trial_metrics(trace, row, mdp)
    body = whole_body_metrics(trace, mdp)
    row['body'] = {key: value for key, value in body.items() if key != 'upper_joint_error'}
    limits = torch.stack((mdp.kin.q_lo, mdp.kin.q_hi)).cpu().numpy()
    row['control'] = control_limits(trace, limits, row['steps'])
    state = unflatten_state(torch.as_tensor(trace['state'], device=mdp.motion.q.device))
    trace['hand_fore_aft'] = local_upper_points(state, mdp.kin)[:, 4:, 0].cpu().numpy()
    smooth = movement_metrics([(trace, row['steps'])], limits, round(.20 / CONTROL_DT))
    row['arm_hf_power'] = smooth.get('arm_hf_power')
    row['arm_hf_fraction'] = smooth.get('arm_hf_fraction')
    row['upper_acceleration_p99_rad_s2'] = smooth.get('acceleration_p99')
    row['upper_acceleration_max_rad_s2'] = smooth.get('acceleration_max')
    row['opposite_hand_correlation'] = smooth.get('opposite_hand_correlation')
    row['same_side_planned_swing_correlation'] = smooth.get('same_side_planned_swing_correlation')
    motion = mdp.motion
    targets = motion.at_time(motion.touchdown_time)['foot_p'].cpu().numpy()
    sides = motion.touchdown_side.cpu().tolist()
    errors = [np.linalg.norm(trace['foot'][round(event['actual_touchdown_s'] / CONTROL_DT), side, :2]
                              - targets[i, side, :2])
              for i, (event, side) in enumerate(zip(row['footsteps'], sides)) if event['actual_touchdown_s'] is not None]
    row['touchdown_xy_rmse_m'] = float(np.sqrt(np.square(errors).mean())) if errors else None


def trial_ok(row):
    return row['survived_full_reference'] and row['consecutive_planned_touchdowns'] == len(row['footsteps'])


def collect_cases(args, source, cases, *, splits, seeds, arrays=None, resets=None, prefix=''):
    resets = {} if resets is None else resets
    for split in splits:
        for entry in read_manifest(args.library)[split]:
            name = entry['clip_id']
            rows = cases.setdefault(name, {})
            pending = [seed for seed in seeds if str(seed) not in rows]
            if not pending:
                continue
            with scene_for(args, source, entry, split) as (env, pi, contacts, solver):
                for seed in pending:
                    trace, row = trial(env, pi, contacts, solver, seed, perturb=seed not in DEFAULT_SEEDS)
                    initial = trace['state'][0]
                    np.testing.assert_allclose(initial, resets.setdefault((name, seed), initial), atol=1.e-7, rtol=0.)
                    if row['termination'] in INVALID_TERMINATIONS:
                        detail = dict(checkpoint=str(source), clip=name, seed=seed, steps=row['steps'],
                                      termination=row['termination'], detail=row['fault_detail'])
                        save_archive(args.out / 'evaluation' / 'fault.npz', detail,
                                     {key: trace[key] for key in TRACE_FIELDS if key in trace})
                        raise FloatingPointError(f'{source.name} {name} seed={seed} {row["termination"]} {row["fault_detail"]}')
                    trial_metrics(trace, row, env.mdp)
                    if arrays is not None:
                        row['prefix'] = f'{prefix}_{name}_s{seed}_'
                        arrays.update({row['prefix']+key: trace[key] for key in TRACE_FIELDS})
                    rows[str(seed)] = row
                    print(f'{source.name} {name} seed={seed} 匹配 {row["consecutive_planned_touchdowns"]}/{len(row["footsteps"])} '
                          f'用时 {row["episode_length_s"]:.2f}s {row["termination"]}', flush=True)
            torch.cuda.empty_cache()


def summarize(cases, seeds):
    result = {}
    for name, rows in cases.items():
        selected = [rows[str(seed)] for seed in seeds if str(seed) in rows]
        if not selected:
            continue
        success = [row for row in selected if trial_ok(row)]
        result[name] = dict(attempts=len(selected), complete=len(success),
                            successful_seeds=[row['seed'] for row in success],
                            survives=sum(row['survived_full_reference'] for row in selected),
                            first_swing=sum(row['first_swing'] for row in selected),
                            first_touchdown=sum(row['first_touchdown'] for row in selected))
        for label, entries in (('observed', selected), ('completed', success)):
            result[name][label] = {key: mean_metrics([row.get(key) for row in entries])
                                   for key in ('episode_length_s', 'foot_position_rmse_m', 'root_xy_rmse_m',
                                               'contact_agreement', 'touchdown_xy_rmse_m', 'normalized_return', 'body',
                                               'upper_smoothness', 'arm_hf_power', 'arm_hf_fraction',
                                               'upper_acceleration_p99_rad_s2', 'upper_acceleration_max_rad_s2',
                                               'opposite_hand_correlation', 'same_side_planned_swing_correlation',
                                               'control', 'per_joint_clamp', 'torque_saturation_ratio',
                                               'joint_limit_excess_rad', 'clamp_ratio')}
    return result


def rank(nominal, perturb=None):
    def rate(metric, data):
        return [row[metric] / row['attempts'] for row in data.values()]

    outcomes = rate('complete', nominal)
    recovered = rate('complete', perturb) if perturb else [0.]
    survived = rate('survives', perturb) if perturb else [0.]
    quality = [row['completed'] if row['complete'] == row['attempts'] else row['observed']
               for row in nominal.values()]
    foot = np.mean([row['foot_position_rmse_m'] for row in quality])
    root = np.mean([row['root_xy_rmse_m'] for row in quality])
    posture = []
    for row in quality:
        body = row['body']
        motion = row['upper_smoothness']['after_startup']
        vals = (body['arm_rmse_rad'] / .30, np.mean(body['hand_local_rmse_m']) / .06,
                (motion['acceleration_rms_rad_s2'] or np.inf) / 35.,
                (motion['jerk_rms_rad_s3'] or np.inf) / 2200.)
        posture.append(np.mean(vals))
    f2 = nominal['F2']
    return (float(np.mean(outcomes)), min(outcomes), float(np.mean(recovered)),
            float(np.mean(survived)), -float(foot), -float(root), -float(np.mean(posture)),
            -f2['observed']['foot_position_rmse_m'])


def assert_stage_ready(milestones, key):
    row = milestones[key]
    iteration = row['iteration']
    if iteration not in (1000, 3000):
        return
    nominal = row['nominal']
    previous = milestones[str(iteration-500)]['nominal']
    degraded = [name for name in nominal if previous[name]['complete'] - nominal[name]['complete'] >= 2]
    if nominal['F2']['complete'] <= 1 and 'F2' not in degraded:
        degraded.append('F2')
    if degraded:
        raise RuntimeError(f'{", ".join(degraded)} 正常完整率下降至少两个种子，已保留课程边界检查点')


def periodic(args):
    out = args.out / 'evaluation'
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'results.json'
    report = json.loads(path.read_text()) if path.exists() else dict(milestones={})
    checkpoint = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    iteration = checkpoint['iteration']
    del checkpoint
    key = 'adaptation' if iteration == 0 else str(iteration)
    if key in report['milestones']:
        assert_stage_ready(report['milestones'], key)
        return
    frozen = args.out / 'checkpoints' / ('after_adaptation.pt' if iteration == 0 else f'iter_{iteration:04d}.pt')
    if not frozen.is_file() or not frozen.samefile(args.checkpoint):
        raise ValueError(f'评估需要已持久化的里程碑 {frozen}')
    extended = iteration in (0, *STAGE_BOUNDARIES)
    cases = {}
    collect_cases(args, frozen, cases, splits=('train',), seeds=DEFAULT_SEEDS + (PERTURB_SEEDS if extended else ()))
    nominal = summarize(cases, DEFAULT_SEEDS)
    perturb = summarize(cases, PERTURB_SEEDS) if extended else None
    report['milestones'][key] = dict(iteration=iteration, nominal=nominal, perturb=perturb, rows=cases,
                                     checkpoint=str(frozen))
    eligible = [row for row in report['milestones'].values() if row['iteration'] > 0]
    if eligible:
        report['best_multi_reference'] = max(eligible, key=lambda row: rank(row['nominal']))['iteration']
    robust = [row for row in eligible if row['perturb']]
    if robust:
        report['best_robust'] = max(robust, key=lambda row: rank(row['nominal'], row['perturb']))['iteration']
    with TemporaryDirectory(dir=out) as directory:
        temp = Path(directory)
        (temp/'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        for alias in ('best_multi_reference', 'best_robust'):
            if alias not in report:
                continue
            name = alias + '.pt'
            chosen = Path(report['milestones'][str(report[alias])]['checkpoint'])
            (temp/name).hardlink_to(chosen)
            (temp/name).replace(args.out / 'checkpoints' / name)
        (temp/'results.json').replace(path)
    print({name: (row['complete'], row['attempts']) for name, row in nominal.items()}, flush=True)
    assert_stage_ready(report['milestones'], key)


def zero_shot(args):
    path = args.out / 'zero_shot.npz'
    info, arrays = load_archive(path)
    cases = info.setdefault('trials', {})
    collect_cases(args, SOURCE, cases, splits=('train',), seeds=DEFAULT_SEEDS, arrays=arrays, prefix='b1250')
    info['nominal'] = summarize(cases, DEFAULT_SEEDS)
    save_archive(path, info, arrays)


def final_candidates(out):
    checkpoints = out / 'checkpoints'
    periodic = json.loads((out / 'evaluation/results.json').read_text())
    candidates = dict(B1250=SOURCE, adaptation=checkpoints/'after_adaptation.pt')
    titles = {'B1250': ['B1250'], 'adaptation': ['适应后']}
    low = 0
    for stage, high in enumerate(STAGE_BOUNDARIES, 1):
        choices = [row for row in periodic['milestones'].values() if low < row['iteration'] <= high]
        winner = max(choices, key=lambda row: rank(row['nominal']))
        name = str(winner['iteration'])
        candidates[name] = checkpoints / f'iter_{int(name):04d}.pt'
        titles.setdefault(name, []).append(f'阶段 {stage} 最佳')
        low = high
    for label, key in (('多参考最佳', 'best_multi_reference'), ('恢复候选', 'best_robust'), ('末轮', None)):
        it = str(periodic[key]) if key else str(STAGE_BOUNDARIES[-1])
        candidates[it] = checkpoints / f'iter_{int(it):04d}.pt'
        titles.setdefault(it, []).append(label)
    return candidates, titles


def paired_errors(info, arrays, manifest, root):
    kin = RobotKinematics('cpu')
    models = info['seen']
    result = {}
    for entry in manifest['train']:
        name = entry['clip_id']
        motion = load_clip(entry, kin, 'cpu', root, 'train')
        foot_plan = motion.foot_p.numpy()
        checkpoint = torch.load(SOURCE, map_location='cpu', mmap=True, weights_only=False)
        config = Namespace(**checkpoint['args'])
        del checkpoint
        spec = MultiReferenceGoalSpec('roban_foot', config.goal_frames, config.goal_dt)
        _, _, mdp, _ = build_reference_tracking(config, kin, motion, spec, MultiReferenceMDP)
        limits = torch.stack((kin.q_lo, kin.q_hi)).numpy()
        seeds = DEFAULT_SEEDS + PERTURB_SEEDS + HOLDOUT_SEEDS
        end = {seed: min(record['trials'][name][str(seed)]['steps'] for record in models.values()) for seed in seeds}
        result[name] = dict(common_end_s={str(seed): steps * CONTROL_DT for seed, steps in end.items()}, models={})
        for model, value in models.items():
            stats = {}
            for seed, steps in end.items():
                if not steps:
                    continue
                prefix = value['trials'][name][str(seed)]['prefix']
                trace = {key: arrays[prefix+key] for key in TRACE_FIELDS}
                f = trace['frame'][1:steps+1].astype(np.int64)
                ef = np.square(trace['foot'][1:steps+1] - foot_plan[f]).sum(-1)
                body = whole_body_metrics(trace, mdp, end=steps)
                smooth = movement_metrics([(trace, steps)], limits, round(.20 / CONTROL_DT))
                stats[str(seed)] = dict(foot_rmse_m=float(np.sqrt(ef.mean())),
                                        root_xy_rmse_m=body['root_xy_rmse_m'],
                                        arm_rmse_rad=body['arm_rmse_rad'], hand_rmse_m=body['hand_local_rmse_m'],
                                        acceleration_rms_rad_s2=smooth.get('acceleration_rms'),
                                        jerk_rms_rad_s3=smooth.get('jerk_rms'),
                                        arm_hf_power=smooth.get('arm_hf_power'))
            result[name]['models'][model] = stats
    return result


def final(args):
    path = args.out / 'cuda_evaluation.npz'
    info, arrays = load_archive(path)
    manifest = read_manifest(args.library)
    cases, titles = final_candidates(args.out)
    seen = info.setdefault('seen', {})
    if info.setdefault('candidates', titles) != titles:
        raise ValueError('独立复评开始后候选集合发生变化')
    expected = DEFAULT_SEEDS + PERTURB_SEEDS + HOLDOUT_SEEDS
    resets = {}
    for branch in (seen, info.get('heldout', {})):
        for entry in branch.values():
            for clip, trials in entry['trials'].items():
                for key, row in trials.items():
                    initial = arrays[row['prefix'] + 'state'][0]
                    np.testing.assert_allclose(initial, resets.setdefault((clip, int(key)), initial), atol=1e-7, rtol=0.)
    try:
        for name, ckpt in cases.items():
            entry = seen.setdefault(name, dict(checkpoint=str(ckpt), trials={}))
            collect_cases(args, ckpt, entry['trials'], splits=('train',), seeds=expected, arrays=arrays,
                          resets=resets, prefix=name)
            entry['summary'] = {label: summarize(entry['trials'], seeds) for label, seeds in (
                ('nominal', DEFAULT_SEEDS), ('perturb', PERTURB_SEEDS), ('holdout_reset', HOLDOUT_SEEDS))}
            save_archive(path, info, arrays)
        new_policy = [name for name in seen if name not in ('B1250', 'adaptation')]
        chosen = max(new_policy, key=lambda name: rank(seen[name]['summary']['nominal'], seen[name]['summary']['perturb']))
        if info.setdefault('selected', chosen) != chosen:
            raise ValueError('未见步态评估开始后选定策略发生变化')
        unseen = info.setdefault('heldout', {}).setdefault(chosen, dict(checkpoint=str(cases[chosen]), trials={}))
        collect_cases(args, cases[chosen], unseen['trials'], splits=('heldout',), seeds=DEFAULT_SEEDS+PERTURB_SEEDS,
                      arrays=arrays, resets=resets, prefix=f'heldout_{chosen}')
        unseen['summary'] = dict(nominal=summarize(unseen['trials'], DEFAULT_SEEDS),
                                 perturb=summarize(unseen['trials'], PERTURB_SEEDS))
        info['paired'] = paired_errors(info, arrays, manifest, args.reference_root)
    except BaseException:
        # 将已经完成的真实试次和诊断一起落盘，恢复时只补做剩余种子。
        save_archive(path, info, arrays)
        raise
    save_archive(path, info, arrays)
    print(f'M5.0 独立复评完成，联合候选第 {chosen} 轮', flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('zero-shot', 'periodic', 'final'))
    parser.add_argument('--library', type=Path, default=MANIFEST)
    parser.add_argument('--reference-root', type=Path, default=REFERENCE_ROOT)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    parser.add_argument('--checkpoint', type=Path)
    cli = parser.parse_args()
    torch.set_num_threads(2)
    {'zero-shot': zero_shot, 'periodic': periodic, 'final': final}[cli.mode](cli)


if __name__ == '__main__':
    main()
