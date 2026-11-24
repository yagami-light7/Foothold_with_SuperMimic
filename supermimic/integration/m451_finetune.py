"""按相同起态独立评估四组短训，保留统一口径的结果。"""

import argparse
from copy import copy
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter

if __name__ == '__main__':
    os.environ['SUPERMIMIC_ROBOT'] = 'roban'

import numpy as np

from .m43_wm_audit import write_json
from .m44_evaluation import DEFAULT_SEEDS, PERTURB_SEEDS, evaluator, scene, trial
from .m45_upper import scored_upper_metrics, upper_metrics
from ..motion import CONTROL_DT


def main():
    from isaaclab.app import add_launcher_args

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m4_reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('outputs/m4_5_1'))
    add_launcher_args(parser)
    args = parser.parse_args()
    starts, results, digests = {}, {}, {}
    physics_contract = None
    measures = (
        'completed', 'full_tracking_successes', 'first_swing_successes', 'first_touchdown_successes',
        'matched_touchdowns', 'episode_length_s', 'foot_position_rmse_m', 'root_xy_rmse_m', 'contact_agreement',
        'actuator_saturation', 'clamp_ratio', 'left_ankle_outside', 'left_ankle_stop_active',
        'left_ankle_pushing_stop', 'left_ankle_torque_saturation', 'actual_joint_limit_violation_max_rad',
        'expected_resets', 'valid_resets', 'valid_for_best', 'epa', 'fatal',
    )
    prefix_steps = math.ceil(.2 / CONTROL_DT)
    per_seed = (
        'seed', 'steps', 'termination', 'first_swing', 'first_touchdown', 'matched_touchdowns',
        'consecutive_planned_touchdowns', 'survived_full_reference', 'foot_position_rmse_m',
        'root_xy_rmse_m', 'contact_agreement', 'torque_saturation_ratio', 'clamp_ratio',
    )
    for weight in (0., .02, .05, .10):
        group = f'w{round(weight * 100):03d}'
        root = args.out / group
        if (root / 'train.exitcode').read_text().strip() != '0':
            raise ValueError(f'{group} 短训未正常完成')
        ranked = json.loads((root / 'evaluation/results.json').read_text())['best_iteration']
        results[group] = dict(body_prior_weight=weight, best_iteration=ranked)
        checkpoints = [('final', 'iter_250.pt', 250)]
        if ranked != 250:
            checkpoints.append(('best', 'best.pt', ranked))
        for role, name, expected in checkpoints:
            checkpoint = root / 'checkpoints' / name
            started = perf_counter()
            with scene(args, checkpoint, expected_iteration=expected) as (env, pi, contacts, solver, iteration, physical):
                if env.mdp.cfg.body_prior_weight != weight or env.mdp.unclipped_pd_target:
                    raise ValueError(f'{group} 与权重或动作映射不符')
                if physics_contract is None:
                    physics_contract = physical
                elif physical != physics_contract:
                    raise ValueError(f'{group} 与统一评估物理契约不同')
                scoring = copy(env.mdp)
                scoring.cfg = replace(scoring.cfg, body_prior_weight=.10)
                record = dict(iteration=iteration, checkpoint=str(checkpoint), settings=str(root / 'config/train.json'))
                for split, seeds in (('nominal', DEFAULT_SEEDS), ('perturb', PERTURB_SEEDS)):
                    traces, rows, up, prefix_arm = [], [], [], []
                    for seed in seeds:
                        trace, row = trial(env, pi, contacts, solver, seed, perturb=split == 'perturb')
                        init = starts.setdefault((split, seed), trace['state'][0])
                        np.testing.assert_allclose(init, trace['state'][0], rtol=0, atol=1e-7)
                        traces.append(trace)
                        rows.append(row)
                        if len(trace['action']):
                            upper = scored_upper_metrics(trace, scoring)
                            row['upper_body'] = upper
                            up.append(upper)
                        if len(trace['action']) >= prefix_steps:
                            # 固定比较前 0.2 秒，防止早停导致各候选的上身误差覆盖不同相位。
                            prefix = {k: trace[k][:prefix_steps + 1] for k in ('state', 'frame')}
                            prefix_arm.append(upper_metrics(prefix, env.motion)['arm_rmse_rad'])
                        if row['termination'] == 'fatal_dynamics':
                            break
                    val = evaluator(traces, rows, len(seeds), env.mdp)['aggregate']
                    avg = {key: val.get(key) for key in measures}
                    if up:
                        avg['upper_body'] = {key: np.mean([u[key] for u in up], axis=0).tolist() for key in up[0]}
                    avg['first_point_two_s'] = dict(n=len(prefix_arm), arm_rmse_rad=float(np.mean(prefix_arm))
                                                    if prefix_arm else None)
                    record[split] = dict(
                        aggregate=avg, trials=[{key: row.get(key) for key in (*per_seed, 'upper_body')} for row in rows],
                    )
                    print(group, role, split, avg['full_tracking_successes'], avg['first_touchdown_successes'], flush=True)
            # 记录实际参加比较的权重摘要。同一文件的别名复用一次计算。
            info = checkpoint.stat()
            identity = (info.st_dev, info.st_ino)
            if identity not in digests:
                with checkpoint.open('rb') as handle:
                    digests[identity] = hashlib.file_digest(handle, 'sha256').hexdigest()
            record.update(wall_s=perf_counter() - started, sha256=digests[identity])
            results[group][role] = record
            write_json(args.out / 'finetune_eval.json', results)


if __name__ == '__main__':
    main()
