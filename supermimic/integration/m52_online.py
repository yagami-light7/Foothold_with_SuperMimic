"""同一 M3.13 在线参考下配对比较冻结策略，无训练或模型更新。"""

from argparse import ArgumentParser, Namespace
from contextlib import chdir
import csv
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

if __name__ == '__main__':
    os.environ['SUPERMIMIC_ROBOT'] = 'roban'

import numpy as np
import torch

from ..kinematics import RobotKinematics
from ..motion import CONTROL_DT
from ..networks import Policy
from ..roban import assert_checkpoint
from .m4_task import M4TrackingSim, make_m4_sim
from .m50_reference import MANIFEST, REFERENCE_ROOT, read_manifest
from .m50_task import build_m50_task
from .m52_orientation import posture_metrics
from .m52_task import OUTPUT, SOURCE, TRANSFER


def load_policy(path, root, device):
    ck = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    assert_checkpoint(ck)
    params = Namespace(**ck['args'])
    params.m50_reference_root, params.m50_library = str(root), str(MANIFEST)
    kin = RobotKinematics(device)
    motion, goals, task, _ = build_m50_task(params, kin, device)
    if (ck['task_contract'] != task.contract() or ck['goal_contract'] != goals.spec.contract()
            or ck['physical_contract'] != task.physics_contract(params)):
        raise ValueError('在线复评需要原任务、目标和物理契约')
    pi = Policy(task.obs_dim, goals.spec.dim, hist=params.hist, obs_adapter=params.obs_adapter).to(device).eval()
    pi.load_state_dict(ck['pi'])
    return Namespace(pi=pi, hist=params.hist), ck, params, motion, task


def spatial(data, end, kin):
    # 位置 FK 无需速度。连续段另比较前后等长窗口的实际根部倾斜。
    if end < 1:
        return {}
    fields = ('p', 'R', 'q')
    actual, target = (tuple(torch.as_tensor(data[prefix+k][1:end+1]) for k in fields) for prefix in ('', 'ref_'))
    zeros = actual[0].new_zeros(actual[0].shape)
    state = (actual[0], actual[1], zeros, zeros, actual[2], torch.zeros_like(actual[2]))
    ref = (target[0], target[1], zeros, zeros, target[2], torch.zeros_like(target[2]))
    width = min(round(5/CONTROL_DT), max(1, end//2))
    tilt = posture_metrics(state, ref, kin, segments=dict(early=slice(0, width), late=slice(-width, None)))
    row = tilt['all']
    values = {key: row[key] for key in ('tilt_rms_deg', 'hand_world_m_rms', 'hand_root_local_m_rms', 'rpy_rmse_deg')}
    values.update(early_tilt_deg=tilt['early']['tilt_rms_deg'], late_tilt_deg=tilt['late']['tilt_rms_deg'],
                  roll_range90_deg=row['angles']['range90_deg'][0], pitch_range90_deg=row['angles']['range90_deg'][1])
    return values


def scenarios():
    entries = read_manifest()['train']
    cmd = [(x['step_length'], x['step_frequency'], x['yaw_rate']) for x in entries]
    length = [x['num_frames']-1 for x in entries]
    return ((False, list(zip(('F1', 'F2', 'F3', 'F2_30s', 'F3_60s'),
                             (*cmd, cmd[1], cmd[2]), (*length, 1500, 3000), (1251, 1252, 1253, 1254, 1255), strict=True))),
            (True, [('F3_60s_dynamic', cmd[2], 3000, 1256)]))


@torch.no_grad()
def run_pair(paths, root, batches):
    from isaaclab.app import launch_simulation
    from roban_motion.m3_11 import PROJECT, setup
    from roban_motion.m312_rollout import _warmup
    from roban_motion.m313_closed import _evaluate, _simulate
    from roban_motion.m313_reward import make_task

    device = 'cuda:0'
    with chdir(PROJECT):
        _, _, solver = setup()
    rows = []
    cpu_kin = RobotKinematics('cpu')
    for dynamic, cases in batches:
        names, commands, lengths, seeds = zip(*cases, strict=True)
        options = Namespace(device=device, visualizer=None, headless=True, dynamic=dynamic, disable_graph=False)
        initial, baseline = None, None
        paired = {}
        for version, path in paths.items():
            learner, ck, params, motion, original = load_policy(path, root, device)
            physics = make_m4_sim(device, params)
            task = make_task(original)
            with launch_simulation(physics, options):
                env = M4TrackingSim(physics, len(cases), motion, original, eval_mode=True, reset_noise=False,
                                    goals=original.goals)
                _warmup(env, device)
                pose, data = _simulate(env, learner, task, motion, solver, options, 'R2',
                                       commands, lengths, seeds, len(cases))
            if initial is None:
                initial, baseline = pose, data
            else:
                for a, b in zip(initial, pose, strict=True):
                    np.testing.assert_allclose(a, b, atol=1e-7, rtol=0.)
                for key in ('target', 'ref_p', 'ref_R', 'ref_q', 'planned_contact', 'command'):
                    np.testing.assert_allclose(baseline[key], data[key], atol=1e-7, rtol=0.)
            run = '动态' if dynamic else '固定'
            for i, (name, length, seed) in enumerate(zip(names, lengths, seeds, strict=True)):
                track = {key: value[:, i] for key, value in data.items()}
                row = _evaluate(track, run, length, name, seed, 'M3.13 R2', ck['iteration'])
                row.update(policy=version, checkpoint=str(path))
                observed = round(row['seconds']/CONTROL_DT)
                row.update(spatial(track, observed, cpu_kin))
                paired.setdefault(name, []).append((row, track, observed))
                print(f'{version} {name} {row["actual_touchdowns"]}/{row["planned_touchdowns"]} 落脚，'
                      f'存活 {row["seconds"]:.2f}/{row["duration_s"]:.2f} s', flush=True)
            del env, task, original, motion, learner, ck
            torch.cuda.empty_cache()
        for pair in paired.values():
            common = min(n for _, _, n in pair)
            for row, track, n in pair:
                row['common_seconds'] = common*CONTROL_DT
                row.update({f'common_{key}': value for key, value in spatial(track, common, cpu_kin).items()})
                rows.append(row)
    return rows


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, help='复评定选后冻结的 M5.2 检查点')
    parser.add_argument('--out', type=Path, default=OUTPUT)
    parser.add_argument('--reference-root', type=Path, default=REFERENCE_ROOT)
    args = parser.parse_args()
    if args.checkpoint is None:
        with np.load(args.out/'cuda_evaluation.npz', allow_pickle=False) as package:
            info = json.loads(package['metadata'].item())
        if not info.get('complete'):
            parser.error('需要完成 M5.2 独立评估，或通过 --checkpoint 指定冻结候选')
        name = next(key for key in info['selection_order'] if key != 'm51')
        args.checkpoint = Path(info['cases'][name]['checkpoint'])
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    if saved.get('transfer_report', {}).get('strategy') != TRANSFER or not saved.get('iteration', 0):
        parser.error('在线正式回归需要已训练的 M5.2 联合检查点')
    torch.set_num_threads(2)
    rows = run_pair({'M5.1-C1500': SOURCE, 'M5.2': args.checkpoint.resolve()}, args.reference_root, scenarios())
    destination = args.out/'online_regression.csv'
    with TemporaryDirectory(dir=destination.parent) as scratch:
        tmp = Path(scratch)/destination.name
        with tmp.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(destination)


if __name__ == '__main__':
    main()
