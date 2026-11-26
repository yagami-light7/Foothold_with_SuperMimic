"""用已归档轨迹检查手部奖励，或在临时目录验证两组 CUDA 微调。"""

import argparse
import csv
from dataclasses import replace
from itertools import combinations
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ['SUPERMIMIC_ROBOT'] = 'roban'
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import mujoco
import numpy as np
import torch

from supermimic.integration.m410_evaluation import ROOT
from supermimic.integration.m410_preflight import select
from supermimic.integration.m410_quality import local_upper_points
from supermimic.integration.m410_reward import HUBER_TERMS, ROOT_TERMS, SHOULDER_JOINTS
from supermimic.integration.m411_hand import GROUPS, HAND_BODIES, HAND_CONFIG, SOURCE, TORSO_BODY, hand_position_huber
from supermimic.integration.m42v3_ac import cosine, gradient, objective
from supermimic.integration.m43_wm_audit import make_policy, make_wm, task_for
from supermimic.integration.m49_calibration import channel_grad, episode
from supermimic.kinematics import RobotKinematics
from supermimic.mdp import unflatten_state
from supermimic.motion import CONTROL_DT
from supermimic.networks import Critic
from supermimic.roban import ELBOW_JOINT_NAMES, JOINT_NAMES, SOURCE_MJCF
from supermimic.rotations import mat_to_quat, quat_to_mat


CASES = ('m48c_best', 'm49c_9500', 'a_10000', 'b_8000', 'c_9000', 'c_10000')
REFERENCE = Path('/m4_reference.npz')
if not REFERENCE.is_file():
    REFERENCE = REPO.parent / 'Humanoid_Online_Motion_Generator/outputs/m3_7_1/forward/F2/reference.npz'


def checkpoint(path):
    return torch.load(path, map_location='cpu', mmap=True, weights_only=False)


def test_huber():
    error = torch.tensor([0., .029, .030, .0300001, .1299999, .13, .1300001, .30],
                         dtype=torch.float64, requires_grad=True)
    delta = torch.nn.functional.pad(error[:, None, None], (0, 2, 0, 1))
    quality = torch.ones_like(error)
    args = (.03, .10)
    raw, distance, _ = hand_position_huber(delta, quality, *args)
    slope, = torch.autograd.grad(raw.sum(), error)
    torch.testing.assert_close(distance[:, 0], error, atol=0, rtol=0)
    torch.testing.assert_close(raw[:3], torch.zeros_like(raw[:3]), atol=0, rtol=0)
    torch.testing.assert_close(slope[:3], torch.zeros_like(slope[:3]), atol=1.e-10, rtol=0)
    torch.testing.assert_close(slope[4:], error.new_tensor([-5., -5., -5., -5.]), atol=1.e-5, rtol=0)
    assert bool((raw[3:].diff() < 0.).all() and torch.isfinite(slope).all())
    torch.testing.assert_close(hand_position_huber(delta.flip(1), quality, *args)[0], raw, atol=0, rtol=0)
    gate = torch.linspace(0., 1., len(raw), dtype=raw.dtype)
    torch.testing.assert_close(hand_position_huber(delta, gate, *args)[0], gate * raw, atol=0, rtol=0)


def verify_fk(state, mdp):
    model = mujoco.MjModel.from_xml_path(str(SOURCE_MJCF))
    data = mujoco.MjData(model)
    torso = model.body(TORSO_BODY).id
    hands = [model.body(name).id for name in HAND_BODIES]
    indices = [int(model.joint(name).qposadr[0]) for name in JOINT_NAMES]
    p, rotation, v, w, q, qd = state
    position, frames, quat, joints = (x.detach().cpu().numpy() for x in (p, rotation, mat_to_quat(rotation), q))
    independently_local = []
    for root, matrix, wxyz, angles in zip(position, frames, quat, joints):
        data.qpos[:3], data.qpos[3:7], data.qpos[indices] = root, wxyz, angles
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.xpos[torso], root, atol=1.e-7, rtol=0)
        np.testing.assert_allclose(data.xmat[torso].reshape(3, 3), matrix, atol=2.e-6, rtol=0)
        independently_local.append((data.xpos[hands] - data.xpos[torso]) @ data.xmat[torso].reshape(3, 3))
    current = local_upper_points(state, mdp.kin)[:, 4:]
    numerical = torch.tensor(np.asarray(independently_local), dtype=current.dtype)
    torch.testing.assert_close(current, numerical, atol=2.e-6, rtol=0)
    offset = quat_to_mat(p.new_tensor([.72, -.30, .51, .36]))
    rotated = local_upper_points((p @ offset.T + p.new_tensor([.75, -.54, .17]),
                                  offset @ rotation, v @ offset.T, w @ offset.T, q, qd), mdp.kin)[:, 4:]
    torch.testing.assert_close(rotated, current, atol=2.e-6, rtol=0)
    return ((current - numerical).norm(dim=-1).amax(0).tolist(),
            (current - rotated).norm(dim=-1).amax(0).tolist())


def verify_reward(task, source_cfg):
    _, motion, _, mdp = task
    f = torch.tensor([13, 45, 144])
    command = motion.anchor(torch.zeros_like(f))
    ref = mdp._reference(f, command)
    reference = tuple(ref[name][:, 0] for name in ('root_p', 'root_R', 'root_v', 'root_w', 'q', 'qd'))
    q = reference[4].clone()
    q[:, JOINT_NAMES.index('zarm_l4_joint')] += .7
    q.requires_grad_()
    state = (*reference[:4], q, reference[5])
    action = torch.zeros_like(q)
    ctx = dict(command=command, ep_step=f, a_prev2=action, pd_target=reference[4])
    mdp.cfg = source_cfg
    table = mdp.hand_reference
    mdp.hand_reference = None
    old = mdp.reward(state, f, action, action, **ctx)
    mdp.prepare_hand_tracking()
    same, _ = mdp.reward(state, f, action, action, True, **ctx)
    assert torch.equal(old, same)
    torch.testing.assert_close(table, mdp.hand_reference, atol=0, rtol=0)
    mdp.cfg = replace(source_cfg, **HAND_CONFIG)
    zero, terms_zero = mdp.reward(reference, f, action, action, True, **ctx)
    assert torch.equal(terms_zero['hand_huber_contribution'], torch.zeros_like(zero))
    value, terms = mdp.reward(state, f, action, action, True, **ctx)
    left = terms['hand_left_error_sq'].detach().sqrt()
    right = terms['hand_right_error_sq'].detach().sqrt()
    assert bool((left > .03).all() and (right < 1.e-6).all())
    u = (left - .03) / .10
    expected = -.5 * .10 * CONTROL_DT * torch.where(u <= 1., .5 * u.square(), u - .5)
    torch.testing.assert_close(terms['hand_huber_contribution'], expected)
    torch.testing.assert_close(value, old + expected)
    grad, = torch.autograd.grad(terms['hand_huber_contribution'].sum(), q)
    assert torch.isfinite(grad).all() and grad[:, JOINT_NAMES.index('zarm_l4_joint')].abs().min() > 0.
    assert grad[:, [JOINT_NAMES.index(name) for name in SHOULDER_JOINTS[:3]]].norm() > 0.
    assert grad[:, mdp.waist_ids].abs().max() < 1.e-8
    with patch.object(mdp, '_reference', return_value=dict(ref, quality=ref['quality'] * 0.)):
        _, masked = mdp.reward(state, f, action, action, True, **ctx)
    assert torch.equal(masked['hand_huber_contribution'], torch.zeros_like(value))
    assert math.isclose(mdp.max_step_reward, .101)
    print('手部奖励单测通过。零误差、容许范围、Huber 导数、左右手、质量门控、权重与 dt、关节梯度及关闭项兼容。', flush=True)


@torch.no_grad()
def score_history(case, archive, task):
    _, motion, _, mdp = task
    rows = []
    for trial in case['full'].values():
        prefix = trial['prefix']
        frames = torch.from_numpy(archive[prefix + 'frame'][1:]).long()
        state = unflatten_state(torch.from_numpy(archive[prefix + 'state'][1:]))
        delta = local_upper_points(state, mdp.kin)[:, 4:] - mdp.hand_reference[frames]
        raw, distance, excess = hand_position_huber(delta, motion.quality[frames], .03, .10)
        contact = torch.from_numpy(archive[prefix + 'contact'][1:]).sum(-1)
        touchdown = ((frames[:, None] * CONTROL_DT - motion.touchdown_time).abs() <= .100001).any(-1)
        masks = torch.stack((frames <= 10, contact == 1, contact == 2, touchdown), dim=1)
        rows.append((raw * (.10 * CONTROL_DT), distance, excess, masks))
    contribution, distance, excess, masks = (torch.cat(items) for items in zip(*rows))
    quantiles = torch.quantile(-contribution, torch.tensor([.95, 1.]))
    return dict(trials=len(rows), steps=len(contribution), contribution=contribution.mean().item(),
                penalty_p95_max=quantiles.tolist(), rmse_m=distance.square().mean(0).sqrt().tolist(),
                outside=(excess > 0.).float().mean(0).tolist(),
                phase={name: [int(mask.sum()), float(contribution[mask].mean()) if mask.any() else None]
                       for name, mask in zip(('startup', 'single', 'double', 'touchdown'), masks.T)})


def verify_bptt(task, ck, case, archive):
    args, motion, goals, mdp = task
    entries = []
    for seed, row in case['full'].items():
        batch, masks, measured = episode(archive, row['prefix'], 'raw_action', task, int(seed) >= 1100)
        masks['startup'] = batch[2] < 10
        masks['single_support'] = torch.from_numpy(archive[row['prefix'] + 'contact'][:-1]).sum(-1) == 1
        entries.append(dict(batch=batch, masks=masks, data=dict(frame=measured[0])))
    pi, wm = make_policy(ck, mdp, goals, 'cpu'), make_wm(ck, 'cpu')
    critic = Critic(mdp.critic_dim + goals.spec.dim).eval().requires_grad_(False)
    critic.load_state_dict(ck['critic'])
    for phase, count in (('all', 24), ('startup', 16), ('single_support', 16), ('touchdown', 16)):
        batch = select(entries, args.horizon, phase, count)
        packet = objective(batch[:6], pi, critic, wm, motion, mdp, goals, args,
                           ck['imagination_contract']['bootstrap_upper_bound'], 41101, action_context=batch[6:])
        direct = packet['direct']
        losses = dict(foot=packet['C'], root=sum(direct[key] for key in ROOT_TERMS),
                      joint=sum(direct[key] for key in HUBER_TERMS), hand=direct['hand_huber'],
                      smoothness=direct['upper_action_smoothness'], bootstrap=packet['bootstrap'], total=packet['A'])
        grads = {name: gradient(loss, tuple(pi.parameters())) for name, loss in losses.items()}
        assert all(g.isfinite().all() for g in grads.values()) and grads['hand'].norm() > 0.
        channels = dict(arm=mdp.upper_ids, waist=mdp.waist_ids,
                        shoulder=[JOINT_NAMES.index(name) for name in SHOULDER_JOINTS],
                        elbow=[JOINT_NAMES.index(name) for name in ELBOW_JOINT_NAMES],
                        leg=[j for j, name in enumerate(JOINT_NAMES) if name.startswith('leg_')])
        actions = {node: channel_grad(losses['hand'], packet[node]) for node in ('actions', 'effective')}
        assert all(value.isfinite().all() for value in actions.values())
        assert all(actions['actions'][..., channels[name]].norm() > 0. for name in ('shoulder', 'elbow'))
        print('BPTT', phase, json.dumps(dict(starts=len(batch[0]),
              norms={key: round(value.norm().item(), 6) for key, value in grads.items()},
              cosines={f'{a}/{b}': cosine(grads[a], grads[b]) for a, b in combinations(grads, 2)},
              hand_channel_rms={node: {name: value[..., ids].square().mean().sqrt().item()
                                       for name, ids in channels.items()} for node, value in actions.items()})), flush=True)


def preflight():
    torch.set_num_threads(4)
    test_huber()
    ck = checkpoint(REPO / SOURCE)
    task = task_for(ck, RobotKinematics('cpu'), REFERENCE, 'cpu')
    _, motion, _, mdp = task
    assert motion.validation == ck['data_contract']
    mdp.prepare_hand_tracking()
    source_cfg = mdp.cfg
    ref = motion.at_time(motion.control_time)
    f2 = tuple(ref[name] for name in ('root_p', 'root_R', 'root_v', 'root_w', 'q', 'qd'))
    torch.testing.assert_close(mdp.hand_reference, local_upper_points(f2, mdp.kin)[:, 4:], atol=0, rtol=0)
    archive_path = REPO / ROOT / 'cuda_evaluation.npz'
    with np.load(archive_path, allow_pickle=False) as archive:
        cases = json.loads(archive['metadata'].item())['cases']
        row = cases['c_9000']['full']['1001']
        measured = unflatten_state(torch.from_numpy(archive[row['prefix'] + 'state']))
        for name, state in (('F2', f2), ('C9000-1001', measured)):
            print('FK', name, len(state[0]), verify_fk(state, mdp), flush=True)
        verify_reward(task, source_cfg)
        for name in CASES:
            print('History', name, json.dumps(score_history(cases[name], archive, task)), flush=True)
        verify_bptt(task, ck, cases['c_9000'], archive)


def run(command):
    subprocess.run(command, cwd=REPO, check=True)


def smoke():
    assert torch.cuda.is_available()
    source = checkpoint(REPO / SOURCE)
    warmup_rows = {}
    with TemporaryDirectory(prefix='m411-smoke-', dir='/tmp') as scratch:
        for key, name in GROUPS.items():
            out = Path(scratch) / name
            weights = out / 'checkpoints'
            trainer = [sys.executable, '-u', 'supermimic/train.py', '--m411_group', key, '--m411_smoke',
                       '--roban_config', f'configs/roban/m4_11_{key}.json', '--m4_reference', str(REFERENCE),
                       '--log_dir', str(weights), '--device', 'cuda:0']
            run(trainer)
            initial, warm, last = [checkpoint(weights / path) for path in ('initial.pt', 'critic_warmup.pt', 'last.pt')]
            for model in ('pi', 'wm', 'critic'):
                for param, value in source[model].items():
                    assert torch.equal(initial[model][param], value), (key, model, param)
            optimizers = ('opt_pi', 'opt_wm', 'opt_v')
            assert all(not initial[name]['state'] for name in optimizers)
            for model in ('pi', 'wm'):
                assert all(torch.equal(warm[model][name], value) for name, value in initial[model].items())
                assert any(not torch.equal(last[model][name], value) for name, value in warm[model].items()
                           if 'norm.' not in name)
            assert not warm['opt_pi']['state'] and not warm['opt_wm']['state'] and warm['opt_v']['state']
            assert warm['iteration'] == 0 and warm['critic_warmup_iterations'] == 2
            assert last['iteration'] == 2 and last['critic_warmup_iterations'] == 2
            assert last['opt_pi']['param_groups'][0]['lr'] == 5.e-5 and (weights / 'iter_0001.pt').exists()
            with (out / 'logs/train.csv').open() as log:
                rows = list(csv.DictReader(log))
            assert [r['stage'] for r in rows] == ['critic_warmup'] * 2 + ['joint'] * 2
            assert all(math.isfinite(float(value)) for row in rows for field, value in row.items()
                       if field != 'stage' and value not in ('', None))
            assert all(float(r['reward_sync_max_error']) <= 1.e-5 and float(r['epa_resets']) == 0
                       and float(r['invalid_transitions']) == 0 and float(r['physics_exceptions']) == 0 for r in rows)
            assert all(float(r['pi_gradnorm']) == 0. for r in rows[:2])
            assert all(float(r['pi_gradnorm']) > 0. for r in rows[2:])
            assert all(float(r['foot_actor_gradnorm']) > 0. for r in rows)
            for row in rows:
                actual, common, hand = (float(row[k]) for k in ('total_reward', 'm410_reward', 'hand_huber_contribution'))
                assert abs(actual - common - hand) < 2.e-7
                assert hand == 0. if key == 'a' else hand < 0.
            warmup_rows[key] = rows[:2]
            print(f'M4.11 {key.upper()} Smoke 通过，'
                  f'手部梯度={rows[-1]["term_hand_huber_actor_gradnorm"]}，'
                  f'Bootstrap 梯度={rows[-1]["actor_bootstrap_gradnorm"]}，'
                  f'Actor 梯度={rows[-1]["pi_gradnorm"]}，'
                  f'最大显存={max(float(row["device_used_mib"]) for row in rows):.0f} MiB', flush=True)
            if key == 'b':
                assert float(rows[-1]['term_hand_huber_actor_gradnorm']) > 0.
                with TemporaryDirectory(prefix='m411-smoke-', dir='/tmp') as resume:
                    dest = Path(resume) / name / 'checkpoints'
                    dest.mkdir(parents=True)
                    warmup_path = dest / 'critic_warmup.pt'
                    warmup_path.hardlink_to(weights / warmup_path.name)
                    trainer[trainer.index('--log_dir') + 1] = str(dest)
                    run(trainer + ['--resume', str(warmup_path)])
                    resumed = checkpoint(dest / 'last.pt')
                    assert resumed['critic_warmup_iterations'] == 2 and resumed['iteration'] == 2
                    with (dest.parent / 'logs/train.csv').open() as log:
                        assert [row['stage'] for row in csv.DictReader(log)] == ['joint'] * 2
                    print('M4.11 B Smoke 已从 Critic Warm-up 检查点恢复，两轮均进入联合更新。', flush=True)
                run([sys.executable, '-m', 'supermimic.integration.m43_evaluation', '--m411-periodic',
                     '--m4_reference', str(REFERENCE), '--out', str(out), '--device', 'cuda:0'])
                evaluation = json.loads((out / 'evaluation/results.json').read_text())
                assert (weights / 'best_foot.pt').is_file()
                if evaluation['milestones']['2']['aggregate']['complete_tracking_rate'] == 1.:
                    assert (weights / 'best_whole_body.pt').is_file()
        # 两个独立 GPU 求解器的采集差值用于报告，同批数据的奖励等式已逐行核对。
        metrics = ('m410_reward', 'foot_rmse_m', 'hand_left_rmse_m', 'hand_right_rmse_m')
        diffs = {name: max(abs(float(a[name]) - float(b[name]))
                           for a, b in zip(warmup_rows['a'], warmup_rows['b'])) for name in metrics}
        print(f'两组 CUDA Smoke 通过，预热采集的共同奖励和位置 RMSE 最大差值 {diffs}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true', help='跑隔离的两组 CUDA 冒烟，不写正式输出目录')
    smoke() if parser.parse_args().smoke else preflight()
