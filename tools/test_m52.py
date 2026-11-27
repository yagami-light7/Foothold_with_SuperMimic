"""M5.2 数学、历史 Replay、M3.13 缓存以及 CUDA 训练冒烟验收。"""

from argparse import ArgumentParser, Namespace
from contextlib import chdir
import json
import os
from pathlib import Path
import sys

os.environ['SUPERMIMIC_ROBOT'] = 'roban'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from supermimic.integration.m50_evaluation import TRACE_FIELDS
from supermimic.integration.m50_reference import MANIFEST, REFERENCE_ROOT, load_clip, read_manifest
from supermimic.integration.m50_task import build_m50_task
from supermimic.integration.m51_audit import bptt_audit, cost_distribution, pre_states, score, wm_predictions
from supermimic.integration.m51_foot import rotation_error
from supermimic.integration.m52_audit import ARCHIVE, frozen_task, update_audit
from supermimic.integration.m52_orientation import angle_error, python, rms, tilt_angle
from supermimic.integration.m52_task import (CONFIGS, GROUPS, OUTPUT, REWARD_KEYS, SOURCE,
                                             RootTiltMDP, RootTiltRewardConfig, tilt_reward)
from supermimic.kinematics import RobotKinematics
from supermimic.mdp import unflatten_state
from supermimic.motion import CONTROL_DT
from supermimic.networks import Critic, Policy, WorldModel
from supermimic.roban import JOINT_NAMES
from supermimic.rotations import mat_to_quat, quat_to_mat, so3_exp

from test_m50 import run_smoke


def mathematics(device):
    def R(axis, angle):
        vector = torch.tensor(axis, device=device, dtype=torch.float64)
        return so3_exp(vector*angle)

    x, y, z = (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)
    ident, pitch, roll = R(x, 0.), R(y, .17), R(x, .21)
    conf = RootTiltRewardConfig(root_tilt=.5)
    quality = lambda a, b: tilt_reward(a, b, a.new_ones(a.shape[:-2]), conf)[1]
    equal = lambda a, b: torch.testing.assert_close(a, b, atol=1e-12, rtol=0.)
    equal(quality(ident, ident), ident.new_tensor(1.))
    equal(quality(roll@pitch, roll@pitch), ident.new_tensor(1.))
    assert 0. < quality(ident, pitch) < 1. and 0. < quality(ident, roll) < 1.
    base = quality(roll, pitch)
    for turn in (-3.142, -3.14, 1.35, 3.14, 3.142):
        rotation = R(z, turn)
        equal(quality(rotation@roll, pitch), base)
        equal(quality(roll, rotation@pitch), base)
        equal(quality(rotation@roll, rotation@pitch), base)
    quat = mat_to_quat(pitch)
    equal(quality(ident, quat_to_mat(-2*quat)), quality(ident, pitch))
    for matrix in (R(y, torch.pi/2), R(y, torch.pi/2-1e-6), R(x, torch.pi), pitch, ident):
        pose = matrix.detach().requires_grad_()
        value = quality(pose, pitch)
        assert value.isfinite() and torch.autograd.grad(value, pose)[0].isfinite().all()
    penalty = RootTiltRewardConfig(root_tilt=.5, root_tilt_penalty=1)
    weight = pitch.new_tensor(.37)
    raw, value = tilt_reward(ident, pitch, weight, penalty)
    assert raw < 0.
    equal(raw, (value-1.)*weight)
    raw, _ = tilt_reward(pitch, pitch, weight, penalty)
    assert raw.item() == 0.
    print(f'{device} 数学验收通过，含姿态响应、独立／共同 Yaw、环绕、四元数正负等价与退化梯度', flush=True)
    return dict(device=device, passed=True)


def parameters(saved, root, group=None):
    args = Namespace(**saved['args'])
    args.m50_library, args.m50_reference_root = str(MANIFEST), str(root)
    if group is not None:
        cfg = json.loads(CONFIGS[group].read_text())
        args.m52_group, args.roban_reward_config = group, cfg['roban_reward_config']
    return args


@torch.no_grad()
def score_cases(models, traces):
    original, a, b, c = models
    reward_max, tilt_quality, tilt_cost = 0., [], []
    for trace in traces:
        x = pre_states(trace, original.default_q.device)
        f = x['frame'][1:]
        anchor = original.motion.anchor(torch.zeros_like(f))
        state = unflatten_state(x['states'][1:])
        i = torch.arange(len(f), device=f.device)
        previous, base = score(original, x, i, state, anchor)
        actual, left = score(a, x, i, state, anchor)
        torch.testing.assert_close(actual, previous, atol=0., rtol=0.)
        for key, value in base.items():
            torch.testing.assert_close(left[key], value, atol=0., rtol=0.)
        reward_max = max(reward_max, float((previous-torch.as_tensor(trace['reward'], device=f.device)).abs().max()))
        for name, model in (('b', b), ('c', c)):
            reward, terms = score(model, x, i, state, anchor)
            assert torch.isfinite(reward).all()
            ref = model._reference(f, anchor)
            q = terms['root_tilt_quality']
            factor = (q-1.) if name == 'b' else q
            wanted = factor*ref['quality'][:, 0]*(.5*CONTROL_DT)
            torch.testing.assert_close(terms['root_tilt_contribution'], wanted, atol=2e-9, rtol=0.)
            removed = base['upright_contribution'] if name == 'c' else 0.
            torch.testing.assert_close(reward, previous-removed+wanted, atol=5e-8, rtol=0.)
            assert torch.equal(terms['survival_contribution'], base['survival_contribution'])
            assert torch.equal(model.termination(state), original.termination(state))
            assert torch.equal(model.soft_termination(state), original.soft_termination(state))
            if name == 'b':
                assert (terms['root_tilt_contribution'] <= 0.).all()
                tilt_quality.append(q.cpu().numpy())
                tilt_cost.append(wanted.cpu().numpy())
            else:
                assert not terms['upright_contribution'].any()
    assert reward_max <= 2e-6, reward_max
    # 明确触发低根高与过大倾斜两种终止，核对 C 仍保留物理终止保护。
    p, R, *tail = state
    down = so3_exp(R.new_tensor([torch.pi, 0., 0.])) @ R
    low = torch.cat((p[:, :2], torch.zeros_like(p[:, 2:])), dim=-1)
    for abnormal in ((p, down, *tail), (low, R, *tail)):
        assert original.termination(abnormal).all()
        assert torch.equal(c.termination(abnormal), original.termination(abnormal))
        assert torch.equal(c.soft_termination(abnormal), original.soft_termination(abnormal))
    return dict(historical_reward_max_error=reward_max, a_exact=True, tracks=len(traces),
                quality=cost_distribution(np.concatenate(tilt_quality)),
                penalty=cost_distribution(np.concatenate(tilt_cost)))


def extra_prediction(predict, actual, simulated, replay):
    roll, pitch = rms(angle_error(predict[1], actual[1]).rad2deg())[:2]
    delta = simulated['root_tilt_contribution']-replay['root_tilt_contribution']
    return dict(root_roll_rmse_deg=python(roll), root_pitch_rmse_deg=python(pitch),
                root_tilt_rmse_deg=python(rms(tilt_angle(predict[1], actual[1]).rad2deg())),
                root_so3_rmse_deg=python(rms(rotation_error(predict[1], actual[1]).rad2deg())),
                root_omega_rmse_rad_s=python(rms((predict[3]-actual[3]).norm(dim=-1))),
                root_tilt_reward_mae=python(delta.abs().mean()), root_tilt_reward_bias=python(delta.mean()))


def history(root, device, saved):
    kin = RobotKinematics(device)
    actor, wm, critic = Policy(69, 320, hist=saved['args']['hist']), WorldModel(), Critic(970)
    for name, network in (('pi', actor), ('wm', wm), ('critic', critic)):
        network.to(device).load_state_dict(saved[name])
        network.eval().requires_grad_(name == 'pi')
    checks = {}
    with np.load(ARCHIVE, allow_pickle=False) as package:
        cases = json.loads(package['metadata'].item())['cases']['c1500']['seen']
        for entry in read_manifest()['train']:
            name = entry['clip_id']
            motion = load_clip(entry, kin, device, root, 'train')
            models = [frozen_task(kin, motion, saved)]
            for group in GROUPS:
                cfg = parameters(saved, root, group)
                models.append(frozen_task(kin, motion, saved, RootTiltMDP, cfg))
            base = models[0].cfg.contract()
            for model in models[1:]:
                effective = {key: value for key, value in model.cfg.contract().items()
                             if key not in REWARD_KEYS and key != 'upright'}
                assert effective == {key: value for key, value in base.items() if key != 'upright'}
                assert model.max_step_reward == models[0].max_step_reward
            arrays = {seed: {key: package[row['prefix']+key] for key in (*TRACE_FIELDS, 'foot_angular_velocity')}
                      for seed, row in cases[name].items()}
            audit = score_cases(models, list(arrays.values()))
            wm_report = wm_predictions(models[2], wm, list(arrays.values()), extend=extra_prediction)
            channels = {key: [i for i, joint in enumerate(JOINT_NAMES) if joint.startswith(pattern)]
                        for key, pattern in (('legs', 'leg_'), ('arms', 'zarm_'), ('waist', 'waist_'))}
            objectives = ('upright', 'root_tilt', 'root_xy', 'root_height', 'root_heading')
            grad = bptt_audit(models[2], actor, wm, critic, saved['args'],
                              [arrays[str(seed)] for seed in (1001, 1101, 1201)],
                              extra_terms=objectives, pair_terms=('root_tilt',), extra_channels=channels)
            assert set(grad) >= {'startup', 'single', 'double', 'touchdown', 'walking'}
            assert all(row['objectives']['root_tilt']['actor_norm'] > 0. for row in grad.values())
            checks[name] = dict(replay=audit, wm=wm_report, bptt=grad)
            data = wm_report['horizons']
            print(f'{name} {audit["tracks"]} 条真实 Replay，1/4/8/16 步 Tilt 预测 RMSE '
                  f'{[round(data[k]["root_tilt_rmse_deg"], 3) for k in ("1", "4", "8", "16")]}°，'
                  f'Tilt Actor 范数 {min(r["objectives"]["root_tilt"]["actor_norm"] for r in grad.values()):.4g}'
                  f'–{max(r["objectives"]["root_tilt"]["actor_norm"] for r in grad.values()):.4g}', flush=True)
    return checks


def cache_parity(root, device, saved):
    from roban_motion.m311_replay import capture_reference
    from roban_motion.m313_reference import HybridReference, LiveCache, StaticBank
    from roban_motion.m313_reward import make_task
    from roban_motion.m313_support import engine
    from roban_motion.m313_training import pack_initial
    from roban_motion.m313_validation import _reward_parity
    from roban_motion.m3_11 import PROJECT, setup

    cfg = parameters(saved, root, 'b')
    kin = RobotKinematics(device)
    motion, _, mdp, _ = build_m50_task(cfg, kin, device)
    ids = torch.arange(3, device=device)
    starts = motion.clip_start
    step = ids.new_tensor([35, 53, 62])
    frames = starts+step
    original = motion.ref_state(frames)
    state = (original[0], so3_exp(motion.q.new_tensor([.04, -.1, .08])) @ original[1], *original[2:])
    bank = StaticBank.from_motion(motion, mdp)
    pack = HybridReference(bank, None, len(ids), device)
    pack.reset(ids, ids, ids < 0)
    cached = make_task(mdp)
    rows = _reward_parity(mdp, cached, motion.anchor(starts), pack.issue(step), state, frames, step,
                          cfg, saved, 'M5.2 静态参考')
    # 用同一时钟对照真正发布的在线节点及未缓存的 quat 查询。
    with chdir(PROJECT):
        _, _, cpu = setup()
    entries = read_manifest()['train']
    g = engine(cpu, len(ids), device, temporal=True)
    g.reset(ids, pack_initial(g, motion.ref_state(starts), ids, mat_to_quat),
            g.fk.tensor([(e['step_length'], e['step_frequency'], e['yaw_rate']) for e in entries]))
    assert g.ensure_future().all()
    for _ in range(40):
        assert g.advance().valid.all()
    cache = LiveCache(g, cached.kin, motion, cached)
    pack = HybridReference(bank, cache, len(ids), device)
    pack.reset(ids, ids, ids >= 0)
    f = g.ring.head.clone()
    ticket = pack.issue(f)
    offline_quat = capture_reference(g).lookup(f[:, None]).root_orientation[:, 0]
    ref = quat_to_mat(offline_quat)
    R = state[1].detach().clone().requires_grad_()
    zero = torch.zeros_like(state[4])
    _, terms = cached.reward((state[0],R,*state[2:]), f, zero, zero, True, command=ticket, ep_step=f,
                             a_prev2=zero, pd_target=mdp.executed_target(zero,state[4],f))
    expected = tilt_reward(R, ref, R.new_ones(len(R)), mdp.cfg)[0]*(mdp.cfg.root_tilt*CONTROL_DT)
    measured = terms['root_tilt_contribution']
    torch.testing.assert_close(measured, expected, atol=2e-9, rtol=0.)
    actual_grad = torch.autograd.grad(measured.sum(), R, retain_graph=True)[0]
    raw_grad = torch.autograd.grad(expected.sum(), R)[0]
    torch.testing.assert_close(actual_grad, raw_grad, atol=2e-8, rtol=0.)
    if device == 'cpu':
        pack.reset(ids[:1], ids[:1], ids[:1] >= 0)
        try:
            ticket.sample(f[:, None], ('root_R',))
        except RuntimeError as error:
            assert 'Episode' in str(error)
        else:
            raise AssertionError('旧 Episode 的在线票据仍可访问')
    result = dict(passed=True, static_metrics_checked=len(rows),
                  static_max_error=max(x['max_abs_difference'] for x in rows),
                  online_reward_delta=python((measured-expected).abs().max()),
                  online_gradient_delta=python((actual_grad-raw_grad).abs().max()))
    print(f'{device} M3.13 缓存差分通过 {result}', flush=True)
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('audit', 'cuda'), required=True)
    parser.add_argument('--root', type=Path, default=REFERENCE_ROOT)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(2)
    path = args.out/'preflight.json'
    saved = torch.load(SOURCE, map_location='cpu', weights_only=False, mmap=True)
    if args.stage == 'audit':
        result = dict(unit=mathematics('cpu'), cache=cache_parity(args.root, 'cpu', saved),
                      dynamics=history(args.root, 'cpu', saved))
        update_audit(path, **result)
    if args.stage == 'cuda':
        assert torch.cuda.is_available(), '未找到 CUDA 实际设备'
        cuda = dict(unit=mathematics('cuda:0'), cache=cache_parity(args.root, 'cuda:0', saved))
        torch.cuda.empty_cache()
        smokes = [run_smoke(args.root, n, group=group, root_tilt=True, resume_test=n == 256)
                  for group, n in (('a', 8), ('b', 32), ('c', 256))]
        update_audit(path, cuda=cuda, smokes=smokes)


if __name__ == '__main__':
    main()
