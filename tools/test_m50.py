"""核对 M5.0 参考、回放、冻结和 CUDA 训练入口。"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

os.environ['SUPERMIMIC_ROBOT'] = 'roban'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
import torch

from supermimic.integration.m43_wm_audit import task_for
from supermimic.integration.m4_task import M4TrackingSim, make_m4_sim
from supermimic.integration.m50_reference import MANIFEST, REFERENCE_ROOT, REPO, SOURCE, load_clip, read_manifest
from supermimic.integration.m50_task import CONFIG, OUTPUT, MultiReferenceSim, build_m50_task, curriculum_at
from supermimic.buffer import ReplayBuffer
from supermimic.kinematics import RobotKinematics
from supermimic.mdp import normalize_xi, unflatten_state
from supermimic.networks import Policy, WorldModel
from supermimic.roban import FOOT_BODY_NAMES, JOINT_NAMES, SOURCE_MJCF


def read_checkpoint(path):
    return torch.load(path, weights_only=False, map_location='cpu', mmap=True)


def independent_fk(clip):
    model = mujoco.MjModel.from_xml_path(str(SOURCE_MJCF))
    data = mujoco.MjData(model)
    joints = [int(model.joint(name).qposadr[0]) for name in JOINT_NAMES]
    feet = [model.body(name).id for name in FOOT_BODY_NAMES]
    hands = [model.body(name).id for name in ('l_handball', 'r_handball')]
    ref = clip.at_time(clip.control_time)
    hand_idx = [clip.kin.tracked_names.index(name) for name in ('l_handball', 'r_handball')]
    hand = clip.body_p[:, hand_idx].cpu().numpy()
    max_foot = max_hand = 0.
    for i, (p, quat, q) in enumerate(zip(ref['root_p'].cpu(), ref['root_quat'].cpu(), ref['q'].cpu())):
        data.qpos[:3], data.qpos[3:7], data.qpos[joints] = p.numpy(), quat.numpy(), q.numpy()
        mujoco.mj_forward(model, data)
        rotation = data.xmat[feet].reshape(2, 3, 3)
        sole = data.xpos[feet] + (rotation @ clip.sole_offsets.cpu().numpy()[..., None]).squeeze(-1)
        max_foot = max(max_foot, float(np.linalg.norm(sole - clip.foot_p[i].cpu().numpy(), axis=-1).max()))
        max_hand = max(max_hand, float(np.linalg.norm(data.xpos[hands] - hand[i], axis=-1).max()))
    if max_foot > .002 or max_hand > 2e-6:
        raise AssertionError(f'{clip.clip_name} 独立 FK 失败')
    return dict(foot_m=max_foot, hand_m=max_hand)


def reset_contract(motion, mdp):
    # 在课程边界检查实际复位，三个片段长度不同但均只从首帧启动。
    sim = MultiReferenceSim.__new__(MultiReferenceSim)
    sim.motion, sim.mdp, sim.eval_mode = motion, mdp, False
    sim.clip_generator = torch.Generator().manual_seed(50003)
    sim.pose_generator = torch.Generator().manual_seed(50021)
    sim.sample_probabilities = motion.q.new_empty(3)
    sim.reset_counts = torch.zeros(3, dtype=torch.long)
    sim.perturbed_counts = torch.zeros_like(sim.reset_counts)
    results = {}
    for iteration, expected in ((1500, 0.), (1501, .25), (6000, .50)):
        sim.set_iteration(iteration)
        before = sim.perturbed_counts.clone()
        frames, ids = sim.sample_frames(4096)
        assert (frames == motion.clip_start[ids]).all()
        state = motion.ref_state(frames)
        actual = sim.prepare_motion_reset(state, frames)
        dp = actual[0] - state[0]
        active = dp[:, 2] > 0
        ratio = active.double().mean().item()
        assert abs(ratio-expected) < .025
        assert int((sim.perturbed_counts-before).sum()) == int(active.sum())
        for a, b in zip(actual, state):
            torch.testing.assert_close(a[~active], b[~active], atol=1e-7, rtol=0.)
        if active.any():
            scale = sim.perturb_scale
            assert dp[active, 2].min() >= .006*scale-1e-6 and dp[:, 2].max() <= .012*scale+1e-6
            assert dp[:, :2].abs().max() <= .02*scale+1e-6
            for i, limit in ((2, .02), (3, .05), (4, .01), (5, .10)):
                assert (actual[i] - state[i]).abs().max() <= limit*scale+1e-6
        results[str(iteration)] = dict(observed_fraction=ratio, scale=sim.perturb_scale)
    return results


def numeric(root, device):
    saved = read_checkpoint(SOURCE)
    manifest = read_manifest()
    config = argparse.Namespace(**saved['args'])
    config.m50_library, config.m50_reference_root = str(MANIFEST), str(root)
    kin = RobotKinematics(device)
    motion, goals, mdp, _ = build_m50_task(config, kin, device)
    f2_entry = manifest['train'][1]
    _, old, old_goals, old_mdp = task_for(saved, kin, root / f2_entry['file_path'], device)
    f2_only, one_goals, _, _ = build_m50_task(config, kin, device, entries=[f2_entry])
    assert (mdp.obs_dim, goals.spec.dim, mdp.critic_dim + goals.spec.dim) == (69, 320, 970)
    assert math.isclose(mdp.max_step_reward, .101) and old_mdp.cfg.contract() == mdp.cfg.contract()
    f = torch.arange(old.num_frames, device=device)
    packed = f + motion.clip_start[1]
    torch.testing.assert_close(mdp.hand_reference[packed], old_mdp.hand_reference, atol=0., rtol=0.)
    original = old.anchor(torch.zeros_like(f))
    new = motion.anchor(torch.full_like(f, int(motion.clip_start[1])))
    state = old.ref_state(f)
    torch.testing.assert_close(goals.goal(state, packed, f, new), old_goals.goal(state, f, f, original),
                               atol=0., rtol=0.)
    actions = torch.sin(torch.arange(len(f) * len(JOINT_NAMES), device=device).reshape(len(f), -1)) * .5
    history = dict(a_prev2=actions * -.1, pd_target=mdp.executed_target(actions, state[4], f))
    old_value, old_terms = old_mdp.reward(state, f, actions, -actions, True, command=original, ep_step=f, **history)
    value, terms = mdp.reward(state, packed, actions, -actions, True, command=new, ep_step=f, **history)
    for key in old_terms:
        torch.testing.assert_close(old_terms[key], terms[key], atol=2e-7, rtol=0.)
    torch.testing.assert_close(value, old_value, atol=2e-7, rtol=0.)
    torch.testing.assert_close(mdp.termination(state, packed), old_mdp.termination(state, f), atol=0., rtol=0.)
    numbers = dict(reward_delta_max=float((value - old_value).abs().max()), dimensions=[69, 320, 970])

    for i, entry in enumerate(manifest['train']):
        clip = load_clip(entry, kin, device, root, 'train')
        last = clip.num_frames - 1
        indices = torch.tensor([0, 20, last-1, last], device=device)
        ids = torch.full_like(indices, i)
        offsets = torch.tensor([0, 5, 16, 50, 66], device=device)
        query = motion.get_future(ids, indices, offsets)
        single = clip.sample((indices[:, None] + offsets).clamp_max(last), clip.anchor(torch.zeros_like(indices)))
        for key in single:
            torch.testing.assert_close(query[key], single[key], atol=0, rtol=0)
    ids = torch.tensor([0, 1, 2], device=device)
    mixed = motion.get_current(ids, motion.clip_len - 1)
    for i in range(len(ids)):
        torch.testing.assert_close(mixed['q'][i, 0], motion.q[motion.clip_end[i]], atol=0, rtol=0)
    assert curriculum_at(1001)[0] == (.30, .40, .30) and curriculum_at(1501)[1:] == (.25, .5)
    assert curriculum_at(6000)[1:] == (.5, 1.)
    if device == 'cpu':
        numbers['reset_curriculum'] = reset_contract(motion, mdp)
        numbers['references'] = {}
        for split in ('train', 'heldout'):
            for entry in manifest[split]:
                clip = load_clip(entry, kin, 'cpu', root, split)
                numbers['references'][entry['clip_id']] = clip.validation | dict(independent_fk=independent_fk(clip))
    else:
        # 三种实现均从 GPU 上已经存在的相同形状状态和时钟查询。
        n = 256
        old_f = torch.full((n,), 102, device=device)
        old_c = old.anchor(torch.zeros_like(old_f))
        only_c = f2_only.anchor(torch.zeros_like(old_f))
        mixed_starts = motion.clip_start[torch.arange(n, device=device) % len(motion.names)]
        mixed_f = mixed_starts + old_f
        mixed_c = motion.anchor(mixed_starts)
        durations = {}
        for name, g, s, f, c in (
                ('F2 原入口', old_goals, old.ref_state(old_f), old_f, old_c),
                ('仅 F2 打包', one_goals, old.ref_state(old_f), old_f, only_c),
                ('三片段混合', goals, motion.ref_state(mixed_f), mixed_f, mixed_c)):
            for _ in range(20):
                g.goal(s, f, old_f, c)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(250):
                g.goal(s, f, old_f, c)
            torch.cuda.synchronize()
            durations[name] = (time.perf_counter() - start) * 1000 / 250
        numbers['goal_query_ms_per_256'] = durations
    return numbers


def benchmark(root):
    from isaaclab.app import launch_simulation

    device, n, frames, repeats = 'cuda:0', 256, 30, 160
    saved = read_checkpoint(SOURCE)
    manifest = read_manifest()
    f2 = manifest['train'][1]
    results = {}
    for label in ('legacy_f2', 'packed_f2', 'mixed'):
        torch.manual_seed(0)
        torch.cuda.reset_peak_memory_stats()
        config = argparse.Namespace(**saved['args'])
        config.device = device
        config.m50_library, config.m50_reference_root = str(MANIFEST), str(root)
        kin = RobotKinematics(device)
        if label == 'legacy_f2':
            _, motion, goals, mdp = task_for(saved, kin, root / f2['file_path'], device)
        else:
            motion, goals, mdp, _ = build_m50_task(config, kin, device, entries=[f2] if label == 'packed_f2' else None)
        actor = Policy(mdp.obs_dim, goals.spec.dim, hist=config.hist).to(device).eval()
        actor.load_state_dict(saved['pi'])
        wm = WorldModel().to(device)
        wm.load_state_dict(saved['wm'])
        opt = torch.optim.Adam(wm.parameters(), lr=2.e-4)
        cfg = make_m4_sim(device, config)
        with launch_simulation(cfg, argparse.Namespace(device=device, visualizer=None)):
            if label == 'mixed':
                env = MultiReferenceSim(cfg, n, motion, mdp, seed=0, goals=goals)
            else:
                env = M4TrackingSim(cfg, n, motion, mdp, goals=goals)
            replay = ReplayBuffer(64, n, device, foot_commands=True, action_context=True,
                                  static_motion=None if label == 'legacy_f2' else motion)
            history = torch.zeros(n, config.hist, mdp.obs_dim, device=device)

            @torch.no_grad()
            def rollout(steps):
                nonlocal history
                for _ in range(steps):
                    state = env.state()
                    obs = mdp.add_noise(goals.obs(mdp, state, env.a_prev, env.xi, env.f, env.command))
                    fresh = env.ep_step[:, None, None] == 0
                    history = torch.where(fresh, obs[:, None], torch.cat((history[:, 1:], obs[:, None]), dim=1))
                    goal = goals.goal(state, env.f, env.ep_step, env.command)
                    action = actor(history, goal, action_noise=config.action_noise)
                    row = env.step(action)
                    if 'valid' in row:
                        raise FloatingPointError('性能短测遇到无效转移')
                    replay.add(row)

            def timer(fn):
                torch.cuda.synchronize()
                start = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                return time.perf_counter() - start

            rollout(6)
            collection = timer(lambda: rollout(frames))

            def sample():
                s, _ = replay.sample_starts(128, config.hist)
                replay.commands(s['command_id'], frames=s['f'], episode_steps=s['ep_step'])

            def sample_many():
                for _ in range(repeats):
                    sample()

            sampling = timer(sample_many)

            def fit():
                for _ in range(12):
                    batch = replay.sample_windows(128, 4)
                    state = unflatten_state(batch['s'][:, 0])
                    err = []
                    for t in range(4):
                        state = wm.step(state, batch['a_exec'][:, t], normalize_xi(batch['xi'][:, t]))
                        delta = wm.state_error(state, unflatten_state(batch['s1'][:, t]))
                        err.append((delta / wm.out_norm.std().clamp_min(.01)).square().mean(-1))
                    loss = (torch.stack(err, dim=1) * batch['mask']).mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0, error_if_nonfinite=True)
                    opt.step()

            learning = timer(fit)
            free, total = torch.cuda.mem_get_info()
            results[label] = dict(collection_fps=frames*n/collection, wm_training_fps=frames*n/(collection+learning),
                                  replay_sample_ms=1000*sampling/repeats, learning_ms=1000*learning/12,
                                  device_mib=(total-free)/2**20,
                                  torch_peak_mib=torch.cuda.max_memory_allocated()/2**20)
        torch.cuda.empty_cache()
    return results


def run_smoke(root, envs, *, group=None, resume_test=None):
    config, source = CONFIG, SOURCE
    initial_name, warmup_name, phase = 'initial_b1250.pt', 'after_adaptation.pt', 'adaptation'
    prefix = 'm50-smoke-'
    if group is not None:
        from supermimic.integration.m51_task import CONFIGS, SOURCE as M51_SOURCE

        config, source = CONFIGS[group], M51_SOURCE
        initial_name, warmup_name, phase = 'initial_m50_5500.pt', 'critic_warmup.pt', 'critic_warmup'
        prefix = 'm51-smoke-'
    if resume_test is None:
        resume_test = envs == 8
    with TemporaryDirectory(prefix=prefix, dir='/tmp') as scratch:
        out = Path(scratch)
        ckpt = out / 'checkpoints'
        cmd = [sys.executable, '-u', 'supermimic/train.py', '--roban_config', str(config), '--device', 'cuda:0',
               '--m50_library', str(MANIFEST), '--m50_reference_root', str(root), '--log_dir', str(ckpt),
               '--m50_smoke', str(envs)]
        subprocess.run(cmd, cwd=REPO, check=True)
        initial, adapt, last = [read_checkpoint(ckpt / name) for name in (initial_name, warmup_name, 'last.pt')]
        source = read_checkpoint(source)

        def equal(a, b, model):
            return all(torch.equal(a[model][name], b[model][name]) for name in a[model])

        assert all(equal(initial, source, name) for name in ('pi', 'critic', 'wm'))
        assert all(not initial[key]['state'] for key in ('opt_pi', 'opt_wm', 'opt_v'))
        assert equal(initial, adapt, 'pi') and not adapt['opt_pi']['state']
        assert equal(adapt, initial, 'wm') == (group is not None) and not equal(adapt, initial, 'critic')
        if group is not None:
            assert not adapt['opt_wm']['state'] and last['opt_wm']['state']
            assert not equal(last, adapt, 'wm')
        assert not equal(last, adapt, 'pi') and last['opt_pi']['param_groups'][0]['lr'] == 5.e-5
        assert last['iteration'] == last['adaptation_iterations'] == 2
        with (out/'logs/train.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        assert [row['stage'] for row in rows] == [phase]*2 + ['joint']*2
        for row in rows:
            assert float(row['reward_sync_max_error']) <= 1e-5 and float(row['goal_sync_max_error']) <= 1e-5
            assert all(float(row[key]) == 0. for key in ('invalid_transitions', 'epa_resets', 'physics_exceptions'))
            assert all(math.isfinite(float(value)) for key, value in row.items() if key != 'stage' and value)
            assert abs(sum(float(row[f'{name}/replay_fraction']) for name in ('F1','F2','F3')) - 1.) < 1e-5
        if group is not None:
            assert all(float(r['reset_probability']) == .5 and float(r['reset_scale']) == 1. for r in rows)
            for key, enabled in (('stance_yaw_huber', group != 'a'), ('stance_angular_slip', group == 'c')):
                assert (float(rows[-1][f'term_{key}_actor_gradnorm']) > 0.) == enabled
                assert (float(rows[-1][f'{key}_contribution']) < 0.) == enabled
        if resume_test:
            with TemporaryDirectory(prefix=prefix, dir='/tmp') as resume:
                dest = Path(resume)/'checkpoints'
                dest.mkdir()
                (dest/'last.pt').hardlink_to(ckpt/warmup_name)
                retry = cmd.copy()
                retry[retry.index('--log_dir')+1] = str(dest)
                subprocess.run(retry + ['--resume',str(dest/'last.pt')], cwd=REPO, check=True)
                resumed = read_checkpoint(dest/'last.pt')
                assert resumed['iteration'] == resumed['adaptation_iterations'] == 2
                with (dest.parent/'logs/train.csv').open() as stream:
                    assert [row['stage'] for row in csv.DictReader(stream)] == ['joint']*2
        result = dict(envs=envs, transitions=last['sim_steps'], epa=0,
                    collection_fps=float(rows[-1]['collect_fps']), training_fps=float(rows[-1]['train_fps']),
                    maximum_device_mib=max(float(row['device_used_mib']) for row in rows),
                    actor_grad=float(rows[-1]['pi_gradnorm']))
        if group is not None:
            result.update(group=group, resumed=resume_test, warmup_actor_wm_frozen=True,
                          warmup_value_loss=float(rows[1]['v_loss']),
                          critic_target_min=float(rows[1]['critic_target_min']),
                          critic_target_negative_ratio=float(rows[1]['critic_target_negative_ratio']),
                          yaw_grad=float(rows[-1]['term_stance_yaw_huber_actor_gradnorm']),
                          angular_grad=float(rows[-1]['term_stance_angular_slip_actor_gradnorm']))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=REFERENCE_ROOT)
    parser.add_argument('--cuda', action='store_true', help='追加三档真实 Newton 测试')
    parser.add_argument('--benchmark', action='store_true', help='完善已通过验收的 CUDA 性能对照')
    parser.add_argument('--save', action='store_true', help='保留汇总，启动器用于核对验收结果')
    opts = parser.parse_args()
    torch.set_num_threads(2)
    if opts.benchmark:
        path = OUTPUT/'preflight.json'
        result = json.loads(path.read_text())
        result['gpu'] = numeric(opts.root, 'cuda:0')
        result['benchmark'] = benchmark(opts.root)
        path.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
        print(json.dumps(result['benchmark'],ensure_ascii=False), flush=True)
        return
    result = dict(cpu=numeric(opts.root, 'cpu'))
    if opts.cuda:
        result['gpu'] = numeric(opts.root, 'cuda:0')
        result['smokes'] = [run_smoke(opts.root, size) for size in (8,32,256)]
    if opts.save:
        if not opts.cuda:
            parser.error('需要 CUDA 实测才生成正式验收汇总')
        (OUTPUT/'preflight.json').write_text(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
    print(json.dumps({key: value for key,value in result.items() if key != 'cpu'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
