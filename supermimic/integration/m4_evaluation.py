"""从 F2 起点重复评估冻结策略，并回放实测 Newton 状态。"""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from ..roban_runtime import M4_GUARDED_MAPPINGS


CONTACT_FORCE_N = 10.0
MIN_SWING_S = 0.10
MIN_SWING_LIFT_M = 0.02
TOUCHDOWN_WINDOW_S = 0.20
MIN_STANCE_S = 0.06
TOUCHDOWN_XY_ERROR_M = 0.10
STANDING_SETTLE_S = 0.20
STANDING_TILT_DEG = 20.0
STANDING_HEIGHT_ERROR_M = 0.12
STANDING_MIN_DOUBLE_SUPPORT = 0.60


@contextmanager
def read_only_pose(robot):
    """允许动作目标写入，禁止评估步内直接写入机器人状态。"""
    names = ('write_root_link_pose_to_sim_index', 'write_root_link_velocity_to_sim_index',
             'write_joint_position_to_sim_index', 'write_joint_velocity_to_sim_index')
    original = {name: getattr(robot, name) for name in names}

    def forbid(*args, **kwargs):
        raise RuntimeError('闭环评估禁止在 Reset 之外写入根部或关节状态')

    try:
        for name in names:
            setattr(robot, name, forbid)
        yield
    finally:
        for name, method in original.items():
            setattr(robot, name, method)


def _swing_and_touchdown(time, foot, contact, target, lift, landing, dt):
    """同一只脚先离地抬升，再在计划窗口内持续承重并接近目标。"""
    if len(time) < 2:
        return False, None
    swing_steps = max(1, int(np.ceil(MIN_SWING_S / dt - 1e-8)))
    stance_steps = max(1, int(np.ceil(MIN_STANCE_S / dt - 1e-8)))
    window = (time >= lift - TOUCHDOWN_WINDOW_S) & (time <= landing + TOUCHDOWN_WINDOW_S)
    transitions = np.diff(np.r_[False, window & ~contact, False].astype(np.int8))
    starts, stops = np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)
    started = False
    for start, stop in zip(starts, stops):
        if stop - start < swing_steps or start == 0 or not contact[start - 1]:
            continue
        if (foot[start:stop, 2] - foot[start - 1, 2]).max() < MIN_SWING_LIFT_M:
            continue
        started = True
        if stop + stance_steps > len(time) or abs(time[stop] - landing) > TOUCHDOWN_WINDOW_S:
            continue
        if contact[stop:stop + stance_steps].all() and np.linalg.norm(foot[stop, :2] - target[:2]) <= TOUCHDOWN_XY_ERROR_M:
            return True, float(time[stop])
    return started, None


def _summarize_trial(trace, reference, seed, reason, steps, dt, reward, max_reward):
    t, p, R, foot, contact = (trace[key] for key in ('time', 'p', 'R', 'foot', 'contact'))
    ref = reference.at_time(t)
    rp, rR, rf = (ref[key].cpu().numpy() for key in ('root_p', 'root_R', 'foot_p'))
    planned, valid = (ref[key].cpu().numpy() for key in ('contact', 'contact_valid'))
    # Reset 后没有实测接触力。接触和位姿统计统一使用积分后的样本。
    valid = valid[1:]
    eq = (contact[1:] == planned[1:])[valid]
    error_p = (p[1:] - rp[1:]) ** 2
    angle = np.rad2deg(np.arccos(np.clip(((R[1:] * rR[1:]).sum(axis=(1, 2)) - 1) / 2, -1, 1)))
    full = reason == 'reference_end'
    lifts = reference.liftoff_time.cpu().numpy()
    landings = reference.touchdown_time.cpu().numpy()
    sides = reference.touchdown_side.cpu().numpy()
    landing_targets = reference.at_time(landings)['foot_p'].cpu().numpy()
    events = []
    for i, (side, lift, landing) in enumerate(zip(sides, lifts, landings)):
        swing, actual = _swing_and_touchdown(
            t, foot[:, side], contact[:, side], landing_targets[i, side], lift, landing, dt)
        events.append(dict(foot=('left', 'right')[side], planned_lift_s=float(lift),
                           planned_touchdown_s=float(landing), swing_observed=swing,
                           actual_touchdown_s=actual))
    stand_samples = (t >= STANDING_SETTLE_S) & (t < lifts[0])
    standing = (t[-1] >= lifts[0] and (full or steps * dt > lifts[0])
                and (np.rad2deg(np.arccos(np.clip(R[stand_samples, 2, 2], -1, 1))) <= STANDING_TILT_DEG).all()
                and (np.abs(p[stand_samples, 2] - rp[stand_samples, 2]) <= STANDING_HEIGHT_ERROR_M).all()
                and contact[stand_samples].all(axis=-1).mean() >= STANDING_MIN_DOUBLE_SUPPORT)
    matched = [event['actual_touchdown_s'] is not None for event in events]
    completed_prefix = next((i for i, event in enumerate(matched) if not event), len(matched))
    xy = p[-1, :2] - p[0, :2]
    return dict(
        seed=seed, steps=steps, episode_length_s=steps * dt, termination=reason,
        survived_full_reference=full, stable_initial_standing=bool(standing),
        first_swing=events[0]['swing_observed'], first_touchdown=matched[0],
        matched_touchdowns=sum(matched), consecutive_planned_touchdowns=completed_prefix,
        root_xy_displacement_m=xy.tolist(), root_xy_displacement_norm_m=float(np.linalg.norm(xy)),
        final_root_xy_error_m=float(np.linalg.norm(p[-1, :2] - rp[-1, :2])),
        root_xy_rmse_m=float(np.sqrt(error_p[:, :2].sum(-1).mean())) if len(error_p) else None,
        root_position_rmse_m=float(np.sqrt(error_p.sum(-1).mean())) if len(error_p) else None,
        root_orientation_rmse_deg=float(np.sqrt((angle ** 2).mean())) if len(angle) else None,
        foot_position_rmse_m=float(np.sqrt(((foot[1:] - rf[1:]) ** 2).sum(-1).mean())) if len(angle) else None,
        contact_agreement=float(eq.mean()) if len(eq) else None,
        contact_valid_samples=int(valid.sum()),
        tracking_return=reward, normalized_return=reward / (steps * max_reward) if steps else None,
        full_clip_normalized_return=reward / (round(reference.duration / dt) * max_reward),
        footsteps=events,
    )


def _trial(env, policy, contacts, seed):
    import torch
    from ..control import bound_action
    from ..motion import CONTROL_DT
    from ..roban import FOOT_BODY_NAMES
    from ..rotations import quat_to_mat
    from .m4_task import M4EpaEvaluationError

    torch.manual_seed(seed)
    reference = env.motion
    env_ids = torch.zeros(1, dtype=torch.long, device=env.device)
    foot_ids = [env.robot.body_names.index(name) for name in FOOT_BODY_NAMES]
    history = None
    trace = {key: [] for key in ('time', 'p', 'R', 'q', 'foot', 'contact')}

    def record(step, state, contact):
        # 用仿真读取的足链路位姿和 F2 偏移获取实际足底参考点。
        pose = env.robot.data.body_link_pose_w.torch[0, foot_ids]
        sole_R = quat_to_mat(pose[:, [6, 3, 4, 5]])
        sole = (pose[:, :3] - env.scene.env_origins[0]
                + (sole_R @ reference.sole_offsets[..., None]).squeeze(-1))
        for key, value in dict(time=step * CONTROL_DT, p=state[0][0].cpu().numpy(),
                               R=state[1][0].cpu().numpy(), q=state[4][0].cpu().numpy(),
                               foot=sole.cpu().numpy(), contact=contact).items():
            trace[key].append(value)

    reward = 0.0
    reason = 'reference_end'
    n_steps = round(reference.duration / CONTROL_DT)
    with torch.no_grad():
        env.reset(env_ids)
        S = env.state()
        record(0, S, np.zeros(2, dtype=bool))
        with read_only_pose(env.robot):
            for step in range(1, n_steps + 1):
                o = env.goals.obs(env.mdp, S, env.a_prev, env.xi, env.f, env.command)
                history = (o[:, None].repeat(1, policy.hist, 1) if history is None
                           else torch.cat((history[:, 1:], o[:, None]), 1))
                goal = env.goals.goal(S, env.f, env.ep_step, env.command)
                action = policy(history, goal)
                if env.mdp.action_mapping not in M4_GUARDED_MAPPINGS:
                    action = bound_action(action)
                try:
                    out = env.step(action)
                except M4EpaEvaluationError as exc:
                    print(f'评估 seed={seed} 因 EPA 中止。{exc}', flush=True)
                    reason = 'epa_horizon'
                    step -= 1
                    break
                S = env.state()
                if not torch.isfinite(out['s1']).all() or not torch.isfinite(out['r']).all():
                    reason = 'nonfinite'
                    break
                _, force, _ = contacts.read()
                if not np.isfinite(force).all():
                    raise FloatingPointError('Newton 地面接触力包含非有限值')
                record(step, S, force[0, :, 2] >= CONTACT_FORCE_N)
                reward += float(out['r'][0])
                if bool(out['term'][0]):
                    flags = [('root_height', S[0][0, 2] < env.mdp.cfg.term_height),
                             ('root_tilt', 1 - S[1][0, 2, 2] > env.mdp.cfg.term_tilt)]
                    reason = '+'.join(name for name, flag in flags if flag) or 'physical_termination'
                    break
                if bool(out['timeout'][0]) and step < n_steps:
                    reason = 'early_timeout'
                    break
    states = {key: np.asarray(value) for key, value in trace.items()}
    report = _summarize_trial(states, reference, seed, reason, step, CONTROL_DT, reward,
                              env.mdp.max_step_reward)
    states.update(termination=reason, termination_time=step * CONTROL_DT, seed=seed)
    return report, states


def _mean(trials, key):
    values = [r[key] for r in trials if r[key] is not None]
    return float(np.mean(values)) if values else None


def _evaluate(env, policy, contacts, seeds):
    trials, video = [], None
    for seed in seeds:
        row, trace = _trial(env, policy, contacts, seed)
        trials.append(row)
        if video is None:
            video = trace
    aggregate = {k: _mean(trials, k) for k in (
        'episode_length_s', 'matched_touchdowns', 'consecutive_planned_touchdowns',
        'root_xy_displacement_norm_m', 'final_root_xy_error_m', 'root_xy_rmse_m',
        'root_position_rmse_m', 'root_orientation_rmse_deg', 'foot_position_rmse_m',
        'contact_agreement', 'tracking_return', 'normalized_return', 'full_clip_normalized_return')}
    aggregate.update(
        num_resets=len(trials), survival_rate=_mean(trials, 'survived_full_reference'),
        early_fall_rate=float(np.mean([r['termination'].startswith('root_') for r in trials])),
        epa_abort_rate=float(np.mean([r['termination'] == 'epa_horizon' for r in trials])),
        standing_rate=_mean(trials, 'stable_initial_standing'),
        first_swing_rate=_mean(trials, 'first_swing'), first_touchdown_rate=_mean(trials, 'first_touchdown'),
        four_touchdown_rate=float(np.mean([r['consecutive_planned_touchdowns'] == 4 for r in trials])),
        complete_tracking_rate=float(np.mean([
            r['survived_full_reference'] and r['consecutive_planned_touchdowns'] == 4 for r in trials])),
        root_xy_displacement_m=np.mean([r['root_xy_displacement_m'] for r in trials], axis=0).tolist())
    return dict(aggregate=aggregate, per_reset=trials), video


def _select_best(result):
    score = result['aggregate']
    foot_rmse = score['foot_position_rmse_m']
    return (score['complete_tracking_rate'], score['survival_rate'],
            score['consecutive_planned_touchdowns'], score['episode_length_s'],
            -foot_rmse if foot_rmse is not None else -np.inf,
            score['full_clip_normalized_return'], result['iteration'])


def _checkpoint_identity(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino


def _render(reference, video_dir, initial=None, trained=None, fps=25):
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    from .contact_diagnostics import RobotGeometry
    from ..rotations import mat_to_quat

    # 复用 M7 的 Roban MJCF 渲染几何。平面网格只用于观察全局位移。
    geom = RobotGeometry(SimpleNamespace(sizes=()))
    w, h = 544, 384
    model = geom.model
    model.vis.global_.offwidth, model.vis.global_.offheight = w, h
    model.vis.headlight.ambient[:] = .5
    model.vis.headlight.diffuse[:] = .7
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 132, -17, 2.70
    option = mujoco.MjvOption()
    option.geomgroup[0], option.geomgroup[1] = 0, 1
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:
        font = ImageFont.load_default()
    times = np.arange(round(reference.duration * fps) + 1) / fps
    ref = reference.at_time(times)
    ref = dict(time=times, p=ref['root_p'].cpu().numpy(), R=mat_to_quat(ref['root_R']).cpu().numpy(),
               q=ref['q'].cpu().numpy())
    center = (ref['p'][0] + ref['p'][-1]) / 2
    camera.lookat[:] = [center[0], center[1], .62]
    sequences = [('f2_reference', [('F2 reference | kinematic playback', ref)], times)]
    if initial is not None and trained is not None:
        import torch

        initial, trained = (dict(trace, R=mat_to_quat(torch.from_numpy(trace['R'])).numpy())
                            for trace in (initial, trained))
        sequences += [
            ('initial_policy', [('Initial policy | Newton', initial)], times[times <= initial['time'][-1]]),
            ('trained_policy', [('Trained policy | Newton', trained)], times[times <= trained['time'][-1]]),
            ('reference_vs_actual', [('F2 reference | kinematic playback', ref),
                                     ('Trained policy | Newton', trained)], times),
        ]
    video_dir.mkdir(parents=True, exist_ok=True)
    identity = np.eye(3).ravel()

    def add_box(renderer, size, pos, rgba):
        scene = renderer.scene
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_BOX,
                            np.array(size), np.array(pos), identity, np.array(rgba, dtype=np.float32))
        scene.ngeom += 1

    def panel(renderer, title, trace, t, final_frame):
        i = int(np.argmin(np.abs(trace['time'] - t)))
        geom.set_robot(trace['p'][i], trace['R'][i], trace['q'][i])
        renderer.update_scene(geom.data, camera=camera, scene_option=option)
        add_box(renderer, [1.5, 1.5, .005], [0., 0., -.006], [.20, .28, .34, 1])
        for v in np.linspace(-1.5, 1.5, 13):
            add_box(renderer, [1.5, .002, .001], [0., v, -.001], [.43, .53, .59, 1])
            add_box(renderer, [.002, 1.5, .001], [v, 0., -.001], [.43, .53, .59, 1])
        frame = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, w, 55), fill='#16232d')
        draw.text((12, 7), title, fill='#edf6fc', font=font)
        timestamp = f"t = {trace['time'][i]:.2f} s"
        if 'seed' in trace:
            timestamp += f"   |   Iteration {trace['iteration']}   |   Reset seed {trace['seed']}"
        draw.text((12, 31), timestamp, fill='#c6dce9', font=font)
        if 'termination' in trace and trace['termination'] != 'reference_end' and (
                final_frame or t >= trace['time'][-1] - 1e-8):
            draw.rectangle((0, h - 33, w, h), fill='#612c21')
            draw.text((12, h - 27), f"Ended at {trace['termination_time']:.2f} s: {trace['termination']}",
                      fill='white', font=font)
        return frame

    with mujoco.Renderer(model, height=h, width=w) as renderer:
        for name, panels, ts in sequences:
            width = w * len(panels)
            # 独立视频在最后一帧显示实际末态，保留奇数控制步结束的画面。
            if len(panels) == 1 and 'termination' in panels[0][1]:
                last_time = panels[0][1]['time'][-1]
                if last_time > ts[-1] + 1e-8:
                    ts = np.r_[ts, last_time]
            with TemporaryDirectory(dir=video_dir) as temp:
                dst = Path(temp) / f'{name}.mp4'
                with subprocess.Popen([
                    'ffmpeg', '-nostdin', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                    '-s', f'{width}x{h}', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
                    '-preset', 'fast', '-crf', '21', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(dst),
                ], stdin=subprocess.PIPE) as encoder:
                    for i, t in enumerate(ts):
                        frame = Image.new('RGB', (width, h))
                        for idx, (label, trace) in enumerate(panels):
                            frame.paste(panel(renderer, label, trace, t, i == len(ts) - 1), (idx * w, 0))
                        encoder.stdin.write(frame.tobytes())
                    encoder.stdin.close()
                    if encoder.wait():
                        raise RuntimeError(f'FFmpeg 编码失败: {name}')
                dst.replace(video_dir / dst.name)
            print(f'{video_dir / dst.name} ({len(ts)} frames, {fps} fps)', flush=True)


def main():
    os.environ['SUPERMIMIC_ROBOT'] = 'roban'
    # 确保首次导入 MuJoCo 前决定 GL 后端。
    os.environ.setdefault('MUJOCO_GL', 'egl')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m4_reference', type=Path)
    parser.add_argument('--out', type=Path, default=Path('outputs/m4_0'))
    parser.add_argument('--reference-only', action='store_true')
    parser.add_argument('--no_video', action='store_true', help='只保存独立评估指标')
    parser.add_argument('--latest_only', action='store_true', help='只评估最新与已有最佳检查点，跳过初始对照和渲染')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1001, 1002, 1003])
    if '--reference-only' not in sys.argv:
        from isaaclab.app import add_launcher_args, launch_simulation

        add_launcher_args(parser)
    args = parser.parse_args()

    import torch
    from ..kinematics import RobotKinematics
    from .m4_reference import F2MotionReference

    if args.reference_only:
        if args.m4_reference is None:
            parser.error('--reference-only 需要 --m4_reference')
        reference = F2MotionReference(args.m4_reference, RobotKinematics('cpu'), 'cpu')
        _render(reference, args.out / 'videos')
        return
    if len(set(args.seeds)) < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error('评估需要至少两个不同种子的独立 Reset')

    from ..networks import Policy
    from ..roban import assert_checkpoint
    from ..transfer import assert_policy_checkpoint
    from .m4_task import M4TrackingSim, build_m4_task, make_m4_sim

    names = ('last',) if args.latest_only else ('initial', 'early', 'last')
    anchor = names[0]
    paths = {name: args.out / 'checkpoints' / f'{name}.pt' for name in names}
    required = [name for name in dict.fromkeys((anchor, 'last')) if not paths[name].is_file()]
    if required:
        parser.error(f'缺少 M4 检查点: {", ".join(required)}')
    identities = {anchor: _checkpoint_identity(paths[anchor])}
    anchor_ck = torch.load(paths[anchor], map_location='cpu', weights_only=False)
    assert_checkpoint(anchor_ck)
    config = SimpleNamespace(**anchor_ck['args'])
    if not getattr(config, 'm4_reference', None):
        parser.error(f'{anchor}.pt 缺少 F2 参考数据配置')
    config.m4_reference = str(args.m4_reference or config.m4_reference)
    device = args.device
    device_type = torch.device(device).type
    if device_type != torch.device(config.device).type:
        parser.error('评估与训练必须选择同类 Newton 求解器设备')
    kin = RobotKinematics(device)
    reference, goals, mdp, _ = build_m4_task(config, kin, [config.m4_reference], device)
    best_path = args.out / 'checkpoints' / 'best.pt'
    if (mdp.action_mapping == 'standing_offset' and best_path.is_file()
            and not any(path.is_file() and path.samefile(best_path) for path in paths.values())):
        paths['best'] = best_path
    physics = mdp.physics_contract(config)
    sim = make_m4_sim(device, config)
    results, videos = {}, {}
    with launch_simulation(sim, args):
        env = M4TrackingSim(sim, 1, reference, mdp, eval_mode=True, reset_noise=False, goals=goals)
        contacts = env.ground_contacts()
        for name, path in paths.items():
            if not path.is_file():
                continue
            if name != anchor:
                identities[name] = _checkpoint_identity(path)
            ck = anchor_ck if name == anchor else torch.load(path, map_location='cpu', weights_only=False)
            assert_checkpoint(ck)
            goals.spec.assert_checkpoint(ck)
            mdp.assert_checkpoint_task(ck)
            if ck['physical_contract'] != physics:
                raise ValueError(f'评估环境与 checkpoint 物理条件不一致: {path}')
            if ck['data_contract'] != reference.validation:
                raise ValueError(f'评估参考数据与 checkpoint 记录不一致: {path}')
            a = ck['args']
            assert_policy_checkpoint(ck, a['hist'], a['obs_adapter'], not a['freeze_decoder'],
                                     obs_dim=mdp.obs_dim, observation_mode=mdp.observation_mode)
            policy = Policy(mdp.obs_dim, goals.spec.dim, hist=a['hist'], obs_adapter=a['obs_adapter'],
                            squash_action=mdp.action_mapping == 'joint_range').to(device)
            policy.load_state_dict(ck['pi'])
            policy.eval()
            result, videos[name] = _evaluate(env, policy, contacts, args.seeds)
            result.update(iteration=ck['iteration'], checkpoint=str(path))
            videos[name]['iteration'] = ck['iteration']
            results[name] = result
            print(f'{name}: {json.dumps(result["aggregate"], allow_nan=False)}', flush=True)

    # 含 EPA 无效轨迹的候选无法与完整评估比较，保留原有最佳检查点。
    eligible = (name for name, result in results.items()
                if name != 'initial' and result['aggregate']['epa_abort_rate'] == 0.)
    best = max(eligible, key=lambda name: _select_best(results[name]), default=None)
    if best is not None and best != 'best':
        with TemporaryDirectory(dir=best_path.parent) as temp:
            alias = Path(temp) / 'best.pt'
            alias.hardlink_to(paths[best])
            if _checkpoint_identity(alias) != identities[best]:
                raise RuntimeError(f'{paths[best]} 在评估期间已更新，现有最佳检查点仍可用')
            alias.replace(best_path)
    output = args.out / 'evaluation'
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(
        device=device, backend='Isaac Lab 3 Newton/' + ('MJWarp' if device_type == 'cuda' else 'MuJoCo CPU'),
        deterministic_unperturbed_resets=args.seeds, video_reset_seed=args.seeds[0],
        action_mapping=mdp.action_mapping,
        reference=str(config.m4_reference), reference_duration_s=reference.duration,
        best_candidate=best, best_checkpoint=str(best_path) if best_path.is_file() else None,
        contact=dict(source='Newton ground contact, final physics substep', force_threshold_n=CONTACT_FORCE_N,
                     planned_validity_only=True),
        steps=dict(minimum_swing_s=MIN_SWING_S, minimum_lift_m=MIN_SWING_LIFT_M,
                   touchdown_window_s=TOUCHDOWN_WINDOW_S, minimum_stance_s=MIN_STANCE_S,
                   touchdown_max_xy_error_m=TOUCHDOWN_XY_ERROR_M),
        standing=dict(settling_time_s=STANDING_SETTLE_S, max_tilt_deg=STANDING_TILT_DEG,
                      max_height_error_m=STANDING_HEIGHT_ERROR_M,
                      minimum_double_support_fraction=STANDING_MIN_DOUBLE_SUPPORT),
        results=results,
    )
    if not (args.latest_only or args.no_video):
        summary['videos'] = dict(
            initial_policy=dict(checkpoint=str(paths['initial']), iteration=results['initial']['iteration']),
            trained_policy_and_comparison=dict(checkpoint=str(paths['last']), iteration=results['last']['iteration']))
    with TemporaryDirectory(dir=output) as temp:
        summary_file = Path(temp) / 'results.json'
        summary_file.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        summary_file.replace(output / 'results.json')
    if not (args.latest_only or args.no_video):
        _render(reference, args.out / 'videos', videos['initial'], videos['last'])


if __name__ == '__main__':
    main()
