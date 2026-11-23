"""以冻结 Roban Actor 运行 Newton，并记录可对齐的 F2 诊断轨迹。"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

if __name__ == '__main__':
    os.environ['SUPERMIMIC_ROBOT'] = 'roban'
    os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import torch

from .m4_evaluation import CONTACT_FORCE_N, _summarize_trial, read_only_pose
from ..mdp import flatten_state, unflatten_state
from ..motion import CONTROL_DT
from ..roban import FOOT_BODY_NAMES, JOINT_NAMES
from ..rotations import quat_to_mat


class EffectiveFriction:
    """从末子步约束读取运动方向两个切向摩擦系数。"""

    def __init__(self, env):
        import warp as wp

        solver = env.physics._solver
        data = solver.mjw_data
        model = solver.mj_model
        self.geom = wp.to_torch(data.contact.geom)
        self.friction = wp.to_torch(data.contact.friction)
        self.address = wp.to_torch(data.contact.efc_address)[:, 0]
        self.count = wp.to_torch(data.nacon)
        self.ground_ids = torch.as_tensor(np.flatnonzero(model.geom_bodyid == 0), device=env.device)

    def read(self):
        size = min(int(self.count[0]), len(self.geom))
        ground = torch.isin(self.geom[:size], self.ground_ids).any(-1)
        values = self.friction[:size, :2][ground & (self.address[:size] >= 0)]
        return (values.amin(0).cpu().numpy() if len(values) else np.full(2, np.nan, np.float32),
                values.amax(0).cpu().numpy() if len(values) else np.full(2, np.nan, np.float32))


@torch.no_grad()
def collect_trial(env, policy, contacts, seed, friction=None, *, solver=None, capture_faults=False,
                  fixed_actions=None, reset_hook=None):
    """状态和接触含 Reset 帧，动作及奖励按其下发的控制步编号。"""
    from .m4_task import M4EpaEvaluationError

    torch.manual_seed(seed)
    env.reset(torch.zeros(1, dtype=torch.long, device=env.device))
    if reset_hook is not None:
        reset_hook(env)
    foot_ids = [env.robot.body_names.index(name) for name in FOOT_BODY_NAMES]
    rows = defaultdict(list)
    motion, mdp = env.motion, env.mdp
    anchor = {f'command_{name}': value[0].cpu().numpy() for name, value in env.command.tensors().items()}
    term_names = ()
    history = None
    reason = 'reference_end'
    fault = {}
    state = env.state()
    steps = motion.num_frames - 1
    if fixed_actions is not None:
        fixed_actions = torch.as_tensor(fixed_actions, device=env.device, dtype=state[4].dtype)
        if fixed_actions.ndim != 2 or fixed_actions.shape[1] != len(JOINT_NAMES) or len(fixed_actions) > steps:
            raise ValueError('固定动作须与 F2 控制步和 Roban 关节顺序一致')

    def record(**values):
        for key, value in values.items():
            rows[key].append(value.detach().to('cpu', copy=True).numpy() if isinstance(value, torch.Tensor) else value)

    def capture_state(state, force=None, count=None, other=None):
        pose = env.robot.data.body_link_pose_w.torch[0, foot_ids]
        rotation = quat_to_mat(pose[:, [6, 3, 4, 5]])
        velocity = env.robot.data.body_link_vel_w.torch[0, foot_ids]
        offset = (rotation @ motion.sole_offsets[..., None]).squeeze(-1)
        position = pose[:, :3] - env.scene.env_origins[0] + offset
        force = np.zeros((2, 3), dtype=np.float32) if force is None else force.astype(np.float32)
        record(state=flatten_state(state)[0], frame=env.f[0], xi=env.xi[0],
               foot=position, foot_R=rotation, foot_angular_velocity=velocity[:, 3:],
               foot_velocity=velocity[:, :3] + torch.cross(velocity[:, 3:], offset, dim=-1),
               force=force, contact=force[:, 2] >= CONTACT_FORCE_N,
               contact_measured=count is not None,
               foot_contact_count=np.zeros(2, dtype=np.int64) if count is None else count,
               nonfoot_ground_count=0 if other is None else other)
        if friction is not None:
            low, high = friction.read() if count is not None else (np.full(2, np.nan, np.float32),) * 2
            record(contact_friction_min=low, contact_friction_max=high)
        if solver is not None:
            # 保存下一控制步用到的 MJWarp 暖启动以及未转换的根四元数，供 Newton 扰动配对复位。
            record(solver_warmstart=solver.warmstart[0], solver_qpos=solver.qpos[0], solver_qvel=solver.qvel[0],
                   actuator_torque=solver.torque[0, solver.dofs],
                   root_quat_xyzw=env.robot.data.root_link_pose_w.torch[0, 3:7])

    capture_state(state)
    with read_only_pose(env.robot):
        for index in range(steps if fixed_actions is None else len(fixed_actions)):
            obs = env.goals.obs(mdp, state, env.a_prev, env.xi, env.f, env.command)
            goal = env.goals.goal(state, env.f, env.ep_step, env.command)
            if fixed_actions is None:
                history = (obs[:, None].expand(-1, policy.hist, -1) if history is None
                           else torch.cat((history[:, 1:], obs[:, None]), dim=1))
                action, raw = policy(history, goal, return_logits=True)
            else:
                # 机制对照用原策略已执行的线性动作，忽略重放状态造成的闭环分叉。
                action = raw = fixed_actions[index:index + 1]
            target = env.pd_target(action)
            prev = env.a_prev.clone()
            try:
                out = env.step(action)
                state = env.state()
                count, force, other = contacts.read()
                if not np.isfinite(force).all() or not bool(out['r'].isfinite().all()):
                    raise FloatingPointError('Newton 足部接触力或奖励含非有限值')
            except FloatingPointError as exc:
                recovered = isinstance(exc, M4EpaEvaluationError)
                if not recovered and not capture_faults:
                    raise
                reason = 'epa_horizon' if recovered else 'fatal_dynamics'
                if capture_faults:
                    # 严重故障的样本单独保存，不进入有效转移或指标分母。
                    fault = dict(fault_detail=np.asarray(str(exc)), fault_state=flatten_state(env.state())[0].cpu().numpy(),
                                 fault_action=action[0].cpu().numpy(), fault_target=target[0].cpu().numpy())
                    if solver is not None:
                        fault.update(fault_solver_qpos=solver.qpos[0].cpu().numpy(),
                                     fault_solver_qvel=solver.qvel[0].cpu().numpy(),
                                     fault_solver_overflow=solver.overflow[0].cpu().numpy())
                    print(f'独立评估 seed={seed} 因 {reason} 停止。{exc}', flush=True)
                break
            term_names = tuple(k for k in out['terms'] if k.endswith('_contribution'))
            terms = torch.stack([out['terms'][key][0] for key in term_names])
            torch.testing.assert_close(terms.sum(), out['r'][0])
            record(observation=obs[0], goal=goal[0], action=action[0], raw_action=raw[0],
                   previous_action=prev[0], target=target[0], clamp_mask=mdp.target_clip_mask(action)[0],
                   reward=out['r'][0], reward_terms=terms, term=out['term'][0], timeout=out['timeout'][0])
            capture_state(state, force[0], count[0], other[0])
            if out['term'][0]:
                low = bool(state[0][0, 2] < mdp.cfg.term_height)
                tilt = bool(1 - state[1][0, 2, 2] > mdp.cfg.term_tilt)
                reason = '+'.join(name for name, flag in (('root_height', low), ('root_tilt', tilt)) if flag) or 'physical_termination'
                break
            if out['timeout'][0]:
                reason = 'reference_end' if int(env.f[0]) == steps else 'early_timeout'
                break
        else:
            if fixed_actions is not None and len(fixed_actions) < steps:
                reason = 'action_sequence_end'

    trace = {key: np.asarray(value) for key, value in rows.items()}
    if not term_names:
        # 首步遇到 EPA 的试次也保留统一的时钟和空转移字段，供批量诊断跳过。
        trace.update(observation=np.empty((0, mdp.obs_dim), np.float32),
                     goal=np.empty((0, env.goals.spec.dim), np.float32),
                     clamp_mask=np.empty((0, len(JOINT_NAMES)), bool),
                     reward=np.empty(0, np.float32), reward_terms=np.empty((0, 0), np.float32),
                     term=np.empty(0, bool), timeout=np.empty(0, bool))
        for key in ('action', 'raw_action', 'previous_action', 'target'):
            trace[key] = np.empty((0, len(JOINT_NAMES)), np.float32)
    trace.update(anchor, **fault)
    trace['termination'], trace['seed'] = np.asarray(reason), np.asarray(seed)
    trace['time'] = trace['frame'].astype(np.float64) * CONTROL_DT
    frames = torch.as_tensor(trace['frame'], device=env.device)
    # 契约中所有目标都由同一个步后时钟给出。
    reference = motion.sample(frames[None], env.command)
    for key in ('root_p', 'root_R', 'foot_p', 'foot_R', 'foot_v', 'contact', 'contact_valid', 'phase'):
        trace['planned_' + key] = reference[key][0].cpu().numpy()
    trace['foot_target_error'] = np.linalg.norm(trace['foot'] - trace['planned_foot_p'], axis=-1)
    trace['reward_term_names'] = np.asarray([key.removesuffix('_contribution') for key in term_names])
    trace['max_step_reward'] = np.asarray(mdp.max_step_reward)
    return trace


def summarize(trace, motion):
    state = unflatten_state(torch.from_numpy(trace['state']))
    t = trace['time']
    summary = _summarize_trial(dict(time=t, p=state[0].numpy(), R=state[1].numpy(),
                                    foot=trace['foot'], contact=trace['contact']),
                               motion, int(trace['seed']), str(trace['termination']), len(t) - 1,
                               CONTROL_DT, float(trace.get('reward', np.empty(0)).sum()),
                               float(trace['max_step_reward']))
    if len(t) == 1:
        return summary
    summary['clamp_ratio'] = float(trace['clamp_mask'].mean())
    summary['nonfoot_ground_frames'] = int((trace['nonfoot_ground_count'] > 0).sum())
    summary['reward_means'] = dict(zip(trace['reward_term_names'], trace['reward_terms'].mean(0).tolist()))
    return summary


def render_comparison(recording, reference, output, *, fps=25, chapters=None):
    """用控制时钟拼接各权重的参考与 Newton 实测视频。"""
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    from .contact_diagnostics import RobotGeometry
    from ..kinematics import RobotKinematics
    from .m4_reference import F2MotionReference

    motion = F2MotionReference(reference, RobotKinematics('cpu'), 'cpu')
    geom = RobotGeometry(SimpleNamespace(sizes=()))
    width, height = 624, 464
    model = geom.model
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    model.vis.headlight.ambient[:] = .53
    model.vis.headlight.diffuse[:] = .72
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation = 135, -22
    option = mujoco.MjvOption()
    option.geomgroup[0], option.geomgroup[1] = 0, 1
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 14)
        title_font = ImageFont.truetype('DejaVuSans-Bold.ttf', 17)
    except OSError:
        font = title_font = ImageFont.load_default()
    colors = ((.16, .85, .94, .95), (1., .57, .18, .95))
    routes = motion.at_time(motion.touchdown_time)['foot_p'].numpy()
    targets = routes[np.arange(len(routes)), motion.touchdown_side.numpy()]
    touch_side = motion.touchdown_side.numpy()
    time = np.arange(motion.num_frames) * CONTROL_DT
    ref = motion.at_time(time)
    reference_trace = dict(time=time, p=ref['root_p'].numpy(), quat=ref['root_quat'].numpy(),
                           q=ref['q'].numpy(), R=ref['root_R'].numpy(), foot=ref['foot_p'].numpy(),
                           contact=ref['contact'].numpy(), phase=ref['phase'].numpy())
    frames = np.rint(np.arange(round(motion.duration * fps) + 1) / fps / CONTROL_DT).astype(int)
    frames = np.minimum(frames, motion.num_frames - 1)

    def add(renderer, kind, size, position, rgba):
        scene = renderer.scene
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], kind, np.asarray(size, dtype=np.float64),
                            np.asarray(position, dtype=np.float64), np.eye(3).ravel(),
                            np.asarray(rgba, dtype=np.float32))
        scene.ngeom += 1

    def panel(renderer, label, trace, frame, requested_time, ref_frame=None):
        i = min(frame, len(trace['time']) - 1)
        geom.set_robot(trace['p'][i], trace['quat'][i], trace['q'][i])
        renderer.update_scene(geom.data, camera=camera, scene_option=option)
        box, sphere, cylinder = (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_SPHERE,
                                 mujoco.mjtGeom.mjGEOM_CYLINDER)
        add(renderer, box, [3.0, 3.0, .005], [camera.lookat[0], camera.lookat[1], -.007], [.19, .25, .30, 1.])
        for value in np.arange(-2.5, 2.6, .25):
            add(renderer, box, [2.5, .001, .001], [camera.lookat[0], camera.lookat[1] + value, -.001], [.31, .39, .44, 1.])
            add(renderer, box, [.001, 2.5, .001], [camera.lookat[0] + value, camera.lookat[1], -.001], [.31, .39, .44, 1.])
        for point, side in zip(targets, touch_side):
            add(renderer, cylinder, [.037, .006, .006], [*point[:2], .006], (*colors[side][:3], .54))
        for point, color in zip(trace['foot'][i], colors):
            add(renderer, sphere, [.023, .023, .023], point + np.array([0, 0, .02]), color)
        if chapters is not None:
            # 最近 0.6 秒的足底轨迹与当前实心标记分开绘制。
            for old in range(max(i - 30, 0), i, 3):
                for point, color in zip(trace['foot'][old], colors):
                    add(renderer, sphere, [.009, .009, .009], point + np.array([0, 0, .01]), (*color[:3], .55))
        image = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, width, 63), fill='#111f2c')
        draw.text((14, 8), label, fill='#eef7fc', font=title_font)
        draw.text((14, 36), f"Episode t {trace['time'][i]:4.2f} s    |    Reference t {requested_time:4.2f} s",
                  fill='#d0e2eb', font=font)
        draw.rectangle((0, height - 104, width, height), fill='#111f2c')
        contact = trace['contact'][i]
        phase = reference_trace['phase'][frame]
        for side, side_name in enumerate(('L', 'R')):
            x = 14 + side * 305
            side_color = tuple(int(255 * c) for c in colors[side][:3])
            status = 'stance' if contact[side] else 'off'
            if ref_frame is not None:
                status = f"{status} ({trace['force'][i, side, 2]:.0f} N)" if i else 'unmeasured'
            draw.text((x, height - 95), f'{side_name}  {status}    |    Plan swing {phase[side]:.2f}',
                      fill=side_color, font=font)
            xyz = trace['foot'][i, side]
            draw.text((x, height - 71), f'Sole m  {xyz[0]:+.2f}  {xyz[1]:+.2f}  {xyz[2]:+.2f}',
                      fill='#d0e2eb', font=font)
        if ref_frame is None:
            bottom = 'Discs mark the four planned footholds. Spheres show sole positions.'
        else:
            err = trace['p'][i] - reference_trace['p'][ref_frame]
            R, R_ref = trace['R'][i], reference_trace['R'][ref_frame]
            heading = np.arctan2(R[1, 0], R[0, 0]) - np.arctan2(R_ref[1, 0], R_ref[0, 0])
            heading = np.rad2deg(np.arctan2(np.sin(heading), np.cos(heading)))
            bottom = f'Root error {np.linalg.norm(err):.2f} m   |   Heading {heading:+.1f} deg'
            if trace['termination'] != 'reference_end' and i == len(trace['time']) - 1:
                bottom = f"Ended at {trace['time'][i]:.2f} s ({trace['termination']})"
        draw.text((14, height - 43), bottom, fill='#eef7fc', font=font)
        return image

    output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(recording) as data, mujoco.Renderer(model, height=height, width=width) as renderer:
        seed = int(data['seeds'][0])
        with TemporaryDirectory(dir=output.parent) as temp:
            path = Path(temp) / output.name
            cmd = ['ffmpeg', '-nostdin', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                   '-s', f'{2 * width}x{height}', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
                   '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)]
            with subprocess.Popen(cmd, stdin=subprocess.PIPE) as encoder:
                sections = chapters if chapters is not None else [
                    (f'i{iteration}_s{seed}_', f'Newton policy  |  iteration {iteration}  |  seed {seed}')
                    for iteration in data['iterations']]
                for prefix, label in sections:
                    trace = {key: data[prefix + key] for key in ('time', 'foot', 'force', 'contact', 'termination')}
                    state = data[prefix + 'state']
                    trace.update(p=state[:, :3], quat=state[:, 3:7], q=state[:, 13:13 + len(JOINT_NAMES)],
                                 R=quat_to_mat(torch.from_numpy(state[:, 3:7])).numpy())
                    xy = np.vstack((reference_trace['p'], trace['p']))[:, :2]
                    camera.lookat[:] = [*((xy.min(0) + xy.max(0)) / 2), .52]
                    camera.distance = 2.45 + .55 * np.linalg.norm(np.ptp(xy, axis=0))
                    for frame in frames:
                        canvas = Image.new('RGB', (2 * width, height))
                        t = time[frame]
                        canvas.paste(panel(renderer, 'F2 reference  |  planned contact', reference_trace, frame, t), (0, 0))
                        canvas.paste(panel(renderer, label, trace, frame, t, ref_frame=frame), (width, 0))
                        encoder.stdin.write(canvas.tobytes())
                encoder.stdin.close()
                if encoder.wait():
                    raise RuntimeError('FFmpeg 编码失败')
            path.replace(output)
    print(f'对比视频已保存至 {output}', flush=True)


def render_main():
    parser = argparse.ArgumentParser(description='从冻结轨迹渲染参考与实测对比视频')
    parser.add_argument('--render-only', action='store_true')
    parser.add_argument('--m4_reference', type=Path, required=True)
    parser.add_argument('--input', type=Path, default=Path('outputs/m4_2/m4_2_v3/rollouts.npz'))
    parser.add_argument('--output', type=Path, default=Path('outputs/m4_2/m4_2_v3/reference_vs_newton.mp4'))
    args = parser.parse_args()
    render_comparison(args.input, args.m4_reference, args.output)


def main():
    from isaaclab.app import add_launcher_args, launch_simulation
    from .m4_task import M4TrackingSim, build_m4_task, make_m4_sim
    from ..kinematics import RobotKinematics
    from ..networks import Policy
    from ..roban import assert_checkpoint
    from ..transfer import assert_policy_checkpoint

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m4_reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/m4_2/m4_2_v3/rollouts.npz'))
    parser.add_argument('--checkpoints', nargs='+', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[1001, 1002, 1003])
    add_launcher_args(parser)
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        parser.error('请使用不同的种子')
    first = torch.load(args.checkpoints[0], map_location='cpu', weights_only=False)
    config = SimpleNamespace(**first['args'])
    device_type = torch.device(args.device).type
    if device_type != torch.device(config.device).type:
        parser.error('冻结评估必须选择与训练相同设备类型的 Newton 求解器')
    ref_path = str(args.m4_reference.resolve())
    motion, goals, mdp, _ = build_m4_task(config, RobotKinematics(args.device), [ref_path], args.device)
    if mdp.action_mapping != 'standing_offset':
        parser.error('此评估需要 M4.2 的 standing_offset 动作映射')
    sim = make_m4_sim(args.device, config)
    output = dict(seeds=args.seeds, dt=CONTROL_DT, joint_names=JOINT_NAMES,
                  foot_names=FOOT_BODY_NAMES,
                  planned_liftoff_time=motion.liftoff_time.cpu().numpy(),
                  planned_touchdown_time=motion.touchdown_time.cpu().numpy(),
                  planned_touchdown_side=motion.touchdown_side.cpu().numpy())
    iterations = []
    with launch_simulation(sim, args):
        env = M4TrackingSim(sim, 1, motion, mdp, gamma=config.gamma, eval_mode=True, reset_noise=False, goals=goals)
        contacts = env.ground_contacts()
        friction = EffectiveFriction(env) if device_type == 'cuda' else None
        for index, path in enumerate(args.checkpoints):
            ck = first if index == 0 else torch.load(path, map_location='cpu', weights_only=False)
            assert_checkpoint(ck)
            goals.spec.assert_checkpoint(ck)
            mdp.assert_checkpoint_task(ck)
            if ck['data_contract'] != motion.validation or ck['physical_contract'] != mdp.physics_contract(config):
                raise ValueError(f'检查点的物理或参考契约与当前评估不同：{path}')
            hist, adapter = ck['args']['hist'], ck['args']['obs_adapter']
            assert_policy_checkpoint(ck, hist, adapter, not ck['args']['freeze_decoder'],
                                     obs_dim=mdp.obs_dim, observation_mode=mdp.observation_mode)
            policy = Policy(mdp.obs_dim, goals.spec.dim, hist=hist, obs_adapter=adapter).to(args.device)
            policy.load_state_dict(ck['pi'])
            policy.eval()
            iteration = int(ck['iteration'])
            if iteration in iterations:
                raise ValueError(f'重复的检查点迭代：{iteration}')
            iterations.append(iteration)
            for seed in args.seeds:
                trace = collect_trial(env, policy, contacts, seed, friction)
                output.update({f'i{iteration}_s{seed}_{key}': value for key, value in trace.items()})
                print(json.dumps(dict(iteration=iteration, **summarize(trace, motion)), ensure_ascii=False, allow_nan=False),
                      flush=True)
    output.update(iterations=iterations, checkpoints=np.asarray([str(p) for p in args.checkpoints]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temp:
        target = Path(temp) / args.output.name
        np.savez_compressed(target, **output)
        target.replace(args.output)
    print(f'Newton 轨迹已保存至 {args.output}', flush=True)


if __name__ == '__main__':
    render_main() if '--render-only' in sys.argv else main()
