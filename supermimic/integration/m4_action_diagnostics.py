"""在相同 F2 起态实测 PD 动作映射，指标输出到终端。"""
import json

import numpy as np
import torch

from ..control import ACTION_LIMIT, bound_action
from ..mdp import unflatten_state
from ..motion import CONTROL_DT
from ..roban import JOINT_NAMES, assert_checkpoint
from ..roban_runtime import M4_GUARDED_MAPPINGS
from ..rotations import mat_to_quat
from .m4_evaluation import CONTACT_FORCE_N, read_only_pose


def _load_actor(env, path):
    from ..networks import Policy
    from ..transfer import assert_policy_checkpoint

    ck = torch.load(path, map_location='cpu', weights_only=False)
    assert_checkpoint(ck)
    env.goals.spec.assert_checkpoint(ck)
    mdp = env.mdp
    joint_range = mdp.action_mapping == 'joint_range'
    if ck['data_contract'] != env.motion.validation:
        raise ValueError('动作对照需要同一 F2 参考的检查点')
    if mdp.action_mapping in M4_GUARDED_MAPPINGS:
        mdp.assert_checkpoint_task(ck)
    elif ck['task_contract'].get('action_mapping', 'neutral') != 'neutral':
        raise ValueError('中立与残差映射对照需要中立映射检查点')
    a = ck['args']
    assert_policy_checkpoint(ck, a['hist'], a['obs_adapter'], not a['freeze_decoder'],
                             obs_dim=mdp.obs_dim, observation_mode=mdp.observation_mode)
    policy = Policy(mdp.obs_dim, env.goals.spec.dim, hist=a['hist'], obs_adapter=a['obs_adapter'],
                    squash_action=joint_range).to(env.device)
    policy.load_state_dict(ck['pi'])
    print(f"冻结 Actor 迭代 {ck['iteration']}，当前测试映射 {mdp.action_mapping}", flush=True)
    return policy.eval()


def _verify_residual(env):
    mdp = env.mdp
    q = env.motion.q
    frames = torch.arange(len(q), device=env.device)
    for sign in (-1, 1):
        raw = torch.full_like(q, sign * 2 * ACTION_LIMIT)
        target = (q + sign * ACTION_LIMIT * mdp.alpha).clamp(mdp.kin.q_lo, mdp.kin.q_hi)
        torch.testing.assert_close(mdp.residual_pd_target(raw, frames), target)
        torch.testing.assert_close(mdp.wm_action(raw, frames) * mdp.alpha + mdp.default_q, target)
    with torch.enable_grad():
        a = torch.zeros_like(q[:1], requires_grad=True)
        grad, = torch.autograd.grad(mdp.wm_action(a, frames[:1]).sum(), a)
        torch.testing.assert_close(grad, torch.ones_like(grad))


def _tilt_deg(R):
    return float(torch.rad2deg(torch.acos(R[2, 2].clamp(-1, 1))))


@torch.no_grad()
def run_action_diagnostics(env, contacts, torque_w, dofs, steps, checkpoint):
    mdp = env.mdp
    residual = mdp.action_mapping == 'reference_residual'
    joint_range = mdp.action_mapping == 'joint_range'
    standing = mdp.action_mapping == 'standing_offset'
    if residual:
        _verify_residual(env)
    policy = _load_actor(env, checkpoint) if checkpoint else None
    ids = torch.arange(env.N, device=env.device)
    initial = env.state()
    print(json.dumps(dict(mapping=mdp.action_mapping, envs=env.N, root_xyz_m=initial[0][0].tolist(),
                          root_wxyz=mat_to_quat(initial[1][:1])[0].tolist(),
                          planned_contacts=env.motion.contact[0].tolist()), allow_nan=False), flush=True)
    cases = [('zero', torch.zeros_like(env.a_prev)),
             ('alternating_0.25', (.25 - .5 * (torch.arange(len(JOINT_NAMES), device=env.device) % 2))
              .expand_as(env.a_prev))]
    if policy is not None:
        cases.append(('frozen_actor', None))

    for name, fixed_action in cases:
        env.reset(ids)
        S = env.state()
        for actual, expected in zip(S, initial):
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
        history = None
        saturated = samples = contact_samples = double_support_samples = 0
        peak = 0.
        first_double_support = None
        lengths = torch.zeros_like(ids)
        with read_only_pose(env.robot):
            for tick in range(steps):
                active = ~env.done_eval
                lengths += active
                action = fixed_action
                if action is None:
                    obs = env.goals.obs(mdp, S, env.a_prev, env.xi, env.f, env.command)
                    history = (obs[:, None].expand(-1, policy.hist, -1) if history is None
                               else torch.cat((history[:, 1:], obs[:, None]), dim=1))
                    goal = env.goals.goal(S, env.f, env.ep_step, env.command)
                    action = policy(history, goal)
                    if mdp.action_mapping not in M4_GUARDED_MAPPINGS:
                        action = bound_action(action)

                if joint_range:
                    lo, hi = mdp.kin.q_lo, mdp.kin.q_hi
                    expected = (lo + hi) * .5 + (hi - lo) * .5 * action
                else:
                    center = env.motion.q[env.f] if residual else env.default_q
                    expected = center + env.alpha * action
                    if residual or standing:
                        expected = expected.clamp(mdp.kin.q_lo, mdp.kin.q_hi)
                error = expected - S[4]
                out = env.step(action)
                target = env.robot.data.joint_pos_target.torch[:, env.canonical_to_sim]
                torch.testing.assert_close(target, expected)
                torch.testing.assert_close(mdp._reward_pd_target(out['a'], out['f'] + 1), target)
                measured = torque_w[:, dofs].abs()
                _, forces, _ = contacts.read()
                finite = out['s1'].isfinite().all() & out['r'].isfinite().all() & measured.isfinite().all()
                if not finite or not np.isfinite(forces).all():
                    raise FloatingPointError('动作对照出现非有限状态、奖励或力矩')
                ratio = measured[active] / mdp.effort
                if (ratio > 1.001).any():
                    raise AssertionError('Newton 实测执行器力矩超出配置限幅')
                saturated += int((ratio >= .98).sum())
                samples += ratio.numel()
                peak = max(peak, float(measured[active].max()))
                force_z = forces[:, :, 2]
                both = (force_z >= CONTACT_FORCE_N).all(axis=1)
                active_cpu = active.cpu().numpy()
                double_support_samples += int(both[active_cpu].sum())
                contact_samples += int(active_cpu.sum())
                if first_double_support is None and both[0] and active_cpu[0]:
                    first_double_support = (tick + 1) * CONTROL_DT
                if tick == 0:
                    tau = (mdp.kp * error - mdp.kd * S[5]).abs()
                    first = dict(
                        target_rms_rad=float(error[0].square().mean().sqrt()),
                        target_peak_rad=float(error[0].abs().max()),
                        target_peak_joint=JOINT_NAMES[int(error[0].abs().argmax())],
                        theory_peak_Nm=float(tau[0].max()),
                        theory_over_limit_joints=int((tau[0] > mdp.effort).sum()),
                        measured_first_peak_Nm=float(measured[0].max()),
                        measured_first_near_limit_joints=int((ratio[0] >= .98).sum()),
                        first_foot_force_z_N=force_z[0].tolist(),
                    )
                S = env.state()
                if env.done_eval.all():
                    break
        replay = mdp.reward(unflatten_state(out['s1']), out['f'] + 1, out['a'], out['a_prev'],
                            command=out['command'], ep_step=out['ep_step'] + 1)
        torch.testing.assert_close(replay, out['r'])
        print(json.dumps(dict(mapping=mdp.action_mapping, case=name, **first,
                              survival_s=(lengths.to(torch.float64) * CONTROL_DT).tolist(),
                              fallen=env.failed_eval.tolist(), env0_final_root_z_m=float(S[0][0, 2]),
                              env0_final_tilt_deg=_tilt_deg(S[1][0]),
                              env0_first_double_support_s=first_double_support,
                              double_support_fraction=double_support_samples / contact_samples,
                              measured_rollout_peak_Nm=peak, measured_near_limit_fraction=saturated / samples),
                         allow_nan=False), flush=True)
