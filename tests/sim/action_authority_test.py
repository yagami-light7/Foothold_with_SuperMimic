"""Batched, simulation-only joint step tests; saves raw traces and measured gains."""

import sys
from pathlib import Path

# 支持直接运行测试文件，无需额外设置PYTHONPATH。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("reports/preflight_2026-10-29/authority"))
parser.add_argument("--offsets", type=float, nargs="+", default=[0.05, 0.1, 0.25])
parser.add_argument("--hold_steps", type=int, default=25)
parser.add_argument("--all_joints", action="store_true", help="Also test every canonical DoF at +/- 0.05 rad")
add_launcher_args(parser)
parser.set_defaults(device="cuda:0")
args = parser.parse_args()

import torch
import warp as wp

from isaaclab.utils import to_dict
from isaaclab_newton.physics import NewtonManager
from tasks.foot_placement.foot_placement_env import S54FootPlacementEnv
from tasks.foot_placement.foot_placement_env_cfg import S54FootPlacementEnvCfg
from tasks.foot_placement.spec import JOINT_NAMES


def main():
    if not args.offsets or min(args.offsets) <= 0 or max(args.offsets) > 0.3 or args.hold_steps < 1:
        raise ValueError("Use positive single-joint offsets <= 0.3 rad and hold_steps >= 1")
    cases = [("zero", {})]
    for side in ("l", "r"):
        for joint in (2, 3, 4, 5, 6):
            for offset in args.offsets:
                for sign in (-1, 1):
                    cases.append((f"{side}{joint}_{sign * offset:+.2f}", {f"leg_{side}{joint}_joint": sign * offset}))
        for knee in (0.1, 0.2, 0.4, 0.6):
            cases.append((f"{side}_flex_{knee:.2f}", {
                f"leg_{side}3_joint": -knee / 2,
                f"leg_{side}4_joint": knee,
                f"leg_{side}5_joint": -knee / 2,
            }))
    if args.all_joints:
        for name in JOINT_NAMES:
            for sign in (-1, 1):
                cases.append((f"all_{name}_{sign:+d}", {name: sign * 0.05}))
    cfg = S54FootPlacementEnvCfg()
    cfg.scene.num_envs = len(cases)
    cfg.sim.device = args.device
    # Diagnostic range only: per-case commanded offsets remain explicit below.
    cfg.action_scale = 0.6
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "env_cfg.json").write_text(json.dumps(to_dict(cfg), default=str, indent=2))
    with launch_simulation(cfg.sim, args):
        env = S54FootPlacementEnv(cfg)
        try:
            env.reset()
            joint_ids = env.joint_ids
            assert [env.robot.joint_names[i] for i in joint_ids.tolist()] == list(JOINT_NAMES)
            offsets = torch.zeros(env.num_envs, 27, device=env.device)
            for i, (_, command) in enumerate(cases):
                for name, offset in command.items():
                    offsets[i, JOINT_NAMES.index(name)] = offset
            actions = offsets / cfg.action_scale
            zeros = torch.zeros_like(actions)
            for _ in range(15):
                env.step(zeros)
            q0 = env.robot.data.joint_pos.torch[:, joint_ids].clone()
            feet0 = torch.stack((env._get_foot_poses()[0], env._get_foot_poses()[2]), dim=1).clone()
            defaults = env.robot.data.default_joint_pos.torch[:, joint_ids]
            limits = env.robot.data.soft_joint_pos_limits.torch[:, joint_ids]
            target = (defaults + offsets).clamp(limits[..., 0], limits[..., 1])
            active = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
            fell = torch.zeros_like(active)
            timed_out = torch.zeros_like(active)
            peak_dq = torch.zeros_like(q0)
            peak_foot = torch.zeros(env.num_envs, 2, 3, device=env.device)
            min_fz = torch.full((env.num_envs, 2), float("inf"), device=env.device)
            peak_effort = torch.zeros_like(q0)
            solver = getattr(NewtonManager, "_solver", None)
            mj_model = getattr(solver, "mj_model", None)
            mj_data = getattr(solver, "mjw_data", None)
            actual_force = getattr(mj_data, "qfrc_actuator", None)
            force_limits = None
            peak_solver_force = None
            solver_joint_ids = []
            solver_dof_ids = None
            if mj_model is not None and actual_force is not None:
                # MJWarp clamps combined P+D torque at the joint, not each actuator.
                solver_joint_ids = [i for i, limited in enumerate(mj_model.jnt_actfrclimited) if limited]
                assert len(solver_joint_ids) == 27
                solver_dof_ids = torch.tensor(mj_model.jnt_dofadr[solver_joint_ids], device=env.device)
                force_limits = torch.tensor(mj_model.jnt_actfrcrange[solver_joint_ids], device=env.device)
                peak_solver_force = torch.zeros(env.num_envs, len(solver_joint_ids), device=env.device)
            final_q = q0.clone()
            trace = []
            for step in range(args.hold_steps):
                command = torch.where(active[:, None], actions, zeros)
                obs, reward, terminated, truncated, _ = env.step(command)
                assert isinstance(obs["policy"], torch.Tensor)
                assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
                fell |= terminated
                timed_out |= truncated
                valid = active & ~(terminated | truncated)
                q = env.robot.data.joint_pos.torch[:, joint_ids]
                feet = torch.stack((env._get_foot_poses()[0], env._get_foot_poses()[2]), dim=1)
                force = torch.stack(env._get_ground_contact_forces(), dim=1)
                effort = env.robot.actuators.applied_effort.torch[:, joint_ids]
                if actual_force is not None and force_limits is not None and peak_solver_force is not None and solver_dof_ids is not None:
                    solved = wp.to_torch(actual_force).reshape(env.num_envs, -1)[:, solver_dof_ids]
                    assert torch.all(solved >= force_limits[:, 0] - 1e-3)
                    assert torch.all(solved <= force_limits[:, 1] + 1e-3)
                    peak_solver_force = torch.maximum(peak_solver_force, solved.abs())
                expected = torch.where(valid[:, None], target, env.joint_targets[:, joint_ids])
                torch.testing.assert_close(env.joint_targets[:, joint_ids], expected)
                dq = q - q0
                displacement = feet - feet0
                peak_dq = torch.where(valid[:, None] & (dq.abs() > peak_dq.abs()), dq, peak_dq)
                peak_foot = torch.where(valid[:, None, None] & (displacement.abs() > peak_foot.abs()), displacement, peak_foot)
                min_fz = torch.where(valid[:, None], torch.minimum(min_fz, force[:, :, 2]), min_fz)
                peak_effort = torch.where(valid[:, None], torch.maximum(peak_effort, effort.abs()), peak_effort)
                final_q = torch.where(valid[:, None], q, final_q)
                for i, (name, _) in enumerate(cases):
                    trace.append({"case": name, "step": step, "valid": bool(valid[i]),
                        "q": q[i].tolist(), "target": target[i].tolist(),
                        "sole_delta": displacement[i].tolist(), "ground_force": force[i].tolist(),
                        "root_z": env.robot.data.root_pos_w.torch[i, 2].item(),
                        "terminated": bool(terminated[i]), "truncated": bool(truncated[i])})
                active = valid
            rows = []
            for i, (name, command) in enumerate(cases):
                ids = [JOINT_NAMES.index(n) for n in command]
                row = {"case": name, "command_rad": command, "fell": bool(fell[i]), "timeout": bool(timed_out[i]),
                    "requested_offsets": offsets[i].tolist(), "target_minus_default": (target[i] - defaults[i]).tolist(),
                    "final_joint_delta": (final_q[i] - q0[i]).tolist(), "peak_joint_delta": peak_dq[i].tolist(),
                    "peak_sole_delta": peak_foot[i].tolist(), "minimum_ground_fz": min_fz[i].tolist(),
                    "peak_estimated_effort": peak_effort[i].tolist(),
                    "mean_target_error_rad": (final_q[i, ids] - target[i, ids]).abs().mean().item() if ids else 0.0}
                rows.append(row)
                if peak_solver_force is not None:
                    row['peak_solver_joint_torque'] = peak_solver_force[i].tolist()
                print(f"{name:16s} fall={row['fell']} q_err={row['mean_target_error_rad']:.4f} "
                      f"sole_dz={peak_foot[i, :, 2].tolist()} min_Fz={row['minimum_ground_fz']}", flush=True)
            hardware = {"joint_names": JOINT_NAMES, "solver_order": env.robot.joint_names,
                "mapping": joint_ids.tolist(), "physics_dt": env.physics_dt, "policy_dt": env.step_dt,
                "stiffness": env.robot.data.joint_stiffness.torch[0, joint_ids].tolist(),
                "damping": env.robot.data.joint_damping.torch[0, joint_ids].tolist(),
                "effort_limits": env.robot.data.joint_effort_limits.torch[0, joint_ids].tolist(),
                "soft_limits": limits[0].tolist()}
            if force_limits is not None and mj_model is not None:
                hardware['solver_force_limits'] = force_limits.tolist()
                hardware['solver_joint_names'] = [mj_model.joint(i).name for i in solver_joint_ids]
                hardware['solver_force_limits_verified'] = True
            else:
                hardware['solver_force_limits_verified'] = False
            (args.output / "results.json").write_text(json.dumps({"hardware": hardware, "cases": rows}, indent=2))
            with (args.output / "trace.jsonl").open("w") as stream:
                for row in trace:
                    stream.write(json.dumps(row) + "\n")
            print("Action authority test completed:", args.output, flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
