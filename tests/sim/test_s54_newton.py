
import sys
from pathlib import Path

# 支持直接运行测试文件，无需额外设置PYTHONPATH。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import math

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(
    description="KUAVO S54 Newton articulation smoke test"
)
parser.add_argument(
    "--num-steps",
    type=int,
    default=0,
    help="Stop after this many physics steps (0: run until the viewer closes).",
)

add_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_steps < 0:
    parser.error("--num-steps must be non-negative")

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorCollection, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab_newton.assets import Articulation
from isaaclab.cloner import CloneCfg, clone_plan_from_env_0, replicate
from isaaclab_newton.physics import NewtonCfg
from isaaclab_newton.sim.schemas import NewtonArticulationCfg

USD_PATH = Path(__file__).resolve().parents[2] / "assets" / "biped_s54.usd"
ROBOT_PATH = "/World/S54"


def main():
    if not USD_PATH.is_file():
        raise FileNotFoundError(f"Robot USD not found: {USD_PATH}")

    # Zero gravity for the initial articulation/control test.
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 240.0,
        device=args_cli.device,
        gravity=(0.0, 0.0, 0.0),
        physics=NewtonCfg(),
    )

    with launch_simulation(sim_cfg, args_cli):
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view(
            (3.0, 3.0, 2.0),
            (0.0, 0.0, 1.0),
        )

        ground_cfg = sim_utils.GroundPlaneCfg()
        sim_utils.spawn_ground_plane("/World/Ground", ground_cfg)

        light_cfg = sim_utils.DomeLightCfg(
            intensity=2500.0
        )
        sim_utils.spawn_light("/World/Light", light_cfg)

        robot_cfg = ArticulationCfg(
            prim_path=ROBOT_PATH,
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(USD_PATH),
                articulation_props=NewtonArticulationCfg(
                    self_collision_enabled=False
                ),
            ),

            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 1.0),
                joint_pos={".*": 0.0},
                joint_vel={".*": 0.0},
            ),
            actuators={
                "joints": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    joint_effort_limit=50.0,
                    stiffness=20.0,
                    damping=2.0,
                )
            },
        )

        # This Isaac Lab version imports only assets declared in a clone plan.
        # Spawning USD prims alone leaves the Newton model empty at sim.reset().
        # These absolute paths declare shared assets, including the single robot.
        plan = clone_plan_from_env_0(
            CloneCfg(),
            (
                robot_cfg,
                AssetBaseCfg(prim_path="/World/Ground"),
                AssetBaseCfg(prim_path="/World/Light"),
            ),
            num_envs=1,
            env_spacing=0.0,
        )
        robot = Articulation(cfg=robot_cfg)

        # Build the Newton simulation.
        try:
            replicate(plan)
            sim.reset()

        except KeyError:
            print("\n========== Newton Model Diagnostics ==========")

            from isaaclab_newton.physics.newton_manager import NewtonManager

            backend = NewtonManager.backend

            if backend is None or backend.model is None:
                print("Newton backend was not initialized.")
            else:
                model = backend.model

                print("Articulation count:", model.articulation_count)
                print("Articulation labels:")

                for i, label in enumerate(model.articulation_label):
                    print(f"  [{i}] {label}")

                print("Body count:", model.body_count)
                print("Joint count:", model.joint_count)

            print("Expected USD articulation:", f"{ROBOT_PATH}/base_link")
            print("================================================\n")

            raise
        print("\n===== S54 Articulation =====")
        print("Joint count:", robot.num_joints)
        print("Joint names:")
        for i, name in enumerate(robot.joint_names):
            print(f"  {i:02d}: {name}")

        if robot.num_joints != 27:
            raise RuntimeError(
                f"Expected 27 moving joints, got {robot.num_joints}"
            )

        # Choose one arm joint for a small motion test.
        joint_name = "zarm_l4_joint"
        if joint_name not in robot.joint_names:
            raise RuntimeError(f"Test joint {joint_name!r} is missing from the robot")
        joint_idx = robot.joint_names.index(joint_name)

        print("\nTesting:", joint_name)

        # Apply the configured initial state explicitly, as in Isaac Lab's
        # articulation examples, before sending the first control command.
        robot.write_root_pose_to_sim_index(
            root_pose=robot.data.default_root_pose.torch.clone()
        )
        robot.write_root_velocity_to_sim_index(
            root_velocity=robot.data.default_root_vel.torch.clone()
        )
        robot.write_joint_position_to_sim_index(
            position=robot.data.default_joint_pos.torch.clone()
        )
        robot.write_joint_velocity_to_sim_index(
            velocity=robot.data.default_joint_vel.torch.clone()
        )
        robot.reset()

        sim_dt = sim.get_physics_dt()
        t = 0.0
        step = 0

        while sim.is_running() and (
            args_cli.num_steps == 0 or step < args_cli.num_steps
        ):
            target = robot.data.default_joint_pos.torch.clone()

            # The S54 left elbow has limits [-150 deg, 0 deg]. Start at zero
            # and move within [-0.2, 0] rad; a zero-centered sine exceeds 0.
            target[:, joint_idx] = (
                -0.10 * (1.0 - math.cos(2.0 * math.pi * 0.25 * t))
            )

            assert isinstance(robot.actuators, ActuatorCollection)
            robot.actuators.target_command.set_position_index(value=target)

            robot.write_data_to_sim()

            sim.step()
            robot.update(sim_dt)

            if step % 120 == 0:
                actual = robot.data.joint_pos.torch[
                    0, joint_idx
                ].item()
                desired = target[0, joint_idx].item()
                if not math.isfinite(actual):
                    raise RuntimeError(f"Non-finite position for {joint_name} at t={t:.2f}")
                print(
                    f"t={t:.2f} "
                    f"target={desired:.3f} "
                    f"actual={actual:.3f}"
                )

            t += sim_dt
            step += 1

        print(f"Completed {step} physics steps ({t:.3f} s simulated).")


if __name__ == "__main__":
    main()
