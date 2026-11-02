
import sys
from pathlib import Path

# 支持直接运行测试文件，无需额外设置PYTHONPATH。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="KUAVO S54 stand test")

add_launcher_args(parser)

parser.set_defaults(device="cuda:0")

args_cli = parser.parse_args()

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorCollection, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab_newton.assets import Articulation
from isaaclab.cloner import CloneCfg, clone_plan_from_env_0, replicate
from isaaclab_newton.physics import NewtonCfg, MJWarpSolverCfg
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg
from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab_newton.sensors import ContactSensor

def main():
    # simulation configuration
    sim_cfg = sim_utils.SimulationCfg(
        dt = 0.002,
        device=args_cli.device,
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                nconmax=256,
                njmax=512,
            )
        ),
    )

    # simulation launch
    with launch_simulation(sim_cfg, args_cli):
        sim = sim_utils.SimulationContext(sim_cfg)

        # camera position and orientation
        sim.set_camera_view(
            (3.0, 3.0, 2.0),
            (0.0, 0.0, 0.9),
        )

        # ground setting
        ground_cfg = sim_utils.GroundPlaneCfg()
        sim_utils.spawn_ground_plane("/World/Ground", ground_cfg)
    
        # light setting
        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0)
        sim_utils.spawn_light("/World/Light", light_cfg)

        # usd 
        usd_path = (
            Path(__file__).resolve().parents[2]
            / "assets"
            / "biped_s54.usd"
        )

        # Kp、Kd、N*m
        joint_params = {
            "leg_[lr]1_joint": (100.0, 5.0, 88.9),
            "leg_[lr]2_joint": (100.0, 5.0, 49.7),
            "leg_[lr]3_joint": (80.0, 5.0, 92.4),
            "leg_[lr]4_joint": (80.0, 6.0, 196.0),
            "leg_[lr]5_joint": (30.0, 7.5, 39.9),
            "leg_[lr]6_joint": (50.0, 7.5, 39.9),

            "waist_yaw_joint": (40.0, 7.0, 71.4),

            "zarm_[lr]1_joint": (15.0, 6.0, 46.2),
            "zarm_[lr]2_joint": (30.0, 3.0, 52.5),
            "zarm_[lr]3_joint": (15.0, 3.0, 39.9),
            "zarm_[lr]4_joint": (30.0, 6.0, 52.5),
            "zarm_[lr][567]_joint": (15.0, 3.0, 9.87),
        }

        # robot configuration
        robot_cfg = ArticulationCfg(
            prim_path="/World/S54",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(usd_path),
                fix_root_link=False,  # 自由基座
                articulation_props=NewtonArticulationCfg(
                    self_collision_enabled=True, 
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.969025),
                # from kuavo-ros-opensource
                joint_pos={
                    "leg_[lr][126]_joint": 0.0,
                    "leg_[lr]3_joint": -0.10,     # 髋 pitch
                    "leg_[lr]4_joint": 0.25,      # 膝 pitch
                    "leg_[lr]5_joint": -0.15,     # 踝 pitch

                    "waist_yaw_joint": 0.0,

                    "zarm_[lr][134567]_joint": 0.0,
                    "zarm_l2_joint": 0.075,      # 左臂略向外展开
                    "zarm_r2_joint": -0.075,     # 右臂略向外展开
                },
                joint_vel={".*": 0.0},
            ),
            actuators={
                "joints": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    # Actual PD(Based on real test)
                    stiffness={
                        name: values[0] * 30.0
                        for name, values in joint_params.items()
                    },
                    damping={
                                name: values[1] * (30.0 ** 0.5)
                                for name, values in joint_params.items()
                            },
                    joint_effort_limit={
                        name: values[2]
                        for name, values in joint_params.items()
                    }
                )
            },
        )

        # import scene to Newton
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

        # build robot
        robot = Articulation(cfg=robot_cfg)

        # foot contact
        foot_contact = ContactSensor(
            ContactSensorCfg(
                prim_path="/World/S54/leg_[lr]6_link", # 左右脚刚体
                filter_shape_prim_expr=["/World/Ground/.*"], # 区分地面接触和机器人自碰撞
                update_period=0.0, # 每个仿真步都可以更新
                history_length=0, 
            )
        )
        replicate(plan)

        # reset simulation
        sim.reset()

        # default state for robot
        target = robot.data.default_joint_pos.torch.clone()

        robot.write_root_pose_to_sim_index(
            root_pose=robot.data.default_root_pose.torch.clone()
        )

        robot.write_root_velocity_to_sim_index(
            root_velocity=robot.data.default_root_vel.torch.clone()
        )

        robot.write_joint_position_to_sim_index(position=target)

        robot.write_joint_velocity_to_sim_index(
            velocity=robot.data.default_joint_vel.torch.clone()
        )

        robot.reset()

        foot_contact.reset()

        foot_names = foot_contact.sensor_names
        assert foot_names is not None

        left_idx = foot_names.index("leg_l6_link")
        right_idx = foot_names.index("leg_r6_link")

        print("脚部传感器:", foot_names)
        print("过滤的地面形状:", foot_contact.filter_object_names)

        assert isinstance(robot.actuators, ActuatorCollection)
        robot.actuators.target_command.set_position_index(value=target)

        # sim running
        dt = sim.get_physics_dt()
        step = 0

        while sim.is_running():
            robot.write_data_to_sim()
            sim.step()

            robot.update(dt)
            foot_contact.update(dt)

            step += 1

            if step % 500 == 0:
                # 原始形状：[环境数, 脚数量, 地面形状数量, xyz]
                # 取第一个环境，再对所有地面形状的作用力求和。
                force_matrix = foot_contact.data.force_matrix_w
                assert force_matrix is not None
                forces = force_matrix.torch[0].sum(dim=1)

                left_force = forces[left_idx].tolist()
                right_force = forces[right_idx].tolist()

                total_fz = left_force[2] + right_force[2]

                print(
                    f"t={step * dt:.1f}s  "
                    f"左脚 F=({left_force[0]:.2f}, "
                    f"{left_force[1]:.2f}, {left_force[2]:.2f}) N  "
                    f"右脚 F=({right_force[0]:.2f}, "
                    f"{right_force[1]:.2f}, {right_force[2]:.2f}) N  "
                    f"双脚 Fz={total_fz:.2f} N"
                )


if __name__ == "__main__":
    main()
