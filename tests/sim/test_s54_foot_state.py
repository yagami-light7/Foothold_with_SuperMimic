
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

import torch
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
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

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

        # Identify the left and right feet
        left_foot_name = "leg_l6_link"
        right_foot_name = "leg_r6_link"

        left_foot_idx = robot.body_names.index(left_foot_name)
        right_foot_idx = robot.body_names.index(right_foot_name)

        print("Left foot index:", left_foot_idx)
        print("Right foot index:", right_foot_idx)

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

            if step % 1000 == 0:
                # 获取所有刚体的世界坐标系下位姿
                body_pos = robot.data.body_pos_w.torch
                body_quat = robot.data.body_quat_w.torch

                # Shapes:
                # body_pos  -> [num_envs, num_bodies, 3]
                # body_quat -> [num_envs, num_bodies, 4] (xyzw)
                left_pos = body_pos[:, left_foot_idx, :]
                right_pos = body_pos[:, right_foot_idx, :]

                left_quat = body_quat[:, left_foot_idx, :]
                right_quat = body_quat[:, right_foot_idx, :]

                num_envs = right_pos.shape[0]

                # Goal Generator生成foot目标位置foot_target_pos，left in right stance
                left_target_offset = torch.tensor(
                    [[0.12, 0.19, 0.0]],
                    device=right_pos.device,
                    dtype=right_pos.dtype,
                ).expand(num_envs, -1) # 将张量扩展到对应GPU的所有并行环境

                # 将左脚目标位姿转换到世界坐标系下
                left_target_pos_w, left_target_quat_w = (
                    combine_frame_transforms(
                        right_pos,
                        right_quat,
                        left_target_offset,
                    )
                )

                # 右脚保持不动 stance foot
                right_target_pos_w = right_pos.clone()
                right_target_quat_w = right_quat.clone()

                # 将左脚目标位姿转换到右脚坐标系下
                l_pos_s, l_quat_s = subtract_frame_transforms( 
                    right_pos, right_quat,
                    left_target_pos_w, left_target_quat_w,
                )

                # 将右脚目标位姿转换到右脚坐标系下
                r_pos_s, r_quat_s = subtract_frame_transforms(
                    right_pos, right_quat,
                    right_target_pos_w, right_target_quat_w,
                )

                # 创建步态相位
                phase = torch.zeros(
                    (num_envs, 1),
                    device=right_pos.device,
                    dtype=right_pos.dtype,
                )

                # 正余弦编码
                phase_obs = torch.cat(
                    (
                        torch.cos(2.0 * torch.pi * phase),
                        torch.sin(2.0 * torch.pi * phase),
                    ),
                    dim=-1,
                )

                # 拼接观测向量（左右脚目标位姿+步态相位编码）
                goal_obs = torch.cat(
                    (
                        l_pos_s,
                        l_quat_s,
                        r_pos_s,
                        r_quat_s,
                        phase_obs,
                    ),
                    dim=-1,
                )
                
                assert goal_obs.shape == (num_envs, 16)
                assert torch.isfinite(goal_obs).all()

                t = dt * step
                print(f"\n[t={t:.2f}s]")
                print("Goal shape:", goal_obs.shape)
                print("Goal vector:", goal_obs[0].tolist())

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
