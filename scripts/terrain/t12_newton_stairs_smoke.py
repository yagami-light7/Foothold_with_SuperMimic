#!/usr/bin/env python3
"""T1.2: Test an exported Isaac Lab stair mesh with actual Newton dynamics.

Put this script and load_stairs_asset.py in the same directory and run via
Isaac Lab 3.0 Python. The scene deliberately contains no additional plane.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Newton stair mesh collision smoke test")
parser.add_argument("--asset-dir", required=True, type=Path)
parser.add_argument("--max-steps", type=int, default=600)
parser.add_argument("--physics", default="newton_mjwarp", choices=["newton_mjwarp"])
parser.add_argument("--drop-x", type=float, default=0.0)
parser.add_argument("--result-json", type=Path, help="Save the contact-test result without overwriting an existing file")
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton_gl"])
args = parser.parse_args()
if args.max_steps < 400:
    parser.error("--max-steps must be >=400 for this contact test")

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.physics import PhysicsCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass, instantiate
# Executed by file path: use the sibling helper instead of ROS's "scripts" package.
from load_stairs_asset import spawn_stairs

# Evaluate the SAME rasterized geometry exported for TCRS, as a diagnostic.
# The Newton collider is still built from exact terrain_mesh.npz below.
with np.load(args.asset_dir / "height_map.npz") as hm:
    xs, ys, heights = hm["x_m"], hm["y_m"], hm["height_m"]
    ix = int(np.argmin(abs(xs - args.drop_x)))
    iy = int(np.argmin(abs(ys)))
    top_z = float(heights[iy, ix])
    if not np.isfinite(top_z):
        raise RuntimeError(f"No platform surface height at XY=({args.drop_x},0)")

# A 10 cm cube dropped from 0.5 m above the top surface.
CUBE_HALF_HEIGHT = 0.05
DROP_HEIGHT = 0.50

@configclass
class StairSmokeSceneCfg(InteractiveSceneCfg):
    # Register the already authored mesh with Newton's scene clone/import plan.
    # An undeclared /World/Stairs prim is visible in USD but absent from physics.
    stairs = AssetBaseCfg(prim_path="/World/Stairs", spawn=None, collision_group=-1)
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=800.0, color=(0.75, 0.75, 0.75)),
    )
    # Do NOT add an Isaac Lab ground plane: the exported terrain includes its own surface.
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.10, 0.10, 0.10),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=1.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.15)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(args.drop_x, 0.0, top_z + DROP_HEIGHT)),
    )


def main():
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args) as physics_cfg:
        # Keep render-only flat geometry available even with an offscreen GL
        # viewer, whose automatic visual-shape detection can happen too early.
        physics_cfg.load_visual_shapes = True
        sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=0.005, device=args.device, physics=physics_cfg)
        )
        sim.set_camera_view(eye=(3.0, 3.0, 2.5), target=(0.0, 0.0, top_z))
        mesh = spawn_stairs(args.asset_dir, prim_path="/World/Stairs")
        scene = instantiate(StairSmokeSceneCfg(num_envs=1, env_spacing=4.0))
        sim.reset()

        target_center_z = top_z + CUBE_HALF_HEIGHT
        count = 0
        final_z = None
        for step in range(args.max_steps):
            if not sim.is_running():
                break
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())
            count += 1
            final_z = float(scene["cube"].data.root_pos_w.torch[0, 2].item())
            if count % 100 == 0:
                print(f"step={count} cube_z={final_z:.4f} expected_rest_z={target_center_z:.4f}")

        if count < 400 or final_z is None:
            raise RuntimeError(f"Simulation stopped early after {count} steps")
        error = abs(final_z - target_center_z)
        result = {
            "asset": str(args.asset_dir),
            "top_z_from_height_map": top_z,
            "drop_x_m": args.drop_x,
            "cube_center_z": final_z,
            "expected_rest_center_z": target_center_z,
            "rest_height_error_m": error,
            "mesh_bounds": np.asarray(mesh.bounds).tolist(),
            "steps": count,
            "passed_height_gate": error <= 0.025,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if args.result_json is not None:
            args.result_json.parent.mkdir(parents=True, exist_ok=True)
            with args.result_json.open("x") as output:
                json.dump(result, output, indent=2)
                output.write("\n")
        if not result["passed_height_gate"]:
            raise RuntimeError("Cube not resting at expected stair top; inspect collision, spawn and contact")
        print("T1.2 basic gravity+collision smoke: PASS (height only; friction not yet validated)")


if __name__ == "__main__":
    main()
