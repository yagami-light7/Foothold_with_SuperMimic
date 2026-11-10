#!/usr/bin/env python3
"""Reproducible Mesh -> USD -> Newton GL A/B, with unchanged renderer settings.

Run each case in a new process. A=native box, B=imported box, C=original
stairs, D=diagnostic exterior. --loader selects an archived/current loader.
No input terrain/height-map files are written.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

# Select EGL before pyglet/Isaac Lab import their window backends. Selecting
# headless only at ViewerGL construction mixes Xlib and EGL when DISPLAY is set.
os.environ.pop('DISPLAY', None)

import numpy as np
import trimesh
from PIL import Image

from terrain_render_diagnostics import audit_mesh, compare_surfaces, continuous_exterior, load_mesh, sha256, top_surface


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def mesh_usd_dump(stage, reference):
    from pxr import UsdGeom, UsdShade, UsdPhysics
    result = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get())
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        normals = mesh.GetNormalsAttr().Get()
        color = UsdGeom.PrimvarsAPI(prim).GetPrimvar('displayColor')
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        shaders = []
        if material:
            for child in material.GetPrim().GetChildren():
                if child.IsA(UsdShade.Shader):
                    shader = UsdShade.Shader(child)
                    shaders.append(dict(path=str(child.GetPath()), id=str(shader.GetIdAttr().Get()),
                                        inputs={v.GetBaseName(): str(v.Get()) for v in shader.GetInputs()}))
        row = dict(path=str(prim.GetPath()), vertices=len(points), triangles=len(indices),
                   subdivision=str(mesh.GetSubdivisionSchemeAttr().Get()),
                   normals_count=0 if normals is None else len(normals),
                   normals_interpolation=str(mesh.GetNormalsInterpolation()),
                   normals_preview=[] if normals is None else np.asarray(normals)[:6].tolist(),
                   display_color_count=len(color.Get()) if color and color.Get() is not None else 0,
                   display_color_interpolation=str(color.GetInterpolation()) if color else None,
                   display_color_preview=np.asarray(color.Get())[:3].tolist() if color and color.Get() is not None else [],
                   material_shaders=shaders, collision_api=prim.HasAPI(UsdPhysics.CollisionAPI),
                   visibility=str(mesh.ComputeVisibility()), purpose=str(mesh.ComputePurpose()))
        if reference is not None and points.shape == reference.vertices.shape:
            row.update(max_vertex_error_m=float(np.max(np.abs(points-reference.vertices))),
                       face_indices_identical=bool(np.array_equal(indices, reference.faces)))
        result.append(row)
    return result


def model_dump(model, reference):
    import newton
    flags, types, bodies = model.shape_flags.numpy(), model.shape_type.numpy(), model.shape_body.numpy()
    rows = []
    for i, label in enumerate(model.shape_label):
        src = model.shape_source[i]
        row = dict(path=label, type=newton.GeoType(int(types[i])).name,
                   collision=bool(int(flags[i]) & int(newton.ShapeFlags.COLLIDE_SHAPES)),
                   visible=bool(int(flags[i]) & int(newton.ShapeFlags.VISIBLE)), body=int(bodies[i]))
        if isinstance(src, newton.Mesh):
            row.update(vertices=len(src.vertices), triangles=len(src.indices)//3,
                       imported_authored_normals=getattr(src, '_normals', None) is not None,
                       color=None if src.color is None else list(src.color),
                       roughness=getattr(src, 'roughness', None), metallic=getattr(src, 'metallic', None))
            row['vertices_sha256'] = hashlib.sha256(np.asarray(src.vertices, dtype=np.float32).tobytes()).hexdigest()
            row['indices_sha256'] = hashlib.sha256(np.asarray(src.indices, dtype=np.int32).tobytes()).hexdigest()
            if reference is not None and src.vertices.shape == reference.vertices.shape:
                row.update(max_vertex_error_m=float(np.max(np.abs(src.vertices-reference.vertices))),
                           face_indices_identical=bool(np.array_equal(src.indices.reshape(-1, 3), reference.faces)))
        if getattr(model, 'shape_color', None) is not None:
            row['shape_display_color'] = model.shape_color.numpy()[i].tolist()
        rows.append(row)
    return rows


def gl_dump(viewer, out):
    result = []
    for i, (name, obj) in enumerate(viewer.objects.items()):
        if not hasattr(obj, 'vertices') or not hasattr(obj, 'num_points'):
            continue
        packed = obj.vertices.numpy()[:obj.num_points]
        index = obj.indices.numpy()[:obj.num_indices].reshape(-1, 3)
        fields = packed.dtype.names
        position = packed['pos'] if 'pos' in fields else packed['position']
        normal = packed['normal']
        face = trimesh.Trimesh(position, index, process=False).face_normals
        corner = normal[index]
        dot = np.sum(corner*face[:, None], axis=-1)
        variation = np.linalg.norm(corner-corner[:, :1], axis=-1)
        np.savez_compressed(out/f'gl_mesh_{i}.npz', vertices=position, faces=index, normals=normal)
        result.append(dict(name=name, vertices=len(position), triangles=len(index),
                           bounds=np.array([position.min(axis=0), position.max(axis=0)]).tolist(),
                           corner_normal_mismatch_count=int((dot < 1-1e-5).sum()),
                           corner_normal_count=int(dot.size),
                           max_corner_normal_angle_deg=float(np.degrees(np.arccos(np.clip(dot, -1, 1))).max()),
                           max_within_triangle_normal_variation=float(variation.max()),
                           recomputed_normals=obj.normals is not None,
                           buffer=f'gl_mesh_{i}.npz'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-dir', type=Path, default=Path('assets/terrains/stairs_straight_4_v1'))
    parser.add_argument('--case', choices=['A', 'B', 'C', 'D'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--loader', type=Path, default=Path(__file__).with_name('load_stairs_asset.py'))
    parser.add_argument('--unweld', action='store_true', help='Diagnostic USD-only corner splitting, not a production collider change')
    parser.add_argument('--no-usd-normals', action='store_true')
    parser.add_argument('--dome-intensity', type=float, default=800.)
    parser.add_argument('--steps', type=int, default=0)
    parser.add_argument('--drop-x', type=float, default=0.)
    parser.add_argument('--diffuse-color', type=float, nargs=3, help='Optional current-loader linear RGB override')
    parser.add_argument('--contact-pipeline', choices=['mujoco', 'newton'], default='mujoco',
                        help='Diagnostic contact comparison only; both use the same MJWarp dynamics solver')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(status='initializing', case=args.case, asset=str(args.asset_dir),
                  loader=str(args.loader), dome_intensity=args.dome_intensity,
                  contact_pipeline=args.contact_pipeline,
                  mesh_sha256=sha256(args.asset_dir/'terrain_mesh.npz'),
                  heightmap_sha256=sha256(args.asset_dir/'height_map.npz'))
    try:
        import isaaclab.sim as sim_utils
        from isaaclab.app import launch_simulation
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.utils import configclass, instantiate
        from isaaclab_newton.physics import NewtonCfg, MJWarpSolverCfg
        from isaaclab_visualizers.newton import NewtonGLVisualizerCfg
        from isaaclab.sim.utils import get_current_stage
        from isaaclab.terrains.utils import create_prim_from_mesh
        from pxr import Gf, UsdGeom

        original = load_mesh(args.asset_dir)
        reference = original
        if args.case in ('A', 'B'):
            reference = trimesh.creation.box((2., 2., .4))
            reference.apply_translation((0., 0., .2))
        elif args.case == 'D':
            reference = continuous_exterior(original)
            report['exterior_surface_comparison'] = compare_surfaces(original, reference, args.asset_dir)
            np.savez_compressed(args.out/'diagnostic_exterior.npz', vertices=reference.vertices, faces=reference.faces)
        report['geometry'] = audit_mesh(reference)
        viz = NewtonGLVisualizerCfg(headless=True, window_width=1280, window_height=800, streaming_view=False)
        # Explicit in headless mode: auto-detection may run before the offscreen
        # visualizer is initialized, and omit all render-only geometry.
        physics = NewtonCfg(load_visual_shapes=True, solver_cfg=MJWarpSolverCfg(
            use_mujoco_contacts=args.contact_pipeline == 'mujoco', save_to_mjcf=str(args.out/'solver.xml')))
        cfg = sim_utils.SimulationCfg(dt=.005, device='cuda:0', physics=physics, visualizer_cfgs=[viz])
        eye, target = (6., -7., 4.), (0., 0., .15)
        report['camera'] = dict(eye=eye, target=target, resolution=[1280, 800])
        material = sim_utils.PreviewSurfaceCfg(diffuse_color=(.28, .33, .40), roughness=.95, metallic=0.)
        with launch_simulation(cfg):
            sim = sim_utils.SimulationContext(cfg)
            sim.set_camera_view(eye=eye, target=target)
            native = None
            if args.case == 'A':
                native = sim_utils.CuboidCfg(size=(2., 2., .4),
                    collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True), visual_material=material)
            elif args.case == 'C':
                spec = importlib.util.spec_from_file_location('diagnosed_terrain_loader', args.loader)
                module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
                options = {} if args.diffuse_color is None else dict(diffuse_color=tuple(args.diffuse_color))
                reference = module.spawn_stairs(args.asset_dir, '/World/Target', **options)
            else:
                create_prim_from_mesh('/World/Target', reference, visual_material=material)
                m = UsdGeom.Mesh.Get(get_current_stage(), '/World/Target/mesh')
                m.GetSubdivisionSchemeAttr().Set('none')
                m.CreateNormalsAttr().Set([Gf.Vec3f(*map(float, n)) for n in reference.face_normals])
                m.SetNormalsInterpolation('uniform')
            if args.case != 'A':
                m = UsdGeom.Mesh.Get(get_current_stage(), '/World/Target/mesh')
                if args.no_usd_normals:
                    m.GetNormalsAttr().Clear()
                if args.unweld:
                    corners = reference.triangles.reshape(-1, 3)
                    m.GetPointsAttr().Set([Gf.Vec3f(*map(float, p)) for p in corners])
                    m.GetFaceVertexIndicesAttr().Set(np.arange(len(corners), dtype=np.int32))
            top = float(top_surface(original, [[args.drop_x, 0.]])[0])
            @configclass
            class SceneCfg(InteractiveSceneCfg):
                terrain = AssetBaseCfg(prim_path='/World/Target', spawn=native, collision_group=-1,
                    init_state=AssetBaseCfg.InitialStateCfg(pos=(0., 0., .2) if args.case == 'A' else (0., 0., 0.)))
                light = AssetBaseCfg(prim_path='/World/Light',
                    spawn=sim_utils.DomeLightCfg(intensity=args.dome_intensity, color=(.75, .75, .75)))
                # MJWarp requires a free joint even for a static render test.
                # Keep the same hidden, unstepped probe below view in A/B/C/D.
                probe = RigidObjectCfg(
                    prim_path='{ENV_REGEX_NS}/Probe', spawn=sim_utils.CuboidCfg(size=(.1, .1, .1),
                        visible=args.steps > 0,
                        rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(), mass_props=sim_utils.MassCfg(mass=1.),
                        collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(args.drop_x, 0., top+.5) if args.steps else (0., 0., -1.)))
            scene = instantiate(SceneCfg(num_envs=1, env_spacing=4.))
            stage = get_current_stage()
            report['usd'] = mesh_usd_dump(stage, reference)
            stage.GetRootLayer().Export(str(args.out/'scene.usda'))
            print('USD_ATTRIBUTES', json.dumps(report['usd']), flush=True)
            sim.reset()
            report['model'] = model_dump(sim.physics_manager.get_model(), reference)
            if not any(s['visible'] and s['path'].startswith('/World/Target') for s in report['model']):
                raise RuntimeError('No visible target mesh imported; check load_visual_shapes')
            trajectory = []
            for _ in range(args.steps):
                scene.write_data_to_sim(); sim.step(); scene.update(sim.get_physics_dt())
                trajectory.append(scene['probe'].data.root_pos_w.torch[0].cpu().numpy().copy())
            if args.steps:
                np.savez_compressed(args.out/'drop_trajectory.npz', position_w=np.asarray(trajectory), dt=.005)
                z = float(scene['probe'].data.root_pos_w.torch[0, 2])
                report['drop'] = dict(steps=args.steps, x=args.drop_x, expected_z=top+.05, actual_z=z,
                                      error_m=abs(z-top-.05), passed=abs(z-top-.05) <= .025)
            sim.render()
            visualizer = sim._visualizers[0]
            image = visualizer.render_rgb_array()
            Image.fromarray(image).save(args.out/'newton_gl.png')
            viewer = visualizer._viewer
            report['gl_meshes'] = gl_dump(viewer, args.out)
            renderer = viewer.renderer
            report['gl_lighting'] = {name: list(getattr(renderer, name)) for name in
                                     ('_light_color', 'ambient_sky', 'ambient_ground')}
            if report.get('drop', {}).get('passed') is False:
                raise RuntimeError('Physics drop regression failed')
        report['status'] = 'PASS'
    except BaseException as exc:
        report.update(status='FAILED', error=repr(exc))
        raise
    finally:
        write_json(args.out/'diagnosis.json', report)
    print('PASS', args.out, flush=True)


if __name__ == '__main__':
    main()
