#!/usr/bin/env python3
"""Load a terrain asset inside an existing Isaac Lab Newton scene.

Call spawn_stairs() after creating SimulationContext, before sim.reset().
Declare the same prim_path in InteractiveSceneCfg with
AssetBaseCfg(prim_path=..., spawn=None, collision_group=-1) so Newton imports
the existing mesh as a shared static collider.

Render representation: flat_visual_v2. The original indexed mesh remains the
collider. Only the non-colliding visual copy has independent triangle corners:
Newton 1.6.1's ordinary USD mesh import ignores authored uniform normals.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import trimesh
import isaaclab.sim as sim_utils


def spawn_stairs(
    asset_dir: str | Path,
    prim_path: str = "/World/Stairs",
    *,
    diffuse_color: tuple[float, float, float] = (0.08, 0.08, 0.08),
):
    from isaaclab.terrains.utils import create_prim_from_mesh

    asset_dir = Path(asset_dir)
    manifest = json.loads((asset_dir / "manifest.json").read_text())
    if manifest["schema"] != "s54_terrain_asset_v1":
        raise ValueError("Unsupported terrain asset schema")
    mesh_file = asset_dir / manifest["mesh_file"]
    sha = hashlib.sha256(mesh_file.read_bytes()).hexdigest()
    if sha != manifest["mesh_sha256"]:
        raise ValueError("Terrain geometry hash mismatch")
    with np.load(mesh_file) as payload:
        mesh = trimesh.Trimesh(
            vertices=payload["vertices"],
            faces=payload["faces"],
            process=False,
        )
    # Creates USD mesh + collision. Do NOT add a second ground-plane
    # covering the stairs when integrating with the S54 task.

    terrain_material = sim_utils.PreviewSurfaceCfg(
        # UsdPreviewSurface takes linear RGB; Newton converts it to display RGB.
        diffuse_color=diffuse_color,
        roughness=0.95,
        metallic=0.0,
    )

    create_prim_from_mesh(
        prim_path,
        mesh,
        visual_material=terrain_material,
    )

    from pxr import Gf, UsdGeom
    from isaaclab.sim.utils import get_current_stage

    # 读取 Isaac Lab 创建的 USD Mesh
    stage = get_current_stage()
    usd_mesh = UsdGeom.Mesh.Get(stage, f"{prim_path}/mesh")

    if not usd_mesh.GetPrim().IsValid():
        raise RuntimeError("Terrain USD Mesh not found")

    # Keep correct USD normals for renderers that honor them.
    usd_mesh.GetSubdivisionSchemeAttr().Set(
        UsdGeom.Tokens.none
    )

    face_normals = mesh.face_normals

    usd_mesh.CreateNormalsAttr().Set([
        Gf.Vec3f(float(n[0]), float(n[1]), float(n[2]))
        for n in face_normals
    ])

    usd_mesh.SetNormalsInterpolation(
        UsdGeom.Tokens.uniform
    )

    # Newton's untextured mesh path drops authored normals, then GL averages
    # normals across shared vertices. Make that averaging harmless on a separate
    # render-only mesh. Never unweld, simplify or remove faces from the collider.
    from pxr import Sdf, UsdShade

    visual = UsdGeom.Mesh.Define(stage, f"{prim_path}/flat_visual")
    corners = mesh.triangles.reshape(-1, 3)
    visual.CreatePointsAttr().Set([Gf.Vec3f(*map(float, p)) for p in corners])
    visual.CreateFaceVertexCountsAttr().Set([3] * len(mesh.faces))
    visual.CreateFaceVertexIndicesAttr().Set(np.arange(len(corners), dtype=np.int32))
    visual.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    visual.CreateNormalsAttr().Set([Gf.Vec3f(*map(float, n)) for n in face_normals])
    visual.SetNormalsInterpolation(UsdGeom.Tokens.uniform)
    material = UsdShade.Material.Get(stage, f"{prim_path}/visualMaterial")
    UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(material)
    visual.GetPrim().CreateAttribute("s54:renderRepresentation", Sdf.ValueTypeNames.String).Set("flat_visual_v2")
    # Visibility changes drawability, not CollisionAPI or collisionEnabled.
    # Hide only this mesh, never its parent or the visual sibling.
    usd_mesh.MakeInvisible()
    print("[INFO] Terrain flat_visual_v2: original collider retained; flat visual mesh has no CollisionAPI")

    return mesh
