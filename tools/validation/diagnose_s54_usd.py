"""Read-only USD/source neutral-pose and collision audit, using the existing importer env."""

# Support both direct execution and python -m from the repository layout.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_root = str(_ToolPath(__file__).resolve().parents[2])
if _tool_root not in _tool_sys.path:
    _tool_sys.path.insert(0, _tool_root)

import json
from pathlib import Path

import mujoco
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation

from tools.validation.check_s54_usd import check
from supermimic.s54 import SOURCE_MJCF, USD_PATH


def main():
    report = check(USD_PATH)
    model = mujoco.MjModel.from_xml_path(str(SOURCE_MJCF)); data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    stage = Usd.Stage.Open(str(USD_PATH)); stage.SetEditTarget(stage.GetSessionLayer())
    for prim in list(stage.Traverse()):
        if 'mujoco' in prim.GetVariantSet('Physics').GetVariantNames():
            prim.GetVariantSet('Physics').SetVariantSelection('mujoco')
    transforms = UsdGeom.XformCache(); collisions = {}; positions = []; angles = []; bounds = {}
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'guide', 'proxy', 'render'], False, True)
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            b = model.body(prim.GetName()).id
            matrix = np.asarray(transforms.GetLocalToWorldTransform(prim)).T
            positions.append(np.linalg.norm(matrix[:3, 3] - data.xpos[b]))
            angles.append(Rotation.from_matrix(matrix[:3, :3].T @ data.xmat[b].reshape(3, 3)).magnitude())
        if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            parent = prim
            while parent and not parent.IsPseudoRoot() and not parent.HasAPI(UsdPhysics.RigidBodyAPI):
                parent = parent.GetParent()
            owner = parent.GetName() if parent and not parent.IsPseudoRoot() else 'world'
            collisions[owner] = collisions.get(owner, 0) + 1
            if owner in ('leg_l6_link', 'leg_r6_link'):
                value = float(bbox.ComputeWorldBound(prim).ComputeAlignedRange().GetMin()[2])
                bounds[owner] = min(bounds.get(owner, np.inf), value)
    report['neutral_body_position_max_m'] = float(max(positions))
    report['neutral_body_rotation_max_rad'] = float(max(angles))
    report['collision_prims_by_body'] = collisions
    report['usd_neutral_foot_collision_min_z_m'] = bounds
    report['mjcf_neutral_foot_collision_min_z_m'] = {}
    for name in bounds:
        b = model.body(name).id
        geoms = np.flatnonzero((model.geom_bodyid == b) & ((model.geom_contype | model.geom_conaffinity) != 0))
        report['mjcf_neutral_foot_collision_min_z_m'][name] = min(
            mujoco.mj_geomDistance(model, data, model.geom('floor').id, int(g), 10.0, None) for g in geoms)
    path = Path('reports/s54_reference_figures/usd_alignment.json')
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('joints', 'variants')}, indent=2))


if __name__ == '__main__':
    main()
