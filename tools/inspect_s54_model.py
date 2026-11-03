"""Dump the compiled MJCF contract and optionally audit an existing USD without Kit."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
from supermimic.s54 import SOURCE_MJCF, model_contract, checkpoint_contract, actuator_parameters, initial_root_height


def inspect(path):
    m, rows = model_contract(path)
    free = next(j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
    return dict(model_path=str(path), **checkpoint_contract(), nq=m.nq, nv=m.nv, nu=m.nu,
                root_body=m.body(int(m.jnt_bodyid[free])).name,
                freejoint=dict(name=m.joint(free).name, qposadr=int(m.jnt_qposadr[free]),
                               qpos0=m.qpos0[:7].tolist()), joints=rows,
                bodies=[dict(name=m.body(b).name, parent=m.body(int(m.body_parentid[b])).name,
                             mass=float(m.body_mass[b]), inertia=m.body_inertia[b].tolist(),
                             pos=m.body_pos[b].tolist(), quat_wxyz=m.body_quat[b].tolist()) for b in range(m.nbody)],
                actuators=[dict(name=m.actuator(a).name, joint=m.joint(int(m.actuator_trnid[a, 0])).name,
                                ctrlrange=m.actuator_ctrlrange[a].tolist(), gear=m.actuator_gear[a].tolist()) for a in range(m.nu)],
                geoms=[dict(name=m.geom(g).name, body=m.body(int(m.geom_bodyid[g])).name,
                            type=int(m.geom_type[g]), contype=int(m.geom_contype[g]),
                            conaffinity=int(m.geom_conaffinity[g]), size=m.geom_size[g].tolist(),
                            pos=m.geom_pos[g].tolist()) for g in range(m.ngeom)],
                simulation_actuators=actuator_parameters(),safe_neutral_spawn_height=initial_root_height())


def inspect_usd(path):
    from pxr import Usd, UsdPhysics, UsdGeom
    s = Usd.Stage.Open(str(path))
    if s is None: raise ValueError(f"Cannot open {path}")
    # Importer defaults may select Physics=none. Audit the same variant as the
    # runtime; author the selection only in the ephemeral session layer.
    s.SetEditTarget(s.GetSessionLayer())
    for p in list(s.Traverse()):
        if 'mujoco' in p.GetVariantSet('Physics').GetVariantNames():
            p.GetVariantSet('Physics').SetVariantSelection('mujoco')
    joints, roots, bodies, variants, props = [], [], [], {}, {}
    for p in s.Traverse():
        if p.GetVariantSets().GetNames():
            variants[str(p.GetPath())] = {n: p.GetVariantSet(n).GetVariantNames() for n in p.GetVariantSets().GetNames()}
        if p.HasAPI(UsdPhysics.ArticulationRootAPI): roots.append(str(p.GetPath()))
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies.append(p.GetName())
            mass=UsdPhysics.MassAPI(p)
            com=mass.GetCenterOfMassAttr().Get(); inertia=mass.GetDiagonalInertiaAttr().Get()
            principal=mass.GetPrincipalAxesAttr().Get()
            props[p.GetName()]=dict(mass=mass.GetMassAttr().Get(),
                center_of_mass=list(com) if com is not None else None,
                diagonal_inertia=list(inertia) if inertia is not None else None,
                principal_axes_xyzw=list(principal.GetImaginary())+[principal.GetReal()] if principal is not None else None,
                schemas=p.GetAppliedSchemas())
        if p.IsA(UsdPhysics.RevoluteJoint):
            j = UsdPhysics.RevoluteJoint(p)
            joints.append(dict(name=p.GetName(), path=str(p.GetPath()), axis=j.GetAxisAttr().Get(),
                               lower_deg=j.GetLowerLimitAttr().Get(), upper_deg=j.GetUpperLimitAttr().Get(),
                               body0=[str(x) for x in j.GetBody0Rel().GetTargets()],
                               body1=[str(x) for x in j.GetBody1Rel().GetTargets()],
                               localPos0=list(j.GetLocalPos0Attr().Get()), localPos1=list(j.GetLocalPos1Attr().Get()),
                               localRot0=list(j.GetLocalRot0Attr().Get().GetImaginary()) + [j.GetLocalRot0Attr().Get().GetReal()],
                               localRot1=list(j.GetLocalRot1Attr().Get().GetImaginary()) + [j.GetLocalRot1Attr().Get().GetReal()]))
    return dict(path=str(path), roots=roots, bodies=bodies, body_properties=props, joints=joints, variants=variants,
                up_axis=UsdGeom.GetStageUpAxis(s),meters_per_unit=UsdGeom.GetStageMetersPerUnit(s),
                alignment_status="INSPECTED_ONLY: require simulator FK/roundtrip before use")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=SOURCE_MJCF)
    ap.add_argument("--usd", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results/s54/model_inspection.json"))
    a = ap.parse_args(); report = inspect(a.model)
    if a.usd:
        try: report["usd"] = inspect_usd(a.usd)
        except ImportError as e: report["usd"] = dict(status="NOT RUN", reason=str(e))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("nq", "nv", "nu", "root_body", "freejoint")}, indent=2))
    print(f"{len(report['joints'])} joints; full mapping/axes/ranges/qpos0/bodies/geoms/actuators: {a.out}")
    if "usd" in report: print("USD:",report["usd"].get("status",report["usd"].get("alignment_status")))


if __name__ == "__main__": main()
