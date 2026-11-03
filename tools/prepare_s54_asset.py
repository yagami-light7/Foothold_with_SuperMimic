"""Create a simulation-only MJCF copy, preserving all source kinematics/inertias."""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from supermimic.s54 import SOURCE_MJCF, MJCF_PATH, model_contract, checkpoint_contract


def main():
    tree = ET.parse(SOURCE_MJCF); root=tree.getroot(); world=root.find("worldbody")
    removed=[]
    for g in list(world.findall("geom")):
        removed.append(g.get("name")); world.remove(g)
    # Source masks are for selective robot self collisions; floor mask=1 misses left feet.
    # Explicitly enable robot↔ground on EVERY collision proxy in the sim copy.
    for geom in root.findall(".//geom"):
        if geom.get("class", "").startswith("collision") or geom.get("class") is None and geom.get("material") == "collision":
            geom.set("contype", "1"); geom.set("conaffinity", "1")
    root.set("model", "s54_sim")
    # The articulation configuration supplies the PD drives. Do not import a second
    # independent torque-motor/controller set from the retargeting viewer model.
    for tag in ('actuator','sensor'):
        element=root.find(tag)
        if element is not None:root.remove(element)
    world.find("body/freejoint").set("name", "floating_base_joint")
    tree.write(MJCF_PATH, encoding="unicode", xml_declaration=True)
    src, _ = model_contract(); dst, _ = model_contract(MJCF_PATH)
    for field in ("body_parentid", "body_pos", "body_quat", "body_mass", "body_inertia", "body_ipos", "body_iquat",
                  "jnt_type", "jnt_bodyid", "jnt_pos", "jnt_axis", "jnt_range", "qpos0",
                  "dof_armature", "dof_damping", "dof_frictionloss", "jnt_actfrclimited", "jnt_actfrcrange"):
        np.testing.assert_array_equal(getattr(src,field), getattr(dst,field), err_msg=field)
    assert not any(dst.geom_bodyid == 0), "Embedded world geoms remain"
    proxies=(dst.geom_contype != 0) | (dst.geom_conaffinity != 0)
    assert np.all(dst.geom_contype[proxies] == 1) and np.all(dst.geom_conaffinity[proxies] == 1)
    report=dict(**checkpoint_contract(), removed_world_geoms=removed, removed_source_motors=src.nu,
                removed_source_sensors=src.nsensor, simulation_mjcf_nu=dst.nu, collision_proxies=int(proxies.sum()),
                kinematics_and_inertias="EXACT", collision_policy="mask=1 for all proxies; Isaac articulation self-collisions disabled; scene supplies ONE ground",
                generated=str(MJCF_PATH))
    out=Path("results/s54/asset_preparation.json");out.parent.mkdir(parents=True, exist_ok=True);out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
