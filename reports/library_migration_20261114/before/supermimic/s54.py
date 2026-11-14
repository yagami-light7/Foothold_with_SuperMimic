"""CPU-safe S54 contract. Names come from Leju-GMR, never from a simulator's ordering."""
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SOURCE_MJCF = REPO / "data/s54_description/xml/biped_s54.xml"
MJCF_PATH = REPO / "data/s54_description/xml/s54_sim.xml"
URDF_PATH = REPO / "data/s54_description/urdf/biped_s54.urdf"
USD_DIR = REPO / "data/s54_newton_usd"
USD_PATH = USD_DIR / "s54_sim/s54_sim.usda"
ROBOT_ID = "kuavo_s54_leju_gmr_v1"
SOURCE_FPS = 30.0
ROOT_BODY = "base_link"
COM_BODY = "waist_yaw"
FOOT_BODY_NAMES = ["leg_l6_link", "leg_r6_link"]
CSV_JOINT_NAMES = ([f"leg_{s}{i}_joint" for s in "lr" for i in range(1, 7)]
                   + ["waist_yaw_joint"]
                   + [f"zarm_{s}{i}_joint" for s in "lr" for i in range(1, 8)])
HEAD_JOINT_NAMES = ["zhead_1_joint", "zhead_2_joint"]
JOINT_NAMES = CSV_JOINT_NAMES + HEAD_JOINT_NAMES
NUM_J = len(JOINT_NAMES)
TRACKED_BODIES = [ROOT_BODY, "leg_l1_link", "leg_l4_link", "leg_l6_link",
                  "leg_r1_link", "leg_r4_link", "leg_r6_link", COM_BODY,
                  "zarm_l2_link", "zarm_l4_link", "zarm_l7_link",
                  "zarm_r2_link", "zarm_r4_link", "zarm_r7_link"]
GOAL_FRAME_DIM = 3 + 6 + 3 + 3 + 2 * NUM_J
XI_DIM = 1 + NUM_J + 3
JOINT_OFFSET = slice(1, 1 + NUM_J)
COM_OFFSET = slice(1 + NUM_J, XI_DIM)
assert NUM_J == 29 and len(TRACKED_BODIES) == 14


def name_mapping(sim_names):
    """canonical→sim and sim→canonical, both checked bijections."""
    names = list(sim_names)
    if len(names) != len(set(names)) or set(names) != set(JOINT_NAMES):
        raise ValueError(f"S54 joint mismatch: missing={set(JOINT_NAMES)-set(names)}, extra={set(names)-set(JOINT_NAMES)}")
    return [names.index(n) for n in JOINT_NAMES], [JOINT_NAMES.index(n) for n in names]


def model_contract(path=SOURCE_MJCF):
    m = mujoco.MjModel.from_xml_path(str(path))
    free = np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1 or m.body(int(m.jnt_bodyid[free[0]])).name != ROOT_BODY:
        raise ValueError("Expected one free base_link")
    moving = [m.joint(j).name for j in range(m.njnt) if j not in free]
    name_mapping(moving)
    if moving != JOINT_NAMES:
        raise ValueError(f"Leju-GMR source ordering changed: {moving}")
    for n in TRACKED_BODIES:
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) < 0:
            raise ValueError(f"missing tracked body {n}")
    rows = []
    for i, n in enumerate(JOINT_NAMES):
        j = m.joint(n).id
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE or not m.jnt_limited[j]:
            raise ValueError(f"Expected limited hinge {n}")
        adr = int(m.jnt_qposadr[j]); lo, hi = m.jnt_range[j]
        neutral = float(m.qpos0[adr])
        if not lo <= neutral <= hi:
            raise ValueError(f"Illegal model neutral {n}={neutral}")
        rows.append(dict(name=n, canonical_index=i, raw_csv_column=7+i if n in CSV_JOINT_NAMES else None,
                         full_csv_column=7+i, mj_joint_id=j, qposadr=adr,
                         dofadr=int(m.jnt_dofadr[j]), axis=m.jnt_axis[j].tolist(),
                         range=[float(lo), float(hi)], neutral=neutral))
    return m, rows


def default_joint_pos_list():
    return [r["neutral"] for r in model_contract()[1]]


def initial_root_height():
    """Raise source qpos0's straight-legged pose above its floor before the first reset."""
    m, _ = model_contract(); d = mujoco.MjData(m); mujoco.mj_forward(m,d)
    floor = m.geom('floor').id
    distances = [mujoco.mj_geomDistance(m,d,floor,g,10.0,None) for g in range(m.ngeom)
                 if m.geom_bodyid[g] != 0 and (m.geom_contype[g] or m.geom_conaffinity[g])]
    return float(m.qpos0[2] + max(0.0,0.02-min(distances)))


def actuator_parameters():
    """MJCF effort/armature; URDF velocity only; PD gains are explicit sim-only assumptions.

    URDF is NOT used for kinematics. Head effort differs from MJCF, so MJCF wins.
    """
    m, rows = model_contract()
    urdf = ET.parse(URDF_PATH)
    velocities = {j.get("name"): float(j.find("limit").get("velocity"))
                  for j in urdf.findall("joint") if j.find("limit") is not None}
    out = {}
    for r in rows:
        n = r["name"]; j = r["mj_joint_id"]
        aids = [a for a in range(m.nu) if m.actuator_trnid[a, 0] == j]
        if len(aids) != 1 or m.actuator_gear[aids[0], 0] != 1 or not m.actuator_ctrllimited[aids[0]]:
            raise ValueError(f"Expected one torque motor {n}")
        effort = min(abs(m.actuator_ctrlrange[aids[0]]))
        if m.jnt_actfrclimited[j]:
            effort = min(effort, min(abs(m.jnt_actfrcrange[j])))
        armature = float(m.dof_armature[m.jnt_dofadr[j]])
        kp = 200.0 if n.startswith(("leg_", "waist_")) else 80.0
        if n.startswith("zhead"): kp = 10.0
        elif n.startswith("zarm") and int(n.split("_")[1][1:]) >= 5: kp = 20.0
        out[n] = dict(effort=float(effort), velocity=velocities[n], armature=armature,
                      stiffness=kp, damping=2 * np.sqrt(kp * armature))
    return out


def checkpoint_contract():
    return dict(robot=ROBOT_ID, joint_names=JOINT_NAMES, tracked_bodies=TRACKED_BODIES,
                source_mjcf_sha256=hashlib.sha256(SOURCE_MJCF.read_bytes()).hexdigest())


def assert_checkpoint(ck):
    if ck.get("robot_contract") != checkpoint_contract():
        raise ValueError("Checkpoint robot contract differs from S54. G1 weights must never be loaded.")
