"""Roban contract derived from the imported MJCF/URDF, never S54 indices."""
from functools import lru_cache
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROBOT_ID = 'roban_s22_0901'
MODEL_ROOT = REPO / 'library/model' / ROBOT_ID
SOURCE_MJCF = MODEL_ROOT / 'xml/biped_s17_verified_handball_fixedhead.xml'
MJCF_PATH = SOURCE_MJCF  # Source has no world ground: scene supplies the only plane.
URDF_PATH = MODEL_ROOT / 'urdf/biped_s17_verified_handball_fixedhead.urdf'
USD_DIR = REPO / 'assets/roban/usd'
USD_PATH = USD_DIR / SOURCE_MJCF.stem / (SOURCE_MJCF.stem + '.usda')
ROOT_BODY = 'base_link'
COM_BODY = 'waist_yaw_link'
_model = mujoco.MjModel.from_xml_path(str(SOURCE_MJCF))
JOINT_NAMES = [_model.joint(j).name for j in range(_model.njnt)
               if _model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
NUM_J = len(JOINT_NAMES)
FOOT_BODY_NAMES = ['leg_l6_link', 'leg_r6_link']  # Named end links in this model.
TRACKED_BODIES = [_model.body(b).name for b in range(1, _model.nbody)]
HEAD_JOINT_NAMES = []  # Head and sensors are fixed, verified below.
CSV_JOINT_NAMES = JOINT_NAMES.copy()
GOAL_FRAME_DIM = 15 + 2 * NUM_J
XI_DIM = 4 + NUM_J
JOINT_OFFSET = slice(1, 1 + NUM_J)
COM_OFFSET = slice(1 + NUM_J, XI_DIM)
SOURCE_FPS = 50.0  # Only datasets whose registry explicitly supplies 50 Hz.


def name_mapping(sim_names):
    names = list(sim_names)
    if len(names) != len(set(names)) or set(names) != set(JOINT_NAMES):
        raise ValueError(f'Roban joint mismatch: expected={JOINT_NAMES}, actual={names}')
    return [names.index(n) for n in JOINT_NAMES], [JOINT_NAMES.index(n) for n in names]


@lru_cache(maxsize=4)
def model_contract(path=SOURCE_MJCF):
    m = mujoco.MjModel.from_xml_path(str(path))
    free = np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1 or m.body(int(m.jnt_bodyid[free[0]])).name != ROOT_BODY:
        raise ValueError('Roban requires one floating chest base_link')
    moving = [m.joint(j).name for j in range(m.njnt) if j not in free]
    name_mapping(moving)
    if moving != JOINT_NAMES:
        raise ValueError('Roban model joint ordering differs from source')
    for name in TRACKED_BODIES + FOOT_BODY_NAMES:
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name) < 0:
            raise ValueError(f'Missing Roban body {name}')
    rows = []
    for i, name in enumerate(JOINT_NAMES):
        j = m.joint(name).id
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE or not m.jnt_limited[j]:
            raise ValueError(f'Expected limited hinge {name}')
        adr = int(m.jnt_qposadr[j])
        rows.append(dict(name=name, canonical_index=i, mj_joint_id=j, qposadr=adr,
                         dofadr=int(m.jnt_dofadr[j]), range=m.jnt_range[j].tolist(),
                         neutral=float(m.qpos0[adr]), axis=m.jnt_axis[j].tolist()))
    return m, rows


def default_joint_pos_list():
    return [r['neutral'] for r in model_contract()[1]]


def initial_root_height():
    return float(model_contract()[0].qpos0[2])


@lru_cache(maxsize=1)
def actuator_parameters():
    """Effort/armature from MJCF, velocity from URDF; PD is a sim assumption.

    Gains match the existing algorithm's documented simulation convention (not
    asserted to be hardware gains). Every actuator is mapped by joint name.
    """
    m, rows = model_contract()
    velocity = {j.get('name'): float(j.find('limit').get('velocity'))
                for j in ET.parse(URDF_PATH).findall('joint') if j.find('limit') is not None}
    result = {}
    for r in rows:
        name, j = r['name'], r['mj_joint_id']
        ids = np.flatnonzero(m.actuator_trnid[:, 0] == j)
        if len(ids) != 1 or not m.actuator_ctrllimited[ids[0]] or m.actuator_gear[ids[0], 0] != 1:
            raise ValueError(f'Expected one unit-gear torque actuator: {name}')
        effort = min(abs(m.actuator_ctrlrange[ids[0]]))
        if m.jnt_actfrclimited[j]:
            effort = min(effort, min(abs(m.jnt_actfrcrange[j])))
        armature = float(m.dof_armature[r['dofadr']])
        kp = 200.0 if name.startswith(('leg_', 'waist_')) else 80.0
        result[name] = dict(effort=float(effort), velocity=velocity[name], armature=armature,
                            stiffness=kp, damping=float(2 * np.sqrt(kp * armature)))
    return result


def foot_geometry():
    """Sole origin, toe/heel and lateral extents from the actual collision proxies.

    Roban feet contain upright cylinders and a box with a common bottom plane.
    Goal orientation is the ankle-link orientation; sole axes are parallel.
    """
    m, _ = model_contract()
    feet = []
    for name in FOOT_BODY_NAMES:
        b = m.body(name).id
        bounds = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            if not np.allclose(m.geom_quat[g], [1, 0, 0, 0]):
                raise ValueError('Rotated foot proxy requires explicit geometry support')
            size = m.geom_size[g]
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                half = size
            elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                half = np.array([size[0], size[0], size[1]])
            else:
                raise ValueError('Unsupported Roban foot collision primitive')
            bounds.append((m.geom_pos[g] - half, m.geom_pos[g] + half))
        lo = np.min([x[0] for x in bounds], axis=0)
        hi = np.max([x[1] for x in bounds], axis=0)
        sole = np.array([(lo[0]+hi[0])/2, (lo[1]+hi[1])/2, lo[2]])
        corners = np.array([[x, y, lo[2]] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])])
        feet.append(dict(body=name, sole=sole.tolist(), heel=float(lo[0]), toe=float(hi[0]),
                         corners=corners.tolist()))
    return feet


def checkpoint_contract():
    return dict(robot=ROBOT_ID, joint_names=JOINT_NAMES, tracked_bodies=TRACKED_BODIES,
                source_mjcf_sha256=hashlib.sha256(SOURCE_MJCF.read_bytes()).hexdigest(),
                foot_frames=foot_geometry())


def assert_checkpoint(ck):
    if ck.get('robot_contract') != checkpoint_contract():
        raise ValueError('Checkpoint robot contract differs from Roban; S54 weights are incompatible')
