"""S54 ankle-frame support geometry, shared with reference grounding tools."""
import itertools

import mujoco as mj
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .s54 import JOINT_NAMES, FOOT_BODY_NAMES


class FootGeometry:
    """Exact support against the horizontal z=0 plane, including visual mesh.

    Convex-hull vertices give the exact mesh support, without its interior
    vertices. Capsules use their analytic support; no bounding-box estimate.
    """
    def __init__(self, model):
        self.m = model
        self.d = mj.MjData(model)
        self.bodies = [model.body(n).id for n in FOOT_BODY_NAMES]
        self.points, self.capsules, self.collision_ids = [], [], []
        for body in self.bodies:
            points, capsules, collision_ids = [], [], []
            for g in np.flatnonzero(model.geom_bodyid == body):
                if model.geom_contype[g] or model.geom_conaffinity[g]:
                    collision_ids.append(int(g))
                kind = model.geom_type[g]
                R = Rotation.from_quat(model.geom_quat[g, [1, 2, 3, 0]]).as_matrix()
                center, size = model.geom_pos[g], model.geom_size[g]
                if kind == mj.mjtGeom.mjGEOM_MESH:
                    mesh = model.geom_dataid[g]
                    start, n = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                    vertices = model.mesh_vert[start:start + n].astype(float)
                    vertices = vertices[ConvexHull(vertices).vertices]
                    points.append(vertices @ R.T + center)
                elif kind == mj.mjtGeom.mjGEOM_BOX:
                    corners = np.array(list(itertools.product((-1., 1.), repeat=3))) * size
                    points.append(corners @ R.T + center)
                elif kind == mj.mjtGeom.mjGEOM_CAPSULE:
                    capsules.append((center.copy(), R[:, 2].copy(), float(size[1]), float(size[0])))
                else:
                    raise ValueError(f'Unsupported S54 foot geom: {g}, type={kind}')
            self.points.append(np.concatenate(points))
            self.capsules.append(capsules)
            self.collision_ids.append(collision_ids)
        self.addresses = [model.jnt_qposadr[model.joint(n).id] for n in JOINT_NAMES]

    def local_low(self, side, orientation):
        vertical = np.asarray(orientation)[2]
        low = float(np.min(self.points[side] @ vertical))
        for center, axis, half, radius in self.capsules[side]:
            low = min(low, float(center @ vertical - abs(axis @ vertical) * half - radius))
        return low

    def gaps(self, csv):
        csv = np.atleast_2d(csv)
        result = np.empty((len(csv), 2))
        for i, row in enumerate(csv):
            self.d.qpos[:] = self.m.qpos0
            self.d.qpos[:3] = row[:3]
            self.d.qpos[3:7] = row[[6, 3, 4, 5]]
            self.d.qpos[self.addresses[:len(row) - 7]] = row[7:]
            mj.mj_kinematics(self.m, self.d)
            for side, body in enumerate(self.bodies):
                result[i, side] = self.d.xpos[body, 2] + self.local_low(side, self.d.xmat[body].reshape(3, 3))
        return result

