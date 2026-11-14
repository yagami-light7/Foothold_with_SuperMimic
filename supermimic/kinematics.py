"""Differentiable compiled-MJCF FK, including fixed bodies, joint anchors and reference angles.

All quaternions internal to this package are wxyz. Root velocities are world velocities of
base_link's origin. No Isaac Lab imports are needed for reference or algorithm tests.
"""
from __future__ import annotations
import mujoco
import numpy as np
import torch
from .robot_contract import JOINT_NAMES, SOURCE_MJCF, TRACKED_BODIES, model_contract
from .rotations import axis_angle_to_mat, quat_to_mat


class RobotKinematics:
    def __init__(self, device='cpu', tracked=None, model_path=SOURCE_MJCF, dtype=torch.float32):
        m, rows = model_contract(model_path)
        self.device = device
        self.joint_names = list(JOINT_NAMES)
        self.body_names = [m.body(b).name for b in range(1, m.nbody)]
        self.num_bodies = len(self.body_names)
        self.parent = [int(m.body_parentid[b])-1 for b in range(1, m.nbody)]
        self.joint_idx = []
        axes, anchors, refs = [], [], []
        for b in range(1, m.nbody):
            if m.body_jntnum[b] > 1: raise ValueError(f'Multi-joint body unsupported: {m.body(b).name}')
            j = int(m.body_jntadr[b]) if m.body_jntnum[b] else -1
            if j >= 0 and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                assert self.parent[b-1] == -1 and b == 1
                self.joint_idx.append(-2)
            elif j >= 0:
                assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
                self.joint_idx.append(JOINT_NAMES.index(m.joint(j).name))
            else:
                self.joint_idx.append(-1)
            axes.append(m.jnt_axis[j].copy() if j >= 0 else [0, 0, 1])
            anchors.append(m.jnt_pos[j].copy() if j >= 0 else [0, 0, 0])
            refs.append(m.qpos0[m.jnt_qposadr[j]] if j >= 0 and self.joint_idx[-1] >= 0 else 0)
        tensor = lambda x: torch.tensor(np.array(x), device=device, dtype=dtype)
        self.axes, self.anchors, self.q_ref = tensor(axes), tensor(anchors), tensor(refs)
        self.off_p = tensor(m.body_pos[1:])
        self.off_R = quat_to_mat(tensor(m.body_quat[1:]))
        self.tracked_names = list(TRACKED_BODIES if tracked is None else tracked)
        self.tracked_idx = [self.body_names.index(n) for n in self.tracked_names]
        self.q_lo = tensor([r['range'][0] for r in rows]); self.q_hi = tensor([r['range'][1] for r in rows])

    def forward(self, p, R, v, w, q, qd, all_bodies=False):
        if q.shape[-1] != len(JOINT_NAMES) or qd.shape != q.shape:
            raise ValueError('Expected canonical robot joint position/velocity vectors')
        P, Rs, V, W = [], [], [], []
        for i, par in enumerate(self.parent):
            j = self.joint_idx[i]
            if j == -2:
                pi, Ri, vi, wi = p, R, v, w
            else:
                if par < 0: raise ValueError('World-fixed body outside floating subtree')
                Rp, pp = Rs[par], P[par]
                R0 = Rp @ self.off_R[i]
                pi = pp + (Rp @ self.off_p[i].unsqueeze(-1)).squeeze(-1)
                if j >= 0:
                    turn = axis_angle_to_mat(self.axes[i], q[..., j]-self.q_ref[i])
                    Ri = R0 @ turn
                    offset = self.anchors[i] - (turn @ self.anchors[i].unsqueeze(-1)).squeeze(-1)
                    pi = pi + (R0 @ offset.unsqueeze(-1)).squeeze(-1)
                    omega = (R0 @ self.axes[i].unsqueeze(-1)).squeeze(-1) * qd[..., j:j+1]
                    wi = W[par] + omega
                    vi = (V[par] + torch.cross(W[par], pi-pp, dim=-1)
                          - torch.cross(omega, (Ri @ self.anchors[i].unsqueeze(-1)).squeeze(-1), dim=-1))
                else:
                    Ri, wi = R0, W[par]
                    vi = V[par] + torch.cross(W[par], pi-pp, dim=-1)
            P.append(pi); Rs.append(Ri); V.append(vi); W.append(wi)
        idx = range(self.num_bodies) if all_bodies else self.tracked_idx
        return tuple(torch.stack([arr[i] for i in idx], dim) for arr,dim in ((P,-2),(Rs,-3),(V,-2),(W,-2)))


# Historical tools retain their public name and the default S54 contract.
S54Kinematics = RobotKinematics
