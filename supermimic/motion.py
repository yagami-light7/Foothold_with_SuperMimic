"""Validated S54 references, resampled from metadata's 30 Hz to 50 Hz control."""

from __future__ import annotations

import glob
import os
import json
import hashlib
from pathlib import Path

import numpy as np
import torch

from .kinematics import S54Kinematics
from .s54 import JOINT_NAMES, SOURCE_FPS, GOAL_FRAME_DIM, checkpoint_contract
from .rotations import mat_to_6d, quat_slerp, quat_to_mat, so3_log, yaw_mat
from .reference_quality import validate_root_motion

CONTROL_DT = 0.02
BIN_FRAMES = 50  # 1 s bins for return-based reset sampling


class MotionLib:
    def __init__(self, files: list[str], kin: S54Kinematics, device: str, src_fps: float = SOURCE_FPS,
                 require_collision_safe: bool = True, require_kinematic_safe: bool = True):
        if not files:
            raise ValueError("No S54 reference CSVs found; run tools/data/generate_s54_reference.py --clips dance1_subject1")
        if src_fps != SOURCE_FPS:
            raise ValueError("Leju-GMR S54 source is 30 FPS")
        self.device = device
        self.names = [os.path.splitext(os.path.basename(f))[0] for f in files]
        ps, qs, js = [], [], []
        lens = []
        for f in files:
            meta = json.loads(Path(f).with_suffix('.json').read_text())
            if any(meta.get(k) != v for k, v in checkpoint_contract().items()):
                raise ValueError(f"S54 reference contract mismatch: {f}")
            if meta.get('fps') != src_fps or meta.get('quat') != 'xyzw' or meta.get('units') != 'radian':
                raise ValueError(f"Reference units/fps/quaternion mismatch: {f}")
            if meta.get('csv_sha256') != hashlib.sha256(Path(f).read_bytes()).hexdigest():
                raise ValueError(f"Reference checksum mismatch: {f}")
            if require_collision_safe:
                certificate = meta.get('self_collision_validation', {})
                required = ('continuous_interpolation', 'control_50hz', 'broadphase_no_missed_contacts',
                            'recorded_adjustments_match', 'geometry_matches_repair', 'all_joint_velocity_limits',
                            'motion_quality', 'heads_remain_neutral', 'finite_motionlib', 'foot_grounding')
                if (certificate.get('version') != 1 or certificate.get('status') != 'PASS'
                        or certificate.get('csv_sha256') != meta['csv_sha256']
                        or any(certificate.get('checks', {}).get(k) is not True for k in required)
                        or certificate.get('continuous_certificate', {}).get('continuous_joint_interpolation') is not True):
                    raise ValueError(f'Reference lacks a valid self-collision certificate: {f}. '
                                     'Regenerate with tools/data/generate_s54_reference.py into reference_startup_fixed, '
                                     'or repair existing data with tools/data/repair_s54_collisions.py.')
            values = np.loadtxt(f, delimiter=",", ndmin=2)
            if require_kinematic_safe:
                validate_root_motion(values, src_fps, f)
            raw = torch.tensor(values, dtype=torch.float32)
            if raw.ndim != 2 or raw.shape[1] != 7+len(JOINT_NAMES) or len(raw) < 2 or not torch.isfinite(raw).all():
                raise ValueError(f"Expected finite complete S54 CSV, got {raw.shape}: {f}")
            if meta.get('frames') != len(raw): raise ValueError(f"Frame count mismatch: {f}")
            if (raw[:,3:7].norm(dim=-1)-1).abs().max() > 1e-5: raise ValueError('Root quaternion not unit')
            p, q_xyzw, j = raw[:, :3], raw[:, 3:7], raw[:, 7:]
            if ((j < kin.q_lo.cpu()-1e-6) | (j > kin.q_hi.cpu()+1e-6)).any():
                raise ValueError(f"Reference joint out of range: {f}")
            quat = torch.cat((q_xyzw[:, 3:4], q_xyzw[:, :3]), -1)
            # resample to 50 Hz
            T_src = raw.shape[0]
            dur = (T_src - 1) / src_fps
            n = int(np.floor(dur / CONTROL_DT)) + 1
            t = torch.arange(n, dtype=torch.float32) * CONTROL_DT * src_fps
            i0 = torch.clamp(t.floor().long(), max=T_src - 2)
            a = (t - i0.float()).clamp(0, 1)
            p = p[i0] * (1 - a[:, None]) + p[i0 + 1] * a[:, None]
            j = j[i0] * (1 - a[:, None]) + j[i0 + 1] * a[:, None]
            quat = quat_slerp(quat[i0], quat[i0 + 1], a)
            ps.append(p), qs.append(quat), js.append(j)
            lens.append(n)
        self.clip_len = torch.tensor(lens, device=device)
        self.clip_start = torch.cumsum(torch.cat((torch.zeros(1, dtype=torch.long), torch.tensor(lens[:-1], dtype=torch.long))), 0).to(device)  # dtype: one clip gives an empty (float) tensor
        self.clip_end = self.clip_start + self.clip_len - 1  # last valid frame (inclusive)
        self.num_frames = int(sum(lens))
        self.clip_id = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(lens)]).to(device)

        root_p, root_R, root_v, root_w, jq, jqd = [], [], [], [], [], []
        for p, quat, j in zip(ps, qs, js):
            R = quat_to_mat(quat)
            v = _gradient(p, CONTROL_DT)
            qd = _gradient(j, CONTROL_DT)
            # angular velocity (world) by central differences of rotations
            w = torch.zeros_like(p)
            w[1:-1] = so3_log(R[2:] @ R[:-2].transpose(-1, -2)) / (2 * CONTROL_DT)
            w[0] = so3_log(R[1] @ R[0].T) / CONTROL_DT
            w[-1] = so3_log(R[-1] @ R[-2].T) / CONTROL_DT
            root_p.append(p), root_R.append(R), root_v.append(v), root_w.append(w), jq.append(j), jqd.append(qd)
        cat = lambda xs: torch.cat(xs).to(device)  # noqa: E731
        self.root_p, self.root_R, self.root_v, self.root_w = cat(root_p), cat(root_R), cat(root_v), cat(root_w)
        self.q, self.qd = cat(jq), cat(jqd)
        with torch.no_grad():
            bp, bR, bv, bw = [], [], [], []
            for s in range(0, self.num_frames, 65536):
                e = min(s + 65536, self.num_frames)
                out = kin.forward(self.root_p[s:e], self.root_R[s:e], self.root_v[s:e], self.root_w[s:e], self.q[s:e], self.qd[s:e])
                bp.append(out[0]), bR.append(out[1]), bv.append(out[2]), bw.append(out[3])
            self.body_p, self.body_R, self.body_v, self.body_w = torch.cat(bp), torch.cat(bR), torch.cat(bv), torch.cat(bw)
        # per-frame static goal parts: q, qd, root velocities in the root frame
        Rt = self.root_R.transpose(-1, -2)
        self.root_v_local = (Rt @ self.root_v.unsqueeze(-1)).squeeze(-1)
        self.root_w_local = (Rt @ self.root_w.unsqueeze(-1)).squeeze(-1)
        self.root_gz = self.root_R[:, 2, 2]  # -(projected gravity z) of the reference root: g_b = R^T (0,0,-1)
        # 1 s bins
        bin_clip, bin_start = [], []
        for c in range(len(lens)):
            nb = int(np.ceil(lens[c] / BIN_FRAMES))
            for b in range(nb):
                bin_clip.append(c)
                bin_start.append(int(self.clip_start[c]) + b * BIN_FRAMES)
        self.bin_clip = torch.tensor(bin_clip, device=device)
        self.bin_start = torch.tensor(bin_start, device=device)
        self.num_bins = len(bin_start)
        self.frame_bin = torch.zeros(self.num_frames, dtype=torch.long, device=device)
        for b in range(self.num_bins):
            c = bin_clip[b]
            s = bin_start[b]
            e = min(s + BIN_FRAMES, int(self.clip_end[c]) + 1)
            self.frame_bin[s:e] = b

    @staticmethod
    def from_dir(path: str, kin: S54Kinematics, device: str) -> MotionLib:
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        return MotionLib(files, kin, device)

    def future_frames(self, f: torch.Tensor, K: int) -> torch.Tensor:
        """Frames f+1..f+K clamped to the clip's last frame, shape (B, K)."""
        end = self.clip_end[self.clip_id[f]]
        fut = f.unsqueeze(-1) + torch.arange(1, K + 1, device=f.device)
        return torch.minimum(fut, end.unsqueeze(-1))

    def goal(self, f: torch.Tensor, K: int) -> torch.Tensor:
        """Goal features from the reference only: K future frames relative to the current reference frame."""
        fut = self.future_frames(f, K)  # (B,K)
        H = yaw_mat(self.root_R[f])  # (B,3,3)
        Ht = H.transpose(-1, -2).unsqueeze(1)
        dp = (Ht @ (self.root_p[fut] - self.root_p[f].unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
        rot = mat_to_6d(Ht @ self.root_R[fut])
        feats = torch.cat(
            (dp, rot, self.root_v_local[fut], self.root_w_local[fut], self.q[fut], self.qd[fut] * 0.1), -1
        )  # (B,K,GOAL_FRAME_DIM)
        assert feats.shape[-1] == GOAL_FRAME_DIM
        return feats.reshape(f.shape[0], -1)

    def ref_state(self, f: torch.Tensor):
        return self.root_p[f], self.root_R[f], self.root_v[f], self.root_w[f], self.q[f], self.qd[f]


def _gradient(x: torch.Tensor, dt: float) -> torch.Tensor:
    g = torch.zeros_like(x)
    g[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    g[0] = (x[1] - x[0]) / dt
    g[-1] = (x[-1] - x[-2]) / dt
    return g
