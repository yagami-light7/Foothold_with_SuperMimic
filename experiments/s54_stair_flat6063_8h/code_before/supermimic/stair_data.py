"""T1.5 final-export audit. No implicit split changes or reference recentering."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
import trimesh

from . import m34
from .kinematics import S54Kinematics
from .rotations import quat_to_mat, quat_slerp, mat_to_quat, so3_log
from .s54 import JOINT_NAMES, FOOT_BODY_NAMES, SOURCE_MJCF, MJCF_PATH, REPO


def geometry_tools():
    path = REPO/'scripts/terrain/terrain_render_diagnostics.py'
    spec = importlib.util.spec_from_file_location('s54_terrain_geometry_local', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_stairs(bundle, kin, device='cpu', ids=None):
    # The upstream bundle is explicitly evaluation-only. A later training
    # designation must be a separate, hashed experiment manifest, never mutation.
    motion = m34.motion_library(bundle, kin, device, split='test', ids=ids)
    meta, manifest, _ = m34.entries(bundle, 'test')
    contacts = []
    for row in motion.rows:
        data = m34.load(bundle, row, meta, manifest)
        contacts.append(np.stack([data['left_contact'], data['right_contact']], -1))
    motion.contacts = torch.as_tensor(np.concatenate(contacts), device=device)
    return motion


def sample_state(motion, clip, times_s):
    """Linear XYZ/joints/velocities and shortest-path quaternion SLERP.

    At the native 50 Hz control clock this returns the exact exported samples.
    Source-retiming times are provenance, not the playback clock.
    """
    device = motion.root_p.device
    t = torch.as_tensor(times_s, device=device, dtype=torch.float32).reshape(-1)
    n = int(motion.clip_len[clip])
    if (t < 0).any() or (t > (n-1)*.02+1e-6).any():
        raise ValueError('Reference time outside this clip')
    u = (t/.02).clamp(0, n-1)
    index = u.floor().long().clamp(max=n-2)
    alpha = (u-index).clamp(0, 1)
    index += motion.clip_start[clip]
    blend = lambda x: x[index]*(1-alpha[:, None])+x[index+1]*alpha[:, None]
    quaternion = mat_to_quat(motion.root_R)
    R = quat_to_mat(quat_slerp(quaternion[index], quaternion[index+1], alpha))
    return blend(motion.root_p), R, blend(motion.root_v), blend(motion.root_w), blend(motion.q), blend(motion.qd)


def pose_collision_audit(data, terrain_mesh):
    """Independent MuJoCo pose collision diagnostic, NOT a dynamics rollout.

    Use the actual simulator proxy geometry, with exact component boxes from the
    source mesh. Self collisions use the source model's authored exclusions.
    This does not change the Newton scene or its contact pipeline.
    """
    spec = mujoco.MjSpec.from_file(str(MJCF_PATH))
    for geom in list(spec.worldbody.geoms):
        spec.delete(geom)
    for i, component in enumerate(terrain_mesh.split(only_watertight=False)):
        if not component.is_watertight:
            continue
        if len(component.faces) != 12 or len(component.vertices) != 8:
            raise ValueError('Unsupported solid for independent pose diagnostic')
        lo, hi = component.bounds
        spec.worldbody.add_geom(name=f'audit_tread_{i}', type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(hi-lo)/2, pos=(hi+lo)/2, contype=1, conaffinity=1)
    model = spec.compile()
    state = mujoco.MjData(model)
    joint_adr = [model.jnt_qposadr[model.joint(n).id] for n in JOINT_NAMES]
    # Source proxies carry exclusions; explicitly keep non-adjacent robot pairs
    # enabled for this offline diagnostic, unlike disabled training self contact.
    rows = []
    for frame in range(len(data['joint_pos'])):
        state.qpos[:3] = data['body_pos_w'][frame, 0]
        state.qpos[3:7] = data['body_quat_w'][frame, 0]
        state.qpos[joint_adr] = data['joint_pos'][frame]
        mujoco.mj_forward(model, state)
        terrain_depth = self_depth = 0.
        for contact in state.contact[:state.ncon]:
            if contact.dist >= 0:
                continue
            b0, b1 = model.geom_bodyid[[contact.geom1, contact.geom2]]
            if b0 == 0 or b1 == 0:
                terrain_depth = max(terrain_depth, -float(contact.dist))
            else:
                self_depth = max(self_depth, -float(contact.dist))
        rows.append(dict(frame=frame, terrain_proxy_penetration_m=terrain_depth,
                         self_proxy_penetration_m=self_depth))
    return rows


def audit_bundle(bundle, terrain_asset, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    bundle, terrain_asset = Path(bundle), Path(terrain_asset)
    meta, manifest, rows = m34.entries(bundle, 'test')
    terrain_meta = json.loads((terrain_asset/'manifest.json').read_text())
    mesh_hash = m34.sha256(terrain_asset/'terrain_mesh.npz')
    if mesh_hash != terrain_meta['mesh_sha256'] or m34.sha256(bundle/'terrain/terrain_mesh.npz') != mesh_hash:
        raise ValueError('Authoritative mesh differs from the Newton terrain')
    tools = geometry_tools(); terrain = tools.load_mesh(terrain_asset)
    kin = S54Kinematics('cpu')
    motion = load_stairs(bundle, kin)
    report = dict(status='initializing', bundle=str(bundle.resolve()), terrain=str(terrain_asset.resolve()),
        mesh_sha256=mesh_hash, root_anchor='unchanged authoritative mesh world; identity import transform',
        control_dt=.02, physics_substep_dt=.005, resampling='none at 50 Hz; explicit linear/SLERP helper for other times',
        source_world_transform='already applied upstream; never applied again',
        source_split='test', train_normalization_uses_test=False, rows=[])
    if not (bundle/meta.get('terrain_manifest', 'terrain/manifest.json')).exists():
        report['upstream_missing_terrain_manifest'] = 'Resolved explicitly with matching authoritative local asset manifest; no source files modified'
    for clip, row in enumerate(motion.rows):
        data = m34.load(bundle, row, meta, manifest)
        n = len(data['joint_pos'])
        if str(data['terrain_mesh_sha256'].item()) != mesh_hash or row['terrain_mesh_sha256'] != mesh_hash:
            raise ValueError('Final NPZ terrain hash mismatch')
        contact = np.stack([data['left_contact'], data['right_contact']], -1)
        if contact.shape != (n, 2) or contact.dtype != bool:
            raise ValueError('Invalid exported contact schedule')
        touchdown = np.flatnonzero((~contact[:-1] & contact[1:]).any(-1))+1
        last = int(touchdown[-1]) if len(touchdown) else 0
        ds = contact.all(-1)
        initial = int(np.flatnonzero(~ds)[0]) if (~ds).any() else n
        terminal = n-last
        if initial < 31 or terminal < 31 or not ds[last:].all() or bool(data['terminal_stance_truncated'].item()):
            raise ValueError('Missing full initial/terminal support interval')
        feet = [meta['body_names'].index(name) for name in FOOT_BODY_NAMES]
        rotations = quat_to_mat(torch.tensor(data['body_quat_w'][:, feet], dtype=torch.float64)).numpy()
        sole = data['body_pos_w'][:, feet] + (rotations @ data['sole_offset_local'][None, None, :, None])[..., 0]
        height = tools.top_surface(terrain, sole[..., :2].reshape(-1, 2)).reshape(n, 2)
        clearance = sole[..., 2]-height
        if not np.isfinite(clearance).all():
            raise ValueError('Reference leaves the terrain coverage')
        collisions = pose_collision_audit(data, terrain)
        terrain_depth = max(r['terrain_proxy_penetration_m'] for r in collisions)
        self_depth = max(r['self_proxy_penetration_m'] for r in collisions)
        # Same strict sampled tolerances as export, applied to final joints again.
        if terrain_depth > .001 or self_depth > .002:
            raise ValueError(f'Final-joint proxy penetration: {row["id"]}, {terrain_depth}, {self_depth}')
        fk = next(v for v in motion.validation if v['id'] == row['id'])
        metrics = dict(id=row['id'], frames=n, duration_s=(n-1)*.02,
            quaternion_order='wxyz; convert exactly once to IsaacLab xyzw at state writes',
            joint_names_identical=meta['joint_names'] == JOINT_NAMES,
            body_names_identical=meta['body_names'] == kin.body_names,
            all_body_fk_and_velocity_max_errors=fk['max_errors'],
            root_start=data['body_pos_w'][0,0].tolist(), root_end=data['body_pos_w'][-1,0].tolist(),
            initial_double_support_frames=initial, last_touchdown_frame=last,
            final_support_frames=terminal, touchdown_frames=touchdown.tolist(),
            source_time_strictly_increasing=bool((np.diff(data['source_times_s']) > 0).all()),
            contact_sole_height_error_max_m=float(np.max(np.abs(clearance[contact]))),
            swing_sole_min_clearance_m=float(clearance[~contact].min()),
            terrain_proxy_penetration_max_m=terrain_depth, self_proxy_penetration_max_m=self_depth,
            joint_velocity_finite_difference_max_error=float(np.max(np.abs(np.gradient(data['joint_pos'],.02,axis=0)-data['joint_vel']))))
        if metrics['contact_sole_height_error_max_m'] > .005:
            raise ValueError('Exported stance sole is not at the source tread')
        with (out/f'{row["id"]}_frames.jsonl').open('w') as stream:
            for f, collision in enumerate(collisions):
                stream.write(json.dumps(dict(**collision, time_s=f*.02, contact=contact[f].tolist(),
                    sole_world=sole[f].tolist(), tread_height_m=height[f].tolist(), sole_clearance_m=clearance[f].tolist()))+'\n')
        report['rows'].append(metrics)
    report['status'] = 'PASS'
    (out/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
