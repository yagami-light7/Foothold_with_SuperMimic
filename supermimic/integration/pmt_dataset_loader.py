"""Strict, read-only consumption of a complete PMT M6 sample.

The manifest owns file discovery and checksums. No PMT Python modules are
imported. Planned ankle targets, final IK poses and measured contacts remain
distinct data products. Offline quality is never a dynamics certificate.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..roban import (JOINT_NAMES, TRACKED_BODIES, ROOT_BODY, FOOT_BODY_NAMES,
                     SOURCE_MJCF, URDF_PATH, foot_geometry, model_contract)
from ..rotations import quat_to_mat

REQUIRED = ('terrain.json', 'terrain_mesh.npz', 'height_map.npz', 'footsteps.json',
            'foot_trajectory.npz', 'body_reference.npz', 'contact_schedule.npz',
            'pillar_events.json', 'pillar_pose_schedule.npz', 'quality.json')
BOX_FACES = np.array([[0,1,2],[1,3,2],[4,6,5],[5,6,7],[0,4,1],[1,4,5],
                      [2,3,6],[3,7,6],[0,2,4],[2,6,4],[1,5,3],[3,5,7]])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def rotation(quaternions):
    return quat_to_mat(torch.as_tensor(quaternions, dtype=torch.float64)).numpy()


def rotate_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


@dataclass
class PMTSample:
    directory: Path
    manifest: dict
    terrain: dict
    mesh: dict
    height_map: dict
    footsteps: dict
    foot: dict
    body: dict
    contact: dict
    schedule: dict
    events: list
    quality: dict
    alignment: dict

    @property
    def name(self):
        return self.manifest['scene_id']

    @property
    def frames(self):
        return len(self.body['time_s'])

    @property
    def dynamic(self):
        return self.manifest['mode'] == 'dynamic_three_pillars'

    @property
    def sizes(self):
        return np.ptp(self.mesh['vertices_local'], axis=1)

    @property
    def planned_sole(self):
        offsets = np.array([f['sole'] for f in foot_geometry()])
        return self.foot['foot_pos_w'] + np.einsum('tfij,fj->tfi', rotation(self.foot['foot_quat_w']), offsets)

    def pillar_layout(self, frame, rotation_w=None, translation=None):
        require(isinstance(frame, (int, np.integer)) and 0 <= frame < self.frames, 'Pillar frame outside sample')
        poses = self.schedule['pose_xyzyaw'][frame].copy()
        if rotation_w is not None or translation is not None:
            require(rotation_w is not None and translation is not None, 'Anchor requires rotation and translation')
            A, t = np.asarray(rotation_w), np.asarray(translation)
            require(A.shape == (3,3) and t.shape == (3,) and np.isfinite(A).all() and np.isfinite(t).all(), 'Invalid scene anchor')
            require(np.allclose(A.T @ A, np.eye(3), atol=1e-7) and np.linalg.det(A)>0 and np.allclose(A[2], [0,0,1], atol=1e-7), 'Pillar anchor must be a rigid Z-up yaw transform')
            poses[:,:3] = poses[:,:3] @ A.T + t
            poses[:,3] += np.arctan2(A[1,0], A[0,0])
        return poses, self.schedule['active'][frame].copy(), self.schedule['state'][frame].copy()


def load_sample(directory, *, catalog_entry=None):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    files = manifest['files']
    require(all(k in files for k in REQUIRED), 'Incomplete sample manifest')
    for name, expected in files.items():
        path = directory/name
        require(path.resolve().is_relative_to(directory.resolve()), f'Unsafe manifest path: {name}')
        require(path.is_file() and sha256(path) == expected, f'Manifest SHA256 mismatch: {name}')
    if catalog_entry is not None:
        require(manifest == {k:v for k,v in catalog_entry.items() if k not in ()}, 'Catalog/sample manifest mismatch')
    require(manifest['accepted'] is True and manifest['status']=='accepted', 'Sample failed offline acceptance')
    require(manifest['mode'] in ('static_pillars','dynamic_three_pillars'), 'Unsupported PMT mode')
    require(manifest['model'] == sha256(SOURCE_MJCF), 'Roban model SHA256 mismatch')
    schema = manifest['schema']
    require(schema['fps']==50. and schema['quaternion']=='wxyz' and schema['dof']==21 and schema['bodies']==37 and schema['units']=='m,rad,s', 'Dataset schema mismatch')
    require(manifest['time_sync']=='50 Hz shared absolute episode time', 'Missing explicit reference clock')
    read = lambda n: json.loads((directory/n).read_text())
    terrain, footsteps, events, quality = (read(n) for n in ('terrain.json','footsteps.json','pillar_events.json','quality.json'))
    mesh, height, foot, body, contact, schedule = (arrays(directory/n) for n in ('terrain_mesh.npz','height_map.npz','foot_trajectory.npz','body_reference.npz','contact_schedule.npz','pillar_pose_schedule.npz'))
    n, count = schema['frames'], len(mesh['pillar_ids'])
    require(n>=3 and quality['frames']==n and quality['fps']==50., 'Frame count/quality clock mismatch')
    require(quality['accepted'] is True and all(v is True for v in quality['checks'].values()), 'Untrusted offline quality')
    require(terrain['frame']=='world_Z_up' and terrain['units']=='m' and terrain['mode']==manifest['mode'] and terrain['ground_plane'] is False and terrain['origin_world']==[0,0,0], 'Terrain frame/ground mismatch')
    require(terrain['mesh_sha256']==files['terrain_mesh.npz'], 'Terrain mesh hash mismatch')
    if 'schedule_sha256' in terrain:
        require(terrain['schedule_sha256']==files['pillar_pose_schedule.npz'], 'Terrain schedule hash mismatch')
    times=np.arange(n)*.02
    for name, data in [('body',body),('foot',foot),('contact',contact),('pillar',schedule)]:
        require(data['time_s'].shape==(n,) and np.allclose(data['time_s'],times,atol=1e-12,rtol=0), f'{name} 50 Hz timeline mismatch')
        for k,value in data.items():
            if np.issubdtype(value.dtype,np.number):
                require(np.isfinite(value).all(), f'Nonfinite {name}.{k}')
        if name!='pillar':
            require(float(data['fps'])==50. and str(data['scene_id'])==manifest['scene_id'], f'{name} scene/FPS mismatch')
            require(str(data['model_sha256'])==manifest['model'] and str(data['coordinate_frame'])=='world_Z_up_metres' and str(data['quaternion_order'])=='wxyz', f'{name} model/frame/quaternion mismatch')
            require(str(data['terrain_mesh_sha256'])==files['terrain_mesh.npz'], f'{name} terrain hash mismatch')
            require(np.array_equal(data['episode_anchor_w'], [0.,0.,0.]), f'{name} unsupported source anchor')
    require(list(body['joint_names'])==JOINT_NAMES, 'Joint names/order mismatch')
    require(list(body['body_names'])==TRACKED_BODIES, 'Body names/order mismatch')
    require(str(body['urdf_sha256'])==sha256(URDF_PATH), 'URDF SHA256 mismatch')
    require(str(body['root_body_name'])==ROOT_BODY, 'Root is not chest base_link')
    require(str(body['schema_version'])=='roban_tcrs_v1', 'Unknown body schema')
    offsets=np.array([f['sole'] for f in foot_geometry()])
    for data in (body,foot):
        require([str(data['left_foot_body_name']),str(data['right_foot_body_name'])]==FOOT_BODY_NAMES, 'Foot body order mismatch')
        require(data['sole_offset_local'].shape==(3,) and np.allclose(data['sole_offset_local'],offsets,atol=1e-12,rtol=0), 'Ankle/sole contract mismatch')
    require(str(foot['foot_frame'])=='named ankle body origins; sole_offset_local transforms to sole centre', 'Unknown planned foot frame')
    shapes={'root_pos':(n,3),'root_quat':(n,4),'joint_pos':(n,21),'joint_vel':(n,21),
            'body_pos_w':(n,37,3),'body_quat_w':(n,37,4),'body_lin_vel_w':(n,37,3),'body_ang_vel_w':(n,37,3),
            'contact':(n,2),'contact_valid':(n,2),'swing_phase':(n,2),'quality':(n,)}
    for k,shape in shapes.items():require(body[k].shape==shape, f'body.{k} shape mismatch')
    for k,shape in {'foot_pos_w':(n,2,3),'foot_quat_w':(n,2,4),'actual_fk_foot_pos_w':(n,2,3),'actual_fk_foot_quat_w':(n,2,4),'contact_mask':(n,2),'phase':(n,2)}.items():
        require(foot[k].shape==shape,f'foot.{k} shape mismatch')
    for value in (body['root_quat'],body['body_quat_w'],foot['foot_quat_w'],foot['actual_fk_foot_quat_w']):
        require(np.allclose(np.linalg.norm(value,axis=-1),1.,atol=1e-7,rtol=0), 'Unnormalized WXYZ quaternion')
    for data in (foot,contact,schedule):
        require(data['contact_mask'].dtype==np.bool_ and np.array_equal(data['contact_mask'],body['contact']), 'Planned contact disagreement')
    require(body['contact_valid'].dtype==np.bool_ and contact['contact_valid'].dtype==np.bool_ and not body['contact_valid'].any() and not contact['contact_valid'].any(), 'Offline contact must not be promoted to measured truth')
    require(np.array_equal(foot['phase'],contact['phase']) and np.all((foot['phase']>=0)&(foot['phase']<=1)), 'Phase mismatch')
    # M6 exports TWO documented phase semantics: foot/contact phase advances
    # during stance AND swing; legacy body.swing_phase is zero during stance.
    # Verify this exact relationship rather than treating them as interchangeable.
    require(np.array_equal(np.where(body['contact'],0.,foot['phase']),body['swing_phase']), 'Legacy swing-only body phase mismatch')
    require(np.array_equal(body['quality'],np.ones(n)), 'Offline quality field mismatch')
    root=TRACKED_BODIES.index(ROOT_BODY);fi=[TRACKED_BODIES.index(f) for f in FOOT_BODY_NAMES]
    require(np.array_equal(body['root_pos'],body['body_pos_w'][:,root]) and np.array_equal(body['root_quat'],body['body_quat_w'][:,root]), 'Chest/root arrays disagree')
    limits=np.array([r['range'] for r in model_contract()[1]])
    require(np.all(body['joint_pos']>=limits[:,0]-1e-6) and np.all(body['joint_pos']<=limits[:,1]+1e-6), 'Reference exceeds joint limits')
    require(np.allclose(body['joint_limits'],limits,atol=1e-12,rtol=0), 'Exported joint limits differ')
    require(mesh['vertices_local'].shape==(count,8,3) and mesh['faces'].shape==(count,12,3) and np.array_equal(mesh['pillar_ids'],np.arange(count)), 'Mesh array shape/id mismatch')
    require(np.isfinite(mesh['vertices_local']).all() and np.array_equal(mesh['faces'],np.broadcast_to(BOX_FACES,(count,12,3))), 'Unsupported mesh topology/winding')
    sizes=np.ptp(mesh['vertices_local'],axis=1)
    corners=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    require(np.all(sizes>0) and np.allclose(mesh['vertices_local'],corners[None]*sizes[:,None]/2,atol=1e-12,rtol=0), 'Mesh is not an authoritative centered closed box')
    for k,shape in {'pose_xyzyaw':(n,count,4),'active':(n,count),'state':(n,count)}.items():require(schedule[k].shape==shape,f'Pillar {k} shape mismatch')
    require(schedule['active'].dtype==np.bool_ and set(np.unique(schedule['state'])) <= {'FREE','RELOCATING','RESERVED_TARGET','SUPPORTING'}, 'Invalid pillar state')
    require(not np.any((schedule['state']=='SUPPORTING')&~schedule['active']), 'Inactive planned support')
    require(np.array_equal(~schedule['active'],schedule['state']=='RELOCATING'), 'Inactive/recycling state mismatch')
    if manifest['mode']=='static_pillars':
        require(count==6 and events==[] and schedule['active'].all() and np.array_equal(schedule['pose_xyzyaw'],np.broadcast_to(schedule['pose_xyzyaw'][0],(n,count,4))), 'Static schedule changes')
    else:require(count==3 and len(events)==4, 'Dynamic three-pillar event count mismatch')
    covered=np.zeros((n,count),dtype=bool)
    for e in events:
        start,end=[int(round(e[k]/.02)) for k in ('timestamp','end_time_s')]
        require(abs(start*.02-e['timestamp'])<1e-12 and abs(end*.02-e['end_time_s'])<1e-12, 'Event off control clock')
        pid=e['pillar_id']
        require(e['event_type']=='FREE_RELOCATION' and 0<start<end<e['target_liftoff_frame']<e['target_touchdown_frame']<n and 0<=pid<count, 'Invalid relocation/event deadline')
        require(not covered[start:end,pid].any(), 'Overlapping events')
        covered[start:end,pid]=True
        require(np.allclose(schedule['pose_xyzyaw'][start-1,pid],e['pose_before'],atol=1e-12,rtol=0) and np.allclose(schedule['pose_xyzyaw'][end,pid],e['pose_after'],atol=1e-12,rtol=0), 'Event pose/schedule mismatch')
    require(np.array_equal(covered,~schedule['active']), 'Inactive intervals do not match event table')
    changes=np.any(np.diff(schedule['pose_xyzyaw'],axis=0)!=0,axis=-1)
    require(not np.any(changes & schedule['active'][1:] & schedule['active'][:-1]), 'Active pillar moved')
    h,valid=height['height_m'],height['valid']
    require(h.shape==(len(height['y_m']),len(height['x_m'])) and valid.shape==h.shape and valid.dtype==np.bool_, 'Height map shape mismatch')
    require(str(height['axes'])=='height[y,x]' and float(height['time_s'])==0. and np.all(np.isfinite(h[valid])) and np.all(np.isnan(h[~valid])), 'Height map void/axis semantics mismatch')
    require(np.allclose(np.diff(height['x_m']),float(height['resolution_m']),atol=1e-12) and np.allclose(np.diff(height['y_m']),float(height['resolution_m']),atol=1e-12) and np.allclose(height['origin_xy_m'],[height['x_m'][0],height['y_m'][0]],atol=1e-12), 'Height map coordinates mismatch')
    # Independent final-IK FK audit against the source MJCF; no PMT code imports.
    import mujoco
    m, rows=model_contract();d=mujoco.MjData(m);bp=[];bR=[];bv=[];bw=[]
    ids=[m.body(name).id for name in TRACKED_BODIES]
    for t in range(n):
        d.qpos[:3]=body['root_pos'][t];d.qpos[3:7]=body['root_quat'][t]
        for r,q in zip(rows,body['joint_pos'][t]):d.qpos[r['qposadr']]=q
        mujoco.mj_forward(m,d);bp.append(d.xpos[ids].copy());bR.append(d.xmat[ids].reshape(-1,3,3).copy())
    bp,bR=np.array(bp),np.array(bR)
    fkerr=float(np.linalg.norm(bp-body['body_pos_w'],axis=-1).max())
    roterr=float(np.max(abs(bR-rotation(body['body_quat_w']))))
    require(fkerr<1e-7 and roterr<1e-7, f'Final IK body FK mismatch: {fkerr}, {roterr}')
    require(np.max(abs(bp[:,fi]-foot['actual_fk_foot_pos_w']))<1e-7 and np.max(abs(bR[:,fi]-rotation(foot['actual_fk_foot_quat_w'])))<1e-7, 'Exported actual foot FK mismatch')
    planned_R=rotation(foot['foot_quat_w'])
    errors=np.linalg.norm(bp[:,fi]-foot['foot_pos_w'],axis=-1)
    angles=np.arccos(np.clip((np.einsum('tfij,tfij->tf',bR[:,fi],planned_R)-1)/2,-1,1))
    config=json.loads(str(body['generator_config_json']))
    require(errors.max()<=config['acceptance']['foot_ik_error_m'] and angles.max()<=np.deg2rad(.5), 'Planned/FK foot error exceeds certified quality gates')
    require(np.max(abs(foot['foot_pos_w']-body['foot_target_pos_w']))<1e-12 and np.max(abs(foot['foot_quat_w']-body['foot_target_quat_w']))<1e-12, 'Body/foot planner target disagreement')
    for g in footsteps['goals']:
        require(0<=g['frame']<n and abs(g['time_s']-g['frame']*.02)<1e-12 and 0<=g['pillar_id']<count, 'Footstep time/pillar mismatch')
    alignment=dict(status='PASS',scene_id=manifest['scene_id'],frames=n,fps=50.,dt=.02,
        model_sha256=manifest['model'],manifest_sha256=sha256(directory/'manifest.json'),mesh_sha256=files['terrain_mesh.npz'],
        schedule_sha256=files['pillar_pose_schedule.npz'],joint_order=JOINT_NAMES,body_order=TRACKED_BODIES,
        body_fk_max_m=fkerr,body_rotation_matrix_max_error=roterr,planned_ankle_fk_max_error_m=float(errors.max()),
        planned_foot_orientation_max_error_deg=float(np.rad2deg(angles.max())),contact_valid_count=0,
        planned_contact_semantics='command only, never Newton measured truth',quality_semantics='offline geometry/kinematics only',
        phase_semantics='Actor uses foot/contact stance+swing phase; body legacy swing_phase is explicitly zero during stance',
        dynamic_events=len(events),verified_files=len(files),foot_goal_source='planned_ankle_to_model_sole')
    return PMTSample(directory,manifest,terrain,mesh,height,footsteps,foot,body,contact,schedule,events,quality,alignment)


def load_catalog(root, *, mode=None, seed=None, split='smoke'):
    root=Path(root)
    catalog=json.loads((root/'dataset_manifest.json').read_text())
    require(catalog['version']==1, 'Unknown dataset manifest version')
    rows=[r for r in catalog['samples'] if (mode is None or r['mode']==mode) and (seed is None or r['seed']==seed) and r['split']==split]
    require(bool(rows), 'No catalog samples match the selection')
    require(len({r['scene_id'] for r in rows})==len(rows), 'Duplicate scene identity')
    samples=[]
    for row in rows:
        path=root/row['path']
        require(path.resolve().is_relative_to(root.resolve()), 'Catalog path leaves dataset root')
        samples.append(load_sample(path,catalog_entry=row))
    return samples


class PMTMotionReferenceProvider:
    """Factory preserving the existing reference API, clocks and 23D goal schema."""
    def __new__(cls, samples, kin, device='cpu', *, trust_offline_body_prior=False):
        from ..motion_reference import MotionReferenceProvider
        from ..motion import _gradient
        require(bool(samples), 'No PMT samples')
        motion=MotionReferenceProvider([s.directory/'body_reference.npz' for s in samples],kin,device,
                                       clip_names=[s.name for s in samples])
        tensor=lambda values,dtype=torch.float32:torch.as_tensor(np.concatenate(values),device=device,dtype=dtype)
        motion.qd=tensor([s.body['joint_vel'] for s in samples])
        root=TRACKED_BODIES.index(ROOT_BODY)
        motion.root_v=tensor([s.body['body_lin_vel_w'][:,root] for s in samples])
        motion.root_w=tensor([s.body['body_ang_vel_w'][:,root] for s in samples])
        with torch.no_grad():
            motion.body_p,motion.body_R,motion.body_v,motion.body_w=kin.forward(*motion.ref_state(torch.arange(motion.num_frames,device=device)))
            motion.foot_p,motion.fk_foot_R,motion.fk_foot_v=motion.feet(motion.ref_state(torch.arange(motion.num_frames,device=device)))
        motion.target_foot_p=tensor([s.planned_sole for s in samples])
        motion.stance_p=motion.target_foot_p.clone()
        motion.foot_R=quat_to_mat(tensor([s.foot['foot_quat_w'] for s in samples]))
        motion.foot_v=torch.cat([_gradient(torch.as_tensor(s.planned_sole,dtype=torch.float32),.02) for s in samples]).to(device)
        motion.contact=tensor([s.foot['contact_mask'] for s in samples],torch.bool)
        motion.contact_valid=torch.zeros_like(motion.contact)
        motion.phase=tensor([s.foot['phase'] for s in samples])
        motion.quality=tensor([s.body['quality'] for s in samples]) if trust_offline_body_prior else torch.zeros(motion.num_frames,device=device)
        Rt=motion.root_R.transpose(-1,-2)
        motion.root_v_local=(Rt@motion.root_v[...,None]).squeeze(-1)
        motion.root_w_local=(Rt@motion.root_w[...,None]).squeeze(-1)
        motion.pmt_samples=tuple(samples)
        motion.foot_goal_source='pmt_planned_ankle_to_sole'
        motion.validation.update(pmt_samples=[s.alignment for s in samples],foot_goal_source=motion.foot_goal_source,
            body_prior_quality='explicitly trusted offline kinematic upper-body quality' if trust_offline_body_prior else 'disabled: offline quality is not automatically trusted for auxiliary prior',
            velocity_source='same PMT body/joint sample; no cross-clip derivative')
        return motion
