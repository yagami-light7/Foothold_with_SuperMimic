"""CPU-safe data contracts and offline foot/contact extraction from S54 MotionLib."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .motion import CONTROL_DT, MotionLib
from .s54 import FOOT_BODY_NAMES, JOINT_NAMES, COM_BODY, checkpoint_contract, model_contract
from .foot_geometry import FootGeometry
from .rotations import mat_to_quat, yaw_mat


def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_manifest(path, split='train'):
    path = Path(path).resolve(); data = json.loads(path.read_text())
    if data.get('robot_contract') != checkpoint_contract(): raise ValueError('Manifest robot contract mismatch')
    seen = {}
    for entry in data['clips']:
        identities=[entry['source_identity'],entry.get('original_clip_id') or Path(entry.get('original_path') or entry['name']).stem]
        for identity in identities:
            if identity in seen and seen[identity] != entry['split']: raise ValueError('Original clip leaks between splits')
            seen[identity] = entry['split']
        for field, digest in [('path','csv_sha256'),('sidecar','sidecar_sha256')]:
            source = Path(entry[field])
            if not source.is_absolute(): source = path.parent/source
            if not source.is_file() or sha256(source) != entry[digest]: raise ValueError(f'Missing/changed manifest source: {source}')
    chosen = [e for e in data['clips'] if e['split']==split and e['status']=='PASS']
    if not chosen: raise ValueError(f'No qualified {split} clips')
    files = [str((path.parent/e['path']).resolve()) for e in chosen]
    return files, dict(version=1,manifest_sha256=sha256(path),split=split,
                       clips=[{k:e[k] for k in ('name','csv_sha256','sidecar_sha256','source_identity')} for e in chosen])


@dataclass(frozen=True)
class ContactConfig:
    enter_height: float = .025
    exit_height: float = .04
    enter_speed: float = .35
    exit_speed: float = .6
    min_duration: float = .06

    def __post_init__(self):
        if not (0<=self.enter_height<self.exit_height and 0<self.enter_speed<self.exit_speed and self.min_duration>0):
            raise ValueError('Invalid contact hysteresis thresholds')


def estimate_contacts(height, speed, cfg=ContactConfig(), dt=CONTROL_DT):
    """Causal debounce and hysteresis. Labels are estimates, never actual forces."""
    height,speed=np.asarray(height),np.asarray(speed)
    contact=np.zeros(height.shape,dtype=bool)
    stable=np.zeros(height.shape[1],dtype=int); state=np.zeros(height.shape[1],dtype=bool)
    required=max(1,int(np.ceil(cfg.min_duration/dt)))
    for t in range(len(height)):
        desired=np.where(state,(height[t]<cfg.exit_height)&(speed[t]<cfg.exit_speed),
                        (height[t]<=cfg.enter_height)&(speed[t]<=cfg.enter_speed))
        stable=np.where(desired!=state,stable+1,0)
        switch=stable>=required; state=np.where(switch,desired,state); stable=np.where(switch,0,stable)
        contact[t]=state
    return contact


def swing_phases(contact):
    phase=np.zeros(contact.shape,dtype=np.float32)
    for side in (0,1):
        mask=~contact[:,side]
        bounds=np.flatnonzero(np.diff(np.r_[False,mask,False]))
        for start,end in zip(bounds[::2],bounds[1::2]):
            phase[start:end,side]=np.arange(end-start,dtype=np.float32)/max(end-start,1)
    return phase


def extract_reference(motion, kin, clip, contact_cfg=ContactConfig()):
    start,end=int(motion.clip_start[clip]),int(motion.clip_end[clip])+1
    ids=[kin.tracked_names.index(n) for n in FOOT_BODY_NAMES]
    p=motion.body_p[start:end,ids].cpu().numpy();R=motion.body_R[start:end,ids].cpu().numpy()
    v=motion.body_v[start:end,ids].cpu().numpy()
    geometry=FootGeometry(model_contract()[0]); gaps=np.empty(p.shape[:2])
    for side in (0,1):
        vertical=R[:,side,2,:]
        low=(vertical@geometry.points[side].T).min(-1)
        for center,axis,half,radius in geometry.capsules[side]:
            low=np.minimum(low,vertical@center-np.abs(vertical@axis)*half-radius)
        gaps[:,side]=p[:,side,2]+low
    contact=estimate_contacts(gaps,np.linalg.norm(v,axis=-1),contact_cfg)
    upper=[i for i,n in enumerate(JOINT_NAMES) if not n.startswith('leg_')]
    H=yaw_mat(motion.root_R[start:end])
    torso=motion.body_R[start:end,kin.tracked_names.index(COM_BODY)]
    return dict(time=np.arange(end-start)*CONTROL_DT, foot_p=p,foot_quat=mat_to_quat(torch.from_numpy(R)).numpy(),
                foot_v=v,sole_gap=gaps,contact=contact,swing_phase=swing_phases(contact),
                root_p=motion.root_p[start:end].cpu().numpy(),root_R=motion.root_R[start:end].cpu().numpy(),
                root_v_local=motion.root_v_local[start:end].cpu().numpy(),
                q=motion.q[start:end].cpu().numpy(),qd=motion.qd[start:end].cpu().numpy(),
                upper_q=motion.q[start:end,upper].cpu().numpy(),upper_qd=motion.qd[start:end,upper].cpu().numpy(),
                torso_local_R=(H.transpose(-1,-2)@torso).cpu().numpy(),upper_joint_names=np.array(JOINT_NAMES)[upper],
                clip_frame=np.arange(end-start),contact_config=json.dumps(asdict(contact_cfg)))
