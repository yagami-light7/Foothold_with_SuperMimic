"""One Roban reference clock/anchor for Actor, Critic, rewards and evaluation.

CSV is selected by header (Frame and retarget diagnostics are not actions).
NPZ v1 keys: root_pos[T,3], root_quat[T,4], joint_pos[T,J], joint_names[J],
fps scalar, quaternion_order scalar ('xyzw'/'wxyz'). Optional contact[T,2],
contact_valid[T,2], swing_phase[T,2], quality[T]. No pickled objects are loaded.
All velocities are recomputed after resampling: world root link origin velocities
and canonical joint radians/second. Control samples are exactly 0.02 seconds.
"""
from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .motion import MotionLib, CONTROL_DT, BIN_FRAMES, _gradient
from .roban import JOINT_NAMES, FOOT_BODY_NAMES, foot_geometry, checkpoint_contract
from .rotations import quat_to_mat, mat_to_quat, quat_slerp, so3_log, yaw_mat, mat_to_6d


@dataclass
class EpisodeAnchor:
    start_frame: torch.Tensor
    clip_id: torch.Tensor
    rotation: torch.Tensor
    translation: torch.Tensor

    def tensors(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def clone(self):
        return EpisodeAnchor(**{k: v.detach().clone() for k, v in self.tensors().items()})

    def index(self, indices):
        if isinstance(indices, int):
            indices = slice(indices, indices + 1)
        return EpisodeAnchor(**{k: v[indices] for k, v in self.tensors().items()})

    def assign(self, indices, command):
        for k, v in self.tensors().items():
            v[indices] = getattr(command, k)

    def repeat(self, count):
        return EpisodeAnchor(**{k: v[None].expand((count,) + v.shape).flatten(0, 1)
                               for k, v in self.tensors().items()})

    @staticmethod
    def cat(commands):
        return EpisodeAnchor(**{k: torch.cat([getattr(c, k) for c in commands])
                               for k in commands[0].tensors()})


class MotionReferenceProvider(MotionLib):
    """MotionLib-compatible replay/sampling API with explicit source FPS."""

    def __init__(self, files, kin, device='cpu', src_fps=None):
        if not files:
            raise ValueError('No Roban references selected')
        self.device, self.kin = device, kin
        self.names = [Path(f).stem for f in files]
        if len(self.names) != len(set(self.names)):
            raise ValueError('Reference clip names must be unique')
        self.validation = dict(robot=checkpoint_contract(), files=[], contact='untrusted estimates unless labelled')
        clips = [self._load(Path(f), src_fps) for f in files]
        lens = [len(c['p']) for c in clips]
        self.clip_len = torch.tensor(lens, device=device)
        self.clip_start = torch.tensor(np.cumsum([0] + lens[:-1]), device=device, dtype=torch.long)
        self.clip_end = self.clip_start + self.clip_len - 1
        self.clip_id = torch.cat([torch.full((n,), i, device=device, dtype=torch.long) for i, n in enumerate(lens)])
        self.num_frames = sum(lens)
        cat = lambda key: torch.cat([c[key] for c in clips]).to(device)
        self.root_p, self.root_R, self.root_v, self.root_w = (cat(k) for k in ('p', 'R', 'v', 'w'))
        self.q, self.qd, self.quality = (cat(k) for k in ('q', 'qd', 'quality'))
        if ((self.q < kin.q_lo - 1e-4) | (self.q > kin.q_hi + 1e-4)).any():
            raise ValueError('Roban reference exceeds model joint limits; repair data explicitly')
        with torch.no_grad():
            self.body_p, self.body_R, self.body_v, self.body_w = kin.forward(*self.ref_state(torch.arange(self.num_frames, device=device)))
        self.foot_indices = [kin.tracked_names.index(n) for n in FOOT_BODY_NAMES]
        self.sole_offsets = self.q.new_tensor([f['sole'] for f in foot_geometry()])
        self.corners = self.q.new_tensor([f['corners'] for f in foot_geometry()])
        self.foot_p, self.foot_R, self.foot_v = self.feet(self.ref_state(torch.arange(self.num_frames, device=device)))
        self.contact, self.contact_valid, self.phase = [], [], []
        for i, c in enumerate(clips):
            sl = slice(int(self.clip_start[i]), int(self.clip_end[i])+1)
            if c['contact'] is None:
                # Multi-signal estimate: sole clearance + 3D speed + 3-frame dwell.
                near = self.foot_p[sl, :, 2] < .035
                slow = self.foot_v[sl].norm(dim=-1) < .25
                estimate = near & slow
                stable = estimate.clone()
                for t in range(1, len(estimate)):
                    stable[t] = estimate[max(0,t-2):t+1].all(0)
                contact = stable
                valid = torch.zeros_like(contact)  # Never promote an estimate to contact truth.
            else:
                contact, valid = c['contact'].to(device), c['contact_valid'].to(device)
            phase = c['phase'].to(device) if c['phase'] is not None else self._phase(contact)
            self.contact.append(contact); self.contact_valid.append(valid); self.phase.append(phase)
        self.contact, self.contact_valid, self.phase = (torch.cat(x) for x in (self.contact, self.contact_valid, self.phase))
        # Fixed world stance target at each planned touchdown; reference ankle
        # jitter during a stance must not become a command to slide the foot.
        previous=torch.cat((torch.zeros_like(self.contact[:1]),self.contact[:-1]))
        new_clip=torch.cat((torch.ones(1,device=device,dtype=torch.bool),self.clip_id[1:]!=self.clip_id[:-1]))
        starts=self.contact&(~previous|new_clip[:,None])
        indices=torch.arange(self.num_frames,device=device)[:,None].expand(-1,2)
        anchored=torch.where(starts,indices,torch.zeros_like(indices)).cummax(0).values
        fixed=self.foot_p[anchored,torch.arange(2,device=device)[None]]
        self.stance_p=torch.where(self.contact[...,None],fixed,self.foot_p)
        self.target_foot_p=torch.where(self.contact[...,None],self.stance_p,self.foot_p)
        Rt = self.root_R.transpose(-1, -2)
        self.root_v_local = (Rt @ self.root_v[..., None]).squeeze(-1)
        self.root_w_local = (Rt @ self.root_w[..., None]).squeeze(-1)
        self.root_gz = self.root_R[:, 2, 2]
        bins = [(i, int(self.clip_start[i])+b) for i, n in enumerate(lens) for b in range(0,n,BIN_FRAMES)]
        self.bin_clip = torch.tensor([b[0] for b in bins], device=device)
        self.bin_start = torch.tensor([b[1] for b in bins], device=device)
        self.num_bins = len(bins)
        self.frame_bin = torch.empty(self.num_frames, device=device, dtype=torch.long)
        for b, (i, start) in enumerate(bins):
            self.frame_bin[start:min(start+BIN_FRAMES,int(self.clip_end[i])+1)] = b
        self.validation['frames'] = self.num_frames
        self.validation['trusted_contact_fraction'] = float(self.contact_valid.float().mean())
        self.validation['body_prior_quality_mean'] = float(self.quality.mean())

    def _load(self, path, explicit_fps):
        contact = valid = phase = None
        if path.suffix == '.csv':
            import csv
            with path.open(newline='',encoding='utf-8') as stream:
                reader=csv.reader(stream); header=next(reader); rows=list(reader)
            if len(header)!=len(set(header)):
                raise ValueError('Duplicate CSV header columns')
            values=np.full((len(rows),len(header)),np.nan)
            for i,row in enumerate(rows):
                if len(row)>len(header): raise ValueError('CSV row wider than header')
                values[i,:len(row)]=[float(x) if x else float('nan') for x in row]
            data=np.rec.fromarrays(values.T,names=header)
            if data.ndim != 1 or len(data) < 3:
                raise ValueError(f'Reference needs >=3 frames: {path}')
            col = lambda names: np.stack([data[n] for n in names], -1)
            p = col(['root_translate'+a for a in 'XYZ'])
            quat = col(['root_rotate'+a for a in 'WXYZ'])
            q = col(JOINT_NAMES)
            if 'Frame' not in data.dtype.names or not np.allclose(np.diff(data['Frame']), 1):
                raise ValueError('CSV Frame must be consecutive')
            fps = explicit_fps
            if fps is None:
                registry = json.loads((Path(__file__).resolve().parents[1]/'library/registry.json').read_text())
                datasets=registry['robots']['roban_s22_0901']['datasets']
                declared=[datasets[p.name]['fps'] for p in path.parents if p.name in datasets]
                fps=declared[0] if declared else None
            # Diagnostics gate the auxiliary prior only, never the primary foot objective.
            pos_names = [n for n in data.dtype.names if n.endswith('_pos_err') and n.startswith(('Spine','LeftArm','RightArm','LeftForeArm','RightForeArm'))]
            rot_names = [n for n in data.dtype.names if n.endswith('_rot_err') and n.startswith(('Spine','LeftArm','RightArm','LeftForeArm','RightForeArm'))]
            quality = np.zeros(len(data), np.float32)
            if pos_names and rot_names:
                ep, er = col(pos_names), col(rot_names)
                quality = (np.isfinite(ep).all(-1) & np.isfinite(er).all(-1) & (np.abs(ep).max(-1)<.10) & (np.abs(er).max(-1)<.6)).astype(np.float32)
        elif path.suffix == '.npz':
            with np.load(path, allow_pickle=False) as z:
                if list(z['joint_names'].astype(str)) != JOINT_NAMES:
                    raise ValueError('NPZ joint names/order must match Roban canonical contract')
                fps = float(z['fps'])
                if explicit_fps is not None and abs(fps-explicit_fps)>1e-6:
                    raise ValueError('NPZ FPS differs from requested FPS')
                p, quat, q = z['root_pos'], z['root_quat'], z['joint_pos']
                order = str(z['quaternion_order'].item())
                if order not in ('wxyz','xyzw'):
                    raise ValueError('NPZ quaternion_order required')
                if order == 'xyzw': quat = quat[:, [3,0,1,2]]
                quality = z['quality'] if 'quality' in z else np.zeros(len(p),np.float32)
                if 'contact' in z:
                    if 'contact_valid' not in z:
                        raise ValueError('Contact labels require an explicit contact_valid mask')
                    contact, valid = z['contact'], z['contact_valid']
                if 'swing_phase' in z: phase = z['swing_phase']
        else:
            raise ValueError('Roban references support CSV or NPZ')
        if fps is None or not np.isfinite(fps) or fps <= 0:
            raise ValueError(f'FPS has no verified provenance for {path}; pass --reference_fps explicitly')
        if p.shape != (len(p),3) or quat.shape != (len(p),4) or q.shape != (len(p),len(JOINT_NAMES)) or len(p)<3:
            raise ValueError('Invalid reference pose shapes')
        for x in (p,quat,q,quality):
            if not np.isfinite(x).all(): raise ValueError('Nonfinite reference pose/quality')
        if quality.shape != (len(p),) or np.any((quality<0)|(quality>1)):
            raise ValueError('quality must be [T] in [0,1]')
        if np.max(np.abs(np.linalg.norm(quat,axis=-1)-1))>.01:
            raise ValueError('Reference quaternion is not normalized')
        if contact is not None:
            if contact.shape != (len(p),2) or valid.shape != contact.shape or not np.isin(contact,[0,1]).all() or not np.isin(valid,[0,1]).all():
                raise ValueError('Contact/mask must be binary [T,2]')
        if phase is not None and (phase.shape!=(len(p),2) or not np.isfinite(phase).all() or np.any((phase<0)|(phase>1))):
            raise ValueError('swing_phase must be finite [T,2] in [0,1]')
        p, quat, q = [torch.tensor(x,dtype=torch.float32) for x in (p,quat,q)]
        t = torch.arange(int(np.floor((len(p)-1)/fps/CONTROL_DT+1e-7))+1,dtype=torch.float64)*CONTROL_DT*fps
        lo=t.floor().long().clamp(max=len(p)-2); a=(t-lo).clamp(0,1).float()
        blend=lambda x: x[lo]*(1-a[:,None])+x[lo+1]*a[:,None]
        p, q, quat = blend(p), blend(q), quat_slerp(quat[lo],quat[lo+1],a)
        R=quat_to_mat(quat)
        w=torch.zeros_like(p)
        w[1:-1]=so3_log(R[2:]@R[:-2].transpose(-1,-2))/(2*CONTROL_DT)
        w[0]=so3_log(R[1]@R[0].T)/CONTROL_DT; w[-1]=so3_log(R[-1]@R[-2].T)/CONTROL_DT
        quality=torch.tensor(quality,dtype=torch.float32)
        # Conservative quality gate over both source endpoints; no interpolated trust.
        quality=torch.minimum(quality[lo],quality[lo+1])
        nearest=t.round().long().clamp(max=int(lo.max()+1))
        convert=lambda x,dtype: None if x is None else torch.tensor(x,dtype=dtype)[nearest]
        library=(Path(__file__).resolve().parents[1]/'library').resolve()
        identity='library/'+str(path.resolve().relative_to(library)) if path.resolve().is_relative_to(library) else str(path.resolve())
        self.validation['files'].append(dict(path=identity,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                            source_fps=float(fps),control_dt=CONTROL_DT,frames=len(p),contact_labelled=contact is not None))
        return dict(p=p,R=R,v=_gradient(p,CONTROL_DT),w=w,q=q,qd=_gradient(q,CONTROL_DT),quality=quality,
                    contact=convert(contact,torch.bool),contact_valid=convert(valid,torch.bool),phase=convert(phase,torch.float32))

    @staticmethod
    def _phase(contact):
        phase=torch.zeros_like(contact,dtype=torch.float32)
        for side in range(2):
            starts=torch.where(torch.cat((~contact[:1,side],contact[:-1,side]&~contact[1:,side])))[0]
            for start in starts.tolist():
                end=int(start)
                while end<len(contact) and not contact[end,side]: end+=1
                phase[start:end,side]=torch.linspace(0,1,max(1,end-start),device=contact.device)
        return phase

    def feet(self, S):
        bp,bR,bv,bw=self.kin.forward(*S)
        R=bR[...,self.foot_indices,:,:]
        offset=(R@self.sole_offsets[...,None]).squeeze(-1)
        p=bp[...,self.foot_indices,:]+offset
        v=bv[...,self.foot_indices,:]+torch.cross(bw[...,self.foot_indices,:],offset,dim=-1)
        return p,R,v

    def anchor(self, frames, rotation=None, translation=None):
        p=self.root_p[frames]
        return EpisodeAnchor(frames.clone(),self.clip_id[frames].clone(),
            torch.eye(3,device=p.device,dtype=p.dtype).expand(len(frames),3,3).clone() if rotation is None else rotation,
            torch.zeros_like(p) if translation is None else translation)

    def indices(self, f, frames=1, dt=CONTROL_DT):
        stride=dt/CONTROL_DT
        if frames<1 or abs(stride-round(stride))>1e-6 or stride<1:
            raise ValueError('Future samples must use a positive integral control-frame stride')
        return torch.minimum(f[:,None]+torch.arange(1,frames+1,device=f.device)*round(stride),self.clip_end[self.clip_id[f]][:,None])

    def sample(self, f, anchor):
        """f[B,K], anchor[B]; all consumers receive this exact same transform."""
        A=anchor.rotation[:,None]; t=anchor.translation[:,None]
        pos=lambda x: (A@x[...,None]).squeeze(-1)+t
        return dict(root_p=pos(self.root_p[f]),root_R=A@self.root_R[f],
                    root_v=(A@self.root_v[f,...,None]).squeeze(-1),root_w=(A@self.root_w[f,...,None]).squeeze(-1),
                    q=self.q[f],qd=self.qd[f],quality=self.quality[f],
                    foot_p=(A[:,:,None]@self.target_foot_p[f,...,None]).squeeze(-1)+t[:,:,None],
                    stance_p=(A[:,:,None]@self.stance_p[f,...,None]).squeeze(-1)+t[:,:,None],
                    foot_R=A[:,:,None]@self.foot_R[f],
                    foot_v=(A[:,:,None]@self.foot_v[f,...,None]).squeeze(-1),
                    contact=self.contact[f],contact_valid=self.contact_valid[f],phase=self.phase[f])

    def body_goal(self, S, f, anchor, frames, dt):
        ref=self.sample(self.indices(f,frames,dt),anchor)
        Ht=yaw_mat(S[1]).transpose(-1,-2)[:,None]
        return torch.cat(((Ht@(ref['root_p']-S[0][:,None])[...,None]).squeeze(-1),
            mat_to_6d(Ht@ref['root_R']), (Ht@ref['root_v'][...,None]).squeeze(-1),
            (Ht@ref['root_w'][...,None]).squeeze(-1), ref['q'],ref['qd']*.1),-1).flatten(1)
