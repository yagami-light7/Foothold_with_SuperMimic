"""Optional velocity/phase-matched upper-body templates; no random lower-body imitation."""
from dataclasses import asdict,dataclass
import json
from pathlib import Path

import numpy as np
import torch

from .foot_mdp import FootTrackingMDP
from .foot_trajectory import query,command_time
from .motion import CONTROL_DT
from .motion_prior_data import load_manifest,sha256
from .rotations import quat_to_mat,mat_to_quat,quat_slerp,so3_log,yaw_mat
from .s54 import JOINT_NAMES,COM_BODY


@dataclass(frozen=True)
class PriorConfig:
    bank: str | None = None
    w_upper_pose: float = 0.
    w_torso: float = 0.
    w_upper_velocity: float = 0.
    pose_std: float = .5
    torso_std: float = .35
    velocity_std: float = 1.5
    speed_tolerance: float = .12
    yaw_rate_tolerance: float = .15
    lateral_speed_tolerance: float = .05

    def __post_init__(self):
        for key,value in asdict(self).items():
            if key=='bank': continue
            if not np.isfinite(value) or value<0 or (key.endswith('_std') and value==0):
                raise ValueError('Invalid prior parameter '+key)
        if self.enabled and not self.bank: raise ValueError('Enabled dynamic prior needs a template bank')

    @property
    def enabled(self): return self.w_upper_pose+self.w_torso+self.w_upper_velocity>0


def resolve_prior(value):
    if isinstance(value,str):
        from .config_paths import resolve_config_path
        value=json.loads(resolve_config_path(value).read_text())
    return PriorConfig(**(value or {}))


def build_template_bank(manifest,data,out,phases=64):
    """Only complete left-to-left gait cycles from TRAIN walking clips.

    No stand template is synthesized from neutral q or from a walking snapshot.
    The observed right liftoff is used to split the gait into its two half cycles.
    """
    _,contract=load_manifest(manifest,'train')
    meta=json.loads(Path(manifest).read_text());q=[];qd=[];R=[];speed=[];names=[]
    for entry in meta['clips']:
        if entry['split']!='train' or entry['status']!='PASS' or not entry['name'].startswith(('walk','slow_walk')): continue
        d=data[entry['name']];c=d['contact']
        left=np.flatnonzero(c[:-1,0]&~c[1:,0])+1;right=np.flatnonzero(c[:-1,1]&~c[1:,1])+1
        for start,end in zip(left[:-1],left[1:]):
            middle=right[(right>start)&(right<end)]
            if len(middle)!=1 or not .45<=(end-start)*CONTROL_DT<=3.: continue
            yaw=np.unwrap(np.arctan2(d['root_R'][start:end,1,0],d['root_R'][start:end,0,0]))
            yaw_rate=abs(yaw[-1]-yaw[0])/max((end-start-1)*CONTROL_DT,CONTROL_DT)
            if yaw_rate>.15 or abs(d['root_v_local'][start:end,1].mean())>.05: continue
            middle=int(middle[0]);grid=np.r_[np.linspace(start,middle,phases//2,endpoint=False),np.linspace(middle,end,phases//2,endpoint=False)]
            i=grid.astype(int);j=np.minimum(i+1,len(c)-1);a=(grid-i)[:,None]
            q.append(d['upper_q'][i]*(1-a)+d['upper_q'][j]*a)
            qd.append(d['upper_qd'][i]*(1-a)+d['upper_qd'][j]*a)
            quats=mat_to_quat(torch.from_numpy(d['torso_local_R']))
            R.append(quat_slerp(quats[i],quats[j],torch.as_tensor(grid-i,dtype=quats.dtype)).numpy())
            speed.append(float(np.mean(d['root_v_local'][start:end,0])));names.append(entry['name'])
    n=len(q);upper=[n for n in JOINT_NAMES if not n.startswith('leg_')]
    np.savez_compressed(out,upper_q=np.asarray(q,dtype=np.float32).reshape(n,phases,17),
        upper_qd=np.asarray(qd,dtype=np.float32).reshape(n,phases,17),
        torso_quat=np.asarray(R,dtype=np.float32).reshape(n,phases,4),speed=np.asarray(speed,dtype=np.float32),
        modes=np.ones(n,dtype=np.int64),source_clips=np.array(names,dtype=str),upper_joint_names=np.array(upper),
        contract=json.dumps(dict(version=1,data=contract,standing_template_available=False,
                                  phase='observed left liftoff=0; right liftoff=.5; next left=1')))


def command_features(command,ep_step):
    """Command-only matching variables, independent of predicted robot state."""
    dtype=command.initial_pos.dtype;t=command_time(ep_step,CONTROL_DT,dtype)
    valid=torch.arange(command.foot.shape[1],device=t.device)[None]<command.lengths[:,None]
    # Forward command velocity derives from fixed final foot placements and plan duration.
    final=[]
    for side in (0,1):
        mask=valid&(command.foot==side)
        idx=torch.where(mask,torch.arange(mask.shape[1],device=t.device)[None],-1).max(-1).values
        target=command.target_pos[torch.arange(len(t),device=t.device),idx.clamp_min(0)]
        final.append(torch.where((idx>=0)[:,None],target,command.initial_pos[:,side]))
    displacement=torch.stack(final,1).mean(1)-command.initial_pos.mean(1)
    duration=torch.where(valid,command.touchdown,0).max(-1).values.clamp_min(.02)
    local=(quat_to_mat(command.heading).transpose(-1,-2)@displacement[...,None]).squeeze(-1)
    speed=local[:,0]/duration
    # Phase is defined by consecutive liftoff times, including Double Support.
    active=valid&(command.liftoff<=t[:,None])
    idx=torch.where(active,torch.arange(active.shape[1],device=t.device)[None],-1).max(-1).values
    row=torch.arange(len(t),device=t.device);safe=idx.clamp_min(0)
    start=command.liftoff[row,safe]
    nxt=(safe+1).clamp_max(active.shape[1]-1)
    end=torch.where(nxt>safe,command.liftoff[row,nxt],command.touchdown[row,safe])
    phase=(command.foot[row,safe].to(dtype)*.5+.5*((t-start)/(end-start).clamp_min(.02)).clamp(0,1)).remainder(1.)
    phase=torch.where(idx>=0,phase,torch.zeros_like(phase))
    mode=(command.lengths>0).long()
    final_R=quat_to_mat(command.target_quat[row,(command.lengths-1).clamp_min(0)])
    relative=quat_to_mat(command.heading).transpose(-1,-2)@final_R
    yaw_rate=torch.atan2(relative[:,1,0],relative[:,0,0])/duration
    return speed,phase,mode,yaw_rate,local[:,1]/duration


class UpperPrior:
    def __init__(self,cfg,device):
        self.cfg=cfg;self.indices=[JOINT_NAMES.index(n) for n in JOINT_NAMES if not n.startswith('leg_')]
        with np.load(cfg.bank,allow_pickle=False) as z:
            names=z['upper_joint_names'].tolist()
            if names!=[JOINT_NAMES[i] for i in self.indices]: raise ValueError('Prior S54 upper joint order mismatch')
            self.contract=json.loads(str(z['contract']));self.hash=sha256(cfg.bank)
            self.q=torch.as_tensor(z['upper_q'],device=device);self.qd=torch.as_tensor(z['upper_qd'],device=device)
            self.R=torch.as_tensor(z['torso_quat'],device=device);self.speed=torch.as_tensor(z['speed'],device=device)
            self.mode=torch.as_tensor(z['modes'],device=device)
        if self.q.shape[0]==0: raise ValueError('Template bank contains no qualified walking cycles')

    def targets(self,command,ep_step):
        speed,phase,mode,yaw_rate,lateral_speed=command_features(command,ep_step)
        error=(speed[:,None]-self.speed[None]).abs()
        eligible=(mode[:,None]==self.mode[None])&(error<=self.cfg.speed_tolerance)
        eligible=eligible&(yaw_rate[:,None].abs()<=self.cfg.yaw_rate_tolerance)
        eligible=eligible&(lateral_speed[:,None].abs()<=self.cfg.lateral_speed_tolerance)
        index=torch.where(eligible,error,float('inf')).argmin(-1);match=eligible.any(-1)
        grid=phase*self.q.shape[1];i=grid.floor().long()%self.q.shape[1];j=(i+1)%self.q.shape[1];a=grid-grid.floor()
        blend=lambda x:x[index,i]*(1-a[:,None])+x[index,j]*a[:,None]
        quaternion=quat_slerp(self.R[index,i],self.R[index,j],a)
        # Follow commanded foot yaw during gentle turns, rather than pinning the
        # torso to the episode's initial heading. This is command-only geometry.
        ref=query(command,command_time(ep_step,CONTROL_DT,command.initial_pos.dtype)[:,None])
        foot_R=quat_to_mat(ref.quaternion[:,0])
        xy=foot_R[:,:,:2,0].mean(1)
        yaw=torch.atan2(xy[:,1],xy[:,0]);cos,sin=yaw.cos(),yaw.sin();zero=torch.zeros_like(yaw);one=torch.ones_like(yaw)
        H=torch.stack((cos,-sin,zero,sin,cos,zero,zero,zero,one),-1).reshape(-1,3,3)
        R=H@quat_to_mat(quaternion)
        return blend(self.q),blend(self.qd),R,match

    def qualities(self,S,torso_R,command,ep_step):
        q,qd,R,match=self.targets(command,ep_step);c=self.cfg
        pose=torch.exp(-(S[4][:,self.indices]-q).square().mean(-1)/c.pose_std**2)
        velocity=torch.exp(-(S[5][:,self.indices]-qd).square().mean(-1)/c.velocity_std**2)
        torso=torch.exp(-so3_log(R.transpose(-1,-2)@torso_R).square().sum(-1)/c.torso_std**2)
        return dict(prior_upper_pose=pose*match,prior_torso=torso*match,prior_upper_velocity=velocity*match,
                    prior_matched=match.to(pose.dtype))


class PriorFootMDP(FootTrackingMDP):
    def __init__(self,kin,default_q,goals,cfg,prior_cfg):
        super().__init__(kin,default_q,goals,cfg)
        self.prior_cfg=prior_cfg;self.prior=UpperPrior(prior_cfg,default_q.device) if prior_cfg.enabled else None
        supported=self.prior is not None and (cfg.command_mode!='standing' or bool((self.prior.mode==0).any()))
        if supported:self.max_step_reward+=CONTROL_DT*(prior_cfg.w_upper_pose+prior_cfg.w_torso+prior_cfg.w_upper_velocity)

    def contract(self):
        result=super().contract()
        if self.prior:
            result=dict(result,upper_prior=dict(version=1,config=asdict(self.prior_cfg),bank_sha256=self.prior.hash,
                data=self.prior.contract,unmatched='disabled; existing default posture penalty remains'))
        return result

    def reward(self,S,f,a,a_prev,terms=False,*,command=None,ep_step=None):
        r,t=super().reward(S,f,a,a_prev,True,command=command,ep_step=ep_step)
        if self.prior:
            torso=self.kin.forward(*S)[1][:,self.kin.tracked_names.index(COM_BODY)]
            raw=self.prior.qualities(S,torso,command,ep_step);c=self.prior_cfg
            for key,weight in [('prior_upper_pose',c.w_upper_pose),('prior_torso',c.w_torso),('prior_upper_velocity',c.w_upper_velocity)]:
                contribution=raw[key]*weight*CONTROL_DT;r=r+contribution;t[key+'_contribution']=contribution
            t.update(raw)
        return (r,t) if terms else r
