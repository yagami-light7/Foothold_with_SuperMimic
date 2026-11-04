"""Optional paired interface: original S54 reference FK, never a quintic refit."""
from dataclasses import dataclass, fields

import torch

from .goals import GoalSource, encode_foot_samples
from .foot_trajectory import FootTrajectory, command_time
from .motion import CONTROL_DT
from .motion_prior_data import ContactConfig, extract_reference
from .rotations import mat_to_quat, quat_slerp, yaw_mat, quat_to_mat, mat_to_6d
from .mdp import MDP, invariant_features, INV_DIM
from .s54 import FOOT_BODY_NAMES


@dataclass
class ReferenceFootCommand:
    start_frame: torch.Tensor
    clip_id: torch.Tensor
    heading: torch.Tensor
    initial_pos: torch.Tensor

    def tensors(self): return {f.name:getattr(self,f.name) for f in fields(self)}
    def clone(self): return ReferenceFootCommand(**{k:v.detach().clone() for k,v in self.tensors().items()})
    def index(self,indices):
        if isinstance(indices,int): indices=slice(indices,indices+1)
        return ReferenceFootCommand(**{k:v[indices] for k,v in self.tensors().items()})
    @staticmethod
    def cat(commands): return ReferenceFootCommand(**{k:torch.cat([getattr(c,k) for c in commands]) for k in commands[0].tensors()})
    def assign(self,indices,command):
        for k,v in self.tensors().items(): v[indices]=getattr(command,k)
    def repeat(self,count): return ReferenceFootCommand(**{k:v[None].expand((count,)+v.shape).flatten(0,1) for k,v in self.tensors().items()})


class ReferenceFootGoalSource(GoalSource):
    def __init__(self,motion,kin,spec,foot_config,contact_cfg=ContactConfig()):
        super().__init__(motion,kin,spec,foot_config,command_mode='standing')
        if spec.mode!='reference_foot': raise ValueError('Paired source needs its own goal contract')
        self.is_foot=True; self.command_mode='paired'; self.contact_cfg=contact_cfg
        data=[extract_reference(motion,kin,i,contact_cfg) for i in range(len(motion.names))]
        device=motion.root_p.device
        self.contact=torch.as_tensor(__import__('numpy').concatenate([d['contact'] for d in data]),device=device)
        self.phase=torch.as_tensor(__import__('numpy').concatenate([d['swing_phase'] for d in data]),device=device)
        self.position=motion.body_p[:,self.foot_indices]
        self.quaternion=mat_to_quat(motion.body_R[:,self.foot_indices])
        self.velocity=motion.body_v[:,self.foot_indices]

    def new_command(self,S,generator=None,frames=None):
        if frames is None: raise ValueError('Paired command requires the saved reference frame')
        return ReferenceFootCommand(frames.clone(),self.motion.clip_id[frames].clone(),
                                    mat_to_quat(yaw_mat(self.motion.root_R[frames])),self.position[frames].clone())

    def query(self,command,times):
        end=self.motion.clip_end[command.clip_id][:,None]
        frame=(command.start_frame[:,None]+times/CONTROL_DT).clamp_min(0)
        rounded=frame.round()
        frame=torch.where((frame-rounded).abs()<1e-7,rounded,frame)
        frame=torch.minimum(frame,end)
        i=frame.floor().long(); j=torch.minimum(i+1,end)
        fraction=(frame-i).to(self.position.dtype)
        lerp=lambda x: x[i]*(1-fraction[...,None,None])+x[j]*fraction[...,None,None]
        q=quat_slerp(self.quaternion[i],self.quaternion[j],fraction[...,None].expand(-1,-1,2))
        return FootTrajectory(lerp(self.position),q,lerp(self.velocity),torch.zeros_like(self.position[i]),
                              self.contact[i],self.phase[i])

    def goal(self,S,f,ep_step=None,command=None):
        if ep_step is None or command is None: raise ValueError('Paired Goal requires immutable frame origin/tick')
        offsets=torch.arange(1,self.spec.frames+1,device=f.device,dtype=torch.float64)*self.spec.dt
        times=command_time(ep_step,CONTROL_DT,torch.float64)[:,None]+offsets[None]
        return encode_foot_samples(S,self.query(command,times),offsets.to(S[0].dtype))


class PairedMotionMDP(MDP):
    """Foot Actor/Critic interface, original Motion reset/reward/termination.

    is_foot=False intentionally retains Motion clip boundaries and reset sampling.
    Full reference is available to analytic reward and privileged Critic only.
    """
    def __init__(self,kin,motion,default_q,goals,cfg):
        super().__init__(kin,motion,default_q,cfg)
        self.goals=goals; self.critic_dim=135
        self.foot_indices=[kin.tracked_names.index(n) for n in FOOT_BODY_NAMES]

    def contract(self):
        return dict(**{k:v for k,v in super().contract().items() if k!='mode'},mode='paired_motion_bridge',
                    goal_source='original_reference_fk',contact_config=__import__('dataclasses').asdict(self.goals.contact_cfg),
                    reset='Motion bins',termination='Motion reference height/tilt + clip end')

    def critic_features(self,S,f,xi,*,command=None,ep_step=None):
        ref=self.goals.query(command,command_time(ep_step,CONTROL_DT,S[0].dtype)[:,None])
        bp,bR,bv,_=self.kin.forward(*S); ids=self.foot_indices
        Ht=yaw_mat(S[1]).transpose(-1,-2)
        pos=(Ht[:,None]@(ref.position[:,0]-bp[:,ids])[...,None]).squeeze(-1).flatten(1)
        vel=(Ht[:,None]@(ref.velocity[:,0]-bv[:,ids])[...,None]).squeeze(-1).flatten(1)
        ori=mat_to_6d(bR[:,ids].transpose(-1,-2)@quat_to_mat(ref.quaternion[:,0])).flatten(1)
        base=self.motion.root_p[f]-S[0]
        xy=(Ht[:,:2,:2]@base[:,:2,None]).squeeze(-1)
        return torch.cat((invariant_features(S),pos,ori,vel,base[:,2:3],xy,
                          ref.contact[:,0].to(S[0].dtype),ref.swing_phase[:,0],xi),-1)
