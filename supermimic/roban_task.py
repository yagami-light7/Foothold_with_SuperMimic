"""Foot-conditioned Actor inputs and a separate, privileged return Critic.

Reward is the same differentiable function for replay/Newton and imagination.
No whole-body reference or xi is read by obs()/goal(); no shared Critic latent.
"""
from dataclasses import dataclass, asdict
import math

import torch

from .motion import CONTROL_DT
from .mdp import flatten_state
from .roban import JOINT_NAMES, NUM_J, XI_DIM, GOAL_FRAME_DIM, actuator_parameters
from .rotations import yaw_mat, mat_to_6d, rot_angle_sq


@dataclass(frozen=True)
class FootRewardConfig:
    swing_position: float = 1.0
    swing_orientation: float = .25
    stance_position: float = 1.0
    contact_schedule: float = .25
    touchdown: float = .5
    upright: float = .5
    survival: float = .2
    slip: float = .1
    joint_limit: float = 2.0
    torque: float = .01
    action_rate: float = .05
    collision: float = 1.0
    self_collision: float = 1.0
    body_prior_weight: float = 0.0
    prior_upper: float = .7
    prior_waist: float = .3
    prior_root_orientation: float = 0.0
    prior_root_height: float = 0.0
    foot_position_std: float = .08
    foot_orientation_std: float = .3
    slip_speed: float = .2
    prior_joint_std: float = .5
    prior_root_height_std: float = .15
    term_height: float = .35
    term_tilt: float = .8
    soft_limit_factor: float = .9

    def __post_init__(self):
        for key,value in asdict(self).items():
            if not math.isfinite(value) or value<0:
                raise ValueError(f'Invalid reward parameter {key}')
            if key.endswith('_std') and value<=0:
                raise ValueError(f'Positive reward scale required: {key}')
        if self.slip_speed<=0 or not 0<self.soft_limit_factor<=1 or not 0<self.term_tilt<2 or self.term_height<=0:
            raise ValueError('Invalid safety/normalization thresholds')
        if self.prior_upper+self.prior_waist+self.prior_root_orientation+self.prior_root_height>1+1e-9:
            raise ValueError('Normalized body-prior component weights must sum to <=1')

    @property
    def positive_sum(self):
        return sum(getattr(self,k) for k in ('swing_position','swing_orientation','stance_position',
                   'contact_schedule','touchdown','upright','survival','body_prior_weight'))


class RobanFootGoals:
    is_foot=True
    command_mode='roban_reference'

    def __init__(self, motion, kin, spec):
        self.motion,self.kin,self.spec=motion,kin,spec
        motion.indices(torch.zeros(1,dtype=torch.long,device=motion.q.device),spec.frames,spec.dt)

    def new_command(self,S,generator=None,frames=None):
        if frames is None: raise ValueError('Episode anchor requires reference frames')
        return self.motion.anchor(frames)

    def goal(self,S,f,ep_step=None,command=None):
        if command is None or ep_step is None:
            raise ValueError('Foot goals require the immutable replay anchor and episode clock')
        torch._assert_async(torch.all(f==command.start_frame+ep_step), 'Roban reference frame/episode clock mismatch')
        frames=self.motion.indices(f,self.spec.frames,self.spec.dt)
        ref=self.motion.sample(frames,command)
        Ht=yaw_mat(S[1]).transpose(-1,-2)[:,None,None]
        p=(Ht@(ref['foot_p']-S[0][:,None,None])[...,None]).squeeze(-1)
        R=mat_to_6d(Ht@ref['foot_R'])
        feet=torch.cat((p,R,ref['contact'][...,None].to(p.dtype),ref['phase'][...,None]),-1)
        offsets=torch.arange(1,self.spec.frames+1,device=p.device,dtype=p.dtype)*self.spec.dt
        return torch.cat((feet.flatten(-2),offsets[None,:,None].expand(len(p),-1,-1)),-1).flatten(1)

    def obs(self,mdp,S,a_prev,xi,f,command=None):
        return mdp.obs(S,a_prev,xi,f)


class RobanFootMDP:
    # Reference-driven resets/clip ends reuse the original Motion bin sampler.
    is_foot=False
    is_reference_foot=True
    canonical_replay_reward=True
    observation_mode='roban_proprio_v1'

    def __init__(self,kin,motion,default_q,goals,cfg=FootRewardConfig(),*,critic_use_body_reference=False,terrain=None):
        self.kin,self.motion,self.default_q,self.goals,self.cfg=kin,motion,default_q,goals,cfg
        self.critic_use_body_reference=critic_use_body_reference
        self.terrain=terrain
        self.obs_dim=3*NUM_J+6  # q, qd, angular velocity, gravity, previous action
        self.critic_dim=13+2*NUM_J+XI_DIM+(GOAL_FRAME_DIM*goals.spec.frames if critic_use_body_reference else 0)+(terrain.dim if terrain else 0)
        self.max_step_reward=CONTROL_DT*cfg.positive_sum
        mid,half=(kin.q_lo+kin.q_hi)/2,(kin.q_hi-kin.q_lo)/2
        self.soft_lo,self.soft_hi=mid-half*cfg.soft_limit_factor,mid+half*cfg.soft_limit_factor
        self.upper_ids=[i for i,n in enumerate(JOINT_NAMES) if n.startswith('zarm_')]
        self.waist_ids=[i for i,n in enumerate(JOINT_NAMES) if n.startswith('waist_')]
        params=actuator_parameters()
        self.kp=default_q.new_tensor([params[n]['stiffness'] for n in JOINT_NAMES])
        self.kd=default_q.new_tensor([params[n]['damping'] for n in JOINT_NAMES])
        self.effort=default_q.new_tensor([params[n]['effort'] for n in JOINT_NAMES])
        self.alpha=.25*self.effort/self.kp
        self.obs_noise=default_q.new_tensor([.01]*NUM_J+[.05]*NUM_J+[.05]*3+[.02]*3+[0.]*NUM_J)
        from .roban import model_contract,FOOT_BODY_NAMES
        model,_=model_contract()
        # Differentiable non-foot ground collision proxies: bounding spheres of
        # authored collision primitives. Real self-collision remains physical.
        self.collision_ids=[]; offsets=[]; radii=[];inscribed=[];bodies=[]
        import numpy as np
        import mujoco
        for g in range(model.ngeom):
            b=int(model.geom_bodyid[g]); name=model.body(b).name
            if b==0 or name in FOOT_BODY_NAMES or not (model.geom_contype[g] or model.geom_conaffinity[g]): continue
            typ=model.geom_type[g]; size=model.geom_size[g]
            if typ==mujoco.mjtGeom.mjGEOM_SPHERE: radius=size[0]
            elif typ==mujoco.mjtGeom.mjGEOM_CAPSULE: radius=size[0]+size[1]
            else: radius=np.linalg.norm(size)
            self.collision_ids.append(kin.tracked_names.index(name)); offsets.append(model.geom_pos[g]); radii.append(radius)
            bodies.append(b)
            inscribed.append(size[0] if typ in (mujoco.mjtGeom.mjGEOM_SPHERE,mujoco.mjtGeom.mjGEOM_CAPSULE)
                             else min(size[:2]) if typ==mujoco.mjtGeom.mjGEOM_CYLINDER else min(size))
        self.collision_offsets=default_q.new_tensor(np.array(offsets))
        self.collision_radii=default_q.new_tensor(radii)
        self.inscribed_radii=default_q.new_tensor(inscribed)
        excluded={int(x) for x in model.exclude_signature}
        pairs=[]
        def near(a,b):
            ancestors={a,int(model.body_parentid[a]),int(model.body_parentid[int(model.body_parentid[a])])}
            return b in ancestors
        for i,a in enumerate(bodies):
            for j in range(i+1,len(bodies)):
                b=bodies[j]
                if near(a,b) or near(b,a) or ((min(a,b)<<16)+max(a,b)) in excluded: continue
                pairs.append((i,j))
        self.self_pairs=torch.tensor(pairs,device=default_q.device,dtype=torch.long).reshape(-1,2)

    def obs(self,S,a_prev,xi=None,f=None,reference_R=None):
        _,R,_,w,q,qd=S
        return torch.cat((q-self.default_q,qd*.1,(R.transpose(-1,-2)@w[...,None]).squeeze(-1)*.25,-R[...,2,:],a_prev),-1)

    def add_noise(self,o):
        return o+(2*torch.rand_like(o)-1)*self.obs_noise

    def critic_features(self,S,f,xi,*,command=None,ep_step=None):
        # Single scalar value: reference is context; labels remain lambda returns.
        if ep_step is not None:
            torch._assert_async(torch.all(f==command.start_frame+ep_step), 'Critic reference clock mismatch')
        values=[flatten_state(S),xi]
        if self.critic_use_body_reference:
            values.append(self.motion.body_goal(S,f,command,self.goals.spec.frames,self.goals.spec.dt))
        if self.terrain is not None:
            values.append(self.terrain(S))
        return torch.cat(values,-1)

    def _reference(self,f,command):
        if command is None: command=self.motion.anchor(f)
        return self.motion.sample(f[:,None].clamp(max=self.motion.num_frames-1),command)

    def reward(self,S,f,a,a_prev,terms=False,*,command=None,ep_step=None):
        c=self.cfg; p,R,_,_,q,qd=S
        if ep_step is not None and command is not None:
            torch._assert_async(torch.all(f==command.start_frame+ep_step), 'Reward reference clock mismatch')
        ref=self._reference(f,command)
        fp,fR,fv=self.motion.feet(S)
        target=ref['foot_p'][:,0]; contact=ref['contact'][:,0].to(p.dtype)
        valid=ref['contact_valid'][:,0].to(p.dtype)
        swing=1-contact
        epos=(fp-target).square().sum(-1)
        # A zero-at-alignment SO(3) chordal error has a finite gradient at identity.
        eori=(fR-ref['foot_R'][:,0]).square().sum((-1,-2))/2
        position=torch.exp(-epos/c.foot_position_std**2)
        stance_position=torch.exp(-(fp-ref['stance_p'][:,0]).square().sum(-1)/c.foot_position_std**2)
        orientation=torch.exp(-eori/c.foot_orientation_std**2)
        local_corners=self.motion.corners-self.motion.sole_offsets[:,None]
        lowest=(fR[:,:,None]@local_corners[None,...,None]).squeeze(-1)[...,2].min(-1).values
        clear=fp[...,2]+lowest
        support=torch.ones_like(clear,dtype=torch.bool)
        if self.terrain:
            h,support,_=self.terrain.surface(fp[...,:2]);clear=clear-h
        soft_contact=torch.sigmoid((.015-clear)/.01)*torch.exp(-fv.square().sum(-1)/.25**2)*support.to(p.dtype)
        previous=torch.maximum(f-1,self.motion.clip_start[self.motion.clip_id[f]])
        event=(self.motion.contact[f]&~self.motion.contact[previous]&self.motion.contact_valid[f]&self.motion.contact_valid[previous]).to(p.dtype)
        bp,bR,_,_=self.kin.forward(*S)
        centers=bp[:,self.collision_ids]+(bR[:,self.collision_ids]@self.collision_offsets[...,None]).squeeze(-1)
        h=torch.zeros_like(centers[...,2]);collision_valid=torch.ones_like(h,dtype=torch.bool)
        if self.terrain: h,collision_valid,_=self.terrain.surface(centers[...,:2])
        penetration=torch.relu(h+self.collision_radii-centers[...,2])*collision_valid
        pair=self.self_pairs
        distance=(centers[:,pair[:,0]]-centers[:,pair[:,1]]).square().sum(-1).clamp_min(1e-12).sqrt()
        self_penetration=torch.relu(self.inscribed_radii[pair[:,0]]+self.inscribed_radii[pair[:,1]]-distance)
        # PD torque proxy shares command/limits with the real simulator. It is
        # differentiable, not a claim of measured Newton constraint force.
        tau=((self.kp*(self.default_q+self.alpha*a-q)-self.kd*qd)/self.effort).clamp(-1,1)
        raw=dict(swing_position=(swing*position).mean(-1),swing_orientation=(swing*orientation).mean(-1),
                 stance_position=(contact*stance_position).mean(-1),
                 contact_schedule=(valid*(1-(soft_contact-contact).square())).mean(-1),
                 touchdown=(event*position).mean(-1),
                 upright=torch.exp(-(1-R[...,2,2])/0.1),survival=1-self.soft_termination(S,f),
                 slip=-(contact*soft_contact*fv[...,:2].square().sum(-1)/c.slip_speed**2).mean(-1),
                 joint_limit=-(torch.relu(q-self.soft_hi)+torch.relu(self.soft_lo-q)).square().mean(-1),
                 torque=-tau.square().mean(-1), action_rate=-(a-a_prev).square().mean(-1),
                 collision=-(penetration/.05).square().mean(-1),
                 self_collision=-(self_penetration/.05).square().mean(-1) if len(pair) else torch.zeros_like(p[:,0]))
        prior=torch.zeros_like(p[:,0])
        if c.body_prior_weight>0:
            target_q=ref['q'][:,0]
            upper=torch.exp(-(q[:,self.upper_ids]-target_q[:,self.upper_ids]).square().mean(-1)/c.prior_joint_std**2)
            waist=torch.exp(-(q[:,self.waist_ids]-target_q[:,self.waist_ids]).square().mean(-1)/c.prior_joint_std**2)
            rootori=torch.exp(-(R-ref['root_R'][:,0]).square().sum((-1,-2))/2/.3**2)
            rootheight=torch.exp(-(p[:,2]-ref['root_p'][:,0,2]).square()/c.prior_root_height_std**2)
            prior=ref['quality'][:,0]*(c.prior_upper*upper+c.prior_waist*waist+c.prior_root_orientation*rootori+c.prior_root_height*rootheight)
        weighted={k:v*getattr(c,k)*CONTROL_DT for k,v in raw.items()}
        weighted['body_prior']=prior*c.body_prior_weight*CONTROL_DT
        reward=sum(weighted.values())
        if terms:
            zero=torch.zeros_like(reward)
            return reward,dict(**raw,**{k+'_contribution':v for k,v in weighted.items()},
                body_prior=prior,foot_objective=sum(weighted[k] for k in ('swing_position','swing_orientation','stance_position','contact_schedule','touchdown')),
                foot_rmse=epos.mean(-1).sqrt(),contact_valid_fraction=valid.mean(-1),
                head_position_error=zero,head_penalty=zero)
        return reward

    def soft_termination(self,S,f=None,*,command=None,ep_step=None):
        p,R=S[:2]
        floor=torch.zeros_like(p[:,2])
        if self.terrain:
            h,valid,_=self.terrain.surface(p[:,:2]); floor=torch.where(valid,h,floor)
        low=torch.sigmoid((self.cfg.term_height-(p[:,2]-floor))/.03)
        tilt=torch.sigmoid(((1-R[:,2,2])-self.cfg.term_tilt)/.05)
        return 1-(1-low)*(1-tilt)

    def termination(self,S,f=None,*,command=None,ep_step=None):
        p,R=S[:2]
        floor=torch.zeros_like(p[:,2])
        if self.terrain:
            h,valid,_=self.terrain.surface(p[:,:2]);floor=torch.where(valid,h,floor)
        return (p[:,2]-floor<self.cfg.term_height)|((1-R[:,2,2])>self.cfg.term_tilt)

    def contract(self):
        return dict(version=1,mode='roban_foot_reference',config=asdict(self.cfg),
                    obs_dim=self.obs_dim,critic_state_dim=self.critic_dim,control_dt=CONTROL_DT,
                    critic_use_body_reference=self.critic_use_body_reference,
                    terrain=self.terrain.contract() if self.terrain else None,
                    foot_frame='model_collision_sole; current_robot_heading',
                    contact_truth='explicit_valid_labels_only',value_target='original_lambda_return')
