"""Condition reset perturbations on original terrain geometry, never alter a physics step."""
from itertools import product
from pathlib import Path
import mujoco
import numpy as np
import torch
from .rotations import quat_to_mat,so3_exp
from .s54 import MJCF_PATH
from .stair_data import geometry_tools


def segment_box_distance_squared(start,end,lo,hi):
    """Exact convex piecewise-quadratic segment/AABB distance, batched."""
    direction=end-start
    safe=torch.where(direction.abs()>1e-9,direction,torch.ones_like(direction))
    cuts=torch.cat(((lo-start)/safe,(hi-start)/safe),-1).clamp(0,1)
    cuts=torch.cat((torch.zeros_like(cuts[...,:1]),cuts,torch.ones_like(cuts[...,:1])),-1).sort(-1).values
    left,right=cuts[...,:-1],cuts[...,1:];mid=(left+right)/2
    point=start[...,None,:]+direction[...,None,:]*mid[...,None]
    active=(point<lo[...,None,:])|(point>hi[...,None,:])
    bound=torch.where(point<lo[...,None,:],lo[...,None,:],hi[...,None,:])
    denominator=(direction[...,None,:].square()*active).sum(-1)
    optimum=((bound-start[...,None,:])*direction[...,None,:]*active).sum(-1)/denominator.clamp_min(1e-20)
    optimum=torch.where(denominator>1e-20,optimum,mid).maximum(left).minimum(right)
    closest=start[...,None,:]+direction[...,None,:]*optimum[...,None]
    error=(lo[...,None,:]-closest).clamp_min(0)+(closest-hi[...,None,:]).clamp_min(0)
    return error.square().sum(-1).amin(-1)


def draw_reset_noise(state,kin):
    """The original Flat reset ranges; only infeasible pose draws are rejected."""
    p,R,v,w,q,qd=(x.clone() for x in state);n=len(p);dev=p.device
    def sample(bounds):
        return torch.stack([torch.empty(n,device=dev).uniform_(-b,b) for b in bounds],-1)
    p+=sample([.05,.05,.01]);R=so3_exp(sample([.1,.1,.2]))@R
    v+=sample([.5,.5,.2]);w+=sample([.52,.52,.78])
    q=(q+torch.empty_like(q).uniform_(-.1,.1)).clamp(kin.q_lo,kin.q_hi)
    return p,R,v,w,q,qd


class TerrainResetValidator:
    """Conservative proxy OBB/AABB SAT against the UNCHANGED source terrain.

    Boxes use all 15 separating axes. Capsules/spheres use exact segment/AABB
    distance; cylinders use conservative oriented boxes. The nominal fallback
    is validated with the same original 1 mm geometry tolerance.
    """
    def __init__(self,kin,terrain,*,mixed_flat=True,clearance=0.,max_attempts=32):
        self.kin=kin;self.clearance=clearance;self.max_attempts=max_attempts
        model=mujoco.MjModel.from_xml_path(str(MJCF_PATH));indices=[];halves=[]
        for i in range(model.ngeom):
            if not (model.geom_contype[i] or model.geom_conaffinity[i]) or model.geom_bodyid[i]==0:continue
            kind=int(model.geom_type[i]);size=model.geom_size[i]
            if kind==mujoco.mjtGeom.mjGEOM_BOX:half=size.copy()
            elif kind==mujoco.mjtGeom.mjGEOM_SPHERE:half=np.full(3,size[0])
            elif kind in (mujoco.mjtGeom.mjGEOM_CAPSULE,mujoco.mjtGeom.mjGEOM_CYLINDER):
                half=np.array([size[0],size[0],size[1]+(size[0] if kind==mujoco.mjtGeom.mjGEOM_CAPSULE else 0)])
            else:raise ValueError(f'Unsupported collision proxy {kind}')
            indices.append(i);halves.append(half)
        tensor=lambda x:torch.as_tensor(np.array(x),device=kin.device,dtype=torch.float32)
        self.body=torch.tensor(model.geom_bodyid[indices]-1,device=kin.device,dtype=torch.long)
        self.local_p=tensor(model.geom_pos[indices]);self.local_R=quat_to_mat(tensor(model.geom_quat[indices]))
        self.half=tensor(halves)
        self.capsule=torch.tensor([model.geom_type[i] in (mujoco.mjtGeom.mjGEOM_CAPSULE,mujoco.mjtGeom.mjGEOM_SPHERE) for i in indices],device=kin.device)
        self.radius=tensor([model.geom_size[i,0] for i in indices])
        self.segment_half=tensor([model.geom_size[i,1] if model.geom_type[i]==mujoco.mjtGeom.mjGEOM_CAPSULE else 0 for i in indices])
        boxes=[]
        for part in geometry_tools().load_mesh(terrain).split(only_watertight=False):
            bounds=part.bounds.copy()
            if part.is_watertight and len(part.vertices)==8 and len(part.faces)==12:boxes.append(bounds)
            elif bounds[0,2]==bounds[1,2] and len(part.faces)==2:
                # Below-surface envelope for an authored open horizontal patch;
                # this is ONLY a reset test, no collider is added or changed.
                bounds[0,2]-=.1;boxes.append(bounds)
            else:raise ValueError('Reset validation requires the audited box/flat-patch stair asset')
        if mixed_flat:boxes.append(np.array([[8.,-12.,-.1],[32.,12.,0.]]))
        boxes=np.asarray(boxes);self.centers=tensor(boxes.mean(1));self.box_half=tensor((boxes[:,1]-boxes[:,0])/2)
        self.stats=dict(proposals=0,rejected=0,fallbacks=0,resets=0)

    @torch.no_grad()
    def valid(self,state,*,return_clearance=False):
        previous=torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32=False
            p,R,_,_=self.kin.forward(*state,all_bodies=True)
            body_R=R[:,self.body];center=p[:,self.body]+(body_R@self.local_p[...,None])[...,0]
            axes=body_R@self.local_R
            delta=center[:,:,None,:]-self.centers[None,None]
            world_half=(axes.abs()@self.half[...,None])[...,0]
            depth=(world_half[:,:,None]+self.box_half[None,None]-delta.abs()).amin(-1)
            local_delta=(axes.transpose(-1,-2)[:,:,None]@delta[...,None])[...,0]
            local_box=(axes.abs().transpose(-1,-2)[:,:,None]@self.box_half[None,None,:,:,None])[...,0]
            depth=torch.minimum(depth,(self.half[None,:,None]+local_box-local_delta.abs()).amin(-1))
            eye=torch.eye(3,device=center.device)
            for i,j in product(range(3),repeat=2):
                axis=torch.cross(eye[i].expand_as(axes[...,j]),axes[...,j],dim=-1)
                norm=axis.norm(dim=-1)
                ra=(axis.abs()[:,:,None]*self.box_half[None,None]).sum(-1)
                rb=((axes.transpose(-1,-2)@axis[...,None])[...,0].abs()*self.half).sum(-1)
                value=(ra+rb[:,:,None]-(delta*axis[:,:,None]).sum(-1).abs())/norm[:,:,None].clamp_min(1e-8)
                value=torch.where(norm[:,:,None]>1e-6,value,torch.full_like(value,float('inf')))
                depth=torch.minimum(depth,value)
            # Exact rounded-proxy test avoids rejecting valid tilted capsules
            # merely because corners of their enclosing OBB cross the surface.
            c=center[:,self.capsule];axis=axes[:,self.capsule,:,2]
            offset=axis*self.segment_half[self.capsule][None,:,None]
            lo=self.centers-self.box_half;hi=self.centers+self.box_half
            distance=segment_box_distance_squared((c-offset)[:,:,None],(c+offset)[:,:,None],lo[None,None],hi[None,None]).sqrt()
            depth[:,self.capsule]=self.radius[self.capsule][None,:,None]-distance
            return -depth.amax(dim=(-1,-2)) if return_clearance else (depth<=-self.clearance).all(dim=(-1,-2))
        finally:torch.backends.cuda.matmul.allow_tf32=previous

    @torch.no_grad()
    def condition(self,candidate,nominal):
        result=tuple(x.clone() for x in candidate)
        pending=torch.arange(len(result[0]),device=result[0].device)
        self.stats['resets']+=len(pending)
        for _ in range(self.max_attempts):
            if not len(pending):break
            valid=self.valid(tuple(x[pending] for x in result))
            self.stats['proposals']+=len(pending);self.stats['rejected']+=int((~valid).sum())
            pending=pending[~valid]
            if len(pending):
                proposal=draw_reset_noise(tuple(x[pending] for x in nominal),self.kin)
                for dst,src in zip(result,proposal):dst[pending]=src
        if len(pending):
            fallback=tuple(x[pending] for x in nominal)
            if not self.valid(fallback).all():
                raise ValueError('Nominal reset violates terrain clearance; refusing an invalid physical reset')
            self.stats['fallbacks']+=len(pending)
            for dst,src in zip(result,fallback):dst[pending]=src
        return result
