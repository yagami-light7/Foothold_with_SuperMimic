"""Actual Newton solver contacts and independent source-model geometry queries."""
from dataclasses import dataclass
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np

from ..roban import SOURCE_MJCF, JOINT_NAMES, ROOT_BODY, FOOT_BODY_NAMES, foot_geometry
from .pmt_dataset_loader import rotate_z


@dataclass
class ContactSnapshot:
    rows: list
    foot_force: np.ndarray
    pillar_force: np.ndarray
    foot_pillar_force: np.ndarray
    robot_contact_pillars: set
    contact_positions: np.ndarray
    cube_force: np.ndarray


def contacted_pillars(snapshot,body_fragment):
    """Actual pair ownership, including contacts with zero net load."""
    result=set()
    for row in snapshot.rows:
        for side in (0,1):
            if body_fragment in row[f'body{side}']:
                match=re.search(r'/Pillars/pillar_(\d+)',row[f'body{1-side}'])
                if match:result.add(int(match.group(1)))
    return result


class SolverContacts:
    """Last solver substep, force on shape0 sign; never planned contact labels."""
    def __init__(self,backend,sample):
        import warp as wp
        self.wp,self.backend,self.sample=wp,backend,sample
        self.contacts=backend.get_contacts()
        if self.contacts is None:raise RuntimeError('Newton did not allocate contact data')
        c=self.contacts
        if c.force is None:c.force=wp.zeros(c.rigid_contact_max+c.soft_contact_max,dtype=wp.spatial_vector,device=backend.get_model().device)
        model=backend.get_model()
        self.shape_body=wp.to_torch(model.shape_body).cpu().numpy()
        self.labels=model.body_label
        self.shape_labels=model.shape_label

    def read(self):
        wp,c=self.wp,self.contacts
        self.backend._solver.update_contacts(c,self.backend.get_state_0())
        count=int(wp.to_torch(c.rigid_contact_count).sum())
        left=wp.to_torch(c.rigid_contact_shape0)[:count].cpu().numpy()
        right=wp.to_torch(c.rigid_contact_shape1)[:count].cpu().numpy()
        force=wp.to_torch(c.force)[:count,:3].cpu().numpy()
        # Surface contact positions are shape-local; reconstruct world using
        # the same Newton SensorContact convention when available.
        state=self.backend.get_state_0()
        poses=wp.to_torch(state.body_q).cpu().numpy()
        points=wp.to_torch(c.rigid_contact_point0)[:count].cpu().numpy()
        rows=[];feet=np.zeros((2,3));pillars=np.zeros((len(self.sample.sizes),3));fp=np.zeros((2,len(pillars),3));cube=np.zeros(3);positions=[];actual=set()
        for i,(s0,s1) in enumerate(zip(left,right)):
            if s0<0 or s1<0:continue
            b0,b1=int(self.shape_body[s0]),int(self.shape_body[s1])
            names=[self.labels[b] if b>=0 else 'world' for b in (b0,b1)]
            from scipy.spatial.transform import Rotation
            point=points[i] if b0<0 else Rotation.from_quat(poses[b0,3:]).apply(points[i])+poses[b0,:3]
            positions.append(point)
            rows.append(dict(shape0=int(s0),shape1=int(s1),body0=names[0],body1=names[1],force_on_body0=force[i].tolist(),position_w=point.tolist()))
            for a in (0,1):
                match=re.search(r'/Pillars/pillar_(\d+)',names[a])
                if match is None:continue
                pid=int(match.group(1));on_other=force[i]*(-1 if a==0 else 1)
                other=names[1-a].split('/')[-1]
                pillars[pid]-=on_other
                if '/Robot/' in names[1-a]:
                    actual.add(pid)
                    if other in FOOT_BODY_NAMES:
                        side=FOOT_BODY_NAMES.index(other);feet[side]+=on_other;fp[side,pid]+=on_other
                if 'TestCube' in names[1-a]:cube+=on_other
        return ContactSnapshot(rows,feet,pillars,fp,actual,np.array(positions).reshape(-1,3),cube)


class RobotGeometry:
    """MJCF FK/distance-only audit; never advances the real Newton robot."""
    def __init__(self,sample):
        import mujoco
        self.mujoco,self.sample=mujoco,sample
        root=ET.parse(SOURCE_MJCF).getroot()
        root.find('compiler').set('meshdir',str(SOURCE_MJCF.parent.parent/'meshes'))
        world=root.find('worldbody')
        for pid,size in enumerate(sample.sizes):
            ET.SubElement(world,'geom',name=f'm7_pillar_{pid}',type='box',size=' '.join(map(str,size/2)),
                          pos='0 0 -100',contype='1',conaffinity='1',rgba='.12 .32 .48 1')
        self.model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
        self.data=mujoco.MjData(self.model)
        self.pillar_ids=[self.model.geom(f'm7_pillar_{pid}').id for pid in range(len(sample.sizes))]
        self.robot_geoms=[g for g in range(self.model.ngeom) if g not in self.pillar_ids and (self.model.geom_contype[g] or self.model.geom_conaffinity[g])]
        self.foot_ids=[self.model.body(n).id for n in FOOT_BODY_NAMES]
        self.joints=[self.model.joint(n).id for n in JOINT_NAMES]

    def set_robot(self,root_pos,root_quat,joint_pos,root_vel=None,joint_vel=None):
        m,d=self.model,self.data
        d.qpos[:3]=root_pos;d.qpos[3:7]=root_quat
        d.qvel[:]=0
        for j,q in zip(self.joints,joint_pos):d.qpos[m.jnt_qposadr[j]]=q
        if root_vel is not None:
            from scipy.spatial.transform import Rotation
            R=Rotation.from_quat(np.asarray(root_quat)[[1,2,3,0]]).as_matrix()
            d.qvel[:3]=root_vel[:3];d.qvel[3:6]=R.T@root_vel[3:]
        if joint_vel is not None:
            for j,v in zip(self.joints,joint_vel):d.qvel[m.jnt_dofadr[j]]=v
        self.mujoco.mj_forward(m,d)

    def set_pillars(self,poses,active):
        for g,p,on in zip(self.pillar_ids,poses,active):
            self.model.geom_pos[g]=p[:3]
            self.model.geom_quat[g]=[np.cos(p[3]/2),0,0,np.sin(p[3]/2)]
            self.model.geom_rgba[g,3]=1. if on else 0.
        self.mujoco.mj_forward(self.model,self.data)

    def distances(self,poses=None,active=None):
        old=self.model.geom_pos[self.pillar_ids].copy(),self.model.geom_quat[self.pillar_ids].copy()
        if poses is not None:self.set_pillars(poses,np.ones(len(poses),dtype=bool) if active is None else active)
        distances=np.full(len(self.pillar_ids),np.inf)
        for pid,g in enumerate(self.pillar_ids):
            for r in self.robot_geoms:
                distance=self.mujoco.mj_geomDistance(self.model,self.data,r,g,10.,None)
                distances[pid]=min(distances[pid],float(distance))
        if poses is not None:
            self.model.geom_pos[self.pillar_ids]=old[0];self.model.geom_quat[self.pillar_ids]=old[1]
            self.mujoco.mj_forward(self.model,self.data)
        return distances

    def feet(self):
        p=[];R=[];v=[];corners=[]
        for b,geom in zip(self.foot_ids,foot_geometry()):
            rot=self.data.xmat[b].reshape(3,3)
            pos=self.data.xpos[b]+rot@geom['sole'];p.append(pos);R.append(rot)
            jac=np.zeros((3,self.model.nv));jr=jac.copy()
            self.mujoco.mj_jac(self.model,self.data,jac,jr,pos,b)
            v.append(jac@self.data.qvel)
            corners.append(np.array(geom['corners'])@rot.T+self.data.xpos[b])
        return np.array(p),np.array(R),np.array(v),np.array(corners)

    @property
    def com(self):
        return self.data.subtree_com[self.model.body(ROOT_BODY).id].copy()


def support_margin(com,feet_corners,foot_force,min_force=10.):
    """Actual force-supported sole polygon; NaN denotes no measured support."""
    from scipy.spatial import ConvexHull
    mask=foot_force[:,2]>=min_force
    points=feet_corners[mask,:,:2].reshape(-1,2)
    if len(points)<3:return float('nan'),np.empty((0,2))
    hull=ConvexHull(points)
    polygon=points[hull.vertices]
    # QHull equations: a*x+b*y+c <= 0 is inside.
    signed=-(hull.equations[:,:2]@np.asarray(com)[:2]+hull.equations[:,2])
    return float(signed.min()),polygon


def contact_patch_margin(com,snapshot):
    """Convex hull of measured load-bearing foot/pillar contact points."""
    from scipy.spatial import ConvexHull,QhullError
    points=[]
    for row in snapshot.rows:
        for side in (0,1):
            if row[f'body{side}'].split('/')[-1] in FOOT_BODY_NAMES and '/Pillars/' in row[f'body{1-side}']:
                force=np.array(row['force_on_body0'])*(1 if side==0 else -1)
                if force[2]>.1:points.append(row['position_w'][:2])
    points=np.unique(np.asarray(points).reshape(-1,2),axis=0)
    if len(points)<3:return float('nan'),np.empty((0,2))
    try:hull=ConvexHull(points)
    except QhullError:return float('nan'),np.empty((0,2))
    margin=-(hull.equations[:,:2]@np.asarray(com)[:2]+hull.equations[:,2])
    return float(margin.min()),points[hull.vertices]
