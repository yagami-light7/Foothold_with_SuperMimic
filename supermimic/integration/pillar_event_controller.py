"""Fail-closed recycling using actual contacts, geometry and consecutive frames."""
import numpy as np

from .pmt_dataset_loader import require, rotate_z


class PillarEventBlocked(RuntimeError):
    pass


def boxes_overlap(a,sa,b,sb,margin=0.):
    """Exact separating-axis test for upright OBBs, including vertical bounds."""
    if abs(a[2]-b[2]) >= (sa[2]+sb[2])/2+margin:return False
    A,B=rotate_z(a[3])[:2,:2],rotate_z(b[3])[:2,:2]
    delta=b[:2]-a[:2]
    for axis in (*A.T,*B.T):
        r0=np.sum(abs(A.T@axis)*sa[:2]/2);r1=np.sum(abs(B.T@axis)*sb[:2]/2)
        if abs(delta@axis)>=r0+r1+margin:return False
    return True


class PillarEventController:
    def __init__(self,sample,apply_collider,*,safe_frames=2,clearance=.001):
        require(isinstance(safe_frames,int) and safe_frames>=1 and clearance>=0,'Invalid recycling guard')
        self.sample,self.apply=sample,apply_collider
        self.safe_frames,self.clearance=safe_frames,clearance
        self.last_frame=-1;self.safe=np.zeros(len(sample.sizes),dtype=int)
        self.log=[];self.blocked=False
        self.poses=sample.schedule['pose_xyzyaw'][0].copy()
        self.active=sample.schedule['active'][0].copy();self.states=sample.schedule['state'][0].copy()

    def advance(self,frame,contacts,geometry,*,source='Newton solver contact set'):
        require(not self.blocked,'Recycling controller already terminated')
        require(frame==self.last_frame+1,'Reference/event frames must advance consecutively from zero')
        require(isinstance(contacts,(set,frozenset)) and all(isinstance(i,(int,np.integer)) and 0<=i<len(self.safe) for i in contacts),'Actual pillar ID contact set required; planned masks are not accepted')
        require(source in ('Newton solver contact set','actual Newton test cube contact','measured no-contact cube world; no robot instantiated','kinematic geometry only (not force truth)'), 'Unrecognized contact provenance')
        distances=geometry.distances(self.poses,self.active)
        for pid in range(len(self.safe)):
            clear=pid not in contacts and distances[pid]>self.clearance
            self.safe[pid]=self.safe[pid]+1 if clear else 0
        target,active,states=self.sample.pillar_layout(frame)
        changed=np.any(target!=self.poses,axis=-1)|(active!=self.active)
        starts=[e for e in self.sample.events if int(round(e['timestamp']/.02))==frame]
        # Validate all updates first; no partially executed multi-pillar frame.
        for e in starts:
            pid=e['pillar_id'];reason=None
            if pid in contacts:reason='actual_robot_contact_on_recycled_pillar'
            elif distances[pid]<=self.clearance:reason='robot_geometry_not_clear_of_old_pillar'
            elif self.safe[pid]<self.safe_frames:reason='insufficient_consecutive_safe_frames'
            final=target.copy();final[pid]=e['pose_after']
            target_dist=geometry.distances(final,np.ones(len(active),dtype=bool))[pid]
            if reason is None and target_dist<=self.clearance:reason='new_target_intersects_robot_geometry'
            if reason is None:
                for other in range(len(active)):
                    if other!=pid and self.active[other] and boxes_overlap(final[pid],self.sample.sizes[pid],self.poses[other],self.sample.sizes[other],.001):
                        reason='new_target_intersects_other_pillar';break
            record=dict(frame=frame,time_s=frame*.02,event=e,contact_source=source,
                actual_contact_pillars=sorted(contacts),safe_frames=int(self.safe[pid]),
                old_geometry_distance_m=float(distances[pid]),target_geometry_distance_m=float(target_dist),
                status='BLOCKED' if reason else 'GUARD_PASS',reason=reason)
            self.log.append(record)
            if reason:
                self.blocked=True
                raise PillarEventBlocked(f'{reason}: pillar {pid} at frame {frame}')
        for pid in np.flatnonzero(changed):
            if pid in contacts:
                self.blocked=True
                self.log.append(dict(frame=frame,pillar_id=int(pid),status='BLOCKED',reason='contact_during_relocation_or_enable',actual_contact_pillars=sorted(contacts)))
                raise PillarEventBlocked(f'Actual contact prevents changing pillar {pid}')
            if active[pid] and not self.active[pid]:
                final_distance=geometry.distances(target,active)[pid]
                if final_distance<=self.clearance:
                    self.blocked=True
                    self.log.append(dict(frame=frame,pillar_id=int(pid),status='BLOCKED',reason='robot_overlap_at_reenable',distance_m=float(final_distance)))
                    raise PillarEventBlocked(f'Robot overlap at reenable pillar {pid}')
        for pid in np.flatnonzero(changed):
            self.apply(int(pid),target[pid],bool(active[pid]))
            self.log.append(dict(frame=frame,time_s=frame*.02,pillar_id=int(pid),
                transition=f'{self.states[pid]} -> {states[pid]}',collision_enabled=bool(active[pid]),
                pose_before=self.poses[pid].tolist(),pose_after=target[pid].tolist(),status='APPLIED'))
        self.poses,self.active,self.states=target,active,states
        self.last_frame=frame
