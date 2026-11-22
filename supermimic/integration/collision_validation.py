"""Real cube dynamics against PMT mesh pillars; no Roban in this experiment."""
import json
import numpy as np
import torch

from .pillar_scene import scene_config,PillarScene,make_pillar_sim
from .contact_diagnostics import SolverContacts,contacted_pillars
from .pmt_dataset_loader import rotate_z
from .pillar_event_controller import PillarEventController,PillarEventBlocked


def run(sample,args,out):
    from isaaclab.app import launch_simulation
    from isaaclab.sim import SimulationContext
    from isaaclab.scene import InteractiveScene
    from ..env import make_sim
    cfg=make_pillar_sim('cuda:0',args)
    traces={};rows=[]
    with launch_simulation(cfg,args):
        sim=SimulationContext(cfg)
        scene=InteractiveScene(scene_config(sample,robot=False,cube=True))
        sim.reset();scene.update(0.)
        terrain=PillarScene(sim,scene,sample)
        contacts=SolverContacts(sim.backend_manager,sample)
        cube=scene['cube'];dt=.02
        def reset_cube(position,velocity=(0,0,0)):
            cube.reset()
            cube.write_root_link_pose_to_sim_index(root_pose=torch.tensor([[*position,0.,0.,0.,1.]],device=sim.device,dtype=torch.float32))
            cube.write_root_link_velocity_to_sim_index(root_velocity=torch.tensor([[*velocity,0.,0.,0.]],device=sim.device,dtype=torch.float32))
            sim.forward();scene.update(0.)
        def episode(name,position,velocity=(0,0,0),steps=75,pid=0):
            reset_cube(position,velocity)
            trace={k:[] for k in ('position','velocity','force','time','contact_position','contact_valid','pillar_pose','pillar_collider_pose','pillar_active')}
            all_contacts=[]
            for tick in range(steps):
                scene.write_data_to_sim();sim.step(render=False);scene.update(dt)
                snapshot=contacts.read()
                p=cube.data.root_link_pose_w.torch[0,:3].cpu().numpy().copy()
                v=cube.data.root_link_vel_w.torch[0,:3].cpu().numpy().copy()
                matches=[r for r in snapshot.rows if ('TestCube' in r['body0'] or 'TestCube' in r['body1']) and (f'/Pillars/pillar_{pid}' in r['body0'] or f'/Pillars/pillar_{pid}' in r['body1'])]
                point=np.mean([r['position_w'] for r in matches],axis=0) if matches else np.zeros(3)
                for k,val in dict(position=p,velocity=v,force=snapshot.cube_force,time=(tick+1)*dt,contact_position=point,contact_valid=bool(matches),
                    pillar_pose=terrain.poses.copy(),pillar_collider_pose=terrain.collider_poses.copy(),pillar_active=terrain.active.copy()).items():trace[k].append(val)
                all_contacts.extend(dict(tick=tick+1,**r) for r in matches)
            trace={k:np.array(v) for k,v in trace.items()}
            for k,v in trace.items():traces[f'{name}__{k}']=v
            top=terrain.poses[pid,2]+sample.sizes[pid,2]/2
            metrics=dict(name=name,pillar_id=pid,final_position=trace['position'][-1].tolist(),final_velocity=trace['velocity'][-1].tolist(),
                max_force_N=np.linalg.norm(trace['force'],axis=-1).max().item(),last10_mean_vertical_force_N=float(trace['force'][-10:,2].mean()),
                contact_samples=int(trace['contact_valid'].sum()),pillar_top_m=float(top),contact_rows=all_contacts)
            rows.append(metrics)
            return metrics,trace
        if not sample.dynamic:
            for pid,p in enumerate(terrain.poses.copy()):
                top=p[2]+sample.sizes[pid,2]/2
                r,t=episode(f'top_drop_{pid}',[*p[:2],top+.22],pid=pid)
                r['passed']=r['contact_samples']>0 and abs(t['position'][-1,2]-(top+.02))<.005 and abs(t['velocity'][-1,2])<.03 and 8.<r['last10_mean_vertical_force_N']<12.
                points=t['contact_position'][t['contact_valid']]
                r['max_top_contact_height_error_m']=float(abs(points[:,2]-top).max()) if len(points) else None
                resting=t['contact_position'][-10:][t['contact_valid'][-10:]]
                r['resting_contact_height_error_m']=float(abs(resting[:,2]-top).max()) if len(resting) else None
                r['passed']=bool(r['passed'] and r['resting_contact_height_error_m']<.005)
            gap=(terrain.poses[0,:2]+terrain.poses[1,:2])/2
            r,t=episode('empty_gap',[*gap,.45],steps=50)
            r['passed']=bool(r['max_force_N']<1e-5 and t['position'][-1,2]<-.5)
            p=terrain.poses[0];normal=rotate_z(p[3])[:,0]
            position=p[:3]+normal*(sample.sizes[0,0]/2+.09)
            r,t=episode('side_collision',position,-normal,steps=30)
            sideforce=np.abs(t['force']@normal)
            r['peak_side_normal_force_N']=float(sideforce.max())
            r['passed']=bool(sideforce.max()>1. and t['contact_valid'].any())
            start=p[:3]+rotate_z(p[3])@np.array([-.04,0,sample.sizes[0,2]/2+.021])
            r,t=episode('friction_slide',start,normal*.25,steps=40)
            r['initial_horizontal_speed_m_s']=.25
            r['final_horizontal_speed_m_s']=float(np.linalg.norm(t['velocity'][-1,:2]))
            r['horizontal_travel_m']=float(np.linalg.norm(t['position'][-1,:2]-start[:2]))
            r['passed']=bool(r['contact_samples']>0 and r['final_horizontal_speed_m_s']<.03 and r['horizontal_travel_m']<.04 and abs(t['position'][-1,2]-(r['pillar_top_m']+.02))<.005)
            event_log=[]
        else:
            pid=2;before=terrain.poses[pid].copy();event=sample.events[0];after=np.array(event['pose_after'])
            r,t=episode('active_old_support',[*before[:2],before[2]+sample.sizes[pid,2]/2+.22],pid=pid)
            r['passed']=bool(r['last10_mean_vertical_force_N']>8.)
            snap=contacts.read()
            # Deliberately use actual cube contacts to test the same fail-closed guard.
            cube_contact=contacted_pillars(snap,'TestCube')
            class ProbeGeometry:
                def distances(self,poses,active):
                    pos=cube.data.root_link_pose_w.torch[0,:3].cpu().numpy()
                    return np.array([np.linalg.norm(np.maximum(abs((pos-p[:3])@rotate_z(p[3]))-sample.sizes[i]/2-.02,0)) for i,p in enumerate(poses)])
            guard=PillarEventController(sample,terrain.apply,safe_frames=2)
            guard.advance(0,cube_contact,ProbeGeometry(),source='actual Newton test cube contact')
            blocked=False
            try:guard.advance(1,cube_contact,ProbeGeometry(),source='actual Newton test cube contact')
            except PillarEventBlocked:blocked=True
            rows.append(dict(name='loaded_relocation_guard',passed=blocked and np.array_equal(terrain.poses[pid],before),actual_cube_force_N=snap.cube_force.tolist()))
            event_log=guard.log.copy()
            reset_cube([0.,2.,2.])
            for _ in range(3):scene.write_data_to_sim();sim.step(render=False);scene.update(dt)
            terrain.apply(pid,before,False)
            r,t=episode('disabled_old_void',[*before[:2],before[2]+sample.sizes[pid,2]/2+.22],pid=pid,steps=50)
            r['passed']=bool(r['max_force_N']<1e-5 and t['position'][-1,2]<-.5)
            reset_cube([0.,2.,2.]);terrain.apply(pid,after,False)
            r,t=episode('relocated_disabled_void',[*after[:2],after[2]+sample.sizes[pid,2]/2+.22],pid=pid,steps=50)
            r['passed']=bool(r['max_force_N']<1e-5 and t['position'][-1,2]<-.5)
            reset_cube([0.,2.,2.]);terrain.apply(pid,after,True)
            r,t=episode('reenabled_new_support',[*after[:2],after[2]+sample.sizes[pid,2]/2+.22],pid=pid)
            r['passed']=bool(r['last10_mean_vertical_force_N']>8. and abs(t['position'][-1,2]-(r['pillar_top_m']+.02))<.005)
            r,t=episode('old_site_after_relocation_void',[*before[:2],before[2]+sample.sizes[pid,2]/2+.22],pid=pid,steps=50)
            r['passed']=bool(r['max_force_N']<1e-5 and t['position'][-1,2]<-.5)
            # Complete all four scheduled events with a distant, noncontact cube.
            reset_cube([0.,3.,3.])
            for i,p in enumerate(sample.schedule['pose_xyzyaw'][0]):terrain.apply(i,p,True)
            controller=PillarEventController(sample,terrain.apply,safe_frames=2)
            for frame in range(sample.frames):
                snap=contacts.read()
                controller.advance(frame,contacted_pillars(snap,'TestCube'),ProbeGeometry(),source='measured no-contact cube world; no robot instantiated')
                scene.write_data_to_sim();sim.step(render=False);scene.update(dt)
            event_log+=controller.log
            rows.append(dict(name='complete_unloaded_event_schedule',passed=controller.last_frame==sample.frames-1 and sum(e['status']=='GUARD_PASS' for e in controller.log)==4))
        evidence=terrain.evidence()
    np.savez_compressed(out/'cube_trajectories.npz',**traces)
    np.savez_compressed(out/'actual_trajectory.npz',**traces)
    np.savez_compressed(out/'reference_trajectory.npz',pillar_pose=sample.schedule['pose_xyzyaw'],pillar_active=sample.schedule['active'],time_s=sample.schedule['time_s'],
                        purpose=np.array('geometry authority; cube drops are not robot reference tracking'))
    np.savez_compressed(out/'contact_diagnostics.npz',**{k:v for k,v in traces.items() if any(k.endswith('__'+n) for n in ('force','contact_position','contact_valid','time'))})
    np.savez_compressed(out/'torque_diagnostics.npz',status=np.array('not applicable: unactuated falling cube; no robot instantiated'))
    (out/'pillar_event_log.json').write_text(json.dumps(event_log,indent=2)+'\n')
    report=dict(status='PASS' if all(r['passed'] for r in rows) else 'FAIL',scope='real Newton dynamics; robot absent',
        scene_id=sample.name,physics_integration_pass=all(r['passed'] for r in rows),locomotion_performance_pass=None,
        cube_size_m=[.04,.04,.04],cube_mass_kg=1.,control_dt=.02,physics_dt=.005,
        contact_source='Newton last-substep solver forces; surface points in world coordinates',terrain=evidence,tests=rows)
    (out/'evaluation_metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report
