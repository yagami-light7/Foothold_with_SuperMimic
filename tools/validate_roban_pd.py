"""Independent Roban PD measurements on the training Newton/MJWarp backend.

Does not train a policy or change production gains. Fixed-chest tests isolate
joint dynamics; free-root holds test nominal standing, without balance control.
"""
import os
os.environ['SUPERMIMIC_ROBOT'] = 'roban'
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(os.environ.get('SUPERMIMIC_REPO', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))


def step_metrics(time, q, target, joint, amp):
    import numpy as np
    y = (q[:, joint] - q[0, joint]) / amp
    first = lambda threshold: float(time[np.flatnonzero(y >= threshold)[0]]) if np.any(y >= threshold) else None
    error = abs(q[:, joint] - target[joint])
    band = max(.05 * abs(amp), .0025)
    outside = np.flatnonzero(error > band)
    settled = 0. if not len(outside) else (float(time[outside[-1]+1]) if outside[-1]+1 < len(time) else None)
    return dict(t10_s=first(.1), t90_s=first(.9),
                rise_10_90_s=None if first(.9) is None else first(.9)-first(.1),
                overshoot_pct=float(max(0., y.max()-1)*100), settling_5pct_s=settled,
                settling_band_rad=band, final_mean_error_rad=float(np.mean(error[-10:])),
                peak_speed_rad_s=float(np.max(abs(np.diff(q[:,joint])/np.diff(time)))))


def source_postures():
    import numpy as np
    from supermimic.roban import JOINT_NAMES, default_joint_pos_list, model_contract
    path=REPO/'library/motion_reference/roban/style100_roban_50hz/Neutral/Neutral_FW.csv'
    with path.open() as f:
        reader=csv.reader(f);names=next(reader);rows=list(reader)
    postures=np.array([default_joint_pos_list()] + [[float(rows[t][names.index(n)]) for n in JOINT_NAMES] for t in (0,50)])
    model,contract=model_contract()
    inertias=[]
    import mujoco
    for q in postures:
        data=mujoco.MjData(model)
        for r,value in zip(contract,q):data.qpos[r['qposadr']]=value
        mujoco.mj_forward(model,data)
        mass=np.zeros((model.nv,model.nv))
        # MuJoCo 3.12 moved the sparse source from data.qM to MjData itself.
        try:mujoco.mj_fullM(model,data,mass)
        except TypeError:mujoco.mj_fullM(model,mass,data.qM)
        ids=[r['dofadr'] for r in contract]
        inertias.append(np.diag(mass[np.ix_(ids,ids)]))
    return postures, np.array(inertias), ['source_neutral','Neutral_FW_frame_0','Neutral_FW_frame_50']


def main():
    from isaaclab.app import add_launcher_args,launch_simulation
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['fixed','standing'],required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--include_damping_probe',action='store_true',help='Also measure an inertia-based diagnostic damping profile')
    add_launcher_args(p);args=p.parse_args()
    if args.out.exists():p.error('Choose a new output directory')
    import numpy as np
    import torch
    import warp as wp
    from isaaclab.sim import SimulationContext
    from isaaclab.scene import InteractiveScene
    from isaaclab.utils import replace
    from supermimic.env import make_sim,TrackingSceneCfg
    from supermimic.robot import ROBOT_CFG
    from supermimic.roban import JOINT_NAMES,NUM_J,actuator_parameters,model_contract,foot_geometry,FOOT_BODY_NAMES,SOURCE_MJCF
    from supermimic.transfer import physical_contract
    torch.manual_seed(0)
    postures,inertias,posture_names=source_postures()
    params=actuator_parameters()
    kp=np.array([params[n]['stiffness'] for n in JOINT_NAMES])
    kd=np.array([params[n]['damping'] for n in JOINT_NAMES])
    effort=np.array([params[n]['effort'] for n in JOINT_NAMES])
    limits=np.array([r['range'] for r in model_contract()[1]])
    probe_kd=1.4*np.sqrt(kp*np.median(inertias,axis=0))
    profiles=[('baseline',kd)]
    if args.include_damping_probe:profiles.append(('inertia_damping_probe',probe_kd))
    if args.mode=='fixed':
        rows=[dict(posture=i,joint=j,direction=d) for i in range(3) for j in range(NUM_J) for d in (-1,1)]
        initial=np.stack([postures[r['posture']] for r in rows])
    else:
        rows=[dict(pitch_deg=a) for a in (0.,-1.,1.,-2.,2.)]
        initial=np.repeat(postures[:1],len(rows),axis=0)
    n=len(rows);dev='cuda:0';dt=.02
    cfg=make_sim(dev,args)
    if args.mode=='fixed':cfg.gravity=(0.,0.,0.)
    args.out.mkdir(parents=True)
    traces={};results=[];holds=[]
    with launch_simulation(cfg,args):
        sim=SimulationContext(cfg)
        rcfg=replace(ROBOT_CFG,prim_path='{ENV_REGEX_NS}/Robot')
        if args.mode=='fixed':rcfg.init_state.pos=(0.,0.,3.)
        if args.mode=='fixed':
            # Author the fixture INSIDE spawning: Newton parses/clones USD
            # during InteractiveScene construction, before sim.reset().
            from pxr import UsdPhysics,Gf,UsdGeom
            from isaaclab.sim.utils import get_current_stage
            spawn=rcfg.spawn.func
            def spawn_with_fixture(*a,**kw):
                prim=spawn(*a,**kw)
                stage=get_current_stage()
                path='/World/envs/env_0/Robot/Geometry/base_link'
                link=stage.GetPrimAtPath(path)
                link.GetAttribute('newton:jointsAddMobility').Set(False)
                joint=UsdPhysics.FixedJoint.Define(stage,'/World/envs/env_0/Robot/PDTestFixture')
                joint.CreateBody1Rel().SetTargets([path])
                point=UsdGeom.XformCache().GetLocalToWorldTransform(link).ExtractTranslation()
                joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*point))
                return prim
            rcfg.spawn.func=spawn_with_fixture
        scene_cfg=TrackingSceneCfg(num_envs=n,env_spacing=3.,replicate_physics=True)
        scene_cfg.robot=rcfg
        scene=InteractiveScene(scene_cfg)
        sim.reset();scene.update(0.)
        robot=scene['robot'];backend=sim.backend_manager
        solver=backend._solver
        c2s=torch.tensor([robot.joint_names.index(name) for name in JOINT_NAMES],device=dev)
        s2c=torch.argsort(c2s)
        def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device=dev)
        def canonical(value):return value.torch[:,c2s].clone().cpu().numpy()
        def gains(damping):
            robot.write_joint_damping_to_sim_index(damping=tensor(np.broadcast_to(damping,(n,NUM_J)).copy())[:,s2c])
            np.testing.assert_allclose(canonical(robot.data.joint_damping),np.broadcast_to(damping,(n,NUM_J)),rtol=1e-5)
            np.testing.assert_allclose(canonical(robot.data.joint_stiffness),np.broadcast_to(kp,(n,NUM_J)),rtol=1e-5)
        def reset(q):
            scene.reset()
            if args.mode=='standing':
                # Put all source-neutral sole corners exactly on the plane.
                from supermimic.kinematics import RobotKinematics
                from supermimic.rotations import so3_exp,mat_to_quat
                kin=RobotKinematics(dev)
                angles=tensor([r['pitch_deg'] for r in rows])*np.pi/180
                rv=torch.zeros(n,3,device=dev);rv[:,1]=angles
                R=so3_exp(rv)
                p0=torch.zeros(n,3,device=dev)
                bp,bR,*_=kin.forward(p0,R,p0,p0,tensor(q),tensor(q*0))
                corners=[]
                from supermimic.roban import TRACKED_BODIES
                for f in foot_geometry():
                    b=TRACKED_BODIES.index(f['body'])
                    corners.append(bp[:,b,None]+(bR[:,b,None]@tensor(f['corners'])[None,:,:,None]).squeeze(-1))
                p0[:,2]=-torch.cat(corners,1)[:,:,2].min(-1).values+.001
                quat=mat_to_quat(R)
                pose=torch.cat((p0+scene.env_origins,quat[:,1:],quat[:,:1]),-1)
                robot.write_root_link_pose_to_sim_index(root_pose=pose)
                robot.write_root_link_velocity_to_sim_index(root_velocity=torch.zeros(n,6,device=dev))
            robot.write_joint_position_to_sim_index(position=tensor(q)[:,s2c])
            robot.write_joint_velocity_to_sim_index(velocity=tensor(q*0)[:,s2c])
            robot.set_joint_position_target_index(target=tensor(q)[:,s2c])
            scene.write_data_to_sim();sim.forward();scene.update(0.)
        contacts=None
        def sample():
            value=dict(q=canonical(robot.data.joint_pos),qd=canonical(robot.data.joint_vel),
                root=robot.data.root_link_pose_w.torch.clone().cpu().numpy(),
                torque=wp.to_torch(solver.mjw_data.qfrc_actuator).clone().cpu().numpy())
            c=solver.mjw_data.contact
            count=int(wp.to_torch(solver.mjw_data.nacon)[0])
            world=wp.to_torch(c.worldid)[:count].long()
            address=wp.to_torch(c.efc_address)[:count]
            active=(address>=0) if address.ndim==1 else (address>=0).any(-1)
            value['active_contact_count']=torch.bincount(world[active],minlength=n).cpu().numpy()
            if contacts is not None:
                counts,force,other=contacts.read()
                value.update(foot_contact_count=counts,foot_ground_force=force,other_ground_contact_count=other)
            return value
        reset(initial)
        mj_names=[solver.mj_model.joint(j).name for j in range(solver.mj_model.njnt)]
        dofs=[]
        for name in JOINT_NAMES:
            matched=[j for j,s in enumerate(mj_names) if s.endswith('/'+name) or s.endswith('_'+name) or s==name]
            if len(matched)!=1:raise ValueError((name,mj_names))
            dofs.append(int(solver.mj_model.jnt_dofadr[matched[0]]))
        if args.mode=='standing':
            from types import SimpleNamespace
            from supermimic.standing_diagnostics import NewtonGroundContacts
            contacts=NewtonGroundContacts(SimpleNamespace(physics=backend,device=dev,N=n))
        def tick(target):
            robot.set_joint_position_target_index(target=tensor(target)[:,s2c])
            scene.write_data_to_sim();sim.step(render=False);scene.update(dt)
            data=sample();data['torque']=data['torque'][:,dofs]
            if not all(np.isfinite(v).all() for v in data.values()):raise FloatingPointError('Nonfinite physics')
            return data
        def record(key,targets):
            first=sample();first['torque']=first['torque'][:,dofs]
            np.testing.assert_allclose(first['q'],initial,atol=1e-5,rtol=0)
            trace={name:[value] for name,value in first.items()}
            roots0=first['root'][:,:3].copy()
            for t,target in enumerate(targets):
                data=tick(target)
                for name,value in data.items():trace[name].append(value)
                if (t+1)%100==0:print(f'{key}: {(t+1)*dt:.2f}s simulated',flush=True)
            trace={name:np.stack(value) for name,value in trace.items()}
            trace['target']=np.concatenate((initial[None],np.array(targets)))
            trace['time']=np.arange(len(targets)+1)*dt
            if args.mode=='fixed':
                drift=float(abs(trace['root']-first['root'][None]).max())
                if drift>1e-5:raise AssertionError(f'Fixed fixture moved {drift}')
            for name,value in trace.items():traces[f'{key}__{name}']=value
            return trace
        metadata=dict(joint_names=JOINT_NAMES,physics=physical_contract(),backend='Newton/MJWarp',
            actual_torque_source='solver.mjw_data.qfrc_actuator, generalized hinge force; last substep only',
            control_dt_s=dt,physics_substeps=4,sample_period_s=dt,mode=args.mode,rows=rows,
            test_gravity=list(cfg.gravity),
            kp=kp.tolist(),baseline_kd=kd.tolist(),inertia_probe_kd=probe_kd.tolist(),
            fixed_base_mass_diagonal_kg_m2=inertias.tolist(),postures=postures.tolist(),posture_names=posture_names,
            source_mjcf_sha256=hashlib.sha256(SOURCE_MJCF.read_bytes()).hexdigest(),
            runtime_model_nv=solver.mj_model.nv,runtime_model_neq=solver.mj_model.neq,
            passive_damping=solver.mj_model.dof_damping.tolist(),
            screening_criteria=dict(t90_s=.15,overshoot_pct=20.,settling_5pct_s=.4,final_mean_error_rad=.005),
            criteria_scope='Engineering screening thresholds, not Roban hardware specifications')
        for profile,damping in profiles:
            gains(damping)
            if args.mode=='fixed':
                for requested in (.05,.15):
                    reset(initial)
                    target=initial.copy()
                    amplitudes=[]
                    for i,r in enumerate(rows):
                        j=r['joint'];target[i,j]=np.clip(initial[i,j]+r['direction']*requested,limits[j,0]+.01,limits[j,1]-.01)
                        amplitudes.append(target[i,j]-initial[i,j])
                    key=f'{profile}_step_{requested:.2f}'
                    trace=record(key,[target]*100)
                    for i,r in enumerate(rows):
                        metric=step_metrics(trace['time'],trace['q'][:,i],target[i],r['joint'],amplitudes[i])
                        j=r['joint'];ratio=abs(trace['torque'][1:,i,j])/effort[j]
                        metric.update(profile=profile,kind='step',requested_amplitude_rad=requested,
                            amplitude_rad=float(amplitudes[i]),joint=JOINT_NAMES[j],posture=posture_names[r['posture']],
                            direction=r['direction'],peak_actual_torque_Nm=float(abs(trace['torque'][1:,i,j]).max()),
                            saturation_fraction_last_substep=float(np.mean(ratio>=.98)),
                            active_contact_samples=int(np.sum(trace['active_contact_count'][1:,i]>0)),
                            cross_joint_peak_rad=float(abs(np.delete(trace['q'][:,i]-initial[i],j,axis=1)).max()))
                        metric['screen_pass']=all(metric[k] is not None and metric[k]<=limit for k,limit in metadata['screening_criteria'].items())
                        metric['contact_free']=metric['active_contact_samples']==0
                        results.append(metric)
                reset(initial)
                frequency=np.array([1. if r['direction']<0 else 2. for r in rows])
                amplitude=np.array([max(.005,min(.1,initial[i,r['joint']]-limits[r['joint'],0]-.01,limits[r['joint'],1]-.01-initial[i,r['joint']])) for i,r in enumerate(rows)])
                targets=[]
                for t in np.arange(400)*dt:
                    target=initial.copy()
                    for i,r in enumerate(rows):target[i,r['joint']]+=amplitude[i]*np.sin(2*np.pi*frequency[i]*t)
                    targets.append(target)
                trace=record(f'{profile}_sine',targets)
                keep=trace['time']>=2.
                for i,r in enumerate(rows):
                    t=trace['time'][keep];f=frequency[i];j=r['joint']
                    X=np.column_stack((np.ones(len(t)),np.sin(2*np.pi*f*t),np.cos(2*np.pi*f*t)))
                    coef=np.linalg.lstsq(X,trace['q'][keep,i,j],rcond=None)[0]
                    # Relative to the scheduled continuous sine, so this phase
                    # includes the 50 Hz command's zero-order hold.
                    phase=-np.arctan2(coef[2],coef[1])
                    results.append(dict(profile=profile,kind='sine',joint=JOINT_NAMES[j],posture=posture_names[r['posture']],
                        frequency_hz=float(f),amplitude_rad=float(amplitude[i]),gain=float(np.hypot(*coef[1:])/amplitude[i]),
                        phase_lag_deg=float(phase*180/np.pi),equivalent_delay_s=float(phase/(2*np.pi*f)),
                        active_contact_samples=int(np.sum(trace['active_contact_count'][keep,i]>0)),
                        rmse_rad=float(np.sqrt(np.mean((trace['q'][keep,i,j]-trace['target'][keep,i,j])**2))),
                        saturation_fraction_last_substep=float(np.mean(abs(trace['torque'][keep,i,j])/effort[j]>=.98))))
            else:
                reset(initial)
                trace=record(f'{profile}_hold',[initial]*250)
                pose=trace['root'];quat=pose[:,:,3:7]
                rzz=1-2*(quat[:,:,0]**2+quat[:,:,1]**2)
                height=pose[:,:,2]-scene.env_origins[:,2].cpu().numpy()
                for i,r in enumerate(rows):
                    failed=(height[:,i]<.35)|(rzz[:,i]<.2)
                    hits=np.flatnonzero(failed)
                    last=int(hits[0]) if len(hits) else len(height)-1
                    holds.append(dict(profile=profile,initial_pitch_deg=r['pitch_deg'],fall_time_s=float(trace['time'][last]) if len(hits) else None,
                        duration_s=5.,joint_rmse_before_fall_rad=float(np.sqrt(np.mean((trace['q'][:last+1,i]-initial[i])**2))),
                        max_tilt_before_fall_deg=float(np.max(np.arccos(np.clip(rzz[:last+1,i],-1,1)))*180/np.pi),
                        final_root_height_m=float(height[last,i]),
                        saturation_fraction_before_fall=float(np.mean(abs(trace['torque'][1:last+1,i])/effort>=.98)),
                        feet_with_measured_ground_force=int(np.sum(np.max(np.linalg.norm(trace['foot_ground_force'][:last+1,i],axis=-1),axis=0)>1.)),
                        peak_foot_ground_vertical_force_N=float(trace['foot_ground_force'][:last+1,i,:,2].max()),
                        nonfoot_ground_contact_samples_before_fall=int(np.sum(trace['other_ground_contact_count'][:last+1,i]>0))))
        metadata.update(results=results,standing=holds,status='MEASURED',production_gains_changed=False)
    np.savez_compressed(args.out/'trajectories.npz',**traces)
    (args.out/'summary.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='MEASURED',mode=args.mode,step_or_sine_rows=len(results),standing=holds),indent=2),flush=True)


if __name__=='__main__':main()
