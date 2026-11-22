"""PMT playback and bounded PD evaluation using TrackingSim, never a policy.

The real PD path writes state only at reset and uses the existing position AND
velocity actuator targets, limits and gains. Kinematic playback is explicitly a
different mode that writes each reference pose and makes no control claim.
"""
import json
import numpy as np
import torch

from .pmt_dataset_loader import PMTMotionReferenceProvider,require,rotate_z
from .pillar_scene import scene_config,PillarScene,PMTTerrainProvider,make_pillar_sim
from .pillar_event_controller import PillarEventController,PillarEventBlocked
from .contact_diagnostics import SolverContacts,RobotGeometry,support_margin,contact_patch_margin
from ..roban import JOINT_NAMES,FOOT_BODY_NAMES,foot_geometry,default_joint_pos_list,actuator_parameters
from ..rotations import mat_to_quat


def finite_json(value):
    if isinstance(value,dict):return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [finite_json(v) for v in value]
    if isinstance(value,np.ndarray):return finite_json(value.tolist())
    if isinstance(value,(np.floating,float)):return float(value) if np.isfinite(value) else None
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(np.bool_,)):return bool(value)
    return value


def run(sample,args,out):
    from isaaclab.app import launch_simulation
    from ..env import TrackingSim,make_sim
    from ..kinematics import RobotKinematics
    from ..roban_task import RobanFootGoals,RobanFootMDP,FootRewardConfig
    from ..goals import GoalSpec
    from ..transfer import physical_contract
    dev='cuda:0';kin=RobotKinematics(dev)
    motion=PMTMotionReferenceProvider([sample],kin,dev,trust_offline_body_prior=args.trust_offline_body_prior)
    goals=RobanFootGoals(motion,kin,GoalSpec('roban_foot',10,.1))
    terrain_provider=PMTTerrainProvider(kin,sample,frame=0)
    reward_cfg=FootRewardConfig(body_prior_weight=.03)
    mdp=RobanFootMDP(kin,motion,torch.tensor(default_joint_pos_list(),device=dev),goals,reward_cfg,
                      critic_use_body_reference=True,terrain=terrain_provider)
    cfg=make_pillar_sim(dev,args);playback=args.mode=='playback'
    support_only=args.experiment=='double_support' and not playback
    require(not (support_only and sample.dynamic),'Double-support diagnostic uses the static sample')
    geometry=RobotGeometry(sample)
    records={};contacts_saved=[];eventlog=[];max_fk=0.;writes=0
    params=actuator_parameters();effort=np.array([params[n]['effort'] for n in JOINT_NAMES])
    hold=args.experiment!='four_step' and not playback
    nsteps=round(args.hold_seconds/.02) if hold else sample.frames-1
    require(not hold or 0<nsteps<sample.frames,'Hold duration must fit the sample clock; default is 5 seconds')
    reason='completed';failure_tick=None;failure_reference=None
    with launch_simulation(cfg,args):
        env=TrackingSim(cfg,1,motion,mdp,eval_mode=True,reset_noise=False,goals=goals,
                          scene_cfg=scene_config(sample,initial_support_only=support_only))
        env.reset(torch.tensor([0],device=dev),torch.tensor([0],device=dev))
        if hold:
            # A holding test starts at rest. This remains part of the sole reset.
            env.robot.write_root_link_velocity_to_sim_index(root_velocity=torch.zeros(1,6,device=dev))
            env.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros(1,21,device=dev))
            env.refresh()
        terrain=PillarScene(env.sim,env.scene,sample,count=2 if support_only else None)
        actual_params={}
        for field,attribute in [('stiffness','joint_stiffness'),('damping','joint_damping'),('armature','joint_armature'),('effort','joint_effort_limits')]:
            values=getattr(env.robot.data,attribute).torch[0,env.canonical_to_sim].cpu().numpy()
            np.testing.assert_allclose(values,[params[n][field] for n in JOINT_NAMES],rtol=1e-6,atol=1e-7,err_msg=f'Runtime actuator {field} differs')
            actual_params[field]=values.tolist()
        contacts=SolverContacts(env.physics,sample)
        solver=env.physics._solver
        dofs=[]
        for name in JOINT_NAMES:
            matches=[j for j in range(solver.mj_model.njnt) if solver.mj_model.joint(j).name.endswith('_'+name)]
            require(len(matches)==1,'Actual torque joint mapping differs from canonical robot')
            dofs.append(int(solver.mj_model.jnt_dofadr[matches[0]]))
        import warp as wp
        if not playback:
            # The evaluator cannot silently animate or reset a falling robot.
            def forbid(*a,**kw):raise RuntimeError('PD rollout attempted non-reset root/joint state write')
            for name in ('write_root_link_pose_to_sim_index','write_root_link_velocity_to_sim_index',
                         'write_joint_position_to_sim_index','write_joint_velocity_to_sim_index'):
                setattr(env.robot,name,forbid)
        controller=PillarEventController(sample,terrain.apply,safe_frames=args.safe_frames) if sample.dynamic else None

        def record(tick,ref_frame,phase):
            nonlocal max_fk
            S=env.state();p,R,v,w,q,qd=[x[0].cpu().numpy() for x in S]
            quat=mat_to_quat(S[1])[0].cpu().numpy()
            geometry.set_robot(p,quat,q,np.r_[v,w],qd)
            # Missing future pillars in PD-0 are far away in geometry queries.
            poses=sample.schedule['pose_xyzyaw'][0].copy()
            active=np.zeros(len(poses),dtype=bool)
            poses[:terrain.count]=terrain.poses;active[:terrain.count]=terrain.active
            geometry.set_pillars(poses,active)
            fp,fR,fv,corners=geometry.feet()
            snap=contacts.read()
            ids=[env.robot.body_names.index(n) for n in FOOT_BODY_NAMES]
            raw=env.robot.data.body_link_pose_w.torch[:,ids]
            from ..rotations import quat_to_mat
            real_R=quat_to_mat(raw[...,3:7][...,[3,0,1,2]])
            sole=raw[...,:3]+(real_R@S[0].new_tensor([f['sole'] for f in foot_geometry()])[...,None]).squeeze(-1)
            max_fk=max(max_fk,float(np.linalg.norm(sole[0].cpu().numpy()-fp,axis=-1).max()))
            margin,polygon=support_margin(geometry.com,corners,snap.foot_force)
            contact_margin,contact_polygon=contact_patch_margin(geometry.com,snap)
            require(len(contact_polygon)<=32,'Contact hull exceeds diagnostic capacity')
            padded_polygon=np.full((32,2),np.nan);padded_polygon[:len(contact_polygon)]=contact_polygon
            from scipy.spatial.transform import Rotation
            fquat=Rotation.from_matrix(fR).as_quat()[:,[3,0,1,2]]
            terrain_provider.frame=ref_frame
            _,term=mdp.reward(S,torch.tensor([ref_frame],device=dev),torch.zeros(1,21,device=dev),torch.zeros(1,21,device=dev),True,
                               command=motion.anchor(torch.tensor([0],device=dev)))
            # Same quality-gated prior; record raw and weighted independently.
            measured_tau=wp.to_torch(solver.mjw_data.qfrc_actuator)[0,dofs].cpu().numpy().copy()
            signed=geometry.distances(poses,active)
            signed=np.where(active,signed,np.inf)
            support_edges=[]
            for side in range(2):
                best=-np.inf
                for pid in np.flatnonzero(active):
                    local=(corners[side,:,:2]-poses[pid,:2])@rotate_z(poses[pid,3])[:2,:2]
                    edge=np.min(sample.sizes[pid,:2]/2-np.abs(local))
                    best=max(best,float(edge))
                support_edges.append(best)
            planned_states=controller.states.copy() if controller else sample.schedule['state'][0].copy()
            actual_states=planned_states.copy()
            if not playback:
                for pid in range(len(active)):
                    actual_states[pid]=('RELOCATING' if not active[pid] else 'SUPPORTING' if pid in snap.robot_contact_pillars
                                        else 'FREE' if planned_states[pid]=='FREE' else 'RESERVED_TARGET')
            value=dict(tick=tick,time_s=tick*.02,reference_frame=ref_frame,record_phase=phase,
                root_pos=p,root_quat=quat,root_lin_vel=v,root_ang_vel=w,joint_pos=q,joint_vel=qd,
                foot_pos=fp,foot_quat=fquat,foot_vel=fv,foot_contact_force=snap.foot_force,
                foot_pillar_force=snap.foot_pillar_force,applied_torque=measured_tau,com=geometry.com,
                support_margin=margin,pillar_pose=poses,pillar_active=active,
                contact_patch_support_margin=contact_margin,contact_support_polygon_xy=padded_polygon,
                contact_support_polygon_vertex_count=len(contact_polygon),
                pillar_collider_pose=np.concatenate((terrain.collider_poses,poses[terrain.count:]),axis=0),
                pillar_state=actual_states,planned_pillar_state=planned_states,
                minimum_robot_pillar_distance=signed,sole_support_edge_margin=np.array(support_edges),
                body_prior_raw=float(term['body_prior'][0]),body_prior_weighted=float(term['body_prior_contribution'][0]),
                body_prior_quality=float(motion.quality[ref_frame]),
                planned_contact=sample.contact['contact_mask'][ref_frame])
            for k,val in value.items():records.setdefault(k,[]).append(val)
            contacts_saved.append(dict(tick=tick,reference_frame=ref_frame,contacts=snap.rows,actual_robot_contact_pillars=sorted(snap.robot_contact_pillars)))
            return snap,signed,corners

        if controller:
            S=env.state();Q=mat_to_quat(S[1])[0].cpu().numpy()
            geometry.set_robot(S[0][0].cpu().numpy(),Q,S[4][0].cpu().numpy())
            controller.advance(0,contacts.read().robot_contact_pillars if not playback else set(),geometry,
                               source='kinematic geometry only (not force truth)' if playback else 'Newton solver contact set')
        snap,_,_=record(0,0,'kinematic' if playback else 'main')
        for tick in range(1,nsteps+1):
            frame=0 if hold else min(tick,sample.frames-1)
            if playback:
                # Kinematic only: each saved frame comes from directly written
                # final-IK root/joints. No physics success is inferred.
                ref=motion.ref_state(torch.tensor([frame],device=dev));quat=mat_to_quat(ref[1])
                env.robot.write_root_link_pose_to_sim_index(root_pose=torch.cat((ref[0],quat[:,1:],quat[:,:1]),-1))
                env.robot.write_root_link_velocity_to_sim_index(root_velocity=torch.cat(ref[2:4],-1))
                env.robot.write_joint_position_to_sim_index(position=ref[4][:,env.sim_to_canonical])
                env.robot.write_joint_velocity_to_sim_index(velocity=ref[5][:,env.sim_to_canonical])
                writes+=1;env.refresh()
                S=env.state();Q=mat_to_quat(S[1])[0].cpu().numpy()
                geometry.set_robot(S[0][0].cpu().numpy(),Q,S[4][0].cpu().numpy())
            if not playback:
                target_frame=frame
                qtarget=motion.q[target_frame:target_frame+1]
                vtarget=torch.zeros_like(qtarget) if hold else motion.qd[target_frame:target_frame+1]
                # Existing implicit actuator: position/velocity target command,
                # existing kp/kd, passive damping and actuator effort clipping.
                env.robot.set_joint_velocity_target_index(target=vtarget[:,env.sim_to_canonical])
                action=(qtarget-env.default_q)/env.alpha
                env.step(action)
                # State tick k is at k*dt. Apply schedule[k] at this boundary,
                # AFTER integrating [k-1,k] with schedule[k-1]. A pre-step
                # schedule[k] update would silently move every event 20ms early.
                S=env.state();Q=mat_to_quat(S[1])[0].cpu().numpy()
                geometry.set_robot(S[0][0].cpu().numpy(),Q,S[4][0].cpu().numpy())
                snap=contacts.read()
            if controller:
                try:
                    controller.advance(frame,set() if playback else snap.robot_contact_pillars,geometry,
                        source='kinematic geometry only (not force truth)' if playback else 'Newton solver contact set')
                except PillarEventBlocked as exc:
                    reason='pillar_event_guard:'+str(exc);failure_tick=tick;failure_reference=frame
                    record(tick,frame,'event_guard_failure')
                    break
            snap,dist,corners=record(tick,frame,'kinematic' if playback else 'main')
            if not playback:
                Rzz=env.state()[1][0,2,2].item();height=records['root_pos'][-1][2]
                if height<.35 or Rzz<.2:
                    reason='fall_root_height' if height<.35 else 'fall_root_tilt'
                    failure_tick=tick;failure_reference=frame
                    # Retain 0.3s after fall with frozen command/pillars. Never
                    # reset or continue accumulating successful footsteps.
                    for after in range(1,16):
                        env.robot.set_joint_position_target_index(target=qtarget[:,env.sim_to_canonical])
                        env.scene.write_data_to_sim();env.sim.step(render=False);env.scene.update(.02)
                        record(tick+after,frame,'post_failure_frozen_target')
                    break
            if tick%50==0:print(f'{args.mode}/{args.experiment}: frame {tick}/{nsteps}',flush=True)
        if controller:eventlog=controller.log
        evidence=terrain.evidence()
        physics=physical_contract()
        physics['solver']=dict(physics['solver'],use_mujoco_contacts=False,collision_decimation=1)
    actual={k:np.array(v) for k,v in records.items()}
    actual['termination_reason']=np.array(reason)
    actual['joint_names']=np.array(JOINT_NAMES)
    frames=actual['reference_frame']
    reference=dict(time_s=actual['time_s'],reference_frame=frames,root_pos=sample.body['root_pos'][frames],
        root_quat=sample.body['root_quat'][frames],joint_pos=sample.body['joint_pos'][frames],joint_vel=sample.body['joint_vel'][frames],
        planned_sole_pos=sample.planned_sole[frames],planned_foot_quat=sample.foot['foot_quat_w'][frames],
        final_ik_ankle_pos=sample.foot['actual_fk_foot_pos_w'][frames],planned_contact=sample.contact['contact_mask'][frames],
        planned_phase=sample.foot['phase'][frames],contact_valid=sample.contact['contact_valid'][frames],
        pillar_pose=sample.schedule['pose_xyzyaw'][frames],pillar_active=sample.schedule['active'][frames])
    np.savez_compressed(out/'actual_trajectory.npz',**actual)
    np.savez_compressed(out/'reference_trajectory.npz',**reference)
    np.savez_compressed(out/'contact_diagnostics.npz',time_s=actual['time_s'],foot_force=actual['foot_contact_force'],foot_pillar_force=actual['foot_pillar_force'],
        planned_contact=actual['planned_contact'],measured_contact=actual['foot_contact_force'][...,2]>10.,planned_contact_valid=reference['contact_valid'],
        support_margin=actual['support_margin'],contact_patch_support_margin=actual['contact_patch_support_margin'],
        contact_support_polygon_xy=actual['contact_support_polygon_xy'],contact_support_polygon_vertex_count=actual['contact_support_polygon_vertex_count'],
        com=actual['com'],sole_support_edge_margin=actual['sole_support_edge_margin'])
    np.savez_compressed(out/'torque_diagnostics.npz',time_s=actual['time_s'],applied_torque=actual['applied_torque'],effort_limit=effort,
        saturated_last_substep=np.abs(actual['applied_torque'])>=.98*effort,
        source=np.array('MJWarp qfrc_actuator generalized hinge force, final substep each 20ms'))
    (out/'pillar_event_log.json').write_text(json.dumps(eventlog,indent=2)+'\n')
    (out/'solver_contact_rows.json').write_text(json.dumps(contacts_saved,indent=2)+'\n')
    main=actual['record_phase']!='post_failure_frozen_target'
    end=int(np.sum(main));rooterr=np.linalg.norm(actual['root_pos']-reference['root_pos'],axis=-1)
    from scipy.spatial.transform import Rotation
    actualR=Rotation.from_quat(actual['root_quat'][:,[1,2,3,0]])
    refR=Rotation.from_quat(reference['root_quat'][:,[1,2,3,0]])
    rootrot=(actualR.inv()*refR).magnitude()
    footerr=np.linalg.norm(actual['foot_pos']-reference['planned_sole_pos'],axis=-1)
    measured=actual['foot_contact_force'][...,2]>10.
    td=[]
    for goal in sample.footsteps['goals']:
        if goal['ordinal']<1:continue
        side=0 if goal['foot']=='left' else 1;fr=goal['frame']
        intended_pillar_support=actual['foot_pillar_force'][:end,side,goal['pillar_id'],2]>10.
        candidates=np.flatnonzero((frames[:end]>=max(0,fr-15))&(frames[:end]<=min(sample.frames-1,fr+25))&intended_pillar_support)
        ix=int(candidates[0]) if len(candidates) else None
        reached=np.flatnonzero(frames[:end]==fr)
        td.append(dict(ordinal=goal['ordinal'],foot=goal['foot'],planned_frame=fr,
            actual_contact_frame=int(frames[ix]) if ix is not None else None,
            contact_timing_error_s=(int(frames[ix])-fr)*.02 if ix is not None else None,
            touchdown_error_m=float(np.linalg.norm(actual['foot_pos'][ix,side]-np.array(goal['sole_center_world']))) if ix is not None else None,
            reference_frame_reached=bool(len(reached)),measured_success=bool(ix is not None and np.linalg.norm(actual['foot_pos'][ix,side]-np.array(goal['sole_center_world']))<.03 and abs(int(frames[ix])-fr)<=10)))
    slip=np.linalg.norm(actual['foot_vel'][...,:2],axis=-1)
    stance=reference['planned_contact']&measured
    margin=actual['support_margin'][:end];valid_margin=margin[np.isfinite(margin)]
    penetration=np.maximum(-actual['minimum_robot_pillar_distance'][:end],0)
    planned_stance=reference['planned_contact'][:end]
    edge=actual['sole_support_edge_margin'][:end]
    kine_ok=playback and reason=='completed' and rooterr.max()<1e-5 and max_fk<1e-5 and penetration.max()<=.001 and np.all(edge[planned_stance]>=-1e-4)
    performance=(not playback and reason=='completed' and
        (hold or all(row['measured_success'] for row in td)))
    report=dict(status=('PASS' if kine_ok else 'FAIL') if playback else ('PASS' if performance else 'FAIL'),
        category='kinematic playback only; no dynamic balance claim' if playback else 'real Newton joint PD; no Actor/WM/Critic',
        scene_id=sample.name,experiment=args.experiment,reference_frames=sample.frames,recorded_frames=len(frames),
        main_control_steps=end-1,requested_control_steps=nsteps,completion_fraction=(end-1)/nsteps,
        dataset_integration_pass=True,physics_control_chain_pass=max_fk<1e-4,
        locomotion_performance_pass=None if playback else performance,kinematic_playback_pass=kine_ok if playback else None,
        termination_reason=reason,fall_time_s=failure_tick*.02 if failure_tick is not None and reason.startswith('fall') else None,
        blocked_event_time_s=failure_tick*.02 if failure_tick is not None and reason.startswith('pillar_event_guard') else None,
        survival_s=(end-1)*.02,state_writes_after_reset=writes,initialization_resets=2,reset_count_during_rollout=0,
        max_sole_fk_vs_newton_error_m=max_fk,
        root_position_rmse_m=float(np.sqrt(np.mean(rooterr[:end]**2))),root_orientation_rmse_rad=float(np.sqrt(np.mean(rootrot[:end]**2))),
        joint_tracking_rmse_rad=float(np.sqrt(np.mean((actual['joint_pos'][:end]-reference['joint_pos'][:end])**2))),
        foot_tracking_rmse_m=np.sqrt(np.mean(footerr[:end]**2,axis=0)).tolist(),
        max_root_tilt_deg=float(np.rad2deg(np.arccos(np.clip(1-2*np.sum(actual['root_quat'][:end,1:3]**2,axis=-1),-1,1))).max()),
        stance_slip_mean_m_s=float(slip[:end][stance[:end]].mean()) if stance[:end].any() else None,
        stance_slip_max_m_s=float(slip[:end][stance[:end]].max()) if stance[:end].any() else None,
        torque_saturation_ratio_last_substep=float((np.abs(actual['applied_torque'][:end])>=.98*effort).mean()),
        peak_actual_torque_Nm=np.abs(actual['applied_torque'][:end]).max(axis=0).tolist(),
        minimum_com_support_margin_m=float(valid_margin.min()) if len(valid_margin) else None,
        measured_support_frames_fraction=float(len(valid_margin)/end),max_robot_pillar_penetration_m=float(penetration.max()),
        minimum_planned_stance_sole_edge_margin_m=float(edge[planned_stance].min()),
        touchdown_diagnostics=td,successful_touchdowns=sum(t['measured_success'] for t in td) if not playback and not hold else None,
        planned_contact_valid_frames=0,body_prior=dict(raw_mean=float(actual['body_prior_raw'][:end].mean()),weighted_mean=float(actual['body_prior_weighted'][:end].mean()),
            quality_valid_fraction=float((actual['body_prior_quality'][:end]>0).mean()),weight=.03,
            trust_offline_kinematic_quality=args.trust_offline_body_prior),
        terrain=evidence,physical_contract=physics,
        runtime_actuator_properties=actual_params,
        com_source='source MJCF mass distribution evaluated at measured Newton root/joints, no reference pose substitution',
        sole_support_polygon_scope='force-supported complete sole corners, optimistic proxy when support is partial',
        pillar_state_scope='kinematic mode uses planned states; PD SUPPORTING conservatively means any measured robot contact, not a force-truth claim',
        contact_support_polygon_scope='convex hull of actual solver foot/pillar contact positions with upward force >0.1N; NaN when fewer than 3 noncollinear points',
        failure_diagnosis='see joint/torque/contact/root/COM time series; PD gains unchanged; dynamic infeasibility is not established by PD failure alone')
    (out/'evaluation_metrics.json').write_text(json.dumps(finite_json(report),indent=2,allow_nan=False)+'\n')
    return finite_json(report)
