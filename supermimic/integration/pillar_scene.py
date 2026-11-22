"""Authoritative PMT meshes in the existing Isaac Lab Newton scene.

Pillars are kinematic rigid bodies. Both contact paths cache pair eligibility in
this version. Inactive bodies use a collision-disabled flag AND a remote parking
pose, through the real rigid-body API. This guarantees absence from the bounded
experiment workspace without pretending live masks are supported by the cache.
"""
import numpy as np
import torch

from .pmt_dataset_loader import require, rotate_z
from ..terrain import TerrainFeatureProvider


class PMTTerrainProvider(TerrainFeatureProvider):
    """Time/anchor-explicit terrain queries; dynamic callers may not omit time."""
    def __init__(self, kin, sample, *, frame=None, anchor=None, **kwargs):
        super().__init__(kin, **kwargs)
        self.sample, self.frame, self.anchor = sample, frame, anchor

    def at(self, frame, anchor=None):
        return PMTTerrainProvider(self.kin,self.sample,frame=frame,anchor=anchor,
                                  patch_points=self.patch_points,patch_width=self.patch_width)

    def surface(self, xy):
        if self.frame is None and self.sample.dynamic:
            raise ValueError('Dynamic PMT terrain needs explicit reference frame; do not cache a rollout-start layout')
        frame=0 if self.frame is None else self.frame
        A,t=(None,None) if self.anchor is None else self.anchor
        p,active,_=self.sample.pillar_layout(frame,A,t)
        p=xy.new_tensor(p);sizes=xy.new_tensor(self.sample.sizes)
        delta=xy[...,None,:]-p[:,:2]
        c,s=p[:,3].cos(),p[:,3].sin()
        local=torch.stack((delta[...,0]*c+delta[...,1]*s,-delta[...,0]*s+delta[...,1]*c),-1)
        edge=(sizes[:,:2]/2-local.abs()).min(-1).values
        active=torch.as_tensor(active,device=xy.device)
        inside=(edge>=0)&active
        h=torch.where(inside,p[:,2]+sizes[:,2]/2,-torch.inf).max(-1).values
        valid=inside.any(-1)
        edge=torch.where(active,edge,-torch.inf).max(-1).values
        # A fully inactive scene still has finite storage/edge values and no support.
        edge=torch.where(torch.isfinite(edge),edge,torch.full_like(edge,-self.patch_width))
        return torch.where(valid,h,torch.zeros_like(h)),valid,edge

    def features_at(self,S,frame,anchor=None):
        return self.at(frame,anchor).features(S)

    def contract(self):
        return dict(**super().contract(),scene_id=self.sample.name,frame=self.frame,
                    mesh_sha256=self.sample.alignment['mesh_sha256'],schedule_sha256=self.sample.alignment['schedule_sha256'],
                    dynamic=self.sample.dynamic,time='explicit control frame; no implicit predicted-time caching')


def mesh_spawner(vertices, faces):
    """Single visual/collision mesh, copied exactly from the certified arrays."""
    def spawn(prim_path,cfg,translation=None,orientation=None,**kwargs):
        from pxr import UsdGeom,UsdPhysics,Gf
        from isaaclab.sim.utils import get_current_stage
        stage=get_current_stage()
        root=UsdGeom.Xform.Define(stage,prim_path)
        root.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0,0,0))))
        q=orientation or (0,0,0,1)
        root.AddOrientOp().Set(Gf.Quatf(q[3],Gf.Vec3f(*q[:3])))
        mesh=UsdGeom.Mesh.Define(stage,prim_path+'/mesh')
        mesh.CreatePointsAttr(vertices.astype(np.float32).tolist())
        mesh.CreateFaceVertexCountsAttr([3]*len(faces))
        mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateDisplayColorAttr([(0.12,.32,.48)])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
        # For a verified closed convex box, convexHull has exactly the same boundary.
        rigid=UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        rigid.CreateRigidBodyEnabledAttr(True);rigid.CreateKinematicEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(1.)
        from isaaclab.sim import RigidBodyMaterialCfg
        from isaaclab.sim.utils import bind_physics_material
        material=RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.,restitution=0.)
        material.func(prim_path+'/material',material)
        bind_physics_material(prim_path+'/mesh',prim_path+'/material')
        return root.GetPrim()
    return spawn


def scene_config(sample, *, robot=True, cube=False, initial_support_only=False):
    """Scene extension of TrackingSceneCfg; robot, actuator and environment reused."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from ..env import TrackingSceneCfg
    cfg=TrackingSceneCfg(num_envs=1,env_spacing=0.,replicate_physics=True)
    cfg.ground=None
    if not robot:cfg.robot=None
    count=2 if initial_support_only else len(sample.sizes)
    for pid in range(count):
        pos=sample.schedule['pose_xyzyaw'][0,pid]
        spawn=sim_utils.CuboidCfg(size=tuple(sample.sizes[pid]))
        spawn.func=mesh_spawner(sample.mesh['vertices_local'][pid],sample.mesh['faces'][pid])
        setattr(cfg,f'pillar_{pid}',RigidObjectCfg(prim_path=f'/World/Pillars/pillar_{pid}',spawn=spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(pos[:3]),rot=(0.,0.,float(np.sin(pos[3]/2)),float(np.cos(pos[3]/2))))))
    if cube:
        cfg.cube=RigidObjectCfg(prim_path='/World/TestCube',spawn=sim_utils.CuboidCfg(size=(.04,.04,.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.,restitution=0.)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.,0.,1.)))
    return cfg


class PillarScene:
    """Runtime adapter writing real poses, best-effort flags and inactive parking."""
    def __init__(self,sim,scene,sample,*,count=None):
        import warp as wp
        self.wp,self.sim,self.scene,self.sample=wp,sim,scene,sample
        self.backend=sim.backend_manager;self.model=self.backend.get_model();self.solver=self.backend._solver
        self.count=len(sample.sizes) if count is None else count
        self.objects=[scene[f'pillar_{i}'] for i in range(self.count)]
        self.poses=sample.schedule['pose_xyzyaw'][0,:self.count].copy()
        self.collider_poses=self.poses.copy()
        self.active=np.ones(self.count,dtype=bool)
        self.shape_ids=[];self.geom_ids=[]
        shape_body=wp.to_torch(self.model.shape_body).cpu().numpy()
        for pid in range(self.count):
            bodies=[i for i,n in enumerate(self.model.body_label) if f'/Pillars/pillar_{pid}' in n]
            require(len(bodies)==1, f'Pillar {pid} Newton body mapping failed')
            shapes=np.flatnonzero(np.isin(shape_body,bodies))
            require(len(shapes)==1, 'Pillar visual/collision must share one mesh shape')
            self.shape_ids.append(int(shapes[0]))
            # Use the backend's explicit shape mapping, never sanitized-name
            # heuristics or an assumed mesh/geom ordering.
            mapping=wp.to_torch(self.solver.mjc_geom_to_newton_shape)[0].cpu().numpy()
            geoms=np.flatnonzero(mapping==shapes[0]).tolist()
            require(len(geoms)==1, f'Pillar {pid} MJWarp collision mapping failed')
            self.geom_ids.append(geoms[0])
        self.base_flags=wp.to_torch(self.model.shape_flags).clone()
        # Explicit geom pairs bypass collision bitmasks in MuJoCo; reject them.
        for ids in (self.solver.mj_model.pair_geom1,self.solver.mj_model.pair_geom2):
            require(not np.isin(ids,self.geom_ids).any(), 'Explicit pillar contact pair bypasses runtime mask')
        for pid in range(self.count):self.apply(pid,self.poses[pid],True)
        self.assert_pose_alignment()

    def apply(self,pid,pose,active):
        import newton
        pose=np.asarray(pose,dtype=float)
        require(0<=pid<self.count and pose.shape==(4,) and np.isfinite(pose).all(), 'Invalid collider pose')
        # Public rigid-object root API notifies JOINT_PROPERTIES for fixed roots;
        # floating kinematic roots use state joint_q. Both paths reach MJWarp.
        # Current pipelines cache candidate eligibility. A supported physical
        # fallback parks the inactive rigid body outside the experiment's finite
        # workspace. Logical/interpolated PMT pose is retained separately.
        physical=pose.copy() if active else np.array([1000.+10*pid,1000.,-1000.,0.])
        q=torch.tensor([[*physical[:3],0.,0.,np.sin(physical[3]/2),np.cos(physical[3]/2)]],dtype=torch.float32,device=self.sim.device)
        self.objects[pid].write_root_link_pose_to_sim_index(root_pose=q)
        self.objects[pid].write_root_link_velocity_to_sim_index(root_velocity=torch.zeros(1,6,device=self.sim.device))
        flags=self.wp.to_torch(self.model.shape_flags)
        sid,gid=self.shape_ids[pid],self.geom_ids[pid]
        value=int(self.base_flags[sid])
        flags[sid]=value if active else value & ~int(newton.ShapeFlags.COLLIDE_SHAPES)
        self.poses[pid]=pose;self.active[pid]=active
        self.collider_poses[pid]=physical

    def assert_pose_alignment(self):
        for pid,obj in enumerate(self.objects):
            p=obj.data.root_link_pose_w.torch[0,:3].cpu().numpy()
            require(np.allclose(p,self.collider_poses[pid,:3],atol=2e-6,rtol=0), f'Actual pillar pose differs: {pid}, {p}')

    def evidence(self):
        return dict(mesh_authority='terrain_mesh.npz, identical visual/collision vertices and faces',
            approximation='convexHull of certified closed convex box (same boundary)',
            pillar_shape_ids=self.shape_ids,pillar_mj_geom_ids=self.geom_ids,
            kinematic_pillars=self.count,mass_kg=1.,mass_role='kinematic fixed collider; no gravity integration',
            friction=1.,restitution=0.,ground_plane=False,
            moving_pose_api='RigidObject.write_root_link_pose_to_sim_index',
            collision_enable_api='inactive shape flag plus supported physical off-scene parking; exact target pose restored on enable',
            parking_position_rule=[1000.,1000.,-1000.],parking_scope='bounded PMT workspace; inactive visual/logical pose is separate from physical collider pose',
            native_collision_pipeline=False,contact_pipeline_change='use_mujoco_contacts=False; collision_decimation=1; same MJWarp solver and all mass/gains/limits/gravity',robot_physics_changes=False,
            mesh_float32_error_m=float(abs(self.sample.mesh['vertices_local'].astype(np.float32)-self.sample.mesh['vertices_local']).max()))


def make_pillar_sim(device,args):
    from ..env import make_sim
    cfg=make_sim(device,args)
    cfg.physics.solver_cfg.use_mujoco_contacts=False
    cfg.physics.collision_decimation=1
    return cfg
