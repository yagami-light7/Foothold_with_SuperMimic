"""Opt-in terrain scene/reward; original flat defaults and network sizes remain valid."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from .mdp import MDP
from .s54 import REPO


def flat_zone_mesh():
    """Exact former finite Box boundary, baked into world-coordinate triangles."""
    import numpy as np
    import trimesh
    transform=np.eye(4);transform[:3,3]=[20.,0.,-.05]
    return trimesh.creation.box(extents=(24.,24.,.1),transform=transform)


class StairTrackingMDP(MDP):
    # Use the exact 71D quaternion state stored in replay for real-step rewards.
    # This avoids rewarding one rotation-matrix rounding and replaying another.
    canonical_replay_reward=True
    def __init__(self, *args, stance_slip_weight=.05, stance_slip_std=.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.stance_slip_weight, self.stance_slip_std = stance_slip_weight, stance_slip_std

    def contract(self):
        return dict(**super().contract(), terrain_task='stair_motion_tracking_v1',
            body_anchor='fixed_reference_world', stance_slip_weight=self.stance_slip_weight,
            stance_slip_std=self.stance_slip_std,
            stance_source='reference_contact_schedule; differentiable foot link XY speed')

    def reference_bodies(self, S, f):
        return self.motion.body_p[f], self.motion.body_R[f]

    def reward(self, S, f, a, a_prev, terms=False, **kwargs):
        reward, values = super().reward(S, f, a, a_prev, terms=True, **kwargs)
        feet_v = self.kin.forward(*S)[2][..., self.foot_indices, :2]
        if hasattr(self.motion, 'contacts'):
            stance = self.motion.contacts[f].to(feet_v.dtype)
        else:
            stance = (self.motion.body_v[f][..., self.foot_indices, :2].norm(dim=-1) < .05).to(feet_v.dtype)
        slip = (feet_v.square().sum(-1)*stance).sum(-1)/stance.sum(-1).clamp_min(1)
        penalty = -.02*self.stance_slip_weight*slip/self.stance_slip_std**2
        values.update(stance_slip_penalty=penalty, stance_slip_speed_squared=slip)
        reward = reward+penalty
        return (reward, values) if terms else reward


def make_stair_sim(device, args, *, contact_pipeline='newton', render=False):
    from .env import make_sim
    cfg = make_sim(device, args)
    # Explicit, recorded transfer difference: keep MJWarp solver, implicit PD,
    # friction, substeps, gravity and limits; use triangle-capable contacts.
    cfg.physics.solver_cfg.use_mujoco_contacts = contact_pipeline == 'mujoco'
    # External Newton contacts otherwise refresh only once per 20 ms control
    # tick and remain stale through all four solver substeps. MuJoCo's internal
    # pipeline refreshes each solver step; preserve that timing for mesh contact.
    if contact_pipeline=='newton':cfg.physics.collision_decimation=1
    cfg.physics.load_visual_shapes = render
    return cfg


def make_stair_env(cfg, num_envs, motion, mdp, terrain_asset, *, goals=None, eval_mode=True,
                   reset_noise=False, gamma=.99, mixed_flat=False, environment_class=None, flat_collider='box', **kwargs):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.utils import configclass
    from .env import TrackingSim, TrackingSceneCfg

    @configclass
    class StairSceneCfg(TrackingSceneCfg):
        ground = None
        stairs = AssetBaseCfg(prim_path='/World/Stairs', spawn=None, collision_group=-1)

    scene_cfg = StairSceneCfg(num_envs=num_envs, env_spacing=0., replicate_physics=True)
    if mixed_flat:
        # Fixed paired transform for flat training clips only. The stair mesh and
        # stair reference stay in the authoritative unmodified source world.
        scene_cfg.flat = AssetBaseCfg(prim_path='/World/FlatZone', collision_group=-1,
            spawn=sim_utils.CuboidCfg(size=(24.,24.,.1),
                collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.08,.08,.08))),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(20.,0.,-.05)))
        if flat_collider=='mesh':scene_cfg.flat=AssetBaseCfg(prim_path='/World/FlatZone',spawn=None,collision_group=-1)

    def setup():
        path = REPO/'scripts/terrain/load_stairs_asset.py'
        spec = importlib.util.spec_from_file_location('s54_stair_loader_local', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        module.spawn_stairs(terrain_asset, '/World/Stairs')
        if mixed_flat and flat_collider=='mesh':
            from isaaclab.terrains.utils import create_prim_from_mesh
            create_prim_from_mesh('/World/FlatZone',flat_zone_mesh(),
                collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.08,.08,.08)))

    return (environment_class or TrackingSim)(cfg, num_envs, motion, mdp, gamma=gamma, eval_mode=eval_mode,
        reset_noise=reset_noise, goals=goals, scene_cfg=scene_cfg, scene_setup=setup, **kwargs)
