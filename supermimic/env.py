"""Motion-tracking simulator on Isaac Lab 3 (kit-less, Newton physics with the MuJoCo-Warp solver).

The environment is a thin batched wrapper: it applies the joint setpoint, steps MuJoCo-Warp, labels
the transition with the analytic reward/termination of :mod:`supermimic.mdp`, and resets episodes to
reference-motion states drawn by return-based bin sampling (paper Appendix D.1).
"""

from __future__ import annotations

import math
import json
import hashlib

import torch
import warp as wp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from newton import ModelFlags

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass, instantiate, replace

from .mdp import MAX_STEP_REWARD, MDP, flatten_state, unflatten_state
from .motion import CONTROL_DT, MotionLib
from .robot import ROBOT_CFG, action_scale
from .robot_contract import (ROBOT_KIND, REPO, JOINT_NAMES, NUM_J, XI_DIM, JOINT_OFFSET, COM_OFFSET, COM_BODY, FOOT_BODY_NAMES,
                  TRACKED_BODIES, ROOT_BODY, name_mapping, USD_PATH, MJCF_PATH, checkpoint_contract)
from .rotations import mat_to_quat, quat_to_mat, so3_exp
from .goals import GoalSource
from .transfer import SIMULATION


@configclass
class TrackingSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=(
        sim_utils.GroundPlaneCfg(usd_path=str(REPO/'configs/roban/flat_ground.usda'), color=None)
        if ROBOT_KIND == 'roban' else sim_utils.GroundPlaneCfg()))
    robot = replace(ROBOT_CFG, prim_path="{ENV_REGEX_NS}/Robot")


def make_sim(device: str, args) -> SimulationContext:
    physics = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(**{k:v for k,v in SIMULATION.items() if k!='num_substeps'}),
        num_substeps=SIMULATION['num_substeps'],
    )
    return sim_utils.SimulationCfg(dt=CONTROL_DT, device=device, physics=physics,gravity=(0.,0.,-9.81))


class TrackingSim:
    """Batched S54 motion-tracking simulator.

    Must be constructed inside ``launch_simulation(sim_cfg, ...)`` with ``sim_cfg`` from :func:`make_sim`.
    """

    def __init__(self, sim_cfg, num_envs: int, motion: MotionLib, mdp: MDP, gamma: float = 0.99, eval_mode: bool = False,
                 max_episode_steps: int | None = None, reset_noise: bool = True, goals: GoalSource | None = None,
                 scene_cfg=None, scene_setup=None):
        if not USD_PATH.is_file():
            raise FileNotFoundError(f"Generate S54 USD first: python tools/mjcf_to_usd.py ({USD_PATH})")
        manifest = json.loads((USD_PATH.parents[1]/"contract.json").read_text())
        if any(manifest.get(k) != v for k,v in checkpoint_contract().items()):
            raise ValueError("USD/source robot contract mismatch")
        if manifest.get("sim_mjcf_sha256") != hashlib.sha256(MJCF_PATH.read_bytes()).hexdigest():
            raise ValueError("Simulation MJCF changed; regenerate USD")
        self.device = sim_cfg.device
        self.sim = SimulationContext(sim_cfg)
        if scene_setup is not None:
            scene_setup()
        self.scene = instantiate(scene_cfg if scene_cfg is not None else
            TrackingSceneCfg(num_envs=num_envs, env_spacing=0.0, replicate_physics=True))
        self.sim.reset()
        # Isaac Lab 3 renamed this public manager; support the original image too.
        self.physics = self.sim.physics_manager if hasattr(self.sim, 'physics_manager') else self.sim.backend_manager
        self.robot = self.scene["robot"]
        canonical_to_sim, sim_to_canonical = name_mapping(self.robot.joint_names)
        self.canonical_to_sim = torch.tensor(canonical_to_sim, device=self.device)
        self.sim_to_canonical = torch.tensor(sim_to_canonical, device=self.device)
        assert all(n in self.robot.body_names for n in (TRACKED_BODIES if ROBOT_KIND == 's54' else [ROOT_BODY, COM_BODY, *FOOT_BODY_NAMES]))
        print("Robot mapping (canonical reference → simulator):", dict(zip(JOINT_NAMES, canonical_to_sim)), flush=True)
        self.N = num_envs
        self.motion, self.mdp, self.gamma = motion, mdp, gamma
        self.goals = goals if goals is not None else GoalSource(motion, mdp.kin)
        if mdp.is_foot and not self.goals.is_foot:
            raise ValueError('FootTrackingMDP requires a foot GoalSource')
        self.command = None
        self.command_id = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        self._next_command_id = 0
        self.eval_mode = eval_mode
        episode_steps = max_episode_steps if max_episode_steps is not None else (mdp.cfg.episode_steps if mdp.is_foot else 500)
        self.max_episode_steps = episode_steps if (mdp.is_foot or not eval_mode) else 10**9
        self.reset_noise = reset_noise and not eval_mode
        self.randomize = not eval_mode and (not mdp.is_foot or mdp.cfg.domain_randomization)
        dev = self.device
        self.default_q = self.robot.data.default_joint_pos.torch[0, self.canonical_to_sim].clone()
        self.alpha = torch.tensor([action_scale(n) for n in JOINT_NAMES], device=dev)
        self.torso_id = self.robot.find_bodies(COM_BODY)[0][0]

        # ---- sim context xi = [friction, joint offsets(NUM_J), torso com offset(3)] ----
        self.xi = torch.zeros(self.N, XI_DIM, device=dev)
        self._setup_friction()
        if not self.randomize:
            self.xi[:, 0] = 1.0
        else:
            self.xi[:, 0] = torch.empty(self.N, device=dev).uniform_(0.3, 1.2)
            com = torch.stack(
                (
                    torch.empty(self.N, device=dev).uniform_(-0.025, 0.025),
                    torch.empty(self.N, device=dev).uniform_(-0.05, 0.05),
                    torch.empty(self.N, device=dev).uniform_(-0.05, 0.05),
                ),
                -1,
            )
            self.xi[:, COM_OFFSET] = com
            default_com = self.robot.data.body_com_pose_b.torch[:, self.torso_id, :3].clone()
            self.robot.set_coms_index(coms=(default_com + com).unsqueeze(1), body_ids=[self.torso_id])
        self._apply_friction()

        # ---- episode bookkeeping ----
        self.f = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_step = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.a_prev = torch.zeros(self.N, NUM_J, device=dev)
        self.start_bin = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_disc_ret = torch.zeros(self.N, device=dev)
        self.ep_ret = torch.zeros(self.N, device=dev)
        self.G_max = mdp.max_step_reward / (1 - gamma)
        num_bins = motion.num_bins if motion is not None else 1
        self.bin_G = torch.full((num_bins,), 0.0, device=dev)
        self.bin_w_avg = torch.full((num_bins,), 1.0 / num_bins, device=dev)
        self.done_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.failed_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.reset(torch.arange(self.N, device=dev))

    # ------------------------------------------------------------------ friction
    def _setup_friction(self):
        mgr = self.physics
        model = mgr.get_model()
        self._mu_view = wp.to_torch(self.robot._root_view.get_attribute("shape_material_mu", model)[:, 0])
        # ground shapes (body == -1): lowest friction of the range, since MuJoCo takes the max of a pair
        shape_body = wp.to_torch(model.shape_body)
        mu_all = wp.to_torch(model.shape_material_mu)
        mu_all[shape_body < 0] = 0.3 if not self.eval_mode else 1.0

    def _apply_friction(self):
        self._mu_view[:] = self.xi[:, :1]
        self.physics.add_model_change(ModelFlags.SHAPE_PROPERTIES)

    # ------------------------------------------------------------------ state
    def state(self):
        d = self.robot.data
        pose = d.root_link_pose_w.torch.clone()
        vel = d.root_link_vel_w.torch.clone()
        p = pose[:, :3] - self.scene.env_origins
        quat = pose[:, 3:7]  # Isaac Lab 3 uses xyzw
        R = quat_to_mat(torch.cat((quat[:, 3:4], quat[:, :3]), -1))
        return (p, R, vel[:, :3], vel[:, 3:6], d.joint_pos.torch[:, self.canonical_to_sim].clone(), d.joint_vel.torch[:, self.canonical_to_sim].clone())

    # ------------------------------------------------------------------ reset
    def sample_frames(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.motion
        if getattr(m, 'split', None) == 'train' and not self.eval_mode:
            from .m34 import balanced_bin_probs
            probs = balanced_bin_probs(m, self.bin_G, self.G_max,
                                       getattr(self, 'm34_iteration', 0),
                                       getattr(self, 'm34_straight_iters', 1000))
            bins = torch.multinomial(probs, n, replacement=True)
            start = torch.minimum(m.bin_start[bins], m.clip_end[m.bin_clip[bins]]-1)
            end = torch.minimum(start+50, m.clip_end[m.bin_clip[bins]])
            return start+(torch.rand(n, device=self.device)*(end-start)).long(), bins
        soft = torch.softmax(-25.0 * self.bin_G / self.G_max, 0)
        self.bin_w_avg.mul_(0.999).add_(0.001 * soft)
        probs = 0.5 / m.num_bins + 0.4 * soft + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
        bins = torch.multinomial(probs, n, replacement=True)
        start = m.bin_start[bins]
        start = torch.minimum(start, m.clip_end[m.bin_clip[bins]] - 1)
        end = torch.minimum(start + 50, m.clip_end[m.bin_clip[bins]])  # exclude last frame
        f = start + (torch.rand(n, device=self.device) * (end - start).float()).long()
        return f, bins

    def reset(self, env_ids: torch.Tensor, frames: torch.Tensor | None = None):
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        dev = self.device
        command = None
        if self.mdp.is_foot:
            if frames is not None:
                raise ValueError('Foot tasks reset from the S54 model, not Motion frame indices')
            frames = torch.zeros(n, dtype=torch.long, device=dev)
            bins = torch.zeros_like(frames)
            nominal = self.mdp.nominal_state(n)
            command = self.goals.new_command(nominal)
            p,R,v,w,q,qd = self.mdp.perturb_reset(nominal) if self.reset_noise else nominal
        elif frames is None:
            frames, bins = self.sample_frames(n)
        else:
            bins = self.motion.frame_bin[frames]
        if not self.mdp.is_foot:
            p, R, v, w, q, qd = (x.clone() for x in self.motion.ref_state(frames))
        if self.reset_noise and not self.mdp.is_foot:
            p = p + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.05, 0.05), torch.empty(n, device=dev).uniform_(-0.05, 0.05),
                 torch.empty(n, device=dev).uniform_(-0.01, 0.01)), -1)  # fmt: skip
            rv = torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.1, 0.1), torch.empty(n, device=dev).uniform_(-0.1, 0.1),
                 torch.empty(n, device=dev).uniform_(-0.2, 0.2)), -1)  # fmt: skip
            R = so3_exp(rv) @ R
            v = v + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.5, 0.5), torch.empty(n, device=dev).uniform_(-0.5, 0.5),
                 torch.empty(n, device=dev).uniform_(-0.2, 0.2)), -1)  # fmt: skip
            w = w + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.52, 0.52), torch.empty(n, device=dev).uniform_(-0.52, 0.52),
                 torch.empty(n, device=dev).uniform_(-0.78, 0.78)), -1)  # fmt: skip
            q = q + torch.empty_like(q).uniform_(-0.1, 0.1)
        q = torch.clamp(q, self.mdp.kin.q_lo, self.mdp.kin.q_hi)
        if not self.mdp.is_foot:
            p,R,v,w,q,qd=self.prepare_motion_reset((p,R,v,w,q,qd),frames)
        quat = mat_to_quat(R)
        pose = torch.cat((p + self.scene.env_origins[env_ids], quat[:, 1:4], quat[:, 0:1]), -1)
        self.robot.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim_index(root_velocity=torch.cat((v, w), -1), env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=q[:, self.sim_to_canonical], env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=qd[:, self.sim_to_canonical], env_ids=env_ids)
        self.robot.set_joint_position_target_index(target=q[:, self.sim_to_canonical], env_ids=env_ids)
        self.f[env_ids] = frames
        self.ep_step[env_ids] = 0
        self.a_prev[env_ids] = 0.0
        self.start_bin[env_ids] = bins
        self.ep_disc_ret[env_ids] = 0.0
        self.ep_ret[env_ids] = 0.0
        if self.randomize:
            self.xi[env_ids, JOINT_OFFSET] = torch.empty(n, NUM_J, device=dev).uniform_(-0.01, 0.01)
        self.scene.reset(env_ids)
        self.refresh()
        if self.goals.is_foot:
            # All state/command coordinates exclude the fixed scene.env_origins.
            if command is None:
                S = tuple(x[env_ids] for x in self.state())
                command = self.goals.new_command(S,frames=frames)
            if self.command is None:
                self.command = command
            else:
                self.command.assign(env_ids, command)
            self.command_id[env_ids] = torch.arange(n, device=dev) + self._next_command_id
            self._next_command_id += n

    def prepare_motion_reset(self,state,frames):
        """Optional terrain feasibility hook; original Flat resets are unchanged."""
        return state

    def refresh(self):
        """Propagate written states so that ``state()`` is consistent after a reset."""
        self.sim.forward()
        self.scene.update(0.0)

    # ------------------------------------------------------------------ step
    def pd_target(self, a: torch.Tensor, xi=None):
        """Canonical joint setpoint applied by the simulator (also used by diagnostics)."""
        xi = self.xi if xi is None else xi
        return self.default_q + xi[:, JOINT_OFFSET] + self.alpha * a

    def step(self, a: torch.Tensor) -> dict:
        S = self.state()
        f = self.f.clone()
        xi = self.xi.clone()
        a_prev = self.a_prev.clone()
        ep_step = self.ep_step.clone()
        command = self.command.clone() if self.goals.is_foot else None
        command_id = self.command_id.clone() if self.goals.is_foot else None
        target = self.pd_target(a, xi)
        self.robot.set_joint_position_target_index(target=target[:, self.sim_to_canonical])
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(CONTROL_DT)
        S1 = self.state()
        saved_s1=flatten_state(S1)
        reward_state=(unflatten_state(saved_s1) if getattr(self.mdp,'canonical_replay_reward',False) else S1)
        f1 = f + 1
        r, terms = self.mdp.reward(reward_state, f1, a, a_prev, terms=True, command=command, ep_step=ep_step+1)
        term = self.mdp.termination(S1, f1, command=command, ep_step=ep_step+1)
        timeout = ep_step + 1 >= self.max_episode_steps
        if not self.mdp.is_foot:
            clip_end = self.motion.clip_end[self.motion.clip_id[f]]
            timeout = timeout | (f1 >= clip_end)
        done = term | timeout
        out = {
            "s": flatten_state(S), "a": a.clone(), "a_prev": a_prev, "f": f, "s1": saved_s1, "r": r,
            "term": term, "timeout": timeout, "done": done, "ep_step": ep_step, "xi": xi, "terms": terms,
        }  # fmt: skip
        if self.goals.is_foot:
            out.update(command=command, command_id=command_id)
        # bookkeeping
        self.ep_disc_ret += (self.gamma ** ep_step.float()) * r
        self.ep_ret += r
        self.f = f1
        self.ep_step = ep_step + 1
        self.a_prev = a.clone()
        if self.eval_mode:
            self.failed_eval |= term & ~self.done_eval
            self.done_eval |= done
            return out
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            # bin return estimate; truncated episodes bootstrap with their mean reward
            T = self.ep_step[ids].float()
            boot = torch.where(term[ids], 0.0, (self.gamma**T) * (self.ep_ret[ids] / T) / (1 - self.gamma))
            G = self.ep_disc_ret[ids] + boot
            if not self.mdp.is_foot:
                b = self.start_bin[ids]
                self.bin_G.index_put_((b,), 0.8 * self.bin_G[b] + 0.2 * G)
            out["ep_len"] = T
            out["ep_ret_norm"] = self.ep_ret[ids] / (T * self.mdp.max_step_reward)
            self.reset(ids)
        return out
