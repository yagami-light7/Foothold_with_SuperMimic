"""S54 Isaac Lab articulation. Physical limits are sourced; PD is sim-only (see s54.py)."""
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from .robot_contract import (ROBOT_KIND, JOINT_NAMES, TRACKED_BODIES, MJCF_PATH, USD_PATH, ROOT_BODY,
                  default_joint_pos_list, actuator_parameters, initial_root_height)

PARAMETERS = actuator_parameters()
DEFAULT_JOINT_POS = dict(zip(JOINT_NAMES, default_joint_pos_list()))


def action_scale(joint_name):
    p = PARAMETERS[joint_name]
    return 0.25 * p['effort'] / p['stiffness']


ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(USD_PATH), variants={'Physics': 'mujoco'}, activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, linear_damping=0.0, angular_damping=0.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=(ROBOT_KIND == 'roban')),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, initial_root_height()), joint_pos=DEFAULT_JOINT_POS, joint_vel={'.*': 0.0}),
    soft_joint_pos_limit_factor=0.9,
    actuators={n: ImplicitActuatorCfg(joint_names_expr=[n], effort_limit_sim=p['effort'],
                  velocity_limit_sim=p['velocity'], stiffness=p['stiffness'], damping=p['damping'],
                  armature=p['armature']) for n, p in PARAMETERS.items()},
)

# Backward-compatible public name used by historical S54 tools.
S54_CFG = ROBOT_CFG
