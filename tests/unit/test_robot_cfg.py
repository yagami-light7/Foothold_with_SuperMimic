
"""Validate the S54 configuration before creating an RL environment."""

import sys
from pathlib import Path

# 支持直接运行测试文件，无需额外设置PYTHONPATH。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.utils import replace
from isaaclab.sim import UsdFileCfg
from isaaclab_newton.physics import NewtonCfg, MJWarpSolverCfg

from configs.s54_physics_cfg import (
    S54_SIM_CFG,
    CONTROL_DECIMATION,
)
from configs.s54_robot_cfg import S54_ROBOT_CFG
from tasks.foot_placement.spec import (
    NUM_JOINTS,
    JOINT_NAMES,
)


def main():
    # Generate a copy for a cloned RL scene.
    robot_cfg = replace(
        S54_ROBOT_CFG,
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    assert isinstance(robot_cfg.spawn, UsdFileCfg)
    assert robot_cfg.init_state is not None
    assert robot_cfg.actuators

    assert len(JOINT_NAMES) == NUM_JOINTS

    print("===== S54 Robot Configuration =====")
    print("Robot prim path:", robot_cfg.prim_path)
    print("USD path:", robot_cfg.spawn.usd_path)
    print("Initial root position:", robot_cfg.init_state.pos)
    print("Actuator groups:", list(robot_cfg.actuators))
    print("Joint count specification:", NUM_JOINTS)

    print("\n===== Newton Configuration =====")
    print("Physics dt:", S54_SIM_CFG.dt)
    print("Control decimation:", CONTROL_DECIMATION)
    print("Policy dt:", S54_SIM_CFG.dt * CONTROL_DECIMATION)

    assert isinstance(S54_SIM_CFG.physics, NewtonCfg)
    solver = S54_SIM_CFG.physics.solver_cfg
    assert isinstance(solver, MJWarpSolverCfg)
    print("MJWarp nconmax:", solver.nconmax)
    print("MJWarp njmax:", solver.njmax)

    print("\nConfiguration validation PASSED")
    print("Note: dynamic stability has not been tested here.")


if __name__ == "__main__":
    main()
