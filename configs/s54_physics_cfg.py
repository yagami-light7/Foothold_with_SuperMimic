"""Newton MJWarp physics configuration for KUAVO S54."""

import isaaclab.sim as sim_utils

from isaaclab_newton.physics import(
    NewtonCfg,
    MJWarpSolverCfg,
)

# Physics runs at 500 Hz.
PHYSICS_DT = 1.0 / 500.0


# Policy runs at 50 Hz.
# 500 / 10 = 50 Hz.
CONTROL_DECIMATION = 10


S54_PHYSICS_CFG = NewtonCfg(
    solver_cfg=MJWarpSolverCfg(
        nconmax=256,
        njmax=512,
    ),
)


S54_SIM_CFG = sim_utils.SimulationCfg(
    dt=PHYSICS_DT,
    gravity=(0.0, 0.0, -9.81),
    device="cuda:0",
    render_interval=CONTROL_DECIMATION,
    physics=S54_PHYSICS_CFG,
)
