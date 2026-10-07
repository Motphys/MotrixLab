# motrix-deploy-mujoco

`motrix-deploy-mujoco` runs deployment policies in MuJoCo. It compiles a `SceneCfg`, exposes the scene's robot through
`RobotInterface`, and supplies physics stepping, joint control, and an optional keyboard viewer.

With `motrix_deploy_tasks` installed and a matching exported artifact:

```bash
motrix-deploy task=go2-walk-flat/sim artifact=artifacts/go2-walk-flat.deploy runtime.viewer=false duration_s=2.0
```

Omit the headless overrides for interactive control. The Go2 flat and rough recipes are installed with the task plugin,
not read from workspace config paths. Task defaults select the `sim` Hydra runtime group. Simulation settings are
`runtime.viewer` and `runtime.realtime`; `runtime.realtime=null` follows `runtime.viewer`. MuJoCo options live directly
under `runtime`, including `runtime.backend=mujoco`, `runtime.deploy_env_id`, `runtime.robot_id`, and `runtime.physics`.
`runtime.backend` is a scalar string selector, not a nested configuration mapping or Hydra group.
`duration_s` budgets `ceil(duration_s / control_period_s)` control ticks, not wall-clock time.

Programmatic callers use `create_simulation_runtime("mujoco", SimulationRuntimeConfig(...))`, assemble a `ControlSession`
with `runtime.robot`, and call `runtime.bind_control_session(control)` before running. The runtime advances physics between
control ticks; the control session handles task observations, policy inference, and robot commands. See
[`examples/deploy_to_sim.py`](../examples/deploy_to_sim.py) for a complete headless or viewer example.
