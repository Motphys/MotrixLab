# Motrix Deploy Tasks

`motrix_deploy_tasks` supplies the Go2 walking task, flat/rough deployment scenes, and installed Hydra recipes for
`motrix-deploy`. Install it alongside the MuJoCo runtime plugin to run an exported policy:

```bash
motrix-deploy task=go2-walk-flat/sim artifact=artifacts/go2-walk-flat.deploy runtime.viewer=false duration_s=2.0
```

The recipes are packaged with the plugin; the command works outside the repository without `--config-path`.
`task=go2-walk-rough/sim` selects the rough scene, while `task=go2-walk-flat/hardware` selects the Unitree runtime.
Task defaults choose the `sim` or `hardware` Hydra runtime group. Target settings live directly under `runtime`;
`runtime.backend` is `mujoco` for simulation or `unitree_go2` for hardware. It is a scalar string selector, not a nested
configuration mapping or Hydra group.
Use an artifact exported for the matching task. Common settings are `artifact`, `duration_s`, and `command`;
`duration_s` budgets `ceil(duration_s / control_period_s)` control ticks, not wall-clock time.
Simulation exposes `runtime.viewer` and `runtime.realtime`; `runtime.realtime=null` follows `runtime.viewer`.
Hardware is always real-time internally and has neither field. For hardware networking, override
`runtime.network_interface=<interface>`.
Follow the [deployment tutorial](../docs/source/en/user_guide/tutorial/advanced/motrix_deploy.md) for physical operator
safety, startup/enable gates, and emergency stop procedures before commanding a robot.

The artifact's `go2_walk/v1` specification defines observation preprocessing, action conversion, gains, command bounds,
and termination settings. `Go2WalkTaskSpec` and `Go2WalkDeployTaskV1` are defined in
`motrix_deploy_tasks.tasks.go2_walk`. Training profile compilers live in `motrix_envs.deploy`.

For a scene/task/policy assembled directly in Python, see
[`examples/deploy_to_sim.py`](../examples/deploy_to_sim.py).
