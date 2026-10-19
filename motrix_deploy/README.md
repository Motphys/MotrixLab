# Motrix Deploy

`motrix-deploy` runs exported policies without a training framework. It provides deployment artifacts,
robot/task contracts, shared control execution, runtime plugin discovery, and the implementation behind
`mxlab deploy`. Install `motrixlab` for the sole `mxlab` CLI; this component retains its Python APIs and
internal callable deployment adapter. Use `mxlab --version` to query the version.
Install the task and runtime plugins needed for your target. `motrix-deploy` does not depend on `motrix-rl`:
training-run export belongs to `motrix-rl`, which depends on this package, while deployment runs independently.
Create artifacts with `mxlab export-deploy run_dir=<run-dir> output=artifacts/policy.deploy`; see the
[deployment tutorial](../docs/source/en/user_guide/tutorial/advanced/motrix_deploy.md) for export.
Environment packages provide profile compilers through package registration callbacks in `motrix_deploy.profiles`;
see [adding a profile compiler](../docs/source/en/user_guide/tutorial/advanced/motrix_deploy.md#add-a-deployment-profile-compiler)
for the provider recipe and discovery requirements.

With the Go2 task and a simulation backend installed:

```bash
mxlab deploy task=go2-walk-flat/sim artifact=artifacts/go2-walk-flat.deploy
```

## Hardware API

Hardware backends return a fully constructed `HardwareRuntime`, independent of the robot model or transport:

```python
from motrix_deploy.runtime.hardware import HardwareRuntime

runtime = HardwareRuntime(robot)  # An already constructed RobotInterface.
```

Construction does not open the robot connection. Bind a `ControlSession` before running;
transport acquisition, output enable, emergency/fault safe stop, and cleanup remain owned by the interface.
Tasks own normal preparation trajectories, servo gains, operator startup gates, and normal shutdown motions.
CLI `task_settings` are passed through `create_task(..., settings=...)` to the task constructor before runtime
construction; each task validates its supported settings independently of the frozen artifact policy contract.
The runtime uses the robot's optional gamepad device unless one is explicitly supplied.
`RobotInterface.read_state()` returns the current/latest snapshot without waiting for a new sample.
Hardware `open()` waits for the first valid state with a bounded timeout; repeated reads preserve the
sample's receive timestamp so `ControlSession` can enforce the configured state-age limit.

For CLI recipes, hardware backend entry points in `motrix_deploy.backends` implement
`factory(config, context) -> HardwareRuntime`. The context supplies the artifact's robot
and control specifications. `create_hardware_runtime` validates and returns the plugin-created
runtime; hardware plugins own runtime construction and may configure its scheduler or gamepad.
Simulation entry points continue to return a `SimulationRuntime`.

Simulation initializes placement, joint pose and velocities through its scene and reset before entering
shared control. The shared `ControlSession` executes one task controller from the measured robot state,
without a fixed-base phase or preparation handoff. Task-owned startup trajectories and operator gates run
as normal controller phases under the same state freshness, command and health checks on simulation and
hardware. Tasks do not receive runtime handles; simulation does not numerically support the base during
those phases.

See the [deployment tutorial](../docs/source/en/user_guide/tutorial/advanced/motrix_deploy.md)
for export, simulation, and hardware workflows, or the [Python example](../examples/deploy.py)
for programmatic control.
