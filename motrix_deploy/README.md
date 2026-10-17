# Motrix Deploy

`motrix-deploy` runs exported policies without a training framework. It provides deployment artifacts,
robot/task contracts, shared control execution, runtime plugin discovery, and the `motrix-deploy` CLI.
Install the task and runtime plugins needed for your target. `motrix-deploy` does not depend on `motrix-rl`:
training-run export belongs to `motrix-rl`, which depends on this package, while deployment runs independently.
Create artifacts with `mxexport deploy run_dir=<run-dir> output=artifacts/policy.deploy`; see the
[deployment tutorial](../docs/source/en/user_guide/tutorial/advanced/motrix_deploy.md) for export and profile compiler plugins.

With the Go2 task and a simulation backend installed:

```bash
motrix-deploy task=go2-walk-flat/sim artifact=artifacts/go2-walk-flat.deploy
```

## Hardware API

Hardware backends return a fully constructed `HardwareRuntime`, independent of the robot model or transport:

```python
from motrix_deploy.runtime.hardware import HardwareRuntime

runtime = HardwareRuntime(robot)  # An already constructed RobotInterface.
```

Construction does not open the robot connection. Bind a `ControlSession` before running;
robot startup, enable, safe stop, and transport cleanup remain owned by the interface.
The runtime uses the robot's optional gamepad device unless one is explicitly supplied.

For CLI recipes, hardware backend entry points in `motrix_deploy.backends` implement
`factory(config, context) -> HardwareRuntime`. The context supplies the artifact's robot
and control specifications. `create_hardware_runtime` validates and returns the plugin-created
runtime; hardware plugins own runtime construction and may configure its scheduler or gamepad.
Simulation entry points continue to return a `SimulationRuntime`.

See the [deployment tutorial](../docs/source/en/user_guide/tutorial/advanced/motrix_deploy.md)
for export, simulation, and hardware workflows, or the [Python example](../examples/deploy_to_sim.py)
for programmatic control.
