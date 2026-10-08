# motrix-deploy-motrixsim

`motrix-deploy-motrixsim` provides the native MotrixSim deployment runtime, registered
as `motrixsim` in the `motrix_deploy.backends` entry-point group. It uses
`MotrixSimSceneCompiler`, the native MSD world/model/data APIs, query/write programs,
and the SDK `RenderApp`; it does not run physics through MuJoCo or a training environment.

## Run the shared example

From an installed workspace environment:

```bash
python examples/deploy_to_sim.py --backend motrixsim --headless --steps 50
python examples/deploy_to_sim.py --backend motrixsim
```

See [the shared simulation example](../examples/deploy_to_sim.py) for the complete
raw ONNX walking controller with explicit task preprocessing and control settings.
The example uses the bundled Go2 policy. For another matching policy, adapt the
policy path in the Python example; this programmatic path does not read a `.deploy` manifest. MuJoCo remains that script's default backend.
Artifact deployment reuses the installed simulation recipes rather than introducing
backend-specific task copies:

```bash
motrix-deploy task=go2-walk-flat/sim artifact=artifacts/go2-walk-flat.deploy \
  runtime.backend=motrixsim runtime.viewer=false runtime.realtime=false duration_s=1
```

The same override applies to `task=go2-walk-rough/sim`. Install this backend package
explicitly for standalone use, alongside `motrix-deploy-tasks` and the robot models.
MotrixSim is supplied by `motrix-env-motrixsim`, using the workspace's pinned SDK version.

## Explicit runtime and controller assembly

`create_simulation_runtime("motrixsim", config)` accepts a complete `SimulationRuntimeConfig`
and returns a prepared host with `runtime.robot` and its read-only `spec` available before
opening. Task, model, and command binding are application-owned components assembled
explicitly:

```python
from motrix_deploy.runtime.control import ControlSession
from motrix_deploy.runtime.factory import create_simulation_runtime
from motrix_deploy.task import create_task

runtime = create_simulation_runtime("motrixsim", config)
robot = runtime.robot
task = create_task(task_spec, robot.spec, model, steps=50)
session = ControlSession(
    robot=robot,
    controller=task,
    command_binding=command_binding,
    period_s=0.02,
    state_timeout_s=0.1,
)
runtime.bind_session(session)
with runtime:
    result = runtime.run()
```

`config` contains the complete scene, physics, state sources, and rendering settings. The
control period must be an integer multiple of `config.physics.dt` and is validated when
the session is bound. The runtime owns the compiled MSD world, model, data, optional
viewer, and physics advancement; robot ports delegate I/O only. See the
[deployment tutorial](../docs/source/en/user_guide/tutorial/advanced/motrix_deploy.md)
for the framework contracts (ControlSession, DeployTask, controller lifecycle, rollout
result semantics).

## Model, placement, and state contract

The primary `scene.objs.robot` must contain a floating-base `RobotCfg` with canonical
key-pose joints and one unit-gear actuator per joint. Position actuators are converted
to torque motors; native torque motors are supported too. Physical joint and actuator
limits determine the scene-derived `RobotSpec`; controller gains are not model defaults.
The runtime does not accept an external `RobotSpec`. Artifact deployment uses the same
runtime; the CLI validates the artifact's robot contract against `runtime.robot.spec`.

Robot `translation` and xyzw `rotation` remain attachment transforms composed once
with the asset's intrinsic base pose. Reset allocates fresh native data, restores the
compiled initial placement, applies the selected joint key pose, and clears velocity
and cached controls. MotrixSim uses xyzw quaternion ordering natively.

State sources are explicit model-local names, resolved through the scene robot's
`resolve_name()` so instance prefixes/suffixes are preserved:

```yaml
sensor_bindings:
  base_angular_velocity: gyro
  base_linear_acceleration: accelerometer
  base_linear_velocity: global_linvel
```

For Go2 these names select local `FrameAngVel`, local `FrameLinAcc`, and world
`FrameLinVel` sensors. IMU sites must belong to the base body and align with its body
frame; the velocity source must target the base link or an aligned base-body site.
Missing or incompatible sources fail rather than fabricating state. Public `RobotState`
does not expose world base position. Optional linear velocity comes only from its bound
sensor and remains `None` when unbound; tasks check optional fields at their usage sites. Public state reads own snapshots of native query buffers.

## Control timing and viewer

ControlSession calls `Controller` with validated state and context and writes its command;
runtime advances one control interval using one scheduler.
Completion consumes no extra interval; task-specific ready and normal exit actions
are decisions of the concrete task, not runtime phases. Every physics
substep recomputes servo torque from current joint state:

```text
tau = kp * (q_target - q) + kd * (dq_target - dq) + tau_ff
```

The result is clipped to `RobotSpec.torque_limit`. Direct torque commands bypass PD
but retain limit checks. Stop clears cached commands and writes zero torque; port
closure does not close the runtime-owned world.

State sample timestamps use simulation time, while receive timestamps and realtime
pacing use the monotonic clock. Headless runs do not pace unless requested. Rendering
uses SDK `RenderApp`, synchronizes native data, and exposes a viewer-owned keyboard
device for the deployment CLI. Closing the viewer interrupts control; leaving the
runtime context closes the viewer and releases model/data. Native interactive rendering
requires a working graphical environment; headless execution can validate control
without opening a window. Keyboard bindings can be assembled before opening the
runtime: `runtime.get_keyboard_device()` returns the viewer's stable keyboard device,
which binds to SDK input when the viewer opens and detaches when it closes.

With MotrixSim SDK `0.10.1`, closing and recreating an interactive viewer in the same
process is unsupported: the native window event loop raises `RecreationAttempt`.
Use a new process for another interactive viewer. Headless runtime close/reopen and
repeated resets are supported and covered by native tests.
