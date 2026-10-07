# Go2 Flat-Terrain Walking: From Training to Physical Deployment

Run an exported Go2 policy in MuJoCo before attempting hardware deployment. With the task and MuJoCo plugins installed,
a bounded headless check is:

```bash
motrix-deploy task=go2-walk-flat/sim artifact=artifacts/go2-walk-flat.deploy runtime.viewer=false duration_s=2.0
```

The following steps create that artifact, explain interactive control, and cover the existing Unitree hardware workflow.

## 1. Install and train

From the repository root:

```bash
sh install.sh --all
source .venv/bin/activate
python scripts/train.py task=go2-walk-flat/rslrl.ppo
```

Training results are saved under `runs/go2-walk-flat/`. Skip training if you already have a metadata-backed run.

## 2. Export and inspect the artifact

```bash
python scripts/export_deploy.py env=go2-walk-flat
motrix-deploy inspect artifact=artifacts/go2-walk-flat.deploy
```

Export selects the latest run and creates `artifacts/go2-walk-flat.deploy/`. To select a specific run, use
`run=<run-dir>` instead of `env`; `output=<new-artifact-dir>` changes the destination. Existing artifact directories are
not overwritten. Confirm that inspection reports `valid: true`.

The artifact contains the ONNX policy, robot/control settings, and task specification. Deployment also needs the installed
`motrix_deploy_tasks` plugin, which supplies the walking implementation, deployment scenes, and Hydra recipes, plus the
runtime plugin for the target. The recipes work outside the repository without a workspace config path.
Task defaults select the Hydra runtime group (`sim` or `hardware`); you do not need a separate runtime override.
Common top-level settings are `artifact`, `duration_s`, and `command`. All target-specific settings live directly under
`runtime`. `runtime.kind` selects the simulation or hardware path; `runtime.backend` selects the plugin (`mujoco`
or `motrixsim` for simulation, or `unitree_go2` for hardware). `runtime.backend` is a scalar string selector, not a nested configuration
mapping or Hydra group.

## 3. Check in MuJoCo

Use the headless command above for a bounded run, or open the viewer:

```bash
motrix-deploy task=go2-walk-flat/sim artifact=artifacts/go2-walk-flat.deploy
```

Hold `W/S` to move forward/backward, `A/D` sideways, and `Q/E` to turn. Close the window, press Esc, or press Ctrl-C to stop.
The recipe selects the flat scene and the `go2` robot. Headless mode uses constant zero velocity; for a moving check, add
`'command.velocity=[0.5,0.0,0.0]'`. Set `duration_s` for a bounded run: it allows
`ceil(duration_s / control_period_s)` control ticks, using the artifact's control period, not a wall-clock timeout.
Startup and shutdown may take additional wall time.

Simulation-only settings are `runtime.viewer` and `runtime.realtime`.
`runtime.realtime=null` follows `runtime.viewer`; use `runtime.realtime=true` to pace a headless run in real time,
or `runtime.realtime=false` to run without real-time pacing.

### Assemble a runtime in Python

The complete `examples/deploy_to_sim.py` example composes a scene, walking task, sample ONNX policy, and input binding:

```bash
python examples/deploy_to_sim.py --headless --steps 100
```

It creates a MuJoCo runtime with `create_simulation_runtime`, attaches a `ControlSession` using
`runtime.bind_control_session(control)`, and runs inside `with runtime:`. The control session handles input, observations,
inference, and robot commands; the runtime handles physics, timing, and the viewer. A raw ONNX file needs explicit task
preprocessing and action settings, whereas the artifact carries those settings for the CLI.

### Select native MotrixSim

With `motrix-deploy-motrixsim` installed, switch the backend to `motrixsim` to reuse the same artifact and scene recipe:

```bash
motrix-deploy task=go2-walk-flat/sim artifact=artifacts/go2-walk-flat.deploy \
  runtime.backend=motrixsim runtime.viewer=true
```

The Python example also supports native MotrixSim:

```bash
python examples/deploy_to_sim.py --backend motrixsim
```

## 4. Run on the real robot

Suspend the robot, put it in low-level/debug mode, connect Ethernet, and keep an independent emergency stop and operator
ready. Startup can physically stand the robot down before the remote Start gate. Inspect the artifact first, then replace
`enp5s0` with the actual interface:

```bash
motrix-deploy task=go2-walk-flat/hardware \
  artifact=artifacts/go2-walk-flat.deploy \
  runtime.network_interface=enp5s0
```

Hardware runs are always real-time internally; the hardware runtime has no `runtime.viewer` or `runtime.realtime` fields.
Press Start, wait for the default-pose transition, then
press A. Hold L1 and move the sticks to command motion. B requests lie-down; Select triggers emergency stop. Stop paths
send damping commands before closing DDS. Software tests do not replace a suspended real-robot check.

The installed hardware recipe sets `runtime.kp=50` and `runtime.kd=1`. Set both to `null` to retain artifact gains, or
supply non-negative scalars or per-joint values in canonical joint order.

For read-only diagnostics without a command publisher:

```bash
motrix-deploy-unitree read-lowstate enp5s0
```

For a bounded single-joint check, with the same hardware precautions:

```bash
motrix-deploy-unitree joint-control enp5s0 FL_thigh_joint 0.9 \
  --artifact artifacts/go2-walk-flat.deploy
```

This helper uses the artifact's robot, gain, timing, and limit settings. It waits for Start and A, moves the joint, holds,
returns to the default pose, and closes with damping.

## Advanced usage

Artifact inspection resolves the versioned task class through `motrix_deploy.tasks` and validates its Pydantic
`spec_type`, without constructing a task or importing training environments or simulator backends.
The JSON wire format remains `{name, config}`. Python uses direct spec attributes such as `spec.kp` and
`spec.action_scale`; action scale and bounds are joint vectors in canonical order. Policy action bounds apply
before scaling and adding the default pose; robot position limits apply to the resulting targets.

### Load a task spec from YAML

For a programmatic application, keep the complete Go2 task fields in `task.yaml` (the contents of
`task.config`, not the manifest envelope). OmegaConf loads and merges configuration and resolves interpolations;
Pydantic then validates the plain dictionary:

```python
from omegaconf import OmegaConf
from motrix_deploy_tasks.tasks.go2_walk import Go2WalkTaskSpec

cfg = OmegaConf.load("task.yaml")
spec = Go2WalkTaskSpec.model_validate(OmegaConf.to_container(cfg, resolve=True))
```

`Go2WalkTaskSpec` uses field constraints and a model validator for joint-vector lengths and ordered bounds.
No separate `validate()` call is needed. Use the artifact reader for a full deployment manifest.

### Startup and array validation

`RobotInterface.enable()` takes no initial command. Unitree owns the transition to
`RobotSpec.default_joint_position`, using zero target velocity/feedforward torque
and the resolved startup gains. `runtime.kp` / `runtime.kd` override artifact gains;
`null` retains the artifact gains. Startup does not call task `process_action()`
with a zero action. Initial-state safety and task-termination checks still run before
enable, and fresh-state checks run afterward; Start/A gating remains unchanged.

`DeployTask` declares neither `observation_size` nor `action_size`. The policy
input boundary validates actual observations, and the task action-conversion
boundary validates actual policy outputs. CLI assembly does not precheck task
dimensions. Manifest tensor specs remain model metadata; `DeploymentProfile`
dimensions remain export-only checks against the compiled environment.

### Hardware sensor and motor wiring

`motrix_deploy_unitree.hardware.UnitreeGo2HardwareCfg` owns asset-free hardware poses and wiring.
`HardwareSensorBinding` maps SDK IMU fields, including `wxyz` quaternion conversion to canonical `xyzw`;
`UnitreeMotorBinding` maps canonical joint names to SDK motor indices. These definitions live in the
hardware plugin's `sensor` and `actuation` modules, not in simulation robot models.
The artifact still owns the policy joint order, default pose, and limits. Hardware configuration imports
neither model assets nor a simulator or SDK eagerly, and does not fabricate base position or linear velocity.

### Select a deployment world

Select a world with `runtime.deploy_env_id` and a robot with the existing registry ID `runtime.robot_id`.
The CLI calls `motrix_deploy.env.assemble_deploy_scene(deploy_env_id, robot_id)`: it creates the robot-free world,
fills `scene.objs.robot` using `motrix_env_core.registry.make_robot_config(robot_id)`.
The core registry lazily discovers installed `motrix_env_core.robots` entry points, shared by training and deployment.
The deployment worlds `flat` and `rough` live in independent `motrix_deploy_tasks.envs` modules. They use core
floor/height-field configuration without importing a training task, robot model, or training contact setup.

To keep the same Go2 recipe and artifact but switch its world:

```bash
motrix-deploy task=go2-walk-flat/sim artifact=artifacts/go2-walk-flat.deploy \
  runtime.deploy_env_id=rough
```

A world-only override leaves robot attachment placement unchanged. Adjust `runtime.robot_translation` if the terrain
needs different clearance; the complete `task=go2-walk-rough/sim` recipe supplies its own attachment defaults.

The installed flat recipe selects the world and robot by ID:

```yaml
runtime:
  kind: simulation
  backend: mujoco
  deploy_env_id: flat
  robot_id: go2
  robot_translation: [0.0, 0.0, -0.114]
  robot_rotation: [0.0, 0.0, 0.0, 1.0]
```

`runtime.physics` and `runtime.sensor_bindings` remain separate settings. MuJoCo accepts only the completed `SceneCfg`;
it neither selects an environment/robot pair nor assembles the scene or queries a training registry.
Programmatic `SimulationRuntimeConfig.scene` receives the same complete scene with the robot declared only in `scene.objs.robot`.

Placement has one source: `RobotCfg.translation` (Vec3) and `rotation` (xyzw) are attachment transforms that compose once
with the model's intrinsic base pose, not absolute world root poses. Go2's intrinsic base height is 0.445 m; the flat
recipe's -0.114 m offset gives 0.331 m, and the rough recipe's -0.025 m offset gives 0.42 m with identity rotation.
The runtime preserves these transforms on its copied scene. `mj_resetData` restores the compiled `model.qpos0`, including
the composed root placement; initialization overrides only articulation joints with the bound `RobotSpec` key pose.

For an external world plugin, declare a zero-argument `SceneCfg` factory in the `motrix_deploy.envs` entry-point group.
`motrix_deploy.env.available_deploy_envs()` lists installed IDs without loading factories;
`create_deploy_env(id)` loads only the selected factory and requires `scene.objs.robot` to be empty.
The core assembly helper inserts the robot selected by the existing robot registry; no generic scene-factory registry
or Hydra scene instantiation is involved. The CLI consumes `deploy_env_id`, `robot_id`, and optional
`robot_translation` / `robot_rotation`, applying attachment overrides to `scene.objs.robot.translation` / `.rotation`
before passing the ready scene to the simulation plugin. These application options are not runtime backend fields.

`SimulationRuntimeConfig` owns scene, rendering, realtime pacing, and state-source settings.
Its `physics` field uses the shared `motrix_env_core.config.sim.SimCfg` for `dt`,
`solver_iterations`, `solver_tolerance`, and `gravity`, defaulting to
`SimCfg(dt=0.002, solver_iterations=100)`. Both backend compilers consume this field directly:
optional solver/gravity fields set to `None` preserve model-source values; explicit values override them.
The YAML layout remains `runtime.physics`, independent of scene and state sensors.
Set physics values before runtime construction: the runtime reads the shared `SimCfg` used for
compilation; changing `dt` after compilation is not a supported runtime retuning workflow.

Programmatic applications use the same helper and pass its result to `SimulationRuntimeConfig`:

```python
from motrix_deploy.env import assemble_deploy_scene
from motrix_deploy.runtime.config import SensorBindings, SimulationRuntimeConfig
from motrix_env_core.config.sim import SimCfg

scene = assemble_deploy_scene("flat", "go2")
scene.objs.robot.translation = (0.0, 0.0, -0.114)
scene.objs.robot.rotation = (0.0, 0.0, 0.0, 1.0)
config = SimulationRuntimeConfig(
    scene=scene,
    physics=SimCfg(dt=0.002, solver_iterations=100),
    sensor_bindings=SensorBindings(
        base_angular_velocity="gyro",
        base_linear_acceleration="accelerometer",
        base_linear_velocity="global_linvel",
    ),
)
```

`SimulationRuntimeConfig.sensor_bindings` uses the frozen `SensorBindings` dataclass,
not a Python dictionary. It supports only `base_angular_velocity`,
`base_linear_acceleration`, and `base_linear_velocity`, each `str | None` with a
`None` default. Names must be nonempty and contain a nonwhitespace character;
unknown roles and non-string values are rejected. YAML `runtime.sensor_bindings`
keeps the same mapping keys and is converted with `SensorBindings(**mapping)`
at the configuration boundary. An absent field or explicit `None` means no binding;
MuJoCo rejects either for its three required roles when opening the robot.
The backend still checks sensor existence, type, dimension, and frame; it does not
synthesize state for other roles or bind every `RobotState` field to a sensor.

### Assemble the runtime and controller explicitly

The runtime factory selects the backend and prepares `runtime.robot`; its bound
`RobotSpec` is available before `open()`. The application creates the task, policy,
and command binding, then constructs and binds the control session. Using the
`config` above and your application-owned `task`, `policy`, and `command_binding`:

```python
from motrix_deploy.runtime.control import ControlSession
from motrix_deploy.runtime.factory import create_simulation_runtime

runtime = create_simulation_runtime("mujoco", config)
robot = runtime.robot
control = ControlSession(
    robot=robot,
    task=task,
    policy=policy,
    command_binding=command_binding,
    period_s=0.02,
)
runtime.bind_control_session(control)
with runtime:
    result = runtime.run(steps=50)
```

Construct a robot-dependent task from `robot.spec` after runtime creation. Simulation
runtimes derive `RobotSpec` only from the scene; they do not accept an external spec.
For artifact deployment, the CLI validates the artifact's robot contract against
`runtime.robot.spec` before assembling control.
`ControlSession.period_s` is the single control-period setting; `SimulationRuntimeConfig`
does not duplicate it. The simulation runtime validates it against the physics timestep
when binding or resolving the session. Runtime creation does not accept controller-building callbacks. The runtime context owns
world resources and simulation advancement; `ControlSession` owns shared control
execution, and `RobotInterface` owns robot I/O without advancing or closing the world.
Hardware applications use `create_hardware_runtime(name, config, context)` followed
by the same explicit construction and binding. Direct callers can also supply a
session to `runtime.run(control, steps=...)`.

### Select native MotrixSim deployment

The installed simulation recipes default to MuJoCo. With `motrix-deploy-motrixsim`
installed, choose the native backend using the same recipe and artifact:

```bash
motrix-deploy task=go2-walk-flat/sim artifact=artifacts/go2-walk-flat.deploy \
  runtime.backend=motrixsim runtime.viewer=false runtime.realtime=false duration_s=1.0
```

The same override works for `task=go2-walk-rough/sim`; no duplicated task recipe is
needed. Programmatic applications call `create_simulation_runtime("motrixsim", config)`
and follow the explicit session-binding sequence above. The Go2 state mappings remain
`gyro`, `accelerometer`, and `global_linvel`: native sensors validate local angular
velocity/acceleration and world-frame linear velocity. The runtime owns native MSD
compilation, model/data, reset, physics, and optional SDK `RenderApp` rendering,
not a MuJoCo runtime or training environment. Removing `runtime.viewer=false` enables the
native viewer and its keyboard provider in a working graphical environment.

For the programmatic ONNX example, run
`python examples/deploy_to_sim.py --backend motrixsim --headless --steps 50`
with a matching ONNX policy; see the example's policy-file options. This path loads raw ONNX, not a
`.deploy` manifest: task preprocessing, tensor specs, and control settings are explicit
in the Python example and must match the model. Its scene and controller are shared
with the default MuJoCo path. Headless uses a zero-velocity command; windowed mode uses
the viewer keyboard provider when available.

### Use an external Hydra config

The native commands above work with the installed task package, without a checkout config directory. The canonical
built-in runtime recipes live in `motrix_deploy_tasks/config/task/{task}/{sim,hardware}.yaml`; prefer `task=...` to select them.
For a custom application's own configuration, Hydra's standard `--config-path` and `--config-name` remain available.
This illustrative command assumes you have authored `/absolute/path/to/my_app/config/deploy.yaml`; it does not select
or copy a built-in recipe:

```bash
motrix-deploy \
  --config-path /absolute/path/to/my_app/config \
  --config-name deploy \
  artifact=artifacts/go2-walk-flat.deploy
```

Use an absolute config path: Hydra resolves relative paths against the CLI module, not the shell's current directory.
External configs own their defaults and complete runtime/backend configuration, and need not use the installed `task`
group. For simulation, select `runtime=sim` in defaults and provide `runtime.backend`, `runtime.deploy_env_id`, and `runtime.robot_id`, plus all
settings required by the selected backend (including MuJoCo state-sensor mappings). The artifact does not supply these
runtime choices. Use Hydra's normal options; there is no additional `--config` argument.
