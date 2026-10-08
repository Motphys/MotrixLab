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

The artifact directory is self-contained:

```text
go2-walk-flat.deploy/
├── manifest.json        # robot/control/task settings, payload checksums
├── policy/model.onnx    # exported actor with input normalization
└── task/                # task-owned payloads (e.g. the G1 motion NPZ)
```

Deployment also needs the installed `motrix_deploy_tasks` plugin, which supplies the walking implementation, deployment
scenes, and Hydra recipes, plus the runtime plugin for the target. The recipes work outside the repository without a
workspace config path. Task defaults select the Hydra runtime group (`sim` or `hardware`); you do not need a separate
runtime override. Common top-level settings are `artifact`, `duration_s`, and `command`. All target-specific settings
live directly under `runtime`. `runtime.kind` selects the simulation or hardware path; `runtime.backend` selects the
plugin (`mujoco` or `motrixsim` for simulation, or `unitree_go2` for hardware). `runtime.backend` is a scalar string
selector, not a nested configuration mapping or Hydra group.

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

The example loads the model externally and injects it into the task; the task computes commands, the control session
owns robot I/O and execution statistics, and the runtime owns physics, scheduling and the viewer. The
[advanced assembly](#assemble-the-runtime-and-task-explicitly) section shows the same composition step by step.
Raw ONNX use requires explicit task preprocessing and action settings; the artifact supplies those settings for the CLI.

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

## Deploy the G1 WBT dance

The G1 whole-body-tracking dance task uses the same export → artifact → `motrix-deploy` pipeline. The artifact embeds
the servo gains, action scales, preparation gains and timing, safety thresholds, and the motion clip itself (as a
`payloads/motion.npz` payload with its checksum in the manifest), so deployment needs neither Torch nor an RL environment.
Export needs the training provider and task environment packages, but does **not** train:

```bash
source .venv/bin/activate
RUN=runs/g1-wbt-dance/motrix/torch/fastsac/<run-dir>
python scripts/export_deploy.py run="$RUN" \
  output=artifacts/g1-wbt-dance-software-pd.deploy validation.atol=3e-5

motrix-deploy task=g1-wbt-dance/sim artifact=artifacts/g1-wbt-dance.deploy
```

Headless validation and artifact inspection:

```bash
motrix-deploy task=g1-wbt-dance/sim artifact=artifacts/g1-wbt-dance.deploy \
  runtime.viewer=false
motrix-deploy inspect artifact=artifacts/g1-wbt-dance.deploy
```

The default duration comes from the embedded motion clip; shorter runs use `duration_s`, longer durations are rejected
before opening the backend instead of looping or holding. No velocity command is needed: the motion is artifact-owned.
The same artifact runs on MuJoCo and MotrixSim (`runtime.backend=motrixsim`, with
`runtime.physics.solver_iterations=3` matching the training budget). It is not a hardware deployment recipe: the current
Unitree transport is Go2-specific, and sim2sim success is not evidence of hardware safety.

The task mirrors real deployment practice with four phases: `prepare` (ramp measured joints to the default standing
pose, gated on continuously measured readiness), `policy_hold`, `playback`, and `damping`. Takeover after readiness is
automatic by default; programmatic applications can stage it manually with `automatic_start=False`, the request
methods, or an attached keyboard (`p` start policy, `m` start motion, `o` stop). Note that the dance ends in a
deliberately leaned pose that only the tracking policy can balance: after the playback budget the task enters measured
damping directly, and ending upright requires the motion clip itself to finish at the standing pose.

## Advanced usage

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

### Hardware sensor and motor wiring

`motrix_deploy_unitree.hardware.UnitreeGo2HardwareCfg` owns asset-free hardware poses and wiring.
`HardwareSensorBinding` maps SDK IMU fields, including `wxyz` quaternion conversion to canonical `xyzw`;
`UnitreeMotorBinding` maps canonical joint names to SDK motor indices. The artifact still owns the policy joint
order, default pose, and limits.

### Select a deployment world

Select a world with `runtime.deploy_env_id` and a robot with `runtime.robot_id`. To keep the same Go2 recipe and
artifact but switch its world:

```bash
motrix-deploy task=go2-walk-flat/sim artifact=artifacts/go2-walk-flat.deploy \
  runtime.deploy_env_id=rough
```

A world-only override leaves robot attachment placement unchanged. Adjust `runtime.robot_translation` if the terrain
needs different clearance; the complete `task=go2-walk-rough/sim` recipe supplies its own attachment defaults.
`RobotCfg.translation`/`rotation` are attachment transforms composed once with the model's intrinsic base pose:
Go2's intrinsic base height is 0.445 m, so the flat recipe's -0.114 m offset gives 0.331 m and the rough recipe's
-0.025 m offset gives 0.42 m.

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

`runtime.physics` uses the shared `SimCfg` (`dt`, `solver_iterations`, `solver_tolerance`, `gravity`, defaulting to
`dt=0.002, solver_iterations=100`); optional fields set to `None` preserve model-source values. Set physics values
before runtime construction — changing `dt` after compilation is not supported.

`runtime.sensor_bindings` maps state roles to model-local sensor names and supports only `base_angular_velocity`,
`base_linear_acceleration`, and `base_linear_velocity` (`str | None`). MuJoCo requires the angular-velocity and
acceleration bindings; the optional `base_linear_velocity` remains `None` when unbound.

Programmatic applications assemble the same scene and pass it to `SimulationRuntimeConfig`:

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

Additional worlds come from installed plugins: declare a zero-argument `SceneCfg` factory in the
`motrix_deploy.envs` entry-point group, and `motrix_deploy.env.available_deploy_envs()` lists the installed IDs.

### Assemble the runtime and task explicitly

The runtime factory selects the backend and prepares `runtime.robot`; its `RobotSpec` is available before `open()`.
Load the model externally, then construct and bind the session. Using the `config` above and your application-owned
`task_spec`, `model`, and Go2 `command_binding`:

```python
from motrix_deploy.runtime.control import ControlSession
from motrix_deploy.runtime.factory import create_simulation_runtime
from motrix_deploy.task import create_task

runtime = create_simulation_runtime("mujoco", config)
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

`create_task(spec, robot, policy, steps=None)` resolves the concrete task class from the spec's task name; tasks that
reference artifact payloads (such as the G1 motion NPZ) receive the artifact through the same call. The control period
must be an integer multiple of the physics timestep. `command_binding` defaults to `None`: autonomous tasks like the
G1 dance omit it, while Go2 walking requires an externally assembled velocity binding whose values the task validates
against its velocity bounds. Hardware applications use `create_hardware_runtime(name, config, context)` and the same
session assembly.

`runtime.run()` returns a rollout result with the success flag, exit reason, completed model steps, wall time, and
latency statistics; simulation runs additionally report `real_time_factor`.

### Select native MotrixSim deployment

The installed simulation recipes default to MuJoCo. With `motrix-deploy-motrixsim`
installed, choose the native backend using the same recipe and artifact:

```bash
motrix-deploy task=go2-walk-flat/sim artifact=artifacts/go2-walk-flat.deploy \
  runtime.backend=motrixsim runtime.viewer=false runtime.realtime=false duration_s=1.0
```

The same override works for `task=go2-walk-rough/sim`; no duplicated task recipe is
needed. Programmatic applications call `create_simulation_runtime("motrixsim", config)`
and follow the explicit session-assembly sequence above. The Go2 state mappings remain
`gyro`, `accelerometer`, and `global_linvel`: native sensors validate local angular
velocity/acceleration and world-frame linear velocity. Removing `runtime.viewer=false` enables the
native viewer and its keyboard provider in a working graphical environment.

For the programmatic ONNX example, run
`python examples/deploy_to_sim.py --backend motrixsim --headless --steps 50`
with a matching ONNX policy; see the example's policy-file options. This path loads raw ONNX, not a
`.deploy` manifest: task preprocessing, tensor specs, and control settings are explicit
in the Python example and must match the model. Headless uses a zero-velocity command; windowed mode uses
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
