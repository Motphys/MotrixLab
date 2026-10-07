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
for simulation or `unitree_go2` for hardware). `runtime.backend` is a scalar string selector, not a nested configuration
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
