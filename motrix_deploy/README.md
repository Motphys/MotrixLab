# Motrix Deploy

`motrix_deploy` is MotrixLab's framework-independent policy deployment library. It defines artifact, backend, policy,
control-loop, deployment-profile registry, task registry, backend plugin discovery, and CLI without importing concrete
tasks, simulator implementations, training environments, or RL frameworks. The simulator and hardware backends live in the
sibling `motrix_deploy_mujoco` and
`motrix_deploy_unitree` plugin packages. Concrete task semantics live in `motrix_deploy_tasks`, which publishes the
`motrix-deploy` executable bootstrap so tasks are registered before the core CLI runs. Built-in environment profile
compilers live under `motrix_envs.deploy`.

The supported vertical slice exports a metadata-backed Go2 RSL-RL walk policy and runs the same artifact in MuJoCo or
on a physical Go2.
From the repository root, install the required development extras:

```bash
sh install.sh --all
```

Export a new artifact directory, validate it without opening a backend, and run an interactive deployment:

```bash
python scripts/export_deploy.py env=go2-walk-rough

motrix-deploy inspect \
  artifact=artifacts/go2-walk-rough.deploy

motrix-deploy sim2sim \
  artifact=artifacts/go2-walk-rough.deploy
```

For the flat-terrain environment, export its own artifact and select the matching runtime recipe:

```bash
python scripts/export_deploy.py env=go2-walk-flat

motrix-deploy inspect \
  artifact=artifacts/go2-walk-flat.deploy

motrix-deploy sim2sim \
  --config-name go2_walk_flat_sim2sim \
  artifact=artifacts/go2-walk-flat.deploy
```

Export defaults are defined in `configs/deploy/export.yaml`. The default rough-terrain workspace runtime is defined in
`configs/deploy/sim2sim/go2_walk_sim2sim.yaml`; the explicit flat-terrain recipe is
`configs/deploy/sim2sim/go2_walk_flat_sim2sim.yaml`. Both can be changed with Hydra overrides. The packaged
`motrix_deploy/config/deploy.yaml` is only a task-agnostic mandatory-field template. The deployment runs until the viewer
closes, the user presses Esc/Ctrl-C, or a runtime failure occurs. It then prints a JSON `RolloutResult`; artifact or
configuration failures exit with code 2.

The minimal MuJoCo GLFW viewer opens by default and uses realtime pacing. Physics stepping remains owned by the backend
control path, and closing the window interrupts deployment safely.

The artifact writer is create-only. Choose a new output path or move an existing artifact before exporting again. Joint,
actuator range, servo gain, tensor shape, checksum, and command-range mismatches fail before the first command.
Headless and physical runs may set either `rollout.steps` or `rollout.duration_s`; viewer runs may leave both unset.
`realtime` defaults to the viewer mode when omitted.

Optional dependencies remain isolated behind extras for smaller runtime environments (ONNX
inference is a core dependency; `mujoco` and `unitree` select the deployment backends):

```bash
uv sync --package motrix-deploy-tasks --extra mujoco --extra unitree
```

Interactive deployment reads keyboard events directly from the focused GLFW viewer window:

```bash
motrix-deploy sim2sim \
  artifact=artifacts/go2-walk-rough.deploy
```

Viewer mode uses realtime pacing. Hold `W/S`, `A/D`, and `Q/E` to set the signed forward, lateral, and yaw-rate
axes to the lower or upper command bound stored in the artifact; releasing a key removes its contribution, and Esc
interrupts the rollout. Losing viewer focus releases every held key, so typing in another window cannot command the robot.
The deployment viewer does not register MuJoCo's built-in visualization shortcuts, so `W/A/S/D` do not have a second effect.
Drag with the left mouse button to rotate the camera, right-drag to move it, and use the middle button or wheel to zoom.
The built-in interactive keyboard path requires `viewer=true`.

The standing-probability random binding is task-specific training behavior and is not exposed by the deployment runtime.

## Physical Go2 Sim2Real

The hardware plugin imports the Unitree SDK2 Python package only when a physical backend is opened. The normal
workspace sync installs the pinned Motphys-maintained fork into the same environment:

```bash
sh install.sh --all
```

Inspect the artifact first. With the robot suspended, low-level/debug mode enabled, Ethernet connected, and an emergency
stop operator ready, explicitly select the interface and confirm the hardware checklist:

```bash
motrix-deploy inspect \
  artifact=artifacts/go2-walk-rough.deploy

motrix-deploy sim2real \
  artifact=artifacts/go2-walk-rough.deploy \
  backend.network_interface=enp3s0 \
  hardware.confirm=true
```

`hardware.confirm` defaults to `false`; `viewer=false` and `realtime=true` are mandatory for this backend. Before creating the `LowCmd` publisher, the backend stands down and releases the active MotionSwitcher mode (MCF), then disables the `sport_mode` service through RobotStateClient; any failure aborts startup. `StandDown` is a physical motion that occurs before the Start-button gate, so the area must be clear and an independent emergency stop must be ready. The Unitree remote is the default command source: left-stick Y controls forward velocity, left-stick X lateral velocity, and right-stick X yaw. The default mapping preserves the forward axis and inverts the lateral and yaw axes to match the policy convention. A 0.12 deadzone is applied and L1 is a deadman switch; releasing L1 immediately commands zero, while artifact bounds remain authoritative. The plugin
publishes zero torque until Start, interpolates from the measured joints to the artifact default pose over two seconds,
holds that pose until A, then enables policy commands. B interpolates to the configured backend-owned lie-down pose, then
enters damping and exits with `exit_reason=lie_down`; this does not widen artifact policy limits. Select remains the
higher-priority emergency stop. Select, Ctrl+C, stale state, or a runtime failure enters damping before both DDS channels
are closed. Artifact order `FL, FR, RL, RR` is mapped explicitly
to SDK motor order `FR, FL, RR, RL`. Software and injected-fake-SDK tests cover this path; a suspended real-robot smoke
test is still required before real-world operation.
