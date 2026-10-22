# Go2 flat-ground walking: sim2sim and sim2real

This page describes the complete path from a `go2-walk-flat` training policy to deployment: export an artifact, check it in MuJoCo or MotrixSim, and then run it on the real robot through the Unitree SDK2 DDS backend. The artifact keeps the policy, joint order, default pose, action scaling, and control period together; passing simulation does not guarantee hardware safety.

## 1. Train the policy

```bash
source .venv/bin/activate
mxlab train task=go2-walk-flat/rslrl.ppo
```

Training outputs are stored under `runs/go2-walk-flat/`, and the trained gait should be replayed with `mxlab play` for inspection.

## 2. Export and inspect the artifact

```bash
mxlab export-deploy env=go2-walk-flat output=artifacts/go2-walk-flat.deploy
mxlab deploy inspect artifact=artifacts/go2-walk-flat.deploy
```

To pin a specific experiment, use `run_dir=/path/to/run` instead of `env`; exactly one may be set. Inspection must report `valid: true`. The artifact is self-contained:

```text
go2-walk-flat.deploy/
├── manifest.json        # robot, control, task settings, and payload checksums
├── policy/model.onnx    # exported actor with input normalization
└── task/                # task-owned payloads (for example, a G1 motion NPZ)
```

Deployment also requires the `motrix_deploy_tasks` plugin, which provides the walking task implementation, deployment scene, and Hydra recipe, together with the runtime plugin for the selected target. The recipe is installed with the plugin and does not require a workspace configuration path when run outside the repository. Task defaults compose the typed runtime configuration, so no separate runtime group override is needed. Common top-level settings are `artifact` and `duration_s`. Runtime settings live under `runtime`; Unitree robot options live under `runtime.robot`, while transport settings such as `runtime.network_interface` remain directly under `runtime`. The configuration distinguishes simulation and hardware execution; `runtime.backend` selects `mujoco` or `motrixsim` for simulation and `unitree` for hardware. It is a scalar string selector, not a nested configuration mapping or Hydra config group.

## 3. sim2sim: MuJoCo

Headless bounded run:

```bash
mxlab deploy task=go2-walk-flat/sim \
  artifact=artifacts/go2-walk-flat.deploy \
  runtime.viewer=false duration_s=2.0
```

For interactive viewing, omit `runtime.viewer=false` to use the MuJoCo viewer:

```bash
mxlab deploy task=go2-walk-flat/sim \
  artifact=artifacts/go2-walk-flat.deploy
```

In the viewer, use `W/S` for forward/backward motion, `A/D` for lateral motion, and `Q/E` for yaw; press Esc or Ctrl-C to exit.

```{video} /_static/videos/go2_sim2sim.mp4
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 60%
:align: center
```

To select native MotrixSim, install `motrix-deploy-motrixsim` and reuse the same artifact:

```bash
mxlab deploy task=go2-walk-flat/sim \
  artifact=artifacts/go2-walk-flat.deploy \
  runtime.backend=motrixsim runtime.viewer=true
```

Use `task=go2-walk-rough/sim` to select the packaged rough-terrain scene. Changing the scene does not mean that a flat-ground policy is suitable for rough terrain; confirm policy applicability before testing.

Velocity tracking in simulation also does not equal hardware tracking. Simulation may randomize action delay, friction, mass, center of mass, observation noise, and other factors. A policy that walks in simulation is not necessarily stable on the real robot. Go2 simulation training and the artifact use approximately `kp=35, kd=0.5` by default, while the hardware recipe overrides these to `kp=50, kd=1`; this is one clear difference between sim2sim and sim2real.

## 4. Hardware deployment: Unitree Go2 sim2real

Before starting, prepare reliable load-bearing supports, sufficient clearance for all legs, an independent emergency stop that the operator can reach, and a second operator. Stop all other low-level command publishers. Damping and deployment shutdown do not provide standing support; do not continue without a supported recovery plan. Before hardware execution, suspend the Go2 and switch it to low-level/debug mode.

Find the wired network interface and replace `enp5s0` in the commands below with the actual interface:

```bash
ip -br addr
```

Connect the PC and Go2 with an Ethernet cable and verify that they can communicate. Set the PC Ethernet address on the same subnet as the robot, for example:

```text
PC address: 192.168.123.100
Go2 address: 192.168.123.161
```

Test the connection to the Go2 onboard computer:

```bash
ping 192.168.123.161
```

### Read-only state check

This diagnostic does not create a command publisher. Replace `enp5s0` with the actual interface:

```bash
motrix-deploy-unitree read-lowstate enp5s0
```

Check that valid state messages continue to arrive and that there are no CRC or decoding errors. The tool does not send motor commands or take official control. While official control remains active, remote buttons may also trigger official actions; do not treat button checks as side-effect-free operations.

### Bounded single-joint check

This test does not require an artifact or policy, but the whole body first transitions to a hold pose. Therefore all 12 joints may move even with `--offset 0`. Prepare supports and clearance, and use gains verified for the current robot and support arrangement, such as `kp=50` and `kd=1`.

```bash
motrix-deploy-unitree joint-control enp5s0 FL_thigh_joint 0.9 \
  --artifact artifacts/go2-walk-flat.deploy
```

`Start` begins the hold-pose transition and `A` starts the selected joint test. `B` returns to the hold pose and then stops; before the test starts or while the startup transition is still running, `B` cancels and applies damping directly.

### Full hardware run

Keep `kp=50` and `kd=1` for the first run; do not increase gains before the suspended check. To reduce the fall risk from a DDS disconnect, press `L2+A` twice first to request lie-down. Replace `enp5s0` with the actual interface:

```bash
mxlab deploy task=go2-walk-flat/hardware \
  artifact=artifacts/go2-walk-flat.deploy \
  runtime.network_interface=enp5s0
```

Press `Start`, wait for the pose transition to finish, then press `A` to enable the policy. Hold `L1` while moving the stick to send commands. `B` requests lie-down and `Select` triggers the emergency stop. Send damping before closing DDS.

```{video} /_static/videos/go2_sim2real_h264.mp4
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 50%
:align: center
```
