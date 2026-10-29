# Physical deployment

This page explains how to deploy the `go2-walk-flat` policy to a Unitree Go2: export and inspect a deployment artifact, check the policy in MuJoCo or MotrixSim, and then connect the real robot using the safety procedure. Simulation success does not establish hardware safety. For the first run, prepare load-bearing support, an independent emergency stop, and an on-site operator.

```{video} /_static/videos/go2_hardware_deployment.mp4
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
:align: center
```

## 1. Train and export the policy

If you do not already have a training run, execute this from the repository root:

```bash
source .venv/bin/activate
mxlab train task=go2-walk-flat/rslrl.ppo
```

Training results are stored under `runs/go2-walk-flat/`. Export the latest run with metadata, or select a specific run with `run_dir`; exactly one of them may be set. When `output` is omitted, the artifact is saved to `artifacts/<env>.deploy` by default:

```bash
mxlab export-deploy env=go2-walk-flat
# or:
mxlab export-deploy run_dir=/path/to/run

mxlab deploy inspect artifact=artifacts/go2-walk-flat.deploy
```

Inspection must report `valid: true`. Add `output=<artifact-path>` only when a custom output location is needed.

Deployment requires `motrix_deploy_tasks` and the backend plugin for the selected target. The deployment recipe uses `runtime.transport.network_interface` to select the hardware Ethernet interface.

## 2. Check the policy in simulation first

Open the MuJoCo viewer:

```bash
mxlab deploy task=go2-walk-flat/sim \
  artifact=artifacts/go2-walk-flat.deploy
```

Use `W/S` for forward/backward motion, `A/D` for lateral motion, and `Q/E` for yaw. Close the window, press Esc, or press Ctrl-C to exit.

After installing `motrix-deploy-motrixsim`, the same artifact can be reused:

```bash
mxlab deploy task=go2-walk-flat/sim \
  artifact=artifacts/go2-walk-flat.deploy \
  runtime.backend=motrixsim
```

`task=go2-walk-rough/sim` selects the rough-terrain scene, but a flat-ground policy is not automatically suitable for rough terrain. Go2 flat-ground training uses approximately `kp=35, kd=0.5` by default, while the hardware recipe uses `kp=50, kd=1`; confirm such differences before hardware execution.

## 3. Connect the robot and prepare safely

Before starting:

- use reliable load-bearing support and provide clearance for every leg;
- prepare an independent emergency stop within the operator's reach and assign a second operator;
- stop every other low-level command publisher;
- remember that damping or deployment shutdown cannot maintain standing support, and prepare a supported recovery plan;
- for the first run, suspend the Go2 and switch it to low-level/debug mode.

Find the wired interface connected to the robot:

```bash
ip -br addr
```

Replace `enp5s0` in the commands below with the actual interface. The PC and Go2 must be on the same subnet, for example:

```text
PC address: 192.168.123.100
Go2 address: 192.168.123.161
```

Connect the Ethernet cable and verify communication:

```bash
ping 192.168.123.161
```

## 4. Perform a read-only check before running the policy

The read-only diagnostic does not create a command publisher or send motor commands:

```bash
motrix-deploy-unitree read-lowstate enp5s0 --robot go2 --duration-s 10
```

Confirm that valid state samples continue to arrive and that there are no CRC or decoding errors. Remote buttons may still trigger the official controller while it is active; do not treat button checks as side-effect-free.

## 5. Start the hardware policy

Keep `kp=50` and `kd=1` for the first run; do not increase gains before the suspended check. Maintain load-bearing protection when stopping. Communication loss, state timeout, Ctrl-C, and other faults do not execute the normal lie-down trajectory.

```bash
mxlab deploy task=go2-walk-flat/hardware \
  artifact=artifacts/go2-walk-flat.deploy \
  runtime.transport.network_interface=enp5s0
```

If official control is still active, deployment asks for two confirmations: type `yes` only when ready to authorize the official StandDown movement; after observing that movement has finished and the body has reliable support, type `down` to authorize release. If official control has already been released, this step is skipped. An error or interruption after release does not restore official control automatically; never enable two controllers at once.

Use these controls during preparation and operation:

| Control | Action |
| --- | --- |
| `Start` | Begin the transition to the default standing pose. |
| `A` | Enable the policy after preparation is complete and measured state is ready. |
| Hold `L1` and move the sticks | Send velocity commands; use small commands on the first run. |
| `B` | Transition toward the built-in lie-down pose, then finish with damping. Before preparation starts, apply damping only. |
| `Select` | Latch a software emergency damping stop; not a substitute for an independent emergency stop. |

The Go2 task maps the left stick to planar motion and the right stick to yaw; `L1` must be held for stick commands to reach the policy. At the end, confirm that all low-level publishers have stopped before restoring official control using the robot's supported procedure.
