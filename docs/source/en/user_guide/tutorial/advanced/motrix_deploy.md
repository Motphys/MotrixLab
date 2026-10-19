# Go2 Flat-Terrain Walking: From Training to Physical Deployment

Use the same deployment artifact to check a Go2 policy in simulation before attempting hardware deployment. With an
artifact and the task/MuJoCo plugins installed, a bounded headless check is:

```bash
mxlab deploy task=go2-walk-flat/sim artifact=artifacts/go2-walk-flat.deploy runtime.viewer=false duration_s=2.0
```

The workflow below covers training, export, simulation, operator-confirmed hardware deployment, and basic diagnostics.
Simulation success is not evidence of hardware safety.

## 1. Install and train

From the repository root:

```bash
sh install.sh --all
source .venv/bin/activate
mxlab train task=go2-walk-flat/rslrl.ppo
```

Training results are saved under `runs/go2-walk-flat/`. Skip training if you already have a run with its metadata.
Export needs the run's training provider and environment package; deployment needs the task package and the backend
plugin for the target.

## 2. Export and inspect

```bash
mxlab export-deploy env=go2-walk-flat output=artifacts/go2-walk-flat.deploy
mxlab deploy inspect artifact=artifacts/go2-walk-flat.deploy
```

`env=go2-walk-flat` selects the latest run with metadata. Use `run_dir=<run-dir>` instead to select a specific run;
exactly one of `env` and `run_dir` is required. Existing artifact directories are not overwritten: choose a new output
path when exporting again. Confirm that inspection reports `valid: true`.

The artifact contains the policy and the robot/task settings needed to interpret it. Copy the complete directory, not
just its ONNX file. `mxlab export-onnx` exports a standalone model and is not a replacement for deployment export.

## 3. Check in simulation

Open the MuJoCo viewer:

```bash
mxlab deploy task=go2-walk-flat/sim artifact=artifacts/go2-walk-flat.deploy
```

Hold `W/S` to move forward/backward, `A/D` sideways, and `Q/E` to turn. Close the window, press Esc, or press Ctrl+C to
stop. Headless mode uses zero velocity. `duration_s` bounds policy execution time, not the entire process's wall time;
startup and shutdown can take additional time.

With the native MotrixSim deployment plugin installed, reuse the same artifact and recipe:

```bash
mxlab deploy task=go2-walk-flat/sim artifact=artifacts/go2-walk-flat.deploy \
  runtime.backend=motrixsim
```

Use `task=go2-walk-rough/sim` for the supplied rough-terrain scene. Changing the scene does not make a flat-terrain policy
suitable for rough terrain; check that the policy is appropriate for the test.

## 4. Run on Go2 hardware

### Prepare the robot and connection

Use reliable load-bearing support, clearance for every leg, an operator-ready independent emergency stop, and a second
operator. Stop all other low-level command publishers. Neither damping nor a stopped deployment maintains standing
support. Do not proceed without a supported recovery plan.

Find the Ethernet interface connected to the robot:

```bash
ip -br addr
```

Replace `enp5s0` below with that interface. First perform the read-only check in section 5. Then start deployment:

```bash
mxlab deploy task=go2-walk-flat/hardware \
  artifact=artifacts/go2-walk-flat.deploy \
  runtime.transport.network_interface=enp5s0
```

### Hand over official control

If official control is active, the command asks for two confirmations:

1. Type `yes` only when ready to authorize the official StandDown movement.
2. Observe that movement has finished and the body has reliable support. Then type `down` to authorize release.

An accepted StandDown request does not prove that movement is complete. If official control is already released, these
prompts and official movements are skipped. Refusal, EOF, or Ctrl+C at a confirmation aborts deployment; cancelling at
`down` preserves official ownership but does not undo StandDown.

**Official control is not restored automatically.** After release, an error or interruption does not restore it either.
This handoff and these prompts are Go2-specific, not a G1 startup procedure.

### Prepare, take over, and stop

| Control | Action |
| --- | --- |
| Start | Begin the transition to the default standing pose. |
| A | Enable the policy after measured readiness. Press it again after preparation is ready; holding it early does not bypass readiness. |
| Hold L1 + sticks | Send motion commands. Start with small commands under protection. |
| B | After preparation has started, transition toward the built-in lie-down pose, then damp and finish. Before Start, finish with damping only. |
| Select | Latch a software emergency damping stop; not a substitute for an independent emergency stop. |

The B transition is constrained by the artifact's joint limits. A startup warning indicates a limited target that may
only produce a shallow crouch; it does not guarantee body contact or stability. Maintain load-bearing protection when
stopping. Communication loss, state timeout, Ctrl+C, and other faults **do not execute the normal lie-down trajectory**.

After exit, ensure that all low-level publishers have stopped before restoring official control using the robot's
supported procedure. Do not enable two controllers simultaneously.

## 5. Diagnose before running a policy

### Read state without commanding motion

```bash
motrix-deploy-unitree read-lowstate enp5s0 --robot go2 --duration-s 10
```

Check for valid state samples and CRC/decode errors. The tool sends no motor commands and does not release official
control. Remote buttons may still affect the official controller while it is active; do not treat button checks as
side-effect-free.

### Run a bounded single-joint test

This test needs no artifact or policy. It can move **all twelve joints**, including with `--offset 0`, because it first
moves the whole robot to a holding pose. Use support, clearance, and gains reviewed for this robot and setup.

First explicitly release official control; follow its `yes` and `down` confirmations:

```bash
motrix-deploy-unitree release-control enp5s0
```

Then run the test with explicitly selected gains:

```bash
motrix-deploy-unitree joint-control enp5s0 FL_thigh_joint \
  --offset 0.05 --kp "${KP:?Set reviewed stiffness}" --kd "${KD:?Set reviewed damping}"
```

The command displays its targets and asks for `yes` before connecting. Start begins the holding-pose transition; A
starts the selected-joint test. `--offset` is in radians relative to the holding pose, not the initial measured angle;
use `--target` instead for an absolute angle. B returns to the holding pose and stops, except before or during startup,
when it cancels directly into damping. **This diagnostic B behavior differs from walking deployment's lie-down.**

Use `motrix-deploy-unitree joint-control --help` for motion/startup timing options. Limit checks and startup protections
do not certify mechanical safety or total actuator torque.

## 6. Common settings and failures

Pass overrides after the selected task, for example:

```bash
mxlab deploy task=go2-walk-flat/sim artifact=artifacts/go2-walk-flat.deploy \
  runtime.viewer=false runtime.realtime=true duration_s=5.0
```

| Setting | Use |
| --- | --- |
| `artifact` | Choose the complete deployment artifact directory. |
| `duration_s` | Bound policy execution; it does not impose a wall-clock timeout. |
| `runtime.backend` | Select `mujoco` or `motrixsim` for simulation. Use the hardware recipe for Unitree. |
| `runtime.viewer`, `runtime.realtime` | Simulation display and pacing only. Hardware execution is always paced in real time. |
| `runtime.transport.network_interface` | Required Ethernet interface for hardware. |
| `task_settings.preparation` | Go2 preparation gains, timing, readiness limits, and buttons. Preparation gains do not replace policy gains. |

Tune hardware preparation only after checking mechanical clearance, support, and the reported measured error. A visually
plausible pose does not establish safe policy takeover. Do not relax communication timeouts to hide missing state.

| Symptom | Next check |
| --- | --- |
| Interface does not match an available interface | Run `ip -br addr` on the machine/environment running deployment. |
| No valid state or CRC/decode errors | Stop at read-only diagnostics and check network/device setup. |
| Standing preparation timed out | Inspect the reported joints, speed, and tilt; check interference and load before tuning. |
| `state_timeout` during execution | Treat it as a fault stop. Preserve the result/error, ensure support, and investigate reception before retrying. |

## 7. Other tasks and Python examples

G1 WBT dance uses its own artifact and task recipe. For a simulation check:

```bash
mxlab deploy task=g1-wbt-dance/sim artifact=artifacts/g1-wbt-dance.deploy runtime.viewer=false
```

The artifact supplies the motion clip; no velocity commands are needed. Its duration cannot exceed that clip. Playback
ends with damping, not a guaranteed standing recovery. Do not apply Go2 controls or its handoff procedure to G1.

Run the programmatic Python example from the repository root; it neither reads an artifact nor invokes the deployment CLI:

```bash
python examples/deploy.py --headless --steps 100
python examples/deploy.py --backend motrixsim --headless --steps 100
```

After completing the hardware safety preparations in section 4, explicitly select the real robot:

```bash
python examples/deploy.py --real --network-interface enp5s0
```

The example uses raw ONNX with robot and task parameters declared in code. When replacing the model, verify its training
contract; matching input/output dimensions alone is not sufficient for hardware use. The real path retains control handoff
and remote confirmations. CLI deployment still uses the artifact workflow above.
