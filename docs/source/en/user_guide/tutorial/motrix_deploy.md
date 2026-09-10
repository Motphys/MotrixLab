# Go2 Flat-Terrain Walking: From Training to Physical Deployment

This tutorial follows the full Go2 flat-terrain workflow: train a policy, export an artifact, check it in MuJoCo, and run
it on a real robot. If you already have a training run, start at “Export the artifact”.

## 1. Install the environment

Run this command from the repository root:

```bash
sh install.sh --all
```

This installs the training, MuJoCo, ONNX Runtime, and Unitree SDK2 dependencies.

## 2. Train the policy

If you do not have a run yet, train the flat-terrain Go2 policy:

```bash
python scripts/train.py task=go2-walk-flat/rslrl.ppo
```

Training results are saved under `runs/go2-walk-flat/`.

## 3. Export the artifact

Export the latest run:

```bash
python scripts/export_deploy.py env=go2-walk-flat
```

The output is written to `artifacts/go2-walk-flat.deploy/`. This artifact is the only policy bundle needed for deployment;
it contains the model and its runtime configuration.

## 4. Inspect the artifact

```bash
motrix-deploy inspect \
  artifact=artifacts/go2-walk-flat.deploy
```

Make sure the output reports `valid: true`.

## 5. Run Sim2Sim first

```bash
motrix-deploy sim2sim \
  --config-name go2_walk_flat_sim2sim \
  artifact=artifacts/go2-walk-flat.deploy
```

This opens the MuJoCo viewer. Use `W/S` to move forward and backward, `A/D` to move sideways, and `Q/E` to turn. Close
the window, press Esc, or press Ctrl-C to stop.

## 6. Run on the real robot

Put the robot in low-level/debug mode, connect Ethernet, and keep an emergency stop ready. Replace `enp5s0` with the
actual network interface:

```bash
motrix-deploy inspect \
  artifact=artifacts/go2-walk-flat.deploy

motrix-deploy sim2real \
  --config-name go2_walk_flat_sim2real \
  artifact=artifacts/go2-walk-flat.deploy \
  backend.network_interface=enp5s0 \
  hardware.confirm=true
```

After startup, press Start on the remote. When the default-pose transition finishes, press A. Hold L1 and move the sticks
to send motion commands. Press B to enter the lie-down sequence; Select triggers the emergency stop.

You can also inspect the robot state before sending policy commands:

```bash
python -m motrix_deploy_unitree.read_lowstate enp5s0
```

## Advanced usage

### Override policy PD gains

The physical-runtime base configuration, `configs/deploy/sim2real/base.yaml`, currently sets `backend.kp=50` and
`backend.kd=1`, overriding the gains stored in the artifact `TaskSpec.config`. Set both fields to `null` to keep the artifact
gains, or pass 12 non-negative values on the command line:

```bash
motrix-deploy sim2real \
  artifact=artifacts/go2-walk-flat.deploy \
  backend.network_interface=enp5s0 \
  'backend.kp=[20,25,30,20,25,30,22,27,32,22,27,32]' \
  'backend.kd=[0.3,0.4,0.5,0.3,0.4,0.5,0.35,0.45,0.55,0.35,0.45,0.55]' \
  hardware.confirm=true
```

### Inspect LowState without sending commands

Run the read-only diagnostic before sending any motion command:

```bash
python -m motrix_deploy_unitree.read_lowstate enp5s0
```

### Send a single-joint motion command

To bypass the policy and debug a single joint-position motion, use the bounded helper installed with
`motrix_deploy_unitree`. By default, it builds the control contract from the current `go2-walk-flat` deployment profile and
does not require a training run, checkpoint, policy, or deployment artifact:

```bash
python -m motrix_deploy_unitree.go2_joint_control \
  enp5s0 \
  FL_thigh_joint \
  0.9 \
  --hardware-confirm
```
