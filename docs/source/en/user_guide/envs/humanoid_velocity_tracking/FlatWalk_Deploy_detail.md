# G1 Flat-Ground Walking: Complete Deployment Guide

This guide covers deploying trained walking policies from simulation to real hardware.

---

## Quick Start

### Prerequisites

- Completed G1 walking training (e.g., `g1-walk-flat`)
- Training run with metadata in `runs/` directory
- MotrixLab deployment packages installed
- Gamepad for velocity control

### Step 1: Export Artifact

```bash
cd /path/to/MotrixLab
source .venv/bin/activate

# Export from latest training run
mxlab export-deploy env=g1-walk-flat output=artifacts/<your_artifact_name>.deploy
```

### Step 2: Validate in Simulation

```bash
# Test in MuJoCo simulation first
mxlab deploy task=g1-walk-flat/sim artifact=artifacts/<your_artifact_name>.deploy
```

**Controls:**
- **P key**: Start walking policy
- **W/A/S/D**: Forward/Left/Backward/Right
- **Q/E**: Turn left/right
- **Esc/Ctrl+C**: Exit

✅ **Only proceed to hardware if simulation succeeds!**

### Step 3: Prepare Hardware

**Safety checklist:**
- [ ] Robot on support frame or sling
- [ ] Emergency stop button ready
- [ ] Two operators present
- [ ] Clear walking area with no obstacles
- [ ] Gamepad connected

**Network setup:**
```bash
# Find your network interface
ip -br addr

# Verify connection to robot (G1 default: 192.168.123.10)
ping 192.168.123.10
```

### Step 4: Install Native SDK (if needed)

**On PC (remote inference):**
```bash
pip install -e motrix_unitree_native

# Verify installation
python -c "from motrix_deploy_unitree.g1 import native; print('OK')"
```

### Step 5: Deploy to Hardware

```bash
# Replace enp5s0 with your network interface name
mxlab deploy task=g1-walk-flat/hardware \
  artifact=artifacts/<your_artifact_name>.deploy \
  runtime.transport.network_interface=enp5s0
```

**Startup sequence:**
1. Program connects to robot
2. **Press Start button** (gamepad) or **R key** → Robot stands up
3. **Press A button** (gamepad) or **P key** → Walking policy enabled
4. Use gamepad to control movement

**Movement control (gamepad):**
- **Left Stick**: Forward/backward and strafe left/right
- **Right Stick**: Yaw rotation
- **Start with small movements to test!**

**Emergency stop:**
- **Select button** (gamepad) or **Physical E-stop** (recommended)

---

## Configuration Options

### Artifact Export Options

**Export from specific run:**
```bash
mxlab export-deploy \
  run_dir=<path/to/metadata.json/parent/directory> \
  output=artifacts/<custom_name>.deploy
```

**Inspect artifact:**
```bash
mxlab deploy inspect artifact=artifacts/<your_artifact>.deploy
```

### Startup Parameters

**Slower transition (safer for first test):**
```bash
mxlab deploy task=g1-walk-flat/hardware \
  artifact=artifacts/<your_artifact>.deploy \
  runtime.transport.network_interface=enp5s0 \
  task_settings.preparation.default_transition_duration_s=5.0
```

**Reduce velocity range (slower movement):**
```bash
runtime.operator.planar_velocity.range_scale=[0.15,0.15,0.15]
```

**Enable deadman button (safety feature):**
```bash
runtime.operator.planar_velocity.deadman_button=LB
```

**Skip manual confirmation (use after testing):**
```bash
task_settings.preparation.wait_for_remote_buttons=false
```

### Configuration File

Edit `/path/to/MotrixLab/motrix_deploy_tasks/src/motrix_deploy_tasks/config/task/g1-walk-flat/hardware.yaml`:

```yaml
runtime:
  transport:
    network_interface: enp5s0  # Your interface
    domain_id: 0
  
  operator:
    planar_velocity:
      linear_x_axis: ly
      linear_y_axis: lx
      yaw_axis: rx
      deadzone: 0.12
      range_scale: [0.25, 0.25, 0.25]  # Reduce for slower movement
      deadman_button: null  # Set to "LB" for safety

task_settings:
  preparation:
    default_transition_duration_s: 3.0  # Transition time
    wait_for_remote_buttons: true       # Manual confirmation
    kp: null  # null uses default gains
    kd: null
```

---

## Hot-Swapping Multi-Policy Mode

Switch between fixstand, walk, and wbt policies without restarting.

**Deploy command:**
```bash
mxlab deploy \
  program=g1-hot-swapping \
  runtime=g1-unitree-state-machine \
  program.walk_artifact=artifacts/g1-flat.deploy \
  program.wbt_artifact=artifacts/g1-wbt-dance.deploy \
  runtime.transport.network_interface=enp5s0
```

**State switching (gamepad):**
- **LT + ↑**: Switch to fixstand
- **RB + X**: Switch to walk
- **RB + Y**: Switch to wbt
- **LT + B**: Enter passive mode

---

## Remote Inference Mode

Run policy inference on PC instead of robot onboard computer.

**Architecture:**
```
[PC: Read state → Read gamepad → Inference → Send commands] ←→ [Robot: Execute]
```

**Deploy from PC:**
```bash
# Connect gamepad to PC
# Ensure native SDK installed
pip install -e motrix_unitree_native

# Run deployment (inference on PC)
mxlab deploy task=g1-walk-flat/hardware \
  artifact=artifacts/<your_artifact>.deploy \
  runtime.transport.network_interface=enp5s0
```

**Network requirements:**
- PC and robot on same LAN
- Wired Ethernet connection (Gigabit)
- Latency < 5ms recommended

---

## Troubleshooting

### Export fails: "metadata not found"

**Solution:**
```bash
# Check for metadata
ls runs/g1-walk-flat/motrix/torch/<trainer>/<run-id>/metadata.json

# If missing, retrain or use complete run
```

### Cannot connect to robot

**Check:**
```bash
# Network interface
ip -br addr

# Ping robot
ping 192.168.123.10

# Test DDS communication
motrix-deploy-unitree read-lowstate enp5s0 --robot g1-29dof

# Firewall
sudo ufw allow from 192.168.123.0/24
```

### Robot not responding to gamepad

**Check:**
```bash
# Test gamepad input
jstest /dev/input/js0

# Increase velocity range
runtime.operator.planar_velocity.range_scale=[0.5,0.5,0.5]

# Reduce deadzone
runtime.operator.planar_velocity.deadzone=0.05

# Disable deadman button for testing
runtime.operator.planar_velocity.deadman_button=null
```

### Robot walks unstably

**Try:**
```bash
# Start with smaller velocities
runtime.operator.planar_velocity.range_scale=[0.1,0.1,0.1]

# Slower transition
task_settings.preparation.default_transition_duration_s=5.0

# Check PD parameters match training
cat artifacts/<your_artifact>.deploy/manifest.json | grep -A 30 '"kp"'
```

### Policy switch fails (hot-swapping)

**Check:**
1. Both artifacts are valid
```bash
mxlab deploy inspect artifact=artifacts/g1-flat.deploy
mxlab deploy inspect artifact=artifacts/g1-wbt-dance.deploy
```

2. Correct button combination
3. Robot in stable state before switching

---

## Key Controls Reference

### Startup Phase

| Action | Gamepad | Keyboard |
|--------|---------|----------|
| Confirm stand | Start | R |
| Enable policy | A | P |
| Cancel | - | O |
| Emergency stop | Select | Physical E-stop |

### Walking Control

| Action | Gamepad | Keyboard |
|--------|---------|----------|
| Forward | Left Stick Down | W |
| Backward | Left Stick Up | S |
| Strafe Left | Left Stick Right | A |
| Strafe Right | Left Stick Left | D |
| Turn Left | Right Stick Left | Q |
| Turn Right | Right Stick Right | E |

### Hot-Swapping

| Action | Gamepad Combination |
|--------|---------------------|
| Fixstand | LT + ↑ |
| Walk | RB + X |
| WBT | RB + Y |
| Passive | LT + B |

---

## Configuration Parameters

### Startup Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `default_transition_duration_s` | 3.0 | Stand-up transition time |
| `wait_for_remote_buttons` | true | Require manual confirmation |
| `kp` | null | Position gains (null = default) |
| `kd` | null | Velocity gains (null = default) |

### Velocity Command Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `deadzone` | 0.12 | Joystick deadzone |
| `range_scale` | [0.25, 0.25, 0.25] | Max velocity [x, y, yaw] |
| `deadman_button` | null | Button to enable movement |

---

## Common Commands

```bash
# Export artifact
mxlab export-deploy env=g1-walk-flat output=artifacts/g1-flat.deploy

# Inspect artifact
mxlab deploy inspect artifact=artifacts/g1-flat.deploy

# Sim validation (MuJoCo)
mxlab deploy task=g1-walk-flat/sim artifact=artifacts/g1-flat.deploy

# Sim validation (MotrixSim)
mxlab deploy task=g1-walk-flat/sim artifact=artifacts/g1-flat.deploy runtime.backend=motrixsim

# Hardware deployment
mxlab deploy task=g1-walk-flat/hardware \
  artifact=artifacts/g1-flat.deploy \
  runtime.transport.network_interface=enp5s0

# With slower velocity (safer)
mxlab deploy task=g1-walk-flat/hardware \
  artifact=artifacts/g1-flat.deploy \
  runtime.transport.network_interface=enp5s0 \
  runtime.operator.planar_velocity.range_scale=[0.15,0.15,0.15]

# Hot-swapping mode
mxlab deploy \
  program=g1-hot-swapping \
  runtime=g1-unitree-state-machine \
  program.walk_artifact=artifacts/g1-flat.deploy \
  program.wbt_artifact=artifacts/g1-wbt-dance.deploy \
  runtime.transport.network_interface=enp5s0
```

---

## Safety Reminders

- ✅ Complete sim validation before hardware deployment
- ✅ Use physical support and independent E-stop
- ✅ At least two operators present
- ✅ Test with small velocity commands first
- ✅ Ensure clear walking area with no obstacles
- ✅ Software E-stop cannot replace physical E-stop

---

## Reference Documentation

- [MotrixLab Training Tutorial](../../../tutorial/basic_frame.md)
- [Motrix Deploy Advanced Guide](../../../tutorial/advanced/motrix_deploy.md)
- [Unitree Deploy README](../../../../../../motrix_deploy_unitree/README.md)
- [WBT Deploy Guide](../whole_body_tracking/WBT_Deploy_detail.md)
