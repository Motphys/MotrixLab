# G1 Whole-Body Tracking (WBT): Complete Deployment Guide

This guide covers deploying trained WBT policies from simulation to real hardware.

---

## Quick Start

### Prerequisites

- Completed G1 WBT training (e.g., `g1-wbt-dance`)
- Training run with metadata in `runs/` directory
- MotrixLab deployment packages installed

### Step 1: Export Artifact

```bash
cd /path/to/MotrixLab
source .venv/bin/activate

# Export from latest training run
mxlab export-deploy env=g1-wbt-dance output=artifacts/<your_artifact_name>.deploy
```

### Step 2: Validate in Simulation

```bash
# Test in MuJoCo simulation first
mxlab deploy task=g1-wbt-dance/sim artifact=artifacts/<your_artifact_name>.deploy
```

**Controls:**
- **P key**: Start WBT policy execution
- **O key**: Cancel startup
- **Esc/Ctrl+C**: Exit

✅ **Only proceed to hardware if simulation succeeds!**

### Step 3: Prepare Hardware

**Safety checklist:**
- [ ] Robot on support frame or sling
- [ ] Emergency stop button ready
- [ ] Two operators present
- [ ] Clear movement space

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
mxlab deploy task=g1-wbt-dance/hardware \
  artifact=artifacts/<your_artifact_name>.deploy \
  runtime.transport.network_interface=enp5s0
```

**Startup sequence:**
1. Program connects to robot
2. **Press Start button** (gamepad) or **R key** → Robot stands up
3. **Press A button** (gamepad) or **P key** → WBT policy starts
4. Robot executes motion sequence (~20 seconds)

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
mxlab deploy task=g1-wbt-dance/hardware \
  artifact=artifacts/<your_artifact>.deploy \
  runtime.transport.network_interface=enp5s0 \
  task_settings.preparation.default_transition_duration_s=5.0
```

**Custom startup gains:**
```bash
task_settings.preparation.kp=80.0 \
task_settings.preparation.kd=3.0
```

**Skip manual confirmation (use after testing):**
```bash
task_settings.preparation.wait_for_remote_buttons=false
```

### Configuration File

Edit `/path/to/MotrixLab/motrix_deploy_tasks/src/motrix_deploy_tasks/config/task/g1-wbt-dance/hardware.yaml`:

```yaml
runtime:
  transport:
    network_interface: enp5s0  # Your interface
    domain_id: 0

task_settings:
  preparation:
    default_transition_duration_s: 3.0  # Transition time
    wait_for_remote_buttons: true       # Manual confirmation
    kp: null  # null uses default gains
    kd: null
```

---

## Remote Inference Mode

Run policy inference on PC instead of robot onboard computer.

**Architecture:**
```
[PC: Read state → Inference → Send commands] ←→ [Robot: Execute]
```

**Deploy from PC:**
```bash
# Ensure native SDK installed
pip install -e motrix_unitree_native

# Run deployment (inference on PC)
mxlab deploy task=g1-wbt-dance/hardware \
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

**Problem:** Training run missing metadata.json

**Solution:**
```bash
# Check for metadata
ls runs/g1-wbt-dance/motrix/torch/fastsac/<run-id>/metadata.json

# If missing, retrain or use complete run
```

### Cannot connect to robot

**Check:**
```bash
# 1. Network interface name
ip -br addr

# 2. Ping robot
ping 192.168.123.10

# 3. Test DDS communication
motrix-deploy-unitree read-lowstate enp5s0 --robot g1-29dof

# 4. Firewall
sudo ufw allow from 192.168.123.0/24
```

### Native SDK import error

```bash
# Install native transport
cd /path/to/MotrixLab
pip install -e motrix_unitree_native

# Install dependencies if build fails
sudo apt-get install build-essential cmake
```

### Robot shakes during startup

**Try:**
```bash
# Reduce gains
task_settings.preparation.kp=60.0
task_settings.preparation.kd=2.5

# Extend transition time
task_settings.preparation.default_transition_duration_s=5.0
```

### Policy tracking unstable

**Check:**
1. PD parameters match training config
```bash
cat artifacts/<your_artifact>.deploy/manifest.json | grep -A 30 '"kp"'
```

2. Ground conditions similar to training
3. Network latency acceptable
```bash
ping -c 100 -i 0.02 192.168.123.10
```

---

## Key Controls Reference

### Startup Phase

| Action | Gamepad | Keyboard |
|--------|---------|----------|
| Confirm stand | Start | R |
| Enable policy | A | P |
| Cancel | - | O |
| Emergency stop | Select | Physical E-stop |

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `default_transition_duration_s` | 3.0 | Stand-up transition time |
| `wait_for_remote_buttons` | true | Require manual confirmation |
| `kp` | null | Position gains (null = default) |
| `kd` | null | Velocity gains (null = default) |

### Network Parameters

| Parameter | Description |
|-----------|-------------|
| `network_interface` | Ethernet interface (e.g., enp5s0) |
| `domain_id` | DDS domain ID (default: 0) |
| `implementation` | SDK type (G1 uses native_cpp) |

---

## Common Commands

```bash
# Export artifact
mxlab export-deploy env=g1-wbt-dance output=artifacts/g1-wbt.deploy

# Inspect artifact
mxlab deploy inspect artifact=artifacts/g1-wbt.deploy

# Sim validation (MuJoCo)
mxlab deploy task=g1-wbt-dance/sim artifact=artifacts/g1-wbt.deploy

# Sim validation (MotrixSim)
mxlab deploy task=g1-wbt-dance/sim artifact=artifacts/g1-wbt.deploy runtime.backend=motrixsim

# Hardware deployment
mxlab deploy task=g1-wbt-dance/hardware \
  artifact=artifacts/g1-wbt.deploy \
  runtime.transport.network_interface=enp5s0

# With slower transition (safer)
mxlab deploy task=g1-wbt-dance/hardware \
  artifact=artifacts/g1-wbt.deploy \
  runtime.transport.network_interface=enp5s0 \
  task_settings.preparation.default_transition_duration_s=5.0
```

---

## Safety Reminders

- ✅ Complete sim validation before hardware deployment
- ✅ Use physical support and independent E-stop
- ✅ At least two operators present
- ✅ WBT may produce large movements - ensure clear space
- ✅ Software E-stop cannot replace physical E-stop

---

## Reference Documentation

- [MotrixLab Training Tutorial](../../../tutorial/basic_frame.md)
- [Motrix Deploy Advanced Guide](../../../tutorial/advanced/motrix_deploy.md)
- [WBT Task Addition Guide](./adding_wbt_task.md)
- [Unitree Deploy README](../../../../../../motrix_deploy_unitree/README.md)
