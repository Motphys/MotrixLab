# motrix-deploy-unitree

`motrix-deploy-unitree` is the Unitree SDK2 DDS hardware plugin for `motrix-deploy`. It exposes the
`unitree_go2` backend through the `motrix_deploy.backends` entry-point group. SDK symbols remain lazily imported until a
physical backend is opened.

The SDK is installed from the pinned Motphys-maintained fork as part of the normal workspace sync:

```bash
sh install.sh --all
```

The Motphys fork includes the CRC shared libraries in built distributions. The SDK metadata declares
`cyclonedds`, `numpy`,
and `opencv-python`; uv therefore installs them into the environment as transitive dependencies. The DDS control backend does not import
OpenCV directly, but the official SDK includes camera/video clients in the same distribution.

Deployment recipes inherit common physical-runtime guards and default policy gains from
`configs/deploy/sim2real/base.yaml`. Concrete recipes such as
`configs/deploy/sim2real/go2_walk_flat_sim2real.yaml` own their robot-specific backend fields and command input configuration.
The SDK is imported only when `open()` is called. Runtime behavior:

- Subscribe to `rt/lowstate`, validate CRC, and map joint `q/dq` plus IMU data into canonical `RobotState`.
- Map canonical artifact joints to explicit Go2 motor indices and publish CRC-protected `rt/lowcmd` joint PD commands.
- Publish zero torque until Start, transition from measured pose to the artifact default pose, then hold until A.
- Latch Select as an emergency stop and report `exit_reason=emergency_stop`.
- Latch B as a lie-down request, interpolate to the configured shutdown pose, then enter damping and report
  `exit_reason=lie_down`. Select retains priority and skips the lie-down trajectory.
- Publish damping commands before closing both DDS channels on all stop paths.

## Direct read/write API

`UnitreeGo2DirectInterface` opens the same production backend without creating a policy runtime. Reading and writing are
separate method calls:

```python
import time

from motrix_deploy_unitree import UnitreeGo2DirectInterface

robot = UnitreeGo2DirectInterface.from_artifact(
    "artifacts/go2-walk-rough.deploy",
    network_interface="enp5s0",
    hardware_confirmed=True,
)

robot.open()
try:
    state = robot.read_data()
    print(state.joint_position, state.joint_velocity)

    # This still requires remote Start, default-pose transition, and A.
    robot.enable_command_output()

    hold = robot.default_pose_command()
    for _ in range(100):
        robot.send_command(hold)
        time.sleep(robot.control_period_s)
finally:
    robot.close()
```

`read_data()` returns canonical `RobotState`. `send_command()` accepts `RobotCommand`; alternatively,
`send_joint_command(position, joint_velocity=..., feedforward_torque=..., kp=..., kd=...)` constructs and sends one
command. Command output is rejected until `enable_command_output()` succeeds, and callers own periodic scheduling.

Opening this direct interface creates a LowCmd publisher and `close()` sends damping. For strictly read-only diagnostics
with no publisher, use `motrix-deploy-unitree read-lowstate <network-interface>`.

### Read LowState without commanding

Use the read-only diagnostic before enabling any command path:

```bash
motrix-deploy-unitree read-lowstate enp5s0
```

It subscribes to `rt/lowstate` for 10 seconds by default and prints at most one sample every 0.5 seconds. It does not need
an artifact or `--hardware-confirm`, and it never creates a `LowCmd` publisher. Each printed sample contains:

| Output           | Contents and units                                                                 |
| ---------------- | ---------------------------------------------------------------------------------- |
| Frame statistics | Received, valid, CRC-error, and decode-error counts                                |
| Joint state      | Canonical names for all 12 joints, position `q` in rad, and velocity `dq` in rad/s |
| IMU orientation  | Quaternion in both SDK `wxyz` and canonical `xyzw` order                           |
| IMU motion       | Gyroscope in rad/s and accelerometer in m/s²                                       |
| Wireless remote  | Pressed buttons and `lx`, `ly`, `rx`, `ry` axes                                    |

Use `--duration-s 0` to run until Ctrl+C and `--print-interval-s` to change output throttling. `--domain-id`, `--topic`, and
`--queue-depth` configure DDS. CRC validation is enabled by default; `--no-validate-crc` is intended only for focused
transport troubleshooting. The command returns `0` after receiving a valid frame, `1` when no valid frame arrives, and
`2` when the Unitree SDK cannot be imported.

### Bounded single-joint motion

The packaged helper requires a frozen deployment artifact as its robot, gain, timing, and limit contract:

```bash
motrix-deploy-unitree joint-control \
  enp5s0 \
  FL_thigh_joint \
  0.9 \
  --artifact artifacts/go2-walk-rough.deploy \
  --hardware-confirm
```

The three positional arguments are the network interface, canonical joint name, and absolute target position in radians.
The helper validates the artifact contract before opening DDS. It does not import task environments or select an
environment profile. After the normal Start/default-pose/A enable sequence, it interpolates to the target, holds it,
returns to the default pose, and sends a damping stop while closing. Every control tick reads a fresh `LowState`, so
state timeout and the remote Select emergency stop remain active.

Use `--move-duration`, `--hold-duration`, and `--return-duration` to change the default `2.0`, `1.0`, and `2.0` second
phases. Setting `--return-duration 0` skips the return trajectory, but closing still changes all joints to damping mode.
Run `motrix-deploy-unitree joint-control --help` for the complete command reference. Although the
helper changes one target, each Unitree `LowCmd` necessarily contains commands for all 12 joints.

## Policy gain overrides

`UnitreeGo2BackendConfig` defaults `backend.kp` and `backend.kd` to `null`, while
`configs/deploy/sim2real/base.yaml` sets them to `50` and `1` for physical deployment recipes. Each accepts either one
non-negative scalar for all joints or 12 values in canonical `FL, FR, RL, RR` order. Overrides apply to the
default-pose transition and policy commands; damping stop still uses `kp=0` and `backend.damping_kd`.

From the repository root:

```bash
motrix-deploy sim2real \
  artifact=artifacts/go2-walk-rough.deploy \
  backend.network_interface=enp5s0 \
  hardware.confirm=true
```

`hardware.confirm=true`, `viewer=false`, and `realtime=true` are enforced before DDS initialization. Before opening LowCmd, the backend uses MotionSwitcher to stand down and release the active MCF mode, then calls `RobotStateClient.ServiceSwitch("sport_mode", False)`; failure aborts startup. Use a suspended
robot, low-level/debug mode, and an operator-ready emergency stop. The implementation is validated with an injected fake
SDK; the suspended real-robot smoke test remains pending.
