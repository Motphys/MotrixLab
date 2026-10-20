# Motrix Unitree Native

This package builds the pybind11 extension used by the G1 `native_cpp` hardware
transport. It owns Unitree SDK2 DDS subscription/publication, LowState CRC and
mode validation, and the 1 kHz LowCmd watchdog writer. Python receives normalized
state snapshots and sends the latest policy command at policy-loop frequency.

Build on a machine with CMake, pybind11, and Unitree SDK2 installed:

```bash
UNITREE_SDK2_ROOT=/opt/unitree_robotics \
  python -m pip install ./motrix_unitree_native
```

The build embeds the SDK2 third-party library directory as a `DT_RPATH` entry,
which the dynamic loader searches **before** `LD_LIBRARY_PATH`. That matters on
robots whose shell sources ROS: `/opt/ros/foxy/lib/*/libddsc.so.0` is CycloneDDS
0.7 and does not export the 0.10 `ddsi_sertype_v0` API, so a `DT_RUNPATH` build
would bind to 0.7 and fail with `undefined symbol: ddsi_sertype_v0`.

If the SDK is moved after installation, export its architecture-specific
library directory before starting deployment, for example:

```bash
export LD_LIBRARY_PATH=/opt/unitree_robotics/thirdparty/lib/x86_64:$LD_LIBRARY_PATH
```

`G1Backend` owns the whole low-level contract: it converts the IMU quaternion to
xyzw and normalizes it, maps canonical joints to motor slots, validates the
LowState CRC and the `mode_pr`/`mode_machine` contract, publishes at
`command_period_s` with the CRC computed outside the writer lock, and damps
(kp=0, kd=`damping_kd`) whenever the state, the command or the mode contract is
stale. Mode/CRC problems are recoverable: the backend damps and reports
unhealthy, then resumes once valid frames return.

For a source checkout of SDK2, set `UNITREE_SDK2_ROOT` to that checkout instead.
The native transport is G1 29-DOF specific; Go2 continues to use the existing
Python SDK transport.
