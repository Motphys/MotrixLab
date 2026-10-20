# Motrix Deploy Unitree

`motrix-deploy-unitree` provides the `unitree` SDK2 DDS hardware backend for `motrix-deploy`,
asset-independent hardware sensor/motor configuration, and direct read/write and diagnostic tools.
G1 hardware recipes select the native C++ SDK2 transport (`transport: native_cpp`), while the
Python SDK transport remains available for compatibility diagnostics.
The entry-point `create_runtime(config, context)` accepts a `UnitreeBackendConfig` with a nested
`UnitreeGo2RobotCfg` or `UnitreeG1_29DOFRobotCfg` in `config.robot`, and returns a `HardwareRuntime`. Robot configurations inherit
`HardwareRobotCfg` through `UnitreeRobotCfg`, exposing `key_pose`, `sensors`, and `actuation` directly
alongside robot-specific control settings. G1 uses HG LowCmd/LowState, validates CRC plus
`mode_pr=0` and `mode_machine=5`, maps artifact joints to motors by name, and repeats the latest
policy command from a native C++ 1 kHz writer. Install the optional `native` extra to build the
`motrix-unitree-native` extension against Unitree SDK2; the extension is loaded only when a native
hardware connection is opened.

## Safety

Use a suspended robot in low-level/debug mode with an operator-ready emergency stop.
Opening a command interface may stand down the robot before the Start gate. Choose hardware-appropriate
startup gains: Start transitions to the default pose, A enables control, Select triggers emergency stop,
and B requests lie-down. Stop paths send damping commands. Even the single-joint helper publishes
commands for every configured joint. G1 unused HG slots remain disabled and zeroed. The backend has
fake-SDK test coverage; a suspended real-robot smoke test remains pending.

## Usage

Check incoming state without creating a command publisher:

```bash
motrix-deploy-unitree read-lowstate <network-interface>
motrix-deploy-unitree read-lowstate <network-interface> --robot g1-29dof
```

With `motrix-deploy-tasks` and a matching artifact installed:

Install the native transport extra before selecting a G1 hardware recipe:

```bash
UNITREE_SDK2_ROOT=/opt/unitree_robotics uv sync --extra unitree
```

When installing the package outside this workspace, use the equivalent
`UNITREE_SDK2_ROOT=/opt/unitree_robotics python -m pip install
"motrix-deploy-unitree[native]"` command. The SDK2 root must contain `include/`,
`lib/<architecture>/`, and `thirdparty/lib/<architecture>/`.

```bash
mxlab deploy task=go2-walk-flat/hardware artifact=artifacts/go2-walk-flat.deploy \
  runtime.network_interface=<network-interface>

mxlab deploy task=g1-wbt-dance/hardware \
  runtime.network_interface=<network-interface>
```

Review configured startup gains and the
[deployment tutorial's safety procedures](../docs/source/en/user_guide/tutorial/advanced/motrix_deploy.md)
before running. Use `motrix-deploy-unitree --help` for diagnostic and joint-control tools.
