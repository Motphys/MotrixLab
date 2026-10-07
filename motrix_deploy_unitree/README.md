# Motrix Deploy Unitree

`motrix-deploy-unitree` provides the `unitree_go2` SDK2 DDS hardware backend for `motrix-deploy`,
asset-independent hardware sensor/motor configuration, and direct read/write and diagnostic tools.
The Unitree SDK is imported only when a hardware connection is opened.

## Safety

Use a suspended robot in low-level/debug mode with an operator-ready emergency stop.
Opening a command interface may stand down the robot before the Start gate. Choose hardware-appropriate
startup gains: Start transitions to the default pose, A enables control, Select triggers emergency stop,
and B requests lie-down. Stop paths send damping commands. Even the single-joint helper publishes
commands for all 12 joints. The backend has fake-SDK test coverage; a real-robot smoke test remains pending.

## Usage

Check incoming state without creating a command publisher:

```bash
motrix-deploy-unitree read-lowstate <network-interface>
```

With `motrix-deploy-tasks` and a matching artifact installed:

```bash
motrix-deploy task=go2-walk-flat/hardware artifact=artifacts/go2-walk-flat.deploy \
  runtime.network_interface=<network-interface>
```

Review configured startup gains and the
[deployment tutorial's safety procedures](../docs/source/en/user_guide/tutorial/advanced/motrix_deploy.md)
before running. Use `motrix-deploy-unitree --help` for diagnostic and joint-control tools.
