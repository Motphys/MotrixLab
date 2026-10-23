# Motrix Deploy Unitree

`motrix-deploy-unitree` provides Unitree hardware backends and diagnostic tools for
`motrix-deploy`. Go2 uses the Python SDK2 DDS transport with independent hardware
and transport configurations. G1 hardware recipes select the native C++ SDK2
transport; Python SDK transport remains available for diagnostics. Go2 uses an independent
periodic publisher with a latest-command slot and state/command watchdog damping, rather
than performing DDS writes in the policy thread. The Python implementation is not hard real-time.

Both robots use `UnitreeRuntimeConfig`: select `robot: go2` or `robot: g1-29dof`,
then configure `control`, shared DDS `transport`, and an optional `operator` profile.
Both robots use `UnitreeControlConfig`; model defaults are loaded from
[`go2/control.yaml`](src/motrix_deploy_unitree/config/go2/control.yaml) and
[`g1/control.yaml`](src/motrix_deploy_unitree/config/g1/control.yaml).
Hydra hardware recipes select these defaults, and settings can be overridden with
`runtime.control.command_timeout_s=0.2`, for example. Python callers can use
`load_control_config("go2")` or `load_control_config("g1")` from
`motrix_deploy_unitree.config.control` to load the same typed configuration.
G1 hardware enable only permits command output; task controllers own standing
preparation and operator confirmation in the regular control loop. Single-task
startup settings live under `task_settings.preparation` (for example,
`task_settings.preparation.default_transition_duration_s=3.0`). Startup `kp`/`kd`
default to the shared FixStand gains and can be overridden there, not under
`runtime.control`. Persistent programs retain their own state-machine preparation.
Each backend owns its fixed motor wiring, protocol modes, command limits and reference
poses; these hardware definitions are not runtime configuration inputs. Select the
SDK implementation with `transport.implementation`. Configuration definitions live
under `motrix_deploy_unitree.config`. The native extension is loaded only when a native
hardware connection is opened. See [native transport installation](../motrix_unitree_native/README.md).

## Safety

Use reliable support and an operator-ready independent emergency stop. Hardware
configuration and command limits do not certify mechanical safety.

Go2 hardware deployment integrates an operator-confirmed official-controller handoff
before low-level control; standalone diagnostics require explicit `release-control`.
Robot opening checks that official control is released and establishes low-level
communication; it does not perform the handoff. Handoff can move the robot or remove
balance support and never automatically restores official control. This flow is Go2-specific.

Go2 diagnostics use Start for the hold-pose transition and A to start the selected
activity. Select triggers backend emergency damping. B follows the task's normal
stop behavior; it does not implicitly lie down. Even single-joint diagnostics
command all twelve joints. G1 uses its own task preparation and operator flow;
do not apply the Go2 release procedure to G1.

## Usage

Check incoming state without creating a command publisher:

```bash
motrix-deploy-unitree read-lowstate <network-interface>
motrix-deploy-unitree read-lowstate <network-interface> --robot g1-29dof
```

With Go2 safely supported, explicitly release official control. Type `yes` before
connecting. If StandDown is requested, observe that movement has settled and the
body is supported, then type `down` before release. Cancelling the second prompt
does not undo StandDown:

```bash
motrix-deploy-unitree release-control <network-interface> --domain-id 0 --timeout-s 5
```

Run Go2 joint diagnostics without an artifact or task plugin. Supply reviewed
gains for your supported robot:

```bash
motrix-deploy-unitree joint-control <network-interface> FL_thigh_joint \
  --offset 0.05 --kp "${KP:?Set reviewed stiffness}" --kd "${KD:?Set reviewed damping}"
```

`--offset` is relative to the named hardware hold pose; `--target` specifies an
absolute angle. The command displays the full-body plan before asking for
confirmation. The diagnostic controller owns the bounded preparation from the measured
pose. Startup poses and command targets must stay within model and hardware position
bounds. This does not certify mechanical safety. See the
[deployment tutorial](../docs/source/en/user_guide/tutorial/advanced/motrix_deploy.md)
for safeguards and timing options.

With `motrix-deploy-tasks` and a matching artifact installed:

```bash
mxlab deploy task=go2-walk-flat/hardware artifact=artifacts/go2-walk-flat.deploy \
  runtime.transport.network_interface=<network-interface>

# Install the native transport before selecting a G1 hardware recipe.
mxlab deploy task=g1-wbt-dance/hardware \
  runtime.transport.network_interface=<network-interface>
```

Review task-specific preparation and safety procedures before running. Use
`motrix-deploy-unitree --help` for diagnostic and joint-control tools.
