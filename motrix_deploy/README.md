# Motrix Deploy

`motrix_deploy` runs exported policies without a training framework. It provides deployment artifacts, robot and task
contracts, `ControlSession`, simulation/hardware runtimes, and the `motrix-deploy` CLI. Install a task plugin and the
runtime plugin for your target; the Go2 recipes below come from `motrix_deploy_tasks` and use `motrix_deploy_mujoco`.

From the repository root, install and activate the environment, then export a trained Go2 flat-terrain run:

```bash
sh install.sh --all
source .venv/bin/activate
python scripts/export_deploy.py env=go2-walk-flat
motrix-deploy inspect artifact=artifacts/go2-walk-flat.deploy
motrix-deploy task=go2-walk-flat/sim artifact=artifacts/go2-walk-flat.deploy runtime.viewer=false duration_s=2.0
```

Export selects the latest metadata-backed run. Use `run=<run-dir> output=<new-artifact-dir>` to select a specific run and
output. Export creates a new directory; it does not overwrite an existing artifact.

The artifact contains the ONNX policy, robot/control contract, and serialized task specification. The installed task
plugin supplies the matching implementation and Hydra recipes, so deployment does not require workspace config files.
Task defaults select the Hydra runtime group (`sim` or `hardware`). All target-specific settings live directly under
`runtime`: `runtime.kind` selects the simulation or hardware path, while `runtime.backend` selects the plugin (`mujoco`
for simulation or `unitree_go2` for hardware). `runtime.backend` is a scalar string selector, not a nested configuration
mapping or Hydra group. The simulation recipe selects the scene and robot
(`runtime.deploy_env_id=flat`, `runtime.robot_id=go2`).

Omit `runtime.viewer=false duration_s=2.0` to open the MuJoCo viewer. Hold `W/S`, `A/D`, and `Q/E` for forward, lateral, and yaw
commands; Esc, Ctrl-C, or closing the window stops the run. Headless operation uses the recipe's constant zero velocity;
override `command.velocity` to change it. Common settings are `artifact`, `duration_s`, and `command`.
`duration_s` is a control-time budget of `ceil(duration_s / control_period_s)` ticks, not a wall-clock timer.
Simulation-only settings are `runtime.viewer` and `runtime.realtime`; `runtime.realtime=null` follows
`runtime.viewer`. Hardware runs are always real-time internally and expose neither simulation field.

## Programmatic control

[`examples/deploy_to_sim.py`](../examples/deploy_to_sim.py) assembles a `SceneCfg`, `SimulationRuntimeConfig`, walking task,
ONNX policy, and command binding, then binds a `ControlSession` to a MuJoCo runtime. It includes a sample policy:

```bash
python examples/deploy_to_sim.py --headless --steps 100
```

`ControlSession` reads input and state, builds observations, runs inference, and writes robot commands. The runtime owns
physics advancement, scheduling, and the viewer. A raw ONNX model needs explicit task preprocessing and action settings;
use an exported artifact when those settings should travel with the policy.

For the training-to-hardware workflow and safety checklist, see the
[deployment tutorial](../docs/source/en/user_guide/tutorial/advanced/motrix_deploy.md) and
[Unitree plugin guide](../motrix_deploy_unitree/README.md).
