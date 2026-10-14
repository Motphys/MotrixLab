# Motrix Deploy Tasks

`motrix-deploy-tasks` supplies the versioned deployment tasks and installed Hydra recipes
for `motrix-deploy`: the Go2 walking task with flat/rough worlds, and the G1 WBT dance
task. It does not provide a simulation or hardware backend.

Install a backend plugin and use an artifact exported for the matching task:

```bash
motrix-deploy task=go2-walk-flat/sim artifact=artifacts/go2-walk-flat.deploy
```

Use `go2-walk-rough/sim` for the rough scene, `runtime.backend=motrixsim` for native
MotrixSim, or `go2-walk-flat/hardware` for Unitree hardware. Recipes are packaged with the
plugin; no workspace config path is required.

See the [deployment tutorial](../docs/source/en/user_guide/tutorial/advanced/motrix_deploy.md)
for export, configuration, and hardware safety procedures.

## G1 WBT dance: MuJoCo / MotrixSim sim2sim

Use the standard **export → artifact → `motrix-deploy`** pipeline. The `g1_wbt/v1` task
embeds the actor's motion references, servo gains, action scales, preparation gains and
timing, and hardware-observable termination thresholds in the artifact. Deployment
requires neither Torch nor an RL environment; export needs the training provider and task
environment packages, but does **not** train.

```bash
source .venv/bin/activate
RUN=runs/g1-wbt-dance/motrix/torch/fastsac/<run-dir>
python scripts/export_deploy.py run="$RUN" \
  output=artifacts/g1-wbt-dance-software-pd.deploy validation.atol=3e-5

motrix-deploy task=g1-wbt-dance/sim \
  artifact=artifacts/g1-wbt-dance-software-pd.deploy
```

Headless validation and artifact inspection:

```bash
motrix-deploy task=g1-wbt-dance/sim \
  artifact=artifacts/g1-wbt-dance-software-pd.deploy runtime.viewer=false
motrix-deploy inspect artifact=artifacts/g1-wbt-dance-software-pd.deploy
```

Notes:

- The default duration comes from the embedded single clip; shorter runs use `duration_s`,
  longer durations are rejected before opening the backend instead of looping or holding.
- No velocity command is needed: WBT motion is artifact-owned.
- The same artifact runs on MuJoCo and MotrixSim (`runtime.backend=motrixsim`, with
  `runtime.physics.solver_iterations=3` matching the training budget). It is not a
  hardware deployment recipe: the current Unitree transport is Go2-specific, and sim2sim
  success is not evidence of hardware safety.
- The recipe's 0.002 s physics step (10 substeps per 0.02 s control tick) keeps the stiff
  preparation PD inside MuJoCo's explicit-damping stability region.

The task runs a four-phase lifecycle mirroring real deployment practice — `prepare` (ramp
measured joints to the robot's default standing pose with artifact-embedded gains and
timing, gated on continuously measured readiness), `policy_hold`, `playback`, `damping`.
The dance ends in a deliberately leaned pose that only the tracking policy can balance;
after playback the task enters measured damping directly. Ending upright requires the
motion clip itself to finish at the standing pose.
Takeover after measured readiness is automatic by default; programmatic applications can
stage it manually via `automatic_start=False`, `attach_keyboard`, or the request methods.
The CLI routes no task-specific options — operator input and preparation timing are task
semantics embedded in the artifact or set programmatically. See the
[deployment tutorial](../docs/source/en/user_guide/tutorial/advanced/motrix_deploy.md)
for the complete runtime, session, and controller contracts.

```bash
python -m pytest motrix_deploy_tasks/tests -q
```
