# Generic Humanoid Velocity-Tracking Environment

`HumanoidVelocityTrackingEnv` is MotrixLab's generic velocity-tracking environment for biped humanoid robots. It provides
one shared task for body-frame `[vx, vy, yaw_rate]` commands, action and observation construction, rewards, termination,
and reset without depending on a particular robot model or a fixed number of joints. A new robot only needs a
`HumanoidRobotCfg`, the task-required foot semantics, and a `HumanoidVelocityTrackingEnvCfg` config; the environment
implementation does not need to be copied.

## Uneven-terrain demo

The following videos show Unitree G1 and Booster K1 tracking velocity commands over procedural uneven terrain.

::::{grid} 1 1 2 2
:gutter: 2 2 2 2

:::{grid-item-card} Unitree G1

```{video} /_static/videos/g1-walk-rough.mp4
:poster: /_static/images/poster/g1-walk-rough.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

:::

:::{grid-item-card} Booster K1

```{video} /_static/videos/k1-walk-rough.mp4
:poster: /_static/images/poster/k1-walk-rough.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

:::

::::

```{toctree}
:hidden:
:maxdepth: 1

humanoid_velocity_tracking/adding_robot
humanoid_velocity_tracking/env_design
humanoid_velocity_tracking/config_tuning

```

## Built-in robots

MotrixLab provides flat-ground and procedural uneven-height-field configs for the following robots. Every environment ID
registers the same `HumanoidVelocityTrackingEnv`, but selects its own complete config. Click a training-curve thumbnail to
open the full-size SVG.

:::{div} task-table
| Environment ID | Robot | Terrain | Available training configs | Training curve |
| --- | --- | --- | --- | --- |
| `g1-walk-flat` | Unitree G1 29-DoF | Flat ground | `motrix.fastsac` | — |
| `g1-walk-rough` | Unitree G1 29-DoF | Procedural uneven height field | `motrix.fastsac` | <button type="button" class="training-curve-thumbnail" data-training-curve-dialog="g1-walk-rough-curve" aria-label="Enlarge the Unitree G1 uneven-terrain training curve"><img src="../../_static/images/performance/g1-walk-rough.svg" alt="Unitree G1 uneven-terrain FastSAC training curve" width="180"></button> |
| `dex-evt-walk-flat` | Dex-EVT | Flat ground | `motrix.fastsac` | — |
| `dex-evt-walk-rough` | Dex-EVT | Procedural uneven height field | `motrix.fastsac` | — |
| `k1-walk-flat` | Booster K1 | Flat ground | `motrix.fastsac` | — |
| `k1-walk-rough` | Booster K1 | Procedural uneven height field | `motrix.fastsac` | <button type="button" class="training-curve-thumbnail" data-training-curve-dialog="k1-walk-rough-curve" aria-label="Enlarge the Booster K1 uneven-terrain training curve"><img src="../../_static/images/performance/k1-walk-rough.svg" alt="Booster K1 uneven-terrain FastSAC training curve" width="180"></button> |
:::

<dialog id="g1-walk-rough-curve" class="training-curve-dialog" aria-labelledby="g1-walk-rough-curve-caption">
  <button type="button" class="training-curve-dialog-close" data-training-curve-close aria-label="Close training curve">×</button>
  <img src="../../_static/images/performance/g1-walk-rough.svg" alt="Unitree G1 uneven-terrain FastSAC training curve">
  <p id="g1-walk-rough-curve-caption">Unitree G1 (<code>g1-walk-rough</code>) training curve</p>
</dialog>

<dialog id="k1-walk-rough-curve" class="training-curve-dialog" aria-labelledby="k1-walk-rough-curve-caption">
  <button type="button" class="training-curve-dialog-close" data-training-curve-close aria-label="Close training curve">×</button>
  <img src="../../_static/images/performance/k1-walk-rough.svg" alt="Booster K1 uneven-terrain FastSAC training curve">
  <p id="k1-walk-rough-curve-caption">Booster K1 (<code>k1-walk-rough</code>) training curve</p>
</dialog>

The flat-ground and uneven-terrain configs for each robot share observation, action, reward, and termination logic; only the
scene terrain and spawn range differ. The curves in the table show asynchronous FastSAC training on uneven terrain: the left
axis reports mean episode return, and the right axis reports the penalty curriculum's `penalty_scale`. Returns rise rapidly
early in training and enter a stable regime in roughly 2–3 minutes. As `penalty_scale` increases, penalty terms receive more
weight and mean return may decrease; this is a change in reward scale rather than policy degradation.

## Commands

Choose an environment ID and one of its training configs from the table, then replace `ENV_ID` and `TRAINING_CONFIG`:

```bash
python scripts/view.py env=ENV_ID num_envs=1
python scripts/train.py task=ENV_ID/TRAINING_CONFIG
python scripts/play.py env=ENV_ID num_envs=16
```

For example, train the K1 uneven-terrain task with asynchronous FastSAC:

```bash
python scripts/train.py task=k1-walk-rough/motrix.fastsac algo.asynchronous=true
```

`view.py` applies random actions and is intended for scene and model inspection. Use `play.py` with a trained policy to
inspect the learned gait.
