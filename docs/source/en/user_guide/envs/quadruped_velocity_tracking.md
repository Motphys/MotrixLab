# Generic Quadruped Velocity-Tracking Environment

`QuadrupedWalkTask` is MotrixLab's generic velocity-tracking environment for quadruped robots. It provides one shared
task for body-frame `[vx, vy, yaw_rate]` commands, a diagonal-trot reference, action and observation construction,
rewards, termination, and reset. The implementation is independent of a concrete robot model and assumes no fixed joint
count. A new robot only needs a `QuadrupedRobotCfg`, the task-required sensors, and a `QuadrupedWalkEnvCfg` config;
the environment implementation does not need to be copied.

## Rough-terrain demos

The following videos replay the `rslrl.ppo` policies for Go1, Go2, and ANYmal-C. Each video shows 16 parallel environments.

::::{grid} 1 1 3 3
:gutter: 2 2 2 2

:::{grid-item-card} Unitree Go1

```{video} /_static/videos/go1-walk-rough.mp4
:poster: /_static/images/poster/go1-walk-rough.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

:::

:::{grid-item-card} Unitree Go2

```{video} /_static/videos/go2-walk-rough.mp4
:poster: /_static/images/poster/go2-walk-rough.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

:::

:::{grid-item-card} ANYmal-C

```{video} /_static/videos/anymalc-walk-rough.mp4
:poster: /_static/images/poster/anymalc-walk-rough.jpg
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

quadruped_velocity_tracking/adding_robot
quadruped_velocity_tracking/env_design
quadruped_velocity_tracking/config_tuning

```

## Built-in robots

MotrixLab provides flat-ground and procedural rough-height-field configs for the following robots. Every environment ID
registers the same `QuadrupedWalkTask`, but selects its own complete configuration. Click a training-curve thumbnail to
open the full-size SVG.

:::{div} task-table
| Environment ID | Robot | Terrain | Available training configs | Training curve |
| --- | --- | --- | --- | --- |
| `go1-walk-flat` | Unitree Go1 | Flat ground | `rslrl.ppo`, `skrl.ppo` | — |
| `go1-walk-rough` | Unitree Go1 | Procedural rough height field | `rslrl.ppo`, `skrl.ppo` | <button type="button" class="training-curve-thumbnail" data-training-curve-dialog="go1-walk-rough-curve" aria-label="Enlarge the Unitree Go1 rough-terrain training curve"><img src="../../_static/images/performance/go1-walk-rough.svg" alt="Unitree Go1 rough-terrain RSL-RL PPO training curve" width="180"></button> |
| `go2-walk-flat` | Unitree Go2 | Flat ground | `motrix.fastsac`, `rslrl.ppo`, `skrl.ppo` | — |
| `go2-walk-rough` | Unitree Go2 | Procedural rough height field | `motrix.fastsac`, `rslrl.ppo`, `skrl.ppo` | <button type="button" class="training-curve-thumbnail" data-training-curve-dialog="go2-walk-rough-curve" aria-label="Enlarge the Unitree Go2 rough-terrain training curve"><img src="../../_static/images/performance/go2-walk-rough.svg" alt="Unitree Go2 rough-terrain RSL-RL PPO training curve" width="180"></button> |
| `anymalc-walk-flat` | ANYmal-C | Flat ground | `rslrl.ppo`, `skrl.ppo` | — |
| `anymalc-walk-rough` | ANYmal-C | Procedural rough height field | `rslrl.ppo`, `skrl.ppo` | <button type="button" class="training-curve-thumbnail" data-training-curve-dialog="anymalc-walk-rough-curve" aria-label="Enlarge the ANYmal-C rough-terrain training curve"><img src="../../_static/images/performance/anymalc-walk-rough.svg" alt="ANYmal-C rough-terrain RSL-RL PPO training curve" width="180"></button> |
:::

<dialog id="go1-walk-rough-curve" class="training-curve-dialog" aria-labelledby="go1-walk-rough-curve-caption">
  <button type="button" class="training-curve-dialog-close" data-training-curve-close aria-label="Close training curve">×</button>
  <img src="../../_static/images/performance/go1-walk-rough.svg" alt="Unitree Go1 rough-terrain RSL-RL PPO training curve">
  <p id="go1-walk-rough-curve-caption">Unitree Go1 (<code>go1-walk-rough</code>) training curve</p>
</dialog>

<dialog id="go2-walk-rough-curve" class="training-curve-dialog" aria-labelledby="go2-walk-rough-curve-caption">
  <button type="button" class="training-curve-dialog-close" data-training-curve-close aria-label="Close training curve">×</button>
  <img src="../../_static/images/performance/go2-walk-rough.svg" alt="Unitree Go2 rough-terrain RSL-RL PPO training curve">
  <p id="go2-walk-rough-curve-caption">Unitree Go2 (<code>go2-walk-rough</code>) training curve</p>
</dialog>

<dialog id="anymalc-walk-rough-curve" class="training-curve-dialog" aria-labelledby="anymalc-walk-rough-curve-caption">
  <button type="button" class="training-curve-dialog-close" data-training-curve-close aria-label="Close training curve">×</button>
  <img src="../../_static/images/performance/anymalc-walk-rough.svg" alt="ANYmal-C rough-terrain RSL-RL PPO training curve">
  <p id="anymalc-walk-rough-curve-caption">ANYmal-C (<code>anymalc-walk-rough</code>) training curve</p>
</dialog>

The flat-ground and rough-terrain configs for each robot share the robot, control, commands, rewards, spawn range, and
termination logic; only the scene terrain differs. The curves in the table show rough-terrain training progress: the x-axis
reports cumulative environment steps, and the y-axis reports mean episode return, making learning progress and convergence
trends easy to inspect. Returns rise rapidly early in training and then stabilize, showing that the policies quickly learn
sustained rough-terrain locomotion. Under the current training configs and hardware, the tasks shown here converge in roughly
1–2 minutes, demonstrating the high throughput of the parallel simulation and training pipeline.

## Commands

Choose an environment ID and one of its training configs from the table, then replace `ENV_ID` and `TRAINING_CONFIG`:

```bash
python scripts/view.py env=ENV_ID num_envs=1
python scripts/train.py task=ENV_ID/TRAINING_CONFIG
python scripts/play.py env=ENV_ID num_envs=16
```

For example, train the Go2 rough-terrain task with RSL-RL PPO:

```bash
python scripts/train.py task=go2-walk-rough/rslrl.ppo
```

`view.py` applies random actions and is intended for scene and model inspection. Use `play.py` with a trained policy to
inspect the learned gait.
