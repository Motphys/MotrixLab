# Whole-Body Tracking Environments

`ManagerEnv` is MotrixLab's generic whole-body tracking (WBT) environment for humanoid robots. A policy follows a
frame-by-frame reference motion under physics simulation while the task compares the global reference-body pose, relative
poses of multiple body parts, body velocities, and joint feasibility. The `RobotCfg` and its assets own the robot model and
physical limits; `WbtManagerEnvCfg` selects the motion, tracked bodies, control scaling, rewards, and termination conditions. The
same environment implementation can therefore support different robots and motion clips.

## Demos

The following videos show Dex-EVT and Unitree G1 tracking dance motions, and Booster K1 tracking a free-kick motion.

::::{grid} 1 1 2 3
:gutter: 2 2 2 2

:::{grid-item-card} Dex-EVT dance

```{video} /_static/videos/dex-evt-wbt-dance.mp4
:alt: Sixteen Dex-EVT humanoid robots tracking a dance motion
:class: wbt-demo-video
:poster: /_static/images/poster/dex-evt-wbt-dance.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

:::

:::{grid-item-card} Unitree G1 dance

```{video} /_static/videos/g1-wbt-dance.mp4
:alt: Sixteen Unitree G1 humanoid robots tracking a dance motion
:class: wbt-demo-video
:poster: /_static/images/poster/g1-wbt-dance.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

:::

:::{grid-item-card} Booster K1 free kick

```{video} /_static/videos/k1-wbt-freekick.mp4
:alt: Sixteen Booster K1 humanoid robots tracking a free-kick motion
:class: wbt-demo-video
:poster: /_static/images/poster/k1-wbt-freekick.jpg
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

env_design
motion_format
adding_wbt_task

```

## Built-in tasks

All built-in tasks use the MotrixSim NumPy backend and provide one `motrix.fastsac` training config. Its `algo.asynchronous`
field selects synchronous or asynchronous execution without changing the algorithm identity. Each environment ID selects a
complete config, including the robot, motion, tracked bodies, rewards, and termination rules. Click a training-curve
thumbnail to open the full-size SVG.

:::{div} task-table
| Environment ID | Robot | Reference motion | Duration | Available training configs | Training curve |
| --- | --- | --- | ---: | --- | --- |
| `g1-29dof-wbt-largebox` | Unitree G1 29-DoF | `sub3_largebox_003.npz` | 6.50&nbsp;s | `motrix.fastsac` | — |
| `g1-wbt-dance` | Unitree G1 29-DoF | `dance1_subject2.npz` | 19.98&nbsp;s | `motrix.fastsac` | <button type="button" class="training-curve-thumbnail" data-training-curve-dialog="g1-wbt-dance-curve" aria-label="Enlarge the Unitree G1 dance WBT training curve"><img src="../../../_static/images/performance/g1-wbt-dance.svg" alt="Unitree G1 dance WBT training curve" width="180"></button> |
| `dex-evt-wbt-dance` | Dex-EVT | `dance1_easy.npz` | 39.72&nbsp;s | `motrix.fastsac` | <button type="button" class="training-curve-thumbnail" data-training-curve-dialog="dex-evt-wbt-dance-curve" aria-label="Enlarge the Dex-EVT dance WBT training curve"><img src="../../../_static/images/performance/dex-evt-wbt-dance.svg" alt="Dex-EVT dance WBT training curve" width="180"></button> |
| `k1-wbt-freekick` | Booster K1 | `freekick_shoot_arc_02.npz` | 2.50&nbsp;s | `motrix.fastsac` | <button type="button" class="training-curve-thumbnail" data-training-curve-dialog="k1-wbt-freekick-curve" aria-label="Enlarge the Booster K1 free-kick WBT training curve"><img src="../../../_static/images/performance/k1-wbt-freekick.svg" alt="Booster K1 free-kick WBT training curve" width="180"></button> |
:::

<dialog id="g1-wbt-dance-curve" class="training-curve-dialog" aria-labelledby="g1-wbt-dance-curve-caption">
  <button type="button" class="training-curve-dialog-close" data-training-curve-close aria-label="Close training curve">×</button>
  <img src="../../../_static/images/performance/g1-wbt-dance.svg" alt="Unitree G1 dance WBT training curve">
  <p id="g1-wbt-dance-curve-caption">Unitree G1 (<code>g1-wbt-dance</code>) training curve</p>
</dialog>

<dialog id="dex-evt-wbt-dance-curve" class="training-curve-dialog" aria-labelledby="dex-evt-wbt-dance-curve-caption">
  <button type="button" class="training-curve-dialog-close" data-training-curve-close aria-label="Close training curve">×</button>
  <img src="../../../_static/images/performance/dex-evt-wbt-dance.svg" alt="Dex-EVT dance WBT training curve">
  <p id="dex-evt-wbt-dance-curve-caption">Dex-EVT (<code>dex-evt-wbt-dance</code>) training curve</p>
</dialog>

<dialog id="k1-wbt-freekick-curve" class="training-curve-dialog" aria-labelledby="k1-wbt-freekick-curve-caption">
  <button type="button" class="training-curve-dialog-close" data-training-curve-close aria-label="Close training curve">×</button>
  <img src="../../../_static/images/performance/k1-wbt-freekick.svg" alt="Booster K1 free-kick WBT training curve">
  <p id="k1-wbt-freekick-curve-caption">Booster K1 (<code>k1-wbt-freekick</code>) training curve</p>
</dialog>

The curves in the table show asynchronous FastSAC training. The x-axis reports total environment steps with elapsed wall
time, and the y-axis reports mean episode return; some curves also show survival relative to the training time limit. Mean
return and survival rise rapidly early in training, reaching an effective tracking stage in roughly 6–7 minutes before
gradually stabilizing with continued training. This demonstrates fast iteration through parallel simulation. Because motion
duration, difficulty, and reward scale differ, each curve is best used to assess its own learning progress rather than for a
direct numerical comparison.

The training CLI does not replace a motion with `motion_file=...`; follow
[Adding a WBT Training Task](adding_wbt_task.md) to register a new environment ID for a custom motion.

## Run a built-in task

Motion files are managed by Git LFS. If an `.npz` is still pointer text after cloning, run `git lfs pull` first.

### Replay the motion kinematically

Check a reference motion with the target robot's `RobotCfg`:

```bash
python scripts/motion/replay.py \
  --robot g1-29dof \
  --motion motrix_envs/src/motrix_envs/locomotion/wbt/assets/motion/g1/dance1_subject2.npz
```

`replay.py` writes the motion's floating-root and joint state directly and runs forward kinematics. It does not execute a
policy, physical controller, reward, or WBT termination rule. Before training, verify a continuous root trajectory, correct
limb sides and joint directions, and playback without name-mapping or quaternion errors. See
[Motion File Format](motion_format.md) for fields and conversion workflows.

### Train

Choose an environment ID and one of its training configs from the built-in task table, then replace `ENV_ID` and
`TRAINING_CONFIG`:

```bash
python scripts/train.py task=ENV_ID/TRAINING_CONFIG
```

For example, train the G1 dance-tracking task with asynchronous FastSAC:

```bash
python scripts/train.py task=g1-wbt-dance/motrix.fastsac algo.asynchronous=true
```

The built-in configs use 2048 parallel environments and task-specific learning-iteration budgets. Use a small smoke test
only to validate registry creation, tensor shapes, and the training entry point; it is not expected to produce a useful policy:

```bash
python scripts/train.py task=g1-wbt-dance/motrix.fastsac \
  algo.asynchronous=true num_envs=64 algo.trainer.num_learning_iterations=100
```

During training, environments start at different points in the motion, and failure records increase the sampling probability
of difficult regions. Weighted reward terms are written to `info["Reward"]`; bad-tracking rates, motion progress, and
adaptive-sampling statistics are written to `info["metrics"]`. See [Task Environment Design](env_design.md) for definitions.

### Play the policy

```bash
python scripts/play.py env=ENV_ID num_envs=16
```

`play.py` selects the best policy from the latest metadata-backed run for the environment. The WBT play config starts at
motion frame 0, disables reset noise and adaptive sampling, and removes the 10-second training time limit. At the clip end it
restarts from frame 0. See [Training Artifacts: the runs Directory and Checkpoint Structure](../../tutorial/runs_and_checkpoints.md)
for run and checkpoint selection.
