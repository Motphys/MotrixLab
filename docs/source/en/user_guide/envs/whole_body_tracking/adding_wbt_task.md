# Adding a WBT Training Task

The reusable unit of `ManagerEnv` is a complete `WbtManagerEnvCfg`. Adding a motion for an existing robot normally requires a motion
file, an environment-config factory, Env registration, and matching Hydra Training Tasks. It does not require a copy of the
environment implementation. This chapter adds G1 motion `dance1_subject1.npz` as Env ID
`g1-wbt-dance1-subject1`.

## 1. Define the complete environment config

Start from the existing WBT config subclass for the target robot. Edit
`motrix_envs/src/motrix_envs/locomotion/wbt/g1.py`:

```python
from pathlib import Path

from motrix_env_core import registry
from motrix_env_core.manager import ManagerEnv

from motrix_envs.locomotion.wbt.g1 import G1WbtManagerCfg


_MOTION_DIR = Path(__file__).parent / "assets" / "motion" / "g1"


@registry.envcfg("g1-wbt-dance1-subject1")
def make_g129dof_wbt_dance1_subject1_cfg() -> G1WbtManagerCfg:
    return G1WbtManagerCfg(motion_file=str(_MOTION_DIR / "dance1_subject1.npz"))
```

`G1WbtManagerCfg` inherits `WbtManagerEnvCfg` and provides the G1 scene, tracked bodies, reference body, control scaling,
rewards, and termination rules. When a new motion uses the same robot and tracking semantics, pass a different `motion_file`
directly to the constructor. Do not copy `ManagerEnv` for each clip.

## 2. Prepare and replay the motion

Place a file that follows the [Motion File Format](motion_format.md) at the path referenced by the factory:

```text
motrix_envs/src/motrix_envs/locomotion/wbt/assets/motion/g1/dance1_subject1.npz
```

Built-in WBT uses `ctrl_dt=0.02` s, so convert this motion to 50 FPS. Run kinematic replay before registering the training
entry point:

```bash
python scripts/motion/replay.py \
  --robot g1-29dof \
  --motion motrix_envs/src/motrix_envs/locomotion/wbt/assets/motion/g1/dance1_subject1.npz
```

Continue to physical training only after the root trajectory, joint directions, limb mapping, and complete clip are correct.
Replay does not use WBT rewards or termination, so it separates data problems from control problems.

## 3. Confirm configuration ownership

A new motion on the same robot usually reuses the existing config subclass. Add or adjust a robot-specific config subclass
only when these semantics change:

| Config                                                     | Responsibility                                                                                        |
| ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `scene.objs.robot`                                         | Robot assets, default key pose, base link, joint/actuator position ranges, and physical `force_range` |
| `motion_file`                                              | MotrixLab Motion NPZ path                                                                             |
| `tracked_body_names`                                       | Links used by relative body-pose and velocity rewards                                                 |
| `reference_body_name`                                      | Link used for global reference pose and local alignment                                               |
| `control_config.action_scale`                              | Base scale from policy action to position target                                                      |
| `control_config.action_scales_by_effort_limit_over_p_gain` | Whether to derive each position-target scale from the actuator `force_range` and `kp`                 |
| `reward_config`                                            | Reward scales, error-kernel widths, soft joint range, and allowed contact links                       |
| `termination_config`                                       | Reference-height/orientation, key-body height, hard joint-range, and velocity thresholds              |
| `observation_noise`, `reset_noise`                         | Actor observation noise and training initial-state noise                                              |
| `adaptive_timestep_sampler`                                | Start-frame curriculum based on failed motion regions                                                 |
| `diagnostics`                                              | Optional robot-specific body and actuator logs                                                        |

Physical actuator force limits belong to RobotCfg/MJCF and are not duplicated in WBT config. When
`action_scales_by_effort_limit_over_p_gain` is enabled, every position actuator must define `force_range`; WBT derives its
effort as the largest absolute endpoint and uses it for position-target scaling. When disabled, WBT uses the scalar
`action_scale` without reading the force range.

Existing robot config classes provide starting points:

-   G1: `G1WbtManagerCfg(motion_file=...)`
-   Dex-EVT: `DexEvtWbtManagerCfg()`
-   K1: `K1WbtManagerCfg(commands=_k1_commands(...), rewards=...)`

## 4. Bind the shared environment implementation

Bind the Env ID to the shared implementation in the same `g1.py` module:

```python
registry.env("g1-wbt-dance1-subject1")(ManagerEnv)
```

The `motrix_envs.locomotion.wbt` package already imports this module, so `import motrix_envs` executes the registration. The
`envcfg` and `env` names must match exactly.

## 5. Add Training Tasks

Create `configs/task/g1-wbt-dance1-subject1/motrix.fastsac.yaml`:

```yaml
# @package _global_
defaults:
    - /algo_base@algo: motrix.fastsac
    - _self_
task:
    env: g1-wbt-dance1-subject1
    rllib: motrix
    algo: fastsac
num_envs: 2048
play_num_envs: 16
seed: 1
checkpoint:
    interval: 1000
algo:
    asynchronous: true
    agent:
        num_updates: 4
        policy_frequency: 2
        gamma: 0.99
        tau: 0.05
        target_entropy_ratio: 0.5
        num_atoms: 501
    trainer:
        num_learning_iterations: 40000
        async_options:
            utd_mode: strict
```

Asynchronous collector/learner execution is the default. Set `algo.asynchronous: false` for synchronous execution. The same
`motrix.fastsac` Task, algorithm configuration, and checkpoint format serve both topologies. Copy an existing WBT Training
Task for the same robot first, and tune algorithm hyperparameters only after the baseline trains reliably.

## 6. Validate training and playback

Use a small smoke test for Hydra composition, registry creation, tensor shapes, and the training path:

```bash
python scripts/train.py task=g1-wbt-dance1-subject1/motrix.fastsac \
  algo.asynchronous=true num_envs=64 algo.trainer.num_learning_iterations=100
```

Check for missing motion, joint, or body names; systematic NaNs, joint-limit violations, or immediate bad-tracking after
reset; and verify that `info["Reward"]` and `info["metrics"]` reach the logs. Then start the default training run:

```bash
python scripts/train.py task=g1-wbt-dance1-subject1/motrix.fastsac algo.asynchronous=true
```

After training creates a metadata-backed run, play the policy:

```bash
python scripts/play.py env=g1-wbt-dance1-subject1 num_envs=16
```

Play mode starts at frame 0, disables reset noise and adaptive sampling, and restarts from the beginning at the clip end. A
separate play environment is not required.

## Acceptance checklist

-   The motion passes schema loading and target-robot kinematic replay, with `fps == 1 / ctrl_dt`.
-   `envcfg`, `env`, and Training Tasks use the same Env ID.
-   The package import path executes the new config and environment registrations.
-   Hydra discovers the `motrix.fastsac` Training Task, and `algo.asynchronous` selects either execution topology.
-   Small-scale observation, action, reward, terminated, and truncated shapes are correct.
-   Resets do not systematically produce NaNs, hard-limit violations, or immediate bad tracking.
-   Reward, termination, and adaptive-sampling metrics reach the logs.
-   `play.py` discovers the run and plays the policy from frame 0 through the complete clip.
-   The new motion includes source and license information and is managed through Git LFS.
