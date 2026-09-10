# Finger Manipulation

Finger is a classic manipulation task from the DeepMind Control Suite. A two-link “finger” applies torques to interact with a rotating spinner. MotrixLab currently provides three Finger environments:

```{video} /_static/videos/dm_finger_spin.mp4
:poster: _static/images/poster/dm_finger_spin.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

```{video} /_static/videos/dm_finger_turn.mp4
:poster: _static/images/poster/dm_finger_turn.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

-   `dm-finger-spin`: make the spinner rotate continuously in the target direction
-   `dm-finger-turn-easy`: align the spinner tip (`tip`) with a target point (larger target radius)
-   `dm-finger-turn-hard`: same as Turn, but with a smaller target radius

---

## Task Description

Finger is a planar (x-z) interaction task:

-   The finger has 2 actuated hinge joints: `proximal` and `distal`
-   The spinner rotates around joint `hinge`, and `tip` denotes the spinner tip position
-   For Turn tasks, a target point is sampled around the spinner at the beginning of each episode

---

## Action Space

| Item          | Details                         |
| ------------- | ------------------------------- |
| **Type**      | `Box(-1.0, 1.0, (2,), float32)` |
| **Dimension** | 2                               |

The actions correspond to:

| Index | Action Description                 | Min Control | Max Control | XML Name | Joint Type |
| ----- | ---------------------------------- | ----------- | ----------- | -------- | ---------- |
| 0     | Torque applied to `proximal` joint | -1          | 1           | proximal | hinge      |
| 1     | Torque applied to `distal` joint   | -1          | 1           | distal   | hinge      |

---

## Observation Space

MotrixLab follows dm_control-style observations, but flattens them into a single vector.

### Spin Observation Space

| Item          | Details                         |
| ------------- | ------------------------------- |
| **Type**      | `Box(-inf, inf, (9,), float32)` |
| **Dimension** | 9                               |

The observation vector contains (in order):

-   **position (4)**: `qpos(proximal, distal)` + `tip_xz` (tip position relative to the spinner in x-z)
-   **velocity (3)**: `qvel(proximal, distal, hinge)` (hinge velocity is used by Spin reward)
-   **touch (2)**: `log(1 + touchtop)`, `log(1 + touchbottom)`

### Turn Observation Space

| Item          | Details                          |
| ------------- | -------------------------------- |
| **Type**      | `Box(-inf, inf, (12,), float32)` |
| **Dimension** | 12                               |

Compared to Spin, Turn adds:

-   **target_position (2)**: target position relative to the spinner in x-z
-   **dist_to_target (1)**: signed distance from tip to the target sphere surface (negative means “inside”)

---

## Reward Function Design

### Spin

In dm_control, Spin is typically defined with a sparse threshold on spinner angular velocity. MotrixLab defaults to a dense/shaped reward for easier training, while also logging the sparse version:

```text
spin_sparse = 1 if hinge_velocity <= -15 else 0
spin = clip(-hinge_velocity / 15, 0, 1)
```

### Turn (Easy / Hard)

Turn aims to bring the spinner tip into a target sphere around the spinner:

-   `turn_sparse = 1` when `dist_to_target <= 0`
-   MotrixLab defaults to a shaped reward based on distance-to-target (exponential decay), and adds auxiliary terms to reduce “no-contact” failure modes and action jitter:
    -   approach-to-spinner shaping
    -   touch bonus
    -   action magnitude / action change penalties

The final shaped reward is clipped to `[0, 1]`.

---

## Initial State

-   `proximal`, `distal` joint angles are sampled uniformly within joint limits
-   spinner `hinge` angle is sampled uniformly in `[-pi, pi]`
-   for Turn tasks, the target is sampled around the spinner on the x-z plane at reset

---

## Episode Termination Conditions

### Termination

If `NaN` appears in the observations

---

## Usage Guide

### 1. Environment Preview (random actions)

```bash
python scripts/view.py env=dm-finger-spin
```

```bash
python scripts/view.py env=dm-finger-turn-easy
```

```bash
python scripts/view.py env=dm-finger-turn-hard
```

### 2. Start Training

```bash
python scripts/train.py task=dm-finger-spin/skrl.ppo task.train_backend=torch
```

```bash
python scripts/train.py task=dm-finger-turn-easy/skrl.ppo task.train_backend=torch
```

```bash
python scripts/train.py task=dm-finger-turn-hard/skrl.ppo task.train_backend=torch
```

### 3. View Training Progress

```bash
tensorboard --logdir runs/dm-finger-spin
```

### 4. Test Training Results

`scripts/play.py` auto-discovers the latest metadata-backed run under `runs/{env-name}/` and loads its `best_policy` artifact. Use `policy=...` to select a checkpoint explicitly:

```bash
python scripts/play.py env=dm-finger-turn-hard
```

---

## Expected Training Results

1. `dm-finger-spin`: stable continuous rotation in the target direction
2. `dm-finger-turn-easy`: consistent contact and alignment with the (larger) target region, with reduced jitter
3. `dm-finger-turn-hard`: successful alignment with a smaller target region, typically requiring more training and better contact behavior
