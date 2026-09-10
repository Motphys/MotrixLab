# Two-Joint Robotic Arm Control

Reacher is a classic robotic arm control task, simulating a robotic arm composed of two links. The goal is to bring the end effector (fingertip) as close as possible to a randomly generated target point.

```{video} /_static/videos/dm_reacher.mp4
:poster: _static/images/poster/dm_reacher.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

## Task Description

The Reacher consists of two joints, with two links connected by hinge joints. The objective of the task is to move the end of the robotic arm to the target position. The target point is randomly sampled at the beginning of each episode.

---

## Action Space

| Item          | Details                         |
| ------------- | ------------------------------- |
| **Type**      | `Box(-1.0, 1.0, (2,), float32)` |
| **Dimension** | 2                               |

The actions correspond to:

| Index | Action Description                               | Min Control | Max Control | XML Name | Joint Type |
| ----- | ------------------------------------------------ | ----------- | ----------- | -------- | ---------- |
| 0     | Torque applied to the first joint (root link)    | -1          | 1           | joint0   | hinge      |
| 1     | Torque applied to the second joint (middle link) | -1          | 1           | joint1   | hinge      |

---

## Observation Space

| Item          | Details                         |
| ------------- | ------------------------------- |
| **Type**      | `Box(-inf, inf, (6,), float32)` |
| **Dimension** | 6                               |

The observation vector contains the following parts (in order):

-   **qpos**: 2 joint angles
-   **fingertip → target vector difference**: x and y dimensions
-   **qvel**: 2 joint angular velocities

| Index | Observation                     | Min  | Max | XML Name   | Joint | Unit  |
| ----- | ------------------------------- | ---- | --- | ---------- | ----- | ----- |
| 0     | First joint angle               | -inf | inf | joint0_pos | hinge | rad   |
| 1     | Second joint angle              | -inf | inf | joint1_pos | hinge | rad   |
| 2     | fingertip - target x difference | -inf | inf | NA         | slide | m     |
| 3     | fingertip - target y difference | -inf | inf | NA         | slide | m     |
| 4     | First joint angular velocity    | -inf | inf | joint0_vel | hinge | rad/s |
| 5     | Second joint angular velocity   | -inf | inf | joint1_vel | hinge | rad/s |

---

## Reward Function Design

The reward for this task is based on the **distance between the fingertip and the target**:

### Distance Reward (tolerance reward)

```text
reward = tolerance(|| fingertip - target ||)
```

-   The closer the distance, the higher the reward

---

## Initial State

The initial state is sampled from random distributions:

-   Arm angles: uniform distribution
-   Arm angular velocities: small random values
-   Target point position: random position in a circular area

---

## Episode Termination Conditions

### Termination

If `NaN` appears in the observations

### Termination Handling

```text
reward = 0
terminated = True
```

---

## Usage Guide

### 1. Environment Preview

```bash
python scripts/view.py env=dm-reacher
```

### 2. Start Training

```bash
python scripts/train.py task=dm-reacher/skrl.ppo
```

### 3. View Training Progress

```bash
tensorboard --logdir runs/dm-reacher
```

### 4. Test Training Results

```bash
python scripts/play.py env=dm-reacher
```

## Expected Training Results

The robotic arm quickly and accurately reaches the target point
