# Single-Leg Hopping Robot

Hopper is a classic single-leg hopping control task in dm-control, simulating a 2D single-leg hopping robot.

```{video} /_static/videos/dm_hopper.mp4
:poster: _static/images/poster/dm_hopper.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

## Task Description

The 2D robot consists of four body segments: torso, pelvis, thigh, calf, and foot. Actions are generated through four articulated joints, including: waist, hip, knee, and ankle. Each joint is driven by a motor with different gear ratios, enabling behaviors such as standing, balancing, and hopping forward.

---

## Action Space

| Item          | Details                         |
| ------------- | ------------------------------- |
| **Type**      | `Box(-1.0, 1.0, (3,), float32)` |
| **Dimension** | 3                               |

Joint mapping:

| Index | Action Description                | Min | Max | XML Name    |
| ----- | --------------------------------- | --- | --- | ----------- |
| 0     | Torque applied on the thigh rotor | -1  | 1   | thigh_joint |
| 1     | Torque applied on the leg rotor   | -1  | 1   | leg_joint   |
| 2     | Torque applied on the foot rotor  | -1  | 1   | foot_joint  |

---

## Observation Space

| Item          | Details                          |
| ------------- | -------------------------------- |
| **Type**      | `Box(-inf, inf, (13,), float64)` |
| **Dimension** | 13                               |

| Component           | Description                   | Dim | Notes                                |
| ------------------- | ----------------------------- | --- | ------------------------------------ |
| **qpos**            | Joint angles and torso height | 5   | torso x-position excluded by default |
| **qvel**            | Joint and torso velocities    | 6   | Velocity as derivative of position   |
| **contact sensors** | Toe and heel ground sensors   | 2   | Normalized by `log1p`                |

The observation vector consists of joint positions (qpos), velocities (qvel), and contact force sensors. The full dimension is 13, including two contact sensors:

| Index | Observation                  | XML Name      | Joint Type | Physical Meaning          |
| ----: | ---------------------------- | ------------- | ---------- | ------------------------- |
|     0 | torso z-position             | `rootz`       | slide      | torso height              |
|     1 | torso angle                  | `rooty`       | hinge      | body pitch angle          |
|     2 | thigh joint angle            | `thigh_joint` | hinge      | thigh rotation            |
|     3 | leg joint angle              | `leg_joint`   | hinge      | calf rotation             |
|     4 | foot joint angle             | `foot_joint`  | hinge      | foot rotation             |
|     5 | torso x-velocity             | `rootx`       | slide      | forward velocity          |
|     6 | torso z-velocity             | `rootz`       | slide      | vertical velocity         |
|     7 | torso angular velocity       | `rooty`       | hinge      | torso angular velocity    |
|     8 | thigh joint angular velocity | `thigh_joint` | hinge      | thigh angular velocity    |
|     9 | leg joint angular velocity   | `leg_joint`   | hinge      | calf angular velocity     |
|    10 | foot joint angular velocity  | `foot_joint`  | hinge      | foot angular velocity     |
|    11 | toe touch sensor             | `touch_toe`   | sensor     | toe ground contact force  |
|    12 | heel touch sensor            | `touch_heel`  | sensor     | heel ground contact force |

---

## Reward Function Design

The Hopper reward consists of the following terms:

### Stand Task

```python
# Stand reward: maintain stable height
```

### Hop Task

```python
# Stand reward: maintain stable height
# Hopping reward: achieve target forward velocity
# Leg movement reward: encourage moderate leg motion
# Knee extension reward: encourage proper knee extension
# Foot contact reward: encourage proper ground reaction forces
# Total reward = stand_reward + hop_reward + leg_motion_reward + knee_reward + contact_reward
```

---

## Initial State

-   Randomize joint angles within allowed ranges during reset

---

## Episode Termination Conditions

-   Observation values contain invalid numerical values (NaN)

---

## Usage Guide

### 1. Environment Preview

```bash
python scripts/view.py env=dm-hopper-stand
python scripts/view.py env=dm-hopper-hop
```

### 2. Start Training

```bash
python scripts/train.py task=dm-hopper-stand/skrl.ppo
python scripts/train.py task=dm-hopper-hop/skrl.ppo
```

### 3. View Training Progress

```bash
tensorboard --logdir runs/dm-hopper-stand
```

### 4. Test Training Results

```bash
python scripts/play.py env=dm-hopper-stand
python scripts/play.py env=dm-hopper-hop
```

---

## Expected Training Results

1. Maintain stable standing behavior
2. Achieve a target hopping speed of **2.0**
