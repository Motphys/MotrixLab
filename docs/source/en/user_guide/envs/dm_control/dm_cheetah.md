# Half-Cheetah Robot

The Half-Cheetah robot is a classic continuous control task in the DeepMind Control Suite. The goal is to train a simulated bipedal robot to run at high speed and stably by controlling its joint torques.

```{video} /_static/videos/dm_cheetah.mp4
:poster: _static/images/poster/dm_cheetah.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

## Task Description

HalfCheetah is a 2D half-cheetah running task, composed of 7 main body parts (1 torso and 3 sections for each of the front and rear legs), with 6 controlled joints (front and rear thighs [connected to the torso], shins [connected to the thighs], and feet [connected to the shins]). The agent applies torques to these joints as actions, aiming to make the cheetah run forward as fast and stably as possible.

---

## Action Space

| Item          | Details                         |
| ------------- | ------------------------------- |
| **Type**      | `Box(-1.0, 1.0, (6,), float32)` |
| **Dimension** | 6                               |

The joints correspond as follows:

| Index | Action Meaning (Torque applied to the joint) | Min Value | Max Value | Corresponding XML Name |
| ----: | -------------------------------------------- | :-------: | :-------: | :--------------------: |
|     0 | Rear Thigh Joint Drive Torque                |    -1     |     1     |        `bthigh`        |
|     1 | Rear Shin Joint Drive Torque                 |    -1     |     1     |        `bshin`         |
|     2 | Rear Foot Joint Drive Torque                 |    -1     |     1     |        `bfoot`         |
|     3 | Front Thigh Joint Drive Torque               |    -1     |     1     |        `fthigh`        |
|     4 | Front Shin Joint Drive Torque                |    -1     |     1     |        `fshin`         |
|     5 | Front Foot Joint Drive Torque                |    -1     |     1     |        `ffoot`         |

---

## Observation Space

| Item          | Details                          |
| ------------- | -------------------------------- |
| **Type**      | `Box(-inf, inf, (17,), float32)` |
| **Dimension** | 17                               |

The observation space of the HalfCheetah environment consists of the following parts (in order):
| Part | Content Description | Dimension | Remarks |
| -------- | ------------- | -- | ------------ |
| **qpos** | Position information of each body joint and the root | 8 | Root x-coordinate is excluded by default |
| **qvel** | Velocity information of each body joint and the root | 9 | Velocity is the derivative of position |

| Index    | Observation                  | Min Value | Max Value | XML Name | Joint Type | Type (Unit)              |
| -------- | ---------------------------- | --------- | --------- | -------- | ---------- | ------------------------ |
| 0        | Front z-coordinate           | -Inf      | Inf       | rootz    | slide      | Position (m)             |
| 1        | Front angle                  | -Inf      | Inf       | rooty    | hinge      | Angle (rad)              |
| 2        | Rear Thigh Angle             | -Inf      | Inf       | bthigh   | hinge      | Angle (rad)              |
| 3        | Rear Shin Angle              | -Inf      | Inf       | bshin    | hinge      | Angle (rad)              |
| 4        | Rear Foot Angle              | -Inf      | Inf       | bfoot    | hinge      | Angle (rad)              |
| 5        | Front Thigh Angle            | -Inf      | Inf       | fthigh   | hinge      | Angle (rad)              |
| 6        | Front Shin Angle             | -Inf      | Inf       | fshin    | hinge      | Angle (rad)              |
| 7        | Front Foot Angle             | -Inf      | Inf       | ffoot    | hinge      | Angle (rad)              |
| 8        | Front x-coordinate Velocity  | -Inf      | Inf       | rootx    | slide      | Velocity (m/s)           |
| 9        | Front z-coordinate Velocity  | -Inf      | Inf       | rootz    | slide      | Velocity (m/s)           |
| 10       | Front Angular Velocity       | -Inf      | Inf       | rooty    | hinge      | Angular Velocity (rad/s) |
| 11       | Rear Thigh Angular Velocity  | -Inf      | Inf       | bthigh   | hinge      | Angular Velocity (rad/s) |
| 12       | Rear Shin Angular Velocity   | -Inf      | Inf       | bshin    | hinge      | Angular Velocity (rad/s) |
| 13       | Rear Foot Angular Velocity   | -Inf      | Inf       | bfoot    | hinge      | Angular Velocity (rad/s) |
| 14       | Front Thigh Angular Velocity | -Inf      | Inf       | fthigh   | hinge      | Angular Velocity (rad/s) |
| 15       | Front Shin Angular Velocity  | -Inf      | Inf       | fshin    | hinge      | Angular Velocity (rad/s) |
| 16       | Front Foot Angular Velocity  | -Inf      | Inf       | ffoot    | hinge      | Angular Velocity (rad/s) |
| excluded | Front x-coordinate           | -Inf      | Inf       | rootx    | slide      | Position (m)             |

---

## Reward Function Design

The cheetah's reward function consists of the following parts:

```python
# Velocity Reward: Tracking target speed
# Posture Reward: Maintaining a stable posture
# Total Reward = Velocity Reward + Posture Reward
```

---

## Initial State

-   Reset all finite joint angles to random values within their allowed ranges, keeping infinite range joints in their default state.
-   Generate the initial observation vector by stabilizing the torso and leg positions through multi-step physics simulation.

## Episode Termination Conditions

-   **No Fall Termination Condition** (Does not end directly due to instability)

---

## Usage Guide

### 1. Environment Preview

```bash
python scripts/view.py env=dm-cheetah
```

### 2. Start Training

```bash
python scripts/train.py task=dm-cheetah/skrl.ppo
```

### 3. View Training Progress

```bash
tensorboard --logdir runs/dm-cheetah
```

### 4. Test Training Results

```bash
python scripts/play.py env=dm-cheetah
```

---

## Expected Training Results

1. Run at a stable horizontal speed close to or exceeding 10.0 m/s
2. Maintain torso stability and coordinated gait, running long distances without falling
3. Running posture close to that of a real cheetah, with a sense of extension during the run
