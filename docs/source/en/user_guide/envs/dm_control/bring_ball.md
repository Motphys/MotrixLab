# Manipulator Bring Ball

Bring Ball is a classic Manipulator task from the DeepMind Control Suite. A planar hand with a thumb and finger must grasp a ball and move it to a target ball position. MotrixLab currently provides one Bring Ball environment:

-   `dm-manipulator-bring-ball`: grasp the ball and move it to the target position

```{video} /_static/videos/bring-ball.mp4
:poster: _static/images/poster/bring-ball.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

---

## Task Description

Bring Ball is a planar (x-z) grasp-and-transport task:

-   The hand has 4 arm joints (`arm_root`, `arm_shoulder`, `arm_elbow`, `arm_wrist`) plus thumb/finger joints
-   Grasping is driven by the `grasp` tendon, coupling the `thumb` and `finger` joints
-   The ball can slide in the plane (`ball_x`, `ball_z`) and rotate about the y-axis (`ball_y`)
-   The target ball is a mocap body, sampled at reset

---

## Action Space

| Item          | Details                         |
| ------------- | ------------------------------- |
| **Type**      | `Box(-1.0, 1.0, (5,), float32)` |
| **Dimension** | 5                               |

Actions correspond to the following actuators:

| Index | Action Meaning                     | Min Control | Max Control |  XML Name  |
| ----: | ---------------------------------- | :---------: | :---------: | :--------: |
|     0 | Root joint drive                   |     -1      |      1      |   `root`   |
|     1 | Shoulder joint drive               |     -1      |      1      | `shoulder` |
|     2 | Elbow joint drive                  |     -1      |      1      |  `elbow`   |
|     3 | Wrist joint drive                  |     -1      |      1      |  `wrist`   |
|     4 | Grasp drive (thumb/finger coupled) |     -1      |      1      |  `grasp`   |

---

## Observation Space

| Item          | Details                          |
| ------------- | -------------------------------- |
| **Type**      | `Box(-inf, inf, (41,), float32)` |
| **Dimension** | 41                               |

The observation vector is composed of the following parts (in order):

| Part           | Content Description           | Dim | Notes                                |
| -------------- | ----------------------------- | --- | ------------------------------------ |
| **arm_pos**    | `sin`/`cos` of 8 joint angles | 16  | Joint order listed below             |
| **arm_vel**    | 8 joint velocities            | 8   | Same order as arm_pos                |
| **touch**      | `log(1 + touch)` sensors      | 5   | palm/finger/thumb/fingertip/thumbtip |
| **hand_pos**   | Grasp site world position     | 3   | x, y, z                              |
| **object_pos** | Ball position                 | 3   | x, y, z                              |
| **target_pos** | Target ball position          | 3   | x, y, z                              |
| **rel**        | `object_pos - target_pos`     | 3   | Relative position                    |

| Index | Observation Range                               | Dim | Notes                             |
| ----- | ----------------------------------------------- | --- | --------------------------------- |
| 0-15  | `sin`/`cos` of 8 joint angles                   | 16  | Joint order: arm_root -> thumbtip |
| 16-23 | 8 joint velocities                              | 8   | Same order as above               |
| 24-28 | Touch: palm, finger, thumb, fingertip, thumbtip | 5   | `log(1 + touch)`                  |
| 29-31 | Grasp position (x, y, z)                        | 3   | hand_pos                          |
| 32-34 | Ball position (x, y, z)                         | 3   | object_pos                        |
| 35-37 | Target ball position (x, y, z)                  | 3   | target_pos                        |
| 38-40 | Ball relative position (x, y, z)                | 3   | rel                               |

Joint order: `arm_root`, `arm_shoulder`, `arm_elbow`, `arm_wrist`, `finger`, `fingertip`, `thumb`, `thumbtip`.

---

## Reward Function Design

Bring Ball uses shaped rewards with multiple components and penalties:

```python
# R1: Reach - fingertips approach the ball
r_reach = tolerance(avg_tip_dist)

# R2: Orient - palm points toward the ball
r_orient = clip(1 - orient_bound + dot(hand_dir, unit_vec_to_ball), 0..1)

# R3: Pause - reduce arm jitter when close to the ball
r_pause = tolerance(arm_speed_step) * is_close_to_ball

# R4: Close - grasp intent and contact together
r_close = r_close_intent * (approach_or_grasp)

# R5: Lift & Transport - height and target distance
r_lift_height = tolerance(ball_z)
r_transport = tolerance(move_dist_to_target)
r_lift = mix(r_lift_height, r_transport)

# Precision/Progress
r_precision = tolerance(move_dist_to_target, gaussian)
r_progress = (prev_dist - curr_dist) * scale

# Penalties
penalty_side + penalty_hover
```

Default weights (from `BringBallCfg`):

-   reach 1.0, orient 1.5, pause 0.5, close 2.0, lift 6.0, precision 1.0
-   lift mixes `lift_height_weight` and `transport_weight`
-   progress reward is controlled by `transport_progress_scale`

---

## Initial State

-   **Arm initialization**: uses the default model pose (`randomize_arm=False`), thumb/finger are symmetric
-   **Target position**: `x in [-0.4, 0.4]`, `z in [0.1, 0.4]`, `y = 0.001`
-   **Ball position**: `x in [-0.4, 0.4]`, `z in [0.2, 0.7]`, with a minimum hand distance
-   **Physics settling**: performs settle steps after reset (`settle_steps=300`)

---

## Episode Termination Conditions

-   Terminate if any `NaN` appears in observations

---

## Usage Guide

### 1. Environment Preview (random actions)

```bash
python scripts/view.py env=dm-manipulator-bring-ball
```

### 2. Start Training

```bash
python scripts/train.py task=dm-manipulator-bring-ball/skrl.ppo.torch
```

### 3. View Training Progress

```bash
tensorboard --logdir runs/dm-manipulator-bring-ball
```

### 4. Test Training Results

```bash
python scripts/play.py env=dm-manipulator-bring-ball
```

---

## Expected Training Results

1. The hand consistently reaches and grasps the ball
2. The ball is lifted off the ground and held at a stable height
3. The ball is reliably moved close to the target ball position
