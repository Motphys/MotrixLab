# Quadruped Robot

The Quadruped robot is a classic continuous control task in the DeepMind Control Suite. In MotrixLab, the `motrix_envs/src/motrix_envs/basic/quadruped` directory currently registers four directly trainable tasks: flat-ground walking `dm-quadruped-walk`, flat-ground running `dm-quadruped-run`, rough-terrain escape `dm-quadruped-escape`, and flat-ground ball pushing `dm-quadruped-fetch`.

## Task Preview

### Walk

```{video} /_static/videos/dm_quadruped_walk.mp4
:poster: _static/images/poster/dm_quadruped_walk.png
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

### Run

```{video} /_static/videos/dm_quadruped_run.mp4
:poster: _static/images/poster/dm_quadruped_run.png
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

### Escape

```{video} /_static/videos/dm_quadruped_escape.mp4
:poster: _static/images/poster/dm_quadruped_escape.png
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

### Fetch

```{video} /_static/videos/dm_quadruped_fetch.mp4
:poster: _static/images/poster/dm_quadruped_fetch.png
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

## Task Overview

| Environment ID        | Task Goal                                                                    | Model File             | Target Speed | Observation Dimension |
| --------------------- | ---------------------------------------------------------------------------- | ---------------------- | ------------ | --------------------- |
| `dm-quadruped-walk`   | Walk forward stably on flat ground while maintaining heading                 | `quadruped_walk.xml`   | 0.5 m/s      | 54                    |
| `dm-quadruped-run`    | Run at high speed on flat ground while maintaining stable posture            | `quadruped_walk.xml`   | 5.0 m/s      | 54                    |
| `dm-quadruped-escape` | Escape outward from the origin area as quickly as possible on uneven terrain | `quadruped_escape.xml` | 3.0 m/s      | 57                    |
| `dm-quadruped-fetch`  | Push a ball into the target area on flat ground                              | `quadruped_fetch.xml`  | 2.0 m/s      | 66                    |

## Task Description

Quadruped is a 3D quadruped robot task. The robot consists of one torso and four legs, and each leg has control dimensions related to yaw, lift, and extension. The underlying XML defines the hip, knee, and ankle joint structure, while the action layer uses a coupled actuator design with 3 actuators per leg:

-   `yaw`: controls leg yaw
-   `lift`: controls leg lifting through tendon coupling
-   `extend`: controls leg extension/retraction through tendon coupling

`walk` and `run` use the same flat-ground model, with the main difference being the target speed. `escape` uses `quadruped_escape.xml` with a heightfield terrain and requires the robot to move away from the world origin quickly while maintaining an upright torso and stable locomotion. `fetch` uses `quadruped_fetch.xml`, which adds a free ball and a target region to the scene, requiring the robot to first approach a suitable position and then push the ball toward the goal.

---

## Action Space

| Item          | Details                          |
| ------------- | -------------------------------- |
| **Type**      | `Box(low, high, (12,), float32)` |
| **Dimension** | 12                               |

Actions are arranged leg by leg, and each leg contains three actuators: `yaw / lift / extend`.

| Index | Action Meaning                    | Min Value | Max Value | Corresponding Actuator |
| ----: | --------------------------------- | :-------: | :-------: | ---------------------- |
|     0 | Front-left leg yaw control        |   -1.0    |    1.0    | `yaw_front_left`       |
|     1 | Front-left leg lift control       |   -1.0    |    1.1    | `lift_front_left`      |
|     2 | Front-left leg extension control  |   -0.8    |    0.8    | `extend_front_left`    |
|     3 | Front-right leg yaw control       |   -1.0    |    1.0    | `yaw_front_right`      |
|     4 | Front-right leg lift control      |   -1.0    |    1.1    | `lift_front_right`     |
|     5 | Front-right leg extension control |   -0.8    |    0.8    | `extend_front_right`   |
|     6 | Rear-right leg yaw control        |   -1.0    |    1.0    | `yaw_back_right`       |
|     7 | Rear-right leg lift control       |   -1.0    |    1.1    | `lift_back_right`      |
|     8 | Rear-right leg extension control  |   -0.8    |    0.8    | `extend_back_right`    |
|     9 | Rear-left leg yaw control         |   -1.0    |    1.0    | `yaw_back_left`        |
|    10 | Rear-left leg lift control        |   -1.0    |    1.1    | `lift_back_left`       |
|    11 | Rear-left leg extension control   |   -0.8    |    0.8    | `extend_back_left`     |

---

## Observation Space

| Environment           | Details                          |
| --------------------- | -------------------------------- |
| `dm-quadruped-walk`   | `Box(-inf, inf, (54,), float32)` |
| `dm-quadruped-run`    | `Box(-inf, inf, (54,), float32)` |
| `dm-quadruped-escape` | `Box(-inf, inf, (57,), float32)` |
| `dm-quadruped-fetch`  | `Box(-inf, inf, (66,), float32)` |

All four tasks share nearly the same proprioceptive observations. `escape` adds 3 task-related dimensions associated with the origin, while `fetch` adds ball state and target position information:

| Part                   | Description                                               | Dimension | `walk/run` | `escape` | `fetch` |
| ---------------------- | --------------------------------------------------------- | --------- | ---------- | -------- | ------- |
| **egocentric dof pos** | Body generalized position state                           | 16        | Yes        | Yes      | Yes     |
| **egocentric dof vel** | Body generalized velocity state                           | 16        | Yes        | Yes      | Yes     |
| **actuator ctrl**      | Current 12-dimensional actuator controls                  | 12        | Yes        | Yes      | Yes     |
| **torso velocity**     | Torso linear velocity sensor `velocimeter`                | 3         | Yes        | Yes      | Yes     |
| **torso upright**      | Scalar representing torso uprightness                     | 1         | Yes        | Yes      | Yes     |
| **imu**                | IMU acceleration and angular velocity                     | 6         | Yes        | Yes      | Yes     |
| **origin**             | World origin position in the body frame                   | 3         | No         | Yes      | No      |
| **ball state**         | Ball position, relative linear velocity, angular velocity | 9         | No         | No       | Yes     |
| **target**             | Relative target position in the body frame                | 3         | No         | No       | Yes     |

The XML also defines foot force/torque sensors and a center-of-mass sensor, but these values are not directly concatenated into the default observation in the current implementation.

---

## Reward Function Design

All four tasks use torso uprightness as the core constraint. In the implementation, `upright_reward` is computed first from `torso_upright`, encouraging the robot to keep the body close to upright.

### Walk / Run

`dm-quadruped-walk` and `dm-quadruped-run` use the same reward structure, with different target speeds:

-   `walk` tracks `0.5 m/s`
-   `run` tracks `5.0 m/s`

The total reward is composed of:

```python
# Speed reward: reach the target forward speed
# Posture reward: keep the torso upright
# Auxiliary rewards: height, lateral stability, heading alignment, action smoothness
# Penalties: backward motion, excessive vertical speed, excessive roll/pitch angular velocity, deviation from default posture
total_reward = upright_reward * move_reward + shaping_terms - penalty_terms
```

The main shaping and penalty terms include:

-   `height_reward`: encourages the torso to stay near the standing height
-   `lateral_reward`: suppresses excessive lateral velocity
-   `heading_reward`: encourages forward motion along the +X direction
-   `smooth_reward`: penalizes large action changes between consecutive timesteps
-   `backward_penalty`: suppresses backward movement
-   `lin_vel_z_penalty` and `ang_vel_xy_penalty`: suppress vertical bouncing and excessive torso roll/pitch
-   `similar_to_default_penalty`: encourages joint posture to stay reasonably close to the default standing pose

### Escape

`dm-quadruped-escape` adds a task term for escaping away from the origin area on top of the locomotion reward. This task uses the heightfield terrain in `quadruped_escape.xml`:

```python
# Base locomotion reward
# + reward for getting farther from the origin
# + reward for outward radial speed
total_reward = locomotion_reward + upright_reward * escape_reward + radial_speed_reward
```

Additional task terms include:

-   `escape_reward`: rewards the robot based on its distance from the origin area
-   `radial_speed_reward`: encourages acceleration along the outward direction away from the origin

This makes `escape` require not only fast locomotion, but also correct outward motion on rough terrain.

### Fetch

`dm-quadruped-fetch` uses a task-specific shaping structure centered on positioning and ball pushing. In the current implementation, the reward is mainly based on the geometric relationship between the robot, the ball, and the target:

```python
# Positioning stage: encourage the robot to move behind or slightly behind the ball
# Ready stage: encourage facing the ball, getting close to it, and aligning with the ball-target line
# Pushing stage: encourage the ball to roll toward the target and eventually enter the target area
# Penalties: moving in the wrong direction, pushing the ball away from the target, getting the legs too close to the ball
total_reward = stage_terms + ready_terms + push_terms - penalty_terms
```

The main terms include:

-   `stage_move`: encourages movement toward the current stage waypoint
-   `stage_reach`: encourages reaching a suitable waypoint behind or to the side of the ball
-   `behind_align`: encourages the robot to position itself behind the ball relative to the target
-   `face_ball`: encourages the torso heading to point toward the ball
-   `near_ball`: encourages the robot to approach the ball
-   `ready` and `ready_gate`: combine position, orientation, and distance into a readiness signal for active pushing
-   `fetch`: encourages the ball to get closer to the target region
-   `push`: encourages the ball to move along the target direction
-   `backward`: penalizes moving opposite to the current stage target
-   `away`: penalizes pushing the ball away from the target
-   `leg_ball`: penalizes leg geometry getting too close to the ball, reducing ball trapping and squeezing behavior

In addition, `fetch` uses a `stability` gate on torso uprightness and torso height so the agent cannot easily exploit obviously collapsed poses to collect task reward.

---

## Initial State

-   `walk`, `run`, and `escape` reset from the default quadruped standing pose defined in the XML
-   For these three tasks, the root orientation is fixed to the initial heading instead of being randomly rotated
-   `fetch` randomizes the robot position and yaw on the plane, and also randomizes the ball position on the ground
-   All tasks initialize joint velocities, and in `fetch` also ball velocities, to zero
-   During reset, the robot is automatically lifted until there is no initial penetration/contact with the ground

## Episode Termination Conditions

-   Maximum episode duration is 20 seconds
-   The episode terminates when `NaN` appears in the observation
-   `walk`, `run`, and `escape` do not currently define a separate fall termination condition
-   `fetch` terminates early when the robot has clearly fallen, based on low torso uprightness or low torso height
-   The current implementation does not yet define a separate success termination condition for "ball enters the target area"

---

## Usage Guide

### 1. Environment Preview

```bash
python scripts/view.py env=dm-quadruped-walk
python scripts/view.py env=dm-quadruped-run
python scripts/view.py env=dm-quadruped-escape
python scripts/view.py env=dm-quadruped-fetch
```

### 2. Start Training

```bash
python scripts/train.py task=dm-quadruped-walk/skrl.ppo
python scripts/train.py task=dm-quadruped-run/skrl.ppo
python scripts/train.py task=dm-quadruped-escape/skrl.ppo
python scripts/train.py task=dm-quadruped-fetch/skrl.ppo
```

### 3. View Training Progress

```bash
tensorboard --logdir runs/dm-quadruped-walk
tensorboard --logdir runs/dm-quadruped-run
tensorboard --logdir runs/dm-quadruped-escape
tensorboard --logdir runs/dm-quadruped-fetch
```

### 4. Test Training Results

```bash
python scripts/play.py env=dm-quadruped-walk
python scripts/play.py env=dm-quadruped-run
python scripts/play.py env=dm-quadruped-escape
python scripts/play.py env=dm-quadruped-fetch
```

---

## Expected Training Results

### Walking Task (`dm-quadruped-walk`)

1. Maintain a stable forward speed close to `0.5 m/s`
2. Keep body posture stable with small lateral sway
3. Sustain walking along the +X direction

### Running Task (`dm-quadruped-run`)

1. Increase speed to near or above `5.0 m/s`
2. Produce larger stride length and more explosive motions
3. Maintain good torso stability under high-speed locomotion

### Escape Task (`dm-quadruped-escape`)

1. Move away from the origin region quickly
2. Maintain stable footholds on the heightfield terrain without easily tipping over
3. Move primarily outward rather than spinning in place

### Fetch Task (`dm-quadruped-fetch`)

1. First move into a reasonable position along the ball-target line instead of randomly colliding with the ball from the side
2. Push the ball toward the target area consistently instead of kicking it away or repeatedly sending it off course
3. Maintain better body stability during pushing, with fewer collapsed poses, flips, or leg-ball entanglement behaviors
