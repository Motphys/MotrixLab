# Stewart Platform Balance Control

The Stewart platform task trains a six-actuator parallel platform to stabilize a rolling ball through tilt control. The repository currently provides two main variants:

-   `stewart-static`: static balance task
-   `stewart-disturb-xy`: balance task with planar disturbances

The environment name `stewart` currently uses the same configuration as `stewart-static`, and can be treated as its alias.

```{video} /_static/videos/stewart_static.mp4
:poster: _static/images/poster/stewart.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

```{video} /_static/videos/stewart_disturb_xy.mp4
:poster: _static/images/poster/stewart.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

---

## Task Description

The scene contains a Stewart parallel platform, a top tray, and a freely rolling ball. The policy does not directly control the six leg lengths. Instead, it outputs a 2D tilt command:

-   target roll
-   target pitch

The environment converts these two action values into a target platform pose, then computes actuator commands for the six sliding legs. The ball rolls under gravity and the platform motion, and the policy must drive it back toward the center and keep it still with sufficiently low velocity.

In the disturbance variant, low-frequency planar motion is additionally applied to the support stage, and the disturbance state can be appended to the observation.

---

## Action Space

All Stewart tasks use the same action space:

| Item     | Details                         |
| -------- | ------------------------------- |
| **Type** | `Box(-1.0, 1.0, (2,), float32)` |
| **Dim**  | 2                               |

| Index | Action Description  | Min  | Max | Notes                           |
| ----- | ------------------- | ---- | --- | ------------------------------- |
| 0     | Roll control input  | -1.0 | 1.0 | Mapped to target platform roll  |
| 1     | Pitch control input | -1.0 | 1.0 | Mapped to target platform pitch |

The action is not written directly to the 6 actuators. It is first converted into a target platform orientation, then transformed into leg-length control commands.

---

## Observation Space

### stewart-static / stewart

| Item     | Details                          |
| -------- | -------------------------------- |
| **Type** | `Box(-inf, inf, (15,), float32)` |
| **Dim**  | 15                               |

The observation vector is composed of:

| Component            | Meaning                                        | Dim |
| -------------------- | ---------------------------------------------- | --- |
| **rel**              | Ball position in the platform local frame      | 3   |
| **rel_vel**          | Relative ball velocity                         | 3   |
| **platform_tilt**    | Normalized roll and pitch of the platform      | 2   |
| **platform_ang_vel** | Platform angular velocity in local coordinates | 3   |
| **target_tilt**      | Current target tilt command                    | 2   |
| **action_exec**      | Smoothed executed action                       | 2   |

### stewart-disturb-xy

| Item     | Details                          |
| -------- | -------------------------------- |
| **Type** | `Box(-inf, inf, (25,), float32)` |
| **Dim**  | 25                               |

The disturbance version adds 10 disturbance-related dimensions on top of the 15-dimensional static observation:

-   `disturb_pos`
-   `disturb_lin_vel`
-   `disturb_rot_deg`
-   `disturb_ang_vel_deg`

---

## Reward Function

The Stewart task currently uses three main reward terms and one terminal penalty:

```python
center_score = k_center * clip(1 - rel_xy / platform_radius, 0, 1)
zero_vel_closer = k_progress * zero_improve_norm
still_bonus = k_still if success else 0

reward = center_score + zero_vel_closer + still_bonus
reward = fall_penalty if fallen else reward
```

Their roles are:

-   **Center reward**: the closer the ball is to the platform center, the higher the reward
-   **Low-velocity progress reward**: additional progress reward when the ball is moving slowly and reaches a better near-center state than the previous low-velocity reference
-   **Still bonus**: reward for remaining near the center with sufficiently low velocity for several consecutive control steps
-   **Fall penalty**: immediate terminal penalty when the ball rolls off the platform or falls

---

## Initial State

At every reset, the environment:

-   initializes the platform with a small random tilt
-   runs a short settle phase so that the legs and constraints reach a stable state
-   places the ball in a circular region near the center of the platform
-   clears initial linear velocity, angular velocity, and action history

In particular:

-   the initial platform tilt is sampled from `min_init_tilt_deg ~ init_tilt_deg`
-   the initial ball radius is controlled by `platform_radius * init_ball_radius_ratio`

---

## Episode Termination Conditions

### Termination

An episode terminates when any of the following is true:

-   the ball leaves the valid platform radius
-   the ball height drops below the fall threshold
-   the ball stays near the center with sufficiently low velocity for several consecutive steps, which is treated as success

### Truncation

-   the maximum episode duration is reached  
    the current default is `max_episode_seconds = 24.0s`

---

## Usage Guide

### 1. Environment Preview

```bash
python scripts/view.py env=stewart-static
python scripts/view.py env=stewart-disturb-xy
```

### 2. Start Training

```bash
python scripts/train.py task=stewart-static/skrl.ppo
python scripts/train.py task=stewart-disturb-xy/skrl.ppo
```

### Training Notes

-   The current default hyperparameters are intended as a runnable baseline.
-   Due to the limited default training budget and generic PPO settings, the final result may not be optimal for this task.
-   For better success rate or smoother stabilization, consider tuning the training timesteps, learning rate, rollouts, and mini-batch settings.

### 3. View Training Progress

```bash
tensorboard --logdir runs/stewart-static
tensorboard --logdir runs/stewart-disturb-xy
```

### 4. Test Training Results

```bash
python scripts/play.py env=stewart-static
python scripts/play.py env=stewart-disturb-xy
```

---

## Expected Training Results

### stewart-static

1. The ball is driven back toward the platform center.
2. The relative velocity decreases and eventually settles near zero.
3. The platform avoids large persistent oscillations.

### stewart-disturb-xy

1. The ball remains controllable near the center under low-frequency planar disturbances.
2. The policy compensates for slow external motion of the stage.
3. Success rate is lower than the static version at first, but improves during training.
