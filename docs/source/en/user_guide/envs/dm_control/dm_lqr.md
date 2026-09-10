# Linear Quadratic Regulator

LQR (Linear Quadratic Regulator) is a classic continuous control and stabilization task. This repository currently provides two variants:

-   `dm-lqr-2-1`: two masses connected by a rope, with only the last mass actuated
-   `dm-lqr-6-2`: six masses connected as a chain, with only the last two masses actuated

The goal is to drive the whole system back to the center and keep it near equilibrium with minimal control effort.

```{video} /_static/videos/dm_lqr_2_1.mp4
:poster: _static/images/poster/dm_lqr_2_1.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

```{video} /_static/videos/dm_lqr_6_2.mp4
:poster: _static/images/poster/dm_lqr_6_2.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

---

## Task Description

Both tasks can be viewed as one-dimensional spring-damper chain stabilization problems. Each mass has a single translational degree of freedom along the x-axis. Neighboring masses are coupled by rope-like spring forces, and the system is affected by:

-   body damping on each mass
-   spring forces and relative damping between neighboring masses
-   a center-restoring force pulling the system toward the origin
-   control inputs applied only to the actuated terminal degrees of freedom

In practice:

-   `dm-lqr-2-1` is the simpler version and is useful for verifying whether the policy can learn a stable equilibrium
-   `dm-lqr-6-2` is more difficult because the controller must propagate its effect through a longer chain

---

## Action Space

### dm-lqr-2-1

| Item          | Details                         |
| ------------- | ------------------------------- |
| **Type**      | `Box(-1.0, 1.0, (1,), float32)` |
| **Dimension** | 1                               |

| Index | Action Description                     | Min  | Max | XML Joint |
| ----- | -------------------------------------- | ---- | --- | --------- |
| 0     | Control input applied to the last mass | -1.0 | 1.0 | `q1`      |

### dm-lqr-6-2

| Item          | Details                         |
| ------------- | ------------------------------- |
| **Type**      | `Box(-1.0, 1.0, (2,), float32)` |
| **Dimension** | 2                               |

| Index | Action Description                            | Min  | Max | XML Joint |
| ----- | --------------------------------------------- | ---- | --- | --------- |
| 0     | Control input applied to the second-last mass | -1.0 | 1.0 | `q4`      |
| 1     | Control input applied to the last mass        | -1.0 | 1.0 | `q5`      |

---

## Observation Space

The observation is formed by concatenating all positions `qpos` and velocities `qvel`.

### dm-lqr-2-1

| Item          | Details                         |
| ------------- | ------------------------------- |
| **Type**      | `Box(-inf, inf, (4,), float32)` |
| **Dimension** | 4                               |

| Index | Observation | Meaning                     |
| ----- | ----------- | --------------------------- |
| 0     | `q0`        | Position of the first mass  |
| 1     | `q1`        | Position of the second mass |
| 2     | `dq0`       | Velocity of the first mass  |
| 3     | `dq1`       | Velocity of the second mass |

### dm-lqr-6-2

| Item          | Details                          |
| ------------- | -------------------------------- |
| **Type**      | `Box(-inf, inf, (12,), float32)` |
| **Dimension** | 12                               |

The first 6 dimensions are `q0 ~ q5`, and the last 6 dimensions are `dq0 ~ dq5`.

---

## Reward Function Design

The current reward is composed of state cost, velocity cost, control cost, success bonus, and out-of-bounds penalty:

```python
state_cost = 0.5 * sum(qpos**2)
velocity_cost = 0.5 * velocity_cost_coef * sum(qvel**2)
control_cost = 0.5 * control_cost_coef * sum(action**2)

reward = 1.0 - (state_cost + velocity_cost + control_cost)
reward += success_bonus
reward -= out_of_bounds_penalty
```

Intuitively:

-   the farther the system is from the origin, the lower the reward
-   larger velocities reduce the reward
-   aggressive control inputs reduce the reward
-   entering a small stable region around the origin yields a success bonus
-   leaving the valid state boundary triggers an additional penalty

---

## Initial State

At reset:

-   the position vector is sampled in a random direction and normalized to a fixed norm
-   all initial velocities are set to zero

With the current configuration:

-   `dm-lqr-2-1` starts with position norm around `0.8`
-   `dm-lqr-6-2` starts with position norm around `1.0`

---

## Episode Termination Conditions

An episode terminates and resets when any of the following conditions is met:

-   success condition is reached:
    the position norm is below the success distance threshold and the velocity norm is below the success velocity threshold
-   out-of-bounds condition is reached:
    any position exceeds the position boundary or any velocity exceeds the velocity boundary
-   the full state is sufficiently close to zero
-   `NaN` appears in the observation or action

---

## Usage Guide

### 1. Environment Preview

```bash
python scripts/view.py env=dm-lqr-2-1
python scripts/view.py env=dm-lqr-6-2
```

### 2. Start Training

```bash
python scripts/train.py task=dm-lqr-2-1/skrl.ppo
python scripts/train.py task=dm-lqr-6-2/skrl.ppo
```

### 3. View Training Progress

```bash
tensorboard --logdir runs/dm-lqr-2-1
tensorboard --logdir runs/dm-lqr-6-2
```

### 4. Test Training Results

```bash
python scripts/play.py env=dm-lqr-2-1
python scripts/play.py env=dm-lqr-6-2
```

---

## Expected Training Results

### dm-lqr-2-1

1. The actuated mass pulls the unactuated mass back toward the center.
2. Both positions and velocities converge to a small neighborhood of zero.
3. The learned policy does not settle at a biased off-center equilibrium.

### dm-lqr-6-2

1. The last two actuated masses gradually pull the entire chain back toward the center.
2. The chain remains stable without obvious divergence or persistent oscillation.
3. Success rate increases during training while the out-of-bounds rate decreases.
