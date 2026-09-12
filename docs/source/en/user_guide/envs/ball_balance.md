# Ball Balance (Microduck)

## Overview

The ball-balance task keeps the 14-DoF biped robot Microduck standing on a free basketball (radius 0.14 m). The robot must coordinate its whole body to stay upright while keeping the rolling ball underneath its feet. The environment is implemented with the Manager workflow, the simulator backend is selected through configuration, and no physical domain randomization is applied.

```{video} /_static/videos/microduck-ball-balance.mp4
:poster: _static/images/poster/microduck-ball-balance.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

## Commands

Preview the environment without training:

```bash
python scripts/view.py env=microduck-ball-balance
```

Train with the built-in Motrix FastSAC config (the only training config provided for this task; `num_envs=2048`, 20000 iterations):

```bash
python scripts/train.py task=microduck-ball-balance/motrix.fastsac
```

Evaluate a trained policy:

```bash
python scripts/play.py env=microduck-ball-balance
```

## Action space

The action space is a 14-dimensional joint position target covering all actuated Microduck joints (left leg hip yaw/roll/pitch, knee, ankle; neck pitch; head pitch/yaw/roll; and the five matching right-leg joints).

Policy actions are converted to joint position targets as:

```
target joint angle = default stance pose + action × 0.5
```

The action scale `action_scale=0.5` is uniform across joints; action-space bounds are derived from each joint's `ctrl_range` in the robot model. The underlying actuators are position actuators; PD gains and effort limits are owned by the robot model.

## Observation space

Policy observations total 54 dims (3+3+3+3+14+14+14); value (critic) observations total 60 dims and add privileged, noise-free terms.

| Observation | Actor | Critic | Meaning |
| ----------- | ----: | -----: | ------- |
| Projected gravity | 3 | 3 | Gravity direction in the robot base frame; uniform noise ±0.05 (actor only) |
| Base angular velocity | 3 | 3 | Base angular velocity; uniform noise ±0.1 (actor only) |
| Base linear velocity | — | 3 | Base linear velocity; privileged, noise-free |
| Ball relative position | 3 | 3 | Ball center position relative to the base (base frame); uniform noise ±0.02 (actor only) |
| Ball relative velocity | 3 | 3 | Ball linear velocity (base frame); uniform noise ±0.1 (actor only) |
| Ball world position | — | 3 | Ball center in world coordinates; privileged |
| Ball world velocity | — | 3 | Ball linear velocity in world coordinates; privileged |
| Joint positions | 14 | 14 | Joint angles relative to the default pose; uniform noise ±0.01 (actor only) |
| Joint velocities | 14 | 14 | Joint velocities; uniform noise ±0.25 (actor only) |
| Last action | 14 | 14 | The currently applied action, noise-free |

## Reward design

The reward combines exponential-kernel terms and penalties. Exponential-kernel terms have the form $\exp(-e^2/\sigma^2)$ where $e$ is the corresponding error; the reward approaches 1 as the error shrinks.

| Reward term | Weight | Computation | Purpose |
| ----------- | -----: | ----------- | ------- |
| `alive` | 1.0 | Constant 1.0 per step | Survival floor so early policies cannot profit from terminating quickly to escape the action-rate penalty |
| `upright` | 4.0 | Exponential kernel on the base-frame gravity direction (σ=0.2) | Keep the trunk upright |
| `ball_under_feet` | 3.0 | Exponential kernel on the horizontal distance from the ball center to the feet midpoint (σ=0.05) | Keep the ball under the feet — the core task objective |
| `base_height` | 1.5 | Exponential kernel on base height against the 0.40 m target (= 0.12 + 2×0.14, the trunk height when standing on the ball apex) (σ=0.05) | Hold the balanced on-ball height |
| `dof_default` | 0.5 | Exponential kernel on the mean squared joint deviation from the default pose (σ=0.5) | Stay near the default stance and discourage extra motion |
| `action_rate_l2` | −0.5 | L2 norm of the difference between consecutive actions | Suppress jittery action sequences |
| `limits_dof_pos` | −5.0 | Soft penalty near joint limits (soft_limit=0.9, cap=5.0) | Avoid slamming into joint limits |
| `undesired_contacts` | −0.2 | Contact forces on non-foot links (threshold 0.5) | Penalize trunk or other body contact with the ground or the ball |

## Termination conditions

The episode ends with `terminated` (failure) when any of the following holds:

| Type | Condition | Meaning |
| ---- | --------- | ------- |
| Height failure | Base height < 0.22 m | Fell off the ball or collapsed |
| Orientation failure | Horizontal component of the base-frame gravity > 0.6 (about 37° tilt) | Severe tilting |
| Ball escaped | Horizontal distance from the ball center to the feet midpoint > 0.20 m | The ball rolled out from under the feet |
| Joint position fault | Any joint exceeds its limit by more than 0.5 rad, or a joint angle is non-finite | Numerical divergence or joint-limit failure |
| Joint velocity fault | Any joint velocity magnitude > 100 rad/s, or non-finite | Numerical divergence |

Reaching 20 s (1000 control steps) ends the episode with `truncated` (time limit), which is not a failure.

## Reset logic

Each episode starts from the following deterministic state plus sampled noise. Reset samples are uniform over [0, scale):

- Robot base pose: position (0, 0, 0.40) with noise of 0.02 m per XY axis and 0.005 m in Z; upright orientation with 0.05 rad Euler noise per axis;
- Robot joints: set to the default stance pose with 0.05 rad noise per joint (clipped to limits), zero joint velocity;
- Base linear velocity: zero plus uniform noise (0.1 m/s XY, 0.05 m/s Z); angular velocity: zero plus 0.2 rad/s noise;
- Basketball: ball center at (0, 0, 0.14) (resting on the ground) with 0.01 m XY noise, linear velocity noise of 0.05 m/s per XY axis, zero angular velocity.

Physical domain randomization: the environment does **not** randomize physical parameters such as mass, inertia, friction, or actuator gains. Randomization above covers the initial state only, and observation noise applies to actor observations only.
