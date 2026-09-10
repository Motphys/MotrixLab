# Shadow Hand Cube Repose

## Overview

This document describes in detail the Shadow Hand dexterous manipulation cube reorientation task environment. This is a classic benchmark test in the field of robotic manipulation, requiring the robot to reorient a cube in-hand to match a randomly sampled target orientation.

```{video} /_static/videos/shadow_hand_repose.mp4
:poster: _static/images/poster/shadow_hand_repose.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

---

## Environment Description

The Shadow Hand cube reorientation task is built based on the realistic Shadow Hand 24-DOF dexterous hand, designed to train robots to rotate a cube in-hand to a target pose through fine multi-finger coordination.

### Robot Structure

Shadow Hand is a high-fidelity 24-DOF dexterous hand composed of the following main parts:

-   **Palm**: The base structure of the hand, fixed to the base
-   **5 Fingers**:
    -   **Thumb**: 5 degrees of freedom, including CMC rotation, MCP, IP joints
    -   **Index**: 4 degrees of freedom, including MCP, PIP, DIP joints
    -   **Middle**: 4 degrees of freedom, including MCP, PIP, DIP joints
    -   **Ring**: 4 degrees of freedom, including MCP, PIP, DIP joints
    -   **Little**: 4 degrees of freedom, including MCP, PIP, DIP joints
-   **Actuator Configuration**: 20 actuated joints (4 coupled joints driven by other joints)

### Finger Joint Details

Each finger (except thumb) contains:

-   **MCP Joint**: Metacarpophalangeal joint, 2 degrees of freedom (abduction/adduction + flexion)
-   **PIP Joint**: Proximal interphalangeal joint, 1 degree of freedom (flexion)
-   **DIP Joint**: Distal interphalangeal joint, 1 degree of freedom (flexion)

Thumb contains:

-   **CMC Joint**: Carpometacarpal joint, 2 degrees of freedom
-   **MCP Joint**: Metacarpophalangeal joint, 1 degree of freedom
-   **IP Joint**: Interphalangeal joint, 1 degree of freedom

### Environment Objects

-   **Cube**: 50mm \* 50mm \* 50mm cube
    -   Mass: approximately 0.028 kg
    -   Friction coefficient: 1.2
    -   Initial position: above palm center `(0.33, 0.00, 0.295)` meters
-   **Target Visualization**: Semi-transparent target pose indicator (mocap body)

### Task Goals

The robot needs to complete the following operation goals:

1. **Maintain Grasp**: Maintain stable grip of the cube in-hand
2. **Perceive Goal**: Observe target pose (visualization indicator)
3. **Fine Manipulation**: Rotate cube through multi-finger coordination
4. **Pose Alignment**: Rotate cube pose to target pose (tolerance �0.1 radian)

---

## Action Space

Action space is `Box(-1, 1, (20,), float32)`, representing position control commands applied to 20 actuated joints (normalized).

### Control Mode

The environment uses position control mode, actions are converted to joint target positions through:

```
1. Scale from [-1, 1] to actuator control range
2. Optional: action smoothing (moving average filter)
3. Clip to joint limits
4. Apply to simulator actuators
```

### Action Processing Flow

```python
# 1. Scale to actuator limits
targets = scale(actions, lower_limits, upper_limits)

# 2. Action smoothing (optional)
if act_moving_average < 1.0:
    targets = α * targets + (1 - α) * prev_actions

# 3. Clip to limits
targets = clip(targets, lower_limits, upper_limits)

# 4. Apply control
actuator_ctrls = targets
```

### Action Dimension Details

| Index | Finger | Joint | DOF | Description   |
| ----- | ------ | ----- | --- | ------------- |
| 0-4   | Thumb  | J0-J4 | 5   | CMC, MCP, IP  |
| 5-8   | Index  | J0-J3 | 4   | MCP, PIP, DIP |
| 9-12  | Middle | J0-J3 | 4   | MCP, PIP, DIP |
| 13-16 | Ring   | J0-J3 | 4   | MCP, PIP, DIP |
| 17-20 | Little | J0-J3 | 4   | MCP, PIP, DIP |

---

## Observation Space

Observation space is `Box(-inf, inf, (157,), float32)`, containing robot proprioceptive information, cube state, target state, and fingertip state.

### Observation Components

Observation vector consists of the following parts (in order):

#### 1. Hand Joint State (48 dimensions)

-   **Joint Position (24 dims)**: Unscaled raw joint angles
-   **Joint Velocity (24 dims)**: Joint angular velocities scaled by 0.2

#### 2. Cube State (17 dimensions)

-   **Position (3 dims)**: Cube position in world coordinates `(x, y, z)`
-   **Orientation (4 dims)**: Quaternion `(x, y, z, w)`
-   **Linear Velocity (3 dims)**: Cube linear velocity
-   **Angular Velocity (3 dims)**: Angular velocity scaled by 0.2
-   **Normalization Factor**: Velocity observations multiplied by `vel_obs_scale = 0.2`

#### 3. Goal State (11 dimensions)

-   **Goal Position (3 dims)**: Fixed at `(0.33, 0.00, 0.295)`
-   **Goal Orientation (4 dims)**: Randomly sampled target quaternion
-   **Relative Rotation (4 dims)**: Relative quaternion from cube to goal

#### 4. Fingertip State (65 dimensions)

State of 5 fingertips, 13 dimensions per fingertip:

-   **Position (3 dims)**: Fingertip position in Cartesian space
-   **Orientation (4 dims)**: Fingertip quaternion
-   **Velocity (6 dims)**: Linear and angular velocities

**Fingertip Link Names**:

-   `rh_ffdistal`: Index fingertip
-   `rh_mfdistal`: Middle fingertip
-   `rh_rfdistal`: Ring fingertip
-   `rh_lfdistal`: Little fingertip
-   `rh_thdistal`: Thumb fingertip

#### 5. Action History (20 dimensions)

-   Previous action values, for temporal context in the policy

### Observation Details

| Index   | Observation Content              | Dimension | Range        | Unit  |
| ------- | -------------------------------- | --------- | ------------ | ----- |
| 0-23    | Hand joint position (unscaled)   | 24        | Joint limits | rad   |
| 24-47   | Hand joint velocity (0.2)        | 24        | ±π/2         | rad/s |
| 48-50   | Cube position                    | 3         | Real         | m     |
| 51-54   | Cube orientation (quat w,x,y,z)  | 4         | Unit norm    | -     |
| 55-57   | Cube linear velocity             | 3         | Real         | m/s   |
| 58-60   | Cube angular velocity (0.2)      | 3         | Real         | rad/s |
| 61-63   | Goal position                    | 3         | Fixed        | m     |
| 64-67   | Goal orientation (quat w,x,y,z)  | 4         | Unit norm    | -     |
| 68-71   | Relative rotation (quat w,x,y,z) | 4         | Unit norm    | -     |
| 72-136  | Fingertip state (5\*13)          | 65        | -            | -     |
| 137-156 | Previous action                  | 20        | [-1, 1]      | -     |

---

## Reward Function

The reward function uses a composite design with multiple reward and penalty terms.

### Main Reward Terms

1. **Rotation Alignment Reward** (core objective)

    ```
    rot_reward = rot_reward_scale / (|rot_dist| + rot_eps)
    ```

    - **Scaling factor**: `1.0`
    - **Epsilon**: `0.1`
    - **Rotation distance calculation**: Using quaternion rotation distance formula
    - **Incentive**: Reward grows inversely as cube orientation approaches target

2. **Position Distance Penalty**

    ```
    dist_reward = dist_reward_scale � goal_dist
    ```

    - **Scaling factor**: `-10.0`
    - **Distance calculation**: Euclidean distance from cube to goal position
    - **Incentive**: Prevent cube from dropping, keep near target position

3. **Action Regularization Penalty**

    ```
    action_penalty = action_penalty_scale � ||actions||�
    ```

    - **Scaling factor**: `-0.0002`
    - **Purpose**: Encourage smooth, energy-efficient motion

### Conditional Rewards

4. **Success Reward**

    ```
    if |rot_dist| d success_tolerance:
        reward += reach_goal_bonus
    ```

    - **Reward value**: `2.0`
    - **Tolerance**: `0.1` radian (approximately 5.7�)
    - **Purpose**: Sparse reward for achieving goal alignment

5. **Drop Penalty**

    ```
    if goal_dist e fall_dist:
        reward += fall_penalty
        terminated = True
    ```

    - **Penalty value**: `0.0` (termination only, no additional penalty)
    - **Distance threshold**: `0.24` meters
    - **Purpose**: Terminate episode when cube is dropped

---

## Initial State

### Hand Initialization

**Position Initialization:**

The palm is fixed in the world coordinate system, position determined by the model file.

**Joint Angle Initialization:**

-   Use model default joint positions
-   Add uniform random noise: `[-0.2, 0.2]` radians
-   Range: All 24 hand degrees of freedom

**Velocity Initialization:**

All joint velocities are initialized to zero.

### Cube Initialization

**Position Initialization:**

-   Fixed position: `(0.33, 0.00, 0.295)` above palm center
-   Add uniform random noise: `[-0.01, 0.01]` (�1cm)

**Orientation Initialization:**

-   Use Shoemake method to generate uniformly distributed random quaternions
-   Ensure uniform sampling on SO(3) space

**Velocity Initialization:**

All linear and angular velocities are initialized to zero.

### Goal Initialization

**Position Initialization:**

-   Fixed position: `(0.33, 0.00, 0.295)` (same as cube initial position)

**Orientation Initialization:**

-   Use Shoemake method to generate uniformly distributed random target quaternions
-   Resampled each reset

---

## Termination Conditions

Episodes terminate under the following conditions:

1. **Drop Termination**: Cube distance from goal position e `fall_dist` (0.24m)
2. **Timeout Termination**: Reaching `max_episode_steps` (default 1000 steps)
3. **NaN Protection**: Detecting rotation distance or position distance as NaN

### Success Holding Mechanism

The environment uses a consecutive success counter:

-   When rotation tolerance is satisfied, counter increments
-   When `max_consecutive_successes` (50) is reached, trigger success termination and reset goal
-   Rotation tolerance: `0.1` radian

---

## Usage

### Training

```bash
python scripts/train.py task=shadow-hand-repose/skrl.ppo
```

### Policy Evaluation

```bash
python scripts/play.py env=shadow-hand-repose
```

### Environment Visualization

```bash
python scripts/view.py env=shadow-hand-repose
```

### TensorBoard

```bash
tensorboard --logdir runs/shadow-hand-repose
```

---

## Configuration Parameters

### Environment Parameters

| Parameter             | Default | Description                |
| --------------------- | ------- | -------------------------- |
| `max_episode_seconds` | 10.0    | Maximum episode length (s) |
| `ctrl_dt`             | 0.01    | Control timestep (s)       |
| `max_episode_steps`   | 1000    | Maximum episode steps      |
| `num_hand_dofs`       | 24      | Total hand DOFs            |
| `num_actuators`       | 20      | Number of actuated joints  |

### Reward Parameters

| Parameter              | Default | Description               |
| ---------------------- | ------- | ------------------------- |
| `dist_reward_scale`    | -10.0   | Position distance reward  |
| `rot_reward_scale`     | 1.0     | Rotation alignment reward |
| `rot_eps`              | 0.1     | Rotation reward epsilon   |
| `action_penalty_scale` | -0.0002 | Action regularization     |
| `success_tolerance`    | 0.1     | Success tolerance (rad)   |
| `reach_goal_bonus`     | 2.0     | Success reward            |
| `fall_dist`            | 0.24    | Drop distance threshold   |
| `fall_penalty`         | 0.0     | Drop penalty              |

### Reset Noise Parameters

| Parameter              | Default | Description                  |
| ---------------------- | ------- | ---------------------------- |
| `reset_position_noise` | 0.01    | Cube position noise (m)      |
| `reset_dof_pos_noise`  | 0.2     | Joint position noise (rad)   |
| `reset_dof_vel_noise`  | 0.0     | Joint velocity noise (rad/s) |

### Observation Scaling Parameters

| Parameter       | Default | Description                |
| --------------- | ------- | -------------------------- |
| `vel_obs_scale` | 0.2     | Velocity observation scale |

---

## References

This environment is based on the following classic works:

-   **OpenAI Dactyl** (2018): First successful in-hand manipulation sim-to-real transfer
-   **Isaac Gym** (2021): High-performance GPU-accelerated physics simulation
-   **Isaac Lab** (2023): Modular robot learning framework
