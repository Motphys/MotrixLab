# RM65 Open Cabinet

## Overview

This document describes the `rm65-open-cabinet` manipulation task environment. The environment uses an RM65 6-DOF robotic arm with a parallel gripper. The goal is to approach the bottom drawer handle, establish a stable grasp, and pull the drawer open.

```{video} /_static/videos/rm65_open_cabinet.mp4
:poster: _static/images/poster/rm65_open_cabinet.png
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

---

## Environment Description

This task is built on an RM65 arm and a cabinet drawer scene. Although the registered environment name is `rm65-open-cabinet`, the current implementation actually targets the bottom drawer, using `drawer_bottom_handle` and `drawer_bottom_joint` in code.

### Robot Structure

The RM65 robot in this environment contains the following major components:

-   **Base (`base_link`)**: Fixed in front of the workspace
-   **6 arm joints**: `joint_1` to `joint_6`
-   **Parallel gripper**: The main driven gripper joint is `gripper_Left_1_Joint`, while the other 5 gripper joints follow through mimic linkage
-   **End effector (TCP)**: The `gripper` site, used to compute the relative pose between the tool center point and the drawer handle
-   **Finger contact sites**: `left_finger_pad` and `right_finger_pad`, used to evaluate handle alignment and penetration

### Scene Objects

-   **Cabinet**: Contains multiple doors and drawers
-   **Target handle**: `drawer_bottom_handle`
-   **Target joint**: `drawer_bottom_joint`, with a sliding range of `0.0 ~ 0.4 m`

### Task Objective

The robot is expected to complete the following stages:

1. **Approach the handle**: Move the TCP close to the drawer handle
2. **Align the pose**: Match the gripper pose to the handle pose
3. **Secure the grasp**: Close the gripper and maintain a stable grasp
4. **Open the drawer**: Pull the bottom drawer along its sliding direction

---

## Action Space

The action space is `Box(-inf, inf, (7,), float32)`.

The first 6 dimensions control arm joint targets, and the last dimension controls gripper opening and closing.

### Control Mode

-   **Arm**: Uses `joint_target` mode by default, with normalized target actions
    Arm actions are clipped to `[-1, 1]` first, then linearly mapped to each joint control range
-   **Gripper**: Uses `binary` mode by default
    The raw action is converted to a closing probability through a Sigmoid function, then turned into a binary open/close command with hysteresis

### Action Dimension Details

| Index | Action Description | Raw Input Range | Controlled Target      |
| ----- | ------------------ | --------------- | ---------------------- |
| 0     | Joint 1 target     | `(-inf, inf)`   | `joint_1`              |
| 1     | Joint 2 target     | `(-inf, inf)`   | `joint_2`              |
| 2     | Joint 3 target     | `(-inf, inf)`   | `joint_3`              |
| 3     | Joint 4 target     | `(-inf, inf)`   | `joint_4`              |
| 4     | Joint 5 target     | `(-inf, inf)`   | `joint_5`              |
| 5     | Joint 6 target     | `(-inf, inf)`   | `joint_6`              |
| 6     | Gripper open/close | `(-inf, inf)`   | `gripper_Left_1_Joint` |

### Control Constraints

-   Control period: `ctrl_dt = 0.025s`, corresponding to 40 Hz
-   The arm uses speed limits, acceleration limits, action delay, and first-order actuator lag by default
-   During training, arm delay, lag, speed limits, and acceleration limits are randomized per episode to improve sim-to-real robustness
-   The gripper uses hysteresis thresholds:
    -   Open-to-close threshold: `0.78`
    -   Close-to-open threshold: `0.62`
    -   Minimum switching interval: `0.25s`

---

## Observation Space

The observation space is `Box(-inf, inf, (84,), float32)`, and the final observation is clipped to `[-5, 5]`.

### Observation Components

The observation is composed of the following 4 parts:

1. **Joint positions (7 dimensions)**
    - 6 arm joints
    - 1 primary gripper joint
    - All normalized to `[-1, 1]`
2. **Joint velocities (7 dimensions)**
    - Estimated by finite differences between consecutive joint positions
    - Then divided by `2` for scaling
3. **Target relative pose (7 dimensions)**
    - Relative position from TCP to handle
    - Relative orientation from TCP to handle in quaternion form
4. **Action history (63 dimensions)**
    - Raw actions from the most recent `9` steps
    - `7` values per step, for `9 × 7 = 63` dimensions

### Observation Dimension Details

| Index Range | Description                          | Dimension |
| ----------- | ------------------------------------ | --------- |
| 0-6         | Normalized joint positions           | 7         |
| 7-13        | Joint velocities                     | 7         |
| 14-16       | Relative position from TCP to handle | 3         |
| 17-20       | Relative orientation quaternion      | 4         |
| 21-83       | Recent 9-step action history         | 63        |

### Observation Noise

Sim-to-real observation perturbations are enabled by default, including:

-   Joint position and velocity noise
-   Handle position and orientation noise
-   Persistent handle observation bias
-   Random handle observation dropout, optionally holding the previous observation on dropout

---

## Reward Function

The reward is a staged composite design that encourages approach, alignment, stable grasping, and continuous drawer opening.

### Main Reward Terms

1. **Distance reward**

    ```python
    dist_reward = 15.0 * (1 - tanh(distance / 0.4))
    ```

    Encourages the TCP to stay close to the handle.

2. **Orientation reward**

    Computed from quaternion similarity between the TCP pose and the handle pose, and only applied when the TCP is sufficiently close to the handle.

3. **Gripper closing reward**

    When the TCP is within `0.035m` and the two fingers are vertically aligned around the handle, closing the gripper is rewarded; otherwise it is penalized. This term is also scaled by the gripper closing amount.

4. **Drawer opening reward**

    ```python
    open_reward = (exp(open_dist) - 1.0) * 420.0
    ```

    This reward is only active when the robot has already grasped the handle, or has entered the grasp-maintenance phase while still staying near the handle.

5. **Open-distance delta reward**

    Provides extra reward for newly gained drawer displacement at the current step, encouraging stable and continuous pulling.

6. **Stable grasp reward**

    When the TCP is within `0.03m`, the gripper close ratio exceeds `0.7`, and this condition is maintained for `6` consecutive steps, the environment considers the handle grasped and provides persistent reward.

7. **Milestone rewards**

    - Reward `35` when drawer opening exceeds `0.15m`
    - Additional reward `70` when drawer opening exceeds `0.22m`

### Penalty Terms

1. **Slip penalty**

    If the environment has entered the grasping phase but the robot later loses the grasp while the drawer is already open, an additional penalty is applied.

2. **Finger penetration penalty**

    Applied when the finger contact points cross the upper or lower handle boundary, discouraging unrealistic penetration.

3. **Gripper switching penalty**

    Penalizes frequent gripper toggling when the TCP is near the handle.

4. **Action-change penalty**

    Penalizes the squared difference between consecutive actions.

5. **Joint-velocity penalty**

    Penalizes the squared sum of joint velocities, with a larger weight in later training.

6. **Termination penalty**

    Applies an additional `-10.0` penalty when a termination condition is triggered.

---

## Initial State

### Robot Initialization

-   The arm starts from its default zero pose
-   The gripper starts in the open state
-   No joint-position reset noise is added by default
-   All joint velocities are initialized to zero

### Scene Initialization

-   The target drawer starts fully closed
-   The drawer and the rest of the cabinet remain at their default scene poses

### Randomized Factors

At reset, the environment re-samples a subset of sim-to-real parameters, including:

-   Arm action delay
-   Arm actuator lag
-   Arm speed and acceleration limits
-   Handle observation bias

---

## Episode Termination Conditions

The episode terminates early if any of the following conditions is met:

1. **TCP moves too far behind the handle**
    - Threshold: `tcp_x - handle_x < -0.02`
2. **Joint velocity becomes too large**
    - Any robot joint velocity magnitude exceeds `3.93 rad/s`

In addition, the maximum episode length is `30s`.

---

## Usage

### Training

```bash
python scripts/train.py task=rm65-open-cabinet/skrl.ppo task.train_backend=torch
```

### Policy Evaluation

```bash
python scripts/play.py env=rm65-open-cabinet
```

### TensorBoard

```bash
tensorboard --logdir runs/rm65_open_cabinet
```
