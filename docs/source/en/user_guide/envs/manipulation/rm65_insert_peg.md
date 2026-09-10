# RM65 Insert Peg

## Overview

This document describes the peg-insertion manipulation task implemented under `rm65_insert_peg`. The environment uses an RM65 6-DOF robotic arm with a parallel gripper. The goal is to approach the peg, grasp it stably, lift it, align it with the socket, and complete insertion. The code registers both `rm65_insert_peg` and `peg-insert`; the two names refer to the same environment and training configuration.

```{video} /_static/videos/rm65_insert_peg.mp4
:poster: _static/images/poster/rm65_insert_peg.png
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

---

## Environment Description

This task is built on an RM65 arm, a free peg, and a fixed socket scene. The examples below use `rm65_insert_peg`, while `peg-insert` remains available as a compatibility alias.

### Robot Structure

The RM65 robot in this environment contains the following major components:

-   **Base (`base_link`)**: Fixed in front of the worktable
-   **6 arm joints**: `joint_1` to `joint_6`
-   **Parallel gripper**: The main driven gripper joint is `gripper_Left_1_Joint`, while the remaining gripper joints follow through mimic linkage
-   **End effector (TCP)**: The `gripper` site, used to compute the relative pose between the tool center point and the peg
-   **Finger contact sites**: `left_finger_pad` and `right_finger_pad`, used to evaluate pre-grasp alignment, grasp symmetry, and release conditions

### Scene Objects

-   **Table**: Provides the support surface for the task scene
-   **Free peg**: `peg`, a cylindrical object with configured length `0.10 m` and radius `0.015 m`
-   **Socket assembly**: Centered on `socket_base`, with the target socket region represented by `socket`
-   **Socket parameters**: Configured socket depth `0.08 m`, nominal socket radius `0.016 m`, insertion threshold `0.045 m`

### Task Objective

The robot is expected to complete the following stages:

1. **Approach the peg**: Move the TCP and finger midpoint above the peg
2. **Prepare the grasp**: Keep the gripper open while improving lateral and height alignment
3. **Secure the grasp**: Close the gripper around the peg and continuously maintain a stable grip
4. **Transport to the socket**: Move the peg toward the socket while keeping it upright
5. **Align and insert**: Reduce the XY error, descend into the socket entry, and complete insertion

---

## Action Space

The action space is `Box(-1, 1, (7,), float32)`.

The first 6 dimensions control arm motion, and the last dimension controls gripper opening and closing.

### Control Mode

-   **Arm**: Uses incremental joint target control
    The first 6 action values are scaled by `0.025` and added to the current joint positions, then clipped to the configured joint limits
-   **Gripper**: Uses gated binary open-close control
    A raw final action greater than `0.2` requests closing. The request is accepted only when the finger midpoint is within `0.032 m` of the peg in XY, its height offset from the peg is in `(0.003, 0.05) m`, and peg uprightness is greater than `0.90`. An accepted request sets the target to `-0.91`; otherwise the target remains open at `0.0`. Once the grasp-success flag is established, the gripper is locked closed.

### Action Dimension Details

| Index | Action Description | Raw Input Range | Controlled Target      |
| ----- | ------------------ | --------------- | ---------------------- |
| 0     | Joint 1 increment  | `[-1, 1]`       | `joint_1`              |
| 1     | Joint 2 increment  | `[-1, 1]`       | `joint_2`              |
| 2     | Joint 3 increment  | `[-1, 1]`       | `joint_3`              |
| 3     | Joint 4 increment  | `[-1, 1]`       | `joint_4`              |
| 4     | Joint 5 increment  | `[-1, 1]`       | `joint_5`              |
| 5     | Joint 6 increment  | `[-1, 1]`       | `joint_6`              |
| 6     | Gripper open/close | `[-1, 1]`       | `gripper_Left_1_Joint` |

### Control Constraints

-   Control period: `ctrl_dt = 0.01s`, corresponding to 100 Hz
-   Physics simulation step: `sim.dt = 0.002s`
-   Arm action scale per step: `0.025`
-   Joint control limits are clipped to `[-2.0, 2.0]`, `[-1.5, 1.5]`, `[-1.5, 1.5]`, `[-2.0, 2.0]`, `[-1.5, 1.5]`, `[-1.57, 1.57]`
-   The gripper joint is clipped to `[-0.91, 0.0]`

---

## Observation Space

The observation space is `Box(-inf, inf, (45,), float32)`.

### Observation Components

The observation is composed of the following 5 parts:

1. **Robot joint state (16 dimensions)**
    - Relative robot joint positions with respect to the reset pose
    - Robot joint velocities used by the policy
2. **Absolute task positions (6 dimensions)**
    - Peg position
    - TCP position
3. **Hand-to-peg grasp features (14 dimensions)**
    - Direction and distance from TCP to peg
    - Direction and distance from finger midpoint to peg
    - Midpoint height offset, finger balance, finger distances, and finger gap
    - Gripper closure ratio
4. **Peg orientation and socket-relative features (7 dimensions)**
    - Peg axis direction
    - Direction and distance from peg to socket
5. **Discrete task indicators (2 dimensions)**
    - Current gripper command
    - Grasp-success flag

### Observation Dimension Details

| Index Range | Description                                 | Dimension |
| ----------- | ------------------------------------------- | --------- |
| 0-7         | Relative robot joint positions              | 8         |
| 8-15        | Robot joint velocities                      | 8         |
| 16-18       | Peg position                                | 3         |
| 19-21       | TCP position                                | 3         |
| 22-24       | Direction from TCP to peg                   | 3         |
| 25-25       | Distance from TCP to peg                    | 1         |
| 26-28       | Direction from finger midpoint to peg       | 3         |
| 29-29       | Distance from finger midpoint to peg        | 1         |
| 30-30       | Vertical offset from finger midpoint to peg | 1         |
| 31-31       | Left-right finger balance                   | 1         |
| 32-32       | Left finger to peg distance                 | 1         |
| 33-33       | Right finger to peg distance                | 1         |
| 34-34       | Finger gap                                  | 1         |
| 35-35       | Gripper closure ratio                       | 1         |
| 36-38       | Peg axis direction                          | 3         |
| 39-41       | Direction from peg to socket                | 3         |
| 42-42       | Distance from peg to socket                 | 1         |
| 43-43       | Current gripper command                     | 1         |
| 44-44       | Grasp-success flag                          | 1         |

### Observation Notes

-   Socket-relative features are only activated after a grasp has been established
-   The current gripper command is the gated actuator target, either open at `0.0` or closed at `-0.91`; it is not the raw seventh policy action
-   A grasp candidate must remain valid for at least 3 control steps before grasp success is established; the flag then stays true for the rest of the episode
-   Any `NaN` or infinite values are replaced with bounded numeric defaults before returning the observation

---

## Reward Function

The reward uses a staged composite design that encourages search, pre-grasp alignment, stable grasping, lifting, transport, and insertion.

### Main Reward Terms

1. **Peg-approach reward**

    Rewards reducing the XY error and height error between the finger midpoint and the peg while the gripper remains open.

2. **Pre-grasp and closing reward**

    Encourages the robot arm to complete pre-grasp alignment with the gripper open, rewards timely closing within the valid grasp region, and penalizes premature closing outside the grasp region, prolonged stalling in the pre-grasp region, or remaining open after entering the valid grasp channel.

3. **Grasp capture reward**

    Rewards symmetric finger placement, increasing gripper closure, stable capture over consecutive steps, and the first successful grasp event.

4. **Lift and carry reward**

    Rewards lifting the peg from the table, keeping the peg upright, maintaining the grasp, and moving toward the socket center in the horizontal plane.

5. **Insertion reward**

    ```python
    insert_reward = insert_depth * reward_cfg.insert_weight * 6.0
    depth_insert_bonus = reward_cfg.depth_insert_bonus * np.tanh(insert_depth / 0.01)
    ```

    Rewards descending toward the socket entry, increasing insertion depth, and maintaining a well-aligned pose near the socket. Insertion depth is tracked only after a grasp has been established, XY error is below `0.008 m`, and peg uprightness is greater than `0.97`.

6. **Alignment and precision reward**

    Additional rewards are provided when the grasped peg reaches the socket region, touches the socket in a valid pose, and reduces XY error to a precise insertion-ready state.

7. **Success bonus**

    The task is marked successful when:

    ```python
    success = (insert_depth > 0.045) & (xy_dist < 0.003) & (peg_uprightness > 0.98)
    ```

    A success bonus of `1500.0` is then added.

### Penalty Terms

1. **Stall and near-socket hover penalties**

    Penalize lingering too long in pre-grasp without committing, stalling while transporting toward the socket, and remaining too high or failing to descend near the socket.

2. **Premature close or weak-grasp penalties**

    Penalize closing outside the reward-defined pre-grasp region or allowing the closure-and-finger-gap grasp strength to become too weak after capture. The environment keeps the close command locked after grasp success.

3. **Knock, tilt, and shake penalties**

    Penalize knocking the peg during pre-grasp, excessive peg tilt, dropping height, and excessive peg velocity.

4. **Move-away and premature-descend penalties**

    Penalize moving away from the socket after grasp or descending before the peg is laterally aligned.

5. **Action and velocity penalties**

    Penalize action changes, robot joint velocity magnitude, and peg linear velocity.

6. **Termination penalty**

    When a termination condition is triggered, an additional `-50.0` penalty is applied. The final reward is clipped to `[-100.0, 3000.0]`.

---

## Initial State

### Robot Initialization

-   The arm starts from the default joint pose `[0.09, 0.71, 0.92, -0.18, 1.19, -0.85]`
-   The gripper starts in the open state
-   Joint reset noise is sampled uniformly with scale `0.005`
-   Robot joint velocities are initialized to zero

### Scene Initialization

-   The socket stays fixed in front of the robot
-   The peg starts upright with identity orientation
-   The peg initial position is sampled on the table with:
    -   `x` range: `0.50 ~ 0.62`
    -   `y` range: `-0.10 ~ 0.10`
    -   `z` position: `0.052 +/- 0.0015`

### Randomized Factors

At reset, the environment re-samples the following factors:

-   Robot joint reset noise
-   Peg XY position
-   Peg Z position
-   Peg-to-socket initial XY distance, constrained to `0.08 ~ 0.18 m`

---

## Episode Termination Conditions

The episode terminates early if any of the following conditions is met:

1. **Peg falls too low**
    - Threshold: `peg_z < -0.15`
2. **Peg is dropped after grasp**
    - Condition: a previous grasp was established, `hand_to_peg_dist > 0.12`, and `peg_z < 0.09`
3. **Robot joint velocity becomes too large**
    - Threshold: any joint velocity magnitude exceeds `30`
4. **Peg linear velocity becomes too large**
    - Threshold: any peg linear velocity component magnitude exceeds `15`

In addition, the maximum episode length is `3s`.

---

## Usage

The task currently provides an Async FastSAC configuration with `2048` parallel training environments, `16` evaluation environments, seed `42`, `20,000` learning iterations, and a checkpoint interval of `1,000` iterations.

### Training

```bash
python scripts/train.py task=rm65_insert_peg/motrix.fastsac
```

### Policy Evaluation

```bash
python scripts/play.py env=rm65_insert_peg num_envs=16
```

### TensorBoard

```bash
tensorboard --logdir runs/rm65_insert_peg/fastsac
```
