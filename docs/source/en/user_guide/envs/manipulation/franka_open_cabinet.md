# Franka Open Cabinet

## Overview

This document describes in detail the cabinet opening task environment based on the Franka Emika Panda robotic arm.

```{video} /_static/videos/franka_open_cabinet.mp4
:poster: _static/images/poster/franka_open_cabinet.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

---

## Environment Description

The Franka open cabinet task environment is built based on the real Franka Emika Panda 7-DOF robotic arm, designed to train robots to approach cabinet door handles, grasp them, and pull open drawers.

### Robot Structure

Franka Emika Panda is a 7-DOF robotic arm composed of the following main parts:

-   **Base**: Robot base fixed to the ground
-   **7 Joints**:
    -   joint1 ~ joint4: Shoulder and arm rotation joints
    -   joint5 ~ joint7: Wrist rotation joints
-   **Gripper**: Two-finger gripper, containing two finger joints
    -   finger_joint1: Left finger joint, with contact pad (left_finger_pad)
    -   finger_joint2: Right finger joint, with contact pad (right_finger_pad)
-   **End Effector (TCP)**: Center point of gripper, used for grasping operations

### Environment Objects

-   **Cabinet**: Contains one openable drawer
-   **Drawer Handle (drawer_top_handle)**: Target part the robot needs to grasp
-   **Drawer Joint (drawer_top_joint)**: Sliding joint of drawer, 1 DOF

### Task Objective

The robot needs to complete the following operation objectives:

1.  **Approach Handle**: Move from initial position to drawer handle position
2.  **Pose Alignment**: Adjust end-effector pose to align with handle
3.  **Grasp Handle**: Close gripper to grasp drawer handle
4.  **Open Drawer**: Pull backward to open drawer

---

## Action Space

The action space is `Box(-inf, inf, (8,), float32)`, representing position control commands applied to 8 joints (offsets relative to current joint positions).

### Control Mode

The environment uses position control mode. Actions are converted to joint target positions as follows:

```
Target Joint Angle = Current Joint Angle + Action Value
```

### Action Dimension Details

| Index | Action Description    | Control Range | Joint Name     | Joint Type |
| ----- | --------------------- | ------------- | -------------- | ---------- |
| 0     | Joint 1 Offset        | -inf ~ inf    | joint1         | revolve    |
| 1     | Joint 2 Offset        | -inf ~ inf    | joint2         | hinge      |
| 2     | Joint 3 Offset        | -inf ~ inf    | joint3         | hinge      |
| 3     | Joint 4 Offset        | -inf ~ inf    | joint4         | hinge      |
| 4     | Joint 5 Offset        | -inf ~ inf    | joint5         | hinge      |
| 5     | Joint 6 Offset        | -inf ~ inf    | joint6         | hinge      |
| 6     | Joint 7 Offset        | -inf ~ inf    | joint7         | hinge      |
| 7     | Gripper Action (Prob) | -inf ~ inf    | finger_joint\* | hinge      |

### Gripper Control

The gripper action uses probabilistic control:

1.  **Sigmoid Mapping**: Map action value to probability in [0, 1] interval

    ```
    p = 1 / (1 + exp(-action))
    ```

2.  **Bernoulli Sampling**: Random sampling based on probability p
    -   Sample result < p: Gripper closes (0.0)
    -   Sample result >= p: Gripper opens (0.04)

---

## Observation Space

The observation space is `Box(-5, 5, (25,), float32)`, containing the robot's proprioceptive information, task-related information, and drawer state.

### Observation Components

The observation vector consists of the following parts (in order):

1.  **Joint Angles (8 dimensions)**

    -   7 robot arm joint angles (normalized to [-1, 1])
    -   Normalization formula: `2 × (Joint Angle - Lower Bound) / (Upper Bound - Lower Bound) - 1`

2.  **Joint Velocities (8 dimensions)**

    -   Angular velocities of 8 joints (divided by 2 for scaling)

3.  **Target Relative Pose (7 dimensions)**

    -   Position Offset (3 dim): Handle position - End-effector position [Δx, Δy, Δz]
    -   Orientation Offset (4 dim): Handle orientation - End-effector orientation (quaternion)

4.  **Drawer Joint Position (1 dimension)**

    -   Current open distance of drawer

5.  **Drawer Joint Velocity (1 dimension)**
    -   Current opening velocity of drawer

### Observation Details

| Index | Observation Content                         | Dimensions | Range        | Unit          |
| ----- | ------------------------------------------- | ---------- | ------------ | ------------- |
| 0-7   | Normalized Joint Angles (8 joints)          | 8          | [-1, 1]      | Dimensionless |
| 8-15  | Normalized Joint Velocities (8 joints)      | 8          | ≈[-π/2, π/2] | rad/s         |
| 16-18 | Relative Position to Handle                 | 3          | [-5, 5]      | m             |
| 19-22 | Relative Orientation to Handle (quaternion) | 4          | [-5, 5]      | Dimensionless |
| 23    | Drawer Joint Position                       | 1          | [-5, 5]      | m             |
| 24    | Drawer Joint Velocity                       | 1          | [-5, 5]      | m/s           |

All observation values are clipped to [-5, 5] range for numerical stability.

---

## Reward Function

The reward function uses a composite design with multiple reward and penalty terms.

### Main Reward Terms

1.  **Distance Reward** (Weight: 10)

    -   Formula: `10 × (1 - tanh(d_gripper_handle / 0.1))`
    -   Encourages robot end-effector to approach drawer handle
    -   d_gripper_handle: Euclidean distance from end-effector to handle

2.  **Orientation Matching Reward**

    -   Formula: Quaternion similarity function
    -   Encourages robot end-effector orientation to align with handle orientation

3.  **Gripper Close Reward** (Conditional Reward)

    -   When distance < 0.025m: Closing gripper receives +100 reward
    -   When distance >= 0.025m: Closing gripper receives -20 penalty
    -   Opening gripper: No reward (0)
    -   Encourages robot to close gripper to grasp when approaching

4.  **Open Drawer Reward** (Exponential Reward)

    -   Formula: `20 × (exp(open_dist) - 1)`
    -   open_dist: Drawer open distance (clipped to [0, 1] range)
    -   Reward grows exponentially as drawer opens more

5.  **Prevent Illegal Opening**
    -   When drawer is already open (open_dist > 0) but end-effector not contacting handle (distance > 0.03m), cancel open reward
    -   Prevents robot from using other methods to force open drawer

### Penalty Terms

1.  **Action Rate Penalty**

    -   Formula: `||current_action - last_action||²`

2.  **Joint Velocity Penalty**

    -   Formula: `||joint_vel||²`

3.  **Finger Position Penetration Penalty**
    -   Applied when finger contact pads are below handle surface
    -   Prevents finger model from penetrating drawer

### Penalty Coefficient Scheduling

Penalty coefficients adjust with training progress:

| Penalty Term           | Early Weight (steps < 8000) | Late Weight (steps >= 8000) |
| ---------------------- | --------------------------- | --------------------------- |
| Action Rate            | 1e-3                        | 2e-3                        |
| Joint Velocity Squared | 0                           | 2e-7                        |

### Termination Penalty

When termination condition is triggered, additional -10.0 penalty is applied.

---

## Initial State

### Robot Initialization

**Position Initialization:**

The robot's initial position in world coordinates is fixed:

-   Base position: Fixed on ground
-   Joint angles: Set to default pose

**Default Joint Pose:**

```
[0.0, -30°, 0°, -156°, 0.0, 186°, -45°, 0.04, 0.04] (radians)
```

**Joint Angle Noise:**

Each joint angle has uniform random noise added in range `[-0.125, 0.125]` radians.

**Velocity Initialization:**

All linear and angular velocities are initialized to zero.

### Cabinet Initialization

Cabinet is fixed on ground with drawer in closed state (joint position at 0).

---

## Usage

### Training

```bash
python scripts/train.py task=franka-open-cabinet/skrl.ppo
```

### Policy Evaluation

```bash
python scripts/play.py env=franka-open-cabinet
```

### TensorBoard

```bash
tensorboard --logdir runs/franka-open-cabinet
```
