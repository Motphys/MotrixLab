# Franka Lift Cube

## Overview

This document describes in detail the cube grasping task environment based on the Franka Emika Panda robotic arm.

```{video} /_static/videos/franka_lift_cube.mp4
:poster: _static/images/poster/franka_lift_cube.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

---

## Environment Description

The Franka lift cube task environment is built based on the real Franka Emika Panda 7-DOF robotic arm, designed to train robots to grasp a cube on a table and lift it to a specified target position.

### Robot Structure

Franka Emika Panda is a 7-DOF robotic arm composed of the following main parts:

-   **Base**: Robot base fixed to the table
-   **7 Joints**:
    -   joint1 ~ joint4: Shoulder and arm rotation joints
    -   joint5 ~ joint7: Wrist rotation joints
-   **Gripper**: Two-finger gripper, containing two finger joints
    -   finger_joint1: Left finger joint
    -   finger_joint2: Right finger joint
-   **End Effector (TCP)**: Center point of gripper, used for grasping operations

### Task Objective

The robot needs to complete the following operation objectives:

1.  **Approach Target**: Move from initial position to cube position
2.  **Grasp Cube**: Close gripper to grasp cube
3.  **Lift Cube**: Lift cube to target height
4.  **Precise Positioning**: Move cube to specified target position (XYZ 3D coordinates)

The environment provides visualization aids:

-   **Cube**: Red cube that can be grasped, initially at random position on table
-   **Target Position**: 3D position where the cube should finally reach

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

### Joint Position Limits

All joint positions are clamped to the following ranges after execution:

| Joint   | Min     | Max     |
| ------- | ------- | ------- |
| 1       | -2.8973 | 2.8973  |
| 2       | -1.7628 | 1.7628  |
| 3       | -2.8973 | 2.8973  |
| 4       | -3.0718 | -0.0698 |
| 5       | -2.8973 | 2.8973  |
| 6       | -0.0175 | 3.7525  |
| 7       | -π/2    | π/2     |
| Gripper | 0       | 0.04    |

---

## Observation Space

The observation space is `Box(-inf, inf, (36,), float32)`, containing the robot's proprioceptive information, object state, and action history.

### Observation Components

The observation vector consists of the following parts (in order):

1.  **Joint Angles (9 dimensions)**

-   7 robot arm joint angle offsets relative to default pose
-   2 gripper joint angles

2.  **Joint Velocities (9 dimensions)**

-   Angular velocities of 9 joints

3.  **Cube Current Pose (9 dimensions)**

-   Position (3 dim): [x, y, z]
-   Quaternion (4 dim): [qx, qy, qz, qw]
-   Rotation (Euler, 2 dim): [roll, pitch]

4.  **Target Position Command (7 dimensions)**

-   Target XYZ coordinates (3 dim)
-   Target quaternion (4 dim)

5.  **Previous Action (8 dimensions)**

### Observation Details

| Index | Observation Content                             | Dimensions | Unit          |
| ----- | ----------------------------------------------- | ---------- | ------------- |
| 0-8   | Joint Angle Offsets (9 joints)                  | 9          | rad           |
| 9-17  | Joint Angular Velocities (9 joints)             | 9          | rad/s         |
| 18-26 | Cube Current Pose (position + orientation)      | 9          | rad           |
| 27-33 | Target Position Command (position + quaternion) | 7          | Dimensionless |
| 34-41 | Previous Action (8 dimensions)                  | 8          | Dimensionless |

---

## Reward Function

The reward function uses a composite design with multiple reward and penalty terms.

### Main Reward Terms

1.  **Approach Reward** (Weight: 1.5)

-   Formula: `1.5 × (1 - tanh(d_hand_cube / 0.1))`
-   Encourages robot end-effector to approach cube
-   d_hand_cube: Euclidean distance from end-effector to cube

2.  **Lifting Reward** (Weight: 30)

-   Condition: Cube height > 0.04m AND end-effector to cube distance < 0.05m
-   Encourages robot to grasp and lift cube

3.  **Target Tracking Reward** (Variable Weight)

-   **Coarse Tracking** (Weight: 10): Uses Sigmoid function, center distance 0.3m
-   **Fine Tracking** (Weight: 20): Uses tanh function, scale factor 0.4m
-   **Approach Reward** (Weight: 10): Used when distance < 0.2m, scale factor 0.05m
-   **Approach Bonus** (Weight: 200): Extra reward, encourages approaching target
-   All tracking rewards only active when cube height > 0.04m and grasp successful

### Penalty Terms

Penalty coefficients adjust with training progress:

| Penalty Term                       | Early Weight (steps < 10000) | Late Weight (steps >= 10000) |
| ---------------------------------- | ---------------------------- | ---------------------------- |
| Action Rate Penalty                | 1e-4                         | 1e-1                         |
| Joint Velocity Squared Sum Penalty | 1e-4                         | 1e-1                         |

### Calculation Formulas

```
Action Rate = ||current_action - last_action||²
Joint Velocity Squared Sum = ||joint_vel||²
```

---

## Initial State

### Robot Initialization

**Position Initialization:**

The robot's initial position in world coordinates is fixed:

-   Base position: Fixed on table
-   Joint angles: Set to default pose with random noise added

**Joint Angle Noise:**

Each joint angle has uniform random noise added in range `[-0.125, 0.125]` radians.

**Velocity Initialization:**

All linear and angular velocities are initialized to zero.

### Cube Initialization

Cube position on table is randomly sampled:

-   X coordinate: `[-0.1, 0.1]`
-   Y coordinate: `[-0.25, 0.25]`
-   Z coordinate: Fixed at 0.05 (above table)

### Target Position Generation

Target position is randomly sampled in the following range:

-   X coordinate: `[0.4, 0.6]`
-   Y coordinate: `[-0.25, 0.25]`
-   Z coordinate: `[0.25, 0.5]`

---

## Usage

### Training

```bash
python scripts/train.py task=franka-lift-cube/skrl.ppo
```

### Policy Evaluation

```bash
python scripts/play.py env=franka-lift-cube
```

### TensorBoard

```bash
tensorboard --logdir runs/franka-lift-cube
```
