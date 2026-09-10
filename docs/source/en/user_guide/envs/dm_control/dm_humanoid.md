# 3D Humanoid Robot

The 3D Humanoid Robot (Humanoid) is a classic bipedal locomotion task from DeepMind Control Suite. The goal is to train a simulated 3D humanoid robot to achieve standing, walking, and running by controlling joint torques.

```{video} /_static/videos/dm_humanoid_run.mp4
:poster: _static/images/poster/dm_humanoid_run.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

## Task Description

Humanoid is a 3D bipedal humanoid robot task. The robot consists of a head, torso, two arms, and two legs, with 21 controlled joints (actuators). The agent controls the robot by applying torques to these joints to achieve standing balance, forward walking, or fast running. This task requires coordinated bipedal gait, balance control, and 3D spatial posture stability.

---

## Action Space

| Item     | Details                          |
| -------- | -------------------------------- |
| **Type** | `Box(-1.0, 1.0, (21,), float32)` |
| **Dim**  | 21                               |

Actions correspond to the following:

| Index | Action (Joint Torque)          | Min  | Max | XML Name          |
| ----: | ------------------------------ | :--: | :-: | :---------------- |
|     0 | Abdomen Y-axis rotation torque | -1.0 | 1.0 | `abdomen_y`       |
|     1 | Abdomen Z-axis rotation torque | -1.0 | 1.0 | `abdomen_z`       |
|     2 | Abdomen X-axis rotation torque | -1.0 | 1.0 | `abdomen_x`       |
|     3 | Right hip X-axis torque        | -1.0 | 1.0 | `right_hip_x`     |
|     4 | Right hip Z-axis torque        | -1.0 | 1.0 | `right_hip_z`     |
|     5 | Right hip Y-axis torque        | -1.0 | 1.0 | `right_hip_y`     |
|     6 | Right knee torque              | -1.0 | 1.0 | `right_knee`      |
|     7 | Right ankle X-axis torque      | -1.0 | 1.0 | `right_ankle_x`   |
|     8 | Right ankle Y-axis torque      | -1.0 | 1.0 | `right_ankle_y`   |
|     9 | Left hip X-axis torque         | -1.0 | 1.0 | `left_hip_x`      |
|    10 | Left hip Z-axis torque         | -1.0 | 1.0 | `left_hip_z`      |
|    11 | Left hip Y-axis torque         | -1.0 | 1.0 | `left_hip_y`      |
|    12 | Left knee torque               | -1.0 | 1.0 | `left_knee`       |
|    13 | Left ankle X-axis torque       | -1.0 | 1.0 | `left_ankle_x`    |
|    14 | Left ankle Y-axis torque       | -1.0 | 1.0 | `left_ankle_y`    |
|    15 | Right shoulder 1 torque        | -1.0 | 1.0 | `right_shoulder1` |
|    16 | Right shoulder 2 torque        | -1.0 | 1.0 | `right_shoulder2` |
|    17 | Right elbow torque             | -1.0 | 1.0 | `right_elbow`     |
|    18 | Left shoulder 1 torque         | -1.0 | 1.0 | `left_shoulder1`  |
|    19 | Left shoulder 2 torque         | -1.0 | 1.0 | `left_shoulder2`  |
|    20 | Left elbow torque              | -1.0 | 1.0 | `left_elbow`      |

---

## Observation Space

| Item     | Details                          |
| -------- | -------------------------------- |
| **Type** | `Box(-inf, inf, (73,), float32)` |
| **Dim**  | 73                               |

The observation space of the Humanoid environment consists of the following components (in order):

| Component          | Description                             | Dim | Notes                                                                 |
| ------------------ | --------------------------------------- | --- | --------------------------------------------------------------------- |
| **joint_angles**   | Joint angles (excluding root's 7 DOF)   | 22  | Angles of 22 joints                                                   |
| **head_height**    | Head height                             | 1   | Height of head relative to ground                                     |
| **extremities**    | Extremity positions (relative to torso) | 12  | Left hand, left foot, right hand, right foot (3D each, in that order) |
| **torso_vertical** | Torso vertical direction vector         | 3   | Vertical direction in local coordinates                               |
| **com_vel**        | Center of mass linear velocity          | 3   | Linear velocity of torso subtree                                      |
| **qvel**           | Velocities of all joints and root       | 29  | Including root's 6 DOF                                                |
| **target_local**   | Target direction (local coordinates)    | 3   | Target direction in torso local frame                                 |

---

## Reward Function Design

The Humanoid reward function varies according to task type (standing, walking, running), but all include the following core components:

### Posture Reward

```python
# Head height reward: keep head above target height (95% of stand_height, ~1.33m)
stand_reward = tolerance(head_height, bounds=(stand_height * 0.95, inf), margin=0.5)

# Torso upright reward: keep torso upright
upright_reward = tolerance(torso_upright, bounds=(0.9, inf), sigmoid="linear", margin=0.9)

# Pelvis height reward: keep pelvis at reasonable height (60% of stand_height, ~0.84m)
pelvis_height_reward = tolerance(
    pelvis_height, bounds=(stand_height * 0.6, inf), sigmoid="linear", margin=stand_height * 0.6
)

# Posture reward = head height reward × torso upright reward × pelvis height reward
posture_reward = stand_reward * upright_reward * pelvis_height_reward
```

### Speed Reward

The speed reward calculation differs by task type:

**Standing Task (move_speed <= 0)**:

```python
# Speed reward: maintain near-zero speed
speed_reward = tolerance(actual_speed, bounds=(0, 0), margin=1.0, value_at_margin=0.01)
```

**Walking Task (0 < move_speed <= 3.0)**:

```python
# Speed reward: achieve target speed (default 1.0 m/s) in target direction (positive X-axis)
actual_speed = dot(com_vel[:2], target_direction[:2])  # Projection of velocity onto target direction
speed_reward = tolerance(actual_speed, bounds=(move_speed, move_speed), margin=move_speed, sigmoid="linear")
```

**Running Task (move_speed > 3.0)**:

```python
# Speed reward: achieve target speed (default 10.0 m/s) or above in target direction
actual_speed = dot(com_vel[:2], target_direction[:2])
speed_reward = tolerance(actual_speed, bounds=(move_speed, inf), margin=move_speed, sigmoid="linear")
```

### Energy Reward

```python
energy_reward = exp(-energy_coef * mean(ctrls ^ 2))
```

### Gait Reward

```python
# Torso heading reward: torso faces target direction
torso_heading_reward = tolerance(dot(torso_forward, target_dir), bounds=(0.9, 1.0), margin=0.3, sigmoid="linear")

# Head heading reward: head faces target direction
head_heading_reward = tolerance(dot(head_forward, target_dir), bounds=(0.9, 1.0), margin=0.3, sigmoid="linear")

# Pelvis yaw reward: pelvis faces target direction
pelvis_yaw_reward = tolerance(dot(pelvis_forward, target_dir), bounds=(0.9, 1.0), margin=0.3, sigmoid="linear")

# Pelvis level reward: pelvis remains level
pelvis_level_reward = tolerance(pelvis_up, bounds=(0.9, 1.0), margin=0.3, sigmoid="linear")

# Feet height reward: feet stay close to ground
feet_height_reward = tolerance(max_foot_height, bounds=(0.0, 0.3), margin=0.5, sigmoid="quadratic")

# Gait reward = product of all heading and posture rewards
gait_reward = torso_heading_reward * head_heading_reward * pelvis_yaw_reward * pelvis_level_reward * feet_height_reward
```

### Total Reward

```python
total_reward = posture_reward * speed_reward * energy_reward * gait_reward
```

---

## Initial State

-   **Robot Position**: Torso initial height is 1.33 meters (95% of standard stand height)
-   **Robot Orientation**: Torso remains upright, quaternion set to (1.0, 0.0, 0.0, 0.0)
-   **Joint Angles**: Randomly initialized within joint limits
    -   Torso/hip base joints: Randomized in small range (±15 degrees)
    -   Leg joints: Symmetrically initialized, ensuring left and right legs are independently randomized, knees initially bent
    -   Arm joints: Symmetrically initialized, using middle 80% of joint limit ranges
-   **Initial Velocities**: All joint velocities and linear velocities initialized to small random values near zero (-0.01 to 0.01)
-   **Initial Controls**: All actuator controls initialized to small random values near zero (-0.02 to 0.02)

## Episode Termination Conditions

-   Robot state observations contain abnormal values (NaN or Inf)
-   Head too low: Head height below 50% of standard stand height (0.7 meters)
-   Torso too tilted: Torso vertical component less than 0.2 (severe torso tilt)
-   Extreme velocity: Absolute value of any joint velocity exceeds 200.0 rad/s or m/s
-   Maximum episode duration: 25 seconds

---

## Usage Guide

### 1. Environment Preview

```bash
python scripts/view.py env=dm-humanoid-stand
python scripts/view.py env=dm-humanoid-walk
python scripts/view.py env=dm-humanoid-run
```

### 2. Start Training

```bash
python scripts/train.py task=dm-humanoid-stand/skrl.ppo
python scripts/train.py task=dm-humanoid-walk/skrl.ppo
python scripts/train.py task=dm-humanoid-run/skrl.ppo
```

### 3. View Training Progress

```bash
tensorboard --logdir runs/dm-humanoid-walk
```

### 4. Test Training Results

```bash
python scripts/play.py env=dm-humanoid-stand
python scripts/play.py env=dm-humanoid-walk
python scripts/play.py env=dm-humanoid-run
```

---

## Expected Training Results

### Standing Task (dm-humanoid-stand)

1. Head height maintained in 1.3-1.5m range
2. Torso upright angle deviation less than 15 degrees
3. Able to stand stably without falling
4. Speed near zero, no significant movement

### Walking Task (dm-humanoid-walk)

1. Actual walking speed close to 1.0 m/s
2. Coordinated gait, no obvious falls
3. Able to walk continuously and stably
4. Torso and head facing target direction

### Running Task (dm-humanoid-run)

1. Running speed reaches 5.0-10.0 m/s
2. Flight phase appears (both feet off ground simultaneously)
3. Coordinated and stable gait
4. Able to maintain high-speed running posture
