# Point Mass Environment

The Point Mass environment is a simple yet fundamental 2D navigation task where an agent controls a point mass to reach a target position. This environment serves as an excellent introduction to reinforcement learning concepts and continuous action spaces.

```{video} /_static/videos/point_mass.mp4
:poster: _static/images/poster/point_mass.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

## Task Description

The Point Mass environment is a 2D navigation task. The agent needs to control a point mass by applying forces to move it to a randomly generated target position. This task requires the agent to learn efficient navigation strategies to reach the target with minimal control cost.

---

## Action Space (Action Space)

| Item          | Details                         |
| ------------- | ------------------------------- |
| **Type**      | `Box(-1.0, 1.0, (2,), float32)` |
| **Dimension** | 2                               |

Actions correspond to:

| Index | Action Meaning (Applied Force) | Min | Max | XML Name  |
| ----: | ------------------------------ | :-: | :-: | :-------: |
|     0 | x-direction force              | -1  |  1  | `x_force` |
|     1 | y-direction force              | -1  |  1  | `y_force` |

---

## Observation Space

| Item          | Details                         |
| ------------- | ------------------------------- |
| **Type**      | `Box(-inf, inf, (9,), float32)` |
| **Dimension** | 9                               |

The observation space of the Point Mass environment consists of the following components (in order):

| Component    | Description                  | Dimension | Notes |
| ------------ | ---------------------------- | --------- | ----- |
| **Position** | Point mass x, y coordinates  | 2         |       |
| **Velocity** | Point mass x, y velocities   | 2         |       |
| **Target**   | Target x, y coordinates      | 2         |       |
| **Distance** | Distance vector to target    | 2         |       |
| **Distance** | Euclidean distance to target | 1         |       |

---

## Reward Function Design

The Point Mass environment's reward function consists of the following components:

### Distance Reward

```python
# Exponential distance reward - stronger as agent gets closer
distance_reward = np.exp(-10 * dist_to_target)
```

### Target Arrival and Stay Reward

```python
# Large bonus for reaching target
target_bonus = 100.0 * in_target

# Continuous reward for staying in target
continuous_reward = 30.0 * in_target
```

### Control and Path Optimization

```python
# Penalty for distance from target center when inside target
center_penalty = np.where(in_target, 10.0 * dist_to_target, 0.0)

# Control penalty to encourage smooth movement
control_penalty = 0.1 * vel_magnitude

# Path optimization reward for straight-line movement
path_reward = 0.5 * direction_alignment
```

### Total Reward Calculation

```python
# Combine all reward components
rwd = distance_reward + target_bonus + continuous_reward + path_reward - center_penalty - control_penalty
```

---

## Initial State

-   Point mass position randomly initialized within [-1.0, 1.0]
-   Target position randomly initialized within [-1.5, 1.5]
-   Point mass velocity initialized to 0

## Episode Termination Conditions

-   Point mass reaches target and stays for 0.5 seconds
-   Simulation time reaches 10 seconds
-   Observation contains abnormal values (NaN)

---

## Usage Guide

### 1. Environment Preview

```bash
python scripts/view.py env=point_mass
```

### 2. Start Training

```bash
python scripts/train.py task=point_mass/skrl.ppo
```

### 3. View Training Progress

```bash
tensorboard --logdir runs/point_mass
```

### 4. Test Training Results

```bash
python scripts/play.py env=point_mass
```

---

## Expected Training Results

### Navigation Performance

1. Agent learns to move directly towards the target
2. Smooth movement with minimal control effort
3. Consistent target reaching within episode duration

### Learning Progress

1. Rapid initial learning phase as agent discovers basic navigation
2. Gradual refinement of control strategy
3. Stable performance across different target positions

### Behavior Characteristics

1. Efficient path planning towards target
2. Smooth approach to target center
3. Minimal overshooting or oscillatory behavior
