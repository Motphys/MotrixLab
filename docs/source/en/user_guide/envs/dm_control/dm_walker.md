# 2D Walker Robot

The 2D Walker Robot (Walker2D) is a classic robot control task from DeepMind Control Suite. The goal is to achieve standing, walking, and running by controlling the robot's joints.

```{video} /_static/videos/dm_walker.mp4
:poster: _static/images/poster/dm_walker.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

## Task Description

Walker2D is a 2D planar bipedal robot with multiple joints and actuators:

-   **State Space**: Includes rotation angles and angular velocities of various robot parts, torso height and velocity, etc.
-   **Action Space**: Control torques for each joint
-   **Reward Function**: Mainly composed of maintaining standing balance and forward speed
-   **Termination Conditions**: Robot falls or joints reach limit positions

### Three Task Modes

1. **dm-stander**: Static standing task (move_speed = 0.0)

```bash
python scripts/train.py task=dm-stander/skrl.ppo
```

2. **dm-walker**: Walking task (move_speed = 1.0)

```bash
python scripts/train.py task=dm-walker/skrl.ppo
```

3. **dm-runner**: Running task (move_speed = 5.0)

```bash
python scripts/train.py task=dm-runner/skrl.ppo
```

## Quick Start

### 1. Environment Preview

```bash
python scripts/view.py env=dm-stander
python scripts/view.py env=dm-walker
python scripts/view.py env=dm-runner
```

### 2. Start Training

```bash
python scripts/train.py task=dm-stander/skrl.ppo
python scripts/train.py task=dm-walker/skrl.ppo
python scripts/train.py task=dm-runner/skrl.ppo
```

### 3. View Training Progress

```bash
tensorboard --logdir runs/dm-walker
```

### 4. Test Training Results

```bash
python scripts/play.py env=dm-stander
python scripts/play.py env=dm-walker
python scripts/play.py env=dm-runner
```

## Reward Function Design

Walker2D's reward function consists of the following components:

### Basic Standing Reward

```python
# Height reward: keep torso at target height

# Upright reward: keep torso upright
```

### Movement Reward (walking and running tasks)

```python
# Speed reward: track target speed

# Total reward = standing reward * movement weight
```

## Expected Results

1. **dm-stander**:

    - Torso height maintained in 1.0-1.4m range

2. **dm-walker**:

    - Actual walking speed close to 1.0 m/s

3. **dm-runner**:
    - Running speed reaches 4.0-5.0 m/s
