# Acrobot

Acrobot is a two-link swing-up and balance task. The goal is to swing both arms up and reach a target position using one motor torque.

```{video} /_static/videos/acrobot.mp4
:poster: _static/images/poster/acrobot.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

---

## Task Description

A two-link acrobot with one hinge joint is driven by a single motor. The motor is installed at the elbow joint, which is the only actuated joint in the system. The motor's torque rotates the rods in a plane, enabling swing-up from arbitrary initial angles and reaching a target position. Torque is limited by the actuator ctrlrange; by modulating its magnitude and direction, the policy must accumulate energy to swing up and reach the target while maintaining stability.

## Action Space

| Item          | Details                         |
| ------------- | ------------------------------- |
| **Type**      | `Box(-1.0, 1.0, (1,), float32)` |
| **Dimension** | 1                               |

---

## Observation Space

| Item          | Details                         |
| ------------- | ------------------------------- |
| **Type**      | `Box(-inf, inf, (6,), float32)` |
| **Dimension** | 6                               |

Order: `upper_arm_horizontal, lower_arm_horizontal, upper_arm_vertical, lower_arm_vertical, shoulder_velocity, elbow_velocity`.

---

## Reward Function Design

-   Base sparse reward: encourages the tip to enter the target region (radius = 0.2)
-   Continuous reward: provides 0.1 reward per step for staying in the target region
-   Distance shaping: 0.3 \* (1.0 - clip(distance / 2.0, 0, 1.0)) to encourage movement towards target
-   Velocity penalty: 0.01 \* max(0, velocity_magnitude - 2.0) to penalize excessive velocities

---

## Initial State

-   Shoulder angle randomized in `[-pi, pi]`
-   Elbow angle randomized in `[-pi, pi]`
-   Angular velocities initialized to zero

## Episode Termination Conditions

-   Episode length limited by `max_episode_seconds`
-   NaN check for observation values

---

### 1. Environment Preview

```bash
python scripts/view.py env=acrobot
```

### 2. Start Training

```bash
# Train with default parameters
python scripts/train.py task=acrobot/skrl.ppo

# Customize parallel environments
python scripts/train.py task=acrobot/skrl.ppo num_envs=1024

# Enable rendering during training
python scripts/train.py task=acrobot/skrl.ppo render=true
```

### 3. View Training Progress

```bash
tensorboard --logdir runs/acrobot
```

### 4. Test Training Results

```bash
# Auto-discover best policy (recommended)
python scripts/play.py env=acrobot

# Manually specify a checkpoint from a metadata-backed run
python scripts/play.py env=acrobot policy=/path/to/run/checkpoints/policy-file
```

> **Tip**: Policies are auto-selected from `runs/acrobot/`. Use `policy=...` to select a checkpoint whose parent run contains `metadata.json`.

---

## Configuration Parameters

### Environment Configuration

```{literalinclude} ../../../../../motrix_envs/src/motrix_envs/basic/acrobot/cfg.py
:language: python
:start-after: '# -- docs-tag-start: acrobot-env-cfg --'
:end-before: '# -- docs-tag-end: acrobot-env-cfg --'
```

### Training Configuration (PPO example)

```{literalinclude} ../../../../../configs/task/acrobot/skrl.ppo.yaml
:language: yaml
```

---

## Expected Training Results

1. Acrobot can swing up both arms to reach the target position
2. The tip can stay within the target region with stability
3. Excessive oscillations are reduced by velocity penalty
4. The policy efficiently approaches the target with smooth movements
