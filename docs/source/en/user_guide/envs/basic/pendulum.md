# Pendulum

Pendulum is a single-joint swing-up and balance task. The goal is to swing the pole up and keep it inverted using one motor torque.

```{video} /_static/videos/pendulum.mp4
:poster: _static/images/poster/pendulum.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

---

## Task Description

A single-link pendulum with one hinge joint is driven by a single motor (configurable gear). The motor’s torque rotates the rod in a plane, enabling swing-up from arbitrary initial angles, inverted balance, and maintenance. Torque is limited by the actuator ctrlrange; by modulating its magnitude and direction, the policy must accumulate energy to swing up and stabilize near the inverted position while damping angular-velocity-induced oscillations.

## Action Space

| Item          | Details                         |
| ------------- | ------------------------------- |
| **Type**      | `Box(-1.0, 1.0, (1,), float32)` |
| **Dimension** | 1                               |

---

## Observation Space

| Item          | Details                         |
| ------------- | ------------------------------- |
| **Type**      | `Box(-inf, inf, (3,), float32)` |
| **Dimension** | 3                               |

Order: `cos(theta), sin(theta), angular velocity`.

---

## Reward Function Design

-   Upright reward: encourages angle near π (inverted)
-   Energy shaping: target energy near inverted position
-   Penalties: `ang_vel^2`, `ctrl^2`, `(ctrl - prev_ctrl)^2` to reduce oscillation and aggressive actuation

---

## Initial State

-   Angle randomized in `[-pi, pi]`
-   Angular velocity small random noise (if configured)
-   Control history (`prev_ctrl`) reset to zero

## Episode Termination Conditions

-   No fall/angle termination; only NaN check
-   Episode length limited by `max_episode_seconds`

---

### 1. Environment Preview

```bash
python scripts/view.py env=pendulum
```

### 2. Start Training

```bash
# Train with default parameters
python scripts/train.py task=pendulum/skrl.ppo

# Customize parallel environments
python scripts/train.py task=pendulum/skrl.ppo num_envs=1024

# Enable rendering during training
python scripts/train.py task=pendulum/skrl.ppo render=true
```

### 3. View Training Progress

```bash
tensorboard --logdir runs/pendulum
```

### 4. Test Training Results

```bash
# Auto-discover best policy (recommended)
python scripts/play.py env=pendulum

# Manually specify a checkpoint from a metadata-backed run
python scripts/play.py env=pendulum policy=/path/to/run/checkpoints/policy-file
```

> **Tip**: Policies are auto-selected from `runs/pendulum/`. Use `policy=...` to select a checkpoint whose parent run contains `metadata.json`.

---

## Configuration Parameters

### Environment Configuration

```{literalinclude} ../../../../../motrix_envs/src/motrix_envs/basic/pendulum/cfg.py
:language: python
:start-after: '# -- docs-tag-start: pendulum-env-cfg --'
:end-before: '# -- docs-tag-end: pendulum-env-cfg --'
```

### Training Configuration (PPO example)

```{literalinclude} ../../../../../configs/task/pendulum/skrl.ppo.yaml
:language: yaml
```

---

## Expected Training Results

1. Pendulum can swing up and stay near inverted
2. Oscillation around upright is reduced by angular-velocity and control-change penalties
