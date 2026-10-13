**Language**: [English](README.md) | [简体中文](README.zh-CN.md)

<div align="center">

# MotrixLab

[![GitHub License](https://img.shields.io/github/license/Motphys/MotrixLab)](LICENSE)
![Python Version](https://img.shields.io/badge/python-3.10-blue)
[![Release](https://img.shields.io/github/v/release/Motphys/MotrixLab?include_prereleases)](https://github.com/Motphys/MotrixLab/releases)
[![Docs](https://img.shields.io/badge/docs-readthedocs-blue)](https://motrixlab.readthedocs.io)

**Train robot policies in simulation, then deploy them to real hardware.**

<img src="docs/source/_static/images/microduck-walk.gif" alt="Microduck robots walking in MotrixRender after training with MotrixLab" width="720">

_Microduck locomotion policies trained with MotrixLab, rendered in MotrixRender — [watch the HD video](https://github.com/user-attachments/assets/4bcf3122-f135-44cb-a966-d2d8e84479da)._

**📖 Documentation**: [简体中文](https://motrixlab.readthedocs.io/zh-cn/stable/) | [English](https://motrixlab.readthedocs.io/en/stable/)

</div>

## Contents

- [What is MotrixLab?](#what-is-motrixlab)
- [Key Features](#key-features)
- [Quick Start](#-quick-start)
- [Task Environments](#-task-environments)
- [Go2 Flat-Walk Baseline](#go2-flat-walk-baseline)
- [Go2 Flat-Walk Updated Version](#go2-flat-walk-updated-version)
- [Built-in Robot Models](#-built-in-robot-models)
- [What's Inside](#-whats-inside)
- [Contributing](#-contributing)
- [Contact](#-contact)

## What is MotrixLab?

**MotrixLab** is an open-source reinforcement learning framework for robot training, built on the high-performance [MotrixSim](https://github.com/Motphys/motrixsim-docs) physics engine. Define an environment once, train it with thousands of parallel environment instances using SKRL, RSL-RL, or the built-in FastSAC, and deploy the resulting policy to MuJoCo, native MotrixSim, or Unitree hardware — all through a single command-line interface.

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/source/_static/images/architecture-dark.svg">
    <img src="docs/source/_static/images/architecture-light.svg" alt="MotrixLab architecture: define an environment once, train it with SKRL, RSL-RL or FastSAC on thousands of parallel MotrixSim environments running on NVIDIA CUDA or AMD ROCm GPUs, then deploy the same policy artifact to MuJoCo or Unitree hardware" width="100%">
  </picture>
</div>

## Key Features

- **Unified Interface**: Provides a concise and unified reinforcement learning training and evaluation interface
- **Multi-framework Support**: Supports SKRL (JAX/PyTorch), RSLRL (PyTorch), and the built-in FastSAC implementation
- **Rich Environments**: Includes various robot simulation environments such as basic control, locomotion, and manipulation tasks
- **Sim-to-Real Deployment**: The same policy code deploys via the deploy CLI — simulation deployment to MuJoCo or native MotrixSim, hardware deployment to Unitree Go2
- **High-precision, High-performance Simulation**: Built on [MotrixSim](https://motrixsim.readthedocs.io/), a high-precision, high-performance physics engine
- **Visual Training**: Supports real-time rendering and training process visualization

## 🚀 Quick Start

### Prerequisites

| Requirement | Notes |
| --- | --- |
| Python **3.10.x** | The workspace pins `==3.10.*` |
| [uv](https://docs.astral.sh/uv/) | Python project and dependency manager — [installation guide](https://docs.astral.sh/uv/getting-started/installation/) |
| [Git LFS](https://git-lfs.com) | Robot meshes, motion data, and videos are tracked by LFS |
| OS | Linux x86_64 or Windows x86_64; the JAX training backend is Linux-only |
| GPU | NVIDIA (CUDA) or AMD (ROCm) — the matching wheels are selected automatically by `sh install.sh` |

### 1. Clone the repository

```bash
git clone https://github.com/Motphys/MotrixLab
cd MotrixLab
git lfs pull
```

### 2. Install dependencies

Linux:

```bash
sh install.sh
```

Windows (PowerShell):

```powershell
.\install.ps1
# if blocked by the execution policy:
powershell -ExecutionPolicy Bypass -File install.ps1
```

This auto-detects your GPU vendor (NVIDIA → CUDA, AMD → ROCm) and installs the runtime workspace packages with the matching PyTorch wheels, required SKRL Torch, and built-in FastSAC. Use `--gpu cuda|rocm` to override detection; `--rslrl` adds RSL-RL, while `--skrl-jax` adds the optional Linux-only SKRL JAX backend. `--skrl-torch` remains accepted as a compatibility flag because SKRL Torch is already required. Flags combine; see `sh install.sh --help`.

### 3. Train your first policy

Activate the installed environment (Windows PowerShell: `.venv\Scripts\Activate.ps1`):

```bash
source .venv/bin/activate
python scripts/train.py task=microduck-walk-flat/motrix.fastsac play=true
```

While training, the built-in dashboard shows live run progress, episode statistics, throughput, rewards, and system health:

<p align="center">
  <img src="docs/source/_static/images/train-console.png" alt="MotrixLab training dashboard for the microduck-walk-flat fastsac task" width="720">
</p>

Training runs thousands of parallel environment instances; when it finishes, the trained policy is loaded and played in the viewer automatically. Checkpoints and TensorBoard logs are saved under `runs/microduck-walk-flat/`; watch the curves with:

```bash
tensorboard --logdir runs/microduck-walk-flat
```

Training finishes in minutes: mean return and episode length typically converge after about 4,000 iterations:

<p align="center">
  <img src="docs/source/_static/images/microduck-training-curves.png" alt="TensorBoard curves of a microduck-walk-flat training run: mean return and episode length converge after about 4,000 iterations" width="720">
</p>

### 4. Replay the trained policy

Replay the latest trained policy without retraining (for example, after stopping training early with Ctrl+C):

```bash
python scripts/play.py env=microduck-walk-flat
```

A trained microduck policy replayed in the viewer:

https://github.com/user-attachments/assets/4bcf3122-f135-44cb-a966-d2d8e84479da

## 🌍 Task Environments

MotrixLab ships 50+ built-in simulation environments spanning basic control, quadruped and humanoid locomotion, whole-body motion tracking, and manipulation. The main categories:

| Preview | Category | Example environments |
| --- | --- | --- |
| <img src="docs/source/_static/images/poster/go2-walk-rough.jpg" alt="go2-walk-rough" width="240"> | Quadruped velocity tracking | `go2-walk-flat` · `go2-walk-rough` · `go1-walk-rough` · `anymalc-walk-flat` |
| <img src="docs/source/_static/images/poster/g1-walk-flat.jpg" alt="g1-walk-flat" width="240"> | Humanoid velocity tracking | `g1-walk-flat` · `k1-walk-rough` · `dex-evt-walk-flat` · `microduck-walk-flat` |
| <img src="docs/source/_static/images/poster/g1-wbt-dance.jpg" alt="g1-wbt-dance" width="240"> | Whole-body tracking (WBT) | `g1-wbt-dance` · `k1-wbt-freekick` · `g1-29dof-wbt-largebox` |

```bash
python scripts/view.py env=go2-walk-rough
```

See the [full environment gallery](https://motrixlab.readthedocs.io/en/latest/user_guide/envs/index.html) for all registered environments and their supported training algorithms.

## Go2 Flat-Walk Baseline

The first `go2-walk-flat` implementation uses MotrixSim, `QuadrupedWalkTask`, and RSL-RL PPO. The policy outputs 12 joint-position residuals with `action_scale=0.25`. The actor receives a 49-D observation; the critic receives the actor observation plus 3-D local body velocity (52-D value observation).

The observation contains gyro (3), projected gravity (3), joint-position error (12), joint velocity (12), previous action (12), velocity command (3), and four trot phase values (4). Commands are `[vx, vy, yaw]` with ranges `[-1, 1]`, `[-0.5, 0.5]`, and `[-0.5, 0.5]`; 10% are standing commands and commands are resampled every 5–10 seconds.

The reward combines linear/angular velocity tracking, vertical velocity, angular velocity, base height, action-rate, default-pose, swing-foot height, contact-phase, and swing-contact terms. Training randomizes joint/state noise, friction, mass, center of mass, PD gains, and 0–1 control-step action delay.

| Component | File |
| --- | --- |
| Go2 task configuration and registration | `motrix_envs/src/motrix_envs/locomotion/quadruped/go2.py` |
| Environment loop, observations, actions, phase, and rewards | `motrix_envs/src/motrix_envs/locomotion/quadruped/walk_np.py` |
| Command, randomization, and reward schemas | `motrix_envs/src/motrix_envs/locomotion/quadruped/cfg.py` |
| Velocity command sampling | `motrix_envs/src/motrix_envs/locomotion/quadruped/velocity_command.py` |
| Go2 model, joint order, and default pose | `motrix_robots/src/motrix_robots/unitree.py` |
| Go2 PPO recipe | `configs/task/go2-walk-flat/rslrl.ppo.yaml`, `configs/algo_base/rslrl.ppo.yaml` |
| Training entry point and RSL-RL provider | `scripts/train.py`, `motrix_rl_rslrl/src/motrix_rl_rslrl/torch/train/ppo.py` |
| Fixed-velocity evaluation | `scripts/play_walk.py`, `configs/play_walk.yaml` |
| Go2 deployment profile | `motrix_envs/src/motrix_envs/deploy/go2_walk.py` |

```bash
python scripts/train.py task=go2-walk-flat/rslrl.ppo
python scripts/play_walk.py env=go2-walk-flat policy=<checkpoint>
```

Evaluation metrics are written to `walk_analysis/velocity_metrics.json` under the run directory.

## Go2 Flat-Walk Updated Version

The updated `go2-walk-flat` configuration is registered from `quadruped/go2_walk.py`. It keeps the 49-D actor observation, 52-D critic observation, and 12-D joint action interfaces so existing PPO and deployment paths remain usable.

Changes from the first version:

- Adds single-axis and boundary command coverage with `single_axis_probability=0.45` and `boundary_probability=0.15`.
- Disables the motion-time `similar_to_default` penalty so it does not suppress leg swing.
- Adds flat-orientation, stand-still, support-foot sliding, joint-acceleration, and second-order action-smoothness terms.
- Reweights tracking and stability terms, including linear-velocity tracking `2.0` and angular-velocity tracking `0.75`.
- Uses a deterministic play configuration with command `[0.5, 0, 0]`, disabled observation noise, and no command resampling.
- Extends `play_walk.py` to repin the command every control step and write schema-v2 command tracking on `[vx, vy, yaw_rate]`, explicit linear/angular velocity metrics, yaw-rate change RMS, planar orthogonal drift, termination counts, orientation RMS, height variation, and reward-term metrics.

| Updated component | File |
| --- | --- |
| Current Go2 configuration and reward weights | `motrix_envs/src/motrix_envs/locomotion/quadruped/go2_walk.py` |
| Single-axis and boundary command sampling | `motrix_envs/src/motrix_envs/locomotion/quadruped/velocity_command.py` |
| New reward terms and action/state history buffers | `motrix_envs/src/motrix_envs/locomotion/quadruped/walk_np.py` |
| New configuration fields | `motrix_envs/src/motrix_envs/locomotion/quadruped/cfg.py` |
| Fixed-command evaluation and JSON metrics | `scripts/play_walk.py`, `configs/play_walk.yaml` |
| Algorithm mapping record | `docs/go2_rl_gym_ppo_mapping.md` |

`play.py` is the generic replay entry point and does not repin a straight-line command every step. Use `play_walk.py` for Go2 flat-walk velocity evaluation; metrics are written to `walk_analysis/velocity_metrics.json`.

### Go2 flat-walk ablations and swing regularization

The flat-walk migration keeps each gait experiment under a separate environment name, so older checkpoints retain their original interpretation and every change can be rolled back by selecting the previous environment:

| Environment | Change relative to the preceding experiment |
| --- | --- |
| `go2-walk-flat-d-legacy-pose` | Uses the legacy RL Gym Go2 default hip angles: left hips `+0.1`, right hips `-0.1`. |
| `go2-walk-flat-d-swing` | Adds rear-foot swing-width and lateral-speed regularization. |
| `go2-walk-flat-d-swing-v2` | Normalizes the rear swing penalties by explicit width and speed tolerances. |
| `go2-walk-flat-d-swing-v3` | Keeps v2 and adds a weak, one-sided front-foot over-extension penalty; normal front swing remains unconstrained. |

The v3 front-foot regularizer starts at a `0.16 m` lateral limit. It penalizes width and lateral speed only when a swinging front foot exceeds that envelope. The implementation and versioned PPO recipes are in `quadruped/go2_swing_ablation.py` and `configs/task/go2-walk-flat-d-swing*/rslrl.ppo.yaml`.

Fixed-command evaluation now reports max, P95, and peak lateral speed for both front and rear swing feet. Use identical command, seed, warmup, and `eval_steps` values when comparing checkpoints:

```bash
python scripts/play_walk.py \
  env=go2-walk-flat-d-swing-v3 \
  policy=<checkpoint> \
  forward_velocity=0.5 lateral_velocity=0 yaw_velocity=0 \
  warmup_seconds=2 eval_steps=5760 \
  analysis_dir=runs/eval/go2_d_swing_v3
```

For reproducible random-command evaluation, `eval_walk_random.py` samples `[vx, vy, yaw]` from configurable uniform ranges with a separate `command_seed` and fixed resampling interval:

```bash
python scripts/eval_walk_random.py \
  env=go2-walk-flat-d-swing-v3 \
  policy=<checkpoint> \
  num_envs=16 seed=1 rand_seed=false command_seed=1 \
  duration_seconds=120 warmup_seconds=2 command_interval_seconds=2 \
  analysis_dir=runs/eval/go2_d_swing_v3_random
```

Keep the environment matching the checkpoint when applying the same evaluator to the baseline or v1/v2 experiments. The random evaluator reports aggregate tracking and stability metrics; use fixed-command evaluation for per-foot swing analysis.

Current simulation results show that v3 reduces front-foot lateral excursion and peak lateral speed relative to v2 while preserving the rear-foot improvement. It improves aggregate random `vx`/`vy` tracking relative to the baseline, but yaw tracking, orientation RMS, and base-height variation remain trade-offs. Nonzero `vy` and yaw under a pure forward command are acceptable only as small periodic errors whose long-horizon means remain near zero. These simulation results are not evidence of real-robot readiness; real-hardware validation remains a separate gated stage.

## 🤖 Built-in Robot Models

Seven reusable robot models are registered out of the box and can be combined into any scene or task:

| Screenshot | Registry name | Type | DoF |
| --- | --- | --- | --- |
| <img src="docs/source/_static/images/robots/anymal_c.png" alt="anymal_c" width="180"> | `anymal_c` | Quadruped | 12 |
| <img src="docs/source/_static/images/robots/dex-evt.png" alt="dex-evt" width="180"> | `dex-evt` | Humanoid | 23 |
| <img src="docs/source/_static/images/robots/g1-29dof.png" alt="g1-29dof" width="180"> | `g1-29dof` | Humanoid | 29 |
| <img src="docs/source/_static/images/robots/go1.png" alt="go1" width="180"> | `go1` | Quadruped | 12 |
| <img src="docs/source/_static/images/robots/go2.png" alt="go2" width="180"> | `go2` | Quadruped | 12 |
| <img src="docs/source/_static/images/robots/k1.png" alt="k1" width="180"> | `k1` | Humanoid | 22 |
| <img src="docs/source/_static/images/robots/microduck.png" alt="microduck" width="180"> | `microduck` | Humanoid | 14 |

```bash
python scripts/view.py robot=go2
```

Use the public configuration API to compose a model directly:

```python
from motrix_robots import Microduck, UnitreeGo2Robot

microduck = Microduck()
go2 = UnitreeGo2Robot()
```

See [Supported Robots](https://motrixlab.readthedocs.io/en/latest/user_guide/robots.html) for configuration details and how to add your own model.

## 🏗️ What's Inside

MotrixLab is a [uv](https://docs.astral.sh/uv/) workspace with the following packages:

| Package | PyPI name | Description |
| --- | --- | --- |
| **motrix_deploy** | `motrix-deploy` | Artifacts, control sessions, runtime contracts, and deployment CLI |
| **motrix_deploy_mujoco** | `motrix-deploy-mujoco` | MuJoCo deployment backend plugin |
| **motrix_deploy_motrixsim** | `motrix-deploy-motrixsim` | Native MotrixSim deployment backend plugin |
| **motrix_deploy_unitree** | `motrix-deploy-unitree` | Asset-free Unitree hardware configuration, sensor/actuation wiring, and SDK2 DDS backend plugin |
| **motrix_deploy_tasks** | `motrix-deploy-tasks` | Walking tasks, deployment scenes, and installed Hydra recipes |
| **motrix_env_core** | `motrix-env-core` | Backend-agnostic environment framework, configuration, registry, and rendering |
| **motrix_env_motrixsim** | `motrix-env-motrixsim` | Live MotrixSim backend, renderer, and Torch frontend |
| **motrix_env_mujoco** | `motrix-env-mujoco` | Compile-only MuJoCo scene backend |
| **motrix_robots** | `motrix-robots` | Reusable robot configurations, default poses, and model assets |
| **motrix_envs** | `motrix-envs` | Built-in environments, task assets, data, and deployment-profile compilers |
| **motrix_rl** | `motrix-rl` | RL control plane, provider/trainer contracts, run/checkpoint handling, and discovery |
| **motrix_rl_builtin** | `motrix-rl-builtin` | Built-in Motrix FastSAC provider |
| **motrix_rl_skrl** | `motrix-rl-skrl` | Required SKRL PPO Torch provider; optional JAX extra |
| **motrix_rl_rslrl** | `motrix-rl-rslrl` | Optional RSL-RL PPO Torch provider |

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for the development environment setup, branch and commit conventions, and the configured checks (`prek`, `ruff`, `dprint`, `mypy`).

## 📬 Contact

Have questions or suggestions? Feel free to contact us through:

- GitHub Issues: [Submit Issues](https://github.com/Motphys/MotrixLab/issues)
- Discussions: [Join Discussion](https://github.com/Motphys/MotrixLab/discussions)

## Project policies

- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Support](SUPPORT.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [Changelog](CHANGELOG.md)
- [Citation metadata](CITATION.cff)
- [License](LICENSE)
