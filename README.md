**Language**: [English](README.md) | [简体中文](README.zh-CN.md)

# MotrixLab

![GitHub License](https://img.shields.io/github/license/Motphys/MotrixLab)
![Python Version](https://img.shields.io/badge/python-3.10-blue)

`MotrixLab` is a reinforcement learning framework based on the [MotrixSim](https://github.com/Motphys/motrixsim-docs) simulation engine, designed specifically for robot simulation and training. This project provides a complete reinforcement learning development platform that integrates multiple simulation environments and training frameworks.

## Project Overview

The project is divided into two core components:

-   **motrix_envs**: Various RL simulation environments built on MotrixSim, defining observation, action, and reward. Framework-agnostic and currently supports MotrixSim's CPU backend
-   **motrix_rl**: Integrates RL frameworks and uses various environment parameters from motrix_envs for training. Currently supports SKRL framework (JAX/PyTorch) and RSLRL framework (PyTorch) PPO algorithms

> Documentation: https://motrixlab.readthedocs.io

## Key Features

-   **Unified Interface**: Provides a concise and unified reinforcement learning training and evaluation interface
-   **Multi-framework Support**: Supports SKRL (JAX/PyTorch) and RSLRL (PyTorch) training frameworks with flexible selection based on hardware environment
-   **Rich Environments**: Includes various robot simulation environments such as basic control, locomotion, and manipulation tasks
-   **High-performance Simulation**: Built on MotrixSim's high-performance physics simulation engine
-   **Visual Training**: Supports real-time rendering and training process visualization

## 🚀 Quick Start

> The following examples use the Python project management tool: [UV](https://docs.astral.sh/uv/)
>
> Before starting, please [install](https://docs.astral.sh/uv/getting-started/installation/) this tool.

### Clone Repository

```bash
git clone https://github.com/Motphys/MotrixLab

cd MotrixLab

git lfs pull
```

### Install Dependencies

Install all dependencies:

```bash
uv sync --all-packages --all-extras
```

SKRL framework supports JAX(Flax) or PyTorch as training backends. You can also choose to install only one training backend based on your hardware environment:

Install JAX as training backend (Linux only):

```bash
uv sync --all-packages --extra skrl-jax
```

Install PyTorch as training backend:

```bash
uv sync --all-packages --extra skrl-torch
```

Install RSLRL framework (PyTorch backend only):

```bash
uv sync --all-packages --extra rslrl
```

## 🎯 Usage Guide

### Environment Visualization

View environments without executing training:

```bash
uv run scripts/view.py --env cartpole
```

### Model Training

Train with SKRL framework (default):

```bash
uv run scripts/train.py --env cartpole
```

Train with RSLRL framework:

```bash
uv run scripts/train.py --env cartpole --rllib rslrl
```

Training results are saved in the `runs/{env-name}/` directory.

View training data through TensorBoard:

```bash
uv run tensorboard --logdir runs/{env-name}
```

### Model Inference

```bash
uv run scripts/play.py --env cartpole
```

For more usage methods, please refer to the [User Documentation](https://motrixlab.readthedocs.io)

## 📬 Contact

Have questions or suggestions? Feel free to contact us through:

-   GitHub Issues: [Submit Issues](https://github.com/Motphys/MotrixLab/issues)
-   Discussions: [Join Discussion](https://github.com/Motphys/MotrixLab/discussions)


## Citation

If you use MotrixLab in your research, please cite it as:

```bibtex
@software{motrixlab2026,
  title  = {MotrixLab: A Reinforcement Learning Framework for Robot Simulation},
  author = {{Motphys Team}},
  year   = {2026},
  url    = {https://motrixlab.readthedocs.io/},
  note   = {Source code available at GitHub - Motphys/MotrixLab: A general-purpose machine learning architecture designed for robot train}
}
```

MotrixLab is built on MotrixSim. If your work also uses MotrixSim directly, please also cite:

```bibtex
@software{motrixsim2026,
  title  = {MotrixSim: A Physics Simulation Engine for Robotics and Embodied AI},
  author = {{Motphys Team}},
  year   = {2026},
  url    = {https://motrixsim.readthedocs.io/},
  note   = {Python binary package}
}
```
