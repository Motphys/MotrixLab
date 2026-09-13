# Tutorial Overview

This section is organized to build the big picture first and unfold details
progressively; the sidebar on the left shows the full reading structure:

- **Basic Framework**: MotrixLab's layered architecture and how environments, RL
  providers, Hydra configs, and the Trainer cooperate.
- **Building Environments**: write environments first (the DirectEnv and ManagerEnv
  guides), then the two cross-cutting concepts — SceneCfg (scene and simulation config)
  and SimBackend (decoupling from the simulator); start from the overview (anatomy,
  lifecycle, workflow choice).
- **Training and Results**: from creating a Task config and running training to
  analyzing the `runs/` artifacts and checkpoints.
- **Advanced Topics**: standalone capabilities to dig into as needed — ONNX export,
  hardware deployment, the command input architecture, and custom training backends.

If you have not run your first training yet, start with the
[Getting Started](../getting_started/installation.md) guide; if you only want to
implement environments, jump straight to the
[Building Environments Overview](building_envs/index.md).
