# Motrix RL SuperMimic

[中文](README.md) | English

`motrix-rl-supermimic` brings SuperMimic-style model-based reinforcement learning to MotrixLab. It learns physical-state dynamics from real simulation rollouts, then optimizes a policy through differentiable imagined trajectories. The package targets tasks such as reference-motion tracking and is under active development.

This package provides the algorithm and training machinery. Task packages define robots, reference motions, rewards, and state representations. It is not a universal world-model interface for every model-based RL algorithm, and it does not implement Dreamer-style visual latent-state learning.

## Installation and usage

The package integrates with MotrixLab's unified CLI through [`motrix-rl-builtin`](../motrix_rl_builtin/README.md), which registers it as `motrix.supermimic`. Installing MotrixLab's builtin component also installs this package; installing this package alone does not register the algorithm with the CLI.

After installing a task package that provides the `g1-supermimic-tracking` environment and its training recipe:

```bash
mxlab train task=g1-supermimic-tracking/motrix.supermimic
```

SuperMimic training requires CUDA. The task package owns the environment registration and training recipe. See the [Motrix RL guide](../motrix_rl/README.md) for shared CLI configuration, run directories, and checkpoint usage.

Training uses asynchronous collection: a collector runs the real environment, while the learner updates the world model, policy, and critic and publishes policy snapshots back to the collector. Model and training settings use [`SuperMimicCfg`](src/motrix_rl_supermimic/config.py); collector settings live under `algo.collector`.

## Connecting your own task

Configure the real environment with [`SuperMimicEnvCfg`](src/motrix_rl_supermimic/env.py). Reuse Manager observation, reward, termination, and reset terms, and declare the physical state that the world model predicts together with its dynamics context. `SuperMimicEnv` records training transitions, so tasks do not need a separate collector implementation.

To run in imagination, the task must also be able to recompute observations, rewards, and soft termination signals from predicted states. These computations must use differentiable tensor operations, without relying on live simulator handles or undeclared sensor outputs. Tasks own their robot kinematics, coordinate conventions, and state encoding; the framework assumes no particular robot layout.

The real environment and imagined task reuse numerical terms but do not share mutable runtime state. Use `for_imagined()` to select the imagination configuration and `for_play()` to express evaluation differences. Tasks that need a residual policy on top of reference actions can provide a base-action module through `reference_action()`.

A minimal task that requires no physics engine is available as a [test example](tests/_supermimic_toy.py). It illustrates task assembly and validation and is not distributed with the runtime package.

## Scope and limitations

- **Explicit physical state:** the world model predicts low-dimensional physical state. Learning latent dynamics from pixels or point clouds is not supported.
- **Reference-motion conditioning:** goals are supplied as an observation group. Reference frames currently advance by `+1` per step, and tasks must identify valid imagination starting points.
- **Differentiable task computation:** imagined trajectories run inside the learner. They do not execute the real environment's command progression, resets, or curriculum updates.
- **Training recovery:** checkpoints preserve models, optimizers, and RNG state, but replay, simulation, and task sampling state are reinitialized. Trajectory-exact continuation is not guaranteed.

Environment stepping, automatic resets, and terminal observations follow the [core environment lifecycle](../wiki/design/architecture.md). See the [randomness design](../wiki/design/randomness-and-seeding.md) for seeding and recovery boundaries.
