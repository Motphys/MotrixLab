# Basic Framework

MotrixLab separates environment implementation, training methods, configuration, and command-line orchestration. This section explains how those pieces fit together before you build a custom environment or training backend.

## Repository Layers

```text
MotrixLab/
├── motrix_envs/                 # Environment configs, implementations, and registry
│   └── src/motrix_envs/
├── motrix_rl/                   # RL frameworks, providers, trainers, and run artifacts
│   └── src/motrix_rl/
├── configs/
│   ├── algo_base/               # Complete typed defaults for each RL provider
│   └── task/<env>/              # Per-environment training recipes
└── scripts/
    ├── train.py                 # Hydra training entry point
    ├── play.py                  # Metadata-backed policy playback
    └── view.py                  # Random-action environment preview
```

The main runtime flow is:

```text
task=<env>/<framework>.<algorithm>
                 │
                 ▼
Hydra composes root config + algorithm base + Task + CLI overrides
                 │
                 ▼
runner resolves an AgentProvider and creates a Trainer
                 │
                 ▼
Trainer creates the registered environment and executes train/play
                 │
                 ▼
runs/... stores metadata, resolved Task config, and checkpoints
```

## Core Components

### Environment Layer

An environment normally consists of:

-   An `EnvCfg` dataclass registered with `@registry.envcfg("name")`.
-   An environment implementation registered with `@registry.env("name")`.
-   Task logic for observations, rewards, termination, reset, and action application.

The environment registry owns environment names and simulation-backend implementations. `scripts/view.py`, trainers, and playback all create environments through this same registry.

### RL Framework and Provider Layer

`RlFramework` defines an RL framework namespace such as `skrl`, `rslrl`, or `motrix`. Each framework contains one or more `AgentProvider` implementations. A provider declares:

-   Its algorithm name, such as `ppo` or `fastsac`.
-   Its training backend, such as `jax` or `torch`.
-   The typed algorithm configuration schema it accepts.
-   Its checkpoint format and how to create a trainer.

Frameworks and providers are registered in Python because they represent executable capabilities. See [Adding a Custom Training Backend](custom_training_backend.md) for the extension interface.

### Hydra Configuration Layer

Training values live in YAML rather than Python Task subclasses:

-   `configs/algo_base/<framework>.<algorithm>.yaml` supplies the complete provider-owned algorithm defaults.
-   `configs/task/<env>/<framework>.<algorithm>.yaml` selects an environment and stores task-specific tuning.
-   An optional `.<backend>.yaml` Task contains only backend-specific differences.
-   CLI `key=value` arguments apply temporary overrides after composition.

The provider's dataclass schema validates field names and types, while YAML remains the source of truth for values. Task files are discovered by scanning `configs/task/`; there is no RL configuration decorator or Python Task registry.

### Runner and Trainer Layer

The shared runner handles framework-neutral orchestration:

1. Read `task.env`, `task.rllib`, `task.algo`, and `task.train_backend` from the composed config.
2. Resolve a compatible provider and training backend.
3. Create a run directory and write `metadata.json` plus `task_config.yaml`.
4. Build a `TrainerContext` and ask the provider to create its trainer.
5. Execute training or playback and register checkpoint artifacts.

The trainer owns framework-specific model construction, optimization, checkpoint serialization, and inference. It should use the environment registry instead of coupling itself to a concrete environment class.

## Training Workflow

For example:

```bash
python scripts/train.py task=cartpole/skrl.ppo num_envs=1024
```

This command performs the following steps:

1. Hydra composes `configs/train.yaml`, `configs/algo_base/skrl.ppo.yaml`, and `configs/task/cartpole/skrl.ppo.yaml`.
2. `num_envs=1024` overrides the composed Task value for this run only.
3. The runner resolves the SKRL PPO provider and an available JAX or Torch backend.
4. The trainer creates the registered `cartpole` environment and starts optimization.
5. Run metadata, the resolved Task snapshot, logs, and checkpoint manifests are written under `runs/cartpole/`.

## Multi-Framework Support

The same environment can have multiple Task recipes without changing its implementation:

```bash
python scripts/train.py task=cartpole/skrl.ppo
python scripts/train.py task=cartpole/rslrl.ppo
```

SKRL supports JAX and Torch providers, RSLRL uses Torch, and `motrix.fastsac` selects its synchronous or asynchronous Torch trainer through `algo.asynchronous`. The selected Task and provider determine the algorithm configuration and output metadata.

## Why This Separation Matters

1. **Environment reuse**: one registered environment can be trained by multiple RL frameworks.
2. **Typed configuration**: provider schemas reject misspelled or incompatible YAML/CLI values before training.
3. **Reproducibility**: each run stores the resolved Task configuration and provider identity.
4. **Extensibility**: new environments add registry entries and Task YAML; new RL integrations add providers and trainers.
5. **Consistent artifacts**: playback and resume use metadata and checkpoint manifests instead of guessing file names.
