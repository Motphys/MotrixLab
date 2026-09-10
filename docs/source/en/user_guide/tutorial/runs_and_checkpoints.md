# Training Artifacts: the runs Directory and Checkpoint Structure

This section describes the files a training run produces on disk: the directory layout under `runs/`, and the meaning of `metadata.json` and the checkpoint manifest. Understanding this structure helps you locate results, resume training, and play back policies.

## Overview

Every time you start training, the framework creates an **independent run directory** under `runs/`, grouping that run's metadata, checkpoints, and TensorBoard logs together. Playback and resume do not rely on fixed file names or directory layouts; instead they read everything from the run's `metadata.json` and `checkpoints/manifest.json`.

## Directory Structure

A run directory is organized as `environment / RL framework / training backend / algorithm / timestamp`:

```text
runs/{env_name}/{rllib}/{train_backend}/{algo}/{timestamp}/
    metadata.json                 # run metadata (source of truth for play/resume)
    checkpoints/
        manifest.json             # checkpoint manifest (registers available artifacts)
        latest.pt                 # checkpoint saved at the end of training
        model_0001000.pt          # periodic checkpoint (optional)
        model_0002000.pt
        ...
    events.out.tfevents.*         # TensorBoard logs
```

A real example:

```text
runs/g1-walk-flat/motrix/torch/fastsac/26-07-06_11-37-50-376526/
```

Meaning of each path segment:

| Path segment    | Meaning                  | Example                    |
| --------------- | ------------------------ | -------------------------- |
| `env_name`      | Environment name         | `g1-walk-flat`, `cartpole` |
| `rllib`         | RL framework / namespace | `skrl`, `rslrl`, `motrix`  |
| `train_backend` | Training backend         | `torch`, `jax`             |
| `algo`          | Provider algorithm name  | `ppo`, `fastsac`           |
| `timestamp`     | Creation timestamp       | `26-07-06_11-37-50-376526` |

The timestamp format is `%y-%m-%d_%H-%M-%S-%f`; if a collision occurs within the same microsecond, a `_1`, `_2`, ... suffix is appended to keep the directory unique.

## metadata.json

`metadata.json` sits at the run root and is the **single source of truth** by which playback, resume, export, and other tools auto-discover training artifacts — the framework uses it to select the correct RL framework, training backend, and algorithm, rather than guessing from file extensions or directory names.

```json
{
    "algo": "fastsac",
    "checkpoint_format": "pt",
    "created_at": "2026-07-06T03:37:50.376570+00:00",
    "env_name": "g1-walk-flat",
    "motrixlab_version": null,
    "rllib": "motrix",
    "seed": 1,
    "sim": null,
    "train_backend": "torch"
}
```

| Field               | Meaning                                                      |
| ------------------- | ------------------------------------------------------------ |
| `env_name`          | Environment name                                             |
| `rllib`             | RL framework name                                            |
| `train_backend`     | Training backend                                             |
| `algo`              | Algorithm name                                               |
| `sim`               | Simulator for manager environments (`null` when unspecified) |
| `seed`              | Random seed after applying CLI/config overrides              |
| `created_at`        | Creation time (UTC, ISO 8601)                                |
| `checkpoint_format` | Checkpoint storage format, e.g. `pt`, `pickle`               |
| `motrixlab_version` | Version field for records (currently `null`)                 |

## checkpoints/ and manifest.json

The actual checkpoint files live under the `checkpoints/` subdirectory, and `manifest.json` registers which files are available and what each one means. Playback and resume only trust the **artifacts** registered in the manifest; they never guess file names.

```json
{
    "version": 1,
    "artifacts": {
        "best_policy": {
            "path": "latest.pt",
            "kind": "policy",
            "format": "pt"
        },
        "latest_training_state": {
            "path": "latest.pt",
            "kind": "training_state",
            "format": "pt"
        }
    }
}
```

-   Paths in `manifest.json` are relative to the `checkpoints/` directory.
-   **`best_policy`** (`kind: policy`): used for playback / export / evaluation; this is what `play.py` loads by default.
-   **`latest_training_state`** (`kind: training_state`): used for resume, and should contain the full state needed to continue training, such as the optimizer, observation normalizer, replay buffer, and `global_step`.
-   The same physical file can be **registered as both artifacts**. For example, FastSAC's `latest.pt` (a complete state dict) can be used for both playback and resume, so `best_policy` and `latest_training_state` both point to it.

## How play and resume Use These Artifacts

-   **Playback (play)**

    -   Without `policy=...`: scans all `metadata.json` files under `runs/{env}`, picks the latest run, and takes `best_policy` from its `manifest.json`.
    -   With `policy=<file>`: walks **upward from the file's directory to locate `metadata.json`**, and uses it to select the correct inference path.

    ```bash
    # Auto-discover and play the best policy of the latest run
    python scripts/play.py env=g1-walk-flat

    # Specify a checkpoint (must be able to locate metadata.json above it)
    python scripts/play.py env=g1-walk-flat policy=/path/to/run/checkpoints/latest.pt
    ```

-   **Resume**: set `resume=` to a run directory or checkpoint path, and the framework resolves `latest_training_state` from it to continue training.

    ```bash
    python scripts/train.py task=g1-walk-flat/motrix.fastsac \
      resume=/path/to/run
    ```

## TensorBoard

TensorBoard logs (`events.out.tfevents.*`) are written directly under the run root and can be viewed per environment:

```bash
tensorboard --logdir runs/g1-walk-flat
```
