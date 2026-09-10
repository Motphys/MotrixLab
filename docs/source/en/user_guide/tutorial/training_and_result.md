# Training Execution and Result Analysis

This section introduces how to execute reinforcement learning training and how to analyze and use training results. For the layout of training artifacts (the `runs/` directory, `metadata.json`, checkpoints), see [Training Artifacts: the runs Directory and Checkpoint Structure](runs_and_checkpoints.md).

## Start Training

### Selecting a Task

The training entry point uses Hydra's `task=<environment>/<framework>.<algorithm>[.<backend>]` option. A Task selects the environment, RL provider, runtime settings, and algorithm hyperparameters as one reproducible recipe:

```bash
# Train the default Cartpole SKRL PPO Task
python scripts/train.py task=cartpole/skrl.ppo

# Select another framework or algorithm
python scripts/train.py task=cartpole/rslrl.ppo
python scripts/train.py task=g1-walk-flat/motrix.fastsac
python scripts/train.py task=g1-walk-flat/motrix.fastsac algo.asynchronous=false
```

Built-in RL methods and their training backends:

| Task suffix      | Training backend | Description                                                 |
| ---------------- | ---------------- | ----------------------------------------------------------- |
| `skrl.ppo`       | `jax` / `torch`  | SKRL PPO                                                    |
| `rslrl.ppo`      | `torch`          | RSLRL PPO                                                   |
| `motrix.fastsac` | `torch`          | FastSAC; `algo.asynchronous` selects the execution topology |

Run `python scripts/train.py --help` to list the Tasks available in the current checkout. See [Task Configuration and CLI Overrides](training_environment_config.md) for the Task file layout and override rules.

### Selecting Training and Simulation Backends

```bash
# Override the training backend (auto-selected when task.train_backend is null)
python scripts/train.py task=cartpole/skrl.ppo task.train_backend=jax
python scripts/train.py task=cartpole/skrl.ppo task.train_backend=torch

# Specify the simulator injected into manager-based environments
python scripts/train.py task=g1-wbt-dance sim=motrixsim
```

### Training Scale and Random Seed

```bash
# Number of parallel environments
python scripts/train.py task=cartpole/skrl.ppo num_envs=1024

# Fixed random seed (reproducible) / choose a random seed at runtime
python scripts/train.py task=cartpole/skrl.ppo seed=42
python scripts/train.py task=cartpole/skrl.ppo seed=null
```

```{note}
Hydra can override typed algorithm fields directly. For example: `algo.agent.learning_rate=5e-4`. Field paths depend on the selected Task; the complete defaults are in `configs/algo_base/`.
```

### Auto-play and Resume

```bash
# After training finishes successfully, play the best policy of this run
python scripts/train.py task=g1-walk-flat/motrix.fastsac play=true

# Resume from a run directory or a checkpoint
python scripts/train.py task=g1-walk-flat/motrix.fastsac \
  resume=/path/to/run

# Enable rendering to monitor the training process
python scripts/train.py task=cartpole/skrl.ppo render=true
```

### Common Hydra Overrides

| Override             | Description                                                               | Default source        |
| -------------------- | ------------------------------------------------------------------------- | --------------------- |
| `task=...`           | Environment, framework, algorithm, backend delta                          | `cartpole/skrl.ppo`   |
| `task.train_backend` | Training backend (`jax` / `torch`)                                        | Selected Task         |
| `sim`                | Simulator for manager environments; leave unset for np-style environments | `null` / auto-select  |
| `num_envs`           | Number of parallel training environments                                  | Selected Task         |
| `seed`               | Fixed seed; `null` chooses one at runtime                                 | Selected Task         |
| `resume`             | Run directory or checkpoint to resume from                                | `null`                |
| `play`               | Play the best policy after training                                       | `false`               |
| `render`             | Enable interactive rendering                                              | `false`               |
| `algo.*`             | Provider-owned typed algorithm settings                                   | Algorithm base + Task |
| `algo.asynchronous`  | Select synchronous or asynchronous FastSAC                                | Selected Task         |
| `logging.*`          | Logging backend and interval                                              | Training root + Task  |
| `checkpoint.*`       | Periodic checkpoint policy                                                | Training root + Task  |

## Training Process Monitoring

### TensorBoard Monitoring

TensorBoard logs are written under the run directory and can be viewed per environment:

```bash
tensorboard --logdir runs/cartpole
```

Besides the standard return and loss curves, if an environment exposes per-term rewards via `info["Reward"]`, they are also logged to TensorBoard during training.

## Model Evaluation and Testing

Playback (play) does not require re-specifying the RL method — the correct `rllib / train_backend / algo` is read automatically from the run's `metadata.json`.

```bash
# Auto-discover and play the best policy of the latest run (recommended)
python scripts/play.py env=cartpole

# Specify a checkpoint (must be able to locate metadata.json above it)
python scripts/play.py env=g1-walk-flat \
  policy=/path/to/run/checkpoints/latest.pt

# Specify the number of playback environments
python scripts/play.py env=cartpole num_envs=100
```

```{note}
Without `policy=...`, the system scans all `metadata.json` files under `runs/{env}/`, picks the latest run, and loads the best policy from its `checkpoints/manifest.json`. See [Training Artifacts: the runs Directory and Checkpoint Structure](runs_and_checkpoints.md) for details.
```
