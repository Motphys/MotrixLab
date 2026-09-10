# Task Configuration and CLI Overrides

After registering an environment, create a training Task for it. A Task selects the environment, RL framework, algorithm, runtime settings, and algorithm hyperparameters.

This chapter starts by creating a runnable Task, then introduces parameter tuning, temporary CLI overrides, backend-specific configuration, and the underlying composition model.

## Create the first training Task for an environment

The following example assumes an environment registered as `my-robot` and trains it with SKRL PPO.

### Select a training method

A Task file name is formed from the RL framework and algorithm. The repository currently provides these algorithm bases:

| Training method | Task file name        |
| --------------- | --------------------- |
| SKRL PPO        | `skrl.ppo.yaml`       |
| RSLRL PPO       | `rslrl.ppo.yaml`      |
| Motrix FastSAC  | `motrix.fastsac.yaml` |

This section uses `skrl.ppo.yaml`. The other configuration structures are covered later.

### Create the Task file

Create:

```text
configs/task/my-robot/skrl.ppo.yaml
```

Add the following content:

```yaml
# @package _global_
defaults:
    - /algo_base@algo: skrl.ppo
    - _self_

task:
    env: my-robot
    rllib: skrl
    algo: ppo
    train_backend: null

num_envs: 1024
play_num_envs: 16
seed: 42

algo:
    trainer:
        timesteps: 100000
```

This configuration:

1. Loads the complete SKRL PPO base from `configs/algo_base/skrl.ppo.yaml`.
2. Selects the environment registered as `my-robot`.
3. Sets the number of parallel environments used for training and play.
4. Overrides the total training timesteps with `100000`.

A Task only needs to contain values that differ from the algorithm base. It does not need to copy every algorithm field.

### Inspect and run the Task

Print the fully composed Hydra configuration before starting training:

```bash
python scripts/train.py --cfg job --resolve task=my-robot/skrl.ppo
```

This command does not start training. After verifying `task.env`, `num_envs`, and `algo.trainer.timesteps`, start the run:

```bash
python scripts/train.py task=my-robot/skrl.ppo
```

List all Tasks available in the repository with:

```bash
python scripts/train.py --help
```

## Adjust Task runtime settings

Fields at the Task root are independent of the selected RL framework.

| Field                 | Meaning                                                                |
| --------------------- | ---------------------------------------------------------------------- |
| `num_envs`            | Number of parallel environments used for training                      |
| `play_num_envs`       | Number of environments used for post-training play or Task restoration |
| `seed`                | Random seed; it may be `null`                                          |
| `logging.backend`     | Logging backend, such as `tensorboard`                                 |
| `logging.interval`    | Logging or training-panel interval; the exact unit is trainer-specific |
| `checkpoint.interval` | Periodic checkpoint interval; `0` disables periodic checkpoints        |

For example, configure logging and checkpoints in the Task:

```yaml
num_envs: 2048
play_num_envs: 4
seed: 7

logging:
    backend: tensorboard
    interval: 20

checkpoint:
    interval: 100
```

The following entry-point fields are usually set from the CLI instead of repeated in each Task:

| Field    | Meaning                                                                                    |
| -------- | ------------------------------------------------------------------------------------------ |
| `render` | Enable interactive rendering during training                                               |
| `play`   | Load and play the policy immediately after training                                        |
| `sim`    | Simulator injected into manager-based environments; `null` resolves the registered default |
| `resume` | Run or checkpoint used to resume training; support depends on the trainer                  |

## Configure algorithm parameters

Algorithm settings live under the Task's `algo` node. The selected RL provider defines the field structure.

### SKRL PPO

SKRL PPO contains these main groups:

-   `models`: policy and value networks and model sharing.
-   `memory`: rollout memory implementation and capacity.
-   `agent`: PPO rollout, optimization, clipping, GAE, entropy, and mixed-precision settings.
-   `trainer`: total training timesteps.

For example, change the networks and learning rate:

```yaml
algo:
    models:
        policy:
            hiddens: [128, 64]
        value:
            hiddens: [128, 64]
    agent:
        learning_rate: 0.0005
        learning_epochs: 8
    trainer:
        timesteps: 20000
```

See [`configs/algo_base/skrl.ppo.yaml`](../../../../configs/algo_base/skrl.ppo.yaml) for the complete field set and per-field comments.

### RSLRL PPO

RSLRL PPO contains these main groups:

-   `num_steps_per_env` and `max_iterations`: rollout length and training iterations.
-   `obs_groups`: environment observation groups consumed by the actor and critic.
-   `actor` and `critic`: model type, hidden layers, activation, and normalization.
-   `algorithm`: PPO optimization, clipping, KL, GAE, RND, and symmetry settings.

For example, change the learning rate and training iterations:

```yaml
algo:
    max_iterations: 300
    algorithm:
        learning_rate: 0.0005
        entropy_coef: 0.005
```

See [`configs/algo_base/rslrl.ppo.yaml`](../../../../configs/algo_base/rslrl.ppo.yaml) for the complete field set and per-field comments.

### Motrix FastSAC

FastSAC uses one algorithm identity for both execution topologies. Its main fields and groups are:

-   `asynchronous`: `false` alternates collection and updates synchronously; `true` runs collector and learner processes.
-   `device`: learning device.
-   `agent`: actor/critic, C51, SAC, replay buffer, update cadence, and performance settings.
-   `trainer`: environment-interaction iterations and async-only `async_options`:
    -   `ring_capacity`: shared-memory capacity between collector and learner.
    -   `utd_mode`: update-to-data policy (`strict` or `learner_bound`).
    -   `weight_publish_interval` and `weight_poll_interval`: policy synchronization cadence.
    -   `max_ingest_per_iter` and `idle_sleep_s`: learner ingestion and idle backoff.

See [`configs/algo_base/motrix.fastsac.yaml`](../../../../configs/algo_base/motrix.fastsac.yaml) for the complete field set and per-field comments.

## Override parameters from the CLI

The MotrixLab CLI uses Hydra's `key=value` syntax. CLI values apply only to the current run and do not modify the YAML source.

### Override runtime settings

```bash
python scripts/train.py \
  task=my-robot/skrl.ppo \
  num_envs=64 \
  seed=7 \
  logging.interval=20 \
  checkpoint.interval=100
```

Enable training rendering and play the policy afterward:

```bash
python scripts/train.py task=my-robot/skrl.ppo render=true play=true
```

### Override algorithm settings

SKRL PPO:

```bash
python scripts/train.py \
  task=my-robot/skrl.ppo \
  algo.agent.learning_rate=5e-4 \
  algo.agent.learning_epochs=8 \
  algo.trainer.timesteps=20000
```

RSLRL PPO:

```bash
python scripts/train.py \
  task=cartpole/rslrl.ppo \
  algo.algorithm.learning_rate=5e-4 \
  algo.algorithm.entropy_coef=0.005 \
  algo.max_iterations=300
```

FastSAC:

```bash
python scripts/train.py \
  task=g1-walk-flat/motrix.fastsac \
  algo.asynchronous=true \
  algo.agent.actor_learning_rate=1e-4 \
  algo.agent.critic_learning_rate=3e-4 \
  algo.trainer.async_options.utd_mode=learner_bound \
  algo.trainer.num_learning_iterations=20000
```

### Override different value types

Use lowercase `true` and `false` for booleans:

```bash
python scripts/train.py task=my-robot/skrl.ppo render=true
```

Use `null` to clear nullable fields:

```bash
python scripts/train.py task=my-robot/skrl.ppo seed=null
python scripts/train.py task=my-robot/skrl.ppo algo.agent.learning_rate_scheduler=null
```

Quote lists so the shell does not interpret brackets:

```bash
python scripts/train.py \
  task=my-robot/skrl.ppo \
  'algo.models.policy.hiddens=[128,64]' \
  'algo.models.value.hiddens=[128,64]'
```

Also quote the complete `key=value` argument when a string contains spaces, parentheses, wildcard characters, or other shell metacharacters.

:::{note}
Training uses structured schemas. Existing fields do not need a `+` prefix. Add a new field to the provider dataclass and algorithm base before using it in a Task.
:::

### Verify CLI overrides

Combine CLI overrides with `--cfg job --resolve` to inspect the exact values that would be used:

```bash
python scripts/train.py \
  --cfg job \
  --resolve \
  task=my-robot/skrl.ppo \
  num_envs=64 \
  algo.agent.learning_rate=1e-3
```

The resolved configuration is saved as `task_config.yaml` in the run directory for experiment tracking and later policy loading.

## Configure different training backends

Shared SKRL Tasks normally set `task.train_backend` to `null`. MotrixLab selects an installed and available backend automatically, currently preferring JAX over Torch.

If a Task needs different hyperparameters on different backends, add backend delta files:

```text
configs/task/go2-walk-flat/skrl.ppo.yaml
configs/task/go2-walk-flat/skrl.ppo.jax.yaml
configs/task/go2-walk-flat/skrl.ppo.torch.yaml
```

A backend delta inherits the shared Task and contains only necessary differences:

```yaml
# @package _global_
defaults:
    - /task/go2-walk-flat/skrl.ppo@_global_
    - _self_

task:
    train_backend: jax

algo:
    agent:
        rollouts: 12
        learning_rate: 0.0008
```

Train with the JAX delta:

```bash
python scripts/train.py task=go2-walk-flat/skrl.ppo.jax
```

Overriding the backend directly also selects the JAX trainer:

```bash
python scripts/train.py task=go2-walk-flat/skrl.ppo task.train_backend=jax
```

This form does not load backend-specific values from `skrl.ppo.jax.yaml`. Select the `.jax` or `.torch` Task directly when a backend delta exists.

## Understand the Task configuration architecture

After creating and tuning a basic Task, the following details explain how its files are composed.

### Configuration directories

```text
configs/
├── train.yaml
├── algo_base/
│   ├── skrl.ppo.yaml
│   ├── rslrl.ppo.yaml
│   └── motrix.fastsac.yaml
└── task/
    └── <env>/
        ├── <rllib>.<algo>.yaml
        └── <rllib>.<algo>.<backend>.yaml
```

Each layer has a separate responsibility:

| Layer               | Location                                           | Purpose                                                               |
| ------------------- | -------------------------------------------------- | --------------------------------------------------------------------- |
| Entry configuration | `configs/train.yaml`                               | Defines shared runtime values and the default Task                    |
| Algorithm base      | `configs/algo_base/<rllib>.<algo>.yaml`            | Defines the complete field set and base values for an algorithm       |
| Shared Task         | `configs/task/<env>/<rllib>.<algo>.yaml`           | Selects the environment and algorithm and stores task-specific tuning |
| Backend delta       | `configs/task/<env>/<rllib>.<algo>.<backend>.yaml` | Stores backend-specific differences                                   |
| CLI override        | `key=value`                                        | Overrides the final value for one run                                 |

The effective precedence is:

```text
algorithm base → shared Task → backend delta → CLI override
```

### Task option naming

The training command uses this Task option format:

```text
task=<env>/<rllib>.<algo>[.<backend>]
```

For example:

```text
configs/task/cartpole/skrl.ppo.yaml
```

is selected with:

```bash
task=cartpole/skrl.ppo
```

Task metadata should match that option:

| Field                | Meaning                                                   |
| -------------------- | --------------------------------------------------------- |
| `task.env`           | Registered environment name                               |
| `task.rllib`         | RL framework name                                         |
| `task.algo`          | Algorithm or training implementation name                 |
| `task.train_backend` | Requested training backend; `null` enables auto-selection |

### Hydra composition directives

A shared Task normally starts with:

```yaml
# @package _global_
defaults:
    - /algo_base@algo: skrl.ppo
    - _self_
```

-   `# @package _global_` merges the Task into the training root.
-   `/algo_base@algo: skrl.ppo` loads the algorithm base under the root `algo` field.
-   `_self_` applies the Task after the base, allowing Task values to override base values.

Hydra validates field names and types against the structured schema registered by the provider. Misspelled fields, incompatible types, and missing required values fail before training starts.

## Play and View overrides

`scripts/play.py` and `scripts/view.py` use the same `key=value` syntax:

```bash
python scripts/view.py env=cartpole num_envs=4
python scripts/play.py env=cartpole num_envs=1
python scripts/play.py policy=/path/to/checkpoint.pt num_envs=1
```

Play loads the `task_config.yaml` stored with a training run. To temporarily override algorithm settings, use `rl` as the algorithm root. Since `rl` starts as an empty mapping, add the path with `+`:

```bash
python scripts/play.py \
  env=cartpole \
  '+rl.agent.learning_rate=1e-4'
```

The `rl` mapping is merged into the stored `algo` node, and its field path depends on the algorithm used to train the policy.
