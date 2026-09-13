# Adding a Custom Training Backend

This guide explains how to integrate a new reinforcement learning framework, algorithm, or training backend with MotrixLab. The integration keeps the shared training entry point, run directory layout, checkpoint metadata, and `TrainResult.play()` behavior.

## Choose the Three Identity Names

Every training implementation is identified by three names:

| Name            | Declared by                   | Purpose                | Examples                 |
| --------------- | ----------------------------- | ---------------------- | ------------------------ |
| `rllib`         | `RlFramework.name`            | RL framework namespace | `skrl`, `rslrl`, `myrl`  |
| `algo`          | `AgentProvider.agent_name`    | Algorithm or agent     | `ppo`, `fastsac`, `sac`  |
| `train_backend` | `AgentProvider.train_backend` | Training backend       | `jax`, `torch`, `custom` |

For example, `task=cartpole/myrl.ppo` loads a Task with `rllib=myrl` and `algo=ppo`; `task.train_backend=torch` selects the Torch provider.

## Step 1: Implement `RlFramework`

`RlFramework` is the registration entry point. It declares the framework name and lists the agents the framework supports. Here, an agent means an RL implementation such as PPO or SAC, not a robot in the simulation environment.

```python
from motrix_rl import frameworks
from motrix_rl.frameworks import RlFramework


class MyRlFramework(RlFramework):
    def __init__(self) -> None:
        # List every agent implementation provided by this framework.
        # Each provider represents one (algo, train_backend) pair.
        super().__init__((MyPpoTorchProvider(),))

    @property
    def name(self) -> str:
        # This name corresponds to Task metadata task.rllib=myrl.
        return "myrl"


def register_framework() -> None:
    # Call once before training so MotrixLab can resolve the framework.
    frameworks.register_framework(MyRlFramework())
```

If the framework supports several algorithms or training backends, pass all their `AgentProvider` instances to `super().__init__()`.

## Step 2: Implement `AgentProvider`

An `AgentProvider` describes one concrete training implementation. It declares the supported backend and algorithm, the checkpoint format, the typed configuration schema, and how to create a trainer.

```python
from dataclasses import dataclass

from omegaconf import MISSING

from motrix_rl.frameworks import AgentProvider, TrainerBase, TrainerContext


@dataclass
class MyPpoCfg:
    total_steps: int = MISSING


class MyPpoTorchProvider(AgentProvider[MyPpoCfg]):
    config_type = MyPpoCfg

    @property
    def train_backend(self) -> str:
        # This name corresponds to Task metadata task.train_backend=torch.
        return "torch"

    @property
    def agent_name(self) -> str:
        # This name corresponds to Task metadata task.algo=ppo.
        return "ppo"

    @property
    def checkpoint_format(self) -> str | None:
        # Default checkpoint suffix produced by the trainer.
        return "pt"

    def create_trainer(self, context: TrainerContext[MyPpoCfg]) -> TrainerBase:
        # Keep training logic in the trainer, not in the provider.
        return MyPpoTrainer(context=context)
```

Keep the provider small: it should declare capabilities and create the trainer, not contain the training loop, environment construction, or model serialization.

## Step 3: Implement `TrainerBase`

The trainer executes training and playback. It implements two methods:

-   `train()`: run training and save checkpoint artifacts.
-   `play(policy)`: load the selected checkpoint and run policy inference.

`TrainerContext` contains the framework-neutral runtime state:

| Field                       | Meaning                                                 |
| --------------------------- | ------------------------------------------------------- |
| `context.run`               | Complete run context, including `metadata`              |
| `context.env_name`          | Registered environment name                             |
| `context.run_dir`           | Root directory for this run                             |
| `context.checkpoint_dir`    | Standard checkpoint directory                           |
| `context.sim`               | Simulator name for manager environments                 |
| `context.checkpoint_format` | Checkpoint format selected for the provider             |
| `context.num_envs`          | Number of training environments                         |
| `context.play_num_envs`     | Number of playback environments                         |
| `context.seed`              | Random seed, or `None`                                  |
| `context.rl_cfg`            | Hydra-composed and type-checked algorithm configuration |
| `context.logging`           | Logging runtime configuration                           |
| `context.checkpoint`        | Periodic checkpoint runtime configuration               |
| `context.render`            | Rendering configuration; `None` disables rendering      |
| `context.resume_from`       | Resume checkpoint path, or `None`                       |

```python
from motrix_rl import checkpoints
from motrix_rl.frameworks import TrainerBase, TrainerContext


class MyPpoTrainer(TrainerBase):
    def __init__(self, *, context: TrainerContext) -> None:
        self._context = context
        self._cfg = context.rl_cfg

    def train(self) -> None:
        # 1. Create the environment from context.env_name/context.sim.
        # 2. Build models, buffers, agents, or a third-party trainer from self._cfg.
        # 3. Restore state from context.resume_from when resume is supported.
        # 4. Execute the training loop.
        # 5. Save the final policy or training state under the run/checkpoint directory.
        # 6. Record BEST_POLICY with checkpoints.record_checkpoint_artifact();
        #    otherwise TrainResult.play() cannot discover the policy.
        pass

    def play(self, policy: str) -> None:
        # 1. Create the evaluation environment, usually with play_num_envs.
        # 2. Load the policy path.
        # 3. Execute inference and render when requested.
        pass
```

If the backend does not support resume, check `context.resume_from` and raise a clear `ValueError` instead of silently starting a new run.

## Step 4: Import the Registration Code

An independent package can register itself during package initialization:

```python
# myrl_backend/__init__.py
from .framework import register_framework

register_framework()
```

The built-in training script does not scan third-party Python entry points automatically. For an external backend, import its registration module before calling `runner.train()` or entering your command-line training path.

## Step 5: Add Hydra Training Configuration

Registering the provider installs the structured Hydra schema for `MyPpoCfg`. Add the algorithm's source-of-truth values:

```yaml
# configs/algo_base/myrl.ppo.yaml
defaults:
    - _myrl_ppo_schema
    - _self_

total_steps: 5000
```

Then add a Task recipe:

```yaml
# configs/task/cartpole/myrl.ppo.yaml
# @package _global_
defaults:
    - /algo_base@algo: myrl.ppo
    - _self_

task:
    env: cartpole
    rllib: myrl
    algo: ppo
    train_backend: torch

num_envs: 2048
play_num_envs: 16
seed: 42
```

The Task's `rllib`, `algo`, and `train_backend` values must match the framework and provider declarations. Hydra handles defaults, composition, and type validation.

## Step 6: Start Training

If the backend is registered on the startup path, use the shared CLI:

```bash
python scripts/train.py task=cartpole/myrl.ppo
```

For an external experiment package, use its Hydra config root and import the backend registration module before entering the training function. No separate Python Task registry is required.

The run is written under `runs/{env}/{rllib}/{train_backend}/{algo}/{timestamp}` with `metadata.json` and `task_config.yaml`. Playback uses those files to resolve the same provider and typed configuration.

## Compare with Built-in Integrations

The built-in implementations follow the same structure:

-   `motrix_rl/skrl/framework.py`: `SkrlFramework` registers `ppo + jax` and `ppo + torch` providers.
-   `motrix_rl/rslrl/framework.py`: `RslrlFramework` registers one `ppo + torch` provider.
-   `motrix_rl/fastsac/framework.py`: `MotrixFramework` registers one `fastsac + torch` provider, which selects its trainer from `algo.asynchronous`.
-   SKRL, RSLRL, and FastSAC trainers receive only `TrainerContext`.
-   Rendering, typed `rl_cfg`, and `resume_from` are read from `TrainerContext`.
-   The provider declares the checkpoint format; the trainer reads `context.checkpoint_format`.
-   SKRL and RSLRL currently reject resume requests; FastSAC restores training state from `context.resume_from`.

When adding an integration, follow this split: `framework.py` registers capabilities, while `train.py` or an equivalent module owns framework-specific execution.

## Common Errors

-   The provider exists, but `frameworks.register_framework()` was never called: the framework cannot be resolved.
-   The provider is registered, but the matching `configs/algo_base/` or `configs/task/` YAML is missing: Hydra cannot compose a usable Task.
-   `rllib`, `algo`, and `train_backend` do not match across the Task and provider: backend resolution or trainer creation fails.
-   `train()` saves a model but does not register `BEST_POLICY`: `TrainResult.play()` cannot auto-discover the policy.
-   Unsupported `resume_from` is ignored: users believe training resumed when it actually restarted.
