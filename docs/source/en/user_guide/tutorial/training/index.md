# Training Overview

Once an environment is registered, training an RL task takes three steps:

```text
Create a Task config       pick the environment, RL framework, algorithm, hyperparameters
      ↓
python scripts/train.py task=...   Hydra composes the config and starts training
      ↓
runs/{env}/...             stores metadata, logs, and checkpoints
```

The pages follow the order of use:

- [Task Configuration and CLI Overrides](task_config.md): create the Task file, tune
  run and algorithm parameters, override values with `key=value`;
- [Running Training and Analyzing Results](training_and_result.md): start training,
  read TensorBoard logs, replay policies with play;
- [Training Artifacts: runs Directories and Checkpoints](runs_and_checkpoints.md):
  the `runs/` layout, best-policy selection, and how to resume.

```{toctree}
:hidden:

task_config
training_and_result
runs_and_checkpoints
```
