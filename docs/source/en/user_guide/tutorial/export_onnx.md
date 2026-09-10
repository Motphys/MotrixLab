# Export an ONNX Policy

MotrixLab can export the best policy from a training run as a standalone ONNX model. The exporter reads the training configuration, metadata, and checkpoint manifest from the run directory, restores the deterministic inference path, and uses ONNX Runtime to check numerical parity with the original framework before writing the file.

The exported model has one input tensor named `obs` and one output tensor named `actions`. Both tensors use `float32` elements and have shapes `[batch_size, observation_dim]` and `[batch_size, action_dim]`, respectively, where `batch_size` is dynamic. Observation normalization used during training is included in the model; do not normalize observations again during deployment.

## Supported Configurations

The following training configurations are currently supported:

| RL framework | Training backend | Algorithm | Limitations                                                                                                                    |
| ------------ | ---------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------ |
| SKRL         | Torch            | PPO       | The policy must have at least one hidden layer and an observation preprocessor                                                 |
| RSL-RL       | Torch            | PPO       | Supports an `MLPModel` actor with a single `policy` observation group; state-dependent standard deviation is not supported yet |
| Motrix       | Torch            | FastSAC   | Exports the actor's deterministic inference path                                                                               |

SKRL/JAX checkpoints cannot be exported yet. The command accepts a complete run directory, not a checkpoint file detached from its `metadata.json` and `checkpoints/manifest.json`. See [Training Artifacts: runs Directories and Checkpoints](runs_and_checkpoints.md) for details about run directories and best-policy selection.

## Install Dependencies

No extra step is needed: the ONNX export and inference dependencies (`onnx`, `onnxruntime`) are part
of the default runtime environment installed by `sh install.sh` (see
[Installation](../getting_started/installation.md)).

## Export the Model

Suppose training produced this run directory:

```text
runs/cartpole/skrl/torch/ppo/<timestamp>/
```

Export it to an explicit output path:

```bash
python scripts/export_onnx.py \
  run_dir=runs/cartpole/skrl/torch/ppo/<timestamp> \
  output=artifacts/cartpole.onnx
```

If a path contains spaces, quote the complete Hydra argument, for example `"run_dir=/path/to/my run"`. The output path must end in `.onnx`; missing parent directories are created automatically.

When `output` is omitted, the model is written next to the best checkpoint as `policy.onnx`:

```bash
python scripts/export_onnx.py \
  run_dir=runs/cartpole/skrl/torch/ppo/<timestamp>
```

After a successful export, the command prints the model path, tensor specifications, and numerical parity results, for example:

```json
{
    "input": {
        "dtype": "float32",
        "name": "obs",
        "shape": [null, 4]
    },
    "max_abs_error": 1.1920928955078125e-7,
    "max_rel_error": 2.384185791015625e-7,
    "output": "artifacts/cartpole.onnx",
    "output_tensor": {
        "dtype": "float32",
        "name": "actions",
        "shape": [null, 1]
    },
    "validation_samples": 32
}
```

The exact feature dimensions and error values depend on the task and exported model.

## Load the Model with ONNX Runtime

This minimal example reads the tensor names from the model and runs one batch of observations:

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("artifacts/cartpole.onnx", providers=["CPUExecutionProvider"])
input_spec = session.get_inputs()[0]
output_spec = session.get_outputs()[0]

observation_dim = input_spec.shape[1]
observations = np.zeros((1, observation_dim), dtype=np.float32)
actions = session.run([output_spec.name], {input_spec.name: observations})[0]
print(actions)
```

The first dimension can be any batch size. Inputs must use `float32`, and the feature dimension must match the model input specification.

## Optional Parameters

In most cases, only `run_dir` and `output` are needed. Override these Hydra parameters when targeting a particular deployment runtime or tuning the parity check:

| Parameter        | Default | Description                                        |
| ---------------- | ------- | -------------------------------------------------- |
| `opset`          | `18`    | ONNX opset; the minimum is `11`                    |
| `parity.seed`    | `1`     | Random seed used to generate validation samples    |
| `parity.samples` | `32`    | Number of parity samples; must be greater than `0` |
| `parity.atol`    | `1e-4`  | Absolute error tolerance                           |
| `parity.rtol`    | `1e-5`  | Relative error tolerance                           |

For example, export with opset 17 and more validation samples:

```bash
python scripts/export_onnx.py \
  run_dir=<run-dir> \
  output=policy.onnx \
  opset=17 \
  parity.samples=64
```

## Troubleshooting

-   **No best policy is found**: make sure the run has saved at least one checkpoint and that `checkpoints/manifest.json` registers a `best_policy` or `latest_training_state` artifact.
-   **The training configuration is unsupported**: inspect the run's `metadata.json`. SKRL/JAX and configurations outside the table above are not currently supported.
-   **`onnx`, `onnxruntime`, `torch`, `skrl`, or `rsl_rl` is missing**: sync the dependencies again with the command matching the training framework.
-   **Numerical parity fails**: the exporter neither writes a new file nor replaces an existing target. First check that the checkpoint is complete and that the run's `metadata.json`, training configuration, and manifest belong to the same run. Do not loosen the tolerances merely to hide a model restoration problem.
-   **The output path is rejected**: make sure `output` ends in `.onnx`.
