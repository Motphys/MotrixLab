# 导出 ONNX 策略

MotrixLab 可以把一次训练的最佳策略导出为独立的 ONNX 模型。导出器会从 run 目录读取训练配置、元数据和 checkpoint 清单，恢复确定性推理路径，并在写入文件前使用 ONNX Runtime 检查导出模型与原训练框架的数值是否一致。

导出的模型包含一个名为 `obs` 的输入张量和一个名为 `actions` 的输出张量。两个张量的元素类型都是 `float32`，形状分别为 `[batch_size, observation_dim]` 和 `[batch_size, action_dim]`，其中 `batch_size` 是动态维度。训练时使用的 observation normalization 已包含在模型中，部署时不要再次归一化 observation。

## 支持范围

当前支持以下训练组合：

| RL 框架 | 训练后端 | 算法    | 限制                                                                                               |
| ------- | -------- | ------- | -------------------------------------------------------------------------------------------------- |
| SKRL    | Torch    | PPO     | 策略必须包含至少一个隐藏层和 observation preprocessor                                              |
| RSL-RL  | Torch    | PPO     | 支持单 `policy` observation group 的 `MLPModel` actor；暂不支持 state-dependent standard deviation |
| Motrix  | Torch    | FastSAC | 导出 actor 的确定性推理路径                                                                        |

SKRL/JAX checkpoint 暂不支持导出。导出命令接收完整的 run 目录，不接收脱离 `metadata.json` 和 `checkpoints/manifest.json` 的单个 checkpoint 文件。有关 run 目录和最佳策略的说明，参见[训练产物：runs 目录与 checkpoint 结构](runs_and_checkpoints.md)。

## 安装依赖

无需额外步骤：ONNX 导出与推理所需的依赖（`onnx`、`onnxruntime`）已包含在 `sh install.sh`
安装的默认运行环境中（参见[安装环境](../getting_started/installation.md)）。

## 导出模型

假设训练产生了以下 run 目录：

```text
runs/cartpole/skrl/torch/ppo/<timestamp>/
```

指定输出文件进行导出：

```bash
python scripts/export_onnx.py \
  run_dir=runs/cartpole/skrl/torch/ppo/<timestamp> \
  output=artifacts/cartpole.onnx
```

路径中含有空格时，需要为整个 Hydra 参数加引号，例如 `"run_dir=/path/to/my run"`。输出路径必须以 `.onnx` 结尾；父目录不存在时会自动创建。

如果省略 `output`，模型会写入最佳 checkpoint 所在目录，文件名为 `policy.onnx`：

```bash
python scripts/export_onnx.py \
  run_dir=runs/cartpole/skrl/torch/ppo/<timestamp>
```

导出成功后，命令会打印模型路径、输入输出规格以及数值检查结果，例如：

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

具体 feature dimension 和误差值由训练任务与导出结果决定。

## 在 ONNX Runtime 中加载

下面的最小示例从模型规格读取输入输出名称，并执行一批 observation：

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

第一维可以替换为任意 batch size。传入数据必须是 `float32`，feature dimension 必须与模型输入规格一致。

## 可选参数

通常只需要设置 `run_dir` 和 `output`。需要适配特定部署运行时或调整数值检查时，可以覆盖以下 Hydra 参数：

| 参数             | 默认值 | 说明                                 |
| ---------------- | ------ | ------------------------------------ |
| `opset`          | `18`   | ONNX opset，最小值为 `11`            |
| `parity.seed`    | `1`    | 生成检查样本的随机种子               |
| `parity.samples` | `32`   | 数值一致性检查的样本数，必须大于 `0` |
| `parity.atol`    | `1e-4` | 绝对误差容限                         |
| `parity.rtol`    | `1e-5` | 相对误差容限                         |

例如，导出为 opset 17，并增加检查样本数：

```bash
python scripts/export_onnx.py \
  run_dir=<run-dir> \
  output=policy.onnx \
  opset=17 \
  parity.samples=64
```

## 常见问题

-   **找不到最佳策略**：确认 run 已完成至少一次 checkpoint 保存，并且 `checkpoints/manifest.json` 中登记了 `best_policy` 或 `latest_training_state`。
-   **提示组合不受支持**：检查 run 的 `metadata.json`。当前不支持 SKRL/JAX，也不支持上表之外的算法或后端。
-   **缺少 `onnx`、`onnxruntime`、`torch`、`skrl` 或 `rsl_rl`**：使用与训练框架匹配的安装命令重新同步依赖。
-   **数值一致性检查失败**：导出器不会写入新文件，也不会覆盖已有的目标文件。优先检查 checkpoint 是否完整，以及 run 中的 `metadata.json`、训练配置和 manifest 是否属于同一次训练；不要直接放宽误差容限掩盖模型恢复问题。
-   **输出路径被拒绝**：确保 `output` 以 `.onnx` 结尾。
