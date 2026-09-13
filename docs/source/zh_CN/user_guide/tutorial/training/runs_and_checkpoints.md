# 训练产物：runs 目录与 checkpoint 结构

本节介绍一次训练在磁盘上产生的文件：`runs/` 下的目录结构、`metadata.json` 与 checkpoint 清单（manifest）的含义。理解这套结构，有助于你定位训练结果、续训（resume）和回放（play）。

## 概述

每启动一次训练，框架都会在 `runs/` 下创建一个**独立的 run 目录**，把该次训练的元数据、checkpoint 和 TensorBoard 日志集中放在一起。回放和续训不依赖固定的文件名或目录布局，而是统一从 run 的 `metadata.json` 和 `checkpoints/manifest.json` 中读取信息。

## 目录结构

run 目录按 `环境 / RL 框架 / 训练后端 / 算法 / 时间戳` 分层：

```text
runs/{env_name}/{rllib}/{train_backend}/{algo}/{timestamp}/
    metadata.json                 # run 元数据（回放/续训的可信来源）
    checkpoints/
        manifest.json             # checkpoint 清单（登记可用的 artifact）
        latest.pt                 # 训练结束时的 checkpoint
        model_0001000.pt          # 周期性 checkpoint（可选）
        model_0002000.pt
        ...
    events.out.tfevents.*         # TensorBoard 日志
```

一个真实例子：

```text
runs/g1-walk-flat/motrix/torch/fastsac/26-07-06_11-37-50-376526/
```

各层含义：

| 路径片段        | 含义               | 示例                       |
| --------------- | ------------------ | -------------------------- |
| `env_name`      | 环境名称           | `g1-walk-flat`、`cartpole` |
| `rllib`         | RL 框架 / 命名空间 | `skrl`、`rslrl`、`motrix`  |
| `train_backend` | 训练后端           | `torch`、`jax`             |
| `algo`          | provider 算法名    | `ppo`、`fastsac`           |
| `timestamp`     | 创建时间戳         | `26-07-06_11-37-50-376526` |

时间戳格式为 `%y-%m-%d_%H-%M-%S-%f`；若同一微秒内发生冲突，会追加 `_1`、`_2` 等后缀以保证目录唯一。

## metadata.json

`metadata.json` 位于 run 根目录，是回放、续训、导出等工具**自动发现训练产物的唯一可信来源**——框架据此选择正确的 RL 框架、训练后端和算法，而不是靠文件扩展名或目录名去猜。

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

| 字段                | 含义                                      |
| ------------------- | ----------------------------------------- |
| `env_name`          | 环境名称                                  |
| `rllib`             | RL 框架名                                 |
| `train_backend`     | 训练后端                                  |
| `algo`              | 算法名                                    |
| `sim`               | manager 环境的仿真器（未指定时为 `null`） |
| `seed`              | 应用 CLI/配置覆盖后的随机种子             |
| `created_at`        | 创建时间（UTC，ISO 8601）                 |
| `checkpoint_format` | checkpoint 存储格式，如 `pt`、`pickle`    |
| `motrixlab_version` | 记录用的版本字段（当前为 `null`）         |

## checkpoints/ 与 manifest.json

实际的 checkpoint 文件都放在 `checkpoints/` 子目录下，并通过 `manifest.json` 登记「哪些文件是可用的、各自是什么语义」。回放/续训只认 manifest 里登记的 **artifact**，不直接猜文件名。

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

-   `manifest.json` 里的 `path` 相对于 `checkpoints/` 目录。
-   **`best_policy`**（`kind: policy`）：用于回放 / 导出 / 评测，是 `play.py` 默认加载的产物。
-   **`latest_training_state`**（`kind: training_state`）：用于续训，应包含 optimizer、观测归一化器、replay buffer、`global_step` 等继续训练所需的完整状态。
-   同一个物理文件可以**同时登记为两个 artifact**。例如 FastSAC 的 `latest.pt`（完整 state dict）既能回放又能续训，因此 `best_policy` 与 `latest_training_state` 都指向它。

## 如何被 play 与 resume 使用

-   **回放（play）**

    -   不设置 `policy=...`：在 `runs/{env}` 下扫描所有 `metadata.json`，选择最新的 run，再按其 `manifest.json` 取 `best_policy`。
    -   设置 `policy=<文件>`：从该文件所在目录**向上逐级查找 `metadata.json`**，据此选择正确的推理路径。

    ```bash
    # 自动发现最新 run 的最佳策略
    python scripts/play.py env=g1-walk-flat

    # 指定某个 checkpoint（需能向上找到 metadata.json）
    python scripts/play.py env=g1-walk-flat policy=/path/to/run/checkpoints/latest.pt
    ```

-   **续训（resume）**：将 `resume=` 设置为 run 目录或 checkpoint 路径，框架据此解析出 `latest_training_state` 继续训练。

    ```bash
    python scripts/train.py task=g1-walk-flat/motrix.fastsac \
      resume=/path/to/run
    ```

## TensorBoard

TensorBoard 日志（`events.out.tfevents.*`）直接写在 run 根目录下，可按环境查看：

```bash
tensorboard --logdir runs/g1-walk-flat
```
