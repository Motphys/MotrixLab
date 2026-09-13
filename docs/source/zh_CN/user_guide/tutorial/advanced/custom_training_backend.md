# 扩展自定义训练后端

本文介绍如何把一个新的强化学习框架、算法或训练后端接入 MotrixLab。接入后，用户仍然可以使用统一的训练入口、运行目录、checkpoint 元数据和 `TrainResult.play()`。

## 先确定三个名字

一次训练由三个名字定位：

| 名称            | 由谁声明                      | 用途                 | 示例                     |
| --------------- | ----------------------------- | -------------------- | ------------------------ |
| `rllib`         | `RlFramework.name`            | 强化学习框架命名空间 | `skrl`、`rslrl`、`myrl`  |
| `algo`          | `AgentProvider.agent_name`    | 算法或 agent 名称    | `ppo`、`fastsac`、`sac`  |
| `train_backend` | `AgentProvider.train_backend` | 训练后端名称         | `jax`、`torch`、`custom` |

例如 `task=cartpole/myrl.ppo` 会加载一份 `rllib=myrl`、`algo=ppo` 的 Task；其中的 `task.train_backend=torch` 会选择 Torch provider。

## Step 1：实现 `RlFramework`

`RlFramework` 是注册入口。它只负责声明框架名字，并把这个框架支持的 agent 列出来。这里的 agent 通常就是 PPO、SAC 这类强化学习算法实现，不是仿真环境里的机器人对象。

```python
from motrix_rl import frameworks
from motrix_rl.frameworks import RlFramework


class MyRlFramework(RlFramework):
    def __init__(self) -> None:
        # 这里列出这个框架提供的所有 agent 实现。
        # 每个 agent 实现对应一个 (algo, train_backend)。
        super().__init__((MyPpoTorchProvider(),))

    @property
    def name(self) -> str:
        # 这个名字对应 Task 元数据 task.rllib=myrl。
        return "myrl"


def register_framework() -> None:
    # 在训练开始前调用一次，让 MotrixLab 能找到这个框架。
    frameworks.register_framework(MyRlFramework())
```

如果你的框架同时支持多个算法或多个训练后端，就在 `super().__init__()` 里传入多个 agent 实现。代码里这些 agent 实现由 `AgentProvider` 承载。

## Step 2：实现 `AgentProvider`

`AgentProvider` 描述一个具体的训练实现。它告诉 MotrixLab：我支持哪个训练后端、哪个算法、产出什么 checkpoint 格式，以及如何创建 trainer。

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
        # 这个名字对应 Task 元数据 task.train_backend=torch。
        return "torch"

    @property
    def agent_name(self) -> str:
        # 这个名字对应 Task 元数据 task.algo=ppo。
        return "ppo"

    @property
    def checkpoint_format(self) -> str | None:
        # 这里写你的 trainer 默认保存的 checkpoint 后缀。
        return "pt"

    def create_trainer(self, context: TrainerContext[MyPpoCfg]) -> TrainerBase:
        # 这里创建实际执行训练的对象。
        # 不要在 provider 里写训练逻辑；训练逻辑放到 trainer 里。
        return MyPpoTrainer(context=context)
```

Provider 应该很薄。它只做“声明能力”和“创建 trainer”，不要把训练循环、环境创建、模型保存放在这里。

## Step 3：实现 `TrainerBase`

Trainer 是真正执行训练和回放的地方。它只需要实现两个方法：

-   `train()`：执行训练，并保存 checkpoint。
-   `play(policy)`：加载指定 checkpoint，并运行策略回放。

`TrainerContext` 已经包含 trainer 需要的运行信息：

| 字段                        | 含义                                     |
| --------------------------- | ---------------------------------------- |
| `context.run`               | 当前 run 的完整上下文，里面有 `metadata` |
| `context.env_name`          | 环境名                                   |
| `context.run_dir`           | 本次训练的运行目录                       |
| `context.checkpoint_dir`    | 标准 checkpoint 目录                     |
| `context.sim`               | manager 环境的仿真器名                   |
| `context.checkpoint_format` | 当前 provider 对应的 checkpoint 格式     |
| `context.num_envs`          | 训练环境数量                             |
| `context.play_num_envs`     | 回放环境数量                             |
| `context.seed`              | 随机种子                                 |
| `context.rl_cfg`            | 已由 Hydra 组合并类型校验的算法配置      |
| `context.logging`           | 日志运行配置                             |
| `context.checkpoint`        | checkpoint 运行配置                      |
| `context.render`            | 渲染配置；`None` 表示禁用                |
| `context.resume_from`       | 断点续训 checkpoint 路径，可能为 `None`  |

```python
from motrix_rl import checkpoints
from motrix_rl.frameworks import TrainerBase, TrainerContext


class MyPpoTrainer(TrainerBase):
    def __init__(self, *, context: TrainerContext) -> None:
        self._context = context
        self._cfg = context.rl_cfg

    def train(self) -> None:
        # 1. 根据 context.env_name/context.sim 创建训练环境。
        # 2. 根据 self._cfg 创建模型、buffer、agent 或第三方训练器。
        # 3. 如果支持断点续训，从 context.resume_from 加载训练状态。
        # 4. 执行你的训练循环。
        # 5. 把最终策略或训练状态保存到 context.run_dir 或 context.checkpoint_dir。
        # 6. 使用 checkpoints.record_checkpoint_artifact() 记录 BEST_POLICY；
        #    否则 TrainResult.play() 找不到自动回放的策略文件。
        pass

    def play(self, policy: str) -> None:
        # 1. 创建评估环境，通常使用 play_num_envs 或等价配置。
        # 2. 从 policy 路径加载策略。
        # 3. 执行推理循环，并按需渲染。
        pass
```

如果你的后端暂时不支持断点续训，建议在 `__init__()` 里检查 `context.resume_from`，并抛出清晰的 `ValueError`，不要静默忽略。

## Step 4：让注册代码被导入

如果这是一个独立 Python 包，可以在包初始化时注册：

```python
# myrl_backend/__init__.py
from .framework import register_framework

register_framework()
```

当前训练脚本不会自动扫描外部 Python 包入口点。使用仓库外部的后端时，必须在调用 `runner.train()` 或命令行训练逻辑之前导入注册模块。

## Step 5：添加 Hydra 训练配置

Provider 注册时会自动安装 `MyPpoCfg` 对应的结构化 schema。然后添加算法默认值：

```yaml
# configs/algo_base/myrl.ppo.yaml
defaults:
    - _myrl_ppo_schema
    - _self_

total_steps: 5000
```

再添加 task 配方：

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

文件中的 `rllib`、`algo` 和 `train_backend` 必须与 framework/provider 声明一致。配置的 defaults、继承和类型校验均由 Hydra 完成。

## Step 6：启动训练

如果后端已经在启动路径中完成注册，可以直接用 CLI：

```bash
python scripts/train.py task=cartpole/myrl.ppo
```

如果是外部实验包，使用自己的 Hydra config root，并在进入训练入口前导入 backend 注册模块。无需额外的 Python task registry。

训练目录会写到 `runs/{env}/{rllib}/{train_backend}/{algo}/{timestamp}`，并带有 `metadata.json`。后续回放会依赖这些元数据找到正确的 provider。

## 和内置实现对照

当前仓库里的实现也是这个结构：

-   `motrix_rl/skrl/framework.py`：`SkrlFramework` 注册两个 provider，分别对应 `ppo + jax` 和 `ppo + torch`。
-   `motrix_rl/rslrl/framework.py`：`RslrlFramework` 注册一个 `ppo + torch` provider。
-   `motrix_rl/fastsac/framework.py`：`MotrixFramework` 注册一个 `fastsac + torch` provider，并根据 `algo.asynchronous` 选择 trainer。
-   SKRL、RSLRL、FastSAC 的 trainer 构造函数都只接收 `TrainerContext`。
-   `render`、类型化的 `rl_cfg`、`resume_from` 都从 `TrainerContext` 读取。
-   checkpoint 格式由 provider 声明，trainer 使用 `context.checkpoint_format`。
-   SKRL/RSLRL 当前不支持 resume，会在收到 `context.resume_from` 时抛错；FastSAC 会从该路径恢复训练状态。

扩展新后端时，优先模仿这些文件的分层：`framework.py` 负责注册，`train.py` 或等价模块负责训练细节。

## 常见错误

-   只写了 provider，但没有 `frameworks.register_framework()`：训练时会找不到框架。
-   provider 已注册，但缺少对应的 `configs/algo_base/` 或 `configs/task/` YAML：Hydra 无法组合出可用 Task。
-   `rllib`、`algo`、`train_backend` 三个名字不一致：自动选择后端或创建 trainer 会失败。
-   `train()` 保存了模型，但没有记录 `BEST_POLICY`：`TrainResult.play()` 无法自动找到策略文件。
-   `resume_from` 不支持却被忽略：用户会以为已经续训，实际是重新训练。
