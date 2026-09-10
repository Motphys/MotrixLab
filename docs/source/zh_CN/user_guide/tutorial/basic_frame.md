# 基础框架

MotrixLab 将环境实现、训练方法、配置和命令行编排分成独立层。本节介绍这些部分如何协作，便于后续开发自定义环境或训练后端。

## 仓库分层

```text
MotrixLab/
├── motrix_envs/                 # 环境配置、实现与注册表
│   └── src/motrix_envs/
├── motrix_rl/                   # RL framework、provider、trainer 与训练产物
│   └── src/motrix_rl/
├── configs/
│   ├── algo_base/               # 各 RL provider 的完整类型化默认配置
│   └── task/<env>/              # 各环境的训练配方
└── scripts/
    ├── train.py                 # Hydra 训练入口
    ├── play.py                  # 基于 metadata 的策略回放入口
    └── view.py                  # 随机动作环境预览入口
```

主要运行流程如下：

```text
task=<环境>/<框架>.<算法>
              │
              ▼
Hydra 组合根配置、算法基础配置、Task 与 CLI 覆盖
              │
              ▼
runner 解析 AgentProvider 并创建 Trainer
              │
              ▼
Trainer 创建已注册环境并执行 train/play
              │
              ▼
runs/... 保存 metadata、最终 Task 配置与 checkpoint
```

## 核心组件

### 环境层

一个环境通常包含：

-   使用 `@registry.envcfg("name")` 注册的 `EnvCfg` 数据类。
-   使用 `@registry.env("name")` 注册的环境实现。
-   observation、reward、termination、reset 和 action application 等任务逻辑。

环境注册表维护环境名及其仿真后端实现。`scripts/view.py`、训练器和回放流程都通过同一注册表创建环境。

### RL Framework 与 Provider 层

`RlFramework` 定义 `skrl`、`rslrl`、`motrix` 等 RL 框架命名空间。每个 framework 包含一个或多个 `AgentProvider`。Provider 声明：

-   算法名，例如 `ppo` 或 `fastsac`。
-   训练后端，例如 `jax` 或 `torch`。
-   它接受的类型化算法配置 schema。
-   checkpoint 格式以及如何创建 trainer。

Framework 与 provider 代表可执行能力，因此在 Python 中注册。扩展接口见[添加自定义训练后端](custom_training_backend.md)。

### Hydra 配置层

训练参数保存在 YAML 中，不再使用 Python Task 子类：

-   `configs/algo_base/<framework>.<algorithm>.yaml` 提供 provider 所有算法字段的完整默认值。
-   `configs/task/<env>/<framework>.<algorithm>.yaml` 选择环境并保存任务调优参数。
-   可选的 `.<backend>.yaml` Task 只保存后端差异。
-   CLI 的 `key=value` 参数在组合完成后应用临时覆盖。

Provider 的数据类 schema 负责校验字段名和类型，YAML 是配置值的唯一来源。Task 通过扫描 `configs/task/` 自动发现，不再存在 RL 配置装饰器或 Python Task 注册表。

### Runner 与 Trainer 层

公共 runner 负责与框架无关的编排：

1. 从组合配置读取 `task.env`、`task.rllib`、`task.algo` 和 `task.train_backend`。
2. 解析兼容的 provider 和训练后端。
3. 创建 run 目录并写入 `metadata.json` 与 `task_config.yaml`。
4. 构造 `TrainerContext`，再由 provider 创建 trainer。
5. 执行训练或回放，并登记 checkpoint artifact。

Trainer 负责框架特有的模型创建、优化、checkpoint 序列化和推理。它应通过环境注册表创建环境，而不是依赖某个具体环境类。

## 训练流程

例如：

```bash
python scripts/train.py task=cartpole/skrl.ppo num_envs=1024
```

该命令会依次执行：

1. Hydra 组合 `configs/train.yaml`、`configs/algo_base/skrl.ppo.yaml` 和 `configs/task/cartpole/skrl.ppo.yaml`。
2. `num_envs=1024` 只覆盖本次运行的 Task 值。
3. runner 解析 SKRL PPO provider，并从可用的 JAX/Torch 后端中选择一个。
4. trainer 创建已注册的 `cartpole` 环境并开始优化。
5. run metadata、最终 Task 快照、日志和 checkpoint manifest 写入 `runs/cartpole/`。

## 多框架支持

同一个环境可以拥有多份 Task 配方，不需要修改环境实现：

```bash
python scripts/train.py task=cartpole/skrl.ppo
python scripts/train.py task=cartpole/rslrl.ppo
```

SKRL 提供 JAX 与 Torch provider，RSLRL 使用 Torch；`motrix.fastsac` 通过 `algo.asynchronous` 选择同步或异步 Torch trainer。所选 Task 与 provider 共同决定算法配置和输出 metadata。

## 分层带来的优势

1. **环境复用**：一个注册环境可以由多个 RL 框架训练。
2. **类型化配置**：provider schema 会在训练前拒绝拼写错误或类型不兼容的 YAML/CLI 值。
3. **实验可复现**：每个 run 保存最终 Task 配置和 provider 身份。
4. **易于扩展**：新增环境只需注册环境并添加 Task YAML；新增 RL 集成则添加 provider 与 trainer。
5. **训练产物一致**：回放和续训依赖 metadata 与 checkpoint manifest，不猜测文件名。
