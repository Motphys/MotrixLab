# 训练执行和结果分析

本节介绍如何执行强化学习训练，以及如何分析和使用训练结果。训练产物（`runs/` 目录、`metadata.json`、checkpoint）的结构与生成逻辑见[训练产物：runs 目录与 checkpoint 结构](runs_and_checkpoints.md)。

## 启动训练

### 选择 Task

训练入口使用 Hydra 的 `task=<环境>/<框架>.<算法>[.<后端>]` 选项。一个 Task 会把环境、RL provider、运行参数和算法超参数组合成一份可复现的训练配方：

```bash
# 训练默认的 Cartpole SKRL PPO Task
python scripts/train.py task=cartpole/skrl.ppo

# 选择其他框架或算法
python scripts/train.py task=cartpole/rslrl.ppo
python scripts/train.py task=g1-walk-flat/motrix.fastsac
python scripts/train.py task=g1-walk-flat/motrix.fastsac algo.asynchronous=false
```

当前内置的 RL method 与训练后端：

| Task 后缀        | 训练后端        | 说明                                            |
| ---------------- | --------------- | ----------------------------------------------- |
| `skrl.ppo`       | `jax` / `torch` | SKRL PPO                                        |
| `rslrl.ppo`      | `torch`         | RSLRL PPO                                       |
| `motrix.fastsac` | `torch`         | FastSAC；`algo.asynchronous` 选择同步或异步拓扑 |

运行 `python scripts/train.py --help` 可以查看当前代码中全部可选 Task。Task 文件结构和覆盖规则见 [Task 配置与命令行参数覆盖](training_environment_config.md)。

### 选择训练后端与仿真后端

```bash
# 覆盖训练后端（task.train_backend 为 null 时自动选择）
python scripts/train.py task=cartpole/skrl.ppo task.train_backend=jax
python scripts/train.py task=cartpole/skrl.ppo task.train_backend=torch

# 指定 manager 环境注入的仿真器
python scripts/train.py task=g1-wbt-dance sim=motrixsim
```

### 训练规模与随机种子

```bash
# 并行环境数量
python scripts/train.py task=cartpole/skrl.ppo num_envs=1024

# 固定随机种子（复现）/ 运行时选择随机种子
python scripts/train.py task=cartpole/skrl.ppo seed=42
python scripts/train.py task=cartpole/skrl.ppo seed=null
```

```{note}
Hydra 可以直接覆盖已经声明的算法字段，例如 `algo.agent.learning_rate=5e-4`。字段路径取决于所选 Task，完整默认值位于 `configs/algo_base/`。
```

### 训练后自动回放与续训

```bash
# 训练成功结束后，用本次 run 的最佳策略自动回放
python scripts/train.py task=g1-walk-flat/motrix.fastsac play=true

# 从某个 run 目录或 checkpoint 续训
python scripts/train.py task=g1-walk-flat/motrix.fastsac \
  resume=/path/to/run

# 启用渲染监控训练过程
python scripts/train.py task=cartpole/skrl.ppo render=true
```

### 常用 Hydra 覆盖项

| 覆盖项               | 说明                                    | 默认值来源          |
| -------------------- | --------------------------------------- | ------------------- |
| `task=...`           | 环境、框架、算法和后端差异配置          | `cartpole/skrl.ppo` |
| `task.train_backend` | 训练后端（`jax` / `torch`）             | 所选 Task           |
| `sim`                | manager 环境的仿真器；np 环境保持未设置 | `null` / 自动选择   |
| `num_envs`           | 并行训练环境数量                        | 所选 Task           |
| `seed`               | 固定种子；`null` 表示运行时随机选择     | 所选 Task           |
| `resume`             | 续训的 run 目录或 checkpoint            | `null`              |
| `play`               | 训练结束后回放最佳策略                  | `false`             |
| `render`             | 启用交互式渲染                          | `false`             |
| `algo.*`             | provider 声明的算法配置                 | 算法基础配置 + Task |
| `algo.asynchronous`  | 选择同步或异步 FastSAC 执行拓扑         | 所选 Task           |
| `logging.*`          | 日志后端与间隔                          | 训练根配置 + Task   |
| `checkpoint.*`       | 周期 checkpoint 策略                    | 训练根配置 + Task   |

## 训练过程监控

### TensorBoard 监控

TensorBoard 日志写在 run 目录下，可按环境查看：

```bash
tensorboard --logdir runs/cartpole
```

除标准的回报、损失曲线外，若环境通过 `info["Reward"]` 暴露了各 reward 分项，训练时也会将其记录到 TensorBoard。

## 模型评估和测试

回放（play）不需要重复指定 RL method——正确的 `rllib / train_backend / algo` 会从 run 的 `metadata.json` 自动读取。

```bash
# 自动发现最新 run 的最佳策略并回放（推荐）
python scripts/play.py env=cartpole

# 指定某个 checkpoint（需能向上找到 metadata.json）
python scripts/play.py env=g1-walk-flat \
  policy=/path/to/run/checkpoints/latest.pt

# 指定回放环境数量
python scripts/play.py env=cartpole num_envs=100
```

```{note}
不设置 `policy=...` 时，系统会在 `runs/{env}/` 下扫描所有 `metadata.json`，选择最新的 run，并按其 `checkpoints/manifest.json` 加载最佳策略。只有带 `metadata.json` 与 manifest 的 run 才会参与自动发现；旧 run 需要显式设置 `policy=...`。详见[训练产物：runs 目录与 checkpoint 结构](runs_and_checkpoints.md)。
```
