# Task 配置与命令行参数覆盖

环境注册完成后，还需要为它创建训练 Task。Task 用于指定训练环境、RL 框架、算法、运行参数和算法超参数。

本章先带你为一个环境创建可运行的 Task，再介绍如何调整参数、使用 CLI 临时覆盖配置，以及 MotrixLab 配置系统的组合方式。

## 为环境创建第一个训练 Task

下面假设已经注册了一个名为 `my-robot` 的环境，并使用 SKRL PPO 进行训练。

### 选择训练方法

Task 文件名由 RL 框架和算法组成。当前仓库提供以下基础配置：

| 训练方法       | Task 文件名           |
| -------------- | --------------------- |
| SKRL PPO       | `skrl.ppo.yaml`       |
| RSLRL PPO      | `rslrl.ppo.yaml`      |
| Motrix FastSAC | `motrix.fastsac.yaml` |

本节使用 `skrl.ppo.yaml`。其他算法的配置结构将在后文介绍。

### 创建 Task 文件

创建文件：

```text
configs/task/my-robot/skrl.ppo.yaml
```

写入以下内容：

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

这个配置完成了四件事：

1. 从 `configs/algo_base/skrl.ppo.yaml` 加载完整的 SKRL PPO 基础配置。
2. 选择注册名为 `my-robot` 的环境。
3. 设置训练和播放使用的并行环境数量。
4. 将训练总步数覆盖为 `100000`。

Task 只需填写与算法基础配置不同的值，不需要复制所有算法参数。

### 检查最终配置

启动训练前，可以先输出 Hydra 组合后的完整配置：

```bash
python scripts/train.py --cfg job --resolve task=my-robot/skrl.ppo
```

该命令不会开始训练。确认 `task.env`、`num_envs` 和 `algo.trainer.timesteps` 符合预期后，再启动训练：

```bash
python scripts/train.py task=my-robot/skrl.ppo
```

运行以下命令可以查看仓库中所有可选 Task：

```bash
python scripts/train.py --help
```

## 调整 Task 的运行参数

Task 根节点中的参数与具体 RL 框架无关，可以在不同算法之间使用。

| 字段                  | 含义                                                |
| --------------------- | --------------------------------------------------- |
| `num_envs`            | 训练时使用的并行环境数量                            |
| `play_num_envs`       | 训练后播放或恢复 Task 时使用的环境数量              |
| `seed`                | 随机种子；可设置为 `null`                           |
| `logging.backend`     | 日志后端，例如 `tensorboard`                        |
| `logging.interval`    | 日志写入或训练面板刷新间隔，具体单位由训练实现决定  |
| `checkpoint.interval` | 周期 checkpoint 间隔；`0` 表示不保存周期 checkpoint |

例如，可以在 Task 中调整日志和 checkpoint 策略：

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

以下训练入口参数通常通过 CLI 设置，不必重复写入每个 Task：

| 字段     | 含义                                                          |
| -------- | ------------------------------------------------------------- |
| `render` | 是否在训练期间启用交互渲染                                    |
| `play`   | 是否在训练完成后立即加载并播放策略                            |
| `sim`    | manager 环境注入的仿真器；`null` 使用注册的默认仿真器         |
| `resume` | 恢复训练所用的 run 或 checkpoint 路径；是否支持取决于训练实现 |

## 配置算法参数

算法参数统一放在 Task 的 `algo` 节点中。字段结构由所选 RL provider 决定。

### SKRL PPO

SKRL PPO 主要包含：

-   `models`：策略网络、价值网络及是否共享模型。
-   `memory`：rollout memory 类型和容量。
-   `agent`：PPO rollout、优化、裁剪、GAE、熵和混合精度参数。
-   `trainer`：总训练步数。

例如，修改网络和学习率：

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

完整字段和逐字段注释见 [`configs/algo_base/skrl.ppo.yaml`](../../../../configs/algo_base/skrl.ppo.yaml)。

### RSLRL PPO

RSLRL PPO 主要包含：

-   `num_steps_per_env`、`max_iterations`：采样长度和训练迭代数。
-   `obs_groups`：Actor 和 Critic 使用的环境 observation group。
-   `actor`、`critic`：模型类型、隐藏层、激活函数和归一化参数。
-   `algorithm`：PPO 优化、裁剪、KL、GAE、RND 和对称增强参数。

例如，修改学习率和训练迭代数：

```yaml
algo:
    max_iterations: 300
    algorithm:
        learning_rate: 0.0005
        entropy_coef: 0.005
```

完整字段和逐字段注释见 [`configs/algo_base/rslrl.ppo.yaml`](../../../../configs/algo_base/rslrl.ppo.yaml)。

### Motrix FastSAC

FastSAC 使用同一个算法身份支持两种执行拓扑，主要字段和配置组包括：

-   `asynchronous`：`false` 表示同步交替采样和更新，`true` 表示 Collector 与 Learner 分进程执行。
-   `device`：学习设备。
-   `agent`：Actor/Critic、C51、SAC、回放缓冲区、更新频率和性能参数。
-   `trainer`：环境交互迭代数，以及仅供异步 Trainer 使用的 `async_options`：
    -   `ring_capacity`：Collector 和 Learner 之间的共享内存容量。
    -   `utd_mode`：update-to-data 策略（`strict` 或 `learner_bound`）。
    -   `weight_publish_interval`、`weight_poll_interval`：策略权重同步频率。
    -   `max_ingest_per_iter`、`idle_sleep_s`：Learner 数据摄取和空闲退避参数。

完整字段和逐字段注释见 [`configs/algo_base/motrix.fastsac.yaml`](../../../../configs/algo_base/motrix.fastsac.yaml)。

## 使用 CLI 临时覆盖参数

MotrixLab CLI 使用 Hydra 的 `key=value` 参数语法。CLI 覆盖只影响本次运行，不会修改原始 YAML。

### 覆盖运行参数

```bash
python scripts/train.py \
  task=my-robot/skrl.ppo \
  num_envs=64 \
  seed=7 \
  logging.interval=20 \
  checkpoint.interval=100
```

开启训练渲染，并在训练完成后播放：

```bash
python scripts/train.py task=my-robot/skrl.ppo render=true play=true
```

### 覆盖算法参数

SKRL PPO：

```bash
python scripts/train.py \
  task=my-robot/skrl.ppo \
  algo.agent.learning_rate=5e-4 \
  algo.agent.learning_epochs=8 \
  algo.trainer.timesteps=20000
```

RSLRL PPO：

```bash
python scripts/train.py \
  task=cartpole/rslrl.ppo \
  algo.algorithm.learning_rate=5e-4 \
  algo.algorithm.entropy_coef=0.005 \
  algo.max_iterations=300
```

FastSAC：

```bash
python scripts/train.py \
  task=g1-walk-flat/motrix.fastsac \
  algo.asynchronous=true \
  algo.agent.actor_learning_rate=1e-4 \
  algo.agent.critic_learning_rate=3e-4 \
  algo.trainer.async_options.utd_mode=learner_bound \
  algo.trainer.num_learning_iterations=20000
```

### 覆盖不同类型的值

布尔值使用小写 `true` 或 `false`：

```bash
python scripts/train.py task=my-robot/skrl.ppo render=true
```

使用 `null` 清空可空字段：

```bash
python scripts/train.py task=my-robot/skrl.ppo seed=null
python scripts/train.py task=my-robot/skrl.ppo algo.agent.learning_rate_scheduler=null
```

列表建议使用引号，避免 shell 解释方括号：

```bash
python scripts/train.py \
  task=my-robot/skrl.ppo \
  'algo.models.policy.hiddens=[128,64]' \
  'algo.models.value.hiddens=[128,64]'
```

字符串中包含空格、括号、通配符或其他 shell 特殊字符时，也应给完整的 `key=value` 参数加引号。

:::{note}
训练配置使用结构化 schema。覆盖已有字段时不需要 `+`；如果需要增加字段，应先在对应 provider 的数据类和算法基础 YAML 中定义它。
:::

### 确认 CLI 覆盖结果

将 CLI 覆盖和 `--cfg job --resolve` 组合，可以查看本次运行最终会使用的值：

```bash
python scripts/train.py \
  --cfg job \
  --resolve \
  task=my-robot/skrl.ppo \
  num_envs=64 \
  algo.agent.learning_rate=1e-3
```

解析后的完整配置会保存到训练 run 的 `task_config.yaml`，用于记录实验配置和后续策略加载。

## 为不同训练后端配置参数

共享 SKRL Task 通常将 `task.train_backend` 设置为 `null`。MotrixLab 会从已安装且可用的后端中自动选择，当前优先顺序为 JAX、Torch。

如果同一个 Task 在不同后端上需要不同的超参数，可以创建后端差异文件：

```text
configs/task/go2-walk-flat/skrl.ppo.yaml
configs/task/go2-walk-flat/skrl.ppo.jax.yaml
configs/task/go2-walk-flat/skrl.ppo.torch.yaml
```

后端差异文件继承共享 Task，只保存必要的增量：

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

使用 JAX 差异配置训练：

```bash
python scripts/train.py task=go2-walk-flat/skrl.ppo.jax
```

直接覆盖后端也可以选择 JAX 训练器：

```bash
python scripts/train.py task=go2-walk-flat/skrl.ppo task.train_backend=jax
```

这种写法不会加载 `skrl.ppo.jax.yaml` 中的后端专用参数。当后端差异文件存在时，应直接选择带 `.jax` 或 `.torch` 后缀的 Task。

## 理解 Task 配置架构

完成基本 Task 配置后，可以进一步了解这些文件是如何组合的。

### 配置目录

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

各层配置的职责如下：

| 配置层       | 位置                                               | 用途                               |
| ------------ | -------------------------------------------------- | ---------------------------------- |
| 入口配置     | `configs/train.yaml`                               | 定义通用训练参数和默认 Task        |
| 算法基础配置 | `configs/algo_base/<rllib>.<algo>.yaml`            | 提供某种算法的完整字段和基础值     |
| 共享 Task    | `configs/task/<env>/<rllib>.<algo>.yaml`           | 选择环境和算法，并保存任务调优参数 |
| 后端差异配置 | `configs/task/<env>/<rllib>.<algo>.<backend>.yaml` | 保存后端独有的增量参数             |
| CLI 覆盖     | `key=value`                                        | 临时覆盖本次运行的最终配置         |

配置组合优先级可以理解为：

```text
算法基础配置 → 共享 Task → 后端差异配置 → CLI 覆盖
```

### Task 选项命名

训练命令中的 Task 选项格式为：

```text
task=<env>/<rllib>.<algo>[.<backend>]
```

例如，文件：

```text
configs/task/cartpole/skrl.ppo.yaml
```

对应命令参数：

```bash
task=cartpole/skrl.ppo
```

Task 文件中的元数据应与该选项保持一致：

| 字段                 | 含义                              |
| -------------------- | --------------------------------- |
| `task.env`           | 环境注册名                        |
| `task.rllib`         | RL 框架名                         |
| `task.algo`          | 算法或训练实现名                  |
| `task.train_backend` | 指定训练后端；`null` 表示自动选择 |

### Hydra 组合指令

共享 Task 开头通常包含：

```yaml
# @package _global_
defaults:
    - /algo_base@algo: skrl.ppo
    - _self_
```

-   `# @package _global_`：将 Task 内容合并到训练配置根节点。
-   `/algo_base@algo: skrl.ppo`：加载算法基础配置，并挂载到根配置的 `algo` 字段。
-   `_self_`：在算法基础配置之后应用当前文件，使 Task 中的值可以覆盖基础值。

Hydra 会根据 provider 注册的结构化 schema 检查字段名和类型。字段拼写错误、类型不匹配或缺少必填字段时，配置会在训练开始前失败。

## Play 和 View 的参数覆盖

`scripts/play.py` 和 `scripts/view.py` 同样使用 `key=value` 语法：

```bash
python scripts/view.py env=cartpole num_envs=4
python scripts/play.py env=cartpole num_envs=1
python scripts/play.py policy=/path/to/checkpoint.pt num_envs=1
```

Play 默认读取训练 run 中保存的 `task_config.yaml`。如需临时覆盖其中的算法参数，使用 `rl` 作为算法配置根节点。因为 `rl` 初始为空，需要用 `+` 添加路径：

```bash
python scripts/play.py \
  env=cartpole \
  '+rl.agent.learning_rate=1e-4'
```

这里的 `rl` 内容会合并到保存配置的 `algo` 节点中，字段路径取决于训练该策略时使用的算法。
