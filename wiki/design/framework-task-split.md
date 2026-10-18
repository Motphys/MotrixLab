# Framework / Task 配置分离设计

## 摘要

把 `motrix_rl` 从"框架 + 内置 task 配置"重构为**纯框架**：框架自带 `algo_base/` 默认值（随包发布），但不再硬编码任何 task 配置根目录。仓库根的 `configs/task/` + `scripts/` 变成**参考应用**（example app），演示外部用户如何接入 `motrix_rl`。外部团队 `pip install motrix_rl` 后，通过公开 API 注册自己的 task 配置目录即可使用，与内置参考应用走完全相同的路径（dogfood）。

## 背景与动机

重构前的配置加载层有一行硬编码：

```python
_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"
```

这行代码假设"仓库根的 `configs/`"就是 task 配置根。当 `motrix_rl` 被作为外部包安装时，`parents[3]` 不再指向用户的仓库根，框架与一个特定的目录布局绑死。

与此同时，task 配置根注册接口在仓库内部没有被完整 dogfood，内置 task 走的是一条依赖仓库布局的私有路径。这意味着：

1. 外部用户接入时，公开 API 没有被内部 dogfood 过，可能踩坑。
2. "框架"与"task 调参"两件事纠缠在同一个包里，职责不清。

## 目标

- `motrix_rl` 负责 schema、trainer、CLI helper 和 Hydra plumbing。
- task 配置由应用自己的 Hydra config root 管理。
- 公开扩展 API 聚焦于 `register_framework`；task 不再维护 Python registry。

## 非目标

- 不改 `motrix_envs` 的结构与职责。
- 不引入新的包（`motrix_tasks` 等）。task YAML 留在仓库根作为 example。

## 架构

### 包布局

```
motrix_rl/src/motrix_rl/
├── configs/
│   ├── train.yaml, play.yaml        ← 随包发布的 RL CLI 顶层配置
│   └── algo_base/                  ← 算法基础配置
│       ├── skrl.ppo.yaml
│       ├── rslrl.ppo.yaml
│       └── motrix.fastsac.yaml
├── frameworks.py                  ← framework/provider 与算法 schema 注册
├── cli.py
└── ... (skrl, rslrl, fastsac, runner, ...)
```

`configs/algo_base/` 随包发布，安装后通过 Hydra 的 `pkg://motrix_rl.configs` schema 协议可解析。

仓库根（example app）：

```
configs/
└── task/                              ← 每 task 的 RL 调参
scripts/                              ← 其他仓库辅助工具
motrixlab/src/motrixlab/               ← 应用集成：统一 mxlab CLI 与跨组件 view
├── commands/view.py                   ← 环境/机器人可视化与 ViewConfig
└── configs/view.yaml                  ← 随包发布的 view 配置
motrix_rl/src/motrix_rl/commands/       ← train/play/export 内部可调用 adapter
```

### 配置根机制

CLI 由 `@hydra.main(config_path="../configs", config_name="train")` 建立应用的
Hydra 上下文。外部应用使用自己的 config root，必要时在 `train.yaml` 中配置
`hydra.searchpath`。框架不保存额外 config root，也不提供脱离应用入口的二次组合 API。

### Hydra searchpath 配置

example app 的 `configs/train.yaml` 使用 Hydra defaults 直接选择 task，并增加 `hydra.searchpath` 条目，指向框架包内：

```yaml
defaults:
  - train_schema
  - _self_
  - task: cartpole/skrl.ppo
  - override hydra/job_logging: stdout

hydra:
  searchpath:
    - pkg://motrix_rl.configs    # 使 motrix_rl 包内的 algo_base/ 可被 task defaults 引用
  output_subdir: null
  ...
```

外部用户复制这条 `searchpath` 即可让自己的 task 配置组合内置 algo 默认值。CLI 选择 task 时使用固定 group：

```bash
mxlab train task=acrobot/skrl.ppo
mxlab train task=acrobot/rslrl.ppo
```

这里 `task` 是 Hydra group，`acrobot/skrl.ppo` 是 option。不要把 env 放到 group key 里写成 `task/acrobot=skrl.ppo`；固定 group 可以让切换环境和算法时 CLI key 保持稳定。

### Task 配置形态

framework-neutral 的默认运行策略直接写在 `configs/train.yaml`，并把
`_self_` 放在 task 前面，使 task 可以覆盖 checkpoint 等默认值：

```yaml
defaults:
  - train_schema
  - _self_
  - task: cartpole/skrl.ppo

logging:
  backend: tensorboard
  interval: 100
checkpoint:
  interval: 0
```

每个 task YAML 是一份完整训练配方：它把算法基础配置挂到 `algo`，同时在 `task` 节点提供训练元信息；需要时可以在根级覆盖 runtime 策略。

```yaml
# configs/task/acrobot/skrl.ppo.yaml
# @package _global_
defaults:
  - /algo_base@algo: skrl.ppo
  - _self_

task:
  env: acrobot
  rllib: skrl
  algo: ppo
  train_backend: null

num_envs: 2048
play_num_envs: 16
seed: 42

algo:
  trainer:
    timesteps: 5000
```

`task` 节点只保存运行元信息，用于 run metadata、backend 选择和 provider 查找；`algo` 是已经组合好的算法配置。`/algo_base@algo` 表示从 `algo_base` group 选择算法默认值，并把结果挂到根配置的 `algo` 节点；runner 校验后再把它作为内部 `rl_cfg` 传给 trainer。

### 单次组合流水线

训练入口由 `@hydra.main(config_name="train")` 一次性组合出 `task` 和 `algo`，然后把完整配置交给 runner。框架不再根据 env、rllib 和 algo 二次推导 YAML 路径；测试需要检查 task option 时，直接调用 Hydra `compose()`，与真实训练入口保持一致。

### 外部应用配置

外部应用应提供自己的 `train.yaml`、`task/` 和需要的 `algo_base/`，或通过
Hydra `searchpath` 引用其他配置源。框架不保存额外的进程级 config-root 列表。

### Train CLI schema

`TrainConfig` 表达 Hydra 已组合出的配置树，而不是再保存一组用于反查 task 的 CLI 字段：

```python
@dataclass
class TaskMeta:
    env: str = MISSING
    rllib: str = MISSING
    algo: str = MISSING
    train_backend: str | None = None


@dataclass
class TrainConfig:
    task: TaskMeta = MISSING
    num_envs: int = MISSING
    play_num_envs: int = MISSING
    seed: int | None = MISSING
    logging: LoggingConfig = MISSING
    checkpoint: CheckpointConfig = MISSING
    algo: Any = MISSING
    render: bool = False
    play: bool = False
    sim: str | None = None
    resume: str | None = None
```

Hydra 运行时仍会把根配置传为 `DictConfig`，因此 CLI 入口必须立即用
`OmegaConf.to_object()` 转成 dataclass。`algo` 在 Hydra schema 中保留为动态槽位，
其具体 dataclass 类型由 `AgentProvider[CfgT].config_type` 声明，并在
`register_framework()` 时自动安装到 Hydra。`RlFramework` 按 agent 持有唯一的
config type 以及各 backend provider，framework 不维护平行的 schema registry；
Trainer 在分发边界仍获得精确的配置类型。

用户覆盖真实配置树：

```bash
mxlab train task=acrobot/skrl.ppo num_envs=4096
mxlab train task=acrobot/skrl.ppo seed=7
mxlab train task=acrobot/skrl.ppo logging.interval=20 checkpoint.interval=100
```

因此训练入口不再需要旧式的独立 method selector 和 `+rl...` 到 `algo` 的转译层。

## 接口与不变量

### 公开 API

- `register_framework(framework) -> RlFramework`：注册 framework、providers，并从 provider 的 `config_type` 自动安装算法 schema。

### 不变量

- `configs/algo_base/<rllib>.<algo>.yaml` 文件名与其内部 `_skrl_ppo_schema` 等私有 schema 名继续以下划线分隔（不与点号 CLI 命名空间冲突）。
- `frameworks.ALGO_GROUP = "algo_base"` 不变；目录名、Hydra group 名、`/algo_base@_here_:` 引用三者保持一致。
- train CLI 使用 `task=<env>/<rllib>.<algo>[.<backend>]` 选择完整训练配方；算法元信息来自 `cfg.task`，算法参数来自 `cfg.algo`。

## 迁移影响

- **`configs/train.yaml`**：defaults 增加 `task: cartpole/skrl.ppo`，训练 CLI 通过 `task=<env>/<rllib>.<algo>` 选择配方。
- **`configs/task/`**：每个 task YAML 改为 `# @package _global_`，写入 `task` 元信息，并通过 `/algo_base@algo` 组合算法默认值。
- **`mxlab train`**：调用 `motrix_rl.commands.train`，直接消费 `cfg.task` 和 `cfg.algo`，不从独立的 env/method 字段二次组合 RL 配置。
- **`mxlab play` / `mxlab view`**：分别承担训练策略回放与环境可视化，按各自 Hydra 配置语义运行。
- **CLI 所有权**：`motrixlab` 提供唯一的 `mxlab` 入口；命令实现与内部可调用 adapter 保持在组件 package 中，版本查询只使用 `mxlab --version`。
- **测试**：Hydra ConfigLoader 枚举 task group option，再通过 Hydra `compose()` 逐项组合和校验。

## 取舍

- **保留 `algo_base` 目录命名**（不改为 `algo`）：`algo_base` 表达算法默认值层，避免与 task 元信息里的 `task.algo` 混淆。
- **CLI 使用 `task=env/algo` 而不是 `task/env=algo`**：两种 Hydra 写法都能定位同一个文件，但固定 `task` group 后，切换环境时 override key 不变，用户命令和 defaults list 更稳定。
- **不引入 `motrix_tasks` 新包**：example app 模式已足够演示外部接入，且不增加包管理复杂度。未来若需要可以再拆。
- **生产代码不做 task discovery**：训练只组合用户明确选择的 option；缺失 option 直接由 Hydra 报错。全量枚举仅存在于配置一致性测试。
- **schema 归 framework 所有**：同一 agent 的所有 backend provider 必须声明同一个 `config_type`；`register_framework()` 将其自动安装到 Hydra，避免独立 schema registry 与 provider 状态不一致。

## 关联

- 系统设计：[RL 多算法架构设计](./rl-multi-algorithm-architecture.md)（issue #151 的基础）
- 实现计划已从 `wiki/plan/` 清理；如需继续推进，请基于本设计重新建立计划。
- 追踪：本仓库 feature 分支
