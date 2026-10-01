# RL 多算法架构设计

## 摘要

本文描述 MotrixLab 在 RL 集成层支持多算法的架构。命令行对用户暴露单一的 RL method selector（例如 `skrl.ppo`、`rslrl.ppo`、`motrix.fastsac`），内部展开为 `env_name + rllib + train_backend + algo` 四个正交维度，用来定位配置、选择训练实现、记录 run metadata 和发现 checkpoint。架构目标是让不同 RL 框架与算法（SKRL PPO、RSLRL PPO、Motrix FastSAC，以及后续的 SAC/TD3/RPO/AMP、多智能体算法）各自独立演进，同时共享环境创建、seed、run 目录、metadata 与 checkpoint 等通用编排，而外部算法接入时不需要改动训练入口、play 入口或 checkpoint 发现逻辑。

本文只描述架构：组件、接口、不变量与取舍。RL 框架整体分层与解耦原则见[项目整体架构设计](./architecture.md)；SAC / FastSAC 的算法原理见[SAC 与 FastSAC 算法原理入门](../research/sac-and-fastsac-primer.md)。

## 1. 设计动机

RL 集成层需要在同一套入口下承载多个 RL 框架和多种算法：

- 一个 `rllib`（RL 框架 namespace）可能支持多个 train backend（如 SKRL 支持 `jax` / `torch`）。
- 一个 `rllib` 下可能有多种算法（PPO、SAC、TD3……），其模型角色、memory 类型、agent 超参数结构彼此不同。
- 自研算法归入 `motrix` namespace，例如 FastSAC 使用 `motrix.fastsac`。
- on-policy（PPO/RPO/AMP）与 off-policy（SAC/TD3）算法在 memory 与并行规模上要求不同，PPO 的字段不应泄漏到 off-policy 配置。

因此架构把"某环境用什么超参数"（配置注册表）、"某 RL 框架提供哪些训练实现"（framework 与 provider）、"如何创建 run 与发现产物"（run/metadata/checkpoint）三类关注点分离，并让它们只通过 `(env_name, rllib, train_backend, algo)` 这组正交键交互。

## 2. 设计目标

1. **RL method 是一等 CLI 维度**：用户通过 `--algo skrl.ppo` 直接选择一套 RL 方法，而非组合多个开关。
2. **内部维度正交**：配置查找、trainer 查找、run metadata、checkpoint 发现都显式包含 `rllib / train_backend / algo`。
3. **默认行为兼容**：不传 `--algo` 时默认 `skrl.ppo`。
4. **算法可独立演进**：每个算法可自定义模型角色、memory、agent 配置和 runtime 配置。
5. **共享通用编排**：环境创建、seed、run 目录、metadata、train/play 主流程不在每个算法里重复。
6. **配置按算法收敛**：每个算法拥有独立配置类型，字段不跨算法泄漏。
7. **外部接入面稳定**：外部算法实现 `TrainerBase`、注册 `RlFramework`/`AgentProvider`、注册配置类后，不需要修改 `scripts/train.py`、`scripts/play.py`、checkpoint 发现或 backend 选择逻辑。

## 3. 非目标

- 不要求所有 SKRL 上游算法一次性实现。
- 不要求 RSLRL 支持 PPO / Distillation 以外的算法。
- 不强制每个算法同时支持所有 train backend；支持矩阵由 framework/provider 显式声明。
- 不在单智能体算法接入时把现有环境接口重构成 multi-agent 接口。

## 4. 分层架构

```
┌─────────────────────────────────────────────┐
│ scripts/train.py · scripts/play.py          │
│ CLI 解析、自动发现、错误展示                 │
├─────────────────────────────────────────────┤
│ motrix_rl.runner                            │
│ 通用 train 编排、TrainRequest -> TrainResult │
├─────────────────────────────────────────────┤
│ motrix_rl.frameworks                        │
│ RlFramework -> AgentProvider 索引与查询      │
│ Hydra algo schema 注册 / TrainerContext / TrainerHandle │
├─────────────────────────────────────────────┤
│ motrix_rl.backend_runtime                   │
│ train backend 可用性判断与自动选择           │
├─────────────────────────────────────────────┤
│ motrix_rl_builtin · motrix_rl_skrl · motrix_rl_rslrl │
│ 插件包：算法特异 model/memory/agent/runtime + Trainer │
├─────────────────────────────────────────────┤
│ motrix_rl.runs · motrix_rl.checkpoints      │
│ run 目录 / metadata / checkpoint manifest    │
└─────────────────────────────────────────────┘
```

`rllib` 是 RL 框架 namespace，也是 run metadata 与 CLI method 中的第一段。`algo` 是该 framework 内部的算法/agent 名称。`train_backend` 是训练实现所用的计算后端（如 `jax` / `torch`，也允许自定义名称）。

### 4.1 Workspace package boundaries

RL 集成拆分为控制平面（同时持有稳定 contracts）和可选 provider 插件两个边界：

| Package | 所有权 | 允许依赖 | 不负责 |
|---|---|---|---|
| `motrix_rl` | 控制平面与稳定 contracts：runner、CLI 协作、run metadata、checkpoint manifest、backend 选择、`motrix_rl.frameworks` 注册/查询与 Hydra schema 安装，以及 `RlFramework`、`AgentProvider`、`TrainerBase`、`TrainerContext` 等 provider/trainer 契约 | 环境与通用运行时；可使用控制平面基础依赖 | 不内置具体 FastSAC、SKRL 或 RSLRL trainer；contracts 不包含 provider 实现细节 |
| `motrix_rl_builtin` | Motrix 内置 FastSAC（`motrix.fastsac`）及其模型、replay、同步/异步 trainer 与导出实现 | 仅依赖 `motrix_rl` 与 Torch 等 FastSAC 运行时依赖 | 不把 FastSAC 类型重新定义为控制平面公共契约 |
| `motrix_rl_skrl` | SKRL PPO provider（JAX/Torch）及 SKRL-specific wrapper、配置和 trainer | 依赖 `motrix_rl`、SKRL 与 Torch；额外 JAX runtime 按 extra 安装 | 不让 SKRL 依赖成为控制平面的强制依赖 |
| `motrix_rl_rslrl` | RSLRL PPO provider（Torch）及原生 `rsl_rl` runner 适配 | 仅依赖 `motrix_rl` 与 RSLRL/Torch | 不把 RSLRL runner 细节泄漏到控制平面 |

控制平面保留稳定的 `motrix_rl.frameworks` 入口与 `motrix_rl.contracts` 契约模块；插件包实现 `RlFramework`，并通过该入口注册，而不是由 `motrix_rl` 直接 import provider 模块。算法库是所属插件包的必需依赖，仅额外 backend runtime 使用可选依赖组；插件依赖控制平面，控制平面不依赖插件。根项目必需安装 builtin 与 SKRL，RSLRL plugin 和 SKRL JAX runtime 通过 extras 选择。

### 4.2 Entry-point discovery

插件以 Python packaging entry point group `motrix_rl.frameworks` 发现。每个插件包在自己的分发声明中提供一个命名 entry point，指向无参数 `register()` callable；以下汇总三个包的条目：

```toml
[project.entry-points."motrix_rl.frameworks"]
builtin = "motrix_rl_builtin.plugin:register"
skrl = "motrix_rl_skrl.plugin:register"
rslrl = "motrix_rl_rslrl.plugin:register"
```

`motrix_rl.plugins.load_plugins()` 在框架查询或 Hydra task 组合前读取已安装 entry points，按名称稳定排序并确保每个 entry point 只加载一次；注册失败向调用方传播，不被静默吞掉。未安装的可选插件不影响控制平面启动，只有选择其 method 或查询其 framework 时才会报告 provider 不可用。entry-point 名称是分发标识，真实 method identity 仍由 `RlFramework.name`、`AgentProvider.agent_name` 和 `train_backend` 声明。

## 5. CLI 与 method 解析

### 5.1 method 形式

用户可见的主选择维度是 RL method，形式为 `rllib.algo`：

```text
skrl.ppo
rslrl.ppo
motrix.fastsac
```

`train.py` 用 `--algo` 承载该 method，默认 `skrl.ppo`；`--train-backend` 单独指定 train backend，未指定时自动选择。示例：

```bash
uv run scripts/train.py task=cartpole/skrl.ppo
uv run scripts/train.py task=cartpole/rslrl.ppo
uv run scripts/train.py task=g1-walk-flat/motrix.fastsac
uv run scripts/train.py task=g1-walk-flat/motrix.fastsac algo.asynchronous=false
```

`--rllib` 作为兼容开关保留：传 `--rllib skrl --algo ppo` 解析为 `skrl.ppo`；传 `--algo skrl.ppo` 时不需要 `--rllib`；两者冲突时报错。

### 5.2 method 解析

```python
@dataclass(frozen=True)
class RlMethod:
    rllib: str
    algo: str


def parse_method(value: str, deprecated_rllib: str | None = None) -> RlMethod: ...
```

规则：

- `skrl.ppo` → `rllib="skrl"`, `algo="ppo"`。
- 裸算法名（如 `ppo`）只在同时传入 `--rllib` 时作为兼容 shorthand 允许。
- 三段形式（如 `skrl.torch.ppo`）不作为主接口，train backend 继续由 `--train-backend` 指定或自动选择。

### 5.3 play 不需要 `--algo`

play 不是训练配置选择，而是加载已训练 policy，正确的 `rllib/train_backend/algo` 来自 run metadata：

- 指定 `--policy` 时，从 checkpoint 同级或上级 run 目录的 `metadata.json` 读取 method，再选择对应 `AgentProvider` 做 inference；找不到 metadata 直接报错。
- 不指定 `--policy` 时，在 `runs/{env}` 下读取所有 run metadata，选择最近 run，读取其 method，再按 `checkpoints/manifest.json` 定位 policy；没有 metadata run 或 manifest artifact 直接报错。

## 6. RL task 配置查询

Hydra task 配置回答"某环境用什么超参数"。

### 6.1 按需组合

系统不维护 Python task registry 或其他全局 task 索引，也不预先检查配置文件。
训练时直接请求 Hydra 组合
`task/<env>/<rllib>.<algo>[.<backend>]`。task YAML 的 defaults、继承、
package 和 schema 均不由 Python 手动解析。

### 6.2 注册与 method 所有权

method identity 由 framework 与 provider 共同定义：`rllib` 来自
`RlFramework.name`，`algo` 来自 `AgentProvider.agent_name`，配置类型来自
`AgentProvider.config_type`。配置 dataclass 不再重复声明 `rllib/algo` ClassVar。

```python
class SkrlPpoJaxProvider(AgentProvider[SkrlCfg]):
    config_type = SkrlCfg
    agent_name = "ppo"
    train_backend = "jax"


class SkrlPpoTorchProvider(AgentProvider[SkrlCfg]):
    config_type = SkrlCfg
    agent_name = "ppo"
    train_backend = "torch"
```

`RlFramework` 按 agent 保存唯一的 `config_type` 和 backend providers；同一 agent
若声明多个不同 config type，framework 构造立即失败。`register_framework()` 同时
注册 provider 能力并把 config type 安装到 Hydra，不维护平行的 schema registry。

外部算法不需要修改 framework 内的固定 Union 或额外 schema 表；只需提供带
`config_type` / `agent_name` / `train_backend` 的 provider，并通过自己的
`RlFramework` 注册即可。

### 6.3 Task option 选择

训练入口直接消费 Hydra 根据 `task=<env>/<rllib>.<algo>[.<backend>]` 组合出的配置。
backend 专用 option 需要显式选择；option 不存在或其 defaults 损坏时直接报告 Hydra
配置错误，不在框架层执行路径猜测或 fallback。

## 7. RlFramework 与 AgentProvider

`motrix_rl.frameworks` 回答"某训练后端、算法如何创建 Trainer"。它是控制平面的稳定入口，不是 provider 实现包的聚合模块；provider 通过 `motrix_rl.frameworks` 的注册 API 接入。

- **`RlFramework`**：外部 RL 框架的注册入口，定义 framework namespace（即 `rllib`），持有一组 framework-scoped `AgentProvider`，并提供 supported agents / supported train backends / provider 查询 API。
- **`AgentProvider`**：framework 内部的训练实现单元，声明 `train_backend`、`agent_name`、`checkpoint_format`，并负责创建 `TrainerBase`。
- **`TrainerBase`**：可运行对象基类，负责框架内部的 train 与 play。外部算法接入时继承它。

这些契约与 `TrainerContext` / `TrainerHandle` 定义于 `motrix_rl.contracts`；`motrix_rl.frameworks` 保留公共兼容入口，并负责将具体的 `RunContext`、运行配置与 result factory 注入接口对象。contracts 模块不 import 控制平面的 `RunContext` / Hydra 配置类型，后者只在控制平面和插件的调用边界被使用。

`AgentProvider` 不要求暴露 `wrap_env` / `make_models` / `make_memory` 等细粒度步骤——这些是内置 trainer 的内部 helper，不是外部接入面。

### 7.1 接口

```python
class TrainerBase(ABC):
    @abstractmethod
    def train(self) -> None: ...
    @abstractmethod
    def play(self, policy: str) -> None: ...


@dataclass(frozen=True)
class TrainerContext:
    run: RunContext
    env_name: str
    run_dir: Path
    checkpoint_dir: Path
    checkpoint_format: str
    num_envs: int
    play_num_envs: int
    seed: int | None
    logging: LoggingConfig
    checkpoint: CheckpointConfig
    rl_cfg: Any
    render: RenderConfig | None = None
    resume_from: str | None = None


class AgentProvider(ABC):
    @property
    @abstractmethod
    def train_backend(self) -> str: ...
    @property
    @abstractmethod
    def agent_name(self) -> str: ...
    @property
    @abstractmethod
    def checkpoint_format(self) -> str | None: ...
    @abstractmethod
    def create_trainer(self, context: TrainerContext) -> TrainerBase: ...


class RlFramework(ABC):
    def __init__(self, providers: Iterable[AgentProvider]) -> None: ...
    @property
    @abstractmethod
    def name(self) -> str: ...
    def supported_agents(self) -> tuple[str, ...]: ...
    def supported_train_backends(self, agent_name: str) -> tuple[str, ...]: ...
    def get_agent_provider(self, agent_name: str, train_backend: str) -> AgentProvider | None: ...


@dataclass(frozen=True)
class TrainerHandle:
    run: RunContext
    trainer: TrainerBase

    def train(self) -> "TrainResult": ...
    def play(self, policy: str) -> None: ...
```

外部实现以 `RlFramework` 作为唯一入口，`register_framework(framework)` 完成注册。`AgentProvider` 不重复声明 `rllib`；provider 索引统一由 `frameworks` 通过 `framework.name`、`provider.train_backend`、`provider.agent_name` 管理。

`TrainerContext` 是传给 trainer 的**单一**上下文对象。它由 `create_trainer_context(...)` 从 `RunContext` 和已组合的 `TaskConfig` 投影得到：`run` 提供 metadata，`num_envs / play_num_envs / seed / logging / checkpoint` 是 framework 统一运行参数，`rl_cfg` 是 provider 的具体 dataclass，`render / resume_from` 是本次运行输入。

### 7.2 trainer 创建

```python
def create_trainer_context(
    run, *, num_envs, play_num_envs, seed, logging, checkpoint, rl_cfg, render=None, resume_from=None
) -> TrainerContext: ...


def create_trainer(context: TrainerContext) -> TrainerHandle:
    metadata = context.run.metadata
    provider = get_agent_provider(metadata.rllib, metadata.algo, metadata.train_backend)
    if provider is None:
        raise ValueError(...)
    context = _validated_trainer_context(provider, context)  # 校验 provider 与 metadata 的 (train_backend, algo) 一致
    return TrainerHandle(run=context.run, trainer=provider.create_trainer(context))
```

`create_trainer` 从 `TrainerContext.run.metadata` 查 provider，校验其 `train_backend / agent_name` 与 metadata 一致，必要时用 provider 的 `checkpoint_format` 补全 context，再创建真实 trainer 并返回框架管理的 `TrainerHandle`。`TrainerHandle.train()` 调用真实 trainer 的 `train()`，并基于 `RunContext` 生成 `TrainResult`。

## 8. 通用编排 runner

`motrix_rl.runner` 提供通用训练编排，入口层不含 framework 分支。

```python
@dataclass(frozen=True)
class TrainRequest:
    config: TrainConfig
    render: RenderConfig | None = None
    runs_root: str | Path = runs.LOG_DIR_PREFIX


def train(request: TrainRequest) -> TrainResult:
    return create_training_handle(request).train()


def create_training_handle(request: TrainRequest) -> TrainerHandle:
    config = request.config
    task = config.task
    method = RlMethod(task.rllib, task.algo)
    train_backend = backend_runtime.resolve_train_backend(
        task.env, method, task.train_backend, utils.get_device_supports()
    )
    provider = frameworks.get_agent_provider(method.rllib, method.algo, train_backend)
    rl_cfg = provider.validate_config(config.algo)
    run = runs.create_run_context(
        env_name=task.env,
        rllib=method.rllib,
        train_backend=provider.train_backend,
        algo=method.algo,
        seed=config.seed,
        checkpoint_format=provider.checkpoint_format,
        runs_root=request.runs_root,
    )
    resume_from = str(checkpoints.resolve_resume_checkpoint_path(config.resume)) if config.resume else None
    return frameworks.create_trainer(
        frameworks.create_trainer_context(
            run,
            num_envs=config.num_envs,
            play_num_envs=config.play_num_envs,
            seed=config.seed,
            rl_cfg=rl_cfg,
            render=request.render,
            resume_from=resume_from,
        )
    )
```

play 编排不引入独立 request 类型：入口（`scripts/play.py`）或 `TrainResult.play(...)` 先解析出 `RunContext`（从 metadata），再调用同一个 `create_trainer(create_trainer_context(run, ...))`，最后 `trainer.play(policy_path)`。best policy 与 resume checkpoint 的定位统一走 `motrix_rl.checkpoints`。

`TrainResult` 是 run 的查询视图：

```python
@dataclass(frozen=True)
class TrainResult:
    run: RunContext

    @property
    def run_dir(self) -> Path: ...
    def find_best_policy(self) -> Path: ...
    def find_resume_checkpoint(self) -> Path: ...
    def play(self, *, cfg_override=None, render=InteractiveRenderConfig()) -> None: ...
```

它只保存 `RunContext`，通过 checkpoint manifest 查询产物；trainer 不手写 result metadata，也不填写 `rllib/train_backend/algo`、`checkpoint_format`、`run_dir` 这类上下文已知字段。

## 9. 配置体系

### 9.1 结构

framework 统一运行参数位于 `TrainConfig` 根节点；算法配置 dataclass 只保存 provider 私有字段：

```python
@dataclass
class TrainConfig:
    task: TaskMeta
    num_envs: int
    play_num_envs: int
    seed: int | None
    logging: LoggingConfig
    checkpoint: CheckpointConfig
    algo: Any


@dataclass
class SkrlCfg:
    models: SkrlModelsCfg = field(default_factory=SkrlModelsCfg)
    memory: SkrlMemoryCfg = field(default_factory=SkrlMemoryCfg)
    agent: SkrlAgentCfg = field(default_factory=SkrlAgentCfg)
    trainer: SkrlTrainerCfg = field(default_factory=SkrlTrainerCfg)


@dataclass
class RslrlCfg(RslrlRunnerCfg):
    pass


@dataclass
class FastSacCfg:
    device: str | None
    agent: FastSacAgentCfg = field(default_factory=FastSacAgentCfg)
    trainer: FastSacTrainerCfg = field(default_factory=FastSacTrainerCfg)
```

provider config 不要求共享内部路径。SKRL 与 FastSAC 直接暴露各组件配置；
RSLRL 的整个 provider config 就是传给上游 `OnPolicyRunner` 的配置，不再额外包装一层 `runner`。
不同算法的 `agent` / `memory` 结构互不共享字段：

| 算法 | agent 配置 | memory |
|---|---|---|
| PPO / RPO | rollouts、gae_lambda、ratio_clip 等 on-policy 字段 | on-policy rollout memory |
| SAC / TD3 | tau、gamma、learning_starts、batch_size、gradient_steps 等 off-policy 字段 | replay memory |
| AMP | PPO 字段 + discriminator / expert dataset 字段 | on-policy + expert memory |

### 9.2 覆写路径

命令行覆写使用 dot-notation，路径以顶层配置为根：

```text
num_envs
play_num_envs
seed
logging.backend
logging.interval
checkpoint.interval
algo.models.<field>          # SKRL
algo.memory.<field>          # SKRL
algo.agent.<field>           # SKRL / FastSAC
algo.trainer.<field>         # SKRL / FastSAC
algo.actor.<field>           # RSLRL
algo.critic.<field>          # RSLRL
algo.algorithm.<field>       # RSLRL
```

## 10. 模型工厂

不同算法需要的模型角色不同：

| 算法 | 模型角色 |
|---|---|
| PPO / RPO | `policy`, `value` |
| SAC | `policy`, `critic_1`, `critic_2`, `target_critic_1`, `target_critic_2` |
| TD3 | `policy`, `target_policy`, `critic_1`, `critic_2`, `target_critic_1`, `target_critic_2` |
| AMP | `policy`, `value`, `discriminator` |
| IPPO | per-agent `policy`, `value` |
| MAPPO | per-agent `policy`, shared/centralized `value` |

模型角色是算法实现内部细节：内置 trainer 在自己的算法 helper 里声明所需角色并把 cfg 映射为模型集合，外部算法在自己的 `TrainerBase` 实现中处理。跨算法只共享**构建块**（activation 解析、MLP body、gaussian/deterministic policy head、Q/V head、obs/state space 校验、privileged critic observation 处理），而不是固定的模型角色组合。

## 11. Memory 体系

- **on-policy（PPO/RPO/AMP）**：rollout memory，`memory_size == -1` 表示使用 `agent.rollouts`。
- **off-policy（SAC/TD3/DDPG）**：replay memory，独立的 `memory_size`、`replacement` 等字段。

Memory factory 不依赖 PPO 字段，`memory_size == -1` 的语义只存在于 on-policy 配置。FastSAC 使用自带的 replay buffer，不经 SKRL memory 抽象。

## 12. Run 目录与 metadata

### 12.1 目录结构

run 目录包含 algorithm 维度，由 `runs.create_run_context(...)` 统一创建：

```text
runs/{env_name}/{rllib}/{train_backend}/{algo}/{timestamp}/
  metadata.json
  checkpoints/
    manifest.json
    latest.pt
    model_0001000.pt
```

对于 SKRL，框架把 `experiment.directory` 设为 `run_dir.parent`、`experiment_name` 设为 `run_dir.name`，避免 SKRL 再追加不可控目录。

### 12.2 metadata.json

每个 run 根目录写入 `RunMetadata`：

```json
{
  "env_name": "g1-walk-flat",
  "rllib": "motrix",
  "train_backend": "torch",
  "algo": "fastsac",
  "seed": 1,
  "created_at": "2026-06-08T19:30:00+00:00",
  "checkpoint_format": "pt",
  "motrixlab_version": null
}
```

`RunMetadata` 是持久化契约，是 play / export / benchmark 自动发现训练产物的唯一可信来源；文件扩展名只能作为 fallback。`RunContext`（定义在 `motrix_rl.runs`）是运行时上下文，包含 `run_dir` 与 `metadata`，并派生 `checkpoint_dir`。`create_run_context(...)` 创建目录、写 metadata、建 `checkpoints/`；`open_run_context(...)` 用于 play / resume / benchmark 读取已有 run。

## 13. Checkpoint 发现与 manifest

checkpoint 发现由 `motrix_rl.checkpoints` 统一实现，入口不做 per-framework 分支。run 在 `checkpoints/manifest.json` 记录实际产出的 artifact；`checkpoints.py` 只提供 manifest 读写与 artifact 解析，不创建 run context，也不内置任何 trainer 名称判断。

manifest 中的路径相对 `checkpoints/manifest.json` 所在目录：

```json
{
  "version": 1,
  "artifacts": {
    "latest_training_state": { "path": "latest.pt", "kind": "training_state", "format": "pt" },
    "best_policy": { "path": "latest.pt", "kind": "policy", "format": "pt" }
  }
}
```

`best_policy` 与 `latest_training_state` 是框架语义，不等同于文件名：

- `best_policy` 用于 play / export / benchmark。
- `latest_training_state` 用于 resume，应包含 optimizer、normalizer、replay buffer、global step 等继续训练所需状态。
- 同一物理文件可同时登记为两个 artifact（例如 FastSAC 的 state dict 既可 play 又可续训）。
- 内置 trainer 在训练结束**以及每次周期性保存**时都应把可播放/可恢复的 checkpoint 保存到标准 `checkpoints/` 目录并调用 `record_checkpoint_artifact(...)` 登记，使中断的 run 也能从 manifest 恢复。

公共 API：

```python
def record_checkpoint_artifact(run_dir, name, path, kind, checkpoint_format=None) -> Path: ...
def best_policy(metadata: RunMetadata, run_dir: Path) -> Path: ...
def resolve_checkpoint_path(checkpoint_or_run_dir, metadata=None) -> Path: ...
def resolve_resume_checkpoint_path(checkpoint_or_run_dir, metadata=None) -> Path: ...
```

manifest 缺失或没有可用 artifact 时，`checkpoints` 直接报错。

## 14. Train backend 自动选择

task 在进入 backend 选择前已经由 Hydra 成功组合。自动选择只需要考虑
"设备能力 + trainer 存在"，由 `motrix_rl.backend_runtime` 集中处理：

```python
def resolve_train_backend(env_name, method, requested_backend, device_supports) -> str:
    if requested_backend is not None:
        _validate_backend(method, requested_backend, device_supports)  # provider 存在 + 设备可用
        return requested_backend

    framework = frameworks.get_framework(method.rllib)
    candidates = [b for b in framework.supported_train_backends(method.algo) if _backend_available(b, device_supports)]
    if not candidates:
        raise ValueError(...)
    return _choose_preferred_backend(candidates)  # 偏好顺序 ("jax", "torch")，其余按名字排序
```

`_backend_available` 对 `jax` / `torch` 做设备能力判断，其他自定义 backend 默认可选。backend 专用 task option 会在 metadata 中明确指定 backend，不进入自动选择。

## 15. 内置 framework 矩阵

当前注册的内置 framework 与其 provider：

| rllib | train_backend | algo | checkpoint 格式 | 说明 |
|---|---|---|---|---|
| `skrl` | `jax` | `ppo` | `pickle` | SKRL PPO，JAX backend |
| `skrl` | `torch` | `ppo` | `pt` | SKRL PPO，PyTorch backend |
| `rslrl` | `torch` | `ppo` | `pt` | RSLRL PPO |
| `motrix` | `torch` | `fastsac` | `pt` | `motrix_rl_builtin` 提供的 Motrix 分布式（C51）FastSAC；配置选择同步或异步拓扑 |

`skrl` 通过 `SkrlPpoTrainerBase` 复用 train/play 主流程与 runtime config，JAX/Torch provider 只提供 backend 特异的 model/memory/agent 构建。`motrix.fastsac` 自带 replay buffer、观测归一化与同步/异步 Trainer；唯一 provider 根据 `FastSacCfg.asynchronous` 选择执行拓扑。

## 16. Multi-agent 扩展点

IPPO/MAPPO 不塞进单智能体 wrapper 的隐式约定，接入前需单独定义环境接口契约：

```python
MultiAgentStep:
    observations: dict[agent_id, obs]
    actions: dict[agent_id, action]
    rewards: dict[agent_id, reward]
    terminated: dict[agent_id, done]
    truncated: dict[agent_id, done]
    infos: dict[agent_id, info]
```

MAPPO 还需 centralized critic state（`global_state | dict[agent_id, critic_state]`）。IPPO/MAPPO 的 Trainer 依赖独立的 `MultiAgentWrapper`，单智能体与 multi-agent 路径在 Trainer 内部分流，不改现有单智能体 wrapper。

## 17. 兼容性与不变量

- `train.py --algo` 接受 `rllib.algo` method，默认 `skrl.ppo`；`--env cartpole` 等价于 SKRL + auto backend + PPO。
- `--algo rslrl.ppo` 等价于 RSLRL Torch PPO；`--rllib rslrl` 作为兼容写法等价，但提示 deprecated。
- `play.py` 不要求 `--algo`，从最新 run metadata 反推 `rllib/train_backend/algo` 选择正确 inference path，可区分 PPO/SAC 等算法。
- `(env, rllib, train_backend, algo)` 可查到独立配置；`(rllib, train_backend, algo)` 可查到 `AgentProvider`。
- run metadata 记录 `env/rllib/train_backend/algo`；没有 metadata/manifest 的旧 run 不参与 play 自动发现。
- 新算法不影响现有 PPO 的默认超参数与训练路径。

## 18. 风险与约束

- **各算法 train backend 支持不一致**：不能假设某算法天然支持所有 train backend，须由 provider 与配置注册显式声明。
- **off-policy 对并行与 replay 设置敏感**：SAC/TD3 的最佳配置不同于大规模并行 PPO 默认值，每个任务需独立配置 `num_envs`、`memory_size`、`learning_starts`、`batch_size`、`gradient_steps` 等。
- **checkpoint 格式不能只靠扩展名推断**：SKRL Torch、RSLRL、FastSAC 都可能用 `.pt` 但内部结构不同，play/export/resume 必须依赖 metadata 与 manifest 中登记的 artifact 语义。
- **multi-agent 是接口级扩展**：涉及环境 step schema、wrapper、配置与 play loop，应在单智能体 off-policy 算法稳定后再推进。

## 19. 相关工作

- SKRL 多算法接入路线：GitLab issue #137。
- SAC 接入：GitLab issue #102。
- TD3 接入：GitLab issue #133。
- RPO 接入：GitLab issue #134。
- IPPO/MAPPO 接入：GitLab issue #135。
- AMP 接入：GitLab issue #136。
