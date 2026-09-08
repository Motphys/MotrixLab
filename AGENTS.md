# AGENTS.md

本文件为 AI 编码 agent 在本仓库中工作时提供指引。

## 项目概览

MotrixLab 是构建在 MotrixSim 仿真后端之上的强化学习框架，提供统一的多仿真后端训练接口，并集成 SKRL 与 RSLRL。
框架面向机器人仿真，内置 cartpole、locomotion、manipulation 等多种环境。

## Workspace 结构

项目使用 UV workspace，包含九个 package：

- `motrix_env_core`：backend 无关的环境框架（不依赖任何 simulator）
- `motrix_env_motrixsim`：MotrixSim 仿真后端（SimBackend、scene compiler、renderer、torch frontend）
- `motrix_env_mujoco`：仅用于编译的 MuJoCo scene 后端
- `motrix_envs`：内置环境、机器人模型与任务资产
- `motrix_rl`：RL 框架集成与训练工具
- `motrix_deploy`：框架无关的 artifact、运行时契约与部署 CLI
- `motrix_deploy_mujoco`：SceneCfg 支撑的 MuJoCo 后端插件
- `motrix_deploy_unitree`：Unitree SDK2 DDS 硬件后端插件
- `motrix_deploy_tasks`：具体部署任务实现

`scripts/` 提供训练、可视化、评估入口。内置环境由环境 package 注册并通过项目 registry 暴露；simulator backend
通过项目 backend registry 选择，core package 不依赖任何具体 simulator 实现。

## 编码规范

框架接口在实现单个功能时可以保持通用，但实现范围必须由当前具体需求驱动。不要为未提出的未来场景添加投机性的
provider、兼容路径、配置变体、registry 或中间抽象。

### Python 代码风格

- `motrix_env_core` 与 `motrix_envs` 声明式配置类使用 `@configclass`；runtime state、result、metadata 等普通数据对象继续使用 `@dataclass`
- 避免过度防御式编程。只在系统边界（例如用户输入、文件、网络和第三方接口）或明确的公共契约处做必要校验；
  内部调用应信任类型标注和上游已建立的契约，不要重复检查类型、shape、有限性或取值范围，也不要为假设性错误增加辅助函数和分支

### Package Initializer

默认保持 `__init__.py` 精简。包内代码（包括兄弟模块）应从定义符号的模块导入符号，而不是把 package initializer
当作内部聚合门面。例如使用 `from package.feature.command import CommandCfg`，而不是为了让附近代码能写
`package.feature.CommandCfg` 而通过 `package.feature` re-export `CommandCfg`。

只有当 package 是有意设计且有文档说明的公共 API 边界时，才从 `__init__.py` re-export 符号。这样的门面应有明确的
package 级用途、显式的 `__all__`，以及预期依赖该稳定 import 路径的真实使用方。不要创建与现有公共命名空间重复的
第二个门面。

当 `__init__.py` 中的 import 是注册或插件副作用所必需时，同样允许，但要用注释标明该意图。不要把副作用模块 import
与从同一模块冗余 import 符号组合在一起——import 那些符号本身就会执行该模块。

Review 或修改 initializer 时：

- 区分公共 API/注册 import 与兄弟代码的便捷 re-export；
- 移除 re-export 前先把内部使用方迁移到具体定义模块；
- 保留已确立的用户可见命名空间和跨 package 契约；
- 检查仓库使用方、文档、环境/插件注册和 import 环风险；
- 清理后验证 package import 与注册行为。

### Manager Kernel 入口方法

Manager fused-kernel 入口方法必须用 `@dispatch` 装饰，绝不使用 `@njit`——Numba 编译、缓存和内联由 manager
编译器统一管理。`@njit` 只用于被编译 kernel 调用的独立 Numba 兼容 helper。哪些方法是 kernel 入口、哪些是普通
Python 方法，由 manager 运行时契约定义（`wiki/design/manager/runtime.md`）。

### 环境配置与注册规范

- 环境配置使用 `@configclass`，并继承与运行时一致的配置基类：直接工作流继承 `DirectEnvCfg`，Manager 工作流继承 `ManagerBasedEnvCfg`；其他公共配置基类不得重复继承 `EnvCfg`。
- 环境配置通过 `@registry.envcfg("name")` 注册。注册对象可以是配置类，也可以是返回具体 `EnvCfg` 子类的零参数 factory；factory 必须提供返回类型标注。
- 配置注册必须先于环境类注册。环境类通过 `@registry.env("name")` 或等价的 `registry.env("name")(EnvClass)` 注册。
- 环境类必须继承受支持的运行时基类：直接 NumPy 工作流使用 `DirectEnv`，Manager 工作流使用 `ManagerEnv`，Torch 工作流使用 `TorchEnv`。registry 根据类继承关系推断前端类型，不要手动指定 backend 类型。
- Simulator backend 是配置级别的字符串选择：`EnvCfg.backend` 使用 registry 中的 backend 名称；`None` 使用默认 backend。Direct/Manager 环境通过 `self.sim` 使用 backend-neutral `SimBackend` 接口，不得让具体 simulator 类型跨越 core 边界。

### RL 任务配置规范

- RL provider 配置使用对应 provider 的 `@dataclass` 配置类：SKRL 使用 `motrix_rl.skrl.config.SkrlCfg`，RSLRL 使用 `motrix_rl.rslrl.cfg.RslrlCfg`。
- provider 配置的字段结构必须与对应的 `configs/algo_base/*.yaml` 保持一致；不要使用已经删除的 `@rlcfg` 或旧的嵌套结构。
- 超参数在配置对象初始化或明确的配置组装流程中赋值；同一环境的多个变体应通过配置复用，子变体只覆写实际差异。
- SKRL 和 RSLRL 的 provider 配置结构不同，不要为了统一接口强行增加无用的兼容层；框架通过 provider 的 `config_type` 和 trainer context 选择对应实现。

### 通用环境要求

- `observation_space` 和 `action_space` 作为 property 定义。
- 常量（初始位置、空间定义、query 名称等）在 `__init__` 或配置构造阶段预计算，不在 step 循环中重复创建。
- NumPy/Array 环境中使用向量化操作，禁止 Python for 循环遍历 environment 维度；遍历固定数量的关节、脚或 term 配置可以使用普通循环。
- reset、step、auto-reset、truncation 和 termination 的语义必须遵循 `ArrayEnv`/`DirectEnv` 的生命周期，不要在子类中重复实现前端生命周期管理。

### 测试规范

- `tests/` 下的测试应面向通用功能、公共接口、行为契约和不变量，确保实现变化不会破坏框架能力。
- 不应针对具体任务配置中的固定数值编写测试，例如 reward scale、默认姿态、训练超参数或其他可能随调参变化的值。
- 不应为纯配置的缺失状态编写负向单元测试，例如断言某个环境注册名、配置字段、preset 或兼容 alias 不存在。删除这类配置时，
  应通过引用扫描、静态检查和现存功能的正向行为测试完成验收。
- 只有当某个配置值本身属于公共契约或用于复现特定缺陷时，才应断言其具体值，并在测试中说明原因。
- 重构过程中用于确认旧模块、旧路径、旧文件或旧 import 已删除的扫描属于一次性迁移验收，不应保留为长期单元测试；应在开发和评审阶段使用 grep、构建产物检查等方式完成。只有旧行为仍属于明确的兼容性公共契约时，才为其保留测试。
- 测试应覆盖 backend 选择、query 声明/解析、reset 生命周期、termination/truncation 区分以及关键的配置契约；不要只验证环境能否实例化。

### 版本与依赖一致性

- 所有 workspace package（见上文 Workspace 结构，共九个）的 `pyproject.toml` 中 `version` 字段必须保持一致。
- MotrixSim 相关依赖版本必须在使用该依赖的 workspace package 之间保持一致。
- 关键第三方依赖使用精确版本锁定（`===`）；新增或升级依赖时同步更新 `uv.lock`。

## 开发环境

项目使用 UV 管理依赖，Python 版本要求 3.10.\*。可用的依赖组与 extras 见 `pyproject.toml`。

完整安装（含全部 package、依赖组与 extras）：

```bash
uv sync --all-packages --all-groups --all-extras
```

按框架安装：

```bash
uv sync --all-packages --extra skrl-jax   # SKRL JAX backend
uv sync --all-packages --extra skrl-torch # SKRL PyTorch backend
uv sync --all-packages --extra rslrl      # RSLRL（PyTorch）
```

## 常用命令

train/play/view CLI 由 [Hydra](https://hydra.cc/) 驱动，参数使用 `key=value` 语法（不是 `--flag`）。可用选项见
`configs/*.yaml`。

### 训练

SKRL（默认）：

```bash
uv run scripts/train.py task=cartpole/skrl.ppo
```

RSLRL：

```bash
uv run scripts/train.py task=cartpole/rslrl.ppo
```

直接覆写框架运行时设置和类型化 RL 参数：

```bash
uv run scripts/train.py task=cartpole/skrl.ppo num_envs=64 algo.agent.learning_rate=1e-3
uv run scripts/train.py task=cartpole/skrl.ppo logging.interval=20 checkpoint.interval=100
```

### 环境可视化

不训练只查看环境：

```bash
uv run scripts/view.py env=cartpole
```

查看内置机器人（不创建 RL 环境）：

```bash
uv run scripts/view.py robot=g1-29dof
```

### 评估

```bash
uv run scripts/play.py env=cartpole
```

指定 policy 文件：

```bash
uv run scripts/play.py env=cartpole policy=<path/to/best.[pickle/pt]>
```

### ONNX 导出

```bash
uv run scripts/export_onnx.py run_dir=<run-dir> output=/tmp/policy.onnx
```

### 渲染

训练时开启可视化：

```bash
uv run scripts/train.py task=cartpole/skrl.ppo render=true
```

### TensorBoard

```bash
uv run tensorboard --logdir runs/{env-name}
```

### 测试

```bash
uv run pytest
```

## 架构要点

- **Backend 无关 core**：`motrix_env_core` 不 import 任何 simulator；前端只通过 `motrix_env_core.sim.registry` 解析 backend。
- **多后端训练**：同一环境支持不同仿真 backend，backend 是配置级别的字符串选择。
- **环境命名**：简单字符串标识（例如 "cartpole"）。
- **自动 backend 选择**：SKRL 下训练脚本根据 GPU 可用性自动选择 JAX 或 PyTorch；RSLRL 仅使用 PyTorch。
- **结果存储**：训练结果保存在 `runs/{env-name}/` 目录结构中，包含 checkpoint 和 TensorBoard 日志。

## 备注

- JAX 与 PyTorch backend 均支持 CUDA。
- MotrixSim package 使用内部 PyPI server。
