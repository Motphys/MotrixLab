# 项目整体架构设计

## 摘要

本文定义 MotrixLab 的整体架构设计。MotrixLab 是一套基于 MotrixSim 仿真引擎的强化学习训练框架，核心目标是让环境定义与 RL 框架解耦，使三类开发者——环境开发者、RL 算法开发者、应用开发者——能够在各自的边界内独立工作。

## 1. 架构目标

MotrixLab 需要同时应对以下挑战：

- 同一个环境可能被多种 RL 框架（SKRL、RSLRL）训练，每种框架又有自己的数据格式和接口协议；
- 环境数量在持续增长，新增环境不应触及框架核心代码；
- 训练超参数需要按环境、按框架灵活配置和继承复用。

因此，架构必须解决以下问题：

1. 环境定义如何与 RL 框架保持独立；
2. 新增环境和新增 RL 框架如何不修改已有代码；
3. 配置如何在保持灵活性的同时避免散乱。

## 2. 总体分层

### 2.1 层级结构

系统从底向上分为四层：

```
┌─────────────────────────────────────────────────┐
│ 应用入口                                         │
│ mxlab train · mxlab play · mxlab view            │
├─────────────────────────────────────────────────┤
│ RL 框架集成层（motrix_rl）                        │
│ Wrapper · Trainer · AgentProvider · Hydra schema │
├─────────────────────────────────────────────────┤
│ 环境定义层（motrix_envs）                         │
│ ABEnv · DirectEnv · Env Registry                 │
├─────────────────────────────────────────────────┤
│ 仿真后端                                         │
│ motrixsim (NumPy)                               │
└─────────────────────────────────────────────────┘
```

每一层只依赖其下层，不依赖上层：

- **仿真后端**提供物理步进、模型加载和场景数据；
- **环境定义层**基于仿真后端定义 obs / reward / termination 逻辑；
- **RL 框架集成层**把环境包装成各 RL 框架所需的接口，管理训练配置和训练流程；
- **应用入口**解析命令行参数，选择环境和框架，启动训练或评估。

### 2.2 两包工作空间

项目由两个 UV workspace 包组成：

- `motrix_envs`：仅依赖仿真后端，不依赖任何 RL 框架；
- `motrix_rl`：依赖 `motrix_envs`，同时依赖 RL 框架。

这个划分确保环境定义的纯粹性——环境代码永远不会引入 PyTorch、JAX 或任何 RL 库。

## 3. 核心架构原则

### 3.1 Model / Data / State 分离

与 MotrixSim 一致，环境层严格区分三个概念：

- **Model（SceneModel / SimModel）**：静态的场景结构、关节拓扑、物理参数。由 MJCF 文件编译产生，整个训练过程不变。
- **Data（SceneData / WorldData）**：运行时可变状态，包括 DoF 位置、速度、执行器控制量。同一个 Model 可以创建多份 Data 实例实现批量仿真。
- **State（ArrayEnvState）**：环境层在 Data 之上附加的训练相关状态——obs、reward、terminated、truncated、
  episode_steps、info。

这个三层分离带来的核心收益：

- 同一个 Model 支撑任意数量的并行环境实例；
- 模型加载和编译只执行一次，运行时只推进状态；
- 环境逻辑（reward、termination）与物理引擎数据表示正交。

### 3.2 环境定义与 RL 框架解耦

环境定义层（`motrix_envs`）和 RL 集成层（`motrix_rl`）之间通过抽象基类 `ABEnv` 建立契约。环境只需暴露：

- `observation_space` / `action_space`
- `step(actions) -> State`
- `init_state() -> State`

RL 框架集成层通过 Wrapper 把这些通用接口适配到具体框架：

```
ABEnv (motrix_envs)
  │
  └── DirectEnv ─→ SkrlNpWrapper   ──→ SKRL Trainer
               ─→ RslrlNpEnvWrap ──→ RSLRL Trainer
```

Wrapper 是唯一允许同时引用 `motrix_envs` 类型和 RL 框架类型的地方。环境代码不知道自己会被哪个框架训练；RL 框架代码通过 Wrapper 屏蔽仿真后端差异。

### 3.3 注册表驱动的松耦合

系统通过两个注册表实现组件发现，避免硬编码的 import 链：

- **Env Registry（`motrix_env_core.registry`）**：将字符串名称映射到 `(EnvCfg factory, EnvCfg type, {backend: EnvClass})`，并通过 `sim` 参数选择 backend。首次环境查询时发现 `motrix_env_core.envs` package entry points，调用各提供方的零参数注册函数；内置提供方为 `builtin = "motrix_envs:register"`。提供方以现有装饰器注册环境，每个进程完成一次发现，支持注册回调重入查询；插件冲突或加载失败明确报错。环境发现不依赖 RL 或 Hydra Task 配置。
- **Hydra task config**：通过 `task=<env>/<rllib>.<algo>[.<backend>]` 组合环境、算法与运行配置。

环境发现按 provider 名称排序并在加载前拒绝同名提供方。回调重入查询只看到当前已注册的条目，不启动另一轮发现。
后续提供方加载或注册失败时，已完成的回调不会重复执行；失败回调的部分注册副作用保留，恢复由提供方负责。

注册通过装饰器触发，发生在模块导入时：

```python
# motrix_envs 侧
@registry.envcfg("cartpole")  # 注册配置
class CartPoleEnvCfg(EnvCfg): ...


@registry.envcfg("g1-wbt")
def make_g1_wbt_cfg() -> WbtEnvCfg:
    return WbtEnvCfg(scene=..., motion_files=(...,))


@registry.env("cartpole")  # 注册实现，backend 自动推断
class CartPoleEnv(DirectEnv): ...


# motrix_rl 侧
@rlcfg("cartpole")  # 注册 RL 配置，框架自动推断
class CartPoleSkrlPpo(SkrlCfg): ...
```

注册表带来的约束：

- 配置必须先于实现注册（`envcfg` 先于 `env`）；
- 同一环境同一后端不允许重复注册；
- RL 配置依赖环境已注册——不存在的环境名会在注册时立即报错。
- `envcfg` 可以注册配置类，也可以注册零参数 factory；factory 的返回类型标注作为 config type，缺少返回类型标注时才需要显式 `cfg_type`。`registry.make()` 每次调用 factory 获得新的配置实例。

### 3.4 配置层次与继承

环境声明式配置使用 `@configclass`，分为两个正交维度：

**环境配置（EnvCfg 体系）**：定义仿真参数、场景来源和任务参数。所有环境统一通过
`scene: SceneCfg` 配置场景；`scene.file` 可指定完整基础模型文件，asset、visual 与 object 配置可继续叠加到该基础 World，`scene.system_camera` 配置 interactive viewer 与视频录制使用的系统相机视角。
物理引擎参数集中在 `sim: SimCfg`，其中 `dt`、solver iterations/tolerance 与 gravity 在 MSD World build 前写入 `simulate_option`；`ctrl_dt` 仍表示环境控制周期。
`SceneCfg` 通过字段式 `SceneAssetsCfg` 和 `SceneObjsCfg` 分别保存具名 asset 与有序 scene object；registry 字段名直接作为 MSD 名称，object 的 dataclass 字段顺序就是组装顺序。所有场景对象配置继承 `SceneObjCfg`。完整 MJCF 场景由 `SceneCfg.file` 表达；`RobotCfg` 通过 `ModelFileCfg` 组合 `MjcfFileCfg` 或 `UrdfFileCfg` 模型来源；程序化几何体由 `GeomCfg` 抽象。
详细设计见 [Scene / Robot 配置设计](./scene-model-config.md)。

```
EnvCfg
 ├── sim: SimCfg
 ├── scene: SceneCfg
 └── CartPoleEnvCfg (reset_noise_scale, ...)
 └── Go1WalkCfg (...)
```

**训练配置（RL Cfg 体系）**：定义超参数。每个 RL 框架有自己的配置树。

```
SkrlCfg
 ├── SkrlModelsCfg (SkrlPolicyCfg, SkrlValueCfg)
 ├── SkrlMemoryCfg
 ├── SkrlAgentCfg (SkrlAgentExperimentCfg)
 └── SkrlTrainerCfg

RslrlCfg (继承 RslrlRunnerCfg)
 ├── RslRlActorCfg
 ├── RslRlCriticCfg
 └── RslRlPpoAlgorithmCfg
```

配置类继承用于扩展 schema，即新增配置字段。仅覆盖已有字段值的环境 preset 使用构造函数和零参数 factory，不为值覆写创建空壳子类：

```python
@registry.envcfg("g1-wbt")
def make_g1_wbt_cfg() -> WbtEnvCfg:
    return WbtEnvCfg(
        motion_files=("...",),
        tracked_body_names=(...),
        reference_body_name="torso_link",
    )
```

训练配置中的类继承仍可用于复用同一 schema 下的初始化逻辑：

```python
@rlcfg("go1-walk-flat")
class Go1Flat(SkrlCfg): ...


@rlcfg("go1-walk-rough")
class Go1Rough(Go1Flat):  # 继承 flat 的基础参数
    def __post_init__(self):
        super().__post_init__()
        self.models.policy.hiddens = [512, 256, 128]  # 只覆写差异
```

配置覆写使用 Hydra dot-notation（如 `algo.agent.learning_rate=1e-3`）；seed、logging、checkpoint 等统一运行参数位于根配置。

## 4. 仿真后端

### 4.1 单后端设计

系统使用 NumPy 作为仿真后端，对应一条完整的数据通路：

|      | NumPy 后端 (np) |
| ---- | ------------- |
| 仿真引擎 | `motrixsim`   |
| 数据类型 | `np.ndarray`  |
| 环境基类 | `DirectEnv`    |
| 状态类型 | `ArrayEnvState` |
| 执行设备 | CPU           |
| 重置方式 | 返回新 obs       |

### 4.2 环境生命周期

所有后端共享相同的 step 生命周期：

```
step(actions)
  1. prev_physics_step()    ← 清零 reward/terminated/truncated
  2. apply_action(actions)  ← 子类：将 action 写入仿真状态
  3. physics_step()         ← 执行 sim_substeps 次物理步进
  4. compute_transition()   ← 子类：计算 transition 输出
  5. update_truncate()      ← 基类：检查 max_episode_steps
  6. reset_done_envs()      ← 基类：自动重置已终止的环境
  7. compute_observation()  ← 子类可选：在重置后计算 observation
```

步骤 2 和 4 是子类实现的环境逻辑，步骤 7 用于需要 post-reset observation 的环境；其余由基类统一管理。
这保证了所有环境共享一致的 episode 管理、truncation 和 auto-reset 语义。

## 5. RL 框架集成

### 5.1 Wrapper 层

Wrapper 的职责是格式转换，不包含任何环境逻辑或训练逻辑：

- **DirectEnv → SKRL**：`np.ndarray` → `torch.Tensor`（通过 `torch.tensor` 拷贝）
- **DirectEnv → RSLRL**：`np.ndarray` → `torch.Tensor`，obs 封装为 `TensorDict`

Wrapper 还负责将渲染器延迟初始化封装在 `render()` 调用中。

### 5.2 Trainer

每个 RL 框架提供一个 `Trainer` 类，封装从配置到训练完成的完整流程：

```
Trainer.__init__(context)
  1. 接收 Hydra 已组合并类型校验的 rl_cfg
  2. 接收日志、checkpoint 和运行上下文
  3. 初始化框架训练组件

Trainer.train()
  1. registry.make() 创建环境
  2. wrap_env() 包装为框架接口
  3. 构建模型、Agent、Runner
  4. 启动训练循环
```

Trainer 是应用入口唯一需要接触的 RL 层类。应用入口通过 `--rllib` 和 `--train-backend` 参数选择引入哪个 Trainer，实现 RL 框架的延迟导入。

## 6. 环境组织

### 6.1 目录结构

环境按任务类别组织为三个分组：

```
motrix_envs/src/motrix_envs/
├── basic/          ← 基础控制任务（cartpole, pendulum, acrobot, ...）
├── locomotion/     ← 运动任务（go1, go2, anymal_c）
└── manipulation/   ← 操作任务（franka, shadow_hand, rm65）
```

每个环境是一个独立子目录：

```
cartpole/
├── cfg.py             ← @envcfg 配置
├── cartpole_np.py     ← @env DirectEnv 实现
├── __init__.py        ← 触发注册
└── cartpole.xml       ← MJCF 模型文件
```

### 6.2 对应的 RL 任务配置

每个环境在 `motrix_rl/src/motrix_rl/tasks/` 下有对应的任务文件，注册该环境的训练超参数。一个任务文件可以包含同一环境面向多个 RL 框架的配置。

## 7. 数据流总览

以 `mxlab train task=cartpole/skrl.ppo` 为例，数据在系统中的完整流动路径：

```
命令行参数
  │
  ▼
mxlab train → motrix_rl.commands.train: 组合训练配置
  │
  ▼
Hydra: 组合 task=cartpole/skrl.ppo → TrainConfig
  │
  ▼
motrix_env_core.registry.make("cartpole"): 查找 CartPoleEnvCfg + CartPoleEnv(DirectEnv)
  │                                      加载 cartpole.xml → SceneModel
  │                                      创建 SceneData(batch=[2048])
  │
  ▼
wrap_env(): DirectEnv → SkrlNpWrapper（torch.Tensor 接口）
  │
  ▼
Trainer.train(): 构建 PPO Agent + SequentialTrainer → 训练循环
  │
  ▼
每个 step:
  SkrlNpWrapper.step(torch.Tensor)
    → actions.cpu().numpy()
    → CartPoleEnv.step(np.ndarray) → ArrayEnvState
    → torch.tensor(state.obs) → 返回给 PPO
```
