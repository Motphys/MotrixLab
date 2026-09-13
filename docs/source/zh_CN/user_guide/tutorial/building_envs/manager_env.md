# 编写 ManagerEnv 环境

`ManagerEnv` 是 Manager 工作流前端：动作、命令、重置、观测、奖励与终止不在环境类中手写，
而是作为逐项（term）的声明式配置写入 `ManagerBasedEnvCfg` 的各个配置组，由 manager 编译器
生成融合的 Numba 任务程序。它适合项数量多、需要组合复用 mdp 项的任务
（如 locomotion、whole-body tracking）。

与 DirectEnv 的选型对比见[环境构建总览](index.md)。本页先用一个最小骨架展示全貌，
再逐组介绍每一项的职责与定义方式。

## 最小骨架

Manager 工作流不需要子类化 `ManagerEnv`：写好一个 `ManagerBasedEnvCfg` 配置，
再把通用环境类注册到环境名上即可：

```python
from motrix_env_core import registry
from motrix_env_core.manager import (
    ManagerBasedEnvCfg,
    ManagerEnv,
)

# —— 各配置组的声明见下文，这里先省略 ——

@registry.envcfg("my-task")
@configclass
class MyTaskEnvCfg(ManagerBasedEnvCfg):
    scene: SceneCfg = SceneCfg(file="my_model.xml")
    actions: ActionsCfg = ActionsCfg()          # 必须至少有一个动作项
    observations: ObservationsCfg = ObservationsCfg()  # 必须有 policy 观测组
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    sim_reset: ManagerResetCfg = ResetCfg()


registry.env("my-task")(ManagerEnv)
```

配置里的每个组是一个 `@configclass` 数据类，字段名即项名，字段值是项配置对象。
下文按组介绍它们各自做什么、如何定义。

## 生命周期

`ManagerEnv` 同样继承 `ArrayEnv`，step 时序与 auto-reset / truncation 语义与 DirectEnv 一致。
下图把各回调整合进流水线：绿色是 manager kernel（`@dispatch`）执行的部分，
紫色是 manager 的主机侧 Python 回调，蓝色是框架——编排各阶段的宿主与其自身的数据通路：

```{image} /_static/images/tutorial/manager-env-lifecycle-light.svg
:alt: ManagerEnv 单个控制步流水线，包含 done 行分支：apply_action → physics_step → compute_transition → 截断判定 → done? 将 done 行送入 auto-reset 微流程、存活行送往 compute_observation
:class: only-light
```

```{image} /_static/images/tutorial/manager-env-lifecycle-dark.svg
:alt: ManagerEnv 单个控制步流水线，包含 done 行分支：apply_action → physics_step → compute_transition → 截断判定 → done? 将 done 行送入 auto-reset 微流程、存活行送往 compute_observation
:class: only-dark
```

与 DirectEnv 的两点区别：各项回调不在环境类里手写，而由配置组声明、manager 编译器
编译成融合 kernel；kernel 的编译与预热发生在首次 `init_state()`（启动阶段），
step 循环内不再编译。

## 动作项（actions）

**做什么**：每个控制步把策略输出的动作批次变换为执行器控制量并写入仿真器。
各项的动作空间（一维 `Box`、float32）按声明顺序拼接成整个环境的 `action_space`，
策略动作也按同样切分分发给各项。

**如何定义**：一个动作项由两部分组成——

1. `ActionCfg` 子类：声明静态参数与执行器路由，`__call__(env, actuators)` 返回运行时 term。
   `actuator_names` 为空元组 `()` 时控制全部执行器，写显式名称时只控制列出的执行器；
   一个执行器只能归属一项。设为 `None` 表示本项不写执行器目标。
2. `@kernel_data` 装饰的 `ActionTerm` 子类：持有持久状态（`np.ndarray` 字段，按环境行）与
   共享模型数据（`SharedArray` 字段），实现三个方法：`action_space()` 给出本项的动作空间、
   `process(actions)` 把动作批次变换为路由执行器的控制量、`reset(env_ids)` 清理持久状态。

```python
@configclass(kw_only=True)
class MyJointActionCfg(ActionCfg):
    actuator_names: tuple[str, ...] = ()
    scale: float = 0.5

    def __call__(self, env: ManagerEnv, actuators) -> ActionTerm:
        return MyJointAction(scale=np.float32(self.scale), num_envs=env.num_envs,
                             actuators=actuators)


@kernel_data
class MyJointAction(ActionTerm):
    current: np.ndarray              # 持久状态：上一动作缓存（num_envs 行）
    scale: SharedArray               # 静态参数：由 __call__ 传入

    def action_space(self, env, actuators) -> gym.spaces.Box:
        ...  # 由执行器 ctrl_range 推导

    def process(self, actions: np.ndarray) -> np.ndarray:
        ...  # 变换动作，返回本项路由的执行器控制量

    def reset(self, env_ids: np.ndarray) -> None:
        self.current[env_ids] = 0.0
```

## 观测项（observations）

**做什么**：把仿真状态与命令拼装成策略/价值网络的输入。只允许 `policy` 与 `value`
两个观测组，`policy` 必须存在：`policy` 面向 actor（可加噪），`value` 面向 critic
（通常无噪声、可含特权信息）。组内各项的输出按声明顺序拼接成观测向量。

**如何定义**：一个观测 term 是 `ObservationTermCfg` 子类，`__call__(env)` 返回
`ObsTerm(size, dispatch, *args)`。`size` 是本 term 的输出宽度；dispatch 形如
`def xxx_obs(ctx, out, *args) -> None`，把本项的观测写入 `out`——`ctx.sim["key"]`
读取 `queries` 声明的数据，噪声等作为标量参数传入、在 dispatch 内部添加。

```python
@dispatch
def projected_gravity_obs(ctx: ManagerContext, out: np.ndarray,
                          noise_amplitude: np.float32) -> None:
    ...  # 计算并写入 out（宽度 3）
    add_uniform_noise(out, noise_amplitude, ctx.rand.state)


@configclass(kw_only=True)
class ProjectedGravityObsCfg(ObservationTermCfg):
    noise: UniformNoiseCfg = UniformNoiseCfg()

    def __call__(self, env: ManagerEnv) -> ObsTerm:
        return ObsTerm(3, projected_gravity_obs, np.float32(self.noise.amplitude))
```

观测依赖的仿真数据通过环境 `queries` 组或项的 `required_sim_queries()` 声明，
见下文「查询（queries）」一节。

## 奖励项（rewards）

**做什么**：每个控制步对当前状态给出一个标量评价，总奖励是各项按 `weight` 的加权和：
正 `weight` 是奖励，负 `weight` 是惩罚。

**如何定义**：一个奖励 term 是 `RewardTermCfg` 子类（自带 `weight: float` 字段），
`__call__(env)` 返回 `RewardTerm(dispatch, *args)`。dispatch 形如
`def xxx_reward(ctx, *args) -> float`，从 `ctx.sim["key"]` 读取 `queries` 声明的数据。

```python
@dispatch
def base_height_reward(ctx: ManagerContext, target_z: np.float32,
                       sigma: np.float32) -> float:
    error = ctx.sim["robot_base_pos"][2] - target_z
    return math.exp(-(error * error) / (sigma * sigma))


@configclass(kw_only=True)
class BaseHeightRewardCfg(RewardTermCfg):
    target_z: float
    sigma: float

    def __call__(self, env: ManagerEnv) -> RewardTerm:
        return RewardTerm(base_height_reward,
                          np.float32(self.target_z), np.float32(self.sigma))
```

## 终止项（terminations）

**做什么**：每个控制步对每个环境输出一个布尔值，任一项为真即回合 `terminated`
（区别于超时 `truncated`）。

**如何定义**：一个终止 term 是 `TerminationTermCfg` 子类，`__call__(env)` 返回
`TerminationTerm(dispatch, *args, metric_names=(...))`。dispatch 形如
`def xxx_termination(ctx, *args) -> bool`；`metric_names` 可选，把 dispatch 写入
`ctx.metrics` 的每环境量登记为可读指标。

```python
@dispatch
def bad_orientation_termination(ctx: ManagerContext, threshold: np.float32) -> bool:
    ...
    return tilt_sq > threshold * threshold


@configclass(kw_only=True)
class BadOrientationTerminationCfg(TerminationTermCfg):
    threshold: float

    def __call__(self, env: ManagerEnv) -> TerminationTerm:
        return TerminationTerm(bad_orientation_termination,
                               np.float32(self.threshold),
                               metric_names=("base_tilt",))
```

## 重置项（sim_reset）

**做什么**：回合重置（以及命令项请求的中途重算，见命令一节）时，改写选中环境行的
仿真器状态——初始位姿、速度、关节角等，由 reset kernel 按声明顺序执行。

**如何定义**：一个重置 term 是 `ResetTermCfg` 子类，`__call__(env)` 返回
`ResetTerm(dispatch, *args, writes={...})`。`writes` 声明本 term 要写哪些仿真状态
（`BodyPositionWrite`、`JointVelocityWrite` 等，键名即 dispatch 里的写入通道）；
dispatch 计算写入值。

```python
@dispatch
def _reset_body_pos(ctx: ManagerContext, sim_writes, spawn, noise) -> None:
    position = sim_writes["position"]
    ...  # 写入带随机化的初始位置


@configclass(kw_only=True)
class BodyPosResetCfg(ResetTermCfg):
    spawn: tuple[float, float, float]
    noise: tuple[float, float, float] = (0.02, 0.02, 0.005)

    def __call__(self, env: ManagerEnv) -> ResetTerm:
        return ResetTerm(
            _reset_body_pos,
            tuple(np.asarray(self.spawn, dtype=np.float32)),
            tuple(np.asarray(self.noise, dtype=np.float32)),
            writes={"position": BodyPositionWrite((base_link,))},
        )
```

## 命令项（commands）

**做什么**：维护命令状态并在每个周期推进——速度指令采样、curriculum、参考动作推进等。
命令项还产出每环境指标：dispatch 内写入 `ctx.metrics["name"]`，即可通过
`env.state.metrics` / `env.metrics` 读取。大多数任务直接复用内置命令即可，
不需要自定义。

**如何定义**：一个命令 term 是 `CommandCfg` 子类，`__call__(env)` 返回
`@kernel_data` 的 `CommandTerm` 子类，实现四个回调：

| 回调                     | 执行位置          | 职责                                       |
| ------------------------ | ----------------- | ------------------------------------------ |
| `update(ctx)`            | evaluate kernel   | 更新本周期派生命令数据                     |
| `advance(ctx)`           | transition kernel | 推进持久命令状态；置 `ctx.sim_reset_requested` 可请求该行在步骤末尾重算仿真状态 |
| `reset(ctx)`             | 主机侧            | 为重置的环境行准备数据（含终止统计）       |
| `reset_env(ctx)`         | reset kernel      | 重置持久命令状态                           |

## 查询（queries）

**做什么**：声明各项需要的仿真数据与结构信息，由框架编译成批量读取程序。
`queries.data` 是仿真数据查询（`JointPositionQuery`、`LinkPositionQuery` 等），
`queries.model` 是模型查询（`ActuatorKpQuery`、`BodyJointPositionLimitsQuery` 等）。

**如何声明**：通常在配置的 `__post_init__` 中根据机器人配置填充：

```python
def __post_init__(self) -> None:
    joint_names = ...  # 由 scene.objs.robot 解析
    base_link = ...    # 基座 link 名
    self.queries.data["robot_dof_pos"] = JointPositionQuery(joints=joint_names)
    self.queries.model["robot_joint_position_limits"] = BodyJointPositionLimitsQuery(body=base_link)
```

观测项也可以通过 `required_sim_queries(env_cfg)` 声明自己需要的查询；
所有声明按 key 合并，同名 key 必须完全一致，否则配置失败。查询的编译与执行机制见
[SimBackend：与仿真器解耦](sim_backend.md)。

## 注册与变体

- **配置先于环境类注册**；registry 根据类继承关系推断前端类型
  （`ManagerEnv` 与 `DirectEnv` 同属 `"np"` 数据后端），不要手动指定。
- 配置 docstring 首行与 `zh_CN:` 行分别作为环境的英文、中文描述。
- 同一环境的多个 preset 通过配置 factory 派生：子变体只覆写与共享配置的差异项
  （例如调低某个奖励权重）。

以内置 `microduck-ball-balance` 为例（源码见
`motrix_envs/src/motrix_envs/locomotion/ball_balance/microduck.py`）：

```python
@registry.envcfg("microduck-ball-balance")
def make_microduck_ball_balance_cfg() -> MicroduckBallBalanceEnvCfg:
    """Balance on top of a basketball with Microduck.

    zh_CN: 让 Microduck 双脚站在篮球上并保持平衡。
    """
    return MicroduckBallBalanceEnvCfg()


registry.env("microduck-ball-balance")(ManagerEnv)
```

注册完成后，`python scripts/view.py env=<name>` 可以预览，按
[Task 配置](../training/task_config.md)创建训练 Task 即可开始训练。
