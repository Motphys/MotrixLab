# Manager Runtime 设计

## 摘要

`ManagerEnv` 是基于 `ArrayEnv` 的 NumPy 前端，使用配置声明 action、command、observation、reward、termination 和 reset，
数值执行由通用 Torch 执行器按整批张量调度（见 [Torch 执行器设计](./torch-executor.md)）。它保持现有环境公共契约：action 和
observation 使用 NumPy，状态使用 `ArrayEnvState`，仿真通过 backend-neutral 的 `SimBackend` 访问。

本文集中说明 Manager 的运行时边界、构造与生命周期、term ownership，以及执行和性能约束。任务配置接口见
[Manager Task API](./task-authoring.md)，kernel 数据 ABI 见 [Manager Kernel ABI](./kernel-abi.md)，仿真后端边界见
[SimBackend](./sim-backend.md)。

## 1. 运行时边界

```text
ArrayEnv lifecycle
  ├── action terms（host）
  ├── SimBackend.step / read program（backend）
  ├── task program（Torch executor，整批张量调度）
  ├── observation / reward / termination outputs
  └── partial reset（host + backend + reset kernel）
```

`ManagerEnv` 与 Direct 环境共享 `ArrayEnv` 生命周期，不是新的数据 backend：

- `ManagerEnv` 继承 `ArrayEnv`，registry 将其归类为 NumPy 前端；
- `DirectEnv` 和 `ManagerEnv` 都通过 registry 的 `sim` 参数选择 `SimBackend`；未指定时使用注册的默认 backend。
- `motrix_env_core` 的 Manager 代码不得 import 具体 simulator 类型；
- MotrixSim 对象、backend handle、Python 容器和 `info` 字典停留在 host 边界；
- term dispatch 只接收固定的 `ManagerContext` 视图和 flat ndarray/scalar leaves（整批张量）。

Manager 不是完整的 `step()` 替代品，也不编译物理推进。物理时间只由 control/training loop 推进一次，reset 不推进物理。

## 2. 配置到运行时

构造阶段由 `ManagerBasedEnvCfg` 完成：

```text
ManagerBasedEnvCfg
  -> normalize manager groups
  -> create concrete terms
  -> canonicalize kernel-data dataclass terms
  -> compile model/read/write/reset programs
  -> construct ManagerContext layout
  -> warm up generated kernels
```

配置只描述 term 及其参数，不保存 environment-local 的运行期 ndarray。每个 concrete term 在环境构造时创建并 canonicalize 一次；
Manager 持有该 canonical instance，host lifecycle 和 compiled kernel 共享同一份 backing arrays。

Manager group 的顺序是行为契约的一部分：action slice、observation layout 和 term 执行顺序均由配置映射顺序决定。配置字段支持普通
`dict` 或对应的 typed group，但归一化后必须得到稳定的有序映射。

## 3. 生命周期

### 初始化

1. backend 构造并完成 scene 编译；
2. 解析并编译 model、read、write 和 reset declarations；
3. 创建 action、command、observation、reward 和 termination terms；
4. 为每个 term 执行一次 `canonicalize_kernel_data`；
5. resolver（`ManagerProgramResolver`）构造 `ManagerContext` layout、read plan 和
   per-term-kind typed call records：term dispatch 以签名校验（`inspect` + 类型标注），不做任何
   代码生成或 dispatcher 编译；
6. Torch 执行器按元数据调度；首次真实调用完成惰性 specialization，`warmup()` 用克隆
   输入执行一次 evaluate/observe 预热。

### Step

1. Manager 按 action layout 切片并分发给 action terms；
2. Manager 按 actuator route 合并 controls，再由 write program 写入 backend；
3. backend 推进 `sim_substeps`；
4. command term 在 host 侧更新 persistent state；
5. read program 刷新 simulator snapshot；
6. Torch 执行器执行 command evaluation、observation、reward 和 termination（一次整批调用/term）；
7. `ArrayEnv` 处理 truncation 和 done reset；`state.metrics` 是常驻活视图（kernel 原地写），跨步持有走 `process_metrics()` 归约快照。

### Partial reset

reset 只处理指定的原始 `env_ids`：

- action/command/reset terms 只修改目标 rows；
- backend reset program 只写目标 rows；
- read program 只刷新目标 rows；
- reset 路径按原始 environment id gather lane 状态，不把 reset batch row 当作环境 id；
- `ArrayEnv` 继续负责 auto-reset、episode steps 和 truncation 语义。

## 4. Term 协议

Term 不要求继承统一实现基类，但必须遵循对应 protocol：

```python
@dataclass(frozen=True, slots=True)
class BaseActionState:
    action_queue: torch.Tensor  # host (N, W, A), lane (W, A); W >= 2
    action_ptr: torch.Tensor    # one shared cursor, advanced once per control step


@dataclass(frozen=True)
class ActionTerm:
    action_space: gym.spaces.Box
    state: BaseActionState
    process: Callable  # process(state, actions_batch) -> controls_batch | None
    reset: Callable    # reset(state, env_ids) -> None


class ManagerContext:
    env_id: np.ndarray
    actions: Map
    commands: Map
    rand: RandValue
    sim: Map


class ObservationTerm:
    size: int
    dispatch: Callable[..., None]
    args: tuple[Any, ...]

    # dispatch(ctx, out, *args) -> None


class RewardTerm:
    dispatch: Callable[..., torch.Tensor]
    args: tuple[Any, ...]

    # dispatch(ctx, *args) -> (N,) float32 tensor


class TerminationTerm:
    dispatch: Callable[..., torch.Tensor]
    args: tuple[Any, ...]
    metrics: dict[str, np.ndarray]

    # dispatch(ctx, *args) -> (N,) bool tensor; metrics written via ctx.metrics


class CommandTerm:
    def reset(ctx: ResetContext) -> None: ...  # host: prepare shared reset data for episode resets
    @staticmethod
    def reset_env(receiver, ctx) -> None: ...  # @dispatch: batched, lane-gathered reset rows
    @staticmethod
    def advance(receiver, ctx) -> None: ...  # @dispatch: batched, whole batch
    @staticmethod
    def update(receiver, ctx) -> None: ...   # @dispatch: batched, whole batch
```

Command term 的生命周期拆成 batched hooks 与 host 侧方法：`advance(receiver, ctx)` 在 evaluate 末尾推进整批状态，
`reset_env(receiver, ctx)` 在 reset 路径开头、sim reset terms 之前重采样（lane-gathered）状态。lane 在 `advance` 中置 `ctx.sim_reset_requested`（Manager 常驻
buffer，每次 physics step 前清零）即可请求 sim-only reset：该 lane 与当步的 episode reset
合并进**同一次** reset 运行（`reset_env` 重采样 → sim reset terms 重写仿真状态 → 按行重读 kernel inputs），但不会
触发 episode bookkeeping（`episode_steps`、truncation）与 command/action 的 host reset。WBT 在 motion clip wrap 时使用
该机制。host `reset(ResetContext)` 只在 episode reset 时准备跨 lane 的共享数据（如采样分布）；per-step 统计折叠
（如 WBT 的失败计数 EMA）由 term 自行内联进 `advance` 图内。

ActionTerm 是 host 侧描述对象：静态 action space、canonicalized state 和批量 process/reset callback。
只有 `term.state` 进入 kernel ABI，`ctx.actions[name]` 直接取得 lane-scoped state；host 通过
`env.action_terms[name].state` 访问同一份 backing。callback 不进入 kernel data，不要求 `@dispatch`。

Manager 在调用 `process(state, actions)` 前推进共享 cursor 一次并写入 raw policy actions；处理回调只计算
route-local controls，不重复推进 history，也不直接写 simulator state。每个 state 的 queue 至少保留两帧，
当前和上一帧分别为 `queue[ptr]` 与 `queue[(ptr - 1) % W]`，observation 和 action-rate reward 均读取 raw history。
延迟 action 可增加 queue 宽度。episode reset 时 Manager 先清空选中 lanes 的整个 queue，再调用
`reset(state, env_ids)`；共享 cursor 和其他 lanes 的 history 不变，sim-only reset 不清 action history。
Manager 根据 actuator route 合并 controls 后写入 backend。Term 不应调用其他 term 的行为方法，跨 term
依赖通过 `ManagerContext` 的数据 store 表达。

Observation terms use the host-side ``ObservationTerm(size, dispatch, *args)`` form. ``size`` is the fixed output width, and additional arguments are passed positionally; simple scalar/array arguments do not require a separate Args class. Reward terms use the analogous ``RewardTerm(dispatch, *args)`` form and return one float32 tensor per environment batch. Termination terms use ``TerminationTerm(dispatch, *args, metrics=...)``; metrics are optional per-environment outputs exposed to the dispatch through ``ctx.metrics``. The resolver validates these values into the static execution metadata, and the Torch executor calls ``dispatch(ctx, out, *args)`` for observations and ``dispatch(ctx, *args)`` for rewards and terminations.

## 5. 执行与性能边界

- 只有纯数组计算进入 term dispatch；不得传入 `SceneModel`、`SceneData`、`ArrayEnvState`、`EnvObs`、config 或 `info`；
- term 一次调用处理整批张量，写入预分配 output 和 term-owned buffers；
- Torch 执行器默认以 `torch.compile` 编译整段 term 调度（可回退 eager），不自动转换任意 Python/NumPy 环境方法；
- 单个环境实例不修改进程级 Torch 线程数，线程策略由训练 runtime 管理；
- 首次调用（惰性 specialization）、warm-up 和稳态 step latency 分开统计；绝对性能门槛属于 benchmark，不属于环境行为测试；
- CI 验证数值行为、reset/step 生命周期和 buffer identity。

## 6. 不变量

- Manager kernel 只为当前环境配置构建，runtime 不重建另一份 manager 配置；
- config 不持有 runtime state，concrete term 是其 persistent state 的唯一 owner；
- action policy layout 与 actuator route 分离，route 非空且不重叠；
- sim snapshot 只来自编译后的 read program；
- `ctx.env_id` 和所有 lane 形状 leaves 使用同一原始 environment id；
- Manager 不引入 generic `ManagerValue` registry 或独立 Hydra backend 维度。
