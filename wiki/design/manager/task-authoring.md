# Manager Task API

## 摘要

本文面向任务开发者，说明如何使用 `ManagerBasedEnvCfg` 和 Manager term 创建一个 config-driven `ManagerEnv`。
内容覆盖任务文件组织、manager groups、term 创建、`required_queries()`、`@kernel_data` 字段和常用 term 接口。
运行时边界见 [Manager Runtime](./runtime.md)，kernel ABI 见 [Manager Kernel ABI](./kernel-abi.md)。

## 1. 任务文件组织

一个简单任务通常拆成：

```text
my_task/
├── __init__.py   # 注册配置和环境
├── cfg.py        # 组合 manager 配置组
└── mdp.py        # 定义 term cfg 和 concrete terms
```

配置负责声明使用哪些 term 及其参数；`mdp.py` 负责实现 term 行为。配置对象不保存每个 environment 的运行期数组。

## 2. 顶层配置

Manager 配置继承 `ManagerBasedEnvCfg`，不需要再次继承 `EnvCfg`：

```python
from motrix_env_core import registry
from motrix_env_core.config import configclass
from motrix_env_core.manager import ManagerBasedEnvCfg


@registry.envcfg("my-task")
@configclass
class MyTaskEnvCfg(ManagerBasedEnvCfg):
    actions: dict = {}
    commands: dict = {}
    observations: dict = {}
    rewards: dict = {}
    terminations: dict = {}
```

Manager group 可以使用普通 `dict[str, TermCfg]`，也可以使用对应的 typed group。归一化会保留声明顺序；该顺序决定 action
切片、observation layout 和同组 term 的执行顺序。

各组职责如下：

| 配置组 | 职责 |
| --- | --- |
| `actions` | 定义 action space、处理 action、维护 action history |
| `commands` | 维护目标、轨迹或采样状态 |
| `queries.data` | 声明 simulator snapshot |
| `queries.model` | 声明构造期模型 metadata |
| `observations` | 生成 policy/value observation |
| `rewards` | 计算 reward term |
| `terminations` | 计算 termination term |
| `sim_reset` | 声明 reset 行为 |

例如：

```python
from motrix_env_core.manager import ActionCfg, CommandCfg, ManagerBasedEnvCfg


@configclass
class MyTaskEnvCfg(ManagerBasedEnvCfg):
    commands: dict[str, CommandCfg] = {}
    actions: dict[str, ActionCfg] = {
        "arm": JointPositionActionCfg(
            actuator_names=("shoulder", "elbow"),
            default_positions=(0.1, -0.2),
            action_scale=0.25,
        ),
    }
```

## 3. Term 创建与字段

各 term cfg 通过 `__call__()` 创建 concrete runtime term。Concrete term 使用 `@kernel_data` 声明固定字段：

```python
@kernel_data
class TrackingErrorTermination:
    threshold: np.float32
    error: np.ndarray = metric()

    def compute(self, ctx: ManagerContext) -> bool:
        value = ...
        self.error[0] = value
        return value > self.threshold
```

`@kernel_data` 的字段必须有类型标注，结构在构造期固定。Term record 不可替换字段，但 owner 可以按契约原地更新 ndarray backing。
支持的 leaf 类型为：

- 普通 `np.ndarray`：默认按 environment 切片，第一维为 `num_envs`；
- `SharedArray`：所有 lane 访问完整数组，适合模型参数、查表数据和 motion clip；
- Python 标量和固定 dtype NumPy scalar；
- 另一个 `@kernel_data` 类型。

需要对外发布逐环境指标时使用 `metric()` 字段。框架为其分配 `(num_envs, 1)` backing，term 在当前 lane 写入 `[0]`。

## 4. Simulator query 声明

任务可以在 `cfg.queries.data` 和 `cfg.queries.model` 中声明 query。Observation term 也可以通过 `required_queries()` 声明自己需要的 query：

```python
@configclass(kw_only=True)
class RobotPositionObsCfg(ObservationTermCfg):
    def required_queries(self) -> SimQueriesCfg:
        return SimQueriesCfg(
            data={"robot_pos": LinkPositionQuery(link=lambda cfg: cfg.scene.objs.robot.resolved_base_link_name)}
        )
```

合并规则：

1. 任务显式声明拥有该 key；
2. 任务未声明时，由 term 提供默认 query；
3. 多个 term 对同一 key 提供不相等 query 时，构造期报错；
4. 相等 query 由 backend read program 折叠为一次 physical read，并保留各 key 的 logical view。

Query key 是 compiled interface；重命名 key 需要同步修改所有 `ctx.sim["key"]` 消费者。
Term 不持有 backend handle，也不在运行期创建或执行 simulator query。

## 5. Action

`ActionCfg.actuator_names` 声明 simulator control route，空 tuple 表示所有 actuator。Manager 在构造期解析并校验未知、重复和重叠 route。

ActionCfg 在构造期返回冻结的 host 描述对象：

```python
ActionTerm(
    action_space=space,
    state=state,  # @kernel_data BaseActionState 子类
    process=process_action,  # process_action(state, actions_batch) -> controls | None
    reset=reset_action,  # reset_action(state, env_ids) -> None
)
```

`BaseActionState` 声明 `action_queue` 和 `action_ptr`，配置为每个环境分配独立 history。
Manager 在 process 回调之前推进共享 cursor 一次并写入 raw action slice；回调只处理该 slice，
返回 route-local controls，不重复维护 history。episode reset 时 Manager 先清目标 lanes 的整个 queue，
再调用 reset 回调，cursor 保持不变。Manager 将各 term 输出合并并写入 simulator。

Host 通过 `env.action_terms[name].state` 访问状态；compiled term 通过 `ctx.actions[name]` 直接读取
lane-scoped state，不能调用 process/reset callback。具体状态可增加 delay、目标缩放和模型参数字段。

## 6. Observation、Reward、Termination

Observation term 将结果写入 caller-owned `out`，必须返回 `None`：

```python
def compute(self, ctx: ManagerContext, out: np.ndarray) -> None: ...
```

Reward term 返回当前 environment 的 scalar，Manager 最终按配置 weight 和 `ctrl_dt` 汇总；加权结果发布到 `state.info["Reward"]`。

Termination term 返回单个 environment 的 `bool`：

```python
@kernel_data
class OutOfBoundsTermination:
    limit: np.float32

    def compute(self, ctx: ManagerContext) -> bool:
        return abs(ctx.sim["dof_pos"][0]) > self.limit
```

Observation、reward、termination term 不应通过共享全局 registry 查找其他 term；需要的数据依赖应通过 `ManagerContext` 的固定 store 表达。

## 7. Reset

Reset 配置声明 simulator state 和 term state 的恢复行为。Term 的 `reset(env_ids)` 只修改指定 environment rows；不要在 term 中实现全局 auto-reset、episode step 或 truncation。
这些职责属于 [Manager Runtime](./runtime.md) 所述的 `ArrayEnv` lifecycle 和 backend reset program。

## 8. 开发验收

新增 Manager task 至少应验证：

- config 能通过 `ManagerBasedEnvCfg` 的 group 和 query 校验；
- action route、observation layout、reward 和 termination 行为符合配置顺序；
- full reset 与 partial reset 只修改目标 rows，buffer identity 保持稳定；
- query key 和 term method 返回契约在 warm-up 阶段能被发现；
- 不支持的 Python object、动态字典 key 或 simulator handle 不进入 kernel ABI；
- 与 reference NumPy 实现进行确定性输入下的数值 parity 检查。
