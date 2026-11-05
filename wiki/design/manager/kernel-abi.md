# Manager Kernel ABI

## 摘要

本文定义 Manager term 的数据 ABI：`@kernel_data` logical tree 如何解析为
path-keyed 的 resolved tree，`ManagerContext` 如何以整批张量
提供固定 namespace，以及 `PER_ENV`、`SHARED` 和 metric backing 的 ownership 语义。任务开发者只需遵守
[Manager Task API](./task-authoring.md) 中的 term 协议；数值执行器见 [Torch 执行器设计](./torch-executor.md)，
backend 读侧和 simulator 边界见 [SimBackend](./sim-backend.md)。

## 1. ManagerContext

`ManagerContext` 是 term dispatch 的唯一 manager/runtime 输入入口：

```python
@kernel_data
class ManagerContext:
    env_id: np.ndarray
    actions: Map
    commands: Map
    rand: RandValue
    sim: Map
    dt: np.float32
    sim_reset_requested: np.ndarray
```

Context 不包含 config、provider、backend handle、dynamic Python container 或 generic `states` / `resources` store。
Persistent runtime data 直接归属 concrete term；framework-owned random state 通过 `rand` 暴露，`dt` 提供当前
`ctrl_dt` 的固定 float32 标量。执行器把该结构解析为固定 layout（path → leaf 引用 + scope），运行时以 namespace 视图（`ctx.sim[key]`、
`ctx.commands[name].field` 等）暴露整批张量。

## 2. 批量语义

Host 侧 `env_id` 是：

```python
np.arange(num_envs, dtype=np.int64)
```

所有 `PER_ENV` leaves 是 `(N, ...)` 整批数组；term 一次调用处理整批。partial reset 采用静态全批 masked 设计：
reset 数学对全部 lane 计算，选中的 lane 由布尔掩码选择，`PER_ENV` leaves 按 clone-on-first-access 修改后以
`where(mask, clone, original)` 合并回常驻缓冲，未选中 lane 保持不变，不会把 reset batch row 当作 environment id。

所有 Map 的 key 顺序、concrete value schema、field tree、leaf dtype、scope 和 resolved path 在 resolve 前固定。Map lookup 使用 literal key：

```python
motion = ctx.commands["motion"]
dof_pos = ctx.sim["robot_dof_pos"]
```

动态字符串 key、运行期插入/删除和 fallback lookup 不属于 ABI；错误 key 在 resolve 或首次执行阶段失败。

## 3. Leaf scope

| 类型 | 默认 scope | term 访问语义 |
| --- | --- | --- |
| `np.ndarray` | `PER_ENV` | 整批 `(N, ...)` 数组/张量；reset 场景为 masked clone 行 |
| `SharedArray` | `SHARED` | 完整可写 ndarray，所有环境共享同一份 |
| Python scalar | `SHARED` | 所有环境使用同一个标量 |
| 固定 dtype NumPy scalar | `SHARED` | 所有环境使用同一个定 dtype 标量 |

`SharedArray` 只描述 scope，不代表 immutable。数组是否可以更新由 concrete term 的 owner/lifecycle 契约决定。
Term receiver 的参数通常按 shared lowering 处理，以免把静态参数误切片。

## 4. Logical tree 与 resolved tree

`@kernel_data` 支持有序、可逆的 record tree：

```python
leaves, tree_def = flatten_kernel_data(value)
rebuilt = unflatten_kernel_data(tree_def, leaves)
```

Record 可以嵌套另一个 KernelData record；不支持 multiple inheritance、覆盖 inherited field、普通 dataclass、递归结构、dict、list、字符串、object array 或 backend handle。

Logical tree 与 execution lowering 分离：同一 concrete value 在 host term 和 context 中保留相同的 tree schema，
但 resolver 根据使用位置决定 scope 并产出 path-keyed 的 resolved tree（leaf 引用 + scope），运行时据此构建
namespace 视图。只有 ndarray/scalar leaves 可以成为 resolved leaves；term 内不应产生稳态临时大数组分配。

## 5. Ownership 与 metric

- config 只创建 term，不持有 environment-local mutable arrays；
- 每个 concrete term 只 canonicalize 一次，host 和执行路径共享 backing identity；
- action/command term 拥有自己的 persistent buffers；
- observation output 由调用方提供，执行器在当前 invocation 写入；
- simulator snapshot 由 `PhysicsReadProgram` 拥有并写入，term 只读；
- `metric()` 字段由 owning term 写入自己的当前行，host collector 负责聚合 scalar metric。

Term 可以读取其他 term 的数据，但不应调用其他 term 的行为方法。Context store 表达数据依赖，不承担 manager scheduling。

## 6. Resolver 职责

`ManagerProgramResolver`（backend-neutral）负责：

- 校验 concrete runtime terms 和 `ManagerContext` 的 KernelData schema（leaf 类型与 scope）；
- 收集 read program 的 sim input map；
- 把 context 与 term 参数解析为 path-keyed resolved trees 和参数描述符（常量/叶子/树/sim key）；
- 校验方法签名、注解契约和 layout；
- 产出 `ManagerLayout`、`ReadPlan`、resolved trees 与 per-term-kind typed call records，供 Torch 执行器调度。

构造期 warm-up 以克隆输入执行一次 evaluate/observe，确认输入/输出契约（shape、dtype、batch 维、可写性）；
schema rename、term type 变化、scope 变化和 query layout 变化都会改变 resolved layout。

## 7. 核心不变量

- Context schema 在一个 environment 生命周期内固定；
- 每个 key 对应唯一 concrete type 和 stable resolved path；
- 所有 `PER_ENV` leaves 共享同一个第 0 维（整批为 N；reset 数学以 masked 全批形式运行）；
- `ctx.env_id` 在 step/reset 中与其他 per-env data 对齐；
- simulator snapshots 只来自编译后的 read program；
- 不建立 generic `states`、`resources` 或 `ManagerValue` registry；
- term dispatch 不接收完整 environment、config、info 或 simulator object；
- 数值行为、buffer identity 和 deterministic parity 属于 CI 验收，绝对性能和 scaling 属于独立 benchmark。
