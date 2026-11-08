# PhysicsReadProgram 读侧编译设计

## 摘要

读侧只有一个编译入口和一个运行时产物：`SimBackend.compile_reads(queries) -> PhysicsReadProgram`。
框架负责 query 的配置级解析和 key 校验；backend 负责 simulator handle 解析、physical memory planning、相等 query 的去重、
logical view 和 full/partial execute。

Manager task 的 query 来源（任务 `queries.data` 与 term 的 `SimDataQuery` 参数）见
[Manager Task API](./task-authoring.md)；整体 simulator 边界见
[Manager SimBackend](./sim-backend.md)。

## 1. 数据流与责任

```text
cfg.queries.data + term args 中的 SimDataQuery
  -> ManagerEnv 构造期合并（_merge_query_declarations）
  -> SimBackend.compile_reads
  -> PhysicsReadProgram
  -> ManagerContext.sim / term 参数视图
```

| 组件 | 职责 |
| --- | --- |
| `SimDataQuery` | 描述 backend-neutral 的固定 simulator quantity |
| `resolve_sim_queries` | 校验 key、解析 `lambda cfg:` 等 config-level resolver |
| `SimBackend.compile_reads` | 解析名称、规划 physical storage、去重并创建 program |
| `PhysicsReadProgram` | 拥有 authoritative storage，刷新 full/partial rows，提供 stable views |
| `ManagerContext.sim` | 为 compiled term 提供固定 literal-key 的 lane-local 输入 |

Term 不持有 backend handle，也不在 step 中创建或执行 query。

## 2. PhysicsReadProgram 服务面

```python
class PhysicsReadProgram(abc.ABC):
    arena_bytes: int
    keys: tuple[str, ...]

    def query(self, key: str) -> SimDataQuery: ...
    def view(self, key: str) -> np.ndarray: ...
    def execute(self, env_ids: np.ndarray | None = None) -> None: ...

    @property
    def inputs(self) -> Mapping[str, np.ndarray]: ...
```

不变量：

- `keys` 包含所有声明 key，包括共享 physical storage 的 alias；
- `view(key)` 返回稳定的 logical view，首维是 `num_envs`，dtype 为 `float32`；
- exact-equal queries 可共享同一个 physical region 和 view object；
- authoritative arena 为 program 私有，`execute` 是唯一写者；
- partial execute 只刷新传入的 environment rows；
- trailing shape 和 strides 直接由 logical view 暴露，不维护第二份 layout 描述。

## 3. Query 声明

```python
@configclass
class MySimQueriesCfg(SimDataQueriesCfg):
    robot_dof_pos: BodyJointPositionQuery = ...
    tracked_body_pos: BatchLinkPositionQuery = ...
```

字段名就是 compiled query key。rename key 是 compiled interface 变化，必须同步修改 `ctx.sim["key"]` 消费者。
一个 key 只能有一个 logical query；完全相等的不同 key 可以由 backend 折叠为一次 physical read。

Query 支持 config-level resolver，例如根据 `scene.objs.robot` 解析最终 link/joint names。resolver 只在 composition/compile 阶段执行，
program 和 kernel 不保留 callable。

## 4. Term required queries

声明来源有两处：任务 `queries.data` / `queries.model`，以及 reward/termination/observation term 参数中的
`SimDataQuery` 对象。Manager 在 backend 编译前合并它们：

- 任务显式声明的 key 由任务拥有；
- term 参数中的 query 按相等去重，登记为合成 key；
- 同一 key 下不相等声明构造期报错；
- 相等声明折叠为一条声明、一次 physical read；
- data query 进入 read program；model query 在 term construction 时解析为静态参数。

具体协议和限制见 [Term 自声明 Sim Query](./term-required-queries.md)。

## 5. Backend 实现步骤

1. 在 `SimBackend.compile_reads` 接收完整声明集；
2. 使用 backend-specific typed resolver 将 query 转为 simulator indices/handles 和 logical shape；
3. 对 exact-equal declarations 建立 canonical physical operation 和 key aliases；
4. 分配 program-owned arena，建立每个 key 的 view、query metadata 和 stride 信息；
5. 实现 full execute 与按 `env_ids` 的 partial execute；
6. 添加 unknown name、unsupported query、shape/layout、view identity 和 full/partial read 测试。

Resolver 只能返回后续 physical planning 所需的 backend-private resolved query，不得把 simulator object 暴露给 frontend 或 kernel。

## 6. 诊断与验收

构造/编译期应明确报告：非法 key、bare query、unknown body/link/sensor/geom、unsupported query、dtype/shape/layout 不匹配以及
`ctx.sim` literal key 不存在。测试至少覆盖：

- query key 和 config resolver 校验；
- exact-equal query 的 physical deduplication；
- stable logical view identity、shape、stride 和 dtype；
- full execute 更新全部 rows；
- partial execute 只更新指定 rows；
- compiled term 通过 `ctx.sim[key]` 读取 snapshot；
- import core/manager 不加载具体 simulator。
