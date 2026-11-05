# Term 自声明 Sim Query 设计（SimDataQuery 参数）

## 摘要

本设计定义 manager term 如何声明自己的 simulator 数据需求：reward、termination 和 observation term 把
`SimDataQuery` 对象直接作为 term 参数传入（`RewardTerm(dispatch, *args)` 中的 `args` 元素）。`ManagerEnv`
构造期扫描全部 term 参数，把每个 distinct query 登记为合成 key `terms.args[i]`，与任务侧
`cfg.queries.data` 声明统一合并：同一 key 下不相等的声明构造期报错，相等声明折叠为一次物理读。resolver
（`ManagerProgramResolver`）随后把 query 参数映射为 read program 的批量 sim 视图，Torch 执行器在参数原位置
把整批张量传给 dispatch。本文覆盖声明机制、合并规则与执行映射；任务侧集中声明方式见
[Manager Task API](./task-authoring.md)，读侧编译契约见 [读侧编译](./sim-read-program.md)。

## 1. 动机：term 声明 vs 任务声明

任务通过 `SimQueriesCfg` 字段名集中声明 data query（key 即字段名，如
`queries.data["robot_dof_pos"]`）。集中声明适合有自身 authority 的任务（如 WBT 的 motion joint 顺序），
但对通用 term 有两个问题：

1. **契约隐式**。term 的 dispatch 需要哪些 sim 输入，任务作者只能阅读 term 源码发现；唯一保障是运行期
   key 访问失败。
2. **声明权与消费权分离**。通用 term（框架 MDP 库中的 robot observation/reward 等）需要的 query 由任务
   代为声明，框架每新增一个 term 都可能要求所有使用它的任务补声明。

机制目标是让 term 自己携带输入声明：把 `SimDataQuery` 作为 term 参数，声明与消费落在同一个对象上；
同时保留任务侧集中声明，二者在构造期合并，且保持"一个 key 只对应一个 logical query"的单一属主规则。

## 2. 声明机制：SimDataQuery 作为 term 参数

reward、termination 和 observation term 的 `args` 元组中可以直接包含 `SimDataQuery` 实例：

```python
joint_pos_query = JointPositionQuery(joints=resolve_robot_joint_names)

RewardTerm(
    dispatch=robot_joint_pos_reward,
    args=(joint_pos_query, -0.5),  # dispatch(ctx, joint_pos, penalty_scale)
)
```

`ManagerEnv` 构造时（`motrix_env_core.manager.env`）在 read program 编译之前扫描 reward、termination 和
全部 observation term 的 `term.args`：

- 每个 distinct query（按 dataclass 相等去重）登记进 `_term_query_keys`，合成 key 为
  `terms.args[i]`（`i` 按收集顺序递增）；多个 term 传入相等 query 时共享同一 key。
- 合成 key 与任务声明一起进入 `_merge_query_declarations` 合并，随后一次性交给
  `SimBackend.compile_reads` 编译为 `PhysicsReadProgram`。

## 3. 合并规则：key 单一属主，不等即报错

`_merge_query_declarations` 对任务声明与 term 贡献统一处理：

| 情形 | 结果 |
| --- | --- |
| 仅任务声明该 key | 使用任务声明，语义权威保持在任务侧 |
| 仅 term 贡献该 key | term 的 query 进入 read program |
| 多个来源对同一 key 声明相等 query | 折叠为一个 logical query、一次物理读（read program 的既有 exact-equal 去重不变量，见 [读侧编译](./sim-read-program.md)） |
| 同一 key 下声明不相等 | 构造期 `ValueError`，报错含 key、两侧 query repr 与来源 |

term-arg 的合成 key（`terms.args[i]`）与任务声明的字段名 key 处于同一命名空间：任务对同一字符串 key
的声明与 term 贡献不等时同样构造期失败。"rename key 是 compiled interface 变更"的既有契约不变。

## 4. 执行映射：resolver 到批量 sim 视图

`ManagerProgramResolver` 解析 term 参数时（`_resolve_term_argument` 路径）识别 `SimDataQuery` 值：

- 以 query 反查 `_term_query_keys` 得到 key，再取其在 read program 中的 `ManagerSimInput`（logical
  view + query 绑定）；
- dispatch 调用时，该参数位置收到的是 context 的**整批** sim 视图张量（`(N, ...)`），与
  `ctx.sim[key]` 的批量语义一致（见 [Torch 执行器设计](./torch-executor.md)）；
- view 的 shape/stride 进入 resolver 的 layout fingerprint，保证输入布局变化触发重新解析。

term 因此不需要在 dispatch 内部用字符串 key 访问 `ctx.sim`：query 声明即参数，参数即批量视图。

## 5. 边界与已知限制

- 收集范围是 reward、termination 和 observation term 的 `args`；command/action term 的 kernel_data
  走各自的 canonicalize 路径，不参与本机制。
- 合成 key `terms.args[i]` 依赖收集顺序（配置映射顺序），属于内部执行元数据，不是面向任务的稳定命名
  契约；任务需要稳定 key 时应在 `queries.data` 中显式声明。
- 不支持单环境多机器人的 per-entity key 命名空间；canonical key 是单数角色（"the robot"）。
- model query 是静态数据，不经 read program：term 在 `__call__(env)` 时从 `env.model` 读出并烘焙进
  term 的 `@kernel_data` params，与本机制无关。

## 验收标准

- 传入 `SimDataQuery` 参数的 term 无需任务在 `queries.data` 中重复声明，环境可构造并训练；dispatch 在
  参数位置收到正确 shape 的批量张量。
- 相等 query（无论来自任务声明还是多个 term 参数）在 read program 中折叠为一次物理读，view 为同一对象。
- 同一 key 下不等声明在构造期报错，报错含 key 与两侧来源。
- 使用任务侧集中声明的现有任务（如 WBT）零迁移。
