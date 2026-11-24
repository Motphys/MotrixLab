# Torch Manager 执行器设计

## 摘要

本文定义 `ManagerEnv` 的数值执行器：通用 Torch 批量执行器
（`motrix_env_core.torch_manager.runtime.TorchManagerRuntime`）。resolver
（`motrix_env_core.manager.resolver.ManagerProgramResolver`）把 manager 配置
解析为 read plan / layout / ManagerContext / typed term calls 等执行元数据，Torch 执行器
以整批张量按固定语句顺序调度 term；host 生命周期与 term 配置 API 不变。
Torch 是唯一执行器——历史上并存过的 Numba fused-kernel 路径与
`executor="numba"|"torch"` 双执行器切换已经移除。相关背景见
[Manager Runtime 设计](./runtime.md)。

执行方式默认为 `torch.compile(fullgraph=False)`：每个 kernel（evaluate/observe/reset）
编译 term 调度的数值计算；`ctx.rand` 直接调用可 trace 的 Torch 默认随机算子，
不人为建立抽样 eager 边界（observe 按组拆图；进程 OMP 池与图内线程数见下文，
图内线程数与 chunk 由 `MOTRIX_TORCH_COMPILE_THREADS`/`MOTRIX_TORCH_MIN_CHUNK` 控制；
reset 以 dynamic gather-compact 形式编译，见下文 reset 路径）。随机序列的可复现性需在
相同执行模式、Torch 版本、设备和随机调用顺序等前提下验证；不承诺 eager 与 compiled
产生相同随机序列，也不保证跨模式数值差异的固定上限。调试可用 `TorchManagerRuntime(env, program,
compile=False)` 或 `runtime.compile_enabled = False` 回到逐 term eager 执行。

## 解析与执行的分工

- `ManagerProgramResolver`（backend-neutral）负责输入准备：组装常驻的基础
  `ManagerContext` 宿主实例（`actions: Mapping[str, ActionState]`、
  `commands: Mapping[str, CommandTerm]` 容纳真实的状态/term 实例，`sim`/
  `metrics: Mapping[str, torch.Tensor]` 为零拷贝 tensor 视图；这些映射运行时均为普通
  `dict`，构造后 key 固定，数值调用只读取条目或原地更新 tensor，不修改映射结构；`dt` 为
  Python float），并以 `iter_leaves` 做一次性 leaf 类型校验；carrier dataclass
  term 参数按常量参数原样透传，Dynamo 直接 trace 其属性访问。构建
  `ManagerLayout`、`ReadPlan`、per-term-kind typed call records（command hooks
  携带 command key 与 hook 函数，reward/termination 携带 buffer column
  index 与参数描述符等），并校验签名注解契约。它不做任何代码生成、缓存或
  dispatcher 编译。
- `TorchManagerRuntime` 消费这些元数据并执行 term 调度；所有写入落回
  host 持有的 Torch 张量缓冲（reward terms/masks/obs/command state/metrics），
  `ManagerEnv` 的公共 API 与状态面保持不变。

## 执行器行为

- evaluate 按固定语句顺序调度：command `update` → rewards
  （raw、`weighted = raw * weight * dt`、Σ）→ terminations（masks、OR）→
  command `advance`（rewards/terminations 读 pre-advance command state）；
  observe 按 group/term 顺序写 obs 切片；reset = command `reset`
  （compact）→ sim-reset terms 写 compact 行。
- RNG：term 使用 `ctx.rand.uniform(shape)` / `ctx.rand.uniform_like(tensor)`。
  `RandomSource()` 无构造参数、不持有 generator，分别直接封装 `torch.rand` 与
  `torch.rand_like`，消费相应设备的默认 Torch 随机流；不提供实例或用途隔离。
  训练入口负责根 seed，ManagerEnv 构造不调用 `torch.manual_seed`。
- 随机算子编译：没有 `@torch.compiler.disable` 边界。Torch 2.7 数值实验已验证
  该入口可由 `fullgraph=True` trace，但不据此承诺整个 ManagerEnv fullgraph 编译成功，
  也不把微基准当作整环境吞吐或 CUDA graph capture 兼容性的证明。
- 常驻 context：evaluate/observe/reset 共用 `ManagerContext`，observe 不替换随机流。
  simulator-backed tensor 视图由 read program 原地重写 arena 保持有效，term 实例
  与标量字段即宿主对象本身（无转译层，guard 落在对象 identity 上）。终态 preview
  消费共享默认流，可改变后续正常观测噪声及 command/reset 抽样。
  seed 与可复现性边界见[训练随机性与 Seed 所有权](../randomness-and-seeding.md)。
- reset 路径是 **dynamic gather-compact** 设计：显式 `env_ids` 参数携带本 pass 的
  K 个原始环境行号，command reset 另接收对齐的 bool `(K,)` `episode_reset`，区分 episode 与 sim-only reset；
  context/query 输入始终为全批 N 行（通过 `mark_dynamic` 与 `dynamic=True` 请求 reset 符号批宽，
  Torch 的零/单行处理仍可能触发特化，不保证所有 K 共享一次编译）。command `reset` 钩子直接在常驻实例上按 ids
  gather/scatter；sim-reset terms 直接填 `(K, *write_shape)` staging（无
  `stage[ids]` 二次 gather）。term 直接写 backend write buffer 的前 K 行视图，不另外分配
  staging storage，也不复制一次 compact 前缀。`SimResetRuntime.apply` 仍先克隆有效前缀，
  再 scatter 到原始 `env_ids`，避免乱序行号的源/目标重叠。write buffer 是串行 reset 调用的
  可覆盖 scratch；term 必须完整写入声明输出，异常时不执行 backend，但不提供状态事务回滚。
  reset term 可直接为本 pass 的 compact 行抽样；随机流会推进，但非选中行的
  环境状态保持不变。以下 reset kernel 性能数据为既有测量（2048 envs）：
  K=1 → 0.089ms、K=16 → 0.097ms、K=2048 → 0.275ms（旧的静态全批 masked
  kernel 不随 K 变化，恒 ~0.40ms）。

## 批量 term 约定

ctx（`ManagerContext` 宿主实例）字段全部是**整批**张量：

- `ctx.sim[key]` → `(N, ...)` 张量（SimDataQuery 参数同样是整批张量）
- `ctx.commands[name].field` → per-env 字段 `(N, ...)`；嵌套 kernel_data（如
  `motion.clip.joint_pos`）为 shared `(F, J)`，term 自行按 `steps[:, 0]` gather
- `ctx.actions[name]` 是真实 `ActionState`；`.current()/previous()` 为纯 tensor
  gather 方法，返回 `(N, A)`；`action_queue/action_ptr` 同理
- `ctx.metrics[name]` → `(N, 1)` 可写张量
- `ctx.dt` float、`ctx.env_id` `(N,)`、`ctx.sim_reset_requested` `(N,1)` bool

函数签名约定（第一个参数注解为 `ManagerContext`，observation 的 `out`
注解为 `torch.Tensor`（`(N, size)` 切片），reset 的 `sim_writes` 注解为
`Mapping[str, torch.Tensor]`——resolver 校验这些契约注解；标量参数到达时为 Python
float，数组参数为批量张量）：

- reward：返回 `(N,)` float32 张量
- termination：返回 `(N,)` bool 张量；metric 写 `ctx.metrics[name][:, 0] = value`
- observation：`out` 为 `(N, size)` 张量切片，直接写入；noise 用
  `(2 * ctx.rand.uniform_like(out) - 1) * amplitude`
- reset：`(ctx: ManagerContext, env_ids: torch.Tensor, sim_writes: Mapping[str, torch.Tensor], *static)`；
  ctx 与 query 参数保持全批 N 行，`env_ids` 为 K 个原始行号，读取时显式 gather；
  `sim_writes[name]` 为 compact `(K, *write_shape)` 张量，第 j 行对应 `env_ids[j]`，不得二次按原始行号 gather；
  `sim_writes` 也是 key 固定的普通 `dict`，dispatch 原地写入 tensor，不修改映射结构

Command hooks（`update`/`advance`/`reset`）是 CommandTerm 类上的实例方法，
`update`/`advance` 签名为 `(self, ctx: ManagerContext)`，reset 为
`(self, ctx: ManagerContext, env_ids: torch.Tensor, episode_reset: torch.Tensor)`
（resolver 按方法名解析，无标记装饰器）；`self` 就是宿主 `CommandTerm` 实例。
三个 kernel 都在常驻实例和全批 context 上执行，编译图对其 identity 建 guard；reset
直接按 `env_ids` gather/scatter，用与 `env_ids` 对齐的 bool `(K,)` `episode_reset` 限定回合级行为：
True 表示回合级重置（含初始化与显式重开），False 表示仅 sim reset、当前回合继续；
它不等于 `terminated`，也不必然表示上一回合已经结束。
`ManagerContext` 不保存 reset 选择字段。
ActionTerm 的 `process/_reset` 保持宿主路径不变。

标量参数（float）以 Python float 传入；字符串参数直接传入。

## 参考 math 模块

- `motrix_envs/.../wbt/mdp/batch_math.py`（张量四元数）、
  `batch_rewards/batch_terminations/batch_observations`（WBT 项的批量数学）
  与 `walk_manager_mdp/terrain.heightfield_lookup_batch`（批量地形插值）。

## 性能特性

以下为 G1 locomotion（2048 envs、zero actions、100 steps）迁移到单 Torch 执行器时的基准记录，
用于解释当前实现中的若干设计选择：

- **机器对比**。EPYC（192 logical CPUs，双方 16 线程，PASSIVE）：Numba 基线 median 12.99ms
  （157,641 steps/s），Torch 迁移后 median 14.0–14.5ms；breakdown 差距集中在 observe
  （3.31 → 1.85ms）与 reset kernel（0.80 → 0.72ms，Numba 0.19ms），evaluate 反超
  （1.94 → 1.66 vs Numba 1.78）。本机（32 核桌面）已打平：16.99 vs 16.98ms
  （evaluate 0.83 反超 Numba 0.92）。
- **`min_chunk_size` 发现**。图本身已是单融合 kernel；真正瓶颈是 Inductor `cpp.min_chunk_size`
  默认 4096——manager 批量循环（2048×3…2048×84）全部低于阈值，被生成为 `#pragma omp single`
  单线程段（observe 图约 19 段）。修复：`min_chunk_size` 默认 64（env `MOTRIX_TORCH_MIN_CHUNK`）、
  observe 改 per-term 连续缓冲 + 单 kernel 组装、输出张量改为图参数使双缓冲槽
  共享编译图、observe 按 group 拆分独立编译图。
- **进程 OMP 池与图内线程**。192 核 EPYC 上不限制进程 OMP 池时，192 线程的 fork/join 开销
  使 median 劣化到 ~34ms；进程池默认按 batch 封顶
  `min(16, max(8, num_envs // 256))` 线程（`OMP_NUM_THREADS` 显式设置时尊重调用方）。
  池宽扫描（gather-compact 后）：2048 envs 时 8 最优（11.3ms；16/32/192 依次
  11.7/12.9/33.8），4096/8192 时 16 最优（16.9/27.3ms；8 线程分别慢 1%/4%）——
  更大的 batch 摊薄宽池的调度开销。曲线浅，启发式在 2048–8192 全程距最优固定值
  ≤4%。`MIN_CHUNK=64` 在 2048–8192 全程最优。图内编译线程默认跟随进程池
  （env `MOTRIX_TORCH_COMPILE_THREADS` 覆盖）。numactl 在单 NUMA node 上无效。
- **冷惩罚测量**。物理后首个编译 kernel 存在冷缓存惩罚：同段计算重复执行 0.29ms vs 物理后执行
  0.59ms（Numba 因单小并行区天然免疫）。
- **测量适用范围**。以上基准是单 Torch 执行器迁移阶段的记录；`ctx.rand` 的数值
  trace 实验也不是整环境吞吐或 CUDA graph capture 验收，相关结论须单独测量。

## Transition reward and nested metrics

The evaluate kernel publishes the weighted transition reward to `ctx.reward` before command
`advance` hooks. It aliases `state.reward`. Nested kernel-data metrics are initialized to zero
and collected recursively so motion curricula can own their statistics.
