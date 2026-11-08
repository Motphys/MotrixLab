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

执行方式默认为 `torch.compile`：每个 kernel（evaluate/observe/reset）把整段
term 调度编译为 Inductor 图（observe 按组拆图；进程 OMP 池默认封顶 32 线程，
图内线程数与 chunk 由 `MOTRIX_TORCH_COMPILE_THREADS`/`MOTRIX_TORCH_MIN_CHUNK` 控制；
reset 以静态
dynamic gather-compact 形式编译，见下文 reset 路径）。RNG draw 顺序与 eager
  在 evaluate/observe 内完全一致；Inductor 重结合仅带来
≤1e-8 量级的 reward 差异。调试可用 `TorchManagerRuntime(env, program,
compile=False)` 或 `runtime.compile_enabled = False` 回到逐 term eager 执行。

## 解析与执行的分工

- `ManagerProgramResolver`（backend-neutral）负责输入准备：组装唯一常驻的
  `ManagerContext` 宿主实例（Map 容纳真实的 command/action term 实例，`sim`/
  `metrics` 为零拷贝 tensor 视图，`rand` 为 tensor 批量 `RandValue`，`dt` 为
  Python float），并以 `iter_leaves` 做一次性 leaf 类型校验；carrier dataclass
  term 参数按常量参数原样透传，Dynamo 直接 trace 其属性访问。构建
  `ManagerLayout`、`ReadPlan`、per-term-kind typed call records（command hooks
  携带 command key 与 hook 函数，reward/termination 携带 buffer column
  index 与参数描述符等），并校验签名注解契约。它不做任何代码生成、缓存或
  dispatcher 编译。
- `TorchManagerRuntime` 消费这些元数据并执行 term 调度；所有写入落回
  host 持有的 numpy 缓冲（reward terms/masks/obs/command state/metrics），
  `ManagerEnv` 的公共 API 与状态面保持不变。

## 执行器行为

- evaluate 按固定语句顺序调度：command `update` → rewards
  （raw、`weighted = raw * weight * dt`、Σ）→ terminations（masks、OR）→
  command `advance`（rewards/terminations 读 pre-advance command state）；
  observe 按 group/term 顺序写 obs 切片；reset = command `reset`
  （compact）→ sim-reset terms 写 compact 行。
- RNG：per-lane SplitMix64（`manager.rand.draw_uniform`），批量语义为
  term 一次 draw `count=k` 等价于 per-lane 连续调用 k 次。
- 常驻 context：三个 kernel 共用同一 `ManagerContext` 实例；simulator-backed
  tensor 视图由 read program 原地重写 arena 保持有效，term 实例与标量字段即
  宿主对象本身（graph-break-free 编译的关键：无转译层，guard 落在对象
  identity 上）。
- reset 路径是 **dynamic gather-compact** 设计：`ctx.reset_ids` 携带本 pass 的
  K 个 reset 行（动态长度，`mark_dynamic` 防 0/1 特化，一次 `dynamic=True`
  编译覆盖所有 K）。command `reset` 钩子直接在常驻实例上按 ids
  gather/scatter；sim-reset terms 直接填 `(K, *write_shape)` staging（无
  `stage[ids]` 二次 gather）；wrapper 把 staging 拷到全批 reset buffer 头部。
  RNG 语义：reset pass 在全批流上 draw 后 gather 对应行，pass 之外的 lane
  的 SplitMix64 序列同样推进。实测 reset kernel 随 K 伸缩（2048 envs）：
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
- `ctx.rand.next_uniform(count=k)` → `(N, k)`，`[-1,1)`，原地推进状态；
  `ctx.rand.uniform_range(count, low, high)` 同理
- `ctx.dt` float、`ctx.env_id` `(N,)`、`ctx.sim_reset_requested` `(N,1)` bool

函数签名约定（第一个参数注解为 `ManagerContext`，observation 的 `out`
注解为 `torch.Tensor`（`(N, size)` 切片），reset 的 `sim_writes` 注解为
`Map[torch.Tensor]`——resolver 校验这些契约注解；标量参数到达时为 Python
float，数组参数为批量张量）：

- reward：返回 `(N,)` float32 张量
- termination：返回 `(N,)` bool 张量；metric 写 `ctx.metrics[name][:, 0] = value`
- observation：`out` 为 `(N, size)` 张量切片，直接写入；noise 用
  `ctx.rand.next_uniform(size) * amplitude`（count == out.shape[1] 保持 RNG 顺序）
- reset：`(ctx, sim_writes, *args)`，ctx 第 0 维 = 本 pass 的 K（`ctx.reset_ids`），
  `sim_writes[name]` 为 compact `(K, *write_shape)` 张量

Command hooks（`update`/`advance`/`reset`）是 CommandTerm 类上的实例方法，
签名 `(self, ctx)`（resolver 按方法名解析，无标记装饰器）；`self` 就是宿主 `CommandTerm` 实例
（三个 kernel 都在常驻实例上执行，编译图对其 identity 建 guard——reset 为
gather-compact，直接按 `ctx.reset_ids` 在常驻实例上 gather/scatter）。
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
- **已知残余**。EPYC 上相对 Numba 约 1ms 残余 = 冷缓存惩罚 + reset 静态全批布尔掩码设计
  （RNG 状态 `where(mask, ...)` 门控）带来的约 3x lane 计算结构性成本；本机持平 16.8–17.0ms。
  拉平高线程数仍需跨图 region 融合或持久 lane kernel（future work）。
