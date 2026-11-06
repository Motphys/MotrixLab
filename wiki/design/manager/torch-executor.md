# Torch Manager 执行器设计

## 摘要

本文定义 `ManagerEnv` 的数值执行器：通用 Torch 批量执行器
（`motrix_env_core.torch_manager.runtime.TorchManagerRuntime`）。resolver
（`motrix_env_core.manager.resolver.ManagerProgramResolver`）把 manager 配置
解析为 read plan / layout / typed term calls 等执行元数据，Torch 执行器
以整批张量按固定语句顺序调度 term；host 生命周期与 term 配置 API 不变。
Torch 是唯一执行器——历史上并存过的 Numba fused-kernel 路径与
`executor="numba"|"torch"` 双执行器切换已经移除。相关背景见
[Manager Runtime 设计](./runtime.md)。

执行方式默认为 `torch.compile`：每个 kernel（evaluate/observe/reset）把整段
term 调度编译为 Inductor 图（observe 限制 `cpp.threads=min(4, nthreads)`，
可用 `MOTRIX_TORCH_COMPILE_THREADS` 覆盖；reset 以 env_ids 为动态维度参数），
context 命名空间、invocation 参数与张量视图按 inputs 身份一次性缓存。
RNG draw 顺序与 eager 完全一致；Inductor 重结合仅带来 ≤1e-8 量级的 reward
差异。调试可用 `TorchManagerRuntime(env, program, compile=False)` 或
`runtime.compile_enabled = False` 回到逐 term eager 执行。

## 解析与执行的分工

- `ManagerProgramResolver`（backend-neutral）负责输入准备：`ManagerContext`
  解析为 path-keyed 的 resolved tree（叶子持有 live value 引用；数组字段已是
  torch tensor，执行器无需再包装）；carrier dataclass term 参数按常量参数
  原样透传，Dynamo 直接 trace 其属性访问，其结构化 fingerprint（字段名/类型/
  张量 dtype+shape+内容哈希/标量值）计入 program fingerprint 以保持缓存失效
  强度。构建 `ManagerLayout`、
  `ReadPlan`、per-term-kind typed call records（command hooks 携带 receiver 与
  static function，reward/termination 携带 buffer column index 与 resolved
  argument 描述符等），并校验签名注解契约与签名契约。它不做任何代码生成、
  缓存或 dispatcher 编译。
- `TorchManagerRuntime` 消费这些元数据并执行 term 调度；所有写入落回
  host 持有的 numpy 缓冲（reward terms/masks/obs/command state/metrics），
  `ManagerEnv` 的公共 API 与状态面保持不变。

## 执行器行为

- evaluate 按固定语句顺序调度：command `update` → rewards
  （raw、`weighted = raw * weight * dt`、Σ）→ terminations（masks、OR）→
  command `advance`（rewards/terminations 读 pre-advance command state）；
  observe 按 group/term 顺序写 obs 切片；reset = command `reset_env`
  （masked）→ sim-reset terms 写 compact 行。
- RNG：per-lane SplitMix64（`torch_manager.rand.draw_uniform`），批量语义为
  term 一次 draw `count=k` 等价于 per-lane 连续调用 k 次。
- 解析树持久复用：resolver 产出的 resolved tree 是三个 kernel 共用的持久
  anchor；simulator-backed 数组叶子由 read program 原地重写，
  `_TensorCache` 在同一 anchor 上按叶子值身份缓存 tensor 视图，0 维
  int/float/bool 叶子转为 Python 标量（graph-break-free 编译的关键）。
- reset 路径是静态全批 masked 设计：选中的 lane 由布尔掩码选择，
  reset write contract 覆盖的可写叶子（reset hook 的 receiver command
  tree、RNG state 与 lane 形状的 metric）在 reset term 首次访问时 clone，
  term 写 clone 行后以 `where(mask, clone, original)` 合并回常驻缓冲，
  未选中 lane（含 RNG 状态）保持不变；compact sim-reset write 缓冲布局与
  `SimResetRuntime.buffers` 一致。

## 批量 term 约定

ctx（`_BatchContext`）字段全部是**整批**张量：

- `ctx.sim[key]` → `(N, ...)` 张量（SimDataQuery 参数同样是整批张量）
- `ctx.commands[name].field` → per-env 字段 `(N, ...)`；嵌套 kernel_data（如
  `motion.clip.joint_pos`）为 shared `(F, J)`，term 自行按 `steps[:, 0]` gather
- `ctx.actions[name].current()/previous()` → `(N, A)`；`action_queue/action_ptr`
  同理
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
- reset：`(ctx, sim_writes, *args)`，ctx 为 lane-gathered（第 0 维 =
  len(env_ids)），`sim_writes[name]` 为 compact `(M, *write_shape)` 张量

Command hooks（`update`/`advance`/`reset_env`）是 CommandTerm 类上的
`@staticmethod` + `@dispatch` 函数，签名为 `(receiver, ctx)`；`receiver` 是该
command 的 kernel-data namespace 视图（batch/reset 场景下分别为整批/gathered
行）。host 侧 `reset(ResetContext)` 与 `on_transition()` 保持 numpy host 实现，
ActionTerm 的 `process/_reset` 也保持 numpy host 路径不变。

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
  单线程段（observe 图约 19 段）。修复：`min_chunk_size=256`（env `MOTRIX_TORCH_MIN_CHUNK`）、
  默认图内线程 4→8、observe 改 per-term 连续缓冲 + 单 kernel 组装、输出张量改为图参数使双缓冲槽
  共享编译图、observe 按 group 拆分独立编译图。图内线程扫描（
  `MOTRIX_TORCH_COMPILE_THREADS=1/2/3/4/8/16`）4 最优；numactl 在单 NUMA node 上无效。
- **冷惩罚测量**。物理后首个编译 kernel 存在冷缓存惩罚：同段计算重复执行 0.29ms vs 物理后执行
  0.59ms（Numba 因单小并行区天然免疫）。
- **已知残余**。EPYC 上相对 Numba 约 1ms 残余 = 冷缓存惩罚 + reset 静态全批布尔掩码设计
  （RNG 状态 `where(mask, ...)` 门控）带来的约 3x lane 计算结构性成本；本机持平 16.8–17.0ms。
