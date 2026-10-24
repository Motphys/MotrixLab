# RL 多算法架构设计

## 摘要

本文描述 MotrixLab 在 RL 集成层支持多算法的架构。命令行对用户暴露单一的 RL method selector（例如 `skrl.ppo`、`rslrl.ppo`、`motrix.fastsac`），内部展开为 `env_name + rllib + train_backend + algo` 四个正交维度，用来定位配置、选择训练实现、记录 run metadata 和发现 checkpoint。架构目标是让不同 RL 框架与算法（SKRL PPO、RSLRL PPO、Motrix FastSAC，以及后续的 SAC/TD3/RPO/AMP、多智能体算法）各自独立演进，同时共享环境创建、seed、run 目录、metadata 与 checkpoint 等通用编排，而外部算法接入时不需要改动训练入口、play 入口或 checkpoint 发现逻辑。

本文只描述架构：组件、接口、不变量与取舍。RL 框架整体分层与解耦原则见[项目整体架构设计](./architecture.md)；SAC / FastSAC 的算法原理见[SAC 与 FastSAC 算法原理入门](../research/sac-and-fastsac-primer.md)。

## 1. 设计动机

RL 集成层需要在同一套入口下承载多个 RL 框架和多种算法：

- 一个 `rllib`（RL 框架 namespace）可能支持多个 train backend（如 SKRL 支持 `jax` / `torch`）。
- 一个 `rllib` 下可能有多种算法（PPO、SAC、TD3……），其模型角色、memory 类型、agent 超参数结构彼此不同。
- 自研算法归入 `motrix` namespace，例如 FastSAC 使用 `motrix.fastsac`。
- on-policy（PPO/RPO/AMP）与 off-policy（SAC/TD3）算法在 memory 与并行规模上要求不同，PPO 的字段不应泄漏到 off-policy 配置。

因此架构把"某环境用什么超参数"（配置注册表）、"某 RL 框架提供哪些训练实现"（framework 与 provider）、"如何创建 run 与发现产物"（run/metadata/checkpoint）三类关注点分离，并让它们只通过 `(env_name, rllib, train_backend, algo)` 这组正交键交互。

## 2. 设计目标

1. **RL method 是一等配置维度**：用户通过 `mxlab train task=cartpole/skrl.ppo` 选择环境与 RL 方法的完整训练配方。
2. **内部维度正交**：配置查找、trainer 查找、run metadata、checkpoint 发现都显式包含 `rllib / train_backend / algo`。
3. **训练配方显式选择**：训练必须提供 `task=...`，不隐式选择环境或算法。
4. **算法可独立演进**：每个算法可自定义模型角色、memory、agent 配置和 runtime 配置。
5. **共享通用编排**：环境创建、seed、run 目录、metadata、train/play 主流程不在每个算法里重复。
6. **配置按算法收敛**：每个算法拥有独立配置类型，字段不跨算法泄漏。
7. **外部接入面稳定**：外部算法实现 `TrainerBase`、注册 `RlFramework`/`AgentProvider`、注册配置类后，不需要修改 `mxlab train`、`mxlab play`、checkpoint 发现或 backend 选择逻辑。

## 3. 非目标

- 不要求所有 SKRL 上游算法一次性实现。
- 不要求 RSLRL 支持 PPO / Distillation 以外的算法。
- 不强制每个算法同时支持所有 train backend；支持矩阵由 framework/provider 显式声明。
- 不在单智能体算法接入时把现有环境接口重构成 multi-agent 接口。

## 4. 分层架构

```
┌─────────────────────────────────────────────┐
│ mxlab train · mxlab play                     │
│ CLI 解析、自动发现、错误展示                 │
├─────────────────────────────────────────────┤
│ motrix_rl.runner                            │
│ 通用 train 编排、TrainRequest -> TrainResult │
├─────────────────────────────────────────────┤
│ motrix_rl.frameworks                        │
│ RlFramework -> AgentProvider 索引与查询      │
│ Hydra algo schema 注册 / TrainerContext / TrainerHandle │
├─────────────────────────────────────────────┤
│ motrix_rl.backend_runtime                   │
│ train backend 可用性判断与自动选择           │
├─────────────────────────────────────────────┤
│ motrix_rl_builtin · motrix_rl_skrl · motrix_rl_rslrl │
│ 插件包：算法特异 model/memory/agent/runtime + Trainer │
├─────────────────────────────────────────────┤
│ motrix_rl.runs · motrix_rl.checkpoints      │
│ run 目录 / metadata / checkpoint manifest    │
└─────────────────────────────────────────────┘
```

`rllib` 是 RL 框架 namespace，也是 run metadata 与 CLI method 中的第一段。`algo` 是该 framework 内部的算法/agent 名称。`train_backend` 是训练实现所用的计算后端（如 `jax` / `torch`，也允许自定义名称）。

### 4.1a 共享执行与 Tensor 传输边界

`motrix_rl.execution` 管理阶段交接、取消和进程生命周期；
`motrix_rl.transport` 提供有界 SPSC tensor slots 与版本化 inference snapshot。
公共 `AsyncTrainerBase` 继承现有 `TrainerBase`，其 `train()` 拥有整个资源作用域与父进程生命周期：
算法声明训练计算与数据流，公共层创建 spawn context 和 process，等待 readiness、放行、监控完成与失败，最后停止并回收进程。
它复用 `ProcessSupervisor`，不另建进程管理器。目标 `WorkerSpec` 只声明 worker ID、模块级入口、可 pickle 参数与资源配置，
不持有 process。算法仅接收只读 worker 状态与算法进度，不自行查询或操作 OS 进程。
算法负责完成决策与指标含义，公共层负责消息消费、日志和面板生命周期；首批更新/暖机门槛不能阻塞 readiness 放行。
本节定义目标边界，具体接口、验收条件及尚未满足的实现差距见下节[异步训练架构目标与验收](#异步训练架构目标与验收)。

公共 worker 工具提供 seed 应用、CPU 资源机制、日志重定向、callback 式 SIGINT 与通用启动闸门；
算法决定 seed 偏移、CPU 分配、拓扑、resource slot 交付内容和训练预算。阻塞式资源交付必须检查 stop 并有限时等待。
共享资源先登记清理，进程回收先于 queues/IPC 关闭；启动部分失败、初始化失败及运行异常不得覆盖原错。
本层不实现 APPO，也不提供通用 replay、UTD 或算法 worker wiring。算法插件提供 payload 布局、采样和学习逻辑，
不需要重新实现共享内存所有权协议。

```text
既有 runner / TrainerContext / metrics / checkpoint artifacts
                  │
           算法 Training Adapter
                  │
        execution：阶段 / 生命周期 / 停止
                  │
Collector ── 有界 SPSC tensor ring ── Learner
    ▲                                  │
    └──── 一致 inference snapshot ──────┘
                  │
       算法拥有 replay / unroll storage
```

- `transport.ring.SharedTensorRing` 接受有序 `TensorField` 布局，支持不同 shape/dtype
  的字段。`read_span` 返回借用 view；只有读取或复制完成后才能 `commit_reads`。
- `transport.cuda_ring.CudaTensorRing` 运输一个固定 shape 的 CUDA slot；打包布局由
  算法定义。issued cursor 用于预留容量，completed cursor 只在 CUDA event 完成后发布。
  producer 必须保持 copy source 不变直至复制完成；consumer 必须在当前 slots-device
  stream 上排完所有读取，再确认释放。CUDA owner 必须保持 slots 存活直到 peer 退出。
- `transport.snapshot` 原子发布 float32 flat model parameters 与具名辅助 tensor，
  共享一个 seqlock/version。CPU staging、CUDA IPC 和双缓冲一致性由通道负责；
  auxiliary state 的语义和加载时机由算法决定。它不是任意 module state_dict 序列化器。
- host cursor/seqlock 仍依赖 x86-64 TSO；不宣称支持 ARM，也不是 MPMC。
  readiness、retry/backoff 和取消属于执行/调用层；数据通道不拥有监控线程。
  构造共享游标与 snapshot 时拒绝非 x86-64 主机，避免在不支持的内存序平台静默运行。
- 公共边界拒绝负数、非整数或超过尚未消费数量的 `commit_reads`，拒绝错误的 tensor 字段布局；
  校验失败不能推进游标。确认仍由调用方保证发生在实际读取完成或 CUDA 读取已排入约定 stream 之后。
- Snapshot 发布前校验参数长度与辅助状态 schema，拒绝缺字段、额外字段或错误 shape/dtype，
  失败不能打开 seqlock；读取先形成私有快照，通过版本复检后才修改推理模型。
- `Parameters.value` 与 `Sample.value` 是 adapter 拥有的 opaque payload；冻结 dataclass 不等于
  递归冻结 tensor。公共执行器不 clone 任意对象，不承担算法模型复制；adapter 必须提供 detached
  immutable snapshot，并在转交后停止复用 sample 存储。这与 ring 借用 view 的显式确认是两种不同的所有权协议。

FastSAC 每个 collector 保持一条 ring、一个参数通道，多个 ring 按原 topology 分配给
唯一 learner。算法适配器保留六字段 transition 布局、CUDA fused float32 packing、
单 pinned staging 和稀疏 final-observation sidecar。sidecar 在 slot 发布前写入，只有
completed read acknowledgement 才允许复用。其 replay 继续利用相邻 observation
构成 next observation；generation 对齐、UTD、DDP、NUMA 和恢复进度不属于通用 ring。

transport 不等于 replay：前者限制在途数据，后者决定历史保留与采样。使用 unroll 的算法
可以提供序列、behavior log probability、mask 和版本布局，而不经过 FastSAC replay。
Sync 可以直接交接本进程数据，无需为接口统一引入 ring copy。共享 transport 本身不代表
已提供所有算法的 worker wiring、学习预算策略或 APPO 实现。

验证覆盖 host 和 CUDA slot 环回、背压、借用 view 生命周期、实际 spawn/IPC、
不同 unroll 布局和 snapshot auxiliary 布局；FastSAC 原有 replay/final observation、
collector、pipeline 和 DDP 测试继续保护算法行为。单 GPU IPC 验证不替代多 GPU NCCL、
跨 NUMA 吞吐与端到端训练性能验收。

### 异步训练架构目标与验收

#### 目标与适用范围

本节是自有异步训练器后续重构的验收基准。目标不是减少 FastSAC 文件行数，而是让算法层仅围绕**算法状态、计算和数据流**编程，不围绕进程对象编程。目标接口尚未全部落地，不能把本节中的类型当作当前可导入 API。

适用对象是继承 `AsyncTrainerBase` 的 process-backed trainer；不强制 SKRL/RSLRL 等上游 runner 改写，不改变既有 provider discovery、`TrainerContext`、run/checkpoint 服务、同步执行器或线程参考执行器。当前具体适配对象仍为 FastSAC，不实现 APPO、通用 replay、拓扑求解器、万能 Worker 类层次或新的 backend registry。

#### 所有权边界

| 责任 | 算法实现层 | 公共执行层 |
|---|---|---|
| 配置与恢复 | 算法参数、checkpoint 内容、恢复进度解释 | run/context 与产物服务、资源作用域 |
| 模型与数据 | actor/critic/optimizer、transition/unroll schema、replay、normalizer、final observation | schema 驱动的 ring/snapshot 机制与借用/确认协议 |
| 布局与连接 | worker 数量/角色、环境分片、collector→learner 关系、placement 约束和传输选择 | 平台事实查询、设备解析工具、placement 应用与通道分配工具 |
| 进程 | 声明 worker ID、入口、参数与所需资源 | spawn context、`ctx.Process`、进程句柄表、start/check/join/terminate/kill |
| Worker 外壳 | 初始化算法对象，决定算法何时 ready，执行采样/学习及其内部同步 | 日志/seed/线程预算/信号应用、错误捕获、ready/start/stop/finish 通信 |
| 进度与结束 | UTD、warmup、预算、锁步规则、允许的 worker 完成边界 | 收集进度与状态、执行结束决策、超时、失败及取消分类 |
| 指标与展示 | 指标字段、权重与聚合口径、进度轴、首批指标有效条件 | queue 消费、系统采样、logger/console 生命周期、traceback 落盘、最终 drain |
| 清理 | 立即登记自有对象的释放函数；worker 内部算法资源释放 | 回收进程后关闭共享资源，保证失败路径与原异常保护 |

算法可声明/分配算法数据通道，但不实现通用 queue factory、feeder 清理规则、错误队列或 readiness 队列。公共构建作用域提供同一 spawn context 下的分配能力；确需 context 的低层共享资源构造可由作用域委托，但不得因此在算法层构造 Process。数据 payload 和 CUDA IPC handle 的含义仍由算法适配器定义，公共层不得导入 FastSAC 类型。

#### 父进程算法扩展点

父进程只围绕以下三个语义扩展点设计；不再要求算法回传公共生命周期 callback。以下签名为目标契约，类型名可在实施计划中细化，职责与输入/输出边界不可隐式扩大。

```python
class AsyncTrainerBase(TrainerBase):
    @abstractmethod
    def build_training(self, runtime: BuildRuntime) -> TrainingPlan:
        ...

    @abstractmethod
    def evaluate_progress(
        self, state: AlgorithmState, progress: TrainingProgress,
    ) -> TrainingDecision:
        ...

    def collect_metrics(
        self, state: AlgorithmState, reports: WorkerReports,
    ) -> TrainingMetrics:
        # 公共默认实现为空；需要算法聚合时覆写。
        ...
```

1. **`build_training` 必需实现**：检查算法配置、获取环境维度、解释 resume、生成布局、分配/声明算法通道并返回 `TrainingPlan`。Plan 持有父进程算法 state、worker 描述与数据连接。不得包含 Process 或由算法提供的 spawn/监控循环。资源获取成功后立即登记清理；prepare 中途失败也能释放，不依赖 plan 成功返回。
2. **`evaluate_progress` 必需实现**：根据带来源和单位的算法计数、worker 语义状态及只读 OS 状态，返回继续或结束决策，并明确哪些 worker 已到合法完成边界。计数的 sum/max/均值不能由公共层猜测。该函数不消费 queue、不等待 worker、不保存/操作 Process，也不以 `exitcode == 0` 代替算法完成证明。所有 worker 成功与否最终仍由公共层裁决。
3. **`collect_metrics` 可选覆写**：从公共层已消费的 worker reports 计算算法指标与展示有效性。返回标准 metrics/progress/diagnostics 及数据有效条件；不调用 Rich、TensorBoard、sleep 或 queue.get。无额外聚合需求时沿用公共默认实现，不强迫算法写空方法。底层 `TrainingLogger`/`MetricsLogger` 和既有进度轴契约继续复用。

`WorkerSpec` 的目标字段是稳定 ID、role、模块级 target、args/kwargs、placement/seed/线程预算等实际需要的配置；日志位置由公共层按 worker ID 推导。role 仅用于分组和状态观察，不隐含 learner 必须均匀拥有 collector 的策略。Process 与 supervisor 句柄不进入 spec、plan 或算法 state。

Worker 数量、collector/learner ownership、环境 shard、learner 消费哪些 endpoint，以及 CPU/NUMA/device placement 属于公共 topology/endpoint wiring；算法不展开和创建 peer worker。算法只声明角色需求、transport capability 偏好和 payload schema，common 将其展开为带有 index/count/rank/world-size/owner 与本地 endpoint 的 worker context。CUDA IPC 的可用性和通用 transport 选择由 common 根据 producer/consumer placement 与 capability 决定；算法保留 transition schema、sidecar、generation、replay ingest、UTD 及其他算法特定 transport 约束。当前 `motrix_rl.execution.topology` 提供 `EndpointWiring`、`cuda_ipc_capable` 与 `wire_endpoints` 作为这一阶段的轻量 endpoint 描述和 capability 选择契约；它不创建算法 ring，也不理解 payload。

`TrainingProgress` 提供 worker ID 关联的只读状态和独立的算法进度，包括未 ready、ready、运行、已报告完成、已退出/失败及退出码诊断；不提供 start/join/terminate 等操作。`TrainingDecision` 表达算法预算是否满足及合法完成集合，不允许忽略非零退出、提前退出或隐藏 worker 错误。取消与失败优先于成功；checkpoint 由算法决定内容与保存时机，公共层登记 artifact 通知。

继承的 `play(policy)` 仍需按已有 `TrainerBase` 契约实现；它不是新增的异步编排 hook。同步/异步可复用 FastSAC builder，不增加第二套 provider 入口。

#### Worker 算法入口与公共外壳

Worker 使用可 spawn 的模块级函数，不 pickle Trainer，也不要求 collector/learner 继承统一算法基类：

```python
def run_collector(runtime: WorkerRuntime, config, resources) -> None:
    ...

def run_learner(runtime: WorkerRuntime, config, resources) -> None:
    ...
```

`WorkerRuntime` 提供 `ready()`（报告初始化就绪并可取消地等待放行）、`should_stop()`、`report(...)` 与 `finish(...)`（声明算法完成）。公共外壳先应用日志/CPU/NUMA/线程/seed/信号设置，再调用算法入口；捕获异常并上报 worker ID、阶段和原 traceback。Seed 偏移、CUDA 初始化时机和 DDP rank 集成由算法声明，公共 seed 工具不得提前为父进程初始化 CUDA。

`ready`、`finish`、`exit` 是三个不同事件：

- 未 ready 即退出（即使退出码 0）属于启动失败。
- 已 ready 只表示初始化完成，不表示 warmup 或首次 learner update 完成。
- `finish` 是算法完成声明，需由预算/进度规则验证；它不替代进程最终退出状态。
- 返回但未报告合法完成属于提前退出，除非公共层已发起停止并按取消/正常收尾协议允许退出。
- 非零退出始终不是成功。公共外壳因已请求 stop 结束资源等待时不重复制造 worker 缺陷；父进程 Ctrl+C 保持取消语义。

不统一 `collect_once`/`learn_once` 调度：FastSAC 的 generation merge、replay ingest、DDP collectives 时序仍在算法入口内。资源交付和背压等待必须可检查 stop 或有有限时退出；不能把普通 start gate 释放误认为能解除所有 queue/NCCL 阻塞。强制回收是最终兜底，不承诺训练状态可恢复。

#### 公共生命周期及清理不变量

```text
公共资源作用域 → build_training → 构造/启动 workers
  → 期间持续消费错误与状态 → 等待全部 readiness → 放行
  → 消费报告 → evaluate_progress / collect_metrics → 执行结束决策
  → 请求 stop / 解除可解除的等待 → 有界 reap
  → 最终 drain / 检查迟到失败 / 登记产物 → 关闭展示与共享资源
```

- 一个 worker ID 对应一个由公共层持有的 process；父进程的 context 与所有同步对象使用一致的启动上下文。
- Base 的 `train()` 不允许被算法复制或通过兼容 callback 绕行；没有 `start_workers`、`wait_for_ready`、`drain_errors`、`render_boot_panel`、`shutdown_workers` 等算法抽象方法。
- 从第一个 spawn 起就有 supervisor 保护；部分 spawn、placement/log 初始化、算法 build 或 poll 失败全部进入统一资源出口。
- Runtime 在启动与运行阶段都消费错误。首批 metrics 的展示门槛只能控制展示，不能延迟 start release。
- 同一轮出现算法预算完成与 worker 错误时，错误优先；进程回收后继续检查迟到的非零退出，最终消息在通道关闭前消费。
- 主异常保留对象与 traceback；清理异常记录且不覆盖主错，没有主错时清理失败必须上报。不得以宽泛吞异常把失败运行变为成功。
- Tensor/IPC owner 保持存活直至所有借用者退出；queue 的 feeder 策略由公共资源作用域统一处理。worker 私有环境/renderer/process-group 的释放按真实接口执行，不虚构底层 `close()` 能力。

#### 当前实现差距（验收基线）

以下是当前磁盘实现与目标的对照，不是已完成声明：

| 当前实现证据 | 目标差距 |
|---|---|
| `motrix_rl/execution/trainer.py` 的 `WorkerSpec.process`，Base 对既有 process 调用 start | spec 必须改为入口与参数；Process 构造及句柄所有权归公共层 |
| FastSAC `async_impl/train.py` 的 `mp.get_context("spawn")`、`ctx.Process`、`p_learners/p_collectors` | 只声明布局与 worker；不维护进程列表 |
| FastSAC 对 `exitcode` 的判断及 `_complete` | 改由带 ID 的只读 runtime 状态和算法完成决策，不访问 raw Process |
| `AsyncTrainingSession` 要求算法提供 release_start、drain_ready、poll、close 等 callback | 公共启动/错误/展示/最终 drain 不再通过算法闭包回传；按三个语义扩展点划分 |
| FastSAC 创建 error queue、handshake、queue 清理及 traceback 落盘，`_poll/_close` 控制 Live/logger | 错误、启动与展示生命周期归公共层，算法只定义 slot 内容、指标与有效条件 |
| Worker 已调用公共 seed/日志/信号工具，但入口仍自行捕获异常、上报 ready/错误/stop | 使用公共 worker 外壳与 runtime 协议，保留算法初始化及内部循环 |

已经存在的 `ProcessSupervisor`、SPSC ring/snapshot、startup barrier、worker runtime helpers、公共 console 均应复用，不重新建立平行实现。现有测试通过只证明已覆盖的行为，不能证明上表的职责迁移已完成。

#### 最终验收矩阵

最终实施报告必须对每项给出实现路径、验证命令/结果及限制；未满足项保持未完成，不得用测试总数或文件行数代替：

| 编号 | 验收条件 | 必需证据 |
|---|---|---|
| A1 | FastSAC 仅提供 worker 描述，不创建或持有 Process，不选择 spawn context | 源码 review 与一次性引用扫描；原始 process 仅在公共 runtime 内 |
| A2 | 算法只实现 build、progress decision、metrics 语义扩展点及 worker 入口，无第二个生命周期循环 | 接口/调用链 review；没有 legacy fallback、名义继承或万能 poll 闭包承载旧 train |
| A3 | 日志、错误、readiness、进程状态、最终 drain 与资源释放由公共 runtime 管理 | 不带 FastSAC 依赖的 toy 算法和真实 CPU spawn 测试，覆盖状态/消息/回收顺序 |
| A4 | ready、算法完成、进程退出不混淆 | 未 ready 的零退出、运行中提前零退出、合法完成、未声明完成返回、迟到非零退出回归 |
| A5 | 启动/运行错误与取消可靠处理，不丢原错、不漏子进程 | 部分 spawn、启动超时、初始化/poll 失败、Ctrl+C、阻塞资源取消及主错+清理错测试 |
| A6 | worker 私有资源与公共 IPC 所有权明确 | 初始化中途失败、资源交付失败及正常退出的 cleanup 测试；reap 后才关闭通道 |
| A7 | FastSAC 算法与用户契约不变 | replay/final observation、generation、normalizer、UTD、恢复进度、checkpoint、live metrics 既有回归；短 run 无更新、首批面板门槛回归 |
| A8 | 原生产数据路径未被通用 Python queue 替换，未引入额外热路径复制 | 代码 review、既有 transport/管线测试、可用单 GPU CUDA/IPC 实际验收；性能结论须有基准 |
| A9 | 文档、测试与实现的边界一致 | 更新本基线差距与 API 文档，执行 lint/格式检查；逐项列出已执行、跳过及硬件限制 |

引用扫描属于一次性迁移验收，不写成“旧符号必须不存在”的长期单元测试。单 GPU/CPU 证据不能替代多 GPU NCCL、跨 NUMA 吞吐或长期训练稳定性；缺少硬件验收时必须明确范围，不能宣称已验证。

### 4.1 Workspace package boundaries

RL 集成拆分为控制平面（同时持有稳定 contracts）和可选 provider 插件两个边界：

| Package | 所有权 | 允许依赖 | 不负责 |
|---|---|---|---|
| `motrix_rl` | 控制平面与稳定 contracts：runner、CLI 协作、run metadata、checkpoint manifest、backend 选择、`motrix_rl.frameworks` 注册/查询与 Hydra schema 安装，以及 `RlFramework`、`AgentProvider`、`TrainerBase`、`TrainerContext` 等 provider/trainer 契约 | 环境与通用运行时；可使用控制平面基础依赖 | 不内置具体 FastSAC、SKRL 或 RSLRL trainer；contracts 不包含 provider 实现细节 |
| `motrix_rl_builtin` | Motrix 内置 FastSAC（`motrix.fastsac`）及其模型、replay、同步/异步 trainer 与导出实现 | 仅依赖 `motrix_rl` 与 Torch 等 FastSAC 运行时依赖 | 不把 FastSAC 类型重新定义为控制平面公共契约 |
| `motrix_rl_skrl` | SKRL PPO provider（JAX/Torch）及 SKRL-specific wrapper、配置和 trainer | 依赖 `motrix_rl`、SKRL 与 Torch；额外 JAX runtime 按 extra 安装 | 不让 SKRL 依赖成为控制平面的强制依赖 |
| `motrix_rl_rslrl` | RSLRL PPO provider（Torch）及原生 `rsl_rl` runner 适配 | 仅依赖 `motrix_rl` 与 RSLRL/Torch | 不把 RSLRL runner 细节泄漏到控制平面 |

控制平面保留稳定的 `motrix_rl.frameworks` 入口与 `motrix_rl.contracts` 契约模块；插件包实现 `RlFramework`，并通过该入口注册，而不是由 `motrix_rl` 直接 import provider 模块。算法库是所属插件包的必需依赖，仅额外 backend runtime 使用可选依赖组；插件依赖控制平面，控制平面不依赖插件。根项目必需安装 builtin 与 SKRL，RSLRL plugin 和 SKRL JAX runtime 通过 extras 选择。

### 4.2 Entry-point discovery

插件以 Python packaging entry point group `motrix_rl.frameworks` 发现。每个插件包在自己的分发声明中提供一个命名 entry point，指向无参数 `register()` callable；以下汇总三个包的条目：

```toml
[project.entry-points."motrix_rl.frameworks"]
builtin = "motrix_rl_builtin.plugin:register"
skrl = "motrix_rl_skrl.plugin:register"
rslrl = "motrix_rl_rslrl.plugin:register"
```

`motrix_rl.plugins.load_plugins()` 在框架查询或 Hydra task 组合前读取已安装 entry points，按名称稳定排序并确保每个 entry point 只加载一次；注册失败向调用方传播，不被静默吞掉。未安装的可选插件不影响控制平面启动，只有选择其 method 或查询其 framework 时才会报告 provider 不可用。entry-point 名称是分发标识，真实 method identity 仍由 `RlFramework.name`、`AgentProvider.agent_name` 和 `train_backend` 声明。

## 5. CLI 与 method 解析

### 5.1 method 形式

用户可见的主选择维度是 RL method，形式为 `rllib.algo`：

```text
skrl.ppo
rslrl.ppo
motrix.fastsac
```

`mxlab train` 通过 Hydra 的 `task=<env>/<rllib>.<algo>` 选择完整训练配方；`task.train_backend` 可单独覆写训练后端，未指定时自动选择。激活安装环境后运行：

```bash
mxlab train task=cartpole/skrl.ppo
mxlab train task=cartpole/rslrl.ppo
mxlab train task=g1-walk-flat/motrix.fastsac
mxlab train task=g1-walk-flat/motrix.fastsac algo.asynchronous=false
```

`mxlab` 是唯一 CLI 入口；train/play/export 的实现与内部可调用 adapter 保留在 `motrix_rl.commands`，不提供独立模块 CLI。
版本查询只使用 `mxlab --version`。

### 5.2 method 解析

```python
@dataclass(frozen=True)
class RlMethod:
    rllib: str
    algo: str


def parse_method(value: str, deprecated_rllib: str | None = None) -> RlMethod: ...
```

规则：

- `skrl.ppo` → `rllib="skrl"`, `algo="ppo"`。
- 裸算法名（如 `ppo`）只在同时传入 `--rllib` 时作为兼容 shorthand 允许。
- 三段形式（如 `skrl.torch.ppo`）不作为主接口，train backend 继续由 `--train-backend` 指定或自动选择。

### 5.3 play 不需要 `--algo`

play 不是训练配置选择，而是加载已训练 policy，正确的 `rllib/train_backend/algo` 来自 run metadata：

- 指定 `--policy` 时，从 checkpoint 同级或上级 run 目录的 `metadata.json` 读取 method，再选择对应 `AgentProvider` 做 inference；找不到 metadata 直接报错。
- 不指定 `--policy` 时，在 `runs/{env}` 下读取所有 run metadata，选择最近 run，读取其 method，再按 `checkpoints/manifest.json` 定位 policy；没有 metadata run 或 manifest artifact 直接报错。

## 6. RL task 配置查询

Hydra task 配置回答"某环境用什么超参数"。

### 6.1 按需组合

系统不维护 Python task registry 或其他全局 task 索引，也不预先检查配置文件。
训练时直接请求 Hydra 组合
`task/<env>/<rllib>.<algo>[.<backend>]`。task YAML 的 defaults、继承、
package 和 schema 均不由 Python 手动解析。

### 6.2 注册与 method 所有权

method identity 由 framework 与 provider 共同定义：`rllib` 来自
`RlFramework.name`，`algo` 来自 `AgentProvider.agent_name`，配置类型来自
`AgentProvider.config_type`。配置 dataclass 不再重复声明 `rllib/algo` ClassVar。

```python
class SkrlPpoJaxProvider(AgentProvider[SkrlCfg]):
    config_type = SkrlCfg
    agent_name = "ppo"
    train_backend = "jax"


class SkrlPpoTorchProvider(AgentProvider[SkrlCfg]):
    config_type = SkrlCfg
    agent_name = "ppo"
    train_backend = "torch"
```

`RlFramework` 按 agent 保存唯一的 `config_type` 和 backend providers；同一 agent
若声明多个不同 config type，framework 构造立即失败。`register_framework()` 同时
注册 provider 能力并把 config type 安装到 Hydra，不维护平行的 schema registry。

外部算法不需要修改 framework 内的固定 Union 或额外 schema 表；只需提供带
`config_type` / `agent_name` / `train_backend` 的 provider，并通过自己的
`RlFramework` 注册即可。

### 6.3 Task option 选择

训练入口直接消费 Hydra 根据 `task=<env>/<rllib>.<algo>[.<backend>]` 组合出的配置。
backend 专用 option 需要显式选择；option 不存在或其 defaults 损坏时直接报告 Hydra
配置错误，不在框架层执行路径猜测或 fallback。

## 7. RlFramework 与 AgentProvider

`motrix_rl.frameworks` 回答"某训练后端、算法如何创建 Trainer"。它是控制平面的稳定入口，不是 provider 实现包的聚合模块；provider 通过 `motrix_rl.frameworks` 的注册 API 接入。

- **`RlFramework`**：外部 RL 框架的注册入口，定义 framework namespace（即 `rllib`），持有一组 framework-scoped `AgentProvider`，并提供 supported agents / supported train backends / provider 查询 API。
- **`AgentProvider`**：framework 内部的训练实现单元，声明 `train_backend`、`agent_name`、`checkpoint_format`，并负责创建 `TrainerBase`。
- **`TrainerBase`**：可运行对象基类，负责框架内部的 train 与 play。外部算法接入时继承它。

这些契约与 `TrainerContext` / `TrainerHandle` 定义于 `motrix_rl.contracts`；`motrix_rl.frameworks` 保留公共兼容入口，并负责将具体的 `RunContext`、运行配置与 result factory 注入接口对象。contracts 模块不 import 控制平面的 `RunContext` / Hydra 配置类型，后者只在控制平面和插件的调用边界被使用。

`AgentProvider` 不要求暴露 `wrap_env` / `make_models` / `make_memory` 等细粒度步骤——这些是内置 trainer 的内部 helper，不是外部接入面。

### 7.1 接口

```python
class TrainerBase(ABC):
    @abstractmethod
    def train(self) -> None: ...
    @abstractmethod
    def play(self, policy: str) -> None: ...


@dataclass(frozen=True)
class TrainerContext:
    run: RunContext
    env_name: str
    run_dir: Path
    checkpoint_dir: Path
    checkpoint_format: str
    num_envs: int
    play_num_envs: int
    seed: int | None
    logging: LoggingConfig
    checkpoint: CheckpointConfig
    rl_cfg: Any
    render: RenderConfig | None = None
    resume_from: str | None = None


class AgentProvider(ABC):
    @property
    @abstractmethod
    def train_backend(self) -> str: ...
    @property
    @abstractmethod
    def agent_name(self) -> str: ...
    @property
    @abstractmethod
    def checkpoint_format(self) -> str | None: ...
    @abstractmethod
    def create_trainer(self, context: TrainerContext) -> TrainerBase: ...


class RlFramework(ABC):
    def __init__(self, providers: Iterable[AgentProvider]) -> None: ...
    @property
    @abstractmethod
    def name(self) -> str: ...
    def supported_agents(self) -> tuple[str, ...]: ...
    def supported_train_backends(self, agent_name: str) -> tuple[str, ...]: ...
    def get_agent_provider(self, agent_name: str, train_backend: str) -> AgentProvider | None: ...


@dataclass(frozen=True)
class TrainerHandle:
    run: RunContext
    trainer: TrainerBase

    def train(self) -> "TrainResult": ...
    def play(self, policy: str) -> None: ...
```

外部实现以 `RlFramework` 作为唯一入口，`register_framework(framework)` 完成注册。`AgentProvider` 不重复声明 `rllib`；provider 索引统一由 `frameworks` 通过 `framework.name`、`provider.train_backend`、`provider.agent_name` 管理。

`TrainerContext` 是传给 trainer 的**单一**上下文对象。它由 `create_trainer_context(...)` 从 `RunContext` 和已组合的 `TaskConfig` 投影得到：`run` 提供 metadata，`num_envs / play_num_envs / seed / logging / checkpoint` 是 framework 统一运行参数，`rl_cfg` 是 provider 的具体 dataclass，`render / resume_from` 是本次运行输入。

### 7.2 trainer 创建

```python
def create_trainer_context(
    run, *, num_envs, play_num_envs, seed, logging, checkpoint, rl_cfg, render=None, resume_from=None
) -> TrainerContext: ...


def create_trainer(context: TrainerContext) -> TrainerHandle:
    metadata = context.run.metadata
    provider = get_agent_provider(metadata.rllib, metadata.algo, metadata.train_backend)
    if provider is None:
        raise ValueError(...)
    context = _validated_trainer_context(provider, context)  # 校验 provider 与 metadata 的 (train_backend, algo) 一致
    return TrainerHandle(run=context.run, trainer=provider.create_trainer(context))
```

`create_trainer` 从 `TrainerContext.run.metadata` 查 provider，校验其 `train_backend / agent_name` 与 metadata 一致，必要时用 provider 的 `checkpoint_format` 补全 context，再创建真实 trainer 并返回框架管理的 `TrainerHandle`。`TrainerHandle.train()` 调用真实 trainer 的 `train()`，并基于 `RunContext` 生成 `TrainResult`。

## 8. 通用编排 runner

`motrix_rl.runner` 提供通用训练编排，入口层不含 framework 分支。

```python
@dataclass(frozen=True)
class TrainRequest:
    config: TrainConfig
    render: RenderConfig | None = None
    runs_root: str | Path = runs.LOG_DIR_PREFIX


def train(request: TrainRequest) -> TrainResult:
    return create_training_handle(request).train()


def create_training_handle(request: TrainRequest) -> TrainerHandle:
    config = request.config
    task = config.task
    method = RlMethod(task.rllib, task.algo)
    train_backend = backend_runtime.resolve_train_backend(
        task.env, method, task.train_backend, utils.get_device_supports()
    )
    provider = frameworks.get_agent_provider(method.rllib, method.algo, train_backend)
    rl_cfg = provider.validate_config(config.algo)
    run = runs.create_run_context(
        env_name=task.env,
        rllib=method.rllib,
        train_backend=provider.train_backend,
        algo=method.algo,
        seed=config.seed,
        checkpoint_format=provider.checkpoint_format,
        runs_root=request.runs_root,
    )
    resume_from = str(checkpoints.resolve_resume_checkpoint_path(config.resume)) if config.resume else None
    return frameworks.create_trainer(
        frameworks.create_trainer_context(
            run,
            num_envs=config.num_envs,
            play_num_envs=config.play_num_envs,
            seed=config.seed,
            rl_cfg=rl_cfg,
            render=request.render,
            resume_from=resume_from,
        )
    )
```

play 编排不引入独立 request 类型：入口（`mxlab play`）或 `TrainResult.play(...)` 先解析出 `RunContext`（从 metadata），再调用同一个 `create_trainer(create_trainer_context(run, ...))`，最后 `trainer.play(policy_path)`。best policy 与 resume checkpoint 的定位统一走 `motrix_rl.checkpoints`。

`TrainResult` 是 run 的查询视图：

```python
@dataclass(frozen=True)
class TrainResult:
    run: RunContext

    @property
    def run_dir(self) -> Path: ...
    def find_best_policy(self) -> Path: ...
    def find_resume_checkpoint(self) -> Path: ...
    def play(self, *, cfg_override=None, render=InteractiveRenderConfig()) -> None: ...
```

它只保存 `RunContext`，通过 checkpoint manifest 查询产物；trainer 不手写 result metadata，也不填写 `rllib/train_backend/algo`、`checkpoint_format`、`run_dir` 这类上下文已知字段。

## 9. 配置体系

### 9.1 结构

framework 统一运行参数位于 `TrainConfig` 根节点；算法配置 dataclass 只保存 provider 私有字段：

```python
@dataclass
class TrainConfig:
    task: TaskMeta
    num_envs: int
    play_num_envs: int
    seed: int | None
    logging: LoggingConfig
    checkpoint: CheckpointConfig
    algo: Any


@dataclass
class SkrlCfg:
    models: SkrlModelsCfg = field(default_factory=SkrlModelsCfg)
    memory: SkrlMemoryCfg = field(default_factory=SkrlMemoryCfg)
    agent: SkrlAgentCfg = field(default_factory=SkrlAgentCfg)
    trainer: SkrlTrainerCfg = field(default_factory=SkrlTrainerCfg)


@dataclass
class RslrlCfg(RslrlRunnerCfg):
    pass


@dataclass
class FastSacCfg:
    device: str | None
    agent: FastSacAgentCfg = field(default_factory=FastSacAgentCfg)
    trainer: FastSacTrainerCfg = field(default_factory=FastSacTrainerCfg)
```

provider config 不要求共享内部路径。SKRL 与 FastSAC 直接暴露各组件配置；
RSLRL 的整个 provider config 就是传给上游 `OnPolicyRunner` 的配置，不再额外包装一层 `runner`。
不同算法的 `agent` / `memory` 结构互不共享字段：

| 算法 | agent 配置 | memory |
|---|---|---|
| PPO / RPO | rollouts、gae_lambda、ratio_clip 等 on-policy 字段 | on-policy rollout memory |
| SAC / TD3 | tau、gamma、learning_starts、batch_size、gradient_steps 等 off-policy 字段 | replay memory |
| AMP | PPO 字段 + discriminator / expert dataset 字段 | on-policy + expert memory |

### 9.2 覆写路径

命令行覆写使用 dot-notation，路径以顶层配置为根：

```text
num_envs
play_num_envs
seed
logging.backend
logging.interval
checkpoint.interval
algo.models.<field>          # SKRL
algo.memory.<field>          # SKRL
algo.agent.<field>           # SKRL / FastSAC
algo.trainer.<field>         # SKRL / FastSAC
algo.actor.<field>           # RSLRL
algo.critic.<field>          # RSLRL
algo.algorithm.<field>       # RSLRL
```

## 10. 模型工厂

不同算法需要的模型角色不同：

| 算法 | 模型角色 |
|---|---|
| PPO / RPO | `policy`, `value` |
| SAC | `policy`, `critic_1`, `critic_2`, `target_critic_1`, `target_critic_2` |
| TD3 | `policy`, `target_policy`, `critic_1`, `critic_2`, `target_critic_1`, `target_critic_2` |
| AMP | `policy`, `value`, `discriminator` |
| IPPO | per-agent `policy`, `value` |
| MAPPO | per-agent `policy`, shared/centralized `value` |

模型角色是算法实现内部细节：内置 trainer 在自己的算法 helper 里声明所需角色并把 cfg 映射为模型集合，外部算法在自己的 `TrainerBase` 实现中处理。跨算法只共享**构建块**（activation 解析、MLP body、gaussian/deterministic policy head、Q/V head、obs/state space 校验、privileged critic observation 处理），而不是固定的模型角色组合。

## 11. Memory 体系

- **on-policy（PPO/RPO/AMP）**：rollout memory，`memory_size == -1` 表示使用 `agent.rollouts`。
- **off-policy（SAC/TD3/DDPG）**：replay memory，独立的 `memory_size`、`replacement` 等字段。

Memory factory 不依赖 PPO 字段，`memory_size == -1` 的语义只存在于 on-policy 配置。FastSAC 使用自带的 replay buffer，不经 SKRL memory 抽象。

## 12. Run 目录与 metadata

### 12.1 目录结构

run 目录包含 algorithm 维度，由 `runs.create_run_context(...)` 统一创建：

```text
runs/{env_name}/{rllib}/{train_backend}/{algo}/{timestamp}/
  metadata.json
  checkpoints/
    manifest.json
    latest.pt
    model_0001000.pt
```

对于 SKRL，框架把 `experiment.directory` 设为 `run_dir.parent`、`experiment_name` 设为 `run_dir.name`，避免 SKRL 再追加不可控目录。

### 12.2 metadata.json

每个 run 根目录写入 `RunMetadata`：

```json
{
  "env_name": "g1-walk-flat",
  "rllib": "motrix",
  "train_backend": "torch",
  "algo": "fastsac",
  "seed": 1,
  "created_at": "2026-06-08T19:30:00+00:00",
  "checkpoint_format": "pt",
  "motrixlab_version": null
}
```

`RunMetadata` 是持久化契约，是 play / export / benchmark 自动发现训练产物的唯一可信来源；文件扩展名只能作为 fallback。`RunContext`（定义在 `motrix_rl.runs`）是运行时上下文，包含 `run_dir` 与 `metadata`，并派生 `checkpoint_dir`。`create_run_context(...)` 创建目录、写 metadata、建 `checkpoints/`；`open_run_context(...)` 用于 play / resume / benchmark 读取已有 run。

## 13. Checkpoint 发现与 manifest

checkpoint 发现由 `motrix_rl.checkpoints` 统一实现，入口不做 per-framework 分支。run 在 `checkpoints/manifest.json` 记录实际产出的 artifact；`checkpoints.py` 只提供 manifest 读写与 artifact 解析，不创建 run context，也不内置任何 trainer 名称判断。

manifest 中的路径相对 `checkpoints/manifest.json` 所在目录：

```json
{
  "version": 1,
  "artifacts": {
    "latest_training_state": { "path": "latest.pt", "kind": "training_state", "format": "pt" },
    "best_policy": { "path": "latest.pt", "kind": "policy", "format": "pt" }
  }
}
```

`best_policy` 与 `latest_training_state` 是框架语义，不等同于文件名：

- `best_policy` 用于 play / export / benchmark。
- `latest_training_state` 用于 resume，应包含 optimizer、normalizer、replay buffer、global step 等继续训练所需状态。
- 同一物理文件可同时登记为两个 artifact（例如 FastSAC 的 state dict 既可 play 又可续训）。
- 内置 trainer 在训练结束**以及每次周期性保存**时都应把可播放/可恢复的 checkpoint 保存到标准 `checkpoints/` 目录并调用 `record_checkpoint_artifact(...)` 登记，使中断的 run 也能从 manifest 恢复。

公共 API：

```python
def record_checkpoint_artifact(run_dir, name, path, kind, checkpoint_format=None) -> Path: ...
def best_policy(metadata: RunMetadata, run_dir: Path) -> Path: ...
def resolve_checkpoint_path(checkpoint_or_run_dir, metadata=None) -> Path: ...
def resolve_resume_checkpoint_path(checkpoint_or_run_dir, metadata=None) -> Path: ...
```

manifest 缺失或没有可用 artifact 时，`checkpoints` 直接报错。

## 14. Train backend 自动选择

task 在进入 backend 选择前已经由 Hydra 成功组合。自动选择只需要考虑
"设备能力 + trainer 存在"，由 `motrix_rl.backend_runtime` 集中处理：

```python
def resolve_train_backend(env_name, method, requested_backend, device_supports) -> str:
    if requested_backend is not None:
        _validate_backend(method, requested_backend, device_supports)  # provider 存在 + 设备可用
        return requested_backend

    framework = frameworks.get_framework(method.rllib)
    candidates = [b for b in framework.supported_train_backends(method.algo) if _backend_available(b, device_supports)]
    if not candidates:
        raise ValueError(...)
    return _choose_preferred_backend(candidates)  # 偏好顺序 ("jax", "torch")，其余按名字排序
```

`_backend_available` 对 `jax` / `torch` 做设备能力判断，其他自定义 backend 默认可选。backend 专用 task option 会在 metadata 中明确指定 backend，不进入自动选择。

## 15. 内置 framework 矩阵

当前注册的内置 framework 与其 provider：

| rllib | train_backend | algo | checkpoint 格式 | 说明 |
|---|---|---|---|---|
| `skrl` | `jax` | `ppo` | `pickle` | SKRL PPO，JAX backend |
| `skrl` | `torch` | `ppo` | `pt` | SKRL PPO，PyTorch backend |
| `rslrl` | `torch` | `ppo` | `pt` | RSLRL PPO |
| `motrix` | `torch` | `fastsac` | `pt` | `motrix_rl_builtin` 提供的 Motrix 分布式（C51）FastSAC；配置选择同步或异步拓扑 |

`skrl` 通过 `SkrlPpoTrainerBase` 复用 train/play 主流程与 runtime config，JAX/Torch provider 只提供 backend 特异的 model/memory/agent 构建。`motrix.fastsac` 自带 replay buffer、观测归一化与同步/异步 Trainer；唯一 provider 根据 `FastSacCfg.asynchronous` 选择执行拓扑。

## 16. Multi-agent 扩展点

IPPO/MAPPO 不塞进单智能体 wrapper 的隐式约定，接入前需单独定义环境接口契约：

```python
MultiAgentStep:
    observations: dict[agent_id, obs]
    actions: dict[agent_id, action]
    rewards: dict[agent_id, reward]
    terminated: dict[agent_id, done]
    truncated: dict[agent_id, done]
    infos: dict[agent_id, info]
```

MAPPO 还需 centralized critic state（`global_state | dict[agent_id, critic_state]`）。IPPO/MAPPO 的 Trainer 依赖独立的 `MultiAgentWrapper`，单智能体与 multi-agent 路径在 Trainer 内部分流，不改现有单智能体 wrapper。

## 17. 兼容性与不变量

- `mxlab train task=cartpole/skrl.ppo` 选择 SKRL PPO，未指定训练后端时自动选择。
- `mxlab train task=cartpole/rslrl.ppo` 选择 RSLRL Torch PPO。
- `mxlab play` 不要求 task 配方，从 run metadata 反推 `rllib/train_backend/algo` 选择正确 inference path，可区分 PPO/SAC 等算法。
- `(env, rllib, train_backend, algo)` 可查到独立配置；`(rllib, train_backend, algo)` 可查到 `AgentProvider`。
- run metadata 记录 `env/rllib/train_backend/algo`；没有 metadata/manifest 的旧 run 不参与 play 自动发现。
- 新算法不影响现有 PPO 的默认超参数与训练路径。

## 18. 风险与约束

- **各算法 train backend 支持不一致**：不能假设某算法天然支持所有 train backend，须由 provider 与配置注册显式声明。
- **off-policy 对并行与 replay 设置敏感**：SAC/TD3 的最佳配置不同于大规模并行 PPO 默认值，每个任务需独立配置 `num_envs`、`memory_size`、`learning_starts`、`batch_size`、`gradient_steps` 等。
- **checkpoint 格式不能只靠扩展名推断**：SKRL Torch、RSLRL、FastSAC 都可能用 `.pt` 但内部结构不同，play/export/resume 必须依赖 metadata 与 manifest 中登记的 artifact 语义。
- **multi-agent 是接口级扩展**：涉及环境 step schema、wrapper、配置与 play loop，应在单智能体 off-policy 算法稳定后再推进。

## 19. 相关工作

- SKRL 多算法接入路线：GitLab issue #137。
- SAC 接入：GitLab issue #102。
- TD3 接入：GitLab issue #133。
- RPO 接入：GitLab issue #134。
- IPPO/MAPPO 接入：GitLab issue #135。
- AMP 接入：GitLab issue #136。
