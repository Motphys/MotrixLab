# 训练随机性与 Seed 所有权

## 摘要

本文定义 MotrixLab Manager 环境的 Torch 原生随机入口、seed 传播与可复现性边界。Manager 数值 term 通过显式 `ctx.rand` 使用 Torch 默认随机流；该入口不提供环境实例、用途或 per-lane 隔离。训练入口负责根 seed，ManagerEnv 构造不重置默认 RNG，也不承诺固定 seed 可以逐位复现异步训练。

## RandomSource 与显式随机入口

[RandomSource](../../motrix_env_core/src/motrix_env_core/manager/random.py) 是无状态的 Torch 随机能力入口，没有构造参数，不持有独立 `torch.Generator` 或自定义 PRNG state。`uniform(shape, dtype=..., device=...)` 直接调用 `torch.rand`；`uniform_like(tensor)` 直接调用 `torch.rand_like`，沿用输入张量的 shape、dtype 与 device，提供 `[0, 1)` 抽样。

`ManagerContext.rand` 让 term 显式表达随机依赖。`RandomSource` 是空的 frozen dataclass，按普通嵌套数据递归校验，没有需要校验的数值叶子，不按字段名设置豁免。它并不意味着实例拥有独立随机流：同一进程、同一设备上的默认 Torch RNG 消费者仍共享状态，创建多个 `RandomSource()` 不产生隔离。例如观测噪声：

```python
out += (2 * ctx.rand.uniform_like(out) - 1) * amplitude
```

其他未显式传入独立 generator 的 Torch 随机操作也消费相应设备的默认流。因此模型初始化、environment sampling、观测噪声、collector exploration 与 learner sampling 是否互相影响，取决于其进程、设备与实际 RNG 使用方式，不能从 `ctx.rand` 入口推导用途隔离。

## Seed 的职责

训练入口负责设置根 seed；[ManagerEnv](../../motrix_env_core/src/motrix_env_core/manager/env.py) 不在构造阶段调用 `torch.manual_seed`。环境构造的 `seed` 参数不为 `ctx.rand` 建立独立流；仅构造相同 seed 的环境不能单独保证默认随机流被重置。`seed=None` 也不会触发 RandomSource 自行熵播种，抽样沿用当前默认流状态。

构造期 `BuildContext.seed` 保留现有语义：环境未指定 seed 时传入 `1`，供 action 等构造期 generator 使用。它与运行期 `ctx.rand` 的默认流不同，不代表默认 Torch RNG 的隐式播种。

## SuperMimic 的组件随机所有权

SuperMimic 在 provider 层管理训练根 seed，不改变上述 Manager 契约。`resolve_seed(None)` 通过系统熵解析一次根 seed；`derive_seed(root, purpose, worker_id)` 使用固定域标识和 SHA-256 派生 Torch 可用的 63-bit 非负整数，不依赖对象构造顺序或 Python 内置 `hash()`。实现见 [randomness](../../motrix_rl_supermimic/src/motrix_rl_supermimic/randomness.py)。

- learner 在训练入口用 `derive_seed(root, "learner")` 初始化默认 Torch RNG。
- collector 使用 `derive_seed(root, "environment", worker_id=0)` 作为环境构造 seed。默认 Torch RNG 的播种仍须由 collector 入口负责，不能依赖 ManagerEnv 构造函数；环境 term 之间仍共享默认流。
- collector exploration 持有独立 CPU `torch.Generator`，用 `derive_seed(root, "exploration", worker_id=0)` 播种。policy action 移到 CPU 后再生成探索噪声，环境 reset/观测随机消耗不推进该 generator。
- world model、policy（含 reference action factory）、critic 分别在 `model/world_model`、`model/policy`、`model/critic` 初始化作用域内构造。`torch.random.fork_rng` 保存并恢复 CPU 和所有可见 CUDA 默认 RNG；模型 device transfer 在作用域外。目标设备是 CUDA 不意味着 Linear 参数初始化消耗 CUDA RNG。

隔离只发生在已有组件边界，不在数值 kernel 或 CUDA Graph 中切换 Generator；环境各 term 与 learner 内部采样/增强仍各自共享所在进程、设备的默认流。它不承诺每行 reset 独立流，也不承诺 learner 内所有用途互相隔离。实现入口见 [trainer](../../motrix_rl_supermimic/src/motrix_rl_supermimic/trainer.py) 和 [collector](../../motrix_rl_supermimic/src/motrix_rl_supermimic/collector.py)。

### Warm resume 边界

SuperMimic learner checkpoint 保存并恢复自身的 CPU/CUDA 默认 RNG、resolved root seed、模型、优化器和训练控制状态；collector 与环境重新初始化，exploration Generator 从保存的 root 派生 seed 重新开始，不恢复已推进的 collector Generator 状态。replay、simulator、policy history 和 reset 采样统计不保存，因此恢复不具备轨迹精确性。resume/play 在模型或 reference factory 构造前读取 checkpoint root；没有 root 字段的 checkpoint 使用本次启动解析的 root。实现见 [learner](../../motrix_rl_supermimic/src/motrix_rl_supermimic/learner.py)。

若要续接远端 collector 随机流，需要独立的控制快照通道和明确的异步采样位置；不能由 learner 伪造远端状态，也不逐 transition 将 RNG 状态写入 ring。当前实现保持 warm resume 边界。

## 终态 Preview 与随机消耗

Task evaluate、command/reset、正常 observe 与终态 observation preview 均通过当前 context 的随机入口消费默认流；observe 不替换独立 stream。终态 preview 不备份或恢复 RNG，额外抽样可以改变同设备上后续正常观测噪声以及 command/reset 的随机样本。不承诺 preview 与正常观测噪声相同或完全独立。

Observation term 不得借观察过程推进 command、历史或其他每步只应提交一次的任务状态。即使 term 不写任务状态，额外噪声抽样仍有共享 RNG 消耗副作用。见[终态观测契约](./array-env-final-observation.md)与[观测历史](./manager/observation-history.md)。

## Torch 编译与验证范围

`RandomSource` 没有 `@torch.compiler.disable` eager 边界，其方法直接表达 Torch 随机算子。Torch 2.7 数值实验表明此入口可被 `torch.compile(fullgraph=True)` trace；默认流避免了显式 generator 对 tracing 的限制，但并不构成整个 ManagerEnv fullgraph 编译成功的证明。

- 数值实验或微基准不替代完整环境、真实设备生命周期与训练的运行验收。
- CUDA graph capture 兼容性与整环境吞吐仍须独立验证，不能由 fullgraph trace 可用直接推导。
- seed 重复性应在相同 Torch 版本、设备、执行方式、shape 和随机调用顺序下验证；不承诺 eager/compiled 或跨设备产生相同随机序列。

执行器约定见 [Torch Manager 执行器](./manager/torch-executor.md)。

## 可复现性边界

- 增加一次共享默认流上的随机调用，可以改变后续抽样结果；不同环境实例与 task/observation 用途不独立。
- partial reset 只修改选中行的环境状态，抽样可按 compact reset 行进行，但不提供每个 environment lane 独立推进 RNG 的保证。改变 reset 行数或批量 shape 可能改变后续抽样。
- 不建立每个 term 一条流或模型、collector、learner 等组件的通用 seed 派生框架。
- 不同进程分别拥有 RNG 状态，同一进程的 CPU 与各 CUDA 设备也有各自的默认 RNG；共享入口不表示跨进程或跨设备共享同一条序列。

需要区分随机源受控、用途隔离与执行确定性。根 seed 可以控制实际播种的随机源，但本入口不提供用途隔离；异步 collector/learner 还可能因 transition 消费批次、权重加载时点、CUDA 算子与并行调度产生不同轨迹。改变每步发布/分块发布的节奏，也可能改变训练组件的 RNG 消耗顺序。

因此不能仅凭相同 seed 承诺位级复现，也不能未做对照就将性能回退归为随机波动。应保留源码 revision、完整配置、执行设备、checkpoint 和评测协议，并通过单变量实验定位差异。
