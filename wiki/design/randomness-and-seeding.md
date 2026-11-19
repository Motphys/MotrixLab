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
