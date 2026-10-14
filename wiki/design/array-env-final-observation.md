# ArrayEnv 终态观测结果契约

## 摘要

ArrayEnv 在时间截断自动 reset 前捕获当前 transition 的终态观测，通过 `ArrayEnvState.final_obs` 发布稀疏行快照。该结果不绑定 RL provider；ManagerEnv 实现捕获，FastSAC adapter 只负责传递和消费，不修改环境结果。本设计仅覆盖 NumPy 前端与 truncated 行，不扩展 Torch 前端或纯 terminated 行。

## 结果结构与生命周期

`FinalObs` 为普通 dataclass，包含 `env_ids: np.ndarray` 和 `obs: NpObs`。观测行顺序与 env_ids 一致，policy/value 保持原始观测结构；没有独立 value 时仍为 None，消费者使用 `value_or_policy`。行数组为独立快照，可跨 step 保留；不需要重复复制 policy 作为 value。消费者不得修改共享结果。

`ArrayEnvState.final_obs` 为可选字段。初始化为 None，每次 step 在推进环境前通过 replace 清空当前结果；此前返回的 state 持有的终态快照不因此清空。step 在 compute_transition、episode_steps 和 truncation 更新后、任何 auto-reset 写入前调用 `_capture_final_obs(env_ids)`。仅有 truncated 行时调用；没有截断或实现不支持捕获时为 None。读取无消费副作用。

ArrayEnv 的默认捕获 hook 返回 None，明确表示当前实现不提供快照；不尝试通过重复调用可能含副作用的通用 compute_observation 自动兼容所有子类。ManagerEnv 使用已有 compiled observe kernel 实现。sim-reset-only 行不进入 final_obs。

## Manager 捕获语义

捕获读取 compute_transition 已刷新的 simulator 数据，并发生在 command on_transition 之后、command reset 之前。它与同一 pre-reset 状态的正常 observe 使用相同输入；不回退 command，不额外刷新 simulator。

Observation term 应仅读取上下文并写观测输出，不推进 command 或历史；现有噪声 term 可以消费 ctx.rand。终态捕获直接使用共享 per-lane RNG，不备份或恢复；额外 observe 消耗噪声随机数，因此可改变同 seed 下后续 reset 和正常观测的随机轨迹，但不要求保持捕获前后的随机序列一致。终态捕获会额外执行一次整批 observe，因此不能用观察函数维护每步只应更新一次的状态。捕获使用独立预绑定 output slot，不写正常 observation 双缓冲；NumPy 高级索引已生成独立行数组，无需额外 copy。

当前 compiled observe 不支持行子集，因此少量截断仍有整批执行成本。性能优化必须先分别测量 kernel 与 compact row gather 成本，不为了潜在用途新增编译入口。

## FastSAC 边界

FastSAC adapter 的 step 返回 `FastSacStepResult`，将本步 obs、critic_obs、reward、terminated、truncated、final_obs 与 info 放在同一个结果中；NumPy adapter 直接传递 state 的 final_obs 引用，不在 wrapper 上缓存旁路状态。reset 仍返回 obs/critic_obs 二元组。collector 将 FinalObs 转换为传输所需的 env_ids、policy 与 value_or_policy。异步传输、背压和 replay 存储的生命周期不属于本次 Env 结构整理，见 [FastSAC 异步训练器设计的终态观测 payload](./fastsac-async-heterogeneous-trainer.md#终态观测-payload)。

## 验收

测试覆盖初始化、无截断、截断前后观测差异、多行选择、policy-only/value 观测、快照跨步寿命、command 时序和只读重复访问。性能测量区分无截断路径、少量/整批截断的 capture 成本；使用预热后的重复采样，不包含编译时间。
