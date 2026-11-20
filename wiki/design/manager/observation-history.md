# History Observation 与单帧传输契约

## 摘要

本文定义 ManagerEnv 的 group 级 observation history：常规 step 只返回 stacked obs。静态描述和固定布局允许 provider 从尾部提取实际单帧，但这只是可选优化，不新增环境单帧读取方法，也不要求改变现有传输或 replay。环境负责观测定义、真实轨迹的历史推进和 episode 边界；RL 层决定存储与采样方式。当前 FastSAC 仍按完整窗口传输和存储 observation，本次不实现单帧 ring/replay 重建。

本文定义 core 的 history 契约。`policy` 必须存在，`value` 可选，任务可声明其他命名 group；各组独立配置整帧 history，统一采用旧到新的时间顺序和 time-major flatten。provider 显式选择消费的组，不由 core 推断网络输入。

本文限定架构、接口与行为不变量，具体实施计划单独维护。基础生命周期见 [Manager Runtime](./runtime.md)，终态快照见 [ArrayEnv 终态观测结果契约](../array-env-final-observation.md)。本文针对当前 Torch 张量前端；相关旧文档中遗留的 NumPy 描述不作为本文接口依据。

## 1. 目标与非目标

### 目标

- 任务声明 observation history，业务 term 不维护历史队列。
- `env.step()` 的 `obs` 可直接交给在线 actor；value group 不强制跟随 policy 的 history 配置。
- 静态描述明确窗口布局，使 provider 在需要时可从 stacked obs 提取实际尾帧，无需环境另行发布单帧。
- 明确真实环境窗口的长度、顺序、布局和 episode 填充语义，供消费者解释完整 observation。
- 正确处理局部 reset、sim-reset-only、truncation bootstrap、借用 buffer 和异步所有权。
- 未启用 history 的路径不引入 H 倍存储或无意义复制。

### 非目标

- 第一阶段不支持 term 级不同历史长度、任意 stride/dilation、可变长度窗口或自定义 flatten 布局。
- 不把 replay、ring、world model 或物理 state schema 搬进环境 core。
- 不统一所有 provider 的训练算法、normalizer 或 replay 实现。
- 不将 observation history 与 RNN hidden state、BPTT、world-model horizon 或未来 reference goal 混为一谈。

## 2. 职责划分

Manager 的 `ObsTerm` 声明单帧宽度，执行器计算各 term 后沿特征维拼接，生命周期层按 group 配置组装 history。`EnvObs` 按组名提供最终 observation，`FinalObs` 提供 truncated 行的 reset 前 owned 全窗快照。

## 3. 配置与静态描述

### 3.1 Group 级配置

`ManagerObservationGroupCfg` 声明 `history_length: int = 1`：

- H >= 1；H=1 表示只包含当前帧，不增加历史。
- H 包含当前帧，不是 H 帧过去数据再加当前帧。
- 每个命名 group 各自配置，缺省均为 1。
- `history_length` 是 group 元数据，不是 observation term；现有 `to_dict()` 的字段遍历须区分元数据与 term 声明。
- term `size`、resolver 数值 kernel 的输出 slice 始终表示单帧宽度，不乘 H。

示意配置：

```python
@configclass
class PolicyObsCfg(ManagerObservationGroupCfg):
    history_length: int = 10
    # joint_pos、joint_vel 等单帧 term 声明

@configclass
class ValueObsCfg(ManagerObservationGroupCfg):
    history_length: int = 1
    # privileged observation term 声明
```

### 3.2 唯一布局

单帧 group 按 term 声明顺序拼成宽度 D。内部逻辑窗口为 `[B, H, D]`，旧帧在前、当前帧在后。公共最终观测为 `[B, H*D]`，按时间优先 flatten：

```text
[term_a(t-H+1), term_b(t-H+1), ..., term_a(t), term_b(t)]
```

不采用 IsaacLab 默认的“各 term 历史分别 flatten，再拼 terms”。整帧布局保持 provider 消费二维 flat observation 的边界。第一阶段不增加 flatten 开关；需要三维输入的模型根据静态描述显式 reshape。

### 3.3 环境公开描述

core 定义具名、不可变的 observation 描述数据，概念接口为：

```text
ObservationGroupSpec
    frame_space: Box(shape=(D,))
    history_length: H
    # 最终 space=(H*D,)，由 frame_space 与 H 推导

ObservationSpec
    groups: dict[str, ObservationGroupSpec]
    # policy 必须存在，value 可选，允许任务自定义组

env.observation_spec: ObservationSpec
```

`spec.groups[name]`（或 `spec[name]`）描述任意命名组；`spec.policy`、`spec.value` 与 `spec.value_or_policy` 是策略/价值组的便捷视图。布局和首帧 repeat-fill 为此接口的固定契约，不再增加只存在一种取值的配置选项。`spec.policy.space` 与可选 `spec.value.space` 描述最终 obs，不能仍声明单帧 D。单帧 space 从 spec 获取，不让 provider 通过模型维度反推 H。

`observation_spec` 是非可选的环境唯一观测描述。DirectEnv 显式构造 H=1 的命名组 spec；ManagerEnv 根据 group 配置提供实际 H。环境不再另设 observation_space、policy_observation_space、value_observation_space、has_value_observation，也不保留 ObsSpace 包装。消费者统一使用 spec.policy.space、spec.value_or_policy.space、spec.value is not None；ArrayEnv 根据 spec 分配结果。只有第三方 RL adapter 按第三方协议提供 observation_space，并从 spec 派生。自定义历史环境须显式提供正确 spec，不从最终维度猜 H。

## 4. 常规结果、单帧读取与所有权

### 4.1 常规结果只返回最终 observation

`ArrayEnvState.obs: EnvObs` 保持原有入口，各命名组为包含对应 history 的最终 `[B, H_g*D_g]` 张量，匹配 `env.observation_spec` 推导的最终 space。不为 ring 给 `ArrayEnvState` 增加 `obs_frame`，也不增加 `env.transition()` 或 `env.observation_transition()`。普通训练、play 和直接消费最终 obs 的 provider 不承担 transport 接口成本。

`EnvObs.value` 在缺少 `value` 组时返回 None，`value_or_policy` 回退 policy；没有独立 value 时不得另配一个隐式 value history。

### 4.2 从既有结果提取单帧

环境仅增加静态 `env.observation_spec` 描述，不新增 `observation_frame()` 或 `final_observation_frame()` 方法。固定整帧 time-major 布局下，最后 D 个元素就是参与 stack 的实际当前帧，RL adapter 如需单帧，可按各组 spec 提取：

```python
frame = result.obs.policy[..., -spec.policy.frame_space.shape[0]:]
# value 独立使用 spec.value_or_policy 的单帧宽度
final_frame = result.final_obs.obs.policy[..., -spec.policy.frame_space.shape[0]:]
```

这里 final 提取仅在 `result.final_obs` 存在时进行。切片不重新计算、不重新抽噪、不推进 history；正常与终态分别对应各自实际追加的那一帧。H=1 时整个 observation 就是单帧。

这一能力依赖固定布局，以及环境不在 stack 后执行混合整个窗口的不可逆变换。provider running normalization 属于模型路径，原始单帧在其之前提取。DirectEnv 的显式 spec 将整个单帧 observation 描述为 H=1，provider 不另设无 spec fallback，也不能猜测隐含 history。Manager 内部单帧 buffer 仅为实现细节，不向 provider 暴露。

### 4.3 生命周期与快照

正常 obs 是环境拥有的只读借用结果，从其切出的 frame 共享同一 storage，有效期至少到下一次会改写 observation 的环境调用开始；不承诺跨 step/reset 保持数值。若 collector 选择单帧优化，需在调用 step 前将起点 frame 保存到自有 staging，step 后再读取下一帧与 final frame；环境不为 transport 额外保留上一动作、reward 或起点 frame。异步设备拷贝完成前，源 buffer 不得被复用，不能依赖当前双缓冲偶然延长的寿命。

`result.final_obs` 仍是最终 stacked observation 的 owned 稀疏快照，其尾帧切片引用同一份 owned storage，与原快照使用同一行序，可跨 step 保留；无需另外保存或发布 final 单帧缓存。

不扩展本阶段的捕获范围：仍按现有约定捕获 truncated 行，不要求纯 terminated 行额外观测。需要终态 bootstrap 的 provider 必须在缺失必要 final 数据时明确报错或使用其已声明的非-bootstrap 模式，不能拿 reset 后观测替代终点。

## 5. History 生命周期

### 初始化与普通 step

1. 初始化 reset 完成后计算单帧；每行窗口用首帧 repeat-fill，发布 stacked obs。
2. 每个 control step 完成 transition、episode bookkeeping 和 auto-reset 后，计算一次正常单帧。
3. 非 episode-reset 行左移一帧并追加；episode-reset 行用新首帧填满整个窗口。
4. 发布最终 flat obs；physics substep、warmup、普通读取不增加历史帧数。

Observation term 始终只计算单帧。初始化和 step 末尾统一调用 `compute_observation(state)`，由 ObservationManager 提交一帧：episode-reset 行 repeat-fill，其余行 append。不增加 step 专用 compute hook，也不使用 pending/stepflag 判断调用来源。普通读取直接访问已发布的 state.obs；显式调用 compute 是一次发布，会重新求值并推进窗口，不承诺重复调用缓存。

局部显式 episode reset 后使用 `reset_observation(state, reset_ids)`，只为指定行初始化窗口，不推进其他行；调用方显式表达范围，不由 manager 猜测。终态 preview 独立求值且不提交历史。外部直接编辑 simulator/task 状态后，须通过 episode reset 与显式目标行发布开始新轨迹，不悄悄替换既有帧。

### 局部 episode reset 与 sim-reset-only

- episode reset 标记目标行窗口待初始化；下一次发布用该 episode 的首帧填满，不混入旧 episode。
- 直接 reset 不推进 control 时间；在下次 observation 发布前不把旧 observation 当新结果暴露。
- sim-reset-only 不构成 episode 边界，保留已有历史；step 内触发的 sim-reset 在本控制步最终追加 reset 后实际观测。独立调用公共 reset 的 sim-only 行不会额外发布；读取 state.obs 仍是原值，下一次普通 compute/step 才追加新实际观测，不对这些行调用 episode 初始化入口。
- 无关行窗口不受影响；多次查询或终态捕获不增加其他行历史。

### Truncation 终态窗口

令 `W_t` 为决策时已经发布的窗口，`x_final` 为执行动作后、reset 前的终点单帧：

```text
final.obs = flatten(append(W_t, x_final))
final.obs[..., -D:] = x_final

正常返回的 reset 行：
obs = flatten(repeat(x_reset, H))
obs[..., -D:] = x_reset
```

终态 append 是非提交的 preview，不能覆盖正常窗口。该 preview 和正常发布分别有自己的单帧计算；只保证各自 stacked obs 的尾帧来自该次单帧计算，不要求终态噪声与 reset 后噪声相同。RNG 消耗沿用终态捕获既有契约，本设计不额外承诺 RNG 备份/恢复。

## 6. Core 与 RL 的实现边界

core 提供无 simulator 依赖的窗口组装张量操作，以及 Manager 的有状态窗口管理：

- repeat-fill、按 reset mask append、flatten/reshape 的固定规则。
- 正常运行允许预分配与原地写，维护正常 stacked obs 与 final 快照的所有权。
- 另有可组合的函数式 append，使 imagined rollout 保留 autograd；不将 detach 或环境原地 buffer 强加给 learner。
- 静态 spec 是唯一长度来源；不在多个 provider 配置里重复维护 H。

### ObservationManager 所有权

`ObservationManager` 持有 observation groups/spec、单帧与全窗 buffer、history、episode reset mask 和独立 preview scratch。构造只依赖 groups 与 batch 数，构造完成即拥有全部观测存储；state 引用 manager.observation，不通过 bind 补依赖，也不接收 reward/terminated、整个 env 或具体 simulator。

生命周期入口分别表达操作：`begin_step` 切换写入槽，`reset` 标记 episode 行，`publish_reset` 发布首帧，`commit` 提交 control 端点，`preview` 构造 owned 终态窗口。需要求值的入口在调用时显式接收 observe callable，数值 observe 仅接收 `EnvObs` 输出。reward/terminated 由环境以 `EvaluationOutputs` 传给 evaluate kernel，与 observation buffer 完全分离。无初始化空值跳过或二次求值协议。

ManagerEnv 负责这些入口的调用时机，不再持有第二份 spec、buffer、history 或 reset mask。group term 的解析与编译仍由既有 resolver/executor 执行，不另建数值调度器；内部 buffer 容器不扩展成通用框架。

RL 负责 trajectory 顺序、存储与采样，不由 core 实现通用 replay；不预先添加通用 provider registry 或兼容抽象。

## 7. 可选的单帧传输边界

core 的固定 time-major 布局允许 provider 提取实际尾帧，但不规定 ring payload 或 replay 存储格式。当前 FastSAC 保留完整窗口传输和存储，不在本次迁移为单帧路径；环境 history 支持不等于单帧 replay 优化已经实现。

若 provider 后续选择单帧优化，传输组装、轨迹连续性、episode 首帧 repeat-fill、缺失前缀及 truncation endpoint 的窗口重建均由 RL 层负责。重建需保留实际采样的噪声与 reset 前终态语义，不能拿 reset 后观测替代终点，也不能为提取尾帧重新求值或推进环境 history。这些边界不构成本次对具体 replay 实现或存储节省的承诺。

## 8. Provider 与部署边界

provider 可继续直接消费最终二维 obs，不必因环境启用 history 就改成序列训练。单帧 replay 是独立的 provider 优化，不是启用环境 history 的前提。无法支持的组合在组装边界明确拒绝。

导出/部署必须记录单帧特征顺序、H、time-major 布局、repeat-fill、预处理和 reset 时机。环境不提前应用 provider 的 running normalization；在线和 learner 对最终输入应用同一模型预处理。normalizer 统计应明确按新单帧更新还是按窗口采样更新，不能在迁移中无意把重叠 H 帧重复计权或二次归一化。具体策略由 provider 决定，不能混入 history core。

## 9. 验收不变量

core 契约以低成本张量/假后端测试验证；可选 provider 存储优化不属于本次验收范围：

1. H=1 输出和原单帧行为等价，允许 alias；policy/value 不同 H 的 space、shape、内容正确。
2. 多 term 使用整帧 time-major 顺序；实际 frame 等于最终窗口最后一帧，同一次噪声计算。
3. 生命周期每 control step 只调用一次普通 compute；读取 state.obs、warmup、终态 preview 不推进窗口，额外显式 compute 按发布契约推进。
4. 局部 episode reset repeat-fill，其他行不变；sim-reset-only 保留前缀。
5. final stacked obs 是 reset 前结果且 owned；正常 stacked obs 对应 reset 后结果；终态窗口内容正确。
6. 函数式 append 的梯度能够回传至前序 observation，不被环境 buffer/detach 切断。
7. 消费者在 borrowed buffer 失效前完成保留或传输；final 快照可跨 step 保留。
