# SuperMimic 外生条件项（ExogenousTerm）设计

## 摘要

本文定义 SuperMimic 数据与想象层的第三类具名 term——`ExogenousTerm`：不受动作影响、按自身规则推进、作为策略与世界模型条件的外生信号列。它与 `WorldStateTerm`（预测对象）和 `WorldContextTerm`（动力学条件）对称，负责「随转移捕获、按 schema 存储、在想象 rollout 中纯函数推进」。`CommandTerm` 继续独占真实环境的生命周期语义；两者通过任务配置关联，同一信号在两侧表现为「活对象」与「可复演数据列」两个投影。本设计为 `design -> plan -> coding` 流程的 settled 设计，实现尚未落地。

## 问题与边界

参考动作帧号是当前唯一的外生条件实例：[`ReferenceCommand`](../../motrix_rl_supermimic/src/motrix_rl_supermimic/reference_command.py) 继承 `CommandTerm` 管理真实环境侧生命周期，同时其 int64 `frame` 被 [`TransitionRecorder`](../../motrix_rl_supermimic/src/motrix_rl_supermimic/recorder.py) 固化为唯一条件列、被想象链路以硬编码 `+1` 推进。该形态把 SuperMimic 绑定在「恰好一个、每步 +1 的 int64 标量条件」上：

- 无参考动作的任务必须伪造一个 frame 才能满足环境校验；
- 固定目标、课程参数、变化的域随机化参数等非计数器型条件无法表达；
- 推进规则散落在 [learner](../../motrix_rl_supermimic/src/motrix_rl_supermimic/learner.py) 与 [actor_graph](../../motrix_rl_supermimic/src/motrix_rl_supermimic/actor_graph.py) 的多处 `frame + 1`，与数据列定义脱节。

`ExogenousTerm` 的职责边界：

- **归属层**：`motrix_rl_supermimic` 的 MBRL 数据/想象层；`motrix_env_core` 的 `CommandTerm` 契约不变。
- **只管两件事**：外生值如何进 replay（捕获与存储）与如何在想象中推进（纯张量规则）。
- **不管**：真实环境内的指令采样、课程与回合生命周期（归 `CommandTerm`）；replay 自身的记账字段 `ep_step`/`done`（历史连续性契约，非条件信号）。

## 类型契约

```python
@dataclass(frozen=True)
class ExogenousTerm(ABC):
    size: int            # 每环境宽度；0 表示该信号不存在
    dtype: torch.dtype

    @abstractmethod
    def advance(self, value: torch.Tensor) -> torch.Tensor:
        """想象推进规则：纯张量函数，可微，不触碰 ctx / simulator / live 状态。"""

    def fields(self, name: str) -> tuple[TransitionField, ...]:
        """该 term 对 transition schema 的字段贡献；size == 0 时为空。"""
```

内置推进规则：

- `Constant`：`advance` 为恒等映射。表达固定目标、不随回合重采样的域参数等。
- `Counter`：`advance(value) = value + 1`，int64。参考帧号是其唯一实例；额外暴露 `offsets(horizon)` 供 goal 序列预编码快路径使用（见下文）。

配置侧与 `world_model_state` / `world_model_context` 同构：`ExogenousCfg`（`@configclass`，`__call__(build) -> ExogenousTerm`）与 `ManagerExogenousCfg` 声明组。每个 term cfg 声明捕获来源：

- `source: "reference.frame"` —— 绑定某个 live `CommandTerm` 的具名字段：真实环境侧的值由该命令的生命周期钩子产生，`ExogenousTerm` 只读快照；
- `source: None` —— 常量初始化，`Constant` 的默认来源。

## 捕获与存储链路

schema 仍由单一工厂构造，collector 与 learner 两侧从同一份任务声明（`env.cfg.exogenous`）构建，共享 ring 的协商自动跟随，[transport](../../motrix_rl_supermimic/src/motrix_rl_supermimic/transport.py) 边界不引入新概念：

```python
transition_schema(*, state_dim, context_dim, action_dim, exogenous: tuple[TransitionField, ...])
```

- 固定字段 `s / s1 / a / a_prev / xi / r / ep_step / done` 不变；外生字段按 term 追加（名称即 term 名）。
- `TransitionRecorder`：`frame` 列替换为 `exogenous: dict[str, Tensor]`；`_capture_begin` 按 `source` 从 `ctx.commands` 收集；`snapshot()` 输出各 term 键。
- `Transitions`：`frame` 字段替换为 `exogenous` 映射，`to_dict / at_step / to` 对映射逐项变换。
- `ReplayBuffer`：分配逻辑按 schema 驱动不变；`sample_windows / sample_starts` 导出整个外生映射而非单列，`valid_frames` 谓词改为接收外生映射。

## 想象推进

想象链路中所有 `frame + 1` 替换为按 term 应用 `advance`；`ImaginationTerm` 的 `goals / valid_starts / reference_view` 与 `ReferenceView` 的契约从单帧号改为外生映射，数值 term 读取的 `ctx.commands[...]` 合成视图不变。

goal 序列批量预编码是唯一依赖推进规则仿射性的快路径：起点帧号张量加等差偏移一次求值整个 horizon 的 goal。该快路径的启用条件收紧为「goal 条件项为 `Counter`」（通过 `offsets(horizon)` 表达）；其余推进规则的 goal 条件退回逐步求值。禁止无声丢弃该快路径，也不为未知推进规则猜测仿射性。

## 环境配置与可选性

`SuperMimicEnvCfg` 新增 `exogenous: ManagerExogenousCfg`，默认空组（0 维）：

- 无 goal 观测组且无外生 term 的任务直接接入，不触发任何 reference 校验；
- `reference_command` 的选择与 int64 校验移入「声明了 goal 条件项」的分支——goal 观测组是否可选属独立演进（观测组放宽），本设计不绑定其结论。

## 不变量与测试策略

长期测试保护三个契约，不复制任务配置或断言字段命名细节：

1. **捕获往返不变量**：term 声明 → recorder 捕获 → snapshot → replay 存储 → 采样读出，全链路 dtype/shape/值一致（动态 schema 真正保护的风险）。
2. **推进正确性与可微性**：`Constant` 恒等、`Counter` 加一；想象 rollout 逐步推进与手工序列一致，`advance` 参与梯度计算不报错。
3. **零维默认可用**：无外生 term 的最小任务全链路（env → replay → learner 单步）可运行。

迁移验收（一次性，不入长期测试）：`_supermimic_toy` 的 frame 断言改为按 term 名访问，并新增无外生条件的 toy 变体贯穿冒烟。

## 相关文档

- [RL 多算法架构设计](./rl-multi-algorithm-architecture.md)：provider / trainer 层次。
- [训练随机性与 Seed 所有权](./randomness-and-seeding.md)：外生条件不含随机流；回合内的重采样语义归 `CommandTerm` 的生命周期钩子。
- [Manager 运行时](./manager/runtime.md)：`CommandTerm` 钩子契约与 `ctx.commands` 读取边界。
