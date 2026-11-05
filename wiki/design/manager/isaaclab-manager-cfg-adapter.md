# IsaacLab ManagerCfg Adapter 设计

## 摘要

本设计定义一个面向 IsaacLab manager-based 任务配置的薄适配层，目标是尽量保留用户熟悉的 ManagerCfg 分组、`func + params`、term 命名、嵌套覆写和 selector 表达。Torch 批量自定义 MDP 函数在程序构建时被特化为私有 concrete `@kernel_data` Term，selector 被归一化为显式 `SimDataQuery`。适配层不直接执行任意 IsaacLab Torch callback，也不改变 [Manager Task API](./task-authoring.md) 和 [Manager Runtime](./runtime.md) 中的 Torch 执行器边界。

本文是目标设计，不表示下述 API 均已实现。兼容基线参考本地 IsaacLab 快照 `21bcb476b2`；该快照检查时落后 `origin/main` 202 个 commit，因此不声称覆盖所有上游版本。

## 1. 目标与边界

适配层应保持以下编译链路：

```text
IsaacLab-like ManagerCfg
  -> configuration normalization
  -> native TermCfg.__call__() / functional term generation
  -> concrete @kernel_data Term
  -> ManagerContext / SimDataQuery / ReadPlan
  -> Torch executor（torch.compile kernel，整批张量）
```

这意味着：

- 用户可以继续按 action、command、observation、reward、termination 和 event 组织配置。
- 配置层可接受 IsaacLab 常见的嵌套属性修改和以 `None` 删除 term。
- 每个被接受的 term 最终必须成为 concrete runtime term；native typed cfg 直接创建，functional cfg 由框架在构建期生成。
- [`@kernel_data`](./kernel-abi.md)、[`ManagerContext`](./kernel-abi.md) 和 [SimDataQuery 读侧编译](./sim-read-program.md) 的契约保持不变。

非目标：

- 不直接执行 IsaacLab 中以 `env` 为参数的通用 Python/Torch `func`。
- 不允许 term 隐式读取整个 environment 或在 kernel 外隐藏 simulator 依赖。
- 不静默忽略 noise、clip、history 等未支持选项。
- 不让 callable、`dict[str, Any]` 或完整 environment 进入 kernel_data ABI 或编译图。
- 不为表面兼容牺牲 typed cfg、compiler-visible dependency 或批量 Torch kernel。

## 2. 当前主要差异

| 配置维度 | IsaacLab 常见形态 | MotrixLab 当前形态 | Adapter 责任 |
| --- | --- | --- | --- |
| Manager group | `@configclass` 容器，主要依赖字段结构 | `dict` 或指定的 `Manager*Cfg` nominal type | 按结构枚举配置字段，不要求必须继承特定 group base |
| Term 定义 | `func` + `params` + 通用 term cfg | concrete typed `*TermCfg.__call__()` | Torch 批量函数自动特化为 concrete Term；foreign callback 通过注册 adapter 映射 |
| 禁用 term | 继承配置中常设为 `None` | term group 目前期望有效 cfg 实例 | normalization 时过滤 `None` |
| Reward 权重 | 常通过 `.weight` 嵌套覆写，0 通常表示关闭 | typed reward cfg，运行时统一加权 | 保留 `.weight` 覆写；`weight == 0` 在 program 构建前裁剪 |
| Scene selector | `SceneEntityCfg` 将 asset/body/joint/sensor 选择放在 term params 内 | `queries.data` 中集中声明 `SimDataQuery` | 解析 backend-neutral selector，产生并去重 `SimDataQuery` |
| Observation group | 任意命名 group，常用 `policy`/`critic`；group 可配 concat/noise/clip/history | 只接受 `policy` 和可选 `value` | 显式处理名称别名和支持子集，其余 fail fast |
| 配置字段 | IsaacLab 存量 group 可不带 MotrixLab 需要的类型标注 | native [`@configclass`](../configclass.md) 要求字段类型标注 | 首阶段接受带明确字段的结构化配置，不承诺无改动加载所有无标注上游类 |

## 3. 配置归一化契约

### 3.1 Manager group

Manager group 是有序的 term 声明集。normalizer 应支持现有 `dict`、native `Manager*Cfg` 以及可枚举字段的结构化 `@configclass` 实例，不以特定 group base 的 `isinstance()` 作为唯一入口条件。归一化后必须保留字段声明顺序，因为 action/observation layout 受顺序影响。

Term 字段允许为 `None`。`None` 表示用户在派生配置中显式删除该 term，应在 term type validation 和 `__call__()` 之前过滤。

### 3.2 Typed term 与 functional term

Native MotrixLab 配置继续直接使用 concrete typed cfg：

```python
@configclass
class RewardsCfg:
    track_lin_vel_xy_exp: mdp.TrackLinVelXYExpCfg = mdp.TrackLinVelXYExpCfg(
        weight=1.0,
        command_name="base_velocity",
        std=0.5,
    )
```

对应的 IsaacLab 形态是：

```python
track_lin_vel_xy_exp = RewTerm(
    func=mdp.track_lin_vel_xy_exp,
    weight=1.0,
    params={"command_name": "base_velocity", "std": 0.5},
)
```

后一种形态应成为 adapter 的一等输入，而不是只能通过逐函数 factory 注册使用的例外路径。用户经常需要自定义 MDP 函数，因此普通 Torch 批量函数不应额外要求 `CommandRef`、参数语义 decorator 或 adapter registry。

#### IsaacLab 的实际解析语义

IsaacLab manager 构建时会校验 `params` 是否匹配函数签名，并解析 `SceneEntityCfg` 中的 body/joint ID，但它不会预先把 `command_name` 转换成 `CommandTerm`。manager 在每次调用 term 时原样执行 `func(env, **params)`，MDP 函数内部再调用 `env.command_manager.get_command(command_name)` 取得 command tensor；只有显式调用 `get_term(name)` 时才返回 `CommandTerm` 对象。

因此 `command_name` 的类型只是 `str`，它的 command lookup 语义由 MDP 函数中的用法决定。adapter 不应依赖参数名称推断语义。

#### 自定义 MDP 函数

MotrixLab 用户应能继续编写普通、无状态函数：

```python
def track_lin_vel_xy_exp(
    ctx: ManagerContext,
    std: float,
    command_name: str,
) -> float:
    command = ctx.commands[command_name]
    velocity = ctx.sim["robot_root_lin_vel_b"]
    error_x = command[0] - velocity[0]
    error_y = command[1] - velocity[1]
    return math.exp(-(error_x * error_x + error_y * error_y) / (std * std))
```

`func + params` normalizer 使用函数签名绑定 keyword params，再按参数值的实际类型生成 concrete Term。对上述配置，等价的内部结果是：

```python
@kernel_data
class _GeneratedTrackLinVelXYExp:
    std: np.float32

    def compute(self, ctx: ManagerContext) -> float:
        return _compiled_track_lin_vel_xy_exp(
            ctx,
            self.std,
            "base_velocity",
        )
```

`"base_velocity"` 在生成的直线调用中是 compile-time literal，因此用户函数内的 `ctx.commands[command_name]` 仍能对异构 command map 完成静态类型解析。这是 MotrixLab 的编译特化，不改变用户对 `command_name` 的 IsaacLab 式理解。

参数按实际值分类，不按参数名称推断：

| `params` 值 | 构建期处理 |
| --- | --- |
| 数值标量 | 转换为类型确定的 Term field，例如 `np.float32` |
| `str` 或 enum | 嵌入生成调用作为 compile-time literal；它在 MDP 函数中可以用作 command/map key 或普通常量 |
| `np.ndarray` | 转换为具有明确 shared/per-env scope 的 typed field |
| `SceneEntityCfg` | 根据值的 selector 类型生成并去重 `SimDataQuery`，将解析后的 kernel 输入传给 MDP 计算 |
| nested `@kernel_data` | 作为 typed subtree flatten/lower |
| 任意 Python object、closure 或未支持容器 | 在构建期报出包含 group/term/param 路径的错误 |

生成的 Term class 是 compiler 实现细节，应按 `(term kind, func fingerprint, param schema, structural literals)` 缓存。数值参数只改变 Term 实例，不生成新类型；改变 string/enum/selector 属于结构配置变化，需要重建 program。调试输出和错误必须引用原始 `func` 与 `params` 路径，不向用户暴露 `_Generated*` 类名。

#### 自动生成与 foreign adapter 边界

Torch 批量纯函数使用上述自动生成路径，无需逐函数注册。原始 IsaacLab callback 若依赖 `ManagerBasedRLEnv` 的 scene/manager object model，不能原样进入 Torch 执行器的编译图；这类 foreign callable 需要注册到等价的批量 Torch function 或 typed `*TermCfg` factory。

自动生成首先覆盖无状态 Reward 和 Termination。Observation 还需要固定输出 `size`，可由 config 或函数 metadata 提供；Action、Command 和其他拥有 reset/update 生命周期的 stateful term 继续使用显式 typed Term class。

### 3.3 SceneEntityCfg selector

适配层可提供 backend-neutral `SceneEntityCfg` 作为配置级 selector。以下为目标 API，尚未实现：

```python
@configclass
class RewardsCfg:
    undesired_contacts: mdp.UndesiredContactsCfg | None = mdp.UndesiredContactsCfg(
        weight=-1.0,
        sensor_cfg=SceneEntityCfg(
            "contact_forces",
            body_names=".*THIGH",
        ),
        threshold=1.0,
    )
```

`SceneEntityCfg` 本身不决定返回 tensor 的语义。concrete term cfg 的字段类型决定它需要的 signal，normalizer 结合 selector 生成 `SimDataQuery`，对等价 query 去重，并把结果交给 `ReadPlan`。因此 selector 是对显式 simulator 依赖的简写，而不是 term 访问 environment 的后门。

### 3.4 继承与嵌套覆写

派生 task 应能保留 IsaacLab 用户熟悉的局部修改：

```python
self.rewards.action_rate_l2.weight = -0.005
self.rewards.undesired_contacts = None
```

第一行修改 typed cfg 中的纯数值字段；第二行在 normalization 时删除 term。对 reward，`weight == 0` 也应在 manager program 构建之前裁剪，避免为明确关闭的 term 创建 runtime object、query 和 kernel slot。

## 4. Observation group 兼容范围

首阶段支持 `policy` 和 `value` 两个 native group，并可将 IsaacLab 常见的 `critic` 显式映射为 `value`。若同时声明 `critic` 与 `value`，必须报冲突而不是静默覆盖。

对 observation group options，初始契约为：

- `concatenate_terms=True`：支持，与当前连续 observation array 一致。
- `history_length=0`：支持。
- 其他 concat/history 组合：在 normalization 阶段 fail fast。
- noise、clip 和 scale：只在存在统一、有测试的 lowering 语义后启用；在此之前不接受或忽略。

## 5. 分阶段范围

### P0：最小迁移表面

- Manager group 从 nominal inheritance 放宽为结构化 `@configclass` 解析。
- term 字段支持 `None`，reward `weight == 0` 在程序构建前裁剪。
- 提供 backend-neutral `SceneEntityCfg`，归一化为显式、可去重的 `SimDataQuery`。
- 保留 `.weight = ...` 与 `term = None` 等嵌套覆写方式。
- 对无状态 Reward/Termination 接受 Torch 批量 `func + params`，自动生成 concrete Term，不要求用户注册 factory。
- `params` 字典只存在于 host config，不进入 kernel_data ABI 或编译图。

### P1：Observation 常用选项

- 接受同名 group option，但只实现经过定义的子集。
- 支持 `policy`/`value` 以及显式 `critic -> value` 别名。
- 初期只支持 `concatenate_terms=True` 和 `history_length=0`。
- 后续按统一 compiler/runtime 语义实现 noise、clip、scale 和 history。
- Curriculum、interval event 等需要新运行时能力的特性单独设计，不伪装成纯配置兼容。

### P2：Foreign callback 与扩展 term

- 为依赖 IsaacLab env/Torch object model 的 foreign callable 提供显式 adapter registry，映射到批量 Torch function 或 native `*TermCfg` factory。
- 对 functional Observation 补充固定 `size` 契约，再按同一生成模型支持。
- Action、Command 与 stateful term 只在能保留其生命周期契约时扩展，不将其伪装为无状态 callback。
- 未注册 foreign callable、不兼容 params 和未支持 option 在 config normalization 阶段立即报错。
- compat config 在 compiler lowering 前转换为 native cfg 或 generated concrete Term，不进入 kernel ABI 或 runtime state。

## 6. 验收准则

适配层完成时至少应验证：

1. 一个典型 IsaacLab locomotion ManagerCfg 可保留 `func + params`、原有分组、term 名称与大部分派生配置覆写。
2. 用户自定义 Torch 批量 Reward/Termination 函数无需创建 Term class、注册 factory 或声明 `CommandRef`。
3. `term = None`、`weight == 0`、`critic -> value` 和 selector 去重有独立配置层测试。
4. 生成的 KernelData/ABI 中不包含 callable、`params` 字典或 environment reference；compiler plan 记录原函数 fingerprint 与生成 term schema。
5. 数值参数、string literal、enum、ndarray 和 selector 有独立 lowering 测试，不支持的值会报出包含 group/term/param 路径的错误。
6. `ctx.commands[command_name]` 能在生成 wrapper 传入 string literal 时进入 torch.compile 编译图；无效 key 在 warm-up 阶段报错。
7. 归一化前后 term 顺序、observation layout、reward 权重和 termination 语义有 parity 测试。
