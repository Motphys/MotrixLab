# Deploy Runtime Command Input 分层设计

## 摘要

本文定义当前 Go2 training/deploy 共用的 command input 最小抽象。`motrix_env_core.input` 提供
`InputDevice`、`KeyboardDevice`、`GamePadDevice`、batch-first `PlanarVelocityCommand`、泛型
`CommandBinding[CommandT]`，以及平面速度的内置 bindings。binding 通过
`read_command(batch_size=...) -> CommandT` 输出高层 command；只有 device-backed binding 才持有 device，
constant binding 直接绑定配置。带 standing probability 的 random binding 属于四足训练任务，不进入 core。

Go2 deploy task 直接消费公共 `PlanarVelocityCommand`，不定义 `Go2WalkCommand`。`InputDevice` 不定义通用
`read()`；keyboard、gamepad 等 subtype 各自暴露 event polling、边沿事件和持续状态语义。command 不需要公共
ABC。第一版也不引入 capabilities、统一 lifecycle、插件 registry、command schema 或其他任务的 command 类型。

## 1. 范围与术语

command input 是 `RobotState` 之外、用于生成策略 observation 的高层目标。当前范围只有 Go2 velocity tracking
所需的平面线速度与 yaw 角速度。公共 command 同时服务 vectorized training environment 和单实例 deploy
runtime，因此 batch 维度是 command contract 的一部分。

高层 command 与 `RobotCommand` 是不同方向的契约：

| 契约 | 数据方向 | 语义 |
| --- | --- | --- |
| 高层 command | input mechanism → task/controller | 策略要完成的目标，例如 `PlanarVelocityCommand` |
| `RobotCommand` | controller → robot backend | 关节位置、速度、力矩和 gains 等低层执行命令 |

本文只定义 input mechanism 到高层 command 的路径。observation 拼接、policy inference 和 action processing 仍由
[Motrix Deploy 框架设计](./motrix-deploy.md) 中的 task-owned `PolicyProcessor`、`PolicyController` 与 `Policy` 负责，机器人 I/O 属于 `ControlSession` 与 `RobotInterface`。

## 2. 最小概念模型

```text
motrix_env_core:
KeyboardDevice ──> KeyboardPlanarVelocityBinding ──> PlanarVelocityCommand
GamePadDevice  ──> GamePadPlanarVelocityBinding  ──> PlanarVelocityCommand
constant config ──> ConstantPlanarVelocityBinding ──> PlanarVelocityCommand

motrix_envs locomotion training:
range + RNG     ──> RandomPlanarVelocityBinding   ──> PlanarVelocityCommand
```

| 概念 | 负责 | 不负责 |
| --- | --- | --- |
| `InputDevice` | 标识输入设备，并承载 subtype-specific API | 通用 `read()`、command 的维度、单位和范围 |
| command dataclass | 表达一类强类型高层目标 | 键位、手柄轴和设备读取 |
| `CommandBinding` | 把 device state、常量或随机采样配置转换成 `CommandT` | observation 和 `RobotCommand` |

必须保持以下不变量：

1. `InputDevice` 不得 import 或构造 command。
2. command dataclass 不得包含 key code、gamepad axis 或设备实现。
3. device 与 command 的组合只出现在 concrete device-backed binding 中；binding 本身不要求 device。
4. runtime 只调用 `CommandBinding.read_command(batch_size=1)`，不包含设备或任务分支。
5. Go2 使用的通用平面速度 command 与 device/constant bindings 放在 `motrix_env_core`；带训练采样策略的 binding
   放在 `motrix_envs` 对应任务模块。

## 3. InputDevice 与设备专属接口

### 3.1 名义基类

`InputDevice` 属于 `motrix_env_core.input`，可同时用于 train/play/view/deploy。它只建立共同的类型边界，不定义
所有设备都必须模拟的 I/O 方法：

```python
class InputDevice(ABC):
    """Nominal base for input devices."""
```

第一版不在 `InputDevice` 上定义：

- `read()`：keyboard 与 gamepad 没有一个有意义的公共返回类型；
- `capabilities`：device subtype 的专属查询接口已经表达当前所需能力；
- `open()` / `reset()`：不是所有设备都需要显式初始化或 episode reset；
- `close()`：资源释放属于具体 provider 或创建它的 application；
- timeout、timestamp、sequence：当前 keyboard vertical slice 不需要；
- registry/plugin metadata：只有内置实现时直接构造更清楚。

具体实现如果持有 listener、file descriptor 或 SDK handle，可以自行实现 context manager 或 provider-specific
`close()`，但不扩大 `InputDevice` ABC。

### 3.2 KeyboardDevice

```python
class KeyboardDevice(InputDevice, ABC):
    @abstractmethod
    def poll(self) -> None: ...

    @abstractmethod
    def is_key_down(self, key: str) -> bool: ...

    @abstractmethod
    def is_key_up(self, key: str) -> bool: ...

    @abstractmethod
    def is_pressing(self, key: str) -> bool: ...
```

`poll()` 为当前 control tick 冻结一个 keyboard event frame。之后的查询都是非消费式的，在下一次 `poll()` 前
重复调用必须返回相同结果：

| 查询 | 语义 |
| --- | --- |
| `is_key_down(key)` | 本 event frame 内发生过未按下 → 按下的边沿 |
| `is_key_up(key)` | 本 event frame 内发生过按下 → 释放的边沿 |
| `is_pressing(key)` | frame 结束时 key 是否仍处于按下状态 |

例如一个按下、保持、释放的序列为：

| event frame | `is_key_down()` | `is_key_up()` | `is_pressing()` |
| --- | --- | --- | --- |
| 收到 press | `True` | `False` | `True` |
| 无新事件 | `False` | `False` | `True` |
| 收到 release | `False` | `True` | `False` |

如果一个 key 在两次 `poll()` 之间完成 press 和 release，则 down/up 都为 `True`，`is_pressing()` 为 `False`。
操作系统 key repeat 不重复产生 down 边沿。MuJoCo deployment 的 concrete device 由 GLFW viewer callback 缓存
transition；window 失焦时必须释放全部 held keys，device 生命周期由 backend viewer 负责。

### 3.3 GamePadDevice

```python
class GamePadDevice(InputDevice, ABC):
    @abstractmethod
    def poll(self) -> None: ...

    @abstractmethod
    def axis_value(self, axis: str) -> float: ...

    @abstractmethod
    def is_button_down(self, button: str) -> bool: ...

    @abstractmethod
    def is_button_up(self, button: str) -> bool: ...

    @abstractmethod
    def is_button_pressing(self, button: str) -> bool: ...
```

`GamePadDevice.poll()` 建立同样稳定的 event frame。button 的 down/up/pressing 语义与 keyboard 一致，
`axis_value()` 返回该 frame 内经过 device 校验的有限归一化轴状态，范围为 `[-1, 1]`。device provider 负责处理
SDK 原始值、非法值和断连状态；binding 信任 device contract，不重复校验 device 输出。deadzone、axis inversion 和
command scale 属于 binding。第一版先定义 ABC；具体 provider 等实际接入手柄时再选择依赖和轴名称。

## 4. Command 类型

### 4.1 不定义公共 Command ABC

command 是普通 immutable dataclass，不继承公共 base。`CommandT` 只用于 `CommandBinding`、`ControlContext` 和
`DeployTask` 之间的静态类型关联。

是否放入 core 由语义复用范围决定，而不是由运行时继承层次决定。当前设计只增加 Go2 所需的公共平面速度类型。

### 4.2 PlanarVelocityCommand

平面速度跟踪具有独立于具体机器人型号的稳定语义，因此由 `motrix_env_core.input` 内置。command 始终使用
batch-first float32 array：

```python
@dataclass(frozen=True)
class PlanarVelocityCommand:
    values: np.ndarray  # float32, shape: (batch_size, 3)

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=np.float32, copy=True)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != 3:
            raise ValueError("values must have shape (batch_size, 3), with batch_size > 0")
        if not np.all(np.isfinite(values)):
            raise ValueError("values must contain only finite numbers")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)

    @property
    def batch_size(self) -> int:
        return self.values.shape[0]

    @property
    def linear_velocity_x_mps(self) -> np.ndarray:
        return self.values[:, 0]

    @property
    def linear_velocity_y_mps(self) -> np.ndarray:
        return self.values[:, 1]

    @property
    def yaw_rate_rad_s(self) -> np.ndarray:
        return self.values[:, 2]
```

最后一维的固定顺序和单位为：

| index | property | 单位 |
| --- | --- | --- |
| `0` | `linear_velocity_x_mps` | m/s |
| `1` | `linear_velocity_y_mps` | m/s |
| `2` | `yaw_rate_rad_s` | rad/s |

使用单个矩阵而不是三个 component arrays，可以从结构上保证三个分量共享同一个 batch size，并直接匹配训练环境
现有的 commands layout。训练侧构造 `values.shape == (num_envs, 3)`，deploy binding 构造
`values.shape == (1, 3)`。不接受 `(3,)`，也不在 command 内执行隐式 broadcast；consumer 必须校验预期 batch
size。constructor 统一复制为 read-only float32 并拒绝非有限值。task 自己的训练范围仍由 versioned task config
定义，不放进 command 或 device。

训练侧 task-specific `RandomPlanarVelocityBinding.read_command(batch_size=num_reset)` 为本次 reset 的 env 子集采样，
并通过 `command.values` 获得 `(num_reset, 3)` 矩阵。deploy task 则在确认 `batch_size == 1` 后使用
`command.values[0]` 构造单实例 observation。

Go2 deploy task 直接消费该类型，不再定义同构的 `Go2WalkCommand`。第一版不设计其他 command 类型，也不增加
command registry、plugin version 或公共 schema；versioned `DeployTask` 和 artifact task config 固定其语义与范围。

## 5. CommandBinding

### 5.1 最小 ABC

`CommandBinding` 与公共 commands 一起定义在 `motrix_env_core.input`：

```python
CommandT = TypeVar("CommandT")


class CommandBinding(ABC, Generic[CommandT]):
    @abstractmethod
    def read_command(self, *, batch_size: int = 1) -> CommandT:
        """Produce one command batch of the requested size."""
```

`CommandBinding` 的构造契约不包含 device。concrete binding 可以持有 `InputDevice`、constant value、采样范围或
RNG。`batch_size` 在调用时传入，是因为 vectorized training 的 partial reset 每次需要采样的 env 数量不同；
deploy runtime 使用默认值 `1`。所有 binding 必须拒绝非正 batch size，并返回 batch size 完全匹配的 command，
不能依赖 consumer 隐式 broadcast。

第一版不声明 required channels、capabilities、reset 或 lifecycle；配置合法性由 concrete binding constructor
检查，command 合法性由 dataclass 或消费它的 task 检查。

### 5.2 Core 内置 planar-velocity bindings

`motrix_env_core.input` 提供以下常用组合：

- `KeyboardPlanarVelocityBinding`：`W/S` → x、`A/D` → y、`Q/E` → yaw，并使用 application factory
  传入的 lower/upper 端点；
- `GamePadPlanarVelocityBinding`：把指定 axes 经过 deadzone、inversion 和 scale 映射到平面速度；
- `ConstantPlanarVelocityBinding`：直接把配置中的三维常量复制成所需 batch。

keyboard 示例：

```python
class KeyboardPlanarVelocityBinding(CommandBinding[PlanarVelocityCommand]):
    def __init__(
        self,
        device: KeyboardDevice,
        *,
        command_lower: Sequence[float],
        command_upper: Sequence[float],
    ) -> None:
        self._device = device
        self._command_lower = validated_float32_vector(command_lower, size=3)
        self._command_upper = validated_float32_vector(command_upper, size=3)

    def read_command(self, *, batch_size: int = 1) -> PlanarVelocityCommand:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._device.poll()
        direction = np.asarray(
            (
                int(self._device.is_pressing("w")) - int(self._device.is_pressing("s")),
                int(self._device.is_pressing("a")) - int(self._device.is_pressing("d")),
                int(self._device.is_pressing("q")) - int(self._device.is_pressing("e")),
            ),
            dtype=np.int8,
        )
        value = np.where(
            direction > 0,
            self._command_upper,
            np.where(direction < 0, self._command_lower, 0.0),
        )
        return PlanarVelocityCommand(values=np.repeat(value[None, :], batch_size, axis=0))
```

`KeyboardPlanarVelocityBinding` 和 `GamePadPlanarVelocityBinding` 每次 `read_command()` 必须且只能调用一次
device `poll()`，确保所有按键或轴查询属于同一个 event frame；constant binding 不需要 poll。
Go2 速度使用 held-state `is_pressing()`；需要单次触发的操作可以使用 down/up 边沿。内置 binding 不 import
任何 deploy task。Go2 task package 只消费它们，不再实现自己的平面速度 binding。

### 5.3 Task-specific random training binding

`RandomPlanarVelocityBinding` 定义在 `motrix_envs.locomotion.quadruped.velocity_command`，不属于
`motrix_env_core.input`。它复用公共 `CommandBinding` 与 `PlanarVelocityCommand`，同时持有四足训练配置中的
velocity ranges、`standing_probability` 和 RNG：

```python
class RandomPlanarVelocityBinding(CommandBinding[PlanarVelocityCommand]):
    def __init__(
        self,
        lower: Sequence[float],
        upper: Sequence[float],
        *,
        rng: np.random.Generator,
        standing_probability: float = 0.0,
    ) -> None:
        self._lower = validated_float32_vector(lower, size=3)
        self._upper = validated_float32_vector(upper, size=3)
        if np.any(self._lower > self._upper):
            raise ValueError("lower must not exceed upper")
        self._standing_probability = validated_probability(standing_probability)
        self._rng = rng

    def read_command(self, *, batch_size: int = 1) -> PlanarVelocityCommand:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        values = self._rng.uniform(self._lower, self._upper, size=(batch_size, 3))
        standing = self._rng.random(batch_size) < self._standing_probability
        values[standing] = 0.0
        return PlanarVelocityCommand(values=values)
```

`standing_probability` 表达训练样本分布，不是 `PlanarVelocityCommand` 的固有属性，也不是所有平面速度任务都
共享的 mapping 规则，因此留在 task package。RNG 由 training application 注入，binding 不定义 `reset(seed)`。
每次 `read_command()` 产生一个新 batch；训练环境只在 command resampling point 调用它，不能在每个
physics/control step 无条件重新采样。partial reset 时传入 `batch_size=num_reset`，因此不需要把固定 batch size
绑定在 constructor 中。

## 6. Runtime 组合

执行层级为 Runtime → ControlSession → DeployTask → phase controllers。Policy（`Policy`）只表示 observation → action；DeployTask 是完整任务 Controller，将机器人状态与控制输入映射为 `RobotCommand`，并编排具体任务需要的 Prepare/Policy/Stop 阶段。`PolicyController` 通过 `PolicyProcessor` 适配模型，不反向引用完整 task。

`ControlSession.period_s` 是唯一执行周期来源，Runtime 据此调度；`ControlContext.dt_s` 携带该周期，`elapsed_time_s` 携带连续执行时间。Controller 在阶段开始时记录时间原点以形成局部时钟，不另设执行周期。Task/artifact 的参考采样周期只定义参考数据语义，不承担调度。

`ControlContext` 携带 control-loop metadata 与具体 command：

```python
@dataclass(frozen=True)
class ControlContext(Generic[CommandT]):
    step: int
    elapsed_time_s: float
    command: CommandT | None
    dt_s: float
```

每个 tick 的 command input 路径只有两步：

```text
1. command = command_binding.read_command(batch_size=1) if command_binding is not None else None
2. context = ControlContext(step=step, elapsed_time_s=elapsed_time_s, dt_s=session.period_s, command=command)
```

ControlSession 的 `command_binding` 默认 `None`；有 binding 时读取指令，没有时直接将
`ControlContext.command` 设为 `None`。G1 自主播放不创建占位 binding；Go2 仍需外部速度
binding，并由 task 语义校验拒绝缺失指令，不回退到零速度。Task 不提供 binding factory。
CLI 从 task 的速度范围组装输入；无速度输入的自主任务校验 `None` 并拒绝 command 配置，
不引入额外 capability 或 registry。
ControlSession 负责读取可选 binding、构造 context、backend open/enable、状态校验、命令写入、清理和统计。
`DeployTask.step(state, context)` 消费实测状态和输入，返回 `ControllerStep`；
其 `PolicyController` 通过 `PolicyProcessor` 调用 command 校验、observation 构造、模型推理与动作转换，
不持有 backend handle。`G1WbtDeployTask` 自行管理 ready、policy 与 damping，不让 ControlSession 或 Runtime 解释阶段名；DeploymentRuntime 负责会话调度、异常处理、清理与宿主生命周期；物理周期配置、推进、时间测量、viewer 与仿真世界资源仅属于 SimulationRuntime。
binding/device 异常统一作为 `input_error` 结束 rollout；`KeyboardInterrupt` 映射为 `interrupted`。
更细的 timeout、disconnect、stale input 错误等真实设备需要区分时再加入。

`ControllerStep.policy_tick` 只标记实际模型推理/actor 决策，不计 ready 或 damping 命令。根 `ControllerStep.error` 对应 `controller_error`；`policy_error` 只用于推理失败。Session 提供 `ControlContext.dt_s` 与 `elapsed_time_s`；Controller 不持有独立执行周期。

`ControlSession.result()` 只返回通用 RolloutResult：success、exit reason、模型完成步数、wall time、overrun、trace、error 与 latency。硬件结果不携带仿真指标。SimulationRuntime 返回 SimulationRolloutResult，增加 policy_simulation_time_s（模型命令下实际推进的物理时间，含故障前部分推进，不含 ready/damping）及其与 wall time 的比值 real_time_factor。该指标不是整个世界时钟；通用循环只通过窄的内部宿主 interval hook 执行宿主工作，不识别 physics。

仿真 realtime/fixed-step scheduler 由 `runtime.realtime` 选择，`null` 跟随 `runtime.viewer`；硬件内部始终实时调度，不从 device capabilities 推断。

## 7. 配置与创建

`KeyboardDeviceProvider` 与 `GamePadDeviceProvider` 定义在 `motrix_deploy.robot.interface`，是可选输入设备
capability protocol，不是 runtime 插件 factory；返回设备的生命周期由 provider 管理。
deploy 应用在 runtime factory 返回 prepared host 后，从 runtime 的可选 `KeyboardDeviceProvider` 取得 GLFW keyboard device 并创建 binding，
在外部加载 Policy，再以 `task = create_task(spec, runtime.robot.spec, policy, steps=50)` 注入任务。
应用创建 `session = ControlSession(robot=runtime.robot, controller=task, command_binding=binding, period_s=0.02, state_timeout_s=0.1)`，
Task 满足构造时必需的 Controller 契约；调用 `runtime.bind_session(session)`，
通过 `runtime.session` 与 `session.controller` 访问会话及 task，在 `with runtime` 中执行 `runtime.run()`。
Task 负责自己的阶段编排，通用 DeployTask 不强制阶段顺序；正常停止由 task 决定，故障立即走 backend fallback。
现有 `motrix_deploy.tasks` entry points 直接提供 task class；`load_task_type` 选择 task 与 spec codec，
`create_task(spec, robot, policy, steps=None)` 构造完整任务，`default_duration_s(spec)` 无需构造 task 即可读取预算。
PolicyController 持有 `policy_io` 与 Policy，不持有完整 task，避免循环依赖。模型预算在 task 构造时传入，
不传给 run。Runtime factory 不承担任务或控制器构造。应用
不建立 input registry；headless CLI 使用 `command.velocity` 配置 constant command。Go2 task 从 artifact 读取
`command_lower`、`command_upper` 和与 `[vx, vy, yaw_rate]` 对齐的三维 `command_scale`，向 factory 暴露逐元素
`range * scale` 后的映射端点；scale 属于 artifact 的 task contract，不在 runtime recipe 重复配置。内置交互式
keyboard path 要求 backend 提供具有焦点的 viewer window；headless CLI 使用 constant binding，程序化调用也可传入其他 binding。

training command sampling 使用：

```yaml
command_binding:
  type: random_planar_velocity
  lower: [-1.0, -0.5, -1.0]
  upper: [1.0, 0.5, 1.0]
  standing_probability: 0.1
```

deploy application 使用 core 的 keyboard、constant 或 bounded gamepad binding；viewer device 和 window 的生命周期由仿真 runtime
持有，Unitree gamepad 由硬件端口提供，binding 不负责关闭 device。
training task 直接创建自己的 `RandomPlanarVelocityBinding`，RNG 由训练入口按现有 seed 机制注入。等第二个外部
package 确实需要独立注册 device/binding 时，再引入 entry point 或 registry。

## 8. 包与依赖边界

```text
motrix_env_core/src/motrix_env_core/
└── input/
    ├── __init__.py             # stable public re-exports
    ├── device.py               # device ABCs
    ├── command.py              # common command dataclasses
    └── bindings.py             # CommandBinding、built-in bindings

motrix_deploy/src/motrix_deploy/
└── runtime/                    # ControlContext；调用 CommandBinding.read_command()

motrix_deploy_tasks/src/motrix_deploy_tasks/
└── tasks/
    └── go2_walk.py             # Go2WalkTaskSpec 与 task 同置；直接消费 PlanarVelocityCommand

motrix_envs/src/motrix_envs/locomotion/quadruped/
└── velocity_command.py         # task-specific RandomPlanarVelocityBinding、standing sampling
```

依赖方向为：

```text
motrix_env_core <- motrix_deploy <- motrix_deploy_tasks
motrix_env_core <- motrix_envs
```

`motrix_env_core.input` 不 import deploy runtime 或具体 task。`motrix_deploy` 复用 input contract；
`motrix_deploy_tasks.tasks.go2_walk` 中的 task 直接消费公共 command，不重复实现 device 或 core binding。
其 `Go2WalkTaskSpec` 持有 typed command range/scale，由 task 类的 `spec_type` 关联；manifest 保持
`{name, config}`，具体 spec 解码与 task class discovery 的边界见
[Manifest 内容](./motrix-deploy.md#82-manifest-内容)。`motrix_envs` 复用公共
contract，并在四足训练模块中实现带训练分布语义的 random binding。

## 9. 第一版测试边界

### 9.1 Device contract

- keyboard press、保持、release 分别产生规定的 down/up/pressing 组合；
- 同一 event frame 内重复查询不会消费事件，下一次 `poll()` 会清除旧边沿但保留 held state；
- 同一 poll interval 内完成 press/release 时 down/up 同时为 true，key repeat 不重复产生 down；
- `GamePadDevice` fake 对 button 提供相同边沿语义，并冻结当前 frame 的 axis value。

### 9.2 Built-in binding

- keyboard 的 `W/S`、`A/D`、`Q/E` 正反键和同时按下语义；
- keyboard scale、显式 batch replication 与 `PlanarVelocityCommand` 数值；
- fake gamepad axis 的 deadzone、inversion 和 scale；
- constant 与 keyboard binding 可以产生数值相同的 command；
- 非正 batch size、非有限 scale和错误 vector length 时失败。

### 9.3 Task-specific training binding

- random binding 的 range、standing probability、seeded reproducibility 与逐 env 独立采样；
- partial reset 请求的 batch size 与返回 shape 一致；
- 非正 batch size、非法 range/probability 时失败；
- `motrix_env_core` 的测试不依赖 standing sampling，相关测试留在四足训练 task package。

### 9.4 Command batch contract

- `(num_envs, 3)` 构造训练 batch，三个命名 property 均为 `(num_envs,)`；
- `(1, 3)` 构造 deploy command，`batch_size == 1`；
- `(3,)`、`(batch_size, 2)`、空 batch 和非有限值均拒绝；
- constructor 复制输入并转为 read-only float32，外部输入数组后续修改不影响 command；
- training consumer 不依赖 deploy binding，deploy consumer 不隐式 squeeze 未校验的 batch。

### 9.5 Runtime integration

- fake `CommandBinding` 可以驱动 control loop，不需要真实 device；
- 替换 keyboard、gamepad 或 constant binding 不修改 `Controller`、`DeployTask` 或 backend；
- training 使用 random binding 不依赖 `InputDevice` 或 deploy runtime；
- Go2 task 直接消费公共 `PlanarVelocityCommand`；
- binding 异常产生 `input_error`，用户中断产生 `interrupted`。

## 10. 延后设计

以下能力不进入第一版公共接口：

- device capabilities 与统一 channel schema；
- 通用 `InputDevice.read()` 与 `InputDevice.open/reset/close` lifecycle；
- timestamp、sequence、freshness、timeout 与 disconnect 状态；
- 包含事件次数与顺序的完整 event stream；
- device/binding plugin registry；
- 公共 command ABC、独立 command schema 与 schema version；
- 通用 declarative mapping DSL；
- ROS 2 和多设备组合；Unitree remote 已通过 gamepad provider 接入。

新增这些能力前必须先有当前最小接口无法表达的真实 consumer，并保持现有 `poll()`、边沿/持续状态查询与
`read_command(batch_size=...)` 语义可兼容演进。
