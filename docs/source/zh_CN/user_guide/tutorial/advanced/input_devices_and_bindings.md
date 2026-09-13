# 指令输入架构

MotrixLab 将“如何读取输入”和“策略需要什么目标”拆成三个独立概念：`InputDevice` 表达原始设备状态，
`CommandBinding` 将设备状态或其他数据源转换为强类型高层指令，任务再消费该指令并构造策略观察。
这套分层让键盘、手柄、常量和训练随机采样通过同一条任务与控制链路驱动策略。
它也是仿真与真机共享的控制抽象：同一种任务指令和策略可以通过 `RobotInterface` 驱动不同 backend。

本文介绍三层接口及其职责关系。具体 API 分为{doc}`内置实现 <input_devices_and_bindings/built_in>`和
{doc}`扩展开发 <input_devices_and_bindings/extending>`两个子文档。所有公共类型都可以从
`motrix_env_core.input` 导入。

```{toctree}
:hidden:
:maxdepth: 1

input_devices_and_bindings/built_in
input_devices_and_bindings/extending
```

## 架构与职责边界

```{figure} /_static/images/input-device-binding-architecture.png
:alt: InputDevice、CommandBinding、任务指令以及仿真和真机 backend 的共享控制链路
:width: 100%
:align: center

InputDevice、Binding 与 backend-neutral task command 控制链路
```

从 `CommandBinding` 到 `RobotInterface` 的控制链路不依赖具体 backend。同一个 binding、任务指令、
`PolicyContext`、task/policy contract 和 `RobotCommand` 语义可以同时用于仿真与真机；切换控制目标时，只需
替换具体 device provider 和 `RobotInterface` 实现。真机 backend 实现 `RobotInterface` 后，即可复用相同的
指令链路、task 和 policy。

仿真和真机可以选择各自合适的具体 device。例如，仿真使用 GLFW keyboard，真机使用遥控器或网络输入；两端
复用同一个 binding 和任务指令语义。

`CommandBinding` 既可以读取键盘和手柄等 `InputDevice`，也可以直接读取固定值、配置或 RNG。只要输出相同的
`CommandT`，下游任务就以同一种方式消费指令。

| 概念                       | 职责                                                   |
| -------------------------- | ------------------------------------------------------ |
| `InputDevice`              | 表达一类原始输入设备及其查询协议                       |
| `CommandBinding[CommandT]` | 轮询输入源并映射、缩放或采样，输出一个指令 batch       |
| 任务专用指令               | 以强类型值对象表达策略目标及其维度、顺序、坐标系和单位 |
| Task                       | 校验指令，并将其写入 observation 或其他任务逻辑        |

这里的高层任务指令与 `RobotCommand` 方向不同：

-   `PlanarVelocityCommand` 等高层指令描述策略要完成的目标，例如机体坐标系下的速度。
-   `RobotCommand` 是策略 action 经 task 处理后发给 backend 的低层关节位置、速度、力矩和增益。

## Binding 如何进入 Task

运行时通过 `CommandBinding` 获取强类型任务指令：

```python
command = command_binding.read_command(batch_size=1)
context = PolicyContext(
    step=step,
    elapsed_time_s=elapsed_time_s,
    command=command,
)
task.validate_command(context.command)
observation = task.build_observation(state, context)
```

例如 MuJoCo deployment 从 backend 的 `KeyboardDeviceProvider` 取得 `MujocoKeyboardDevice`，再创建
`KeyboardPlanarVelocityBinding`。训练环境则可以为同一个 `PlanarVelocityCommand` 使用 random binding。两条路径最终
进入同一个 typed task contract。

程序化运行可以直接使用 constant binding。例如固定速度写作：

```python
from motrix_env_core.input import ConstantPlanarVelocityBinding

binding = ConstantPlanarVelocityBinding([0.5, 0.0, 0.0])
command = binding.read_command(batch_size=4)
assert command.values.shape == (4, 3)
```

## 继续阅读

-   {doc}`内置实现 <input_devices_and_bindings/built_in>`：了解现有任务指令、`InputDevice` contract 和 binding。
-   {doc}`扩展开发 <input_devices_and_bindings/extending>`：选择扩展点，并实现新的 provider、设备类型或任务指令。
