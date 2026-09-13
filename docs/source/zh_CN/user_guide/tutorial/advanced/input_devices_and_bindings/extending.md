# 扩展输入组件

本页说明如何沿用{doc}`InputDevice、Binding 与任务指令架构 <../input_devices_and_bindings>`接入新输入源。
扩展应从行为实际变化的层开始：优先复用已有 device contract 和 command 语义，只在原始查询协议或任务目标确实
变化时增加新类型。

## 如何选择扩展点

先根据变化发生在哪一层选择最小扩展：

| 需求                          | 应扩展的类型                                          | 复用关系                          |
| ----------------------------- | ----------------------------------------------------- | --------------------------------- |
| 接入新的键盘窗口或 listener   | 实现 `KeyboardDevice`                                 | 复用现有 keyboard binding 和 task |
| 接入新的手柄 SDK              | 实现 `GamePadDevice`                                  | 复用现有 gamepad binding 和 task  |
| 修改键位、轴或缩放规则        | 新增 `CommandBinding[现有 CommandT]`                  | 复用现有 device 和 task           |
| 输入源已直接提供 task command | 直接实现 `CommandBinding`                             | 复用现有 task                     |
| 新硬件具有独立的查询协议      | 新增 `InputDevice` subtype，再实现对应 binding        | task 继续消费输出 command         |
| 新任务需要不同目标语义        | 新增 command dataclass、binding，并让 task 消费该类型 | 需要新的 typed task contract      |

如果一个 ROS topic、网络服务或配置文件已经提供 `[vx, vy, yaw_rate]`，可以直接实现
`CommandBinding[PlanarVelocityCommand]`，将输入源映射到已有的任务指令。

## 新增一个具体 Device Provider

最常见的扩展是为已有 ABC 实现新的 provider。下面的 callback-backed keyboard 示例展示 event frame 所需状态；
窗口系统或 SDK callback 调用 `on_press()` / `on_release()`，control loop 通过 `poll()` 冻结结果：

```python
from motrix_env_core.input import KeyboardDevice


class CallbackKeyboardDevice(KeyboardDevice):
    def __init__(self) -> None:
        self._pressing: set[str] = set()
        self._pending_down: set[str] = set()
        self._pending_up: set[str] = set()
        self._frame_pressing: frozenset[str] = frozenset()
        self._frame_down: frozenset[str] = frozenset()
        self._frame_up: frozenset[str] = frozenset()

    def on_press(self, key: str) -> None:
        key = key.lower()
        if key not in self._pressing:
            self._pressing.add(key)
            self._pending_down.add(key)

    def on_release(self, key: str) -> None:
        key = key.lower()
        if key in self._pressing:
            self._pressing.remove(key)
            self._pending_up.add(key)

    def poll(self) -> None:
        self._frame_down = frozenset(self._pending_down)
        self._frame_up = frozenset(self._pending_up)
        self._frame_pressing = frozenset(self._pressing)
        self._pending_down.clear()
        self._pending_up.clear()

    def is_key_down(self, key: str) -> bool:
        return key.lower() in self._frame_down

    def is_key_up(self, key: str) -> bool:
        return key.lower() in self._frame_up

    def is_pressing(self, key: str) -> bool:
        return key.lower() in self._frame_pressing
```

实现后可以直接复用公共 mapping：

```python
from motrix_env_core.input import KeyboardPlanarVelocityBinding

device = CallbackKeyboardDevice()
binding = KeyboardPlanarVelocityBinding(
    device,
    command_lower=[-0.5, -0.4, -1.0],
    command_upper=[1.0, 0.4, 1.0],
)
```

provider 还应定义断连、窗口失焦和资源关闭策略。创建 device 的对象拥有其 lifecycle，负责释放 listener、
file descriptor 或 SDK handle，也负责关闭共享 device。

## 新增一种 Device 类型和 Binding

当原始查询语义需要键盘或手柄 contract 之外的数据时，新增 `InputDevice` subtype。以下 SpaceMouse 示例定义
自己的 frame 查询，再将所需分量映射到公共平面速度 command：

```python
from abc import abstractmethod

import numpy as np

from motrix_env_core.input import CommandBinding, InputDevice, PlanarVelocityCommand


class SpaceMouseDevice(InputDevice):
    @abstractmethod
    def poll(self) -> None: ...

    @abstractmethod
    def translation(self) -> np.ndarray: ...

    @abstractmethod
    def rotation(self) -> np.ndarray: ...


class SpaceMousePlanarVelocityBinding(CommandBinding[PlanarVelocityCommand]):
    def __init__(self, device: SpaceMouseDevice, scale: np.ndarray) -> None:
        self._device = device
        self._scale = np.asarray(scale, dtype=np.float32)

    def read_command(self, *, batch_size: int = 1) -> PlanarVelocityCommand:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self._device.poll()
        translation = self._device.translation()
        rotation = self._device.rotation()
        value = (
            np.asarray(
                [translation[0], translation[1], rotation[2]],
                dtype=np.float32,
            )
            * self._scale
        )
        return PlanarVelocityCommand(np.repeat(value[None, :], batch_size, axis=0))
```

生产实现还应在 constructor 验证 scale shape/有限性，并由具体 device provider 保证 translation/rotation 的 shape、
单位、归一化范围和 frame 内稳定性。若这个 subtype 和 binding 只服务一个外部应用，可以留在该应用包中；确认存在
跨任务复用需求后再加入 `motrix_env_core.input` 的公共导出。

## 新增任务专用 Command

当任务目标使用平面速度之外的语义时，定义新的 immutable dataclass，并依次完成：

1. 选择 command 所属 package；以语义复用范围为准。
2. 明确 batch-first shape、各分量顺序、坐标系和单位。
3. 在构造时复制并验证 dtype、shape 和有限性，避免输入被外部修改。
4. 实现 `CommandBinding[NewCommand]`；binding 可以读取 device，也可以读取配置、网络或 RNG。
5. 让 training task 或 `DeployTask[NewCommand]` 校验任务范围并消费该类型。
6. 为 command、binding 和 task integration 分别添加测试。

task range、键位和 SDK channel 分别由 task contract、binding 和 device 管理。
