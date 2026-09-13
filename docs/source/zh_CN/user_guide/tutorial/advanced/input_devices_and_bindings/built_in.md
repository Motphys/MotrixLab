# 内置输入组件

本页介绍 MotrixLab 已提供的任务指令、设备 contract 和 binding。它们遵循{doc}`总览页
<../input_devices_and_bindings>`中的分层关系，可在训练、仿真 deployment 和实现了 `RobotInterface` 的真机 backend
之间复用相同的任务指令语义。

## 内置任务指令

### `PlanarVelocityCommand`

当前公共 command 是 `PlanarVelocityCommand`，表示机体坐标系下的平面速度目标。其 `values` 是只读
`float32` 数组，shape 固定为 `(batch_size, 3)`：

| 索引 | 属性                    | 含义       | 单位  |
| ---: | ----------------------- | ---------- | ----- |
|  `0` | `linear_velocity_x_mps` | 前后线速度 | m/s   |
|  `1` | `linear_velocity_y_mps` | 左右线速度 | m/s   |
|  `2` | `yaw_rate_rad_s`        | yaw 角速度 | rad/s |

训练环境通常构造 `(num_envs, 3)`；单机器人 deployment 使用 `(1, 3)`。构造函数会复制输入、转换为
`float32`、拒绝空 batch 或非有限值，并将数组设为只读。速度范围由具体 task contract 管理。

以 `go2_walk/v1` 为例，artifact 保存 `command_lower`、`command_upper` 和三维 `command_scale`。task 将范围两端
逐元素乘以 scale 后向 binding 提供可用端点，并在消费 command 时再次检查 batch size 和最终范围。

## 内置 InputDevice

### `InputDevice`

`InputDevice` 是输入设备的公共标记基类。具体 subtype 根据设备特征定义查询协议：键盘表达离散事件和持续按下
状态，手柄还表达连续轴。创建具体 device 的 provider 或 application 管理设备资源和生命周期。

### `KeyboardDevice`

`KeyboardDevice` 定义 event-frame 查询接口：

```python
class KeyboardDevice(InputDevice, ABC):
    def poll(self) -> None: ...
    def is_key_down(self, key: str) -> bool: ...
    def is_key_up(self, key: str) -> bool: ...
    def is_pressing(self, key: str) -> bool: ...
```

`poll()` 冻结当前 control tick 的事件帧；在下一次 `poll()` 之前，所有查询都是非消费式的并返回稳定结果。

| 查询               | 当前 event frame 的语义  |
| ------------------ | ------------------------ |
| `is_key_down(key)` | 本帧发生按下边沿         |
| `is_key_up(key)`   | 本帧发生释放边沿         |
| `is_pressing(key)` | 本帧结束时仍处于按下状态 |

需要持续速度控制时使用 `is_pressing()`；菜单切换或一次性动作通常使用 down/up 边沿。操作系统 key repeat 保持
held state，首次按下产生一次 down 边沿。

当前具体实现 `MujocoKeyboardDevice` 位于 `motrix_deploy_mujoco`。它读取 deployment GLFW window 的 callback，
只接收该窗口的键盘事件；窗口失焦会立即释放所有 held keys，Esc 或关闭窗口会中断 rollout。其生命周期由
MuJoCo backend viewer 管理，而不是由 binding 管理。

### `GamePadDevice`

`GamePadDevice` 提供相同的 `poll()` 和 button down/up/pressing 语义，并增加：

```python
value = device.axis_value("left_y")
```

具体 provider 把 SDK 原始轴值校验并归一化到有限的 `[-1, 1]`。deadzone、轴反向和 task command scale
属于 binding；binding 按照 device contract 使用 provider 的输出。不同手柄 SDK 的 provider 通过该 ABC 接入。

## 内置 Binding

`CommandBinding` 的公共接口只有一个方法：

```python
CommandT = TypeVar("CommandT")


class CommandBinding(ABC, Generic[CommandT]):
    def read_command(self, *, batch_size: int = 1) -> CommandT: ...
```

`batch_size` 是调用参数，因为 vectorized training 在 partial reset 时需要动态数量的 command，而 deployment
固定请求一个 command。一个 device-backed binding 每次 `read_command()` 应只调用一次 `poll()`，确保该次映射
读取的所有 key、button 和 axis 属于同一个 event frame。

### 公共 Binding

| Binding                         | 输入源           | 输出                    | 映射行为                                    |
| ------------------------------- | ---------------- | ----------------------- | ------------------------------------------- |
| `KeyboardPlanarVelocityBinding` | `KeyboardDevice` | `PlanarVelocityCommand` | `W/S`、`A/D`、`Q/E` 映射到各轴 lower/upper  |
| `GamePadPlanarVelocityBinding`  | `GamePadDevice`  | `PlanarVelocityCommand` | 三个指定轴经过 deadzone、inversion 和 scale |
| `ConstantPlanarVelocityBinding` | 三维常量         | `PlanarVelocityCommand` | 将同一个值复制到请求的 batch                |

键盘映射使用 held state。正反方向同时按下时相互抵消，松开后该轴立即归零：

| 轴         | 正方向                   | 负方向                   |
| ---------- | ------------------------ | ------------------------ |
| `vx`       | `W` → `command_upper[0]` | `S` → `command_lower[0]` |
| `vy`       | `A` → `command_upper[1]` | `D` → `command_lower[1]` |
| `yaw_rate` | `Q` → `command_upper[2]` | `E` → `command_lower[2]` |

手柄 binding 依赖 `axis_value()` 的 `[-1, 1]` 契约，在应用 deadzone 和 inversion 后逐轴乘以 scale。task 校验
最终 command 是否位于自己的允许范围。

### Task-specific Binding

`RandomPlanarVelocityBinding` 位于 `motrix_envs.locomotion.quadruped.velocity_command`。它从训练配置的 lower/upper
范围独立采样每个环境，并按 `standing_probability` 将部分 command 设为零。随机分布和 standing sampling 是四足
训练语义，因此随四足任务 package 提供。

这条边界也适用于其他任务：如果 mapping 包含特定 curriculum、目标采样或 reset 分布，应把 binding 留在 task
package；如果 mapping 对多个任务具有稳定相同的含义，才考虑放入 core。

下一步可阅读{doc}`扩展开发 <extending>`，了解如何复用这些 contract 接入新设备或定义新的任务指令。
