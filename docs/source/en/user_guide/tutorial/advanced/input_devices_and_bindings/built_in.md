# Built-in Input Components

This page describes the task commands, device contracts, and bindings provided by MotrixLab. They follow the layering in the
{doc}`overview <../input_devices_and_bindings>` and preserve the same task-command semantics across training, simulation
deployment, and a physical backend that implements `RobotInterface`.

## Built-in Task Command

### `PlanarVelocityCommand`

The current shared command is `PlanarVelocityCommand`. It represents a body-frame planar-velocity target. Its `values` field is
a read-only `float32` array with shape `(batch_size, 3)`:

| Index | Property                | Meaning                          | Unit  |
| ----: | ----------------------- | -------------------------------- | ----- |
|   `0` | `linear_velocity_x_mps` | Forward/backward linear velocity | m/s   |
|   `1` | `linear_velocity_y_mps` | Left/right linear velocity       | m/s   |
|   `2` | `yaw_rate_rad_s`        | Yaw angular velocity             | rad/s |

Training environments normally construct `(num_envs, 3)`, while single-robot deployment uses `(1, 3)`. Construction copies
the input, converts it to `float32`, rejects empty batches and non-finite values, and makes the array read-only. The concrete
task contract owns the velocity limits.

For example, a `go2_walk/v1` artifact stores `command_lower`, `command_upper`, and a three-dimensional `command_scale`. The task
multiplies both range endpoints by the scale before exposing them to a binding, then checks the command batch size and final
range again when consuming it.

## Built-in InputDevice Contracts

### `InputDevice`

`InputDevice` is the common marker base class for input devices. Each subtype defines a query protocol suited to the device:
a keyboard represents discrete events and held state, while a gamepad also represents continuous axes. The provider or
application that creates a concrete device owns its resources and lifecycle.

### `KeyboardDevice`

`KeyboardDevice` defines event-frame queries:

```python
class KeyboardDevice(InputDevice, ABC):
    def poll(self) -> None: ...
    def is_key_down(self, key: str) -> bool: ...
    def is_key_up(self, key: str) -> bool: ...
    def is_pressing(self, key: str) -> bool: ...
```

`poll()` freezes the event frame for the current control tick. Every query is non-consuming and stable until the next poll.

| Query              | Meaning in the current event frame       |
| ------------------ | ---------------------------------------- |
| `is_key_down(key)` | A press edge occurred in this frame      |
| `is_key_up(key)`   | A release edge occurred in this frame    |
| `is_pressing(key)` | The key is held at the end of this frame |

Continuous velocity control uses `is_pressing()`. Menu toggles and one-shot actions commonly use down/up edges. Operating-system
key repeat preserves held state, with the initial press producing one down edge.

The current concrete implementation, `MujocoKeyboardDevice`, lives in `motrix_deploy_mujoco`. It reads callbacks from the
deployment GLFW window and only receives that window's events. Losing focus immediately releases every held key; Esc or
closing the window interrupts the rollout. The MuJoCo backend viewer owns its lifecycle, not the binding.

### `GamePadDevice`

`GamePadDevice` provides the same `poll()` and button down/up/pressing semantics, plus axis queries:

```python
value = device.axis_value("left_y")
```

A concrete provider validates and normalizes raw SDK values to finite values in `[-1, 1]`. Deadzone, axis inversion, and
task-command scale belong to the binding. The binding consumes provider output according to the device contract. Providers
for different gamepad SDKs integrate through this ABC.

## Built-in Bindings

`CommandBinding` has one public method:

```python
CommandT = TypeVar("CommandT")


class CommandBinding(ABC, Generic[CommandT]):
    def read_command(self, *, batch_size: int = 1) -> CommandT: ...
```

`batch_size` is a call argument because vectorized training may request a dynamic number of commands during a partial reset,
while deployment requests one. A device-backed binding should call `poll()` exactly once per `read_command()` so every key,
button, and axis query used by that mapping belongs to the same event frame.

### Shared Bindings

| Binding                         | Input source               | Output                  | Mapping behavior                                              |
| ------------------------------- | -------------------------- | ----------------------- | ------------------------------------------------------------- |
| `KeyboardPlanarVelocityBinding` | `KeyboardDevice`           | `PlanarVelocityCommand` | `W/S`, `A/D`, and `Q/E` select each axis lower/upper endpoint |
| `GamePadPlanarVelocityBinding`  | `GamePadDevice`            | `PlanarVelocityCommand` | Three named axes with deadzone, inversion, and scale          |
| `ConstantPlanarVelocityBinding` | Three-dimensional constant | `PlanarVelocityCommand` | Replicates the same value across the requested batch          |

The keyboard mapping reads held state. Opposite directions cancel, and releasing a key immediately returns that axis to zero:

| Axis       | Positive direction       | Negative direction       |
| ---------- | ------------------------ | ------------------------ |
| `vx`       | `W` → `command_upper[0]` | `S` → `command_lower[0]` |
| `vy`       | `A` → `command_upper[1]` | `D` → `command_lower[1]` |
| `yaw_rate` | `Q` → `command_upper[2]` | `E` → `command_lower[2]` |

The gamepad binding relies on the `axis_value()` contract of `[-1, 1]`, applies deadzone and inversion, and multiplies each
axis by its scale. The task validates the final command against its allowed range.

### Task-specific Binding

`RandomPlanarVelocityBinding` lives in `motrix_envs.locomotion.quadruped.velocity_command`. It independently samples each
training environment from configured lower/upper ranges and replaces some samples with zero according to
`standing_probability`. The quadruped task package owns these training-specific distribution and standing semantics.

Apply the same boundary to other tasks. Keep a binding with its task package when its mapping contains a task-specific
curriculum, target sampler, or reset distribution. Consider moving it to core only after its meaning is stably shared by
several tasks.

Continue with the {doc}`extension guide <extending>` to reuse these contracts for a new device or task command.
