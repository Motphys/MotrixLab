# Extending Input Components

This page explains how to integrate another input source while preserving the
{doc}`InputDevice, binding, and task-command architecture <../input_devices_and_bindings>`. Start at the layer whose behavior
actually changes. Reuse an existing device contract and command semantics where possible, and add a type only when the raw
query protocol or task objective genuinely differs.

## Choosing an Extension Point

Choose the smallest extension at the layer where behavior changes:

| Requirement                                    | Extend                                                           | Reuse relationship                               |
| ---------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------ |
| Integrate another keyboard window or listener  | Implement `KeyboardDevice`                                       | Reuse the existing keyboard binding and task     |
| Integrate another gamepad SDK                  | Implement `GamePadDevice`                                        | Reuse the existing gamepad binding and task      |
| Change key, axis, or scaling rules             | Add `CommandBinding[existing CommandT]`                          | Reuse the existing device and task               |
| Source already supplies a task command         | Implement `CommandBinding` directly                              | Reuse the existing task                          |
| Hardware has a distinct query protocol         | Add an `InputDevice` subtype and its binding                     | The task continues to consume the output command |
| A new task needs different objective semantics | Add a command dataclass and binding, then consume it in the task | A new typed task contract is required            |

If a ROS topic, network service, or configuration file already supplies `[vx, vy, yaw_rate]`, implement
`CommandBinding[PlanarVelocityCommand]` directly to map that source to the existing task command.

## Adding a Concrete Device Provider

The most common extension is a new provider for an existing ABC. This callback-backed keyboard example shows the state needed
for event frames. A window system or SDK callback invokes `on_press()` / `on_release()`, while the control loop freezes the
result through `poll()`:

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

The provider can immediately reuse the shared mapping:

```python
from motrix_env_core.input import KeyboardPlanarVelocityBinding

device = CallbackKeyboardDevice()
binding = KeyboardPlanarVelocityBinding(
    device,
    command_lower=[-0.5, -0.4, -1.0],
    command_upper=[1.0, 0.4, 1.0],
)
```

A provider should also define disconnection, focus-loss, and resource-cleanup behavior. The object that creates a device owns
its lifecycle, closes listeners, file descriptors, or SDK handles, and closes a shared device when appropriate.

## Adding a Device Category and Binding

Add an `InputDevice` subtype when the raw query semantics require data beyond the keyboard and gamepad contracts. This
SpaceMouse example defines a device-specific frame query and maps selected components to the shared planar-velocity command:

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

A production implementation should also validate scale shape and finiteness in the constructor. The concrete device provider
owns translation/rotation shape, units, normalized range, and frame stability. Keep a subtype and binding in an application
package if only that application uses them; add them to the public `motrix_env_core.input` exports only after a cross-task
reuse requirement exists.

## Adding a Task-specific Command

When the task objective is not planar velocity, define another immutable dataclass and then:

1. Choose its package based on semantic reuse scope.
2. Specify batch-first shape, component order, coordinate frame, and units.
3. Copy and validate dtype, shape, and finite values during construction so the command retains immutable data.
4. Implement `CommandBinding[NewCommand]`; it may read a device, configuration, network source, or RNG.
5. Make the training task or `DeployTask[NewCommand]` validate task ranges and consume the type.
6. Add separate command, binding, and task-integration tests.

The task contract, binding, and device own task ranges, key mappings, and SDK channel names respectively.
