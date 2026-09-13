# Command Input Architecture

MotrixLab separates how input is read from what a policy is asked to achieve. An `InputDevice` exposes raw device state, a
`CommandBinding` converts a device or another source into a typed high-level command, and the task consumes that command when
building the policy observation. This split lets keyboards, gamepads, constants, and randomized training samples drive the
policy through one task and control path. It is also the shared control abstraction for
simulation and physical robots: the same task command and policy can drive different backends through `RobotInterface`.

This overview explains the three contracts and their responsibilities. The concrete API is divided between the
{doc}`built-in implementations <input_devices_and_bindings/built_in>` and the
{doc}`extension guide <input_devices_and_bindings/extending>`. All public types are importable from
`motrix_env_core.input`.

```{toctree}
:hidden:
:maxdepth: 1

input_devices_and_bindings/built_in
input_devices_and_bindings/extending
```

## Architecture and Responsibilities

```{figure} /_static/images/input-device-binding-architecture.png
:alt: Shared InputDevice, CommandBinding, task-command, simulation-backend, and physical-robot-backend pipeline
:width: 100%
:align: center

Backend-neutral InputDevice, binding, and task-command pipeline
```

The control path from `CommandBinding` through `RobotInterface` is backend-neutral. The same binding, task command,
`PolicyContext`, task/policy contract, and `RobotCommand` semantics can control simulation and physical-robot targets. Moving
between the two requires replacing only the concrete device provider and `RobotInterface` implementation. A physical backend
that implements `RobotInterface` therefore reuses the same command pipeline, task, and policy.

A simulation and a physical robot can select different concrete devices. For example, simulation can use a GLFW keyboard
while a physical robot uses a remote controller or network input; both reuse the same binding and task-command semantics.

A `CommandBinding` can read an `InputDevice` such as a keyboard or gamepad, or directly read constants, configuration, or an
RNG. As long as they produce the same `CommandT`, the downstream task consumes each command in the same way.

| Concept                    | Responsibility                                                                            |
| -------------------------- | ----------------------------------------------------------------------------------------- |
| `InputDevice`              | Represents a raw input category and its query protocol                                    |
| `CommandBinding[CommandT]` | Polls, maps, scales, or samples an input source and emits one command batch               |
| Task-specific command      | Represents the policy objective, dimensions, component order, coordinate frame, and units |
| Task                       | Validates the command and uses it in observations or other task logic                     |

A high-level task command also has the opposite direction from `RobotCommand`:

-   A high-level command such as `PlanarVelocityCommand` describes the policy objective, such as body-frame velocity.
-   `RobotCommand` is the low-level joint position, velocity, torque, and gain command produced after the task processes the
    policy action.

## How a Binding Reaches the Task

The runtime obtains a typed task command through `CommandBinding`:

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

For example, MuJoCo deployment obtains a `MujocoKeyboardDevice` from the backend's `KeyboardDeviceProvider` and constructs a
`KeyboardPlanarVelocityBinding`. Training can use a random binding for the same `PlanarVelocityCommand`. Both paths end at the
same typed task contract.

Programmatic operation can use a constant binding directly. A fixed velocity uses:

```python
from motrix_env_core.input import ConstantPlanarVelocityBinding

binding = ConstantPlanarVelocityBinding([0.5, 0.0, 0.0])
command = binding.read_command(batch_size=4)
assert command.values.shape == (4, 3)
```

## Continue Reading

-   {doc}`Built-in implementations <input_devices_and_bindings/built_in>`: existing task commands, `InputDevice` contracts,
    and bindings.
-   {doc}`Extension guide <input_devices_and_bindings/extending>`: choose an extension point and implement a provider, device
    category, or task command.
