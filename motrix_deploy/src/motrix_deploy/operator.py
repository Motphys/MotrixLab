# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral operator gestures and command-input configuration."""

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from motrix_deploy.errors import ValidationError
from motrix_env_core.input import CommandBinding, GamePadDevice, InputDevice, KeyboardDevice, PlanarVelocityCommand


@dataclass(frozen=True)
class OperatorCommand:
    """Backend-neutral operator input consumed by a persistent control program."""

    intents: frozenset[str] = frozenset()
    planar_velocity: PlanarVelocityCommand | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.intents, frozenset) or any(
            not isinstance(intent, str) or not intent for intent in self.intents
        ):
            raise ValidationError("operator.intents", "a frozenset of non-empty strings", self.intents)
        if self.planar_velocity is not None and not isinstance(self.planar_velocity, PlanarVelocityCommand):
            raise ValidationError(
                "operator.planar_velocity", "a PlanarVelocityCommand or None", type(self.planar_velocity).__name__
            )


@dataclass
class ButtonGestureCfg:
    """One edge-triggered button gesture with optional held modifiers."""

    held: list[str] = field(default_factory=list)
    pressed: str | None = None
    released: str | None = None
    held_for_s: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = [*self.held, *self.held_for_s]
        if self.pressed is not None:
            names.append(self.pressed)
        if self.released is not None:
            names.append(self.released)
        if not names:
            raise ValidationError("operator.gesture", "at least one button condition", names)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValidationError("operator.gesture.buttons", "non-empty button names", names)
        if len(set(self.held)) != len(self.held):
            raise ValidationError("operator.gesture.held", "unique button names", self.held)
        for button, seconds in self.held_for_s.items():
            if (
                not isinstance(seconds, (int, float))
                or isinstance(seconds, bool)
                or not np.isfinite(seconds)
                or seconds <= 0
            ):
                raise ValidationError(f"operator.gesture.held_for_s.{button}", "a positive finite duration", seconds)


@dataclass
class PlanarVelocityOperatorCfg:
    """Map three normalized device axes into one planar velocity command."""

    linear_x_axis: str = "ly"
    linear_y_axis: str = "lx"
    yaw_axis: str = "rx"
    invert_linear_x: bool = False
    invert_linear_y: bool = False
    invert_yaw: bool = False
    deadzone: float = 0.0
    range_scale: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    deadman_button: str | None = None

    def __post_init__(self) -> None:
        for name in (self.linear_x_axis, self.linear_y_axis, self.yaw_axis):
            if not isinstance(name, str) or not name:
                raise ValidationError("operator.planar_velocity.axes", "non-empty axis names", name)
        if (
            not isinstance(self.deadzone, (int, float))
            or isinstance(self.deadzone, bool)
            or not np.isfinite(self.deadzone)
            or not 0 <= self.deadzone < 1
        ):
            raise ValidationError("operator.planar_velocity.deadzone", "a finite value in [0, 1)", self.deadzone)
        scale = np.asarray(self.range_scale, dtype=np.float32)
        if scale.shape != (3,) or not np.all(np.isfinite(scale)) or np.any(scale < 0) or np.any(scale > 1):
            raise ValidationError(
                "operator.planar_velocity.range_scale", "three finite values in [0, 1]", self.range_scale
            )
        if self.deadman_button is not None and (not isinstance(self.deadman_button, str) or not self.deadman_button):
            raise ValidationError(
                "operator.planar_velocity.deadman_button", "null or a non-empty button name", self.deadman_button
            )


@dataclass
class OperatorProfileCfg:
    """User-configurable gestures plus optional continuous command wiring."""

    intents: dict[str, ButtonGestureCfg] = field(default_factory=dict)
    planar_velocity: PlanarVelocityOperatorCfg | None = None

    def __post_init__(self) -> None:
        if any(not isinstance(name, str) or not name for name in self.intents):
            raise ValidationError("operator.intents", "non-empty intent names", tuple(self.intents))
        signatures: dict[tuple[object, ...], str] = {}
        for intent, gesture in self.intents.items():
            if not isinstance(gesture, ButtonGestureCfg):
                raise ValidationError(f"operator.intents.{intent}", "a ButtonGestureCfg", type(gesture).__name__)
            signature = (
                tuple(sorted(gesture.held)),
                gesture.pressed,
                gesture.released,
                tuple(sorted(gesture.held_for_s.items())),
            )
            previous = signatures.get(signature)
            if previous is not None:
                raise ValidationError(
                    "operator.intents", "distinct gestures for each intent", f"{previous!r} and {intent!r}"
                )
            signatures[signature] = intent


class OperatorIntentEvaluator:
    """Evaluate configured button gestures from timestamped button snapshots."""

    def __init__(self, gestures: dict[str, ButtonGestureCfg]) -> None:
        self._gestures = dict(gestures)
        self._previous: frozenset[str] | None = None
        self._pressed_since_ns: dict[str, int] = {}

    def reset(self) -> None:
        self._previous = None
        self._pressed_since_ns.clear()

    def update(self, pressed_buttons: frozenset[str], timestamp_ns: int) -> frozenset[str]:
        """Return intents whose edge/hold predicates became true in this snapshot."""

        if not isinstance(timestamp_ns, int) or isinstance(timestamp_ns, bool) or timestamp_ns < 0:
            raise ValidationError("operator.timestamp_ns", "a non-negative integer", timestamp_ns)
        previous = self._previous
        for button in pressed_buttons:
            self._pressed_since_ns.setdefault(button, timestamp_ns)
        for button in tuple(self._pressed_since_ns):
            if button not in pressed_buttons:
                del self._pressed_since_ns[button]
        self._previous = pressed_buttons
        if previous is None:
            return frozenset()

        matched = []
        for intent, gesture in self._gestures.items():
            if any(button not in pressed_buttons for button in gesture.held):
                continue
            if gesture.pressed is not None and not (
                gesture.pressed in pressed_buttons and gesture.pressed not in previous
            ):
                continue
            if gesture.released is not None and not (
                gesture.released not in pressed_buttons and gesture.released in previous
            ):
                continue
            if any(
                button not in pressed_buttons
                or timestamp_ns - self._pressed_since_ns.get(button, timestamp_ns) < round(seconds * 1e9)
                for button, seconds in gesture.held_for_s.items()
            ):
                continue
            matched.append(intent)
        return frozenset(matched)


class OperatorCommandBinding(CommandBinding[OperatorCommand]):
    """Read one input frame and produce semantic intents plus bounded velocity."""

    def __init__(
        self,
        device: InputDevice,
        profile: OperatorProfileCfg,
        *,
        command_lower: Sequence[float],
        command_upper: Sequence[float],
    ) -> None:
        if not isinstance(device, (KeyboardDevice, GamePadDevice)):
            raise ValidationError("operator.device", "a keyboard or gamepad", type(device).__name__)
        self._device = device
        self._profile = profile
        self._evaluator = OperatorIntentEvaluator(profile.intents)
        self._buttons = frozenset(
            button
            for gesture in profile.intents.values()
            for button in (*gesture.held, *gesture.held_for_s, gesture.pressed, gesture.released)
            if button is not None
        )
        self._lower = np.asarray(command_lower, dtype=np.float32)
        self._upper = np.asarray(command_upper, dtype=np.float32)
        if self._lower.shape != (3,) or self._upper.shape != (3,):
            raise ValidationError("operator.command_bounds", "two vectors with shape (3,)", (self._lower, self._upper))
        if np.any(self._lower > 0) or np.any(self._upper < 0) or np.any(self._lower > self._upper):
            raise ValidationError(
                "operator.command_bounds", "ordered bounds containing zero", (self._lower, self._upper)
            )

    def _pressed_buttons(self) -> frozenset[str]:
        if isinstance(self._device, KeyboardDevice):
            return frozenset(button for button in self._buttons if self._device.is_pressing(button))
        return frozenset(button for button in self._buttons if self._device.is_button_pressing(button))

    def _keyboard_velocity(self) -> np.ndarray:
        assert isinstance(self._device, KeyboardDevice)
        direction = np.asarray(
            (
                int(self._device.is_pressing("w")) - int(self._device.is_pressing("s")),
                int(self._device.is_pressing("a")) - int(self._device.is_pressing("d")),
                int(self._device.is_pressing("q")) - int(self._device.is_pressing("e")),
            ),
            dtype=np.int8,
        )
        return np.where(direction > 0, self._upper, np.where(direction < 0, self._lower, 0)).astype(np.float32)

    def _gamepad_velocity(self) -> np.ndarray:
        assert isinstance(self._device, GamePadDevice)
        config = self._profile.planar_velocity
        if config is None:
            return np.zeros(3, dtype=np.float32)
        if config.deadman_button is not None and not self._device.is_button_pressing(config.deadman_button):
            return np.zeros(3, dtype=np.float32)
        axes = np.asarray(
            (
                self._device.axis_value(config.linear_x_axis),
                self._device.axis_value(config.linear_y_axis),
                self._device.axis_value(config.yaw_axis),
            ),
            dtype=np.float32,
        )
        direction = np.asarray(
            (
                -1.0 if config.invert_linear_x else 1.0,
                -1.0 if config.invert_linear_y else 1.0,
                -1.0 if config.invert_yaw else 1.0,
            ),
            dtype=np.float32,
        )
        axes = np.clip(axes * direction, -1.0, 1.0)
        magnitude = np.maximum(np.abs(axes) - config.deadzone, 0.0) / (1.0 - config.deadzone)
        normalized = np.copysign(magnitude, axes)
        bounded = np.where(normalized >= 0, normalized * self._upper, (-normalized) * self._lower)
        return (bounded * np.asarray(config.range_scale, dtype=np.float32)).astype(np.float32)

    def read_command(self, *, batch_size: int = 1) -> OperatorCommand:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValidationError("operator.batch_size", "a positive integer", batch_size)
        self._device.poll()
        intents = self._evaluator.update(self._pressed_buttons(), time.monotonic_ns())
        velocity = self._keyboard_velocity() if isinstance(self._device, KeyboardDevice) else self._gamepad_velocity()
        return OperatorCommand(
            intents=intents,
            planar_velocity=PlanarVelocityCommand(np.repeat(velocity[None, :], batch_size, axis=0)),
        )


class ConstantOperatorCommandBinding(CommandBinding[OperatorCommand]):
    """Keep a headless program passive until an external binding is supplied."""

    def read_command(self, *, batch_size: int = 1) -> OperatorCommand:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValidationError("operator.batch_size", "a positive integer", batch_size)
        return OperatorCommand(planar_velocity=PlanarVelocityCommand(np.zeros((batch_size, 3), dtype=np.float32)))


__all__ = [
    "ButtonGestureCfg",
    "ConstantOperatorCommandBinding",
    "OperatorCommand",
    "OperatorCommandBinding",
    "OperatorIntentEvaluator",
    "OperatorProfileCfg",
    "PlanarVelocityOperatorCfg",
]
