# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Bindings that produce high-level commands from devices or configuration."""

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar

import torch

from motrix_env_core.input.command import PlanarVelocityCommand
from motrix_env_core.input.device import GamePadDevice, KeyboardDevice

CommandT = TypeVar("CommandT")


def _finite_vec3(name: str, values) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.shape != (3,) or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain three finite values")
    return tensor


class CommandBinding(ABC, Generic[CommandT]):
    """Produce task commands from a device, configuration, or another input mechanism."""

    @abstractmethod
    def read_command(self, *, batch_size: int = 1) -> CommandT:
        """Produce one command batch of the requested size."""


class KeyboardPlanarVelocityBinding(CommandBinding[PlanarVelocityCommand]):
    """Map held ``W/S``, ``A/D``, and ``Q/E`` keys to planar velocity."""

    def __init__(
        self,
        device: KeyboardDevice,
        *,
        command_lower: Sequence[float] | torch.Tensor,
        command_upper: Sequence[float] | torch.Tensor,
    ) -> None:
        self._device = device
        self._command_lower = _finite_vec3("command_lower", command_lower)
        self._command_upper = _finite_vec3("command_upper", command_upper)
        if bool((self._command_lower > 0.0).any()) or bool((self._command_upper < 0.0).any()):
            raise ValueError("keyboard command range must contain zero on every axis")

    def read_command(self, *, batch_size: int = 1) -> PlanarVelocityCommand:
        _validate_batch_size(batch_size)
        self._device.poll()
        direction = torch.tensor(
            (
                int(self._device.is_pressing("w")) - int(self._device.is_pressing("s")),
                int(self._device.is_pressing("a")) - int(self._device.is_pressing("d")),
                int(self._device.is_pressing("q")) - int(self._device.is_pressing("e")),
            ),
            dtype=torch.int8,
        )
        zero = torch.zeros(3, dtype=torch.float32)
        value = torch.where(
            direction > 0,
            self._command_upper,
            torch.where(direction < 0, self._command_lower, zero),
        )
        return PlanarVelocityCommand(value[None, :].repeat(batch_size, 1))


class GamePadPlanarVelocityBinding(CommandBinding[PlanarVelocityCommand]):
    """Map three gamepad axes to planar velocity."""

    def __init__(
        self,
        device: GamePadDevice,
        *,
        linear_x_axis: str,
        linear_y_axis: str,
        yaw_axis: str,
        linear_x_scale: float,
        linear_y_scale: float,
        yaw_scale: float,
        deadzone: float = 0.0,
        invert_linear_x: bool = False,
        invert_linear_y: bool = False,
        invert_yaw: bool = False,
    ) -> None:
        axes = (linear_x_axis, linear_y_axis, yaw_axis)
        if any(not isinstance(axis, str) or not axis for axis in axes):
            raise ValueError("axis names must be non-empty strings")
        self._device = device
        self._axes = axes
        self._scale = _finite_vec3("scale values", (linear_x_scale, linear_y_scale, yaw_scale))
        if bool((self._scale < 0.0).any()):
            raise ValueError("scale values must be non-negative")
        if isinstance(deadzone, bool) or not isinstance(deadzone, (int, float)):
            raise ValueError("deadzone must be a finite value inside [0, 1)")
        self._deadzone = float(deadzone)
        if not math.isfinite(self._deadzone) or not 0.0 <= self._deadzone < 1.0:
            raise ValueError("deadzone must be a finite value inside [0, 1)")
        inversions = (invert_linear_x, invert_linear_y, invert_yaw)
        if any(not isinstance(value, bool) for value in inversions):
            raise ValueError("axis inversion flags must be bool values")
        self._direction = torch.tensor(tuple(-1.0 if value else 1.0 for value in inversions), dtype=torch.float32)

    def read_command(self, *, batch_size: int = 1) -> PlanarVelocityCommand:
        _validate_batch_size(batch_size)
        self._device.poll()
        axes = torch.tensor(tuple(self._device.axis_value(axis) for axis in self._axes), dtype=torch.float32)
        axes = torch.where(axes.abs() < self._deadzone, torch.zeros_like(axes), axes)
        value = axes * self._direction * self._scale
        return PlanarVelocityCommand(value[None, :].repeat(batch_size, 1))


class BoundedGamePadPlanarVelocityBinding(CommandBinding[PlanarVelocityCommand]):
    """Map normalized gamepad axes into artifact-bounded planar velocity commands."""

    def __init__(
        self,
        device: GamePadDevice,
        *,
        linear_x_axis: str,
        linear_y_axis: str,
        yaw_axis: str,
        command_lower: Sequence[float] | torch.Tensor,
        command_upper: Sequence[float] | torch.Tensor,
        deadzone: float = 0.0,
        range_scale: Sequence[float] | torch.Tensor = (1.0, 1.0, 1.0),
        invert_linear_x: bool = False,
        invert_linear_y: bool = False,
        invert_yaw: bool = False,
        deadman_button: str | None = None,
    ) -> None:
        axes = (linear_x_axis, linear_y_axis, yaw_axis)
        if any(not isinstance(axis, str) or not axis for axis in axes):
            raise ValueError("axis names must be non-empty strings")
        self._device = device
        self._axes = axes
        self._command_lower = _finite_vec3("command_lower", command_lower)
        self._command_upper = _finite_vec3("command_upper", command_upper)
        self._range_scale = _finite_vec3("range_scale", range_scale)
        if bool((self._command_lower > 0.0).any()) or bool((self._command_upper < 0.0).any()):
            raise ValueError("gamepad command range must contain zero on every axis")
        if bool((self._command_lower > self._command_upper).any()):
            raise ValueError("gamepad command lower must not exceed upper")
        if bool((self._range_scale < 0.0).any()) or bool((self._range_scale > 1.0).any()):
            raise ValueError("range_scale values must be inside [0, 1]")
        if isinstance(deadzone, bool) or not isinstance(deadzone, (int, float)):
            raise ValueError("deadzone must be a finite value inside [0, 1)")
        self._deadzone = float(deadzone)
        if not math.isfinite(self._deadzone) or not 0.0 <= self._deadzone < 1.0:
            raise ValueError("deadzone must be a finite value inside [0, 1)")
        inversions = (invert_linear_x, invert_linear_y, invert_yaw)
        if any(not isinstance(value, bool) for value in inversions):
            raise ValueError("axis inversion flags must be bool values")
        self._direction = torch.tensor(tuple(-1.0 if value else 1.0 for value in inversions), dtype=torch.float32)
        if deadman_button is not None and (not isinstance(deadman_button, str) or not deadman_button):
            raise ValueError("deadman_button must be null or a non-empty string")
        self._deadman_button = deadman_button

    def read_command(self, *, batch_size: int = 1) -> PlanarVelocityCommand:
        _validate_batch_size(batch_size)
        self._device.poll()
        if self._deadman_button is not None and not self._device.is_button_pressing(self._deadman_button):
            value = torch.zeros(3, dtype=torch.float32)
        else:
            axes = torch.tensor(tuple(self._device.axis_value(axis) for axis in self._axes), dtype=torch.float32)
            axes = torch.clamp(axes * self._direction, -1.0, 1.0)
            magnitude = (axes.abs() - self._deadzone).clamp(min=0.0) / (1.0 - self._deadzone)
            normalized = torch.copysign(magnitude, axes)
            bounded = torch.where(
                normalized >= 0.0,
                normalized * self._command_upper,
                (-normalized) * self._command_lower,
            )
            value = bounded * self._range_scale
        return PlanarVelocityCommand(value[None, :].repeat(batch_size, 1))


class ConstantPlanarVelocityBinding(CommandBinding[PlanarVelocityCommand]):
    """Replicate one configured planar velocity across the requested batch."""

    def __init__(self, value: Sequence[float] | torch.Tensor) -> None:
        self._value = _finite_vec3("value", value)

    def read_command(self, *, batch_size: int = 1) -> PlanarVelocityCommand:
        _validate_batch_size(batch_size)
        return PlanarVelocityCommand(self._value[None, :].repeat(batch_size, 1))


def _validate_batch_size(batch_size: int) -> None:
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")


__all__ = [
    "BoundedGamePadPlanarVelocityBinding",
    "CommandBinding",
    "ConstantPlanarVelocityBinding",
    "GamePadPlanarVelocityBinding",
    "KeyboardPlanarVelocityBinding",
]
