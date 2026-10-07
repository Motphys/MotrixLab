# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Public robot I/O and device capability contracts."""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from motrix_deploy.contracts import HealthStatus, RobotCapabilities, RobotCommand, RobotSpec, RobotState
from motrix_env_core.input import GamePadDevice, KeyboardDevice


class RobotInterface(ABC):
    """Robot boundary shared by simulation and hardware runtimes."""

    @property
    @abstractmethod
    def spec(self) -> RobotSpec:
        """Return the canonical robot specification bound to this interface."""

    @property
    @abstractmethod
    def capabilities(self) -> RobotCapabilities:
        """Return immutable capabilities used during startup negotiation."""

    @abstractmethod
    def open(self) -> None:
        """Open resources for the robot specification supplied at construction."""

    def enable(self) -> None:
        """Enable command output using backend-owned startup safety behavior."""
        if self.capabilities.requires_enable:
            raise NotImplementedError("robot declares requires_enable=True but does not implement enable()")

    @abstractmethod
    def read_state(self, timeout_s: float) -> RobotState:
        """Read one state sample or raise on timeout."""

    @abstractmethod
    def write_command(self, command: RobotCommand) -> None:
        """Apply one canonical command."""

    @abstractmethod
    def health(self) -> HealthStatus:
        """Return current robot I/O health."""

    @abstractmethod
    def stop(self) -> None:
        """Enter the declared safe stopped state; must be idempotent."""

    @abstractmethod
    def close(self) -> None:
        """Release robot I/O resources; must be idempotent."""


@runtime_checkable
class KeyboardDeviceProvider(Protocol):
    """Optional runtime capability supplying keyboard input."""

    def get_keyboard_device(self) -> KeyboardDevice:
        """Return a device whose lifecycle is owned by its provider."""


@runtime_checkable
class GamePadDeviceProvider(Protocol):
    """Optional robot capability supplying gamepad input."""

    def get_gamepad_device(self) -> GamePadDevice:
        """Return a device whose lifecycle is owned by its provider."""


__all__ = ["GamePadDeviceProvider", "KeyboardDeviceProvider", "RobotInterface"]
