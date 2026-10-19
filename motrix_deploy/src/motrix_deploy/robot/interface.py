# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Public robot I/O and device capability contracts."""

from abc import ABC, abstractmethod

from motrix_deploy.contracts import HealthStatus, RobotCapabilities, RobotCommand, RobotSpec, RobotState
from motrix_env_core.input import GamePadDevice


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

    def get_gamepad_device(self) -> GamePadDevice | None:
        """Return an optional operator device supplied by the robot transport."""
        return None

    def acquire_control(self) -> None:
        """Acquire command authority before open; backends without a handoff do nothing.

        This explicit operation may change device mode. Backends requiring operator
        consent must receive it explicitly and reject acquisition without it.
        """

    @abstractmethod
    def open(self) -> None:
        """Open resources for the robot specification supplied at construction."""

    def enable(self) -> None:
        """Enable output after transport safety checks, without task trajectories or operator gates."""
        if self.capabilities.requires_enable:
            raise NotImplementedError("robot declares requires_enable=True but does not implement enable()")

    @abstractmethod
    def read_state(self) -> RobotState:
        """Return the current/latest state snapshot without waiting for a new sample.

        Repeated reads of a cached sample retain its original receive timestamp.
        """

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


__all__ = ["RobotInterface"]
