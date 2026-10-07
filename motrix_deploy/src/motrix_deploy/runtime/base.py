# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Deployment runtime lifecycle contracts."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from motrix_deploy.errors import ValidationError

if TYPE_CHECKING:
    from motrix_deploy.robot.interface import RobotInterface
    from motrix_deploy.runtime.control import ControlSession
    from motrix_deploy.runtime.result import RolloutResult


class DeploymentRuntime(ABC):
    """Own runtime resources while a control session owns the robot backend."""

    def __init__(self) -> None:
        self._opened = False
        self._control: ControlSession | None = None

    @property
    def control(self) -> "ControlSession | None":
        """Control session bound to this runtime, if any."""
        return self._control

    def bind_control_session(self, control: "ControlSession") -> None:
        """Bind one control session, rejecting conflicting robot ownership."""
        if control.robot is not self.robot:
            raise ValidationError("runtime.control.robot", self.robot, control.robot)
        if self._control is not None and self._control is not control:
            raise ValidationError("runtime.control", "the already bound control session", control)
        self._control = control

    def _resolve_control(self, control: "ControlSession | None") -> "ControlSession":
        if control is not None and self.control is not None and control is not self.control:
            raise ValidationError("runtime.control", "the already bound control session", control)
        resolved = control if control is not None else self.control
        if resolved is None:
            raise ValidationError("runtime.control", "a supplied or bound control session", None)
        if resolved.robot is not self.robot:
            raise ValidationError("runtime.robot", "the control session robot", resolved.robot)
        return resolved

    @property
    @abstractmethod
    def robot(self) -> "RobotInterface":
        """Robot interface associated with this runtime."""
        raise NotImplementedError

    @property
    def opened(self) -> bool:
        """Whether this runtime lifecycle is open."""
        return self._opened

    def open(self) -> None:
        """Open runtime resources; safe to call repeatedly."""
        self._opened = True

    def close(self) -> None:
        """Close runtime resources; safe to call repeatedly."""
        self._opened = False

    def __enter__(self) -> "DeploymentRuntime":
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()

    @abstractmethod
    def run(self, control: "ControlSession | None" = None, *, steps: int | None = None) -> "RolloutResult":
        """Run a control session and return its structured result."""


class SimulationRuntime(DeploymentRuntime, ABC):
    """Base for runtimes that own simulation advancement and robot creation."""


__all__ = ["DeploymentRuntime", "SimulationRuntime"]
