# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Task-specific preprocessing surrounding observation-to-action inference."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from motrix_deploy.contracts import FloatArray, RobotCommand, RobotState

if TYPE_CHECKING:
    from motrix_deploy.runtime.context import ControlContext

CommandT = TypeVar("CommandT")


class PolicyProcessor(ABC, Generic[CommandT]):
    """Own observation/action history and policy input validity, not task phases."""

    @abstractmethod
    def reset(self, state: RobotState, context: ControlContext[CommandT]) -> None: ...

    @abstractmethod
    def build_observation(self, state: RobotState, context: ControlContext[CommandT]) -> FloatArray: ...

    @abstractmethod
    def process_action(self, action: FloatArray) -> RobotCommand: ...

    @abstractmethod
    def validate_command(self, command: CommandT | None) -> None: ...

    def check_termination(self, state: RobotState) -> str | None:
        """Return a policy termination reason, or None when no check is required."""
        return None
