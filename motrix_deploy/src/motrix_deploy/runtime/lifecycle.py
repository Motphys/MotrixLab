# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Robot-level controller decisions, independent of backend resources and scheduling."""

from dataclasses import dataclass, field
from typing import Protocol, TypeVar

from motrix_deploy.contracts import RobotCommand, RobotState
from motrix_deploy.runtime.context import ControlContext

CommandT = TypeVar("CommandT", contravariant=True)


@dataclass(frozen=True)
class ControllerStep:
    """One robot decision; completion consumes no physical interval."""

    command: RobotCommand | None = None
    complete: bool = False
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    policy_tick: bool = False
    trace_bytes: bytes = b""
    latency_ns: dict[str, int] = field(default_factory=dict)
    success: bool = True


class Controller(Protocol[CommandT]):
    """Map robot state and command input to robot commands, including composite stages."""

    def reset(self, state: RobotState) -> None: ...

    def step(self, state: RobotState, context: ControlContext[CommandT]) -> ControllerStep: ...

    def request_stop(self) -> None: ...
