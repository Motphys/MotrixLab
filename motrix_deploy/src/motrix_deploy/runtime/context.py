# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Task-typed per-tick controller inputs."""

from dataclasses import dataclass
from typing import Generic, TypeVar

CommandT = TypeVar("CommandT")


@dataclass(frozen=True)
class ControlContext(Generic[CommandT]):
    """Deterministic per-tick inputs that do not come from robot state."""

    step: int
    elapsed_time_s: float
    command: CommandT | None
    dt_s: float


__all__ = ["ControlContext"]
