# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Runtime contract implemented by concrete deployment-task packages."""

from abc import ABC, abstractmethod
from importlib import metadata
from typing import Any, ClassVar, Generic, TypeVar

from motrix_deploy.artifact.schema import TaskSpec
from motrix_deploy.contracts import FloatArray, RobotCommand, RobotSpec, RobotState
from motrix_deploy.runtime.context import PolicyContext

CommandT = TypeVar("CommandT")


class DeployTask(ABC, Generic[CommandT]):
    """Own task-specific observation, action, command, and termination semantics.

    ControlSession checks backend state and calls validate_command before passing
    inputs to reset and build_observation. Tasks check optional state fields at
    their usage sites when a feature needs them. Backend capabilities describe
    which fields are privileged ground truth.
    """

    spec_type: ClassVar[type[TaskSpec]]

    def check_termination(self, state: RobotState) -> str | None:
        """Return a task termination reason, or None to continue, without side effects."""
        return None

    @abstractmethod
    def reset(self, state: RobotState, context: PolicyContext[CommandT]) -> None:
        """Reset episode-local preprocessing and action state."""

    @abstractmethod
    def build_observation(self, state: RobotState, context: PolicyContext[CommandT]) -> FloatArray:
        """Build one flat policy observation."""

    @abstractmethod
    def process_action(self, action: FloatArray) -> RobotCommand:
        """Convert a policy action into a servo or torque command supported by the backend."""

    @abstractmethod
    def validate_command(self, command: CommandT) -> None:
        """Validate one external high-level command before using it."""


TASK_ENTRY_POINT_GROUP = "motrix_deploy.tasks"


def available_tasks() -> tuple[str, ...]:
    """List installed task plugins without importing their implementations."""
    return tuple(sorted({entry.name for entry in metadata.entry_points(group=TASK_ENTRY_POINT_GROUP)}))


def load_task_type(name: str) -> type[DeployTask[Any]]:
    """Load only the selected installed task class, shared by codec and runtime."""
    entries = tuple(metadata.entry_points(group=TASK_ENTRY_POINT_GROUP))
    matches = [entry for entry in entries if entry.name == name]
    if not matches:
        supported = ", ".join(sorted({entry.name for entry in entries})) or "none"
        raise ValueError(f"Unsupported deployment task {name}; available tasks: {supported}")
    if len(matches) != 1:
        raise ValueError(f"Multiple deployment task plugins provide {name}")
    task_type = matches[0].load()
    if not isinstance(task_type, type) or not issubclass(task_type, DeployTask):
        raise TypeError(f"Deployment task plugin {name} must provide a DeployTask class")
    return task_type


def create_task(spec: TaskSpec, robot: RobotSpec) -> DeployTask[Any]:
    """Instantiate the installed task class selected by an artifact."""
    return load_task_type(spec.name)(spec, robot)


__all__ = ["DeployTask", "create_task", "available_tasks"]
