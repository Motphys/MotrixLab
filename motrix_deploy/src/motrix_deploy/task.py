# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Complete task control entry point and installed task discovery."""

from abc import ABC, abstractmethod
from importlib import metadata
from typing import Any, ClassVar, Generic, TypeVar

from motrix_deploy.artifact.schema import TaskSpec
from motrix_deploy.contracts import RobotSpec, RobotState
from motrix_deploy.policy import Policy
from motrix_deploy.runtime.context import ControlContext
from motrix_deploy.runtime.lifecycle import ControllerStep

CommandT = TypeVar("CommandT")


class DeployTask(ABC, Generic[CommandT]):
    """Implement Controller and own task phases, without robot handles or scheduling."""

    spec_type: ClassVar[type[TaskSpec]]

    @classmethod
    def default_duration_s(cls, spec: TaskSpec) -> float | None:
        """Artifact-owned playback budget, if no CLI duration is supplied."""
        return None

    @abstractmethod
    def reset(self, state: RobotState) -> None: ...

    @abstractmethod
    def step(self, state: RobotState, context: ControlContext[CommandT]) -> ControllerStep: ...

    @abstractmethod
    def request_stop(self) -> None: ...

    @abstractmethod
    def validate_command(self, command: CommandT | None) -> None:
        """Validate external input at the application assembly boundary."""


TASK_ENTRY_POINT_GROUP = "motrix_deploy.tasks"


def available_tasks() -> tuple[str, ...]:
    """List installed tasks without importing their implementations."""
    return tuple(sorted({entry.name for entry in metadata.entry_points(group=TASK_ENTRY_POINT_GROUP)}))


def load_task_type(name: str) -> type[DeployTask[Any]]:
    entries = tuple(metadata.entry_points(group=TASK_ENTRY_POINT_GROUP))
    matches = [entry for entry in entries if entry.name == name]
    if not matches:
        supported = ", ".join(sorted({entry.name for entry in entries})) or "none"
        raise ValueError(f"Unsupported deployment task {name}; available tasks: {supported}")
    if len(matches) != 1:
        raise ValueError(f"Multiple deployment task plugins provide {name}")
    task_type = matches[0].load()
    if not isinstance(task_type, type) or not issubclass(task_type, DeployTask):
        raise TypeError(f"Deployment task plugin {name} must provide a DeployTask type")
    return task_type


def create_task(
    spec: TaskSpec,
    robot: RobotSpec,
    policy: Policy,
    steps: int | None = None,
    artifact: Any | None = None,
) -> DeployTask[Any]:
    """Construct a task; ``artifact`` supplies file payloads for tasks that reference them."""
    return load_task_type(spec.name)(spec, robot, policy, steps=steps, artifact=artifact)


__all__ = ["DeployTask", "create_task", "available_tasks", "load_task_type"]
