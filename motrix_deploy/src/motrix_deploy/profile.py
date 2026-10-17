# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Task-agnostic result of compiling one training environment for deployment."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import metadata

from motrix_deploy.artifact.schema import ControlSpec, TaskSpec
from motrix_deploy.contracts import RobotSpec


@dataclass(frozen=True)
class DeploymentProfile:
    """Robot, task runtime, and timing contracts used to create an artifact."""

    robot: RobotSpec
    task: TaskSpec
    control: ControlSpec
    # Training-space dimensions used only to validate ONNX export, not serialized in TaskSpec.
    observation_size: int
    action_size: int
    # Task-owned binary payloads (path -> bytes) written into the artifact next to the policy.
    payloads: Mapping[str, bytes] = field(default_factory=dict)


ProfileCompiler = Callable[[str], DeploymentProfile]

PROFILE_COMPILER_ENTRY_POINT_GROUP = "motrix_deploy.profiles"


def build_deployment_profile(env_name: str) -> DeploymentProfile:
    """Load only the installed compiler for this exact environment name and call it."""
    entries = tuple(metadata.entry_points(group=PROFILE_COMPILER_ENTRY_POINT_GROUP))
    matches = [entry for entry in entries if entry.name == env_name]
    if not matches:
        supported = ", ".join(sorted({entry.name for entry in entries})) or "none"
        raise ValueError(
            f"No deployment profile compiler for environment {env_name!r}; supported environments: {supported}"
        )
    if len(matches) != 1:
        raise ValueError(f"Multiple deployment profile compiler plugins provide {env_name!r}")
    compiler = matches[0].load()
    if not callable(compiler):
        raise TypeError(f"Deployment profile compiler plugin {env_name!r} must provide a callable")
    return compiler(env_name)


def registered_profile_compilers() -> tuple[str, ...]:
    """List installed compiler environment names without importing implementations."""
    return tuple(sorted({entry.name for entry in metadata.entry_points(group=PROFILE_COMPILER_ENTRY_POINT_GROUP)}))


__all__ = [
    "DeploymentProfile",
    "ProfileCompiler",
    "build_deployment_profile",
    "registered_profile_compilers",
]
