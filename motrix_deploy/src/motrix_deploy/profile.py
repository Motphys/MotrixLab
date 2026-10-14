# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Task-agnostic result of compiling one training environment for deployment."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

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

_PROFILE_COMPILERS: dict[str, ProfileCompiler] = {}


def register_profile_compiler(*env_names: str) -> Callable[[ProfileCompiler], ProfileCompiler]:
    """Register one profile compiler for one or more environment names."""
    if not env_names:
        raise ValueError("At least one environment name is required")
    if any(not env_name for env_name in env_names):
        raise ValueError("Environment names must be non-empty")
    if len(set(env_names)) != len(env_names):
        raise ValueError("Environment names must be unique")

    def register(compiler: ProfileCompiler) -> ProfileCompiler:
        duplicates = set(env_names) & set(_PROFILE_COMPILERS)
        if duplicates:
            names = ", ".join(sorted(duplicates))
            raise ValueError(f"Deployment profile compiler already registered for: {names}")
        for env_name in env_names:
            _PROFILE_COMPILERS[env_name] = compiler
        return compiler

    return register


def build_deployment_profile(env_name: str) -> DeploymentProfile:
    """Build a deployment profile using the compiler registered for an environment."""
    try:
        compiler = _PROFILE_COMPILERS[env_name]
    except KeyError as error:
        supported = ", ".join(sorted(_PROFILE_COMPILERS)) or "none"
        raise ValueError(
            f"No deployment profile compiler for environment {env_name!r}; supported environments: {supported}"
        ) from error
    return compiler(env_name)


def registered_profile_compilers() -> tuple[str, ...]:
    """Return environment names with a compiler registered in this process."""
    return tuple(sorted(_PROFILE_COMPILERS))


__all__ = [
    "DeploymentProfile",
    "ProfileCompiler",
    "build_deployment_profile",
    "register_profile_compiler",
    "registered_profile_compilers",
]
