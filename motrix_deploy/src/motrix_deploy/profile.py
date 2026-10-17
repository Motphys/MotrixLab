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


_profile_compilers: dict[str, ProfileCompiler] = {}
_profile_compilers_discovered = False
_profile_compilers_discovering = False
_profile_entry_points: tuple[metadata.EntryPoint, ...] | None = None
_completed_profile_entry_points = 0


def register_profile_compiler(env_name: str, compiler: ProfileCompiler) -> None:
    """Register a compiler for an exact environment name without discovering providers."""
    if env_name in _profile_compilers:
        raise ValueError(f"Deployment profile compiler for environment {env_name!r} is already registered")
    if not callable(compiler):
        raise TypeError(f"Deployment profile compiler for environment {env_name!r} must be callable")
    _profile_compilers[env_name] = compiler


def _discover_profile_compilers() -> None:
    """Invoke installed package registration callbacks once, on the first query.

    Provider names must be unique. Completed callbacks are skipped on retry if a
    later load/callback fails. Failed callbacks retain registration side effects,
    so providers own recovery from partial registration. Reentrant queries see
    registrations completed so far without starting another discovery pass.
    """
    global _profile_compilers_discovered, _profile_compilers_discovering
    global _profile_entry_points, _completed_profile_entry_points
    if _profile_compilers_discovered or _profile_compilers_discovering:
        return
    _profile_compilers_discovering = True
    try:
        if _profile_entry_points is None:
            providers = tuple(
                sorted(
                    metadata.entry_points(group=PROFILE_COMPILER_ENTRY_POINT_GROUP),
                    key=lambda ep: (ep.name, ep.value),
                )
            )
            names = set()
            for provider in providers:
                if provider.name in names:
                    raise ValueError(
                        f"Duplicate deployment profile provider name {provider.name!r} in entry point group "
                        f"{PROFILE_COMPILER_ENTRY_POINT_GROUP!r}"
                    )
                names.add(provider.name)
            _profile_entry_points = providers
        for entry_point in _profile_entry_points[_completed_profile_entry_points:]:
            register = entry_point.load()
            if not callable(register):
                raise TypeError(f"Deployment profile entry point {entry_point.name!r} must load a callable")
            register()
            _completed_profile_entry_points += 1
        _profile_compilers_discovered = True
    finally:
        _profile_compilers_discovering = False


def build_deployment_profile(env_name: str) -> DeploymentProfile:
    """Discover package providers, then invoke the compiler for this exact environment."""
    _discover_profile_compilers()
    try:
        compiler = _profile_compilers[env_name]
    except KeyError:
        supported = ", ".join(sorted(_profile_compilers)) or "none"
        raise ValueError(
            f"No deployment profile compiler for environment {env_name!r}; registered environments: {supported}"
        ) from None
    return compiler(env_name)


def registered_profile_compilers() -> tuple[str, ...]:
    """Discover installed package providers and list registered compiler environment names."""
    _discover_profile_compilers()
    return tuple(sorted(_profile_compilers))


__all__ = [
    "DeploymentProfile",
    "ProfileCompiler",
    "build_deployment_profile",
    "register_profile_compiler",
    "registered_profile_compilers",
]
