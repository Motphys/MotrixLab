# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Discover deployment runtimes through installed plugins."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import metadata
from typing import Any

from motrix_deploy.artifact.schema import ControlSpec
from motrix_deploy.contracts import RobotSpec
from motrix_deploy.runtime.base import DeploymentRuntime, SimulationRuntime
from motrix_deploy.runtime.config import SimulationRuntimeConfig
from motrix_deploy.runtime.hardware import HardwareRuntime


@dataclass(frozen=True)
class RuntimeCreateContext:
    """Artifact-owned robot and control contracts for hardware creation."""

    robot: RobotSpec
    control: ControlSpec


RUNTIME_ENTRY_POINT_GROUP = "motrix_deploy.backends"


def _runtime_entry_points() -> tuple[metadata.EntryPoint, ...]:
    return tuple(metadata.entry_points(group=RUNTIME_ENTRY_POINT_GROUP))


def registered_runtimes() -> tuple[str, ...]:
    return tuple(sorted(entry.name for entry in _runtime_entry_points()))


def create_simulation_runtime(
    name: str,
    config: SimulationRuntimeConfig,
) -> SimulationRuntime:
    """Create a simulation host with its robot ready for control-session assembly."""
    runtime = _validate_runtime(name, _load_factory(name)(config))
    if not isinstance(runtime, SimulationRuntime):
        runtime.close()
        raise TypeError(f"Deployment runtime plugin {name!r} does not provide a SimulationRuntime")
    return runtime


def create_hardware_runtime(
    name: str,
    config: Mapping[str, Any],
    context: RuntimeCreateContext,
) -> HardwareRuntime:
    """Create a hardware host, preserving plugin operator safety checks."""
    runtime = _validate_runtime(name, _load_factory(name)(config, context))
    if not isinstance(runtime, HardwareRuntime):
        runtime.close()
        raise TypeError(f"Deployment runtime plugin {name!r} does not provide a HardwareRuntime")
    return runtime


def _load_factory(name: str) -> Callable[..., DeploymentRuntime]:
    """Resolve the selected plugin; simulation and hardware have distinct contracts."""
    matches = [entry for entry in _runtime_entry_points() if entry.name == name]
    if not matches:
        available = ", ".join(registered_runtimes()) or "none"
        raise ValueError(f"Unsupported deployment runtime {name!r}; installed runtimes: {available}")
    if len(matches) != 1:
        values = sorted(entry.value for entry in matches)
        raise ValueError(f"Multiple deployment runtime plugins are registered as {name!r}: {values}")
    factory = matches[0].load()
    if not callable(factory):
        raise TypeError(f"Deployment runtime plugin {name!r} must load a callable factory")
    return factory


def _validate_runtime(name: str, runtime: DeploymentRuntime) -> DeploymentRuntime:
    if not isinstance(runtime, DeploymentRuntime):
        raise TypeError(
            f"Deployment runtime plugin {name!r} returned {type(runtime).__name__}, expected DeploymentRuntime"
        )
    return runtime
