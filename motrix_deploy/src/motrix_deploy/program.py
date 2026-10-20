# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Persistent multi-policy deployment program discovery and assembly."""

from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from typing import Any

from motrix_deploy.artifact.schema import ControlSpec
from motrix_deploy.contracts import FloatArray, RobotSpec
from motrix_deploy.operator import (
    ConstantOperatorCommandBinding,
    OperatorCommand,
    OperatorCommandBinding,
    OperatorProfileCfg,
)
from motrix_deploy.runtime.lifecycle import Controller
from motrix_env_core.input import CommandBinding, InputDevice

PROGRAM_ENTRY_POINT_GROUP = "motrix_deploy.programs"


@dataclass(frozen=True)
class DeploymentProgram:
    """Fully loaded persistent controller plus its backend-neutral contracts."""

    robot: RobotSpec
    control: ControlSpec
    controller: Controller[OperatorCommand]
    command_lower: FloatArray
    command_upper: FloatArray
    operator: OperatorProfileCfg

    def prepare(self, device: InputDevice | None) -> CommandBinding[OperatorCommand]:
        if device is None:
            return ConstantOperatorCommandBinding()
        return OperatorCommandBinding(
            device,
            self.operator,
            command_lower=self.command_lower,
            command_upper=self.command_upper,
        )


def create_program(config: Any) -> DeploymentProgram:
    """Load a program factory by the typed config's stable name."""
    name = getattr(config, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError("program config must provide a non-empty name")
    matches = [entry for entry in metadata.entry_points(group=PROGRAM_ENTRY_POINT_GROUP) if entry.name == name]
    if len(matches) != 1:
        available = sorted(entry.name for entry in metadata.entry_points(group=PROGRAM_ENTRY_POINT_GROUP))
        raise ValueError(f"Deployment program {name!r} has {len(matches)} providers; available: {available}")
    factory: Callable[[Any], DeploymentProgram] = matches[0].load()
    program = factory(config)
    if not isinstance(program, DeploymentProgram):
        raise TypeError(f"Deployment program {name!r} returned {type(program).__name__}")
    return program


__all__ = ["DeploymentProgram", "create_program"]
