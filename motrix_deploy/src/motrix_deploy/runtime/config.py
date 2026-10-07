# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Simulator-independent deployment scene and state-source configuration."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from motrix_deploy.errors import ValidationError
from motrix_env_core.config.scene import SceneCfg
from motrix_env_core.config.sim import SimCfg


@dataclass(frozen=True)
class SensorBindings:
    """Semantic robot state fields bound to exact model-local sensor names."""

    base_angular_velocity: str | None = None
    base_linear_acceleration: str | None = None
    base_linear_velocity: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "base_angular_velocity",
            "base_linear_acceleration",
            "base_linear_velocity",
        ):
            name = getattr(self, field_name)
            if name is None:
                continue
            if not isinstance(name, str):
                raise TypeError(f"{field_name} must be a string or None")
            if not name.strip():
                raise ValueError(f"{field_name} must be a non-empty sensor name")


@dataclass(frozen=True)
class SimulationRuntimeConfig:
    scene: SceneCfg
    physics: SimCfg = field(default_factory=lambda: SimCfg(dt=0.002, solver_iterations=100))
    render: bool = False
    realtime: bool | None = None
    # Deployment recipe: semantic RobotState fields mapped to model-local sensor names.
    sensor_bindings: SensorBindings = field(default_factory=SensorBindings)

    def __post_init__(self) -> None:
        self.physics.validate()

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, render: bool = False, realtime: bool | None = None
    ) -> "SimulationRuntimeConfig":
        """Decode an assembled simulation recipe at the application boundary."""
        values = dict(value)
        values.pop("name", None)
        values.setdefault("sensor_bindings", {})
        expected = {"scene", "physics", "sensor_bindings"}
        if set(values) != expected:
            raise ValidationError(
                "runtime",
                f"fields {sorted(expected)}",
                f"missing={sorted(expected - set(values))}, unknown={sorted(set(values) - expected)}",
            )
        if not isinstance(values["scene"], SceneCfg):
            raise ValidationError("runtime.scene", "a complete SceneCfg", values["scene"])
        missing = {"dt", "solver_iterations"} - set(values["physics"])
        if missing:
            raise ValidationError("runtime.physics", "dt and solver_iterations", f"missing={sorted(missing)}")
        return cls(
            scene=values["scene"],
            physics=SimCfg(**values["physics"]),
            sensor_bindings=SensorBindings(**values["sensor_bindings"]),
            render=render,
            realtime=realtime,
        )
