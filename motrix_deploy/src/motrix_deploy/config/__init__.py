# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Typed application configuration for the deployment CLI.

The simulated runtime recipe is a structured schema installed in Hydra's
ConfigStore ``runtime`` group, mirroring how RL providers install their
algorithm schemas: ``runtime/sim.yaml`` composes ``_sim_schema`` so a task
recipe's runtime section is typed and validated at compose time. Hardware
runtimes stay a plain mapping because their schema is owned by the installed
backend plugin, which decodes it strictly at the factory boundary.
"""

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from motrix_deploy.env import assemble_deploy_scene
from motrix_deploy.runtime.config import SensorBindings, SimulationRuntimeConfig
from motrix_env_core.config.sim import SimCfg

SIM_SCHEMA_GROUP = "runtime"


@dataclass
class HardwareRuntimeCfg:
    """Base schema for hardware runtime recipes; plugins subclass with their fields.

    The schema type itself discriminates hardware from simulation recipes.
    """

    backend: str = MISSING


@dataclass
class SimRuntimeCfg:
    """Structured schema for one simulated deployment runtime recipe."""

    backend: str = MISSING
    viewer: bool = True
    realtime: bool | None = None
    deploy_env_id: str = MISSING
    robot_id: str = MISSING
    robot_init_key_pose: str | None = None
    robot_translation: tuple[float, float, float] | None = None
    robot_rotation: tuple[float, float, float, float] | None = None
    physics: SimCfg = field(default_factory=lambda: SimCfg(dt=0.002, solver_iterations=100))
    sensor_bindings: SensorBindings = field(default_factory=SensorBindings)

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("runtime.backend must be a non-empty string")
        if not self.deploy_env_id or not self.robot_id:
            raise ValueError("runtime.deploy_env_id and runtime.robot_id must be non-empty registered IDs")
        self.physics.validate()

    def build_runtime_config(self) -> SimulationRuntimeConfig:
        """Assemble the recipe's scene and decode it into the frozen runtime contract."""
        scene = assemble_deploy_scene(self.deploy_env_id, self.robot_id)
        if self.robot_init_key_pose is not None:
            scene.objs.robot.init_key_pose = self.robot_init_key_pose
        for field_name in ("translation", "rotation"):
            value = getattr(self, f"robot_{field_name}")
            if value is not None:
                setattr(scene.objs.robot, field_name, tuple(value))
        return SimulationRuntimeConfig(
            scene=scene,
            physics=self.physics,
            sensor_bindings=self.sensor_bindings,
            render=self.viewer,
            realtime=self.viewer if self.realtime is None else self.realtime,
        )


@dataclass
class DeployRunConfig:
    """Resolved execution settings with one selected runtime configuration."""

    artifact: str
    # Sim recipes compose to SimRuntimeCfg; hardware recipes compose to a
    # HardwareRuntimeCfg subclass owned by the backend plugin. The union is not
    # expressible in OmegaConf, so the CLI narrows it by isinstance.
    runtime: Any = MISSING
    duration_s: float | None = None

    def __post_init__(self) -> None:
        if not self.artifact:
            raise ValueError("artifact must be a non-empty path")
        if not isinstance(self.runtime, (SimRuntimeCfg, HardwareRuntimeCfg)):
            raise ValueError(
                "runtime must be a composed SimRuntimeCfg or HardwareRuntimeCfg recipe, "
                f"got {type(self.runtime).__name__}"
            )


@dataclass
class ProgramRunConfig:
    """Resolved persistent multi-policy execution settings."""

    program: Any = MISSING
    runtime: Any = MISSING

    def __post_init__(self) -> None:
        if self.program is MISSING or not getattr(self.program, "name", None):
            raise ValueError("program must be a composed program recipe")
        if not isinstance(self.runtime, (SimRuntimeCfg, HardwareRuntimeCfg)):
            raise ValueError(
                "runtime must be a composed SimRuntimeCfg or HardwareRuntimeCfg recipe, "
                f"got {type(self.runtime).__name__}"
            )


def _install_sim_runtime_schema() -> None:
    ConfigStore.instance().store(group=SIM_SCHEMA_GROUP, name="_sim_schema", node=SimRuntimeCfg)


_install_sim_runtime_schema()

__all__ = [
    "DeployRunConfig",
    "HardwareRuntimeCfg",
    "ProgramRunConfig",
    "SIM_SCHEMA_GROUP",
    "SimRuntimeCfg",
]
