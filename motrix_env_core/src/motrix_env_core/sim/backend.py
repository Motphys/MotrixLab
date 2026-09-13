# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral simulator boundary consumed by environment frontends.

The :class:`SimBackend` interface owns simulator construction, model metadata,
read-program compilation, and write-program compilation. Runtime simulator
objects and backend-specific types remain behind this boundary; environments
interact with the backend through the shared query and write interfaces.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from motrix_env_core.config import SimCfg
from motrix_env_core.config.scene import SceneCfg, SystemCameraCfg
from motrix_env_core.sim.model import ModelQuery, SimModelQueryCompiler
from motrix_env_core.sim.read import PhysicsReadProgram
from motrix_env_core.sim.write import SimWrite, SimWriteCompiler, WriteProgram


class ActuatorType(str, Enum):
    """Supported actuator control semantics."""

    POSITION = "position"
    VELOCITY = "velocity"
    MOTOR = "motor"
    GENERAL = "general"
    ADHESION = "adhesion"


@dataclass(frozen=True)
class ActuatorSpec:
    """Static per-actuator metadata resolved from the simulator model."""

    name: str
    actuator_type: ActuatorType
    target_name: str
    ctrl_range: tuple[float, float] | None
    force_range: tuple[float, float] | None


@dataclass(frozen=True)
class GeomSpec:
    """Static per-geom metadata resolved from the simulator model."""

    size: tuple[float, ...] | None
    local_pose: tuple[float, ...] | None


@dataclass(frozen=True)
class SimModel:
    """Typed model surface every environment consumes as ``env.model``.

    The typed fields are the required core metadata: backends must fill them,
    while simulator layout remains encapsulated behind declared queries and programs.
    ``others`` carries the resolved results of the environment's declared
    :class:`~motrix_env_core.sim.model.ModelQuery` set — present keys
    are exactly what the environment declared.
    """

    actuators: tuple[ActuatorSpec, ...]
    init_dof_pos: np.ndarray
    others: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderConfig:
    """Rendering settings; ``headless`` selects windowed vs offscreen mode.

    Interactive (``headless=False``) rendering opens a viewer window. Headless
    rendering captures offscreen frames and records them to a video file, so
    ``path`` / ``fps`` / ``num_frames`` are required in that mode.
    """

    headless: bool = False
    path: Path | None = None
    fps: int | None = None
    num_frames: int | None = None
    width: int = 256
    height: int = 256
    camera_lookat: Sequence[float] | None = None
    camera_distance: float | None = None
    camera_elevation: float | None = None
    camera_azimuth: float | None = None

    def __post_init__(self) -> None:
        if not self.headless:
            if self.path is not None:
                raise ValueError("path is a headless recording field; pass headless=True or remove it.")
            return
        if self.path is None:
            raise ValueError("headless recording requires a path.")
        if self.fps is None or self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.num_frames is None or self.num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {self.num_frames}")
        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"height must be positive, got {self.height}")
        if self.width % 2 != 0:
            raise ValueError(f"width must be even for yuv420p encoding, got {self.width}")
        if self.height % 2 != 0:
            raise ValueError(f"height must be even for yuv420p encoding, got {self.height}")


class SimRenderer(abc.ABC):
    """Backend-owned renderer bound to one environment's live state.

    Renderers bind the backend's model and simulator data at creation and pull
    state themselves; the frontend never touches simulator render types. The
    rendering mode is fixed by the :class:`RenderConfig` used at creation:
    interactive renderers present frames through :meth:`render`, headless
    renderers return frames through :meth:`capture`. :meth:`render`
    presents one frame in both modes (windowed viewer input, or an offscreen
    sync when headless). :meth:`capture` returns frames and exists only
    in headless mode: windowed renderers set no system render target, so there
    are no pixels to return.
    """

    @abc.abstractmethod
    def render(self) -> None:
        """Present one frame from the current simulator state (sync + viewer input)."""

    @abc.abstractmethod
    def capture(self) -> np.ndarray:
        """Return the current system-camera view as an HxWx3 uint8 image (headless only)."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release renderer resources."""


class SimBackend(abc.ABC):
    """One simulator: construction-time scene translation plus live behavior.

    Constructing a backend with ``(scene, sim, num_envs)`` compiles the
    neutral ``SceneCfg`` into the simulator's own model and batched data —
    scene compilation is the backend's internal affair and no compiled
    artifact crosses this boundary. Afterwards the backend serves both
    faces: static translation (:attr:`model_query_compiler`,
    :meth:`compile_reads`, :attr:`write_compiler`) and live
    behavior (:meth:`step`, :meth:`reset`, shape properties).
    """

    name: str

    def __init__(self, scene: SceneCfg, sim: SimCfg, num_envs: int) -> None:
        """Compile ``scene`` and allocate ``num_envs`` batched data rows.

        Concrete backends own this contract: they must finish all scene
        translation here so every member below is usable immediately after
        construction. ``num_envs`` fixes the batch width for the backend's
        lifetime.
        """
        del scene, sim, num_envs

    @property
    @abc.abstractmethod
    def num_dof_pos(self) -> int:
        """Number of dof-position channels in the canonical DOF order."""

    @property
    @abc.abstractmethod
    def num_dof_vel(self) -> int:
        """Number of dof-velocity channels in the canonical DOF order."""

    @property
    @abc.abstractmethod
    def num_actuators(self) -> int:
        """Number of actuators in the canonical actuator order."""

    @abc.abstractmethod
    def compile_reads(self, queries) -> PhysicsReadProgram:
        """Lower the complete declared query set into one fixed read program.

        ``queries`` carries concrete simulator-data query declarations;
        duplicates may be present. The backend owns
        the memory planning: it must serve ``view(key)`` for **every** declared
        key — equal queries may share one physical region — and each served
        view must be float32 with leading dimension ``num_envs``.
        """

    @abc.abstractmethod
    def step(self, substeps: int) -> None:
        """Advance every row by ``substeps`` physics steps.

        Time advancement belongs exclusively to the control/training loop;
        reset never steps physics.
        """

    def create_renderer(
        self,
        config: RenderConfig,
        *,
        num_envs: int,
        render_spacing: float,
        system_camera: SystemCameraCfg,
    ) -> SimRenderer:
        """Create this backend's renderer using its own model and data.

        Optional capability: render-less backends report the gap loudly.
        """
        raise NotImplementedError(f"{type(self).__name__} does not provide rendering")

    def sample_terrain_height(self, geom_name: str, env_ids: np.ndarray, xy: np.ndarray) -> np.ndarray:
        """Sample terrain height at ``(rows, ..., 2)`` query points for selected rows.

        Optional capability covering both height-field and flat ground geoms;
        backends without terrain sampling report the gap loudly.
        """
        raise NotImplementedError(f"{type(self).__name__} does not provide terrain sampling")

    @property
    @abc.abstractmethod
    def model_query_compiler(self) -> SimModelQueryCompiler:
        """Return the compiler bound to this backend's static model."""
        raise NotImplementedError(f"{type(self).__name__} does not provide model query compilation")

    def compile_model(self, queries: Mapping[str, ModelQuery]) -> SimModel:
        """Lower static model-query declarations into one backend-neutral model."""
        return self.model_query_compiler.compile(queries)

    def compile_writes(
        self,
        writes: Mapping[str, SimWrite],
        *,
        reset: bool = False,
        forward_kinematics: bool = True,
    ) -> WriteProgram:
        """Lower named write declarations into one executable write program.

        Convenience twin of :meth:`compile_reads` and :meth:`compile_model`
        for the write path: it forwards to :attr:`write_compiler`, so
        backends without live write support report the gap when the property
        is accessed.
        """
        return self.write_compiler.compile(writes, reset=reset, forward_kinematics=forward_kinematics)

    @property
    def write_compiler(self) -> SimWriteCompiler:
        """Return the compiler bound to this backend's model and live data.

        Optional capability: backends without live write support report the
        gap when the property is accessed.
        """
        raise NotImplementedError(f"{type(self).__name__} does not provide write compilation")


__all__ = [
    "ActuatorSpec",
    "ActuatorType",
    "GeomSpec",
    "RenderConfig",
    "SimBackend",
    "SimModel",
    "SimRenderer",
]
