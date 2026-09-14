# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""The typed model surface, declared model-metadata queries, and dispatch.

The surface types (:class:`SimModel`, :class:`BodyModel`, :class:`ActuatorSpec`,
...) define what every backend must produce as ``env.model``; the query
classes declare environment-owned metadata lookups; the compiler base wires
the two together. Runtime behavior lives in ``sim.backend``.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from motrix_env_core.config.scene import SceneCfg


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


@dataclass(frozen=True, slots=True)
class BodyModel:
    """Per-body static model data resolved from one ``BodyCfg`` at compile time.

    Two alignment axes govern every array field:

    - **joint axis**: ``joint_names`` lists the body's named joints in the
      backend's joint-DOF order; the floating base is not a named joint and
      is not included. ``joint_pos_limits`` and ``init_joint_pos`` are
      aligned to this order;
    - **link axis**: ``init_link_positions`` and ``init_link_quats`` are
      aligned to ``link_names``.

    The joint/link orderings are the backend's own and may differ across
    backends for the same scene — consumers must resolve joints and links
    by name, never by hard-coded index (precompute ``.index(name)`` in ``__init__``).

    The container carries only the init snapshot: joint angles come from the
    key pose designated by the originating ``RobotCfg.init_key_pose``
    (engine defaults for bodies without key poses), and link poses from one
    forward-kinematics evaluation of those angles with the body at its
    default placement, in world frame. Non-init key poses stay in the cfg;
    consumers that need them read ``RobotCfg.key_pose`` directly.

    Attributes:
        name: The ``SceneObjsCfg`` field name of the body (e.g. ``"robot"``);
            identical across backends for the same scene, so it is the stable
            addressing key into ``SimModel.bodies``.
        base_link_name: Resolved engine name of the attach root link (after
            the cfg's prefix/suffix decoration); backend-specific.
        link_names: All links of the body, including the base link; the
            alignment axis of the FK arrays.
        joint_names: The body's named joints in joint-DOF order; the
            alignment axis of every joint-array field.
        joint_pos_limits: ``(lower, upper)`` float32 arrays aligned to
            ``joint_names``, or ``None`` when the body declares no joints.
        actuators: Body-scoped view of ``SimModel.actuators``: the specs
            whose ``target_name`` is one of this body's joints, keeping the
            scene-wide engine model order. Actuators outside every body
            (e.g. declared in the base scene file) appear only globally.
        init_base_position: ``(3,)`` float32 world-frame position of the
            default base placement from the originating ``BodyCfg``.
        init_base_quat: ``(4,)`` float32 ``(x, y, z, w)`` orientation of the
            default base placement; a unit quaternion.
        init_joint_pos: ``(len(joint_names),)`` float32 init joint angles
            from the designated init key pose (permuted into body order),
            or zero angles for bodies without key poses.
        init_link_positions: ``(num_links, 3)`` float32 world-frame link
            positions from the init-pose FK evaluation.
        init_link_quats: ``(num_links, 4)`` float32 world-frame link
            rotations in ``(x, y, z, w)`` order from the init-pose FK
            evaluation; each row is a unit quaternion.
    """

    name: str
    base_link_name: str
    link_names: tuple[str, ...]
    joint_names: tuple[str, ...]
    joint_pos_limits: tuple[np.ndarray, np.ndarray] | None
    actuators: tuple[ActuatorSpec, ...]
    init_base_position: np.ndarray
    init_base_quat: np.ndarray
    init_joint_pos: np.ndarray
    init_link_positions: np.ndarray
    init_link_quats: np.ndarray


@dataclass(frozen=True)
class SimModel:
    """Typed model surface every environment consumes as ``env.model``.

    The typed fields are the required core metadata: backends must fill them,
    while simulator layout remains encapsulated behind declared queries and
    programs. ``others`` carries the resolved results of the environment's
    declared :class:`ModelQuery` set — present keys are exactly what the
    environment declared.
    """

    actuators: tuple[ActuatorSpec, ...]
    init_dof_pos: np.ndarray
    others: Mapping[str, Any] = field(default_factory=dict)
    # Keyed by SceneObjsCfg field name; backends assemble entries at compile
    # time from their engine model plus the originating BodyCfg.
    bodies: Mapping[str, BodyModel] = field(default_factory=dict)


class ModelQuery(abc.ABC):
    """Base class for declared static model-metadata queries."""

    @abc.abstractmethod
    def compile_with(self, compiler: SimModelCompiler, *, key: str) -> None:
        """Record this declaration on the compiler through its typed hook."""


@dataclass(frozen=True)
class GeomSpecsQuery(ModelQuery):
    """Selected ``GeomSpec`` values keyed by geom name."""

    names: tuple[str, ...]

    def compile_with(self, compiler: SimModelCompiler, *, key: str) -> None:
        compiler.compile_geom_specs(key, self.names)


@dataclass(frozen=True)
class BodyJointPositionLimitsQuery(ModelQuery):
    """``(lower, upper)`` float32 arrays in one body's joint-DOF order."""

    body: str

    def compile_with(self, compiler: SimModelCompiler, *, key: str) -> None:
        compiler.compile_body_joint_position_limits(key, self.body)


@dataclass(frozen=True)
class DofPositionLimitsQuery(ModelQuery):
    """``(lower, upper)`` float32 arrays in global DOF-position order."""

    def compile_with(self, compiler: SimModelCompiler, *, key: str) -> None:
        compiler.compile_dof_position_limits(key)


@dataclass(frozen=True)
class ActuatorKpQuery(ModelQuery):
    """Nominal position gains in declared-name or full model order."""

    names: tuple[str, ...] | None = None

    def compile_with(self, compiler: SimModelCompiler, *, key: str) -> None:
        compiler.compile_actuator_kp(key, self.names)


@dataclass(frozen=True)
class ActuatorKdQuery(ModelQuery):
    """Nominal damping gains in declared-name or full model order."""

    names: tuple[str, ...] | None = None

    def compile_with(self, compiler: SimModelCompiler, *, key: str) -> None:
        compiler.compile_actuator_kd(key, self.names)


@dataclass(frozen=True)
class BodyMassQuery(ModelQuery):
    """Scalar ``float`` nominal mass of one named link."""

    name: str

    def compile_with(self, compiler: SimModelCompiler, *, key: str) -> None:
        compiler.compile_body_mass(key, self.name)


@dataclass(frozen=True)
class BodyCenterOfMassQuery(ModelQuery):
    """``(3,)`` float32 nominal center-of-mass offset of one named link."""

    name: str

    def compile_with(self, compiler: SimModelCompiler, *, key: str) -> None:
        compiler.compile_body_center_of_mass(key, self.name)


@dataclass(frozen=True)
class GeomFrictionQuery(ModelQuery):
    """``(3,)`` float32 nominal friction parameters."""

    name: str

    def compile_with(self, compiler: SimModelCompiler, *, key: str) -> None:
        compiler.compile_geom_friction(key, self.name)


class SimModelCompiler(abc.ABC):
    """Assemble the backend-neutral model surface for one scene and query set.

    ``compile`` receives both inputs that meet in :class:`SimModel`: the
    scene configuration drives the unconditional surface (core facts and
    ``bodies``), while the declared queries drive ``others``. Backends bind
    the compiler to their engine model at construction.
    """

    def compile(self, scene: SceneCfg, queries: Mapping[str, ModelQuery]) -> SimModel:
        """Compile the scene surface and every named metadata query.

        Args:
            scene: The scene configuration the bound engine model was
                compiled from; it drives the unconditional model surface.
            queries: Model queries keyed by their logical result names.

        Returns:
            The backend-neutral simulator model and compiled metadata values.
        """
        self._begin_compile()
        for key, query in queries.items():
            query.compile_with(self, key=key)
        return self._build_model(scene)

    def _begin_compile(self) -> None:
        """Reset per-compile accumulation before dispatch; default no-op."""

    @abc.abstractmethod
    def _build_model(self, scene: SceneCfg) -> SimModel:
        """Assemble the model from the scene and the values recorded in dispatch.

        Returns:
            The backend-neutral simulator model.
        """

    @abc.abstractmethod
    def compile_geom_specs(self, key: str, geom_names: tuple[str, ...]) -> None:
        """Compile geometry specifications for an ordered set of geometries.

        Args:
            key: Logical key under which the result is stored.
            geom_names: Ordered geometry names to inspect.
        """

    @abc.abstractmethod
    def compile_body_joint_position_limits(self, key: str, body: str) -> None:
        """Compile joint-position limits in one body's joint-DOF order.

        Args:
            key: Logical key under which the result is stored.
            body: Name of the body whose joint limits are read.
        """

    @abc.abstractmethod
    def compile_dof_position_limits(self, key: str) -> None:
        """Compile position limits in canonical DOF-position order.

        Args:
            key: Logical key under which the result is stored.
        """

    @abc.abstractmethod
    def compile_actuator_kp(self, key: str, actuator_names: tuple[str, ...] | None) -> None:
        """Compile nominal proportional gains for selected or all actuators.

        Args:
            key: Logical key under which the result is stored.
            actuator_names: Ordered actuator names, or ``None`` for all actuators.
        """

    @abc.abstractmethod
    def compile_actuator_kd(self, key: str, actuator_names: tuple[str, ...] | None) -> None:
        """Compile nominal damping gains for selected or all actuators.

        Args:
            key: Logical key under which the result is stored.
            actuator_names: Ordered actuator names, or ``None`` for all actuators.
        """

    @abc.abstractmethod
    def compile_body_mass(self, key: str, body: str) -> None:
        """Compile the nominal mass of one body link.

        Args:
            key: Logical key under which the result is stored.
            body: Name of the body link whose mass is read.
        """

    @abc.abstractmethod
    def compile_body_center_of_mass(self, key: str, body: str) -> None:
        """Compile the nominal center-of-mass offset of one body link.

        Args:
            key: Logical key under which the result is stored.
            body: Name of the body link whose center of mass is read.
        """

    @abc.abstractmethod
    def compile_geom_friction(self, key: str, geom: str) -> None:
        """Compile nominal friction parameters for one geometry.

        Args:
            key: Logical key under which the result is stored.
            geom: Name of the geometry whose friction is read.
        """
