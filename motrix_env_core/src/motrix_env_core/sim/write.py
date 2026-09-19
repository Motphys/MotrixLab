# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Declarative sim writes: the mirror of :mod:`motrix_env_core.sim.read`.

Reads compile into one bulk read program; writes compile into one write
program per execution point. An environment declares the writes it needs
(``SimWrite`` ops), compiles them once through the backend's write
compiler, then fills the program-owned value buffers and executes at the
right moment — per control step, at reset, or between physics substeps.

Reset-before-write and forward-kinematics behavior are fixed by
:meth:`SimWriteCompiler.compile`; execution only selects rows. Target names
are validated at compile time and fail loudly.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


class SimWrite(abc.ABC):
    """Base of every declarative sim write op."""

    @abc.abstractmethod
    def compile_with(self, compiler: SimWriteCompiler, name: str) -> None:
        """Record this declaration on the compiler through its typed hook."""


@dataclass(frozen=True)
class BodyJointPositionWrite(SimWrite):
    """One body's articulated-joint position write."""

    body: str

    def compile_with(self, compiler: SimWriteCompiler, name: str) -> None:
        compiler.compile_body_joint_position(name, self)


@dataclass(frozen=True)
class BodyJointVelocityWrite(SimWrite):
    """One body's articulated-joint velocity write."""

    body: str

    def compile_with(self, compiler: SimWriteCompiler, name: str) -> None:
        compiler.compile_body_joint_velocity(name, self)


@dataclass(frozen=True)
class JointPositionWrite(SimWrite):
    """Position write for declared one-DOF joints."""

    joints: tuple[str, ...]

    def compile_with(self, compiler: SimWriteCompiler, name: str) -> None:
        compiler.compile_joint_position(name, self)


@dataclass(frozen=True)
class JointVelocityWrite(SimWrite):
    """Velocity write for declared one-DOF joints."""

    joints: tuple[str, ...]

    def compile_with(self, compiler: SimWriteCompiler, name: str) -> None:
        compiler.compile_joint_velocity(name, self)


@dataclass(frozen=True)
class JointQuaternionWrite(SimWrite):
    """Local orientation quaternions (xyzw) of declared ball joints: ``(N, J, 4)``.

    The backend normalizes each quaternion on write.
    """

    joints: tuple[str, ...]

    def compile_with(self, compiler: SimWriteCompiler, name: str) -> None:
        compiler.compile_joint_quaternion(name, self)


@dataclass(frozen=True)
class JointAngularVelocityWrite(SimWrite):
    """Local angular velocities of declared ball joints: ``(N, J, 3)``."""

    joints: tuple[str, ...]

    def compile_with(self, compiler: SimWriteCompiler, name: str) -> None:
        compiler.compile_joint_angular_velocity(name, self)


@dataclass(frozen=True)
class CtrlTargetsWrite(SimWrite):
    """Actuator ctrl targets in declared name order.

    ``actuators=None`` selects every actuator in canonical order. An explicit
    tuple produces a local ``(num_envs, len(actuators))`` float32 buffer whose
    columns are routed to those named actuators at compile time.
    """

    actuators: tuple[str, ...] | None = None

    def compile_with(self, compiler: SimWriteCompiler, name: str) -> None:
        compiler.compile_ctrl_targets(name, self)


@dataclass(frozen=True)
class BodyPositionWrite(SimWrite):
    """Floating-base world positions in declared body order: ``(N, B, 3)``."""

    bodies: tuple[str, ...]

    def compile_with(self, compiler: SimWriteCompiler, name: str) -> None:
        compiler.compile_body_position(name, self)


@dataclass(frozen=True)
class BodyRotationWrite(SimWrite):
    """Floating-base world quaternions in declared body order: ``(N, B, 4)``."""

    bodies: tuple[str, ...]

    def compile_with(self, compiler: SimWriteCompiler, name: str) -> None:
        compiler.compile_body_rotation(name, self)


@dataclass(frozen=True)
class BodyLinearVelocityWrite(SimWrite):
    """Floating-base world linear velocities in body order: ``(N, B, 3)``."""

    bodies: tuple[str, ...]

    def compile_with(self, compiler: SimWriteCompiler, name: str) -> None:
        compiler.compile_body_linear_velocity(name, self)


@dataclass(frozen=True)
class BodyAngularVelocityWrite(SimWrite):
    """Floating-base world angular velocities in body order: ``(N, B, 3)``."""

    bodies: tuple[str, ...]

    def compile_with(self, compiler: SimWriteCompiler, name: str) -> None:
        compiler.compile_body_angular_velocity(name, self)


@dataclass(frozen=True)
class KinematicBodyPositionWrite(SimWrite):
    """Kinematic body world positions in declared order: ``(N, B, 3)`` float32."""

    bodies: tuple[str, ...]

    def compile_with(self, compiler: SimWriteCompiler, name: str) -> None:
        compiler.compile_kinematic_body_position(name, self)


@dataclass(frozen=True)
class KinematicBodyRotationWrite(SimWrite):
    """Kinematic body world quaternions in declared order: ``(N, B, 4)`` float32."""

    bodies: tuple[str, ...]

    def compile_with(self, compiler: SimWriteCompiler, name: str) -> None:
        compiler.compile_kinematic_body_rotation(name, self)


@dataclass(frozen=True)
class ActuatorKpWrite(SimWrite):
    """Actuator kp overrides in declared order: ``(N, A)``."""

    actuators: tuple[str, ...]

    def compile_with(self, compiler: SimWriteCompiler, name: str) -> None:
        compiler.compile_actuator_kp(name, self)


@dataclass(frozen=True)
class ActuatorDampingWrite(SimWrite):
    """Actuator damping overrides in declared order: ``(N, A)``."""

    actuators: tuple[str, ...]

    def compile_with(self, compiler: SimWriteCompiler, name: str) -> None:
        compiler.compile_actuator_damping(name, self)


@dataclass(frozen=True)
class BodyMassWrite(SimWrite):
    """Link mass overrides in declared order: ``(N, L)``."""

    links: tuple[str, ...]

    def compile_with(self, compiler: SimWriteCompiler, name: str) -> None:
        compiler.compile_body_mass(name, self)


@dataclass(frozen=True)
class BodyComWrite(SimWrite):
    """Link center-of-mass overrides in declared order: ``(N, L, 3)``."""

    links: tuple[str, ...]

    def compile_with(self, compiler: SimWriteCompiler, name: str) -> None:
        compiler.compile_body_com(name, self)


@dataclass(frozen=True)
class GeomFrictionWrite(SimWrite):
    """Geom friction overrides in declared order: ``(N, G, 3)``."""

    geoms: tuple[str, ...]

    def compile_with(self, compiler: SimWriteCompiler, name: str) -> None:
        compiler.compile_geom_friction(name, self)


class SimWriteCompiler(abc.ABC):
    """Compile named neutral write declarations into an executable program.

    ``compile`` is the only consumer entry point. The ``compile_*`` methods
    are the backend side of the internal dispatch protocol —
    :meth:`SimWrite.compile_with` routes each declaration to its typed hook,
    which records the backend op for the program being compiled. They form a
    backend SPI, not a consumer API.
    """

    def compile(
        self,
        writes: Mapping[str, SimWrite],
        *,
        reset: bool = False,
        forward_kinematics: bool = True,
    ) -> WriteProgram:
        """Compile a named write set with fixed execution semantics.

        Dispatches every declaration to its ``compile_*`` hook in order, then
        assembles the program from the recorded ops. ``reset=True`` restores
        selected rows to backend defaults before applying the writes.
        ``forward_kinematics`` controls whether execution refreshes derived
        kinematic state when the compiled program can make it stale.
        """
        self._begin_compile()
        for name, write in writes.items():
            write.compile_with(self, name)
        return self._build_program(reset=reset, forward_kinematics=forward_kinematics)

    def _begin_compile(self) -> None:
        """Reset per-compile accumulation before dispatch; default no-op."""

    @abc.abstractmethod
    def _build_program(self, *, reset: bool, forward_kinematics: bool) -> WriteProgram:
        """Assemble the program from the ops recorded during dispatch."""

    @abc.abstractmethod
    def compile_body_joint_position(self, name: str, write: BodyJointPositionWrite) -> None:
        """Record one body's articulated DOF position write."""

    @abc.abstractmethod
    def compile_body_joint_velocity(self, name: str, write: BodyJointVelocityWrite) -> None:
        """Record one body's articulated DOF velocity write."""

    @abc.abstractmethod
    def compile_joint_position(self, name: str, write: JointPositionWrite) -> None:
        """Record named one-DOF joint position writes."""

    @abc.abstractmethod
    def compile_joint_velocity(self, name: str, write: JointVelocityWrite) -> None:
        """Record named one-DOF joint velocity writes."""

    @abc.abstractmethod
    def compile_joint_quaternion(self, name: str, write: JointQuaternionWrite) -> None:
        """Record ball-joint local orientation writes."""

    @abc.abstractmethod
    def compile_joint_angular_velocity(self, name: str, write: JointAngularVelocityWrite) -> None:
        """Record ball-joint local angular velocity writes."""

    @abc.abstractmethod
    def compile_ctrl_targets(self, name: str, write: CtrlTargetsWrite) -> None:
        """Record actuator control targets."""

    @abc.abstractmethod
    def compile_body_position(self, name: str, write: BodyPositionWrite) -> None:
        """Record floating-body world position writes."""

    @abc.abstractmethod
    def compile_body_rotation(self, name: str, write: BodyRotationWrite) -> None:
        """Record floating-body world rotation writes."""

    @abc.abstractmethod
    def compile_body_linear_velocity(self, name: str, write: BodyLinearVelocityWrite) -> None:
        """Record floating-body world linear velocity writes."""

    @abc.abstractmethod
    def compile_body_angular_velocity(self, name: str, write: BodyAngularVelocityWrite) -> None:
        """Record floating-body world angular velocity writes."""

    @abc.abstractmethod
    def compile_kinematic_body_position(self, name: str, write: KinematicBodyPositionWrite) -> None:
        """Record kinematic-body world position writes."""

    @abc.abstractmethod
    def compile_kinematic_body_rotation(self, name: str, write: KinematicBodyRotationWrite) -> None:
        """Record kinematic-body world rotation writes."""

    @abc.abstractmethod
    def compile_actuator_kp(self, name: str, write: ActuatorKpWrite) -> None:
        """Record actuator kp overrides."""

    @abc.abstractmethod
    def compile_actuator_damping(self, name: str, write: ActuatorDampingWrite) -> None:
        """Record actuator damping overrides."""

    @abc.abstractmethod
    def compile_body_mass(self, name: str, write: BodyMassWrite) -> None:
        """Record body mass overrides."""

    @abc.abstractmethod
    def compile_body_com(self, name: str, write: BodyComWrite) -> None:
        """Record body center-of-mass overrides."""

    @abc.abstractmethod
    def compile_geom_friction(self, name: str, write: GeomFrictionWrite) -> None:
        """Record geom friction overrides."""


class WriteProgram(abc.ABC):
    """Compiled set of writes executed together at one point in the loop.

    Owns one value buffer per declared write; environments fill buffers
    (row-scoped ops read only the ``execute(env_ids)`` rows) and call
    :meth:`execute` to push everything to the backend in one crossing.
    """

    @abc.abstractmethod
    def buffer(self, name: str) -> np.ndarray:
        """Return the program-owned buffer of one declaration."""

    @abc.abstractmethod
    def execute(self, env_ids: np.ndarray | None = None) -> None:
        """Write every op's buffer to the backend.

        ``env_ids`` selects rows; ``None`` means every row. Reset-before-write
        and derived-state refresh behavior are fixed when the program is compiled.
        """
