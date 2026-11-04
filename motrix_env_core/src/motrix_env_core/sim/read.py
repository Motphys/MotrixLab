# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt


# Name fields hold concrete strings so query dataclasses remain valid,
# serializable declarations before backend compilation.
@dataclass(frozen=True, kw_only=True)
class SimDataQuery(abc.ABC):
    """Backend-neutral descriptor for one fixed simulator quantity."""

    @abc.abstractmethod
    def compile_with(self, compiler: SimDataQueryCompiler, *, key: str) -> None:
        """Dispatch this query to its typed backend compilation method."""


@dataclass(frozen=True, kw_only=True)
class BodyJointPositionQuery(SimDataQuery):
    """Articulated DOF positions for one body, excluding its floating base."""

    body: str

    def compile_with(self, compiler: SimDataQueryCompiler, *, key: str) -> None:
        compiler.compile_body_joint_position(key, self.body)


@dataclass(frozen=True, kw_only=True)
class BodyJointVelocityQuery(SimDataQuery):
    """Articulated DOF velocities for one body, excluding its floating base."""

    body: str

    def compile_with(self, compiler: SimDataQueryCompiler, *, key: str) -> None:
        compiler.compile_body_joint_velocity(key, self.body)


@dataclass(frozen=True, kw_only=True)
class JointPositionQuery(SimDataQuery):
    joints: tuple[str, ...]

    def compile_with(self, compiler: SimDataQueryCompiler, *, key: str) -> None:
        compiler.compile_joints_position(key, self.joints)


@dataclass(frozen=True, kw_only=True)
class JointVelocityQuery(SimDataQuery):
    joints: tuple[str, ...]

    def compile_with(self, compiler: SimDataQueryCompiler, *, key: str) -> None:
        compiler.compile_joints_velocity(key, self.joints)


@dataclass(frozen=True, kw_only=True)
class LinkPositionQuery(SimDataQuery):
    link: str

    def compile_with(self, compiler: SimDataQueryCompiler, *, key: str) -> None:
        compiler.compile_links_position(key, self.link)


@dataclass(frozen=True, kw_only=True)
class BatchLinkPositionQuery(SimDataQuery):
    links: tuple[str, ...]

    def compile_with(self, compiler: SimDataQueryCompiler, *, key: str) -> None:
        compiler.compile_links_position(key, self.links)


@dataclass(frozen=True, kw_only=True)
class LinkQuaternionQuery(SimDataQuery):
    link: str

    def compile_with(self, compiler: SimDataQueryCompiler, *, key: str) -> None:
        compiler.compile_links_quaternion(key, self.link)


@dataclass(frozen=True, kw_only=True)
class BatchLinkQuaternionQuery(SimDataQuery):
    links: tuple[str, ...]

    def compile_with(self, compiler: SimDataQueryCompiler, *, key: str) -> None:
        compiler.compile_links_quaternion(key, self.links)


@dataclass(frozen=True, kw_only=True)
class LinkLinearVelocityQuery(SimDataQuery):
    link: str

    def compile_with(self, compiler: SimDataQueryCompiler, *, key: str) -> None:
        compiler.compile_links_linear_velocity(key, self.link)


@dataclass(frozen=True, kw_only=True)
class BatchLinkLinearVelocityQuery(SimDataQuery):
    links: tuple[str, ...]

    def compile_with(self, compiler: SimDataQueryCompiler, *, key: str) -> None:
        compiler.compile_links_linear_velocity(key, self.links)


@dataclass(frozen=True, kw_only=True)
class LinkAngularVelocityQuery(SimDataQuery):
    link: str

    def compile_with(self, compiler: SimDataQueryCompiler, *, key: str) -> None:
        compiler.compile_links_angular_velocity(key, self.link)


@dataclass(frozen=True, kw_only=True)
class BatchLinkAngularVelocityQuery(SimDataQuery):
    links: tuple[str, ...]

    def compile_with(self, compiler: SimDataQueryCompiler, *, key: str) -> None:
        compiler.compile_links_angular_velocity(key, self.links)


@dataclass(frozen=True, kw_only=True)
class LinkNetContactForceQuery(SimDataQuery):
    link: str

    def compile_with(self, compiler: SimDataQueryCompiler, *, key: str) -> None:
        compiler.compile_links_net_contact_force(key, self.link)


@dataclass(frozen=True, kw_only=True)
class BatchLinkNetContactForceQuery(SimDataQuery):
    links: tuple[str, ...]

    def compile_with(self, compiler: SimDataQueryCompiler, *, key: str) -> None:
        compiler.compile_links_net_contact_force(key, self.links)


@dataclass(frozen=True, kw_only=True)
class BodyLinkNetContactForceQuery(SimDataQuery):
    """Read contact forces for one body's links except an explicit allow-list."""

    body: str
    exclude_links: tuple[str, ...] = ()

    def compile_with(self, compiler: SimDataQueryCompiler, *, key: str) -> None:
        compiler.compile_body_link_net_contact_force(key, self.body, self.exclude_links)


@dataclass(frozen=True, kw_only=True)
class SensorValuesQuery(SimDataQuery):
    sensors: tuple[str, ...]

    def compile_with(self, compiler: SimDataQueryCompiler, *, key: str) -> None:
        compiler.compile_sensor_values(key, self.sensors)


@dataclass(frozen=True, kw_only=True)
class RaycastCameraImageQuery(SimDataQuery):
    """One camera output: scalar ``(N,H,W)`` or world vector ``(N,H,W,3)``.

    Distances are in metres, misses are -1. Points and normals are world-space
    and zero on misses. Pixels are row-major in the optical frame (+X right,
    +Y down, +Z forward). Read-only views are updated by program execution;
    querying never advances physics or forces sensor sampling. This reads the
    backend cache: without native refresh/status support it may still contain
    pre-reset measurements, and -1 cannot distinguish unready from all misses.
    """

    sensor: str
    field: str = "depth"

    def compile_with(self, compiler: SimDataQueryCompiler, *, key: str) -> None:
        compiler.compile_raycast_camera_image(key, self.sensor, self.field)


@dataclass(frozen=True, kw_only=True)
class SitePositionQuery(SimDataQuery):
    """World position of one named site, shape ``(rows, 3)``."""

    site: str

    def compile_with(self, compiler: SimDataQueryCompiler, *, key: str) -> None:
        compiler.compile_site_position(key, self.site)


@dataclass(frozen=True, kw_only=True)
class SiteQuaternionQuery(SimDataQuery):
    """World orientation quaternion (x, y, z, w) of one named site, shape ``(rows, 4)``."""

    site: str

    def compile_with(self, compiler: SimDataQueryCompiler, *, key: str) -> None:
        compiler.compile_site_quaternion(key, self.site)


@dataclass(frozen=True, kw_only=True)
class GeomPositionQuery(SimDataQuery):
    """World position of one named geom, shape ``(rows, 3)``."""

    geom: str

    def compile_with(self, compiler: SimDataQueryCompiler, *, key: str) -> None:
        compiler.compile_geom_position(key, self.geom)


@dataclass(frozen=True, kw_only=True)
class GeomQuaternionQuery(SimDataQuery):
    """World orientation quaternion (x, y, z, w) of one named geom, shape ``(rows, 4)``."""

    geom: str

    def compile_with(self, compiler: SimDataQueryCompiler, *, key: str) -> None:
        compiler.compile_geom_quaternion(key, self.geom)


@dataclass(frozen=True, kw_only=True)
class GeomLinearVelocityQuery(SimDataQuery):
    """World-frame linear velocity of a geom (includes rotational contribution)."""

    geom: str

    def compile_with(self, compiler: SimDataQueryCompiler, *, key: str) -> None:
        compiler.compile_geom_linear_velocity(key, self.geom)


@dataclass(frozen=True, kw_only=True)
class GeomPairCollidingQuery(SimDataQuery):
    """Per-pair collision indicator for named geom pairs, shape ``(rows, num_pairs)``.

    Each pair is an ordered 2-tuple of geom names; the output carries 1.0 where
    the two geoms collide in that row and 0.0 otherwise.
    """

    # The resolved value is tuple[tuple[str, str], ...]. This stays Any because
    # OmegaConf structured configs cannot represent nested tuple annotations.
    pairs: Any

    def compile_with(self, compiler: SimDataQueryCompiler, *, key: str) -> None:
        compiler.compile_geom_pair_colliding(key, self.pairs)


@dataclass(frozen=True, kw_only=True)
class DofPositionQuery(SimDataQuery):
    """Whole canonical DOF position space (includes floating bases)."""

    def compile_with(self, compiler: SimDataQueryCompiler, *, key: str) -> None:
        compiler.compile_dof_position(key)


@dataclass(frozen=True, kw_only=True)
class DofVelocityQuery(SimDataQuery):
    """Whole canonical DOF velocity space (includes floating bases)."""

    def compile_with(self, compiler: SimDataQueryCompiler, *, key: str) -> None:
        compiler.compile_dof_velocity(key)


@dataclass(frozen=True, kw_only=True)
class ActuatorCtrlQuery(SimDataQuery):
    """Canonical-order full-width actuator control targets."""

    def compile_with(self, compiler: SimDataQueryCompiler, *, key: str) -> None:
        compiler.compile_actuator_ctrl(key)


class SimDataQueryCompiler(abc.ABC):
    """Compile neutral simulator-data queries into one backend read program."""

    def compile(self, queries: Mapping[str, SimDataQuery]) -> PhysicsReadProgram:
        """Compile every named query through its typed backend hook.

        Args:
            queries: Simulator-data queries keyed by their logical read names.

        Returns:
            The executable backend read program.
        """
        self._begin_compile()
        for name, query in queries.items():
            query.compile_with(self, key=name)
        return self._build_program()

    def _begin_compile(self) -> None:
        """Reset per-compile accumulation before dispatch; default no-op."""

    @abc.abstractmethod
    def _build_program(self) -> PhysicsReadProgram:
        """Assemble the read program from values recorded during dispatch.

        Returns:
            The executable backend read program.
        """

    @abc.abstractmethod
    def compile_body_joint_position(self, key: str, body: str) -> None:
        """Compile articulated position DOFs for one body, excluding its floating base.

        Args:
            key: Logical name under which the read is exposed.
            body: Name of the body whose articulated DOFs are read.
        """

    @abc.abstractmethod
    def compile_body_joint_velocity(self, key: str, body: str) -> None:
        """Compile articulated velocity DOFs for one body, excluding its floating base.

        Args:
            key: Logical name under which the read is exposed.
            body: Name of the body whose articulated DOFs are read.
        """

    @abc.abstractmethod
    def compile_joints_position(self, key: str, joints: tuple[str, ...]) -> None:
        """Compile positions for an ordered set of one-DOF joints.

        Args:
            key: Logical name under which the read is exposed.
            joints: Ordered joint names to read.
        """

    @abc.abstractmethod
    def compile_joints_velocity(self, key: str, joints: tuple[str, ...]) -> None:
        """Compile velocities for an ordered set of one-DOF joints.

        Args:
            key: Logical name under which the read is exposed.
            joints: Ordered joint names to read.
        """

    @abc.abstractmethod
    def compile_links_position(self, key: str, links: str | tuple[str, ...]) -> None:
        """Compile world positions for one link or an ordered link batch.

        Args:
            key: Logical name under which the read is exposed.
            links: One link name or an ordered tuple of link names.
        """

    @abc.abstractmethod
    def compile_links_quaternion(self, key: str, links: str | tuple[str, ...]) -> None:
        """Compile world quaternions for one link or an ordered link batch.

        Args:
            key: Logical name under which the read is exposed.
            links: One link name or an ordered tuple of link names.
        """

    @abc.abstractmethod
    def compile_links_linear_velocity(self, key: str, links: str | tuple[str, ...]) -> None:
        """Compile world linear velocities for one link or an ordered link batch.

        Args:
            key: Logical name under which the read is exposed.
            links: One link name or an ordered tuple of link names.
        """

    @abc.abstractmethod
    def compile_links_angular_velocity(self, key: str, links: str | tuple[str, ...]) -> None:
        """Compile world angular velocities for one link or an ordered link batch.

        Args:
            key: Logical name under which the read is exposed.
            links: One link name or an ordered tuple of link names.
        """

    @abc.abstractmethod
    def compile_links_net_contact_force(self, key: str, links: str | tuple[str, ...]) -> None:
        """Compile net contact-force vectors for one link or an ordered link batch.

        Args:
            key: Logical name under which the read is exposed.
            links: One link name or an ordered tuple of link names.
        """

    @abc.abstractmethod
    def compile_body_link_net_contact_force(
        self,
        key: str,
        body: str,
        exclude_links: tuple[str, ...],
    ) -> None:
        """Compile net contact forces for a body's links after exclusions.

        Args:
            key: Logical name under which the read is exposed.
            body: Name of the body whose links are read.
            exclude_links: Body link names omitted from the read.
        """

    @abc.abstractmethod
    def compile_sensor_values(self, key: str, sensors: tuple[str, ...]) -> None:
        """Compile concatenated values for an ordered set of sensors.

        Args:
            key: Logical name under which the read is exposed.
            sensors: Ordered sensor names to read.
        """

    def compile_raycast_camera_image(self, key: str, sensor: str, field: str) -> None:
        """Compile a camera image field; unsupported backends fail explicitly."""
        raise NotImplementedError(f"{type(self).__name__} does not support raycast camera images")

    @abc.abstractmethod
    def compile_site_position(self, key: str, site: str) -> None:
        """Compile the world position of one site.

        Args:
            key: Logical name under which the read is exposed.
            site: Name of the site to read.
        """

    @abc.abstractmethod
    def compile_site_quaternion(self, key: str, site: str) -> None:
        """Compile the world quaternion of one site.

        Args:
            key: Logical name under which the read is exposed.
            site: Name of the site to read.
        """

    @abc.abstractmethod
    def compile_geom_position(self, key: str, geom: str) -> None:
        """Compile the world position of one geometry.

        Args:
            key: Logical name under which the read is exposed.
            geom: Name of the geometry to read.
        """

    @abc.abstractmethod
    def compile_geom_quaternion(self, key: str, geom: str) -> None:
        """Compile the world quaternion of one geometry.

        Args:
            key: Logical name under which the read is exposed.
            geom: Name of the geometry to read.
        """

    @abc.abstractmethod
    def compile_geom_linear_velocity(self, key: str, geom: str) -> None:
        """Compile the world linear velocity of one geometry.

        Args:
            key: Logical name under which the read is exposed.
            geom: Name of the geometry to read.
        """

    @abc.abstractmethod
    def compile_geom_pair_colliding(self, key: str, pairs: tuple[tuple[str, str], ...]) -> None:
        """Compile collision indicators for ordered geometry pairs.

        Args:
            key: Logical name under which the read is exposed.
            pairs: Ordered pairs of geometry names to test for collision.
        """

    @abc.abstractmethod
    def compile_dof_position(self, key: str) -> None:
        """Compile the complete canonical DOF-position vector.

        Args:
            key: Logical name under which the read is exposed.
        """

    @abc.abstractmethod
    def compile_dof_velocity(self, key: str) -> None:
        """Compile the complete canonical DOF-velocity vector.

        Args:
            key: Logical name under which the read is exposed.
        """

    @abc.abstractmethod
    def compile_actuator_ctrl(self, key: str) -> None:
        """Compile the complete canonical actuator-control vector.

        Args:
            key: Logical name under which the read is exposed.
        """


FloatArray: TypeAlias = npt.NDArray[np.float32]
IntArray: TypeAlias = npt.NDArray[np.int64]


class PhysicsReadProgram(abc.ABC):
    """One backend's compiled executable physical read program.

    Built once by :meth:`SimBackend.compile_reads` from the complete
    declared query set and executed many times. The program owns its physical
    memory planning: equal declared queries are collapsed onto shared storage
    while every declared key stays servable, the authoritative arena is
    private, ``execute`` is the only writer, and consumers read only through
    the stable read-only logical views the program serves.
    """

    @property
    @abc.abstractmethod
    def arena_bytes(self) -> int:
        """Size of the program's authoritative destination storage in bytes."""

    @property
    @abc.abstractmethod
    def keys(self) -> tuple[str, ...]:
        """Every served declared query key, including aliases of shared storage."""

    @abc.abstractmethod
    def query(self, key: str) -> SimDataQuery:
        """Return the resolved concrete declaration served under ``key``."""

    @abc.abstractmethod
    def view(self, key: str) -> FloatArray:
        """Return the stable read-only logical view for one declared query key.

        The view aliases the arena with the query's declared trailing shape and
        leading batch dimension, is float32, and stays valid for the program's
        lifetime.
        """

    def __getitem__(self, key: str) -> FloatArray:
        """Read :meth:`view` by key — the canonical env-facing read spelling."""
        if not isinstance(key, str):
            raise TypeError(f"Simulator query keys must be strings, got {type(key).__name__}.")
        return self.view(key)

    @abc.abstractmethod
    def execute(self, env_ids: IntArray | None = None) -> None:
        """Refresh the authoritative arena — the full batch, or selected rows.

        ``env_ids`` is either ``None`` (full batch) or a one-dimensional int64
        row index array.
        """
