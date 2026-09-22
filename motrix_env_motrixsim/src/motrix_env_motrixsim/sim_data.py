# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

import motrixsim as mtx
import numpy as np
import numpy.typing as npt

from motrix_env_core.perf import active_perf_scope
from motrix_env_core.sim import PhysicsReadProgram, SimDataQuery, SimDataQueryCompiler

FloatArray: TypeAlias = npt.NDArray[np.float32]


def _c_element_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = [0] * len(shape)
    stride = 1
    for index in range(len(shape) - 1, -1, -1):
        strides[index] = stride
        stride *= shape[index]
    return tuple(strides)


def _arena_view(
    arena: np.ndarray,
    num_envs: int,
    row_width: int,
    element_offset: int,
    trailing_shape: tuple[int, ...],
) -> np.ndarray:
    """Build one read-only logical view of shape ``(num_envs, *trailing_shape)`` over the arena."""
    itemsize = arena.dtype.itemsize
    strides = (row_width, *_c_element_strides(trailing_shape))
    view = np.ndarray(
        shape=(num_envs, *trailing_shape),
        dtype=arena.dtype,
        buffer=arena,
        offset=element_offset * itemsize,
        strides=tuple(stride * itemsize for stride in strides),
    )
    view.flags.writeable = False
    return view


@dataclass(frozen=True)
class _ResolvedQuery:
    key: str
    source: mtx.query.QuerySource
    # ``trailing_shape is None`` adopts the compiled native field shape; atomic
    # link queries keep their historical flat ``(width,)`` shape while native
    # sources report ``(1, width)``.
    trailing_shape: tuple[int, ...] | None = None


class _MotrixSimDataQueryCompiler(SimDataQueryCompiler):
    """Compile semantic queries into one MotrixSim physical read program."""

    def __init__(
        self,
        model: mtx.SceneModel,
        source: mtx.SceneData,
        declared: Mapping[str, SimDataQuery],
        canonical_keys: Mapping[str, str],
    ) -> None:
        self._model = model
        self._source = source
        self._declared = declared
        self._canonical_keys = canonical_keys
        self._resolved: dict[str, _ResolvedQuery] = {}

    def _begin_compile(self) -> None:
        self._resolved = {}

    def _build_program(self) -> PhysicsReadProgram:
        native = tuple(self._resolved.values())
        if any(isinstance(query.source, mtx.query.LinkNetContactForces) for query in native):
            self._model.reports.contact_forces = True
        plan = self._model.compile_query({query.key: query.source for query in native})
        program = plan.allocate(self._source)
        read = _BatchQueryRead(program, self._resolved)
        return _MotrixSimReadProgram(
            read,
            self._source,
            declared=self._declared,
            canonical_keys=self._canonical_keys,
        )

    def _record(
        self,
        key: str,
        source: mtx.query.QuerySource,
        trailing_shape: tuple[int, ...] | None = None,
    ) -> None:
        self._resolved[key] = _ResolvedQuery(key, source, trailing_shape)

    def compile_body_joint_position(self, key: str, body: str) -> None:
        body = self._get_body(self._validate_name(body, key), key)
        self._record(key, mtx.query.BodyDofPosition(body.name, include_base=False))

    def compile_body_joint_velocity(self, key: str, body: str) -> None:
        body = self._get_body(self._validate_name(body, key), key)
        self._record(key, mtx.query.BodyDofVelocity(body.name, include_base=False))

    def compile_joints_position(self, key: str, joints: tuple[str, ...]) -> None:
        joints = self._one_dof_joints(key, joints, position=True)
        self._record(key, mtx.query.JointPosition(tuple(joint.name for joint in joints)))

    def compile_joints_velocity(self, key: str, joints: tuple[str, ...]) -> None:
        joints = self._one_dof_joints(key, joints, position=False)
        self._record(key, mtx.query.JointVelocity(tuple(joint.name for joint in joints)))

    def compile_links_position(self, key: str, links: str | tuple[str, ...]) -> None:
        if isinstance(links, str):
            self._atomic_link(key, links, 3, mtx.query.LinkPosition)
        else:
            self._batch_links(key, links, 3, mtx.query.LinkPosition)

    def compile_links_quaternion(self, key: str, links: str | tuple[str, ...]) -> None:
        if isinstance(links, str):
            self._atomic_link(key, links, 4, mtx.query.LinkRotation)
        else:
            self._batch_links(key, links, 4, mtx.query.LinkRotation)

    def compile_links_linear_velocity(self, key: str, links: str | tuple[str, ...]) -> None:
        if isinstance(links, str):
            self._atomic_link(key, links, 3, mtx.query.LinkLinearVelocity)
        else:
            self._batch_links(key, links, 3, mtx.query.LinkLinearVelocity)

    def compile_links_angular_velocity(self, key: str, links: str | tuple[str, ...]) -> None:
        if isinstance(links, str):
            self._atomic_link(key, links, 3, mtx.query.LinkAngularVelocity)
        else:
            self._batch_links(key, links, 3, mtx.query.LinkAngularVelocity)

    def compile_links_net_contact_force(self, key: str, links: str | tuple[str, ...]) -> None:
        if isinstance(links, str):
            self._atomic_link(key, links, 6, mtx.query.LinkNetContactForces)
        else:
            self._batch_links(key, links, 6, mtx.query.LinkNetContactForces)

    def compile_body_link_net_contact_force(self, key: str, body: str, exclude_links: tuple[str, ...]) -> None:
        body = self._get_body(self._validate_name(body, key), key)
        excluded_names = self._validate_names(exclude_links, key, "excluded links", allow_empty=True)
        excluded = set(excluded_names)
        unknown = sorted(excluded.difference(link.name for link in body.links))
        if unknown:
            raise KeyError(f"Simulator query {key!r} excludes unknown body links: {unknown}.")
        indices = tuple(link.index for link in body.links if link.name not in excluded)
        self._record(key, mtx.query.LinkNetContactForces(indices), (len(indices), 6))

    def compile_sensor_values(self, key: str, sensors: tuple[str, ...]) -> None:
        sensor_names = self._validate_names(sensors, key, "sensors")
        self._record(key, mtx.query.SensorValues(sensor_names), None)

    def compile_dof_position(self, key: str) -> None:
        self._record(key, mtx.query.DofPosition())

    def compile_dof_velocity(self, key: str) -> None:
        self._record(key, mtx.query.DofVelocity())

    def compile_actuator_ctrl(self, key: str) -> None:
        self._record(key, mtx.query.ActuatorControls())

    def compile_geom_position(self, key: str, geom: str) -> None:
        geom = self._get_geom(self._validate_name(geom, key), key)
        self._record(key, mtx.query.GeomPosition([geom.index]), (3,))

    def compile_geom_quaternion(self, key: str, geom: str) -> None:
        geom = self._get_geom(self._validate_name(geom, key), key)
        self._record(key, mtx.query.GeomRotation([geom.index]), (4,))

    def compile_geom_linear_velocity(self, key: str, geom: str) -> None:
        geom = self._get_geom(self._validate_name(geom, key), key)
        self._record(key, mtx.query.GeomLinearVelocity([geom.index]), (3,))

    def compile_site_position(self, key: str, site: str) -> None:
        site = self._get_site(self._validate_name(site, key), key)
        self._record(key, mtx.query.SitePosition((site.name,)), (3,))

    def compile_site_quaternion(self, key: str, site: str) -> None:
        site = self._get_site(self._validate_name(site, key), key)
        self._record(key, mtx.query.SiteRotation((site.name,)), (4,))

    def compile_geom_pair_colliding(self, key: str, pairs: tuple[tuple[str, str], ...]) -> None:
        pairs = self._validate_name_pairs(pairs, key)
        indices = tuple(
            (self._get_geom(first, key).index, self._get_geom(second, key).index) for first, second in pairs
        )
        self._record(key, mtx.query.GeomPairCollision(indices), (len(pairs),))

    def _atomic_link(
        self,
        key: str,
        name: str,
        width: int,
        source_type: type[mtx.query.QuerySource],
    ) -> None:
        index = self._get_link_index(self._validate_name(name, key), key)
        self._record(key, source_type((index,)), (width,))

    def _batch_links(
        self,
        key: str,
        names: tuple[str, ...],
        width: int,
        source_type: type[mtx.query.QuerySource],
    ) -> None:
        indices = self._link_indices(key, names)
        self._record(key, source_type(indices), (len(indices), width))

    def _link_indices(self, key: str, names: tuple[str, ...]) -> tuple[int, ...]:
        link_names = self._validate_names(names, key, "links")
        return tuple(self._get_link_index(name, key) for name in link_names)

    @staticmethod
    def _validate_name(name: str, key: str) -> str:
        if not isinstance(name, str) or not name:
            raise ValueError(f"Simulator query {key!r} names must be non-empty strings.")
        return name

    @classmethod
    def _validate_names(
        cls,
        names: tuple[str, ...],
        key: str,
        kind: str,
        *,
        allow_empty: bool = False,
    ) -> tuple[str, ...]:
        if not isinstance(names, tuple):
            raise TypeError(f"Simulator query {key!r} {kind} must be tuple[str, ...].")
        if not names and not allow_empty:
            raise ValueError(f"Simulator query {key!r} must contain at least one {kind} entry.")
        for name in names:
            cls._validate_name(name, key)
        if len(set(names)) != len(names):
            raise ValueError(f"Simulator query {key!r} contains duplicate {kind}: {names}.")
        return names

    @staticmethod
    def _validate_name_pairs(pairs: tuple[tuple[str, str], ...], key: str) -> tuple[tuple[str, str], ...]:
        if not isinstance(pairs, tuple):
            raise TypeError(f"Simulator query {key!r} geom pairs must be tuple[tuple[str, str], ...].")
        if not pairs:
            raise ValueError(f"Simulator query {key!r} must contain at least one geom pair entry.")
        for pair in pairs:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TypeError(f"Simulator query {key!r} geom pairs must contain two-name tuples.")
            if any(not isinstance(name, str) or not name for name in pair):
                raise TypeError(f"Simulator query {key!r} geom pairs must contain non-empty string names.")
        if len(set(pairs)) != len(pairs):
            raise ValueError(f"Simulator query {key!r} contains duplicate geom pairs: {pairs}.")
        return pairs

    def _one_dof_joints(self, key: str, names: tuple[str, ...], *, position: bool) -> tuple[mtx.Joint, ...]:
        joint_names = self._validate_names(names, key, "joints")
        joints = []
        for name in joint_names:
            joint = self._model.get_joint(name)
            if joint is None:
                raise KeyError(f"Simulator query {key!r} references unknown joint {name!r}.")
            width = joint.num_dof_pos if position else joint.num_dof_vel
            if width != 1:
                quantity = "position" if position else "velocity"
                raise ValueError(
                    f"Simulator query {key!r} joint {name!r} has {width} {quantity} DOFs; exactly one is required."
                )
            joints.append(joint)
        return tuple(joints)

    def _get_body(self, name: str, key: str) -> mtx.Body:
        body = self._model.get_body(name)
        if body is None:
            raise KeyError(f"Simulator query {key!r} references unknown body {name!r}.")
        return body

    def _get_site(self, name: str, key: str) -> mtx.Site:
        site = self._model.get_site(name)
        if site is None:
            raise KeyError(f"Simulator query {key!r} references unknown site {name!r}.")
        return site

    def _get_geom(self, name: str, key: str) -> mtx.Geom:
        geom = self._model.get_geom(name)
        if geom is None:
            raise KeyError(f"Simulator query {key!r} references unknown geom {name!r}.")
        return geom

    def _get_link_index(self, name: str, key: str) -> int:
        link = self._model.get_link(name)
        if link is None:
            raise KeyError(f"Simulator query {key!r} references unknown link {name!r}.")
        return link.index


@dataclass(frozen=True)
class _BatchQueryRead:
    """One native batch query program and its resolved semantic fields."""

    program: mtx.query.QueryProgram
    resolved: Mapping[str, _ResolvedQuery]


class _MotrixSimReadProgram(PhysicsReadProgram):
    """Executable batch-query program bound to one MotrixSim scene state.

    The program-owned native buffer is the authoritative arena; logical query
    views alias the same storage.
    """

    def __init__(
        self,
        read: _BatchQueryRead,
        source: mtx.SceneData,
        declared: Mapping[str, SimDataQuery],
        canonical_keys: Mapping[str, str],
    ) -> None:
        self._read = read
        self._source = source
        self._declared = dict(declared)
        num_envs = int(source.shape[0])
        arena = read.program.buffer.reshape(-1)
        row_width = read.program.buffer.shape[-1]
        self._views: dict[str, FloatArray] = {}
        for field in read.program.fields:
            query = read.resolved[field.name]
            shape = query.trailing_shape
            if shape is None:
                shape = tuple(field.shape)
            self._views[field.name] = _arena_view(
                arena,
                num_envs,
                row_width,
                field.offset,
                shape,
            )
        # Duplicate declarations alias the canonical region: one physical read.
        for alias, canonical in canonical_keys.items():
            if alias != canonical:
                self._views[alias] = self._views[canonical]
        for key, view in self._views.items():
            if view.dtype != np.float32:
                raise RuntimeError(f"Program view for simulator query {key!r} must be float32, got {view.dtype}.")
            if view.shape[0] != num_envs:
                raise RuntimeError(
                    f"Program view for simulator query {key!r} must have "
                    f"leading dimension {num_envs}, got {view.shape}."
                )

    @property
    def arena_bytes(self) -> int:
        return int(self._read.program.buffer.nbytes)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self._views)

    def query(self, key: str) -> SimDataQuery:
        return self._declared[key]

    def view(self, key: str) -> FloatArray:
        return self._views[key]

    def execute(self, env_ids: npt.NDArray[np.int64] | None = None) -> None:
        if env_ids is not None and (env_ids.dtype != np.int64 or env_ids.ndim != 1):
            raise TypeError("Partial simulator read env_ids must be a one-dimensional int64 ndarray.")
        # One native call writes the authoritative arena — the full batch, or
        # only the rows selected by env_ids. The perf scope nests under the
        # caller's stage (e.g. transition/reset in the console timing tree).
        with active_perf_scope("read"):
            self._read.program.execute(self._source, env_ids=env_ids)


def compile_read_program(
    model: mtx.SceneModel,
    source: mtx.SceneData,
    queries: Mapping[str, SimDataQuery],
) -> PhysicsReadProgram:
    """Compile fixed semantic queries into one MotrixSim batch-query program."""
    # Memory planning is backend-owned: collapse equal declarations onto
    # one region, then serve every declared key as an alias of it.
    canonical_queries: dict[str, SimDataQuery] = {}
    canonical_keys: dict[str, str] = {}
    for key, query in queries.items():
        duplicate = next((c for c, q in canonical_queries.items() if q == query), None)
        canonical_keys[key] = key if duplicate is None else duplicate
        if duplicate is None:
            canonical_queries[key] = query
    compiler = _MotrixSimDataQueryCompiler(model, source, queries, canonical_keys)
    return compiler.compile(canonical_queries)
