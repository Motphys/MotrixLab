# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from types import SimpleNamespace

import motrixsim as mtx
import numpy as np
import pytest

from motrix_env_core.sim import (
    ActuatorCtrlQuery,
    BatchLinkAngularVelocityQuery,
    BatchLinkLinearVelocityQuery,
    BatchLinkNetContactForceQuery,
    BatchLinkPositionQuery,
    BatchLinkQuaternionQuery,
    BodyJointPositionQuery,
    BodyJointVelocityQuery,
    DofPositionQuery,
    DofVelocityQuery,
    GeomLinearVelocityQuery,
    GeomPairCollidingQuery,
    GeomPositionQuery,
    GeomQuaternionQuery,
    LinkQuaternionQuery,
    ModelQuery,
    PhysicsReadProgram,
    SimDataQuery,
    SimDataQueryCompiler,
    SitePositionQuery,
    SiteQuaternionQuery,
)
from motrix_env_core.sim.backend import SimBackend
from motrix_env_motrixsim.sim_data import compile_read_program

# Native source kinds the fakes below know how to fill, mapped to their
# per-link width and value offset inside one query-buffer row.
_LINK_SOURCES = {
    mtx.query.LinkPosition: (3, 0.0),
    mtx.query.LinkRotation: (4, 10.0),
    mtx.query.LinkLinearVelocity: (3, 20.0),
    mtx.query.LinkAngularVelocity: (3, 30.0),
    mtx.query.LinkNetContactForces: (6, 30.0),
}

# Fake geom offsets by geom index, mirroring _Model.geoms below.
_GEOM_OFFSETS = {4: 50.0, 5: 60.0}
_SITE_OFFSETS = {"tip": 40.0}


class _Data:
    def __init__(
        self,
        values: list[float],
        *,
        dof_pos=None,
        dof_vel=None,
        actuator_ctrls=None,
    ) -> None:
        self.values = np.asarray(values, dtype=np.float32)
        self.shape = (len(values),)
        self.dof_pos = self.values[:, None] if dof_pos is None else np.asarray(dof_pos, dtype=np.float32)
        self.dof_vel = self.values[:, None] if dof_vel is None else np.asarray(dof_vel, dtype=np.float32)
        self.actuator_ctrls = (
            self.values[:, None] if actuator_ctrls is None else np.asarray(actuator_ctrls, dtype=np.float32)
        )


class _Link:
    def __init__(self, index: int, name: str) -> None:
        self.index = index
        self.name = name


class _Body:
    index = 5
    name = "robot"
    num_joint_dof_pos = 2
    num_joint_dof_vel = 2


class _QueryField:
    def __init__(self, name: str, shape: tuple[int, ...], offset: int, size: int) -> None:
        self.name = name
        self.shape = shape
        self.offset = offset
        self.size = size


class _PoseObject:
    """Fake named scene object exposing a batched pose getter (site or geom)."""

    def __init__(self, index: int, name: str, offset: float, calls: list[str], kind: str) -> None:
        self.index = index
        self.name = name
        self._offset = offset
        self._calls = calls
        self._kind = kind

    def get_pose(self, data: _Data) -> np.ndarray:
        self._calls.append(f"{self._kind}:{self.name}")
        pose = np.zeros((data.values.shape[0], 7), dtype=np.float32)
        pose[:, :3] = data.values[:, None] + np.float32(self._offset) + np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
        pose[:, 3:7] = data.values[:, None] + np.float32(self._offset) + np.float32(100.0)
        return pose


class _ContactQuery:
    def __init__(self, data: _Data, calls: list[str]) -> None:
        self._data = data
        self._calls = calls

    def is_colliding(self, pairs: np.ndarray) -> np.ndarray:
        self._calls.append("contact_query")
        rows = self._data.values.shape[0]
        result = np.zeros((rows, pairs.shape[0]), dtype=bool)
        result[:, 0] = self._data.values > 0.5
        return result


# Whole-space native sources: one full-batch array on the fake data each.
_WHOLE_SPACE_SOURCES = {
    mtx.query.DofPosition: lambda data, row: data.dof_pos[row],
    mtx.query.DofVelocity: lambda data, row: data.dof_vel[row],
    mtx.query.ActuatorControls: lambda data, row: data.actuator_ctrls[row],
}

# Native geom sources: per-slot width and value base, mirroring _Model.geoms.
_GEOM_SOURCES = {
    mtx.query.GeomPosition: (3, np.float32([0.0, 1.0, 2.0])),
    mtx.query.GeomRotation: (4, np.float32(100.0)),
    mtx.query.GeomLinearVelocity: (3, np.float32([3.0, 4.0, 5.0])),
}
_SITE_SOURCES = {
    mtx.query.SitePosition: (3, np.float32([0.0, 1.0, 2.0])),
    mtx.query.SiteRotation: (4, np.float32(100.0)),
}


def _source_fill(source: object) -> tuple[int, object]:
    """Per-slot width and value base for one selection-based native source."""
    if type(source) in _LINK_SOURCES:
        return _LINK_SOURCES[type(source)]
    if type(source) in _GEOM_SOURCES:
        return _GEOM_SOURCES[type(source)]
    return _SITE_SOURCES[type(source)]


def _source_identity(source: object) -> tuple:
    if isinstance(source, (mtx.query.BodyDofPosition, mtx.query.BodyDofVelocity)):
        return (type(source), source.body, source.include_base)
    if isinstance(source, mtx.query.SensorValues):
        return (type(source), tuple(source.selection) if source.selection else None)
    if type(source) in _WHOLE_SPACE_SOURCES:
        return (type(source),)
    if isinstance(source, mtx.query.GeomPairCollision):
        return (type(source), tuple(source.pairs))
    width, _ = _source_fill(source)
    return (type(source), tuple(source.selection or ()), width)


def _source_shape(source: object) -> tuple[int, ...]:
    if isinstance(source, (mtx.query.BodyDofPosition, mtx.query.BodyDofVelocity)):
        return (_Body.num_joint_dof_pos,)
    if isinstance(source, mtx.query.DofPosition):
        return (_Model.num_dof_pos,)
    if isinstance(source, mtx.query.DofVelocity):
        return (_Model.num_dof_vel,)
    if isinstance(source, mtx.query.ActuatorControls):
        return (_Model.num_actuators,)
    if isinstance(source, mtx.query.GeomPairCollision):
        return (len(source.pairs),)
    width, _ = _source_fill(source)
    return (len(source.selection or ()), width)


class _QueryProgram:
    """Fake native program: contiguous buffer, env-indexed row fill."""

    def __init__(self, sources: Mapping[str, object], fields: list[_QueryField], rows: int, total_width: int) -> None:
        self._sources = dict(sources)
        self.fields = fields
        self.buffer = np.zeros((rows, total_width), dtype=np.float32)
        self.execute_calls = 0

    def execute(self, data: _Data, env_ids: np.ndarray | None = None) -> "_QueryProgram":
        self.execute_calls += 1
        rows = range(self.buffer.shape[0]) if env_ids is None else (int(row) for row in env_ids)
        for row in rows:
            self._fill_row(data, row)
        return self

    def _fill_row(self, data: _Data, row: int) -> None:
        values = data.values[row, None]
        for field in self.fields:
            source = self._sources[field.name]
            region = self.buffer[row, field.offset : field.offset + field.size]
            if isinstance(source, mtx.query.BodyDofPosition):
                region[:] = np.concatenate((values + 1.0, values + 2.0))
            elif isinstance(source, mtx.query.BodyDofVelocity):
                region[:] = np.concatenate((values + 3.0, values + 4.0))
            elif type(source) in _WHOLE_SPACE_SOURCES:
                region[:] = _WHOLE_SPACE_SOURCES[type(source)](data, row)
            elif type(source) in _GEOM_SOURCES:
                width, base = _GEOM_SOURCES[type(source)]
                for slot, index in enumerate(source.selection or ()):
                    region[slot * width : (slot + 1) * width] = values + _GEOM_OFFSETS[index] + base
            elif type(source) in _SITE_SOURCES:
                width, base = _SITE_SOURCES[type(source)]
                for slot, name in enumerate(source.selection or ()):
                    offset = _SITE_OFFSETS[name]
                    region[slot * width : (slot + 1) * width] = values + offset + base
            elif isinstance(source, mtx.query.GeomPairCollision):
                region[:] = np.asarray([data.values[row] > 0.5] + [False] * (len(source.pairs) - 1), dtype=np.float32)
            else:
                width, base = _LINK_SOURCES[type(source)]
                for slot, index in enumerate(source.selection or ()):
                    region[slot * width : (slot + 1) * width] = values + index + base


class _QueryPlan:
    """Fake native plan: contiguous layout, identical-source dedup."""

    def __init__(self, sources: Mapping[str, object]) -> None:
        self._sources = dict(sources)
        self.programs: list[_QueryProgram] = []
        self.fields: list[_QueryField] = []
        self.total_width = 0
        regions: dict[tuple, int] = {}
        for name, source in sources.items():
            identity = _source_identity(source)
            offset = regions.get(identity)
            if offset is None:
                offset = self.total_width
                regions[identity] = offset
                self.total_width += int(np.prod(_source_shape(source), dtype=np.int64))
            self.fields.append(_QueryField(name, _source_shape(source), offset, self._source_size(source)))

    @staticmethod
    def _source_size(source: object) -> int:
        return int(np.prod(_source_shape(source), dtype=np.int64))

    def allocate(self, data: _Data) -> _QueryProgram:
        program = _QueryProgram(self._sources, self.fields, data.shape[0], self.total_width)
        self.programs.append(program)
        return program


class _Model:
    num_dof_pos = 2
    num_dof_vel = 2
    num_actuators = 2

    def __init__(self) -> None:
        self.body = _Body()
        self.links = (_Link(2, "root"), _Link(7, "foot"))
        self._links = {link.name: link for link in self.links}
        self.reports = SimpleNamespace(contact_forces=False)
        self.plans: list[_QueryPlan] = []
        self.calls: list[str] = []
        self.sites = (_PoseObject(1, "tip", 40.0, self.calls, "site"),)
        self._sites = {site.name: site for site in self.sites}
        self.geoms = (
            _PoseObject(4, "ball", 50.0, self.calls, "geom"),
            _PoseObject(5, "hand", 60.0, self.calls, "geom"),
        )
        self._geoms = {geom.name: geom for geom in self.geoms}

    def get_body(self, name: str) -> _Body:
        if name != "robot":
            raise KeyError(name)
        return self.body

    def get_link(self, name: str) -> _Link:
        return self._links[name]

    def get_site(self, name: str) -> _PoseObject | None:
        return self._sites.get(name)

    def get_geom(self, name: str) -> _PoseObject | None:
        return self._geoms.get(name)

    def get_contact_query(self, data: _Data) -> _ContactQuery:
        return _ContactQuery(data, self.calls)

    def compile_query(self, fields: Mapping[str, object]) -> _QueryPlan:
        plan = _QueryPlan(fields)
        self.plans.append(plan)
        return plan


def _queries():
    links = ("root", "foot")
    return {
        "dof_pos": BodyJointPositionQuery(body="robot"),
        "dof_pos_alias": BodyJointPositionQuery(body="robot"),
        "tracked_pos": BatchLinkPositionQuery(links=links),
        "tracked_quat": BatchLinkQuaternionQuery(links=links),
        "tracked_ang_vel": BatchLinkAngularVelocityQuery(links=links),
        "contacts": BatchLinkNetContactForceQuery(links=links),
    }


class _FrameworkTestProgram(PhysicsReadProgram):
    """Minimal program whose memory planning collapses equal declarations."""

    def __init__(self, declared: Mapping[str, SimDataQuery], canonical_keys: Mapping[str, str]) -> None:
        self.storage = np.zeros((2 * 2,), dtype=np.float32)
        self._output = self.storage.reshape(2, 2)
        self._declared = dict(declared)
        self._views: dict[str, np.ndarray] = {}
        for key, canonical in canonical_keys.items():
            if canonical not in self._views:
                view = self.storage.reshape(2, 2)
                view.flags.writeable = False
                self._views[canonical] = view
            self._views[key] = self._views[canonical]

    @property
    def arena_bytes(self) -> int:
        return int(self.storage.nbytes)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self._views)

    def query(self, key: str) -> SimDataQuery:
        return self._declared[key]

    def view(self, key: str) -> np.ndarray:
        return self._views[key]

    def execute(self, env_ids: np.ndarray | None = None) -> None:
        assert env_ids is None
        self._output[:] = [[1.0, 2.0], [3.0, 4.0]]


class _FrameworkTestBackend(SimBackend):
    def __init__(self) -> None:
        super().__init__(None, None, 2)

    @property
    def num_dof_pos(self) -> int:
        return 2

    @property
    def num_dof_vel(self) -> int:
        return 2

    @property
    def num_actuators(self) -> int:
        return 2

    def step(self, substeps: int) -> None:
        raise AssertionError("not exercised")

    @property
    def model_compiler(self):
        return self

    def compile(self, queries: Mapping[str, ModelQuery]) -> object:
        raise AssertionError("not exercised")

    @property
    def num_envs(self) -> int:
        return 2

    def compile_reads(self, queries: Mapping[str, SimDataQuery]) -> PhysicsReadProgram:
        canonical: dict[str, SimDataQuery] = {}
        canonical_keys: dict[str, str] = {}
        for key, query in queries.items():
            duplicate = next((c for c, q in canonical.items() if q == query), None)
            canonical_keys[key] = key if duplicate is None else duplicate
            if duplicate is None:
                canonical[key] = query
        assert tuple(canonical) == ("value",)
        return _FrameworkTestProgram(queries, canonical_keys)


def test_program_surface_collapses_duplicates_without_exposing_the_arena() -> None:
    query = BodyJointPositionQuery(body="robot")
    program = _FrameworkTestBackend().compile_reads({"value": query, "alias": query})

    assert not hasattr(program, "arena")
    assert program["value"] is program["alias"]
    assert program.keys == ("value", "alias")
    assert program.query("alias") is query
    assert np.shares_memory(program["value"], program.storage)

    program.execute()

    np.testing.assert_array_equal(program["value"], [[1.0, 2.0], [3.0, 4.0]])


def test_motrixsim_provider_reads_joint_dof_positions_and_velocities() -> None:
    compiled = compile_read_program(
        _Model(),
        _Data([0.0, 1.0]),
        {
            "dof_pos": BodyJointPositionQuery(body="robot"),
            "dof_vel": BodyJointVelocityQuery(body="robot"),
        },
    )

    compiled.execute()

    np.testing.assert_array_equal(compiled["dof_pos"], [[1.0, 2.0], [2.0, 3.0]])
    np.testing.assert_array_equal(compiled["dof_vel"], [[3.0, 4.0], [4.0, 5.0]])


def test_motrixsim_provider_compiles_stable_aliases_and_one_batch_query_plan() -> None:
    model = _Model()
    source = _Data([0.0, 1.0, 2.0, 3.0])
    compiled = compile_read_program(model, source, _queries())

    assert compiled["dof_pos"] is compiled["dof_pos_alias"]
    assert not hasattr(compiled, "arena")
    assert not compiled["dof_pos"].flags.writeable
    assert np.shares_memory(compiled["tracked_pos"], model.plans[0].programs[0].buffer)
    assert compiled.arena_bytes == 4 * 34 * 4
    assert compiled["tracked_pos"].shape[1:] == (2, 3)
    assert compiled["tracked_quat"].shape[1:] == (2, 4)
    assert compiled["tracked_ang_vel"].shape[1:] == (2, 3)
    assert compiled["contacts"].shape[1:] == (2, 6)
    assert model.reports.contact_forces
    assert compiled.arena_bytes == model.plans[0].programs[0].buffer.nbytes

    compiled.execute()
    np.testing.assert_array_equal(compiled["dof_pos"], [[1, 2], [2, 3], [3, 4], [4, 5]])
    np.testing.assert_array_equal(compiled["tracked_pos"][:, :, 0], [[2, 7], [3, 8], [4, 9], [5, 10]])
    np.testing.assert_array_equal(compiled["tracked_quat"][:, :, 0], [[12, 17], [13, 18], [14, 19], [15, 20]])
    assert model.plans[0].programs[0].execute_calls == 1


def test_motrixsim_provider_partial_read_scatter_preserves_other_rows_and_view_identity() -> None:
    model = _Model()
    source = _Data([0.0, 1.0, 2.0, 3.0])
    compiled = compile_read_program(model, source, _queries())
    compiled.execute()
    dof_pos = compiled["dof_pos"]
    before = dof_pos.copy()
    tracked_before = compiled["tracked_pos"].copy()

    source.values[[3, 1]] = [100.0, 200.0]
    compiled.execute(np.asarray([3, 1], dtype=np.int64))

    assert compiled["dof_pos"] is dof_pos
    np.testing.assert_array_equal(dof_pos[0], before[0])
    np.testing.assert_array_equal(dof_pos[2], before[2])
    np.testing.assert_array_equal(dof_pos[3], [101.0, 102.0])
    np.testing.assert_array_equal(dof_pos[1], [201.0, 202.0])
    np.testing.assert_array_equal(compiled["tracked_pos"][0], tracked_before[0])
    np.testing.assert_array_equal(compiled["tracked_pos"][2], tracked_before[2])
    assert model.plans[0].programs[0].execute_calls == 2  # one full-batch execution plus one partial-row execution
    np.testing.assert_array_equal(compiled["tracked_pos"][3, 0], [102.0, 102.0, 102.0])


def test_motrixsim_provider_uses_one_batch_query_as_zero_copy_authoritative_storage() -> None:
    links = ("root", "foot")
    model = _Model()
    compiled = compile_read_program(
        model,
        _Data([0.0, 1.0, 2.0, 3.0]),
        {
            "tracked_pos": BatchLinkPositionQuery(links=links),
            "tracked_quat": BatchLinkQuaternionQuery(links=links),
            "tracked_linear_vel": BatchLinkLinearVelocityQuery(links=links),
            "tracked_angular_vel": BatchLinkAngularVelocityQuery(links=links),
            "root_quat": LinkQuaternionQuery(link="root"),
        },
    )

    assert compiled.arena_bytes == 4 * 30 * 4
    assert compiled["root_quat"].shape[1:] == (4,)
    assert all(np.shares_memory(compiled[key], model.plans[0].programs[0].buffer) for key in compiled.keys)

    compiled.execute()

    np.testing.assert_array_equal(compiled["tracked_pos"][:, :, 0], [[2, 7], [3, 8], [4, 9], [5, 10]])
    np.testing.assert_array_equal(compiled["root_quat"][:, 0], [12, 13, 14, 15])
    assert model.plans[0].programs[0].execute_calls == 1


def test_motrixsim_provider_aliases_identical_native_sources() -> None:
    compiled = compile_read_program(
        _Model(),
        _Data([0.0, 1.0]),
        {
            "root_quat": LinkQuaternionQuery(link="root"),
            "root_quat_batch": BatchLinkQuaternionQuery(links=("root",)),
        },
    )

    assert compiled.arena_bytes == 2 * 4 * 4
    compiled.execute()
    assert np.shares_memory(compiled["root_quat"], compiled["root_quat_batch"])
    assert compiled["root_quat"].shape == (2, 4)
    assert compiled["root_quat_batch"].shape == (2, 1, 4)
    np.testing.assert_array_equal(compiled["root_quat"], compiled["root_quat_batch"].reshape(2, 4))
    np.testing.assert_array_equal(compiled["root_quat"], [[12, 12, 12, 12], [13, 13, 13, 13]])


def test_motrixsim_provider_validates_query_and_read_contracts() -> None:
    with pytest.raises(ValueError, match="at least one links"):
        compile_read_program(
            _Model(),
            _Data([0.0, 1.0]),
            {"empty": BatchLinkPositionQuery(links=())},
        )

    compiled = compile_read_program(
        _Model(),
        _Data([0.0, 1.0]),
        {"dof_pos": BodyJointPositionQuery(body="robot")},
    )
    with pytest.raises(TypeError, match="int64"):
        compiled.execute(np.asarray([0], dtype=np.int32))
    compiled.execute(np.asarray([0, 0], dtype=np.int64))  # native duplicates are idempotent
    with pytest.raises(ValueError):
        compiled["dof_pos"][0, 0] = 1.0


def test_query_compiler_is_abstract() -> None:
    with pytest.raises(TypeError, match="abstract"):
        SimDataQueryCompiler()


def test_whole_space_reads_compile_and_execute() -> None:
    data = _Data(
        [1.0, 2.0],
        dof_pos=[[10.0, 11.0], [12.0, 13.0]],
        dof_vel=[[20.0, 21.0], [22.0, 23.0]],
        actuator_ctrls=[[30.0, 31.0], [32.0, 33.0]],
    )
    compiled = compile_read_program(
        _Model(),
        data,
        {
            "dof_pos": DofPositionQuery(),
            "dof_vel": DofVelocityQuery(),
            "actuator_ctrls": ActuatorCtrlQuery(),
        },
    )

    compiled.execute()

    np.testing.assert_allclose(compiled["dof_pos"], data.dof_pos)
    np.testing.assert_allclose(compiled["dof_vel"], data.dof_vel)
    np.testing.assert_allclose(compiled["actuator_ctrls"], data.actuator_ctrls)

    data.dof_pos[1] = [99.0, 99.0]
    compiled.execute(np.asarray([1], dtype=np.int64))

    np.testing.assert_allclose(compiled["dof_pos"][1], [99.0, 99.0])
    np.testing.assert_allclose(compiled["dof_pos"][0], [10.0, 11.0])


def test_site_and_geom_pose_queries_group_getter_fills() -> None:
    model = _Model()
    compiled = compile_read_program(
        model,
        _Data([0.0, 1.0, 2.0]),
        {
            "tip_pos": SitePositionQuery(site="tip"),
            "tip_quat": SiteQuaternionQuery(site="tip"),
            "ball_pos": GeomPositionQuery(geom="ball"),
            "ball_quat": GeomQuaternionQuery(geom="ball"),
        },
    )

    assert compiled["tip_pos"].shape[1:] == (3,)
    assert compiled["tip_quat"].shape[1:] == (4,)
    assert compiled["ball_pos"].shape[1:] == (3,)
    assert compiled["ball_quat"].shape[1:] == (4,)

    compiled.execute()

    np.testing.assert_allclose(compiled["tip_pos"], [[40, 41, 42], [41, 42, 43], [42, 43, 44]])
    np.testing.assert_allclose(compiled["tip_quat"][:, 0], [140, 141, 142])
    np.testing.assert_allclose(compiled["ball_pos"], [[50, 51, 52], [51, 52, 53], [52, 53, 54]])
    np.testing.assert_allclose(compiled["ball_quat"][:, 0], [150, 151, 152])
    # Site and geom pose queries all ride the native batch query.
    assert model.calls == []

    compiled.execute(np.asarray([2], dtype=np.int64))
    np.testing.assert_allclose(compiled["tip_pos"][2], [42, 43, 44])
    np.testing.assert_allclose(compiled["tip_pos"][0], [40, 41, 42])
    assert model.calls == []


def test_geom_linear_velocity_query_reads_world_frame_velocity() -> None:
    model = _Model()
    compiled = compile_read_program(
        model,
        _Data([0.0, 1.0, 2.0]),
        {"ball_lin_vel": GeomLinearVelocityQuery(geom="ball")},
    )

    assert compiled["ball_lin_vel"].shape[1:] == (3,)

    compiled.execute()

    np.testing.assert_allclose(compiled["ball_lin_vel"], [[53, 54, 55], [54, 55, 56], [55, 56, 57]])
    assert np.shares_memory(compiled["ball_lin_vel"], model.plans[0].programs[0].buffer)

    compiled.execute(np.asarray([2], dtype=np.int64))
    np.testing.assert_allclose(compiled["ball_lin_vel"][2], [55, 56, 57])
    np.testing.assert_allclose(compiled["ball_lin_vel"][0], [53, 54, 55])


def test_motrixsim_provider_rejects_unknown_site_and_geom_names() -> None:
    with pytest.raises(KeyError, match="unknown site"):
        compile_read_program(_Model(), _Data([0.0, 1.0]), {"site": SitePositionQuery(site="missing")})
    with pytest.raises(KeyError, match="unknown geom"):
        compile_read_program(_Model(), _Data([0.0, 1.0]), {"geom": GeomQuaternionQuery(geom="missing")})


def test_geom_pair_colliding_reads_boolean_pairs_as_float() -> None:
    model = _Model()
    data = _Data([0.0, 1.0])
    compiled = compile_read_program(
        model,
        data,
        {"colliding": GeomPairCollidingQuery(pairs=(("hand", "ball"), ("ball", "hand")))},
    )

    assert compiled["colliding"].shape[1:] == (2,)
    compiled.execute()

    np.testing.assert_allclose(compiled["colliding"], [[0.0, 0.0], [1.0, 0.0]])
    assert compiled["colliding"].dtype == np.float32
    assert "contact_query" not in model.calls

    compiled.execute(np.asarray([0], dtype=np.int64))
    np.testing.assert_allclose(compiled["colliding"], [[0.0, 0.0], [1.0, 0.0]])


def test_geom_pair_colliding_query_validates_pair_structure() -> None:
    provider = (_Model(), _Data([0.0, 1.0]))
    for pairs in (
        ("hand", "ball"),
        (("hand", "ball", "floor"),),
        (("hand", 1),),
        (["hand", "ball"],),
        (("", "ball"),),
    ):
        with pytest.raises(TypeError, match="geom pairs"):
            compile_read_program(*provider, {"colliding": GeomPairCollidingQuery(pairs=pairs)})
    with pytest.raises(ValueError, match="duplicate"):
        compile_read_program(
            *provider,
            {"colliding": GeomPairCollidingQuery(pairs=(("hand", "ball"), ("hand", "ball")))},
        )
    with pytest.raises(ValueError, match="at least one"):
        compile_read_program(*provider, {"colliding": GeomPairCollidingQuery(pairs=())})


def test_sim_query_compiler_is_abstract() -> None:
    with pytest.raises(TypeError, match="abstract"):
        SimDataQueryCompiler()
