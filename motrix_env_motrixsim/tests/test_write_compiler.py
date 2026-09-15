# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Execution-level behavior of compiled MotrixSim write programs."""

import numpy as np

from motrix_env_motrixsim.write_compiler import _CompiledWrite, _MotrixSimWriteProgram


class _Model:
    num_dof_pos = 0
    num_dof_vel = 0

    def __init__(self) -> None:
        self.forward_kinematic_rows = []

    def compute_init_dof_pos(self) -> np.ndarray:
        return np.zeros((0,), dtype=np.float32)

    def forward_kinematic(self, rows) -> None:
        self.forward_kinematic_rows.append(rows)


class _Data:
    shape = (3,)


class _ResetData(_Data):
    def __init__(self) -> None:
        self.reset_calls = []

    def reset(self, model, **kwargs) -> None:
        self.reset_calls.append((model, kwargs))


class _Op:
    def __init__(self) -> None:
        self.rows = []

    def __call__(self, buffers, idx, rows) -> None:
        del buffers, idx
        self.rows.append(rows)


def test_write_program_refreshes_kinematics_once_after_all_ops() -> None:
    model = _Model()
    data = _Data()
    first = _Op()
    second = _Op()
    program = _MotrixSimWriteProgram(
        model,
        data,
        lambda env_ids: ("rows", tuple(env_ids)),
        {},
        [(_CompiledWrite(first), {}), (_CompiledWrite(second), {})],
        reset=False,
        refresh_kinematics=True,
    )

    program.execute(np.asarray([2, 0], dtype=np.int64))

    expected_rows = ("rows", (0, 2))
    assert first.rows == [expected_rows]
    assert second.rows == [expected_rows]
    assert model.forward_kinematic_rows == [expected_rows]


def test_reset_program_passes_compile_time_kinematics_flag_to_native_reset() -> None:
    model = _Model()
    data = _ResetData()
    program = _MotrixSimWriteProgram(model, data, lambda env_ids: env_ids, {}, [], reset=True, refresh_kinematics=False)

    program.execute()

    assert data.reset_calls == [(model, {"forward_kinematic": False})]
    assert model.forward_kinematic_rows == []


def test_reset_program_folds_ops_into_the_reset_state() -> None:
    model = _Model()
    data = _ResetData()
    folded = []

    class _FoldOp:
        def apply(self, dof_pos, dof_vel, buffers, env_ids) -> None:
            folded.append((dof_pos.shape, dof_vel.shape, list(env_ids)))

    program = _MotrixSimWriteProgram(
        model,
        data,
        lambda env_ids: data,
        {},
        [(_CompiledWrite(_Op(), _FoldOp()), np.zeros((3, 2), dtype=np.float32))],
        reset=True,
        refresh_kinematics=True,
    )

    program.execute(np.asarray([0, 2], dtype=np.int64))

    # The reset fires once with the folded state; no op runs post-reset.
    assert data.reset_calls == [(model, {"forward_kinematic": True})]
    assert folded == [((2, 0), (2, 0), [0, 2])]


def test_write_program_skips_kinematic_refresh_when_no_op_requires_it() -> None:
    model = _Model()
    program = _MotrixSimWriteProgram(
        model, _Data(), lambda env_ids: env_ids, {}, [], reset=False, refresh_kinematics=False
    )

    program.execute()

    assert model.forward_kinematic_rows == []


class _NativeProgram:
    def __init__(self, label: str, log: list) -> None:
        self._label = label
        self._log = log

    def execute(self, data, env_ids) -> None:
        self._log.append((self._label, data, None if env_ids is None else env_ids.copy()))


class _LoggedOp:
    def __init__(self, label: str, log: list) -> None:
        self._label = label
        self._log = log

    def alloc(self, num_envs: int) -> np.ndarray:
        return np.zeros((num_envs, 1), dtype=np.float32)

    def __call__(self, buffers, idx, rows) -> None:
        del buffers, idx
        self._log.append((self._label, rows))


def _native_case(
    log: list, ops: list, *, reset: bool, refresh: bool, lead: int, env_ids=None
) -> _MotrixSimWriteProgram:
    model = _Model()
    program = _MotrixSimWriteProgram(
        model,
        _Data(),
        lambda env_ids: ("rows", tuple(env_ids)),
        {},
        ops,
        native=_NativeProgram("native", log),
        reset=reset,
        refresh_kinematics=refresh,
        lead_op_count=lead,
    )
    program.execute(env_ids)
    return program


def test_native_program_runs_lead_ops_then_native_then_tail_ops() -> None:
    log = []
    ops = [(_CompiledWrite(_LoggedOp("lead", log)), {}), (_CompiledWrite(_LoggedOp("tail", log)), {})]

    _native_case(log, ops, reset=False, refresh=False, lead=1, env_ids=np.asarray([1], dtype=np.int64))

    assert [entry[0] for entry in log] == ["lead", "native", "tail"]
    assert log[1][2].tolist() == [1]


def test_native_reset_program_runs_native_pass_before_numpy_ops() -> None:
    log = []
    ops = [(_CompiledWrite(_LoggedOp("op", log)), {})]

    _native_case(log, ops, reset=True, refresh=False, lead=0)

    assert [entry[0] for entry in log] == ["native", "op"]
    assert log[0][2] is None


def test_native_program_refreshes_kinematics_once_after_every_write() -> None:
    log = []
    ops = [(_CompiledWrite(_LoggedOp("op", log)), {})]
    model = _Model()
    data = _Data()

    program = _MotrixSimWriteProgram(
        model,
        data,
        lambda env_ids: ("rows", tuple(env_ids)),
        {},
        ops,
        native=_NativeProgram("native", log),
        reset=False,
        refresh_kinematics=True,
        lead_op_count=0,
    )
    program.execute()

    assert [entry[0] for entry in log] == ["native", "op"]
    assert model.forward_kinematic_rows == [data]


def test_native_buffer_lookup_falls_through_to_native_views() -> None:
    class _ViewNative:
        def __getitem__(self, name: str) -> np.ndarray:
            return {"ctrl": np.zeros((2, 3), dtype=np.float32)}[name]

    program = _MotrixSimWriteProgram(
        _Model(),
        _Data(),
        lambda env_ids: env_ids,
        {"state": np.zeros((2, 2), dtype=np.float32)},
        [],
        native=_ViewNative(),
        reset=False,
        refresh_kinematics=False,
    )

    assert program.buffer("state").shape == (2, 2)
    assert program.buffer("ctrl").shape == (2, 3)
    try:
        program.buffer("missing")
    except KeyError:
        pass
    else:
        raise AssertionError("Unknown buffer name must raise KeyError.")
