# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Execution-level behavior of compiled MotrixSim write programs."""

import numpy
import pytest

from motrix_env_motrixsim.write_compiler import _MotrixSimWriteProgram


class _Data:
    shape = (3,)


class _NativeProgram:
    def __init__(self) -> None:
        self.execute_calls = []

    def execute(self, data, env_ids=None) -> None:
        self.execute_calls.append((data, None if env_ids is None else tuple(env_ids)))


def _program(native: _NativeProgram | None = None, data: _Data | None = None) -> _MotrixSimWriteProgram:
    return _MotrixSimWriteProgram(data if data is not None else _Data(), {}, native)


def test_write_program_executes_native_program_with_selected_ids() -> None:
    native = _NativeProgram()
    data = _Data()
    program = _program(native, data)

    program.execute(numpy.asarray([2, 0], dtype=numpy.int64))

    assert native.execute_calls == [(data, (0, 2))]


def test_write_program_executes_full_batch_without_env_ids() -> None:
    native = _NativeProgram()
    data = _Data()
    program = _program(native, data)

    program.execute()

    assert native.execute_calls == [(data, None)]


def test_write_program_without_native_plan_is_a_no_op() -> None:
    program = _program()

    program.execute(numpy.asarray([0, 1], dtype=numpy.int64))


@pytest.mark.parametrize(
    "env_ids",
    [
        numpy.asarray([0], dtype=numpy.int32),
        numpy.asarray([[0, 1]], dtype=numpy.int64),
        [0, 1],
    ],
)
def test_write_program_rejects_non_int64_1d_ids(env_ids) -> None:
    with pytest.raises(TypeError, match="int64 ndarray"):
        _program().execute(env_ids)


def test_write_program_rejects_out_of_range_ids() -> None:
    with pytest.raises(IndexError, match="out of range"):
        _program().execute(numpy.asarray([3], dtype=numpy.int64))


def test_write_program_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        _program().execute(numpy.asarray([0, 0], dtype=numpy.int64))


def test_write_program_skips_empty_selection() -> None:
    native = _NativeProgram()
    _program(native).execute(numpy.asarray([], dtype=numpy.int64))

    assert native.execute_calls == []
