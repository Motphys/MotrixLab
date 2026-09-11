# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Execution-level behavior of compiled MotrixSim write programs."""

import numpy
import pytest

from motrix_env_motrixsim.write_compiler import _MotrixSimWriteProgram


@pytest.mark.parametrize("targets", [("first", "second"), ("second", "first")])
def test_native_batched_ctrl_write_preserves_environment_and_actuator_axes(tmp_path, targets) -> None:
    from motrix_env_core.base import SimCfg
    from motrix_env_core.config.scene import SceneCfg
    from motrix_env_core.sim.write import CtrlTargetsWrite
    from motrix_env_motrixsim.runtime import MotrixSimBackend

    model = tmp_path / "controls.xml"
    model.write_text(
        "<mujoco><worldbody>"
        '<body><joint name="j0"/><geom type="sphere" size="0.1" mass="1"/></body>'
        '<body pos="1 0 0"><joint name="j1"/><geom type="sphere" size="0.1" mass="1"/></body>'
        "</worldbody><actuator>"
        '<position name="first" joint="j0" kp="10"/>'
        '<position name="second" joint="j1" kp="10"/>'
        "</actuator></mujoco>",
        encoding="utf-8",
    )
    backend = MotrixSimBackend(SceneCfg(file=model), SimCfg(dt=0.005), num_envs=3)
    program = backend.write_compiler.compile({"ctrl": CtrlTargetsWrite(targets)})
    values = numpy.asarray([[1, 2], [3, 4], [5, 6]], numpy.float32)
    program.buffer("ctrl")[:] = values
    program.execute()

    expected = values if targets[0] == "first" else values[:, ::-1]
    numpy.testing.assert_array_equal(numpy.asarray(backend._data.actuator_ctrls), expected)


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
