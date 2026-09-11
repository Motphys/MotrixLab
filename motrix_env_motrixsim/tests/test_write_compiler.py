# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Execution-level behavior of compiled MotrixSim write programs."""

import numpy as np
import pytest

from motrix_env_motrixsim.write_compiler import _CompiledWrite, _CtrlOp, _MotrixSimWriteProgram


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
    values = np.asarray([[1, 2], [3, 4], [5, 6]], np.float32)
    program.buffer("ctrl")[:] = values
    program.execute()

    expected = values if targets[0] == "first" else values[:, ::-1]
    np.testing.assert_array_equal(np.asarray(backend._data.actuator_ctrls), expected)


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


@pytest.mark.parametrize("indices", [[0, 1, 2], [2, 0, 1], [2, 0]])
def test_ctrl_targets_preserve_declared_order_for_full_and_partial_writes(indices) -> None:
    rows = _Data()
    rows.actuator_ctrls = np.full((2, 3), -1.0, np.float32)
    op = _CtrlOp(np.asarray(indices, np.int64))
    values = np.arange(3 * len(indices), dtype=np.float32).reshape(3, len(indices))
    selected = np.asarray([2, 0], np.int64)
    expected = rows.actuator_ctrls.copy()
    expected[:, indices] = values[selected]

    op(values, selected, rows)

    np.testing.assert_array_equal(rows.actuator_ctrls, expected)


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


def test_reset_program_applies_non_fused_writes_after_native_reset_and_refreshes_once() -> None:
    model = _Model()
    data = _ResetData()
    op = _Op()
    program = _MotrixSimWriteProgram(
        model,
        data,
        lambda env_ids: env_ids,
        {},
        [(_CompiledWrite(op), {})],
        reset=True,
        refresh_kinematics=True,
    )

    program.execute()

    assert data.reset_calls == [(model, {"forward_kinematic": False})]
    assert op.rows == [data]
    assert model.forward_kinematic_rows == [data]


def test_write_program_skips_kinematic_refresh_when_no_op_requires_it() -> None:
    model = _Model()
    program = _MotrixSimWriteProgram(
        model, _Data(), lambda env_ids: env_ids, {}, [], reset=False, refresh_kinematics=False
    )

    program.execute()

    assert model.forward_kinematic_rows == []
