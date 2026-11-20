# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError, dataclass, field

import numpy as np
import pytest
import torch

from motrix_env_core.manager.kernel_data import (
    Map,
    is_kernel_data,
    iter_leaves,
    leaf_value_type,
    validate_kernel_data,
)


@dataclass(frozen=True, slots=True)
class _Pose:
    position: torch.Tensor
    orientation: torch.Tensor


@dataclass(frozen=True, slots=True)
class _RobotBase:
    velocity: torch.Tensor


@dataclass(frozen=True, slots=True)
class _Robot(_RobotBase):
    pose: _Pose


@dataclass(frozen=True, slots=True)
class _Tracking:
    robot: _Robot
    phase: float


@dataclass(frozen=True, slots=True)
class _Cyclic:
    child: "_Cyclic"


def test_kernel_data_is_frozen_and_compiles_nested_inherited_leaves() -> None:
    value = _Tracking(
        _Robot(torch.asarray([[1.0, 2.0]], dtype=torch.float32), _Pose(torch.asarray([[3.0]]), torch.asarray([4.0]))),
        0.5,
    )
    with pytest.raises(FrozenInstanceError):
        value.phase = 1.0  # type: ignore[misc]

    leaves = [(path, leaf_value) for path, _, leaf_value in iter_leaves(value)]
    assert [path for path, _ in leaves] == [
        ("robot", "velocity"),
        ("robot", "pose", "position"),
        ("robot", "pose", "orientation"),
        ("phase",),
    ]
    annotations = [annotation for _, annotation, _ in iter_leaves(value)]
    assert leaf_value_type(annotations[0]) is torch.Tensor
    assert leaf_value_type(annotations[-1]) is float


def test_kernel_data_accepts_tensor_leaves() -> None:
    @dataclass(frozen=True, slots=True)
    class _Tensors:
        buffer: torch.Tensor

    buffer = torch.zeros((2, 3), dtype=torch.float32)
    value = _Tensors(buffer)
    leaves = list(iter_leaves(value))
    assert [leaf_value for _, _, leaf_value in leaves] == [buffer]
    assert leaf_value_type(leaves[0][1]) is torch.Tensor


def test_kernel_data_excludes_declared_host_resources_but_walks_numeric_state() -> None:
    @dataclass(frozen=True, slots=True)
    class _WithHostResource:
        resource: object = field(metadata={"kernel_data": False}, repr=False, compare=False)
        pose: _Pose
        phase: float

    resource = object()
    position = torch.zeros(2, 3)
    orientation = torch.ones(2, 4)
    value = _WithHostResource(resource, _Pose(position, orientation), 0.5)

    assert validate_kernel_data(value, context="host resource") is value
    leaves = list(iter_leaves(value))
    assert [path for path, _, _ in leaves] == [("pose", "position"), ("pose", "orientation"), ("phase",)]
    assert leaves[0][2] is position
    assert leaves[1][2] is orientation
    assert leaves[2][2] == 0.5
    assert value.resource is resource


def test_kernel_data_host_resource_exclusion_does_not_skip_other_field_validation() -> None:
    @dataclass(frozen=True, slots=True)
    class _WithInvalidState:
        resource: object = field(metadata={"kernel_data": False}, repr=False, compare=False)
        state: str

    with pytest.raises(TypeError, match="state expected a tensor or a numeric scalar"):
        validate_kernel_data(_WithInvalidState(object(), "invalid"), context="invalid state")


def test_kernel_data_rejects_ndarray_leaves() -> None:
    @dataclass(frozen=True, slots=True)
    class _Numpy:
        position: np.ndarray

    with pytest.raises(TypeError, match="expected a tensor or a numeric scalar"):
        validate_kernel_data(_Numpy(np.zeros(2)), context="numpy")


def test_iter_leaves_rejects_wrong_class_field() -> None:
    @dataclass(frozen=True, slots=True)
    class _Inner:
        x: torch.Tensor

    @dataclass(frozen=True, slots=True)
    class _Outer:
        inner: _Inner

    @dataclass(frozen=True, slots=True)
    class _Impostor:
        x: torch.Tensor

    with pytest.raises(TypeError, match="must be _Inner, got _Impostor"):
        list(iter_leaves(_Outer(_Impostor(torch.zeros(1)))))  # type: ignore[arg-type]


def test_iter_leaves_rejects_class_cycle() -> None:
    @dataclass(frozen=True, slots=True)
    class _Node:
        child: _Cyclic

    with pytest.raises(TypeError, match="cycle detected"):
        list(iter_leaves(_Node(_Cyclic(None))))  # type: ignore[arg-type]


def test_iter_leaves_rejects_map_annotation_without_map_value() -> None:
    @dataclass(frozen=True, slots=True)
    class _HoldsMap:
        values: Map

    with pytest.raises(TypeError, match="requires a concrete Map value"):
        list(iter_leaves(_HoldsMap({"a": 1.0})))  # type: ignore[arg-type]


def test_kernel_data_rejects_unsupported_leaf() -> None:
    @dataclass(frozen=True, slots=True)
    class _Invalid:
        label: str

    with pytest.raises(TypeError, match="expected a tensor"):
        validate_kernel_data(_Invalid("invalid"), context="invalid")


def test_ndarrays_and_tensors_are_not_kernel_data_values() -> None:
    assert not is_kernel_data(np.zeros(2))
    assert not is_kernel_data(torch.zeros(2))
    assert is_kernel_data(_Pose(torch.zeros(2), torch.zeros(2)))
