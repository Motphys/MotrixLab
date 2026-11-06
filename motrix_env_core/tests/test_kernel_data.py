# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError, dataclass

import numpy as np
import pytest
import torch

from motrix_env_core.manager.kernel_data import (
    KernelDataLowering,
    Map,
    flatten_kernel_data,
    is_kernel_data,
    iter_layout_leaves,
    unflatten_kernel_data,
)


@dataclass(frozen=True, slots=True)
class _Pose:
    position: np.ndarray
    orientation: np.ndarray


@dataclass(frozen=True, slots=True)
class _RobotBase:
    velocity: np.ndarray


@dataclass(frozen=True, slots=True)
class _Robot(_RobotBase):
    pose: _Pose


@dataclass(frozen=True, slots=True)
class _Tracking:
    robot: _Robot
    phase: np.float32


@dataclass(frozen=True, slots=True)
class _MappedValues:
    values: Map


@dataclass(frozen=True, slots=True)
class _ArrayMapValues:
    values: Map[np.ndarray]


def test_kernel_data_is_frozen_and_compiles_nested_inherited_leaves() -> None:
    value = _Tracking(
        _Robot(np.asarray([[1.0, 2.0]], dtype=np.float32), _Pose(np.asarray([[3.0]]), np.asarray([4.0]))),
        np.float32(0.5),
    )
    with pytest.raises(FrozenInstanceError):
        value.phase = np.float32(1.0)  # type: ignore[misc]

    _, tree_def = flatten_kernel_data(value)
    layout = KernelDataLowering().lower(tree_def, context="test")
    leaves = iter_layout_leaves(layout)
    assert [leaf.path for leaf in leaves] == [
        ("robot", "velocity"),
        ("robot", "pose", "position"),
        ("robot", "pose", "orientation"),
        ("phase",),
    ]


def test_kernel_data_flatten_is_zero_copy_and_rebuilds_host_value() -> None:
    velocity = np.asarray([[1.0]], dtype=np.float32)
    position = np.asarray([[2.0]], dtype=np.float32)
    orientation = np.asarray([3.0], dtype=np.float32)
    value = _Tracking(_Robot(velocity, _Pose(position, orientation)), np.float32(0.25))
    leaves, tree_def = flatten_kernel_data(value)
    assert leaves[:3] == (velocity, position, orientation)
    rebuilt = unflatten_kernel_data(tree_def, leaves)
    KernelDataLowering().lower(tree_def, context="test")
    assert rebuilt.robot.pose.position is position
    assert rebuilt.robot.pose.orientation is orientation
    assert rebuilt.phase == np.float32(0.25)


def test_kernel_data_accepts_tensor_leaves() -> None:
    @dataclass(frozen=True, slots=True)
    class _Tensors:
        buffer: torch.Tensor

    buffer = torch.zeros((2, 3), dtype=torch.float32)
    value = _Tensors(buffer)
    leaves, tree_def = flatten_kernel_data(value)
    assert leaves == (buffer,)
    layout = KernelDataLowering().lower(tree_def, context="test")
    assert iter_layout_leaves(layout)[0].value_type is torch.Tensor


def test_kernel_data_schema_fingerprint_ignores_values() -> None:
    first_value = _Tracking(
        _Robot(np.asarray([[1.0]], dtype=np.float32), _Pose(np.asarray([[2.0]]), np.asarray([3.0]))),
        np.float32(0.5),
    )
    second_value = _Tracking(
        _Robot(np.asarray([[4.0]], dtype=np.float32), _Pose(np.asarray([[5.0]]), np.asarray([6.0]))),
        np.float32(0.75),
    )
    _, first = flatten_kernel_data(first_value)
    _, second = flatten_kernel_data(second_value)
    assert first is second
    assert first.fingerprint == second.fingerprint
    first_layout = KernelDataLowering().lower(first, context="first")
    second_layout = KernelDataLowering().lower(second, context="second")
    assert first_layout is second_layout
    assert first_layout.logical_type.__module__ == _Tracking.__module__


def test_map_schema_changes_tree_and_layout_identity() -> None:
    first = _MappedValues(Map({"value": np.float32(1.0)}))
    second = _MappedValues(Map({"value": np.asarray([[1.0]], dtype=np.float32)}))
    _, first_tree = flatten_kernel_data(first)
    _, second_tree = flatten_kernel_data(second)
    first_layout = KernelDataLowering().lower(first_tree, context="first")
    second_layout = KernelDataLowering().lower(second_tree, context="second")
    assert first_tree.fingerprint != second_tree.fingerprint
    assert first_layout.fingerprint != second_layout.fingerprint
    assert first_layout.fields[0].child.fingerprint != second_layout.fields[0].child.fingerprint


def test_kernel_data_rejects_unsupported_leaf() -> None:
    @dataclass(frozen=True, slots=True)
    class _Invalid:
        label: str

    _, tree_def = flatten_kernel_data(_Invalid("invalid"))
    with pytest.raises(TypeError, match="expected a tensor"):
        KernelDataLowering().lower(tree_def, context="invalid")


def test_ndarrays_and_tensors_are_not_kernel_data_values() -> None:
    assert not is_kernel_data(np.zeros(2))
    assert not is_kernel_data(torch.zeros(2))
    assert is_kernel_data(_Pose(np.zeros(2), np.zeros(2)))
