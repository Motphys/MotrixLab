# Copyright Motphys Technology Co., Ltd. 2025, 2026

from dataclasses import FrozenInstanceError, dataclass

import numpy as np
import pytest
import torch

from motrix_env_core.manager.kernel_data import (
    Map,
    canonicalize_kernel_data,
    flatten_kernel_data,
    is_kernel_data,
    iter_tree_leaves,
    leaf_value_type,
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


def test_kernel_data_is_frozen_and_compiles_nested_inherited_leaves() -> None:
    value = _Tracking(
        _Robot(np.asarray([[1.0, 2.0]], dtype=np.float32), _Pose(np.asarray([[3.0]]), np.asarray([4.0]))),
        np.float32(0.5),
    )
    with pytest.raises(FrozenInstanceError):
        value.phase = np.float32(1.0)  # type: ignore[misc]

    _, tree_def = flatten_kernel_data(value)
    leaves = iter_tree_leaves(tree_def)
    assert [leaf.path for leaf in leaves] == [
        ("robot", "velocity"),
        ("robot", "pose", "position"),
        ("robot", "pose", "orientation"),
        ("phase",),
    ]
    assert leaf_value_type(leaves[0].annotation) is np.ndarray
    assert leaf_value_type(leaves[-1].annotation) is np.float32


def test_kernel_data_flatten_is_zero_copy_and_rebuilds_host_value() -> None:
    velocity = np.asarray([[1.0]], dtype=np.float32)
    position = np.asarray([[2.0]], dtype=np.float32)
    orientation = np.asarray([3.0], dtype=np.float32)
    value = _Tracking(_Robot(velocity, _Pose(position, orientation)), np.float32(0.25))
    leaves, tree_def = flatten_kernel_data(value)
    assert leaves[:3] == (velocity, position, orientation)
    rebuilt = unflatten_kernel_data(tree_def, leaves)
    canonicalize_kernel_data(value, context="test")
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
    assert leaf_value_type(iter_tree_leaves(tree_def)[0].annotation) is torch.Tensor


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
    assert first.fingerprint == second.fingerprint
    assert first.logical_type.__module__ == _Tracking.__module__


def test_map_schema_changes_tree_identity() -> None:
    first = _MappedValues(Map({"value": np.float32(1.0)}))
    second = _MappedValues(Map({"value": np.asarray([[1.0]], dtype=np.float32)}))
    _, first_tree = flatten_kernel_data(first)
    _, second_tree = flatten_kernel_data(second)
    assert first_tree.fingerprint != second_tree.fingerprint
    first_map = first_tree.fields[0].tree_def
    second_map = second_tree.fields[0].tree_def
    assert first_map.fingerprint != second_map.fingerprint


def test_kernel_data_rejects_unsupported_leaf() -> None:
    @dataclass(frozen=True, slots=True)
    class _Invalid:
        label: str

    with pytest.raises(TypeError, match="expected a tensor"):
        canonicalize_kernel_data(_Invalid("invalid"), context="invalid")


def test_ndarrays_and_tensors_are_not_kernel_data_values() -> None:
    assert not is_kernel_data(np.zeros(2))
    assert not is_kernel_data(torch.zeros(2))
    assert is_kernel_data(_Pose(np.zeros(2), np.zeros(2)))
