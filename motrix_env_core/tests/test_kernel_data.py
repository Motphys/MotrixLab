# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError, dataclass

import numpy as np
import pytest
import torch

from motrix_env_core.manager.kernel_data import (
    Map,
    canonicalize_kernel_data,
    clone_tree,
    is_kernel_data,
    iter_leaves,
    leaf_value_type,
    tree_def_for,
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

    leaves = [(leaf.path, leaf_value) for leaf, leaf_value in iter_leaves(value, tree_def_for(value))]
    assert [path for path, _ in leaves] == [
        ("robot", "velocity"),
        ("robot", "pose", "position"),
        ("robot", "pose", "orientation"),
        ("phase",),
    ]
    annotations = [leaf.annotation for leaf, _ in iter_leaves(value, tree_def_for(value))]
    assert leaf_value_type(annotations[0]) is np.ndarray
    assert leaf_value_type(annotations[-1]) is np.float32


def test_kernel_data_clone_copies_leaves_and_rebuilds_host_value() -> None:
    velocity = np.asarray([[1.0]], dtype=np.float32)
    position = np.asarray([[2.0]], dtype=np.float32)
    orientation = np.asarray([3.0], dtype=np.float32)
    value = _Tracking(_Robot(velocity, _Pose(position, orientation)), np.float32(0.25))
    cloned = clone_tree(value, tree_def_for(value))
    assert cloned.robot.velocity is not velocity
    assert not np.shares_memory(cloned.robot.velocity, velocity)
    assert np.array_equal(cloned.robot.velocity, velocity)
    assert cloned.robot.pose.position is not position
    assert np.array_equal(cloned.robot.pose.position, position)
    assert cloned.phase is value.phase
    assert cloned == value


def test_kernel_data_accepts_tensor_leaves_and_clones_them() -> None:
    @dataclass(frozen=True, slots=True)
    class _Tensors:
        buffer: torch.Tensor

    buffer = torch.zeros((2, 3), dtype=torch.float32)
    value = _Tensors(buffer)
    leaves = list(iter_leaves(value, tree_def_for(value)))
    assert [leaf_value for _, leaf_value in leaves] == [buffer]
    assert leaf_value_type(leaves[0][0].annotation) is torch.Tensor
    cloned = clone_tree(value, tree_def_for(value))
    assert cloned.buffer is not buffer
    assert cloned.buffer.data_ptr() != buffer.data_ptr()
    assert torch.equal(cloned.buffer, buffer)


def test_kernel_data_schema_fingerprint_ignores_values() -> None:
    first_value = _Tracking(
        _Robot(np.asarray([[1.0]], dtype=np.float32), _Pose(np.asarray([[2.0]]), np.asarray([3.0]))),
        np.float32(0.5),
    )
    second_value = _Tracking(
        _Robot(np.asarray([[4.0]], dtype=np.float32), _Pose(np.asarray([[5.0]]), np.asarray([6.0]))),
        np.float32(0.75),
    )
    first = tree_def_for(first_value)
    second = tree_def_for(second_value)
    assert first.fingerprint == second.fingerprint
    assert first.logical_type.__module__ == _Tracking.__module__


def test_map_schema_changes_tree_identity() -> None:
    first = _MappedValues(Map({"value": np.float32(1.0)}))
    second = _MappedValues(Map({"value": np.asarray([[1.0]], dtype=np.float32)}))
    first_tree = tree_def_for(first)
    second_tree = tree_def_for(second)
    assert first_tree.fingerprint != second_tree.fingerprint
    first_map = first_tree.fields[0].tree_def
    second_map = second_tree.fields[0].tree_def
    assert first_map.fingerprint != second_map.fingerprint


def test_iter_leaves_rejects_changed_map_keys() -> None:
    schema = tree_def_for(_MappedValues(Map({"a": np.float32(1.0), "b": np.float32(2.0)})))
    with pytest.raises(TypeError, match="has keys"):
        list(iter_leaves(_MappedValues(Map({"a": np.float32(1.0)})), schema))


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
