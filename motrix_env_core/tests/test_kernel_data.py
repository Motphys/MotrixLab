# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from motrix_env_core.manager.kernel_data import (
    KernelDataLowering,
    KernelDataScope,
    Map,
    SharedArray,
    flatten_kernel_data,
    iter_layout_leaves,
    kernel_data,
    rebuild_lowered,
    unflatten_kernel_data,
)


@kernel_data
class _Pose:
    position: np.ndarray
    orientation: SharedArray


@kernel_data
class _RobotBase:
    velocity: np.ndarray


@kernel_data
class _Robot(_RobotBase):
    pose: _Pose


@kernel_data
class _Tracking:
    robot: _Robot
    phase: np.float32


@kernel_data
class _MappedValues:
    values: Map


@kernel_data
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
    assert [leaf.scope for leaf in leaves] == [
        KernelDataScope.PER_ENV,
        KernelDataScope.PER_ENV,
        KernelDataScope.SHARED,
        KernelDataScope.SHARED,
    ]


def test_kernel_data_flatten_is_zero_copy_and_rebuilds_both_views() -> None:
    velocity = np.asarray([[1.0]], dtype=np.float32)
    position = np.asarray([[2.0]], dtype=np.float32)
    orientation = np.asarray([3.0], dtype=np.float32)
    value = _Tracking(_Robot(velocity, _Pose(position, orientation)), np.float32(0.25))
    leaves, tree_def = flatten_kernel_data(value)
    assert leaves[:3] == (velocity, position, orientation)
    rebuilt = unflatten_kernel_data(tree_def, leaves)
    layout = KernelDataLowering().lower(tree_def, context="test")
    lowered = rebuild_lowered(layout, leaves)
    assert rebuilt.robot.pose.position is position
    assert lowered.robot.pose.orientation is orientation
    assert lowered.phase == np.float32(0.25)


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
    assert first_layout.lowered_type.__module__ == _Tracking.__module__


def test_map_schema_changes_tree_and_proxy_identity() -> None:
    first = _MappedValues(Map({"value": np.float32(1.0)}))
    second = _MappedValues(Map({"value": np.asarray([[1.0]], dtype=np.float32)}))
    _, first_tree = flatten_kernel_data(first)
    _, second_tree = flatten_kernel_data(second)
    first_layout = KernelDataLowering().lower(first_tree, context="first")
    second_layout = KernelDataLowering().lower(second_tree, context="second")
    assert first_tree.fingerprint != second_tree.fingerprint
    assert first_layout.lowered_type is not second_layout.lowered_type
    assert first_layout.fields[0].child.lowered_type is not second_layout.fields[0].child.lowered_type


def test_kernel_data_rejects_unsupported_leaf() -> None:
    @kernel_data
    class _Invalid:
        label: str

    _, tree_def = flatten_kernel_data(_Invalid("invalid"))
    with pytest.raises(TypeError, match="expected np.ndarray"):
        KernelDataLowering().lower(tree_def, context="invalid")


def test_kernel_data_rejects_inherited_field_override_before_slots_transform() -> None:
    @kernel_data
    class _Base:
        value: int = 1

    with pytest.raises(TypeError, match="must not override inherited fields: value"):

        @kernel_data
        class _AnnotationOverride(_Base):
            value: int = 2

    with pytest.raises(TypeError, match="must not override inherited fields: value"):

        @kernel_data
        class _DefaultOverride(_Base):
            value = 2
