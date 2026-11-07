# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Structured values passed directly to manager kernels.

Kernel data is made of frozen dataclasses, :class:`Map` containers, tensors,
and numeric scalar leaves. This module validates that structure at
construction time; it does not build a runtime lookup or resolved tree.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import fields, is_dataclass
from typing import Any, TypeVar, get_origin, get_type_hints

import numpy as np
import torch

from motrix_env_core.manager.kernel_data.map import Map
from motrix_env_core.sim.read import SimDataQuery

KernelDataType = TypeVar("KernelDataType")


def is_kernel_data_type(value_type: object) -> bool:
    """Return whether a type is a concrete kernel-data dataclass."""
    return isinstance(value_type, type) and is_dataclass(value_type)


def is_kernel_data(value: object) -> bool:
    """Return whether a value is a kernel-data dataclass instance."""
    return (
        is_dataclass(value)
        and not isinstance(value, (np.ndarray, torch.Tensor))
        and not isinstance(value, SimDataQuery)
    )


def leaf_value_type(annotation: Any) -> type | None:
    """Return the concrete value type accepted for one leaf annotation."""
    if annotation is torch.Tensor:
        return annotation
    if annotation in {bool, int, float}:
        return annotation
    if isinstance(annotation, type) and issubclass(annotation, np.generic):
        try:
            dtype = np.dtype(annotation)
        except TypeError:
            return None
        if dtype.type is annotation and dtype.kind in "biuf":
            return annotation
    return None


def iter_leaves(value: Any) -> Iterator[tuple[tuple[str, ...], Any, Any]]:
    """Yield ``(path, annotation, value)`` leaves from one kernel-data value.

    The structure is walked live in dataclass field order and Map insertion
    order. Structural mismatches raise ``TypeError`` for wrong nested classes,
    invalid Map values, and class cycles.
    """
    if not is_kernel_data(value):
        raise TypeError(f"KernelData root must be a dataclass value, got {type(value).__name__}.")
    yield from _iter_class_leaves(type(value), value, (), ())


def _iter_class_leaves(
    declared_type: type,
    value: Any,
    path: tuple[str, ...],
    stack: tuple[type, ...],
) -> Iterator[tuple[tuple[str, ...], Any, Any]]:
    if declared_type in stack:
        chain = " -> ".join(item.__name__ for item in (*stack, declared_type))
        raise TypeError(f"KernelData cycle detected: {chain}.")
    if type(value) is not declared_type:
        raise TypeError(f"KernelData value must be {declared_type.__name__}, got {type(value).__name__}.")
    stack = (*stack, declared_type)
    hints = get_type_hints(declared_type, include_extras=True)
    for data_field in fields(declared_type):
        annotation = hints[data_field.name]
        field_path = (*path, data_field.name)
        field_value = getattr(value, data_field.name)
        if annotation is Map or get_origin(annotation) is Map:
            if not isinstance(field_value, Map):
                raise TypeError(
                    f"KernelData {declared_type.__name__}.{'.'.join(field_path)} requires a concrete Map value."
                )
            yield from _iter_map_leaves(field_value, field_path, stack)
        elif is_kernel_data_type(annotation):
            yield from _iter_class_leaves(annotation, field_value, field_path, stack)
        else:
            yield field_path, annotation, field_value


def _iter_map_leaves(
    value: Map,
    path: tuple[str, ...],
    stack: tuple[type, ...],
) -> Iterator[tuple[tuple[str, ...], Any, Any]]:
    for key, entry_value in zip(value.keys_tuple, value.values_tuple, strict=True):
        entry_path = (*path, key)
        if isinstance(entry_value, Map):
            yield from _iter_map_leaves(entry_value, entry_path, stack)
        elif is_kernel_data(entry_value):
            yield from _iter_class_leaves(type(entry_value), entry_value, entry_path, stack)
        else:
            yield entry_path, type(entry_value), entry_value


def validate_kernel_data(value: KernelDataType, *, context: str) -> KernelDataType:
    """Validate one kernel-data value against the read-only term contract."""
    if not is_kernel_data(value):
        raise TypeError(f"{context} must return a concrete kernel-data dataclass, got {type(value).__name__}.")
    for leaf_path, annotation, _leaf_value in iter_leaves(value):
        if leaf_value_type(annotation) is None:
            raise TypeError(f"{context} KernelData {'.'.join(leaf_path)} expected a tensor or a numeric scalar.")
    return value


__all__ = [
    "is_kernel_data",
    "is_kernel_data_type",
    "iter_leaves",
    "leaf_value_type",
    "validate_kernel_data",
]
