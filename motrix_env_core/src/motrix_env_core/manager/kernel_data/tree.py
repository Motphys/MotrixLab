# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import fields, is_dataclass
from functools import cache
from typing import Any, get_origin, get_type_hints

import numpy as np
import torch

from motrix_env_core.manager.kernel_data.map import Map
from motrix_env_core.sim.read import SimDataQuery


def is_kernel_data_type(value_type: object) -> bool:
    """Return whether a type is a concrete kernel-data dataclass."""
    return isinstance(value_type, type) and is_dataclass(value_type)


def is_kernel_data(value: object) -> bool:
    """Return whether a value is an instance of a kernel-data dataclass.

    Simulator read queries are dataclasses too; they resolve through the read
    program instead of the kernel-data tree and are excluded here.
    """
    return (
        is_dataclass(value)
        and not isinstance(value, (np.ndarray, torch.Tensor))
        and not isinstance(value, SimDataQuery)
    )


def leaf_value_type(annotation: Any) -> type | None:
    """Return the concrete value type accepted for one leaf annotation."""
    if annotation in (np.ndarray, torch.Tensor):
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


@cache
def _type_hints(cls: type) -> dict[str, Any]:
    return get_type_hints(cls, include_extras=True)


def iter_leaves(value: Any) -> Iterator[tuple[tuple[str, ...], Any, Any]]:
    """Yield (path, annotation, leaf value) triples for one kernel-data value.

    The tree is walked live along dataclass field declaration order and Map
    insertion order. Structural mismatches raise ``TypeError``: a field whose
    value is not its declared kernel-data class, a ``Map``-annotated field
    without a concrete ``Map`` value, and class cycles.
    """
    if not is_kernel_data(value):
        raise TypeError(f"KernelData tree root must be a dataclass value, got {type(value).__name__}.")
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
    hints = _type_hints(declared_type)
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
