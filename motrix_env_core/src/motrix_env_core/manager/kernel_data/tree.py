# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, get_origin, get_type_hints

import numpy as np
import torch

from motrix_env_core.manager.kernel_data.map import Map
from motrix_env_core.sim.read import SimDataQuery

_TREE_SCHEMA_VERSION = 3


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


class BaseDef:
    """Base class for nodes contained in a ``TreeClassDef`` schema tree."""


@dataclass(frozen=True)
class LeafDef(BaseDef):
    """Definition of one terminal value in a logical KernelData tree.

    Attributes:
        path: Field and map-key components from the tree root to this leaf.
        annotation: Declared leaf annotation.
    """

    path: tuple[str, ...]
    annotation: Any


@dataclass(frozen=True)
class FieldDef(BaseDef):
    """Named child of a ``TreeClassDef``.

    Attributes:
        name: Python dataclass field name on the logical kernel-data class.
        tree_def: Logical tree definition for the field value.
    """

    name: str
    tree_def: BaseDef


@dataclass(frozen=True)
class TreeClassDef(BaseDef):
    """Root definition of one concrete kernel-data class tree.

    Every kernel-data value is described by a ``TreeClassDef`` root.
    Its fields recursively contain nested class, map, and leaf definitions.

    Attributes:
        logical_type: Concrete dataclass type described by this definition.
        fields: Field definitions in dataclass declaration order.
        fingerprint: Deterministic identity of the complete class tree schema.
    """

    logical_type: type
    fields: tuple[FieldDef, ...]
    fingerprint: str


@dataclass(frozen=True)
class MapEntryDef(BaseDef):
    """Named child of a value-dependent ``KernelMapDef``.

    Attributes:
        key: Immutable string key exposed by ``Map.__getitem__``.
        tree_def: Logical tree definition for the value associated with ``key``.
    """

    key: str
    tree_def: BaseDef


@dataclass(frozen=True)
class KernelMapDef(BaseDef):
    """Value-derived definition of one immutable heterogeneous ``Map``.

    Attributes:
        path: Field and map-key components from the tree root to this map.
        entries: Map children in stable insertion order.
        fingerprint: Deterministic identity including keys and child schemas.
    """

    path: tuple[str, ...]
    entries: tuple[MapEntryDef, ...]
    fingerprint: str


def tree_def_for(value: Any) -> TreeClassDef:
    """Build and validate the class tree schema for one kernel-data value."""
    if not is_kernel_data(value):
        raise TypeError(f"KernelData tree root must be a dataclass value, got {type(value).__name__}.")
    return _build_class_def(type(value), value, (), ())


def iter_leaves(value: Any, tree_def: BaseDef) -> Iterator[tuple[LeafDef, Any]]:
    """Yield (leaf definition, leaf value) pairs in declaration order.

    The value is walked alongside the schema; structural mismatches (wrong
    class type or changed Map keys) raise ``TypeError``.
    """
    if isinstance(tree_def, LeafDef):
        yield tree_def, value
        return
    if isinstance(tree_def, KernelMapDef):
        if not isinstance(value, Map):
            raise TypeError(f"KernelData value at {'.'.join(tree_def.path)!r} must be Map.")
        if value.keys_tuple != tuple(entry.key for entry in tree_def.entries):
            raise TypeError(
                f"Map at {'.'.join(tree_def.path)!r} has keys {value.keys_tuple!r}, expected "
                f"{tuple(entry.key for entry in tree_def.entries)!r}."
            )
        for entry in tree_def.entries:
            yield from iter_leaves(value[entry.key], entry.tree_def)
        return
    assert isinstance(tree_def, TreeClassDef)
    if type(value) is not tree_def.logical_type:
        raise TypeError(f"KernelData value must be {tree_def.logical_type.__name__}, got {type(value).__name__}.")
    for field in tree_def.fields:
        yield from iter_leaves(getattr(value, field.name), field.tree_def)


def clone_tree(value: Any, tree_def: BaseDef) -> Any:
    """Deep-clone one kernel-data value following its tree schema.

    Tensor leaves are cloned, ndarray leaves are copied, and scalar leaves are
    reused as-is; dataclass fields and Map entries are rebuilt around the
    cloned children.
    """
    if isinstance(tree_def, LeafDef):
        if isinstance(value, torch.Tensor):
            return value.clone()
        if isinstance(value, np.ndarray):
            return value.copy()
        return value
    if isinstance(tree_def, KernelMapDef):
        return Map({entry.key: clone_tree(value[entry.key], entry.tree_def) for entry in tree_def.entries})
    assert isinstance(tree_def, TreeClassDef)
    cloned = tuple(clone_tree(getattr(value, field.name), field.tree_def) for field in tree_def.fields)
    return tree_def.logical_type(*cloned)


def _build_class_def(
    logical_type: type,
    value: Any | None,
    path: tuple[str, ...],
    stack: tuple[type, ...],
) -> TreeClassDef:
    if logical_type in stack:
        chain = " -> ".join(item.__name__ for item in (*stack, logical_type))
        raise TypeError(f"KernelData cycle detected: {chain}.")
    annotations = get_type_hints(logical_type, include_extras=True)
    field_defs: list[FieldDef] = []
    fingerprint_fields: list[Any] = []
    for data_field in fields(logical_type):
        annotation = annotations[data_field.name]
        field_path = (*path, data_field.name)
        field_value = getattr(value, data_field.name) if value is not None else None
        child: BaseDef
        if annotation is Map or get_origin(annotation) is Map:
            if not isinstance(field_value, Map):
                raise TypeError(
                    f"KernelData {logical_type.__name__}.{'.'.join(field_path)} requires a concrete Map value."
                )
            child = _build_map_def(field_value, field_path, (*stack, logical_type))
        elif is_kernel_data_type(annotation):
            child = _build_class_def(annotation, field_value, field_path, (*stack, logical_type))
        else:
            child = LeafDef(field_path, annotation)
        field_defs.append(FieldDef(data_field.name, child))
        fingerprint_fields.append((data_field.name, _tree_fingerprint_part(child)))
    identity = f"{logical_type.__module__}.{logical_type.__qualname__}"
    raw_fingerprint = repr((_TREE_SCHEMA_VERSION, identity, tuple(fingerprint_fields)))
    fingerprint = hashlib.sha256(raw_fingerprint.encode()).hexdigest()
    return TreeClassDef(logical_type, tuple(field_defs), fingerprint)


def _build_map_def(
    value: Map,
    path: tuple[str, ...],
    stack: tuple[type, ...],
) -> KernelMapDef:
    entries: list[MapEntryDef] = []
    fingerprint_entries: list[Any] = []
    for key, entry_value in zip(value.keys_tuple, value.values_tuple, strict=True):
        entry_path = (*path, key)
        if isinstance(entry_value, Map):
            child = _build_map_def(entry_value, entry_path, stack)
        elif is_kernel_data(entry_value):
            child = _build_class_def(type(entry_value), entry_value, entry_path, stack)
        else:
            child = LeafDef(entry_path, type(entry_value))
        entries.append(MapEntryDef(key, child))
        fingerprint_entries.append((key, _tree_fingerprint_part(child)))
    raw_fingerprint = repr((_TREE_SCHEMA_VERSION, "map", path, tuple(fingerprint_entries)))
    fingerprint = hashlib.sha256(raw_fingerprint.encode()).hexdigest()
    return KernelMapDef(path, tuple(entries), fingerprint)


def _tree_fingerprint_part(tree_def: BaseDef) -> Any:
    if isinstance(tree_def, LeafDef):
        return "leaf", tree_def.path, _annotation_identity(tree_def.annotation)
    if isinstance(tree_def, KernelMapDef):
        return (
            "map",
            tree_def.path,
            tuple((entry.key, _tree_fingerprint_part(entry.tree_def)) for entry in tree_def.entries),
        )
    assert isinstance(tree_def, TreeClassDef)
    return (
        "class",
        f"{tree_def.logical_type.__module__}.{tree_def.logical_type.__qualname__}",
        tuple((field.name, _tree_fingerprint_part(field.tree_def)) for field in tree_def.fields),
    )


def _annotation_identity(annotation: Any) -> Any:
    if isinstance(annotation, type):
        return f"{annotation.__module__}.{annotation.__qualname__}"
    return repr(annotation)
