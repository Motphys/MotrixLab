# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields, is_dataclass
from typing import Annotated, Any, get_args, get_origin, get_type_hints

import numpy as np
import torch

from motrix_env_core.manager.kernel_data.map import Map
from motrix_env_core.sim.read import SimDataQuery

_TREE_SCHEMA_VERSION = 2
_TREE_DEF_CACHE: dict[type, TreeClassDef] = {}


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


class BaseDef:
    """Base class for nodes contained in a ``TreeClassDef`` schema tree."""


@dataclass(frozen=True)
class LeafDef(BaseDef):
    """Definition of one terminal value in a logical KernelData tree.

    Attributes:
        path: Field and map-key components from the tree root to this leaf.
        annotation: Declared leaf annotation, including any ``Annotated`` metadata.
    """

    path: tuple[str, ...]
    annotation: Any


@dataclass(frozen=True)
class FieldDef:
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

    Every flattened kernel-data value is represented by a ``TreeClassDef`` root.
    Its fields recursively contain nested class, map, and leaf definitions.

    Attributes:
        logical_type: Frozen dataclass type reconstructed by host unflattening.
        fields: Field definitions in dataclass declaration order.
        fingerprint: Deterministic identity of the complete class tree schema.
    """

    logical_type: type
    fields: tuple[FieldDef, ...]
    fingerprint: str

    def flatten(self, value: Any) -> tuple[Any, ...]:
        """Flatten a matching class value without copying leaves."""
        return _flatten(value, self)

    def unflatten(self, leaves: tuple[Any, ...]) -> Any:
        """Rebuild one class value from ordered leaves."""
        return unflatten_kernel_data(self, leaves)


@dataclass(frozen=True)
class MapEntryDef:
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


def flatten_kernel_data(value: Any) -> tuple[tuple[Any, ...], TreeClassDef]:
    """Return zero-copy leaves and the concrete class tree definition for one kernel-data value."""
    if not is_kernel_data(value):
        raise TypeError(f"KernelData tree root must be a dataclass value, got {type(value).__name__}.")
    tree_def, dynamic = _build_class_def(type(value), value, (), ())
    if not dynamic:
        cached = _TREE_DEF_CACHE.get(type(value))
        if cached is None:
            _TREE_DEF_CACHE[type(value)] = tree_def
        else:
            tree_def = cached
    return _flatten(value, tree_def), tree_def


def unflatten_kernel_data(tree_def: TreeClassDef, leaves: tuple[Any, ...]) -> Any:
    """Rebuild a logical host value from a tree definition and ordered leaves."""
    value, next_index = _unflatten(tree_def, leaves, 0)
    if next_index != len(leaves):
        raise ValueError("KernelData unflatten received extra leaves.")
    return value


def iter_tree_leaves(tree_def: BaseDef) -> tuple[LeafDef, ...]:
    """Return logical leaves in stable declaration order."""
    if isinstance(tree_def, LeafDef):
        return (tree_def,)
    if isinstance(tree_def, TreeClassDef):
        return tuple(leaf for field in tree_def.fields for leaf in iter_tree_leaves(field.tree_def))
    assert isinstance(tree_def, KernelMapDef)
    return tuple(leaf for entry in tree_def.entries for leaf in iter_tree_leaves(entry.tree_def))


def _flatten(value: Any, tree_def: BaseDef) -> tuple[Any, ...]:
    if isinstance(tree_def, LeafDef):
        return (value,)
    if isinstance(tree_def, KernelMapDef):
        if not isinstance(value, Map):
            raise TypeError(f"KernelData value at {'.'.join(tree_def.path)!r} must be Map.")
        if value.keys_tuple != tuple(entry.key for entry in tree_def.entries):
            raise TypeError(
                f"Map at {'.'.join(tree_def.path)!r} has keys {value.keys_tuple!r}, expected "
                f"{tuple(entry.key for entry in tree_def.entries)!r}."
            )
        return tuple(leaf for entry in tree_def.entries for leaf in _flatten(value[entry.key], entry.tree_def))
    assert isinstance(tree_def, TreeClassDef)
    if type(value) is not tree_def.logical_type:
        raise TypeError(f"KernelData value must be {tree_def.logical_type.__name__}, got {type(value).__name__}.")
    return tuple(leaf for field in tree_def.fields for leaf in _flatten(getattr(value, field.name), field.tree_def))


def _unflatten(
    tree_def: BaseDef,
    leaves: tuple[Any, ...],
    index: int,
) -> tuple[Any, int]:
    if isinstance(tree_def, LeafDef):
        if index >= len(leaves):
            raise ValueError("KernelData unflatten received too few leaves.")
        return leaves[index], index + 1
    if isinstance(tree_def, KernelMapDef):
        entries: dict[str, Any] = {}
        for entry in tree_def.entries:
            value, index = _unflatten(entry.tree_def, leaves, index)
            entries[entry.key] = value
        return Map(entries), index
    assert isinstance(tree_def, TreeClassDef)
    values = []
    for field in tree_def.fields:
        value, index = _unflatten(field.tree_def, leaves, index)
        values.append(value)
    return tree_def.logical_type(*values), index


def _build_class_def(
    logical_type: type,
    value: Any | None,
    path: tuple[str, ...],
    stack: tuple[type, ...],
) -> tuple[TreeClassDef, bool]:
    if logical_type in stack:
        chain = " -> ".join(item.__name__ for item in (*stack, logical_type))
        raise TypeError(f"KernelData cycle detected: {chain}.")
    annotations = get_type_hints(logical_type, include_extras=True)
    field_defs: list[FieldDef] = []
    fingerprint_fields: list[Any] = []
    dynamic = False
    for data_field in fields(logical_type):
        annotation = annotations[data_field.name]
        field_path = (*path, data_field.name)
        node_annotation = _node_annotation(annotation)
        field_value = getattr(value, data_field.name) if value is not None else None
        child: BaseDef
        if node_annotation is Map or get_origin(node_annotation) is Map:
            if get_origin(annotation) is Annotated:
                raise TypeError(
                    f"KernelData {logical_type.__name__}.{'.'.join(field_path)} map fields cannot declare metadata."
                )
            if not isinstance(field_value, Map):
                raise TypeError(
                    f"KernelData {logical_type.__name__}.{'.'.join(field_path)} requires a concrete Map value."
                )
            child, _ = _build_map_def(field_value, field_path, (*stack, logical_type))
            dynamic = True
        elif is_kernel_data_type(node_annotation):
            if get_origin(annotation) is Annotated:
                raise TypeError(
                    f"KernelData {logical_type.__name__}.{'.'.join(field_path)} class fields cannot declare "
                    "leaf metadata."
                )
            child, child_dynamic = _build_class_def(
                node_annotation,
                field_value,
                field_path,
                (*stack, logical_type),
            )
            dynamic = dynamic or child_dynamic
        else:
            child = LeafDef(field_path, annotation)
        field_defs.append(FieldDef(data_field.name, child))
        fingerprint_fields.append((data_field.name, _tree_fingerprint_part(child)))
    identity = f"{logical_type.__module__}.{logical_type.__qualname__}"
    raw_fingerprint = repr((_TREE_SCHEMA_VERSION, identity, tuple(fingerprint_fields)))
    fingerprint = hashlib.sha256(raw_fingerprint.encode()).hexdigest()
    return TreeClassDef(logical_type, tuple(field_defs), fingerprint), dynamic


def _build_map_def(
    value: Map,
    path: tuple[str, ...],
    stack: tuple[type, ...],
) -> tuple[KernelMapDef, bool]:
    entries: list[MapEntryDef] = []
    fingerprint_entries: list[Any] = []
    for key, entry_value in zip(value.keys_tuple, value.values_tuple, strict=True):
        entry_path = (*path, key)
        if isinstance(entry_value, Map):
            child, _ = _build_map_def(entry_value, entry_path, stack)
        elif is_kernel_data(entry_value):
            child, _ = _build_class_def(type(entry_value), entry_value, entry_path, stack)
        else:
            child = LeafDef(entry_path, type(entry_value))
        entries.append(MapEntryDef(key, child))
        fingerprint_entries.append((key, _tree_fingerprint_part(child)))
    raw_fingerprint = repr((_TREE_SCHEMA_VERSION, "map", path, tuple(fingerprint_entries)))
    fingerprint = hashlib.sha256(raw_fingerprint.encode()).hexdigest()
    return KernelMapDef(path, tuple(entries), fingerprint), True


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
    if get_origin(annotation) is Annotated:
        value_type, *metadata = get_args(annotation)
        return "annotated", _annotation_identity(value_type), tuple(_metadata_identity(value) for value in metadata)
    if isinstance(annotation, type):
        return f"{annotation.__module__}.{annotation.__qualname__}"
    return repr(annotation)


def _metadata_identity(value: Any) -> Any:
    value_type = type(value)
    if hasattr(value, "value"):
        return f"{value_type.__module__}.{value_type.__qualname__}", value.value
    return repr(value)


def _node_annotation(annotation: Any) -> Any:
    return get_args(annotation)[0] if get_origin(annotation) is Annotated else annotation
