# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, TypeAlias, get_args, get_origin

import numpy as np

from motrix_env_core.manager.fingerprint import function_fingerprint
from motrix_env_core.manager.kernel_data.methods import dispatch_methods
from motrix_env_core.manager.kernel_data.tree import KernelMapDef, LeafDef, TreeClassDef

_LOWERING_SCHEMA_VERSION = 3
_LAYOUT_CACHE: dict[
    tuple[str, bool],
    KernelRecordLayout,
] = {}


class KernelDataScope(str, Enum):
    """How one resolved KernelData leaf is exposed to execution.

    Attributes:
        PER_ENV: Leaf is an ``(N, ...)`` array covering the whole batch; the
            reset schedule clones it for selected lanes.
        SHARED: Pass the complete leaf unchanged to every environment lane.
    """

    PER_ENV = "per_env"
    SHARED = "shared"


SharedArray: TypeAlias = Annotated[np.ndarray, KernelDataScope.SHARED]
"""A writable ndarray passed unchanged to every environment lane."""


@dataclass(frozen=True)
class KernelLeafLayout:
    """Lowered ABI description of one logical KernelData leaf.

    Attributes:
        tree_def: Logical leaf definition that owns the path and annotation.
        value_type: Concrete Python/NumPy type accepted for the leaf value.
        scope: Whether the leaf is per-environment batched or shared.
        slot_index: Zero-based declaration index relative to the root layout.
    """

    tree_def: LeafDef
    value_type: type
    scope: KernelDataScope
    slot_index: int

    @property
    def path(self) -> tuple[str, ...]:
        """Return the logical path inherited from ``tree_def``."""
        return self.tree_def.path


@dataclass(frozen=True)
class KernelFieldLayout:
    """Named lowered child of a ``KernelRecordLayout``.

    Attributes:
        name: Python dataclass field name on the logical record.
        child: Lowered layout for the field value.
    """

    name: str
    child: KernelDataLayout


@dataclass(frozen=True)
class KernelMapEntryLayout:
    """Named lowered child of a ``KernelMapLayout``.

    Attributes:
        key: Immutable string key used for compile-time map lookup.
        child: Lowered layout for the value associated with ``key``.
    """

    key: str
    child: KernelDataLayout


@dataclass(frozen=True)
class KernelMapLayout:
    """Lowered schema for one heterogeneous ``Map``.

    Attributes:
        tree_def: Value-derived logical map definition.
        entries: Lowered children in the same order as the logical map entries.
        fingerprint: Deterministic identity including keys and child schemas.
    """

    tree_def: KernelMapDef
    entries: tuple[KernelMapEntryLayout, ...]
    fingerprint: str


@dataclass(frozen=True)
class KernelRecordLayout:
    """Lowered schema for one concrete ``@kernel_data`` class tree.

    Attributes:
        tree_def: Logical record definition being lowered.
        fields: Lowered children in logical dataclass declaration order.
        fingerprint: Deterministic identity including child types, scopes,
            and dispatch-method code fingerprints.
    """

    tree_def: TreeClassDef
    fields: tuple[KernelFieldLayout, ...]
    fingerprint: str

    @property
    def logical_type(self) -> type:
        """Return the host ``@kernel_data`` type represented by this layout."""
        return self.tree_def.logical_type


KernelDataLayout = KernelRecordLayout | KernelMapLayout | KernelLeafLayout


def iter_layout_leaves(layout: KernelDataLayout) -> tuple[KernelLeafLayout, ...]:
    """Return lowered leaves in stable logical declaration order."""
    if isinstance(layout, KernelLeafLayout):
        return (layout,)
    if isinstance(layout, KernelMapLayout):
        return tuple(leaf for entry in layout.entries for leaf in iter_layout_leaves(entry.child))
    assert isinstance(layout, KernelRecordLayout)
    return tuple(leaf for field in layout.fields for leaf in iter_layout_leaves(field.child))


def _tree_method_fingerprint(tree_def: TreeClassDef | KernelMapDef | LeafDef) -> Any:
    """Include nested records and map values before consulting the layout cache."""
    if isinstance(tree_def, LeafDef):
        return ()
    if isinstance(tree_def, KernelMapDef):
        return tuple((entry.key, _tree_method_fingerprint(entry.tree_def)) for entry in tree_def.entries)
    return (
        tuple(
            (name, function_fingerprint(method))
            for name, method in sorted(dispatch_methods(tree_def.logical_type).items())
        ),
        tuple((field.name, _tree_method_fingerprint(field.tree_def)) for field in tree_def.fields),
    )


class KernelDataLowering:
    """Validate one logical ``TreeClassDef`` and derive its leaf layout.

    The lowering checks that every leaf annotation is a supported
    ndarray/scalar type, assigns each leaf its data scope, and produces a
    deterministic structural fingerprint (field names, types, scopes, and
    dispatch-method code) used for resolved-program invalidation.
    """

    def lower(
        self,
        tree_def: TreeClassDef,
        *,
        context: str,
        force_shared: bool = False,
    ) -> KernelRecordLayout:
        method_key = hashlib.sha256(repr(_tree_method_fingerprint(tree_def)).encode()).hexdigest()
        cache_key = (f"{tree_def.fingerprint}:{method_key}", force_shared)
        cached = _LAYOUT_CACHE.get(cache_key)
        if cached is not None:
            return cached
        layout, _ = self._lower_record(tree_def, 0, force_shared, context)
        _LAYOUT_CACHE[cache_key] = layout
        return layout

    def _lower_record(
        self,
        tree_def: TreeClassDef,
        slot_index: int,
        force_shared: bool,
        context: str,
    ) -> tuple[KernelRecordLayout, int]:
        field_layouts: list[KernelFieldLayout] = []
        fingerprint_fields: list[Any] = []
        for field in tree_def.fields:
            child: KernelDataLayout
            if isinstance(field.tree_def, TreeClassDef):
                child, slot_index = self._lower_record(
                    field.tree_def,
                    slot_index,
                    force_shared,
                    context,
                )
            elif isinstance(field.tree_def, KernelMapDef):
                child, slot_index = self._lower_map(
                    field.tree_def,
                    slot_index,
                    force_shared,
                    context,
                )
            else:
                assert isinstance(field.tree_def, LeafDef)
                value_type, scope = self._leaf_type(
                    field.tree_def.annotation,
                    context=f"{context} KernelData {tree_def.logical_type.__name__}.{'.'.join(field.tree_def.path)}",
                    force_shared=force_shared,
                )
                if value_type is None:
                    raise TypeError(
                        f"{context} KernelData {tree_def.logical_type.__name__}."
                        f"{'.'.join(field.tree_def.path)} expected "
                        "np.ndarray, SharedArray, or numeric scalar."
                    )
                child = KernelLeafLayout(field.tree_def, value_type, scope, slot_index)
                slot_index += 1
            field_layouts.append(KernelFieldLayout(field.name, child))
            fingerprint_fields.append((field.name, self._fingerprint_part(child)))
        identity = f"{tree_def.logical_type.__module__}.{tree_def.logical_type.__qualname__}"
        method_fingerprints = tuple(
            (name, function_fingerprint(method))
            for name, method in sorted(dispatch_methods(tree_def.logical_type).items())
        )
        raw_fingerprint = repr(
            (_LOWERING_SCHEMA_VERSION, tree_def.fingerprint, identity, tuple(fingerprint_fields), method_fingerprints)
        )
        fingerprint = hashlib.sha256(raw_fingerprint.encode()).hexdigest()
        return KernelRecordLayout(tree_def, tuple(field_layouts), fingerprint), slot_index

    def _lower_map(
        self,
        tree_def: KernelMapDef,
        slot_index: int,
        force_shared: bool,
        context: str,
    ) -> tuple[KernelMapLayout, int]:
        entry_layouts: list[KernelMapEntryLayout] = []
        fingerprint_entries: list[Any] = []
        for entry in tree_def.entries:
            if isinstance(entry.tree_def, TreeClassDef):
                child, slot_index = self._lower_record(
                    entry.tree_def,
                    slot_index,
                    force_shared,
                    context,
                )
            elif isinstance(entry.tree_def, KernelMapDef):
                child, slot_index = self._lower_map(
                    entry.tree_def,
                    slot_index,
                    force_shared,
                    context,
                )
            else:
                assert isinstance(entry.tree_def, LeafDef)
                value_type, scope = self._leaf_type(
                    entry.tree_def.annotation,
                    context=f"{context} Map {'.'.join(entry.tree_def.path)}",
                    force_shared=force_shared,
                )
                if value_type is None:
                    raise TypeError(
                        f"{context} Map {'.'.join(entry.tree_def.path)} expected np.ndarray, "
                        "SharedArray, or numeric scalar."
                    )
                child = KernelLeafLayout(entry.tree_def, value_type, scope, slot_index)
                slot_index += 1
            entry_layouts.append(KernelMapEntryLayout(entry.key, child))
            fingerprint_entries.append((entry.key, self._fingerprint_part(child)))
        raw_fingerprint = repr((_LOWERING_SCHEMA_VERSION, tree_def.fingerprint, tuple(fingerprint_entries)))
        fingerprint = hashlib.sha256(raw_fingerprint.encode()).hexdigest()
        return KernelMapLayout(tree_def, tuple(entry_layouts), fingerprint), slot_index

    @staticmethod
    def _fingerprint_part(layout: KernelDataLayout) -> Any:
        if isinstance(layout, KernelLeafLayout):
            return (
                "leaf",
                layout.path,
                f"{layout.value_type.__module__}.{layout.value_type.__qualname__}",
                layout.scope.value,
                layout.slot_index,
            )
        if isinstance(layout, KernelMapLayout):
            return (
                "map",
                layout.tree_def.path,
                layout.fingerprint,
                tuple((entry.key, KernelDataLowering._fingerprint_part(entry.child)) for entry in layout.entries),
            )
        assert isinstance(layout, KernelRecordLayout)
        return (
            "record",
            f"{layout.logical_type.__module__}.{layout.logical_type.__qualname__}",
            layout.fingerprint,
            tuple((field.name, KernelDataLowering._fingerprint_part(field.child)) for field in layout.fields),
        )

    @staticmethod
    def _leaf_type(
        annotation: Any,
        *,
        context: str,
        force_shared: bool,
    ) -> tuple[type[Any] | None, KernelDataScope]:
        metadata: tuple[Any, ...] = ()
        if get_origin(annotation) is Annotated:
            annotation, *metadata_values = get_args(annotation)
            metadata = tuple(metadata_values)
        if metadata and metadata != (KernelDataScope.SHARED,):
            raise TypeError(f"{context} has unsupported or conflicting metadata {metadata!r}.")
        if annotation is np.ndarray:
            scope = KernelDataScope.SHARED if metadata or force_shared else KernelDataScope.PER_ENV
            return annotation, scope
        if metadata:
            raise TypeError(f"{context} scope metadata is only valid on np.ndarray leaves.")
        if annotation in {bool, int, float}:
            return annotation, KernelDataScope.SHARED
        if isinstance(annotation, type) and issubclass(annotation, np.generic):
            try:
                dtype = np.dtype(annotation)
            except TypeError:
                return None, KernelDataScope.SHARED
            if dtype.type is annotation and dtype.kind in "biuf":
                return annotation, KernelDataScope.SHARED
        return None, KernelDataScope.SHARED
