# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, NamedTuple, TypeAlias, get_args, get_origin

import numpy as np

from motrix_env_core.numba.fingerprint import function_fingerprint
from motrix_env_core.numba.kernel_data.map import map_proxy
from motrix_env_core.numba.kernel_data.methods import dispatch_methods, register_proxy_method
from motrix_env_core.numba.kernel_data.tree import KernelMapDef, LeafDef, TreeClassDef

_LOWERING_SCHEMA_VERSION = 2
_LAYOUT_CACHE: dict[
    tuple[str, bool],
    KernelRecordLayout,
] = {}


class KernelDataScope(str, Enum):
    """How one flattened KernelData leaf is exposed to a Numba execution lane.

    Attributes:
        PER_ENV: Select ``leaf[env_id]`` while reconstructing each execution lane.
        SHARED: Pass the complete leaf unchanged to every execution lane.
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
        value_type: Concrete Python/NumPy type accepted by the kernel input slot.
        scope: Whether lane reconstruction indexes by environment or shares the leaf.
        slot_index: Zero-based slot relative to the root layout's flattened inputs.
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
        lowered_type: Schema-specific NamedTuple proxy constructed in Numba lanes.
        entries: Lowered children in the same order as the logical map entries.
        fingerprint: Deterministic identity including child types, scopes, and slots.
    """

    tree_def: KernelMapDef
    lowered_type: type[tuple[Any, ...]]
    entries: tuple[KernelMapEntryLayout, ...]
    fingerprint: str


@dataclass(frozen=True)
class KernelRecordLayout:
    """Lowered schema for one concrete ``@kernel_data`` class tree.

    Attributes:
        tree_def: Logical record definition being lowered.
        lowered_type: Schema-specific NamedTuple proxy constructed in Numba lanes.
        fields: Lowered children in logical dataclass declaration order.
        fingerprint: Deterministic identity including child types, scopes, and slots.
    """

    tree_def: TreeClassDef
    lowered_type: type[tuple[Any, ...]]
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


def rebuild_lowered(layout: KernelDataLayout, leaves: tuple[Any, ...]) -> Any:
    """Rebuild compiler-owned lowered proxies from ordered leaves."""
    value, next_index = _rebuild_lowered(layout, leaves, 0)
    if next_index != len(leaves):
        raise ValueError("KernelData lowered rebuild received extra leaves.")
    return value


def _rebuild_lowered(layout: KernelDataLayout, leaves: tuple[Any, ...], index: int) -> tuple[Any, int]:
    if isinstance(layout, KernelLeafLayout):
        if index >= len(leaves):
            raise ValueError("KernelData lowered rebuild received too few leaves.")
        return leaves[index], index + 1
    if isinstance(layout, KernelMapLayout):
        values = []
        for entry in layout.entries:
            value, index = _rebuild_lowered(entry.child, leaves, index)
            values.append(value)
        return layout.lowered_type(*values), index
    assert isinstance(layout, KernelRecordLayout)
    values = []
    for field in layout.fields:
        value, index = _rebuild_lowered(field.child, leaves, index)
        values.append(value)
    return layout.lowered_type(*values), index


def lane_expression(layout: KernelDataLayout, input_offset: int, env_symbol: str = "env_id") -> str:
    """Generate a bottom-up lane-local reconstruction expression."""
    if isinstance(layout, KernelLeafLayout):
        input_symbol = f"input_{input_offset + layout.slot_index}"
        return f"{input_symbol}[{env_symbol}]" if layout.scope is KernelDataScope.PER_ENV else input_symbol
    if isinstance(layout, KernelMapLayout):
        arguments = ", ".join(lane_expression(entry.child, input_offset, env_symbol) for entry in layout.entries)
        return f"{proxy_symbol(layout.lowered_type)}({arguments})"
    assert isinstance(layout, KernelRecordLayout)
    arguments = ", ".join(lane_expression(field.child, input_offset, env_symbol) for field in layout.fields)
    return f"{proxy_symbol(layout.lowered_type)}({arguments})"


def proxy_types(layout: KernelDataLayout) -> tuple[type[tuple[Any, ...]], ...]:
    """Return all lowered record proxy types in child-first order."""
    if isinstance(layout, KernelLeafLayout):
        return ()
    if isinstance(layout, KernelMapLayout):
        return tuple(proxy for entry in layout.entries for proxy in proxy_types(entry.child)) + (layout.lowered_type,)
    assert isinstance(layout, KernelRecordLayout)
    return tuple(proxy for field in layout.fields for proxy in proxy_types(field.child)) + (layout.lowered_type,)


def proxy_symbol(proxy: type[tuple[Any, ...]]) -> str:
    return proxy.__name__


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
    """Lower one logical ``TreeClassDef`` into a fixed flat Numba ABI layout."""

    def lower(
        self,
        tree_def: TreeClassDef,
        *,
        context: str,
        force_shared: bool = False,
    ) -> KernelDataLayout:
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
                    tree_def.logical_type.__module__,
                )
            else:
                assert isinstance(field.tree_def, LeafDef)
                value_type, scope = self._leaf_type(
                    field.tree_def.annotation,
                    context=f"{context} KernelData {tree_def.logical_type.__name__}.{'.'.join(field.tree_def.path)}",
                    force_shared=force_shared,
                    force_per_env=False,
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
        lowered_type = self._lowered_proxy(
            tree_def.logical_type,
            tuple(field.name for field in field_layouts),
            fingerprint,
        )
        return KernelRecordLayout(tree_def, lowered_type, tuple(field_layouts), fingerprint), slot_index

    def _lower_map(
        self,
        tree_def: KernelMapDef,
        slot_index: int,
        force_shared: bool,
        context: str,
        module_name: str,
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
                    module_name,
                )
            else:
                assert isinstance(entry.tree_def, LeafDef)
                value_type, scope = self._leaf_type(
                    entry.tree_def.annotation,
                    context=f"{context} Map {'.'.join(entry.tree_def.path)}",
                    force_shared=force_shared,
                    force_per_env=False,
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
        namespace = ".".join(tree_def.path) or "root"
        lowered_type = map_proxy(
            namespace,
            tuple(entry.key for entry in entry_layouts),
            module_name=module_name,
            schema_fingerprint=fingerprint,
        )
        return KernelMapLayout(tree_def, lowered_type, tuple(entry_layouts), fingerprint), slot_index

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
    def _lowered_proxy(
        logical_type: type,
        field_names: tuple[str, ...],
        fingerprint: str,
    ) -> type[tuple[Any, ...]]:
        sanitized = "_".join(
            part for part in logical_type.__qualname__.replace("<locals>", "locals").split(".") if part
        )
        name = f"_{sanitized}KernelData_{fingerprint[:12]}"
        module = sys.modules.get(logical_type.__module__)
        if module is None:
            raise TypeError(f"KernelData module {logical_type.__module__!r} is not loaded.")
        existing = getattr(module, name, None)
        if existing is not None:
            if (
                not isinstance(existing, type)
                or not issubclass(existing, tuple)
                or getattr(existing, "_fields", None) != field_names
            ):
                raise TypeError(f"KernelData lowered proxy name collision: {logical_type.__module__}.{name}.")
            return existing
        lowered = NamedTuple(name, [(field_name, Any) for field_name in field_names])  # type: ignore[misc]
        lowered.__module__ = logical_type.__module__
        lowered.__qualname__ = name
        setattr(module, name, lowered)
        for method_name, method in dispatch_methods(logical_type).items():
            register_proxy_method(lowered, method_name, method)
        return lowered

    @staticmethod
    def _leaf_type(
        annotation: Any,
        *,
        context: str,
        force_shared: bool,
        force_per_env: bool = False,
    ) -> tuple[type[Any] | None, KernelDataScope]:
        metadata: tuple[Any, ...] = ()
        if get_origin(annotation) is Annotated:
            annotation, *metadata_values = get_args(annotation)
            metadata = tuple(metadata_values)
        if metadata and metadata != (KernelDataScope.SHARED,):
            raise TypeError(f"{context} has unsupported or conflicting metadata {metadata!r}.")
        if annotation is np.ndarray:
            if metadata and force_per_env:
                raise TypeError(f"{context} cannot be both SharedArray and explicitly per-environment.")
            scope = (
                KernelDataScope.PER_ENV
                if force_per_env
                else (KernelDataScope.SHARED if metadata or force_shared else KernelDataScope.PER_ENV)
            )
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
