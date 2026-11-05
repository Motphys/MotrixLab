# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import sys
from collections.abc import Iterator, Mapping
from typing import Any, Generic, NamedTuple, TypeVar

_MAP_KEYS = "__motrix_kernel_data_map_keys__"
_MAP_SCHEMA = "__motrix_kernel_data_map_schema__"

ValueT = TypeVar("ValueT")


class Map(Mapping[str, ValueT], Generic[ValueT]):
    """Immutable named heterogeneous collection used by kernel contexts."""

    def __init__(self, entries: Mapping[str, ValueT]) -> None:
        self._keys = tuple(entries)
        self._values = tuple(entries.values())
        self._indices = {key: index for index, key in enumerate(self._keys)}

    def __getitem__(self, key: str) -> ValueT:
        try:
            return self._values[self._indices[key]]
        except KeyError as error:
            raise KeyError(f"Map has no entry {key!r}; available entries: {list(self._keys)}.") from error

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    @property
    def keys_tuple(self) -> tuple[str, ...]:
        return self._keys

    @property
    def values_tuple(self) -> tuple[ValueT, ...]:
        return self._values


def map_proxy(
    namespace: str,
    keys: tuple[str, ...],
    *,
    module_name: str,
    schema_fingerprint: str = "",
) -> type[tuple[Any, ...]]:
    """Return a stable lowered NamedTuple type for one static map schema."""
    identity = (module_name, namespace, keys, schema_fingerprint)
    fingerprint = hashlib.sha256(repr(identity).encode()).hexdigest()[:12]
    sanitized = "_".join(part for part in namespace.replace(".", "_").split("_") if part)
    name = f"_{sanitized}Map_{fingerprint}"
    module = sys.modules.get(module_name)
    if module is None:
        raise TypeError(f"Map proxy module {module_name!r} is not loaded.")
    existing = getattr(module, name, None)
    field_names = tuple(f"value_{index}" for index in range(len(keys)))
    if existing is not None:
        if (
            not isinstance(existing, type)
            or not issubclass(existing, tuple)
            or getattr(existing, "_fields", None) != field_names
            or getattr(existing, _MAP_KEYS, None) != keys
            or getattr(existing, _MAP_SCHEMA, None) != schema_fingerprint
        ):
            raise TypeError(f"Map lowered proxy name collision: {module_name}.{name}.")
        return existing
    proxy = NamedTuple(name, [(field_name, Any) for field_name in field_names])  # type: ignore[misc]
    proxy.__module__ = module_name
    proxy.__qualname__ = name
    setattr(proxy, _MAP_KEYS, keys)
    setattr(proxy, _MAP_SCHEMA, schema_fingerprint)
    setattr(module, name, proxy)
    return proxy


def map_keys(value_type: type[Any]) -> tuple[str, ...] | None:
    """Return static map keys for a lowered proxy type."""
    keys = getattr(value_type, _MAP_KEYS, None)
    return keys if isinstance(keys, tuple) else None


__all__ = [
    "Map",
    "map_keys",
    "map_proxy",
]
