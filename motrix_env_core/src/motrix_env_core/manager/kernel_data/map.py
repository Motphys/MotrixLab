# Copyright Motphys Technology Co., Ltd. 2025, 2026

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Generic, TypeVar

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


__all__ = ["Map"]
