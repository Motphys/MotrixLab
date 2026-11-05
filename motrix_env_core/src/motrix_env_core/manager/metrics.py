# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

import numpy as np

_METRIC_METADATA_KEY = "motrix_metric"
_UNBOUND_METRIC = object()


@dataclass(frozen=True)
class Metric:
    """Export metadata for one per-environment ndarray field on a manager term."""

    name: str | None = None
    dtype: type[np.generic] | None = None


def metric(
    *,
    name: str | None = None,
    dtype: type[np.generic] | None = None,
) -> Any:
    """Declare one per-environment metric field with compiler-managed default backing."""

    return field(
        default=_UNBOUND_METRIC,
        repr=False,
        compare=False,
        metadata={_METRIC_METADATA_KEY: Metric(name=name, dtype=dtype)},
    )


def _field_metric(data_field: Any) -> Metric | None:
    declared = data_field.metadata.get(_METRIC_METADATA_KEY)
    if declared is not None and not isinstance(declared, Metric):
        raise TypeError(f"Metric field {data_field.name!r} has invalid metadata {declared!r}.")
    return declared


def materialize_metrics(value: Any, num_envs: int, *, context: str) -> Any:
    """Replace metric() placeholders with writable ``(num_envs,)`` arrays."""

    if not is_dataclass(value):
        return value
    values = []
    changed = False
    for data_field in fields(value):
        current = getattr(value, data_field.name)
        spec = _field_metric(data_field)
        if current is _UNBOUND_METRIC:
            if spec is None:
                raise TypeError(f"{context} field {data_field.name!r} has an unbound non-metric value.")
            dtype = np.dtype(np.float32 if spec.dtype is None else spec.dtype)
            current = np.empty(num_envs, dtype=dtype)
            changed = True
        values.append(current)
    return type(value)(*values) if changed else value


def collect_metrics(
    value: Any,
    num_envs: int,
    *,
    context: str,
    context_path: tuple[str, ...],
) -> tuple[tuple[str, np.ndarray], ...]:
    """Validate direct metric fields on one canonical manager term."""

    del context_path
    bindings = []
    for data_field in fields(value):
        spec = _field_metric(data_field)
        if spec is None:
            continue
        array = getattr(value, data_field.name)
        if not isinstance(array, np.ndarray):
            raise TypeError(f"{context} metric field {data_field.name!r} must be an ndarray.")
        if array.shape not in {(num_envs,), (num_envs, 1)}:
            raise ValueError(
                f"{context} metric field {data_field.name!r} must have shape "
                f"({num_envs},) or ({num_envs}, 1), got {array.shape}."
            )
        export_dtype = np.dtype(array.dtype if spec.dtype is None else spec.dtype)
        if export_dtype.kind not in "bif":
            raise TypeError(f"{context} metric field {data_field.name!r} export dtype must be bool, integer, or float.")
        bindings.append((data_field.name if spec.name is None else spec.name, array))
    return tuple(bindings)


__all__ = ["Metric", "collect_metrics", "materialize_metrics", "metric"]
