# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

import numpy as np
import torch

_METRIC_METADATA_KEY = "motrix_metric"
_UNBOUND_METRIC = object()
_NUMPY_DTYPES = {
    torch.bool: np.dtype(np.bool_),
    torch.int8: np.dtype(np.int8),
    torch.int16: np.dtype(np.int16),
    torch.int32: np.dtype(np.int32),
    torch.int64: np.dtype(np.int64),
    torch.uint8: np.dtype(np.uint8),
    torch.uint16: np.dtype(np.uint16),
    torch.uint32: np.dtype(np.uint32),
    torch.float16: np.dtype(np.float16),
    torch.float32: np.dtype(np.float32),
    torch.float64: np.dtype(np.float64),
}


@dataclass(frozen=True)
class Metric:
    """Export metadata for one per-environment tensor field on a manager term."""

    name: str | None = None
    dtype: torch.dtype | None = None


def metric(
    *,
    name: str | None = None,
    dtype: torch.dtype | None = None,
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
    """Replace metric() placeholders with writable ``(num_envs,)`` tensor views."""

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
            dtype = torch.float32 if spec.dtype is None else spec.dtype
            current = torch.empty(num_envs, dtype=dtype)
            changed = True
        values.append(current)
    return type(value)(*values) if changed else value


def collect_metrics(
    value: Any,
    num_envs: int,
    *,
    context: str,
    context_path: tuple[str, ...],
) -> tuple[tuple[str, torch.Tensor], ...]:
    """Validate direct metric fields on one canonical manager term.

    Metric fields remain torch tensors for kernel execution. The manager
    frontend creates zero-copy NumPy views only when exposing host metrics.
    """
    del context_path
    bindings = []
    for data_field in fields(value):
        spec = _field_metric(data_field)
        if spec is None:
            continue
        tensor = getattr(value, data_field.name)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{context} metric field {data_field.name!r} must hold a torch.Tensor.")
        if tensor.device.type != "cpu":
            raise TypeError(f"{context} metric field {data_field.name!r} must be CPU-backed.")
        if tensor.shape not in {(num_envs,), (num_envs, 1)}:
            raise ValueError(
                f"{context} metric field {data_field.name!r} must have shape "
                f"({num_envs},) or ({num_envs}, 1), got {tensor.shape}."
            )
        export_dtype = tensor.numpy().dtype if spec.dtype is None else _NUMPY_DTYPES[spec.dtype]
        if export_dtype.kind not in "bif":
            raise TypeError(f"{context} metric field {data_field.name!r} export dtype must be bool, integer, or float.")
        bindings.append((data_field.name if spec.name is None else spec.name, tensor))
    return tuple(bindings)


__all__ = ["Metric", "collect_metrics", "materialize_metrics", "metric"]
