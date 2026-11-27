# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

import torch

from motrix_env_core.manager.kernel_data import is_kernel_data

_METRIC_METADATA_KEY = "motrix_metric"
_UNBOUND_METRIC = object()
_EXPORT_DTYPES = frozenset(
    {
        torch.bool,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.float16,
        torch.float32,
        torch.float64,
    }
)


@dataclass(frozen=True)
class Metric:
    """Export metadata for one tensor field on a manager term.

    Scalar metrics (``scalar=True``) hold one batch-level value in a ``(1,)``
    tensor — batch-level gauges that kernels write directly, without the
    per-environment lane semantics.
    """

    name: str | None = None
    dtype: torch.dtype | None = None
    scalar: bool = False


def metric(
    *,
    name: str | None = None,
    dtype: torch.dtype | None = None,
    scalar: bool = False,
) -> Any:
    """Declare one metric field with compiler-managed default backing."""

    return field(
        default=_UNBOUND_METRIC,
        repr=False,
        compare=False,
        metadata={_METRIC_METADATA_KEY: Metric(name=name, dtype=dtype, scalar=scalar)},
    )


def _field_metric(data_field: Any) -> Metric | None:
    declared = data_field.metadata.get(_METRIC_METADATA_KEY)
    if declared is not None and not isinstance(declared, Metric):
        raise TypeError(f"Metric field {data_field.name!r} has invalid metadata {declared!r}.")
    return declared


def materialize_metrics(value: Any, num_envs: int, *, context: str) -> Any:
    """Replace metric() placeholders with writable ``(num_envs,)`` tensor views.

    Recurses into nested kernel-data members, so a term holding a structured
    sub-object (e.g. a curriculum sampler) rebuilds that child with its
    metric fields materialized; parents are rebuilt only when a child changed.
    """

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
            shape = () if spec.scalar else (num_envs,)
            current = torch.empty(shape, dtype=dtype)
            changed = True
        elif is_kernel_data(current):
            materialized = materialize_metrics(current, num_envs, context=f"{context} field {data_field.name!r}")
            if materialized is not current:
                current = materialized
                changed = True
        values.append(current)
    return type(value)(*values) if changed else value


def collect_metrics(
    value: Any,
    num_envs: int,
    *,
    context: str,
    context_path: tuple[str, ...],
    device: torch.device = torch.device("cpu"),
) -> tuple[tuple[str, torch.Tensor], ...]:
    """Collect metric fields from one canonical manager term, recursing into
    nested kernel-data members.

    Metric fields remain torch tensors for kernel execution. The manager
    frontend creates zero-copy NumPy views only when exposing host metrics.
    Names come from each field's declaration and must stay unique across the
    whole term tree (the manager frontend rejects duplicates).
    """
    bindings = []
    for data_field in fields(value):
        spec = _field_metric(data_field)
        child = getattr(value, data_field.name)
        if spec is None:
            if is_kernel_data(child):
                bindings.extend(
                    collect_metrics(
                        child,
                        num_envs,
                        context=f"{context} field {data_field.name!r}",
                        context_path=(*context_path, data_field.name),
                        device=device,
                    )
                )
            continue
        tensor = child
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{context} metric field {data_field.name!r} must hold a torch.Tensor.")
        if tensor.device != device:
            raise TypeError(f"{context} metric field {data_field.name!r} must be on {device}.")
        if spec.scalar:
            if tensor.ndim != 0:
                raise ValueError(
                    f"{context} scalar metric field {data_field.name!r} must be 0-d, got shape {tensor.shape}."
                )
        elif tensor.shape not in {(num_envs,), (num_envs, 1)}:
            raise ValueError(
                f"{context} metric field {data_field.name!r} must have shape "
                f"({num_envs},) or ({num_envs}, 1), got {tensor.shape}."
            )
        dtype = tensor.dtype if spec.dtype is None else spec.dtype
        if dtype not in _EXPORT_DTYPES:
            raise TypeError(
                f"{context} metric field {data_field.name!r} export dtype must be bool, integer, or float, got {dtype}."
            )
        bindings.append((data_field.name if spec.name is None else spec.name, tensor))
    return tuple(bindings)


__all__ = ["Metric", "collect_metrics", "materialize_metrics", "metric"]
