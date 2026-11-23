# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Small, dependency-free statistical helpers for benchmark measurements."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def percentile(values: Iterable[float], q: float) -> float:
    """Return a linearly interpolated percentile using a normalized quantile.

    ``q`` is expressed from 0.0 to 1.0, and the calculation matches NumPy's
    ``method="linear"`` convention. Values are sorted internally, so callers
    may provide samples in measurement order.
    """

    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one sample")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be between 0.0 and 1.0")
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def percentiles(values: Sequence[float] | Iterable[float], *quantiles: float) -> tuple[float, ...]:
    """Return several percentiles from one sorted sample set.

    Quantiles use the same normalized 0.0--1.0 convention as :func:`percentile`.
    """

    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentiles requires at least one sample")
    if any(not 0.0 <= q <= 1.0 for q in quantiles):
        raise ValueError("quantiles must be between 0.0 and 1.0")
    result = []
    for q in quantiles:
        position = (len(ordered) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        result.append(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))
    return tuple(result)


__all__ = ["percentile", "percentiles"]
