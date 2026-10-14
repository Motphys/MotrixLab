# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Window-aggregated training diagnostics.

Mirrors holosoma's ``TensorAverageMeterDict`` plus per-key extremes: logging
only the last gradient step's values hides single-update events (e.g. a
one-iteration Q spike or entropy spike) inside a multi-iteration logging
window. Every key drains as the window mean; range-tracked keys additionally
drain ``{key}_max`` / ``{key}_min`` so transient spikes stay visible.

Values are accumulated on their producing device (a few tensor adds per
update); the only host synchronization is the ``.item()`` at the drain, which
callers place at their logging boundary.
"""

from __future__ import annotations

import torch


class WindowMetrics:
    """Accumulates per-update diagnostic scalars, drained once per log window."""

    def __init__(self) -> None:
        self._sums: dict[str, torch.Tensor] = {}
        self._counts: dict[str, int] = {}
        self._max: dict[str, torch.Tensor] = {}
        self._min: dict[str, torch.Tensor] = {}

    def add(self, key: str, value: torch.Tensor, track_range: bool = False) -> None:
        """Accumulate one scalar observation under ``key``.

        ``track_range=True`` also keeps the window max/min (drained as
        ``{key}_max`` / ``{key}_min``) — used for the spike-sensitive
        diagnostics (Q extrema and entropy).
        """
        v = value.detach().float()
        s = self._sums.get(key)
        if s is None:
            self._sums[key] = v.clone()
            self._counts[key] = 1
            if track_range:
                self._max[key] = v.clone()
                self._min[key] = v.clone()
        else:
            self._sums[key] = s + v
            self._counts[key] += 1
            if track_range:
                self._max[key] = torch.maximum(self._max[key], v)
                self._min[key] = torch.minimum(self._min[key], v)

    def stats_and_reset(self) -> dict[str, float]:
        """Drain window means (plus tracked max/min) accumulated since the last call.

        Returns ``{key}`` window means plus ``{key}_max`` / ``{key}_min`` for
        range-tracked keys, converted to floats; empty when nothing was
        accumulated. Resets the accumulator for the next window.
        """
        stats = {key: (total / self._counts[key]).item() for key, total in self._sums.items()}
        stats.update({f"{key}_max": value.item() for key, value in self._max.items()})
        stats.update({f"{key}_min": value.item() for key, value in self._min.items()})
        self._sums, self._counts = {}, {}
        self._max, self._min = {}, {}
        return stats
