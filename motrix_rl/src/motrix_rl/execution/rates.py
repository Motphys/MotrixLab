# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Small clock-independent rolling rate helpers for execution presenters."""

from __future__ import annotations

from collections import deque


class RollingRate:
    """Calculate a rolling counter rate from caller-supplied samples.

    The caller owns the clock and the counter scale.  ``sample`` stores one
    ``(now, counter)`` point and returns the rate across the retained window;
    it returns ``None`` until two samples have a non-zero time span.  A stalled
    counter is therefore a measured ``0.0`` once time has elapsed.
    """

    def __init__(self, window_seconds: float = 10.0) -> None:
        self.window_seconds = window_seconds
        self._samples: deque[tuple[float, float]] = deque()

    def sample(self, now: float, counter: float) -> float | None:
        """Add a sample and return counter units per second, if measurable."""
        self._samples.append((now, counter))
        while len(self._samples) >= 2 and self._samples[1][0] <= now - self.window_seconds:
            self._samples.popleft()
        first_time, first_counter = self._samples[0]
        span = now - first_time
        if span <= 0.0:
            return None
        return (counter - first_counter) / span

    def reset(self) -> None:
        """Discard retained samples."""
        self._samples.clear()
