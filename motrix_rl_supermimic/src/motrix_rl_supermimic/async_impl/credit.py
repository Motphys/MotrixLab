# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0
# Adapted from SuperMimic (supermimic_vibed), Copyright (c) 2026 ETH Zurich,
# Chong Zhang. MIT License.

"""Async governance: strict post-warmup update credit.

Collect n environment steps → earn ``n * utd / num_envs`` updates. Before
warmup completes no credit accrues, so the learner never trains on empty
replay; after warmup the learner may never run more updates than earned,
preventing it from spinning on stale data.
"""

from __future__ import annotations


class UpdateCredit:
    def __init__(self, utd: float, warmup_steps: int):
        if utd <= 0:
            raise ValueError("utd must be positive")
        self.utd = utd
        self.warmup_steps = warmup_steps
        self.steps = 0
        self.updates = 0
        self._credit = 0.0

    def receive(self, n: int) -> None:
        """Record n collected environment steps (one per env of a batch counts as num_envs)."""
        self.steps += n
        if self.steps >= self.warmup_steps:
            self._credit += n * self.utd

    @property
    def warm(self) -> bool:
        return self.steps >= self.warmup_steps

    @property
    def available(self) -> int:
        """Whole updates currently earned but not yet consumed."""
        return int(self._credit) if self.warm else 0

    def consume(self) -> None:
        if not self.warm:
            raise RuntimeError("Update credit consumed before warmup")
        self._credit -= 1.0
        self.updates += 1
