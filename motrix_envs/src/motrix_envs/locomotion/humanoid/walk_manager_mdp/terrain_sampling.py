# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Bounded, host-only terrain-column sampling for the mixed walk curriculum."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TerrainSamplingState:
    """Track completed episodes by column and difficulty, then sample columns.

    Uniform column probabilities preserve the terrain-type proportions encoded
    by the grid's column counts (including duplicate flat/stair columns). Each
    column/level cell starts with a failure-rate pseudoprior of 0.5. Only mature
    cells influence adaptation; the maximum ``4*p*(1-p)`` across those cells
    favors a learning frontier rather than mastered or consistently impossible
    terrain. Columns without mature evidence remain neutral with score 1.

    Difficulty levels must be recorded *before* the row curriculum updates them.
    Keeping cell EMAs separate prevents changes in the observed level mixture
    from masquerading as changes in a terrain type's failure rate. Duplicate
    columns remain independent; no terrain-name or simulator dependency is needed.
    """

    num_cols: int
    num_levels: int
    adaptive_fraction: float = 0.25
    max_multiplier: float = 2.0
    ema_alpha: float = 0.05
    min_count: int = 20
    base_probabilities: np.ndarray = field(init=False)
    failure_ema: np.ndarray = field(init=False)
    evidence_counts: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.num_cols < 1 or self.num_levels < 1:
            raise ValueError("terrain sampling requires positive column and level counts")
        if not 0.0 <= self.adaptive_fraction <= 1.0:
            raise ValueError("adaptive_fraction must lie in [0, 1]")
        if not np.isfinite(self.max_multiplier) or self.max_multiplier < 1.0:
            raise ValueError("max_multiplier must be finite and at least 1")
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must lie in (0, 1]")
        if self.min_count < 1:
            raise ValueError("min_count must be positive")
        self.base_probabilities = np.full(self.num_cols, 1.0 / self.num_cols, dtype=np.float64)
        self.failure_ema = np.full((self.num_cols, self.num_levels), 0.5, dtype=np.float64)
        self.evidence_counts = np.zeros((self.num_cols, self.num_levels), dtype=np.int64)

    def update(
        self,
        cols: np.ndarray,
        levels: np.ndarray,
        terminated: np.ndarray,
        active: np.ndarray,
    ) -> None:
        """Observe only completed/reset lanes, ignoring inactive initial resets.

        Arguments are aligned reset-lane slices, not full running-environment
        arrays. ``terminated`` is the actual failure flag: a surviving timeout
        is a successful completion, not a failure. ``active`` is normally
        ``episode_steps[reset_ids, 0] > 0``. Lane indices and levels are assumed
        valid upstream. Inputs may be 1-D or manager-style ``(N, 1)`` arrays.

        Repeated cell indices are reduced with bincount. A batch of n outcomes
        uses its mean failure rate and gain ``1-(1-alpha)**n``; this is
        order-independent, counts every episode and matches n identical scalar
        EMA updates without Python loops over environments.
        """
        active = np.asarray(active, dtype=np.bool_).reshape(-1)
        cols = np.asarray(cols).reshape(-1)[active]
        levels = np.asarray(levels).reshape(-1)[active]
        failed = np.asarray(terminated, dtype=np.bool_).reshape(-1)[active]
        cells = cols * self.num_levels + levels
        size = self.num_cols * self.num_levels
        counts = np.bincount(cells, minlength=size)
        failures = np.bincount(cells, weights=failed, minlength=size)
        observed = counts > 0
        mean_failure = failures[observed] / counts[observed]
        gain = 1.0 - (1.0 - self.ema_alpha) ** counts[observed]
        ema = self.failure_ema.reshape(-1)
        ema[observed] += gain * (mean_failure - ema[observed])
        self.evidence_counts += counts.reshape(self.num_cols, self.num_levels)

    def probabilities(self) -> np.ndarray:
        """Return a normalized, bounded mixed distribution over actual columns.

        For hardness h in [0, 1], adaptive weights are
        ``base * (1 + (max_multiplier-1)*h)``. Their normalization factor is
        in [1, max_multiplier], so mixing fraction f with base guarantees
        ``(1-f)*base <= p <= (1+f*(max_multiplier-1))*base``. In particular,
        f=.25 and max_multiplier=2 retain at least .75x baseline and at most
        1.25x baseline, tighter than the requested 2x cap. Bounds apply *after*
        normalization; no clipping followed by a bound-breaking renormalization.
        """
        mature = self.evidence_counts >= self.min_count
        failure = np.clip(self.failure_ema, 0.0, 1.0)
        cell_hardness = 4.0 * failure * (1.0 - failure)
        hardness = np.max(np.where(mature, cell_hardness, 0.0), axis=1)
        hardness = np.where(np.any(mature, axis=1), hardness, 1.0)
        adaptive = self.base_probabilities * (1.0 + (self.max_multiplier - 1.0) * hardness)
        adaptive /= adaptive.sum()
        return (1.0 - self.adaptive_fraction) * self.base_probabilities + self.adaptive_fraction * adaptive

    def resample(self, reset_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return one column per reset ID; caller assigns only those bindings.

        The caller owns the seeded RNG. This does not mutate bindings, levels,
        or evidence, and an empty reset consumes no draws.
        """
        return rng.choice(self.num_cols, size=reset_ids.size, p=self.probabilities())
