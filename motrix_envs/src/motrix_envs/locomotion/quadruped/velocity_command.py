# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Task-specific random planar-velocity command binding for quadruped training."""

import torch

from motrix_env_core.input import CommandBinding, PlanarVelocityCommand


class RandomPlanarVelocityBinding(CommandBinding[PlanarVelocityCommand]):
    """Sample independent planar velocities and optional standing commands."""

    def __init__(
        self,
        lower: torch.Tensor,
        upper: torch.Tensor,
        *,
        rng: torch.Generator,
        standing_probability: float = 0.0,
        hard_case_probability: float = 0.0,
        hard_cases: tuple[tuple[float, float, float], ...] = (),
    ) -> None:
        self._lower = lower
        self._upper = upper
        self._standing_probability = standing_probability
        self._hard_case_probability = hard_case_probability
        self._hard_cases = torch.as_tensor(hard_cases, dtype=torch.float32)
        self._rng = rng

        if hard_case_probability > 0.0 and not hard_cases:
            raise ValueError("hard_case_probability requires at least one hard case")
        if self._hard_cases.numel() and (
            bool(torch.any(self._hard_cases < lower)) or bool(torch.any(self._hard_cases > upper))
        ):
            raise ValueError("hard-case commands must lie inside the configured velocity bounds")

    def read_command(self, *, batch_size: int = 1) -> PlanarVelocityCommand:
        u = torch.rand((batch_size, 3), generator=self._rng)
        values = self._lower + (self._upper - self._lower) * u
        if self._hard_case_probability > 0.0:
            hard = torch.rand((batch_size,), generator=self._rng) < self._hard_case_probability
            choices = torch.randint(0, len(self._hard_cases), (batch_size,), generator=self._rng)
            values[hard] = self._hard_cases[choices[hard]]
        if self._standing_probability > 0.0:
            standing = torch.rand((batch_size,), generator=self._rng) < self._standing_probability
            values[standing] = 0.0
        return PlanarVelocityCommand(values)

    @property
    def hard_case_probability(self) -> float:
        return self._hard_case_probability

    def set_hard_case_probability(self, probability: float) -> None:
        """Update the hard-case sampling probability for a training curriculum."""
        if not bool(torch.isfinite(torch.tensor(probability))) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"hard-case probability must be inside [0, 1], got {probability}")
        if probability > 0.0 and not self._hard_cases.numel():
            raise ValueError("hard-case probability requires at least one hard case")
        self._hard_case_probability = float(probability)


__all__ = ["RandomPlanarVelocityBinding"]
