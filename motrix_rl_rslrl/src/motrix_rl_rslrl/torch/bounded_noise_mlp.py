# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""RSL-RL actor with bounded state-independent scalar exploration noise."""

from __future__ import annotations

import torch
from rsl_rl.models import MLPModel


class BoundedNoiseMLPModel(MLPModel):
    """``MLPModel`` whose state-independent noise std stays in a fixed band."""

    def __init__(self, *args, min_noise_std: float = 0.25, max_noise_std: float | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if min_noise_std <= 0.0:
            raise ValueError(f"min_noise_std must be positive, got {min_noise_std}")
        if max_noise_std is not None and max_noise_std < min_noise_std:
            raise ValueError(f"max_noise_std ({max_noise_std}) < min_noise_std ({min_noise_std})")
        self.min_noise_std = float(min_noise_std)
        self.max_noise_std = None if max_noise_std is None else float(max_noise_std)
        if not self.stochastic or self.state_dependent_std or self.noise_std_type != "scalar":
            raise ValueError("BoundedNoiseMLPModel requires state-independent scalar exploration noise")
        self._clamp_noise()

    @torch.no_grad()
    def _clamp_noise(self) -> None:
        self.std.clamp_(min=self.min_noise_std, max=self.max_noise_std)

    def _update_distribution(self, obs: torch.Tensor) -> None:
        self._clamp_noise()
        super()._update_distribution(obs)
